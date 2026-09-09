"""CSV transaction import with row-level validation.

Validation is strict and reported per row: a file with three bad rows imports
the other rows and tells the operator exactly which three failed and why,
rather than rejecting the upload wholesale or - worse - silently skipping them.

Every counter on the summary panel is a real tally produced here.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import ImportError_
from app.db.enums import ImportStatus, PaymentMethod, TransactionStatus
from app.db.models.customer import Beneficiary, Customer, Device
from app.db.models.ops import ImportJob
from app.db.models.transaction import Transaction
from app.db.models.user import User

logger = logging.getLogger("sentinel.import")

REQUIRED_COLUMNS = ("transaction_id", "customer_id", "amount", "timestamp")
OPTIONAL_COLUMNS = (
    "currency",
    "location",
    "region",
    "device_id",
    "beneficiary_id",
    "payment_method",
    "channel",
    "latitude",
    "longitude",
)

MAX_ROWS = 50_000
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_REPORTED_ERRORS = 200

MIN_AMOUNT = Decimal("0.01")
MAX_AMOUNT = Decimal("99999999999.99")

#: Accepted timestamp layouts, tried in order. ISO 8601 first.
TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
)


@dataclass
class RowError:
    row: int
    field: str
    message: str

    def as_dict(self) -> dict[str, str | int]:
        return {"row": self.row, "field": self.field, "message": self.message}


@dataclass
class ImportResult:
    received: int = 0
    valid: int = 0
    invalid: int = 0
    duplicate: int = 0
    processed: int = 0
    flagged: int = 0
    errors: list[RowError] = field(default_factory=list)
    inserted_ids: list[int] = field(default_factory=list)

    def add_error(self, row: int, field_name: str, message: str) -> None:
        if len(self.errors) < MAX_REPORTED_ERRORS:
            self.errors.append(RowError(row, field_name, message))


def parse_timestamp(raw: str) -> datetime:
    """Parse a timestamp in any accepted layout, always returning UTC-aware."""
    text = raw.strip()
    if not text:
        raise ValueError("Timestamp is required.")
    for fmt in TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    try:  # Last resort: anything else fromisoformat understands.
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    except ValueError as exc:
        raise ValueError(
            "Timestamp is not a recognised format. Use ISO 8601, e.g. 2026-08-20T14:21:00Z."
        ) from exc


def parse_amount(raw: str) -> Decimal:
    """Parse and range-check a monetary amount."""
    text = (raw or "").strip().replace(",", "")
    if not text:
        raise ValueError("Amount is required.")
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Amount is not a valid number.") from exc
    if not value.is_finite():
        raise ValueError("Amount is not a valid number.")
    if value < MIN_AMOUNT:
        raise ValueError("Amount must be greater than zero.")
    if value > MAX_AMOUNT:
        raise ValueError("Amount exceeds the maximum supported value.")
    return value.quantize(Decimal("0.01"))


def decode_csv(content: bytes) -> list[dict[str, str]]:
    """Decode and parse the upload into dict rows."""
    if len(content) > MAX_FILE_BYTES:
        raise ImportError_(
            "The uploaded file is too large. The maximum supported size is 20 MB."
        )
    if not content.strip():
        raise ImportError_("The uploaded file is empty.")

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ImportError_(
            "Unable to read the uploaded file. Please save it as UTF-8 encoded CSV."
        )

    try:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise ImportError_("The uploaded file has no header row.")
        headers = {(h or "").strip().lower() for h in reader.fieldnames}
        missing = [c for c in REQUIRED_COLUMNS if c not in headers]
        if missing:
            raise ImportError_(
                "Unable to process transaction data. The file is missing required "
                f"column(s): {', '.join(missing)}."
            )
        rows = []
        for index, row in enumerate(reader):
            if index >= MAX_ROWS:
                raise ImportError_(
                    f"The file contains more than {MAX_ROWS:,} rows. "
                    "Please split it into smaller batches."
                )
            rows.append({(k or "").strip().lower(): (v or "") for k, v in row.items()})
        return rows
    except csv.Error as exc:
        logger.warning("CSV parse failure: %s", exc)
        raise ImportError_(
            "Unable to process transaction data. Please verify the uploaded file and try again."
        ) from exc


def validate_and_import(
    db: Session, content: bytes, filename: str, uploaded_by: User
) -> tuple[ImportJob, ImportResult]:
    """Validate every row and insert the ones that pass.

    The job row is written whatever the outcome, so a failed import is as
    auditable as a successful one.
    """
    job = ImportJob(
        filename=filename[:255],
        uploaded_by=uploaded_by.id,
        status=ImportStatus.PROCESSING,
        started_at=datetime.now(UTC),
    )
    db.add(job)
    db.flush()

    result = ImportResult()
    try:
        rows = decode_csv(content)
    except ImportError_ as exc:
        job.status = ImportStatus.FAILED
        job.error_message = exc.message
        job.finished_at = datetime.now(UTC)
        db.commit()
        raise

    result.received = len(rows)
    if not rows:
        job.status = ImportStatus.COMPLETED
        job.finished_at = datetime.now(UTC)
        db.commit()
        return job, result

    customers = {
        ref: cid for ref, cid in db.execute(select(Customer.customer_ref, Customer.id)).all()
    }
    existing_refs = {
        r[0]
        for r in db.execute(
            select(Transaction.transaction_ref).where(
                Transaction.transaction_ref.in_(
                    [r.get("transaction_id", "").strip() for r in rows if r.get("transaction_id")]
                )
            )
        ).all()
    }
    devices = {
        (cid, ref): did
        for did, cid, ref in db.execute(
            select(Device.id, Device.customer_id, Device.device_ref)
        ).all()
    }
    beneficiaries = {
        (cid, ref): bid
        for bid, cid, ref in db.execute(
            select(Beneficiary.id, Beneficiary.customer_id, Beneficiary.beneficiary_ref)
        ).all()
    }

    seen_in_file: set[str] = set()
    to_insert: list[dict] = []

    for index, row in enumerate(rows, start=2):  # row 1 is the header
        transaction_ref = row.get("transaction_id", "").strip()
        customer_ref = row.get("customer_id", "").strip()

        if not transaction_ref:
            result.invalid += 1
            result.add_error(index, "transaction_id", "Transaction ID is required.")
            continue
        if transaction_ref in seen_in_file:
            result.duplicate += 1
            result.add_error(
                index, "transaction_id", f"Duplicate transaction ID {transaction_ref} in file."
            )
            continue
        if transaction_ref in existing_refs:
            result.duplicate += 1
            result.add_error(
                index, "transaction_id", f"Transaction {transaction_ref} already exists."
            )
            continue
        if customer_ref not in customers:
            result.invalid += 1
            result.add_error(
                index, "customer_id", f"Unknown customer reference {customer_ref or '(blank)'}."
            )
            continue

        try:
            amount = parse_amount(row.get("amount", ""))
        except ValueError as exc:
            result.invalid += 1
            result.add_error(index, "amount", str(exc))
            continue

        try:
            occurred_at = parse_timestamp(row.get("timestamp", ""))
        except ValueError as exc:
            result.invalid += 1
            result.add_error(index, "timestamp", str(exc))
            continue

        method_raw = (row.get("payment_method") or "UPI").strip().upper().replace(" ", "_")
        try:
            payment_method = PaymentMethod(method_raw)
        except ValueError:
            result.invalid += 1
            result.add_error(
                index,
                "payment_method",
                f"Unsupported payment method {method_raw!r}. Expected one of: "
                f"{', '.join(m.value for m in PaymentMethod)}.",
            )
            continue

        customer_id = customers[customer_ref]
        currency = (row.get("currency") or "INR").strip().upper()[:3] or "INR"
        location = (row.get("location") or "").strip() or "Unknown"

        seen_in_file.add(transaction_ref)
        result.valid += 1
        to_insert.append(
            {
                "transaction_ref": transaction_ref,
                "customer_id": customer_id,
                "amount": amount,
                "currency": currency,
                "occurred_at": occurred_at,
                "location_city": location[:64],
                "location_region": (row.get("region") or location).strip()[:64] or "Unknown",
                "latitude": _optional_float(row.get("latitude")),
                "longitude": _optional_float(row.get("longitude")),
                "device_id": devices.get((customer_id, (row.get("device_id") or "").strip())),
                "beneficiary_id": beneficiaries.get(
                    (customer_id, (row.get("beneficiary_id") or "").strip())
                ),
                "payment_method": payment_method,
                "channel": (row.get("channel") or "IMPORT").strip()[:32],
                "status": TransactionStatus.COMPLETED,
                "import_job_id": job.id,
                "is_demo": False,
                "injected_scenario": None,
            }
        )

    if to_insert:
        db.bulk_insert_mappings(Transaction, to_insert)
        db.flush()
        result.inserted_ids = [
            r[0]
            for r in db.execute(
                select(Transaction.id).where(Transaction.import_job_id == job.id)
            ).all()
        ]
        result.processed = len(result.inserted_ids)

    job.records_received = result.received
    job.records_valid = result.valid
    job.records_invalid = result.invalid
    job.records_duplicate = result.duplicate
    job.records_processed = result.processed
    job.error_report = [e.as_dict() for e in result.errors]
    job.status = (
        ImportStatus.COMPLETED
        if not result.errors
        else ImportStatus.COMPLETED_WITH_ERRORS
    )
    job.finished_at = datetime.now(UTC)
    db.flush()
    return job, result


def _optional_float(value: str | None) -> float | None:
    if not value or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None
