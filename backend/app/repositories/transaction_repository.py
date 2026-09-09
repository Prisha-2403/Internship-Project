"""Transaction querying: filtering, sorting, pagination and lookups.

All filtering happens in SQL. The monitoring table is expected to sit over tens
of thousands of rows, so pulling them into Python to filter would not survive
contact with a real dataset.

Every filter value is bound as a parameter through SQLAlchemy - no query text is
ever assembled from user input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.db.enums import RiskLevel, TransactionStatus
from app.db.models.customer import Beneficiary, Customer, Device
from app.db.models.transaction import RiskScore, Transaction, TransactionFeature

SortField = Literal["occurred_at", "amount", "risk_score", "customer"]
SortDirection = Literal["asc", "desc"]

MAX_PAGE_SIZE = 100
ALLOWED_PAGE_SIZES = (20, 50, 100)


@dataclass
class TransactionFilters:
    """Everything the monitoring page can filter on."""

    search: str | None = None
    risk_level: RiskLevel | None = None
    min_risk_score: int | None = None
    max_risk_score: int | None = None
    status: TransactionStatus | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_amount: Decimal | None = None
    max_amount: Decimal | None = None
    location: str | None = None
    customer_ref: str | None = None
    customer_id: int | None = None
    detection_reason: str | None = None
    flagged_only: bool = False


def _base_query() -> Select:
    """Transactions joined to their customer and score.

    An outer join on ``risk_scores`` keeps not-yet-scored transactions visible
    rather than silently dropping them from the monitoring table.
    """
    return (
        select(Transaction, RiskScore, Customer, Device)
        .join(Customer, Transaction.customer_id == Customer.id)
        .outerjoin(RiskScore, RiskScore.transaction_id == Transaction.id)
        .outerjoin(Device, Transaction.device_id == Device.id)
    )


def apply_filters(stmt: Select, filters: TransactionFilters) -> Select:
    """Attach every active filter to ``stmt``."""
    if filters.search:
        pattern = f"%{filters.search.strip()}%"
        stmt = stmt.where(
            or_(
                Transaction.transaction_ref.ilike(pattern),
                Customer.customer_ref.ilike(pattern),
                Customer.full_name.ilike(pattern),
                Transaction.location_city.ilike(pattern),
            )
        )
    if filters.risk_level is not None:
        stmt = stmt.where(RiskScore.risk_level == filters.risk_level)
    if filters.min_risk_score is not None:
        stmt = stmt.where(RiskScore.business_score >= filters.min_risk_score)
    if filters.max_risk_score is not None:
        stmt = stmt.where(RiskScore.business_score <= filters.max_risk_score)
    if filters.status is not None:
        stmt = stmt.where(Transaction.status == filters.status)
    if filters.date_from is not None:
        stmt = stmt.where(Transaction.occurred_at >= filters.date_from)
    if filters.date_to is not None:
        stmt = stmt.where(Transaction.occurred_at <= filters.date_to)
    if filters.min_amount is not None:
        stmt = stmt.where(Transaction.amount >= filters.min_amount)
    if filters.max_amount is not None:
        stmt = stmt.where(Transaction.amount <= filters.max_amount)
    if filters.location:
        stmt = stmt.where(Transaction.location_city == filters.location)
    if filters.customer_ref:
        stmt = stmt.where(Customer.customer_ref == filters.customer_ref)
    if filters.customer_id is not None:
        stmt = stmt.where(Transaction.customer_id == filters.customer_id)
    if filters.detection_reason:
        stmt = stmt.where(RiskScore.primary_reason == filters.detection_reason)
    if filters.flagged_only:
        stmt = stmt.where(
            Transaction.status.in_([TransactionStatus.FLAGGED, TransactionStatus.UNDER_REVIEW])
        )
    return stmt


def apply_sort(stmt: Select, sort_by: SortField, direction: SortDirection) -> Select:
    """Order results, always with a unique tiebreak.

    Without the trailing id, rows with equal sort keys can shuffle between pages
    and an analyst paging through a table would see duplicates and gaps.
    """
    columns = {
        "occurred_at": Transaction.occurred_at,
        "amount": Transaction.amount,
        "risk_score": RiskScore.business_score,
        "customer": Customer.customer_ref,
    }
    column = columns.get(sort_by, Transaction.occurred_at)
    ordering = column.desc() if direction == "desc" else column.asc()
    # Nulls last for score so unscored rows do not head a descending sort.
    if sort_by == "risk_score":
        ordering = ordering.nullslast()
    return stmt.order_by(ordering, Transaction.id.desc())


def count(db: Session, filters: TransactionFilters) -> int:
    stmt = (
        select(func.count(Transaction.id))
        .select_from(Transaction)
        .join(Customer, Transaction.customer_id == Customer.id)
        .outerjoin(RiskScore, RiskScore.transaction_id == Transaction.id)
    )
    return int(db.execute(apply_filters(stmt, filters)).scalar() or 0)


def list_transactions(
    db: Session,
    filters: TransactionFilters,
    *,
    page: int = 1,
    page_size: int = 20,
    sort_by: SortField = "occurred_at",
    direction: SortDirection = "desc",
) -> list[tuple[Transaction, RiskScore | None, Customer, Device | None]]:
    """One page of the monitoring table."""
    page_size = min(max(page_size, 1), MAX_PAGE_SIZE)
    offset = max(page - 1, 0) * page_size
    stmt = apply_sort(apply_filters(_base_query(), filters), sort_by, direction)
    return list(db.execute(stmt.offset(offset).limit(page_size)).all())


def iter_for_export(
    db: Session,
    filters: TransactionFilters,
    *,
    sort_by: SortField = "occurred_at",
    direction: SortDirection = "desc",
    limit: int = 50_000,
) -> list[tuple[Transaction, RiskScore | None, Customer, Device | None]]:
    """Rows for a CSV/XLSX export.

    Capped so one click cannot try to materialise an unbounded result set.
    """
    stmt = apply_sort(apply_filters(_base_query(), filters), sort_by, direction)
    return list(db.execute(stmt.limit(limit)).all())


def get_by_ref(db: Session, transaction_ref: str) -> Transaction | None:
    return db.execute(
        select(Transaction)
        .options(
            joinedload(Transaction.customer),
            joinedload(Transaction.device),
            joinedload(Transaction.beneficiary),
        )
        .where(Transaction.transaction_ref == transaction_ref)
    ).scalar_one_or_none()


def get_by_id(db: Session, transaction_id: int) -> Transaction | None:
    return db.execute(
        select(Transaction)
        .options(
            joinedload(Transaction.customer),
            joinedload(Transaction.device),
            joinedload(Transaction.beneficiary),
        )
        .where(Transaction.id == transaction_id)
    ).scalar_one_or_none()


def get_risk_and_features(
    db: Session, transaction_id: int
) -> tuple[RiskScore | None, TransactionFeature | None]:
    risk = db.execute(
        select(RiskScore).where(RiskScore.transaction_id == transaction_id)
    ).scalar_one_or_none()
    features = db.execute(
        select(TransactionFeature).where(TransactionFeature.transaction_id == transaction_id)
    ).scalar_one_or_none()
    return risk, features


def distinct_locations(db: Session) -> list[str]:
    """City list for the location filter dropdown."""
    return [
        row[0]
        for row in db.execute(
            select(Transaction.location_city)
            .distinct()
            .order_by(Transaction.location_city)
        ).all()
        if row[0]
    ]


def search_global(db: Session, term: str, limit: int = 5) -> dict[str, list[dict]]:
    """Backing query for the top-bar global search."""
    pattern = f"%{term.strip()}%"

    transactions = db.execute(
        select(Transaction.id, Transaction.transaction_ref, Transaction.amount)
        .where(Transaction.transaction_ref.ilike(pattern))
        .order_by(Transaction.occurred_at.desc())
        .limit(limit)
    ).all()

    customers = db.execute(
        select(Customer.id, Customer.customer_ref, Customer.full_name, Customer.home_city)
        .where(
            or_(Customer.customer_ref.ilike(pattern), Customer.full_name.ilike(pattern))
        )
        .limit(limit)
    ).all()

    return {
        "transactions": [
            {"id": t.id, "ref": t.transaction_ref, "amount": float(t.amount)}
            for t in transactions
        ],
        "customers": [
            {"id": c.id, "ref": c.customer_ref, "name": c.full_name, "city": c.home_city}
            for c in customers
        ],
    }


def beneficiary_for(db: Session, beneficiary_id: int | None) -> Beneficiary | None:
    if beneficiary_id is None:
        return None
    return db.get(Beneficiary, beneficiary_id)
