"""CSV data ingestion endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile
from sqlalchemy import select

from app.auth.dependencies import CurrentUser, DbSession, require_permission
from app.auth.permissions import Permission
from app.core.exceptions import ImportError_, NotFoundError
from app.core.middleware import get_client_ip, get_user_agent
from app.db.enums import AuditAction
from app.db.models.ops import ImportJob
from app.ml import pipeline
from app.schemas.analytics import ImportSummary
from app.services import audit_service, import_service

router = APIRouter(tags=["Data Ingestion"])

ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",  # what many browsers send for .csv
    "text/plain",
    "application/octet-stream",
}


def _to_summary(job: ImportJob, flagged: int | None = None) -> ImportSummary:
    return ImportSummary(
        id=job.id,
        filename=job.filename,
        status=job.status,
        records_received=job.records_received,
        records_valid=job.records_valid,
        records_invalid=job.records_invalid,
        records_duplicate=job.records_duplicate,
        records_processed=job.records_processed,
        records_flagged=flagged if flagged is not None else job.records_flagged,
        errors=job.error_report or [],
        error_message=job.error_message,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.post(
    "/transactions/import",
    response_model=ImportSummary,
    summary="Import transactions from a CSV file",
    responses={400: {"description": "The file could not be read or is malformed."}},
)
def import_transactions(
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.IMPORT_DATA))],
    file: UploadFile = File(description="CSV file of transactions."),
    run_detection: bool = Query(
        default=True, description="Score the imported rows immediately."
    ),
) -> ImportSummary:
    """Validate and ingest a CSV, then score what was imported.

    Required columns: ``transaction_id``, ``customer_id``, ``amount``,
    ``timestamp``. Optional: ``currency``, ``location``, ``region``,
    ``device_id``, ``beneficiary_id``, ``payment_method``, ``channel``,
    ``latitude``, ``longitude``.

    Invalid rows are reported individually and the valid ones still import.
    """
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise ImportError_(
            "Only CSV files are supported. Please export your data as CSV and try again."
        )
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ImportError_(
            "Unable to process transaction data. Please upload a plain CSV file."
        )

    content = file.file.read()
    job, result = import_service.validate_and_import(db, content, filename, user)

    flagged = 0
    if run_detection and result.inserted_ids:
        # Score only the new rows; the pipeline pulls in each affected
        # customer's full history so features remain correct.
        run = pipeline.run_detection(db, transaction_ids=result.inserted_ids)
        flagged = run.alerts_raised
        job = db.get(ImportJob, job.id)
        job.records_flagged = flagged

    audit_service.record(
        db,
        action=AuditAction.IMPORT_PERFORMED,
        actor=user,
        resource_type="import_job",
        resource_id=job.id,
        description=(
            f"Imported {filename}: {result.processed} of {result.received} rows processed, "
            f"{result.invalid} invalid, {result.duplicate} duplicate."
        ),
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        new_state={
            "filename": filename,
            "received": result.received,
            "processed": result.processed,
            "invalid": result.invalid,
            "duplicate": result.duplicate,
            "flagged": flagged,
        },
    )
    db.commit()
    db.refresh(job)
    return _to_summary(job, flagged)


@router.get(
    "/imports",
    response_model=list[ImportSummary],
    summary="Recent import jobs",
)
def list_imports(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.IMPORT_DATA))],
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ImportSummary]:
    jobs = (
        db.execute(select(ImportJob).order_by(ImportJob.created_at.desc()).limit(limit))
        .unique()
        .scalars()
        .all()
    )
    return [_to_summary(j) for j in jobs]


@router.get(
    "/imports/{job_id}",
    response_model=ImportSummary,
    summary="Import job detail including the row-level error report",
)
def get_import(
    job_id: int,
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.IMPORT_DATA))],
) -> ImportSummary:
    job = db.get(ImportJob, job_id)
    if job is None:
        raise NotFoundError("That import job could not be found.")
    return _to_summary(job)


@router.get(
    "/transactions/import/template",
    summary="Download a CSV template with the expected columns",
)
def import_template(
    _: Annotated[CurrentUser, Depends(require_permission(Permission.IMPORT_DATA))],
) -> Response:
    """A ready-to-fill CSV header row with one example."""
    header = (
        "transaction_id,customer_id,amount,currency,timestamp,location,region,"
        "device_id,beneficiary_id,payment_method,channel,latitude,longitude\r\n"
    )
    example = (
        "TX-900001,CUST-1000,12500.00,INR,2026-08-20T14:21:00Z,Delhi,Delhi NCR,"
        "DEV-0123456789,BEN-012345678,UPI,MOBILE_APP,28.6139,77.2090\r\n"
    )
    return Response(
        content=(header + example).encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="sentinel-import-template.csv"'
        },
    )
