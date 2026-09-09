"""Transaction monitoring, investigation and export endpoints."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy import select

from app.auth.dependencies import CurrentUser, DbSession, require_permission
from app.auth.permissions import Permission
from app.core.exceptions import NotFoundError
from app.core.middleware import get_client_ip, get_user_agent
from app.db.enums import AuditAction, RiskLevel, TransactionStatus
from app.db.models.case import Case, CaseTransaction
from app.schemas.common import MessageResponse, Page
from app.schemas.transaction import (
    TransactionDetail,
    TransactionListItem,
    TransactionStatusUpdate,
)
from app.repositories import transaction_repository as repo
from app.repositories.transaction_repository import TransactionFilters
from app.services import audit_service, export_service
from app.services.masking import mask_device_ref, mask_name

router = APIRouter(prefix="/transactions", tags=["Transactions"])

SortField = Literal["occurred_at", "amount", "risk_score", "customer"]
SortDirection = Literal["asc", "desc"]
ExportFormat = Literal["csv", "xlsx"]


def _filters(
    search: str | None = Query(default=None, max_length=120),
    risk_level: RiskLevel | None = Query(default=None),
    min_risk_score: int | None = Query(default=None, ge=0, le=100),
    max_risk_score: int | None = Query(default=None, ge=0, le=100),
    status: TransactionStatus | None = Query(default=None),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    min_amount: Decimal | None = Query(default=None, ge=0),
    max_amount: Decimal | None = Query(default=None, ge=0),
    location: str | None = Query(default=None, max_length=64),
    customer_ref: str | None = Query(default=None, max_length=32),
    detection_reason: str | None = Query(default=None, max_length=32),
    flagged_only: bool = Query(default=False),
) -> TransactionFilters:
    """Collect the monitoring filters into one object."""
    return TransactionFilters(
        search=search,
        risk_level=risk_level,
        min_risk_score=min_risk_score,
        max_risk_score=max_risk_score,
        status=status,
        date_from=date_from,
        date_to=date_to,
        min_amount=min_amount,
        max_amount=max_amount,
        location=location,
        customer_ref=customer_ref,
        detection_reason=detection_reason,
        flagged_only=flagged_only,
    )


Filters = Annotated[TransactionFilters, Depends(_filters)]


def _to_list_item(row) -> TransactionListItem:
    transaction, risk, customer, device = row
    from app.db.enums import DETECTION_REASON_LABELS, DetectionReason

    label = None
    if risk and risk.primary_reason:
        try:
            label = DETECTION_REASON_LABELS[DetectionReason(risk.primary_reason)]
        except ValueError:
            label = risk.primary_reason

    return TransactionListItem(
        id=transaction.id,
        transaction_ref=transaction.transaction_ref,
        customer_ref=customer.customer_ref,
        customer_name=mask_name(customer.full_name) or customer.customer_ref,
        amount=transaction.amount,
        currency=transaction.currency,
        occurred_at=transaction.occurred_at,
        location_city=transaction.location_city,
        device_ref=mask_device_ref(device.device_ref) if device else None,
        payment_method=transaction.payment_method,
        status=transaction.status,
        risk_score=risk.business_score if risk else None,
        risk_level=risk.risk_level if risk else None,
        primary_reason=risk.primary_reason if risk else None,
        primary_reason_label=label,
    )


@router.get(
    "",
    response_model=Page[TransactionListItem],
    summary="List transactions with filtering, sorting and pagination",
)
def list_transactions(
    db: DbSession,
    filters: Filters,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_TRANSACTIONS))],
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort_by: SortField = Query(default="occurred_at"),
    direction: SortDirection = Query(default="desc"),
) -> Page[TransactionListItem]:
    """One page of the monitoring table. All filtering happens in SQL."""
    total = repo.count(db, filters)
    rows = repo.list_transactions(
        db, filters, page=page, page_size=page_size, sort_by=sort_by, direction=direction
    )
    return Page.build([_to_list_item(r) for r in rows], total, page, page_size)


@router.get(
    "/locations",
    response_model=list[str],
    summary="Distinct transaction locations for the filter dropdown",
)
def locations(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_TRANSACTIONS))],
) -> list[str]:
    return repo.distinct_locations(db)


@router.get(
    "/export",
    summary="Export the filtered transaction list as CSV or Excel",
    response_class=Response,
)
def export(
    request: Request,
    db: DbSession,
    filters: Filters,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.EXPORT_DATA))],
    format: ExportFormat = Query(default="csv"),
    sort_by: SortField = Query(default="occurred_at"),
    direction: SortDirection = Query(default="desc"),
) -> Response:
    """Download the current view.

    The export is audited - who exported what, and how many rows - and the data
    carries the same masking as the screen.
    """
    records = repo.iter_for_export(db, filters, sort_by=sort_by, direction=direction)
    rows = export_service.build_rows(records)

    if format == "xlsx":
        payload = export_service.to_xlsx(rows)
    else:
        payload = export_service.to_csv(rows)
    filename = export_service.export_filename(format)

    audit_service.record_and_commit(
        db,
        action=AuditAction.EXPORT_PERFORMED,
        actor=user,
        resource_type="transactions",
        resource_id=filename,
        description=f"Exported {len(rows)} transactions as {format.upper()}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        new_state={"format": format, "row_count": len(rows)},
    )

    return Response(
        content=payload,
        media_type=export_service.MEDIA_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{transaction_ref}",
    response_model=TransactionDetail,
    summary="Transaction detail with its risk breakdown",
    responses={404: {"description": "No such transaction."}},
)
def get_transaction(
    transaction_ref: str,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_TRANSACTIONS))],
) -> TransactionDetail:
    """Everything the investigation page shows, including why it was flagged.

    Opening a transaction is itself an audited event: an investigation record
    should show who looked at what.
    """
    transaction = repo.get_by_ref(db, transaction_ref)
    if transaction is None:
        raise NotFoundError(f"Transaction {transaction_ref} could not be found.")

    risk, features = repo.get_risk_and_features(db, transaction.id)
    linked = [
        r[0]
        for r in db.execute(
            select(Case.case_ref)
            .join(CaseTransaction, CaseTransaction.case_id == Case.id)
            .where(CaseTransaction.transaction_id == transaction.id)
        ).all()
    ]

    audit_service.record_and_commit(
        db,
        action=AuditAction.TRANSACTION_VIEWED,
        actor=user,
        resource_type="transaction",
        resource_id=transaction.transaction_ref,
        description=f"Viewed transaction {transaction.transaction_ref}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
    )
    return TransactionDetail.from_models(transaction, risk, features, linked)


@router.patch(
    "/{transaction_ref}/status",
    response_model=MessageResponse,
    summary="Update a transaction's review status",
)
def update_status(
    transaction_ref: str,
    payload: TransactionStatusUpdate,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.FLAG_TRANSACTION))],
) -> MessageResponse:
    """Move a transaction through the review workflow."""
    transaction = repo.get_by_ref(db, transaction_ref)
    if transaction is None:
        raise NotFoundError(f"Transaction {transaction_ref} could not be found.")

    previous = transaction.status
    if previous == payload.status:
        return MessageResponse(
            message=f"Transaction is already {payload.status.value.replace('_', ' ').lower()}."
        )

    transaction.status = payload.status
    audit_service.record(
        db,
        action=AuditAction.TRANSACTION_STATUS_CHANGED,
        actor=user,
        resource_type="transaction",
        resource_id=transaction.transaction_ref,
        description=(
            f"Status changed from {previous.value} to {payload.status.value}."
            + (f" Note: {payload.note}" if payload.note else "")
        ),
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        previous_state={"status": previous.value},
        new_state={"status": payload.status.value},
    )
    db.commit()
    return MessageResponse(
        message=f"Transaction {transaction_ref} marked as "
        f"{payload.status.value.replace('_', ' ').lower()}."
    )
