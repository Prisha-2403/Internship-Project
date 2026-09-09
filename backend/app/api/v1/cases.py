"""Investigation case management endpoints.

Permission checks for status, priority and assignment live in
``case_service`` - they depend on which field is changing, so a single
route-level dependency could not express them.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status as http_status
from sqlalchemy import func, select

from app.auth.dependencies import CurrentUser, DbSession, require_permission
from app.auth.permissions import Permission
from app.core.middleware import get_client_ip, get_user_agent
from app.db.enums import AuditAction, CasePriority, CaseStatus
from app.db.models.case import Case, CaseNote, CaseTransaction
from app.db.models.transaction import RiskScore
from app.schemas.auth import UserSummary
from app.schemas.case import (
    CaseCreate,
    CaseDetail,
    CaseEventOut,
    CaseListItem,
    CaseNoteCreate,
    CaseNoteOut,
    CaseTransactionLink,
    CaseTransactionOut,
    CaseUpdate,
)
from app.schemas.common import MessageResponse, Page
from app.services import audit_service, case_service
from app.services.masking import mask_name

router = APIRouter(prefix="/cases", tags=["Investigation Cases"])


def _to_list_item(db: DbSession, case: Case) -> CaseListItem:
    transaction_count = int(
        db.execute(
            select(func.count(CaseTransaction.id)).where(CaseTransaction.case_id == case.id)
        ).scalar()
        or 0
    )
    note_count = int(
        db.execute(
            select(func.count(CaseNote.id)).where(CaseNote.case_id == case.id)
        ).scalar()
        or 0
    )
    return CaseListItem(
        id=case.id,
        case_ref=case.case_ref,
        title=case.title,
        status=case.status,
        priority=case.priority,
        customer_ref=case.customer.customer_ref if case.customer else None,
        customer_name=mask_name(case.customer.full_name) if case.customer else None,
        assignee=UserSummary.model_validate(case.assignee) if case.assignee else None,
        peak_risk_score=case.peak_risk_score,
        transaction_count=transaction_count,
        note_count=note_count,
        opened_at=case.opened_at,
        closed_at=case.closed_at,
        updated_at=case.updated_at,
    )


def _to_detail(db: DbSession, case: Case) -> CaseDetail:
    rows = db.execute(
        select(CaseTransaction, RiskScore)
        .outerjoin(RiskScore, RiskScore.transaction_id == CaseTransaction.transaction_id)
        .where(CaseTransaction.case_id == case.id)
        .order_by(CaseTransaction.created_at)
    ).all()

    transactions = [
        CaseTransactionOut(
            id=link.transaction.id,
            transaction_ref=link.transaction.transaction_ref,
            amount=float(link.transaction.amount),
            currency=link.transaction.currency,
            occurred_at=link.transaction.occurred_at,
            location_city=link.transaction.location_city,
            risk_score=risk.business_score if risk else None,
            risk_level=risk.risk_level if risk else None,
            linked_at=link.created_at,
        )
        for link, risk in rows
    ]

    return CaseDetail(
        id=case.id,
        case_ref=case.case_ref,
        title=case.title,
        summary=case.summary,
        status=case.status,
        priority=case.priority,
        customer_id=case.customer_id,
        customer_ref=case.customer.customer_ref if case.customer else None,
        customer_name=mask_name(case.customer.full_name) if case.customer else None,
        assignee=UserSummary.model_validate(case.assignee) if case.assignee else None,
        creator=UserSummary.model_validate(case.creator),
        peak_risk_score=case.peak_risk_score,
        opened_at=case.opened_at,
        closed_at=case.closed_at,
        resolution_note=case.resolution_note,
        created_at=case.created_at,
        updated_at=case.updated_at,
        transactions=transactions,
        notes=[CaseNoteOut.model_validate(n) for n in case.notes],
        timeline=[CaseEventOut.model_validate(e) for e in case.events],
    )


@router.get("", response_model=Page[CaseListItem], summary="List investigation cases")
def list_cases(
    db: DbSession,
    current: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_CASES))],
    status: CaseStatus | None = Query(default=None),
    priority: CasePriority | None = Query(default=None),
    assigned_to_me: bool = Query(default=False),
    open_only: bool = Query(default=False),
    search: str | None = Query(default=None, max_length=120),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Page[CaseListItem]:
    stmt = case_service.build_query(
        status=status,
        priority=priority,
        assigned_to=current.id if assigned_to_me else None,
        open_only=open_only,
        search=search,
    )
    total = int(
        db.execute(select(func.count()).select_from(stmt.subquery())).scalar() or 0
    )
    cases = (
        db.execute(
            stmt.order_by(Case.opened_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .unique()
        .scalars()
        .all()
    )
    return Page.build([_to_list_item(db, c) for c in cases], total, page, page_size)


@router.post(
    "",
    response_model=CaseDetail,
    status_code=http_status.HTTP_201_CREATED,
    summary="Open an investigation case",
)
def create_case(
    payload: CaseCreate,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.CREATE_CASE))],
) -> CaseDetail:
    case = case_service.create_case(db, user, payload)
    audit_service.record(
        db,
        action=AuditAction.CASE_CREATED,
        actor=user,
        resource_type="case",
        resource_id=case.case_ref,
        description=f"Opened investigation {case.case_ref}: {case.title}",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        new_state={
            "status": case.status.value,
            "priority": case.priority.value,
            "title": case.title,
        },
    )
    db.commit()
    return _to_detail(db, case_service.get_case(db, case.case_ref))


@router.get(
    "/{case_ref}",
    response_model=CaseDetail,
    summary="Case detail with evidence, notes and the investigation timeline",
    responses={404: {"description": "No such case."}},
)
def get_case(
    case_ref: str,
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_CASES))],
) -> CaseDetail:
    """The timeline is read straight from ``case_events``, never reconstructed."""
    return _to_detail(db, case_service.get_case(db, case_ref))


@router.patch(
    "/{case_ref}",
    response_model=CaseDetail,
    summary="Update a case (status, priority, assignment, details)",
)
def update_case(
    case_ref: str,
    payload: CaseUpdate,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_CASES))],
) -> CaseDetail:
    """Partial update. Each field is authorised separately in the service."""
    case = case_service.get_case(db, case_ref)
    previous = {
        "status": case.status.value,
        "priority": case.priority.value,
        "assigned_to": case.assigned_to,
    }
    case_service.update_case(db, case, user, payload)

    action = AuditAction.CASE_UPDATED
    if payload.status in (CaseStatus.RESOLVED, CaseStatus.FALSE_POSITIVE, CaseStatus.CLOSED):
        action = AuditAction.CASE_CLOSED
    elif payload.status in (CaseStatus.NEW, CaseStatus.UNDER_REVIEW) and previous["status"] in (
        CaseStatus.RESOLVED.value,
        CaseStatus.FALSE_POSITIVE.value,
        CaseStatus.CLOSED.value,
    ):
        action = AuditAction.CASE_REOPENED
    elif payload.assigned_to is not None:
        action = AuditAction.CASE_ASSIGNED

    audit_service.record(
        db,
        action=action,
        actor=user,
        resource_type="case",
        resource_id=case.case_ref,
        description=f"Updated case {case.case_ref}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        previous_state=previous,
        new_state={
            "status": case.status.value,
            "priority": case.priority.value,
            "assigned_to": case.assigned_to,
        },
    )
    db.commit()
    return _to_detail(db, case_service.get_case(db, case_ref))


@router.post(
    "/{case_ref}/notes",
    response_model=CaseNoteOut,
    status_code=http_status.HTTP_201_CREATED,
    summary="Add an investigation note",
)
def add_note(
    case_ref: str,
    payload: CaseNoteCreate,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.ADD_CASE_NOTE))],
) -> CaseNoteOut:
    case = case_service.get_case(db, case_ref)
    case_service.ensure_open(case)
    note = case_service.add_note(db, case, user, payload.body, payload.evidence_ref)

    audit_service.record(
        db,
        action=AuditAction.CASE_NOTE_ADDED,
        actor=user,
        resource_type="case",
        resource_id=case.case_ref,
        description=f"Added a note to case {case.case_ref}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
    )
    db.commit()
    db.refresh(note)
    return CaseNoteOut.model_validate(note)


@router.post(
    "/{case_ref}/transactions",
    response_model=CaseDetail,
    summary="Link transactions to a case as evidence",
)
def link_transactions(
    case_ref: str,
    payload: CaseTransactionLink,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.LINK_TRANSACTION))],
) -> CaseDetail:
    case = case_service.get_case(db, case_ref)
    case_service.ensure_open(case)
    linked = case_service.link_transactions(db, case, user, payload.transaction_ids)

    audit_service.record(
        db,
        action=AuditAction.CASE_UPDATED,
        actor=user,
        resource_type="case",
        resource_id=case.case_ref,
        description=f"Linked {linked} transaction(s) to case {case.case_ref}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        new_state={"linked_count": linked},
    )
    db.commit()
    return _to_detail(db, case_service.get_case(db, case_ref))


@router.delete(
    "/{case_ref}/transactions/{transaction_id}",
    response_model=MessageResponse,
    summary="Remove a transaction from a case",
)
def unlink_transaction(
    case_ref: str,
    transaction_id: int,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.LINK_TRANSACTION))],
) -> MessageResponse:
    case = case_service.get_case(db, case_ref)
    case_service.ensure_open(case)
    case_service.unlink_transaction(db, case, user, transaction_id)

    audit_service.record(
        db,
        action=AuditAction.CASE_UPDATED,
        actor=user,
        resource_type="case",
        resource_id=case.case_ref,
        description=f"Removed transaction {transaction_id} from case {case.case_ref}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
    )
    db.commit()
    return MessageResponse(message="Transaction removed from the case.")


@router.get(
    "/{case_ref}/timeline",
    response_model=list[CaseEventOut],
    summary="Investigation timeline for a case",
)
def timeline(
    case_ref: str,
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_CASES))],
) -> list[CaseEventOut]:
    case = case_service.get_case(db, case_ref)
    return [CaseEventOut.model_validate(e) for e in case.events]
