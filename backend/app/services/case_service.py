"""Investigation case lifecycle.

Every mutation goes through this module, and every mutation writes a
``CaseEvent``. The investigation timeline is therefore a plain read of that
table - never reconstructed, inferred, or assembled in the UI.

Permission checks live here rather than in the routes because the rules depend
on *what* is changing: editing a title and closing a case arrive on the same
PATCH but need different authority.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.auth.permissions import Permission, has_permission
from app.core.exceptions import ConflictError, NotFoundError, PermissionError_, ValidationError
from app.db.enums import (
    CLOSED_CASE_STATUSES,
    OPEN_CASE_STATUSES,
    CaseEventType,
    CasePriority,
    CaseStatus,
)
from app.db.models.case import Case, CaseEvent, CaseNote, CaseTransaction
from app.db.models.transaction import RiskScore, Transaction
from app.db.models.user import User
from app.schemas.case import CaseCreate, CaseUpdate

#: Which permission each status transition demands.
STATUS_PERMISSIONS: dict[CaseStatus, Permission] = {
    CaseStatus.NEW: Permission.CREATE_CASE,
    CaseStatus.UNDER_REVIEW: Permission.CREATE_CASE,
    CaseStatus.ESCALATED: Permission.ESCALATE_CASE,
    CaseStatus.RESOLVED: Permission.RESOLVE_CASE,
    CaseStatus.FALSE_POSITIVE: Permission.RESOLVE_CASE,
    CaseStatus.CLOSED: Permission.CLOSE_CASE,
}


def _next_case_ref(db: Session) -> str:
    highest = db.execute(select(func.max(Case.id))).scalar() or 0
    return f"CASE-{2000 + highest + 1}"


def _record_event(
    db: Session,
    case: Case,
    actor: User | None,
    event_type: CaseEventType,
    description: str,
    *,
    from_value: str | None = None,
    to_value: str | None = None,
) -> CaseEvent:
    event = CaseEvent(
        case_id=case.id,
        actor_id=actor.id if actor else None,
        event_type=event_type,
        description=description[:400],
        from_value=from_value,
        to_value=to_value,
        occurred_at=datetime.now(UTC),
    )
    db.add(event)
    return event


def _refresh_peak_score(db: Session, case: Case) -> None:
    """Recompute the case's highest linked risk score."""
    peak = db.execute(
        select(func.max(RiskScore.business_score))
        .select_from(CaseTransaction)
        .join(RiskScore, RiskScore.transaction_id == CaseTransaction.transaction_id)
        .where(CaseTransaction.case_id == case.id)
    ).scalar()
    case.peak_risk_score = int(peak or 0)


def create_case(db: Session, actor: User, payload: CaseCreate) -> Case:
    """Open an investigation, optionally linking evidence immediately."""
    if payload.assigned_to is not None and not has_permission(
        actor.role_code, Permission.ASSIGN_CASE
    ):
        raise PermissionError_("Your role does not permit assigning cases.")

    now = datetime.now(UTC)
    case = Case(
        case_ref=_next_case_ref(db),
        title=payload.title.strip(),
        summary=(payload.summary or "").strip(),
        customer_id=payload.customer_id,
        status=CaseStatus.NEW,
        priority=payload.priority,
        assigned_to=payload.assigned_to,
        created_by=actor.id,
        peak_risk_score=0,
        opened_at=now,
    )
    db.add(case)
    db.flush()

    _record_event(
        db,
        case,
        actor,
        CaseEventType.CREATED,
        f"Investigation {case.case_ref} opened by {actor.full_name}.",
        to_value=CaseStatus.NEW.value,
    )
    if payload.transaction_ids:
        link_transactions(db, case, actor, payload.transaction_ids)
    if payload.assigned_to is not None:
        assignee = db.get(User, payload.assigned_to)
        if assignee is None:
            raise ValidationError("The selected assignee does not exist.")
        _record_event(
            db,
            case,
            actor,
            CaseEventType.ASSIGNED,
            f"Case assigned to {assignee.full_name}.",
            to_value=assignee.full_name,
        )

    # Derive the customer from linked evidence when not given explicitly.
    if case.customer_id is None and payload.transaction_ids:
        first = db.get(Transaction, payload.transaction_ids[0])
        if first is not None:
            case.customer_id = first.customer_id

    db.flush()
    return case


def update_case(db: Session, case: Case, actor: User, payload: CaseUpdate) -> Case:
    """Apply a partial update, checking each field against its own permission."""
    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        return case

    if "assigned_to" in changes and not has_permission(actor.role_code, Permission.ASSIGN_CASE):
        raise PermissionError_("Your role does not permit assigning cases.")
    if "priority" in changes and not has_permission(
        actor.role_code, Permission.CHANGE_CASE_PRIORITY
    ):
        raise PermissionError_("Your role does not permit changing case priority.")

    if "status" in changes and changes["status"] is not None:
        _apply_status_change(db, case, actor, CaseStatus(changes["status"]))

    if "priority" in changes and changes["priority"] is not None:
        new_priority = CasePriority(changes["priority"])
        if new_priority != case.priority:
            _record_event(
                db,
                case,
                actor,
                CaseEventType.PRIORITY_CHANGED,
                f"Priority changed from {case.priority.value.title()} to "
                f"{new_priority.value.title()}.",
                from_value=case.priority.value,
                to_value=new_priority.value,
            )
            case.priority = new_priority

    if "assigned_to" in changes:
        _apply_assignment(db, case, actor, changes["assigned_to"])

    if changes.get("title"):
        case.title = changes["title"].strip()
    if "summary" in changes and changes["summary"] is not None:
        case.summary = changes["summary"].strip()
    if "resolution_note" in changes and changes["resolution_note"] is not None:
        case.resolution_note = changes["resolution_note"].strip()

    db.flush()
    return case


def _apply_status_change(db: Session, case: Case, actor: User, new_status: CaseStatus) -> None:
    if new_status == case.status:
        return

    required = STATUS_PERMISSIONS.get(new_status)
    if required and not has_permission(actor.role_code, required):
        raise PermissionError_(
            f"Your role does not permit moving a case to "
            f"{new_status.value.replace('_', ' ').title()}."
        )

    reopening = case.status in CLOSED_CASE_STATUSES and new_status in OPEN_CASE_STATUSES
    if reopening and not has_permission(actor.role_code, Permission.REOPEN_CASE):
        raise PermissionError_("Your role does not permit reopening a closed case.")

    previous = case.status
    case.status = new_status

    if new_status in CLOSED_CASE_STATUSES:
        case.closed_at = datetime.now(UTC)
        _record_event(
            db,
            case,
            actor,
            CaseEventType.CLOSED,
            f"Case closed as {new_status.value.replace('_', ' ').title()}.",
            from_value=previous.value,
            to_value=new_status.value,
        )
        return

    if reopening:
        case.closed_at = None
        _record_event(
            db,
            case,
            actor,
            CaseEventType.REOPENED,
            f"Case reopened as {new_status.value.replace('_', ' ').title()}.",
            from_value=previous.value,
            to_value=new_status.value,
        )
        return

    _record_event(
        db,
        case,
        actor,
        CaseEventType.STATUS_CHANGED,
        f"Status changed from {previous.value.replace('_', ' ').title()} to "
        f"{new_status.value.replace('_', ' ').title()}.",
        from_value=previous.value,
        to_value=new_status.value,
    )


def _apply_assignment(db: Session, case: Case, actor: User, assignee_id: int | None) -> None:
    if assignee_id == case.assigned_to:
        return

    previous = db.get(User, case.assigned_to) if case.assigned_to else None
    if assignee_id is None:
        case.assigned_to = None
        _record_event(
            db,
            case,
            actor,
            CaseEventType.UNASSIGNED,
            "Case unassigned.",
            from_value=previous.full_name if previous else None,
        )
        return

    assignee = db.get(User, assignee_id)
    if assignee is None:
        raise ValidationError("The selected assignee does not exist.")
    if not assignee.is_active:
        raise ValidationError("Cannot assign a case to a deactivated account.")

    case.assigned_to = assignee.id
    _record_event(
        db,
        case,
        actor,
        CaseEventType.ASSIGNED,
        f"Case assigned to {assignee.full_name}.",
        from_value=previous.full_name if previous else None,
        to_value=assignee.full_name,
    )


def add_note(
    db: Session, case: Case, actor: User, body: str, evidence_ref: str | None = None
) -> CaseNote:
    """Attach an analyst note and record it on the timeline."""
    note = CaseNote(
        case_id=case.id,
        author_id=actor.id,
        body=body.strip(),
        evidence_ref=(evidence_ref or None),
        created_at=datetime.now(UTC),
    )
    db.add(note)
    detail = (
        f"Note added with evidence reference {evidence_ref}."
        if evidence_ref
        else "Investigation note added."
    )
    _record_event(db, case, actor, CaseEventType.NOTE_ADDED, detail)
    db.flush()
    return note


def link_transactions(
    db: Session, case: Case, actor: User, transaction_ids: list[int]
) -> int:
    """Link transactions as evidence, skipping any already attached.

    Every link writes a timeline event, including those made while the case is
    being opened - a case created from a flagged transaction must still show
    when that evidence was attached.
    """
    if not transaction_ids:
        return 0

    existing = {
        row[0]
        for row in db.execute(
            select(CaseTransaction.transaction_id).where(CaseTransaction.case_id == case.id)
        ).all()
    }
    found = {
        t.id: t
        for t in db.execute(
            select(Transaction).where(Transaction.id.in_(transaction_ids))
        ).scalars()
    }
    missing = set(transaction_ids) - set(found)
    if missing:
        raise NotFoundError(
            f"{len(missing)} of the selected transactions could not be found."
        )

    linked = 0
    for transaction_id in transaction_ids:
        if transaction_id in existing:
            continue
        db.add(
            CaseTransaction(
                case_id=case.id, transaction_id=transaction_id, linked_by=actor.id
            )
        )
        _record_event(
            db,
            case,
            actor,
            CaseEventType.TRANSACTION_LINKED,
            f"Transaction {found[transaction_id].transaction_ref} linked as evidence.",
            to_value=found[transaction_id].transaction_ref,
        )
        linked += 1

    db.flush()
    _refresh_peak_score(db, case)
    return linked


def unlink_transaction(db: Session, case: Case, actor: User, transaction_id: int) -> None:
    link = db.execute(
        select(CaseTransaction).where(
            CaseTransaction.case_id == case.id,
            CaseTransaction.transaction_id == transaction_id,
        )
    ).scalar_one_or_none()
    if link is None:
        raise NotFoundError("That transaction is not linked to this case.")

    transaction = db.get(Transaction, transaction_id)
    db.delete(link)
    _record_event(
        db,
        case,
        actor,
        CaseEventType.TRANSACTION_UNLINKED,
        f"Transaction {transaction.transaction_ref if transaction else transaction_id} "
        f"removed from evidence.",
    )
    db.flush()
    _refresh_peak_score(db, case)


def get_case(db: Session, case_ref: str) -> Case:
    case = db.execute(
        select(Case)
        .options(
            joinedload(Case.customer),
            joinedload(Case.assignee),
            joinedload(Case.creator),
            selectinload(Case.notes).joinedload(CaseNote.author),
            selectinload(Case.events).joinedload(CaseEvent.actor),
            selectinload(Case.links).joinedload(CaseTransaction.transaction),
        )
        .where(Case.case_ref == case_ref)
        # Without this, a Case already in the identity map is returned with its
        # cached relationships, so a just-changed assignee would read as stale.
        .execution_options(populate_existing=True)
    ).unique().scalar_one_or_none()
    if case is None:
        raise NotFoundError(f"Case {case_ref} could not be found.")
    return case


def build_query(
    *,
    status: CaseStatus | None = None,
    priority: CasePriority | None = None,
    assigned_to: int | None = None,
    customer_id: int | None = None,
    open_only: bool = False,
    search: str | None = None,
) -> Select[tuple[Case]]:
    stmt = select(Case).options(joinedload(Case.assignee), joinedload(Case.customer))
    if status is not None:
        stmt = stmt.where(Case.status == status)
    if priority is not None:
        stmt = stmt.where(Case.priority == priority)
    if assigned_to is not None:
        stmt = stmt.where(Case.assigned_to == assigned_to)
    if customer_id is not None:
        stmt = stmt.where(Case.customer_id == customer_id)
    if open_only:
        stmt = stmt.where(Case.status.in_(OPEN_CASE_STATUSES))
    if search:
        pattern = f"%{search.strip()}%"
        stmt = stmt.where(Case.title.ilike(pattern) | Case.case_ref.ilike(pattern))
    return stmt


def ensure_open(case: Case) -> None:
    """Guard mutations that make no sense on a closed case."""
    if case.status in CLOSED_CASE_STATUSES:
        raise ConflictError(
            "This case is closed. Reopen it before making further changes."
        )
