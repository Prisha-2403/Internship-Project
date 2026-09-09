"""Real-time alert centre endpoints.

The frontend polls ``GET /alerts`` on an interval; the payload carries the
unread count so the topbar badge and the feed stay in step.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select

from app.auth.dependencies import CurrentUser, DbSession, require_permission
from app.auth.permissions import Permission
from app.core.exceptions import NotFoundError, ValidationError
from app.core.middleware import get_client_ip, get_user_agent
from app.db.enums import AlertStatus, AuditAction, RiskLevel
from app.db.models.alert import Alert
from app.db.models.user import User
from app.schemas.alert import AlertActionRequest, AlertFeed, AlertOut
from app.schemas.auth import UserSummary
from app.services import alert_service, audit_service
from app.services.masking import mask_name

router = APIRouter(prefix="/alerts", tags=["Alerts"])


def _to_out(alert: Alert) -> AlertOut:
    transaction = alert.transaction
    customer = transaction.customer
    return AlertOut(
        id=alert.id,
        alert_ref=alert.alert_ref,
        transaction_id=transaction.id,
        transaction_ref=transaction.transaction_ref,
        customer_ref=customer.customer_ref,
        customer_name=mask_name(customer.full_name) or customer.customer_ref,
        amount=float(transaction.amount),
        currency=transaction.currency,
        risk_level=alert.risk_level,
        risk_score=alert.risk_score,
        headline=alert.headline,
        reason_summary=alert.reason_summary,
        status=alert.status,
        assignee=UserSummary.model_validate(alert.assignee) if alert.assignee else None,
        triggered_at=alert.triggered_at,
        acknowledged_at=alert.acknowledged_at,
    )


def _get_alert(db: DbSession, alert_ref: str) -> Alert:
    alert = db.execute(
        alert_service.build_query().where(Alert.alert_ref == alert_ref)
    ).unique().scalar_one_or_none()
    if alert is None:
        raise NotFoundError(f"Alert {alert_ref} could not be found.")
    return alert


@router.get("", response_model=AlertFeed, summary="Alert feed for the alert centre")
def list_alerts(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_ALERTS))],
    status: AlertStatus | None = Query(default=None),
    risk_level: RiskLevel | None = Query(default=None),
    unresolved_only: bool = Query(default=False),
    assigned_to_me: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    current: CurrentUser = None,  # type: ignore[assignment]
) -> AlertFeed:
    """Latest alerts, newest first, with the unread count for the badge."""
    stmt = alert_service.build_query(
        status=status,
        risk_level=risk_level,
        unresolved_only=unresolved_only,
        assigned_to=current.id if (assigned_to_me and current) else None,
    )
    total = int(
        db.execute(select(func.count()).select_from(stmt.subquery())).scalar() or 0
    )
    alerts = (
        db.execute(stmt.order_by(Alert.triggered_at.desc()).limit(limit)).unique().scalars().all()
    )
    return AlertFeed(
        items=[_to_out(a) for a in alerts],
        total=total,
        unread=alert_service.unread_count(db),
        generated_at=datetime.now(UTC),
    )


@router.post(
    "/{alert_ref}/acknowledge",
    response_model=AlertOut,
    summary="Mark an alert as reviewed",
)
def acknowledge(
    alert_ref: str,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.ACKNOWLEDGE_ALERT))],
) -> AlertOut:
    """Record that an analyst has looked at this alert."""
    alert = _get_alert(db, alert_ref)
    previous = alert.status
    alert.status = AlertStatus.VIEWED
    alert.acknowledged_by = user.id
    alert.acknowledged_at = datetime.now(UTC)

    audit_service.record(
        db,
        action=AuditAction.ALERT_ACKNOWLEDGED,
        actor=user,
        resource_type="alert",
        resource_id=alert.alert_ref,
        description=f"Acknowledged alert {alert.alert_ref}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        previous_state={"status": previous.value},
        new_state={"status": alert.status.value},
    )
    db.commit()
    db.refresh(alert)
    return _to_out(alert)


@router.post(
    "/{alert_ref}/assign",
    response_model=AlertOut,
    summary="Assign an alert to an analyst",
)
def assign(
    alert_ref: str,
    payload: AlertActionRequest,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.ASSIGN_CASE))],
) -> AlertOut:
    """Assign to a named analyst, or to yourself when none is given."""
    alert = _get_alert(db, alert_ref)
    target = user
    if payload.assigned_to is not None:
        candidate = db.get(User, payload.assigned_to)
        if candidate is None or not candidate.is_active:
            raise ValidationError("The selected analyst could not be found or is inactive.")
        target = candidate

    previous = alert.status
    alert.assigned_to = target.id
    alert.status = AlertStatus.ASSIGNED

    audit_service.record(
        db,
        action=AuditAction.CASE_ASSIGNED,
        actor=user,
        resource_type="alert",
        resource_id=alert.alert_ref,
        description=f"Alert {alert.alert_ref} assigned to {target.full_name}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        previous_state={"status": previous.value},
        new_state={"status": alert.status.value, "assigned_to": target.email},
    )
    db.commit()
    db.refresh(alert)
    return _to_out(alert)


@router.post(
    "/{alert_ref}/dismiss",
    response_model=AlertOut,
    summary="Dismiss an alert as not requiring investigation",
)
def dismiss(
    alert_ref: str,
    payload: AlertActionRequest,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.DISMISS_ALERT))],
) -> AlertOut:
    """Close an alert without opening a case.

    The alert is retained, not deleted: dismissing is a decision, and the audit
    trail needs to show who made it.
    """
    alert = _get_alert(db, alert_ref)
    previous = alert.status
    alert.status = AlertStatus.DISMISSED
    alert.acknowledged_by = user.id
    alert.acknowledged_at = datetime.now(UTC)

    audit_service.record(
        db,
        action=AuditAction.ALERT_DISMISSED,
        actor=user,
        resource_type="alert",
        resource_id=alert.alert_ref,
        description=(
            f"Dismissed alert {alert.alert_ref}."
            + (f" Reason: {payload.note}" if payload.note else "")
        ),
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        previous_state={"status": previous.value},
        new_state={"status": alert.status.value},
    )
    db.commit()
    db.refresh(alert)
    return _to_out(alert)
