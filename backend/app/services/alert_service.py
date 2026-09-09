"""Alert lifecycle.

An alert is raised when a scored transaction reaches ``ALERT_MIN_SCORE``. There
is at most one alert per transaction, so re-running detection refreshes the
existing row rather than filling the queue with duplicates.

Analyst decisions are never overwritten by a rescore: once someone has viewed,
assigned, escalated or dismissed an alert, only its score and text refresh.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.db.enums import AlertStatus, RiskLevel, TransactionStatus
from app.db.models.alert import Alert
from app.db.models.transaction import RiskScore, Transaction

logger = logging.getLogger("sentinel.alerts")

CHUNK_SIZE = 5000


def _next_alert_number(db: Session) -> int:
    highest = db.execute(select(func.max(Alert.id))).scalar()
    return int(highest or 0) + 1


def _headline(transaction: Transaction, score: RiskScore) -> str:
    level = score.risk_level.value.replace("_", " ").title()
    return f"{level} risk transaction {transaction.transaction_ref}"


def sync_alerts(db: Session, transaction_ids: list[int]) -> int:
    """Create, refresh or retract alerts for the given transactions.

    Returns the number of alerts newly raised.
    """
    if not transaction_ids:
        return 0

    raised = 0
    next_number = _next_alert_number(db)

    for start in range(0, len(transaction_ids), CHUNK_SIZE):
        chunk = transaction_ids[start : start + CHUNK_SIZE]

        rows = db.execute(
            select(Transaction, RiskScore)
            .join(RiskScore, RiskScore.transaction_id == Transaction.id)
            .where(Transaction.id.in_(chunk))
        ).all()

        existing = {
            alert.transaction_id: alert
            for alert in db.execute(
                select(Alert).where(Alert.transaction_id.in_(chunk))
            ).scalars()
        }

        for transaction, score in rows:
            alert = existing.get(transaction.id)
            qualifies = score.business_score >= settings.alert_min_score

            if not qualifies:
                # Only retract alerts nobody has acted on yet.
                if alert is not None and alert.status == AlertStatus.NEW:
                    db.delete(alert)
                    if transaction.status == TransactionStatus.FLAGGED:
                        transaction.status = TransactionStatus.COMPLETED
                continue

            summary = _summarise(score)
            if alert is None:
                alert = Alert(
                    alert_ref=f"ALRT-{next_number:05d}",
                    transaction_id=transaction.id,
                    risk_level=score.risk_level,
                    risk_score=score.business_score,
                    headline=_headline(transaction, score),
                    reason_summary=summary,
                    status=AlertStatus.NEW,
                    triggered_at=score.scored_at or datetime.now(UTC),
                )
                db.add(alert)
                next_number += 1
                raised += 1
            else:
                alert.risk_level = score.risk_level
                alert.risk_score = score.business_score
                alert.headline = _headline(transaction, score)
                alert.reason_summary = summary

            # Flagging is a system observation, so it must not clobber a status
            # an analyst has deliberately set.
            if transaction.status == TransactionStatus.COMPLETED:
                transaction.status = TransactionStatus.FLAGGED

    db.flush()
    return raised


def _summarise(score: RiskScore) -> str:
    """One-line reason text built from the stored factors."""
    factors = score.factors or []
    labels = [f.get("label", "") for f in factors if f.get("label")]
    if not labels:
        return "Risk threshold exceeded."
    if len(labels) == 1:
        return f"{labels[0]} detected."
    head = ", ".join(label.lower() for label in labels[:3])
    if len(labels) > 3:
        return f"Multiple behavioural anomalies detected: {head} and {len(labels) - 3} more."
    return f"Multiple behavioural anomalies detected: {head}."


def build_query(
    *,
    status: AlertStatus | None = None,
    risk_level: RiskLevel | None = None,
    assigned_to: int | None = None,
    unresolved_only: bool = False,
    min_score: int | None = None,
) -> Select[tuple[Alert]]:
    """Filtered alert query with the relations the list view renders."""
    stmt = select(Alert).options(
        joinedload(Alert.transaction).joinedload(Transaction.customer),
        joinedload(Alert.assignee),
    )
    if status is not None:
        stmt = stmt.where(Alert.status == status)
    if risk_level is not None:
        stmt = stmt.where(Alert.risk_level == risk_level)
    if assigned_to is not None:
        stmt = stmt.where(Alert.assigned_to == assigned_to)
    if unresolved_only:
        stmt = stmt.where(Alert.status.in_([AlertStatus.NEW, AlertStatus.VIEWED]))
    if min_score is not None:
        stmt = stmt.where(Alert.risk_score >= min_score)
    return stmt


def unread_count(db: Session) -> int:
    """Alerts nobody has looked at - the topbar badge."""
    return int(
        db.execute(
            select(func.count(Alert.id)).where(Alert.status == AlertStatus.NEW)
        ).scalar()
        or 0
    )
