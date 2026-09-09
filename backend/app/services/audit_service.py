"""Audit trail writes.

The only supported operation is *append*. There is no update or delete path
here, and the database refuses both anyway (see the append-only rules in
migration 0001).

Any state snapshot passed in is scrubbed of credential-like keys before it is
persisted, so a caller that hands over a whole ORM row cannot accidentally
write a password hash into the trail.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.db.enums import AuditAction
from app.db.models.audit import AuditLog
from app.db.models.user import User

logger = logging.getLogger("sentinel.audit")

#: Keys never written to the audit trail, whatever the caller passes.
SENSITIVE_KEYS = frozenset(
    {
        "password",
        "password_hash",
        "passwd",
        "pwd",
        "secret",
        "secret_key",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "api_key",
        "session",
    }
)

#: Guards against one oversized payload bloating the table.
MAX_STATE_KEYS = 40
MAX_VALUE_LENGTH = 500


def scrub(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop sensitive keys and truncate long values."""
    if not state:
        return None

    cleaned: dict[str, Any] = {}
    for key, value in list(state.items())[:MAX_STATE_KEYS]:
        if key.lower() in SENSITIVE_KEYS:
            cleaned[key] = "[REDACTED]"
            continue
        if isinstance(value, datetime):
            cleaned[key] = value.isoformat()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > MAX_VALUE_LENGTH:
                cleaned[key] = value[:MAX_VALUE_LENGTH] + "..."
            else:
                cleaned[key] = value
        else:
            # Enums, Decimals, dates and the like.
            text = str(getattr(value, "value", value))
            cleaned[key] = text[:MAX_VALUE_LENGTH]
    return cleaned


def record(
    db: Session,
    *,
    action: AuditAction,
    actor: User | None = None,
    resource_type: str | None = None,
    resource_id: str | int | None = None,
    description: str = "",
    ip_address: str | None = None,
    user_agent: str | None = None,
    previous_state: dict[str, Any] | None = None,
    new_state: dict[str, Any] | None = None,
    actor_email: str | None = None,
) -> AuditLog:
    """Append one audit entry.

    The row is added to the caller's session but not committed - it lands in the
    same transaction as the change it describes, so an action and its audit
    record are never written apart.
    """
    entry = AuditLog(
        actor_id=actor.id if actor else None,
        actor_email=(actor.email if actor else actor_email),
        actor_role=(actor.role_code.value if actor else None),
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        description=description[:400],
        ip_address=ip_address,
        user_agent=user_agent,
        previous_state=scrub(previous_state),
        new_state=scrub(new_state),
        created_at=datetime.now(UTC),
    )
    db.add(entry)
    return entry


def record_and_commit(db: Session, **kwargs: Any) -> AuditLog:
    """Append an entry and commit immediately.

    For events with nothing else to persist alongside them - a login, a logout,
    a read that must be traceable.
    """
    entry = record(db, **kwargs)
    db.commit()
    return entry


def build_query(
    *,
    action: AuditAction | None = None,
    actor_id: int | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    search: str | None = None,
) -> Select[tuple[AuditLog]]:
    """Build the filtered audit-log query used by the audit endpoint."""
    stmt = select(AuditLog)
    if action is not None:
        stmt = stmt.where(AuditLog.action == action)
    if actor_id is not None:
        stmt = stmt.where(AuditLog.actor_id == actor_id)
    if resource_type:
        stmt = stmt.where(AuditLog.resource_type == resource_type)
    if resource_id:
        stmt = stmt.where(AuditLog.resource_id == resource_id)
    if date_from is not None:
        stmt = stmt.where(AuditLog.created_at >= date_from)
    if date_to is not None:
        stmt = stmt.where(AuditLog.created_at <= date_to)
    if search:
        pattern = f"%{search.strip()}%"
        stmt = stmt.where(
            AuditLog.description.ilike(pattern) | AuditLog.actor_email.ilike(pattern)
        )
    return stmt
