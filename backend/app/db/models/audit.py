"""The ``audit_logs`` table.

Append-only by construction:

* The ORM class exposes no update path and the service layer only ever inserts.
* The migration installs PostgreSQL rules that turn UPDATE and DELETE on this
  table into no-ops, and revokes both privileges from the application role.

Never write credentials, tokens or password hashes into ``previous_state`` /
``new_state`` - ``audit_service`` strips known-sensitive keys before persisting.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.enums import AuditAction

if TYPE_CHECKING:
    from app.db.models.user import User


class AuditLog(Base):
    """One immutable record of a security- or data-relevant action."""

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_action_created", "action", "created_at"),
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Actor id may be NULL (failed login for an unknown address); the email is
    # denormalised so the trail survives user deletion.
    actor_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    actor_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    actor_role: Mapped[str | None] = mapped_column(String(32), nullable=True)

    action: Mapped[AuditAction] = mapped_column(
        Enum(AuditAction, name="audit_action"), nullable=False, index=True
    )
    resource_type: Mapped[str | None] = mapped_column(String(48), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    description: Mapped[str] = mapped_column(String(400), nullable=False, default="")

    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)

    previous_state: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    new_state: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    actor: Mapped["User | None"] = relationship(lazy="joined")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AuditLog {self.action.value} by={self.actor_email}>"
