"""Investigation case tables.

``cases`` + ``case_transactions`` + ``case_notes`` + ``case_events``.

Every mutation of a case writes a ``CaseEvent``; the investigation timeline is
a straight read of that table, never reconstructed or synthesised in the UI.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, CreatedAtMixin, TimestampMixin
from app.db.enums import CaseEventType, CasePriority, CaseStatus

if TYPE_CHECKING:
    from app.db.models.customer import Customer
    from app.db.models.transaction import Transaction
    from app.db.models.user import User


class Case(Base, TimestampMixin):
    """An investigation opened against suspicious activity."""

    __tablename__ = "cases"
    __table_args__ = (Index("ix_cases_status_priority", "status", "priority"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Human-facing identifier, e.g. CASE-2041
    case_ref: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    customer_id: Mapped[int | None] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[CaseStatus] = mapped_column(
        Enum(CaseStatus, name="case_status"),
        nullable=False,
        default=CaseStatus.NEW,
        index=True,
    )
    priority: Mapped[CasePriority] = mapped_column(
        Enum(CasePriority, name="case_priority"),
        nullable=False,
        default=CasePriority.MEDIUM,
        index=True,
    )

    assigned_to: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_by: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )

    # Highest risk score among linked transactions, refreshed on link/unlink so
    # the case list can sort by severity without an aggregate subquery.
    peak_risk_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)

    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    customer: Mapped["Customer | None"] = relationship()
    assignee: Mapped["User | None"] = relationship(
        back_populates="assigned_cases", foreign_keys=[assigned_to], lazy="joined"
    )
    creator: Mapped["User"] = relationship(foreign_keys=[created_by], lazy="joined")

    links: Mapped[list["CaseTransaction"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    notes: Mapped[list["CaseNote"]] = relationship(
        back_populates="case", cascade="all, delete-orphan", order_by="CaseNote.created_at"
    )
    events: Mapped[list["CaseEvent"]] = relationship(
        back_populates="case", cascade="all, delete-orphan", order_by="CaseEvent.occurred_at"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Case {self.case_ref} {self.status.value}>"


class CaseTransaction(Base, CreatedAtMixin):
    """Join row linking a transaction into a case as evidence."""

    __tablename__ = "case_transactions"
    __table_args__ = (
        UniqueConstraint("case_id", "transaction_id", name="uq_case_transactions_case_txn"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    linked_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    case: Mapped["Case"] = relationship(back_populates="links")
    transaction: Mapped["Transaction"] = relationship()
    linker: Mapped["User | None"] = relationship()


class CaseNote(Base, CreatedAtMixin):
    """An analyst's written observation on a case."""

    __tablename__ = "case_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    author_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # Free-text pointer to external evidence (ticket ref, document id, ...).
    evidence_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)

    case: Mapped["Case"] = relationship(back_populates="notes")
    author: Mapped["User"] = relationship(lazy="joined")


class CaseEvent(Base):
    """One entry in the investigation timeline.

    Written by ``case_service`` on every mutation. ``from_value`` / ``to_value``
    capture status, priority and assignment transitions.
    """

    __tablename__ = "case_events"
    __table_args__ = (Index("ix_case_events_case_occurred", "case_id", "occurred_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    actor_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    event_type: Mapped[CaseEventType] = mapped_column(
        Enum(CaseEventType, name="case_event_type"), nullable=False
    )
    description: Mapped[str] = mapped_column(String(400), nullable=False)
    from_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    to_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    case: Mapped["Case"] = relationship(back_populates="events")
    actor: Mapped["User | None"] = relationship(lazy="joined")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CaseEvent {self.event_type.value} case={self.case_id}>"
