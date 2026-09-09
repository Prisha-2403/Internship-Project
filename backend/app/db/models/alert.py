"""The ``alerts`` table backing the real-time alert centre."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.enums import AlertStatus, RiskLevel

if TYPE_CHECKING:
    from app.db.models.transaction import Transaction
    from app.db.models.user import User


class Alert(Base, TimestampMixin):
    """Raised when a scored transaction crosses ``ALERT_MIN_SCORE``.

    One alert per transaction: re-running detection updates the existing row
    rather than flooding the queue with duplicates.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Human-facing identifier, e.g. ALRT-10245
    alert_ref: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    risk_level: Mapped[RiskLevel] = mapped_column(
        Enum(RiskLevel, name="risk_level"), nullable=False, index=True
    )
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    headline: Mapped[str] = mapped_column(String(160), nullable=False)
    reason_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, name="alert_status"),
        nullable=False,
        default=AlertStatus.NEW,
        index=True,
    )
    assigned_to: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    acknowledged_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    transaction: Mapped["Transaction"] = relationship(back_populates="alert")
    assignee: Mapped["User | None"] = relationship(foreign_keys=[assigned_to])
    acknowledger: Mapped["User | None"] = relationship(foreign_keys=[acknowledged_by])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Alert {self.alert_ref} score={self.risk_score}>"
