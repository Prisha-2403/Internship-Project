"""Alert centre schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.db.enums import AlertStatus, RiskLevel
from app.schemas.auth import UserSummary


class AlertOut(BaseModel):
    """One entry in the alert feed."""

    id: int
    alert_ref: str
    transaction_id: int
    transaction_ref: str
    customer_ref: str
    customer_name: str
    amount: float
    currency: str
    risk_level: RiskLevel
    risk_score: int
    headline: str
    reason_summary: str
    status: AlertStatus
    assignee: UserSummary | None = None
    triggered_at: datetime
    acknowledged_at: datetime | None = None


class AlertActionRequest(BaseModel):
    """Assignment target for the assign action."""

    assigned_to: int | None = Field(
        default=None,
        description="User to assign the alert to. Omit to assign to yourself.",
    )
    note: str | None = Field(default=None, max_length=400)


class AlertFeed(BaseModel):
    """Polling payload for the alert centre."""

    items: list[AlertOut]
    total: int
    unread: int
    generated_at: datetime
