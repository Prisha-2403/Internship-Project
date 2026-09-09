"""Investigation case schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.db.enums import CaseEventType, CasePriority, CaseStatus, RiskLevel
from app.schemas.auth import UserSummary
from app.schemas.common import ORMModel


class CaseCreate(BaseModel):
    title: str = Field(min_length=4, max_length=200)
    summary: str = Field(default="", max_length=4000)
    priority: CasePriority = CasePriority.MEDIUM
    customer_id: int | None = None
    transaction_ids: list[int] = Field(
        default_factory=list,
        max_length=100,
        description="Transactions to link as evidence when the case is opened.",
    )
    assigned_to: int | None = None


class CaseUpdate(BaseModel):
    """Partial update. Only the supplied fields change.

    Each field maps to a distinct permission, checked in the service layer:
    status transitions, priority changes and assignment are not all available
    to every role.
    """

    title: str | None = Field(default=None, min_length=4, max_length=200)
    summary: str | None = Field(default=None, max_length=4000)
    status: CaseStatus | None = None
    priority: CasePriority | None = None
    assigned_to: int | None = None
    resolution_note: str | None = Field(default=None, max_length=4000)


class CaseNoteCreate(BaseModel):
    body: str = Field(min_length=1, max_length=4000)
    evidence_ref: str | None = Field(
        default=None,
        max_length=255,
        description="Optional pointer to external evidence, e.g. a ticket reference.",
    )


class CaseTransactionLink(BaseModel):
    transaction_ids: list[int] = Field(min_length=1, max_length=100)


class CaseNoteOut(ORMModel):
    id: int
    body: str
    evidence_ref: str | None = None
    created_at: datetime
    author: UserSummary


class CaseEventOut(ORMModel):
    """One timeline entry, read straight from ``case_events``."""

    id: int
    event_type: CaseEventType
    description: str
    from_value: str | None = None
    to_value: str | None = None
    occurred_at: datetime
    actor: UserSummary | None = None


class CaseTransactionOut(BaseModel):
    """A transaction linked to a case."""

    id: int
    transaction_ref: str
    amount: float
    currency: str
    occurred_at: datetime
    location_city: str
    risk_score: int | None = None
    risk_level: RiskLevel | None = None
    linked_at: datetime


class CaseListItem(BaseModel):
    id: int
    case_ref: str
    title: str
    status: CaseStatus
    priority: CasePriority
    customer_ref: str | None = None
    customer_name: str | None = None
    assignee: UserSummary | None = None
    peak_risk_score: int
    transaction_count: int
    note_count: int
    opened_at: datetime
    closed_at: datetime | None = None
    updated_at: datetime


class CaseDetail(BaseModel):
    id: int
    case_ref: str
    title: str
    summary: str
    status: CaseStatus
    priority: CasePriority
    customer_ref: str | None = None
    customer_name: str | None = None
    customer_id: int | None = None
    assignee: UserSummary | None = None
    creator: UserSummary
    peak_risk_score: int
    opened_at: datetime
    closed_at: datetime | None = None
    resolution_note: str | None = None
    created_at: datetime
    updated_at: datetime
    transactions: list[CaseTransactionOut] = Field(default_factory=list)
    notes: list[CaseNoteOut] = Field(default_factory=list)
    timeline: list[CaseEventOut] = Field(default_factory=list)
