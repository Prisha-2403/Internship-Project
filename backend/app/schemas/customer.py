"""Customer risk-profile schemas.

Contact details are masked at this boundary. An analyst investigating a case
needs a stable handle and behavioural context, not a customer's full phone
number or account number.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, computed_field

from app.db.enums import RiskLevel
from app.schemas.common import ORMModel, TimeSeriesPoint
from app.services.masking import (
    mask_account_number,
    mask_email,
    mask_name,
    mask_phone,
)


class CustomerListItem(BaseModel):
    """One row in the customer risk list."""

    id: int
    customer_ref: str
    display_name: str
    home_city: str
    segment: str
    transaction_count: int
    average_amount: float
    max_risk_score: int
    average_risk_score: float
    risk_level: RiskLevel
    high_risk_count: int
    last_activity_at: datetime | None = None


class CustomerIdentity(ORMModel):
    """Masked identity block."""

    customer_ref: str
    # Read from the ORM so the masked computed fields below can be derived, but
    # excluded from serialisation - the raw values must never leave the API.
    full_name: str = Field(exclude=True)
    email: str = Field(exclude=True)
    phone: str = Field(exclude=True)
    account_number: str = Field(exclude=True)
    home_city: str
    home_region: str
    segment: str
    kyc_level: str
    onboarded_at: datetime
    is_active: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display_name(self) -> str:
        return mask_name(self.full_name) or self.customer_ref

    @computed_field  # type: ignore[prop-decorator]
    @property
    def masked_email(self) -> str | None:
        return mask_email(self.email)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def masked_phone(self) -> str | None:
        return mask_phone(self.phone)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def masked_account(self) -> str | None:
        return mask_account_number(self.account_number)


class BehaviourBaseline(BaseModel):
    """What normal looks like for this customer, over their full history."""

    average_amount: float
    median_amount: float
    average_daily_transactions: float
    active_hours_start: int
    active_hours_end: int
    usual_locations: list[str]
    known_devices: int
    known_beneficiaries: int
    total_transactions: int
    history_days: int


class RecentBehaviour(BaseModel):
    """The last 7 days, and how far they deviate from the baseline."""

    window_days: int
    transaction_count: int
    average_amount: float
    daily_transactions: float
    new_locations: int
    new_devices: int
    new_beneficiaries: int
    amount_change_pct: float = Field(
        description="Percentage change in average amount versus baseline. Positive is an increase."
    )
    frequency_change_pct: float
    behaviour_change_level: RiskLevel = Field(
        description="Severity of drift from baseline, on the shared risk scale."
    )


class CustomerRiskProfile(BaseModel):
    """The full customer risk page."""

    identity: CustomerIdentity
    risk_level: RiskLevel
    risk_score: int
    average_risk_score: float
    total_transactions: int
    flagged_transactions: int
    high_risk_transactions: int
    open_cases: int
    baseline: BehaviourBaseline
    recent: RecentBehaviour
    amount_history: list[TimeSeriesPoint]
    risk_history: list[TimeSeriesPoint]
    top_reasons: list[dict]
