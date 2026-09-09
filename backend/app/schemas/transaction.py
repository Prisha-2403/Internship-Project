"""Transaction, feature and risk-score schemas.

Customer-identifying fields are masked here, at the serialisation boundary, so
no endpoint can leak them by omission.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, computed_field

from app.db.enums import (
    DETECTION_REASON_LABELS,
    AnomalyScenario,
    DetectionReason,
    PaymentMethod,
    RiskLevel,
    TransactionStatus,
)
from app.schemas.common import ORMModel
from app.services.masking import mask_account_number, mask_device_ref, mask_name


class RiskFactorOut(BaseModel):
    """One rule that fired, as stored on the risk score."""

    code: str
    label: str
    points: int = Field(description="Contribution to the 0-100 score.")
    detail: str
    raw_points: int | None = Field(
        default=None,
        description="Contribution before scaling onto the 0-100 scale, when scaling applied.",
    )


class RiskScoreOut(ORMModel):
    """The scoring verdict.

    ``business_score`` is the analyst-facing number and always equals the sum of
    ``factors``. ``ml_anomaly_score`` is a separate 0-1 measure of how unusual
    the transaction looks to the Isolation Forest - it is not a probability that
    fraud occurred.
    """

    business_score: int
    risk_level: RiskLevel
    rule_score: int
    ml_uplift_points: int
    ml_anomaly_score: float
    is_ml_anomaly: bool
    primary_reason: str | None = None
    factors: list[RiskFactorOut] = Field(default_factory=list)
    scored_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def primary_reason_label(self) -> str | None:
        if not self.primary_reason:
            return None
        try:
            return DETECTION_REASON_LABELS[DetectionReason(self.primary_reason)]
        except ValueError:
            return self.primary_reason


class TransactionCustomerOut(ORMModel):
    """Customer reference shown alongside a transaction, masked."""

    customer_ref: str
    # Source for the masked display name below; never serialised itself.
    full_name: str = Field(exclude=True)
    home_city: str
    segment: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display_name(self) -> str:
        return mask_name(self.full_name) or self.customer_ref


class TransactionListItem(BaseModel):
    """One row in the monitoring table."""

    id: int
    transaction_ref: str
    customer_ref: str
    customer_name: str
    amount: Decimal
    currency: str
    occurred_at: datetime
    location_city: str
    device_ref: str | None = None
    payment_method: PaymentMethod
    status: TransactionStatus
    risk_score: int | None = None
    risk_level: RiskLevel | None = None
    primary_reason: str | None = None
    primary_reason_label: str | None = None


class TransactionDetail(BaseModel):
    """Everything the investigation page needs for one transaction."""

    id: int
    transaction_ref: str
    amount: Decimal
    currency: str
    occurred_at: datetime
    location_city: str
    location_region: str
    latitude: float | None = None
    longitude: float | None = None
    payment_method: PaymentMethod
    channel: str
    status: TransactionStatus
    is_demo: bool
    injected_scenario: AnomalyScenario | None = Field(
        default=None,
        description=(
            "Which synthetic scenario generated this row. Demo bookkeeping only - "
            "not a fraud label and not used as ground truth."
        ),
    )

    customer: TransactionCustomerOut
    device_ref: str | None = None
    device_type: str | None = None
    beneficiary_name: str | None = None
    beneficiary_account: str | None = None
    beneficiary_bank: str | None = None

    risk: RiskScoreOut | None = None
    features: "TransactionFeatureOut | None" = None
    linked_case_refs: list[str] = Field(default_factory=list)

    @classmethod
    def from_models(
        cls,
        transaction: Any,
        risk: Any | None,
        features: Any | None,
        linked_case_refs: list[str],
    ) -> "TransactionDetail":
        beneficiary = transaction.beneficiary
        device = transaction.device
        return cls(
            id=transaction.id,
            transaction_ref=transaction.transaction_ref,
            amount=transaction.amount,
            currency=transaction.currency,
            occurred_at=transaction.occurred_at,
            location_city=transaction.location_city,
            location_region=transaction.location_region,
            latitude=transaction.latitude,
            longitude=transaction.longitude,
            payment_method=transaction.payment_method,
            channel=transaction.channel,
            status=transaction.status,
            is_demo=transaction.is_demo,
            injected_scenario=transaction.injected_scenario,
            customer=TransactionCustomerOut.model_validate(transaction.customer),
            device_ref=mask_device_ref(device.device_ref) if device else None,
            device_type=device.device_type if device else None,
            beneficiary_name=beneficiary.display_name if beneficiary else None,
            beneficiary_account=(
                mask_account_number(beneficiary.account_number) if beneficiary else None
            ),
            beneficiary_bank=beneficiary.bank_name if beneficiary else None,
            risk=RiskScoreOut.model_validate(risk) if risk else None,
            features=TransactionFeatureOut.model_validate(features) if features else None,
            linked_case_refs=linked_case_refs,
        )


class TransactionFeatureOut(ORMModel):
    """Engineered features behind a score, for the explainability panel."""

    amount_value: float
    customer_avg_amount: float
    customer_std_amount: float
    amount_zscore: float
    amount_vs_customer_avg_ratio: float
    customer_p99_amount: float
    txn_count_10m: int
    txn_count_1h: int
    txn_count_24h: int
    customer_avg_daily_txns: float
    velocity_score: float
    hour_of_day: int
    day_of_week: int
    is_unusual_hour: bool
    is_new_device: bool
    is_new_location: bool
    is_new_beneficiary: bool
    device_usage_count: int
    distance_from_usual_km: float
    unique_devices_30d: int
    unique_locations_30d: int
    behavior_change_score: float
    amount_trend_ratio: float
    frequency_trend_ratio: float
    computed_at: datetime


class TransactionStatusUpdate(BaseModel):
    status: TransactionStatus
    note: str | None = Field(default=None, max_length=400)


TransactionDetail.model_rebuild()
