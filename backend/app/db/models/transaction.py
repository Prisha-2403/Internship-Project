"""Transaction tables: ``transactions``, ``transaction_features``, ``risk_scores``.

The three are one-to-one and deliberately kept apart:

* ``transactions``          - the immutable financial record as ingested.
* ``transaction_features``  - engineered behavioural signals, recomputable.
* ``risk_scores``           - the scoring verdict, recomputable and versioned.

Re-running detection rewrites the latter two without touching the source record.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, CreatedAtMixin, TimestampMixin
from app.db.enums import (
    AnomalyScenario,
    PaymentMethod,
    RiskLevel,
    TransactionStatus,
)

if TYPE_CHECKING:
    from app.db.models.alert import Alert
    from app.db.models.customer import Beneficiary, Customer, Device
    from app.db.models.ops import ModelVersion


class Transaction(Base, TimestampMixin):
    """A single monetary movement under monitoring."""

    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("amount > 0", name="amount_positive"),
        # Feature engineering always walks a customer's history in time order.
        Index("ix_transactions_customer_occurred", "customer_id", "occurred_at"),
        Index("ix_transactions_occurred_at_desc", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Human-facing identifier, e.g. TX-84921
    transaction_ref: Mapped[str] = mapped_column(
        String(32), unique=True, nullable=False, index=True
    )
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True
    )

    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    location_city: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    location_region: Mapped[str] = mapped_column(String(64), nullable=False)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)

    device_id: Mapped[int | None] = mapped_column(
        ForeignKey("devices.id", ondelete="SET NULL"), nullable=True, index=True
    )
    beneficiary_id: Mapped[int | None] = mapped_column(
        ForeignKey("beneficiaries.id", ondelete="SET NULL"), nullable=True, index=True
    )
    payment_method: Mapped[PaymentMethod] = mapped_column(
        Enum(PaymentMethod, name="payment_method"), nullable=False
    )
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="MOBILE_APP")
    status: Mapped[TransactionStatus] = mapped_column(
        Enum(TransactionStatus, name="transaction_status"),
        nullable=False,
        default=TransactionStatus.COMPLETED,
        index=True,
    )

    import_job_id: Mapped[int | None] = mapped_column(
        ForeignKey("import_jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    # Which synthetic scenario produced this row. Demo bookkeeping only - this is
    # NOT a fraud label and must never be presented as ground truth.
    injected_scenario: Mapped[AnomalyScenario | None] = mapped_column(
        Enum(AnomalyScenario, name="anomaly_scenario"), nullable=True, index=True
    )

    customer: Mapped["Customer"] = relationship(back_populates="transactions")
    device: Mapped["Device | None"] = relationship(back_populates="transactions")
    beneficiary: Mapped["Beneficiary | None"] = relationship(back_populates="transactions")
    features: Mapped["TransactionFeature | None"] = relationship(
        back_populates="transaction", cascade="all, delete-orphan", uselist=False
    )
    risk_score: Mapped["RiskScore | None"] = relationship(
        back_populates="transaction", cascade="all, delete-orphan", uselist=False
    )
    alert: Mapped["Alert | None"] = relationship(
        back_populates="transaction", cascade="all, delete-orphan", uselist=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Transaction {self.transaction_ref} {self.amount}>"


class TransactionFeature(Base, CreatedAtMixin):
    """Engineered behavioural features for one transaction.

    Every column is documented in ``app/ml/features.py`` and in the README's
    feature dictionary. All are computed from the customer's history *strictly
    before* ``occurred_at``, so no future information leaks into a score.
    """

    __tablename__ = "transaction_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # --- Amount behaviour ---------------------------------------------------
    amount_value: Mapped[float] = mapped_column(Float, nullable=False)
    customer_avg_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    customer_std_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    amount_zscore: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    amount_vs_customer_avg_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    customer_p99_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # --- Velocity -----------------------------------------------------------
    txn_count_10m: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    txn_count_1h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    txn_count_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    customer_avg_daily_txns: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    velocity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # --- Timing -------------------------------------------------------------
    hour_of_day: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_unusual_hour: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- Novelty ------------------------------------------------------------
    is_new_device: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_new_location: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_new_beneficiary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    device_usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    distance_from_usual_km: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unique_devices_30d: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unique_locations_30d: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # --- Drift --------------------------------------------------------------
    behavior_change_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    amount_trend_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    frequency_trend_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    transaction: Mapped["Transaction"] = relationship(back_populates="features")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<TransactionFeature txn={self.transaction_id}>"


class RiskScore(Base, CreatedAtMixin):
    """The scoring verdict for one transaction.

    ``business_score`` is the analyst-facing 0-100 number and is exactly the sum
    of the points in ``factors`` (clamped to 100). ``ml_anomaly_score`` is the
    normalised Isolation Forest output kept as a *separate* signal - it is an
    anomaly measure, never a fraud probability.
    """

    __tablename__ = "risk_scores"
    __table_args__ = (
        CheckConstraint(
            "business_score >= 0 AND business_score <= 100", name="business_score_range"
        ),
        CheckConstraint(
            "ml_anomaly_score >= 0 AND ml_anomaly_score <= 1", name="ml_anomaly_score_range"
        ),
        Index("ix_risk_scores_level_score", "risk_level", "business_score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    business_score: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    risk_level: Mapped[RiskLevel] = mapped_column(
        Enum(RiskLevel, name="risk_level"), nullable=False, index=True
    )
    rule_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ml_uplift_points: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    ml_anomaly_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    ml_raw_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_ml_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # [{"code","label","points","detail"}] - what the explanation panel renders.
    factors: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    # Highest-scoring factor, denormalised so the monitoring table can filter on
    # detection reason without unnesting JSONB on every row.
    primary_reason: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)

    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    transaction: Mapped["Transaction"] = relationship(back_populates="risk_score")
    model_version: Mapped["ModelVersion | None"] = relationship(back_populates="risk_scores")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<RiskScore txn={self.transaction_id} {self.business_score}/{self.risk_level.value}>"
