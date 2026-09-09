"""Operational tables: ``model_versions`` and ``import_jobs``."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.enums import ImportStatus

if TYPE_CHECKING:
    from app.db.models.transaction import RiskScore
    from app.db.models.user import User


class ModelVersion(Base, TimestampMixin):
    """A trained Isolation Forest, with the facts needed to reproduce it.

    ``params`` carries the normalisation bounds captured at fit time so scores
    produced later sit on the same 0-1 scale as the training run.

    Deliberately absent: precision, recall, F1 and confusion matrices. Those
    require labelled ground truth which this platform does not have, and the
    model page states as much rather than inventing numbers.
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Human-facing identifier, e.g. if-20260820-1
    version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False, default="IsolationForest")

    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    training_record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    feature_names: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)

    contamination: Mapped[float] = mapped_column(Float, nullable=False, default=0.03)
    n_estimators: Mapped[int] = mapped_column(Integer, nullable=False, default=200)
    random_state: Mapped[int] = mapped_column(Integer, nullable=False, default=42)
    # Normalised anomaly score at or above which a transaction is an ML anomaly.
    anomaly_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Observed share of training rows the model marked anomalous.
    anomaly_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    artifact_path: Mapped[str | None] = mapped_column(String(400), nullable=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)

    risk_scores: Mapped[list["RiskScore"]] = relationship(back_populates="model_version")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ModelVersion {self.version} active={self.is_active}>"


class ImportJob(Base, TimestampMixin):
    """One CSV upload and the outcome of validating it.

    The counters are exactly what the import summary panel shows; each is a real
    tally produced by ``import_service``, not an estimate.
    """

    __tablename__ = "import_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    uploaded_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[ImportStatus] = mapped_column(
        Enum(ImportStatus, name="import_status"),
        nullable=False,
        default=ImportStatus.PENDING,
        index=True,
    )

    records_received: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_valid: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_invalid: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_duplicate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_flagged: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # [{"row": 12, "field": "amount", "message": "..."}] - capped by the service
    # so one bad file cannot bloat the table.
    error_report: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    uploader: Mapped["User | None"] = relationship(lazy="joined")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ImportJob {self.filename} {self.status.value}>"
