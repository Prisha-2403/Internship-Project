"""End-to-end detection pipeline.

    load -> clean -> engineer features -> scale -> train -> score ->
    normalise -> combine with business rules -> persist

Two entry points:

* :func:`run_detection` - engineer features and score transactions with the
  active model (or rules only, if none is trained).
* :func:`train_and_detect` - fit a new model first, register it, then score.

Both are safe to re-run: features and risk scores are replaced for the
transactions in scope, never duplicated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models.customer import Beneficiary, Device
from app.db.models.ops import ModelVersion
from app.db.models.transaction import RiskScore, Transaction, TransactionFeature
from app.ml import registry
from app.ml.features import FEATURE_COLUMNS, build_features
from app.ml.isolation_forest import AnomalyDetector, InsufficientTrainingData
from app.services import risk_scoring_service
from app.services.risk_scoring_service import FeatureSnapshot

logger = logging.getLogger("sentinel.ml.pipeline")

# Rows per bulk write. Large enough to be fast, small enough to bound memory.
CHUNK_SIZE = 5000


@dataclass
class DetectionRun:
    """What a detection run did."""

    transactions_scored: int = 0
    features_computed: int = 0
    model_version: str | None = None
    model_trained: bool = False
    alerts_raised: int = 0
    level_counts: dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    used_ml: bool = False

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()

    def as_dict(self) -> dict[str, Any]:
        return {
            "transactions_scored": self.transactions_scored,
            "features_computed": self.features_computed,
            "model_version": self.model_version,
            "model_trained": self.model_trained,
            "alerts_raised": self.alerts_raised,
            "level_counts": self.level_counts,
            "used_ml": self.used_ml,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def load_transaction_frame(
    db: Session, transaction_ids: list[int] | None = None
) -> pd.DataFrame:
    """Read transactions into a frame shaped for feature engineering.

    Even when only some transactions need rescoring, the *whole* history of the
    customers involved is loaded: a transaction's features are defined by what
    came before it, so a partial history would produce different numbers.
    """
    stmt = (
        select(
            Transaction.id.label("transaction_id"),
            Transaction.customer_id,
            Transaction.amount,
            Transaction.occurred_at,
            Transaction.location_city,
            Transaction.latitude,
            Transaction.longitude,
            Device.device_ref,
            Beneficiary.beneficiary_ref,
        )
        .outerjoin(Device, Transaction.device_id == Device.id)
        .outerjoin(Beneficiary, Transaction.beneficiary_id == Beneficiary.id)
    )

    if transaction_ids is not None:
        if not transaction_ids:
            return pd.DataFrame(
                columns=[
                    "transaction_id", "customer_id", "amount", "occurred_at",
                    "location_city", "latitude", "longitude", "device_ref",
                    "beneficiary_ref",
                ]
            )
        customer_ids = select(Transaction.customer_id).where(
            Transaction.id.in_(transaction_ids)
        ).distinct()
        stmt = stmt.where(Transaction.customer_id.in_(customer_ids))

    rows = db.execute(stmt.order_by(Transaction.customer_id, Transaction.occurred_at)).all()
    frame = pd.DataFrame(rows, columns=list(rows[0]._fields) if rows else [
        "transaction_id", "customer_id", "amount", "occurred_at", "location_city",
        "latitude", "longitude", "device_ref", "beneficiary_ref",
    ])
    if not frame.empty:
        frame["amount"] = frame["amount"].astype(float)
    return frame


def persist_features(db: Session, features: pd.DataFrame) -> int:
    """Replace stored features for the transactions in ``features``."""
    if features.empty:
        return 0

    transaction_ids = features["transaction_id"].astype(int).tolist()
    computed_at = datetime.now(UTC)

    for start in range(0, len(transaction_ids), CHUNK_SIZE):
        chunk = transaction_ids[start : start + CHUNK_SIZE]
        db.execute(
            delete(TransactionFeature).where(TransactionFeature.transaction_id.in_(chunk))
        )

    records = features.to_dict(orient="records")
    payload = [
        {
            "transaction_id": int(r["transaction_id"]),
            "computed_at": computed_at,
            **{c: _coerce(c, r[c]) for c in FEATURE_COLUMNS},
        }
        for r in records
    ]
    for start in range(0, len(payload), CHUNK_SIZE):
        db.bulk_insert_mappings(TransactionFeature, payload[start : start + CHUNK_SIZE])
    return len(payload)


def _coerce(column: str, value: Any) -> Any:
    """Convert numpy scalars to plain Python for the driver."""
    if column in {"is_new_device", "is_new_location", "is_new_beneficiary", "is_unusual_hour"}:
        return bool(value)
    if column in {
        "txn_count_10m", "txn_count_1h", "txn_count_24h", "device_usage_count",
        "unique_devices_30d", "unique_locations_30d", "hour_of_day", "day_of_week",
    }:
        return int(value)
    return float(value)


def score_transactions(
    db: Session,
    features: pd.DataFrame,
    detector: AnomalyDetector | None,
    model_version: ModelVersion | None,
) -> tuple[int, dict[str, int]]:
    """Combine ML anomaly scores with business rules and persist the verdicts."""
    if features.empty:
        return 0, {}

    if detector is not None and detector.is_fitted:
        result = detector.score(features)
        anomaly_scores = result.anomaly_scores
        raw_scores = result.raw_scores
        threshold = detector.threshold
    else:
        # No model yet: rules alone still produce a complete, explainable score.
        anomaly_scores = [0.0] * len(features)
        raw_scores = [0.0] * len(features)
        threshold = 1.0

    scored_at = datetime.now(UTC)
    level_counts: dict[str, int] = {}
    payload: list[dict[str, Any]] = []

    records = features.to_dict(orient="records")
    for index, record in enumerate(records):
        snapshot = FeatureSnapshot.from_any(record)
        assessment = risk_scoring_service.score_transaction(
            snapshot,
            ml_anomaly_score=float(anomaly_scores[index]),
            ml_raw_score=float(raw_scores[index]),
            ml_threshold=threshold,
        )
        level_counts[assessment.risk_level.value] = (
            level_counts.get(assessment.risk_level.value, 0) + 1
        )
        payload.append(
            {
                "transaction_id": int(record["transaction_id"]),
                "business_score": assessment.business_score,
                "risk_level": assessment.risk_level,
                "rule_score": assessment.rule_score,
                "ml_uplift_points": assessment.ml_uplift_points,
                "ml_anomaly_score": assessment.ml_anomaly_score,
                "ml_raw_score": assessment.ml_raw_score,
                "is_ml_anomaly": assessment.is_ml_anomaly,
                "factors": assessment.factors_as_dicts(),
                "primary_reason": assessment.primary_reason,
                "model_version_id": model_version.id if model_version else None,
                "scored_at": scored_at,
                "created_at": scored_at,
            }
        )

    transaction_ids = [p["transaction_id"] for p in payload]
    for start in range(0, len(transaction_ids), CHUNK_SIZE):
        db.execute(
            delete(RiskScore).where(
                RiskScore.transaction_id.in_(transaction_ids[start : start + CHUNK_SIZE])
            )
        )
    for start in range(0, len(payload), CHUNK_SIZE):
        db.bulk_insert_mappings(RiskScore, payload[start : start + CHUNK_SIZE])

    return len(payload), level_counts


def run_detection(
    db: Session,
    *,
    transaction_ids: list[int] | None = None,
    raise_alerts: bool = True,
) -> DetectionRun:
    """Engineer features and score transactions using the active model."""
    run = DetectionRun()

    frame = load_transaction_frame(db, transaction_ids)
    if frame.empty:
        run.finished_at = datetime.now(UTC)
        return run

    features = build_features(frame)
    run.features_computed = persist_features(db, features)

    loaded = registry.load_active_detector(db)
    detector, version = loaded if loaded else (None, None)
    run.used_ml = detector is not None
    run.model_version = version.version if version else None

    scored, level_counts = score_transactions(db, features, detector, version)
    run.transactions_scored = scored
    run.level_counts = level_counts

    db.flush()
    if raise_alerts:
        from app.services import alert_service

        run.alerts_raised = alert_service.sync_alerts(
            db, [int(t) for t in features["transaction_id"]]
        )

    db.commit()
    run.finished_at = datetime.now(UTC)
    logger.info("Detection run complete: %s", run.as_dict())
    return run


def train_and_detect(
    db: Session, *, notes: str = "", raise_alerts: bool = True
) -> DetectionRun:
    """Fit a fresh model on all history, register it, then score everything."""
    run = DetectionRun(model_trained=True)

    frame = load_transaction_frame(db)
    if frame.empty:
        run.finished_at = datetime.now(UTC)
        return run

    features = build_features(frame)
    run.features_computed = persist_features(db, features)

    detector = AnomalyDetector(
        n_estimators=settings.ml_n_estimators,
        contamination=settings.ml_contamination,
        random_state=settings.ml_random_state,
    )
    try:
        detector.fit(features)
    except InsufficientTrainingData as exc:
        # Not an error: a young deployment simply scores on rules until it has
        # enough history to isolate anything.
        logger.warning("Skipping model training: %s", exc)
        run.model_trained = False
        scored, level_counts = score_transactions(db, features, None, None)
        run.transactions_scored = scored
        run.level_counts = level_counts
        db.commit()
        run.finished_at = datetime.now(UTC)
        return run

    version = registry.register(db, detector, notes=notes, activate=True)
    run.model_version = version.version
    run.used_ml = True

    scored, level_counts = score_transactions(db, features, detector, version)
    run.transactions_scored = scored
    run.level_counts = level_counts

    db.flush()
    if raise_alerts:
        from app.services import alert_service

        run.alerts_raised = alert_service.sync_alerts(
            db, [int(t) for t in features["transaction_id"]]
        )

    db.commit()
    run.finished_at = datetime.now(UTC)
    logger.info("Training run complete: %s", run.as_dict())
    return run
