"""Model version registry.

Records what was trained, when, on how much data and with which features, and
keeps exactly one version marked active. The artifact lives on disk; the row in
``model_versions`` is the durable record of its provenance.

What is deliberately *not* stored: precision, recall, F1 or a confusion matrix.
Those require labelled ground truth about which transactions were genuinely
fraudulent, which this platform does not have. Rather than invent them, the
model page states their absence.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.ml.isolation_forest import AnomalyDetector
from app.db.models.ops import ModelVersion

logger = logging.getLogger("sentinel.ml.registry")


def next_version_label(db: Session) -> str:
    """Sequential label for today, e.g. ``if-20260820-2``."""
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    prefix = f"if-{stamp}"
    existing = db.execute(
        select(ModelVersion.version).where(ModelVersion.version.like(f"{prefix}%"))
    ).scalars().all()
    return f"{prefix}-{len(existing) + 1}"


def artifact_filename(version: str) -> str:
    return f"model_{version}.joblib"


def artifact_path_for(version: str) -> Path:
    return settings.artifact_path / artifact_filename(version)


def resolve_artifact(stored: str | None) -> Path | None:
    """Turn a stored artifact reference into a path on this machine.

    Only the filename is persisted, so a model trained on a developer machine
    still resolves on a server where the artifacts directory lives somewhere
    else. Absolute paths written by older versions are still honoured, falling
    back to the filename when that path does not exist here.
    """
    if not stored:
        return None
    candidate = Path(stored)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        candidate = Path(candidate.name)
    return settings.artifact_path / candidate.name


def register(
    db: Session,
    detector: AnomalyDetector,
    *,
    version: str | None = None,
    notes: str = "",
    activate: bool = True,
) -> ModelVersion:
    """Persist a trained detector and its metadata.

    Writes the artifact to disk first: a row pointing at a missing file is worse
    than a file with no row, since the former breaks scoring at request time.
    """
    label = version or next_version_label(db)
    path = artifact_path_for(label)
    detector.save(path)

    assert detector.bounds is not None
    record = ModelVersion(
        version=label,
        algorithm="IsolationForest",
        trained_at=datetime.now(UTC),
        training_record_count=detector.training_rows,
        feature_count=len(detector.feature_columns),
        feature_names=list(detector.feature_columns),
        contamination=detector.contamination,
        n_estimators=detector.n_estimators,
        random_state=detector.random_state,
        anomaly_threshold=detector.threshold,
        anomaly_rate=detector.anomaly_rate,
        # Filename only - see resolve_artifact for why.
        artifact_path=artifact_filename(label),
        params=detector.params(),
        notes=notes,
        is_active=False,
    )
    db.add(record)
    db.flush()

    if activate:
        activate_version(db, record)

    logger.info("Registered model version %s (%d rows)", label, detector.training_rows)
    return record


def activate_version(db: Session, version: ModelVersion) -> None:
    """Make ``version`` the active model, deactivating any other."""
    db.execute(update(ModelVersion).where(ModelVersion.is_active.is_(True)).values(is_active=False))
    version.is_active = True
    db.flush()


def get_active_version(db: Session) -> ModelVersion | None:
    return db.execute(
        select(ModelVersion).where(ModelVersion.is_active.is_(True))
    ).scalar_one_or_none()


def load_active_detector(db: Session) -> tuple[AnomalyDetector, ModelVersion] | None:
    """Load the active model from disk.

    Returns ``None`` when no model has been trained yet or its artifact has gone
    missing. Callers fall back to rules-only scoring, which still produces a
    complete and explainable score.
    """
    version = get_active_version(db)
    if version is None:
        return None

    path = resolve_artifact(version.artifact_path)
    if path is None or not path.exists():
        logger.warning(
            "Active model %s references a missing artifact at %s; "
            "falling back to rules-only scoring.",
            version.version,
            path,
        )
        return None

    try:
        return AnomalyDetector.load(path), version
    except Exception:
        logger.exception("Failed to load model artifact %s; falling back to rules-only.", path)
        return None


def list_versions(db: Session, limit: int = 20) -> list[ModelVersion]:
    return list(
        db.execute(
            select(ModelVersion).order_by(ModelVersion.trained_at.desc()).limit(limit)
        ).scalars()
    )
