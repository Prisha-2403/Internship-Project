"""Model status, detection runs and anomaly review.

The status payload is deliberately conservative. It reports what is genuinely
known - version, training size, features, threshold, observed anomaly rate - and
states plainly that precision, recall and confusion matrices are unavailable,
because this platform has no labelled fraud outcomes to compute them from.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import case as sql_case, func, select

from app.auth.dependencies import CurrentUser, DbSession, require_permission
from app.auth.permissions import Permission
from app.core.config import settings
from app.core.middleware import get_client_ip, get_user_agent
from app.db.enums import (
    ANOMALY_SCENARIO_LABELS,
    AnomalyScenario,
    AuditAction,
)
from app.db.models.transaction import RiskScore, Transaction
from app.ml import pipeline, registry
from app.ml.features import FEATURE_DEFINITIONS, ML_FEATURE_COLUMNS
from app.schemas.analytics import ModelFeatureInfo, ModelStatus, ScenarioCoverage
from app.schemas.common import Page
from app.schemas.transaction import TransactionListItem
from app.repositories import transaction_repository as repo
from app.repositories.transaction_repository import TransactionFilters
from app.services import audit_service

router = APIRouter(tags=["Anomaly Detection & Model"])


@router.get(
    "/model/status",
    response_model=ModelStatus,
    summary="Model version, training facts and honest metric availability",
)
def model_status(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_MODEL_STATUS))],
) -> ModelStatus:
    """Everything genuinely known about the active model.

    Supervised metrics are reported as unavailable rather than fabricated.
    """
    version = registry.get_active_version(db)

    scored = int(db.execute(select(func.count(RiskScore.id))).scalar() or 0)
    ml_flagged = int(
        db.execute(
            select(func.count(RiskScore.id)).where(RiskScore.is_ml_anomaly.is_(True))
        ).scalar()
        or 0
    )

    features = [
        ModelFeatureInfo(
            name=name,
            label=meta["label"],
            description=meta["description"],
            unit=meta["unit"],
            used_by_model=name in ML_FEATURE_COLUMNS,
        )
        for name, meta in FEATURE_DEFINITIONS.items()
    ]

    history = [
        {
            "version": v.version,
            "trainedAt": v.trained_at.isoformat(),
            "trainingRecords": v.training_record_count,
            "featureCount": v.feature_count,
            "anomalyRate": round(v.anomaly_rate, 4),
            "isActive": v.is_active,
        }
        for v in registry.list_versions(db, limit=10)
    ]

    return ModelStatus(
        has_model=version is not None,
        version=version.version if version else None,
        algorithm=version.algorithm if version else None,
        trained_at=version.trained_at if version else None,
        training_record_count=version.training_record_count if version else 0,
        feature_count=version.feature_count if version else 0,
        contamination=version.contamination if version else None,
        n_estimators=version.n_estimators if version else None,
        anomaly_threshold=round(version.anomaly_threshold, 4) if version else None,
        anomaly_rate=round(version.anomaly_rate, 4) if version else None,
        scored_transactions=scored,
        ml_flagged_transactions=ml_flagged,
        features=features,
        # No labelled ground truth exists, so these stay false/unavailable.
        supervised_metrics_available=False,
        scenario_coverage=_scenario_coverage(db),
        history=history,
    )


def _scenario_coverage(db: DbSession) -> list[ScenarioCoverage]:
    """Coverage against injected synthetic scenarios.

    DEMO DATA ONLY. These are a record of what the generator produced, not
    verified fraud outcomes, and this is not a model accuracy measure. It is
    reported separately from - and never as a substitute for - the supervised
    metrics that genuinely require labelled data.
    """
    def _count_at_or_above(threshold: int):
        return func.coalesce(
            func.sum(sql_case((RiskScore.business_score >= threshold, 1), else_=0)), 0
        )

    rows = db.execute(
        select(
            Transaction.injected_scenario,
            func.count(Transaction.id),
            _count_at_or_above(settings.risk_level_medium_min),
            _count_at_or_above(settings.risk_level_high_min),
        )
        .outerjoin(RiskScore, RiskScore.transaction_id == Transaction.id)
        .where(Transaction.injected_scenario.isnot(None))
        .group_by(Transaction.injected_scenario)
    ).all()

    coverage: list[ScenarioCoverage] = []
    for scenario, generated, review, high in rows:
        coverage.append(
            ScenarioCoverage(
                scenario=scenario.value,
                label=ANOMALY_SCENARIO_LABELS.get(AnomalyScenario(scenario), scenario.value),
                generated=int(generated or 0),
                reached_review=int(review or 0),
                reached_high_risk=int(high or 0),
            )
        )
    coverage.sort(key=lambda c: c.scenario)
    return coverage


@router.get(
    "/anomalies",
    response_model=Page[TransactionListItem],
    summary="Transactions ranked by anomaly signal",
)
def anomalies(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_TRANSACTIONS))],
    ml_only: bool = Query(
        default=False, description="Restrict to transactions the model itself flagged."
    ),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Page[TransactionListItem]:
    """The anomaly detection worklist, highest risk first."""
    from app.api.v1.transactions import _to_list_item

    filters = TransactionFilters(min_risk_score=settings.risk_level_medium_min)
    total = repo.count(db, filters)
    rows = repo.list_transactions(
        db, filters, page=page, page_size=page_size, sort_by="risk_score", direction="desc"
    )
    if ml_only:
        rows = [r for r in rows if r[1] is not None and r[1].is_ml_anomaly]
    return Page.build([_to_list_item(r) for r in rows], total, page, page_size)


@router.post(
    "/detection/run",
    summary="Re-run detection over all transactions with the active model",
)
def run_detection(
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.RUN_DETECTION))],
) -> dict:
    """Recompute features and rescore.

    Existing analyst decisions on alerts and cases are preserved.
    """
    run = pipeline.run_detection(db)
    audit_service.record_and_commit(
        db,
        action=AuditAction.DETECTION_RUN,
        actor=user,
        resource_type="detection",
        resource_id="manual",
        description=(
            f"Detection run scored {run.transactions_scored:,} transactions "
            f"and raised {run.alerts_raised} alert(s)."
        ),
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        new_state=run.as_dict(),
    )
    return run.as_dict()


@router.post(
    "/model/train",
    summary="Train a new model version and rescore all transactions",
)
def train_model(
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.TRAIN_MODEL))],
    notes: str = Query(default="", max_length=400),
) -> dict:
    """Fit a fresh Isolation Forest, register it as active, then rescore."""
    run = pipeline.train_and_detect(db, notes=notes)
    audit_service.record_and_commit(
        db,
        action=AuditAction.MODEL_TRAINED,
        actor=user,
        resource_type="model_version",
        resource_id=run.model_version or "none",
        description=(
            f"Trained model {run.model_version or '(insufficient data)'} on "
            f"{run.transactions_scored:,} transactions."
        ),
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        new_state=run.as_dict(),
    )
    return run.as_dict()
