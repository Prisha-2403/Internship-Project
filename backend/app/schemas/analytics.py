"""Dashboard, analytics, model-status and import schemas.

Where a number cannot be derived from the data, it is not invented. The model
status schema in particular carries an explicit statement about the absence of
ground-truth-dependent metrics rather than fabricating them.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.db.enums import ImportStatus, RiskLevel
from app.schemas.common import CountByLabel, TimeSeriesPoint

#: Rendered verbatim on the model performance page. Required by the product
#: brief: never fabricate precision, recall, F1 or a confusion matrix.
NO_GROUND_TRUTH_STATEMENT = (
    "Model performance metrics requiring labeled ground truth are unavailable."
)


class DashboardKpis(BaseModel):
    """Headline counters. Every value is a real aggregate over stored data."""

    total_transactions: int
    suspicious_transactions: int = Field(
        description="Transactions scored MEDIUM or above (risk score 30+)."
    )
    high_risk_transactions: int = Field(
        description="Transactions scored HIGH or CRITICAL (risk score 60+)."
    )
    open_investigations: int
    average_risk_score: float
    detection_rate: float = Field(
        description=(
            "Share of scored transactions that reached the alert threshold, as a "
            "percentage. Computed from stored scores, not estimated."
        )
    )
    scored_transactions: int
    total_customers: int
    total_alerts: int
    unread_alerts: int


class DashboardSummary(BaseModel):
    kpis: DashboardKpis
    transaction_volume: list[TimeSeriesPoint]
    risk_distribution: list[CountByLabel]
    suspicious_trend: list[TimeSeriesPoint]
    risk_by_location: list[CountByLabel]
    detection_reasons: list[CountByLabel]
    recent_alerts: list[dict]
    generated_at: datetime
    is_demo_data: bool = Field(
        default=True, description="True when the dataset is synthetic demo data."
    )


class AnalyticsSummary(BaseModel):
    """The analytics page."""

    transaction_volume: list[TimeSeriesPoint]
    risk_trend: list[TimeSeriesPoint] = Field(
        description="Average risk score per bucket."
    )
    high_risk_trend: list[TimeSeriesPoint]
    detection_reasons: list[CountByLabel]
    risk_by_location: list[CountByLabel]
    amount_by_risk_level: list[CountByLabel]
    payment_method_breakdown: list[CountByLabel]
    outcome_analysis: "OutcomeAnalysis"
    generated_at: datetime


class OutcomeAnalysis(BaseModel):
    """False-positive analysis from real analyst decisions on cases.

    ``pending_review`` is alerts with no case yet. These are genuinely
    undetermined and are reported as such rather than folded into either bucket.
    """

    total_alerts: int
    cases_opened: int
    reviewed: int = Field(description="Cases reaching a terminal status.")
    confirmed_suspicious: int
    false_positive: int
    closed_no_action: int
    still_open: int
    pending_review: int
    false_positive_rate: float | None = Field(
        default=None,
        description="False positives as a share of reviewed cases. Null when nothing is reviewed.",
    )


class ModelFeatureInfo(BaseModel):
    name: str
    label: str
    description: str
    unit: str
    used_by_model: bool


class ScenarioCoverage(BaseModel):
    """Detection coverage against injected synthetic scenarios.

    DEMO DATA ONLY. These scenarios are a record of what the generator produced;
    they are not verified fraud outcomes. This is not a model accuracy metric
    and must never be presented as one.
    """

    scenario: str
    label: str
    generated: int
    reached_review: int = Field(description="Scored 30 or above.")
    reached_high_risk: int = Field(description="Scored 60 or above.")


class ModelStatus(BaseModel):
    """The model performance page."""

    has_model: bool
    version: str | None = None
    algorithm: str | None = None
    trained_at: datetime | None = None
    training_record_count: int = 0
    feature_count: int = 0
    contamination: float | None = None
    n_estimators: int | None = None
    anomaly_threshold: float | None = None
    anomaly_rate: float | None = None
    scored_transactions: int = 0
    ml_flagged_transactions: int = 0
    features: list[ModelFeatureInfo] = Field(default_factory=list)

    # Honesty block - see NO_GROUND_TRUTH_STATEMENT.
    supervised_metrics_available: bool = Field(
        default=False,
        description="True only if labelled ground truth exists. It does not here.",
    )
    supervised_metrics_statement: str = NO_GROUND_TRUTH_STATEMENT

    scenario_coverage: list[ScenarioCoverage] = Field(default_factory=list)
    scenario_coverage_disclaimer: str = (
        "Synthetic scenario coverage is measured against transactions this platform "
        "generated for demonstration. It describes generated demo data only and is "
        "not a measure of fraud detection accuracy."
    )
    history: list[dict] = Field(default_factory=list)


class ImportSummary(BaseModel):
    """The import result panel. Every counter is a real tally."""

    id: int
    filename: str
    status: ImportStatus
    records_received: int
    records_valid: int
    records_invalid: int
    records_duplicate: int
    records_processed: int
    records_flagged: int
    errors: list[dict] = Field(default_factory=list)
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ReportRequest(BaseModel):
    date_from: datetime | None = None
    date_to: datetime | None = None
    risk_level: RiskLevel | None = None


class RiskSummaryReport(BaseModel):
    """A generated risk summary over a period."""

    title: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    total_transactions: int
    total_value: float
    suspicious_transactions: int
    high_risk_transactions: int
    average_risk_score: float
    alerts_raised: int
    cases_opened: int
    cases_closed: int
    risk_distribution: list[CountByLabel]
    top_detection_reasons: list[CountByLabel]
    top_locations: list[CountByLabel]
    highest_risk_transactions: list[dict]
    is_demo_data: bool = True


AnalyticsSummary.model_rebuild()
