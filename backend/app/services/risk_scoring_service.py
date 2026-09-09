"""The business risk scoring engine.

Deliberately free of database and HTTP concerns: it takes a feature snapshot and
an optional ML anomaly score, and returns an assessment. That makes it directly
testable and keeps scoring identical whether it runs in a batch backfill, during
a CSV import, or behind an API call.

The contract that matters:

    business_score == sum(factor.points for factor in factors)

Every point an analyst sees is attributed to a named rule that actually fired.
Nothing is added anonymously, and no score is invented.

Raw signal strength can exceed 100 when many rules fire at once. Rather than
silently clipping - which would leave the displayed factors summing to more than
the displayed score - contributions are scaled proportionally onto the 0-100
scale using largest-remainder rounding, and the assessment records that it
happened along with the raw total.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.db.enums import DETECTION_REASON_LABELS, DetectionReason, RiskLevel

# --- Rule thresholds ------------------------------------------------------
# Amount deviation, by z-score and by plain multiple of the customer's average.
# The stronger of the two applies, so a customer with near-constant spending
# (zero variance, undefined z-score) is still scored on the multiple.
AMOUNT_ZSCORE_BANDS: tuple[tuple[float, int], ...] = ((6.0, 28), (4.0, 22), (3.0, 16), (2.0, 10))
AMOUNT_RATIO_BANDS: tuple[tuple[float, int], ...] = ((10.0, 28), (5.0, 22), (3.0, 16), (2.0, 10))

# Beyond anything the customer has previously done.
ABSOLUTE_P99_MULTIPLE = 1.5
ABSOLUTE_MIN_RATIO = 3.0
ABSOLUTE_POINTS = 10

NEW_DEVICE_POINTS = 20
RARELY_USED_DEVICE_POINTS = 12
RARELY_USED_DEVICE_MAX_USES = 3

NEW_LOCATION_POINTS = 15
DISTANT_LOCATION_BONUS = 5
DISTANT_LOCATION_KM = 500.0

NEW_BENEFICIARY_POINTS = 15
UNUSUAL_TIME_POINTS = 12

BURST_10M_COUNT = 3
BURST_10M_POINTS = 14
BURST_1H_COUNT = 6
BURST_1H_POINTS = 10
VELOCITY_SCORE_MIN = 2.0
VELOCITY_POINTS = 8

BEHAVIOUR_BANDS: tuple[tuple[float, int], ...] = ((0.7, 15), (0.5, 10), (0.3, 6))

MAX_SCORE = 100


@dataclass(frozen=True)
class RiskFactor:
    """One rule that fired, with the points it contributed."""

    code: DetectionReason
    label: str
    points: int
    detail: str
    raw_points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "label": self.label,
            "points": self.points,
            "detail": self.detail,
            "rawPoints": self.raw_points,
        }


@dataclass
class RiskAssessment:
    """The verdict for one transaction."""

    business_score: int
    risk_level: RiskLevel
    rule_score: int
    ml_uplift_points: int
    ml_anomaly_score: float
    ml_raw_score: float
    is_ml_anomaly: bool
    factors: list[RiskFactor] = field(default_factory=list)
    raw_total: int = 0
    was_scaled: bool = False

    @property
    def primary_reason(self) -> str | None:
        """The highest-scoring factor - what the monitoring table filters on."""
        return self.factors[0].code.value if self.factors else None

    def factors_as_dicts(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.factors]

    def summary(self) -> str:
        """One-line explanation used for alert text."""
        if not self.factors:
            return "No risk signals were triggered for this transaction."
        if len(self.factors) == 1:
            return f"{self.factors[0].label} detected."
        leading = ", ".join(f.label.lower() for f in self.factors[:3])
        if len(self.factors) > 3:
            return f"Multiple behavioural anomalies detected: {leading} and {len(self.factors) - 3} more."
        return f"Multiple behavioural anomalies detected: {leading}."


@dataclass(frozen=True)
class FeatureSnapshot:
    """The feature values the rules read.

    Decouples scoring from the ORM so the engine can be driven from a database
    row, a pandas record, or a literal dict in a test.
    """

    amount_value: float = 0.0
    customer_avg_amount: float = 0.0
    customer_std_amount: float = 0.0
    amount_zscore: float = 0.0
    amount_vs_customer_avg_ratio: float = 1.0
    customer_p99_amount: float = 0.0
    txn_count_10m: int = 0
    txn_count_1h: int = 0
    txn_count_24h: int = 0
    customer_avg_daily_txns: float = 0.0
    velocity_score: float = 0.0
    hour_of_day: int = 0
    day_of_week: int = 0
    is_unusual_hour: bool = False
    is_new_device: bool = False
    is_new_location: bool = False
    is_new_beneficiary: bool = False
    device_usage_count: int = 0
    distance_from_usual_km: float = 0.0
    unique_devices_30d: int = 0
    unique_locations_30d: int = 0
    behavior_change_score: float = 0.0
    amount_trend_ratio: float = 1.0
    frequency_trend_ratio: float = 1.0

    @classmethod
    def from_any(cls, source: Mapping[str, Any] | Any) -> "FeatureSnapshot":
        """Build from a mapping or any object exposing the feature attributes."""
        getter = (
            (lambda k, d: source.get(k, d))
            if isinstance(source, Mapping)
            else (lambda k, d: getattr(source, k, d))
        )
        kwargs: dict[str, Any] = {}
        for name, spec in cls.__dataclass_fields__.items():
            value = getter(name, spec.default)
            if value is None:
                value = spec.default
            kwargs[name] = value
        return cls(**kwargs)


def _band(value: float, bands: tuple[tuple[float, int], ...]) -> int:
    """Points for the first band whose threshold ``value`` meets."""
    for threshold, points in bands:
        if value >= threshold:
            return points
    return 0


def _rule_amount_deviation(f: FeatureSnapshot) -> RiskFactor | None:
    """Amount far outside what this customer normally transacts."""
    by_zscore = _band(f.amount_zscore, AMOUNT_ZSCORE_BANDS)
    by_ratio = _band(f.amount_vs_customer_avg_ratio, AMOUNT_RATIO_BANDS)
    points = max(by_zscore, by_ratio)
    if points == 0:
        return None

    if by_ratio >= by_zscore:
        detail = (
            f"Amount is {f.amount_vs_customer_avg_ratio:.1f}x this customer's "
            f"average of {f.customer_avg_amount:,.0f}."
        )
    else:
        detail = (
            f"Amount is {f.amount_zscore:.1f} standard deviations above this "
            f"customer's average of {f.customer_avg_amount:,.0f}."
        )
    return RiskFactor(
        DetectionReason.AMOUNT_DEVIATION,
        DETECTION_REASON_LABELS[DetectionReason.AMOUNT_DEVIATION],
        points,
        detail,
        points,
    )


def _rule_amount_absolute(f: FeatureSnapshot) -> RiskFactor | None:
    """Amount beyond anything in the customer's own history."""
    if f.customer_p99_amount <= 0:
        return None
    if f.amount_value <= f.customer_p99_amount * ABSOLUTE_P99_MULTIPLE:
        return None
    if f.amount_vs_customer_avg_ratio < ABSOLUTE_MIN_RATIO:
        return None
    return RiskFactor(
        DetectionReason.AMOUNT_ABSOLUTE,
        DETECTION_REASON_LABELS[DetectionReason.AMOUNT_ABSOLUTE],
        ABSOLUTE_POINTS,
        f"Amount exceeds this customer's historical 99th percentile of "
        f"{f.customer_p99_amount:,.0f}.",
        ABSOLUTE_POINTS,
    )


def _rule_new_device(f: FeatureSnapshot) -> RiskFactor | None:
    """Transaction from a device the customer has not established."""
    if f.is_new_device:
        return RiskFactor(
            DetectionReason.NEW_DEVICE,
            DETECTION_REASON_LABELS[DetectionReason.NEW_DEVICE],
            NEW_DEVICE_POINTS,
            "This device has not been seen on this customer's account before.",
            NEW_DEVICE_POINTS,
        )
    if 0 < f.device_usage_count < RARELY_USED_DEVICE_MAX_USES:
        return RiskFactor(
            DetectionReason.NEW_DEVICE,
            DETECTION_REASON_LABELS[DetectionReason.NEW_DEVICE],
            RARELY_USED_DEVICE_POINTS,
            f"Device has only been used {f.device_usage_count} time(s) previously.",
            RARELY_USED_DEVICE_POINTS,
        )
    return None


def _rule_new_location(f: FeatureSnapshot) -> RiskFactor | None:
    """Transaction from a city the customer has not used before."""
    if not f.is_new_location:
        return None
    points = NEW_LOCATION_POINTS
    detail = "Transaction originated from a location not previously used by this customer."
    if f.distance_from_usual_km > DISTANT_LOCATION_KM:
        points += DISTANT_LOCATION_BONUS
        detail = (
            f"Location is {f.distance_from_usual_km:,.0f} km from this customer's "
            f"usual area and has not been used before."
        )
    return RiskFactor(
        DetectionReason.NEW_LOCATION,
        DETECTION_REASON_LABELS[DetectionReason.NEW_LOCATION],
        points,
        detail,
        points,
    )


def _rule_new_beneficiary(f: FeatureSnapshot) -> RiskFactor | None:
    """First payment to this beneficiary."""
    if not f.is_new_beneficiary:
        return None
    return RiskFactor(
        DetectionReason.NEW_BENEFICIARY,
        DETECTION_REASON_LABELS[DetectionReason.NEW_BENEFICIARY],
        NEW_BENEFICIARY_POINTS,
        "This is the first transfer to this beneficiary from this account.",
        NEW_BENEFICIARY_POINTS,
    )


def _rule_unusual_time(f: FeatureSnapshot) -> RiskFactor | None:
    """Transaction outside the customer's established active hours."""
    if not f.is_unusual_hour:
        return None
    return RiskFactor(
        DetectionReason.UNUSUAL_TIME,
        DETECTION_REASON_LABELS[DetectionReason.UNUSUAL_TIME],
        UNUSUAL_TIME_POINTS,
        f"Transaction occurred at {f.hour_of_day:02d}:00, outside this customer's "
        f"usual activity hours.",
        UNUSUAL_TIME_POINTS,
    )


def _rule_high_frequency(f: FeatureSnapshot) -> RiskFactor | None:
    """Unusually rapid sequence of transactions."""
    if f.txn_count_10m >= BURST_10M_COUNT:
        points, detail = (
            BURST_10M_POINTS,
            f"{f.txn_count_10m} transactions in a 10-minute window.",
        )
    elif f.txn_count_1h >= BURST_1H_COUNT:
        points, detail = BURST_1H_POINTS, f"{f.txn_count_1h} transactions within one hour."
    elif f.velocity_score >= VELOCITY_SCORE_MIN:
        points, detail = (
            VELOCITY_POINTS,
            f"24-hour transaction count is {f.velocity_score + 1:.1f}x this "
            f"customer's normal daily rate.",
        )
    else:
        return None
    return RiskFactor(
        DetectionReason.HIGH_FREQUENCY,
        DETECTION_REASON_LABELS[DetectionReason.HIGH_FREQUENCY],
        points,
        detail,
        points,
    )


def _rule_behaviour_change(f: FeatureSnapshot) -> RiskFactor | None:
    """Sustained drift from the customer's established baseline."""
    points = _band(f.behavior_change_score, BEHAVIOUR_BANDS)
    if points == 0:
        return None
    parts: list[str] = []
    if f.amount_trend_ratio > 1.2:
        parts.append(f"average amount up {(f.amount_trend_ratio - 1) * 100:.0f}%")
    if f.frequency_trend_ratio > 1.2:
        parts.append(f"frequency up {(f.frequency_trend_ratio - 1) * 100:.0f}%")
    detail = (
        f"Recent behaviour differs from baseline ({'; '.join(parts)})."
        if parts
        else "Recent activity differs from this customer's established baseline."
    )
    return RiskFactor(
        DetectionReason.BEHAVIOR_CHANGE,
        DETECTION_REASON_LABELS[DetectionReason.BEHAVIOR_CHANGE],
        points,
        detail,
        points,
    )


#: Applied in order; the resulting factor list is then sorted by points.
RULES = (
    _rule_amount_deviation,
    _rule_amount_absolute,
    _rule_new_device,
    _rule_new_location,
    _rule_new_beneficiary,
    _rule_unusual_time,
    _rule_high_frequency,
    _rule_behaviour_change,
)


def ml_uplift(anomaly_score: float, threshold: float, max_points: int | None = None) -> int:
    """Points contributed by the Isolation Forest.

    The model contributes only above its own anomaly threshold, and its
    contribution is bounded so a single unexplainable signal can never dominate
    a score an analyst has to justify. This is an anomaly measure, not a
    probability of fraud.
    """
    cap = settings.ml_max_uplift_points if max_points is None else max_points
    if anomaly_score < threshold or threshold >= 1.0:
        return 0
    proportion = (anomaly_score - threshold) / (1.0 - threshold)
    return int(round(min(max(proportion, 0.0), 1.0) * cap))


def _rule_ml_anomaly(anomaly_score: float, points: int) -> RiskFactor:
    return RiskFactor(
        DetectionReason.ML_ANOMALY,
        DETECTION_REASON_LABELS[DetectionReason.ML_ANOMALY],
        points,
        f"Isolation Forest flagged this transaction as anomalous relative to "
        f"the wider population (anomaly score {anomaly_score:.2f}).",
        points,
    )


def classify(score: int) -> RiskLevel:
    """Map a 0-100 score onto its risk band."""
    if score >= settings.risk_level_critical_min:
        return RiskLevel.CRITICAL
    if score >= settings.risk_level_high_min:
        return RiskLevel.HIGH
    if score >= settings.risk_level_medium_min:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _scale_to_cap(factors: list[RiskFactor], raw_total: int) -> list[RiskFactor]:
    """Scale contributions onto the 0-100 scale, preserving their exact sum.

    Uses largest-remainder rounding so the scaled points add up to exactly
    ``MAX_SCORE`` rather than drifting by a point or two.
    """
    exact = [f.raw_points * MAX_SCORE / raw_total for f in factors]
    floors = [int(v) for v in exact]
    shortfall = MAX_SCORE - sum(floors)
    # Hand the remaining points to the largest fractional parts.
    order = sorted(range(len(factors)), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in order[:shortfall]:
        floors[i] += 1
    return [
        RiskFactor(f.code, f.label, points, f.detail, f.raw_points)
        for f, points in zip(factors, floors)
    ]


def score_transaction(
    features: Mapping[str, Any] | FeatureSnapshot | Any,
    *,
    ml_anomaly_score: float = 0.0,
    ml_raw_score: float = 0.0,
    ml_threshold: float = 1.0,
) -> RiskAssessment:
    """Score one transaction and explain the result.

    ``ml_threshold >= 1.0`` (the default) disables the ML factor entirely, which
    is what happens before any model has been trained. Business rules still
    produce a complete, explainable score in that state.
    """
    snapshot = (
        features
        if isinstance(features, FeatureSnapshot)
        else FeatureSnapshot.from_any(features)
    )

    factors: list[RiskFactor] = []
    for rule in RULES:
        factor = rule(snapshot)
        if factor is not None:
            factors.append(factor)

    rule_score = sum(f.points for f in factors)

    uplift = ml_uplift(ml_anomaly_score, ml_threshold)
    is_ml_anomaly = ml_threshold < 1.0 and ml_anomaly_score >= ml_threshold
    if uplift > 0:
        factors.append(_rule_ml_anomaly(ml_anomaly_score, uplift))

    factors.sort(key=lambda f: f.points, reverse=True)

    raw_total = sum(f.raw_points for f in factors)
    was_scaled = raw_total > MAX_SCORE
    if was_scaled:
        factors = _scale_to_cap(factors, raw_total)

    business_score = sum(f.points for f in factors)
    # The invariant this whole module exists to guarantee.
    assert 0 <= business_score <= MAX_SCORE, "risk score outside 0-100"

    return RiskAssessment(
        business_score=business_score,
        risk_level=classify(business_score),
        rule_score=rule_score,
        ml_uplift_points=uplift,
        ml_anomaly_score=float(ml_anomaly_score),
        ml_raw_score=float(ml_raw_score),
        is_ml_anomaly=is_ml_anomaly,
        factors=factors,
        raw_total=raw_total,
        was_scaled=was_scaled,
    )
