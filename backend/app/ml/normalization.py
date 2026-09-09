"""Turning Isolation Forest output into a stable 0-1 anomaly score.

scikit-learn's ``decision_function`` is unbounded and centred on zero, with
negative values meaning "anomalous". That is awkward to display and impossible
to compare across training runs, so scores are mapped onto 0-1 using percentile
bounds captured at fit time and stored with the model version.

Reusing the training bounds is the point: a transaction scored next week lands
on the same scale as one scored during training, so a stored score stays
meaningful and the threshold does not silently move.

The result is an *anomaly* score - how unlike the wider population a
transaction looks. It is not a probability that fraud occurred, and nothing in
this module should ever present it as one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Percentile bounds. Trimming the extremes stops a handful of outliers from
# compressing every ordinary transaction into the bottom of the range.
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0

# Guards against a degenerate spread when every training row scores alike.
MIN_SPREAD = 1e-9


@dataclass(frozen=True)
class NormalizationBounds:
    """The scale captured at training time."""

    lower: float
    upper: float

    def to_dict(self) -> dict[str, float]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizationBounds":
        return cls(lower=float(data["lower"]), upper=float(data["upper"]))

    @classmethod
    def fit(cls, raw_scores: np.ndarray) -> "NormalizationBounds":
        """Derive bounds from the training population's raw scores."""
        if raw_scores.size == 0:
            return cls(lower=0.0, upper=1.0)
        lower = float(np.percentile(raw_scores, LOWER_PERCENTILE))
        upper = float(np.percentile(raw_scores, UPPER_PERCENTILE))
        if upper - lower < MIN_SPREAD:
            upper = lower + 1.0
        return cls(lower=lower, upper=upper)


def to_raw_anomaly(decision_values: np.ndarray) -> np.ndarray:
    """Flip ``decision_function`` so larger means more anomalous."""
    return -np.asarray(decision_values, dtype=float)


def normalize(raw_scores: np.ndarray, bounds: NormalizationBounds) -> np.ndarray:
    """Map raw anomaly scores onto 0-1 using fixed training-time bounds."""
    raw = np.asarray(raw_scores, dtype=float)
    spread = max(bounds.upper - bounds.lower, MIN_SPREAD)
    return np.clip((raw - bounds.lower) / spread, 0.0, 1.0)


def threshold_from_bounds(bounds: NormalizationBounds) -> float:
    """The normalised score matching the model's own decision boundary.

    ``decision_function == 0`` is where scikit-learn separates inlier from
    outlier, which is raw score 0 after the flip. Expressing that same boundary
    on the 0-1 scale keeps "the model considers this anomalous" and "the score
    is at or above the threshold" the same statement.
    """
    return float(normalize(np.array([0.0]), bounds)[0])
