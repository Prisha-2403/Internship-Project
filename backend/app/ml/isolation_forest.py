"""Isolation Forest detector: scaling, fitting, scoring and persistence.

Wraps scikit-learn so the rest of the application deals in normalised 0-1
anomaly scores and never touches estimator internals. Scaler and model travel
together in one artifact, because scoring with a scaler that does not match the
fitted model produces confidently wrong numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.ml.features import ML_FEATURE_COLUMNS
from app.ml.normalization import (
    NormalizationBounds,
    normalize,
    threshold_from_bounds,
    to_raw_anomaly,
)

logger = logging.getLogger("sentinel.ml")

# Below this, an Isolation Forest has nothing meaningful to isolate.
MIN_TRAINING_ROWS = 50


class InsufficientTrainingData(Exception):
    """Raised when there are too few rows to fit a model."""


@dataclass
class ScoringResult:
    """Per-transaction model output."""

    anomaly_scores: np.ndarray  # normalised 0-1, higher is more anomalous
    raw_scores: np.ndarray  # flipped decision_function
    is_anomaly: np.ndarray  # bool, model's own inlier/outlier call


class AnomalyDetector:
    """A fitted scaler + Isolation Forest with its normalisation scale."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        contamination: float = 0.03,
        random_state: int = 42,
        feature_columns: tuple[str, ...] = ML_FEATURE_COLUMNS,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.feature_columns = tuple(feature_columns)
        self.scaler: StandardScaler | None = None
        self.model: IsolationForest | None = None
        self.bounds: NormalizationBounds | None = None
        self.threshold: float = 1.0
        self.anomaly_rate: float = 0.0
        self.training_rows: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.model is not None and self.scaler is not None and self.bounds is not None

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Select and order the model's columns, coercing to finite floats."""
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Feature frame is missing required columns: {missing}")
        matrix = features.loc[:, list(self.feature_columns)].to_numpy(dtype=float, copy=True)
        # Isolation Forest cannot consume NaN/inf; features.py should never emit
        # them, so this is a guard rather than an expected code path.
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, features: pd.DataFrame) -> "AnomalyDetector":
        """Fit the scaler, the forest, and the normalisation scale."""
        if len(features) < MIN_TRAINING_ROWS:
            raise InsufficientTrainingData(
                f"At least {MIN_TRAINING_ROWS} transactions are required to train a model; "
                f"received {len(features)}."
            )

        matrix = self._matrix(features)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(matrix)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(scaled)

        raw = to_raw_anomaly(self.model.decision_function(scaled))
        self.bounds = NormalizationBounds.fit(raw)
        self.threshold = threshold_from_bounds(self.bounds)
        self.training_rows = len(features)
        self.anomaly_rate = float((self.model.predict(scaled) == -1).mean())

        logger.info(
            "Trained IsolationForest on %d rows, %d features; anomaly rate %.3f, threshold %.3f",
            self.training_rows,
            len(self.feature_columns),
            self.anomaly_rate,
            self.threshold,
        )
        return self

    def score(self, features: pd.DataFrame) -> ScoringResult:
        """Score transactions with the fitted model."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")
        assert self.scaler is not None and self.model is not None and self.bounds is not None

        if features.empty:
            empty_f = np.empty(0, dtype=float)
            return ScoringResult(empty_f, empty_f, np.empty(0, dtype=bool))

        scaled = self.scaler.transform(self._matrix(features))
        decision = self.model.decision_function(scaled)
        raw = to_raw_anomaly(decision)
        return ScoringResult(
            anomaly_scores=normalize(raw, self.bounds),
            raw_scores=raw,
            is_anomaly=(decision < 0),
        )

    def save(self, path: Path) -> Path:
        """Persist scaler, model and bounds as one artifact."""
        if not self.is_fitted:
            raise RuntimeError("Refusing to save an unfitted detector.")
        assert self.bounds is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": 1,
                "scaler": self.scaler,
                "model": self.model,
                "bounds": self.bounds.to_dict(),
                "threshold": self.threshold,
                "feature_columns": list(self.feature_columns),
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "random_state": self.random_state,
                "anomaly_rate": self.anomaly_rate,
                "training_rows": self.training_rows,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        """Restore a detector saved by :meth:`save`."""
        payload = joblib.load(path)
        detector = cls(
            n_estimators=payload["n_estimators"],
            contamination=payload["contamination"],
            random_state=payload["random_state"],
            feature_columns=tuple(payload["feature_columns"]),
        )
        detector.scaler = payload["scaler"]
        detector.model = payload["model"]
        detector.bounds = NormalizationBounds.from_dict(payload["bounds"])
        detector.threshold = float(payload["threshold"])
        detector.anomaly_rate = float(payload.get("anomaly_rate", 0.0))
        detector.training_rows = int(payload.get("training_rows", 0))
        return detector

    def params(self) -> dict[str, Any]:
        """Reproducibility metadata stored on the model version row."""
        assert self.bounds is not None
        return {
            "normalization_bounds": self.bounds.to_dict(),
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "feature_columns": list(self.feature_columns),
            "scaler": "StandardScaler",
        }
