"""Isolation Forest wrapper, score normalisation and the synthetic generator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.features import ML_FEATURE_COLUMNS
from app.ml.isolation_forest import (
    MIN_TRAINING_ROWS,
    AnomalyDetector,
    InsufficientTrainingData,
)
from app.ml.normalization import (
    NormalizationBounds,
    normalize,
    threshold_from_bounds,
    to_raw_anomaly,
)
from app.services.synthetic_data import SyntheticDataGenerator


@pytest.fixture(scope="module")
def training_frame() -> pd.DataFrame:
    """A normal population with 60 unmistakable outliers injected."""
    rng = np.random.default_rng(42)
    size = 2000
    frame = pd.DataFrame({column: rng.normal(0, 1, size) for column in ML_FEATURE_COLUMNS})
    outliers = rng.choice(size, 60, replace=False)
    for column in ML_FEATURE_COLUMNS:
        frame.loc[outliers, column] = rng.normal(9, 1, 60)
    frame.attrs["outliers"] = outliers
    return frame


@pytest.fixture(scope="module")
def fitted(training_frame) -> AnomalyDetector:
    return AnomalyDetector(n_estimators=100, contamination=0.03, random_state=42).fit(
        training_frame
    )


class TestNormalization:
    def test_scores_land_inside_zero_to_one(self, fitted, training_frame):
        result = fitted.score(training_frame)
        assert result.anomaly_scores.min() >= 0.0
        assert result.anomaly_scores.max() <= 1.0

    def test_bounds_survive_a_degenerate_distribution(self):
        bounds = NormalizationBounds.fit(np.array([5.0] * 100))
        assert bounds.upper > bounds.lower

    def test_empty_input_is_handled(self):
        bounds = NormalizationBounds.fit(np.array([]))
        assert bounds.lower == 0.0 and bounds.upper == 1.0

    def test_threshold_matches_the_models_own_decision_boundary(self, fitted, training_frame):
        """`is_anomaly` and `score >= threshold` must be the same statement.

        If they diverge, the number shown on screen stops meaning what the model
        actually decided.
        """
        result = fitted.score(training_frame)
        agreement = (result.is_anomaly == (result.anomaly_scores >= fitted.threshold)).mean()
        assert agreement > 0.99

    def test_raw_scores_flip_sign_so_higher_means_more_anomalous(self):
        assert to_raw_anomaly(np.array([-0.5, 0.5]))[0] > to_raw_anomaly(np.array([-0.5, 0.5]))[1]

    def test_normalisation_is_monotonic(self):
        bounds = NormalizationBounds(lower=-1.0, upper=1.0)
        values = normalize(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), bounds)
        assert list(values) == sorted(values)

    def test_threshold_from_bounds_is_within_range(self):
        bounds = NormalizationBounds(lower=-1.0, upper=1.0)
        assert 0.0 <= threshold_from_bounds(bounds) <= 1.0


class TestDetection:
    def test_injected_outliers_score_higher_than_the_population(self, fitted, training_frame):
        result = fitted.score(training_frame)
        outliers = training_frame.attrs["outliers"]
        mask = np.ones(len(training_frame), dtype=bool)
        mask[outliers] = False
        assert result.anomaly_scores[outliers].mean() > result.anomaly_scores[mask].mean() + 0.2

    def test_most_injected_outliers_are_flagged(self, fitted, training_frame):
        result = fitted.score(training_frame)
        outliers = training_frame.attrs["outliers"]
        assert result.is_anomaly[outliers].sum() >= 50

    def test_anomaly_rate_tracks_the_configured_contamination(self, fitted):
        assert fitted.anomaly_rate == pytest.approx(0.03, abs=0.01)

    def test_training_is_deterministic_for_a_fixed_seed(self, training_frame, fitted):
        again = AnomalyDetector(n_estimators=100, contamination=0.03, random_state=42).fit(
            training_frame
        )
        assert np.allclose(
            fitted.score(training_frame).anomaly_scores,
            again.score(training_frame).anomaly_scores,
        )

    def test_new_data_is_scored_on_the_training_scale(self, fitted):
        """Stored scores stay comparable across runs, and the threshold holds."""
        rng = np.random.default_rng(7)
        fresh = pd.DataFrame({c: rng.normal(0, 1, 400) for c in ML_FEATURE_COLUMNS})
        scores = fitted.score(fresh).anomaly_scores
        assert 0.0 <= scores.min() and scores.max() <= 1.0
        assert scores.mean() < 0.5


class TestGuardrails:
    def test_too_little_data_is_refused(self, training_frame):
        with pytest.raises(InsufficientTrainingData):
            AnomalyDetector().fit(training_frame.head(MIN_TRAINING_ROWS - 1))

    def test_scoring_before_fitting_is_refused(self, training_frame):
        with pytest.raises(RuntimeError):
            AnomalyDetector().score(training_frame)

    def test_missing_feature_columns_are_rejected(self, fitted, training_frame):
        with pytest.raises(ValueError, match="missing required columns"):
            fitted.score(training_frame.drop(columns=[ML_FEATURE_COLUMNS[0]]))

    def test_empty_frame_scores_to_an_empty_result(self, fitted, training_frame):
        assert fitted.score(training_frame.head(0)).anomaly_scores.shape == (0,)

    def test_unfitted_detector_refuses_to_save(self):
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(RuntimeError):
                AnomalyDetector().save(Path(directory) / "model.joblib")


class TestPersistence:
    def test_round_trip_preserves_scores_and_threshold(self, fitted, training_frame):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            fitted.save(path)
            restored = AnomalyDetector.load(path)

            assert np.allclose(
                fitted.score(training_frame).anomaly_scores,
                restored.score(training_frame).anomaly_scores,
            )
            assert restored.threshold == pytest.approx(fitted.threshold)
            assert restored.feature_columns == fitted.feature_columns

    def test_params_capture_what_is_needed_to_reproduce(self, fitted):
        params = fitted.params()
        assert "normalization_bounds" in params
        assert params["contamination"] == fitted.contamination
        assert params["random_state"] == fitted.random_state


class TestSyntheticGenerator:
    @pytest.fixture(scope="class")
    def dataset(self):
        return SyntheticDataGenerator(
            customer_count=40, target_transactions=4000, history_days=60, seed=42
        ).generate()

    def test_produces_roughly_the_requested_volume(self, dataset):
        assert 3200 <= len(dataset.transactions) <= 4800

    def test_all_six_scenarios_are_represented(self, dataset):
        assert len(dataset.scenario_counts) == 6

    def test_transaction_references_are_unique(self, dataset):
        refs = [t.transaction_ref for t in dataset.transactions]
        assert len(set(refs)) == len(refs)

    def test_amounts_are_always_positive(self, dataset):
        assert all(t.amount > 0 for t in dataset.transactions)

    def test_output_is_ordered_by_time(self, dataset):
        times = [t.occurred_at for t in dataset.transactions]
        assert times == sorted(times)

    def test_every_referenced_device_and_payee_belongs_to_its_customer(self, dataset):
        """Anomaly scenarios introduce new references; they must be registered."""
        for transaction in dataset.transactions:
            customer = dataset.customers[transaction.customer_index]
            if transaction.device_ref:
                assert transaction.device_ref in {d.device_ref for d in customer.devices}
            if transaction.beneficiary_ref:
                assert transaction.beneficiary_ref in {
                    b.beneficiary_ref for b in customer.beneficiaries
                }

    def test_generation_is_reproducible(self):
        first = SyntheticDataGenerator(
            customer_count=20, target_transactions=1000, history_days=30, seed=7
        ).generate()
        second = SyntheticDataGenerator(
            customer_count=20, target_transactions=1000, history_days=30, seed=7
        ).generate()
        assert len(first.transactions) == len(second.transactions)
        assert [t.transaction_ref for t in first.transactions[:20]] == [
            t.transaction_ref for t in second.transactions[:20]
        ]

    def test_night_time_activity_stays_rare(self, dataset):
        """Ordinary customers keep ordinary hours, so unusual-hour means something."""
        night = sum(1 for t in dataset.transactions if t.occurred_at.hour < 6)
        assert night / len(dataset.transactions) < 0.05
