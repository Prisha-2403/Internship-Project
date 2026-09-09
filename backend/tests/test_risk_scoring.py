"""Risk scoring engine.

The contract these tests defend: the score an analyst is shown is exactly the
sum of the factors they are shown alongside it. If that ever stops holding, the
explanation panel is lying, which is worse than having no explanation at all.
"""

from __future__ import annotations

import random

import pytest

from app.db.enums import DetectionReason, RiskLevel
from app.services.risk_scoring_service import (
    MAX_SCORE,
    FeatureSnapshot,
    classify,
    ml_uplift,
    score_transaction,
)


def codes(assessment) -> set[str]:
    return {factor.code.value for factor in assessment.factors}


class TestScoreInvariant:
    def test_clean_transaction_scores_zero(self):
        assessment = score_transaction({})
        assert assessment.business_score == 0
        assert assessment.risk_level is RiskLevel.LOW
        assert assessment.factors == []

    def test_factors_always_sum_to_the_displayed_score(self):
        assessment = score_transaction(
            {
                "amount_zscore": 4.5,
                "amount_vs_customer_avg_ratio": 12.0,
                "customer_avg_amount": 8500,
                "is_new_device": True,
                "is_new_location": True,
                "is_new_beneficiary": True,
                "is_unusual_hour": True,
                "txn_count_10m": 4,
                "behavior_change_score": 0.8,
            }
        )
        assert sum(f.points for f in assessment.factors) == assessment.business_score

    @pytest.mark.parametrize("seed", range(25))
    def test_invariant_holds_for_random_inputs(self, seed):
        rng = random.Random(seed)
        features = {
            "amount_value": rng.uniform(0, 500_000),
            "customer_avg_amount": rng.uniform(0, 50_000),
            "customer_std_amount": rng.uniform(0, 20_000),
            "amount_zscore": rng.uniform(-20, 20),
            "amount_vs_customer_avg_ratio": rng.uniform(0, 100),
            "customer_p99_amount": rng.uniform(0, 80_000),
            "txn_count_10m": rng.randint(0, 12),
            "txn_count_1h": rng.randint(0, 30),
            "velocity_score": rng.uniform(0, 20),
            "is_unusual_hour": rng.random() < 0.5,
            "is_new_device": rng.random() < 0.5,
            "is_new_location": rng.random() < 0.5,
            "is_new_beneficiary": rng.random() < 0.5,
            "device_usage_count": rng.randint(0, 20),
            "distance_from_usual_km": rng.uniform(0, 3000),
            "behavior_change_score": rng.uniform(0, 1),
            "amount_trend_ratio": rng.uniform(0, 50),
            "frequency_trend_ratio": rng.uniform(0, 50),
        }
        assessment = score_transaction(
            features, ml_anomaly_score=rng.random(), ml_threshold=rng.choice([0.5, 0.7, 1.0])
        )
        assert 0 <= assessment.business_score <= MAX_SCORE
        assert sum(f.points for f in assessment.factors) == assessment.business_score
        assert all(f.points >= 0 for f in assessment.factors)

    def test_score_is_capped_at_100_with_proportional_scaling(self):
        """Every rule firing at maximum still produces a valid, attributable score."""
        assessment = score_transaction(
            {
                "amount_zscore": 20,
                "amount_vs_customer_avg_ratio": 90,
                "customer_avg_amount": 1000,
                "customer_p99_amount": 1200,
                "amount_value": 500_000,
                "is_new_device": True,
                "is_new_location": True,
                "distance_from_usual_km": 2000,
                "is_new_beneficiary": True,
                "is_unusual_hour": True,
                "txn_count_10m": 10,
                "behavior_change_score": 1.0,
            },
            ml_anomaly_score=1.0,
            ml_threshold=0.5,
        )
        assert assessment.business_score == MAX_SCORE
        assert assessment.was_scaled is True
        assert assessment.raw_total > MAX_SCORE
        assert sum(f.points for f in assessment.factors) == MAX_SCORE


class TestRiskBands:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0, RiskLevel.LOW),
            (29, RiskLevel.LOW),
            (30, RiskLevel.MEDIUM),
            (59, RiskLevel.MEDIUM),
            (60, RiskLevel.HIGH),
            (79, RiskLevel.HIGH),
            (80, RiskLevel.CRITICAL),
            (100, RiskLevel.CRITICAL),
        ],
    )
    def test_band_boundaries(self, score, expected):
        assert classify(score) is expected


class TestIndividualRules:
    def test_amount_deviation_fires_on_zscore(self):
        assessment = score_transaction(
            {"amount_zscore": 4.2, "customer_avg_amount": 8500, "amount_vs_customer_avg_ratio": 1.1}
        )
        assert DetectionReason.AMOUNT_DEVIATION.value in codes(assessment)

    def test_amount_deviation_falls_back_to_ratio_for_zero_variance_customers(self):
        """A customer with near-constant spending has an undefined z-score.

        Without the ratio fallback a 50x transaction on such an account would
        score zero, which is exactly the case most worth catching.
        """
        assessment = score_transaction(
            {
                "amount_zscore": 0.0,
                "customer_std_amount": 0.0,
                "amount_vs_customer_avg_ratio": 50.0,
                "customer_avg_amount": 1000,
            }
        )
        assert DetectionReason.AMOUNT_DEVIATION.value in codes(assessment)
        assert assessment.business_score > 0

    def test_new_device_scores_higher_than_a_rarely_used_one(self):
        brand_new = score_transaction({"is_new_device": True})
        rarely_used = score_transaction({"is_new_device": False, "device_usage_count": 1})
        assert brand_new.business_score > rarely_used.business_score > 0

    def test_distant_new_location_scores_above_a_nearby_one(self):
        near = score_transaction({"is_new_location": True, "distance_from_usual_km": 20})
        far = score_transaction({"is_new_location": True, "distance_from_usual_km": 1500})
        assert far.business_score > near.business_score

    def test_known_location_does_not_fire(self):
        assessment = score_transaction(
            {"is_new_location": False, "distance_from_usual_km": 1500}
        )
        assert DetectionReason.NEW_LOCATION.value not in codes(assessment)

    def test_velocity_burst_fires(self):
        assessment = score_transaction({"txn_count_10m": 5})
        assert DetectionReason.HIGH_FREQUENCY.value in codes(assessment)

    def test_behaviour_change_bands_are_ordered(self):
        mild = score_transaction({"behavior_change_score": 0.35})
        severe = score_transaction({"behavior_change_score": 0.9})
        assert severe.business_score > mild.business_score > 0

    def test_every_factor_carries_a_human_explanation(self):
        assessment = score_transaction(
            {
                "amount_zscore": 5,
                "customer_avg_amount": 5000,
                "is_new_device": True,
                "is_new_beneficiary": True,
                "is_unusual_hour": True,
                "hour_of_day": 3,
            }
        )
        assert assessment.factors
        for factor in assessment.factors:
            assert factor.label and len(factor.label) > 2
            assert factor.detail and len(factor.detail) > 10
            assert factor.points > 0

    def test_factors_are_ordered_by_contribution(self):
        assessment = score_transaction(
            {
                "amount_zscore": 6.5,
                "customer_avg_amount": 5000,
                "is_new_device": True,
                "is_unusual_hour": True,
            }
        )
        points = [f.points for f in assessment.factors]
        assert points == sorted(points, reverse=True)


class TestMlContribution:
    def test_model_does_not_contribute_below_its_threshold(self):
        assert ml_uplift(0.50, 0.60) == 0
        assert ml_uplift(0.60, 0.60) == 0

    def test_contribution_is_bounded(self):
        """An unexplainable signal must never dominate a score.

        The cap is what keeps a high-risk verdict defensible to the customer it
        affects: most of the score always traces to a stated rule.
        """
        assert ml_uplift(1.0, 0.60) == 15
        assert all(ml_uplift(score / 100, 0.6) <= 15 for score in range(101))

    def test_threshold_of_one_disables_the_model_entirely(self):
        """Before any model is trained, rules alone must still produce a score."""
        assert ml_uplift(0.99, 1.0) == 0
        assessment = score_transaction(
            {"is_new_device": True}, ml_anomaly_score=0.99, ml_threshold=1.0
        )
        assert DetectionReason.ML_ANOMALY.value not in codes(assessment)
        assert assessment.business_score > 0
        assert assessment.is_ml_anomaly is False

    def test_model_factor_is_labelled_and_attributed(self):
        assessment = score_transaction(
            {"is_new_device": True}, ml_anomaly_score=0.95, ml_threshold=0.6
        )
        model_factors = [
            f for f in assessment.factors if f.code is DetectionReason.ML_ANOMALY
        ]
        assert len(model_factors) == 1
        assert "anomal" in model_factors[0].detail.lower()
        # Never described as a probability of fraud.
        assert "fraud" not in model_factors[0].detail.lower()

    def test_ml_score_is_recorded_separately_from_the_business_score(self):
        assessment = score_transaction(
            {"is_new_device": True}, ml_anomaly_score=0.82, ml_threshold=0.6
        )
        assert assessment.ml_anomaly_score == pytest.approx(0.82)
        assert assessment.business_score != assessment.ml_anomaly_score
        assert assessment.rule_score == 20


class TestFeatureSnapshot:
    def test_builds_from_a_mapping(self):
        snapshot = FeatureSnapshot.from_any({"amount_zscore": 3.5, "is_new_device": True})
        assert snapshot.amount_zscore == 3.5
        assert snapshot.is_new_device is True

    def test_missing_values_fall_back_to_neutral_defaults(self):
        snapshot = FeatureSnapshot.from_any({})
        assert snapshot.amount_vs_customer_avg_ratio == 1.0
        assert snapshot.amount_zscore == 0.0

    def test_none_values_do_not_leak_through(self):
        snapshot = FeatureSnapshot.from_any({"amount_zscore": None, "txn_count_10m": None})
        assert snapshot.amount_zscore == 0.0
        assert snapshot.txn_count_10m == 0

    def test_builds_from_an_object(self):
        class Row:
            amount_zscore = 2.5
            is_new_beneficiary = True

        snapshot = FeatureSnapshot.from_any(Row())
        assert snapshot.amount_zscore == 2.5
        assert snapshot.is_new_beneficiary is True


class TestAssessmentSummary:
    def test_summary_avoids_asserting_fraud(self):
        assessment = score_transaction(
            {
                "is_new_device": True,
                "is_new_location": True,
                "is_new_beneficiary": True,
                "is_unusual_hour": True,
            }
        )
        summary = assessment.summary().lower()
        assert "fraud" not in summary
        assert "anomal" in summary or "detected" in summary

    def test_primary_reason_is_the_leading_factor(self):
        assessment = score_transaction(
            {"amount_zscore": 6.5, "customer_avg_amount": 5000, "is_unusual_hour": True}
        )
        assert assessment.primary_reason == assessment.factors[0].code.value
