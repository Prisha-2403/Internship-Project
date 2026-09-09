"""Feature engineering.

The property that matters most here is the absence of lookahead: a feature must
depend only on what happened before its transaction. If that breaks, every score
computed during a backfill is quietly different from the score the transaction
would have received live, and the whole system becomes untrustworthy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ml.features import (
    FEATURE_COLUMNS,
    MIN_HISTORY_FOR_NOVELTY,
    ML_FEATURE_COLUMNS,
    build_features,
    haversine_km,
)

BASE = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)


def make_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def steady_history(
    customer_id: int = 1,
    count: int = 30,
    amount: float = 1000.0,
    start_id: int = 1,
) -> list[dict]:
    """A customer transacting the same amount, daily, at the same hour."""
    return [
        {
            "transaction_id": start_id + index,
            "customer_id": customer_id,
            "amount": amount,
            "occurred_at": BASE + timedelta(days=index),
            "location_city": "Delhi",
            "device_ref": "DEV-1",
            "beneficiary_ref": "BEN-1",
            "latitude": 28.61,
            "longitude": 77.21,
        }
        for index in range(count)
    ]


class TestOutputShape:
    def test_empty_input_returns_empty_frame(self):
        result = build_features(pd.DataFrame())
        assert result.empty

    def test_every_documented_column_is_produced(self):
        result = build_features(make_frame(steady_history()))
        assert set(result.columns) == {"transaction_id", *FEATURE_COLUMNS}

    def test_row_count_is_preserved(self):
        result = build_features(make_frame(steady_history(count=17)))
        assert len(result) == 17

    def test_no_nan_or_infinite_values(self):
        """Isolation Forest cannot consume NaN, and a NaN score is meaningless."""
        result = build_features(make_frame(steady_history()))
        numeric = result.select_dtypes(include=[np.number])
        assert numeric.isna().sum().sum() == 0
        assert not np.isinf(numeric.to_numpy()).any()

    def test_single_customer_frame_is_handled(self):
        """One group is the edge case pandas groupby handles differently."""
        result = build_features(make_frame(steady_history(count=8)))
        assert len(result) == 8
        assert result["txn_count_24h"].iloc[0] == 1


class TestNoLookahead:
    def test_features_do_not_change_when_later_rows_are_removed(self):
        rows = steady_history(count=30)
        rows.append(
            {
                "transaction_id": 31,
                "customer_id": 1,
                "amount": 50_000.0,
                "occurred_at": BASE + timedelta(days=30),
                "location_city": "Mumbai",
                "device_ref": "DEV-9",
                "beneficiary_ref": "BEN-9",
                "latitude": 19.07,
                "longitude": 72.87,
            }
        )
        full = build_features(make_frame(rows)).set_index("transaction_id")
        truncated = build_features(make_frame(rows[:20])).set_index("transaction_id")

        for transaction_id in (1, 5, 10, 19):
            for column in FEATURE_COLUMNS:
                a, b = full.loc[transaction_id, column], truncated.loc[transaction_id, column]
                if isinstance(a, (float, np.floating)):
                    assert float(a) == pytest.approx(float(b)), f"{transaction_id}.{column}"
                else:
                    assert a == b, f"{transaction_id}.{column}"

    def test_baseline_excludes_the_transaction_being_described(self):
        """A large transaction must not inflate the average it is measured against."""
        rows = steady_history(count=10, amount=1000.0)
        rows.append(
            {
                "transaction_id": 11,
                "customer_id": 1,
                "amount": 100_000.0,
                "occurred_at": BASE + timedelta(days=10),
                "location_city": "Delhi",
                "device_ref": "DEV-1",
                "beneficiary_ref": "BEN-1",
                "latitude": 28.61,
                "longitude": 77.21,
            }
        )
        result = build_features(make_frame(rows)).set_index("transaction_id")
        assert result.loc[11, "customer_avg_amount"] == pytest.approx(1000.0)
        assert result.loc[11, "amount_vs_customer_avg_ratio"] == pytest.approx(100.0)


class TestColdStart:
    def test_novelty_is_suppressed_until_some_history_exists(self):
        """A customer's first device is their baseline, not an anomaly."""
        result = build_features(make_frame(steady_history(count=10))).set_index(
            "transaction_id"
        )
        for transaction_id in range(1, MIN_HISTORY_FOR_NOVELTY + 1):
            assert not result.loc[transaction_id, "is_new_device"]
            assert not result.loc[transaction_id, "is_new_location"]
            assert not result.loc[transaction_id, "is_new_beneficiary"]

    def test_deviation_features_stay_neutral_on_thin_history(self):
        result = build_features(make_frame(steady_history(count=10))).set_index(
            "transaction_id"
        )
        assert result.loc[1, "amount_zscore"] == 0.0
        assert result.loc[1, "amount_vs_customer_avg_ratio"] == 1.0


class TestNoveltyDetection:
    @pytest.fixture
    def anomalous(self) -> pd.DataFrame:
        rows = steady_history(count=30)
        rows.append(
            {
                "transaction_id": 31,
                "customer_id": 1,
                "amount": 50_000.0,
                # 03:00 - well outside the 10:00 pattern
                "occurred_at": BASE + timedelta(days=30, hours=-7),
                "location_city": "Mumbai",
                "device_ref": "DEV-9",
                "beneficiary_ref": "BEN-9",
                "latitude": 19.07,
                "longitude": 72.87,
            }
        )
        return build_features(make_frame(rows)).set_index("transaction_id")

    def test_new_device_is_detected(self, anomalous):
        assert bool(anomalous.loc[31, "is_new_device"]) is True
        assert int(anomalous.loc[31, "device_usage_count"]) == 0

    def test_new_location_is_detected(self, anomalous):
        assert bool(anomalous.loc[31, "is_new_location"]) is True

    def test_new_beneficiary_is_detected(self, anomalous):
        assert bool(anomalous.loc[31, "is_new_beneficiary"]) is True

    def test_unusual_hour_is_detected(self, anomalous):
        assert bool(anomalous.loc[31, "is_unusual_hour"]) is True

    def test_distance_from_usual_area(self, anomalous):
        # Delhi to Mumbai is roughly 1,150 km.
        assert 1100 < float(anomalous.loc[31, "distance_from_usual_km"]) < 1200

    def test_zero_variance_history_yields_a_zero_zscore_not_infinity(self, anomalous):
        assert float(anomalous.loc[31, "amount_zscore"]) == 0.0
        assert float(anomalous.loc[31, "amount_vs_customer_avg_ratio"]) == pytest.approx(50.0)


class TestVelocity:
    def test_window_counts_include_the_current_transaction(self):
        rows = steady_history(count=25, amount=2000.0)
        burst_ids = []
        for step in range(5):
            burst_ids.append(100 + step)
            rows.append(
                {
                    "transaction_id": 100 + step,
                    "customer_id": 1,
                    "amount": 2100.0,
                    "occurred_at": BASE + timedelta(days=25, minutes=step),
                    "location_city": "Delhi",
                    "device_ref": "DEV-1",
                    "beneficiary_ref": "BEN-1",
                    "latitude": 28.61,
                    "longitude": 77.21,
                }
            )
        result = build_features(make_frame(rows)).set_index("transaction_id")
        assert int(result.loc[burst_ids[0], "txn_count_10m"]) == 1
        assert int(result.loc[burst_ids[-1], "txn_count_10m"]) == 5
        assert float(result.loc[burst_ids[-1], "velocity_score"]) > 0

    def test_isolated_transactions_have_a_count_of_one(self):
        result = build_features(make_frame(steady_history(count=5))).set_index(
            "transaction_id"
        )
        assert int(result.loc[1, "txn_count_10m"]) == 1
        assert int(result.loc[1, "txn_count_1h"]) == 1


class TestMultipleCustomers:
    def test_customers_do_not_contaminate_each_other(self):
        rows = steady_history(customer_id=1, count=10, amount=1000.0, start_id=1)
        rows += steady_history(customer_id=2, count=10, amount=90_000.0, start_id=100)
        result = build_features(make_frame(rows)).set_index("transaction_id")

        # Customer 2's large amounts must not raise customer 1's baseline.
        assert result.loc[10, "customer_avg_amount"] == pytest.approx(1000.0)
        assert result.loc[109, "customer_avg_amount"] == pytest.approx(90_000.0)


class TestHaversine:
    def test_known_distance(self):
        # Delhi to Mumbai, roughly 1,150 km.
        assert 1100 < haversine_km(28.6139, 77.2090, 19.0760, 72.8777) < 1200

    def test_zero_distance(self):
        assert haversine_km(28.6, 77.2, 28.6, 77.2) == pytest.approx(0.0)

    def test_missing_coordinates_return_zero(self):
        assert haversine_km(float("nan"), 77.2, 19.0, 72.8) == 0.0


class TestModelInputs:
    def test_model_columns_are_a_subset_of_produced_features(self):
        assert set(ML_FEATURE_COLUMNS).issubset(set(FEATURE_COLUMNS))

    def test_model_sees_behaviour_not_identity(self):
        """Raw amounts and identity flags are deliberately withheld.

        The model should learn from behavioural shape; the boolean novelty flags
        are already covered explicitly by the business rules.
        """
        for column in ("amount_value", "customer_avg_amount", "is_new_device"):
            assert column not in ML_FEATURE_COLUMNS
