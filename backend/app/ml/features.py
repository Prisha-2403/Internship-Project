"""Behavioural feature engineering.

Every feature is derived from a customer's history *strictly before* the
transaction being described. Nothing looks ahead, so a score computed during a
backfill is the same score the transaction would have received live.

Two passes produce the frame:

1. A vectorised pandas pass for statistics that expanding/rolling windows
   express directly (means, standard deviations, time-window counts).
2. A single ordered pass per customer for the stateful signals - novelty,
   30-day distinct counts and the hour-of-day distribution - each of which
   needs memory of what has been seen so far.

``FEATURE_DEFINITIONS`` documents every column and is surfaced verbatim on the
model page, so the dictionary in the UI can never drift from the code.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

# A customer needs some history before their "normal" means anything. Below
# this, deviation features stay neutral instead of firing on thin evidence.
MIN_HISTORY_FOR_BASELINE = 5

# Novelty needs a little history too: the first device a customer ever uses is
# their baseline, not an anomaly.
MIN_HISTORY_FOR_NOVELTY = 3

# An hour counts as unusual when it holds less than this share of a customer's
# prior activity (and they have enough history for the share to be meaningful).
UNUSUAL_HOUR_SHARE = 0.05
MIN_HISTORY_FOR_HOUR = 20

RECENT_WINDOW_DAYS = 7
DISTINCT_WINDOW_DAYS = 30

EARTH_RADIUS_KM = 6371.0


#: name -> (human label, description, unit)
FEATURE_DEFINITIONS: dict[str, dict[str, str]] = {
    "amount_value": {
        "label": "Transaction amount",
        "description": "The transaction value in the account currency.",
        "unit": "currency",
    },
    "customer_avg_amount": {
        "label": "Customer average amount",
        "description": "Mean of the customer's prior transaction amounts (expanding, excludes this transaction).",
        "unit": "currency",
    },
    "customer_std_amount": {
        "label": "Customer amount deviation",
        "description": "Standard deviation of the customer's prior amounts. Zero until enough history exists.",
        "unit": "currency",
    },
    "amount_zscore": {
        "label": "Amount z-score",
        "description": "How many standard deviations this amount sits above the customer's prior mean. Zero when history is too thin to judge.",
        "unit": "sigma",
    },
    "amount_vs_customer_avg_ratio": {
        "label": "Amount vs average",
        "description": "This amount divided by the customer's prior average. 1.0 means typical.",
        "unit": "ratio",
    },
    "customer_p99_amount": {
        "label": "Customer 99th percentile",
        "description": "The customer's prior 99th-percentile amount, used to spot values beyond anything they have done before.",
        "unit": "currency",
    },
    "txn_count_10m": {
        "label": "Transactions in 10 minutes",
        "description": "Count in the 10 minutes up to and including this transaction.",
        "unit": "count",
    },
    "txn_count_1h": {
        "label": "Transactions in 1 hour",
        "description": "Count in the hour up to and including this transaction.",
        "unit": "count",
    },
    "txn_count_24h": {
        "label": "Transactions in 24 hours",
        "description": "Count in the 24 hours up to and including this transaction.",
        "unit": "count",
    },
    "customer_avg_daily_txns": {
        "label": "Average daily transactions",
        "description": "The customer's historical transactions per active day.",
        "unit": "count/day",
    },
    "velocity_score": {
        "label": "Velocity score",
        "description": "Short-window transaction rate relative to the customer's own norm. 0 is normal pace.",
        "unit": "score",
    },
    "hour_of_day": {
        "label": "Hour of day",
        "description": "Local hour the transaction occurred, 0-23.",
        "unit": "hour",
    },
    "day_of_week": {
        "label": "Day of week",
        "description": "0 is Monday through 6 for Sunday.",
        "unit": "index",
    },
    "is_unusual_hour": {
        "label": "Unusual hour",
        "description": f"True when under {UNUSUAL_HOUR_SHARE:.0%} of the customer's prior activity fell in this hour, and they have at least {MIN_HISTORY_FOR_HOUR} prior transactions.",
        "unit": "boolean",
    },
    "is_new_device": {
        "label": "New device",
        "description": "True when this device has not been seen for this customer before.",
        "unit": "boolean",
    },
    "is_new_location": {
        "label": "New location",
        "description": "True when the customer has not transacted from this city before.",
        "unit": "boolean",
    },
    "is_new_beneficiary": {
        "label": "New beneficiary",
        "description": "True when the customer has not paid this beneficiary before.",
        "unit": "boolean",
    },
    "device_usage_count": {
        "label": "Device usage count",
        "description": "How many times the customer has previously used this device.",
        "unit": "count",
    },
    "distance_from_usual_km": {
        "label": "Distance from usual area",
        "description": "Great-circle distance from the centroid of the customer's prior transaction locations.",
        "unit": "km",
    },
    "unique_devices_30d": {
        "label": "Distinct devices (30d)",
        "description": "Distinct devices the customer used in the preceding 30 days.",
        "unit": "count",
    },
    "unique_locations_30d": {
        "label": "Distinct locations (30d)",
        "description": "Distinct cities the customer transacted from in the preceding 30 days.",
        "unit": "count",
    },
    "amount_trend_ratio": {
        "label": "Amount trend",
        "description": f"Average amount over the last {RECENT_WINDOW_DAYS} days divided by the customer's longer-run average.",
        "unit": "ratio",
    },
    "frequency_trend_ratio": {
        "label": "Frequency trend",
        "description": f"Transactions per day over the last {RECENT_WINDOW_DAYS} days divided by the customer's longer-run rate.",
        "unit": "ratio",
    },
    "behavior_change_score": {
        "label": "Behavioural change",
        "description": "Combined drift in spend size and frequency versus the customer's established baseline, 0 to 1.",
        "unit": "score",
    },
}

#: Columns fed to the Isolation Forest. Identifiers, raw currency values and
#: booleans that duplicate a rule are excluded; the model sees behaviour, not
#: identity.
ML_FEATURE_COLUMNS: tuple[str, ...] = (
    "amount_zscore",
    "amount_vs_customer_avg_ratio",
    "txn_count_10m",
    "txn_count_1h",
    "txn_count_24h",
    "velocity_score",
    "hour_of_day",
    "day_of_week",
    "distance_from_usual_km",
    "unique_devices_30d",
    "unique_locations_30d",
    "behavior_change_score",
    "amount_trend_ratio",
    "frequency_trend_ratio",
)

#: Columns written to ``transaction_features``.
FEATURE_COLUMNS: tuple[str, ...] = tuple(FEATURE_DEFINITIONS)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (lat1, lon1, lat2, lon2)):
        return 0.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _window_counts(times_ns: np.ndarray, window_ns: int) -> np.ndarray:
    """Count rows falling in ``(t - window, t]`` for each ascending timestamp.

    ``searchsorted`` on the already-sorted timestamps makes this exact and
    O(n log n), and - unlike ``groupby.apply`` with a rolling window - it
    behaves identically whether the frame holds one customer or ten thousand.
    """
    if times_ns.size == 0:
        return np.empty(0, dtype=np.int64)
    left = np.searchsorted(times_ns, times_ns - window_ns, side="right")
    return (np.arange(times_ns.size, dtype=np.int64) - left + 1).astype(np.int64)


def _stateful_pass(group: pd.DataFrame) -> dict[str, list[Any]]:
    """Single ordered walk computing signals that need memory of the past.

    Handles novelty, 30-day distinct counts, the hour-of-day distribution and
    the location centroid, all of which depend on everything seen so far and so
    cannot be expressed as a plain rolling window.
    """
    n = len(group)
    out: dict[str, list[Any]] = {
        "is_new_device": [False] * n,
        "is_new_location": [False] * n,
        "is_new_beneficiary": [False] * n,
        "device_usage_count": [0] * n,
        "unique_devices_30d": [0] * n,
        "unique_locations_30d": [0] * n,
        "is_unusual_hour": [False] * n,
        "distance_from_usual_km": [0.0] * n,
    }

    seen_devices: dict[str, int] = {}
    seen_locations: set[str] = set()
    seen_beneficiaries: set[str] = set()
    hour_counts = np.zeros(24, dtype=np.int64)
    # (timestamp, device, city) for the trailing 30-day distinct counts.
    window: deque[tuple[pd.Timestamp, str, str]] = deque()
    lat_sum = lon_sum = 0.0
    coord_count = 0

    times = group["occurred_at"].to_numpy()
    devices = group["device_ref"].fillna("").to_numpy()
    cities = group["location_city"].fillna("").to_numpy()
    beneficiaries = group["beneficiary_ref"].fillna("").to_numpy()
    hours = group["hour_of_day"].to_numpy()
    lats = group["latitude"].to_numpy()
    lons = group["longitude"].to_numpy()

    for i in range(n):
        prior = i
        ts = pd.Timestamp(times[i])
        device, city, beneficiary = devices[i], cities[i], beneficiaries[i]

        # --- Novelty, judged against history only -------------------------
        if prior >= MIN_HISTORY_FOR_NOVELTY:
            out["is_new_device"][i] = bool(device) and device not in seen_devices
            out["is_new_location"][i] = bool(city) and city not in seen_locations
            out["is_new_beneficiary"][i] = (
                bool(beneficiary) and beneficiary not in seen_beneficiaries
            )
        out["device_usage_count"][i] = seen_devices.get(device, 0)

        # --- Unusual hour, share of prior activity ------------------------
        if prior >= MIN_HISTORY_FOR_HOUR:
            share = hour_counts[hours[i]] / prior
            out["is_unusual_hour"][i] = bool(share < UNUSUAL_HOUR_SHARE)

        # --- Distance from the centroid of prior locations ----------------
        if coord_count > 0 and not pd.isna(lats[i]) and not pd.isna(lons[i]):
            out["distance_from_usual_km"][i] = haversine_km(
                float(lats[i]), float(lons[i]), lat_sum / coord_count, lon_sum / coord_count
            )

        # --- Trailing 30-day distinct counts (window excludes this row) ----
        cutoff = ts - pd.Timedelta(days=DISTINCT_WINDOW_DAYS)
        while window and window[0][0] < cutoff:
            window.popleft()
        out["unique_devices_30d"][i] = len({d for _, d, _ in window if d})
        out["unique_locations_30d"][i] = len({c for _, _, c in window if c})

        # --- Fold this row into the running state -------------------------
        if device:
            seen_devices[device] = seen_devices.get(device, 0) + 1
        if city:
            seen_locations.add(city)
        if beneficiary:
            seen_beneficiaries.add(beneficiary)
        hour_counts[hours[i]] += 1
        window.append((ts, device, city))
        if not pd.isna(lats[i]) and not pd.isna(lons[i]):
            lat_sum += float(lats[i])
            lon_sum += float(lons[i])
            coord_count += 1

    return out


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute the full feature frame.

    ``transactions`` must contain: ``transaction_id``, ``customer_id``,
    ``amount``, ``occurred_at``, ``location_city``, ``device_ref``,
    ``beneficiary_ref``, ``latitude``, ``longitude``.

    Returns one row per input transaction, indexed the same way, with every
    column in :data:`FEATURE_COLUMNS`.
    """
    if transactions.empty:
        return pd.DataFrame(columns=["transaction_id", *FEATURE_COLUMNS])

    df = transactions.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)
    df = df.sort_values(["customer_id", "occurred_at"], kind="mergesort").reset_index(drop=True)

    df["hour_of_day"] = df["occurred_at"].dt.hour.astype(int)
    df["day_of_week"] = df["occurred_at"].dt.dayofweek.astype(int)
    df["amount_value"] = df["amount"]

    grouped = df.groupby("customer_id", sort=False)["amount"]
    # shift(1) so the current transaction never contributes to its own baseline.
    prior = grouped.shift(1)
    prior_by_customer = prior.groupby(df["customer_id"], sort=False)

    df["prior_count"] = grouped.cumcount()
    df["customer_avg_amount"] = prior_by_customer.expanding().mean().reset_index(level=0, drop=True)
    df["customer_std_amount"] = prior_by_customer.expanding().std().reset_index(level=0, drop=True)
    df["customer_p99_amount"] = (
        prior_by_customer.expanding().quantile(0.99).reset_index(level=0, drop=True)
    )
    df[["customer_avg_amount", "customer_std_amount", "customer_p99_amount"]] = df[
        ["customer_avg_amount", "customer_std_amount", "customer_p99_amount"]
    ].fillna(0.0)

    thin_history = df["prior_count"] < MIN_HISTORY_FOR_BASELINE
    safe_std = df["customer_std_amount"].where(df["customer_std_amount"] > 0, np.nan)
    df["amount_zscore"] = ((df["amount"] - df["customer_avg_amount"]) / safe_std).fillna(0.0)
    df.loc[thin_history, "amount_zscore"] = 0.0
    # Extreme z-scores add nothing beyond "far outside normal" and would
    # dominate the scaler, so clip to a band the model can use.
    df["amount_zscore"] = df["amount_zscore"].clip(-20, 20)

    safe_avg = df["customer_avg_amount"].where(df["customer_avg_amount"] > 0, np.nan)
    df["amount_vs_customer_avg_ratio"] = (df["amount"] / safe_avg).fillna(1.0).clip(0, 100)
    df.loc[thin_history, "amount_vs_customer_avg_ratio"] = 1.0

    # --- Velocity ---------------------------------------------------------
    times_ns = df["occurred_at"].astype("int64").to_numpy()
    windows = (
        (int(pd.Timedelta(minutes=10).value), "txn_count_10m"),
        (int(pd.Timedelta(hours=1).value), "txn_count_1h"),
        (int(pd.Timedelta(hours=24).value), "txn_count_24h"),
    )
    for _, column in windows:
        df[column] = 0
    # The frame is sorted by (customer_id, occurred_at), so each customer owns a
    # contiguous slice and their timestamps are already ascending.
    for _, positions in df.groupby("customer_id", sort=False).indices.items():
        group_times = times_ns[positions]
        for window_ns, column in windows:
            df.iloc[positions, df.columns.get_loc(column)] = _window_counts(
                group_times, window_ns
            )

    # Pace so far, measured only against what preceded each transaction. Using
    # the customer's whole span here would let later activity change an earlier
    # transaction's features - the exact lookahead this module exists to avoid.
    # The customer's first timestamp is safe to use: it precedes every row.
    first_seen = df.groupby("customer_id")["occurred_at"].transform("min")
    elapsed_days = (
        (df["occurred_at"] - first_seen).dt.total_seconds() / 86400.0
    ).clip(lower=1.0)
    df["customer_avg_daily_txns"] = (df["prior_count"] / elapsed_days).astype(float)

    expected_24h = df["customer_avg_daily_txns"].clip(lower=0.5)
    df["velocity_score"] = ((df["txn_count_24h"] / expected_24h) - 1.0).clip(0, 20)
    # Too little history to say what "fast" means for this customer yet.
    df.loc[thin_history, "velocity_score"] = 0.0

    # --- Stateful signals -------------------------------------------------
    stateful_frames = []
    for _, group in df.groupby("customer_id", sort=False):
        computed = _stateful_pass(group)
        stateful_frames.append(pd.DataFrame(computed, index=group.index))
    stateful = pd.concat(stateful_frames).sort_index()
    for column in stateful.columns:
        df[column] = stateful[column]

    # --- Drift ------------------------------------------------------------
    df = _add_trend_features(df)

    df["behavior_change_score"] = _behaviour_change(
        df["amount_trend_ratio"], df["frequency_trend_ratio"], df["prior_count"]
    )

    result = df[["transaction_id", *FEATURE_COLUMNS]].copy()
    for column in ("txn_count_10m", "txn_count_1h", "txn_count_24h", "device_usage_count",
                   "unique_devices_30d", "unique_locations_30d", "hour_of_day", "day_of_week"):
        result[column] = result[column].astype(int)
    for column in ("is_new_device", "is_new_location", "is_new_beneficiary", "is_unusual_hour"):
        result[column] = result[column].astype(bool)
    float_columns = [
        c for c in FEATURE_COLUMNS
        if c not in {"is_new_device", "is_new_location", "is_new_beneficiary", "is_unusual_hour"}
        and not c.startswith(("txn_count", "device_usage", "unique_", "hour_of", "day_of"))
    ]
    result[float_columns] = result[float_columns].astype(float).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    return result


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recent-window spend and pace versus the customer's longer-run baseline."""
    amount_trend = np.ones(len(df))
    frequency_trend = np.ones(len(df))

    for _, group in df.groupby("customer_id", sort=False):
        times = group["occurred_at"].to_numpy()
        amounts = group["amount"].to_numpy(dtype=float)
        positions = group.index.to_numpy()
        n = len(group)

        # ``recent_start``/``recent_sum`` track the trailing RECENT_WINDOW_DAYS
        # of *prior* transactions, maintained incrementally across every row so
        # the window is correct from the first scored transaction onward.
        recent_start = 0
        recent_sum = 0.0
        prior_sum = 0.0

        for i in range(n):
            cutoff = pd.Timestamp(times[i]) - pd.Timedelta(days=RECENT_WINDOW_DAYS)
            while recent_start < i and pd.Timestamp(times[recent_start]) < cutoff:
                recent_sum -= amounts[recent_start]
                recent_start += 1

            recent_n = i - recent_start
            if i >= MIN_HISTORY_FOR_BASELINE and recent_n > 0:
                baseline_mean = prior_sum / i
                recent_mean = recent_sum / recent_n
                if baseline_mean > 0:
                    amount_trend[positions[i]] = min(recent_mean / baseline_mean, 50.0)

                baseline_days = max(
                    (pd.Timestamp(times[i]) - pd.Timestamp(times[0])).total_seconds() / 86400.0,
                    1.0,
                )
                baseline_rate = i / baseline_days
                recent_rate = recent_n / float(RECENT_WINDOW_DAYS)
                if baseline_rate > 0:
                    frequency_trend[positions[i]] = min(recent_rate / baseline_rate, 50.0)

            recent_sum += amounts[i]
            prior_sum += amounts[i]

    df["amount_trend_ratio"] = amount_trend
    df["frequency_trend_ratio"] = frequency_trend
    return df


def _behaviour_change(
    amount_trend: pd.Series, frequency_trend: pd.Series, prior_count: pd.Series
) -> pd.Series:
    """Blend spend and pace drift into a single 0-1 score.

    Only upward drift counts: a customer spending less than usual is not a risk
    signal. A ratio of 1.0 (no change) scores 0; 4x or more scores 1.
    """
    amount_drift = ((amount_trend - 1.0) / 3.0).clip(0, 1)
    frequency_drift = ((frequency_trend - 1.0) / 3.0).clip(0, 1)
    score = (0.6 * amount_drift + 0.4 * frequency_drift).clip(0, 1)
    return score.where(prior_count >= MIN_HISTORY_FOR_BASELINE, 0.0)


def features_to_records(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert the feature frame to rows ready for bulk insert."""
    now = datetime.now(UTC)
    records = features.to_dict(orient="records")
    for record in records:
        record["computed_at"] = now
    return records
