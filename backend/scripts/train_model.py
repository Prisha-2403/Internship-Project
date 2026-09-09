"""Retrain the anomaly detection model and rescore all transactions.

    python -m scripts.train_model
    python -m scripts.train_model --notes "Quarterly retrain"
    python -m scripts.train_model --detect-only    # score with the active model

Registers a new model version, marks it active, and rewrites features and risk
scores. Existing analyst decisions on alerts and cases are preserved.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import configure_logging  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.ml import pipeline, registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or re-run anomaly detection.")
    parser.add_argument("--notes", default="", help="Note stored on the model version.")
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Rescore using the active model instead of training a new one.",
    )
    args = parser.parse_args()

    configure_logging("INFO")
    logging.getLogger("sentinel").setLevel(logging.INFO)

    db = SessionLocal()
    try:
        run = (
            pipeline.run_detection(db)
            if args.detect_only
            else pipeline.train_and_detect(db, notes=args.notes)
        )

        print("Detection run complete")
        print("=" * 50)
        print(f"  features computed  : {run.features_computed:,}")
        print(f"  transactions scored: {run.transactions_scored:,}")
        print(f"  alerts raised      : {run.alerts_raised}")
        print(f"  model version      : {run.model_version or 'rules-only (no model)'}")
        print(f"  used ML            : {run.used_ml}")
        print(f"  duration           : {run.duration_seconds:.1f}s")
        print("  risk distribution  :")
        for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            print(f"      {level:9s} {run.level_counts.get(level, 0):,}")

        active = registry.get_active_version(db)
        if active:
            print()
            print(f"  active model       : {active.version}")
            print(f"  trained at         : {active.trained_at:%Y-%m-%d %H:%M UTC}")
            print(f"  training records   : {active.training_record_count:,}")
            print(f"  features           : {active.feature_count}")
            print(f"  contamination      : {active.contamination}")
            print(f"  anomaly threshold  : {active.anomaly_threshold:.4f}")
            print(f"  observed rate      : {active.anomaly_rate:.4f}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
