"""Initial Sentinel Finance schema.

Creates the sixteen core tables, their enum types, indexes and constraints, and
installs the append-only protection on ``audit_logs``.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Enum types are created once up front and referenced with ``create_type=False``
# so that a type shared by several tables (risk_level) is not created twice.
ENUM_TYPES: dict[str, tuple[str, ...]] = {
    "role_code": ("ANALYST", "SENIOR_ANALYST", "MANAGER", "ADMIN"),
    "risk_level": ("LOW", "MEDIUM", "HIGH", "CRITICAL"),
    "transaction_status": (
        "COMPLETED",
        "PENDING",
        "FLAGGED",
        "UNDER_REVIEW",
        "CLEARED",
        "BLOCKED",
    ),
    "payment_method": ("UPI", "CARD", "NET_BANKING", "IMPS", "NEFT", "RTGS", "WALLET"),
    "anomaly_scenario": (
        "AMOUNT_SPIKE",
        "NEW_DEVICE",
        "UNUSUAL_LOCATION",
        "UNUSUAL_TIME",
        "VELOCITY_SPIKE",
        "MULTI_SIGNAL",
    ),
    "alert_status": ("NEW", "VIEWED", "ASSIGNED", "ESCALATED", "DISMISSED"),
    "case_status": (
        "NEW",
        "UNDER_REVIEW",
        "ESCALATED",
        "RESOLVED",
        "FALSE_POSITIVE",
        "CLOSED",
    ),
    "case_priority": ("LOW", "MEDIUM", "HIGH", "CRITICAL"),
    "case_event_type": (
        "CREATED",
        "OPENED",
        "STATUS_CHANGED",
        "PRIORITY_CHANGED",
        "ASSIGNED",
        "UNASSIGNED",
        "NOTE_ADDED",
        "TRANSACTION_LINKED",
        "TRANSACTION_UNLINKED",
        "TRANSACTION_FLAGGED",
        "REOPENED",
        "CLOSED",
    ),
    "audit_action": (
        "LOGIN",
        "LOGIN_FAILED",
        "LOGOUT",
        "TRANSACTION_VIEWED",
        "TRANSACTION_FLAGGED",
        "TRANSACTION_STATUS_CHANGED",
        "CUSTOMER_VIEWED",
        "CASE_CREATED",
        "CASE_UPDATED",
        "CASE_ASSIGNED",
        "CASE_CLOSED",
        "CASE_REOPENED",
        "CASE_NOTE_ADDED",
        "ALERT_ACKNOWLEDGED",
        "ALERT_DISMISSED",
        "USER_CREATED",
        "USER_UPDATED",
        "USER_ROLE_CHANGED",
        "USER_DEACTIVATED",
        "EXPORT_PERFORMED",
        "IMPORT_PERFORMED",
        "DETECTION_RUN",
        "MODEL_TRAINED",
    ),
    "import_status": (
        "PENDING",
        "PROCESSING",
        "COMPLETED",
        "COMPLETED_WITH_ERRORS",
        "FAILED",
    ),
}


def _enum(name: str) -> postgresql.ENUM:
    """Reference an already-created enum type."""
    return postgresql.ENUM(*ENUM_TYPES[name], name=name, create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    for name, values in ENUM_TYPES.items():
        postgresql.ENUM(*values, name=name).create(bind, checkfirst=True)

    now = sa.text("now()")

    # --- roles ---------------------------------------------------------------
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", _enum("role_code"), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_roles"),
    )
    op.create_index("ix_roles_code", "roles", ["code"], unique=True)

    # --- users ---------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=128), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.ForeignKeyConstraint(
            ["role_id"], ["roles.id"], name="fk_users_role_id_roles", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role_id", "users", ["role_id"])

    # --- customers -----------------------------------------------------------
    op.create_table(
        "customers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("customer_ref", sa.String(length=32), nullable=False),
        sa.Column("full_name", sa.String(length=128), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("phone", sa.String(length=24), nullable=False),
        sa.Column("account_number", sa.String(length=32), nullable=False),
        sa.Column("home_city", sa.String(length=64), nullable=False),
        sa.Column("home_region", sa.String(length=64), nullable=False),
        sa.Column("home_latitude", sa.Float(), nullable=False),
        sa.Column("home_longitude", sa.Float(), nullable=False),
        sa.Column("segment", sa.String(length=32), nullable=False),
        sa.Column("kyc_level", sa.String(length=16), nullable=False),
        sa.Column("onboarded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_customers"),
    )
    op.create_index("ix_customers_customer_ref", "customers", ["customer_ref"], unique=True)
    op.create_index("ix_customers_home_city", "customers", ["home_city"])
    op.create_index("ix_customers_is_demo", "customers", ["is_demo"])

    # --- devices -------------------------------------------------------------
    op.create_table(
        "devices",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("device_ref", sa.String(length=48), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("device_type", sa.String(length=32), nullable=False),
        sa.Column("operating_system", sa.String(length=32), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_trusted", sa.Boolean(), nullable=False),
        sa.Column("usage_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name="fk_devices_customer_id_customers",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_devices"),
        sa.UniqueConstraint("customer_id", "device_ref", name="uq_devices_customer_device"),
    )
    op.create_index("ix_devices_device_ref", "devices", ["device_ref"])
    op.create_index("ix_devices_customer_id", "devices", ["customer_id"])

    # --- beneficiaries -------------------------------------------------------
    op.create_table(
        "beneficiaries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("beneficiary_ref", sa.String(length=48), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("account_number", sa.String(length=32), nullable=False),
        sa.Column("bank_name", sa.String(length=64), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payment_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name="fk_beneficiaries_customer_id_customers",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_beneficiaries"),
        sa.UniqueConstraint(
            "customer_id", "beneficiary_ref", name="uq_beneficiaries_customer_beneficiary"
        ),
    )
    op.create_index("ix_beneficiaries_beneficiary_ref", "beneficiaries", ["beneficiary_ref"])
    op.create_index("ix_beneficiaries_customer_id", "beneficiaries", ["customer_id"])

    # --- model_versions ------------------------------------------------------
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("algorithm", sa.String(length=64), nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("training_record_count", sa.Integer(), nullable=False),
        sa.Column("feature_count", sa.Integer(), nullable=False),
        sa.Column("feature_names", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("contamination", sa.Float(), nullable=False),
        sa.Column("n_estimators", sa.Integer(), nullable=False),
        sa.Column("random_state", sa.Integer(), nullable=False),
        sa.Column("anomaly_threshold", sa.Float(), nullable=False),
        sa.Column("anomaly_rate", sa.Float(), nullable=False),
        sa.Column("artifact_path", sa.String(length=400), nullable=True),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_model_versions"),
    )
    op.create_index("ix_model_versions_version", "model_versions", ["version"], unique=True)
    op.create_index("ix_model_versions_is_active", "model_versions", ["is_active"])

    # --- import_jobs ---------------------------------------------------------
    op.create_table(
        "import_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("uploaded_by", sa.Integer(), nullable=True),
        sa.Column("status", _enum("import_status"), nullable=False),
        sa.Column("records_received", sa.Integer(), nullable=False),
        sa.Column("records_valid", sa.Integer(), nullable=False),
        sa.Column("records_invalid", sa.Integer(), nullable=False),
        sa.Column("records_duplicate", sa.Integer(), nullable=False),
        sa.Column("records_processed", sa.Integer(), nullable=False),
        sa.Column("records_flagged", sa.Integer(), nullable=False),
        sa.Column("error_report", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.ForeignKeyConstraint(
            ["uploaded_by"],
            ["users.id"],
            name="fk_import_jobs_uploaded_by_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_import_jobs"),
    )
    op.create_index("ix_import_jobs_uploaded_by", "import_jobs", ["uploaded_by"])
    op.create_index("ix_import_jobs_status", "import_jobs", ["status"])

    # --- transactions --------------------------------------------------------
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("transaction_ref", sa.String(length=32), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=False),
        sa.Column("amount", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("location_city", sa.String(length=64), nullable=False),
        sa.Column("location_region", sa.String(length=64), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("device_id", sa.Integer(), nullable=True),
        sa.Column("beneficiary_id", sa.Integer(), nullable=True),
        sa.Column("payment_method", _enum("payment_method"), nullable=False),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("status", _enum("transaction_status"), nullable=False),
        sa.Column("import_job_id", sa.Integer(), nullable=True),
        sa.Column("is_demo", sa.Boolean(), nullable=False),
        sa.Column("injected_scenario", _enum("anomaly_scenario"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.CheckConstraint("amount > 0", name="ck_transactions_amount_positive"),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name="fk_transactions_customer_id_customers",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["device_id"],
            ["devices.id"],
            name="fk_transactions_device_id_devices",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["beneficiary_id"],
            ["beneficiaries.id"],
            name="fk_transactions_beneficiary_id_beneficiaries",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["import_job_id"],
            ["import_jobs.id"],
            name="fk_transactions_import_job_id_import_jobs",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_transactions"),
    )
    op.create_index(
        "ix_transactions_transaction_ref", "transactions", ["transaction_ref"], unique=True
    )
    op.create_index("ix_transactions_customer_id", "transactions", ["customer_id"])
    op.create_index("ix_transactions_location_city", "transactions", ["location_city"])
    op.create_index("ix_transactions_device_id", "transactions", ["device_id"])
    op.create_index("ix_transactions_beneficiary_id", "transactions", ["beneficiary_id"])
    op.create_index("ix_transactions_status", "transactions", ["status"])
    op.create_index("ix_transactions_import_job_id", "transactions", ["import_job_id"])
    op.create_index("ix_transactions_is_demo", "transactions", ["is_demo"])
    op.create_index("ix_transactions_injected_scenario", "transactions", ["injected_scenario"])
    op.create_index(
        "ix_transactions_customer_occurred", "transactions", ["customer_id", "occurred_at"]
    )
    op.create_index("ix_transactions_occurred_at_desc", "transactions", ["occurred_at"])

    # --- transaction_features ------------------------------------------------
    op.create_table(
        "transaction_features",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("transaction_id", sa.Integer(), nullable=False),
        sa.Column("amount_value", sa.Float(), nullable=False),
        sa.Column("customer_avg_amount", sa.Float(), nullable=False),
        sa.Column("customer_std_amount", sa.Float(), nullable=False),
        sa.Column("amount_zscore", sa.Float(), nullable=False),
        sa.Column("amount_vs_customer_avg_ratio", sa.Float(), nullable=False),
        sa.Column("customer_p99_amount", sa.Float(), nullable=False),
        sa.Column("txn_count_10m", sa.Integer(), nullable=False),
        sa.Column("txn_count_1h", sa.Integer(), nullable=False),
        sa.Column("txn_count_24h", sa.Integer(), nullable=False),
        sa.Column("customer_avg_daily_txns", sa.Float(), nullable=False),
        sa.Column("velocity_score", sa.Float(), nullable=False),
        sa.Column("hour_of_day", sa.Integer(), nullable=False),
        sa.Column("day_of_week", sa.Integer(), nullable=False),
        sa.Column("is_unusual_hour", sa.Boolean(), nullable=False),
        sa.Column("is_new_device", sa.Boolean(), nullable=False),
        sa.Column("is_new_location", sa.Boolean(), nullable=False),
        sa.Column("is_new_beneficiary", sa.Boolean(), nullable=False),
        sa.Column("device_usage_count", sa.Integer(), nullable=False),
        sa.Column("distance_from_usual_km", sa.Float(), nullable=False),
        sa.Column("unique_devices_30d", sa.Integer(), nullable=False),
        sa.Column("unique_locations_30d", sa.Integer(), nullable=False),
        sa.Column("behavior_change_score", sa.Float(), nullable=False),
        sa.Column("amount_trend_ratio", sa.Float(), nullable=False),
        sa.Column("frequency_trend_ratio", sa.Float(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["transactions.id"],
            name="fk_transaction_features_transaction_id_transactions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_transaction_features"),
    )
    op.create_index(
        "ix_transaction_features_transaction_id",
        "transaction_features",
        ["transaction_id"],
        unique=True,
    )
    op.create_index("ix_transaction_features_created_at", "transaction_features", ["created_at"])

    # --- risk_scores ---------------------------------------------------------
    op.create_table(
        "risk_scores",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("transaction_id", sa.Integer(), nullable=False),
        sa.Column("business_score", sa.Integer(), nullable=False),
        sa.Column("risk_level", _enum("risk_level"), nullable=False),
        sa.Column("rule_score", sa.Integer(), nullable=False),
        sa.Column("ml_uplift_points", sa.Integer(), nullable=False),
        sa.Column("ml_anomaly_score", sa.Float(), nullable=False),
        sa.Column("ml_raw_score", sa.Float(), nullable=False),
        sa.Column("is_ml_anomaly", sa.Boolean(), nullable=False),
        sa.Column("factors", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("primary_reason", sa.String(length=32), nullable=True),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.CheckConstraint(
            "business_score >= 0 AND business_score <= 100",
            name="ck_risk_scores_business_score_range",
        ),
        sa.CheckConstraint(
            "ml_anomaly_score >= 0 AND ml_anomaly_score <= 1",
            name="ck_risk_scores_ml_anomaly_score_range",
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["transactions.id"],
            name="fk_risk_scores_transaction_id_transactions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_risk_scores_model_version_id_model_versions",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_risk_scores"),
    )
    op.create_index(
        "ix_risk_scores_transaction_id", "risk_scores", ["transaction_id"], unique=True
    )
    op.create_index("ix_risk_scores_business_score", "risk_scores", ["business_score"])
    op.create_index("ix_risk_scores_risk_level", "risk_scores", ["risk_level"])
    op.create_index("ix_risk_scores_primary_reason", "risk_scores", ["primary_reason"])
    op.create_index("ix_risk_scores_model_version_id", "risk_scores", ["model_version_id"])
    op.create_index("ix_risk_scores_created_at", "risk_scores", ["created_at"])
    op.create_index(
        "ix_risk_scores_level_score", "risk_scores", ["risk_level", "business_score"]
    )

    # --- alerts --------------------------------------------------------------
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("alert_ref", sa.String(length=32), nullable=False),
        sa.Column("transaction_id", sa.Integer(), nullable=False),
        sa.Column("risk_level", _enum("risk_level"), nullable=False),
        sa.Column("risk_score", sa.Integer(), nullable=False),
        sa.Column("headline", sa.String(length=160), nullable=False),
        sa.Column("reason_summary", sa.Text(), nullable=False),
        sa.Column("status", _enum("alert_status"), nullable=False),
        sa.Column("assigned_to", sa.Integer(), nullable=True),
        sa.Column("acknowledged_by", sa.Integer(), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["transactions.id"],
            name="fk_alerts_transaction_id_transactions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["assigned_to"], ["users.id"], name="fk_alerts_assigned_to_users", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["acknowledged_by"],
            ["users.id"],
            name="fk_alerts_acknowledged_by_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_alerts"),
    )
    op.create_index("ix_alerts_alert_ref", "alerts", ["alert_ref"], unique=True)
    op.create_index("ix_alerts_transaction_id", "alerts", ["transaction_id"], unique=True)
    op.create_index("ix_alerts_risk_level", "alerts", ["risk_level"])
    op.create_index("ix_alerts_risk_score", "alerts", ["risk_score"])
    op.create_index("ix_alerts_status", "alerts", ["status"])
    op.create_index("ix_alerts_assigned_to", "alerts", ["assigned_to"])
    op.create_index("ix_alerts_triggered_at", "alerts", ["triggered_at"])

    # --- cases ---------------------------------------------------------------
    op.create_table(
        "cases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("case_ref", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("customer_id", sa.Integer(), nullable=True),
        sa.Column("status", _enum("case_status"), nullable=False),
        sa.Column("priority", _enum("case_priority"), nullable=False),
        sa.Column("assigned_to", sa.Integer(), nullable=True),
        sa.Column("created_by", sa.Integer(), nullable=False),
        sa.Column("peak_risk_score", sa.Integer(), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name="fk_cases_customer_id_customers",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["assigned_to"], ["users.id"], name="fk_cases_assigned_to_users", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], name="fk_cases_created_by_users", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cases"),
    )
    op.create_index("ix_cases_case_ref", "cases", ["case_ref"], unique=True)
    op.create_index("ix_cases_customer_id", "cases", ["customer_id"])
    op.create_index("ix_cases_status", "cases", ["status"])
    op.create_index("ix_cases_priority", "cases", ["priority"])
    op.create_index("ix_cases_assigned_to", "cases", ["assigned_to"])
    op.create_index("ix_cases_created_by", "cases", ["created_by"])
    op.create_index("ix_cases_peak_risk_score", "cases", ["peak_risk_score"])
    op.create_index("ix_cases_status_priority", "cases", ["status", "priority"])

    # --- case_transactions ---------------------------------------------------
    op.create_table(
        "case_transactions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("transaction_id", sa.Integer(), nullable=False),
        sa.Column("linked_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["cases.id"],
            name="fk_case_transactions_case_id_cases",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["transactions.id"],
            name="fk_case_transactions_transaction_id_transactions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["linked_by"],
            ["users.id"],
            name="fk_case_transactions_linked_by_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_case_transactions"),
        sa.UniqueConstraint("case_id", "transaction_id", name="uq_case_transactions_case_txn"),
    )
    op.create_index("ix_case_transactions_case_id", "case_transactions", ["case_id"])
    op.create_index(
        "ix_case_transactions_transaction_id", "case_transactions", ["transaction_id"]
    )
    op.create_index("ix_case_transactions_created_at", "case_transactions", ["created_at"])

    # --- case_notes ----------------------------------------------------------
    op.create_table(
        "case_notes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("author_id", sa.Integer(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("evidence_ref", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.ForeignKeyConstraint(
            ["case_id"], ["cases.id"], name="fk_case_notes_case_id_cases", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["author_id"], ["users.id"], name="fk_case_notes_author_id_users", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_case_notes"),
    )
    op.create_index("ix_case_notes_case_id", "case_notes", ["case_id"])
    op.create_index("ix_case_notes_author_id", "case_notes", ["author_id"])
    op.create_index("ix_case_notes_created_at", "case_notes", ["created_at"])

    # --- case_events ---------------------------------------------------------
    op.create_table(
        "case_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("event_type", _enum("case_event_type"), nullable=False),
        sa.Column("description", sa.String(length=400), nullable=False),
        sa.Column("from_value", sa.String(length=64), nullable=True),
        sa.Column("to_value", sa.String(length=64), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["case_id"], ["cases.id"], name="fk_case_events_case_id_cases", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["actor_id"], ["users.id"], name="fk_case_events_actor_id_users", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_case_events"),
    )
    op.create_index("ix_case_events_case_id", "case_events", ["case_id"])
    op.create_index("ix_case_events_occurred_at", "case_events", ["occurred_at"])
    op.create_index("ix_case_events_case_occurred", "case_events", ["case_id", "occurred_at"])

    # --- audit_logs ----------------------------------------------------------
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("actor_email", sa.String(length=255), nullable=True),
        sa.Column("actor_role", sa.String(length=32), nullable=True),
        sa.Column("action", _enum("audit_action"), nullable=False),
        sa.Column("resource_type", sa.String(length=48), nullable=True),
        sa.Column("resource_id", sa.String(length=64), nullable=True),
        sa.Column("description", sa.String(length=400), nullable=False),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column("previous_state", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("new_state", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["actor_id"], ["users.id"], name="fk_audit_logs_actor_id_users", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_audit_logs"),
    )
    op.create_index("ix_audit_logs_actor_id", "audit_logs", ["actor_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])
    op.create_index("ix_audit_logs_action_created", "audit_logs", ["action", "created_at"])
    op.create_index("ix_audit_logs_resource", "audit_logs", ["resource_type", "resource_id"])

    # --- audit_logs: append-only protection ----------------------------------
    # Rules turn UPDATE/DELETE into no-ops for every caller including the owner,
    # so an ordinary application bug or a compromised app role cannot rewrite
    # history. Dropping the rules requires table ownership (a DBA action).
    op.execute("CREATE RULE audit_logs_no_update AS ON UPDATE TO audit_logs DO INSTEAD NOTHING")
    op.execute("CREATE RULE audit_logs_no_delete AS ON DELETE TO audit_logs DO INSTEAD NOTHING")


def downgrade() -> None:
    op.execute("DROP RULE IF EXISTS audit_logs_no_delete ON audit_logs")
    op.execute("DROP RULE IF EXISTS audit_logs_no_update ON audit_logs")

    for table in (
        "audit_logs",
        "case_events",
        "case_notes",
        "case_transactions",
        "cases",
        "alerts",
        "risk_scores",
        "transaction_features",
        "transactions",
        "import_jobs",
        "model_versions",
        "beneficiaries",
        "devices",
        "customers",
        "users",
        "roles",
    ):
        op.drop_table(table)

    bind = op.get_bind()
    for name in ENUM_TYPES:
        postgresql.ENUM(name=name).drop(bind, checkfirst=True)
