"""Domain enumerations shared by the ORM models, schemas and services.

These are stored as native PostgreSQL enums.  Values are the stable wire format
used by the API and the frontend, so renaming one is a breaking change that
requires a migration.
"""

from __future__ import annotations

import enum


class RoleCode(str, enum.Enum):
    """The four access tiers, ordered least to most privileged."""

    ANALYST = "ANALYST"
    SENIOR_ANALYST = "SENIOR_ANALYST"
    MANAGER = "MANAGER"
    ADMIN = "ADMIN"


#: Privilege ordering.  ``require_role`` compares these integers, so adding a
#: tier only means inserting it here with the right level.
ROLE_LEVELS: dict[RoleCode, int] = {
    RoleCode.ANALYST: 10,
    RoleCode.SENIOR_ANALYST: 20,
    RoleCode.MANAGER: 30,
    RoleCode.ADMIN: 40,
}

ROLE_LABELS: dict[RoleCode, str] = {
    RoleCode.ANALYST: "Analyst",
    RoleCode.SENIOR_ANALYST: "Senior Analyst",
    RoleCode.MANAGER: "Manager",
    RoleCode.ADMIN: "Administrator",
}


class RiskLevel(str, enum.Enum):
    """Risk bands derived from the 0-100 business risk score."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TransactionStatus(str, enum.Enum):
    """Where a transaction sits in the review workflow."""

    COMPLETED = "COMPLETED"
    PENDING = "PENDING"
    FLAGGED = "FLAGGED"
    UNDER_REVIEW = "UNDER_REVIEW"
    CLEARED = "CLEARED"
    BLOCKED = "BLOCKED"


class PaymentMethod(str, enum.Enum):
    UPI = "UPI"
    CARD = "CARD"
    NET_BANKING = "NET_BANKING"
    IMPS = "IMPS"
    NEFT = "NEFT"
    RTGS = "RTGS"
    WALLET = "WALLET"


class AlertStatus(str, enum.Enum):
    NEW = "NEW"
    VIEWED = "VIEWED"
    ASSIGNED = "ASSIGNED"
    ESCALATED = "ESCALATED"
    DISMISSED = "DISMISSED"


class CaseStatus(str, enum.Enum):
    NEW = "NEW"
    UNDER_REVIEW = "UNDER_REVIEW"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    CLOSED = "CLOSED"


#: Statuses that count as "open investigations" for dashboards and workload.
OPEN_CASE_STATUSES: tuple[CaseStatus, ...] = (
    CaseStatus.NEW,
    CaseStatus.UNDER_REVIEW,
    CaseStatus.ESCALATED,
)

#: Terminal statuses.  Reopening moves a case back to UNDER_REVIEW.
CLOSED_CASE_STATUSES: tuple[CaseStatus, ...] = (
    CaseStatus.RESOLVED,
    CaseStatus.FALSE_POSITIVE,
    CaseStatus.CLOSED,
)


class CasePriority(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CaseEventType(str, enum.Enum):
    """Timeline event kinds.  Every case mutation writes one of these."""

    CREATED = "CREATED"
    OPENED = "OPENED"
    STATUS_CHANGED = "STATUS_CHANGED"
    PRIORITY_CHANGED = "PRIORITY_CHANGED"
    ASSIGNED = "ASSIGNED"
    UNASSIGNED = "UNASSIGNED"
    NOTE_ADDED = "NOTE_ADDED"
    TRANSACTION_LINKED = "TRANSACTION_LINKED"
    TRANSACTION_UNLINKED = "TRANSACTION_UNLINKED"
    TRANSACTION_FLAGGED = "TRANSACTION_FLAGGED"
    REOPENED = "REOPENED"
    CLOSED = "CLOSED"


class AuditAction(str, enum.Enum):
    """Audit trail verbs.  Append-only; never remove a value."""

    LOGIN = "LOGIN"
    LOGIN_FAILED = "LOGIN_FAILED"
    LOGOUT = "LOGOUT"
    TRANSACTION_VIEWED = "TRANSACTION_VIEWED"
    TRANSACTION_FLAGGED = "TRANSACTION_FLAGGED"
    TRANSACTION_STATUS_CHANGED = "TRANSACTION_STATUS_CHANGED"
    CUSTOMER_VIEWED = "CUSTOMER_VIEWED"
    CASE_CREATED = "CASE_CREATED"
    CASE_UPDATED = "CASE_UPDATED"
    CASE_ASSIGNED = "CASE_ASSIGNED"
    CASE_CLOSED = "CASE_CLOSED"
    CASE_REOPENED = "CASE_REOPENED"
    CASE_NOTE_ADDED = "CASE_NOTE_ADDED"
    ALERT_ACKNOWLEDGED = "ALERT_ACKNOWLEDGED"
    ALERT_DISMISSED = "ALERT_DISMISSED"
    USER_CREATED = "USER_CREATED"
    USER_UPDATED = "USER_UPDATED"
    USER_ROLE_CHANGED = "USER_ROLE_CHANGED"
    USER_DEACTIVATED = "USER_DEACTIVATED"
    EXPORT_PERFORMED = "EXPORT_PERFORMED"
    IMPORT_PERFORMED = "IMPORT_PERFORMED"
    DETECTION_RUN = "DETECTION_RUN"
    MODEL_TRAINED = "MODEL_TRAINED"


class ImportStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
    FAILED = "FAILED"


class DetectionReason(str, enum.Enum):
    """Risk rule identifiers.

    Each value is emitted by exactly one rule in ``risk_scoring_service`` and is
    what the "Why was this flagged?" panel renders.  ``ML_ANOMALY`` is the
    bounded Isolation Forest contribution.
    """

    AMOUNT_DEVIATION = "AMOUNT_DEVIATION"
    AMOUNT_ABSOLUTE = "AMOUNT_ABSOLUTE"
    NEW_DEVICE = "NEW_DEVICE"
    NEW_LOCATION = "NEW_LOCATION"
    NEW_BENEFICIARY = "NEW_BENEFICIARY"
    UNUSUAL_TIME = "UNUSUAL_TIME"
    HIGH_FREQUENCY = "HIGH_FREQUENCY"
    BEHAVIOR_CHANGE = "BEHAVIOR_CHANGE"
    ML_ANOMALY = "ML_ANOMALY"


DETECTION_REASON_LABELS: dict[DetectionReason, str] = {
    DetectionReason.AMOUNT_DEVIATION: "Unusual Amount",
    DetectionReason.AMOUNT_ABSOLUTE: "High Value Transaction",
    DetectionReason.NEW_DEVICE: "New Device",
    DetectionReason.NEW_LOCATION: "New Location",
    DetectionReason.NEW_BENEFICIARY: "New Beneficiary",
    DetectionReason.UNUSUAL_TIME: "Unusual Time",
    DetectionReason.HIGH_FREQUENCY: "High Frequency",
    DetectionReason.BEHAVIOR_CHANGE: "Behavioural Change",
    DetectionReason.ML_ANOMALY: "ML Anomaly Signal",
}


class AnomalyScenario(str, enum.Enum):
    """Which synthetic scenario produced a demo transaction.

    Recorded only for generated data so the demo can show scenario coverage.
    These are NOT fraud labels and must never be presented as ground truth.
    """

    AMOUNT_SPIKE = "AMOUNT_SPIKE"
    NEW_DEVICE = "NEW_DEVICE"
    UNUSUAL_LOCATION = "UNUSUAL_LOCATION"
    UNUSUAL_TIME = "UNUSUAL_TIME"
    VELOCITY_SPIKE = "VELOCITY_SPIKE"
    MULTI_SIGNAL = "MULTI_SIGNAL"


ANOMALY_SCENARIO_LABELS: dict[AnomalyScenario, str] = {
    AnomalyScenario.AMOUNT_SPIKE: "Large transaction spike",
    AnomalyScenario.NEW_DEVICE: "New device",
    AnomalyScenario.UNUSUAL_LOCATION: "Unusual location",
    AnomalyScenario.UNUSUAL_TIME: "Unusual transaction time",
    AnomalyScenario.VELOCITY_SPIKE: "Transaction velocity spike",
    AnomalyScenario.MULTI_SIGNAL: "Multiple anomalies combined",
}
