"""The permission matrix.

One place defines what each role may do. The API depends on this, and the
frontend fetches the same matrix from ``/api/auth/me`` so the UI can disable
what a role cannot do - but the server check is what actually enforces it.
"""

from __future__ import annotations

import enum

from app.db.enums import ROLE_LEVELS, RoleCode


class Permission(str, enum.Enum):
    """Capabilities that can be granted to a role."""

    # Read access - every authenticated role has these.
    VIEW_DASHBOARD = "VIEW_DASHBOARD"
    VIEW_TRANSACTIONS = "VIEW_TRANSACTIONS"
    VIEW_CUSTOMERS = "VIEW_CUSTOMERS"
    VIEW_ALERTS = "VIEW_ALERTS"
    VIEW_CASES = "VIEW_CASES"
    VIEW_ANALYTICS = "VIEW_ANALYTICS"
    VIEW_MODEL_STATUS = "VIEW_MODEL_STATUS"

    # Analyst working set.
    CREATE_CASE = "CREATE_CASE"
    ADD_CASE_NOTE = "ADD_CASE_NOTE"
    LINK_TRANSACTION = "LINK_TRANSACTION"
    EXPORT_DATA = "EXPORT_DATA"
    ACKNOWLEDGE_ALERT = "ACKNOWLEDGE_ALERT"
    FLAG_TRANSACTION = "FLAG_TRANSACTION"

    # Senior analyst and above.
    ASSIGN_CASE = "ASSIGN_CASE"
    CHANGE_CASE_PRIORITY = "CHANGE_CASE_PRIORITY"
    ESCALATE_CASE = "ESCALATE_CASE"
    RESOLVE_CASE = "RESOLVE_CASE"
    DISMISS_ALERT = "DISMISS_ALERT"

    # Manager and above.
    CLOSE_CASE = "CLOSE_CASE"
    REOPEN_CASE = "REOPEN_CASE"
    VIEW_AUDIT_LOGS = "VIEW_AUDIT_LOGS"
    RUN_DETECTION = "RUN_DETECTION"
    VIEW_REPORTS = "VIEW_REPORTS"

    # Administrator only.
    MANAGE_USERS = "MANAGE_USERS"
    CHANGE_USER_ROLE = "CHANGE_USER_ROLE"
    MANAGE_SETTINGS = "MANAGE_SETTINGS"
    TRAIN_MODEL = "TRAIN_MODEL"
    IMPORT_DATA = "IMPORT_DATA"


_ANALYST: frozenset[Permission] = frozenset(
    {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_TRANSACTIONS,
        Permission.VIEW_CUSTOMERS,
        Permission.VIEW_ALERTS,
        Permission.VIEW_CASES,
        Permission.VIEW_ANALYTICS,
        Permission.VIEW_MODEL_STATUS,
        Permission.CREATE_CASE,
        Permission.ADD_CASE_NOTE,
        Permission.LINK_TRANSACTION,
        Permission.EXPORT_DATA,
        Permission.ACKNOWLEDGE_ALERT,
        Permission.FLAG_TRANSACTION,
    }
)

_SENIOR_ANALYST: frozenset[Permission] = _ANALYST | {
    Permission.ASSIGN_CASE,
    Permission.CHANGE_CASE_PRIORITY,
    Permission.ESCALATE_CASE,
    Permission.RESOLVE_CASE,
    Permission.DISMISS_ALERT,
}

_MANAGER: frozenset[Permission] = _SENIOR_ANALYST | {
    Permission.CLOSE_CASE,
    Permission.REOPEN_CASE,
    Permission.VIEW_AUDIT_LOGS,
    Permission.RUN_DETECTION,
    Permission.VIEW_REPORTS,
}

_ADMIN: frozenset[Permission] = _MANAGER | {
    Permission.MANAGE_USERS,
    Permission.CHANGE_USER_ROLE,
    Permission.MANAGE_SETTINGS,
    Permission.TRAIN_MODEL,
    Permission.IMPORT_DATA,
}

ROLE_PERMISSIONS: dict[RoleCode, frozenset[Permission]] = {
    RoleCode.ANALYST: _ANALYST,
    RoleCode.SENIOR_ANALYST: _SENIOR_ANALYST,
    RoleCode.MANAGER: _MANAGER,
    RoleCode.ADMIN: _ADMIN,
}


def permissions_for(role: RoleCode) -> frozenset[Permission]:
    return ROLE_PERMISSIONS.get(role, frozenset())


def has_permission(role: RoleCode, permission: Permission) -> bool:
    return permission in permissions_for(role)


def role_at_least(role: RoleCode, minimum: RoleCode) -> bool:
    """True when ``role`` sits at or above ``minimum`` in the hierarchy."""
    return ROLE_LEVELS.get(role, 0) >= ROLE_LEVELS.get(minimum, 0)
