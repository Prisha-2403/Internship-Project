"""User management, audit log and system settings schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core.security import MAX_PASSWORD_BYTES
from app.db.enums import AuditAction, RoleCode
from app.schemas.common import ORMModel

MIN_PASSWORD_LENGTH = 12


def _validate_password_strength(value: str) -> str:
    """Reject passwords that would be trivially guessable.

    Length does most of the work; the character-class checks stop the obvious
    cases without pushing users toward predictable substitutions.
    """
    if len(value) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must not exceed {MAX_PASSWORD_BYTES} bytes.")
    if not any(c.isalpha() for c in value):
        raise ValueError("Password must contain at least one letter.")
    if not any(c.isdigit() for c in value):
        raise ValueError("Password must contain at least one digit.")
    return value


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=128)
    password: str
    role_code: RoleCode

    @field_validator("password")
    @classmethod
    def _password(cls, value: str) -> str:
        return _validate_password_strength(value)


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=128)
    role_code: RoleCode | None = None
    is_active: bool | None = None


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _password(cls, value: str) -> str:
        return _validate_password_strength(value)


class AuditLogOut(ORMModel):
    """One audit entry. Append-only; there is no update schema by design."""

    id: int
    actor_id: int | None = None
    actor_email: str | None = None
    actor_role: str | None = None
    action: AuditAction
    resource_type: str | None = None
    resource_id: str | None = None
    description: str
    ip_address: str | None = None
    previous_state: dict[str, Any] | None = None
    new_state: dict[str, Any] | None = None
    created_at: datetime


class SystemSettings(BaseModel):
    """Read-only view of the effective configuration.

    Secrets are never included - not the signing key, not the database
    password, not any credential. Only operational thresholds appear here.
    """

    app_name: str
    environment: str
    api_version: str

    risk_level_medium_min: int
    risk_level_high_min: int
    risk_level_critical_min: int
    alert_min_score: int

    ml_contamination: float
    ml_n_estimators: int
    ml_max_uplift_points: int

    access_token_expire_minutes: int
    login_rate_limit_attempts: int
    login_rate_limit_window_seconds: int

    demo_data_enabled: bool
    masking_enabled: bool = True
    audit_append_only: bool = True


class RoleInfo(BaseModel):
    """A role and what it may do - drives the settings page matrix."""

    code: RoleCode
    name: str
    description: str
    level: int
    permissions: list[str]
    user_count: int
