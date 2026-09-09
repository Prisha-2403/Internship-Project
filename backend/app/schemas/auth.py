"""Authentication and user-identity schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, computed_field

from app.auth.permissions import permissions_for
from app.db.enums import RoleCode
from app.schemas.common import ORMModel


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)
    remember_me: bool = Field(
        default=False,
        description="Extends the refresh-token lifetime; does not widen access.",
    )


class RoleOut(ORMModel):
    code: RoleCode
    name: str
    description: str
    level: int


class UserOut(ORMModel):
    """A platform operator. Never includes ``password_hash``."""

    id: int
    email: EmailStr
    full_name: str
    is_active: bool
    role: RoleOut
    last_login_at: datetime | None = None
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def permissions(self) -> list[str]:
        """Capabilities for this user's role.

        The frontend uses these to disable actions the role cannot perform; the
        server enforces the same matrix independently on every request.
        """
        return sorted(p.value for p in permissions_for(self.role.code))


class UserSummary(ORMModel):
    """Compact user reference for assignment pickers and timelines."""

    id: int
    full_name: str
    email: EmailStr
    role_code: RoleCode


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access-token lifetime in seconds.")
    user: UserOut


class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
