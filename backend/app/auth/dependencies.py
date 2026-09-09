"""FastAPI dependencies for authentication and authorisation.

``get_current_user`` re-reads the user (and their role) from the database on
every request, so deactivating an account or demoting a role takes effect
immediately rather than when the token happens to expire.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.auth.permissions import Permission, has_permission, role_at_least
from app.core.exceptions import AuthenticationError, PermissionError_
from app.core.security import TokenError, decode_token
from app.db.enums import RoleCode
from app.db.models.user import User
from app.db.session import get_db

# auto_error=False so a missing header raises our own AuthenticationError with a
# consistent body, rather than Starlette's bare 403.
_bearer = HTTPBearer(auto_error=False, description="JWT access token")


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Resolve the authenticated user from the Bearer access token."""
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Authentication is required to access this resource.")

    try:
        payload = decode_token(credentials.credentials, expected_type="access")
    except TokenError as exc:
        raise AuthenticationError(str(exc)) from exc

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthenticationError("Token is invalid.") from exc

    user = db.execute(
        select(User).options(joinedload(User.role)).where(User.id == user_id)
    ).scalar_one_or_none()

    if user is None:
        raise AuthenticationError("Token is invalid.")
    if not user.is_active:
        raise AuthenticationError("This account has been deactivated.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
DbSession = Annotated[Session, Depends(get_db)]


def require_permission(permission: Permission) -> Callable[[User], User]:
    """Dependency factory enforcing a single capability."""

    def _dependency(user: CurrentUser) -> User:
        if not has_permission(user.role_code, permission):
            raise PermissionError_(
                f"Your role ({user.role.name}) does not permit this action."
            )
        return user

    return _dependency


def require_role(minimum: RoleCode) -> Callable[[User], User]:
    """Dependency factory enforcing a minimum position in the role hierarchy."""

    def _dependency(user: CurrentUser) -> User:
        if not role_at_least(user.role_code, minimum):
            raise PermissionError_(
                f"This action requires the {minimum.value.replace('_', ' ').title()} role or above."
            )
        return user

    return _dependency


def get_request_context(request: Request) -> dict[str, str | None]:
    """IP and user-agent for audit records."""
    from app.core.middleware import get_client_ip, get_user_agent

    return {"ip_address": get_client_ip(request), "user_agent": get_user_agent(request)}


RequestContext = Annotated[dict, Depends(get_request_context)]
