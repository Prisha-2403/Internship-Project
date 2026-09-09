"""Authentication endpoints: login, refresh, logout, current user.

Every outcome - success or failure - is written to the audit trail. Failed
logins record the attempted address but never the submitted password.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Cookie, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.auth.dependencies import CurrentUser, DbSession
from app.core.config import settings
from app.core.exceptions import AuthenticationError, RateLimitError
from app.core.middleware import get_client_ip, get_user_agent
from app.core.rate_limit import login_limiter, login_rate_limit_key
from app.core.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.db.enums import AuditAction
from app.db.models.user import User
from app.schemas.auth import LoginRequest, LoginResponse, RefreshResponse, UserOut
from app.schemas.common import MessageResponse
from app.services import audit_service

router = APIRouter(prefix="/auth", tags=["Authentication"])

REFRESH_COOKIE_NAME = "sentinel_refresh"


def _set_refresh_cookie(response: Response, token: str, max_age: int) -> None:
    """Store the refresh token httpOnly so script cannot read it."""
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        max_age=max_age,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path=f"{settings.api_v1_prefix}/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=REFRESH_COOKIE_NAME,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path=f"{settings.api_v1_prefix}/auth",
    )


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Sign in and receive an access token",
    responses={
        401: {"description": "Invalid credentials or deactivated account."},
        429: {"description": "Too many failed attempts from this address."},
    },
)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: DbSession,
) -> LoginResponse:
    """Authenticate and issue tokens.

    Rate limited per (IP, email). The same generic message is returned for an
    unknown address and a wrong password, so the endpoint cannot be used to
    enumerate accounts.
    """
    ip_address = get_client_ip(request)
    user_agent = get_user_agent(request)
    email = payload.email.strip().lower()
    rate_key = login_rate_limit_key(ip_address, email)

    allowed, retry_after = login_limiter.check(rate_key)
    if not allowed:
        audit_service.record_and_commit(
            db,
            action=AuditAction.LOGIN_FAILED,
            actor_email=email,
            description="Login blocked: rate limit exceeded.",
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise RateLimitError(
            f"Too many failed sign-in attempts. Please try again in {retry_after} seconds.",
            details={"retry_after_seconds": retry_after},
        )

    user = db.execute(
        select(User).options(joinedload(User.role)).where(User.email == email)
    ).scalar_one_or_none()

    # Always run the hash comparison, even for an unknown address, so response
    # timing does not reveal whether the account exists.
    stored_hash = user.password_hash if user else "$2b$12$" + "." * 53
    password_ok = verify_password(payload.password, stored_hash)

    if user is None or not password_ok:
        login_limiter.record(rate_key)
        audit_service.record_and_commit(
            db,
            action=AuditAction.LOGIN_FAILED,
            actor=user if user else None,
            actor_email=email,
            description="Sign-in failed: invalid credentials.",
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise AuthenticationError("The email address or password is incorrect.")

    if not user.is_active:
        login_limiter.record(rate_key)
        audit_service.record_and_commit(
            db,
            action=AuditAction.LOGIN_FAILED,
            actor=user,
            description="Sign-in failed: account deactivated.",
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise AuthenticationError(
            "This account has been deactivated. Contact an administrator."
        )

    login_limiter.reset(rate_key)

    access_token, expires_in = create_access_token(user.id, user.role_code.value)
    refresh_token, max_age = create_refresh_token(user.id, remember_me=payload.remember_me)
    _set_refresh_cookie(response, refresh_token, max_age)

    user.last_login_at = datetime.now(UTC)
    audit_service.record(
        db,
        action=AuditAction.LOGIN,
        actor=user,
        resource_type="user",
        resource_id=user.id,
        description=f"Signed in as {user.role.name}.",
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.commit()
    db.refresh(user)

    return LoginResponse(
        access_token=access_token,
        expires_in=expires_in,
        user=UserOut.model_validate(user),
    )


@router.post(
    "/refresh",
    response_model=RefreshResponse,
    summary="Exchange the refresh cookie for a new access token",
    responses={401: {"description": "Missing, expired or invalid refresh token."}},
)
def refresh(
    db: DbSession,
    sentinel_refresh: Annotated[str | None, Cookie(alias=REFRESH_COOKIE_NAME)] = None,
) -> RefreshResponse:
    """Issue a fresh access token from the httpOnly refresh cookie."""
    if not sentinel_refresh:
        raise AuthenticationError("Your session has expired. Please sign in again.")

    try:
        claims = decode_token(sentinel_refresh, expected_type="refresh")
    except TokenError as exc:
        raise AuthenticationError("Your session has expired. Please sign in again.") from exc

    user = db.execute(
        select(User).options(joinedload(User.role)).where(User.id == int(claims["sub"]))
    ).scalar_one_or_none()

    if user is None or not user.is_active:
        raise AuthenticationError("Your session is no longer valid. Please sign in again.")

    access_token, expires_in = create_access_token(user.id, user.role_code.value)
    return RefreshResponse(access_token=access_token, expires_in=expires_in)


@router.post("/logout", response_model=MessageResponse, summary="Sign out")
def logout(
    request: Request,
    response: Response,
    db: DbSession,
    user: CurrentUser,
) -> MessageResponse:
    """Clear the refresh cookie and record the logout."""
    _clear_refresh_cookie(response)
    audit_service.record_and_commit(
        db,
        action=AuditAction.LOGOUT,
        actor=user,
        resource_type="user",
        resource_id=user.id,
        description="Signed out.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
    )
    return MessageResponse(message="You have been signed out.")


@router.get(
    "/me",
    response_model=UserOut,
    summary="Current user, role and effective permissions",
)
def me(user: CurrentUser) -> UserOut:
    """Return the signed-in user together with their permission list."""
    return UserOut.model_validate(user)
