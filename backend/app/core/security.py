"""Password hashing and JWT issuing/validation.

bcrypt is used directly rather than through passlib, which mis-detects the
version of bcrypt 4.x and emits spurious warnings.

Two token types are issued, distinguished by their ``type`` claim so a refresh
token can never be replayed as an access token:

* ``access``  - short-lived, returned in the response body, held in memory by
  the SPA and sent as a Bearer header.
* ``refresh`` - longer-lived, set as an httpOnly cookie so JavaScript (and
  therefore any XSS payload) cannot read it.
"""

from __future__ import annotations

import hmac
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import bcrypt
import jwt

from app.core.config import settings

TokenType = Literal["access", "refresh"]

# bcrypt silently truncates at 72 bytes; reject longer input rather than let two
# different passwords hash identically.
MAX_PASSWORD_BYTES = 72


class TokenError(Exception):
    """Raised when a token is missing, malformed, expired or the wrong type."""


def hash_password(password: str) -> str:
    """Return a bcrypt digest for ``password``."""
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must not exceed {MAX_PASSWORD_BYTES} bytes.")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Check ``password`` against a stored digest.

    Never raises on malformed input - a corrupt hash is simply a failed login.
    """
    try:
        encoded = password.encode("utf-8")
        if len(encoded) > MAX_PASSWORD_BYTES:
            return False
        return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _create_token(
    subject: str,
    token_type: TokenType,
    expires_delta: timedelta,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "iss": "sentinel-finance",
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.require_secret_key(), algorithm=settings.jwt_algorithm)


def create_access_token(
    user_id: int, role: str, expires_minutes: int | None = None
) -> tuple[str, int]:
    """Issue an access token. Returns ``(token, expires_in_seconds)``.

    The role is embedded for display only; every request re-reads the user's
    real role from the database, so a stale token cannot escalate privilege.
    """
    minutes = expires_minutes or settings.access_token_expire_minutes
    token = _create_token(str(user_id), "access", timedelta(minutes=minutes), {"role": role})
    return token, minutes * 60


def create_refresh_token(user_id: int, remember_me: bool = False) -> tuple[str, int]:
    """Issue a refresh token. Returns ``(token, max_age_seconds)``."""
    days = (
        settings.refresh_token_remember_me_days
        if remember_me
        else settings.refresh_token_expire_days
    )
    token = _create_token(str(user_id), "refresh", timedelta(days=days))
    return token, days * 24 * 60 * 60


def decode_token(token: str, expected_type: TokenType) -> dict[str, Any]:
    """Validate a token and return its claims.

    Raises ``TokenError`` for any failure; callers translate that into a 401
    without leaking which specific check failed.
    """
    try:
        payload = jwt.decode(
            token,
            settings.require_secret_key(),
            algorithms=[settings.jwt_algorithm],
            issuer="sentinel-finance",
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Token is invalid.") from exc

    # Constant-time compare so token type cannot be probed by timing.
    actual = str(payload.get("type", ""))
    if not hmac.compare_digest(actual, expected_type):
        raise TokenError("Token is not valid for this operation.")
    return payload
