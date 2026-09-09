"""HTTP middleware: security headers and client-IP resolution."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings

# This service returns JSON only, so the CSP can forbid essentially everything.
# It protects the Swagger UI and any browser that renders a response directly.
_API_CSP = (
    "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)
# Swagger UI needs its own bundled assets, so /docs gets a slightly wider policy.
_DOCS_CSP = (
    "default-src 'self'; img-src 'self' data: https://fastapi.tiangolo.com; "
    "script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' "
    "https://cdn.jsdelivr.net; frame-ancestors 'none'; base-uri 'none'"
)
_DOC_PATHS = ("/docs", "/redoc", "/openapi.json")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds defence-in-depth response headers."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-site"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
        )
        is_docs = request.url.path.startswith(_DOC_PATHS)
        response.headers["Content-Security-Policy"] = _DOCS_CSP if is_docs else _API_CSP

        # Risk data must never be cached by an intermediary.
        if request.url.path.startswith(settings.api_v1_prefix):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"

        # HSTS is meaningless (and harmful to set) outside HTTPS.
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


def get_client_ip(request: Request) -> str:
    """Best-effort client address for audit logging and rate limiting.

    ``X-Forwarded-For`` is only honoured in production, where the API is
    expected to sit behind a trusted proxy. In development the header is
    ignored so a client cannot spoof its way past the login limiter.
    """
    if settings.is_production:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()[:64]
    if request.client and request.client.host:
        return request.client.host[:64]
    return "unknown"


def get_user_agent(request: Request) -> str | None:
    agent = request.headers.get("user-agent")
    return agent[:255] if agent else None
