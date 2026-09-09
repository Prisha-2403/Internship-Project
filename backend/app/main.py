"""FastAPI application factory.

Wires middleware, CORS, exception handlers and the v1 router, and exposes
OpenAPI docs at ``/docs``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import SecurityHeadersMiddleware

logger = logging.getLogger("sentinel")

DESCRIPTION = """
Internal transaction monitoring and investigation platform.

**This is a decision-support system.** Scores and anomaly signals indicate that
a transaction warrants review. They are not a determination that fraud has
occurred, and nothing in this API should be presented as one.

* **Business risk score** (0-100) - the sum of named rules that fired, always
  fully attributable to the factors shown alongside it.
* **ML anomaly score** (0-1) - how unusual a transaction looks to an Isolation
  Forest, relative to the wider population. It is not a fraud probability.

All demo data is synthetic. No real customer information is present.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging("DEBUG" if settings.debug else "INFO")
    logger.info(
        "%s starting (environment=%s, docs=%s)",
        settings.app_name,
        settings.environment,
        "enabled" if not settings.is_production else "disabled",
    )
    # Fail fast on a missing signing key rather than at the first login.
    try:
        settings.require_secret_key()
    except RuntimeError as exc:
        logger.error("Configuration error: %s", exc)
        raise
    yield
    logger.info("%s shutting down", settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        # Interactive docs are a development affordance, not a production one.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
        contact={"name": "Sentinel Finance Risk Engineering"},
        license_info={"name": "Internal use"},
    )

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,  # required for the httpOnly refresh cookie
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
        expose_headers=["Content-Disposition"],  # so the browser can name downloads
        max_age=600,
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["System"], summary="Liveness probe")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    return app


app = create_app()
