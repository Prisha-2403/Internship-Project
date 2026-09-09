"""Application error types and the handlers that render them.

Users receive a short, professional message and a stable machine-readable code.
Stack traces and database detail go to the server log only - never the wire.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("sentinel.errors")


class AppError(Exception):
    """Base class for errors that are safe to show a user."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "APP_ERROR"
    message: str = "The request could not be completed."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.details = details or {}
        super().__init__(self.message)


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "NOT_FOUND"
    message = "The requested record could not be found."


class ValidationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "VALIDATION_ERROR"
    message = "Some of the submitted values are not valid. Please review and try again."


class AuthenticationError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "AUTHENTICATION_FAILED"
    message = "Authentication failed. Please sign in again."


class PermissionError_(AppError):
    """Named with a trailing underscore so it does not shadow the builtin."""

    status_code = status.HTTP_403_FORBIDDEN
    code = "PERMISSION_DENIED"
    message = "Your role does not permit this action."


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "CONFLICT"
    message = "This action conflicts with the current state of the record."


class RateLimitError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "RATE_LIMITED"
    message = "Too many attempts. Please wait before trying again."


class ImportError_(AppError):
    """Raised when an uploaded file cannot be parsed or validated."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "IMPORT_FAILED"
    message = (
        "Unable to process transaction data. Please verify the uploaded file and try again."
    )


def _payload(
    code: str, message: str, details: dict[str, Any] | None = None, trace_id: str | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    if trace_id:
        body["error"]["traceId"] = trace_id
    return body


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to ``app``."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Surface field paths and messages, but never the submitted values -
        # those can contain credentials or customer data.
        fields = [
            {
                "field": ".".join(str(p) for p in err.get("loc", ()) if p != "body"),
                "message": err.get("msg", "Invalid value."),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_payload(
                "VALIDATION_ERROR",
                "Some of the submitted values are not valid. Please review and try again.",
                {"fields": fields[:20]},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {
            401: "AUTHENTICATION_FAILED",
            403: "PERMISSION_DENIED",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            429: "RATE_LIMITED",
        }
        detail = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(codes.get(exc.status_code, "HTTP_ERROR"), detail),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(IntegrityError)
    async def _integrity(request: Request, exc: IntegrityError) -> JSONResponse:
        trace_id = uuid.uuid4().hex[:12]
        logger.warning(
            "Integrity error [%s] on %s %s: %s",
            trace_id,
            request.method,
            request.url.path,
            exc.orig,
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=_payload(
                "CONFLICT",
                "This action conflicts with existing data. The record may already exist.",
                trace_id=trace_id,
            ),
        )

    @app.exception_handler(SQLAlchemyError)
    async def _database(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        trace_id = uuid.uuid4().hex[:12]
        logger.exception(
            "Database error [%s] on %s %s", trace_id, request.method, request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_payload(
                "DATABASE_ERROR",
                "A data storage error occurred. The action was not completed.",
                trace_id=trace_id,
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        trace_id = uuid.uuid4().hex[:12]
        logger.exception(
            "Unhandled error [%s] on %s %s", trace_id, request.method, request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_payload(
                "INTERNAL_ERROR",
                "An unexpected error occurred. The issue has been logged for investigation.",
                trace_id=trace_id,
            ),
        )
