"""Logging setup with redaction of sensitive values.

Financial applications must not leak credentials into logs. The filter below is
a backstop: application code should never log a secret in the first place, but
if one reaches a log record it is masked before it is written.
"""

from __future__ import annotations

import logging
import re
import sys

# Matches ``password=...``/``"token": "..."`` style pairs in a formatted message.
_SENSITIVE_PATTERN = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|authorization|api[_-]?key|refresh_token"
    r"|access_token|password_hash)\b\s*[:=]\s*[\"']?([^\s,;\"'}]+)"
)

# Bearer tokens and bcrypt digests wherever they appear.
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+")
_BCRYPT_PATTERN = re.compile(r"\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}")

REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    """Mask credential-looking substrings in ``text``.

    Order matters. ``Authorization: Bearer <token>`` matches the key/value
    pattern too, but that pattern would consume only the word ``Bearer`` and
    leave the token exposed - so the bearer rule runs first.
    """
    text = _BEARER_PATTERN.sub(f"Bearer {REDACTED}", text)
    text = _BCRYPT_PATTERN.sub(REDACTED, text)
    text = _SENSITIVE_PATTERN.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    return text


class RedactingFilter(logging.Filter):
    """Applies :func:`redact` to every record before it is emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Install a single stderr handler with redaction enabled."""
    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(RedactingFilter())
    root.addHandler(handler)

    # SQLAlchemy echoes bound parameters at INFO, which can include customer
    # data; keep it at WARNING regardless of the app log level.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
