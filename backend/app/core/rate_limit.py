"""In-memory sliding-window rate limiting for authentication endpoints.

Deliberately process-local: this deployment runs a single API process, so a
Redis dependency would add operational weight without buying anything. If the
API is ever scaled horizontally, swap ``_WINDOWS`` for a shared store - the
public interface here does not need to change.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from app.core.config import settings


class SlidingWindowLimiter:
    """Allow at most ``max_attempts`` events per key within ``window_seconds``."""

    def __init__(self, max_attempts: int, window_seconds: int) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        window = self._windows[key]
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()
        return window

    def check(self, key: str) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)`` without recording an attempt."""
        now = time.monotonic()
        with self._lock:
            window = self._prune(key, now)
            if len(window) >= self.max_attempts:
                retry_after = int(self.window_seconds - (now - window[0])) + 1
                return False, max(retry_after, 1)
            return True, 0

    def record(self, key: str) -> None:
        """Record one failed attempt against ``key``."""
        now = time.monotonic()
        with self._lock:
            self._prune(key, now)
            self._windows[key].append(now)

    def reset(self, key: str) -> None:
        """Clear a key's history - called after a successful login."""
        with self._lock:
            self._windows.pop(key, None)

    def clear(self) -> None:
        """Drop all state. Used by tests."""
        with self._lock:
            self._windows.clear()


login_limiter = SlidingWindowLimiter(
    max_attempts=settings.login_rate_limit_attempts,
    window_seconds=settings.login_rate_limit_window_seconds,
)


def login_rate_limit_key(ip_address: str, email: str) -> str:
    """Bucket by IP *and* account.

    Keying on both means one attacker cannot lock out a legitimate user by
    spraying their address, while still capping attempts per source.
    """
    return f"{ip_address}|{email.strip().lower()}"
