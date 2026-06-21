"""In-process fixed-window rate limiting per client IP.

Opt-in via ``NLU_RATE_LIMIT_ENABLED``. This is a lightweight, single-process limiter
suitable for a single instance; for multi-instance deployments use a shared store
(e.g. Redis) behind the same interface.
"""

from __future__ import annotations

import threading
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings
from app.errors import error_response
from app.request_context import get_request_id

# Only the public NLU surface is limited; admin/metrics/health are exempt.
_LIMITED_PREFIXES = (
    "/nlu",
    "/transfer",
    "/contacts",
    "/v1/nlu",
    "/v1/transfer",
    "/v1/contacts",
)


class _FixedWindow:
    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            count, start = self._hits.get(key, (0, now))
            if now - start >= self.window:
                count, start = 0, now
            count += 1
            self._hits[key] = (count, start)
            return count <= self.limit


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)
        self._limiter = _FixedWindow(settings.rate_limit_per_minute)

    async def dispatch(self, request: Request, call_next) -> Response:
        if settings.rate_limit_enabled and request.url.path.startswith(
            _LIMITED_PREFIXES
        ):
            self._limiter.limit = settings.rate_limit_per_minute
            client = request.client.host if request.client else "unknown"
            if not self._limiter.allow(client):
                return error_response(
                    429,
                    "Rate limit exceeded. Please retry later.",
                    get_request_id(),
                    code="rate_limited",
                )
        return await call_next(request)
