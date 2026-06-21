"""Prometheus metrics: request counts and latency, exposed at GET /metrics."""

from __future__ import annotations

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# A dedicated registry keeps test runs isolated and avoids duplicate-timeseries
# errors when the app module is imported more than once.
registry = CollectorRegistry()

REQUESTS = Counter(
    "nlu_http_requests_total",
    "Total HTTP requests.",
    labelnames=("method", "path", "status"),
    registry=registry,
)

LATENCY = Histogram(
    "nlu_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "path"),
    registry=registry,
)


def _route_template(request: Request) -> str:
    """Use the matched route template (not the raw path) to bound cardinality."""

    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else request.url.path


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            path = _route_template(request)
            elapsed = time.perf_counter() - start
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(request.method, path).observe(elapsed)


def metrics_response() -> Response:
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
