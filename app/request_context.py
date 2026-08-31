"""Per-request context: one id shared by logs, audit events and turn rows.

A ``ContextVar`` holds the current request id so any log record emitted while
handling a request can be correlated with the audit event for that same request.

When the caller sends a W3C ``traceparent`` the trace id in it is adopted as the
request id, and a fresh span id is minted for this service. That is what lets one
customer request be followed across the conversation layer and the Banking Core
in whatever APM the bank already runs, instead of each service inventing its own
identifier. Without the header nothing changes: an inbound ``X-Request-ID`` is
honoured as before, and a random id is generated otherwise.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

REQUEST_ID_HEADER = "x-request-id"
TRACEPARENT_HEADER = "traceparent"

# version-traceid-spanid-flags, version 00 (RFC: unknown versions are ignored).
_TRACEPARENT = re.compile(
    r"^00-(?P<trace>[0-9a-f]{32})-(?P<span>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)
_ALL_ZERO_TRACE = "0" * 32
_ALL_ZERO_SPAN = "0" * 16


@dataclass(frozen=True, slots=True)
class TraceContext:
    """The trace this request belongs to, and this service's place in it."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    flags: str

    def header(self) -> str:
        """This service's ``traceparent``, for a call it makes onwards."""

        return f"00-{self.trace_id}-{self.span_id}-{self.flags}"


_trace: ContextVar[TraceContext | None] = ContextVar("trace_context", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def _new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def get_request_id() -> str | None:
    return _request_id.get()


def set_request_id(value: str | None) -> None:
    _request_id.set(value)


def get_trace_context() -> TraceContext | None:
    return _trace.get()


def outbound_traceparent() -> str | None:
    """The header to attach to a downstream call, if this request has a trace."""

    context = _trace.get()
    return context.header() if context else None


def parse_traceparent(value: str | None) -> TraceContext | None:
    """Read an inbound ``traceparent``, or ``None`` if it is absent or invalid.

    An unparseable or all-zero header is treated as no trace at all rather than
    as a trace to join: a malformed id propagated onwards is worse than a new one.
    """

    if not value:
        return None
    match = _TRACEPARENT.match(value.strip().lower())
    if not match:
        return None
    trace_id = match.group("trace")
    parent_span = match.group("span")
    if trace_id == _ALL_ZERO_TRACE or parent_span == _ALL_ZERO_SPAN:
        return None
    return TraceContext(
        trace_id=trace_id,
        span_id=_new_span_id(),
        parent_span_id=parent_span,
        flags=match.group("flags"),
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id per request, joining an inbound W3C trace if sent."""

    async def dispatch(self, request: Request, call_next) -> Response:
        trace = parse_traceparent(request.headers.get(TRACEPARENT_HEADER))
        request_id = (
            trace.trace_id
            if trace
            else request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        )
        token = _request_id.set(request_id)
        trace_token = _trace.set(trace)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        finally:
            _request_id.reset(token)
            _trace.reset(trace_token)
        response.headers[REQUEST_ID_HEADER] = request_id
        if trace:
            response.headers[TRACEPARENT_HEADER] = trace.header()
        return response
