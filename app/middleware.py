"""Cross-cutting HTTP middleware: request size limits."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds ``settings.max_request_bytes``."""

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        limit = settings.max_request_bytes
        if content_length is not None:
            try:
                if int(content_length) > limit:
                    return self._too_large(limit)
            except ValueError:
                pass
        return await call_next(request)

    @staticmethod
    def _too_large(limit: int) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "code": "payload_too_large",
                    "message": f"Request body exceeds the {limit} byte limit.",
                    "request_id": None,
                }
            },
        )
