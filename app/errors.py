"""Consistent error responses and global exception handling.

All errors are returned with a uniform envelope so clients can rely on a stable
shape regardless of where the error originated::

    {"error": {"code": "not_found", "message": "...", "request_id": "abc123"}}
"""

from __future__ import annotations

import logging
from typing import cast

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

_CODE_BY_STATUS = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}


def _request_id(request: Request) -> str | None:
    return request.headers.get("x-request-id")


def error_response(
    status_code: int, message: str, request_id: str | None, *, code: str | None = None
) -> JSONResponse:
    body = {
        "error": {
            "code": code or _CODE_BY_STATUS.get(status_code, "error"),
            "message": message,
            "request_id": request_id,
        }
    }
    headers = {"x-request-id": request_id} if request_id else None
    return JSONResponse(status_code=status_code, content=body, headers=headers)


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    http_exc = cast(StarletteHTTPException, exc)
    response = error_response(
        http_exc.status_code, str(http_exc.detail), _request_id(request)
    )
    if http_exc.headers:
        response.headers.update(http_exc.headers)
    return response


async def _validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    return error_response(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "Request validation failed.",
        _request_id(request),
        code="validation_error",
    )


async def _unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "An internal error occurred.",
        _request_id(request),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install the uniform error handlers on the application."""

    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
