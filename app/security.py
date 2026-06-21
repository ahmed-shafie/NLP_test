"""API access control: API-key authentication for public and admin endpoints.

Authentication is opt-in via ``NLU_AUTH_ENABLED``. When disabled (the default for
local development) the dependencies are no-ops so the service behaves as before.
When enabled, requests must carry a valid key in the relevant header and the service
fails closed (HTTP 503) if authentication is enabled but no keys are configured.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
ADMIN_KEY_HEADER = "X-Admin-Key"


def _matches(provided: str | None, allowed: list[str]) -> bool:
    if not provided:
        return False
    return any(secrets.compare_digest(provided, key) for key in allowed)


def _authorize(provided: str | None, allowed: list[str], scope: str) -> str:
    if not settings.auth_enabled:
        return "anonymous"
    if not allowed:
        logger.error("Authentication enabled but no %s keys configured", scope)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Authentication is enabled but no {scope} API keys are configured.",
        )
    if not _matches(provided, allowed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return scope


def require_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> str:
    """Guard public endpoints (/nlu/*, /transfer/*, /contacts/*)."""

    return _authorize(x_api_key, settings.api_keys_list(), "api")


def require_admin_key(
    x_admin_key: str | None = Header(default=None, alias=ADMIN_KEY_HEADER),
) -> str:
    """Guard admin endpoints (/admin/api/*)."""

    return _authorize(x_admin_key, settings.admin_api_keys_list(), "admin")
