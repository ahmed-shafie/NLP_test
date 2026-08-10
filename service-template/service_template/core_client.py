"""Thin HTTP client for an external "core" validation service (pre-flight).

Modelled on ``app/banking_core_client.py``. In the main app the NLU layer never
executes a transaction; before confirming it asks a separate service whether the
action is fundable / valid, and shows any *advisory* warnings (low funds, FX).

This client:
* is a no-op when ``settings.core_enabled`` is false (the template still runs);
* returns a typed ``PreflightResult`` your engine can act on;
* treats network/HTTP errors as "no opinion" (returns ``None``) so a flaky core
  service never hard-blocks the conversation — matching the product rule that
  pre-flight is advisory.

Point ``SVC_CORE_BASE_URL`` at the ``banking-core`` service (or your own) whose
``POST /preflight/transfer`` returns ``{ok, warnings, blocking}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import httpx

from service_template.config import settings


@dataclass
class PreflightResult:
    """Outcome of a pre-flight check.

    * ``ok`` — no hard blocks.
    * ``warnings`` — advisory notes shown to the user; they DO NOT block.
    * ``blocking`` — hard stops (e.g. account inactive) that SHOULD block.
    """

    ok: bool
    warnings: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)


def _headers() -> dict[str, str]:
    return {"x-api-key": settings.core_api_key} if settings.core_api_key else {}


def preflight_transfer(
    owner_user: str,
    amount: Decimal,
    currency: str,
    source_account: str | None,
) -> PreflightResult | None:
    """Ask the core service to validate a pending transfer.

    Returns ``None`` when pre-flight is disabled or the service is unreachable,
    which the engine interprets as "proceed without extra checks".
    """

    if not settings.core_enabled:
        return None
    payload = {
        "owner_user": owner_user,
        "amount": str(amount),
        "currency": currency,
        "source_account": source_account,
    }
    try:
        response = httpx.post(
            f"{settings.core_base_url.rstrip('/')}/preflight/transfer",
            json=payload,
            headers=_headers(),
            timeout=settings.core_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        # Network error / bad JSON / non-2xx: no opinion, let the flow continue.
        return None
    return PreflightResult(
        ok=bool(data.get("ok", True)),
        warnings=list(data.get("warnings", [])),
        blocking=list(data.get("blocking", [])),
    )


def health() -> bool:
    """Return True if the core service answers ``GET /health`` with 2xx."""

    if not settings.core_enabled:
        return False
    try:
        response = httpx.get(
            f"{settings.core_base_url.rstrip('/')}/health",
            headers=_headers(),
            timeout=settings.core_timeout_seconds,
        )
        return response.is_success
    except httpx.HTTPError:
        return False
