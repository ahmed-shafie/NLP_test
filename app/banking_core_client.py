"""HTTP client for the standalone Banking Core service.

Used for the API-backed cases: account balance, pre-flight funds/FX checks, and
adding a beneficiary. Every call degrades gracefully — on any network/HTTP error
it returns ``None`` (or an unavailable result) so a turn is never broken by the
external service being down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

import httpx

from app.config import settings
from app.observability import signals
from app.request_context import outbound_traceparent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccountInfo:
    account_id: str
    account_type: str
    number: str
    currency: str
    balance: Decimal
    status: str


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    source_account: AccountInfo | None = None
    warnings: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.banking_core_api_key:
        headers["x-api-key"] = settings.banking_core_api_key
    # Continue the caller's trace into the Core so one request is followable
    # across both services.
    traceparent = outbound_traceparent()
    if traceparent:
        headers["traceparent"] = traceparent
    return headers


def _post(path: str, payload: dict) -> dict | None:
    if not settings.banking_core_enabled:
        return None
    url = f"{settings.banking_core_url.rstrip('/')}{path}"
    try:
        resp = httpx.post(
            url, json=payload, headers=_headers(), timeout=settings.banking_core_timeout
        )
    except Exception as exc:  # noqa: BLE001 - service optional; never break a turn
        logger.warning("Banking Core call to %s failed: %s", path, exc)
        signals.record_call(signals.BANKING_CORE, ok=False)
        return None
    # A 4xx is the Core answering — an unknown account is an answer, not an
    # outage — so only a 5xx or an unreachable service counts against health.
    signals.record_call(signals.BANKING_CORE, ok=resp.status_code < 500)
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        logger.warning("Banking Core %s -> HTTP %s", path, resp.status_code)
        return None
    return resp.json()


def health() -> bool:
    """Return ``True`` if the Banking Core /health endpoint responds ok."""

    url = f"{settings.banking_core_url.rstrip('/')}/health"
    try:
        resp = httpx.get(url, timeout=settings.banking_core_timeout)
    except Exception as exc:  # noqa: BLE001 - report unreachable, never raise
        logger.warning("Banking Core health check failed: %s", exc)
        signals.record_call(signals.BANKING_CORE, ok=False)
        return False
    signals.record_call(signals.BANKING_CORE, ok=resp.status_code == 200)
    return resp.status_code == 200


def _to_account(data: dict | None) -> AccountInfo | None:
    if not data:
        return None
    return AccountInfo(
        account_id=str(data["account_id"]),
        account_type=str(data["account_type"]),
        number=str(data["number"]),
        currency=str(data["currency"]),
        balance=Decimal(str(data["balance"])),
        status=str(data["status"]),
    )


def get_balance(
    owner_user: str, account: str | None = None, account_type: str | None = None
) -> AccountInfo | None:
    data = _post(
        "/accounts/balance",
        {"owner_user": owner_user, "account": account, "account_type": account_type},
    )
    return _to_account(data)


def list_accounts(owner_user: str) -> list[AccountInfo]:
    """List the customer's accounts, in the order the Core returns them.

    An empty list means "no accounts to choose from" — either the Core is
    disabled/unreachable or the customer has none — so the caller must not read
    it as "the customer has no money".
    """

    data = _post("/accounts/list", {"owner_user": owner_user})
    if not data:
        return []
    accounts = [_to_account(row) for row in data.get("accounts") or []]
    return [a for a in accounts if a is not None]


def _to_preflight(data: dict | None) -> PreflightResult | None:
    if data is None:
        return None
    return PreflightResult(
        ok=bool(data.get("ok")),
        source_account=_to_account(data.get("source_account")),
        warnings=list(data.get("warnings") or []),
        blocking=list(data.get("blocking") or []),
    )


def preflight_transfer(
    owner_user: str,
    amount: Decimal,
    currency: str,
    recipient_account: str | None = None,
    source_account: str | None = None,
    source_account_type: str | None = None,
) -> PreflightResult | None:
    data = _post(
        "/preflight/transfer",
        {
            "owner_user": owner_user,
            "amount": str(amount),
            "currency": currency,
            "recipient_account": recipient_account,
            "source_account": source_account,
            "source_account_type": source_account_type,
        },
    )
    return _to_preflight(data)


def preflight_bill(
    owner_user: str,
    amount: Decimal,
    currency: str,
    biller_code: str | None,
    reference_number: str,
    source_account: str | None = None,
    source_account_type: str | None = None,
) -> PreflightResult | None:
    data = _post(
        "/preflight/bill",
        {
            "owner_user": owner_user,
            "amount": str(amount),
            "currency": currency,
            "biller_code": biller_code,
            "reference_number": reference_number,
            "source_account": source_account,
            "source_account_type": source_account_type,
        },
    )
    return _to_preflight(data)


def add_beneficiary(
    owner_user: str,
    name: str,
    account: str,
    bank: str | None = None,
    currency: str = "SAR",
) -> dict | None:
    return _post(
        "/beneficiary/add",
        {
            "owner_user": owner_user,
            "name": name,
            "account": account,
            "bank": bank,
            "currency": currency,
        },
    )
