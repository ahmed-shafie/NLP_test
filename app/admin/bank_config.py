"""Per-case Banking Core configuration, persisted in the admin store.

The chat/NLU layer reads its Banking Core settings from the global ``settings``
object. This module lets the admin GUI edit those settings per case (transfer /
pay bill / beneficiary lookup / add beneficiary) and persists them so they
survive restarts. Saving applies the values to ``settings`` immediately and
clears the cached beneficiary directory so the next turn uses them.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from app.admin.store import AppSetting, get_sessionmaker
from app.config import settings

logger = logging.getLogger(__name__)

_KEY = "banking_core"


class BankingCoreConfig(BaseModel):
    """Editable Banking Core connection settings (API cases + DB lookup case)."""

    # Shared external API (transfer / pay-bill / balance / add-beneficiary).
    api_enabled: bool = Field(default=False)
    api_url: str = Field(default="http://localhost:8100")
    api_timeout: float = Field(default=10.0, gt=0)
    # Whether an API key is configured (the value itself is never returned).
    api_key_set: bool = Field(default=False)

    # Beneficiary lookup case: direct DB read (no API).
    lookup_enabled: bool = Field(default=False)
    lookup_db_url: str = Field(default="")
    lookup_table: str = Field(default="beneficiaries")
    lookup_owner_column: str = Field(default="owner_user")


class BankingCoreConfigUpdate(BaseModel):
    """Partial update; ``api_key`` is write-only (blank/omitted leaves it as-is)."""

    api_enabled: bool | None = None
    api_url: str | None = None
    api_timeout: float | None = None
    api_key: str | None = None
    lookup_enabled: bool | None = None
    lookup_db_url: str | None = None
    lookup_table: str | None = None
    lookup_owner_column: str | None = None


def current_config() -> BankingCoreConfig:
    """Return the live config as reflected on the global ``settings`` object."""

    return BankingCoreConfig(
        api_enabled=settings.banking_core_enabled,
        api_url=settings.banking_core_url,
        api_timeout=settings.banking_core_timeout,
        api_key_set=bool(settings.banking_core_api_key),
        lookup_enabled=settings.beneficiary_lookup_enabled,
        lookup_db_url=settings.beneficiary_db_url,
        lookup_table=settings.beneficiary_table,
        lookup_owner_column=settings.beneficiary_owner_column,
    )


def _apply(data: dict[str, object]) -> None:
    """Overlay a persisted/updated blob onto the global settings object."""

    if "api_enabled" in data:
        settings.banking_core_enabled = bool(data["api_enabled"])
    if "api_url" in data and data["api_url"]:
        settings.banking_core_url = str(data["api_url"])
    if "api_timeout" in data and data["api_timeout"]:
        settings.banking_core_timeout = float(data["api_timeout"])  # type: ignore[arg-type]
    if data.get("api_key"):
        settings.banking_core_api_key = str(data["api_key"])
    if "lookup_enabled" in data:
        settings.beneficiary_lookup_enabled = bool(data["lookup_enabled"])
    if "lookup_db_url" in data and data["lookup_db_url"]:
        settings.beneficiary_db_url = str(data["lookup_db_url"])
    if "lookup_table" in data and data["lookup_table"]:
        settings.beneficiary_table = str(data["lookup_table"])
    if "lookup_owner_column" in data and data["lookup_owner_column"]:
        settings.beneficiary_owner_column = str(data["lookup_owner_column"])
    _reset_directory()


def _reset_directory() -> None:
    from app.db.directory import get_beneficiary_directory

    get_beneficiary_directory.cache_clear()


def load_persisted_into_settings() -> None:
    """On startup, overlay any persisted Banking Core config onto ``settings``."""

    with get_sessionmaker()() as session:
        row = session.get(AppSetting, _KEY)
        if row is None:
            return
        try:
            data = json.loads(row.value or "{}")
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed persisted banking_core config")
            return
    _apply(data)


def save_config(update: BankingCoreConfigUpdate) -> BankingCoreConfig:
    """Persist a partial update, apply it to ``settings``, and return the result."""

    data = update.model_dump(exclude_none=True)
    with get_sessionmaker()() as session:
        row = session.get(AppSetting, _KEY)
        stored: dict[str, object] = {}
        if row is not None:
            try:
                stored = json.loads(row.value or "{}")
            except json.JSONDecodeError:
                stored = {}
        stored.update(data)
        payload = json.dumps(stored)
        if row is None:
            session.add(AppSetting(key=_KEY, value=payload))
        else:
            row.value = payload
        session.commit()
    _apply(data)
    return current_config()
