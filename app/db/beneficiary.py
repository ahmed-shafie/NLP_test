"""Configurable beneficiary account lookup.

Resolves the destination beneficiary by account number against a SQL database. The
provider (PostgreSQL, Oracle, SQL Server, Impala, Hive, SQLite, ...) is selected by
the SQLAlchemy URL, and the connection, lookup query, and column-to-field mapping are
all driven from :mod:`app.config` — so switching providers needs no code changes.

The repository degrades gracefully: if the database is disabled, the URL/driver is
missing, or a query fails, :func:`get_beneficiary_repository` returns ``None`` (or a
lookup returns ``None``) and the caller falls back to the LiteLLM handler.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import settings
from app.schemas import Beneficiary

logger = logging.getLogger(__name__)

# Beneficiary fields that may be populated from a result row.
_BENEFICIARY_FIELDS = ("id", "name", "account", "bank", "branch", "currency")


class BeneficiaryRepository:
    """SQLAlchemy-backed, provider-agnostic beneficiary lookup."""

    def __init__(
        self,
        url: str,
        query: str,
        account_param: str,
        column_map: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        from sqlalchemy import create_engine, text

        self._text = text
        self._query = text(query)
        self._account_param = account_param
        # Invert config mapping (result-column -> field) to (field -> result-column).
        self._field_to_column = dict(column_map or {})
        self._engine = create_engine(
            url,
            pool_pre_ping=True,
            connect_args=self._connect_args(url, timeout),
        )

    @staticmethod
    def _connect_args(url: str, timeout: float) -> dict:
        """Best-effort per-driver connect timeout (ignored where unsupported)."""

        seconds = int(timeout)
        if url.startswith("postgresql"):
            return {"connect_timeout": seconds}
        if url.startswith("mysql"):
            return {"connect_timeout": seconds}
        return {}

    def _row_to_beneficiary(self, mapping: dict) -> Beneficiary | None:
        """Map a result row to a :class:`Beneficiary` using the configured columns."""

        values: dict[str, str] = {}
        for field in _BENEFICIARY_FIELDS:
            column = self._field_to_column.get(field, field)
            if column in mapping and mapping[column] is not None:
                values[field] = str(mapping[column])
        if "name" not in values:
            logger.warning(
                "Beneficiary row missing a 'name' column (looked for %r). "
                "Check NLU_DB_QUERY / NLU_DB_COLUMN_MAP.",
                self._field_to_column.get("name", "name"),
            )
            return None
        return Beneficiary(**values)

    def lookup(self, account_number: str) -> Beneficiary | None:
        """Return the beneficiary for ``account_number``, or ``None`` if not found."""

        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    self._query, {self._account_param: account_number}
                )
                row = result.mappings().first()
        except Exception as exc:  # noqa: BLE001 - DB issues must not break the request
            logger.warning("Beneficiary lookup failed for %r: %s", account_number, exc)
            return None
        if row is None:
            return None
        return self._row_to_beneficiary(dict(row))


@lru_cache(maxsize=1)
def get_beneficiary_repository() -> BeneficiaryRepository | None:
    """Build the configured repository once, or ``None`` if disabled/unavailable."""

    if not settings.db_enabled:
        return None
    if not settings.db_url:
        logger.warning("NLU_DB_ENABLED is true but NLU_DB_URL is not set.")
        return None
    try:
        return BeneficiaryRepository(
            url=settings.db_url,
            query=settings.db_query,
            account_param=settings.db_account_param,
            column_map=settings.db_column_map,
            timeout=settings.db_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - missing driver, bad URL, etc.
        logger.warning(
            "Beneficiary repository unavailable (%s); falling back to LLM.", exc
        )
        return None
