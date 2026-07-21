"""Direct beneficiary lookup by name (read path for transfer disambiguation).

For a transfer, the recipient name is resolved by querying the beneficiaries table
*directly* (not via the Banking Core API). A first-name match that returns several
people triggers a "which one?" disambiguation in the conversation engine.

Degrades gracefully: if the directory is disabled or the DB is unreachable, the
lookup returns an empty list and the engine keeps the free-text recipient.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

# Table / column names are interpolated into the lookup SQL, so they must be
# plain SQL identifiers (optionally schema-qualified). This rejects anything
# that could inject SQL via the admin-configurable settings.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _safe_identifier(value: str, kind: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid {kind} identifier: {value!r}")
    return value


@dataclass(frozen=True)
class BeneficiaryHit:
    """A beneficiary row matched by name."""

    id: str
    name: str
    account: str
    bank: str | None = None
    currency: str = "SAR"
    name_ar: str | None = None
    is_favorite: bool = False


class BeneficiaryDirectory:
    """SQLAlchemy-backed name lookup over the beneficiaries table."""

    def __init__(self, url: str, table: str, owner_column: str) -> None:
        from sqlalchemy import create_engine

        self._table = _safe_identifier(table, "table")
        self._owner_column = _safe_identifier(owner_column, "owner column")
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)

    def search(self, name: str, owner_user: str | None) -> list[BeneficiaryHit] | None:
        """Return beneficiaries whose name starts with / contains ``name``.

        Matching is case-insensitive and works on the first token, so "Ahmed"
        matches "Ahmed Hassan", "Ahmed Khaled", ... (both EN ``name`` and AR
        ``name_ar`` columns are searched).

        Returns ``None`` when the directory could not be queried (DB/table
        unavailable) so the caller can fall back to the free-text recipient,
        versus ``[]`` when the query ran but matched nobody.
        """

        from sqlalchemy import text

        needle = name.strip()
        if not needle:
            return None
        prefix = f"{needle.lower()}%"
        contains = f"%{needle.lower()}%"
        owner_clause = ""
        params: dict[str, object] = {"prefix": prefix, "contains": contains}
        if owner_user:
            owner_clause = f"AND {self._owner_column} = :owner"
            params["owner"] = owner_user
        query = text(
            f"SELECT id, name, name_ar, account, bank, currency, "  # noqa: S608
            f"COALESCE(is_favorite, 0) AS is_favorite "
            f"FROM {self._table} "
            f"WHERE (lower(name) LIKE :prefix OR lower(name) LIKE :contains "
            f"OR lower(COALESCE(name_ar,'')) LIKE :prefix "
            f"OR lower(COALESCE(name_ar,'')) LIKE :contains) {owner_clause}"
        )
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(query, params).mappings().all()
        except Exception as exc:  # noqa: BLE001 - DB issues must not break a turn
            logger.warning("Beneficiary directory lookup failed for %r: %s", name, exc)
            return None
        return [
            BeneficiaryHit(
                id=str(r["id"]),
                name=str(r["name"]),
                account=str(r["account"]),
                bank=(str(r["bank"]) if r["bank"] is not None else None),
                currency=str(r["currency"] or "SAR"),
                name_ar=(str(r["name_ar"]) if r["name_ar"] is not None else None),
                is_favorite=bool(r["is_favorite"]),
            )
            for r in rows
        ]


@lru_cache(maxsize=1)
def get_beneficiary_directory() -> BeneficiaryDirectory | None:
    """Build the directory once, or ``None`` when disabled/unavailable."""

    if not settings.beneficiary_lookup_enabled or not settings.beneficiary_db_url:
        return None
    try:
        return BeneficiaryDirectory(
            url=settings.beneficiary_db_url,
            table=settings.beneficiary_table,
            owner_column=settings.beneficiary_owner_column,
        )
    except Exception as exc:  # noqa: BLE001 - missing driver / bad URL
        logger.warning("Beneficiary directory unavailable (%s).", exc)
        return None
