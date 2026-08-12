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
from typing import TYPE_CHECKING

from app.config import settings
from app.nlu.normalize import normalize

if TYPE_CHECKING:
    from sqlalchemy import RowMapping

logger = logging.getLogger(__name__)


def _name_matches(needle: str, name: object, name_ar: object) -> bool:
    """True when the needle appears in either name column as whole word(s).

    Both sides are run through :func:`normalize`, so Arabic letter-form variants
    (e.g. alef with/without hamza), diacritics, and casing all collapse before
    the comparison. Matching stops at word boundaries so a fragment can never
    stand in for a person: "no" must not reach "Mohammed Nour".
    """

    pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)")
    for value in (name, name_ar):
        if value and pattern.search(normalize(str(value))):
            return True
    return False


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


def _row_to_hit(r: RowMapping) -> BeneficiaryHit:
    """Map a beneficiaries table row (keyed by column name) to a hit."""

    return BeneficiaryHit(
        id=str(r["id"]),
        name=str(r["name"]),
        account=str(r["account"]),
        bank=(str(r["bank"]) if r["bank"] is not None else None),
        currency=str(r["currency"] or "SAR"),
        name_ar=(str(r["name_ar"]) if r["name_ar"] is not None else None),
        is_favorite=bool(r["is_favorite"]),
    )


class BeneficiaryDirectory:
    """SQLAlchemy-backed name lookup over the beneficiaries table."""

    def __init__(self, url: str, table: str, owner_column: str) -> None:
        from sqlalchemy import create_engine

        self._table = _safe_identifier(table, "table")
        self._owner_column = _safe_identifier(owner_column, "owner column")
        if url.startswith("sqlite"):
            self._engine = create_engine(
                url, pool_pre_ping=True, connect_args={"check_same_thread": False}
            )
        else:
            # Server-backed (Postgres): recycle pooled connections so an idle
            # worker doesn't reuse one the server has already dropped.
            self._engine = create_engine(url, pool_pre_ping=True, pool_recycle=1800)

    def search(self, name: str, owner_user: str | None) -> list[BeneficiaryHit] | None:
        """Return beneficiaries whose name contains ``name`` as whole word(s).

        Matching is case-insensitive and works on the first token, so "Ahmed"
        matches "Ahmed Hassan", "Ahmed Khaled", ... (both EN ``name`` and AR
        ``name_ar`` columns are searched), while a fragment of a name ("no")
        matches nobody.

        Returns ``None`` when the directory could not be queried (DB/table
        unavailable) so the caller can fall back to the free-text recipient,
        versus ``[]`` when the query ran but matched nobody.
        """

        from sqlalchemy import text

        needle = normalize(name)
        if not needle:
            return None
        owner_clause = ""
        params: dict[str, object] = {}
        if owner_user:
            owner_clause = f"WHERE {self._owner_column} = :owner"
            params["owner"] = owner_user
        query = text(
            f"SELECT id, name, name_ar, account, bank, currency, "  # noqa: S608
            f"is_favorite FROM {self._table} {owner_clause}"
        )
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(query, params).mappings().all()
        except Exception as exc:  # noqa: BLE001 - DB issues must not break a turn
            logger.warning("Beneficiary directory lookup failed for %r: %s", name, exc)
            return None
        # Match on the normalized form so Arabic alef/hamza variants (احمد vs أحمد),
        # diacritics, and casing all collapse to the same key before comparing.
        return [
            _row_to_hit(r)
            for r in rows
            if _name_matches(needle, r["name"], r["name_ar"])
        ]

    def list_all(self, owner_user: str | None) -> list[BeneficiaryHit] | None:
        """Return every saved beneficiary for ``owner_user`` (favorites first).

        Returns ``None`` when the directory could not be queried (DB/table
        unavailable), versus ``[]`` when the owner has no beneficiaries saved.
        """

        from sqlalchemy import text

        owner_clause = ""
        params: dict[str, object] = {}
        if owner_user:
            owner_clause = f"WHERE {self._owner_column} = :owner"
            params["owner"] = owner_user
        query = text(
            f"SELECT id, name, name_ar, account, bank, currency, "  # noqa: S608
            f"is_favorite FROM {self._table} {owner_clause}"
        )
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(query, params).mappings().all()
        except Exception as exc:  # noqa: BLE001 - DB issues must not break a turn
            logger.warning("Beneficiary directory list failed: %s", exc)
            return None
        hits = [_row_to_hit(r) for r in rows]
        hits.sort(key=lambda h: (not h.is_favorite, h.name.lower()))
        return hits


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
