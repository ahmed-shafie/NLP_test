"""Record one operational row per customer turn, and read it back.

What is written is deliberately thin. A turn row answers *what the layer decided*
— which flow, which slot it is waiting on, which ``ReasonCode`` stopped it, where
each filled slot came from — and nothing about the money itself. The amount, the
account, the beneficiary and the customer's own words stay in the Banking Core and
in the session; a report built on this table can count decisions but can never be
mistaken for a second source of financial truth.

The identifiers are salted digests, so one conversation can be followed end to end
without the table naming the customer.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select

from app.config import settings
from app.conversation.state import ConversationState
from app.observability.store import ConversationTurnRow, get_sessionmaker
from app.request_context import get_request_id

logger = logging.getLogger(__name__)

# Slots recorded as "filled, from this source" only. Their values (an amount, an
# IBAN, a person's name, a bill reference) are never copied here.
_TRACKED_SLOTS: tuple[str, ...] = (
    "amount",
    "currency",
    "recipient",
    "source_account",
    "account_number",
    "biller",
    "biller_category",
    "biller_code",
    "reference_number",
    "note",
)


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """A turn row as read back out of the store."""

    id: int
    timestamp: datetime
    trace_id: str | None
    session_ref: str
    customer_ref: str | None
    language: str
    intent: str | None
    status: str
    pending_slot: str | None
    reason_code: str | None
    latency_ms: float | None
    slots: dict[str, str]


# How many recent rows a windowed count scans. Timestamps are compared in
# Python (the store may be SQLite, where a stored offset is text), so the scan
# is bounded rather than open-ended.
_WINDOW_SCAN_LIMIT = 20_000


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _digest(value: str) -> str:
    """Salted digest of an identifier, stable within one deployment."""

    salted = f"{settings.turn_store_salt}:{value}".encode()
    return hashlib.sha256(salted).hexdigest()[:32]


def masked_slots(state: ConversationState) -> dict[str, str]:
    """Which slots are filled and where each came from — never their values."""

    values = state.slots.model_dump()
    filled: dict[str, str] = {}
    for slot in _TRACKED_SLOTS:
        value = values.get(slot)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        filled[slot] = state.slot_provenance.get(slot, "unknown")
    return filled


def record_turn(
    state: ConversationState,
    *,
    reason_code: str | None,
    latency_ms: float | None = None,
) -> None:
    """Persist one turn. Never raises: observability must not break a payment."""

    if not settings.turn_observability_enabled:
        return
    try:
        with get_sessionmaker()() as session:
            session.add(
                ConversationTurnRow(
                    timestamp=datetime.now(UTC),
                    trace_id=get_request_id(),
                    session_ref=_digest(state.session_id),
                    customer_ref=_digest(state.user_id) if state.user_id else None,
                    language=state.language.value,
                    intent=state.intent.value if state.intent else None,
                    status=state.status.value,
                    pending_slot=state.pending_slot,
                    reason_code=reason_code,
                    latency_ms=latency_ms,
                    slots_masked=json.dumps(masked_slots(state)),
                )
            )
            session.commit()
    except Exception as exc:  # noqa: BLE001 - a turn row is never worth a 500
        logger.warning("Failed to record conversation turn: %s", exc)


def _to_record(row: ConversationTurnRow) -> TurnRecord:
    try:
        slots = json.loads(row.slots_masked or "{}")
    except json.JSONDecodeError:
        slots = {}
    return TurnRecord(
        id=row.id,
        timestamp=_aware(row.timestamp),
        trace_id=row.trace_id,
        session_ref=row.session_ref,
        customer_ref=row.customer_ref,
        language=row.language,
        intent=row.intent,
        status=row.status,
        pending_slot=row.pending_slot,
        reason_code=row.reason_code,
        latency_ms=row.latency_ms,
        slots=slots,
    )


def list_turns(
    *,
    limit: int = 100,
    session_ref: str | None = None,
    reason_code: str | None = None,
) -> list[TurnRecord]:
    """Return recent turns, newest first."""

    with get_sessionmaker()() as session:
        stmt = select(ConversationTurnRow).order_by(ConversationTurnRow.id.desc())
        if session_ref:
            stmt = stmt.where(ConversationTurnRow.session_ref == session_ref)
        if reason_code:
            stmt = stmt.where(ConversationTurnRow.reason_code == reason_code)
        rows = session.scalars(stmt.limit(limit)).all()
        return [_to_record(row) for row in rows]


def session_ref_for(session_id: str) -> str:
    """The digest an operator can search a known session by."""

    return _digest(session_id)


def counts(window_minutes: int) -> dict[str, int]:
    """Turn counts within the window, keyed for the SLO catalogue.

    ``turns`` is the denominator; ``reason:<code>`` and ``status:<status>`` are
    the numerators, so a new ``ReasonCode`` needs no change here.
    """

    cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
    with get_sessionmaker()() as session:
        rows = session.scalars(
            select(ConversationTurnRow)
            .order_by(ConversationTurnRow.id.desc())
            .limit(_WINDOW_SCAN_LIMIT)
        ).all()

    recent = [row for row in rows if _aware(row.timestamp) >= cutoff]
    tally: Counter[str] = Counter({"turns": len(recent)})
    for row in recent:
        tally[f"status:{row.status}"] += 1
        if row.reason_code:
            tally[f"reason:{row.reason_code}"] += 1
    return dict(tally)


def purge_older_than(days: int) -> int:
    """Delete rows past the retention window; returns how many went.

    A retained conversation log is a regulated record: it needs a stated life,
    not an unbounded one, so this is the deletion side of that policy.
    """

    cutoff = datetime.now(UTC) - timedelta(days=days)
    with get_sessionmaker()() as session:
        rows = session.scalars(select(ConversationTurnRow)).all()
        stale = [row.id for row in rows if _aware(row.timestamp) < cutoff]
        if not stale:
            return 0
        session.execute(
            delete(ConversationTurnRow).where(ConversationTurnRow.id.in_(stale))
        )
        session.commit()
        return len(stale)
