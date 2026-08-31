"""The durable turn store, the SLO catalogue, and the authenticated read API."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import settings
from app.conversation.reasons import ReasonCode
from app.conversation.state import ConversationStatus
from app.main import app
from app.observability import alerts, turns
from app.observability.store import ConversationTurnRow, get_sessionmaker, reset_engine
from app.schemas import Intent

client = TestClient(app)

OPS_KEY = "test-ops-key"


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the turn store at a throwaway database for each test."""

    monkeypatch.setattr(
        settings, "turn_store_url", f"sqlite:///{tmp_path / 'turns.db'}"
    )
    monkeypatch.setattr(settings, "turn_observability_enabled", True)
    reset_engine()
    yield
    reset_engine()


def _talk(text: str, session_id: str, user_id: str | None = None) -> dict[str, object]:
    payload = {"text": text, "session_id": session_id, "user_id": user_id}
    response = client.post("/conversation/text", json=payload)
    assert response.status_code == 200
    return response.json()


# --------------------------------------------------------------------------- #
# What a turn row holds — and what it must never hold
# --------------------------------------------------------------------------- #
def test_a_turn_is_recorded_with_the_engine_s_own_reason_code() -> None:
    """The stored code is the one the engine attached, not one inferred later."""

    _talk("send 250 dollars", "obs-reason")

    rows = turns.list_turns(limit=5)
    assert len(rows) == 1
    row = rows[0]
    assert row.intent == Intent.TRANSFER_MONEY.value
    assert row.status == ConversationStatus.COLLECTING.value
    assert row.pending_slot == "recipient"
    assert row.reason_code == ReasonCode.SLOT_REQUIRED.value
    assert row.latency_ms is not None


def test_the_row_holds_no_customer_text_amount_account_or_payee() -> None:
    """A slot is recorded as filled and sourced — never as its value."""

    _talk("transfer 250 SAR to Sara Adel", "obs-pii", user_id="u-1")

    row = turns.list_turns(limit=1)[0]
    stored = str(row)
    for secret in ("Sara", "250", "obs-pii", "u-1", "transfer 250"):
        assert secret not in stored

    assert row.slots["amount"] == "user_text"
    assert set(row.slots) <= {"amount", "currency", "recipient", "note"}
    assert all(source for source in row.slots.values())


def test_identifiers_are_digests_that_still_correlate_one_session() -> None:
    """An operator can follow a known session without the store naming it."""

    _talk("send 100 riyals", "obs-corr")
    _talk("to Sara Adel", "obs-corr")
    _talk("pay my stc bill", "obs-other")

    reference = turns.session_ref_for("obs-corr")
    assert reference != "obs-corr"
    assert len(turns.list_turns(session_ref=reference)) == 2

    monkeyed = turns.session_ref_for("obs-other")
    assert monkeyed != reference


def test_recording_never_breaks_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store failure is a warning, not a 500 on a payment."""

    def unreachable_store() -> None:
        raise RuntimeError("turn store is down")

    monkeypatch.setattr(turns, "get_sessionmaker", unreachable_store)
    body = _talk("send 250 dollars", "obs-resilient")
    assert body["pending_slot"] == "recipient"


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
def test_purge_removes_rows_past_the_retention_window() -> None:
    _talk("send 250 dollars", "obs-retention")
    with get_sessionmaker()() as session:
        row = session.scalars(select(ConversationTurnRow)).one()
        row.timestamp = datetime.now(UTC) - timedelta(days=120)
        session.commit()

    assert turns.purge_older_than(90) == 1
    assert turns.list_turns() == []


# --------------------------------------------------------------------------- #
# SLO catalogue (transport-free)
# --------------------------------------------------------------------------- #
def test_every_objective_counts_real_reason_codes_or_statuses() -> None:
    """An objective may only be defined over codes the engine actually emits."""

    known = {f"reason:{code.value}" for code in ReasonCode}
    known |= {f"status:{status.value}" for status in ConversationStatus}
    for slo in alerts.CATALOGUE:
        assert slo.numerator
        assert set(slo.numerator) <= known, slo.key


def test_an_objective_breaches_only_above_its_bound() -> None:
    failed = f"status:{ConversationStatus.FAILED.value}"
    counts = {
        60: {"turns": 100, failed: 2},
        1440: {"turns": 100, failed: 2},
    }
    reading = {m.slo.key: m for m in alerts.evaluate(counts)}

    assert reading["failed_writes"].ratio == 0.02
    assert reading["failed_writes"].breached
    assert not reading["understanding_gap"].breached


def test_an_empty_window_is_not_a_breach() -> None:
    """No traffic must not page anybody."""

    counts = dict.fromkeys(alerts.windows(), {"turns": 0})
    assert not any(m.breached for m in alerts.evaluate(counts))


def test_counts_are_keyed_for_the_catalogue() -> None:
    _talk("send 250 dollars", "obs-counts")

    tally = turns.counts(60)
    assert tally["turns"] == 1
    assert tally[f"reason:{ReasonCode.SLOT_REQUIRED.value}"] == 1
    assert tally[f"status:{ConversationStatus.COLLECTING.value}"] == 1


def test_counts_ignore_rows_outside_the_window() -> None:
    _talk("send 250 dollars", "obs-window")
    with get_sessionmaker()() as session:
        row = session.scalars(select(ConversationTurnRow)).one()
        row.timestamp = datetime.now(UTC) - timedelta(hours=5)
        session.commit()

    assert turns.counts(60)["turns"] == 0
    assert turns.counts(1440)["turns"] == 1


# --------------------------------------------------------------------------- #
# The read API is closed by default
# --------------------------------------------------------------------------- #
def test_the_endpoints_are_unavailable_with_no_key_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: an unconfigured key means no service, not open access."""

    monkeypatch.setattr(settings, "ops_api_key", None)
    for path in ("/ops/observability/turns", "/ops/observability/slo"):
        assert client.get(path).status_code == 503


def test_a_wrong_or_missing_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ops_api_key", OPS_KEY)
    assert client.get("/ops/observability/turns").status_code == 401
    assert (
        client.get(
            "/ops/observability/turns", headers={"x-ops-key": "nope"}
        ).status_code
        == 401
    )


def test_turns_and_slo_read_back_with_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ops_api_key", OPS_KEY)
    _talk("send 250 dollars", "obs-api")

    headers = {"x-ops-key": OPS_KEY}
    listed = client.get(
        "/ops/observability/turns",
        params={"session_id": "obs-api"},
        headers=headers,
    )
    assert listed.status_code == 200
    body = listed.json()
    assert len(body) == 1
    assert body[0]["reason_code"] == ReasonCode.SLOT_REQUIRED.value
    assert body[0]["session_ref"] == turns.session_ref_for("obs-api")

    report = client.get("/ops/observability/slo", headers=headers)
    assert report.status_code == 200
    slo = report.json()
    assert slo["breaching"] == 0
    assert {o["key"] for o in slo["objectives"]} == {s.key for s in alerts.CATALOGUE}
