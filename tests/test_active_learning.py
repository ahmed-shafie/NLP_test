"""Tests for the Active Learning block: store, policy, daemon, rebuild, and API."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.active_learning import daemon, service, store
from app.active_learning.schemas import CaseStatus
from app.config import settings
from app.main import app
from app.schemas import Intent, Language, NLUResponse, TransferEntities

client = TestClient(app)


@pytest.fixture()
def al_store(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Point the review-queue store at an isolated SQLite file and reset caches."""

    db_url = f"sqlite:///{tmp_path}/al.db"
    monkeypatch.setattr(settings, "active_learning_store_url", db_url)
    monkeypatch.setattr(settings, "active_learning_enabled", True)
    monkeypatch.setattr(settings, "active_learning_auto_approve_confidence", 0.85)
    monkeypatch.setattr(settings, "active_learning_log_confidence", 0.6)

    store.get_engine.cache_clear()
    store.get_sessionmaker.cache_clear()
    store.get_store.cache_clear()

    yield store.get_store()

    store.get_engine.cache_clear()
    store.get_sessionmaker.cache_clear()
    store.get_store.cache_clear()


def _response(
    text: str,
    *,
    intent: Intent = Intent.TRANSFER_MONEY,
    confidence: float = 0.9,
    llm_assisted: bool = False,
) -> NLUResponse:
    return NLUResponse(
        text=text,
        language=Language.EN,
        intent=intent,
        confidence=confidence,
        intent_source="semantic",
        entities=TransferEntities(),
        llm_assisted=llm_assisted,
    )


# --------------------------------- store ----------------------------------- #


def test_log_and_list_case(al_store):
    case = al_store.log_case(
        text="send 5 to mom",
        language=Language.EN,
        predicted_intent=Intent.TRANSFER_MONEY,
        confidence=0.4,
        intent_source="semantic",
        llm_assisted=True,
        clarification=None,
        status=CaseStatus.PENDING,
        source="nlu.parse",
    )
    assert case.id is not None
    pending = al_store.list_cases(status=CaseStatus.PENDING)
    assert [c.id for c in pending] == [case.id]


def test_decide_and_approved_examples(al_store):
    case = al_store.log_case(
        text="ship 9 to dad",
        language=Language.EN,
        predicted_intent=Intent.FALLBACK,
        confidence=0.2,
        intent_source="semantic",
        llm_assisted=True,
        clarification=None,
        status=CaseStatus.PENDING,
        source="nlu.parse",
    )
    updated = al_store.decide(
        case.id,
        CaseStatus.APPROVED,
        corrected_intent=Intent.TRANSFER_MONEY,
        reviewer="alice",
    )
    assert updated.status is CaseStatus.APPROVED
    assert updated.corrected_intent is Intent.TRANSFER_MONEY
    assert updated.reviewer == "alice"

    # The correction is the effective training label.
    examples = al_store.approved_examples()
    assert ("ship 9 to dad", Intent.TRANSFER_MONEY) in examples


def test_stats_counts(al_store):
    al_store.log_case(
        text="a",
        language=Language.EN,
        predicted_intent=Intent.FALLBACK,
        confidence=0.1,
        intent_source="semantic",
        llm_assisted=True,
        clarification=None,
        status=CaseStatus.PENDING,
        source="nlu.parse",
    )
    al_store.log_case(
        text="b",
        language=Language.EN,
        predicted_intent=Intent.TRANSFER_MONEY,
        confidence=0.9,
        intent_source="semantic",
        llm_assisted=True,
        clarification=None,
        status=CaseStatus.AUTO_APPROVED,
        source="nlu.parse",
    )
    stats = al_store.stats()
    assert stats.total == 2
    assert stats.pending == 1
    assert stats.auto_approved == 1
    assert stats.learned_examples == 1


# --------------------------------- policy ---------------------------------- #


def test_confident_deterministic_case_not_logged(al_store):
    logged = service.record_case(_response("send 500 to Ahmed"), source="t")
    assert logged is None
    assert al_store.stats().total == 0


def test_fallback_case_enqueued_for_review(al_store):
    logged = service.record_case(
        _response("weather please", intent=Intent.FALLBACK, confidence=0.2),
        source="t",
    )
    assert logged is not None
    assert logged.status is CaseStatus.PENDING


def test_high_confidence_llm_case_auto_approved(al_store):
    logged = service.record_case(
        _response("wire 10 to Sara", confidence=0.92, llm_assisted=True),
        source="t",
    )
    assert logged is not None
    assert logged.status is CaseStatus.AUTO_APPROVED


def test_record_case_respects_disabled_flag(al_store, monkeypatch):
    monkeypatch.setattr(settings, "active_learning_enabled", False)
    assert service.record_case(_response("x", intent=Intent.FALLBACK), "t") is None


# --------------------------------- daemon ---------------------------------- #


def test_next_run_time_rolls_to_tomorrow(monkeypatch):
    monkeypatch.setattr(settings, "index_rebuild_hour_utc", 3)
    monkeypatch.setattr(settings, "index_rebuild_minute_utc", 30)
    now = datetime(2026, 6, 22, 5, 0, 0, tzinfo=UTC)
    nxt = daemon.next_run_time(now)
    assert (nxt.year, nxt.month, nxt.day, nxt.hour, nxt.minute) == (
        2026,
        6,
        23,
        3,
        30,
    )
    assert daemon.seconds_until_next_run(now) > 0


def test_next_run_time_later_today(monkeypatch):
    monkeypatch.setattr(settings, "index_rebuild_hour_utc", 23)
    monkeypatch.setattr(settings, "index_rebuild_minute_utc", 0)
    now = datetime(2026, 6, 22, 5, 0, 0, tzinfo=UTC)
    nxt = daemon.next_run_time(now)
    assert nxt.day == 22 and nxt.hour == 23


# ----------------------------------- API ----------------------------------- #


def test_queue_and_decision_endpoints(al_store):
    service.record_case(
        _response("send my rent", intent=Intent.FALLBACK, confidence=0.3),
        source="t",
    )
    listed = client.get("/active-learning/queue?status=pending")
    assert listed.status_code == 200
    body = listed.json()
    assert len(body) == 1
    case_id = body[0]["id"]

    approved = client.post(
        f"/active-learning/{case_id}/approve",
        json={"corrected_intent": "transfer_money", "reviewer": "web"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    stats = client.get("/active-learning/stats").json()
    assert stats["approved"] == 1
    assert stats["learned_examples"] == 1


def test_approve_unknown_case_returns_404(al_store):
    resp = client.post("/active-learning/999999/approve", json={})
    assert resp.status_code == 404


def test_manual_rebuild_endpoint(al_store):
    resp = client.post("/active-learning/rebuild", json={})
    assert resp.status_code == 200
    body = resp.json()
    # Either the index rebuilt (embedder available) or it degraded gracefully.
    assert "base_examples" in body
    assert isinstance(body["ok"], bool)
