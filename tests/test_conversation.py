"""Tests for the multi-turn conversation engine, session store, and endpoints."""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.conversation.engine import ConversationEngine
from app.conversation.state import (
    ConversationSlots,
    ConversationState,
    ConversationStatus,
)
from app.conversation.store import InMemorySessionStore, get_session_store
from app.main import app
from app.schemas import Intent, Language
from app.voice import asr, tts

client = TestClient(app)


@pytest.fixture()
def engine() -> ConversationEngine:
    return ConversationEngine()


# --------------------------- slot-filling engine --------------------------- #


def test_single_utterance_reaches_confirmation(engine: ConversationEngine):
    result = engine.handle("send 500 dollars to Ahmed", "t1")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.amount == Decimal("500")
    assert result.state.slots.currency == "USD"
    assert result.state.slots.recipient == "Ahmed"


def test_progressive_slot_filling(engine: ConversationEngine):
    assert (
        engine.handle("I want to transfer money", "t2").state.pending_slot == "amount"
    )
    assert engine.handle("500 dollars", "t2").state.pending_slot == "recipient"
    confirming = engine.handle("Ahmed", "t2")
    assert confirming.state.status is ConversationStatus.CONFIRMING
    done = engine.handle("yes", "t2")
    assert done.state.status is ConversationStatus.COMPLETED
    assert done.transfer is not None
    assert done.transfer.amount == Decimal("500")
    assert done.transfer.recipient == "Ahmed"


def test_confirmation_yes_completes(engine: ConversationEngine):
    engine.handle("send 500 USD to Ahmed", "t3")
    done = engine.handle("yes", "t3")
    assert done.state.status is ConversationStatus.COMPLETED
    assert done.transfer is not None


def test_confirmation_no_cancels(engine: ConversationEngine):
    engine.handle("send 500 USD to Ahmed", "t4")
    cancelled = engine.handle("no", "t4")
    assert cancelled.state.status is ConversationStatus.CANCELLED
    assert cancelled.transfer is None


def test_unrecognised_confirmation_reasks(engine: ConversationEngine):
    engine.handle("send 500 USD to Ahmed", "t5")
    again = engine.handle("what do you mean", "t5")
    assert again.state.status is ConversationStatus.CONFIRMING


def test_cancel_midflow(engine: ConversationEngine):
    engine.handle("I want to send 500 USD", "t6")
    cancelled = engine.handle("cancel", "t6")
    assert cancelled.state.status is ConversationStatus.CANCELLED
    assert cancelled.state.slots.recipient is None


def test_bare_recipient_fills_pending(engine: ConversationEngine):
    engine.handle("send 500 USD", "t7")  # missing recipient
    result = engine.handle("Sara", "t7")
    assert result.state.slots.recipient == "Sara"


def test_non_transfer_is_fallback(engine: ConversationEngine):
    result = engine.handle("what is my balance", "t8")
    assert result.state.intent is not Intent.TRANSFER_MONEY
    assert "transfer" in result.reply.lower()


def test_completed_then_new_dialogue_resets(engine: ConversationEngine):
    engine.handle("send 500 USD to Ahmed", "t9")
    engine.handle("yes", "t9")
    fresh = engine.handle("send 20 EUR to Sara", "t9")
    assert fresh.state.slots.amount == Decimal("20")
    assert fresh.state.slots.currency == "EUR"
    assert fresh.state.slots.recipient == "Sara"


def test_arabic_dialogue(engine: ConversationEngine):
    result = engine.handle("حوّل 500 دولار إلى أحمد", "t10")
    assert result.state.language is Language.AR
    assert result.state.status is ConversationStatus.CONFIRMING
    done = engine.handle("نعم", "t10")
    assert done.state.status is ConversationStatus.COMPLETED
    assert done.transfer is not None
    assert done.transfer.recipient == "أحمد"


def test_sessions_are_isolated(engine: ConversationEngine):
    engine.handle("send 500 USD to Ahmed", "iso-a")
    other = engine.handle("I want to transfer money", "iso-b")
    assert other.state.slots.amount is None


def test_unsupported_currency_recollects(engine: ConversationEngine):
    state = ConversationState(
        session_id="t11",
        intent=Intent.TRANSFER_MONEY,
        status=ConversationStatus.CONFIRMING,
        slots=ConversationSlots(amount=Decimal("5"), currency="XYZ", recipient="Ahmed"),
    )
    engine._store.save(state)
    result = engine.handle("yes", "t11")
    assert result.state.status is ConversationStatus.COLLECTING
    assert result.state.pending_slot == "currency"


# ------------------------------- session store ----------------------------- #


def test_in_memory_store_roundtrip():
    store = InMemorySessionStore()
    state = ConversationState(
        session_id="x", slots=ConversationSlots(amount=Decimal("9"))
    )
    store.save(state)
    loaded = store.load("x")
    assert loaded is not None
    assert loaded.slots.amount == Decimal("9")
    store.delete("x")
    assert store.load("x") is None


def test_default_store_is_in_memory():
    assert isinstance(get_session_store(), InMemorySessionStore)


# ------------------------------- API endpoints ----------------------------- #


def test_conversation_text_endpoint():
    first = client.post("/conversation/text", json={"text": "I want to transfer money"})
    assert first.status_code == 200
    sid = first.json()["session_id"]
    assert first.json()["pending_slot"] == "amount"

    second = client.post(
        "/conversation/text", json={"text": "500 USD to Ahmed", "session_id": sid}
    )
    body = second.json()
    assert body["session_id"] == sid
    assert body["status"] == "confirming"


def test_conversation_v1_alias():
    resp = client.post("/v1/conversation/text", json={"text": "send 5 USD to Ahmed"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "confirming"


def test_conversation_disabled_returns_503(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "conversation_enabled", False)
    resp = client.post("/conversation/text", json={"text": "hi"})
    assert resp.status_code == 503


def test_voice_endpoint_503_when_asr_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(asr, "asr_available", lambda: False)
    resp = client.post(
        "/conversation/voice",
        files={"audio": ("clip.wav", b"\x00\x00", "audio/wav")},
    )
    assert resp.status_code == 503


# ----------------------------------- voice --------------------------------- #


def test_tts_disabled_returns_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "voice_enabled", False)
    assert tts.synthesize("hello", Language.EN) is None


def test_asr_transcribe_none_when_model_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(asr, "_get_model", lambda: None)
    assert asr.transcribe("/tmp/does-not-exist.wav") is None


def test_voice_endpoint_returns_transcript_and_audio(monkeypatch: pytest.MonkeyPatch):
    """Happy path: the async endpoint offloads ASR/TTS so a spoken reply is produced.

    The fake TTS calls ``asyncio.run`` like edge-tts does; this regresses the bug
    where synthesis ran inside the request's event loop and raised "asyncio.run()
    cannot be called from a running event loop".
    """
    import asyncio

    monkeypatch.setattr(asr, "asr_available", lambda: True)
    monkeypatch.setattr(asr, "transcribe", lambda *a, **k: "send 500 USD to Ahmed")

    def fake_synthesize(text: str, language: Language) -> tuple[bytes, str]:
        async def _run() -> bytes:
            return b"audio-bytes"

        return asyncio.run(_run()), "audio/mpeg"

    monkeypatch.setattr("app.conversation.router.tts.synthesize", fake_synthesize)

    resp = client.post(
        "/conversation/voice",
        files={"audio": ("clip.webm", b"\x00\x00", "audio/webm")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "send 500 USD to Ahmed"
    assert body["audio_base64"]
    assert body["audio_mime"] == "audio/mpeg"
