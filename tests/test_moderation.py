"""Tests for abusive ("ribald") input moderation in the conversation engine."""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.conversation import moderation
from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.main import app
from app.nlu.semantic_intents import get_semantic_classifier
from app.schemas import Intent

client = TestClient(app)


@pytest.fixture()
def engine() -> ConversationEngine:
    return ConversationEngine()


# ------------------------------ detector ---------------------------------- #


def test_clean_text_is_not_flagged():
    assert moderation.detect("send 500 to Ahmed").flagged is False
    assert moderation.detect("this is frustrating, pay my bill").flagged is False
    assert moderation.is_clean("ادفع فاتورة الكهرباء") is True


def test_mild_and_severe_severity():
    mild = moderation.detect("you are stupid")
    assert mild.flagged and mild.severity == "mild"
    severe = moderation.detect("this is bullshit")
    assert severe.flagged and severe.severity == "severe"
    # A severe term anywhere makes the whole message severe.
    mixed = moderation.detect("you stupid bastard")
    assert mixed.severity == "severe"


def test_leetspeak_and_arabic_variants_flagged():
    assert moderation.detect("you are stup1d").flagged is True
    assert moderation.detect("انت غبي").flagged is True
    # Multi-word entry.
    assert moderation.detect("just shut up").flagged is True


def test_detect_surfaces_original_terms():
    result = moderation.detect("You IDIOT")
    assert result.terms == ("IDIOT",)


# ------------------------------ engine ------------------------------------ #


def test_abuse_is_refused_and_not_processed(engine: ConversationEngine):
    result = engine.handle("you stupid idiot, send 500 to Ahmed", "m1")
    # Steered back to banking; nothing executed; no slot polluted.
    assert result.transfer is None
    assert result.state.slots.recipient is None
    assert result.flagged_terms
    assert result.state.status is not ConversationStatus.CONFIRMING


def test_abuse_midflow_preserves_state(engine: ConversationEngine):
    confirming = engine.handle("send 500 USD to Ahmed", "m2")
    assert confirming.state.status is ConversationStatus.CONFIRMING
    # An abusive turn during confirmation is refused but keeps the pending action.
    interrupted = engine.handle("this is stupid", "m2")
    assert interrupted.state.status is ConversationStatus.CONFIRMING
    assert interrupted.state.slots.amount == Decimal("500")
    # A following "yes" still completes the held transfer.
    done = engine.handle("yes", "m2")
    assert done.state.status is ConversationStatus.COMPLETED
    assert done.transfer is not None


def test_moderation_disabled_does_not_redirect(engine: ConversationEngine, monkeypatch):
    # With moderation off, abusive input is processed normally — neither the
    # blocklist nor the semantic safety net should issue a redirect.
    monkeypatch.setattr(settings, "moderation_enabled", False)
    result = engine.handle("you are stupid, send 500 to Ahmed", "m_off")
    assert not result.flagged_terms
    assert result.state.flagged_count == 0


def test_legitimate_word_not_over_blocked(engine: ConversationEngine):
    # "ادفع المخالفة" = "pay the fine": semantically near the abuse cluster but a
    # real banking request. It must NOT be flagged, and should route to a bill.
    result = engine.handle("ادفع المخالفة 12345", "m_fine")
    assert not result.flagged_terms
    assert result.state.flagged_count == 0
    assert result.state.intent is not Intent.INAPPROPRIATE


def test_genuine_abuse_still_flagged_above_threshold(engine: ConversationEngine):
    # This phrase is NOT in the deterministic blocklist, so it can only be caught
    # by the semantic safety net — and it must still clear the 0.80 bar and be
    # refused. Proves the higher threshold did not let real abuse through.
    text = "I think you are completely worthless and pathetic"
    assert not moderation.detect(text).flagged  # bypasses the blocklist
    classifier = get_semantic_classifier()
    if classifier is not None:
        intent, confidence = classifier.classify(text)
        assert intent is Intent.INAPPROPRIATE
        assert confidence >= settings.moderation_semantic_threshold
    result = engine.handle(text, "m_sem_abuse")
    assert result.state.flagged_count == 1  # the turn was refused (a strike)


def test_replies_vary_between_turns(engine: ConversationEngine, monkeypatch):
    monkeypatch.setattr(settings, "moderation_max_strikes", 10)
    first = engine.handle("you are stupid", "m3").reply
    second = engine.handle("you are stupid", "m3").reply
    assert first != second  # no immediate repeat


def test_mild_reply_names_the_flagged_word(engine: ConversationEngine, monkeypatch):
    monkeypatch.setattr(settings, "moderation_max_strikes", 10)
    monkeypatch.setattr(settings, "reply_variation_seed", 0)
    result = engine.handle("you are stupid", "m4")
    assert "stupid" in result.reply.lower()


def test_repeat_offense_ends_session(engine: ConversationEngine, monkeypatch):
    monkeypatch.setattr(settings, "moderation_max_strikes", 3)
    engine.handle("you stupid", "m5")
    engine.handle("you idiot", "m5")
    third = engine.handle("you moron", "m5")
    assert third.state.status is ConversationStatus.CANCELLED
    assert "continue" in third.reply.lower() or "أكمل" in third.reply


def test_conversation_endpoint_returns_flagged_terms():
    res = client.post(
        "/conversation/text", json={"text": "you are an idiot", "session_id": "m6"}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["flagged_terms"]
    assert body["intent"] != "transfer_money"
