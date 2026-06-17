"""Tests for the Haystack-orchestrated pipeline and the LLM exception handler hook."""

from __future__ import annotations

from decimal import Decimal

import pytest

import app.orchestration as orchestration
from app.llm import LLMResult
from app.schemas import Intent, TransferEntities


class _FakeHandler:
    def __init__(self, result: LLMResult | None) -> None:
        self._result = result
        self.called = False

    def extract(
        self, text: str, language: str, known: TransferEntities
    ) -> LLMResult | None:
        self.called = True
        return self._result


def _install(monkeypatch: pytest.MonkeyPatch, handler: _FakeHandler) -> None:
    monkeypatch.setattr(orchestration, "get_llm_handler", lambda: handler)
    orchestration.get_nlu_pipeline.cache_clear()


def test_pipeline_has_connected_components():
    pipe = orchestration.get_nlu_pipeline()
    assert {"detect", "intent", "entities", "contacts", "llm"} <= set(pipe.graph.nodes)


def test_llm_fills_missing_word_amount(monkeypatch):
    handler = _FakeHandler(
        LLMResult(
            intent="transfer_money",
            amount=Decimal("1000"),
            currency="EGP",
            recipient=None,
            source_account=None,
            clarification=None,
        )
    )
    _install(monkeypatch, handler)

    result = orchestration.run_pipeline("حوّل ألف جنيه إلى محمد")

    assert handler.called is True
    assert result.intent is Intent.TRANSFER_MONEY
    assert result.entities.amount == Decimal("1000")
    assert result.entities.currency == "EGP"
    assert result.llm_assisted is True


def test_llm_reclassifies_fallback_and_resolves_recipient(monkeypatch):
    handler = _FakeHandler(
        LLMResult(
            intent="transfer_money",
            amount=Decimal("50"),
            currency="USD",
            recipient="Sara",
            source_account=None,
            clarification="Shall I send 50 USD to Sara?",
        )
    )
    _install(monkeypatch, handler)

    result = orchestration.run_pipeline("what is the weather today")

    assert result.intent is Intent.TRANSFER_MONEY
    assert result.entities.recipient == "Sara"
    assert result.resolved_recipient is not None
    assert result.llm_assisted is True


def test_llm_adds_clarification_on_fallback(monkeypatch):
    handler = _FakeHandler(
        LLMResult(
            intent="fallback",
            amount=None,
            currency=None,
            recipient=None,
            source_account=None,
            clarification="I can only help with money transfers.",
        )
    )
    _install(monkeypatch, handler)

    result = orchestration.run_pipeline("what is the weather today")

    assert result.intent is Intent.FALLBACK
    assert result.clarification == "I can only help with money transfers."
    assert result.llm_assisted is True


def test_complete_transfer_skips_llm(monkeypatch):
    handler = _FakeHandler(None)
    _install(monkeypatch, handler)

    result = orchestration.run_pipeline("send 500 dollars to Ahmed")

    assert handler.called is False
    assert result.entities.amount == Decimal("500")
    assert result.llm_assisted is False


def test_graceful_degradation_without_llm():
    # autouse fixture disables the LLM handler; the word-amount stays unparsed.
    result = orchestration.run_pipeline("حوّل ألف جنيه إلى محمد")
    assert result.entities.amount is None
    assert result.llm_assisted is False
    assert result.clarification is None
