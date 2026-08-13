"""Tests for the Haystack-orchestrated pipeline and the LLM exception handler hook."""

from __future__ import annotations

from decimal import Decimal

import pytest

import app.orchestration as orchestration
from app.llm import LLMResult
from app.schemas import Beneficiary, Intent, TransferEntities


class _FakeHandler:
    def __init__(self, result: LLMResult | None, response: str | None = None) -> None:
        self._result = result
        self._response = response
        self.called = False
        self.responded = False

    def extract(
        self, text: str, language: str, known: TransferEntities
    ) -> LLMResult | None:
        self.called = True
        return self._result

    def respond_unresolved(
        self, text: str, language: str, account_number: str, known: TransferEntities
    ) -> str | None:
        self.responded = True
        return self._response


class _FakeRepo:
    def __init__(self, beneficiary: Beneficiary | None) -> None:
        self._beneficiary = beneficiary
        self.looked_up: str | None = None

    def lookup(self, account_number: str) -> Beneficiary | None:
        self.looked_up = account_number
        return self._beneficiary


def _install_repo(monkeypatch: pytest.MonkeyPatch, repo: object | None) -> None:
    monkeypatch.setattr(orchestration, "get_beneficiary_repository", lambda: repo)
    orchestration.get_nlu_pipeline.cache_clear()


def _install(monkeypatch: pytest.MonkeyPatch, handler: _FakeHandler) -> None:
    monkeypatch.setattr(orchestration, "get_llm_handler", lambda: handler)
    orchestration.get_nlu_pipeline.cache_clear()


def test_pipeline_has_connected_components():
    pipe = orchestration.get_nlu_pipeline()
    assert {
        "detect",
        "intent",
        "entities",
        "contacts",
        "beneficiary",
        "llm",
    } <= set(pipe.graph.nodes)


def test_llm_fills_a_name_the_rules_missed(monkeypatch):
    """A slot the customer did type may be filled by the model."""

    handler = _FakeHandler(
        LLMResult(
            intent="transfer_money",
            amount=None,
            currency=None,
            recipient="karim",
            source_account=None,
            clarification=None,
        )
    )
    _install(monkeypatch, handler)

    result = orchestration.run_pipeline("the money is for karim please")

    assert handler.called is True
    assert result.entities.recipient == "karim"
    assert result.llm_assisted is True


def test_llm_reclassifies_fallback_but_invents_no_slots(monkeypatch):
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
    # Nobody typed "Sara", "50" or "USD": ungrounded slots are dropped, so the
    # model can route a message but never author its financial content.
    assert result.entities.recipient is None
    assert result.entities.amount is None
    assert result.entities.currency is None
    assert result.resolved_recipient is None
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
    # autouse fixture disables the LLM handler; the rules read the word-amount.
    result = orchestration.run_pipeline("حوّل ألف جنيه إلى محمد")
    assert result.entities.amount == Decimal("1000")
    assert result.llm_assisted is False
    assert result.clarification is None


def test_beneficiary_found_in_database(monkeypatch):
    repo = _FakeRepo(
        Beneficiary(id="b1", name="Sara Adel", account="EG1003", bank="CIB")
    )
    _install_repo(monkeypatch, repo)

    result = orchestration.run_pipeline("transfer 500 dollars", account_number="EG1003")

    assert repo.looked_up == "EG1003"
    assert result.resolved_beneficiary is not None
    assert result.resolved_beneficiary.name == "Sara Adel"
    assert result.beneficiary_source == "database"
    # The DB beneficiary fills the recipient slot.
    assert result.entities.recipient == "Sara Adel"


def test_beneficiary_not_found_delegates_to_llm(monkeypatch):
    repo = _FakeRepo(None)
    handler = _FakeHandler(None, response="لم يتم العثور على الحساب، تحقق من الرقم.")
    _install_repo(monkeypatch, repo)
    monkeypatch.setattr(orchestration, "get_llm_handler", lambda: handler)
    orchestration.get_nlu_pipeline.cache_clear()

    result = orchestration.run_pipeline("حوّل 500 جنيه", account_number="EG9999")

    assert repo.looked_up == "EG9999"
    assert handler.responded is True
    assert result.resolved_beneficiary is None
    assert result.beneficiary_source == "llm"
    assert result.clarification == "لم يتم العثور على الحساب، تحقق من الرقم."
    assert result.llm_assisted is True


def test_beneficiary_not_found_without_llm_degrades(monkeypatch):
    # DB miss but the LLM is unavailable (autouse fixture): no crash, no response.
    repo = _FakeRepo(None)
    _install_repo(monkeypatch, repo)

    result = orchestration.run_pipeline("transfer 500 dollars", account_number="EG9999")

    assert result.resolved_beneficiary is None
    assert result.beneficiary_source is None
    assert result.clarification is None


def test_no_account_number_skips_beneficiary_lookup(monkeypatch):
    repo = _FakeRepo(Beneficiary(name="Should Not Be Used"))
    _install_repo(monkeypatch, repo)

    result = orchestration.run_pipeline("send 500 dollars to Ahmed")

    assert repo.looked_up is None
    assert result.resolved_beneficiary is None
    assert result.beneficiary_source is None
