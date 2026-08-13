"""Unit tests for the LiteLLM exception handler (no live LLM required)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.llm import (
    LLMExceptionHandler,
    _coerce_amount,
    _parse_json_object,
)
from app.schemas import TransferEntities


def test_parse_json_object_handles_code_fence():
    content = 'Sure!\n```json\n{"intent": "fallback", "amount": null}\n```'
    assert _parse_json_object(content) == {"intent": "fallback", "amount": None}


def test_parse_json_object_raises_without_json():
    with pytest.raises(ValueError):
        _parse_json_object("no json here")


@pytest.mark.parametrize(
    "value,expected",
    [
        (1000, Decimal("1000")),
        ("250.5", Decimal("250.5")),
        (0, None),
        (-5, None),
        (None, None),
        (True, None),
        ("abc", None),
    ],
)
def test_coerce_amount(value, expected):
    assert _coerce_amount(value) == expected


def test_extract_normalises_llm_json(monkeypatch):
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"intent": "transfer_money", "amount": 1000, '
                        '"currency": "egp", "recipient": "محمد", '
                        '"source_account": null, "clarification": "ok"}'
                    )
                }
            }
        ]
    }
    monkeypatch.setattr("litellm.completion", lambda **kwargs: fake_response)

    handler = LLMExceptionHandler("ollama/x", "http://localhost:11434", 5.0, 0.0)
    result = handler.extract("حوّل ألف جنيه إلى محمد", "ar", TransferEntities())

    assert result is not None
    assert result.intent == "transfer_money"
    assert result.amount == Decimal("1000")
    assert result.currency == "EGP"
    assert result.recipient == "محمد"
    assert result.clarification == "ok"


def test_extract_returns_none_on_error(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr("litellm.completion", _boom)

    handler = LLMExceptionHandler("ollama/x", "http://localhost:11434", 5.0, 0.0)
    assert handler.extract("hello", "en", TransferEntities()) is None
