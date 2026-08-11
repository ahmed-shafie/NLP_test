"""The curated example corpus: loading, and the routing it must not break."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import settings
from app.conversation.engine import decide_action, route_fresh_turn
from app.nlu import corpus
from app.nlu.lang import Language
from app.schemas import Intent


@pytest.fixture(autouse=True)
def _clear_cache():
    corpus.load_corpus_examples.cache_clear()
    yield
    corpus.load_corpus_examples.cache_clear()


def _write(tmp_path: Path, *rows: dict[str, str]) -> Path:
    path = tmp_path / "example_corpus.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return path


def test_loads_rows(tmp_path, monkeypatch):
    path = _write(
        tmp_path,
        {"text": "حوّل 100 لسارة", "intent": "transfer_money"},
        {"text": "ليش انخصمت رسوم؟", "intent": "fallback", "topic": "الرسوم"},
    )
    monkeypatch.setattr(corpus, "CORPUS_PATH", path)
    assert corpus.load_corpus_examples() == (
        ("حوّل 100 لسارة", Intent.TRANSFER_MONEY),
        ("ليش انخصمت رسوم؟", Intent.FALLBACK),
    )


def test_missing_file_degrades_to_builtins(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "CORPUS_PATH", tmp_path / "absent.jsonl")
    assert corpus.load_corpus_examples() == ()


def test_malformed_rows_are_skipped(tmp_path, monkeypatch):
    path = _write(tmp_path, {"text": "hi", "intent": "no_such_intent"})
    path.write_text(
        path.read_text(encoding="utf-8") + "\n{ not json\n", encoding="utf-8"
    )
    monkeypatch.setattr(corpus, "CORPUS_PATH", path)
    assert corpus.load_corpus_examples() == ()


def test_disabled_by_setting(tmp_path, monkeypatch):
    path = _write(tmp_path, {"text": "hi", "intent": "small_talk"})
    monkeypatch.setattr(corpus, "CORPUS_PATH", path)
    monkeypatch.setattr(settings, "example_corpus_enabled", False)
    assert corpus.load_corpus_examples() == ()


# Indexing 31k customer-service rows makes the classifier refuse anything that
# looks like a banking question, including underspecified requests it should
# instead ask about. These cues keep that decision off the classifier.
@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        ("I need to transfer some money", Language.EN, Intent.TRANSFER_MONEY),
        ("أرغب في تحويل مبلغ", Language.AR, Intent.TRANSFER_MONEY),
        ("send 500 USD", Language.EN, Intent.TRANSFER_MONEY),
        ("pay my fine", Language.EN, Intent.PAY_BILL),
        # A question about the action is customer service, not a request.
        ("كيف أحوّل فلوس لحسابي؟", Language.AR, None),
        ("how do I transfer money?", Language.EN, None),
        # Currency conversion reads like a transfer but names no beneficiary.
        ("حوّل ٥٠٠ ريال لدولار", Language.AR, None),
        # "send" alone sends things that are not money.
        ("send me my account statement", Language.EN, None),
    ],
)
def test_underspecified_requests_route_deterministically(text, lang, expected):
    assert decide_action(text, lang, Intent.FALLBACK) is expected


def test_customer_service_question_never_opens_a_money_flow():
    assert (
        route_fresh_turn("ليش التحويل ما وصل؟", Language.AR, Intent.FALLBACK, 0.9)
        is Intent.FALLBACK
    )
