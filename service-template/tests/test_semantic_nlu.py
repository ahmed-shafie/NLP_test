"""Tests for the spaCy + FAISS NLU tier (no LLM).

These opt out of the deterministic-NLU fixture and exercise the real models.
Each test skips cleanly when the corresponding model is not installed, so CI
stays green whether or not the embedding / spaCy models were downloaded.
"""

from __future__ import annotations

import pytest
from service_template import extractor
from service_template.config import settings
from service_template.embeddings import get_embedder
from service_template.schemas import Intent
from service_template.semantic_intents import (
    get_semantic_classifier,
    reset_semantic_classifier,
)


@pytest.fixture()
def semantic_enabled(monkeypatch: pytest.MonkeyPatch):
    """Enable the semantic tier; skip the test if the embedder can't load."""

    monkeypatch.setattr(settings, "use_semantic_intent", True)
    reset_semantic_classifier()
    if get_embedder() is None:
        pytest.skip("embedding model unavailable (offline); semantic tier skipped")
    classifier = get_semantic_classifier()
    if classifier is None:
        pytest.skip("semantic classifier unavailable")
    yield classifier
    reset_semantic_classifier()


@pytest.fixture()
def spacy_enabled(monkeypatch: pytest.MonkeyPatch):
    """Enable spaCy NER; skip the test if the model isn't installed."""

    monkeypatch.setattr(settings, "use_spacy_ner", True)
    extractor._load_spacy.cache_clear()
    if extractor._load_spacy() is None:
        pytest.skip("spaCy model not installed; run: python -m spacy download ...")
    yield
    extractor._load_spacy.cache_clear()


def test_faiss_classifier_indexes_examples(semantic_enabled) -> None:
    assert len(semantic_enabled) > 0  # FAISS index built from the examples


@pytest.mark.parametrize(
    "text,expected",
    [
        # Phrasings NOT present verbatim in the examples — semantic recall.
        ("could you move 200 to my colleague", Intent.TRANSFER_MONEY),
        ("good evening!", Intent.SMALL_TALK),
        ("حابب ابعت مبلغ لأخويا", Intent.TRANSFER_MONEY),  # Arabic paraphrase
    ],
)
def test_semantic_intent_classification(
    semantic_enabled, text: str, expected: Intent
) -> None:
    intent, confidence = semantic_enabled.classify(text)
    assert intent is expected
    assert 0.0 <= confidence <= 1.0


def test_spacy_recipient_ner(spacy_enabled) -> None:
    # No "to"/"for" cue for the regex to latch onto — spaCy PERSON NER handles it.
    slots = extractor.extract_slots("please wire 300 USD, recipient is Michael Scott")
    assert slots.recipient is not None
    assert "Michael" in slots.recipient
