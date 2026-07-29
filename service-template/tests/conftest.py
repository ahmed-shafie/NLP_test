"""Shared fixtures for the template tests.

The core FSM tests must be deterministic and fast, so by default we pin the NLU
to the regex/keyword tier (no model downloads, no spaCy/FAISS). Tests that want
to exercise the semantic/spaCy tier opt in explicitly and skip when the models
are not installed. This mirrors the main app's conftest, which disables the LLM
unless a test opts in.
"""

from __future__ import annotations

import pytest
from service_template import extractor
from service_template.config import settings
from service_template.semantic_intents import reset_semantic_classifier


@pytest.fixture(autouse=True)
def _pin_deterministic_nlu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the deterministic keyword/regex NLU for predictable core tests."""

    monkeypatch.setattr(settings, "use_semantic_intent", False)
    monkeypatch.setattr(settings, "use_spacy_ner", False)
    extractor._load_spacy.cache_clear()
    reset_semantic_classifier()
    yield
    extractor._load_spacy.cache_clear()
    reset_semantic_classifier()
