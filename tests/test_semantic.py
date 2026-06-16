"""Tests for semantic intent classification and contact matching.

These require the multilingual embedding model. When it cannot be loaded (e.g.
offline CI), the tests are skipped rather than failing.
"""

import pytest

from app.embeddings import get_embedder
from app.nlu.contacts import get_default_matcher
from app.nlu.semantic_intents import get_semantic_classifier
from app.schemas import Intent

pytestmark = pytest.mark.skipif(
    get_embedder() is None,
    reason="embedding model unavailable",
)


def test_semantic_intent_english_transfer():
    clf = get_semantic_classifier()
    assert clf is not None
    intent, conf = clf.classify("please send 300 dollars to my brother")
    assert intent == Intent.TRANSFER_MONEY
    assert conf > 0.4


def test_semantic_intent_arabic_transfer():
    clf = get_semantic_classifier()
    assert clf is not None
    intent, _ = clf.classify("عايز احول مبلغ لصديقي محمد")
    assert intent == Intent.TRANSFER_MONEY


def test_semantic_intent_fallback():
    clf = get_semantic_classifier()
    assert clf is not None
    intent, _ = clf.classify("what time does the branch close today")
    assert intent == Intent.FALLBACK


def test_similar_returns_neighbours():
    clf = get_semantic_classifier()
    assert clf is not None
    neighbours = clf.similar("transfer 100 to Sara", k=3)
    assert len(neighbours) == 3
    assert neighbours[0].score >= neighbours[-1].score


def test_contact_match_cross_lingual():
    matcher = get_default_matcher()
    matched, candidates = matcher.resolve("Ahmed")
    assert matched is not None
    # "Ahmed" should resolve to one of the Ahmed entries (either script).
    assert "أحمد" in matched.contact.name or "Ahmed" in matched.contact.name
    assert candidates[0].score >= candidates[-1].score


def test_contact_match_arabic_query():
    matcher = get_default_matcher()
    matched, _ = matcher.resolve("سارة")
    assert matched is not None
    assert matched.contact.account == "EG1003"
