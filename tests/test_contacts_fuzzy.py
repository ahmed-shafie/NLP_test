"""Tests for the contact matcher's fuzzy fallback (no embedding model needed)."""

from app.nlu.contacts import ContactMatcher
from app.schemas import Contact

CONTACTS = [
    Contact(id="1", name="John Smith", account="A1"),
    Contact(id="2", name="Jane Doe", account="A2"),
    Contact(id="3", name="Johnny Appleseed", account="A3"),
]


def test_fuzzy_ranks_closest_name_first():
    matcher = ContactMatcher(CONTACTS)
    ranked = matcher._fuzzy("Jon Smith", top_k=3)
    assert ranked[0].contact.name == "John Smith"
    assert ranked[0].score >= ranked[-1].score


def test_resolve_empty_contacts():
    matcher = ContactMatcher([])
    matched, candidates = matcher.resolve("anyone")
    assert matched is None
    assert candidates == []
