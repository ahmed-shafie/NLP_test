"""Tests for the FAISS vector store (no embedding model required)."""

import numpy as np

from app.vectorstore import FaissVectorStore


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype="float32")
    return arr / np.linalg.norm(arr)


def test_empty_search_returns_nothing():
    store: FaissVectorStore[str] = FaissVectorStore(3)
    assert store.search(_unit([1, 0, 0]), k=3) == []


def test_add_and_search_orders_by_cosine():
    store: FaissVectorStore[str] = FaissVectorStore(3)
    vectors = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([1, 1, 0])])
    store.add(vectors, ["x", "y", "xy"])

    hits = store.search(_unit([1, 0, 0]), k=3)
    assert len(hits) == 3
    assert hits[0].payload == "x"
    assert hits[0].score > hits[1].score


def test_length_mismatch_raises():
    store: FaissVectorStore[str] = FaissVectorStore(2)
    try:
        store.add(np.zeros((2, 2), dtype="float32"), ["only-one"])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched lengths")
