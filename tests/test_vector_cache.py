"""The example-vector cache must be fast when valid and ignored when stale."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.nlu import vector_cache


class FakeEmbedder:
    """Deterministic stand-in that counts how many texts it encoded."""

    dimension = 4

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encoded.extend(texts)
        rows = [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]
        return np.asarray(rows, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    path = tmp_path / "vectors.npy"
    monkeypatch.setattr(settings, "semantic_vector_cache_path", str(path))
    monkeypatch.setattr(settings, "semantic_vector_cache_enabled", True)
    monkeypatch.setattr(settings, "semantic_vector_cache_write", True)
    return path


def test_second_build_reads_the_cache_instead_of_encoding(cache_path):
    texts = ["حول 500 لأحمد", "what is my balance"]

    first = FakeEmbedder()
    written = vector_cache.encode_cached(first, texts)
    assert first.encoded == texts
    assert cache_path.exists()

    second = FakeEmbedder()
    read = vector_cache.encode_cached(second, texts)
    assert second.encoded == []  # nothing re-encoded
    np.testing.assert_array_equal(read, written)
    assert read.dtype == np.float32


def test_an_edited_corpus_is_re_encoded(cache_path):
    vector_cache.encode_cached(FakeEmbedder(), ["one", "two"])

    embedder = FakeEmbedder()
    vectors = vector_cache.encode_cached(embedder, ["one", "two", "three"])
    assert embedder.encoded == ["one", "two", "three"]
    assert vectors.shape[0] == 3


def test_a_model_swap_is_re_encoded(cache_path, monkeypatch):
    texts = ["one", "two"]
    vector_cache.encode_cached(FakeEmbedder(), texts)

    monkeypatch.setattr(settings, "embedding_model", "some/other-model")
    embedder = FakeEmbedder()
    vector_cache.encode_cached(embedder, texts)
    assert embedder.encoded == texts


def test_a_corrupt_cache_falls_back_to_encoding(cache_path):
    texts = ["one", "two"]
    vector_cache.encode_cached(FakeEmbedder(), texts)
    cache_path.write_bytes(b"not a numpy file")

    embedder = FakeEmbedder()
    vectors = vector_cache.encode_cached(embedder, texts)
    assert embedder.encoded == texts
    assert vectors.shape == (2, 4)


def test_disabled_cache_never_writes(tmp_path, monkeypatch):
    path = tmp_path / "vectors.npy"
    monkeypatch.setattr(settings, "semantic_vector_cache_path", str(path))
    monkeypatch.setattr(settings, "semantic_vector_cache_enabled", False)

    embedder = FakeEmbedder()
    vector_cache.encode_cached(embedder, ["one"])
    assert embedder.encoded == ["one"]
    assert not path.exists()


def test_empty_input_needs_no_model(cache_path):
    embedder = FakeEmbedder()
    vectors = vector_cache.encode_cached(embedder, [])
    assert vectors.shape == (0, 4)
    assert embedder.encoded == []
