"""A minimal FAISS-backed vector store with generic payloads.

Vectors are assumed to be L2-normalised, so inner product equals cosine
similarity. Payloads (arbitrary objects) are kept in a parallel list and
returned alongside scores at search time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import faiss
import numpy as np

P = TypeVar("P")


@dataclass(frozen=True)
class SearchHit(Generic[P]):
    """A single nearest-neighbour result."""

    payload: P
    score: float


class FaissVectorStore(Generic[P]):
    """In-process cosine-similarity index over unit vectors."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._payloads: list[P] = []

    def __len__(self) -> int:
        return len(self._payloads)

    def add(self, vectors: np.ndarray, payloads: list[P]) -> None:
        """Add a batch of vectors with their associated payloads."""

        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length")
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        self._index.add(vectors)
        self._payloads.extend(payloads)

    def search(self, vector: np.ndarray, k: int) -> list[SearchHit[P]]:
        """Return up to ``k`` nearest payloads ranked by cosine similarity."""

        if len(self._payloads) == 0:
            return []
        query = np.asarray(vector, dtype="float32").reshape(1, -1)
        k = min(k, len(self._payloads))
        scores, indices = self._index.search(query, k)
        hits: list[SearchHit[P]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx == -1:
                continue
            hits.append(SearchHit(payload=self._payloads[idx], score=float(score)))
        return hits
