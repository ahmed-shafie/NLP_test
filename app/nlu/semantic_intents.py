"""Semantic intent classification backed by FAISS + multilingual embeddings.

At startup the labeled examples in :mod:`app.nlu.examples` are embedded and
indexed. An utterance is classified by retrieving its nearest example vectors
and aggregating their similarity scores per intent. If embeddings are
unavailable, :func:`get_semantic_classifier` returns ``None`` and the caller
falls back to the keyword classifier.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass

from app.config import settings
from app.embeddings import get_embedder
from app.nlu.examples import INTENT_EXAMPLES
from app.schemas import Intent
from app.vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimilarExample:
    """A retrieved example utterance and its similarity to the query."""

    text: str
    intent: Intent
    score: float


class SemanticIntentClassifier:
    """Nearest-neighbour intent classifier over embedded example utterances."""

    def __init__(self, extra_examples: list[tuple[str, Intent]] | None = None) -> None:
        embedder = get_embedder()
        if embedder is None:
            raise RuntimeError(
                "Embedder unavailable; cannot build semantic classifier."
            )
        self._embedder = embedder
        self._store: FaissVectorStore[tuple[str, Intent]] = FaissVectorStore(
            embedder.dimension
        )

        examples = list(INTENT_EXAMPLES) + list(extra_examples or [])
        self.base_count = len(INTENT_EXAMPLES)
        self.extra_count = len(extra_examples or [])
        texts = [text for text, _ in examples]
        vectors = embedder.encode(texts)
        self._store.add(vectors, examples)
        logger.info(
            "Indexed %d intent examples (%d base + %d learned).",
            len(examples),
            self.base_count,
            self.extra_count,
        )

    def __len__(self) -> int:
        return len(self._store)

    def similar(self, text: str, k: int | None = None) -> list[SimilarExample]:
        """Return the ``k`` nearest example utterances to ``text``."""

        top_k = k or settings.semantic_top_k
        query = self._embedder.encode_one(text)
        hits = self._store.search(query, top_k)
        return [
            SimilarExample(text=hit.payload[0], intent=hit.payload[1], score=hit.score)
            for hit in hits
        ]

    def classify(self, text: str) -> tuple[Intent, float]:
        """Classify ``text`` by aggregating nearest-neighbour similarity per intent.

        Confidence is the summed similarity of the winning intent divided by the
        total similarity across retrieved neighbours, scaled by the top hit's
        score. Returns ``FALLBACK`` if the best neighbour is below threshold.
        """

        neighbours = self.similar(text)
        if not neighbours:
            return Intent.FALLBACK, 0.0

        top = neighbours[0]
        if top.score < settings.semantic_intent_threshold:
            return Intent.FALLBACK, round(max(top.score, 0.0), 4)

        weights: dict[Intent, float] = defaultdict(float)
        for n in neighbours:
            weights[n.intent] += max(n.score, 0.0)
        total = sum(weights.values()) or 1.0

        best_intent = max(weights, key=lambda i: weights[i])
        share = weights[best_intent] / total
        # Confidence blends neighbour agreement (share) with the top similarity.
        confidence = round(min(0.5 * share + 0.5 * top.score, 0.99), 4)
        return best_intent, confidence


# The live classifier is held behind a lock so the active-learning daemon can swap
# in a freshly rebuilt index atomically while requests are in flight.
_lock = threading.Lock()
_classifier: SemanticIntentClassifier | None = None
_built = False


def _build(
    extra_examples: list[tuple[str, Intent]] | None = None,
) -> SemanticIntentClassifier | None:
    if not settings.use_semantic_intent:
        return None
    try:
        return SemanticIntentClassifier(extra_examples)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("Semantic intent classifier unavailable (%s).", exc)
        return None


def get_semantic_classifier() -> SemanticIntentClassifier | None:
    """Return the live semantic classifier (built once), or ``None`` if unavailable."""

    global _classifier, _built
    with _lock:
        if not _built:
            _classifier = _build()
            _built = True
        return _classifier


def rebuild_semantic_classifier(
    extra_examples: list[tuple[str, Intent]] | None = None,
) -> SemanticIntentClassifier | None:
    """Rebuild the index from base + ``extra_examples`` and hot-swap it in.

    The new classifier is built outside the lock (the slow part) and then swapped
    atomically, so in-flight ``get_semantic_classifier`` callers never see a
    half-built index. Returns the new classifier (or ``None`` if unavailable).
    """

    global _classifier, _built
    new = _build(extra_examples)
    with _lock:
        _classifier = new
        _built = True
    return new


def reset_semantic_classifier() -> None:
    """Drop the cached classifier so the next access rebuilds it (used by tests)."""

    global _classifier, _built
    with _lock:
        _classifier = None
        _built = False
