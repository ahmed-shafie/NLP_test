"""Semantic intent classification backed by FAISS + multilingual embeddings.

At startup the labeled examples in :mod:`app.nlu.examples` are embedded and
indexed. An utterance is classified by retrieving its nearest example vectors
and aggregating their similarity scores per intent. If embeddings are
unavailable, :func:`get_semantic_classifier` returns ``None`` and the caller
falls back to the keyword classifier.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

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

    def __init__(self) -> None:
        embedder = get_embedder()
        if embedder is None:
            raise RuntimeError(
                "Embedder unavailable; cannot build semantic classifier."
            )
        self._embedder = embedder
        self._store: FaissVectorStore[tuple[str, Intent]] = FaissVectorStore(
            embedder.dimension
        )

        texts = [text for text, _ in INTENT_EXAMPLES]
        vectors = embedder.encode(texts)
        self._store.add(vectors, list(INTENT_EXAMPLES))
        logger.info("Indexed %d intent examples.", len(INTENT_EXAMPLES))

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


@lru_cache(maxsize=1)
def get_semantic_classifier() -> SemanticIntentClassifier | None:
    """Build and cache the semantic classifier, or ``None`` if unavailable."""

    if not settings.use_semantic_intent:
        return None
    try:
        return SemanticIntentClassifier()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("Semantic intent classifier unavailable (%s).", exc)
        return None
