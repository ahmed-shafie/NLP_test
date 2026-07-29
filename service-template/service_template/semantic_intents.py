"""Semantic intent classification backed by FAISS + multilingual embeddings.

Mirrors ``app/nlu/semantic_intents.py`` (minus the abuse-moderation special
case). At first use the labelled examples in ``examples.py`` are embedded and
indexed in a FAISS store. An utterance is classified by retrieving its nearest
example vectors and aggregating their similarity per intent.

If embeddings are unavailable, :func:`get_semantic_classifier` returns ``None``
and the caller (``extractor.detect_intent``) falls back to the keyword
classifier. **No LLM is involved** — this is pure nearest-neighbour retrieval.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass

from service_template.config import settings
from service_template.embeddings import get_embedder
from service_template.examples import INTENT_EXAMPLES
from service_template.schemas import Intent
from service_template.vectorstore import FaissVectorStore

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
            raise RuntimeError("Embedder unavailable; cannot build classifier.")
        self._embedder = embedder
        self._store: FaissVectorStore[tuple[str, Intent]] = FaissVectorStore(
            embedder.dimension
        )
        examples = list(INTENT_EXAMPLES)
        texts = [text for text, _ in examples]
        self._store.add(embedder.encode(texts), examples)
        logger.info("Indexed %d intent examples.", len(examples))

    def __len__(self) -> int:
        return len(self._store)

    def similar(self, text: str, k: int | None = None) -> list[SimilarExample]:
        """Return the ``k`` nearest example utterances to ``text``."""

        query = self._embedder.encode_one(text)
        hits = self._store.search(query, k or settings.semantic_top_k)
        return [
            SimilarExample(text=h.payload[0], intent=h.payload[1], score=h.score)
            for h in hits
        ]

    def classify(self, text: str) -> tuple[Intent, float]:
        """Classify ``text`` by aggregating nearest-neighbour similarity per intent.

        Returns ``(FALLBACK, score)`` when the best neighbour is below the
        configured similarity threshold, so the caller can defer to keywords.
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


# Built once and cached behind a lock (so a background rebuild could hot-swap it,
# as the main app's active-learning daemon does).
_lock = threading.Lock()
_classifier: SemanticIntentClassifier | None = None
_built = False


def _build() -> SemanticIntentClassifier | None:
    if not settings.use_semantic_intent:
        return None
    try:
        return SemanticIntentClassifier()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("Semantic intent classifier unavailable (%s).", exc)
        return None


def get_semantic_classifier() -> SemanticIntentClassifier | None:
    """Return the live classifier (built once), or ``None`` if unavailable."""

    global _classifier, _built
    with _lock:
        if not _built:
            _classifier = _build()
            _built = True
        return _classifier


def reset_semantic_classifier() -> None:
    """Drop the cached classifier so the next access rebuilds it (used by tests)."""

    global _classifier, _built
    with _lock:
        _classifier = None
        _built = False
