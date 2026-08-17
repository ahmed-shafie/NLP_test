"""A trained head that names the answer a customer-service question deserves.

Retrieval alone cannot say how sure it is: cosine similarity is a distance, so
the gate in :mod:`app.conversation.topic_replies` has to set one bar high enough
for its worst subject and every other subject pays for it — 85% of Arabic
service questions get the generic menu. This head is a small supervised
classifier over the **same** vector the index already computes for the query, so
it costs no extra model and no extra encode, and it does produce a probability.

It predicts the *answer* (a reviewed family reply, or a topic's specific reply)
rather than the raw topic, because that is what the customer reads: two topics
sharing one reply cannot mislead anybody. It also predicts an explicit "no
answer" class, trained on the executable rows (transfer, bill, balance, ...), so
a head that answers more questions cannot start answering a transfer request.

The weights ship as plain arrays and the forward pass is written out here, so
the customer-facing path needs numpy alone: no scikit-learn at runtime and no
pickle to trust. ``scripts/train_topic_classifier.py`` produces the file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).parent / "data" / "topic_head.npz"

# The label meaning "this question gets no topic answer".
NO_ANSWER = ""


@dataclass(frozen=True)
class Prediction:
    """The answer key the head names, and how sure it is."""

    key: str
    probability: float


class TopicHead:
    """One hidden layer over an utterance embedding; numpy only."""

    def __init__(
        self,
        weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        keys: tuple[str, ...],
    ) -> None:
        self._w1, self._b1, self._w2, self._b2 = weights
        self._keys = keys

    @property
    def dimension(self) -> int:
        return int(self._w1.shape[0])

    @property
    def answers(self) -> tuple[str, ...]:
        return self._keys

    def predict(self, vector: np.ndarray) -> Prediction:
        """Name the most likely answer key for one utterance vector."""

        hidden = np.maximum(vector @ self._w1 + self._b1, 0.0)
        logits = hidden @ self._w2 + self._b2
        logits -= logits.max()
        exp = np.exp(logits)
        probabilities = exp / exp.sum()
        best = int(probabilities.argmax())
        return Prediction(key=self._keys[best], probability=float(probabilities[best]))


@lru_cache(maxsize=1)
def get_topic_head() -> TopicHead | None:
    """Load the trained head, or ``None`` if it cannot be used.

    A missing file, a file trained for a different embedding model, or a
    dimension mismatch all degrade to ``None``: the retrieval gate alone still
    answers, and refusing to start would take the assistant down over an
    optional model file.
    """

    if not settings.topic_head_enabled:
        return None
    try:
        with np.load(WEIGHTS_PATH, allow_pickle=False) as data:
            trained_for = str(data["embedding_model"])
            if trained_for != settings.embedding_model:
                logger.warning(
                    "Topic head was trained for '%s' but the embedder is '%s'; "
                    "disabling it.",
                    trained_for,
                    settings.embedding_model,
                )
                return None
            head = TopicHead(
                weights=(data["w1"], data["b1"], data["w2"], data["b2"]),
                keys=tuple(str(key) for key in data["keys"]),
            )
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Topic head unavailable (%s); using retrieval alone.", exc)
        return None
    logger.info(
        "Loaded topic head: %d answers over %d dimensions.",
        len(head.answers),
        head.dimension,
    )
    return head
