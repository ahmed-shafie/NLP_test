"""Multilingual sentence embeddings (Arabic + English) via sentence-transformers.

Mirrors ``app/embeddings.py``. The embedder loads lazily and is cached. If the
model cannot be loaded (dependency missing, or no network on first download),
:func:`get_embedder` returns ``None`` and callers fall back to non-semantic
logic — so the template still runs. There is **no LLM** anywhere here; this is a
plain embedding model used only to measure sentence similarity.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from service_template.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper around a SentenceTransformer producing L2-normalised vectors.

    L2-normalised vectors mean an inner product equals cosine similarity, which
    is what the FAISS ``IndexFlatIP`` store expects (see ``vectorstore.py``).
    """

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model
        dim = model.get_sentence_embedding_dimension()
        if dim is None:
            dim = int(model.encode(["x"]).shape[-1])
        self.dimension: int = int(dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), dimension)`` float32 array of unit vectors."""

        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        """Embed a single string into a ``(dimension,)`` unit vector."""

        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder | None:
    """Load and cache the multilingual embedder, or ``None`` if unavailable."""

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embedding_model)
        logger.info("Loaded embedding model '%s'.", settings.embedding_model)
        return Embedder(model)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning(
            "Embedding model '%s' unavailable (%s); semantic features disabled.",
            settings.embedding_model,
            exc,
        )
        return None
