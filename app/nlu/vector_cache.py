"""On-disk cache for the embedded example corpus.

Encoding the ~32k indexed examples costs minutes of CPU and happens in every
process that builds the semantic index (each worker, each test session, each
active-learning rebuild). The vectors are a pure function of the example texts
and the embedding model, so they can be computed once and read back.

The cache is only used when its fingerprint matches the texts, the model name
and the dimension it was written for, so a corpus edit or a model swap falls
back to encoding instead of serving stale vectors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from app.config import settings
from app.embeddings import Embedder

logger = logging.getLogger(__name__)


def fingerprint(texts: list[str], model: str, dimension: int) -> str:
    """Identify exactly which texts, model and dimension a cache was built for."""

    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(dimension).encode("utf-8"))
    digest.update(b"\0")
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def load(path: Path, expected: str) -> np.ndarray | None:
    """Read cached vectors, or ``None`` when absent, stale or unreadable."""

    meta_path = _meta_path(path)
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != expected:
            logger.info("Vector cache %s is stale; re-encoding.", path)
            return None
        vectors = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - a bad cache must never break startup
        logger.warning("Vector cache %s unusable (%s); re-encoding.", path, exc)
        return None
    if vectors.ndim != 2 or vectors.shape[0] != meta.get("count"):
        logger.warning("Vector cache %s has an unexpected shape; re-encoding.", path)
        return None
    return np.ascontiguousarray(vectors, dtype="float32")


def save(path: Path, vectors: np.ndarray, expected: str) -> None:
    """Write vectors and their fingerprint, atomically and best-effort."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        np.save(tmp, vectors, allow_pickle=False)
        # ``np.save`` appends .npy when the name lacks it; keep the real name.
        written = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npy")
        os.replace(written, path)
        _meta_path(path).write_text(
            json.dumps(
                {
                    "fingerprint": expected,
                    "count": int(vectors.shape[0]),
                    "dimension": int(vectors.shape[1]),
                    "model": settings.embedding_model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote vector cache %s (%d vectors).", path, vectors.shape[0])
    except Exception as exc:  # noqa: BLE001 - the cache is an optimisation only
        logger.warning("Could not write vector cache %s (%s).", path, exc)


def encode_cached(embedder: Embedder, texts: list[str]) -> np.ndarray:
    """Return vectors for ``texts``, reading (and populating) the on-disk cache."""

    if not texts:
        return np.zeros((0, embedder.dimension), dtype="float32")
    if not settings.semantic_vector_cache_enabled:
        return embedder.encode(texts)

    path = Path(settings.semantic_vector_cache_path)
    expected = fingerprint(texts, settings.embedding_model, embedder.dimension)
    cached = load(path, expected)
    if cached is not None:
        logger.info("Loaded %d example vectors from %s.", cached.shape[0], path)
        return cached

    vectors = embedder.encode(texts)
    if settings.semantic_vector_cache_write:
        save(path, vectors, expected)
    return vectors
