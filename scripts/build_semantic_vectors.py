"""Encode the indexed example corpus once and write the vector cache.

Run this at image-build time (or once locally) so every process that builds the
semantic index reads the vectors instead of spending minutes encoding them.

    python -m scripts.build_semantic_vectors

The output is keyed by the example texts and the embedding model, so it is
ignored automatically after a corpus edit or a model swap.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from app.config import settings
from app.embeddings import get_embedder
from app.nlu.corpus import load_corpus_examples
from app.nlu.examples import INTENT_EXAMPLES
from app.nlu.vector_cache import fingerprint, save

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    embedder = get_embedder()
    if embedder is None:
        print(f"Embedding model {settings.embedding_model!r} unavailable.")
        return 1

    texts = [text for text, _ in INTENT_EXAMPLES] + [
        example.text for example in load_corpus_examples()
    ]
    print(f"Encoding {len(texts)} examples with {settings.embedding_model}...")
    started = time.perf_counter()
    vectors = embedder.encode(texts)
    elapsed = time.perf_counter() - started

    path = Path(settings.semantic_vector_cache_path)
    stamp = fingerprint(texts, settings.embedding_model, embedder.dimension)
    save(path, vectors, stamp)
    size_mb = path.stat().st_size / 1_000_000 if path.exists() else 0.0
    print(
        f"Wrote {vectors.shape[0]}x{vectors.shape[1]} vectors to {path} "
        f"({size_mb:.1f} MB) in {elapsed:.1f}s."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
