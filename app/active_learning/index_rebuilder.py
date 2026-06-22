"""Rebuild the semantic intent index from base + approved examples, and hot-swap it.

The rebuild gathers approved review-queue examples, appends them to the built-in
examples, builds a fresh FAISS index, and atomically swaps it into the live
classifier (see :func:`app.nlu.semantic_intents.rebuild_semantic_classifier`).
No restart is required.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from app.active_learning.schemas import RebuildResult
from app.active_learning.store import get_store
from app.nlu.examples import INTENT_EXAMPLES
from app.nlu.semantic_intents import rebuild_semantic_classifier

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_last_result: RebuildResult | None = None


def last_result() -> RebuildResult | None:
    """Return the most recent rebuild result, if any has run this process."""

    with _lock:
        return _last_result


def rebuild_index() -> RebuildResult:
    """Rebuild the intent index from base + approved examples and hot-swap it in."""

    global _last_result
    now = datetime.now(UTC)
    learned = get_store().approved_examples()
    base = len(INTENT_EXAMPLES)

    classifier = rebuild_semantic_classifier(learned)
    if classifier is None:
        result = RebuildResult(
            ok=False,
            at=now,
            base_examples=base,
            learned_examples=len(learned),
            message=(
                "Semantic classifier unavailable (embedding model not loaded); "
                "index not rebuilt."
            ),
        )
        logger.warning("Index rebuild skipped: classifier unavailable.")
    else:
        result = RebuildResult(
            ok=True,
            at=now,
            total_examples=len(classifier),
            base_examples=base,
            learned_examples=len(learned),
            message=(
                f"Rebuilt index with {len(classifier)} examples "
                f"({base} base + {len(learned)} learned)."
            ),
        )
        logger.info(result.message)

    with _lock:
        _last_result = result
    return result
