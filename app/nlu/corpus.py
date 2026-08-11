"""Curated example corpus loaded into the semantic intent index at startup.

``data/example_corpus.jsonl`` holds bilingual, multi-dialect utterances (MSA,
Palestinian, Gulf, Moroccan, Tunisian, Egyptian, Levantine, English). Rows whose
``topic`` is set are customer-service questions the engine cannot execute; they
carry the ``fallback`` intent so the classifier learns to refuse them instead of
opening a money flow. See ``scripts/build_example_corpus.py`` for how the file is
produced and ``research/vector_db_v08/`` for the measurements behind the cap.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.schemas import Intent

logger = logging.getLogger(__name__)

CORPUS_PATH = Path(__file__).parent / "data" / "example_corpus.jsonl"


@lru_cache(maxsize=1)
def load_corpus_examples() -> tuple[tuple[str, Intent], ...]:
    """Return ``(utterance, intent)`` pairs from the corpus file.

    A missing or malformed file degrades to an empty corpus: the built-in
    examples alone still classify, and refusing to start would take the whole
    assistant down over an optional data file.
    """

    if not settings.example_corpus_enabled:
        return ()
    try:
        lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Example corpus unavailable (%s); using built-ins only.", exc)
        return ()

    rows: list[tuple[str, Intent]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows.append((str(row["text"]), Intent(row["intent"])))
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping malformed corpus row (%s).", exc)
    return tuple(rows)
