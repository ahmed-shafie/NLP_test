"""English NLU backed by spaCy, with a regex fallback when no model is present."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import settings
from app.nlu import entities
from app.schemas import Language, TransferEntities

if TYPE_CHECKING:
    from spacy.language import Language as SpacyPipeline

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SpacyPipeline | None:
    """Load and cache the spaCy English model, or ``None`` if unavailable."""

    try:
        import spacy

        return spacy.load(settings.spacy_model)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully to regex
        logger.warning(
            "spaCy model '%s' unavailable (%s); using regex fallback.",
            settings.spacy_model,
            exc,
        )
        return None


def _recipient(text: str) -> str | None:
    """Prefer a spaCy PERSON entity for the beneficiary; fall back to regex."""

    nlp = _load_model()
    if nlp is not None:
        doc = nlp(text)
        people = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
        if people:
            return people[0]
    return entities.extract_recipient(text, Language.EN)


def extract_entities(text: str) -> TransferEntities:
    """Extract transfer slots from an English utterance."""

    return TransferEntities(
        amount=entities.extract_amount(text),
        currency=entities.extract_currency(text),
        recipient=_recipient(text),
        source_account=entities.extract_source_account(text, Language.EN),
    )
