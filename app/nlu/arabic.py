"""Arabic NLU backed by Stanza, with a regex fallback when no model is present."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import settings
from app.nlu import entities
from app.schemas import Language, TransferEntities

if TYPE_CHECKING:
    from stanza import Pipeline as StanzaPipeline

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> "StanzaPipeline | None":
    """Load and cache the Stanza Arabic pipeline, or ``None`` if unavailable.

    Stanza models are downloaded separately (see README). When absent, the
    pipeline degrades to regex-based slot extraction.
    """

    try:
        import stanza

        return stanza.Pipeline(
            lang=settings.stanza_lang,
            processors="tokenize,ner",
            download_method=None,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully to regex
        logger.warning(
            "Stanza Arabic model unavailable (%s); using regex fallback.", exc
        )
        return None


def _recipient(text: str) -> str | None:
    """Prefer a Stanza PER entity for the beneficiary; fall back to regex."""

    nlp = _load_model()
    if nlp is not None:
        doc = nlp(text)
        people = [ent.text.strip() for ent in doc.ents if ent.type in {"PER", "PERSON"}]
        if people:
            return people[0]
    return entities.extract_recipient(text, Language.AR)


def extract_entities(text: str) -> TransferEntities:
    """Extract transfer slots from an Arabic utterance."""

    return TransferEntities(
        amount=entities.extract_amount(text),
        currency=entities.extract_currency(text),
        recipient=_recipient(text),
        source_account=entities.extract_source_account(text, Language.AR),
    )
