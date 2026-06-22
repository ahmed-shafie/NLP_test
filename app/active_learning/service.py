"""Active Learning policy: decide which cases to log and which to auto-approve.

Runs passively at the tail of the pipeline. A case is *interesting* (worth
logging) when the deterministic layers were uncertain — the LLM was invoked, the
intent fell back, or the intent confidence was below
``NLU_ACTIVE_LEARNING_LOG_CONFIDENCE``. Confident, deterministic results are not
logged (there is nothing to learn). An interesting case with a concrete,
high-confidence intent is auto-approved; otherwise it waits for human review.
"""

from __future__ import annotations

import logging

from app.active_learning.schemas import CaseStatus, ReviewCase
from app.active_learning.store import get_store
from app.config import settings
from app.schemas import Intent, NLUResponse

logger = logging.getLogger(__name__)


def _classify(response: NLUResponse) -> CaseStatus | None:
    """Return the status to log this case under, or ``None`` to skip logging."""

    is_uncertain = (
        response.llm_assisted
        or response.intent is Intent.FALLBACK
        or response.confidence < settings.active_learning_log_confidence
    )
    if not is_uncertain:
        return None

    auto_ok = (
        response.intent is not Intent.FALLBACK
        and response.confidence >= settings.active_learning_auto_approve_confidence
    )
    return CaseStatus.AUTO_APPROVED if auto_ok else CaseStatus.PENDING


def record_case(response: NLUResponse, source: str) -> ReviewCase | None:
    """Log ``response`` to the review queue if it is worth learning from.

    Returns the logged :class:`ReviewCase`, or ``None`` when the case was skipped
    or active learning is disabled. Never raises: a logging failure must not break
    the request that produced it.
    """

    if not settings.active_learning_enabled:
        return None
    status = _classify(response)
    if status is None:
        return None
    try:
        return get_store().log_case(
            text=response.text,
            language=response.language,
            predicted_intent=response.intent,
            confidence=response.confidence,
            intent_source=response.intent_source,
            llm_assisted=response.llm_assisted,
            clarification=response.clarification,
            status=status,
            source=source,
        )
    except Exception:  # noqa: BLE001 - active learning must never break a request
        logger.warning("Failed to log active-learning case", exc_info=True)
        return None
