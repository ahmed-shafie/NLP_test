"""Pydantic schemas for the Active Learning review queue and index rebuild."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas import Intent, Language


class CaseStatus(str, Enum):
    """Lifecycle of a logged case in the review queue."""

    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewCase(BaseModel):
    """A single logged case awaiting (or having received) a review decision."""

    id: int
    created_at: datetime
    text: str
    language: Language
    predicted_intent: Intent
    confidence: float
    intent_source: str
    llm_assisted: bool
    clarification: str | None = None
    status: CaseStatus
    corrected_intent: Intent | None = Field(
        default=None,
        description="Reviewer-supplied label that overrides the predicted intent.",
    )
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    source: str = Field(description="Where the case originated, e.g. 'nlu.parse'.")

    @property
    def effective_intent(self) -> Intent:
        """Training label: the correction if present, else the prediction."""

        return self.corrected_intent or self.predicted_intent


class ReviewDecision(BaseModel):
    """Body for approving a case, optionally correcting the intent label."""

    corrected_intent: Intent | None = Field(
        default=None,
        description="Override the predicted intent before adding it to the index.",
    )
    reviewer: str | None = Field(
        default=None, description="Identifier of the human reviewer."
    )


class ActiveLearningStats(BaseModel):
    """Aggregate counts for the review-queue dashboard."""

    total: int = 0
    pending: int = 0
    approved: int = 0
    auto_approved: int = 0
    rejected: int = 0
    learned_examples: int = 0
    index_base_examples: int = 0
    index_learned_examples: int = 0
    index_loaded: bool = False
    next_rebuild_utc: str | None = None
    last_rebuild_at: datetime | None = None


class RebuildResult(BaseModel):
    """Outcome of a (manual or scheduled) intent-index rebuild + hot-swap."""

    ok: bool
    total_examples: int = 0
    base_examples: int = 0
    learned_examples: int = 0
    at: datetime
    message: str
