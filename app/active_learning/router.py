"""FastAPI routes for the Active Learning review queue and index rebuild."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.active_learning.daemon import next_run_time
from app.active_learning.index_rebuilder import last_result, rebuild_index
from app.active_learning.schemas import (
    ActiveLearningStats,
    CaseStatus,
    RebuildResult,
    ReviewCase,
    ReviewDecision,
)
from app.active_learning.store import get_store
from app.config import settings
from app.nlu.semantic_intents import get_semantic_classifier

router = APIRouter(tags=["active-learning"], prefix="/active-learning")


def _require_active_learning() -> None:
    if not settings.active_learning_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Active learning is disabled.",
        )


@router.get("/queue", response_model=list[ReviewCase])
def review_queue(
    status_filter: CaseStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[ReviewCase]:
    """List logged cases (newest first), optionally filtered by status."""

    _require_active_learning()
    return get_store().list_cases(status=status_filter, limit=limit)


@router.get("/stats", response_model=ActiveLearningStats)
def stats() -> ActiveLearningStats:
    """Queue counts plus live index size and next scheduled rebuild."""

    _require_active_learning()
    result = get_store().stats()
    classifier = get_semantic_classifier()
    if classifier is not None:
        result.index_loaded = True
        result.index_base_examples = classifier.base_count
        result.index_learned_examples = classifier.extra_count
    if settings.index_rebuild_enabled:
        result.next_rebuild_utc = next_run_time().isoformat()
    last = last_result()
    if last is not None:
        result.last_rebuild_at = last.at
    return result


@router.post("/{case_id}/approve", response_model=ReviewCase)
def approve_case(case_id: int, decision: ReviewDecision) -> ReviewCase:
    """Approve a case so it joins the index on the next rebuild."""

    _require_active_learning()
    updated = get_store().decide(
        case_id,
        CaseStatus.APPROVED,
        corrected_intent=decision.corrected_intent,
        reviewer=decision.reviewer,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return updated


@router.post("/{case_id}/reject", response_model=ReviewCase)
def reject_case(case_id: int, decision: ReviewDecision) -> ReviewCase:
    """Reject a case so it is excluded from the index."""

    _require_active_learning()
    updated = get_store().decide(
        case_id, CaseStatus.REJECTED, reviewer=decision.reviewer
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return updated


@router.post("/rebuild", response_model=RebuildResult)
def rebuild() -> RebuildResult:
    """Manually rebuild + hot-swap the intent index now (no restart)."""

    _require_active_learning()
    return rebuild_index()
