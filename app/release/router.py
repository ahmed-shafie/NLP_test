"""Authenticated read: which release is live, and does the tree still match it.

Same key and same fail-closed rule as the turn store. Answering "what is
running" is an operational question, and ``drift`` is the one field worth paging
on: a live version whose data files no longer hash to the manifest means the
running behaviour is not the behaviour that was measured and approved.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.observability.router import require_ops_key
from app.release import registry
from app.release.build import REPO_ROOT
from app.release.manifest import drifted_files

router = APIRouter(
    prefix="/ops/release",
    tags=["release"],
    dependencies=[Depends(require_ops_key)],
)


class ReleaseEvent(BaseModel):
    """One promote/rollback entry."""

    at: str
    action: str
    version: str
    actor: str
    reason: str


class LiveRelease(BaseModel):
    """The live release as operations needs to see it."""

    version: str | None
    since: str | None
    action: str | None
    actor: str | None
    approved_by: str | None = None
    gate_split: str | None = None
    calibration_split: str | None = None
    intent_accuracy: float | None = None
    money_flow_breaches: int | None = None
    thresholds: dict[str, float | bool] = {}
    drift: list[str] = []
    versions: list[str] = []


@router.get("", response_model=LiveRelease)
def live_release() -> LiveRelease:
    """What is running, how it was measured, and whether it still matches."""

    live = registry.current()
    if live is None:
        return LiveRelease(
            version=None,
            since=None,
            action=None,
            actor=None,
            versions=registry.versions(),
        )
    manifest = registry.load(live.version)
    return LiveRelease(
        version=live.version,
        since=live.since,
        action=live.action,
        actor=live.actor,
        approved_by=manifest.approved_by,
        gate_split=manifest.gate_split,
        calibration_split=manifest.calibration_split,
        intent_accuracy=manifest.metrics.intent_accuracy,
        money_flow_breaches=manifest.metrics.money_flow_breaches,
        thresholds=manifest.thresholds,
        drift=drifted_files(manifest, REPO_ROOT),
        versions=registry.versions(),
    )


@router.get("/history", response_model=list[ReleaseEvent])
def release_history() -> list[ReleaseEvent]:
    """Every promote and rollback, oldest first."""

    return [
        ReleaseEvent(
            at=entry.at,
            action=entry.action,
            version=entry.version,
            actor=entry.actor,
            reason=entry.reason,
        )
        for entry in registry.history()
    ]
