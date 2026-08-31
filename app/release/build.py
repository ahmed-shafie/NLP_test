"""Build a candidate release: measure it on held-out rows, then describe it.

The order is the point. Metrics come from the **test** half of the gold set,
which nothing in the runtime is allowed to be tuned on, plus the hard negatives
(zero tolerance). The dev half is measured too, but only so a reviewer can see
the gap between the rows the thresholds saw and the rows they did not — a dev
score far above test is overfitting, visible instead of hidden.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.eval.harness import GOLD_PATH, Report, check_thresholds, evaluate
from app.eval.splits import (
    HARD_NEGATIVES_PATH,
    Split,
    load_hard_negatives,
    load_split,
    money_flow_breaches,
)
from app.release.manifest import (
    FileDigest,
    Manifest,
    Metrics,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The non-code inputs the customer-facing decision actually reads. A change to
# any of these changes answers without changing a line of Python, so a release
# that does not pin them pins nothing.
RUNTIME_DATA_FILES: tuple[str, ...] = (
    "app/nlu/data/topic_head.npz",
    "app/nlu/data/example_corpus.jsonl",
    "app/nlu/data/semantic_vectors.npy",
    "app/data/sadad_billers.csv",
    "app/data/names.csv",
    "app/data/blocklist.csv",
    "app/eval/nlu_gold.jsonl",
    "app/eval/hard_negatives.jsonl",
)

# Every setting the routing decision is calibrated on. Recorded by name so a
# release can be compared with the one before it field by field.
THRESHOLD_FIELDS: tuple[str, ...] = (
    "intent_threshold",
    "semantic_intent_threshold",
    "semantic_route_threshold",
    "topic_reply_threshold",
    "topic_reply_unanimous_threshold",
    "topic_reply_unanimous_threshold_en",
    "topic_head_enabled",
    "topic_head_threshold",
    "topic_head_score_floor",
    "contact_match_threshold",
    "biller_match_threshold",
    "moderation_semantic_threshold",
)


def thresholds_snapshot() -> dict[str, float | bool]:
    """The calibrated settings in force, by name."""

    values = settings.model_dump()
    snapshot: dict[str, float | bool] = {}
    for name in THRESHOLD_FIELDS:
        value = values[name]
        if isinstance(value, bool):
            snapshot[name] = value
        else:
            snapshot[name] = float(value)
    return snapshot


def _metrics(report: Report, breaches: int) -> Metrics:
    return Metrics(
        intent_accuracy=round(report.intent_accuracy, 6),
        rows=report.total,
        slot_f1={slot: round(score.f1, 6) for slot, score in report.slots.items()},
        over_blocks=len(report.over_blocks),
        wrong_flow_starts=len(report.wrong_flow_starts),
        money_flow_breaches=breaches,
    )


def shipped_files(root: Path = REPO_ROOT) -> list[FileDigest]:
    """Digest every runtime data file that exists."""

    digests: list[FileDigest] = []
    for relative in RUNTIME_DATA_FILES:
        path = root / relative
        if not path.exists():
            continue
        digests.append(
            FileDigest(
                path=relative,
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
            )
        )
    return digests


def gate_violations(report: Report, breaches: list[str]) -> list[str]:
    """Everything that must be empty before a candidate may be promoted."""

    return check_thresholds(report) + breaches


def build_candidate(
    version: str,
    *,
    notes: str = "",
    parent_version: str | None = None,
    profile: str = "default",
    root: Path = REPO_ROOT,
) -> tuple[Manifest, list[str]]:
    """Measure the current tree and return ``(manifest, gate violations)``.

    Never promotes and never writes: building is measurement. Promotion is a
    separate, named act (:mod:`app.release.registry`).
    """

    test_report = evaluate(load_split(Split.TEST))
    dev_report = evaluate(load_split(Split.DEV))
    negatives = load_hard_negatives()
    breaches = money_flow_breaches(negatives)

    manifest = Manifest(
        version=version,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        manifest_version=1,
        thresholds=thresholds_snapshot(),
        files=shipped_files(root),
        metrics=_metrics(test_report, len(breaches)),
        dev_metrics=_metrics(dev_report, len(breaches)),
        calibration_split=Split.DEV.value,
        gate_split=Split.TEST.value,
        gold_sha256=sha256_file(GOLD_PATH),
        hard_negatives_sha256=sha256_file(HARD_NEGATIVES_PATH),
        notes=notes,
        parent_version=parent_version,
        profile=profile,
        hard_negative_count=len(negatives),
    )
    return manifest, gate_violations(test_report, breaches)
