"""What a release is: a version, the bytes it shipped, and how it was measured.

A model layer that cannot be pinned cannot be rolled back. The runtime reads
several data files that are not code — the embedded example corpus, the
precomputed vectors, the trained topic head, the biller catalogue — and it reads
a dozen thresholds out of settings. "Which of those was live when the customer
was told the wrong thing?" has to be answerable from a record, not from a guess.

So a manifest names, for one version:

* every runtime data file with its **sha256** — the bytes, not the filename,
* the **thresholds** in force,
* the metrics, and **which split produced them**,
* the checksums of the gold and hard-negative sets the metrics came from.

The split fields are the part with teeth. ``calibration_split`` must not equal
``gate_split``: a threshold fitted on the rows that later grade it cannot fail,
which is exactly how an imported evaluation service was scoring itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


class LeakyManifest(Exception):
    """Raised when a manifest calibrated on the rows it was gated with."""


@dataclass(frozen=True, slots=True)
class FileDigest:
    """One shipped file, by content."""

    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class Metrics:
    """The measurements a release is allowed to be judged on."""

    intent_accuracy: float
    rows: int
    slot_f1: dict[str, float]
    over_blocks: int
    wrong_flow_starts: int
    money_flow_breaches: int


@dataclass(frozen=True, slots=True)
class Manifest:
    """The full, verifiable description of one release."""

    version: str
    created_at: str
    manifest_version: int
    thresholds: dict[str, float | bool]
    files: list[FileDigest]
    metrics: Metrics
    calibration_split: str
    gate_split: str
    gold_sha256: str
    hard_negatives_sha256: str
    notes: str = ""
    approved_by: str | None = None
    approved_at: str | None = None
    parent_version: str | None = None
    profile: str = "default"
    hard_negative_count: int = 0
    dev_metrics: Metrics | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.calibration_split == self.gate_split:
            raise LeakyManifest(
                "calibration_split and gate_split are both "
                f"{self.gate_split!r}: a threshold fitted on the rows that "
                "grade it cannot fail the gate."
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> Manifest:
        obj = json.loads(raw)
        files = [FileDigest(**entry) for entry in obj.pop("files")]
        metrics = Metrics(**obj.pop("metrics"))
        dev = obj.pop("dev_metrics", None)
        return cls(
            files=files,
            metrics=metrics,
            dev_metrics=Metrics(**dev) if dev else None,
            **obj,
        )

    def approve(self, approved_by: str) -> Manifest:
        """A copy of this manifest signed off by a named human."""

        return Manifest(
            version=self.version,
            created_at=self.created_at,
            manifest_version=self.manifest_version,
            thresholds=self.thresholds,
            files=self.files,
            metrics=self.metrics,
            calibration_split=self.calibration_split,
            gate_split=self.gate_split,
            gold_sha256=self.gold_sha256,
            hard_negatives_sha256=self.hard_negatives_sha256,
            notes=self.notes,
            approved_by=approved_by,
            approved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            parent_version=self.parent_version,
            profile=self.profile,
            hard_negative_count=self.hard_negative_count,
            dev_metrics=self.dev_metrics,
            tags=list(self.tags),
        )


def sha256_file(path: Path) -> str:
    """Content digest of one file, read in chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def drifted_files(manifest: Manifest, root: Path) -> list[str]:
    """Files whose bytes on disk no longer match the manifest."""

    drift: list[str] = []
    for entry in manifest.files:
        path = root / entry.path
        if not path.exists():
            drift.append(f"MISSING {entry.path}")
            continue
        if sha256_file(path) != entry.sha256:
            drift.append(f"CHANGED {entry.path}")
    return drift
