"""Where releases live, and the only two ways the live one changes.

``promote`` and ``rollback`` are the whole API. Both append to a log; neither
edits history. Promotion refuses a candidate unless:

* the gate was clean on the held-out split,
* the manifest is not self-graded (``calibration_split != gate_split``),
* the bytes on disk still match the manifest, and
* a **named human** approved it.

Rollback is deliberately dumber than promotion: it names an earlier version and
takes effect immediately, with a reason. Rolling back is what you do at 2am, so
it must not be able to fail a gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.release.manifest import (
    MANIFEST_NAME,
    Manifest,
    drifted_files,
)

CURRENT_NAME = "current.json"
LOG_NAME = "history.jsonl"


class ReleaseRefused(Exception):
    """Raised when a release may not be promoted."""


class UnknownRelease(Exception):
    """Raised when a version is not in the registry."""


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One line of the release log."""

    at: str
    action: str
    version: str
    actor: str
    reason: str


@dataclass(frozen=True, slots=True)
class Current:
    """Which version is live, and how it got there."""

    version: str
    since: str
    action: str
    actor: str


def registry_root() -> Path:
    """Directory holding every stored release."""

    return Path(settings.release_dir)


def _version_dir(version: str) -> Path:
    if "/" in version or version in {"", ".", ".."}:
        raise UnknownRelease(f"{version!r} is not a valid version")
    return registry_root() / version


def _append_log(entry: LogEntry) -> None:
    root = registry_root()
    root.mkdir(parents=True, exist_ok=True)
    with (root / LOG_NAME).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "at": entry.at,
                    "action": entry.action,
                    "version": entry.version,
                    "actor": entry.actor,
                    "reason": entry.reason,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def store(manifest: Manifest) -> Path:
    """Write a candidate manifest into the registry without promoting it."""

    directory = _version_dir(manifest.version)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def load(version: str) -> Manifest:
    """Read a stored manifest."""

    path = _version_dir(version) / MANIFEST_NAME
    if not path.exists():
        raise UnknownRelease(f"no release {version!r} in {registry_root()}")
    return Manifest.from_json(path.read_text(encoding="utf-8"))


def versions() -> list[str]:
    """Every stored version, oldest first by creation time in the manifest."""

    root = registry_root()
    if not root.exists():
        return []
    stored = [
        child.name for child in root.iterdir() if (child / MANIFEST_NAME).exists()
    ]
    return sorted(stored, key=lambda version: load(version).created_at)


def current() -> Current | None:
    """The live release, or ``None`` when nothing was ever promoted."""

    path = registry_root() / CURRENT_NAME
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return Current(
        version=obj["version"],
        since=obj["since"],
        action=obj["action"],
        actor=obj["actor"],
    )


def history() -> list[LogEntry]:
    """The append-only promote/rollback log, oldest first."""

    path = registry_root() / LOG_NAME
    if not path.exists():
        return []
    entries: list[LogEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        entries.append(
            LogEntry(
                at=obj["at"],
                action=obj["action"],
                version=obj["version"],
                actor=obj["actor"],
                reason=obj["reason"],
            )
        )
    return entries


def _set_current(version: str, action: str, actor: str, reason: str) -> Current:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    root = registry_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / CURRENT_NAME).write_text(
        json.dumps(
            {"version": version, "since": now, "action": action, "actor": actor},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _append_log(
        LogEntry(at=now, action=action, version=version, actor=actor, reason=reason)
    )
    return Current(version=version, since=now, action=action, actor=actor)


def promote(
    manifest: Manifest,
    *,
    approved_by: str,
    violations: list[str],
    reason: str = "",
    root: Path | None = None,
) -> Current:
    """Store, sign and make ``manifest`` the live release, or refuse to."""

    if not approved_by.strip():
        raise ReleaseRefused("a release needs a named approver")
    if violations:
        raise ReleaseRefused(
            "the gate did not pass on the held-out split: " + "; ".join(violations)
        )
    if manifest.metrics.money_flow_breaches:
        raise ReleaseRefused(
            f"{manifest.metrics.money_flow_breaches} hard negative(s) opened a "
            "money flow; that budget is zero"
        )
    from app.release.build import REPO_ROOT  # local: build imports the harness

    drift = drifted_files(manifest, root or REPO_ROOT)
    if drift:
        raise ReleaseRefused(
            "the measured files changed since the build: " + str(drift)
        )

    signed = manifest.approve(approved_by)
    store(signed)
    return _set_current(signed.version, "promote", approved_by, reason or signed.notes)


def rollback(version: str, *, actor: str, reason: str) -> Current:
    """Make an already-stored earlier version live again."""

    if not reason.strip():
        raise ReleaseRefused("a rollback needs a reason; it is the incident record")
    load(version)  # raises UnknownRelease when it was never stored
    return _set_current(version, "rollback", actor, reason)
