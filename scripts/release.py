#!/usr/bin/env python
"""CLI for the measured release loop.

Usage::

    python scripts/release.py build   --version 2026.06.1 [--notes "..."]
    python scripts/release.py promote --version 2026.06.1 --approved-by "Ahmed N"
    python scripts/release.py rollback --version 2026.05.3 --reason "wrong biller"
    python scripts/release.py status

``build`` measures the tree on the **held-out** gold split plus the hard
negatives and writes a manifest; it exits non-zero when the gate fails, so CI
can run it. ``promote`` refuses anything the gate did not pass, anything with a
hard-negative breach, anything whose files changed since the build, and anything
without a named approver. ``rollback`` needs a reason and nothing else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.release import registry  # noqa: E402
from app.release.build import build_candidate  # noqa: E402
from app.release.manifest import drifted_files  # noqa: E402


def _build(args: argparse.Namespace) -> int:
    live = registry.current()
    manifest, violations = build_candidate(
        args.version,
        notes=args.notes,
        parent_version=live.version if live else None,
        profile=args.profile,
    )
    path = registry.store(manifest)
    metrics = manifest.metrics
    dev = manifest.dev_metrics
    print(f"release {manifest.version} -> {path}")
    print(
        f"  gate split      : {manifest.gate_split} "
        f"({metrics.rows} rows, held out from calibration)"
    )
    print(f"  intent accuracy : {metrics.intent_accuracy:.3f}")
    if dev is not None:
        print(
            f"  dev accuracy    : {dev.intent_accuracy:.3f} "
            f"({dev.rows} rows, calibration split)"
        )
    print(f"  wrong flow start: {metrics.wrong_flow_starts}")
    print(f"  over-blocks     : {metrics.over_blocks}")
    print(
        f"  hard negatives  : {manifest.hard_negative_count} "
        f"({metrics.money_flow_breaches} opened a money flow)"
    )
    print(f"  files pinned    : {len(manifest.files)}")
    if violations:
        for violation in violations:
            print(f"  GATE: {violation}")
        print("\nGATE FAILED — not promotable")
        return 1
    print("\nGATE PASSED — promotable")
    return 0


def _promote(args: argparse.Namespace) -> int:
    manifest = registry.load(args.version)
    _, violations = build_candidate(
        args.version, notes=manifest.notes, profile=manifest.profile
    )
    try:
        live = registry.promote(
            manifest,
            approved_by=args.approved_by,
            violations=violations,
            reason=args.reason,
        )
    except registry.ReleaseRefused as refusal:
        print(f"REFUSED: {refusal}")
        return 1
    print(f"live: {live.version} (promoted by {live.actor} at {live.since})")
    return 0


def _rollback(args: argparse.Namespace) -> int:
    try:
        live = registry.rollback(args.version, actor=args.actor, reason=args.reason)
    except (registry.ReleaseRefused, registry.UnknownRelease) as refusal:
        print(f"REFUSED: {refusal}")
        return 1
    print(f"live: {live.version} (rolled back by {live.actor} at {live.since})")
    return 0


def _status(_: argparse.Namespace) -> int:
    live = registry.current()
    print(f"stored versions: {', '.join(registry.versions()) or '(none)'}")
    if live is None:
        print("live: (nothing promoted)")
        return 0
    manifest = registry.load(live.version)
    print(f"live: {live.version} since {live.since} by {live.actor} ({live.action})")
    print(f"  approved_by     : {manifest.approved_by}")
    print(
        f"  measured on     : {manifest.gate_split} "
        f"(calibrated on {manifest.calibration_split})"
    )
    print(f"  intent accuracy : {manifest.metrics.intent_accuracy:.3f}")
    from app.release.build import REPO_ROOT

    drift = drifted_files(manifest, REPO_ROOT)
    print(f"  drift           : {', '.join(drift) if drift else 'none'}")
    for entry in registry.history()[-5:]:
        print(f"  {entry.at} {entry.action} {entry.version} by {entry.actor}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="measure the tree and write a manifest")
    build.add_argument("--version", required=True)
    build.add_argument("--notes", default="")
    build.add_argument("--profile", default="default")
    build.set_defaults(func=_build)

    promote = sub.add_parser("promote", help="make a stored candidate live")
    promote.add_argument("--version", required=True)
    promote.add_argument("--approved-by", required=True)
    promote.add_argument("--reason", default="")
    promote.set_defaults(func=_promote)

    rollback = sub.add_parser("rollback", help="make an earlier version live again")
    rollback.add_argument("--version", required=True)
    rollback.add_argument("--actor", default="operations")
    rollback.add_argument("--reason", required=True)
    rollback.set_defaults(func=_rollback)

    status = sub.add_parser("status", help="what is live, and does it still match")
    status.set_defaults(func=_status)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
