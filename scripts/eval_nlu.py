#!/usr/bin/env python
"""CLI for the deterministic NLU evaluation harness.

Runs the NLU pipeline over the bilingual gold set, prints intent accuracy, a
confusion matrix and per-slot F1, and exits non-zero when a CI threshold is
breached (see :mod:`app.eval.harness`).

Usage::

    python scripts/eval_nlu.py            # score the packaged gold set
    python scripts/eval_nlu.py --gold path/to/other.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``app`` importable when run as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.eval.harness import (  # noqa: E402
    GOLD_PATH,
    check_thresholds,
    evaluate,
    format_report,
    load_gold,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=GOLD_PATH,
        help="Path to the gold JSONL file (defaults to the packaged gold set).",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Print the report but always exit 0 (skip threshold enforcement).",
    )
    args = parser.parse_args()

    report = evaluate(load_gold(args.gold))
    print(format_report(report))

    failures = check_thresholds(report)
    if failures and not args.no_gate:
        print("\nGATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nGATE PASSED" if not args.no_gate else "\n(gate skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
