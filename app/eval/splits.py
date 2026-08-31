"""A fixed dev/test partition of the gold set, and the hard-negative set.

A threshold tuned on the same rows that later judge it cannot fail: the numbers
are fitted to the answer sheet. So the gold set is split once, deterministically,
and the two halves have different jobs:

* **dev** — the only rows a threshold, a cue or a calibration may look at.
* **test** — held out; it scores a release and nothing else reads it.

The partition is derived from the row itself (a digest of its normalised text,
stratified by intent and language) rather than stored in the file, so adding gold
rows cannot quietly move an existing row from held-out to visible, and no split
column has to be maintained by hand.

The hard negatives are separate on purpose. They are not scored for accuracy —
they carry one property, taken from real transcripts: *this sentence must never
open a money flow*. A regression there is a customer being asked for an amount
they never mentioned, so the release gate allows none.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.eval.harness import GOLD_PATH, GoldRow, load_gold, predict_intent
from app.nlu.lang import detect_language
from app.nlu.normalize import normalize
from app.schemas import Intent, Language

HARD_NEGATIVES_PATH = Path(__file__).resolve().parent / "hard_negatives.jsonl"

# Share of each (intent, language) stratum that goes to dev. The held-out half is
# large enough that a single flipped row cannot swing the gate.
DEV_SHARE = 0.6

# Fixed: changing it repartitions the gold set and invalidates every recorded
# release metric, so it is a deliberate, reviewable edit.
_SPLIT_SALT = "nlu-gold-split-v1"


class Split(str, Enum):
    """Which half of the gold set a row belongs to."""

    DEV = "dev"
    TEST = "test"


class MoneyFlowOpened(Exception):
    """Raised when a hard negative routed into a money flow."""


@dataclass(frozen=True, slots=True)
class HardNegative:
    """An utterance that must not be read as a request to move money."""

    text: str
    language: Language | None
    reason: str


def _rank(text: str) -> str:
    return hashlib.sha256(f"{_SPLIT_SALT}:{normalize(text)}".encode()).hexdigest()


def assign_splits(rows: list[GoldRow]) -> dict[str, Split]:
    """Map each row's text to its split, stratified by intent and language."""

    strata: dict[tuple[Intent, str], list[GoldRow]] = {}
    for row in rows:
        language = row.language or detect_language(row.text)
        strata.setdefault((row.intent, language.value), []).append(row)

    assignment: dict[str, Split] = {}
    for members in strata.values():
        ordered = sorted(members, key=lambda row: _rank(row.text))
        cut = round(len(ordered) * DEV_SHARE)
        for index, row in enumerate(ordered):
            assignment[row.text] = Split.DEV if index < cut else Split.TEST
    return assignment


def load_split(split: Split, path: Path = GOLD_PATH) -> list[GoldRow]:
    """Load only the rows belonging to ``split``."""

    rows = load_gold(path)
    assignment = assign_splits(rows)
    return [row for row in rows if assignment[row.text] is split]


def load_hard_negatives(path: Path = HARD_NEGATIVES_PATH) -> list[HardNegative]:
    """Parse the hard-negative set."""

    negatives: list[HardNegative] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            language = obj.get("language")
            negatives.append(
                HardNegative(
                    text=obj["text"],
                    language=Language(language) if language else None,
                    reason=obj["reason"],
                )
            )
    return negatives


def money_flow_breaches(negatives: list[HardNegative] | None = None) -> list[str]:
    """Hard negatives that were routed into a transfer or a bill payment."""

    negatives = negatives if negatives is not None else load_hard_negatives()
    breaches: list[str] = []
    for negative in negatives:
        language = negative.language or detect_language(negative.text)
        routed, _ = predict_intent(negative.text, language)
        if routed in (Intent.TRANSFER_MONEY, Intent.PAY_BILL):
            breaches.append(
                f"MONEY-FLOW {negative.text!r} -> {routed.value} ({negative.reason})"
            )
    return breaches
