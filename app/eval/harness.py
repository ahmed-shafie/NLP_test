"""Deterministic NLU evaluation harness.

Runs the real NLU pipeline over a labelled bilingual gold set and scores it:

* **intent accuracy** + a gold-vs-predicted confusion matrix, and
* **per-slot precision / recall / F1** (amount, currency, recipient, biller_code,
  reference_number).

The scorer is deterministic and needs no LLM: intents come from the semantic
classifier (or the keyword classifier when the embedding model is unavailable)
and slots come from the regex/gazetteer extractors. The same input always yields
the same score, so it can gate CI. See :mod:`scripts.eval_nlu` for the CLI and
``tests/test_eval_nlu.py`` for the pytest gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.config import DEFAULT_CURRENCY, settings
from app.nlu.entities import (
    extract_amount,
    extract_bill_entities,
    extract_currency,
    extract_recipient,
)
from app.nlu.intents import classify_intent
from app.nlu.lang import detect_language
from app.nlu.normalize import normalize
from app.nlu.semantic_intents import get_semantic_classifier
from app.schemas import Intent, Language

GOLD_PATH = Path(__file__).resolve().parent / "nlu_gold.jsonl"

# Slots scored per intent. Only slots present in a gold row are scored, so an
# assumed default (e.g. SAR currency) is never penalised when the row omits it.
_TRANSFER_SLOTS = ("amount", "currency", "recipient")
_BILL_SLOTS = ("biller_code", "reference_number", "amount", "currency")

# CI gate thresholds. Slot thresholds are deterministic and always enforced;
# intent accuracy is enforced only when the embedding model is available (the
# keyword fallback cannot distinguish pay_bill / small_talk). The intent floor
# sits a little below the current baseline (~0.96) so a real regression trips it
# without the gate being knife-edge.
MIN_INTENT_ACCURACY = 0.93
# Over-blocking guard: no non-abusive gold utterance may be flagged INAPPROPRIATE.
# Flagging a legitimate request as abuse is a high-harm failure, so the gate is
# zero-tolerance and catches over-blocking regressions for *any* word in CI.
MAX_INAPPROPRIATE_FALSE_POSITIVES = 0
MIN_SLOT_F1 = {
    "amount": 0.97,
    "currency": 0.97,
    "recipient": 0.90,
    "biller_code": 0.85,
    "reference_number": 0.95,
}


@dataclass
class GoldRow:
    """One labelled gold utterance."""

    text: str
    intent: Intent
    slots: dict[str, str]
    language: Language | None = None


@dataclass
class SlotScore:
    """Precision/recall/F1 accumulator for a single slot."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        # Count-based form so an all-wrong slot (tp=0, fp>0, fn>0) scores 0.0,
        # not 1.0; only a slot with no gold and no prediction is vacuously 1.0.
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp) / denom if denom else 1.0


@dataclass
class Report:
    """Aggregate evaluation result."""

    total: int = 0
    intent_correct: int = 0
    semantic: bool = False
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    slots: dict[str, SlotScore] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)
    # Non-abusive gold rows that were wrongly predicted INAPPROPRIATE (over-block).
    over_blocks: list[str] = field(default_factory=list)

    @property
    def intent_accuracy(self) -> float:
        return self.intent_correct / self.total if self.total else 1.0

    def slot_f1(self, slot: str) -> float:
        return self.slots[slot].f1 if slot in self.slots else 1.0


def load_gold(path: Path = GOLD_PATH) -> list[GoldRow]:
    """Parse the JSONL gold set into typed rows."""

    rows: list[GoldRow] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lang = obj.get("language")
            rows.append(
                GoldRow(
                    text=obj["text"],
                    intent=Intent(obj["intent"]),
                    slots={k: str(v) for k, v in (obj.get("slots") or {}).items()},
                    language=Language(lang) if lang else None,
                )
            )
    return rows


def predict_intent(text: str, language: Language) -> tuple[Intent, bool]:
    """Predict the intent deterministically; second value is ``semantic_used``."""

    from app.conversation.moderation import detect as detect_abuse

    # Deterministic moderation guard mirrors the live pipeline (orchestration).
    if detect_abuse(text).flagged:
        return Intent.INAPPROPRIATE, False
    classifier = get_semantic_classifier()
    if classifier is not None:
        intent, _ = classifier.classify(text)
        return intent, True
    intent, confidence = classify_intent(text, language)
    if confidence < settings.intent_threshold:
        intent = Intent.FALLBACK
    return intent, False


def _predict_slots(text: str, language: Language, intent: Intent) -> dict[str, str]:
    """Extract the slots relevant to ``intent`` (gold intent drives extraction)."""

    if intent is Intent.TRANSFER_MONEY:
        amount = extract_amount(text)
        currency = extract_currency(text)
        if amount is not None and currency is None:
            currency = DEFAULT_CURRENCY
        recipient = extract_recipient(text, language)
        return _clean({"amount": amount, "currency": currency, "recipient": recipient})
    if intent is Intent.PAY_BILL:
        bill = extract_bill_entities(text, language, allow_semantic=True)
        currency = bill.currency
        if bill.amount is not None and currency is None:
            currency = DEFAULT_CURRENCY
        return _clean(
            {
                "biller_code": bill.biller_code,
                "reference_number": bill.reference_number,
                "amount": bill.amount,
                "currency": currency,
            }
        )
    return {}


def _clean(values: dict[str, object]) -> dict[str, str]:
    return {k: str(v) for k, v in values.items() if v is not None and str(v) != ""}


def _values_match(slot: str, gold: str, pred: str) -> bool:
    if slot == "amount":
        try:
            return Decimal(gold) == Decimal(pred)
        except (InvalidOperation, ValueError):
            return gold == pred
    if slot == "currency":
        return gold.strip().upper() == pred.strip().upper()
    if slot == "recipient":
        return normalize(gold) == normalize(pred)
    return gold.strip() == pred.strip()


def evaluate(rows: list[GoldRow] | None = None) -> Report:
    """Score the pipeline over the gold rows and return an aggregate report."""

    rows = rows if rows is not None else load_gold()
    report = Report(total=len(rows))
    for intent in Intent:
        report.confusion[intent.value] = {}

    for row in rows:
        language = row.language or detect_language(row.text)
        pred_intent, semantic = predict_intent(row.text, language)
        report.semantic = report.semantic or semantic

        bucket = report.confusion[row.intent.value]
        bucket[pred_intent.value] = bucket.get(pred_intent.value, 0) + 1
        if pred_intent is row.intent:
            report.intent_correct += 1
        else:
            report.misses.append(
                f"INTENT {row.text!r}: gold={row.intent.value} pred={pred_intent.value}"
            )
        if (
            pred_intent is Intent.INAPPROPRIATE
            and row.intent is not Intent.INAPPROPRIATE
        ):
            report.over_blocks.append(
                f"OVER-BLOCK {row.text!r}: gold={row.intent.value} flagged as abuse"
            )

        pred_slots = _predict_slots(row.text, language, row.intent)
        scored = _TRANSFER_SLOTS if row.intent is Intent.TRANSFER_MONEY else _BILL_SLOTS
        for slot in scored:
            if slot not in row.slots:
                continue
            score = report.slots.setdefault(slot, SlotScore())
            gold_val = row.slots[slot]
            pred_val = pred_slots.get(slot)
            if pred_val is None:
                score.fn += 1
                report.misses.append(
                    f"SLOT  {row.text!r}: {slot} gold={gold_val!r} pred=<missing>"
                )
            elif _values_match(slot, gold_val, pred_val):
                score.tp += 1
            else:
                score.fp += 1
                score.fn += 1
                report.misses.append(
                    f"SLOT  {row.text!r}: {slot} gold={gold_val!r} pred={pred_val!r}"
                )
    return report


def format_report(report: Report) -> str:
    """Render a human-readable summary (used by the CLI and on gate failure)."""

    lines: list[str] = []
    mode = "semantic" if report.semantic else "keyword"
    lines.append(
        f"intent_accuracy: {report.intent_accuracy:.3f} "
        f"({report.intent_correct}/{report.total})  [classifier: {mode}]"
    )
    if report.slots:
        slot_bits = " | ".join(
            f"{slot} {report.slots[slot].f1:.3f}" for slot in sorted(report.slots)
        )
        lines.append(f"slot F1: {slot_bits}")
        for slot in sorted(report.slots):
            s = report.slots[slot]
            lines.append(
                f"  {slot:<17} P={s.precision:.3f} R={s.recall:.3f} "
                f"F1={s.f1:.3f} (support={s.support})"
            )
    lines.append("confusion (gold -> predicted):")
    for gold, preds in report.confusion.items():
        if not preds:
            continue
        rendered = ", ".join(f"{p}:{c}" for p, c in sorted(preds.items()))
        lines.append(f"  {gold:<16} -> {rendered}")
    lines.append(
        f"inappropriate over-blocks: {len(report.over_blocks)} "
        "(non-abusive rows flagged as abuse)"
    )
    for ob in report.over_blocks:
        lines.append(f"  {ob}")
    if report.misses:
        lines.append(f"misses ({len(report.misses)}):")
        lines.extend(f"  {m}" for m in report.misses)
    return "\n".join(lines)


def check_thresholds(report: Report) -> list[str]:
    """Return a list of threshold violations (empty when the gate passes).

    Intent accuracy is only enforced when the semantic classifier ran, since the
    keyword fallback cannot separate pay_bill / small_talk.
    """

    failures: list[str] = []
    if report.semantic and report.intent_accuracy < MIN_INTENT_ACCURACY:
        failures.append(
            f"intent_accuracy {report.intent_accuracy:.3f} < {MIN_INTENT_ACCURACY}"
        )
    # Over-blocking is enforced only under the semantic classifier (the keyword
    # fallback cannot label INAPPROPRIATE), and is zero-tolerance.
    if report.semantic and len(report.over_blocks) > MAX_INAPPROPRIATE_FALSE_POSITIVES:
        failures.append(
            f"inappropriate_false_positives {len(report.over_blocks)} "
            f"> {MAX_INAPPROPRIATE_FALSE_POSITIVES}"
        )
    for slot, minimum in MIN_SLOT_F1.items():
        if slot in report.slots and report.slots[slot].f1 < minimum:
            failures.append(f"slot_f1[{slot}] {report.slots[slot].f1:.3f} < {minimum}")
    return failures
