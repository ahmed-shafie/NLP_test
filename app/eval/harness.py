"""Deterministic NLU evaluation harness.

Runs the real routing over a labelled bilingual gold set and scores it:

* **intent accuracy**, a gold-vs-predicted confusion matrix and **per-intent
  precision / recall / F1** (one global number hides an intent that is broken),
* **per-slot precision / recall / F1** (amount, currency, recipient, biller_code,
  reference_number), and
* **slices**: per language and per tag (``colloquial``, ``arabic_digits``,
  ``typo``, ``out_of_scope`` …), since the failures live in the long tail.

The scored intent is the one the customer actually gets
(:func:`app.conversation.engine.route_fresh_turn`), not the raw classifier
verdict: the engine's deterministic cues overrule the classifier, so scoring the
classifier alone measures a decision the engine never makes.

The scorer is deterministic and needs no LLM, so the same input always yields
the same score and it can gate CI. See :mod:`scripts.eval_nlu` for the CLI and
``tests/test_eval_nlu.py`` for the pytest gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.config import DEFAULT_CURRENCY, settings
from app.conversation.engine import route_fresh_turn
from app.nlu.entities import (
    extract_amount,
    extract_bill_entities,
    extract_currency,
    extract_recipient,
)
from app.nlu.examples import INTENT_EXAMPLES
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
# Per-intent recall floors. A global average lets one intent collapse unnoticed
# (before this gate existed, balance_inquiry was routed by cues no metric
# covered), so every supported intent carries its own floor. ``fallback`` is the
# abstain path: its floor is deliberately lower because containing every
# unsupported request is the hardest slice, and its failures are the least
# harmful (we guess a flow instead of asking).
MIN_INTENT_RECALL = {
    Intent.TRANSFER_MONEY: 0.90,
    Intent.PAY_BILL: 0.90,
    Intent.BALANCE_INQUIRY: 0.90,
    Intent.LIST_BENEFICIARIES: 0.90,
    Intent.ADD_BENEFICIARY: 0.85,
    Intent.SMALL_TALK: 0.85,
    Intent.INAPPROPRIATE: 0.90,
    Intent.FALLBACK: 0.50,
}
# Mis-routing a supported banking request into a *money-moving* flow is the
# highest-harm intent error (we start asking for an amount for a request that
# was never a transfer), so it carries its own zero-ish budget.
MAX_WRONG_FLOW_STARTS = 0
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
    tags: tuple[str, ...] = ()


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
class Slice:
    """Intent accuracy over a subset of rows (a language or a tag)."""

    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 1.0


@dataclass
class Report:
    """Aggregate evaluation result."""

    total: int = 0
    intent_correct: int = 0
    semantic: bool = False
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    intents: dict[Intent, SlotScore] = field(default_factory=dict)
    slots: dict[str, SlotScore] = field(default_factory=dict)
    languages: dict[str, Slice] = field(default_factory=dict)
    tags: dict[str, Slice] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)
    # Non-abusive gold rows that were wrongly predicted INAPPROPRIATE (over-block).
    over_blocks: list[str] = field(default_factory=list)
    # Non-transfer/bill requests wrongly routed into a money-moving flow.
    wrong_flow_starts: list[str] = field(default_factory=list)

    @property
    def intent_accuracy(self) -> float:
        return self.intent_correct / self.total if self.total else 1.0

    def slot_f1(self, slot: str) -> float:
        return self.slots[slot].f1 if slot in self.slots else 1.0

    def intent_recall(self, intent: Intent) -> float:
        return self.intents[intent].recall if intent in self.intents else 1.0


def leaked_rows(rows: list[GoldRow]) -> list[str]:
    """Gold utterances that are also classifier training examples.

    Such a row scores the index's memory of it rather than the pipeline's ability
    to generalise, which is how a gold set quietly starts flattering itself.
    """

    trained = {normalize(text) for text, _ in INTENT_EXAMPLES}
    return [row.text for row in rows if normalize(row.text) in trained]


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
                    tags=tuple(obj.get("tags") or ()),
                )
            )
    return rows


def predict_intent(text: str, language: Language) -> tuple[Intent, bool]:
    """Predict the routed intent; second value is ``semantic_used``.

    The classifier verdict is fed through the engine's routing so the score
    reflects what the customer is actually shown.
    """

    classifier = get_semantic_classifier()
    if classifier is not None:
        classified, confidence = classifier.classify(text)
        semantic = True
    else:
        classified, confidence = classify_intent(text, language)
        if confidence < settings.intent_threshold:
            classified = Intent.FALLBACK
        semantic = False
    return route_fresh_turn(text, language, classified, confidence), semantic


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
        gold_score = report.intents.setdefault(row.intent, SlotScore())
        pred_score = report.intents.setdefault(pred_intent, SlotScore())
        correct = pred_intent is row.intent
        slices = [report.languages.setdefault(language.value, Slice())]
        slices += [report.tags.setdefault(tag, Slice()) for tag in row.tags]
        for sl in slices:
            sl.total += 1
            sl.correct += int(correct)
        if correct:
            report.intent_correct += 1
            gold_score.tp += 1
        else:
            gold_score.fn += 1
            pred_score.fp += 1
            report.misses.append(
                f"INTENT {row.text!r}: gold={row.intent.value} pred={pred_intent.value}"
            )
            if (
                pred_intent in (Intent.TRANSFER_MONEY, Intent.PAY_BILL)
                and row.intent not in (Intent.TRANSFER_MONEY, Intent.PAY_BILL)
                and "out_of_scope" not in row.tags
            ):
                report.wrong_flow_starts.append(
                    f"WRONG-FLOW {row.text!r}: gold={row.intent.value} "
                    f"started {pred_intent.value}"
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
    if report.intents:
        lines.append("per-intent (recall = share of gold rows routed correctly):")
        for intent in Intent:
            score = report.intents.get(intent)
            if score is None or score.support == 0:
                continue
            floor = MIN_INTENT_RECALL.get(intent)
            mark = "" if floor is None or score.recall >= floor else "  <-- BELOW FLOOR"
            lines.append(
                f"  {intent.value:<19} P={score.precision:.3f} R={score.recall:.3f} "
                f"F1={score.f1:.3f} (support={score.support}){mark}"
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
    if report.languages:
        rendered = " | ".join(
            f"{lang} {sl.accuracy:.3f} ({sl.correct}/{sl.total})"
            for lang, sl in sorted(report.languages.items())
        )
        lines.append(f"by language: {rendered}")
    if report.tags:
        lines.append("by tag (weakest first):")
        for tag, sl in sorted(report.tags.items(), key=lambda kv: kv[1].accuracy):
            lines.append(f"  {tag:<21} {sl.accuracy:.3f} ({sl.correct}/{sl.total})")
    lines.append(
        f"inappropriate over-blocks: {len(report.over_blocks)} "
        "(non-abusive rows flagged as abuse)"
    )
    for ob in report.over_blocks:
        lines.append(f"  {ob}")
    lines.append(
        f"wrong flow starts: {len(report.wrong_flow_starts)} "
        "(supported non-payment request routed into transfer/bill)"
    )
    for wf in report.wrong_flow_starts:
        lines.append(f"  {wf}")
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
    if report.semantic:
        for intent, minimum in MIN_INTENT_RECALL.items():
            score = report.intents.get(intent)
            if score is None or score.support == 0:
                continue
            if score.recall < minimum:
                failures.append(
                    f"intent_recall[{intent.value}] {score.recall:.3f} < {minimum} "
                    f"(support={score.support})"
                )
        if len(report.wrong_flow_starts) > MAX_WRONG_FLOW_STARTS:
            failures.append(
                f"wrong_flow_starts {len(report.wrong_flow_starts)} "
                f"> {MAX_WRONG_FLOW_STARTS}"
            )
    for slot, minimum in MIN_SLOT_F1.items():
        if slot in report.slots and report.slots[slot].f1 < minimum:
            failures.append(f"slot_f1[{slot}] {report.slots[slot].f1:.3f} < {minimum}")
    return failures
