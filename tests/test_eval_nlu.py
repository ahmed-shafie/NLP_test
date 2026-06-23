"""CI gate: the NLU pipeline must meet accuracy/slot-F1 thresholds on the gold set.

Slot F1 is deterministic (regex/gazetteer) and always enforced. Intent accuracy
is enforced only when the embedding model is available, since the keyword
fallback cannot separate pay_bill / small_talk (see ``app.eval.harness``).
"""

from __future__ import annotations

from app.eval.harness import (
    check_thresholds,
    evaluate,
    format_report,
    load_gold,
)
from app.schemas import Intent


def test_gold_set_is_bilingual_and_covers_all_intents() -> None:
    rows = load_gold()
    assert len(rows) >= 50
    intents = {row.intent for row in rows}
    assert intents == {Intent.TRANSFER_MONEY, Intent.PAY_BILL, Intent.SMALL_TALK}
    languages = {row.language for row in rows if row.language is not None}
    assert len(languages) == 2  # English and Arabic both represented


def test_nlu_eval_meets_ci_thresholds() -> None:
    report = evaluate(load_gold())
    failures = check_thresholds(report)
    assert not failures, "NLU gate failed:\n" + format_report(report)


def test_eval_is_reproducible() -> None:
    gold = load_gold()
    first = evaluate(gold)
    second = evaluate(gold)
    assert first.intent_accuracy == second.intent_accuracy
    assert {s: first.slots[s].f1 for s in first.slots} == {
        s: second.slots[s].f1 for s in second.slots
    }
