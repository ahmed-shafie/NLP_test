"""CI gate: the NLU pipeline must meet accuracy/slot-F1 thresholds on the gold set.

Slot F1 is deterministic (regex/gazetteer) and always enforced. Intent accuracy
is enforced only when the embedding model is available, since the keyword
fallback cannot separate pay_bill / small_talk (see ``app.eval.harness``).
"""

from __future__ import annotations

from app.eval.harness import (
    MIN_INTENT_RECALL,
    GoldRow,
    Slice,
    SlotScore,
    check_thresholds,
    evaluate,
    format_report,
    leaked_rows,
    load_gold,
)
from app.schemas import Intent, Language


def test_slot_f1_zero_when_every_prediction_is_wrong() -> None:
    # All predictions wrong (tp=0, fp>0, fn>0) must score 0.0, not a vacuous 1.0,
    # otherwise the gate would silently pass a total slot regression.
    assert SlotScore(tp=0, fp=5, fn=5).f1 == 0.0
    # No gold and no prediction is vacuously perfect.
    assert SlotScore().f1 == 1.0
    assert SlotScore(tp=4, fp=0, fn=0).f1 == 1.0


def test_gold_set_is_bilingual_and_covers_every_supported_intent() -> None:
    rows = load_gold()
    assert len(rows) >= 250
    # Every intent the assistant can route to needs gold rows, otherwise its
    # recall is unmeasured and can collapse without failing the gate.
    assert {row.intent for row in rows} == set(MIN_INTENT_RECALL)
    languages = {row.language for row in rows if row.language is not None}
    assert len(languages) == 2  # English and Arabic both represented
    for intent in MIN_INTENT_RECALL:
        support = [row for row in rows if row.intent is intent]
        assert len(support) >= 10, f"{intent.value} has too few gold rows"


def test_gold_rows_are_not_classifier_training_examples() -> None:
    # A gold row that is also a training example scores the index's memory of it.
    assert leaked_rows(load_gold()) == []


def test_gold_rows_are_unique() -> None:
    texts = [row.text for row in load_gold()]
    assert len(texts) == len(set(texts))


def test_report_slices_rows_by_language_and_tag() -> None:
    rows = [
        GoldRow(
            text="send 500 to Ahmed",
            intent=Intent.TRANSFER_MONEY,
            slots={},
            language=Language.EN,
            tags=("plain",),
        ),
        GoldRow(
            text="وش الجو اليوم",
            intent=Intent.FALLBACK,
            slots={},
            language=Language.AR,
            tags=("off_topic",),
        ),
    ]
    report = evaluate(rows)
    assert set(report.languages) == {"en", "ar"}
    assert set(report.tags) == {"plain", "off_topic"}
    assert report.languages["en"].total == 1
    assert report.tags["off_topic"].total == 1


def test_slice_accuracy_is_vacuously_perfect_when_empty() -> None:
    assert Slice().accuracy == 1.0
    assert Slice(total=4, correct=1).accuracy == 0.25


def test_threshold_check_flags_a_collapsed_intent() -> None:
    # Ten balance questions all answered as "transfer or bill?" keep overall
    # accuracy high but must still fail the gate on that intent's recall.
    rows = [
        GoldRow(
            text="قل لي نكتة",
            intent=Intent.BALANCE_INQUIRY,
            slots={},
            language=Language.AR,
        )
    ] * 10
    failures = check_thresholds(evaluate(rows))
    assert any("balance_inquiry" in f for f in failures)


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
