"""Measure what the v0.8 vector-DB CSV does to the semantic intent classifier.

Two questions, and the second one is the reason this script exists:

1. does the CSV improve **out-of-scope rejection**? Scored on the ArBanking77
   dialect test splits, which are held out of every index built here — every row
   there must come back ``FALLBACK``;
2. does it **damage the intents we actually execute**? 31k customer-service rows
   against ~260 transfer rows is a 100:1 imbalance in a nearest-neighbour vote,
   so the plausible outcome is better rejection bought with lost recall. Scored
   on the existing gold set.

Run: ``python -m research.vector_db_v08.eval_index`` from the repo root.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.nlu.semantic_intents import SemanticIntentClassifier
from app.schemas import Intent
from research.vector_db_v08.load_csv import Row, cap_topics, partition, read_rows

GOLD = Path("app/eval/nlu_gold.jsonl")


def gold_rows() -> list[tuple[str, Intent]]:
    out: list[tuple[str, Intent]] = []
    with GOLD.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out.append((row["text"], Intent(row["intent"])))
    return out


def score(
    classifier: SemanticIntentClassifier, cases: list[tuple[str, Intent]]
) -> tuple[float, Counter[str]]:
    hits = 0
    wrong: Counter[str] = Counter()
    for text, expected in cases:
        predicted, _ = classifier.classify(text)
        if predicted is expected:
            hits += 1
        else:
            wrong[f"{expected.value} -> {predicted.value}"] += 1
    return hits / len(cases), wrong


def money_leaks(
    classifier: SemanticIntentClassifier, cases: list[tuple[str, Intent]]
) -> int:
    money = {Intent.TRANSFER_MONEY, Intent.PAY_BILL, Intent.ADD_BENEFICIARY}
    return sum(1 for text, _ in cases if classifier.classify(text)[0] in money)


def report(name: str, extra: list[Row] | None, oos: list[tuple[str, Intent]]) -> None:
    examples = [(row.text, row.intent) for row in extra] if extra else None
    classifier = SemanticIntentClassifier(examples)
    gold_acc, gold_wrong = score(classifier, gold_rows())
    oos_acc, _ = score(classifier, oos)
    leaks = money_leaks(classifier, oos)

    print(f"\n### {name}  (index = {len(classifier)} rows)")
    print(f"  gold intent accuracy      : {gold_acc:.3f}")
    print(f"  out-of-scope -> fallback  : {oos_acc:.3f}")
    print(f"  money flow opened on OOS  : {leaks}/{len(oos)} = {leaks / len(oos):.1%}")
    for pair, count in gold_wrong.most_common(8):
        print(f"    gold miss {count:3d}  {pair}")


def main() -> None:
    rows = read_rows()
    index, held_out = partition(rows)
    authored = [row for row in index if row.source != "ArBanking77-SinaLab"]
    oos = [(row.text, Intent.FALLBACK) for row in held_out]

    print(f"csv rows          : {len(rows)}")
    print(f"indexable         : {len(index)}  (authored {len(authored)})")
    print(f"held out (OOS)    : {len(oos)}")
    print(f"dialects held out : {Counter(row.dialect for row in held_out)}")

    report("baseline: built-in examples only", None, oos)
    report("+ authored CSV rows", authored, oos)
    report("+ full CSV (ArBanking77 corpus included)", index, oos)
    for cap in (10, 30, 100):
        report(f"+ CSV capped at {cap} rows per topic", cap_topics(index, cap), oos)


if __name__ == "__main__":
    main()
