"""Score the shipped configuration on the held-out dialect slice, end to end.

Every row is a banking customer-service question in a dialect that is *not* in
the index (Gulf, Moroccan, Tunisian), so the only correct outcome is "do not
start a money flow". Scored through :func:`route_fresh_turn`, i.e. the decision
the customer actually gets, not the raw classifier verdict.
"""

from __future__ import annotations

from collections import Counter

from app.conversation.engine import route_fresh_turn
from app.nlu.lang import detect_language
from app.nlu.semantic_intents import get_semantic_classifier
from app.schemas import Intent
from research.vector_db_v08.load_csv import partition, read_rows

MONEY = {Intent.TRANSFER_MONEY, Intent.PAY_BILL, Intent.ADD_BENEFICIARY}


def main() -> None:
    classifier = get_semantic_classifier()
    if classifier is None:
        raise SystemExit("semantic classifier unavailable")

    _, held_out = partition(read_rows())
    routed: Counter[str] = Counter()
    leaks: Counter[str] = Counter()
    examples: list[str] = []

    for done, row in enumerate(held_out):
        if done % 250 == 0:
            print(f"  {done}/{len(held_out)}", flush=True)
        language = detect_language(row.text)
        intent, confidence = classifier.classify(row.text)
        decision = route_fresh_turn(row.text, language, intent, confidence)
        routed[decision.value] += 1
        if decision in MONEY:
            leaks[row.dialect] += 1
            if len(examples) < 15:
                examples.append(f"{decision.value:16s} {row.text[:70]}")

    total = len(held_out)
    opened = sum(leaks.values())
    print(f"held-out rows      : {total}")
    print(f"routed             : {routed.most_common()}")
    print(f"money flow opened  : {opened}/{total} = {opened / total:.1%}")
    print(f"  by dialect       : {leaks.most_common()}")
    for line in examples:
        print("  ", line)


if __name__ == "__main__":
    main()
