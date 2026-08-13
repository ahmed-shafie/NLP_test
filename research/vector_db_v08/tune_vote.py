"""Compare neighbour-vote aggregations under a heavily imbalanced index.

Summing similarity per intent counts votes, so with 85% of the index labelled
out of scope a correct-but-outnumbered neighbour loses: "أرغب في تحويل مبلغ"
retrieves the right transfer example at 0.753 and is still refused. Taking each
intent's *best* neighbour instead makes an intent as strong as its best evidence,
which is the standard answer to class imbalance in k-NN — but it also lets a
single spurious neighbour win, so both slices are scored here.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from app.conversation.engine import route_fresh_turn
from app.nlu.lang import detect_language
from app.nlu.semantic_intents import SimilarExample, get_semantic_classifier
from app.schemas import Intent
from research.vector_db_v08.load_csv import partition, read_rows

GOLD = Path("app/eval/nlu_gold.jsonl")
MONEY = {Intent.TRANSFER_MONEY, Intent.PAY_BILL, Intent.ADD_BENEFICIARY}


def vote(
    neighbours: list[SimilarExample], aggregate: str, threshold: float
) -> tuple[Intent, float]:
    if not neighbours or neighbours[0].score < threshold:
        return Intent.FALLBACK, round(max(neighbours[0].score, 0.0), 4)
    weights: dict[Intent, float] = defaultdict(float)
    for n in neighbours:
        score = max(n.score, 0.0)
        if aggregate == "max":
            weights[n.intent] = max(weights[n.intent], score)
        else:
            weights[n.intent] += score
    total = sum(weights.values()) or 1.0
    best = max(weights, key=lambda i: weights[i])
    share = weights[best] / total
    return best, round(min(0.5 * share + 0.5 * neighbours[0].score, 0.99), 4)


def main() -> None:
    classifier = get_semantic_classifier()
    if classifier is None:
        raise SystemExit("semantic classifier unavailable")
    _, held_out = partition(read_rows())
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()]

    for aggregate in ("sum", "max"):
        for top_k in (5, 15):
            hits = 0
            for row in gold:
                neighbours = classifier.similar(row["text"], top_k)
                intent, confidence = vote(neighbours, aggregate, 0.45)
                language = detect_language(row["text"])
                decision = route_fresh_turn(row["text"], language, intent, confidence)
                hits += decision is Intent(row["intent"])

            leaks: Counter[str] = Counter()
            for csv_row in held_out:
                neighbours = classifier.similar(csv_row.text, top_k)
                intent, confidence = vote(neighbours, aggregate, 0.45)
                language = detect_language(csv_row.text)
                decision = route_fresh_turn(csv_row.text, language, intent, confidence)
                if decision in MONEY:
                    leaks[csv_row.dialect] += 1

            opened = sum(leaks.values())
            print(
                f"{aggregate:4s} k={top_k:<3d} gold {hits / len(gold):.3f} "
                f"({hits}/{len(gold)})  money flow on OOS "
                f"{opened}/{len(held_out)} = {opened / len(held_out):.1%}",
                flush=True,
            )


if __name__ == "__main__":
    main()
