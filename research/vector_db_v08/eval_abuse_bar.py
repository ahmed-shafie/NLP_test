"""How high can the abuse bar sit before real complaints trip it?

Deciding abuse by its nearest example (instead of its share of the retrieved
pool) needs a similarity bar. The held-out slice is the right place to set it:
7,667 genuine banking complaints, many of them angry, none of them abuse.
"""

from __future__ import annotations

from collections import Counter

from app.nlu.semantic_intents import get_semantic_classifier
from app.schemas import Intent
from research.vector_db_v08.load_csv import partition, read_rows

BARS = (0.70, 0.75, 0.80, 0.85)


def main() -> None:
    classifier = get_semantic_classifier()
    if classifier is None:
        raise SystemExit("semantic classifier unavailable")

    _, held_out = partition(read_rows())
    tripped: Counter[float] = Counter()
    worst: list[tuple[float, str]] = []

    for row in held_out:
        abuse = max(
            (
                n.score
                for n in classifier.similar(row.text, 5)
                if n.intent is Intent.INAPPROPRIATE
            ),
            default=0.0,
        )
        for bar in BARS:
            if abuse >= bar:
                tripped[bar] += 1
        if abuse >= BARS[0]:
            worst.append((abuse, row.text))

    total = len(held_out)
    for bar in BARS:
        print(f"bar {bar:.2f}: {tripped[bar]}/{total} complaints flagged as abuse")
    for score, text in sorted(worst, reverse=True)[:10]:
        print(f"  {score:.3f}  {text[:80]}")


if __name__ == "__main__":
    main()
