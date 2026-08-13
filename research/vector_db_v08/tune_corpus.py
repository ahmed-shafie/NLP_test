"""Compare corpus shaping options on both slices at once.

The trade-off is one-dimensional and has to be measured on both sides: every
extra out-of-scope row bought safety on the held-out dialect slice and cost
recall on the intents the assistant executes.
"""

from __future__ import annotations

from app.schemas import Intent
from research.vector_db_v08.eval_index import report
from research.vector_db_v08.load_csv import cap_topics, partition, read_rows


def main() -> None:
    index, held_out = partition(read_rows())
    oos = [(row.text, Intent.FALLBACK) for row in held_out]
    money_free = [
        row
        for row in index
        if not row.topic or not any(w in row.text for w in ("تحويل", "حوال", "رصيد"))
    ]

    report("cap 30 (current)", cap_topics(index, 30), oos)
    report("cap 10", cap_topics(index, 10), oos)
    report("cap 30, no money-adjacent topics", cap_topics(money_free, 30), oos)


if __name__ == "__main__":
    main()
