"""Score the shipped topic head against retrieval alone, on both held-out slices.

The retrieval gate votes over the nearest indexed rows and answers only when the
vote is decisive *and* the top similarity clears a bar. Similarity is a raw
distance, not a probability: it cannot say "70% sure", so the bar has to sit high
enough for the worst subject and every other subject pays for it — which is why
85% of Arabic service questions still get the generic menu.

``app/nlu/topic_head.py`` adds a supervised head over the *same* query vector,
which does produce a probability. This script measures what that buys, through
the shipped :func:`app.conversation.topic_replies.decide` — not a copy of it — so
the numbers quoted in ``app/config.py`` describe the code that runs:

* **coverage** — refused questions answered about their subject, not the menu;
* **wrong answer** — the delivered text differs from the text the row's *gold*
  topic would have produced (stricter than "wrong topic": two topics sharing a
  family answer produce identical words, which cannot mislead anybody).

Both slices are held out of the index: the ArBanking77 dialect test splits (the
Saudi one is reported separately — it is the audience of record) and the English
Banking77 test split, whose 77 categories are the same subjects because
ArBanking77 is its translation.

Run (needs the neighbour caches of ``topic_gate_sweep`` and
``topic_gate_english``, which both build on first use)::

    VECTOR_DB_CSV=/path/to/banking_nlu_vector_db_v08_final.csv \\
    BANKING77_TEST_CSV=/path/to/banking77_test.csv \\
        python -m research.vector_db_v08.topic_classifier
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.conversation.topic_replies import decide, topic_reply, topic_reply_top_k
from app.embeddings import get_embedder
from app.nlu.topic_head import Prediction, get_topic_head
from app.schemas import Language
from research.vector_db_v08 import topic_gate_english, topic_gate_sweep
from research.vector_db_v08.load_csv import partition, read_rows


@dataclass(frozen=True)
class Row:
    """A held-out question with its gold topic, neighbours and head prediction."""

    text: str
    gold: str
    language: Language
    neighbours: tuple[tuple[str, float], ...]
    prediction: Prediction | None

    def evidence(self, k: int) -> tuple[float, dict[str, int], int]:
        window = self.neighbours[:k]
        votes: dict[str, int] = {}
        for topic, _ in window:
            if topic:
                votes[topic] = votes.get(topic, 0) + 1
        return (window[0][1] if window else 0.0), votes, len(window)


def predictions(texts: list[str]) -> list[Prediction | None]:
    """Read every question with the shipped head (batched, one encode each)."""

    head = get_topic_head()
    embedder = get_embedder()
    if head is None or embedder is None:
        return [None] * len(texts)
    vectors = embedder.encode(texts)
    return [head.predict(np.asarray(vector)) for vector in vectors]


def arabic_rows() -> list[Row]:
    if not topic_gate_sweep.CACHE.exists():
        topic_gate_sweep.build_cache()
    cached = list(topic_gate_sweep.read_cache())
    heads = predictions([row.text for row in cached])
    return [
        Row(
            text=row.text,
            gold=row.gold,
            language=row.language,
            neighbours=row.neighbours,
            prediction=head,
        )
        for row, head in zip(cached, heads, strict=True)
    ]


def english_rows() -> list[Row]:
    if not topic_gate_english.CACHE.exists():
        topic_gate_english.build_cache()
    cached = list(topic_gate_english.read_cache())
    heads = predictions([row.text for row in cached])
    return [
        Row(
            text=row.text,
            gold=row.gold,
            language=Language.EN,
            neighbours=row.neighbours,
            prediction=head,
        )
        for row, head in zip(cached, heads, strict=True)
    ]


def score(rows: list[Row], *, with_head: bool) -> tuple[int, int]:
    """Return (answered, wrong) for the shipped gate over ``rows``."""

    answered = wrong = 0
    for row in rows:
        top, votes, retrieved = row.evidence(topic_reply_top_k(row.language))
        answer = decide(
            row.text,
            top,
            votes,
            retrieved,
            row.language,
            row.prediction if with_head else None,
        )
        if answer is None:
            continue
        answered += 1
        if answer.reply != topic_reply(row.gold, row.language):
            wrong += 1
    return answered, wrong


def report(name: str, rows: list[Row]) -> None:
    total = len(rows)
    if not total:
        print(f"{name}: no rows")
        return
    print(f"\n{name}: {total} held-out questions")
    for label, with_head in (("retrieval only", False), ("+ trained head", True)):
        answered, wrong = score(rows, with_head=with_head)
        print(
            f"  {label:15} answered {answered:5}/{total} = {answered / total:5.1%}"
            f"   wrong {wrong:4}/{max(answered, 1)} = {wrong / max(answered, 1):5.2%}"
        )


def main() -> None:
    head = get_topic_head()
    if head is None:
        raise SystemExit(
            "no topic head loaded; run python -m scripts.train_topic_classifier"
        )
    print(
        f"head: {len(head.answers)} answers, probability >= "
        f"{settings.topic_head_threshold}, similarity floor "
        f"{settings.topic_head_score_floor}"
    )

    arabic = arabic_rows()
    _, held_out = partition(read_rows())
    saudi = {row.text for row in held_out if row.split == "test_saudi"}
    report("Arabic — Saudi split", [row for row in arabic if row.text in saudi])
    report("Arabic — all dialects", arabic)
    report("English — Banking77 test", english_rows())


if __name__ == "__main__":
    main()
