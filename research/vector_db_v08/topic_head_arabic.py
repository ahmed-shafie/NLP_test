"""Recalibrate the head's gate for Arabic, where one shared setting is too tight.

The head answers a question retrieval refused, but only when it is almost certain
(``topic_head_threshold``) *and* the retrieved rows back it. That probability bar
is currently one number for both languages, and it was fixed by the English
slice: at 0.999 Arabic answers 28.9% of held-out service questions, and simply
lowering it to 0.99 reaches 37.7% at 2.3% wrong — an error rate we rejected.

The trade is not the probability alone, though. The other half of the gate is how
the retrieved rows have to back the head, and today that is the weakest possible
form: the single most-voted topic must map to the head's answer. This script
sweeps the two together, on the Arabic slices only, to find whether a *stronger*
agreement rule buys back the error a lower probability costs.

Agreement rules swept (``support`` = rows whose topic maps to the head's answer):

* ``argmax``  — shipped: the most-voted topic maps to the head's answer;
* ``share``   — support / retrieved >= s, so scattered neighbours cannot pass;
* ``margin``  — ``share`` and support strictly beats every other answer's votes.

Both numbers are measured exactly as ``topic_gate_sweep`` measures them: coverage
is refused questions answered about their subject, and an answer is wrong when
its *text* differs from the text the row's gold topic would have produced.

Neighbours come from the committed cache; the head predictions are encoded once
and cached beside them. Run::

    VECTOR_DB_CSV=/path/to/banking_nlu_vector_db_v08_final.csv \\
        python -m research.vector_db_v08.topic_head_arabic
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings
from app.conversation.topic_replies import (
    answer_key,
    decide,
    reply_for_key,
    topic_reply,
)
from app.embeddings import get_embedder
from app.nlu.topic_head import NO_ANSWER, get_topic_head
from app.schemas import Language
from research.vector_db_v08 import topic_gate_sweep
from research.vector_db_v08.load_csv import partition, read_rows

HEAD_CACHE = Path(__file__).with_name("topic_head_arabic.jsonl")

PROBABILITIES = (0.98, 0.99, 0.995, 0.997, 0.998, 0.999)
SHARES = (0.3, 0.4, 0.5, 0.6, 0.7)
FLOORS = (0.80, 0.84, 0.86, 0.88, 0.90)


@dataclass(frozen=True)
class Row:
    """A held-out Arabic question, its gold topic, neighbours and head reading."""

    text: str
    gold: str
    neighbours: tuple[tuple[str, float], ...]
    key: str
    probability: float

    def evidence(self, k: int) -> tuple[float, dict[str, int], int]:
        window = self.neighbours[:k]
        votes: dict[str, int] = {}
        for topic, _ in window:
            if topic:
                votes[topic] = votes.get(topic, 0) + 1
        return (window[0][1] if window else 0.0), votes, len(window)


def build_head_cache() -> None:
    """Encode every cached Arabic question once and store the head's reading."""

    head = get_topic_head()
    embedder = get_embedder()
    if head is None or embedder is None:
        raise SystemExit(
            "no topic head loaded; run python -m scripts.train_topic_classifier"
        )
    if not topic_gate_sweep.CACHE.exists():
        topic_gate_sweep.build_cache()
    cached = [
        row for row in topic_gate_sweep.read_cache() if row.language is Language.AR
    ]
    vectors = embedder.encode([row.text for row in cached])
    with HEAD_CACHE.open("w", encoding="utf-8") as handle:
        for row, vector in zip(cached, vectors, strict=True):
            prediction = head.predict(np.asarray(vector))
            handle.write(
                json.dumps(
                    {
                        "text": row.text,
                        "gold": row.gold,
                        "neighbours": [list(pair) for pair in row.neighbours],
                        "key": prediction.key if prediction else NO_ANSWER,
                        "probability": prediction.probability if prediction else 0.0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"cached {len(cached)} Arabic questions -> {HEAD_CACHE.name}")


def read_head_cache() -> Iterator[Row]:
    with HEAD_CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            yield Row(
                text=record["text"],
                gold=record["gold"],
                neighbours=tuple(
                    (topic, float(score)) for topic, score in record["neighbours"]
                ),
                key=record["key"],
                probability=float(record["probability"]),
            )


def support(votes: Mapping[str, int], key: str) -> tuple[int, int]:
    """Votes for the head's answer, and the best votes for any other answer.

    Counted per *answer*, not per topic: two topics sharing a reply are not a
    disagreement, because the words the customer reads are identical either way.
    """

    per_answer: dict[str, int] = {}
    for topic, count in votes.items():
        mapped = answer_key(topic)
        if mapped == NO_ANSWER:
            continue
        per_answer[mapped] = per_answer.get(mapped, 0) + count
    mine = per_answer.pop(key, 0)
    return mine, max(per_answer.values(), default=0)


def accepts(
    row: Row,
    k: int,
    probability: float,
    rule: str,
    share: float,
    floor: float | None = None,
) -> bool:
    """Would this gate answer ``row`` from the head?"""

    if row.key == NO_ANSWER or row.probability < probability:
        return False
    top, votes, retrieved = row.evidence(k)
    if top < (settings.topic_head_score_floor if floor is None else floor):
        return False
    if not votes or not retrieved:
        return False
    if rule == "argmax":
        return answer_key(max(votes, key=lambda t: votes[t])) == row.key
    mine, other = support(votes, row.key)
    if mine / retrieved < share:
        return False
    return mine > other if rule == "margin" else True


def score(
    rows: list[Row],
    k: int,
    probability: float,
    rule: str,
    share: float,
    floor: float | None = None,
) -> tuple[int, int]:
    """Coverage and wrong answers for the whole gate: retrieval, then the head."""

    answered = wrong = 0
    for row in rows:
        top, votes, retrieved = row.evidence(k)
        delivered = decide(row.text, top, votes, retrieved, Language.AR, None)
        reply = delivered.reply if delivered is not None else None
        if reply is None and accepts(row, k, probability, rule, share, floor):
            from_head = reply_for_key(row.key, row.text, Language.AR)
            reply = None if from_head is None else from_head[1]
        if reply is None:
            continue
        answered += 1
        if reply != topic_reply(row.gold, Language.AR):
            wrong += 1
    return answered, wrong


def report(name: str, rows: list[Row], k: int) -> None:
    total = len(rows)
    print(f"\n{name}: {total} held-out questions")
    print(f"{'rule':>8} {'share':>6} {'p':>7}   answered            wrong")
    for rule, share in (
        [("argmax", 0.0)]
        + [("share", s) for s in SHARES]
        + [("margin", s) for s in SHARES]
    ):
        for probability in PROBABILITIES:
            answered, wrong = score(rows, k, probability, rule, share)
            flag = (
                "  <- shipped"
                if rule == "argmax" and probability == settings.topic_head_threshold
                else ""
            )
            print(
                f"{rule:>8} {share:6.1f} {probability:7.3f}   "
                f"{answered:5}/{total} = {answered / total:5.1%}   "
                f"{wrong:4}/{max(answered, 1)} = {wrong / max(answered, 1):5.2%}{flag}"
            )


def report_floors(name: str, rows: list[Row], k: int) -> None:
    """Sweep the similarity floor against the probability, shipped rule."""

    total = len(rows)
    print(f"\n{name}: {total} held-out questions, rule=argmax")
    print(f"{'floor':>6} {'p':>7}   answered            wrong")
    for floor in FLOORS:
        for probability in PROBABILITIES:
            answered, wrong = score(rows, k, probability, "argmax", 0.0, floor)
            print(
                f"{floor:6.2f} {probability:7.3f}   "
                f"{answered:5}/{total} = {answered / total:5.1%}   "
                f"{wrong:4}/{max(answered, 1)} = {wrong / max(answered, 1):5.2%}"
            )


def main() -> None:
    if not HEAD_CACHE.exists():
        build_head_cache()
    rows = list(read_head_cache())
    k = settings.topic_reply_top_k
    _, held_out = partition(read_rows())
    saudi = {row.text for row in held_out if row.split == "test_saudi"}
    print(
        f"similarity floor {settings.topic_head_score_floor}, k={k}; "
        f"retrieval runs first and the head only speaks where it refused."
    )
    saudi_rows = [row for row in rows if row.text in saudi]
    report("Arabic - Saudi split", saudi_rows, k)
    report("Arabic - all dialects", rows, k)
    report_floors("Arabic - Saudi split", saudi_rows, k)
    report_floors("Arabic - all dialects", rows, k)


if __name__ == "__main__":
    main()
