"""Calibrate the topic-answer gate against the gold topics of the held-out slice.

Retrieval is the expensive part and it does not depend on the gate, so the 15
nearest neighbours of every held-out question are cached once to
``topic_evidence.jsonl``; every candidate gate — including a different ``k`` — is
then scored offline in seconds.

Two numbers are traded off:

* **coverage** — refused questions answered about their subject instead of the
  generic menu;
* **wrong answer** — the delivered text differs from the text the row's *gold*
  topic would have produced. That is stricter than "wrong topic": two topics
  sharing a family answer produce the same words, and identical words cannot
  mislead anybody.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.conversation.topic_replies import decide, topic_reply
from app.nlu.lang import detect_language
from app.nlu.semantic_intents import get_semantic_classifier
from app.schemas import Language
from research.vector_db_v08.load_csv import partition, read_rows

CACHE = Path(__file__).with_name("topic_evidence.jsonl")
CACHE_K = 15


@dataclass(frozen=True)
class Cached:
    """A held-out question, its gold topic, and its cached neighbours."""

    text: str
    gold: str
    language: Language
    neighbours: tuple[tuple[str, float], ...]

    def evidence(self, k: int) -> tuple[float, dict[str, int], int]:
        """Votes over the nearest ``k`` cached neighbours."""

        window = self.neighbours[:k]
        votes: dict[str, int] = defaultdict(int)
        for topic, _ in window:
            if topic:
                votes[topic] += 1
        top = window[0][1] if window else 0.0
        return top, dict(votes), len(window)


def build_cache() -> None:
    classifier = get_semantic_classifier()
    if classifier is None:
        raise SystemExit("semantic classifier unavailable")
    _, held_out = partition(read_rows())
    with CACHE.open("w", encoding="utf-8") as handle:
        for done, row in enumerate(held_out):
            if done % 500 == 0:
                print(f"  {done}/{len(held_out)}", flush=True)
            neighbours = classifier.similar(row.text, k=CACHE_K)
            handle.write(
                json.dumps(
                    {
                        "text": row.text,
                        "gold": row.topic,
                        "language": detect_language(row.text).value,
                        "neighbours": [
                            [n.topic, round(n.score, 4)] for n in neighbours
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_cache() -> Iterator[Cached]:
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            yield Cached(
                text=raw["text"],
                gold=raw["gold"],
                language=Language(raw["language"]),
                neighbours=tuple((t, s) for t, s in raw["neighbours"]),
            )


def score(rows: list[Cached], k: int) -> tuple[int, int, list[tuple[str, str, str]]]:
    """Return (answered, wrong text, wrong examples) for the current gate."""

    answered = 0
    wrong = 0
    examples: list[tuple[str, str, str]] = []
    for row in rows:
        top, votes, retrieved = row.evidence(k)
        answer = decide(row.text, top, votes, retrieved, row.language)
        if answer is None:
            continue
        answered += 1
        # The gold reply is read from the gold topic alone: correcting it with
        # the same cues the gate uses would score the correction against itself.
        if answer.reply != topic_reply(row.gold, row.language):
            wrong += 1
            if len(examples) < 12:
                examples.append((row.gold, answer.subject, row.text))
    return answered, wrong, examples


def main() -> None:
    if not CACHE.exists():
        build_cache()
    rows = list(read_cache())
    total = len(rows)
    print(f"held-out rows: {total}\n")
    header = f"{'k':>3} {'unanimous':>9} {'majority':>8} {'agree':>5}"
    print(f"{header} {'answered':>18} {'wrong':>16}")

    shipped = (
        settings.topic_reply_top_k,
        settings.topic_reply_unanimous_threshold,
        settings.topic_reply_threshold,
        settings.topic_reply_agreement,
    )
    for k in (3, 5, 10, 15):
        for uni in (0.74, 0.78, 0.82, 0.86):
            for maj in (0.90, 0.94):
                for agree in (0.6, 0.8):
                    settings.topic_reply_unanimous_threshold = uni
                    settings.topic_reply_threshold = maj
                    settings.topic_reply_agreement = agree
                    answered, wrong, _ = score(rows, k)
                    if not answered:
                        continue
                    share = f"{answered}/{total} = {answered / total:5.1%}"
                    bad = f"{wrong}/{answered} = {wrong / answered:5.1%}"
                    mark = " <- shipped" if (k, uni, maj, agree) == shipped else ""
                    print(
                        f"{k:3d} {uni:9.2f} {maj:8.2f} {agree:5.1f} "
                        f"{share:>18} {bad:>16}{mark}"
                    )

    k, uni, maj, agree = shipped
    settings.topic_reply_unanimous_threshold = uni
    settings.topic_reply_threshold = maj
    settings.topic_reply_agreement = agree
    answered, wrong, examples = score(rows, k)
    print(f"\nshipped gate: k={k} unanimous={uni} majority={maj} agreement={agree}")
    print(f"  answered   : {answered}/{total} = {answered / total:.1%}")
    print(f"  wrong text : {wrong}/{answered} = {wrong / answered:.1%}")
    for gold, given, text in examples:
        print(f"   asked about {gold!r}, answered about {given!r}: {text[:60]}")


if __name__ == "__main__":
    main()
