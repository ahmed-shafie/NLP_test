"""Turn the v0.8 vector-DB CSV into the curated example corpus the index loads.

The raw CSV cannot be indexed as it stands (see
``research/vector_db_v08/load_csv.py``). This script applies the three
corrections and writes ``app/nlu/data/example_corpus.jsonl``:

* the 77 ArBanking77 customer-service topics are projected onto ``fallback`` —
  the engine cannot execute a topic, and a customer-service question must never
  open a money flow;
* the dialect test splits are dropped, so they stay a valid measurement set;
* ``--cap`` limits how many rows one topic may contribute (default: no cap).
  Capping trades safety for recall in one direction only — measured end to end,
  the uncapped corpus opens a money flow on 1.8% of the held-out complaints
  against 11.4% before, at no cost to gold accuracy (1.000).

The output is committed for the PoC. 39k of its rows derive from ArBanking77,
whose dataset card carries no licence — see the README before shipping it.

Usage::

    python -m scripts.build_example_corpus --csv path/to/vector_db.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.vector_db_v08.load_csv import CSV_PATH, capped_corpus

OUT = Path("app/nlu/data/example_corpus.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--cap", type=int, default=0, help="0 = no cap")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rows = capped_corpus(args.csv, args.cap or None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "text": row.text,
                        "intent": row.intent.value,
                        "topic": row.topic,
                        "language": row.language,
                        "dialect": row.dialect,
                        "source": row.source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
