"""Load the v0.8 banking-NLU vector-DB CSV into indexable rows + a held-out slice.

The CSV mixes three things that must not be mixed:

* **two label spaces** — the 77 ArBanking77 customer-service topics (Arabic
  strings such as "وصول البطاقة") sit in the same ``intent`` column as the
  executable intents. The engine cannot act on a topic, so topics map to
  ``FALLBACK`` and the fine label is kept separately as ``topic``;
* **train and test** — the ArBanking77 dialect test splits are in the file. If
  they are indexed, any score measured on them is retrieval of a memorised row,
  so they are held out by their ``split=`` tag and never returned as index rows;
* **single tokens with contradictory labels** — "حوّل" is labelled both
  ``transfer_money`` and ``currency_conversion``. A one-word row is a near
  neighbour of every short utterance, so contradictory ones are dropped.
"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.schemas import Intent

# The CSV is not in the repo; point at your copy with --csv or $VECTOR_DB_CSV.
CSV_PATH = Path(
    os.environ.get("VECTOR_DB_CSV", "data/banking_nlu_vector_db_v08_final.csv")
)

# The CSV's own intent vocabulary, projected onto what the engine can execute.
# Everything absent from this map (including all 77 ArBanking77 topics) is out of
# scope and must land on FALLBACK.
INTENT_MAP: dict[str, Intent] = {
    "transfer_money": Intent.TRANSFER_MONEY,
    "pay_bill": Intent.PAY_BILL,
    "check_balance": Intent.BALANCE_INQUIRY,
    "list_beneficiaries": Intent.LIST_BENEFICIARIES,
    "add_beneficiary": Intent.ADD_BENEFICIARY,
    "greet": Intent.SMALL_TALK,
    "goodbye": Intent.SMALL_TALK,
    "ask_help": Intent.SMALL_TALK,
}

HELD_OUT_SPLITS = {
    "test_saudi",
    "test_moroccan",
    "test_tunisian",
    "test_msa",
    "test_pal",
}

DEFAULT_CAP = 30

# Action verbs that make a row a request rather than a greeting. The CSV labels
# "أهلاً، أريد تحويل مبلغ" as a greeting, which teaches the index to answer a
# transfer request with small talk.
_ACTION_WORDS = (
    "حول",
    "حوّل",
    "ابعت",
    "أرسل",
    "ارسل",
    "تحويل",
    "ادفع",
    "سدد",
    "transfer",
    "send",
    "pay",
)


@dataclass(frozen=True)
class Row:
    text: str
    intent: Intent
    topic: str
    language: str
    dialect: str
    category: str
    weight: float
    source: str
    split: str


def _split_of(notes: str) -> str:
    for part in notes.split("|"):
        part = part.strip()
        if part.startswith("split="):
            return part[len("split=") :]
    return ""


def read_rows(path: Path = CSV_PATH) -> list[Row]:
    out: list[Row] = []
    with path.open(encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            label = raw["intent"]
            intent = INTENT_MAP.get(label, Intent.FALLBACK)
            out.append(
                Row(
                    text=raw["text"].strip(),
                    intent=intent,
                    topic="" if label in INTENT_MAP else label,
                    language=raw["language"],
                    dialect=raw["dialect"],
                    category=raw["category"],
                    weight=float(raw["weight"] or 0),
                    source=raw["source"],
                    split=_split_of(raw["notes"]),
                )
            )
    return out


# "How do I transfer money into my account?" is a customer-service question in
# the corpus, but as an indexed fallback neighbour it teaches the classifier to
# refuse "أرغب في تحويل مبلغ" too. How-to rows about an action we support are
# therefore left out; complaints about the same action ("the transfer never
# arrived") stay, since refusing those is exactly what we want.
_HOWTO_WORDS = ("كيف", "خطوات", "طريقة", "how do i", "how can i")


def is_indexable(row: Row) -> bool:
    """Reject rows that are unusable as nearest-neighbour evidence.

    Three kinds do measurable damage:

    * **fragments** — bare verbs and stubs ("ابعتلو", "اعمل دفعة") labelled with a
      money intent. Embedded, they sit close to *any* short Arabic sentence, so
      "الغِ الامر المستمر" (cancel a standing order) retrieves "عايزة ابعت" at
      0.86 and routes into a transfer;
    * **greetings that are really requests** — "أهلاً، أريد تحويل مبلغ" carries
      the ``greet`` label, which drags genuine transfer wording to small talk;
    * **how-to questions about actions we support** — see ``_HOWTO_WORDS``.
    """

    if row.category != "sentence" or len(row.text.split()) < 3:
        return False
    lowered = row.text.lower()
    has_action = any(word in lowered for word in _ACTION_WORDS)
    if row.intent is Intent.SMALL_TALK:
        return not has_action
    if row.topic and has_action:
        return not any(word in lowered for word in _HOWTO_WORDS)
    return True


def partition(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    """Split into (indexable, held-out) and drop contradictory duplicates.

    A text carrying two different intents is dropped from the index entirely
    rather than resolved: with no evidence for either label, indexing one of them
    is a coin flip that the nearest-neighbour vote then treats as fact.
    """

    labels: dict[str, set[Intent]] = defaultdict(set)
    for row in rows:
        labels[row.text].add(row.intent)
    contradictory = {text for text, seen in labels.items() if len(seen) > 1}

    index: list[Row] = []
    held_out: list[Row] = []
    seen_text: set[str] = set()
    for row in rows:
        if row.split in HELD_OUT_SPLITS:
            held_out.append(row)
            continue
        if row.text in contradictory or row.text in seen_text:
            continue
        if not is_indexable(row):
            continue
        seen_text.add(row.text)
        index.append(row)
    return index, held_out


def cap_topics(index: list[Row], per_topic: int | None) -> list[Row]:
    """Keep every executable-intent row but at most ``per_topic`` rows per topic.

    ``None`` keeps them all, which is what ships: the ~500 near-paraphrases per
    customer-service topic do outvote the transfer examples in the top-k pool,
    but the resulting refusals are underspecified requests the engine now routes
    deterministically, so the safety gain comes free.
    """

    if per_topic is None:
        return list(index)
    kept: list[Row] = []
    seen: Counter[str] = Counter()
    for row in index:
        if not row.topic:
            kept.append(row)
            continue
        if seen[row.topic] < per_topic:
            seen[row.topic] += 1
            kept.append(row)
    return kept


def capped_corpus(path: Path = CSV_PATH, per_topic: int | None = None) -> list[Row]:
    """Read the CSV and return the rows that are safe to index."""

    index, _ = partition(read_rows(path))
    return cap_topics(index, per_topic)
