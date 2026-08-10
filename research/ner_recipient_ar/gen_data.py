"""Generate labelled Arabic banking utterances for PER token-classification.

Labels are exact by construction: the generator knows which characters it wrote
as the beneficiary name, so no manual annotation is involved.

Two deliberate design choices:
* the name pool is split, so the evaluation half of the names never appears in
  training - a model that only memorised the gazetteer cannot score;
* roughly a third of the rows carry no name at all (balance, bills, small talk,
  own-account transfers), because the expensive error in production is inventing
  a recipient, not missing one.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

SURNAMES = [
    "حسن",
    "خالد",
    "محمود",
    "نور",
    "سعد",
    "علي",
    "عادل",
    "عمر",
    "الحربي",
    "القحطاني",
    "الشمري",
    "الغامدي",
    "العتيبي",
    "الدوسري",
]

VERBS = [
    "حول",
    "حوّل",
    "حولي",
    "ارسل",
    "أرسل",
    "ابعت",
    "ابعث",
    "سدد",
    "ودّي",
    "طيّر",
    "انقل",
    "اصرف",
]
PREAMBLES = [
    "",
    "ابغى ",
    "أبغى ",
    "اريد ",
    "أريد ",
    "لو سمحت ",
    "تكفى ",
    "ممكن ",
    "بدي ",
    "عايز ",
    "ودي ",
    "من فضلك ",
]
PREPS = ["إلى ", "الى ", "الي ", "ل", "إلي ", "لـ"]
CURRENCIES = ["", " ريال", " دولار", " درهم", " جنيه", " يورو", " ريال سعودي"]
RELATIONS = ["", "أخوي ", "اخوي ", "صاحبي ", "زميلي ", "ابني ", "اختي ", "المستفيد "]
TAILS = [
    "",
    " من حسابي الجاري",
    " من حساب التوفير",
    " من حسابي",
    " اليوم",
    " بسرعة لو سمحت",
    " الحوالة مستعجلة",
]

# Utterances that must yield no beneficiary at all.
NEGATIVES = [
    "كم رصيدي",
    "ايش رصيدي الحالي",
    "كم فلوسي في الجاري",
    "ابغى ادفع فاتورة الكهرباء",
    "سدد فاتورة الماء",
    "ابغى ادفع فاتورة الاتصالات",
    "عايز اسدد فاتورة النت",
    "ابغى اضيف مستفيد جديد",
    "اضف مستفيد",
    "ابغى اشوف المستفيدين",
    "وين اقرب فرع لكم",
    "ايش دوام الفروع",
    "شكرا على مساعدتك",
    "شكرا لك",
    "مرحبا",
    "اهلا وسهلا",
    "كيف حالك",
    "مين انت وتقدر تسوي ايش",
    "الي فيه مشكلة في التطبيق كلمني",
    "التطبيق الي عندي مايفتح",
    "حول 500 إلى حسابي",
    "حول 300 إلى حساب التوفير",
    "انقل 1000 من الجاري إلى التوفير",
    "حول 250 لحسابي الجاري",
    "ابغى احول بين حساباتي",
    "الغاء الحوالة",
    "لا الغها",
    "نعم اكد",
    "ايش عمولة الحوالة",
    "كم حد التحويل اليومي",
]


def arabic_given_names(limit: int = 1200) -> list[str]:
    names: list[str] = []
    with (REPO / "app/data/names.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ar = (row.get("name_ar") or "").strip()
            # Skip the dictionary noise that made the gazetteer match plain
            # speech: single/double-letter "names" and function words.
            if len(ar) >= 4 and " " not in ar:
                names.append(ar)
    random.shuffle(names)
    return names[:limit]


def amount() -> str:
    value = random.choice(
        [
            50,
            75,
            90,
            100,
            120,
            150,
            200,
            250,
            300,
            500,
            750,
            1000,
            1500,
            2000,
            3200,
            5000,
            10000,
        ]
    )
    text = str(value)
    if random.random() < 0.25:  # Arabic-Indic digits
        text = text.translate({ord(str(i)): "٠١٢٣٤٥٦٧٨٩"[i] for i in range(10)})
    return text


def full_name(pool: list[str]) -> str:
    name = random.choice(pool)
    if random.random() < 0.45:
        name = f"{name} {random.choice(SURNAMES)}"
    return name


def with_prep(name: str) -> tuple[str, int]:
    """Return the rendered "preposition + name" and the offset of the name."""

    prep = random.choice(PREPS)
    if prep in {"ل", "لـ"}:
        # Attached preposition: "لمحمد". Arabic writes "لل" before a name
        # starting with alef-lam.
        return f"{prep}{name}", len(prep)
    return f"{prep}{name}", len(prep)


def render_positive(pool: list[str]) -> tuple[str, int, int]:
    name = full_name(pool)
    shape = random.random()
    pre = random.choice(PREAMBLES)
    verb = random.choice(VERBS)
    amt = amount()
    cur = random.choice(CURRENCIES)
    tail = random.choice(TAILS)
    relation = random.choice(RELATIONS) if random.random() < 0.2 else ""

    if shape < 0.55:  # verb, amount, then the name
        head = f"{pre}{verb} {amt}{cur} "
        prep_name, offset = with_prep(f"{relation}{name}")
        text = f"{head}{prep_name}{tail}"
        start = len(head) + offset + len(relation)
    elif shape < 0.75:  # verb, name, then the amount
        head = f"{pre}{verb} "
        prep_name, offset = with_prep(f"{relation}{name}")
        text = f"{head}{prep_name} {amt}{cur}{tail}"
        start = len(head) + offset + len(relation)
    elif shape < 0.88:  # name first, no preposition at all
        verb_phrase = random.choice(
            [
                f"يحتاج {amt}{cur}",
                f"محتاج {amt}{cur}",
                f"حول له {amt}{cur}",
                f"ابغى احول له {amt}{cur}",
                f"تحتاج تحويل {amt}{cur}",
            ]
        )
        text = f"{name} {verb_phrase}{tail}"
        start = 0
    else:  # no amount at all - the engine will ask for it
        head = f"{pre}{verb} "
        prep_name, offset = with_prep(name)
        text = f"{head}{prep_name}"
        start = len(head) + offset

    assert text[start : start + len(name)] == name, (text, start, name)
    return text, start, start + len(name)


def build(rows: int, pool: list[str], seed: int) -> list[dict[str, object]]:
    random.seed(seed)
    out: list[dict[str, object]] = []
    for _ in range(rows):
        if random.random() < 0.3:
            text = random.choice(NEGATIVES)
            out.append({"text": text, "spans": []})
            continue
        text, start, end = render_positive(pool)
        out.append({"text": text, "spans": [[start, end]]})
    return out


_PERSON_SLOT = re.compile(r"\[person : ([^\]]+)\]")


def massive_rows(
    partition: str, limit_neg: int, limit_per: int
) -> list[dict[str, object]]:
    """Real Saudi utterances from MASSIVE ar-SA as out-of-domain supervision.

    Negatives (no person annotation) are the important half: a model trained only
    on our own generated banking sentences labels any unfamiliar token after a
    preposition as a beneficiary, which is exactly the failure the gazetteer had.
    Split by MASSIVE's own partition so the evaluation rows are never trained on.
    """

    path = HERE / "data" / "ar-SA.jsonl"
    negatives: list[dict[str, object]] = []
    persons: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["partition"] != partition:
                continue
            match = _PERSON_SLOT.search(row["annot_utt"])
            if match is None:
                negatives.append({"text": row["utt"], "spans": []})
                continue
            name = match.group(1).strip()
            start = row["utt"].find(name)
            if start >= 0:
                persons.append(
                    {"text": row["utt"], "spans": [[start, start + len(name)]]}
                )
    random.shuffle(negatives)
    random.shuffle(persons)
    return negatives[:limit_neg] + persons[:limit_per]


def main() -> None:
    names = arabic_given_names()
    half = len(names) // 2
    train_pool, holdout_pool = names[:half], names[half:]

    train = build(6000, train_pool, seed=13)
    holdout = build(600, holdout_pool, seed=77)

    random.seed(21)
    external = massive_rows("train", limit_neg=3000, limit_per=700)
    print(f"external MASSIVE(train partition) rows: {len(external)}")
    train += external
    random.shuffle(train)

    for name, rows in (("train.jsonl", train), ("holdout.jsonl", holdout)):
        with (HERE / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        named = sum(1 for r in rows if r["spans"])
        print(f"{name}: {len(rows)} rows, {named} named, {len(rows) - named} without")
    print(f"name pools: train={len(train_pool)} holdout={len(holdout_pool)} (disjoint)")
    for row in train[:5]:
        print(" ", row)


if __name__ == "__main__":
    main()
