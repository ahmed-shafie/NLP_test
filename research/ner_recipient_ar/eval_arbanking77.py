"""Blind out-of-scope check on ArBanking77 (SinaLab / Birzeit University).

The 77 intents are bank *customer-service* topics ("card swallowed", "why verify
identity", "exchange rate"). Our assistant supports five actions, none of which
is in that list, so the correct answer for every row is: do not start a money
flow. Anything that lands on transfer/bill/beneficiary is a false action, and a
recipient or amount extracted from these sentences is invented outright.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/repos/NLP-test")

from app.nlu.pipeline import parse  # noqa: E402
from app.schemas import Intent  # noqa: E402

HERE = Path(__file__).resolve().parent / "arbanking"
FILES = {
    "MSA": "Banking77_Arabized_MSA_test_sample.csv",
    "Palestinian": "Banking77_Arabized_PAL_test_sample.csv",
}
MONEY = {
    Intent.TRANSFER_MONEY,
    Intent.PAY_BILL,
    Intent.ADD_BENEFICIARY,
}


def main() -> None:
    for dialect, name in FILES.items():
        with (HERE / name).open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        intents: Counter[str] = Counter()
        bad: list[tuple[str, str, str | None]] = []
        for row in rows:
            response = parse(row["text"])
            intents[response.intent.value] += 1
            if response.intent in MONEY:
                bad.append(
                    (row["text"], response.intent.value, response.entities.recipient)
                )
        print(f"\n### {dialect}: {len(rows)} customer-service questions")
        for intent, count in intents.most_common():
            print(f"  {intent}: {count} ({count / len(rows):.1%})")
        invented = sum(1 for _, _, recipient in bad if recipient is not None)
        print(f"  -> money flow started on {len(bad)}/{len(rows)}")
        print(f"  -> recipient invented on {invented}/{len(rows)}")
        for text, intent, recipient in bad[:10]:
            print(f"     {text[:70]} -> {intent} recipient={recipient}")


if __name__ == "__main__":
    main()
