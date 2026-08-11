"""Harvest Saudi banking-app reviews from Apple's public customer-reviews feed.

This is Apple's own published RSS/JSON endpoint (no scraping of rendered pages,
no login, no personal data beyond the public nickname, which we drop). It is the
cheapest source of *real* Saudi customer language about banking that we can use
without a customer of our own.

Output: reviews.jsonl with {app, rating, text} and a banking-relevant subset.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "reviews.jsonl"

APPS = {
    "alrajhi": 1472508112,
    "anb": 6446777169,
    "alinma": 1668637683,
    "riyad": 6451439906,
    "albilad": 6502697024,
    "snb": 1593644558,
    "sab": 1451809913,
    "aljazira": 6584521799,
    "stcbank": 6458546150,
    "urpay": 1585778338,
    "alinmapay": 1492900777,
}

SORTS = ("mostRecent", "mostHelpful")

# The path segments are order-sensitive: "page=" before "id=" silently returns an
# empty feed for most apps.
FEED = (
    "https://itunes.apple.com/sa/rss/customerreviews/"
    "id={app_id}/sortBy={sort}/page={page}/json"
)

ARABIC = re.compile(r"[\u0600-\u06FF]")
# Turns of phrase that put the reviewer in our domain.
BANKING = re.compile(
    r"حوال|حول|أحول|احول|تحويل|ارسل|أرسل|ابعت|ابعث|مستفيد|المستفيدين|"
    r"ايبان|آيبان|رصيد|فاتور|سداد|سدد|حساب|بطاق|تحويلات"
)


def fetch(app_id: int, page: int, sort: str = "mostRecent") -> list[dict[str, object]]:
    """One page of the feed. Empty pages are common and are not an error: the
    endpoint is CDN-cached and returns an empty feed intermittently, so callers
    retry rather than treating the first empty page as the end."""

    url = FEED.format(page=page, app_id=app_id, sort=sort)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        feed = json.load(response)["feed"]
    entries = feed.get("entry", [])
    # Page 1 puts the app's own metadata in the first entry.
    return [e for e in entries if "im:rating" in e]


def main() -> None:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for name, app_id in APPS.items():
        got = 0
        for sort in SORTS:
            for page in range(1, 11):  # Apple caps the feed at 10 pages
                entries: list[dict[str, object]] = []
                for _ in range(3):
                    try:
                        entries = fetch(app_id, page, sort)
                    except Exception as exc:  # noqa: BLE001 - one dead id must not stop the run
                        print(f"  {name} {sort} page {page}: {exc}")
                    if entries:
                        break
                    time.sleep(1.0)
                for entry in entries:
                    text = entry["content"]["label"].strip()
                    if text in seen:
                        continue
                    seen.add(text)
                    rows.append(
                        {
                            "app": name,
                            "rating": int(entry["im:rating"]["label"]),
                            "text": text,
                        }
                    )
                    got += 1
                time.sleep(0.3)
        print(f"{name}: {got} reviews")

    arabic = [r for r in rows if ARABIC.search(str(r["text"]))]
    banking = [r for r in arabic if BANKING.search(str(r["text"]))]
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\ntotal {len(rows)}, arabic {len(arabic)}, banking-relevant {len(banking)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
