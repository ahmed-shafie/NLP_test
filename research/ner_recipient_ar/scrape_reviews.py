"""Harvest Arabic banking-app reviews from Apple's public customer-reviews feed.

This is Apple's own published RSS/JSON endpoint (no page scraping, no login, no
personal data beyond the public nickname, which we drop). It is the cheapest
source of *real* Arabic customer language about banking that we can use without
having customers of our own.

Apps are discovered per storefront with the public iTunes Search API and kept
only if Apple files them under Finance.

Output: reviews.jsonl with {country, app, app_id, rating, text}.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT = HERE / "reviews.jsonl"

# Arabic-speaking storefronts.
COUNTRIES = [
    "ae",
    "kw",
    "qa",
    "bh",
    "om",
    "jo",
    "lb",
    "iq",
    "ma",
    "dz",
    "tn",
    "eg",
    "sa",
]

TERMS = ["بنك", "مصرف", "بنكي", "محفظة", "bank", "mobile banking", "wallet"]

SORTS = ("mostRecent", "mostHelpful")
PAGES = 6
# One request per second, single-threaded: anything faster gets 403-throttled
# within a few hundred requests and then the whole IP is blocked for minutes.
THROTTLE = 1.0

SEARCH = "https://itunes.apple.com/search"
# The path segments are order-sensitive: "page=" before "id=" silently returns an
# empty feed for most apps.
FEED = (
    "https://itunes.apple.com/{country}/rss/customerreviews/"
    "id={app_id}/sortBy={sort}/page={page}/json"
)

ARABIC = re.compile(r"[\u0600-\u06FF]")
# Turns of phrase that put the reviewer in our domain.
BANKING = re.compile(
    r"حوال|حول|أحول|احول|تحويل|ارسل|أرسل|ابعت|ابعث|مستفيد|المستفيدين|"
    r"ايبان|آيبان|رصيد|فاتور|سداد|سدد|حساب|بطاق|تحويلات"
)


def _get(url: str) -> dict[str, Any]:
    """GET with backoff. The feed rate-limits with 403 under any real load, so a
    403 is a "come back later", not a permanent failure."""

    time.sleep(THROTTLE)
    delay = 15.0
    for attempt in range(4):
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 403 or attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def discover(country: str, per_term: int = 12) -> dict[int, str]:
    """Finance apps on one storefront, keyed by track id."""

    apps: dict[int, str] = {}
    for term in TERMS:
        query = urllib.parse.urlencode(
            {"term": term, "country": country, "entity": "software", "limit": per_term}
        )
        try:
            results = _get(f"{SEARCH}?{query}").get("results", [])
        except Exception as exc:  # noqa: BLE001 - one bad storefront must not stop the run
            print(f"  search {country}/{term}: {exc}")
            continue
        for row in results:
            if row.get("primaryGenreName") == "Finance":
                apps[int(row["trackId"])] = str(row["trackName"])
        time.sleep(0.3)
    return apps


def fetch(
    country: str, app_id: int, page: int, sort: str = "mostRecent"
) -> list[dict[str, Any]]:
    """One page of the feed. Empty pages are common and are not an error: the
    endpoint is CDN-cached and returns an empty feed intermittently, so callers
    retry rather than treating the first empty page as the end."""

    url = FEED.format(country=country, page=page, app_id=app_id, sort=sort)
    entries = _get(url)["feed"].get("entry", [])
    # Page 1 puts the app's own metadata in the first entry.
    return [e for e in entries if "im:rating" in e]


def harvest(country: str, app_id: int, name: str) -> list[dict[str, object]]:
    """Every review the feed will give up for one app."""

    out: list[dict[str, object]] = []
    for sort in SORTS:
        for page in range(1, PAGES + 1):  # Apple caps the feed at 10 pages
            entries: list[dict[str, Any]] = []
            for _ in range(2):
                try:
                    entries = fetch(country, app_id, page, sort)
                except Exception as exc:  # noqa: BLE001 - one dead app must not stop the run
                    print(f"  {country}/{app_id} {sort} p{page}: {exc}")
                if entries:
                    break
                time.sleep(2.0)
            for entry in entries:
                out.append(
                    {
                        "country": country,
                        "app": name,
                        "app_id": app_id,
                        "rating": int(entry["im:rating"]["label"]),
                        "text": entry["content"]["label"].strip(),
                    }
                )
    return out


def main() -> None:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    if OUT.exists():  # resume: the feed rate-limits, so runs are incremental
        for line in OUT.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["text"] not in seen:
                seen.add(row["text"])
                rows.append(row)
        print(f"resuming with {len(rows)} rows")
    for country in COUNTRIES:
        apps = discover(country)
        before = len(rows)

        for app_id, name in apps.items():
            for row in harvest(country, app_id, name):
                text = str(row["text"])
                if text in seen:
                    continue
                seen.add(text)
                rows.append(row)
        got = rows[before:]
        arabic = [r for r in got if ARABIC.search(str(r["text"]))]
        print(f"{country}: {len(apps)} apps, {len(got)} reviews, {len(arabic)} arabic")

        with OUT.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    arabic = [r for r in rows if ARABIC.search(str(r["text"]))]
    banking = [r for r in arabic if BANKING.search(str(r["text"]))]
    print(f"\ntotal {len(rows)}, arabic {len(arabic)}, banking-relevant {len(banking)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
