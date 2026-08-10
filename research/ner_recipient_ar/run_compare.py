"""Compare recipient extraction: current regex+gazetteer vs Stanza vs HF Arabic NER.

Slices:
  gold   - the 37 Arabic transfer utterances with a recipient in app/eval/nlu_gold.jsonl
  hard   - hand-written harder cases (hard_ar.jsonl), including negatives

Metric: exact match after app.nlu.normalize.normalize (the same folding the
downstream beneficiary lookup uses), so أحمد == احمد. Negatives must yield None.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from app.nlu import entities  # noqa: E402
from app.nlu.normalize import normalize  # noqa: E402
from app.schemas import Language  # noqa: E402

HF_MODELS = [
    "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
    "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
    str(HERE / "model-per"),  # fine-tuned on generated banking utterances
]


def load_gold() -> list[dict[str, object]]:
    rows = []
    with (REPO / "app/eval/nlu_gold.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            recipient = row.get("slots", {}).get("recipient")
            if row.get("language") == "ar" and recipient:
                rows.append({"text": row["text"], "recipient": recipient})
    return rows


def load_massive(kind: str, limit: int, seed: int = 5) -> list[dict[str, object]]:
    """Blind external slice from MASSIVE ar-SA (CC BY 4.0, Amazon Science).

    ``kind="neg"``: real Saudi utterances with no person annotation -> the
    recipient must be None. Nobody on this project wrote them, so they are the
    honest measure of how often a system invents a beneficiary.
    ``kind="per"``: utterances whose gold annotation carries a ``person`` slot.
    """

    import random
    import re

    path = HERE / "data" / "ar-SA.jsonl"
    person = re.compile(r"\[person : ([^\]]+)\]")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["partition"] != "test":  # never seen during fine-tuning
                continue
            match = person.search(row["annot_utt"])
            if kind == "per" and match is not None:
                rows.append({"text": row["utt"], "recipient": match.group(1).strip()})
            elif kind == "neg" and "[person : " not in row["annot_utt"]:
                rows.append({"text": row["utt"], "recipient": None})
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def load_hard() -> list[dict[str, object]]:
    with (HERE / "hard_ar.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eq(predicted: str | None, gold: str | None) -> bool:
    if gold is None:
        return predicted is None
    if predicted is None:
        return False
    return normalize(predicted) == normalize(gold)


# ---- systems ------------------------------------------------------------- #


def sys_regex(text: str) -> str | None:
    return entities.extract_recipient(text, Language.AR)


def make_stanza():
    try:
        import stanza

        nlp = stanza.Pipeline(
            lang="ar", processors="tokenize,ner", download_method=None, verbose=False
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (stanza unavailable: {exc})")
        return None

    def run(text: str) -> str | None:
        doc = nlp(text)
        people = [e.text.strip() for e in doc.ents if e.type in {"PER", "PERSON"}]
        return people[0] if people else None

    return run


def make_hf(model_id: str, *, strip: bool = True):
    from transformers import pipeline

    ner = pipeline(
        "token-classification",
        model=model_id,
        aggregation_strategy="simple",
        device=-1,
    )

    def run(text: str) -> str | None:
        spans = [s for s in ner(text) if s["entity_group"] in {"PER", "PERS", "PERSON"}]
        if not spans:
            return None
        # Slice the original text by character offsets: the pipeline's ``word``
        # field carries subword artefacts ("##حمد") that the offsets do not.
        merged = _merge_adjacent(spans, text)
        start, span = max(merged, key=lambda item: len(item[1]))
        # Only strip an attached preposition when the span really starts at a word
        # boundary: a model that already excluded the "ل" must not lose a letter.
        at_boundary = start == 0 or text[start - 1].isspace()
        return (_strip_proclitic(span) if strip and at_boundary else span) or None

    return run


def _merge_adjacent(spans: list[dict[str, int]], text: str) -> list[tuple[int, str]]:
    """Join PER spans separated only by whitespace ("منى" + "علي" -> "منى علي")."""

    out: list[tuple[int, str]] = []
    start, end = spans[0]["start"], spans[0]["end"]
    for span in spans[1:]:
        if text[end : span["start"]].strip() == "":
            end = span["end"]
            continue
        out.append((start, text[start:end].strip()))
        start, end = span["start"], span["end"]
    out.append((start, text[start:end].strip()))
    return out


def _strip_proclitic(span: str) -> str:
    """Drop the attached preposition ("لمحمد" -> "محمد", "لليلى" -> "ليلى")."""

    head, _, rest = span.partition(" ")
    if head.startswith("لل") and len(head) > 4:
        head = head[2:]
    elif head.startswith("ل") and len(head) > 3:
        head = head[1:]
    return f"{head} {rest}".strip() if rest else head


def _proclitic_candidates(span: str) -> list[str]:
    """Readings of a span whose first word may carry an attached "ل".

    "لمحمد" -> محمد, and "لليلى" is ambiguous: ل+ليلى (a name starting with ل)
    or لل+يلى. Both readings are offered; the beneficiary list decides.
    """

    head, _, rest = span.partition(" ")
    out = []
    for cut in (1, 2):
        if head.startswith("ل" * cut) and len(head) - cut >= 3:
            candidate = head[cut:]
            out.append(f"{candidate} {rest}".strip() if rest else candidate)
    out.append(span)
    return out


BENEFICIARIES = [
    "أحمد حسن",
    "أحمد خالد",
    "أحمد محمود",
    "محمد نور",
    "محمد سعد",
    "منى علي",
    "سارة عادل",
    "ليلى عمر",
]


def with_directory(fn):
    """Resolve the "لـ" ambiguity ("ليلى" vs "لليلى") against the customer's list.

    The NER span may or may not include the attached preposition; the customer's
    own beneficiary list is the cheapest tie-breaker, and it is data the engine
    already has at that point in the turn.
    """

    from rapidfuzz import fuzz, process

    known = [normalize(name) for name in BENEFICIARIES]

    def run(text: str) -> str | None:
        span = fn(text)
        if span is None:
            return None
        candidates = _proclitic_candidates(span)
        best = None
        best_score = 0.0
        for candidate in candidates:
            match = process.extractOne(
                normalize(candidate),
                known,
                scorer=fuzz.token_set_ratio,
                score_cutoff=80,
            )
            if match is not None and match[1] > best_score:
                best, best_score = candidate, match[1]
        # No beneficiary matched (a new payee): the attached preposition is far
        # more common than a name that genuinely starts with "ل", so prefer the
        # stripped form.
        return best or candidates[0]

    return run


def evaluate(name: str, fn, slices: dict[str, list[dict[str, object]]]) -> None:
    print(f"\n### {name}")
    total_time = 0.0
    turns = 0
    for slice_name, rows in slices.items():
        hits = 0
        fp = 0
        misses: list[str] = []
        for row in rows:
            gold = row["recipient"]
            start = time.perf_counter()
            try:
                pred = fn(str(row["text"]))
            except Exception as exc:  # noqa: BLE001
                pred = None
                print(f"  error on {row['text']!r}: {exc}")
            total_time += time.perf_counter() - start
            turns += 1
            if eq(pred, gold if gold is None else str(gold)):
                hits += 1
            else:
                misses.append(f"{row['text']}  gold={gold!r} pred={pred!r}")
                if gold is None and pred is not None:
                    fp += 1
        print(f"  {slice_name}: {hits}/{len(rows)} = {hits / len(rows):.3f}", end="")
        negatives = sum(1 for r in rows if r["recipient"] is None)
        if negatives:
            print(f"   (false positives on {negatives} negatives: {fp})")
        else:
            print()
        for miss in misses[:8]:
            print(f"     x {miss}")
        if len(misses) > 8:
            print(f"     ... {len(misses) - 8} more")
    print(f"  mean latency: {total_time / max(turns, 1) * 1000:.1f} ms/utterance")


def main() -> None:
    slices = {
        "gold(37)": load_gold(),
        "hard(20)": load_hard(),
        "massive-neg(400)": load_massive("neg", 400),
        "massive-per(200)": load_massive("per", 200),
    }
    print(f"slices: {[(k, len(v)) for k, v in slices.items()]}")

    evaluate("current: regex cue + name gazetteer", sys_regex, slices)

    stanza_fn = make_stanza()
    if stanza_fn is not None:
        evaluate("stanza ar NER", stanza_fn, slices)

    for model_id in HF_MODELS:
        try:
            fn = make_hf(model_id)
        except Exception as exc:  # noqa: BLE001
            print(f"\n### {model_id}\n  unavailable: {exc}")
            continue
        evaluate(f"HF {model_id}", fn, slices)

    tuned = str(HERE / "model-per")
    evaluate(
        "HF fine-tuned + beneficiary-directory tie-break",
        with_directory(make_hf(tuned, strip=False)),
        slices,
    )


if __name__ == "__main__":
    main()
