# Guide — Build a Bilingual NLU "Case" from Scratch

> A guided, teach-yourself walkthrough of the `service-template/` package.
> Read it top-to-bottom the first time; use it as a reference afterwards.
>
> **What you'll learn:** how a natural-language request ("send 500 SAR to Ahmed")
> becomes a validated JSON action object — using an intent classifier, slot
> extraction (regex + spaCy), a FAISS semantic search, and a finite-state
> machine (FSM). **No LLM is involved anywhere.**

---

## 0. The 60-second mental model

The whole system is a pipeline that turns *text* into a *typed action object*:

```
 "send 500 SAR to Ahmed"
        │
        ▼
 ┌───────────────┐   detect_language()  → EN / AR
 │  1. NLU        │   detect_intent()    → transfer_money        (spaCy + FAISS)
 │  (extractor)   │   extract_slots()    → {amount, currency, …} (regex + spaCy NER)
 └───────┬───────┘
         ▼
 ┌───────────────┐   fill slots one question at a time,
 │  2. ENGINE     │   resolve the recipient (disambiguate if
 │  (FSM)         │   several match), ask for confirmation
 └───────┬───────┘
         ▼
 ┌───────────────┐   TransferAction(amount=…, currency=…, …)
 │  3. VALIDATE   │   → raises if anything is wrong
 │  (schemas)     │   → otherwise a clean JSON object
 └───────┬───────┘
         ▼
 { "intent": "transfer_money", "amount": "500", "currency": "SAR",
   "recipient": "Ahmed Khaled", ... }   ← handed to a downstream system to EXECUTE
```

Two ideas to hold onto:

1. **The template never executes anything.** It only *understands* the request
   and produces a validated JSON object. A separate banking/core system does the
   real money movement. This keeps the NLU layer safe to experiment with.
2. **Each concern lives in its own file.** You can rewrite the NLU without
   touching the FSM, swap the storage without touching the prompts, etc. Learn
   the boundaries and the rest is easy.

---

## 1. The vocabulary (glossary)

| Term | Plain-English meaning |
|------|-----------------------|
| **Intent** | *What the user wants to do* — a category like `transfer_money`, `small_talk`, `fallback`. |
| **Slot** | *A single piece of information* the action needs — `amount`, `currency`, `recipient`. |
| **Slot filling** | Collecting slots across several turns, asking one question at a time. |
| **Entity / NER** | A named thing in the text (a **PERSON**, a number). *Named-Entity Recognition* finds them. |
| **Embedding** | A sentence turned into a list of numbers (a *vector*) so similar sentences have similar vectors. |
| **FAISS** | A library that, given a query vector, quickly finds the nearest stored vectors (nearest-neighbour search). |
| **Semantic classification** | Guessing the intent by finding the most *similar* example sentences (via embeddings + FAISS), instead of matching exact keywords. |
| **FSM (finite-state machine)** | A tiny state chart: the conversation is always in exactly one *status* (COLLECTING, CONFIRMING, …) and each message moves it to the next. |
| **Pre-flight** | An advisory check (e.g. "enough balance?") done *before* confirming. It can *warn* but here never *blocks*. |
| **Action object** | The final, strictly-validated result — the JSON the downstream system consumes. |

---

## 2. Tour of the files (what to read, in order)

Read them in this order the first time — each builds on the previous one.

| # | File | What it teaches |
|---|------|-----------------|
| 1 | `config.py` | Settings & feature flags (which NLU tiers are on). |
| 2 | `schemas.py` | The **data contracts**: `Intent`, `ActionSlots`, `TransferAction`. |
| 3 | `state.py` | The **FSM state** that is saved between turns. |
| 4 | `store.py` | Where that state is kept (in-memory here). |
| 5 | `extractor.py` | The **NLU**: language, intent, slots (regex + spaCy + FAISS). |
| 6 | `examples.py` + `embeddings.py` + `vectorstore.py` + `semantic_intents.py` | The **FAISS semantic classifier** internals. |
| 7 | `prompts.py` | Every user-facing string (EN + AR), separated from logic. |
| 8 | `core_client.py` | Optional external **pre-flight** HTTP call. |
| 9 | `engine.py` | The **FSM** that ties everything together. |
| 10 | `api.py` / `cli.py` | Two headless front-ends over the same engine. |

---

## 3. Layer by layer

### 3.1 Data contracts — `schemas.py`

Everything starts with *types*. There are three layers, on purpose:

```python
class Intent(str, Enum):
    TRANSFER_MONEY = "transfer_money"   # the worked example
    SMALL_TALK = "small_talk"           # greetings / thanks
    FALLBACK = "fallback"               # nothing actionable

class ActionSlots(BaseModel):           # LOOSE: everything Optional
    amount: Decimal | None = None
    currency: str | None = None
    recipient: str | None = None
    source_account: str | None = None
    note: str | None = None

class TransferAction(BaseModel):        # STRICT: the final contract
    amount: Decimal = Field(..., gt=0)
    currency: str = Field(...)          # validated against SUPPORTED_CURRENCIES
    recipient: str = Field(..., min_length=1)
    ...
```

**Why two models (`ActionSlots` vs `TransferAction`)?** Because during the
conversation you only have *partial* data — you can't validate what you don't
have yet. So `ActionSlots` is permissive (all `Optional`). Only at the very end
do you try to build the *strict* `TransferAction`. If that construction raises a
`ValidationError`, you know exactly which slot to ask about again.

> **Key insight:** validation is not scattered through the code — it's
> concentrated in one place (constructing the strict model). That's what makes
> the flow reliable.

`ActionSlots.first_missing(...)` is the helper that powers "ask one question at
a time":

```python
def first_missing(self, required: tuple[str, ...]) -> str | None:
    for slot in required:
        value = getattr(self, slot)
        if value is None or (isinstance(value, str) and not value.strip()):
            return slot   # this is the next question to ask
    return None           # nothing missing → ready to confirm
```

### 3.2 The saved state — `state.py`

A conversation must remember what happened on earlier turns. That memory is
`ConversationState`: the current `status` (an FSM node), the `slots` collected so
far, a `pending_slot` (the question we're waiting on), disambiguation
`candidates`, advisory `warnings`, and a `turns` counter (a safety valve against
infinite loops). It's a Pydantic model, so it serialises cleanly to whatever
store you choose.

The `status` values *are* the FSM's nodes:

```python
class ConversationStatus(str, Enum):
    COLLECTING = "collecting"          # gathering slots
    DISAMBIGUATING = "disambiguating"  # "which Ahmed?"
    CONFIRMING = "confirming"          # "shall I proceed? yes/no"
    COMPLETED = "completed"            # action emitted
    CANCELLED = "cancelled"            # user backed out
```

### 3.3 Storage — `store.py`

Trivial on purpose: an in-memory dict keyed by `session_id`. It returns *deep
copies* so callers can't mutate stored state by accident. In production you'd
swap this for Redis or SQL — the engine doesn't care, it only uses
`load()` / `save()`.

### 3.4 The NLU — `extractor.py` (the interesting part)

This file answers three questions about a piece of text:

1. **What language is it?** — `detect_language()` counts Arabic characters.
2. **What does the user want?** — `detect_intent()`.
3. **What details did they give?** — `extract_slots()`.

#### Two-tier intent detection (spaCy + FAISS, **no LLM**)

```python
def detect_intent(text: str) -> Intent:
    if settings.use_semantic_intent:
        classifier = _get_semantic()          # FAISS + embeddings
        if classifier is not None:
            intent, _conf = classifier.classify(text)
            if intent is not Intent.FALLBACK:
                return intent                  # semantic tier won
    return _keyword_intent(text)               # deterministic fallback
```

- **Tier 1 (semantic):** understands *meaning*. "could you move 200 to my
  colleague" has none of the keywords, yet FAISS finds it's most similar to the
  labelled transfer examples → `transfer_money`. (See §3.5 for how.)
- **Tier 2 (keyword):** a plain set-membership check (`send`, `transfer`, `حوّل`,
  …). Always available, even with no models installed.

The pattern to notice: **prefer the smart tier, but degrade gracefully.** If the
embedding model isn't installed, `_get_semantic()` returns `None` and you fall
back to keywords automatically. The template always runs.

#### Slot extraction (regex + spaCy NER)

```python
def _extract_recipient(text: str) -> str | None:
    nlp = _load_spacy()
    if nlp is not None:                        # spaCy PERSON entity, if available
        people = [e.text for e in nlp(text).ents if e.label_ == "PERSON"]
        if people:
            return people[0]
    match = _RECIPIENT_RE.search(text)         # else regex: "to <Name>"
    ...
```

Amount and currency are pure regex + a small word map (`dollars→USD`,
`ريال→SAR`). Arabic gets two pre-processing steps that are easy to miss but
essential:

```python
_TASHKEEL_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")  # diacritics
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")               # digits
```

Without normalising `حوّل` → `حول` and `٢٠٠` → `200`, matching fails. This is a
recurring lesson in multilingual NLU: **normalise before you match.**

### 3.5 How the FAISS semantic classifier actually works

Four small files cooperate. Follow the data:

```
examples.py            embeddings.py                 vectorstore.py
[(text, intent), …] ─▶ embed each text to a vector ─▶ store vectors + payloads
                                                        in a FAISS index
                                    ▲                          │
query text ─────────────────────────┘                          ▼
                       embed the query, then FAISS returns the k
                       nearest example vectors → vote on their intents
                                    (semantic_intents.py)
```

1. **`examples.py`** — a hand-written list of `(utterance, intent)` pairs in
   English *and* Arabic. This is your "training data" — except there is no
   training. More varied examples = better recall.

2. **`embeddings.py`** — wraps a multilingual `sentence-transformers` model.
   `encode()` turns text into **L2-normalised** vectors (unit length). Because
   they're unit length, the dot product between two vectors equals their *cosine
   similarity* (1.0 = identical meaning, 0 = unrelated). It loads lazily and, if
   unavailable, returns `None` (→ keyword fallback).

3. **`vectorstore.py`** — a thin wrapper over `faiss.IndexFlatIP` ("IP" = inner
   product). You `add()` the example vectors once, then `search(query, k)`
   returns the `k` closest with their similarity scores.

4. **`semantic_intents.py`** — the classifier. At first use it embeds every
   example and adds them to the index. To classify:

   ```python
   neighbours = self.similar(text)              # k nearest examples
   if neighbours[0].score < threshold:          # nothing close enough?
       return Intent.FALLBACK, score            # → let keywords decide
   weights = defaultdict(float)
   for n in neighbours:
       weights[n.intent] += max(n.score, 0.0)   # each neighbour votes, weighted
   best = max(weights, key=weights.get)         # the winning intent
   ```

   So classification is literally *"which labelled examples is this most similar
   to, and what were they labelled?"* — nearest-neighbour voting. No model
   weights are updated; you "teach" it by editing `examples.py`.

> **Why no LLM?** For a fixed set of intents, embedding-based retrieval is fast,
> cheap, deterministic, offline, and needs no prompt engineering. The production
> app adds an LLM only as a *last-resort fallback*; this template stops here on
> purpose.

### 3.6 Prompts — `prompts.py`

Every string the user sees lives here, keyed by `Language`. Logic never contains
copy. This is why adding Arabic was cheap and why you can retheme wording without
touching the engine.

### 3.7 Optional pre-flight — `core_client.py`

If `SVC_CORE_ENABLED=true`, the engine calls an external service before
confirming (e.g. "is there enough balance?"). It returns `warnings` (advisory)
and `blocking` (hard stops). **Crucially, network errors are treated as "no
opinion"** — a flaky dependency must not break the conversation. When disabled,
it returns `None` and the flow proceeds with no warnings.

### 3.8 The engine (FSM) — `engine.py`

This is the heart. One method, `handle()`, runs every turn:

```
load state
   │
   ├─ finished last time?  → reset and start fresh
   ├─ user said "cancel"?  → reset → CANCELLED   (checked in EVERY state)
   │
   └─ dispatch on current status:
        COLLECTING     → extract+merge slots → ask next missing slot
                         │ slots complete → resolve recipient
                         │    • many matches → DISAMBIGUATING
                         │    • one match    → lock it
                         └─ run pre-flight → CONFIRMING
        DISAMBIGUATING → interpret "2" or a name → CONFIRMING
        CONFIRMING     → "yes" → build TransferAction → COMPLETED
                         "no"  → CANCELLED
                         else  → re-ask (state preserved)
save state → return (reply, optional action)
```

Three rules the code carefully upholds — internalise these, they're the whole
point of an FSM:

1. **Never clobber a filled slot.** `_merge_slots` only writes a field that is
   still empty, so a later vague message can't erase earlier detail.
2. **One question at a time**, driven by `first_missing`.
3. **Warnings never block; only `blocking` does.** And the engine *emits* the
   action — it never executes it (`# NOTE: we never move money here`).

The disambiguation path is real (not a stub): a tiny in-memory `_DIRECTORY` has
three people named "Ahmed", so a first-name match returns several candidates and
the FSM enters `DISAMBIGUATING`. Replace `_lookup_recipients` with a real DB
query and nothing else changes.

### 3.9 Two front-ends — `api.py` and `cli.py`

Both are *thin*. They parse input, call `engine.handle(...)`, and render the
result. `api.py` is a headless FastAPI `POST /conversation/text`; `cli.py` is a
terminal REPL. They share one engine, which is why behaviour is identical across
them. **This is the payoff of keeping the engine UI-free.**

---

## 4. Trace a real conversation (end to end)

Follow `"pay for Ahmed 100"` then `"2"` then `"yes"`:

| Turn | Input | What happens | New status |
|------|-------|--------------|------------|
| 1 | `pay for Ahmed 100` | `detect_intent`→`transfer_money`; `extract_slots`→ amount=100, currency defaults SAR, recipient="Ahmed"; recipient matches **3** rows → list them | `DISAMBIGUATING` |
| 2 | `2` | `_resolve_pick` → index 1 → "Ahmed Khaled"; pre-flight runs (advisory) | `CONFIRMING` |
| 3 | `yes` | build `TransferAction` (validates); emit JSON | `COMPLETED` |

Output of turn 3:

```json
{ "intent": "transfer_money", "amount": "100", "currency": "SAR",
  "recipient": "Ahmed Khaled", "source_account": null, "note": null }
```

Try it yourself:

```bash
cd service-template
../.venv/bin/python -m service_template.cli
# then type:  pay for Ahmed 100   ↵   2   ↵   yes
```

---

## 5. Guided exercise — add a `pay_bill` case

The best way to learn the structure is to extend it. Each step maps to one file;
the `# >>> EDIT PER CASE` markers show you exactly where.

1. **`schemas.py`** — add `PAY_BILL = "pay_bill"` to `Intent`; add
   `biller` / `reference_number` to `ActionSlots`; copy `TransferAction` →
   `BillPaymentAction` with the new required fields.
2. **`state.py`** — add
   `BILL_REQUIRED_SLOTS = ("biller", "reference_number", "amount", "currency")`.
3. **`examples.py`** — add ~8 EN + AR bill utterances labelled `PAY_BILL`
   (this teaches the FAISS classifier — no retraining).
4. **`extractor.py`** — add a `_BILL_CUES` set to `_keyword_intent` (the
   fallback) and a regex to pull the reference number.
5. **`prompts.py`** — add slot prompts + a `confirm_bill(...)`.
6. **`engine.py`** — in `_collect`, dispatch to a new `_collect_bill(...)` when
   `state.intent is Intent.PAY_BILL` (there's an `# >>> EDIT PER CASE` marker
   right there).
7. **`api.py`** — allow the response to carry a `BillPaymentAction`.
8. **tests** — copy an engine test and assert the bill flow completes.

Run `../.venv/bin/python -m pytest service-template/tests -q` after each step —
red/green feedback is the fastest teacher.

---

## 6. Where reality differs (so you're not surprised)

This template is deliberately smaller than the production app. What's the same,
what's simplified:

| Concern | Template | Production (`app/…`) |
|---------|----------|----------------------|
| Intent | spaCy + FAISS (no LLM) | + LLM fallback, active-learning |
| Recipient data | in-memory `_DIRECTORY` | real DB (`app/db/directory.py`) |
| Session store | in-memory dict | pluggable persistent store |
| Pre-flight | one optional HTTP call | full Banking Core service |
| Front-ends | API + CLI | + web UI, admin, audit |

The *shapes* match, so what you learn here transfers directly.

---

## 7. Cheat-sheet

```bash
# run the tests (living documentation of expected behaviour)
.venv/bin/python -m pytest service-template/tests -q

# talk to it in the terminal
cd service-template && ../.venv/bin/python -m service_template.cli

# run the HTTP API
cd service-template && ../.venv/bin/uvicorn service_template.api:app --port 8200

# enable the full semantic tier (optional; falls back to regex without it)
../.venv/bin/python -m spacy download en_core_web_sm
# the embedding model auto-downloads from HuggingFace on first use

# turn tiers off to see the fallback behaviour
SVC_USE_SEMANTIC_INTENT=false SVC_USE_SPACY_NER=false \
  ../.venv/bin/python -m service_template.cli
```

**One-line summary:** *text → (language, intent, slots) → an FSM fills & confirms
→ a strictly-validated JSON action object — spaCy + FAISS for understanding, no
LLM, and it never executes anything.*
