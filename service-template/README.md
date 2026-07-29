# Service Template — a headless NLU "case" starter kit

A **self-contained, GUI-free template** for building a new conversational
"service" (a *case*/intent) like **transfer money** or **pay a bill**, extracted
from — and structured exactly like — the production code in this repo
(`app/conversation` + `banking-core`).

It turns a natural-language request (English **or** Arabic) into a **validated,
structured JSON action object** for a downstream system to execute. It does
**not** move money or mutate anything — it only understands, validates, and
emits an action, same as the main app.

Everything is heavily commented. Every place you need to change to add your own
case is marked with `# >>> EDIT PER CASE`.

> **New to this? Start with the [Learning Edition](LEARNING.md)** — a guided,
> teach-yourself walkthrough of every concept (intents, slots, spaCy NER,
> embeddings + FAISS, the FSM) with a glossary, an end-to-end trace, and a
> worked "add a new case" exercise.

```
text ─▶ language detect ─▶ intent detect ─▶ slot extraction
     ─▶ ask for missing slots (one at a time)
     ─▶ disambiguate (if a name matches several people)
     ─▶ external pre-flight validation (optional, advisory-only)
     ─▶ confirm (yes/no)
     ─▶ structured JSON action object
```

---

## Why it exists

This template keeps the *same architecture and naming* as the full app so you
can copy it, rename it, and fill in the `# >>> EDIT PER CASE` blocks to stand up
a new conversational case (transfer, pay, …) fast.

### NLU tiers (spaCy + FAISS, **no LLM**)

The NLU in `extractor.py` has two tiers and degrades gracefully:

1. **Semantic tier** *(preferred)* — a multilingual **sentence-transformers**
   embedder indexes the labelled examples in `examples.py` into a **FAISS**
   vector store (`vectorstore.py` + `semantic_intents.py`); an utterance's
   intent is the aggregated nearest-neighbour match. **spaCy** PERSON NER pulls
   the recipient name.
2. **Deterministic tier** *(always-available fallback)* — regex + keyword logic.

The semantic tier is used when the model is available and confident; otherwise
the deterministic tier takes over. There is **no LLM** — the production app adds
an LLM fallback (`app/orchestration.py`), which is intentionally omitted here.

Because each tier degrades to the next, the template still runs even with no
models installed (it just relies on regex/keywords). To enable the full
semantic tier, install the models once:

```bash
python -m spacy download en_core_web_sm   # spaCy NER model
# the sentence-transformers embedding model downloads automatically on first use
```

Disable either tier via env vars (`SVC_USE_SEMANTIC_INTENT=false`,
`SVC_USE_SPACY_NER=false`) — see `config.py`.

---

## Run it

From the repo root, using the repo's virtualenv (`.venv`):

```bash
# 1) Unit tests (also serve as living docs of the expected behaviour)
.venv/bin/python -m pytest service-template/tests -q

# 2) Headless REPL — talk to it in the terminal
cd service-template && ../.venv/bin/python -m service_template.cli

# 3) Headless HTTP API
cd service-template && ../.venv/bin/uvicorn service_template.api:app --port 8200
```

Drive the API with plain HTTP (keep the same `session_id` to continue a
dialogue):

```bash
curl -s localhost:8200/conversation/text -H 'content-type: application/json' \
  -d '{"text":"send 500 SAR to Ahmed","session_id":"s1"}'
# → disambiguation: "which Ahmed? 1..2..3.."
curl -s localhost:8200/conversation/text -H 'content-type: application/json' \
  -d '{"text":"2","session_id":"s1"}'
# → "I'll send 500 SAR to Ahmed Khaled. Shall I proceed? (yes/no)"
curl -s localhost:8200/conversation/text -H 'content-type: application/json' \
  -d '{"text":"yes","session_id":"s1"}'
# → completed, with the JSON "action" object in the response
```

---

## File map

| File | Role | Production analogue |
|------|------|---------------------|
| `schemas.py` | `Intent`/`Language` enums, `ActionSlots`, validated `TransferAction` | `app/schemas.py` |
| `state.py` | FSM `ConversationStatus` + persisted `ConversationState` | `app/conversation/state.py` |
| `store.py` | Session persistence (in-memory here) | `app/conversation/store.py` |
| `extractor.py` | Language/intent detection (semantic + keyword) + slot extraction | `app/nlu`, `app/orchestration.py` |
| `embeddings.py` | Multilingual sentence-transformers embedder (lazy, cached) | `app/embeddings.py` |
| `vectorstore.py` | FAISS cosine-similarity index | `app/vectorstore.py` |
| `examples.py` | Labelled example utterances that seed the FAISS index | `app/nlu/examples.py` |
| `semantic_intents.py` | FAISS nearest-neighbour intent classifier | `app/nlu/semantic_intents.py` |
| `prompts.py` | All EN/AR user-facing strings | `app/conversation/templates.py` |
| `core_client.py` | HTTP client for external pre-flight validation | `app/banking_core_client.py` |
| `engine.py` | The finite-state machine (the heart) | `app/conversation/engine.py` |
| `api.py` | Headless FastAPI endpoint | `app/conversation/router.py` |
| `cli.py` | Headless terminal REPL | — |

---

## The state machine

```
                 ┌──────────────┐
   new message ─▶│  COLLECTING  │  extract + merge slots
                 └──────┬───────┘  ask next missing slot (one at a time)
                        │ all required slots present
              ┌─────────┴──────────┐
              │ recipient ambiguous│ yes ─▶ ┌────────────────┐
              └─────────┬──────────┘        │ DISAMBIGUATING │─┐
                        │ no                 └────────────────┘ │ pick
                        ▼                                       │
                 ┌──────────────┐   (advisory pre-flight)  ◀────┘
                 │  CONFIRMING  │  show summary + warnings
                 └──────┬───────┘
                yes ─────┼───── no
                 ▼               ▼
          ┌───────────┐   ┌───────────┐
          │ COMPLETED │   │ CANCELLED │
          │ (emit JSON│   └───────────┘
          │  action)  │
          └───────────┘
```

**Invariants worth preserving when you extend it** (all enforced in `engine.py`):

1. **One question at a time** — `ActionSlots.first_missing(...)` drives prompts.
2. **Never clobber a filled slot** — `_merge_slots` only fills empties.
3. **Pre-flight is advisory** — `warnings` are shown but never block `yes`; only
   `blocking` hard-stops would. (Matches the product rule: low funds / FX just
   warn.)
4. **Cancel wins everywhere** — checked in every state before dispatch.
5. **Emit, don't execute** — the engine returns a validated action object; a
   downstream system performs the real operation.

---

## Add a new case (worked example: "pay a bill")

Say you want a `pay_bill` case that collects `biller`, `reference_number`,
`amount`, `currency`. Touch only the `# >>> EDIT PER CASE` spots:

**1. `schemas.py` — declare the intent, slots, and a validated model**

```python
class Intent(str, Enum):
    TRANSFER_MONEY = "transfer_money"
    PAY_BILL = "pay_bill"            # <-- new

class ActionSlots(BaseModel):
    ...
    biller: str | None = None        # <-- new
    reference_number: str | None = None

class BillPaymentAction(BaseModel):  # copy of TransferAction
    intent: Intent = Intent.PAY_BILL
    biller: str = Field(..., min_length=1)
    reference_number: str = Field(..., min_length=1)
    amount: Decimal = Field(..., gt=0)
    currency: str = Field(...)       # reuse the currency validator
```

**2. `state.py` — required slots for the new case**

```python
BILL_REQUIRED_SLOTS = ("biller", "reference_number", "amount", "currency")
```

**3a. `examples.py` — teach the FAISS semantic classifier the new intent**

Add ~6-10 varied English + Arabic example utterances for the new case. No
retraining or LLM needed — they are embedded and indexed at startup:

```python
INTENT_EXAMPLES += [
    ("pay my electricity bill", Intent.PAY_BILL),
    ("settle internet bill reference 4455", Intent.PAY_BILL),
    ("ادفع فاتورة الكهرباء", Intent.PAY_BILL),
    # ...
]
```

**3b. `extractor.py` — keyword fallback cues + how to pull the new slots**

The keyword classifier is the fallback for when the embedding model is absent:

```python
_BILL_CUES = {"bill", "pay", "sadad", "فاتورة", "سداد"}

def _keyword_intent(text):
    tokens = _tokens(_normalize(text))
    if tokens & _BILL_CUES:        # check the more specific case first
        return Intent.PAY_BILL
    if tokens & _TRANSFER_CUES:
        return Intent.TRANSFER_MONEY
    ...
```

**4. `prompts.py` — prompts for the new slots + a `confirm_bill(...)`**

Add `biller` / `reference_number` entries to `_SLOT_PROMPTS`, plus a
`confirm_bill(...)` and `completed(...)` string.

**5. `engine.py` — dispatch to a `_collect_bill(...)` collector**

```python
def _collect(self, state, text, lang):
    if state.intent is None:
        state.intent = extractor.detect_intent(text)
    ...
    if state.intent is Intent.PAY_BILL:      # <-- new branch
        return self._collect_bill(state, text, lang)
    return self._collect_transfer(state, text, lang)
```

`_collect_bill` is a near-copy of `_collect_transfer` using
`BILL_REQUIRED_SLOTS` and building a `BillPaymentAction` at confirmation. Bills
have no name disambiguation, so you can skip the `DISAMBIGUATING` step (or reuse
it to pick among biller candidates).

**6. `api.py` — expose the new action type** in `ConversationResponse.action`
(make it a union, e.g. `TransferAction | BillPaymentAction | None`).

**7. Tests** — copy a case from `tests/test_engine.py`.

That's it — the FSM, session store, merging, cancel handling, pre-flight, and
localisation all keep working unchanged.

---

## Wiring the external pre-flight service (optional)

By default `core_enabled` is `false` and the engine skips pre-flight. To turn it
on, point it at a service exposing `POST /preflight/transfer` returning
`{ok, warnings, blocking}` (the repo's `banking-core` service does exactly
this):

```bash
SVC_CORE_ENABLED=true \
SVC_CORE_BASE_URL=http://localhost:8100 \
SVC_CORE_API_KEY=... \
../.venv/bin/uvicorn service_template.api:app --port 8200
```

Network errors are swallowed (the client returns "no opinion") so a flaky core
service can never hard-block a conversation — pre-flight is advisory by design.

---

## What this template deliberately leaves out

- **No GUI** — API + REPL only, by request.
- **No LLM** — spaCy + FAISS semantic NLU with a regex/keyword fallback, but no LLM fallback (the app has one in `app/orchestration.py`).
- **No real DB** — an in-memory directory + in-memory session store. Swap
  `store.py` and `engine._lookup_recipients`.
- **No execution** — it emits a validated action object; it never moves money.
