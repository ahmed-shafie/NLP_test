# Bilingual Banking NLU Layer — Full Architecture

Repository: <https://github.com/ahmed-shafie/NLP_test> · revision `a9d8fa0` (`main`, after PR #40 and #41)
Scope: 180 tracked files · 647 automated tests · Python 3.12 · FastAPI · load-tested on 8 vCPU

This document is the complete architecture reference: every diagram, every technical component, the
data model, the configuration surface, the measured performance envelope, and the security posture as
it actually is in the code today.

---

## 0. Reading guide

| § | Diagram | Answers |
|---|---|---|
| 1 | — | What the system is responsible for, and what it deliberately is not |
| 2 | 1. System context | Which services exist and who talks to whom |
| 3 | 2a-2b. Internal components | What is inside the NLU service, file by file |
| 4 | 3. Turn lifecycle | What happens between the request and the reply |
| 5 | 4. NLU pipeline | How an utterance becomes intent + slots |
| 6 | 5. State machine | How a multi-turn conversation is tracked |
| 7 | 6. Determinism boundary | Exactly where a model is allowed to influence output |
| 8 | 7. Topic gate | How a customer-service question gets answered (or refused) |
| 9 | 8. Data stores | Where every byte of state lives |
| 10 | 9. Deployment | How to run it and how it scales |
| 11–14 | — | Component inventory, configuration, API, quality gates, performance, limitations |

---

## 1. Responsibility boundary

This is **the language layer in front of a bank**, not a bank. It converts free text — Saudi Arabic,
MSA, or English — into a **validated, strictly typed action object** that a separate Banking Core
executes, and turns the Core's answer back into a reply.

```
Natural-language request
        ↓
Language + intent detection
        ↓
Entity and slot extraction
        ↓
Beneficiary / biller / account resolution
        ↓
Banking Core validation
        ↓
Validated JSON action object
        ↓
Banking Core / payment API (separate service)
```

**Deliberate non-goals:** it holds no balances, moves no money, decides no limits or fees, and never
lets a model author a financial value.

### 1.1 The invariant everything else serves

> Amounts, currencies, IBANs, beneficiary names, balances, confirmations, rejection reasons and
> operation results are **deterministic and template-rendered**. A model may *select* or *nominate*;
> it may never *author*.

By construction the system cannot:

- rewrite `500` as "about five hundred";
- invent a fee, limit, FX rate, duration, balance or result;
- silently repair an invalid IBAN, or substitute a saved beneficiary for a typed IBAN;
- resolve an ambiguous beneficiary without asking;
- report a rejected Banking Core write as `completed`;
- let a customer-service answer swallow an executable transfer or bill payment.

Sections 7 and 8 show the mechanisms that enforce this sentence.

---

## 2. System context

![System context](diagrams/01-context.png)

| Service | Technology | Role |
|---|---|---|
| **NLU app** | FastAPI + Uvicorn (`app/`) | Language, intent, slots, dialogue state, resolution, replies, web UIs |
| **Banking Core** | FastAPI (`banking-core/`) | **Source of financial truth**: accounts, balances, beneficiaries, billers, pre-flight, writes, refusals |
| **PostgreSQL** | SQLAlchemy 2 + psycopg 3 | Core's accounts / beneficiaries / SADAD billers (SQLite for local dev) |
| **Redis** | redis-py | Conversation sessions (TTL 1800 s) + memory cache |
| **FAISS** | faiss-cpu, in-process | Vector index: example corpus, contacts, billers |
| **Ollama** | `qwen2.5:3b` via LiteLLM | Optional local LLM: exception-path slot recovery, Tier-B rephrasing |
| **Elasticsearch** | elasticsearch-py | Optional async audit-event shipping (`nlu-audit` index) |
| **Prometheus** | prometheus-client | `/metrics`: request counter + latency histogram per route |

Local ports used in development and load testing: app `8001`, Banking Core `8100` (container) /
`8101` (from source), Postgres `55432`, Redis `56379`, Ollama `11434`, Elasticsearch `9200`.

---

## 3. Internal components

![Components — edge and dialogue](diagrams/02a-components-dialogue.png)

![Components — understanding and platform](diagrams/02b-components-nlu.png)

### 3.1 Component inventory

| Area | Module | Responsibility |
|---|---|---|
| Edge | `app/main.py` | App assembly, routers, `/health`, `/health/ready`, `/metrics`, static UIs |
| Edge | `app/request_context.py` | `request-id` generation/propagation into logs |
| Edge | `app/middleware.py` | Body-size limit (`max_request_bytes`, default 1 MB) → HTTP 413 |
| Edge | `app/metrics.py` | Prometheus request counter + latency histogram, route-templated |
| Edge | `app/errors.py` | One error envelope: `{error: {code, message, request_id}}` |
| Edge | `app/logging_config.py` | Structured JSON logging |
| Dialogue | `app/conversation/engine.py` | Turn orchestration: restore → moderate → understand → route → fill → confirm → execute |
| Dialogue | `app/conversation/state.py` | `ConversationState`, `ConversationSlots`, `ConversationStatus`, required-slot sets |
| Dialogue | `app/conversation/store.py` | Session persistence (Redis or in-process) |
| Dialogue | `app/conversation/moderation.py` | Abuse gate with strike counting, tuned against over-blocking |
| Dialogue | `app/conversation/templates.py` | Bilingual reply tables incl. the session-opening table |
| Dialogue | `app/conversation/phrasing.py` | Two-tier phrasing + post-generation validator |
| Dialogue | `app/conversation/topic_replies.py` | 28 reviewed customer-service reply families |
| Dialogue | `app/conversation/router.py` | `POST /conversation/text`, `GET /conversation/opening` |
| Understanding | `app/orchestration.py` | Haystack pipeline wiring (6 components, diagram 4) |
| Understanding | `app/nlu/lang.py` | Language detection (script + cue based, per utterance) |
| Understanding | `app/nlu/normalize.py`, `arabic.py` | Indo-Arabic digits → ASCII, alef/hamza/teh-marbuta folding, proclitic & article stripping |
| Understanding | `app/nlu/english.py` | English-side normalisation helpers |
| Understanding | `app/nlu/entities.py` | Deterministic readers: amount, currency, IBAN, reference, biller, recipient |
| Understanding | `app/nlu/accounts.py` | IBAN ISO 13616 length + mod-97, account-type words |
| Understanding | `app/nlu/intents.py` | Keyword/rule intent fallback |
| Understanding | `app/nlu/semantic_intents.py` | Embedding + FAISS kNN intent routing and thresholds |
| Understanding | `app/nlu/corpus.py`, `examples.py` | The 31,781-row example index and its loaders |
| Understanding | `app/nlu/topic_head.py` | Trained MLP head (384→1024→28), numpy-only forward pass |
| Understanding | `app/nlu/contacts.py` | Name resolution: gazetteer + rapidfuzz + cross-lingual vectors |
| Understanding | `app/nlu/pipeline.py` | spaCy/Stanza analysis pipelines |
| Platform | `app/embeddings.py` | SentenceTransformer wrapper, single/batch encode |
| Platform | `app/vectorstore.py` | Thin generic FAISS store (`add`, `search`, payloads) |
| Platform | `app/banking_core_client.py` | Typed HTTP client: balance, pre-flight, add-beneficiary, write |
| Platform | `app/db/beneficiary.py`, `db/directory.py` | Read-only SQL beneficiary directory (exact / ambiguous / none) |
| Platform | `app/memory/` | Learned shortcuts and habit counters (`memory_brain.db`) |
| Platform | `app/active_learning/` | Review queue, reviewer decisions, index rebuild daemon |
| Platform | `app/admin/` | Connection providers, bank config, audit store, ELK shipper |
| Platform | `app/trace.py` | `BlockTracer` / `BlockTrace`: per-stage timing + decision notes |
| Platform | `app/data_loader.py` | SADAD biller catalogue + name gazetteer loading/resolution |
| UI | `app/static/assistant.html` | Assistant UI + Developer mode (Inspect panel shows `block_trace`) |
| Eval | `app/eval/`, `scripts/eval_nlu.py` | Gold-set harness and the blocking quality gate |

### 3.2 `block_trace` — the audit surface

Every stage appends a span: name, wall-clock ms, and the decision it made (or why it was skipped).
For any reply you can point at the block that produced the amount, the block that resolved the
beneficiary, and the block that refused. It is returned in the API response and rendered in the
Developer-mode Inspect panel — and it is also how the load-test bottleneck was located (§13).

---

## 4. Lifecycle of one turn

![Turn lifecycle](diagrams/03-turn-flow.png)

```
POST /conversation/text  {text, user_id, session_id?, language?}
  → middleware (request-id · size limit · metrics · audit)
  → ConversationEngine.handle()
      → memory_restore (Redis session + learned shortcuts)
      → moderation
      → NLU pipeline (diagram 4)
      → decide_action: executable request vs. customer-service question
      → slot filling / disambiguation / confirmation
      → Banking Core: balance · pre-flight · write
  → ConversationResponse {reply, state, action?, block_trace}
```

Two routing rules carry most of the safety weight:

1. **A question never starts a flow.** The corpus is deliberately skewed (31,292 of 31,781 rows are
   labelled `fallback`) so the router learns to *refuse to act* far more often than to act.
2. **An aside does not lose the flow.** "كم رصيدي؟" in the middle of a transfer is answered and the
   transfer resumes at the same missing slot.

---

## 5. NLU pipeline

![NLU pipeline](diagrams/04-nlu-pipeline.png)

Six Haystack components, wired linearly, each passing an `NLUState`:

| # | Component | Method | Output |
|---|---|---|---|
| 1 | `detect` | `LanguageDetector` | script + cue based language per utterance |
| 2 | `intent` | `IntentClassifier` | embedding → FAISS kNN vote, keyword fallback, thresholds |
| 3 | `entities` | `EntityExtractor` | amount, currency, IBAN, reference, biller, recipient — deterministic readers only |
| 4 | `contacts` | `ContactResolver` | name candidates: gazetteer + rapidfuzz (score 88, margin 5) + cross-lingual vectors |
| 5 | `beneficiary` | `BeneficiaryLookup` | SQL directory verdict: exact / ambiguous / none |
| 6 | `llm` | `LLMExceptionHandler` | runs **only** when the deterministic readers found nothing |

Notable deterministic rules inside stage 3–5:

- **Amount vs reference (PR #37).** A number the customer *priced in a currency* is an amount, even
  after the word "bill": `pay my mobily bill 100 sar` → `100.00 SAR`. An unpriced number stays a
  reference and the assistant asks for the amount — reading it as money would put a figure on the
  confirmation screen the customer never called money.
- **Fuzzy matching with a margin.** On a ~28k-name gazetteer a near-miss is usually several
  *different* people (`noura` is one edit from `nouran`, `nora`, `nour`, `nura`). If the best match
  does not lead the runner-up by 5 points, the correction is **declined** and the customer's own
  spelling is kept.
- **Arabic proclitics and the article.** `لأحمد` → `أحمد`, `الايجار` → `ايجار`, so a name or biller
  glued to a preposition is still found (PR #36).
- **IBAN.** ISO 13616 length + mod-97. Invalid is reported as invalid; never repaired, and a typed
  IBAN is never replaced by a saved beneficiary's account.

---

## 6. Dialogue state machine

![State machine](diagrams/05-state-machine.png)

`ConversationStatus` = `SELECTING · COLLECTING · DISAMBIGUATING · CONFIRMING · COMPLETED · CANCELLED ·
FAILED`.

Required slots: transfer → `amount, currency, recipient`; bill → `biller, reference_number, amount,
currency`.

`FAILED` exists precisely so that a refused write is terminal **without** claiming anything happened —
including the paths where a balance is unavailable or a beneficiary list fails (PR #37).

---

## 7. Determinism boundary

![Determinism boundary](diagrams/06-determinism.png)

| Tier | Content | Who writes it |
|---|---|---|
| **A — deterministic** | amounts, currencies, IBANs, names, balances, confirmations, rejection reasons, results | fixed templates only |
| **B — conversational** | greetings, the opening line, explanations, clarifying questions, apologies | may be varied or model-rephrased |
| **Validator** | runs after generation | rejects generated text containing any digit, name or IBAN not present **verbatim** in the source object, and restores the fixed template |

Safety comes from a programmatic check *after* the model, not from prompt discipline. The trained
topic head fits the same rule: it **selects one of 28 stored replies** — it does not write text.

---

## 8. Customer-service answer gate

![Topic gate](diagrams/07-topic-gate.png)

**Problem.** Retrieval cannot state confidence — cosine similarity is a distance. A single voting bar
must be set high enough for the worst subject, and every other subject pays for it. Result before
PR #40: **85% of Arabic service questions received the generic "transfer or bill?" menu.**

**Design (PR #40, merged).**

- **No extra encode, no extra model.** The head consumes the query vector the index already computed;
  the added cost is one 384×1024 matmul.
- **It predicts the answer, not the raw topic.** 28 classes = reviewed reply families, not the 77 raw
  subjects: two subjects sharing one reply cannot mislead anyone.
- **An explicit `""` no-answer class**, trained on the executable rows (transfer, bill, balance …), so
  broader answering still cannot swallow a transfer request.
- **Numpy-only at runtime.** Weights load with `allow_pickle=False`; the forward pass is ~6 lines. No
  scikit-learn and no pickle on the customer path (sklearn is a *training* dependency of
  `scripts/train_topic_classifier.py`).
- **Fails safe.** Missing file, dimension mismatch, or weights stamped for a different embedding model
  ⇒ head disabled, retrieval alone still answers. Kill switch `NLU_TOPIC_HEAD_ENABLED=false`.
- **Three conditions before it may speak:** `p ≥ 0.999`, best retrieved score `≥ 0.80`, and the
  retrieved majority must agree on the same reply.

Artefact: `app/nlu/data/topic_head.npz`, W1 384×1024 + W2 1024×28, float32, **1.6 MB**.

**Measured improvement** on held-out slices that are *not* in the index ("answered" = contextual
answer instead of the generic menu; "wrong" = answered about a different subject):

| Slice | Retrieval only | + trained head | Delta |
|---|---|---|---|
| **Saudi Arabic** (3,526 questions) | 19.1% answered · 2 wrong (0.30%) | **28.9% · 8 wrong (0.78%)** | **+9.8 pts (+51% relative)** |
| **English** (Banking77 test, 3,079) | 39.4% · 12 wrong (0.99%) | **50.3% · 14 wrong (0.90%)** | **+10.9 pts, error rate down** |
| All dialects (7,667) | 14.9% · 0.79% | 23.3% · 1.40% | includes Moroccan/Tunisian, not indexed |

Head accuracy: 85% English · 74% Saudi · 61% Moroccan · 40% Tunisian. Threshold sensitivity — why
0.999 and not 0.99: at 0.99 Arabic coverage reaches 37.7% but at **2.3% wrong**, roughly triple the
error for 9 more points. Rejected.

### 8.1 Models actually in use

| Layer | Model / method | Notes |
|---|---|---|
| Sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d) | **Multilingual on purpose**: `أحمد` and `Ahmed` share one space, so one index serves both languages |
| Vector search | FAISS, in-process | Example corpus, contacts, billers |
| Topic classification | trained MLP head, numpy runtime | §8 |
| Arabic morphology | Stanza (`ar`) | analysis only, not representation |
| English parsing | spaCy `en_core_web_sm` | analysis only |
| Fuzzy matching | rapidfuzz | names/billers, with the margin rule |
| LLM | `ollama/qwen2.5:3b` via LiteLLM | exception path + Tier B only |

**Not used:** AraBERT / MARBERT (Arabic-only ⇒ would break cross-lingual matching) and
`all-mpnet-base-v2` — measured, not shipped:

| | Shipped gate | (A) head on MiniLM — **shipped** | (B) mpnet encoder + head |
|---|---|---|---|
| English | 39.4% · 1.0% wrong | 50.3% · 0.9% | **65.7% · 0.9%** |
| Saudi | 19.1% · 0.3% | **28.9% · 0.8%** | 31.0% · **1.1%** |
| Runtime cost | — | **zero** (same vector) | +~20 s startup, +230 MB RAM |
| Risk | — | retrieval untouched | **all retrieval scores shift** ⇒ every threshold recalibrated |

(B) buys 2 points of Saudi coverage for ~3× the Saudi error rate and forces a full recalibration of a
working system. The Arabic ceiling is a **data** problem: the weak slices are Moroccan and Tunisian
dialects absent from the index.

---

## 9. Data stores and artefacts

![Data stores](diagrams/08-data.png)

| Store | Owner | Contents |
|---|---|---|
| `accounts`, `beneficiaries`, `billers` (PostgreSQL) | Banking Core | balances, IBANs, SADAD codes — the financial truth |
| Redis | NLU app | conversation sessions (`session_ttl_seconds = 1800`), memory cache |
| `memory_brain.db` | NLU app | learned shortcuts ("rent" → landlord) and habit counters |
| `active_learning.db` | NLU app | low-confidence utterances and reviewer decisions |
| `app_config.db` | NLU app | connection providers, bank config, audit rows |
| `app/nlu/data/example_corpus.jsonl` | repo artefact | 31,781 rows — 31,450 AR (MSA + Palestinian, Saudi, Moroccan, Tunisian) / 331 EN; by label: 31,292 `fallback`, 153 `transfer_money`, 131 `pay_bill`, 79 `balance_inquiry`, 47 `add_beneficiary`, 47 `small_talk`, 32 `list_beneficiaries` |
| `app/nlu/data/topic_head.npz` | repo artefact | trained head weights, stamped with the embedding-model name |
| gazetteers | repo artefact | ~28k names, SADAD biller catalogue |

**Licensing note:** ArBanking77-derived rows are inside the shipped index — its licence must be
cleared before commercial use. English Banking77 is Apache-2.0.

---

## 10. Deployment and scaling

![Deployment](diagrams/09-deployment.png)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python -c "import stanza; stanza.download('ar')"

docker compose up -d postgres redis banking-core

NLU_PRELOAD_MODELS=true \
NLU_BANKING_CORE_ENABLED=true NLU_BANKING_CORE_URL=http://localhost:8100 \
NLU_BENEFICIARY_LOOKUP_ENABLED=true \
NLU_BENEFICIARY_DB_URL=postgresql+psycopg://banking:banking@localhost:55432/banking_core \
NLU_SESSION_BACKEND=redis NLU_REDIS_URL=redis://localhost:56379/0 \
NLU_MEMORY_CACHE_BACKEND=redis \
OMP_NUM_THREADS=1 \
.venv/bin/uvicorn app.main:app --port 8001 --workers 8
# UI: http://localhost:8001/assistant?dev=1   ·   Swagger: /docs
```

Operational facts that matter in production:

- **Stateless app.** Sessions live in Redis ⇒ N hosts ≈ N × capacity behind any load balancer.
- **Inference is in-process.** Throughput is CPU-bound on the encoder; scale with **workers**, and set
  `OMP_NUM_THREADS=1` (8 workers × 1 thread beat 2 workers × 4 threads by **+76%**).
- **Startup cost is real.** Each worker independently encodes the 31,781-row corpus: **~4 min 40 s**
  and **~600 MB RSS per worker**. Auto-scaling a new host therefore takes ~5 minutes to serve.
  Persisting the vectors/FAISS index to disk would make startup seconds — the top open optimisation.

---

## 11. Public API

| Endpoint | Purpose |
|---|---|
| `POST /conversation/text` (+ `/v1/…`) | Main dialogue turn |
| `GET /conversation/opening?language=ar\|en` | Line shown before the first message (PR #41) |
| `POST /nlu/*` (+ `/v1`) | Stateless intent/entity inspection |
| `/memory/*` | Learned shortcuts and habits |
| `/active-learning/*` | Review queue, approve/reject, index rebuild |
| `/admin/*` | Connection providers, Banking Core config, audit events/stats |
| `GET /health`, `/health/ready` | Liveness / readiness (models + dependencies) |
| `GET /metrics` | Prometheus |
| `/assistant?dev=1`, `/brain`, `/brain/monitor`, `/admin`, `/active-learning` | Web UIs |

Banking Core (separate service, API-key protected): `GET /health`,
`POST /accounts/balance`, `POST /preflight/transfer`, `POST /preflight/bill`,
`POST /beneficiary/add`.

`Intent` = `transfer_money · pay_bill · balance_inquiry · list_beneficiaries · add_beneficiary ·
small_talk · inappropriate · fallback`. The emitted action object is versioned and strictly typed
(Pydantic); every financial field is copied verbatim from the customer's text (amount, currency,
reference) or from Banking Core (account, balance, status) — never generated.

### 11.1 Session opening (PR #41)

`GET /conversation/opening` returns one of a reviewed bilingual variant table
(`templates.py::_OPENING`), registered in the **conversational** tier. The UI calls it on load and on
**New chat**; a language toggle *before* the first message **replaces** the bubble rather than
appending. It is **display-only** and tested as such: no `session_id`, no slots, no `block_trace`, no
flow started, and a unit test asserts **no variant contains a digit** — a greeting precedes any
Banking Core call, so a number in it could only have been invented.

---

## 12. Configuration surface (selected, `app/config.py`, prefix `NLU_`)

| Setting | Default | Effect |
|---|---|---|
| `embedding_model` | `paraphrase-multilingual-MiniLM-L12-v2` | shared encoder for every vector path |
| `preload_models` | `false` | load models at startup instead of on the first request |
| `semantic_intent_threshold` / `semantic_route_threshold` | `0.45` / `0.80` | intent acceptance and routing bars |
| `topic_replies_enabled`, `topic_reply_top_k`, `topic_reply_threshold`, `topic_reply_agreement` | `true`, `10`, `0.94`, `0.8` | retrieval voting gate for service questions |
| `topic_reply_top_k_en`, `topic_reply_unanimous_threshold_en` | `7`, `0.78` | English-specific gate |
| `topic_head_enabled`, `topic_head_threshold`, `topic_head_score_floor` | `true`, `0.999`, `0.80` | trained head and its two numeric conditions |
| `name_match_score`, `name_match_margin` | `88`, `5` | fuzzy name acceptance and the runner-up margin |
| `biller_fuzzy_min_ratio`, `biller_match_threshold` | `90`, `0.55` | biller resolution |
| `llm_enabled`, `llm_model`, `llm_temperature` | `true`, `ollama/qwen2.5:3b`, `0.0` | exception-path LLM |
| `reply_variation_enabled`, `reply_rewrite_enabled` | `true`, `false` | Tier-B variation / model rewriting |
| `banking_core_enabled`, `banking_core_url`, `banking_core_api_key` | `false`, `:8100`, — | financial source of truth |
| `beneficiary_lookup_enabled`, `beneficiary_db_url` | `false`, SQLite | read-only beneficiary directory |
| `session_backend`, `redis_url`, `session_ttl_seconds` | `memory`, `:6379`, `1800` | dialogue sessions |
| `moderation_enabled`, `moderation_max_strikes` | `true`, `3` | abuse gate |
| `audit_enabled`, `audit_sink`, `elk_enabled` | `true`, `elasticsearch`, `true` | audit shipping |
| `max_request_bytes`, `log_json`, `metrics_enabled` | `1_000_000`, `true`, `true` | edge hardening and observability |

---

## 13. Quality gates and measured performance

| Gate | What it enforces |
|---|---|
| **Gold set** — 303 hand-labelled bilingual cases (`scripts/eval_nlu.py`) | intent accuracy, per-slot F1, 0 over-blocks, 0 wrong flow starts; tagged by phenomenon (colloquial, typo, arabic_digits, attached_lam, amount_last, codeswitch …). **Blocking.** |
| **pytest** — 647 tests | flows, slots, identity, moderation, phrasing, trace, Banking Core client, topic head |
| **Held-out slices** | ArBanking77 dialect splits (never indexed) + Banking77 English test split |
| **CI** | `quality` (pytest + ruff format + ruff check + mypy) and `security-audit` |
| **Runtime UI testing** | scripted browser runs against a real Banking Core, recorded, demo directory verified unchanged |

Current state on `main`: gold `intent 1.000 (303/303)`, all slot F1 `1.000`, `GATE PASSED`; 647 tests
pass; ruff and mypy clean.

### 13.1 Load test (8 vCPU / ~31 GB, LLM disabled, real Banking Core, k6)

| Path | Peak throughput | p95 there | Failures |
|---|---|---|---|
| Service question (encode + FAISS + head) | **55 req/s** @ 8 VUs | 210 ms | 0% |
| Executable request (transfer / balance / bill) | **165 req/s** @ 16 VUs | 290 ms | 0% |
| `GET /conversation/opening` (template only) | **453 req/s** | 5.5 ms | 0% |

Per-stage cost from `block_trace` (single request, no load): `intent_classification` **33 ms**
(encode), `topic_answer` **33 ms** (a *second* encode of the same sentence), `contact_resolution`
**25 ms** (encode of the name) — everything else, including Redis, Postgres, Banking Core HTTP and
template rendering, sums to **< 3 ms**.

Capacity, on one host, mixed load (30% service questions / 70% executable), 50% headroom, one message
per active user every 25 s: **≈1,250 concurrent conversations**, **≈200k daily active users**; at
saturation (p95 ≈ 600 ms) ≈2,600 concurrent. These are estimates from measurement, not guarantees:
they scale with hardware, worker count, request mix, message frequency, peak-hour distribution and
whether the LLM is enabled.

**Two known optimisations, both unimplemented:**

1. **The same sentence is encoded twice per turn** (intent, then topic) — memoising one vector per
   turn roughly halves the service-question path and should raise mixed capacity by ~50% with no
   behavioural change.
2. **Every worker rebuilds the index at startup** — persisting embeddings/FAISS to disk turns ~4 min
   40 s into seconds and cuts per-worker memory.

---

## 14. Security posture — as it is today

Stated plainly, because the gap matters for a bank:

- **The NLU app itself has no authentication, no CORS policy and no rate limiting.** Its edge
  protections are a `request-id`, a body-size limit (413 above 1 MB), structured logs, Prometheus
  metrics and optional audit shipping. It is designed to sit **behind** an authenticated API gateway
  that supplies `user_id`; do not expose it directly to the internet.
- **Banking Core is API-key protected** (`banking_core_api_key`), and it — not the language layer —
  owns every financial decision.
- **No financial value is ever model-authored** (§7), and no secret or credential is committed;
  connections come from environment variables or the admin connection store.
- **CI runs a `security-audit` job** on every PR.

Recommended before production: gateway authentication (mTLS or OAuth2) in front of the app, per-user
rate limiting, PII-aware log redaction, and a licence clearance for the ArBanking77-derived corpus
rows.

---

## 15. Known limitations (measured, not hidden)

1. **Arabic service coverage is 28.9%** — better than 19.1%, still the biggest gap; the bottleneck is
   dialect coverage in the index, not classifier capacity.
2. `بطاقتي ما تشتغل` still gets the menu (head probability 0.9735 < 0.999) while the English
   equivalent is answered. Lowering the bar raises coverage *and* error — needs a sweep, not a hunch.
3. `freeze my card` gets the menu: the head is certain but the retrieved majority names another reply
   and the agreement rule refuses. That rule is what holds error near 1%.
4. `can i cancel a transfer i made yesterday` is read as a cancel command by the cancel vocabulary
   although it is a question and nothing is pending. Pre-existing; needs its own fix.
5. Moroccan/Tunisian accuracy (61% / 40%) is not production-grade — outside the agreed scope
   (Saudi + English).
6. Developer **Inspect** panel does not reset on *New chat* (dev-only cosmetic).
7. Arabic slot extraction is rule-based: robust on the gold set (F1 1.000) but every new phrasing
   costs a new rule — the reason a trained Arabic NER is next on the roadmap.
8. No auth/rate limiting at the app edge (§14), and ~5-minute worker startup (§13).

---

## 16. Roadmap (agreed order)

1. **Recalibrate the Arabic head threshold + agreement rule alone** — cheap, no model change, targets
   limitations 2 and 3, decided by measurement.
2. **Performance PR** — per-turn vector memoisation + persisted FAISS index (§13.1).
3. **Arabic NER for recipient/amount** — replaces brittle rule accretion. Hard constraint: output is
   **candidates only**; the directory still decides identity and amounts are read deterministically.
4. **mpnet encoder swap** — its own PR with a full recalibration sweep, or preceded by adding
   Saudi/Gulf rows so coverage rises without error rising.
5. **LLM phrasing for Tier B** — already architected (validator in place); remaining work is enabling
   and measuring it.
6. **Neo4j** — deferred. Useful for ranking ambiguous beneficiaries by real relationship
   (`Customer -[SENT_TO {count, last, avg}]-> Beneficiary`) and habit shortcuts; **not** as a financial
   source of truth. The same ranking is achievable in Postgres from a transfer counter without a
   fourth service.
