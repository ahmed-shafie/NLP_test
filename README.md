# Banking NLU Brain

An NLU (Natural Language Understanding) microservice for a mobile-banking AI assistant, focused on **money-transfer** intent.

## Features

- **Bilingual** — handles English (spaCy) and Arabic (Stanza) natively.
- **Semantic intent classification** — a FAISS vector index over multilingual sentence embeddings classifies intent by nearest labeled examples; falls back to keyword matching.
- **Cross-lingual contact matching** — resolves a spoken/typed recipient to an address-book contact across scripts (e.g. `Ahmed` → `أحمد حسن`); falls back to fuzzy string matching.
- **Entity/slot extraction** — extracts `amount`, `currency`, `recipient`, `source_account`, `note`.
- **Strict validation** — Pydantic schemas enforce business rules and produce human-friendly prompts for missing/invalid slots.
- **Haystack orchestration** — the NLU steps are wired as Haystack components in a `Pipeline`, so the flow is explicit and extensible.
- **Configurable beneficiary lookup** — when a request includes an `account_number`, the destination beneficiary is resolved by a SQL lookup against a database provider (PostgreSQL, Oracle, SQL Server, Impala, Hive, SQLite, …). The connection, query, and column mapping are all set from config, so switching providers needs no code changes.
- **LLM exception handling** — a LiteLLM-backed safety net (local LLM via Ollama by default) fires only when the deterministic path falls short (e.g. an unparsed Arabic word-amount like "ألف", or a fallback intent), filling missing slots and proposing a clarification.
- **Resource connection GUI** — a browser admin page (`/admin`) to add, edit, **test**, and **activate** external database & datalake connections without touching code. The active connection drives the beneficiary lookup.
- **Audit log monitor + ELK** — every system action (HTTP request + domain events) is recorded and shipped to **Elasticsearch/Logstash/Kibana**, with a built-in observability dashboard (`/admin/audit`) of charts and reports that falls back to the local store when ELK is down.
- **Multi-turn conversation** — a slot-filling dialogue engine (`/conversation/text`) drives a transfer to confirmation over several turns, asking targeted follow-ups in English or Arabic; sessions persist in Redis (in-memory fallback).
- **Memory Brain** — per-user **habits** (favourite recipient, usual currency, default source account, recent amounts — learned automatically) and **shortcuts** (named transfer templates like `rent`), backed by SQL (durable) + a Redis cache. Supplying a `user_id` lets the assistant pre-fill slots and expand shortcuts.
- **Graceful degradation** — runs in regex/fuzzy-only mode when NLP/embedding models are not downloaded, skips the LLM entirely when no LLM server is reachable, and keeps auditing locally when ELK is unavailable.

## Tech Stack

| Layer | Tool |
|-------|------|
| API | FastAPI |
| Validation | Pydantic v2 |
| English NLU | spaCy |
| Arabic NLU | Stanza |
| Embeddings | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector DB | FAISS (in-process) |
| Orchestration | Haystack (`haystack-ai`) |
| Beneficiary DB | SQLAlchemy (any dialect/provider) |
| LLM gateway | LiteLLM → local Ollama (`qwen2.5:3b`) |
| Config store | SQLAlchemy (SQLite by default) |
| Observability | Elasticsearch + Logstash + Kibana (ELK) + Chart.js dashboard |

## Quick Start

```bash
# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLP models (optional — service works without them via regex/fuzzy)
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('ar')"
# The multilingual embedding model auto-downloads on first use (~470MB).

# (optional) local LLM for the exception handler — offline, no API key
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b   # ~2GB, strong Arabic; LiteLLM targets ollama/qwen2.5:3b

# Run the server
uvicorn app.main:app --reload --port 8000
```

The LLM handler is optional and configurable via env vars (prefix `NLU_`):
`NLU_LLM_ENABLED` (default `true`), `NLU_LLM_MODEL` (default `ollama/qwen2.5:3b`),
`NLU_LLM_API_BASE` (default `http://localhost:11434`). Set `NLU_LLM_ENABLED=false`
to disable it; if the server is unreachable the pipeline degrades automatically.

## Beneficiary account lookup (configurable provider)

When `POST /nlu/parse` receives an `account_number`, the pipeline resolves the
destination **beneficiary** by querying a SQL database before building the response.
The provider is selected purely by the SQLAlchemy URL, and the query and result
column mapping are configuration — so you can switch between PostgreSQL, Oracle,
SQL Server, Impala, Hive, SQLite, etc. **without code changes**:

```bash
NLU_DB_ENABLED=true
NLU_DB_URL="postgresql+psycopg://user:pass@host:5432/bank"
NLU_DB_QUERY="SELECT id, name, account, bank FROM beneficiaries WHERE account = :account_number"
NLU_DB_ACCOUNT_PARAM="account_number"           # bind param name in the query
NLU_DB_COLUMN_MAP='{"id":"id","name":"name","account":"account","bank":"bank"}'
```

The database drivers are optional installs (only the one you use is needed):

| Provider | SQLAlchemy URL prefix | Driver to `pip install` |
|----------|----------------------|--------------------------|
| PostgreSQL | `postgresql+psycopg://` | `psycopg[binary]` |
| Oracle | `oracle+oracledb://` | `oracledb` |
| SQL Server | `mssql+pyodbc://` | `pyodbc` (+ ODBC driver) |
| MySQL/MariaDB | `mysql+pymysql://` | `pymysql` |
| Impala | `impala://` | `impyla` |
| Hive | `hive://` | `pyhive[hive]` |
| SQLite (dev/test) | `sqlite:///` | built in |

If the account is **found**, the beneficiary is returned in `resolved_beneficiary`
(`beneficiary_source="database"`) and fills the recipient slot. If it is **not found**
(or the DB is disabled/unreachable), the request is delegated to the LiteLLM handler,
which processes the query and generates a response (`beneficiary_source="llm"`). DB
lookup is skipped entirely when no `account_number` is supplied.

## Resource connection GUI (`/admin`)

Instead of editing `NLU_DB_*` env vars by hand, you can manage connections from a
browser at <http://localhost:8000/admin>:

- **Add / edit** a connection: name, provider (preset fills the URL template), the
  SQLAlchemy URL, lookup query, account bind-parameter, and column map.
- **Test connection** — opens the connection and (optionally) runs the lookup for a
  sample account, reporting elapsed time and the result columns.
- **Activate** — marks one connection as the active beneficiary provider. The
  beneficiary lookup uses the **active stored connection** first, and only falls back
  to the `NLU_DB_*` env settings when none is active (toggle with
  `NLU_USE_STORED_CONNECTION`).

Connections are persisted in a local SQLAlchemy store (`NLU_ADMIN_STORE_URL`,
default `sqlite:///./app_config.db`). Datalake engines (Impala, Hive, Trino, Presto)
appear alongside the relational databases as presets.

## Audit log monitor + ELK observability (`/admin/audit`)

Every system action is audited: an HTTP middleware records each request (method,
path, status, latency, client, request id, actor) and domain code records events such
as `nlu.parse`, `connection.test`, and `connection.activate`. Events are persisted to
the local store **and** shipped to ELK.

```bash
# Bring up Elasticsearch + Logstash + Kibana locally
docker compose -f deploy/elk/docker-compose.yml up -d
# Install the index template + import Kibana data view, visualizations & dashboard
bash deploy/elk/provision.sh
```

Relevant settings (prefix `NLU_`):

| Setting | Default | Purpose |
|---------|---------|---------|
| `NLU_AUDIT_ENABLED` | `true` | Record audit events at all. |
| `NLU_AUDIT_SINK` | `elasticsearch` | `elasticsearch` (direct), `logstash` (TCP), or `none`. |
| `NLU_ELK_ENABLED` | `true` | Ship to Elasticsearch / power dashboard aggregations. |
| `NLU_ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch endpoint. |
| `NLU_ELK_INDEX` | `nlu-audit` | Target index. |
| `NLU_LOGSTASH_HOST` / `NLU_LOGSTASH_PORT` | `localhost` / `50000` | Logstash TCP (json_lines). |

The in-app dashboard at <http://localhost:8000/admin/audit> shows charts and reports
(actions over time, status-code split, by category, top actions, latency, recent
events). It reads aggregations from **Elasticsearch** when reachable and **falls back
to the local store** otherwise, so observability keeps working even with ELK down. The
same data is available in Kibana at <http://localhost:5601/app/dashboards> ("Banking
NLU — Audit Observability").

## Orchestration & LLM fallback

```
text ─▶ LanguageDetector ─▶ IntentClassifier ─▶ EntityExtractor ─▶ ContactResolver ─▶ BeneficiaryLookup ─▶ LLMExceptionHandler
                                                                                          │ (account_number)        │ (only if intent=fallback,
                                                                                          ▼                         │  slots missing, or the
                                                                            DB provider (SQLAlchemy)                │  account was not found)
                                                                                                                    ▼
                                                                                                       LiteLLM → local LLM
```

The Haystack pipeline (`app/orchestration.py`) threads a shared `state` through each
component. The final `LLMExceptionHandler` runs the LiteLLM call (`app/llm.py`) only
when the deterministic result is incomplete, then merges any slots it recovered
(never overwriting values the rules already found) and attaches a `clarification`.
The response gains two fields: `llm_assisted` (bool) and `clarification` (string).

## Web simulator

Open <http://localhost:8000/> in a browser for a built-in simulation & testing
page. It lets you try utterances in English or Arabic (with one-click samples),
shows the parsed intent, `intent_source`, confidence, extracted entities and the
resolved contact, and has panels for `/nlu/similar` and `/contacts/resolve`. The
raw JSON for each call is available under a collapsible section. Interactive
OpenAPI docs remain at <http://localhost:8000/docs>.

## How semantic NLU works

```
utterance ── embed ──▶ FAISS index of labeled examples ──▶ top-k neighbours ──▶ aggregate per intent
recipient ── embed ──▶ FAISS index of contact names    ──▶ best match (cosine ≥ threshold)
```

Labeled intent examples live in `app/nlu/examples.py` (English + Arabic). The demo
address book lives in `app.config.DEMO_CONTACTS`. Because the embedding model is
multilingual, Arabic and English map into a shared vector space, so a transfer
phrased in either language lands near the same examples and `Ahmed`/`أحمد` resolve
to the same contact. Set `NLU_USE_SEMANTIC_INTENT=false` to force the keyword path.

### Curated example corpus

Alongside the built-in examples the index loads `app/nlu/data/example_corpus.jsonl`
— 31,781 bilingual, multi-dialect utterances (MSA, Palestinian, Gulf, Moroccan,
Tunisian, Egyptian, Levantine, English). Most of them are banking *customer-service*
questions carrying the `fallback` intent, which is what teaches the assistant to
answer "لماذا خُصمت رسوم؟" as a question instead of opening a transfer: measured on
7,667 held-out complaints in dialects absent from the index, wrongly opening a money
flow drops from **11.4% to 1.8%**, with gold intent accuracy at **1.000**. Set
`NLU_EXAMPLE_CORPUS_ENABLED=false` to index the built-ins only; a missing corpus file
degrades to the same thing. Rebuild it with:

```bash
python -m scripts.build_example_corpus --csv path/to/banking_nlu_vector_db_v08_final.csv
```

> **Licence — PoC use only.** 39,083 of the source rows come from
> [ArBanking77](https://sina.birzeit.edu/arbanking77/) (SinaLab, Birzeit University).
> The dataset card states no licence, so this corpus ships for research and
> demonstration only. Confirm terms with SinaLab before any commercial use.
> `research/vector_db_v08/README.md` documents the cleaning and the measurements.

### Precomputed example vectors

Embedding those 31,893 examples takes ~170s of CPU and happens in every process
that builds the index — each worker, each test session, each active-learning
rebuild. The vectors are written once to `app/nlu/data/semantic_vectors.npy`
(49 MB, git-ignored) and read back on later builds, which takes milliseconds.
The cache is keyed by the example texts, the model name and its dimension, so a
corpus edit or an encoder swap re-encodes instead of serving stale vectors, and
an unreadable file degrades to encoding. Build it ahead of time (e.g. as a
Docker build step, after the embedding model is baked in):

```bash
python -m scripts.build_semantic_vectors
```

`NLU_SEMANTIC_VECTOR_CACHE_ENABLED=false` disables it;
`NLU_SEMANTIC_VECTOR_CACHE_WRITE=false` reads a baked file without writing
(read-only filesystems). An active-learning rebuild now only encodes the rows
that were approved, not the whole corpus.

### Answering a refused question in its own context

Every corpus row carries the *subject* it was collected under ("charged twice",
"card payment reversed", "lost or stolen card"), so a question the assistant cannot
execute is answered about that subject instead of with the generic
"transfer or bill?" menu. The answers live in `app/conversation/topic_replies.py` —
hand-written per subject, bilingual, and free of any invented fee, duration, limit
or policy; a test rejects a reply that mentions one. They never touch money: a
topical answer is only reachable once no action was recognised, and it states what
the assistant can actually do (show recent operations, hand over to support).

The risk is answering about the *wrong* subject, so the gate is strict — 8 of the 10
retrieved rows must name the same subject, at high similarity.
`research/vector_db_v08/topic_gate_sweep.py` calibrates it against the gold topics
of the held-out slice: the shipped point answers **14.9%** of the questions in
context with **0.8%** of those answers about the wrong subject. Answering at
*family* level when the subjects disagree was measured and dropped — it reaches
~37% coverage but ~15% wrong answers, because "where is the card I ordered" and
"my card was stolen" share a family and not a question.

Two thirds of the wrong answers that remained were pairs of subjects retrieval
cannot separate (1.4% → 0.8% at the same coverage). Two kinds of fix, both
deterministic: subjects whose *answer* is the same no longer have two answers to
swap (the exchange rate and its fee; a blocked card and a blocked PIN), and the
question's own words settle the card cases where the answers must differ — a card
that "doesn't work" is never answered as a theft, and "freeze my card" always is.

An English question meets its own gate. The index is 98.9% Arabic, so English is
answered by *cross-lingual* retrieval, which scores lower than the same-language
retrieval those numbers come from: "my card is not working" retrieves
"البطاقة لا تعمل" nine times out of ten and peaks at 0.9213, under a bar set at
0.94. `research/vector_db_v08/topic_gate_english.py` calibrates the English gate
against the **English Banking77 test split** — ArBanking77 is its translation, so
the 77 subjects are the same and none of that split is indexed. A narrower window
(7 rows) with a higher unanimity bar (0.78) answers **39.4%** of those questions
with **1.0%** about the wrong subject, against 34.6% / 1.7% for the Arabic gate.
Both directions improved, so nothing was traded away; the Arabic gate is unchanged,
where the same window costs twice the wrong answers.

## API Endpoints

### `POST /nlu/parse`

Parse a user utterance into an intent and extracted entities.

**Request:** (`account_number` is optional; include it to resolve the beneficiary from the database)
```json
{"text": "transfer 500 dollars to John", "account_number": "EG1003"}
```

**Response:**
```json
{
  "text": "transfer 500 dollars to John",
  "language": "en",
  "intent": "transfer_money",
  "confidence": 0.94,
  "intent_source": "semantic",
  "entities": {
    "amount": "500",
    "currency": "USD",
    "recipient": "Sara Adel",
    "source_account": null,
    "note": null
  },
  "resolved_recipient": {
    "contact": {"id": "c5", "name": "Sara Adel", "account": "EG1003"},
    "score": 0.71
  },
  "resolved_beneficiary": {
    "id": "b1", "name": "Sara Adel", "account": "EG1003", "bank": "CIB"
  },
  "beneficiary_source": "database",
  "llm_assisted": false,
  "clarification": null
}
```

When the account number is **not found**, `resolved_beneficiary` is `null`,
`beneficiary_source` is `"llm"`, and `clarification` holds the LLM-generated reply.

### `GET /nlu/similar`

Inspect the nearest labeled example utterances for a query (semantic debug/eval).

`GET /nlu/similar?text=send money to Sara&k=3` →
```json
[
  {"text": "send money to my friend Sara", "intent": "transfer_money", "score": 0.82},
  {"text": "can you send 50 to Mohamed", "intent": "transfer_money", "score": 0.74}
]
```

### `POST /contacts/resolve`

Resolve a recipient name to an address-book contact (cross-lingual). Pass your own
`contacts` or omit to use the demo address book.

**Request:**
```json
{"name": "Ahmed", "contacts": [{"id": "1", "name": "أحمد حسن", "account": "A1"}]}
```

**Response:**
```json
{"matched": {"contact": {"id": "1", "name": "أحمد حسن", "account": "A1"}, "score": 0.85}, "candidates": [...]}
```

### `POST /transfer/validate`

Validate gathered slots and return a ready transfer or follow-up prompts.

**Request:**
```json
{"amount": 500, "currency": "USD", "recipient": "John"}
```

**Response (success):**
```json
{"valid": true, "transfer": {"amount": "500", "currency": "USD", "recipient": "John"}, "missing": [], "errors": []}
```

**Response (missing slots):**
```json
{"valid": false, "transfer": null, "missing": ["recipient"], "errors": [...]}
```

### `GET /health`

Returns `{"status": "ok", "version": "0.1.0"}`.

## Error handling & request limits

- All endpoints are open — there is no authentication, rate limiting, or CORS layer.
- Every error returns a uniform envelope: `{"error": {"code", "message", "request_id"}}`.
- A request body-size limit (`NLU_MAX_REQUEST_BYTES`) is applied via middleware.

See `.env.example` for the full configuration reference.

## Deployment (Docker)

The container ships the **web client** — no app install needed. Bring up the whole
stack with one command and open the assistant in any browser (desktop or phone):

```bash
docker compose up --build
```

Then open:

- **http://localhost:8000/** — the simulator (parse / similar / resolve).
- **http://localhost:8000/docs** — OpenAPI/Swagger (text conversation + Memory Brain).

**Use it from your phone:** make sure the phone is on the same Wi-Fi, find the host's
LAN IP (`ipconfig` / `ip addr`, e.g. `192.168.1.20`), then browse to
`http://192.168.1.20:8000/`.

The spaCy English model is baked in; the multilingual sentence-transformer and Stanza
Arabic model download on first use into a persisted `model_cache` volume (so they're
fetched only once). The same volume holds the Memory Brain SQLite store and Redis backs
both conversation sessions and the memory cache.

```bash
# Optional: also run the local LLM exception handler (Ollama)
docker compose --profile llm up --build
docker compose exec ollama ollama pull qwen2.5:3b   # one-time model pull

# Just the API image (no compose)
docker build -t banking-nlu .
docker run -p 8000:8000 -v banking_nlu_cache:/home/appuser/.cache banking-nlu
```

The image runs as a non-root user with a `/health` container healthcheck. The ELK stack
is separate under `deploy/elk/`. CI (lint, format, type-check, tests, dependency audit)
runs on every push/PR via `.github/workflows/ci.yml`.

## Observability & operations

- **Structured logging** — JSON logs (`NLU_LOG_JSON=true`) with a `request_id` on every
  record, correlated with the matching audit event. Set `NLU_LOG_JSON=false` for
  human-readable local logs.
- **Request correlation** — every response carries an `X-Request-ID` header (an inbound
  one is honoured), shared by logs and audit.
- **Health probes** — `GET /health` (liveness) and `GET /health/ready` (readiness:
  checks the config/audit store, reports the embedder; returns 503 when not ready).
- **Metrics** — Prometheus exposition at `GET /metrics` (`nlu_http_requests_total`,
  `nlu_http_request_duration_seconds`), toggled by `NLU_METRICS_ENABLED`.
- **Async audit shipping** — events ship to ELK on a background worker thread so network
  I/O never blocks the request path (`NLU_AUDIT_ASYNC=true`).
- **API versioning** — public endpoints are served under the canonical `/v1` prefix
  (e.g. `POST /v1/nlu/parse`) and the original unversioned paths (kept for back-compat).

## Conversation (multi-turn)

Beyond the single-shot `/nlu/parse`, the service runs a multi-turn slot-filling dialogue
that drives a transfer to confirmation, asking targeted follow-up questions for any
missing slot (amount, currency, recipient) in English or Arabic.

- `POST /conversation/text` — `{ "text": "...", "session_id"?, "language"?, "user_id"? }`
  → `{ session_id, reply, status, slots, pending_slot, complete, transfer? }`. Omit
  `session_id` on the first turn; reuse the returned one to continue. `status` moves
  `collecting → confirming → completed` (or `cancelled`). Pass `user_id` to engage the
  **Memory Brain** (below).

Sessions persist in Redis when `NLU_SESSION_BACKEND=redis` (auto-falls back to an
in-process store if Redis is unreachable).

## Reply phrasing: two tiers

Every reply is classified in `app/conversation/phrasing.py`, and the tier decides how
much freedom the wording has:

| Tier | Covers | Wording |
| --- | --- | --- |
| `CRITICAL` | confirmations, amounts, IBANs/masked accounts, write outcomes, balances, rendered lists | one fixed template, asserted verbatim by tests, **never** sent to a model |
| `CONVERSATIONAL` | greetings, thanks, capability answers, "which slot is missing" questions, rejection explanations, out-of-scope redirects | several hand-written phrasings, rotated without immediate repeats; optionally re-worded by the local model |

`phrasing.rewrite()` raises for a critical key, so a money reply cannot reach the model
even by mistake — `tests/test_phrasing.py` proves it, including with a hostile fake model
installed. A rewrite is only used if `phrasing.guard()` accepts it: every number and
every code/masked account in the candidate must already appear in the template, the reply
must stay in the customer's script, and it must stay short. Anything else silently falls
back to the template, as does a timeout (`NLU_REPLY_REWRITE_TIMEOUT`, default 0.8 s) or a
missing Ollama. Rewriting is off by default (`NLU_REPLY_REWRITE_ENABLED=false`);
variation alone (`NLU_REPLY_VARIATION_ENABLED`) needs no model and costs no latency.

## Memory Brain (per-user habits + shortcuts)

When a `user_id` is supplied, the assistant remembers each user across conversations so
it can pre-fill slots and offer shortcuts. It is backed by **SQL** (durable source of
truth, any SQLAlchemy provider) plus a **Redis cache** for fast reads, with an in-memory
fallback when Redis is down.

- **Habits (learned automatically)** — on every completed transfer the brain records the
  favourite recipient, usual currency, default source account, preferred language and
  recent amounts. These then fill missing slots on later turns, e.g. a user whose habit
  currency is `EGP` can say *"send 50 to Ahmed"* and skip the currency question, or
  *"send 20 to my usual"* to reuse their favourite recipient.
- **Shortcuts (user-defined)** — named transfer templates, e.g. `rent` → 5000 EGP to
  Landlord. Saying *"pay rent"* expands the shortcut and jumps straight to confirmation.

API surface:

- `GET /memory/{user_id}` — read a user's habits + shortcuts.
- `PUT /memory/{user_id}/habits` — set habit defaults (currency, source account,
  favourite recipient, language).
- `PUT /memory/{user_id}/shortcuts` — create/update a shortcut
  `{ name, amount?, currency?, recipient?, source_account?, note? }`.
- `DELETE /memory/{user_id}/shortcuts/{name}` — remove a shortcut.

Configure via `NLU_MEMORY_*` (see `.env.example`): `NLU_MEMORY_STORE_URL` (durable SQL),
`NLU_MEMORY_CACHE_BACKEND` (`memory`/`redis`), and `NLU_MEMORY_FAVORITE_MIN_COUNT`.
The whole feature degrades gracefully and is a no-op when `user_id` is omitted.

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

## Project Structure

```
app/
  main.py         — FastAPI application
  config.py       — Settings, supported currencies, DB lookup config
  errors.py       — Uniform error envelope + global exception handlers
  middleware.py   — Request body-size limit
  request_context.py — Per-request id (logs + audit correlation)
  logging_config.py  — Structured JSON logging
  metrics.py      — Prometheus metrics + middleware
  conversation/   — Multi-turn slot-filling dialogue
    engine.py     —   state machine (collecting → confirming → completed)
    state.py      —   serializable session state + slots
    store.py      —   Redis / in-memory session store
    templates.py  —   bilingual EN/AR prompts
    router.py     —   /conversation/text
  memory/         — Memory Brain (per-user habits + shortcuts)
    store.py      —   SQL durable store + Redis/in-memory cache
    service.py    —   learn habits, resolve shortcuts, apply defaults
    schemas.py    —   habits/shortcuts Pydantic models
    router.py     —   /memory/{user_id} (+ habits, shortcuts)
  schemas.py      — Pydantic models (request/response, validation, Beneficiary)
  orchestration.py— Haystack pipeline (incl. BeneficiaryLookup + LLM handler)
  llm.py          — LiteLLM exception handler & beneficiary-not-found responder
  embeddings.py   — Multilingual sentence-embedding wrapper (lazy, cached)
  vectorstore.py  — Generic FAISS cosine-similarity index
  db/
    beneficiary.py     — Configurable SQLAlchemy beneficiary repository
  admin/
    store.py           — SQLAlchemy config store (connections + audit events)
    connections.py     — Connection CRUD / test / activate service
    audit.py           — Audit middleware, recorder & store-backed stats
    elk.py             — Elasticsearch client, shipping & aggregations
    router.py          — /admin/api routes (connections + audit)
    schemas.py         — Admin Pydantic models + provider presets
  static/
    index.html         — NLU simulator
    connections.html   — Resource connection GUI
    audit.html         — Audit monitor / observability dashboard
  nlu/
    pipeline.py        — Orchestration (lang detect → intent → entities → contact)
    lang.py            — Arabic vs. English language detection
    intents.py         — Keyword-based intent classifier (fallback)
    semantic_intents.py— FAISS + embeddings intent classifier
    examples.py        — Labeled example utterances (EN + AR)
    contacts.py        — Semantic/fuzzy contact matcher
    entities.py        — Regex-based slot extraction
    english.py         — spaCy NER augmentation
    arabic.py          — Stanza NER augmentation
tests/
  test_lang.py
  test_intents.py
  test_entities.py
  test_pipeline.py
  test_api.py
  test_vectorstore.py
  test_semantic.py
  test_contacts_fuzzy.py
  test_admin_connections.py
  test_audit.py
  test_beneficiary.py
  test_http_errors.py
  test_observability.py
  test_conversation.py
deploy/
  elk/
    docker-compose.yml — Elasticsearch + Logstash + Kibana stack
    index-template.json— Elasticsearch mappings for nlu-audit*
    provision.sh       — Install template + import Kibana dashboard
    logstash/pipeline/logstash.conf
    kibana/dashboards.ndjson
```

## License

MIT
