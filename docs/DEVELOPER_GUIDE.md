# Developer Guide — Bilingual Banking NLU Layer

Repository: <https://github.com/ahmed-shafie/NLP_test> · `main` @ `a9d8fa0` · Python 3.12 · FastAPI
Companion documents: [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md) (diagrams and component
inventory) and [`CHANGE_GUIDE.md`](CHANGE_GUIDE.md) (worked examples for making a change).
This guide is the *working* document: how to run it, how to change it safely, and what will reject
your change.

---

## 1. Read this first — the rule that governs every change

> Amounts, currencies, IBANs, beneficiary names, balances, confirmations, rejection reasons and
> operation results are **deterministic and template-rendered**. A model may *select* or *nominate*;
> it may never *author*.

![Determinism boundary](diagrams/06-determinism.png)

Practical consequences for you as a contributor:

| You want to… | Allowed? | How |
|---|---|---|
| add a phrasing for a greeting or clarifying question | yes | `templates.py`, conversational tier |
| add a phrasing that contains an amount/IBAN/name | only via placeholders in a **critical** template | `templates.py` + `phrasing.py` tier `CRITICAL` |
| let the LLM write a confirmation line | **no** | `tier_of()` returns `CRITICAL`; `rewrite()` refuses |
| let a classifier pick which stored reply is sent | yes | it selects a key, never text (`topic_replies.TOPIC_REPLIES`) |
| have the model choose between two beneficiaries | **no** | ambiguity ⇒ ask the customer (`DISAMBIGUATING`) |
| return `completed` when Banking Core refused | **no** | status must be `FAILED` |
| put a digit in the opening greeting | **no** | a test asserts no variant contains a digit |

If a change would violate this, the correct move is to stop and raise it — not to work around the
guard.

---

## 2. Local setup

```bash
git clone https://github.com/ahmed-shafie/NLP_test && cd NLP_test
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt          # dev = runtime + pytest/ruff/mypy/sklearn
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python -c "import stanza; stanza.download('ar')"
# the multilingual embedding model (~470 MB) downloads on first use
```

### 2.1 Three ways to run

```bash
# (a) fastest inner loop — no Banking Core, no Redis, models load lazily
.venv/bin/uvicorn app.main:app --reload --port 8001

# (b) realistic single process — real Banking Core + Postgres + Redis
docker compose up -d postgres redis banking-core
NLU_BANKING_CORE_ENABLED=true NLU_BANKING_CORE_URL=http://localhost:8100 \
NLU_BENEFICIARY_LOOKUP_ENABLED=true \
NLU_BENEFICIARY_DB_URL=postgresql+psycopg://banking:banking@localhost:55432/banking_core \
NLU_SESSION_BACKEND=redis NLU_REDIS_URL=redis://localhost:56379/0 \
NLU_MEMORY_CACHE_BACKEND=redis \
.venv/bin/uvicorn app.main:app --port 8001

# (c) whole stack in containers (app on :8000)
docker compose up --build
```

Useful URLs: `/assistant?dev=1` (assistant + Inspect panel), `/docs` (Swagger), `/brain`,
`/admin`, `/active-learning`, `/metrics`, `/health/ready`.

Compose host-port overrides used in this project: `POSTGRES_HOST_PORT=55432`,
`REDIS_HOST_PORT=56379`.

### 2.2 Things that will bite you once

| Symptom | Cause | Fix |
|---|---|---|
| first request takes ~45 s | models load lazily | `NLU_PRELOAD_MODELS=true` (pays ~4 min 40 s at startup instead) |
| sessions vanish with `--workers > 1` | in-memory session store | `NLU_SESSION_BACKEND=redis` |
| balance replies "تعذّر جلب الرصيد" | stale `banking-core` image (no `/accounts/balance`) | rebuild the image, or run the Core from source on `:8101` |
| beneficiary/account lookups return nothing | seeded demo rows belong to owner `demo` | use `user_id="demo"` |
| CPU pinned, low throughput | torch intra-op threads fighting workers | `OMP_NUM_THREADS=1`, scale with `--workers` |

---

## 3. Repository map

```
app/
  main.py                 app assembly, /health, /metrics, /nlu/*, static UIs
  config.py               every setting (env prefix NLU_)
  orchestration.py        Haystack pipeline: detect → intent → entities → contacts → beneficiary → llm
  trace.py                block_trace spans
  embeddings.py           SentenceTransformer wrapper
  vectorstore.py          generic FAISS store
  banking_core_client.py  typed HTTP client for the Core
  data/                   names.csv · sadad_billers.csv · blocklist.csv
  conversation/           engine.py · state.py · store.py · templates.py · phrasing.py
                          topic_replies.py · moderation.py · router.py
  nlu/                    lang.py · normalize.py · arabic.py · english.py · entities.py
                          accounts.py · intents.py · semantic_intents.py · corpus.py
                          contacts.py · topic_head.py · pipeline.py
                          data/example_corpus.jsonl · data/topic_head.npz
  db/                     beneficiary.py · directory.py (read-only SQL directory)
  memory/ active_learning/ admin/ eval/ static/
banking-core/             separate FastAPI service (main.py · service.py · db.py · seed.py)
scripts/                  eval_nlu.py · train_topic_classifier.py · build_example_corpus.py
tests/                    39 test modules, 647 tests
docs/                     ARCHITECTURE.md · DESIGN*.md · GAP_ANALYSIS.md
```

---

## 4. The turn contract

![Turn lifecycle](diagrams/03-turn-flow.png)

```http
POST /conversation/text
Content-Type: application/json

{"text": "حول 500 لأحمد", "user_id": "demo", "session_id": "s-1", "language": "ar"}
```

```jsonc
{
  "session_id": "s-1",
  "reply": "في ثلاثة باسم أحمد: (1) أحمد حسن (2) أحمد علي (3) أحمد سالم — مين منهم؟",
  "status": "disambiguating",          // selecting|collecting|disambiguating|confirming|completed|cancelled|failed
  "language": "ar",
  "intent": "transfer_money",
  "pending_slot": "recipient",
  "complete": false,
  "slots": {"amount": "500.00", "currency": "SAR", "recipient": null, "...": null},
  "transfer": null,                     // populated (typed) once the flow completes
  "bill": null,
  "flagged_terms": [],
  "warnings": [],                       // advisory pre-flight notes; never blocking
  "block_trace": [{"block": "intent_classification", "status": "ok", "duration_ms": 33.1, "note": "…"}]
}
```

Contract rules to preserve when you touch the engine:

- `session_id` is created when omitted; **the caller owns it** and it is the Redis key.
- `user_id` scopes the Memory Brain (habits + shortcuts). Without it, no shortcut expansion.
- `status = failed` is terminal and must never carry a success sentence.
- `warnings` are advisory (low funds, FX) and must not block.
- `block_trace` is append-only: every new stage adds a span. Don't add silent work.

Other endpoints: `GET /conversation/opening?language=ar|en`, `POST /nlu/parse`,
`POST /transfer/validate`, `/memory/*`, `/active-learning/*`, `/admin/api/*`.
Banking Core: `POST /accounts/balance`, `/preflight/transfer`, `/preflight/bill`, `/beneficiary/add`.

---

## 5. Recipes

### 5.1 Add a new intent

1. `app/schemas.py` → add to `Intent`.
2. `app/nlu/data/example_corpus.jsonl` → add labelled example rows (both languages; Saudi phrasing
   matters more than MSA). Row shape:
   ```json
   {"text": "…", "intent": "new_intent", "topic": "", "language": "ar", "dialect": "saudi", "source": "manual"}
   ```
3. `app/nlu/intents.py` → keyword fallback cues, so the flow still works with no embedding model.
4. `app/conversation/state.py` → its required-slot tuple if it collects slots.
5. `app/conversation/engine.py` → routing in `decide_action` + the flow branch.
6. `app/conversation/templates.py` → prompts/confirmation for both languages (critical tier if a
   financial value appears).
7. `app/eval/nlu_gold.jsonl` → gold rows for it, and `tests/` → unit tests for the flow.
8. Run the gate: `pytest -q` **and** `python scripts/eval_nlu.py` (must print `GATE PASSED`).

Watch out: the corpus is deliberately 98% `fallback`. Adding executable examples moves the routing
boundary — check `wrong_flow_starts` in the eval output, not just your own case.

### 5.2 Add or change a slot

- Field: `app/conversation/state.py::ConversationSlots` (+ the required tuple).
- Reader: `app/nlu/entities.py` — deterministic only. If a number is involved, decide explicitly
  whether an unpriced number may become money (today: **no**, it stays a reference).
- Prompt: `templates.py` for the "ask for the missing slot" line, both languages.
- Gold: add slot values to `app/eval/nlu_gold.jsonl`; per-slot F1 is enforced.
- Tests: `tests/test_entities.py` + the relevant flow test.

### 5.3 Add a biller or names

- Billers: `app/data/sadad_billers.csv` (SADAD code, English + Arabic name, category). Resolution is
  exact → category → fuzzy (`biller_fuzzy_min_ratio = 90`, max edit distance 1).
- Names gazetteer: `app/data/names.csv`. Remember the **margin rule**: a fuzzy correction is declined
  unless it leads the runner-up by `name_match_margin = 5` points.
- Arabic forms: normalisation strips `ال` and proclitics, so add the bare form (`ايجار`), not `الايجار`.
- Tests: `tests/test_names_billers.py`, `tests/test_contacts_fuzzy.py`.

### 5.4 Add or edit a customer-service answer

1. `app/conversation/topic_replies.py::TOPIC_REPLIES` — add the topic key with **both** languages.
   Replies are reviewed text; they must contain no invented number, fee, limit or duration.
2. `answer_key()` maps a topic to its reply family. Two topics sharing one reply is intentional and
   good (it cannot mislead).
3. If you added a **new reply family**, the trained head's 28 classes no longer cover it — retrain
   (§5.7) or the head simply won't select it (retrieval still can).
4. Tests: `tests/test_topic_replies.py`; if you touched the gate, also `tests/test_topic_head.py`.

### 5.5 Change reply wording safely

`app/conversation/phrasing.py` decides who may touch a reply:

- `CRITICAL_REPLIES` — fixed templates, no variation, no rewriting.
- `CONVERSATIONAL_REPLIES` — variation allowed (`pick`), model rewriting allowed when enabled.
- `guard(template, candidate, language)` — the post-generation validator: rejects a candidate that
  introduces any digit, name or IBAN not present verbatim in the template, and restores the template.

Adding a **new reply key** means classifying it in one of the two sets. An unclassified key is a bug:
decide explicitly. Note the test suite runs with `reply_variation_enabled=false` (see
`tests/conftest.py`), so assertions compare the *first* phrasing exactly.

### 5.6 Touch Banking Core

Never let the language layer decide money. To add a Core capability:

1. `banking-core/banking_core/schemas.py` + `service.py` + `main.py` (endpoint, API-key dependency).
2. `app/banking_core_client.py` — typed call, timeout (`banking_core_timeout`), explicit failure
   mapping. A failure must surface as a refusal/`FAILED`, never as a silent default.
3. `app/conversation/engine.py` — call it, and quote its reason text rather than paraphrasing.
4. Tests: `tests/test_banking_core.py` (+ Postgres path — CI runs the suite against Postgres too).

### 5.7 Retrain the topic head

```bash
.venv/bin/python -m scripts.train_topic_classifier          # writes app/nlu/data/topic_head.npz
```

- Features are the corpus embeddings the index already computes; labels come from `answer_key()`.
- The artefact stores the embedding-model name; a head trained for a different encoder is **refused**
  at load time (and the app still starts — retrieval alone answers).
- Runtime is numpy-only, `allow_pickle=False`. Never introduce pickle or sklearn on the request path.
- After retraining, re-measure coverage/error on the held-out slices before shipping — "the head is
  more confident" is not a result.

### 5.8 Rebuild the example corpus

```bash
.venv/bin/python -m scripts.build_example_corpus --csv path/to/vector_db.csv [--cap N]
```

The script projects the 77 customer-service topics onto `fallback`, and **drops the dialect test
splits** so they remain a valid measurement set. Do not re-add them: indexing your test set makes
every coverage number meaningless.

### 5.9 Tune a threshold

All gates live in `app/config.py` (`topic_head_threshold`, `topic_reply_*`, `semantic_*`,
`name_match_*`, `biller_*`). Rules of engagement:

1. Never tune by intuition on one example. Every threshold trades coverage against error.
2. Measure on the held-out slices (ArBanking77 dialect splits, Banking77 test split) and report both
   numbers: *answered %* and *wrong %*.
3. Re-run the gold gate — a threshold that fixes a question can start a wrong money flow.
4. Put the before/after table in the PR description. (Example: dropping the head threshold from
   0.999 to 0.99 buys +9 pts Arabic coverage for ~3× the error — that is why it is 0.999.)

---

## 6. Tests and quality gates

```bash
.venv/bin/pytest -q                      # 647 tests
.venv/bin/pytest tests/test_topic_head.py -q          # focused
.venv/bin/python scripts/eval_nlu.py     # gold gate — must print GATE PASSED
.venv/bin/ruff format --check app banking-core tests
.venv/bin/ruff check app banking-core tests
.venv/bin/mypy app banking-core
```

What the **gold gate** enforces (`app/eval/harness.py`, 303 hand-labelled bilingual rows in
`app/eval/nlu_gold.jsonl`):

| Check | Bar |
|---|---|
| intent accuracy | ≥ `MIN_INTENT_ACCURACY` = 0.93 (actual on `main`: 1.000) |
| per-intent recall | `MIN_INTENT_RECALL` per intent |
| per-slot F1 | amount, currency, recipient, biller_code, reference_number |
| inappropriate false positives (over-blocks) | **0** |
| wrong flow starts | **0** |
| leakage | gold rows must not appear in the index (`leaked_rows`) |

CI (`.github/workflows/ci.yml`) runs two jobs: `quality` (ruff format + ruff check + mypy + pytest
against a **Postgres 16 service** + the gold eval) and `security-audit`. Both must be green.

Test conventions worth knowing before you write one:

- `tests/conftest.py` disables the live LLM and reply variation for determinism; opt in per test with
  monkeypatch.
- Tests never assert on model probabilities — assert on the *decision* (answered vs. menu, flow
  started vs. refused).
- Do not modify an existing test to make a change pass; if a test is wrong, say so in the PR.

---

## 7. Debugging

**`block_trace` first.** Every response carries per-stage timing and the decision each stage made.
In the UI: `/assistant?dev=1` → Inspect panel. Via curl:

```bash
curl -s localhost:8001/conversation/text -H 'content-type: application/json' \
  -d '{"text":"pay my mobily bill 100 sar","user_id":"demo"}' \
  | python -m json.tool | head -40
```

Typical questions and where to look:

| Question | Look at |
|---|---|
| why did it open a transfer for a question? | `intent_classification` span + corpus neighbours; `decide_action` |
| why did it not answer a service question? | `topic_answer` span — which of the three head conditions failed |
| why did it pick that person? | `contact_resolution` (fuzzy score + margin) and `beneficiary_lookup` verdict |
| why was the amount read as a reference? | `entity_extraction` — was the number priced in a currency? |
| why is it slow? | span durations: encode stages dominate (33 ms each) |
| is the Core reachable? | `GET /health/ready`, `/admin/api/banking-core/health` |
| production rates/latency | `GET /metrics` (Prometheus) |

Logs are JSON with a `request-id` that also appears in the error envelope
`{"error": {"code", "message", "request_id"}}` — grep by it.

---

## 8. Performance notes for contributors

Measured on 8 vCPU (see the architecture doc §13): service questions **55 req/s**, executable
requests **165 req/s**, opening endpoint **453 req/s**, all at 0% failures; p95 210–290 ms.

- **Encoding is the entire cost.** `intent_classification` 33 ms, `topic_answer` 33 ms,
  `contact_resolution` 25 ms; Redis + Postgres + Core HTTP + templates sum to **< 3 ms**.
- Therefore: **never add a second encode of the same string**. There is already one duplicate
  (intent + topic on the same utterance) — memoising one vector per turn is the top open optimisation.
- Startup encodes the 31,781-row corpus per worker: ~4 min 40 s, ~600 MB RSS each. Persisting the
  index to disk is the second open optimisation.
- Load testing: the k6 script (`k6-conversation.js`) has three scenarios and pins
  `NLU_LLM_ENABLED=false`, `NLU_REPLY_VARIATION_ENABLED=false`, `NLU_PRELOAD_MODELS=true`, a distinct
  `session_id` per VU/iteration, and `user_id=demo`.

```bash
SCENARIO=service VUS=8 DURATION=25s k6 run k6-conversation.js
```

---

## 9. Coding standards

- **ruff format + ruff check + mypy** must be clean over `app`, `banking-core`, `tests`.
- Full type annotations. **No `Any`, `getattr`, `setattr`** to dodge a type — read the type instead.
- Docstrings state *why*, not *what*. Comments are rare; never comment the diff.
- Focused edits: don't reformat unrelated files, don't refactor opportunistically.
- No pickle, no `eval`, no new dependency without a reason in the PR (and prefer versions ≥ 7 days old).
- Never commit secrets, `.env`, `*.db`, caches, `.venv`, or the built index.
- Floating-point defensiveness matters here: FAISS cosine can return `1.0000001`, which a Pydantic
  `score ≤ 1` field rejects — clamp at the boundary (`min(max(score, 0.0), 1.0)`).
- PRs: one concern each, before/after numbers for anything that touches a threshold or a model, and a
  note on which gate proves it. Branch naming used in this repo: `devin/<timestamp>-<topic>`.

---

## 10. Configuration cheat-sheet (env prefix `NLU_`)

| Group | Keys |
|---|---|
| models | `EMBEDDING_MODEL`, `PRELOAD_MODELS`, `SPACY_MODEL`, `STANZA_LANG` |
| intent routing | `USE_SEMANTIC_INTENT`, `SEMANTIC_TOP_K`, `SEMANTIC_INTENT_THRESHOLD` (0.45), `SEMANTIC_ROUTE_THRESHOLD` (0.80), `EXAMPLE_CORPUS_ENABLED` |
| service answers | `TOPIC_REPLIES_ENABLED`, `TOPIC_REPLY_TOP_K` (10), `TOPIC_REPLY_THRESHOLD` (0.94), `TOPIC_REPLY_AGREEMENT` (0.8), `TOPIC_REPLY_TOP_K_EN` (7), `TOPIC_HEAD_ENABLED`, `TOPIC_HEAD_THRESHOLD` (0.999), `TOPIC_HEAD_SCORE_FLOOR` (0.80) |
| identity | `NAME_MATCH_SCORE` (88), `NAME_MATCH_MARGIN` (5), `CONTACT_MATCH_THRESHOLD` (0.5), `BILLER_*` |
| bank | `BANKING_CORE_ENABLED/URL/API_KEY/TIMEOUT`, `BENEFICIARY_LOOKUP_ENABLED`, `BENEFICIARY_DB_URL` |
| dialogue | `SESSION_BACKEND` (memory\|redis), `REDIS_URL`, `SESSION_TTL_SECONDS` (1800), `MEMORY_*`, `MODERATION_*` |
| replies | `REPLY_VARIATION_ENABLED`, `REPLY_REWRITE_ENABLED`, `LLM_ENABLED/MODEL/API_BASE/TIMEOUT/TEMPERATURE` |
| ops | `MAX_REQUEST_BYTES` (1 MB), `LOG_JSON`, `LOG_LEVEL`, `METRICS_ENABLED`, `AUDIT_*`, `ELK_*`, `ELASTICSEARCH_*` |

Banking Core uses its own prefix: `BANKING_CORE_DB_URL`, `BANKING_CORE_SEED_ON_STARTUP`,
`BANKING_CORE_API_KEY`.

---

## 11. Security notes for developers

- The NLU app has **no authentication, no CORS policy and no rate limiting** today; it is designed to
  sit behind an authenticated gateway that supplies `user_id`. Do not add features that assume the app
  authenticates the caller, and do not expose it directly.
- Banking Core is API-key protected and owns every financial decision.
- Customer text reaches logs and the audit store — do not add new fields that widen PII exposure
  without redaction.
- ArBanking77-derived rows are inside the shipped index; its licence must be cleared before commercial
  use. English Banking77 is Apache-2.0.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **action object** | the validated, strictly typed JSON the Core executes (`transfer`, `bill`) |
| **slot** | one required field of a flow (amount, currency, recipient, biller, reference) |
| **aside** | a question asked mid-flow; answered, then the flow resumes at the same slot |
| **gold set** | 303 hand-labelled bilingual cases that block CI |
| **topic / reply family** | a customer-service subject and the reviewed answer it maps to (28 families) |
| **topic head** | trained MLP (384→1024→28) that may *select* a stored reply |
| **block_trace** | per-stage timing + decision record returned with every turn |
| **Memory Brain** | per-user habits and named shortcuts (`rent` → landlord) |
| **pre-flight** | non-blocking Core checks (low funds, FX) surfaced as `warnings` |
| **margin rule** | a fuzzy name correction is declined unless it leads the runner-up by 5 points |
| **Tier A / Tier B** | deterministic financial text vs. conversational text |
