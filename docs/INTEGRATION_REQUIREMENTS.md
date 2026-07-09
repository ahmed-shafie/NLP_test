# NLU + LLM Middleware — Integration Guide

**Component:** NLU + LLM Middleware (bilingual EN/AR banking assistant)
**Repository:** `ahmed-shafie/NLP_test`
**Audience:** IT / Infrastructure team and Mobile / Frontend developers integrating with this service.

> This document describes **what is implemented today** and exactly **what we need from the
> IT team and the Mobile developers** to integrate the application with them.

---

## 1. What this application is
A FastAPI service that turns a bilingual (English/Arabic) natural-language banking message
into a **structured JSON result**: it detects the intent (transfer, bill payment, small-talk),
extracts entities (amount, currency, recipient, biller, reference), resolves the beneficiary,
and drives a multi-turn conversation (asking for missing details, disambiguating, confirming).
It exposes both a stateless parse endpoint and a stateful chat endpoint for a mobile/web client.

---

## 2. What is implemented now
| Capability | Status | Detail |
|---|---|---|
| **REST API (FastAPI)** | ✅ | `/nlu/parse`, `/conversation/text`, `/health`, `/health/ready`, `/metrics` (mirrored under `/v1`). |
| **Language detection (EN/AR)** | ✅ | Auto-detected per message; can be hinted. |
| **Intent classification** | ✅ | FAISS semantic classifier + keyword fallback. Intents: `transfer_money`, `pay_bill`, `small_talk`, `inappropriate`, `fallback`. |
| **Entity extraction** | ✅ | Amount, currency, recipient, source account, biller, biller category/code, reference number — EN + AR (incl. Arabic-Indic digits, spelled-out amounts). |
| **Transfer flow** | ✅ | Collect amount/currency/recipient → confirm → complete. |
| **Bill-payment flow (SADAD)** | ✅ | Biller resolution by name / category / 3-digit code / fuzzy typo; disambiguation when ambiguous. |
| **Contact & beneficiary resolution** | ✅ | Name matching (EN↔AR transliteration); beneficiary lookup by account number (⚠️ off by default — see §4). |
| **Conversation engine (multi-turn)** | ✅ | Stateful sessions; states `selecting/collecting/disambiguating/confirming/completed/cancelled`. |
| **Moderation / abuse guard** | ✅ | Refuses abusive input with bilingual redirects; tuned to avoid over-blocking legitimate words. |
| **Memory Brain** | ✅ | Saved shortcuts + learned habits (favorite recipient, common amounts). |
| **LLM fallback** | ✅ | On-prem Ollama `qwen2.5:3b`; fires only when the deterministic path is incomplete. |
| **Observability** | ✅ | Per-turn `block_trace` + `trace_id`; Prometheus `/metrics`. |
| **Quality gate** | ✅ | Deterministic eval gate in CI + Active-Learning review queue. |
| **Supported currencies** | ✅ | USD, EUR, GBP, EGP, SAR, AED, KWD, QAR — **default SAR**. |

---

## 3. What we need from the Mobile / Frontend developers
The mobile app is the **channel** that talks to this service. To integrate, the mobile team needs
the API contract below and must agree to a few responsibilities.

### 3.1 Endpoints to call
- **Stateful chat (recommended for the app):** `POST /conversation/text`
- **Stateless single parse (optional):** `POST /nlu/parse`

### 3.2 Request contract (what the app sends)
```jsonc
// POST /conversation/text
{
  "text": "حول 500 ريال إلى أحمد",   // required: the user's message this turn
  "session_id": "sess_abc123",        // keep constant across a conversation; create per new chat
  "language": "ar",                   // optional hint ("en" | "ar"); auto-detected if omitted
  "user_id": "CUST_10045"             // the authenticated customer id (see 3.5)
}
```
```jsonc
// POST /nlu/parse
{ "text": "pay my STC bill 12345, 200 riyals", "language": "en" /*optional*/, "account_number": "3000009999" /*optional*/ }
```

### 3.3 Response contract (what the app receives)
`POST /conversation/text` returns:
```jsonc
{
  "session_id": "sess_abc123",
  "reply": "You're sending 500 SAR to Ahmed. Confirm? (yes/no)", // show this to the user
  "status": "confirming",   // selecting | collecting | disambiguating | confirming | completed | cancelled
  "language": "ar",
  "intent": "transfer_money",
  "pending_slot": null,     // if set, the field we still need (prompt the user for it)
  "complete": false,        // true only when the action is fully collected + confirmed
  "slots": { "amount": 500, "currency": "SAR", "recipient": "أحمد" },
  "transfer": { /* validated transfer object when applicable */ },
  "bill": null,             // validated bill object for pay_bill
  "flagged_terms": [],      // non-empty if the message was moderated
  "block_trace": [ /* per-step trace for debugging */ ]
}
```

### 3.4 How the app should drive the conversation
- Send each user message with the **same `session_id`**; start a new `session_id` for a new chat.
- Always render `reply` to the user.
- Loop on `status`: while `collecting`/`disambiguating`/`confirming`, keep sending the user's
  next message. Treat `complete: true` (status `completed`) as "action ready".
- If `flagged_terms` is non-empty, the turn was a moderation redirect — just show `reply`.
- Pass `language` if the app already knows the user's preference; otherwise omit it.

### 3.5 Mobile team responsibilities
- **Authentication is done by the app/channel**, not by this service. The app must only call the
  API for an **already-authenticated** user and pass the verified `user_id` (customer id).
- Manage `session_id` lifecycle (one per conversation) and conversation timeout on the UI side.
- Do not send secrets or tokens in `text`; this service does not authenticate.
- Handle standard HTTP errors (400 validation, 5xx) and show a friendly fallback message.

---

## 4. What we need from the IT / Infrastructure team
To host and operate the service, IT must provide the following environment. Config is via `NLU_*`
environment variables; secrets come from the bank's vault (nothing is committed to the repo).

### 4.1 Runtime & hosting
- **Python 3.11+** container (Dockerfile provided), run with **Uvicorn** behind the API gateway /
  reverse proxy with **TLS terminated upstream**.
- Run **≥2 replicas** for availability; the service is stateless once session/memory state is
  externalized (see 4.3).
- **Liveness/readiness probes:** `GET /health` and `GET /health/ready`.

### 4.2 NLP / LLM models (must be reachable / pre-cached)
| Asset | Requirement |
|---|---|
| spaCy + Stanza language models (EN/AR) | baked into the image or a mounted volume |
| sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` | pre-downloaded; set `NLU_PRELOAD_MODELS=true` to warm at boot |
| faiss-cpu | CPU-only, no GPU required |
| **On-prem Ollama** running `qwen2.5:3b` | reachable at `NLU_LLM_API_BASE`; keep the LLM **in-bank** — no external LLM API |

### 4.3 Data stores & services
- **Redis** for sessions in production — set `NLU_SESSION_BACKEND=redis` + `NLU_REDIS_URL`
  (default is single-process in-memory; not suitable for multi-replica).
- **Database** for Memory Brain — `NLU_MEMORY_STORE_URL` (SQLite by default → managed DB in prod).
- **Elasticsearch / ELK** for the audit trail — `NLU_AUDIT_SINK=elasticsearch` + `NLU_ELASTICSEARCH_URL`.
- **Beneficiary / account-details lookup** — a **read-only** lookup API or DB, enabled with
  `NLU_DB_ENABLED=true` + endpoint/credential. IT must provide: access method, connection details +
  read-only credential, queryable keys (account number / IBAN / customer id), returned fields
  (name, account, currency, status), and PII/logging constraints.

### 4.4 Networking & security
- **Egress allow-list:** only the Ollama endpoint and the beneficiary lookup need outbound access;
  no public internet at runtime once models are cached.
- **Secrets** (Redis URL, DB creds, ELK/Ollama endpoints, beneficiary credential) delivered via the
  bank's **secret manager / vault** as env vars.
- **Request size cap** `NLU_MAX_REQUEST_BYTES` (default 1 MB) and structured JSON logging enabled.
- Confirm **PII redaction, data-residency, retention, and encryption** policies (audit logs +
  `block_trace` masking) before go-live.

### 4.5 Key configuration (env vars)
| Variable | Production value / note |
|---|---|
| `NLU_PRELOAD_MODELS` | `true` |
| `NLU_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `NLU_LLM_ENABLED` / `NLU_LLM_MODEL` / `NLU_LLM_API_BASE` | `true` / `ollama/qwen2.5:3b` / in-bank Ollama URL |
| `NLU_DB_ENABLED` | `true` + beneficiary lookup endpoint/credential |
| `NLU_SESSION_BACKEND` / `NLU_REDIS_URL` | `redis` / managed Redis URL |
| `NLU_AUDIT_ENABLED` / `NLU_AUDIT_SINK` / `NLU_ELASTICSEARCH_URL` | `true` / `elasticsearch` / ELK URL |
| `NLU_METRICS_ENABLED` | `true` (Prometheus `/metrics`) |
| `NLU_MODERATION_SEMANTIC_THRESHOLD` | `0.80` |
| `NLU_MAX_REQUEST_BYTES` | `1000000` |
| `NLU_LOG_JSON` / `NLU_LOG_LEVEL` | `true` / `INFO` |

---

## 5. Integration checklist
**Mobile / Frontend**
- [ ] Call `POST /conversation/text` with `text` + a stable `session_id`.
- [ ] Pass the authenticated `user_id`; never send credentials in `text`.
- [ ] Render `reply`; loop on `status` until `complete: true`.
- [ ] Handle `flagged_terms` (moderation) and HTTP errors gracefully.

**IT / Infrastructure**
- [ ] Deploy the container (≥2 replicas) behind the gateway with TLS + health probes.
- [ ] Pre-cache models; `NLU_PRELOAD_MODELS=true`.
- [ ] Stand up on-prem Ollama (`qwen2.5:3b`) and allow-list it.
- [ ] Provide Redis (sessions), a managed DB (memory), and ELK (audit).
- [ ] Provide read-only beneficiary lookup access + credential (`NLU_DB_ENABLED=true`).
- [ ] Deliver all secrets via the vault; set the `NLU_*` config above.
- [ ] Sign off PII redaction / data-residency / retention with Security.
