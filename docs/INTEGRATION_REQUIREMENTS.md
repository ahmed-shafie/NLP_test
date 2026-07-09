# NLU + LLM Middleware — Integration Requirements & Technical Specification

**Component:** NLU + LLM Middleware (bilingual EN/AR banking assistant)
**Repository:** `ahmed-shafie/NLP_test`
**Status legend:** ✅ built & running · ⚠️ partial / demo-only · ❌ target, not built

> This document mirrors the standard integration-contract format, but every schema,
> endpoint, intent, and field below is reconciled against **what is actually in this
> codebase today**. Idealized/aspirational items are explicitly marked ❌ *target* so
> they are not mistaken for delivered features.

---

## 1. Executive Summary
This service is the intelligent conversational layer of the banking architecture. It
processes bilingual (English/Arabic) natural-language queries, extracts intent, resolves
entities (e.g. beneficiaries, SADAD billers), and returns a **structured, versioned JSON
Action Object** for downstream orchestration to build and execute the final core-banking
call.

The component is **stateless regarding business logic**. It does **not** authenticate the
user, enforce business rules/limits, construct the final bank payload, handle transaction
orchestration (retries/idempotency), or execute the transaction. Those are delegated to
downstream systems.

**What is real today (beyond a plain NLU):** a deterministic-first NLU core with an LLM
fallback, a stateful conversation engine (multi-turn transfer & bill flows with
confirmation/disambiguation), an abuse-moderation guard, a Memory Brain (shortcuts/habits),
`block_trace` observability, and an Active-Learning review queue + eval gate in CI.

## 2. Component Scope and Architecture

### 2.1 Architecture flow (as-built)
```
[ Channel / App / Web UI ]
        |  (text; today: {text, language?, account_number?})   ❌ target: authenticated envelope
        v
[ FastAPI ]  ->  [ Conversation Engine ]  --lookup-->  [ Beneficiary / Account-Details source ] ⚠️ off by default
        |               |
        |               v
        |        [ Haystack Pipeline: detect-lang -> moderation -> intent (FAISS)
        |          -> entities -> contacts -> beneficiary -> LLM fallback ]
        v
[ Structured JSON Action Object ]  --->  ❌ [ Request Builder -> Orchestration -> Bank API Adapter -> Core Banking ]
```
Key correction vs. an idealized diagram: the **LLM is the last stage inside the linear
pipeline** (fires only when deterministic slots are incomplete / fallback), not a parallel
branch. Everything to the right of the Action Object (Request Builder onward) is **not built**.

### 2.2 Deliverables (our side of the contract)
1. **JSON Action Object** — validated, versioned (`schema_version`), with intent,
   confidence, entities, `resolved_beneficiary`, and a conversational `status`.
   ✅ built (see `NLUResponse` / `ConversationResponse`).
2. **Intent enum & slot dictionary** — stable list of intents + required/optional slots. ✅ built.
3. **Conversational status semantics** — state machine consumers use to know when an action
   is ready (`complete`) vs. still clarifying. ✅ built.
4. **Observability** — per-turn `trace_id` + `block_trace`. ✅ built (NLU steps only; bank-call
   sub-fields ❌ until a bank call exists).

### 2.3 Exclusions (explicitly not ours)
- **Authentication & authorization** — assumed done by the channel. ❌ not in this component.
- **Business-limit enforcement** — funds/daily-limit checks are downstream. ❌ not here.
- **Payload mapping** — we emit a canonical object; mapping to proprietary core-banking fields is downstream. ❌ not here.
- **Transaction execution** — we never initiate/commit a transaction. ❌ not here.

---

## 3. Data Contracts: Input and Output

### 3.1 Input — as-built vs. target
**As-built** — `POST /nlu/parse` (also mirrored under `/v1`) and `POST /conversation/text`:
```jsonc
// POST /nlu/parse
{ "text": "حول 500 ريال إلى أحمد", "language": "ar" /*optional*/, "account_number": "..." /*optional*/ }

// POST /conversation/text  (stateful, multi-turn)
{ "text": "...", "session_id": "sess_...", "language": "ar", "user_id": "CUST_10045" }
```
There is **no** authenticated request envelope today (no `request_id`, `channel`,
`user_context`, `source_accounts`). ⚠️

**❌ Target input envelope (recommended for production)** — the channel should send an
authenticated context wrapper:
```jsonc
{
  "request_id": "req_987654321",
  "timestamp": "2026-07-09T14:30:00Z",
  "channel": "MOBILE_APP | WEB_PORTAL | WHATSAPP | IVR",
  "user_context": {
    "customer_id": "CUST_10045",          // verified, authenticated
    "session_id": "sess_abc123xyz",
    "language": "en",
    "source_accounts": ["1000001234", "2000005678"]
  },
  "message": { "text": "Transfer 500 riyals to my brother Ahmed" }
}
```
*Gap to close:* accept and thread `request_id → trace_id`, `session_id`, `customer_id`, and
`source_accounts` (today `session_id`/`user_id` exist on `/conversation/text`; the rest do not).

### 3.2 Output — the Action Object (as-built)
Formal schema: [`action_object.schema.json`](./action_object.schema.json); examples:
[`action_object.example.json`](./action_object.example.json). Grounded in the real
`NLUResponse` model:

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | `"1.0.0"` (consumers pin major). |
| `text` | string | Raw utterance (audit/trace). |
| `language` | enum | `en` \| `ar`. |
| `intent` | enum | `transfer_money` \| `pay_bill` \| `small_talk` \| `inappropriate` \| `fallback`. |
| `confidence` | number 0..1 | Intent confidence. |
| `intent_source` | enum | `semantic` \| `keyword` \| `moderation`. |
| `status` | enum | `complete` \| `needs_clarification` \| `fallback` \| `refused`. |
| `entities` | object | Transfer: `amount, currency, recipient, source_account, note`. Bill: `biller, biller_category, biller_code, biller_name, reference_number, amount, currency, note`. |
| `resolved_beneficiary` | object\|null | `beneficiary_id, name, account, bank, branch, currency`. |
| `beneficiary_source` | enum\|null | `database` \| `contacts` \| `llm`. |
| `llm_assisted` | bool | LLM filled/corrected slots. |
| `clarification` | string\|null | Follow-up text when `needs_clarification`. |
| `session_id`, `trace_id` | string\|null | Session + correlation IDs. |

**Naming note vs. your idealized PDF (kept honest):** the app uses a **flat** `intent` +
`confidence` (not a nested `intent: {name, confidence}`), `entities.recipient` (not
`recipient_name_raw`), and `resolved_beneficiary.{id,name,account,bank}` (not
`{beneficiary_id, account_number, bank_code, full_name, match_confidence}`). The intent enum
is **lowercase** and does **not** split `TRANSFER_INTERNAL`/`TRANSFER_EXTERNAL`. These are
the concrete deltas to decide on before publishing v1 externally.

**Supported currencies (as-built):** USD, EUR, GBP, EGP, SAR, AED, KWD, QAR — **default SAR**.

Example (transfer, complete):
```json
{
  "schema_version": "1.0.0", "text": "حول 500 ريال إلى أحمد", "language": "ar",
  "intent": "transfer_money", "confidence": 0.98, "intent_source": "semantic",
  "status": "complete",
  "entities": { "amount": 500, "currency": "SAR", "recipient": "أحمد" },
  "resolved_beneficiary": { "beneficiary_id": "BEN_4001", "name": "Ahmed",
    "account": "SA03...519", "bank": "Al Rajhi Bank", "currency": "SAR" },
  "beneficiary_source": "database", "llm_assisted": false,
  "session_id": "sess_9f3c", "trace_id": "trc_2b71"
}
```

### 3.3 Conversation engine status (multi-turn, as-built)
`/conversation/text` returns `ConversationResponse` with its own state machine
(`ConversationStatus`): `selecting`, `collecting`, `disambiguating`, `confirming`,
`completed`, `cancelled` — plus `reply`, `pending_slot`, `slots`, `transfer`/`bill`
validated objects, `flagged_terms`, `block_trace`. This is the live chat contract; the
Action Object `status` above is the normalized, consumer-facing projection.

---

## 4. External Dependencies and Prerequisites

### 4.1 Core-Banking API team *(highest priority — blocking)*
- **API specifications** (OpenAPI/Swagger) for every action: internal transfer, external
  transfer, SADAD bill payment, balance inquiry, beneficiary management.
- **Exact request/response schemas** — field names, types, required/optional, formats (IBAN
  rules, amount precision, ISO-4217 currency, dates).
- **Authentication model** — OAuth2 / mTLS / API key; token issuer + scopes; explicit
  confirmation auth is **not** owned by this component.
- **Idempotency & correlation** — required idempotency key / correlation-ID header.
- **Error catalogue** — codes/messages to map backend failures to user-friendly replies.
- **Test environment** — stable sandbox/UAT with test accounts, beneficiaries, billers.
- **Operational metrics** — rate limits, timeouts, SLAs.
- **Field-mapping rules** — how our canonical slots map to their fields
  (e.g. `resolved_beneficiary.beneficiary_id → beneficiaryId`, `entities.amount → txn.amount`).

### 4.2 Beneficiary / Account-Details data owners *(blocking)*
- **Access method** — highly-available, **read-only lookup API** (preferred over direct DB).
  ⚠️ Today an internal lookup exists but is **OFF by default** (`db_enabled=false`, demo data).
- **Connection details** — endpoint URLs / connection strings + a secure read-only credential.
- **Schema & lookup keys** — query by account number / IBAN / alias / customer ID; returned
  fields (name, account, currency, **account type**, status). *(account-type is ❌ not consumed today.)*
- **Matching rules** — disambiguation + canonical EN↔AR name formats.
- **Data scope** — global directory vs. scoped to the authenticated customer's beneficiaries.
- **PII constraints** — what may be logged / masked / cached.

### 4.3 Identity / Session / Channel team *(blocking)*
- **Context delivery mechanism** — how authenticated context (customer ID, source accounts,
  entitlements) reaches us: secure session token (JWT) or dedicated HTTP headers.
- **Identity confirmation** — formal assurance every inbound request is already
  authenticated; the middleware performs **no** login/credential validation.

### 4.4 Product / Business team *(blocking)*
- **Canonical lists** — finalized intent list + slot dictionary (mandatory vs optional per
  intent), aligned to our enum. *(Decide the `transfer_money` vs `TRANSFER_INTERNAL/EXTERNAL` split.)*
- **Business rules** — formal confirmation limits (e.g. "max 10,000 SAR") are enforced
  **downstream**, not in the NLU middleware.
- **SADAD catalogue** — official, up-to-date biller catalogue (codes + categories) as source
  of truth. ⚠️ We ship a working `sadad_billers.csv`; production must sync the official list.

### 4.5 Security / Compliance team *(blocking)*
- **Data policies** — PII redaction + data-residency rules.
- **Audit requirements** — logging requirements, retention, encryption in transit & at rest.

---

## 5. Responsibilities & Blocking Prerequisites

### 5.1 The boundary agreement
> **We deliver** a validated, versioned JSON Action Object from natural-language input.
> **They deliver** API specs, auth, sandbox, and beneficiary-data access to map that JSON to
> a real bank call.
> **Not ours:** user authentication, limit enforcement, final payload construction,
> orchestration, and executing the bank call.

### 5.2 Prerequisites checklist
| # | Prerequisite | Owner | Blocking? |
|---|---|---|---|
| 1 | Bank API spec + sandbox + auth | Core-Banking | **Yes** |
| 2 | Beneficiary lookup access (read-only) | Data owners | **Yes** |
| 3 | Authenticated identity/context passed in | Identity/Channel | **Yes** |
| 4 | Canonical intents/slots + SADAD catalogue | Product | **Yes** |
| 5 | PII / audit / residency policy | Security | **Yes** |
| 6 | Field-mapping rules (our slots → their fields) | Core-Banking + us | **Yes** |

### 5.3 Our own build gaps to reach full end-to-end (❌ not built)
1. **Authenticated input envelope** — accept `request_id/channel/user_context/source_accounts`.
2. **Request Builder** — map the Action Object → validated bank payload (+ account-type check).
3. **Orchestration** — idempotency, retries/timeouts, response normalization.
4. **Bank API Adapter** — actually call transfer / SADAD / balance endpoints (start with a mock).
5. **Transactional response** — return real transaction status + bank reference ID (`status: complete` today stops at a *confirmed* action, not an *executed* one).
