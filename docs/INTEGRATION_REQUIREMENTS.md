# Integration Requirements & Prerequisites

**Scope of this component (the NLU + LLM middleware).** This service understands a
bilingual (EN/AR) natural-language banking query, converts it into a **structured,
versioned JSON action object** (see [`action_object.schema.json`](./action_object.schema.json)),
and resolves the **beneficiary** from an account-details source. It does **not**
authenticate the user, enforce business limits, build the final bank payload,
orchestrate retries/idempotency, or execute the transaction. Those belong to
downstream parties.

```
[ Channel/App ] --auth'd identity + text-->  [ THIS: NLU + LLM ]  --Action Object JSON-->  [ Request Builder -> Orchestration -> Bank API Adapter -> Core Banking ]
                                                     |  ^
                                          beneficiary lookup (read-only)
```

This document is the checklist of what must be **requested from / agreed with other
parties** for the solution to interact end-to-end.

---

## 0. What WE deliver (our side of the contract)
- A validated, **versioned** JSON action object: `intent + confidence + entities +
  resolved_beneficiary + status`. Schema: [`action_object.schema.json`](./action_object.schema.json);
  examples: [`action_object.example.json`](./action_object.example.json).
- A stable **intent enum** and **slot dictionary**.
- `status` semantics so consumers know when a result is safe to act on
  (`complete`) vs. still in dialogue (`needs_clarification` / `fallback` / `refused`).
- Per-turn observability (`trace_id`, block trace).

**Explicitly NOT ours:** authentication, limit/eligibility enforcement, payload
mapping to bank fields, orchestration, and the bank call itself.

---

## 1. From the Bank / Core-Banking API team  *(highest priority)*
- [ ] **API specification** (OpenAPI/Swagger) for every supported action: internal
      transfer, external transfer, SADAD bill payment, balance inquiry, beneficiary mgmt.
- [ ] **Exact request/response schemas** — field names, types, required/optional,
      formats (IBAN, amount precision, ISO-4217 currency, dates).
- [ ] **Authentication model** — OAuth2 / mTLS / API key; token issuer + scopes.
      Confirm **who owns auth** (should not be this component).
- [ ] **Idempotency & correlation** — required idempotency key / correlation-ID header?
- [ ] **Error catalogue** — codes/messages, to map to user-friendly replies.
- [ ] **Sandbox/test environment + test data** — test accounts, beneficiaries, billers.
- [ ] **Rate limits, timeouts, SLAs.**
- [ ] **Field-mapping rules** — how our slots map to their fields
      (e.g. `recipient/resolved_beneficiary → beneficiaryId`, `amount → txn.amount`).

## 2. From the Beneficiary / Account-Details data owners
- [ ] **Access method** — read-only lookup **API** (preferred) or DB connection.
- [ ] **Connection details + a read-only service credential.**
- [ ] **Schema & lookup keys** — query by account no. / IBAN / alias / customer ID;
      returned fields (name, account, currency, **account type**, status).
- [ ] **Matching/disambiguation rules** and canonical name formats (EN↔AR).
- [ ] **Data scope & freshness** — global vs. scoped to the authenticated customer.
- [ ] **PII constraints** — what may be logged / masked / cached.

## 3. From the Identity / Session / Channel team
- [ ] **How the authenticated customer context reaches us** — customer ID, the
      user's own/source account(s), entitlements — via session token or header.
- [ ] Confirmation we receive a **trusted, already-authenticated identity**
      (we do not perform login).

## 4. From Product / Business
- [ ] **Canonical intent list** and **slot dictionary** (mandatory slots per intent)
      — must match our enum.
- [ ] **Business rules / limits** — confirmed to be enforced **downstream**, not in NLU.
- [ ] **Official SADAD biller catalogue** (codes + categories) as source of truth.

## 5. From Security / Compliance
- [ ] **PII redaction & data-residency rules.**
- [ ] **Audit/logging requirements** — what to log, retention, encryption in
      transit/at rest.

---

## The one-line boundary to agree with everyone
> **We deliver** a validated, versioned JSON action object.
> **They deliver** the API specs, auth, sandbox, and beneficiary-data access so that
> JSON can be mapped to a real bank call.
> **Not ours:** authentication, limit enforcement, payload building, orchestration,
> and the bank call itself.

## Prerequisites checklist (blocking for end-to-end)
| # | Prerequisite | Owner | Blocking? |
|---|---|---|---|
| 1 | Bank API spec + sandbox + auth | Core-Banking | **Yes** |
| 2 | Beneficiary lookup access (read-only) | Data owners | **Yes** |
| 3 | Authenticated identity/context passed in | Identity/Channel | **Yes** |
| 4 | Canonical intents/slots + SADAD catalogue | Product | Yes |
| 5 | PII / audit / residency policy | Security | Yes |
| 6 | Field-mapping rules (our slots → their fields) | Core-Banking + us | Yes |
