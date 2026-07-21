# Banking Core service

A small, standalone FastAPI + SQLAlchemy service that stands in for the downstream
core-banking system during development. It owns a demo database of **accounts**,
**beneficiaries**, and **billers**, and exposes an external API the NLU app calls to:

- read an **account balance** (`POST /accounts/balance`),
- run **pre-flight checks** before a transfer / bill (`POST /preflight/transfer`,
  `POST /preflight/bill`) — funds and currency are advisory (**warn only** / **FX note**),
- **add a beneficiary** (`POST /beneficiary/add`).

Beneficiary *lookups* are done by the NLU app directly against this database (read path);
this service owns the *writes* and the funds/pre-flight logic.

The service validates only — it never moves real money.

## Run

```bash
cd banking-core
python -m banking_core.seed            # create + seed banking_core.db (idempotent)
uvicorn banking_core.main:app --port 8100
```

Configuration (env, prefix `BANKING_CORE_`):

- `BANKING_CORE_DB_URL` (default `sqlite:///./banking_core.db`) — any SQLAlchemy URL.
- `BANKING_CORE_API_KEY` (optional) — when set, callers must send `x-api-key`.
