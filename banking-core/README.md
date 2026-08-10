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

### Everything on Postgres (recommended)

`docker compose up --build` from the repository root starts Postgres, this service
and the NLU app already wired together — Postgres is provisioned, the schema is
created and the demo rows are loaded on first boot.

To run against your own Postgres instead:

```bash
createdb banking_core                  # plus a role that owns it
cd banking-core
export BANKING_CORE_DB_URL="postgresql+psycopg://banking:banking@localhost:5432/banking_core"
python -m banking_core.seed --if-empty   # loads demo rows only into an empty DB
uvicorn banking_core.main:app --port 8100
```

Point the NLU app's beneficiary read path at the same database:

```bash
export NLU_BENEFICIARY_DB_URL="postgresql+psycopg://banking:banking@localhost:5432/banking_core"
```

### Local SQLite (zero setup)

```bash
cd banking-core
python -m banking_core.seed            # create + seed banking_core.db
uvicorn banking_core.main:app --port 8100
```

Configuration (env, prefix `BANKING_CORE_`):

- `BANKING_CORE_DB_URL` (default `sqlite:///./banking_core.db`) — any SQLAlchemy URL;
  use `postgresql+psycopg://user:pass@host:5432/banking_core` for Postgres.
- `BANKING_CORE_SEED_ON_STARTUP` (default `false`) — load the demo rows when the
  database is empty. An already-populated database is never modified.
- `BANKING_CORE_AUTO_CREATE_TABLES` (default `true`) — `CREATE TABLE IF NOT EXISTS`
  on startup. Turn off if a migration tool owns the schema.
- `BANKING_CORE_DB_POOL_SIZE` / `BANKING_CORE_DB_MAX_OVERFLOW` — pool sizing (Postgres).
- `BANKING_CORE_API_KEY` (optional) — when set, callers must send `x-api-key`.

`python -m banking_core.seed` **drops and re-creates** the tables; pass `--if-empty`
to load demo data only into an empty database, which is what the startup hook uses.

## Data model

| Table | Holds | Key columns |
|---|---|---|
| `accounts` | balances the assistant reads and pre-flights against | `account_id`, `owner_user`, `account_type`, `currency`, `balance` |
| `beneficiaries` | saved payees (the add-beneficiary write target) | `id`, `owner_user`, `name`, `name_ar`, `account`, `bank`, `is_favorite` |
| `billers` | SADAD billers | `biller_code`, `name`, `category` |
