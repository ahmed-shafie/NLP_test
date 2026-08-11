"""Application configuration and shared constants."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, overridable via environment variables (prefix ``NLU_``)."""

    model_config = SettingsConfigDict(
        env_prefix="NLU_", env_file=".env", extra="ignore"
    )

    app_name: str = "Banking NLU Brain"

    # spaCy English model. ``en_core_web_sm`` is downloaded as a separate step.
    spacy_model: str = "en_core_web_sm"

    # Stanza Arabic pipeline language code.
    stanza_lang: str = "ar"

    # Eagerly load NLP models on startup. When False, models load lazily on first use.
    preload_models: bool = False

    # Minimum confidence for an intent to be considered non-fallback.
    intent_threshold: float = 0.4

    # Multilingual sentence-embedding model (Arabic + English) for the vector store.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Use semantic (vector) intent classification; falls back to keywords if the
    # embedding model is unavailable.
    use_semantic_intent: bool = True

    # Nearest neighbours considered when classifying an utterance.
    semantic_top_k: int = 5

    # Index the curated multi-dialect example corpus (app/nlu/data) alongside the
    # built-in examples. It is what teaches the classifier to refuse banking
    # customer-service questions instead of opening a transfer on them.
    example_corpus_enabled: bool = True

    # Cosine-similarity floor for accepting a semantic intent match.
    semantic_intent_threshold: float = 0.45

    # Confidence floor for letting the semantic classifier alone pick a flow
    # when no deterministic cue fired. Higher than
    # ``semantic_intent_threshold``: a weak neighbour should abstain (ask) rather
    # than route, e.g. "close my account" scores ~0.74 against the balance
    # phrasings and must not be answered with a balance.
    semantic_route_threshold: float = 0.80

    # Cosine-similarity floor for accepting a contact match.
    contact_match_threshold: float = 0.5

    # ---- Biller catalogue (SADAD) + name gazetteer ----
    # Resolve billers against the SADAD catalogue (gazetteer + FAISS fallback).
    biller_catalog_enabled: bool = True

    # Cosine-similarity floor for accepting a FAISS biller match.
    biller_match_threshold: float = 0.55

    # FAISS/embedding semantic biller fallback. Off by default: on this small
    # catalogue it returns unreliable nearest-neighbours (e.g. "internet" ->
    # "Nesma Internet"); rapidfuzz typo matching is used instead.
    biller_semantic_enabled: bool = False

    # rapidfuzz typo-tolerant biller matching ("egar" -> "Ejar").
    biller_fuzzy_enabled: bool = True
    # Max edit distance for a typo match (1 = a single insert/delete/substitute).
    biller_fuzzy_max_distance: int = 1
    # rapidfuzz ratio floor (0-100) accepted for a longer fuzzy biller match.
    biller_fuzzy_min_ratio: float = 90.0

    # Use the name gazetteer to canonicalise/typo-correct recipient names.
    names_gazetteer_enabled: bool = True

    # rapidfuzz ratio floor (0-100) for accepting a fuzzy name match.
    name_match_score: float = 88.0

    # ---- LiteLLM exception handler (local LLM via Ollama by default) ----
    # Route the LLM fallback through LiteLLM. Disable to skip the LLM entirely.
    llm_enabled: bool = True

    # LiteLLM model string. Default targets a local Ollama model (offline, no key).
    llm_model: str = "ollama/qwen2.5:3b"

    # Base URL for the local LLM server (Ollama). Used for the reachability probe.
    llm_api_base: str = "http://localhost:11434"

    # Per-request timeout (seconds) for the LLM call.
    llm_timeout: float = 30.0

    # Sampling temperature; 0 for deterministic slot extraction.
    llm_temperature: float = 0.0

    # ---- Beneficiary account lookup (configurable database provider) ----
    # Resolve the destination beneficiary by account number against a SQL database.
    # Switching providers (PostgreSQL, Oracle, SQL Server, Impala, Hive, ...) is done
    # entirely through these settings; no code changes are required.
    db_enabled: bool = False

    # SQLAlchemy database URL. The dialect+driver selects the provider, e.g.:
    #   postgresql+psycopg://user:pass@host:5432/bank
    #   oracle+oracledb://user:pass@host:1521/?service_name=ORCL
    #   mssql+pyodbc://user:pass@host:1433/bank?driver=ODBC+Driver+18+for+SQL+Server
    #   impala://host:21050/default          (needs the impyla SQLAlchemy dialect)
    #   hive://host:10000/default            (needs the PyHive SQLAlchemy dialect)
    #   sqlite:///./beneficiaries.db         (handy for local development/tests)
    db_url: str | None = None

    # Parameterized lookup query. Must accept a single bind parameter named after
    # ``db_account_param`` (default ``account_number``).
    db_query: str = (
        "SELECT id, name, account, bank FROM beneficiaries "
        "WHERE account = :account_number"
    )

    # Name of the bind parameter carrying the account number in ``db_query``.
    db_account_param: str = "account_number"

    # JSON mapping of result-set column names -> Beneficiary fields
    # (id, name, account, bank, branch, currency). Example:
    #   {"id": "ben_id", "name": "full_name", "account": "acct_no", "bank": "bank_name"}
    # When empty, columns are read by their Beneficiary field names directly.
    db_column_map: dict[str, str] = {}

    # Per-query timeout (seconds) for the database lookup.
    db_timeout: float = 10.0

    # ---- Banking Core service (separate app: balance + pre-flight + add beneficiary)
    # The transfer/bill/balance cases call this external HTTP service. Beneficiary
    # *lookups* are read directly from the database (see beneficiary_lookup_* below);
    # this service owns balance, pre-flight funds/FX checks, and adding beneficiaries.
    banking_core_enabled: bool = False
    banking_core_url: str = "http://localhost:8100"
    banking_core_api_key: str | None = None
    banking_core_timeout: float = 10.0

    # ---- Beneficiary directory (direct DB read for transfer disambiguation) ----
    # For a transfer, the recipient is resolved by querying the beneficiaries table
    # directly (by name, EN/AR); several people sharing a first name triggers a
    # "which one?" disambiguation. This is intentionally a DB read, not an API call.
    beneficiary_lookup_enabled: bool = False
    # SQLAlchemy URL for the beneficiaries table. Defaults to the Banking Core demo DB.
    beneficiary_db_url: str = "sqlite:///./banking-core/banking_core.db"
    # Table + columns for the name lookup (overridable for a real schema).
    beneficiary_table: str = "beneficiaries"
    beneficiary_owner_column: str = "owner_user"

    # ---- Admin config store (external resource connections + audit log) ----
    # SQLAlchemy URL for the local store that persists configured connections and
    # audit events. Defaults to a SQLite file next to the project.
    admin_store_url: str = "sqlite:///./app_config.db"

    # Prefer the active stored connection over the NLU_DB_* env settings for the
    # beneficiary lookup. When no active connection exists, the env settings are used.
    use_stored_connection: bool = True

    # ---- Audit logging + ELK observability ----
    # Record every system action (HTTP request + domain events) to the audit store.
    audit_enabled: bool = True

    # Where audit events are shipped: "elasticsearch" (direct), "logstash" (TCP), or
    # "none". Events are always persisted to the local store regardless of this.
    audit_sink: str = "elasticsearch"

    # Ship audit events on a background worker thread so slow network I/O never blocks
    # the request path. Disable for fully synchronous, deterministic shipping.
    audit_async: bool = True

    # Ship audit events to Elasticsearch (ELK). Requires the elasticsearch client.
    elk_enabled: bool = True

    # Elasticsearch base URL.
    elasticsearch_url: str = "http://localhost:9200"

    # Optional Elasticsearch basic-auth credentials.
    elasticsearch_username: str | None = None
    elasticsearch_password: str | None = None

    # Index that receives audit events.
    elk_index: str = "nlu-audit"

    # Logstash TCP endpoint (json_lines codec) used when audit_sink == "logstash".
    logstash_host: str = "localhost"
    logstash_port: int = 50000

    # Maximum accepted request body size in bytes (rejects larger with HTTP 413).
    max_request_bytes: int = 1_000_000

    # ---- Observability (logging, metrics) ----
    # Root log level.
    log_level: str = "INFO"

    # Emit structured JSON logs (recommended for production / ELK ingestion).
    log_json: bool = True

    # Expose Prometheus metrics at GET /metrics.
    metrics_enabled: bool = True

    # ---- Content moderation (abusive / ribald input) ----
    # Detect profanity/abuse and reply with a calm, professional redirect instead
    # of processing the turn (and never let abusive text leak into a slot).
    moderation_enabled: bool = True

    # End the conversation after this many flagged turns in one session.
    moderation_max_strikes: int = 3

    # Similarity a turn must reach against the *nearest known abusive phrase*
    # before the semantic safety net labels it INAPPROPRIATE. It is a distance,
    # not a vote share: with tens of thousands of out-of-scope rows in the index,
    # share measures index composition more than it measures abuse. Measured on
    # the 7,667 held-out customer complaints, 0.75 flags 1 of them; legitimate
    # nearby words ("المخالفة" = a fine) retrieve no abuse neighbour at all.
    # The deterministic blocklist is unaffected and always flags.
    moderation_semantic_threshold: float = 0.75

    # Seed for canned-reply variation. ``None`` -> random (varied) replies;
    # set an int for deterministic replies (used by tests).
    reply_variation_seed: int | None = None

    # Rotate between the hand-written phrasings of a conversational reply so the
    # assistant doesn't sound canned. Disable for byte-stable replies (tests).
    reply_variation_enabled: bool = True

    # Let the local LLM re-word CONVERSATIONAL replies only (see
    # app/conversation/phrasing.py). Money-critical replies are never rewritten.
    # Off by default: fluency is opt-in, determinism is the default.
    reply_rewrite_enabled: bool = False

    # Latency budget for a rewrite. On timeout the template is sent as-is, so this
    # is the worst case a customer can wait for the extra fluency.
    reply_rewrite_timeout: float = 0.8

    # ---- Conversation (multi-turn slot filling) ----
    conversation_enabled: bool = True

    # Session backend: "redis" (shared, multi-instance) or "memory" (single process).
    # Falls back to in-memory automatically when Redis is unavailable.
    session_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 1800

    # ---- Memory Brain (per-user habits + shortcuts) ----
    # Remember each user's habits (favourite recipients, usual currency, common
    # amounts, default source account, preferred language) and user-defined shortcuts
    # (named transfer templates). SQL is the durable source of truth; Redis caches
    # reads for speed and falls back to an in-memory cache when unavailable.
    memory_enabled: bool = True

    # SQLAlchemy URL for the durable memory store. Any provider works
    # (PostgreSQL/Oracle/SQL Server/...); defaults to a local SQLite file.
    memory_store_url: str = "sqlite:///./memory_brain.db"

    # Cache backend for memory reads: "redis" or "memory" (process-local). Falls back
    # to in-memory automatically when Redis is unavailable. Reuses ``redis_url``.
    memory_cache_backend: str = "memory"
    memory_cache_ttl_seconds: int = 900

    # Minimum number of completed transfers to a recipient before it is offered as a
    # learned "favourite" default.
    memory_favorite_min_count: int = 2

    # Automatically create a shortcut (alias) once a transfer pattern repeats.
    # Rule B: same recipient (any amount) completed ``memory_auto_alias_min_count``
    # times -> a recipient-only alias. Rule A: the same recipient+amount+currency
    # completed that many times -> a full template alias. Existing aliases are never
    # overwritten and each pattern is only auto-created once.
    memory_auto_alias_enabled: bool = True
    memory_auto_alias_min_count: int = 3

    # ---- Active Learning (passive review queue + nightly index rebuild) ----
    # Log LLM-assisted / low-confidence cases to a review queue, auto-approve the
    # confident ones, and feed approved examples back into the semantic intent index.
    active_learning_enabled: bool = True

    # SQLAlchemy URL for the durable review-queue store. Any provider works
    # (PostgreSQL/Oracle/SQL Server/...); defaults to a local SQLite file.
    active_learning_store_url: str = "sqlite:///./active_learning.db"

    # A logged case with intent confidence at or above this floor (and a concrete,
    # non-fallback intent) is auto-approved and skips human review.
    active_learning_auto_approve_confidence: float = 0.85

    # A case is enqueued for review when the LLM was invoked, the intent was a
    # fallback, or the intent confidence is below this ceiling. Confident,
    # deterministic results are not logged (nothing to learn from).
    active_learning_log_confidence: float = 0.6

    # ---- Nightly FAISS intent-index rebuild (hot-swap, no restart) ----
    # Run a background daemon that rebuilds the semantic intent index from the base
    # examples plus approved review-queue examples, then atomically swaps it in.
    index_rebuild_enabled: bool = True

    # UTC hour/minute the nightly rebuild runs at.
    index_rebuild_hour_utc: int = 3
    index_rebuild_minute_utc: int = 0


# Currencies the assistant understands, keyed by ISO-4217 code.
SUPPORTED_CURRENCIES: dict[str, set[str]] = {
    "USD": {"usd", "dollar", "dollars", "$", "بالدولار", "دولار", "دولارات"},
    "EUR": {"eur", "euro", "euros", "€", "يورو"},
    "GBP": {"gbp", "pound", "pounds", "sterling", "£", "استرليني", "جنيه استرليني"},
    "EGP": {"egp", "le", "جنيه", "جنيها", "جنيهات", "جنيه مصري"},
    "SAR": {"sar", "riyal", "riyals", "ريال", "ريالات", "ريال سعودي"},
    "AED": {"aed", "dirham", "dirhams", "درهم", "دراهم", "درهم اماراتي"},
    "KWD": {"kwd", "dinar", "dinars", "دينار", "دنانير", "دينار كويتي"},
    "QAR": {"qar", "ريال قطري"},
}

# Recognised bill categories, keyed by canonical name -> trigger keywords (EN+AR).
# Matching is substring-based on the lowered utterance; anything not matched is
# kept as the free-text biller the customer typed.
BILLER_CATEGORIES: dict[str, set[str]] = {
    "electricity": {"electricity", "power", "كهرباء", "الكهرباء"},
    # "موية"/"مويه" are the colloquial Gulf spellings of "مياه".
    "water": {
        "water",
        "مياه",
        "المياه",
        "مية",
        "موية",
        "الموية",
        "مويه",
        "المويه",
    },
    "gas": {"gas", "غاز", "الغاز"},
    "internet": {"internet", "wifi", "نت", "النت", "انترنت", "إنترنت"},
    "mobile": {
        "mobile",
        "phone",
        "vodafone",
        "orange",
        "etisalat",
        "موبايل",
        "الموبايل",
        "تليفون",
        "فودافون",
        "اورنج",
    },
    "tv": {"tv", "cable", "تليفزيون", "التليفزيون"},
}

# Default currency assumed when the user gives an amount with no currency.
# SAR (Saudi Riyal) is the default for all flows (transfers and SADAD bills).
DEFAULT_CURRENCY = "SAR"

# Currency symbol -> ISO code, used while tokenizing amounts like ``$50``.
CURRENCY_SYMBOLS: dict[str, str] = {"$": "USD", "€": "EUR", "£": "GBP"}

# Demo address book used when a request does not supply its own contacts.
# Names are stored in both scripts so cross-lingual matching can be demonstrated.
DEMO_CONTACTS: list[dict[str, str]] = [
    {"id": "c1", "name": "Ahmed Hassan", "account": "EG1001"},
    {"id": "c2", "name": "أحمد حسن", "account": "EG1001"},
    {"id": "c3", "name": "Mohamed Ali", "account": "EG1002"},
    {"id": "c4", "name": "محمد علي", "account": "EG1002"},
    {"id": "c5", "name": "Sara Adel", "account": "EG1003"},
    {"id": "c6", "name": "سارة عادل", "account": "EG1003"},
    {"id": "c7", "name": "Laila Mansour", "account": "EG1004"},
    {"id": "c8", "name": "ليلى منصور", "account": "EG1004"},
]


settings = Settings()
