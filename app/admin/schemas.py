"""Pydantic schemas for the admin API (connections + audit observability)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Supported provider presets surfaced in the GUI. The value is the SQLAlchemy
# dialect/driver prefix; ``kind`` groups databases vs. datalake engines.
PROVIDER_PRESETS: list[dict[str, str]] = [
    {
        "provider": "postgresql",
        "kind": "database",
        "label": "PostgreSQL",
        "url_template": "postgresql+psycopg://user:pass@host:5432/dbname",
        "driver": "psycopg[binary]",
    },
    {
        "provider": "mysql",
        "kind": "database",
        "label": "MySQL / MariaDB",
        "url_template": "mysql+pymysql://user:pass@host:3306/dbname",
        "driver": "pymysql",
    },
    {
        "provider": "oracle",
        "kind": "database",
        "label": "Oracle",
        "url_template": "oracle+oracledb://user:pass@host:1521/?service_name=ORCL",
        "driver": "oracledb",
    },
    {
        "provider": "mssql",
        "kind": "database",
        "label": "SQL Server",
        "url_template": "mssql+pyodbc://user:pass@host:1433/db?driver=ODBC+Driver+18+for+SQL+Server",
        "driver": "pyodbc",
    },
    {
        "provider": "sqlite",
        "kind": "database",
        "label": "SQLite (local)",
        "url_template": "sqlite:///./beneficiaries.db",
        "driver": "(built in)",
    },
    {
        "provider": "impala",
        "kind": "datalake",
        "label": "Impala",
        "url_template": "impala://host:21050/default",
        "driver": "impyla",
    },
    {
        "provider": "hive",
        "kind": "datalake",
        "label": "Hive",
        "url_template": "hive://host:10000/default",
        "driver": "pyhive[hive]",
    },
    {
        "provider": "trino",
        "kind": "datalake",
        "label": "Trino",
        "url_template": "trino://user@host:8080/catalog/schema",
        "driver": "trino[sqlalchemy]",
    },
    {
        "provider": "presto",
        "kind": "datalake",
        "label": "Presto",
        "url_template": "presto://user@host:8080/catalog/schema",
        "driver": "pyhive[presto]",
    },
]

_DEFAULT_QUERY = (
    "SELECT id, name, account, bank FROM beneficiaries WHERE account = :account_number"
)


class ConnectionBase(BaseModel):
    """Editable fields of a resource connection."""

    name: str = Field(..., min_length=1, max_length=200)
    kind: str = Field(default="database", description="'database' or 'datalake'.")
    provider: str = Field(
        ..., min_length=1, description="SQLAlchemy dialect, e.g. postgresql."
    )
    url: str = Field(..., min_length=1, description="SQLAlchemy connection URL.")
    query: str = Field(
        default=_DEFAULT_QUERY, description="Parameterized lookup query."
    )
    account_param: str = Field(default="account_number")
    column_map: dict[str, str] = Field(default_factory=dict)


class ConnectionCreate(ConnectionBase):
    """Payload to create a connection."""


class ConnectionUpdate(BaseModel):
    """Partial update for a connection (all fields optional)."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    kind: str | None = None
    provider: str | None = None
    url: str | None = None
    query: str | None = None
    account_param: str | None = None
    column_map: dict[str, str] | None = None


class Connection(ConnectionBase):
    """A stored connection as returned by the API."""

    id: int
    is_active: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ConnectionTestResult(BaseModel):
    """Outcome of a 'test connection' probe."""

    ok: bool
    message: str
    elapsed_ms: float | None = None
    sample_columns: list[str] = Field(default_factory=list)


class AuditEvent(BaseModel):
    """A single audit event as returned by the API."""

    id: int | None = None
    timestamp: datetime
    action: str
    category: str = "http"
    method: str | None = None
    path: str | None = None
    status_code: int | None = None
    duration_ms: float | None = None
    client_ip: str | None = None
    actor: str | None = None
    request_id: str | None = None
    outcome: str = "success"
    detail: dict = Field(default_factory=dict)


class AuditStats(BaseModel):
    """Aggregated metrics powering the observability charts."""

    source: str = Field(
        description="Where the stats came from: 'elasticsearch' or 'store'."
    )
    total: int = 0
    success: int = 0
    errors: int = 0
    avg_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    by_status: dict[str, int] = Field(default_factory=dict)
    by_action: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)
    timeline: list[dict] = Field(default_factory=list)
    top_paths: list[dict] = Field(default_factory=list)


class ElkStatus(BaseModel):
    """Health/availability of the ELK pipeline."""

    enabled: bool
    sink: str
    reachable: bool
    cluster: str | None = None
    index: str | None = None
    doc_count: int | None = None
    message: str | None = None
