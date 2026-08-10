"""Configuration for the Banking Core service."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, overridable via ``BANKING_CORE_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="BANKING_CORE_", env_file=".env", extra="ignore"
    )

    app_name: str = "Banking Core service"

    # SQLAlchemy URL for the database holding accounts, beneficiaries and billers.
    # Postgres is the intended deployment target:
    #   postgresql+psycopg://banking:banking@postgres:5432/banking_core
    # SQLite remains the default so local development stays zero-setup.
    db_url: str = "sqlite:///./banking_core.db"

    # Create the tables on startup. Harmless on Postgres (CREATE TABLE IF NOT
    # EXISTS semantics) and required for the zero-setup SQLite flow.
    auto_create_tables: bool = True

    # Load the demo rows on startup when the database is still empty. Never
    # touches an already-populated database, so it is safe against Postgres.
    seed_on_startup: bool = False

    # Connection pool sizing (ignored by SQLite).
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Optional API key. When set, every request must carry a matching ``x-api-key``.
    api_key: str | None = None


settings = Settings()
