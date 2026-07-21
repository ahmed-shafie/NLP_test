"""Configuration for the Banking Core service."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, overridable via ``BANKING_CORE_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="BANKING_CORE_", env_file=".env", extra="ignore"
    )

    app_name: str = "Banking Core service"

    # SQLAlchemy URL for the demo database. Any provider works (Postgres/Oracle/...);
    # SQLite keeps local development zero-setup.
    db_url: str = "sqlite:///./banking_core.db"

    # Optional API key. When set, every request must carry a matching ``x-api-key``.
    api_key: str | None = None


settings = Settings()
