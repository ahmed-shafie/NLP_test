"""Central configuration for aiapply_lite.

All settings are read from environment variables (optionally loaded from a
local ``.env`` file) so the app runs fully locally by default and can be
pointed at cloud services later without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
PROMPTS_DIR = PACKAGE_DIR / "prompts"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


@dataclass
class Settings:
    """Runtime settings, resolved once from the environment."""

    # --- LLM (local Ollama by default) ---
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("AIAPPLY_LLM_MODEL", "qwen2.5:3b"))
    embed_model: str = field(default_factory=lambda: _env("AIAPPLY_EMBED_MODEL", "nomic-embed-text"))
    llm_timeout: int = field(default_factory=lambda: int(_env("AIAPPLY_LLM_TIMEOUT", "120")))

    # --- Job data sources ---
    # Comma-separated list; supported: remotive, remoteok, adzuna, mock
    job_sources: str = field(default_factory=lambda: _env("AIAPPLY_JOB_SOURCES", "remotive,remoteok"))
    adzuna_app_id: str = field(default_factory=lambda: _env("ADZUNA_APP_ID", ""))
    adzuna_app_key: str = field(default_factory=lambda: _env("ADZUNA_APP_KEY", ""))
    adzuna_country: str = field(default_factory=lambda: _env("ADZUNA_COUNTRY", "gb"))

    # --- Web search for company research (optional) ---
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY", ""))

    # --- Storage ---
    data_dir: Path = field(default_factory=lambda: Path(_env("AIAPPLY_DATA_DIR", str(DATA_DIR))))

    @property
    def job_source_list(self) -> list[str]:
        return [s.strip().lower() for s in self.job_sources.split(",") if s.strip()]

    @property
    def adzuna_enabled(self) -> bool:
        return bool(self.adzuna_app_id and self.adzuna_app_key)

    @property
    def tavily_enabled(self) -> bool:
        return bool(self.tavily_api_key)


settings = Settings()
