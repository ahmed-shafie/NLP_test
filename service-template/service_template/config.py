"""Runtime configuration for the template service.

Mirrors ``app/config.py`` and ``banking-core/banking_core/config.py``: a single
``settings`` object built from environment variables (prefixed ``SVC_``) with
sensible zero-config defaults so the template runs out of the box.

Example overrides::

    SVC_CORE_ENABLED=true SVC_CORE_BASE_URL=http://localhost:8100 \\
        uvicorn service_template.api:app --port 8200
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

# Currencies the validated action object will accept. Keep this in one place so
# both the extractor and the schema validators agree. (In the main app this
# lives in ``app/config.py`` as ``SUPPORTED_CURRENCIES``.)
SUPPORTED_CURRENCIES: frozenset[str] = frozenset(
    {"SAR", "USD", "EUR", "GBP", "AED", "EGP"}
)

# The default currency assumed when the user does not state one (e.g. "send 500
# to Ahmed"). Matches the main app's SAR default.
DEFAULT_CURRENCY: str = "SAR"


class Settings(BaseSettings):
    """All knobs are overridable via ``SVC_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="SVC_", env_file=".env", extra="ignore"
    )

    app_name: str = "Service Template"

    # ------------------------------------------------------------------ #
    # External "core" service (the thing that owns balances/accounts and
    # performs pre-flight validation). When disabled, the engine skips the
    # pre-flight step entirely and still works — handy for local demos.
    # Point this at the banking-core service (or your own) to enable it.
    # ------------------------------------------------------------------ #
    core_enabled: bool = False
    core_base_url: str = "http://localhost:8100"
    core_api_key: str | None = None
    core_timeout_seconds: float = 5.0

    # After this many collected turns without completing, the engine gives up on
    # the current action and resets (prevents infinite loops). 0 disables it.
    max_turns: int = 25

    # ------------------------------------------------------------------ #
    # NLU backends (spaCy + FAISS). Both are OPTIONAL and degrade gracefully:
    # if the model/dependency is missing, the extractor silently falls back to
    # the regex/keyword logic, so the template always runs. There is NO LLM.
    # ------------------------------------------------------------------ #

    # Semantic intent classification: embed the labelled examples in
    # ``examples.py`` and classify by nearest neighbour in a FAISS index.
    use_semantic_intent: bool = True
    # Multilingual (Arabic + English) sentence-embedding model. Downloaded from
    # the HuggingFace hub on first use and cached on disk.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Nearest neighbours considered when classifying an utterance.
    semantic_top_k: int = 5
    # Cosine-similarity floor for accepting a semantic match; below it we treat
    # the result as FALLBACK and let the keyword classifier have the final say.
    semantic_intent_threshold: float = 0.45

    # spaCy named-entity recognition for the recipient (PERSON entities).
    use_spacy_ner: bool = True
    # spaCy English model. Install once with:
    #   python -m spacy download en_core_web_sm
    spacy_model: str = "en_core_web_sm"


settings = Settings()
