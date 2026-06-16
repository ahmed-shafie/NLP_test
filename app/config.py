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

    # Cosine-similarity floor for accepting a semantic intent match.
    semantic_intent_threshold: float = 0.45

    # Cosine-similarity floor for accepting a contact match.
    contact_match_threshold: float = 0.5


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

# Default currency assumed when the user gives an amount with no currency.
DEFAULT_CURRENCY = "USD"

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
