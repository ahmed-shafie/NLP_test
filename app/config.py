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


settings = Settings()
