"""Pydantic schemas for NLU requests, responses, and transfer validation."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import SUPPORTED_CURRENCIES


class Language(str, Enum):
    """Languages handled by the NLU pipeline."""

    EN = "en"
    AR = "ar"


class Intent(str, Enum):
    """Intents recognised by the assistant (v1 focuses on transfers)."""

    TRANSFER_MONEY = "transfer_money"
    FALLBACK = "fallback"


class TransferEntities(BaseModel):
    """Raw slots extracted from an utterance, before strict validation."""

    amount: Decimal | None = None
    currency: str | None = None
    recipient: str | None = None
    source_account: str | None = None
    note: str | None = None


class ParseRequest(BaseModel):
    """Input for ``POST /nlu/parse``."""

    text: str = Field(..., min_length=1, description="Raw user utterance.")
    language: Language | None = Field(
        default=None,
        description="Optional language hint; auto-detected when omitted.",
    )


class NLUResponse(BaseModel):
    """Structured NLU result for an utterance."""

    text: str
    language: Language
    intent: Intent
    confidence: float = Field(..., ge=0.0, le=1.0)
    entities: TransferEntities


class TransferRequest(BaseModel):
    """A validated money-transfer instruction.

    Construction fails with a ``ValidationError`` if any slot is missing or invalid,
    which the API turns into actionable follow-up prompts for the assistant.
    """

    amount: Decimal = Field(..., gt=0, description="Amount to transfer, must be > 0.")
    currency: str = Field(..., description="ISO-4217 currency code.")
    recipient: str = Field(
        ..., min_length=1, description="Beneficiary name or contact."
    )
    source_account: str | None = None
    note: str | None = None

    @field_validator("currency")
    @classmethod
    def _known_currency(cls, value: str) -> str:
        code = value.strip().upper()
        if code not in SUPPORTED_CURRENCIES:
            supported = ", ".join(sorted(SUPPORTED_CURRENCIES))
            raise ValueError(f"Unsupported currency '{value}'. Supported: {supported}.")
        return code

    @field_validator("recipient")
    @classmethod
    def _clean_recipient(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Recipient must not be empty.")
        return cleaned


class ValidateRequest(BaseModel):
    """Input for ``POST /transfer/validate`` (slots gathered so far)."""

    amount: Decimal | None = None
    currency: str | None = None
    recipient: str | None = None
    source_account: str | None = None
    note: str | None = None


class SlotError(BaseModel):
    """A single missing/invalid slot, with a human prompt to resolve it."""

    field: str
    message: str
    prompt: str


class ValidationResult(BaseModel):
    """Outcome of validating transfer slots."""

    valid: bool
    transfer: TransferRequest | None = None
    missing: list[str] = Field(default_factory=list)
    errors: list[SlotError] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "ValidationResult":
        if self.valid and self.transfer is None:
            raise ValueError("A valid result must include a transfer.")
        return self
