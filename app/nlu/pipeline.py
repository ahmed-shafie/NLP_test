"""Orchestration: detect language, classify intent, extract and validate slots."""

from __future__ import annotations

from decimal import Decimal

from pydantic import ValidationError

from app.config import DEFAULT_CURRENCY, settings
from app.nlu import arabic, english
from app.nlu.contacts import ContactMatcher, get_default_matcher
from app.nlu.intents import classify_intent
from app.nlu.lang import detect_language
from app.nlu.semantic_intents import get_semantic_classifier
from app.schemas import (
    Contact,
    ContactMatch,
    Intent,
    Language,
    NLUResponse,
    SlotError,
    TransferEntities,
    TransferRequest,
    ValidationResult,
)

# Human-friendly follow-up prompts the assistant can speak when a slot is missing.
_SLOT_PROMPTS: dict[str, str] = {
    "amount": "How much would you like to transfer?",
    "currency": "Which currency should I use?",
    "recipient": "Who should I send the money to?",
}


def _classify(text: str, lang: Language) -> tuple[Intent, float, str]:
    """Classify intent, preferring the semantic classifier with keyword fallback."""

    classifier = get_semantic_classifier()
    if classifier is not None:
        intent, confidence = classifier.classify(text)
        return intent, confidence, "semantic"

    intent, confidence = classify_intent(text, lang)
    if confidence < settings.intent_threshold:
        intent = Intent.FALLBACK
    return intent, confidence, "keyword"


def parse(text: str, language: Language | None = None) -> NLUResponse:
    """Run the full NLU pipeline over a single utterance."""

    lang = language or detect_language(text)
    intent, confidence, source = _classify(text, lang)

    entities = TransferEntities()
    resolved: ContactMatch | None = None
    if intent is Intent.TRANSFER_MONEY:
        module = arabic if lang is Language.AR else english
        entities = module.extract_entities(text)
        # Assume a default currency when an amount is given without one.
        if entities.amount is not None and entities.currency is None:
            entities.currency = DEFAULT_CURRENCY
        if entities.recipient:
            resolved, _ = get_default_matcher().resolve(entities.recipient)

    return NLUResponse(
        text=text,
        language=lang,
        intent=intent,
        confidence=confidence,
        intent_source=source,
        entities=entities,
        resolved_recipient=resolved,
    )


def resolve_contact(
    name: str,
    contacts: list[Contact] | None = None,
    top_k: int = 3,
) -> tuple[ContactMatch | None, list[ContactMatch]]:
    """Resolve a recipient name against an address book (default: demo contacts)."""

    matcher = ContactMatcher(contacts) if contacts else get_default_matcher()
    return matcher.resolve(name, top_k=top_k)


def validate_transfer(
    *,
    amount: Decimal | None,
    currency: str | None,
    recipient: str | None,
    source_account: str | None = None,
    note: str | None = None,
) -> ValidationResult:
    """Validate gathered slots into a :class:`TransferRequest`.

    Returns a structured result listing missing slots (with prompts) and any
    field errors, so the assistant can either execute the transfer or ask a
    targeted follow-up question.
    """

    missing = [
        field
        for field, value in (
            ("amount", amount),
            ("currency", currency),
            ("recipient", recipient),
        )
        if value is None or (isinstance(value, str) and not value.strip())
    ]

    try:
        transfer = TransferRequest(
            amount=amount,  # type: ignore[arg-type]
            currency=currency,  # type: ignore[arg-type]
            recipient=recipient,  # type: ignore[arg-type]
            source_account=source_account,
            note=note,
        )
    except ValidationError as exc:
        errors: list[SlotError] = []
        for err in exc.errors():
            field = str(err["loc"][0]) if err["loc"] else "unknown"
            errors.append(
                SlotError(
                    field=field,
                    message=err["msg"],
                    prompt=_SLOT_PROMPTS.get(field, f"Please provide a valid {field}."),
                )
            )
        return ValidationResult(valid=False, missing=missing, errors=errors)

    return ValidationResult(valid=True, transfer=transfer)
