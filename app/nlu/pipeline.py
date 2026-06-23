"""Orchestration: detect language, classify intent, extract and validate slots."""

from __future__ import annotations

from decimal import Decimal

from pydantic import ValidationError

from app.nlu.contacts import ContactMatcher, get_default_matcher
from app.schemas import (
    BillPaymentRequest,
    Contact,
    ContactMatch,
    Language,
    NLUResponse,
    SlotError,
    TransferRequest,
    ValidationResult,
)

# Human-friendly follow-up prompts the assistant can speak when a slot is missing.
_SLOT_PROMPTS: dict[str, str] = {
    "amount": "How much would you like to transfer?",
    "currency": "Which currency should I use?",
    "recipient": "Who should I send the money to?",
    "biller": "Which bill would you like to pay?",
    "reference_number": "What's the bill/reference number?",
}


def parse(
    text: str,
    language: Language | None = None,
    account_number: str | None = None,
) -> NLUResponse:
    """Run the full NLU pipeline over a single utterance.

    Delegates to the Haystack-orchestrated pipeline (which adds the beneficiary
    account lookup and the LiteLLM exception handler on top of the deterministic
    steps).
    """

    from app.orchestration import run_pipeline

    return run_pipeline(text, language, account_number)


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


def validate_bill_payment(
    *,
    biller: str | None,
    reference_number: str | None,
    amount: Decimal | None,
    currency: str | None,
    biller_category: str | None = None,
    note: str | None = None,
) -> tuple[BillPaymentRequest | None, list[str], list[SlotError]]:
    """Validate gathered bill slots into a :class:`BillPaymentRequest`.

    Returns ``(payment, missing, errors)`` so the assistant can either submit the
    bill payment or ask a targeted follow-up for the first missing/invalid slot.
    """

    missing = [
        field
        for field, value in (
            ("biller", biller),
            ("reference_number", reference_number),
            ("amount", amount),
            ("currency", currency),
        )
        if value is None or (isinstance(value, str) and not value.strip())
    ]

    try:
        payment = BillPaymentRequest(
            biller=biller,  # type: ignore[arg-type]
            biller_category=biller_category,
            reference_number=reference_number,  # type: ignore[arg-type]
            amount=amount,  # type: ignore[arg-type]
            currency=currency,  # type: ignore[arg-type]
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
        return None, missing, errors

    return payment, [], []
