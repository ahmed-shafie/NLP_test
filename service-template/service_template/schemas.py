"""Pydantic schemas: the enums, the raw slots, and the *validated action object*.

This is the contract of the whole service. Three layers, mirroring
``app/schemas.py``:

* ``Language`` / ``Intent`` — enumerations the classifier can produce.
* ``ActionSlots`` — the loosely-typed values gathered across turns. Everything is
  ``Optional`` because slots are filled incrementally.
* ``TransferAction`` — the **strictly validated** result. Constructing it raises a
  ``ValidationError`` if anything is missing or invalid; the engine catches that
  to decide which follow-up question to ask. When it constructs successfully you
  have a clean JSON action object ready for a downstream system.

To add a NEW case you typically:
  1. add a value to ``Intent``;
  2. add any new fields to ``ActionSlots``;
  3. add a new validated ``*Action`` model (copy ``TransferAction``).
See ``README.md`` → "Add a new case".
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from service_template.config import SUPPORTED_CURRENCIES


class Language(str, Enum):
    """Languages the pipeline understands. Extend as needed."""

    EN = "en"
    AR = "ar"


class Intent(str, Enum):
    """Every "case"/service the assistant can handle is an intent.

    # >>> EDIT PER CASE: add your new case here, e.g. ``PAY_BILL = "pay_bill"``.
    """

    TRANSFER_MONEY = "transfer_money"  # the fully-worked example in this template
    SMALL_TALK = "small_talk"  # greetings / thanks / anything with no action
    FALLBACK = "fallback"  # recognised as nothing actionable


class ActionSlots(BaseModel):
    """Values gathered across turns. All optional — filled incrementally.

    Add fields here for your case's inputs. Keep them permissive (``str`` /
    ``Decimal | None``); strict validation happens only when we build the
    ``*Action`` object below.
    """

    amount: Decimal | None = None
    currency: str | None = None
    recipient: str | None = None
    source_account: str | None = None
    note: str | None = None
    # >>> EDIT PER CASE: add slots for your new case, e.g.
    #     biller: str | None = None
    #     reference_number: str | None = None

    def first_missing(self, required: tuple[str, ...]) -> str | None:
        """Return the first required slot that is still empty, else ``None``.

        Drives the "ask one question at a time" behaviour: the engine keeps
        prompting for ``first_missing(...)`` until it returns ``None``.
        """

        for slot in required:
            value = getattr(self, slot)
            if value is None or (isinstance(value, str) and not value.strip()):
                return slot
        return None


class TransferAction(BaseModel):
    """A fully validated "transfer money" action — the JSON emitted on success.

    This is the downstream contract. Construction fails with a
    ``ValidationError`` when a slot is missing/invalid, which the engine turns
    into a follow-up prompt. Copy this class to create a validated model for a
    new case (e.g. ``BillPaymentAction``).
    """

    intent: Intent = Intent.TRANSFER_MONEY
    amount: Decimal = Field(..., gt=0, description="Amount to send; must be > 0.")
    currency: str = Field(..., description="ISO-4217 currency code.")
    recipient: str = Field(..., min_length=1, description="Beneficiary name.")
    source_account: str | None = Field(
        default=None, description="Account to debit; resolved downstream if omitted."
    )
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
