"""Stable codes for why a turn refused, paused or asked for something.

The reply text is written for the customer and is free to change wording or
language; these codes are the machine-readable half of the same decision, so a
report can count "how many turns stopped on ``INSUFFICIENT_FUNDS``" without
matching on Arabic or English prose.

A code is only attached when the turn did not move the money forward: a missing
slot, an ambiguity, a validation failure, or a Banking Core refusal. A plain
confirmation or a completed action carries no code.
"""

from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    """Why this turn did not carry the request forward."""

    # Understanding: the request itself is not actionable yet.
    INTENT_UNCLEAR = "intent_unclear"
    SLOT_REQUIRED = "slot_required"
    CHOICE_NOT_RECOGNISED = "choice_not_recognised"
    CONFIRMATION_NOT_RECOGNISED = "confirmation_not_recognised"
    INAPPROPRIATE_INPUT = "inappropriate_input"

    # Identity: who or what is being paid is not settled.
    AMBIGUOUS_BENEFICIARY = "ambiguous_beneficiary"
    BENEFICIARY_NOT_FOUND = "beneficiary_not_found"
    AMBIGUOUS_BILLER = "ambiguous_biller"
    BILLER_NOT_IN_CATALOGUE = "biller_not_in_catalogue"

    # Values the deterministic layer rejected.
    INVALID_SLOT_VALUE = "invalid_slot_value"
    ACCOUNT_INVALID = "account_invalid"
    IBAN_CHECKSUM = "iban_checksum"

    # The Banking Core refused, or could not answer.
    INSUFFICIENT_FUNDS = "insufficient_funds"
    PREFLIGHT_BLOCKED = "preflight_blocked"
    BENEFICIARY_ADD_FAILED = "beneficiary_add_failed"
    BALANCE_UNAVAILABLE = "balance_unavailable"
    DIRECTORY_UNAVAILABLE = "directory_unavailable"

    # A fresh instruction arrived mid-flow, so which one to serve is unsettled.
    FLOW_SWITCH_REQUIRED = "flow_switch_required"

    # The request is for a product this assistant does not open (loan, card,
    # investment wallet, new account).
    PRODUCT_NOT_SUPPORTED = "product_not_supported"
    # "The full amount" was asked for and no authoritative figure exists here.
    AMOUNT_DUE_UNAVAILABLE = "amount_due_unavailable"

    # The customer stopped, or the session was ended for repeated abuse.
    CANCELLED_BY_CUSTOMER = "cancelled_by_customer"
    SESSION_ENDED = "session_ended"
