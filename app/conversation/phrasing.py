"""Two-tier reply phrasing.

Every reply the assistant can send belongs to exactly one tier:

``Tier.CRITICAL``
    Carries a money fact — an amount, an account, a confirmation question, or the
    outcome of a write. One fixed wording, asserted verbatim by tests, and *never*
    handed to a language model. If a model turned "500" into "about five hundred"
    the customer would be misled, so the wording is not negotiable.

``Tier.CONVERSATIONAL``
    Carries no money fact — greetings, thanks, capability answers, "which slot is
    missing" questions, rejection explanations, out-of-scope redirects. These may
    be phrased several equivalent ways, and may optionally be re-worded at runtime
    by the local LLM when ``settings.reply_rewrite_enabled`` is on.

Two mechanisms keep the fluency safe:

``guard``
    A rewrite is discarded unless every number and every code/masked account in it
    already appears in the template, it stays in the customer's script, and it
    stays short. Rejection is silent — the template is sent instead.

``rewrite``
    Raises for a critical key. The tier is therefore not a convention but a
    call-site guarantee: money replies cannot reach the model even by mistake.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Mapping, Sequence
from enum import Enum

from app.config import settings
from app.schemas import Language

logger = logging.getLogger(__name__)

_RNG = random.Random()


class Tier(str, Enum):
    """Risk tier of a reply."""

    CRITICAL = "critical"
    CONVERSATIONAL = "conversational"


# Money facts, confirmations, write outcomes and rendered catalogues/lists. A
# list is critical too: a rewrite could silently drop a row or a masked account.
CRITICAL_REPLIES: frozenset[str] = frozenset(
    {
        "balance_reply",
        "beneficiary_added",
        "beneficiary_add_completed",
        "beneficiary_add_failed",
        "bill_completed",
        "bill_confirm_prompt",
        "biller_not_found",
        "choose_beneficiary",
        "choose_biller",
        # Prints the Core's balances and the numbered list a pick resolves
        # against: re-ordering or re-wording a row would move the money.
        "choose_source_account",
        "choose_transfer_purpose",
        "completed",
        # Names a person the customer did not type, so it must read exactly.
        "confirm_beneficiary_match",
        "confirm_add_beneficiary",
        "confirm_add_then_transfer",
        "confirm_prompt",
        "list_beneficiaries",
        # Reads the payees back by name before any of them is paid.
        "one_payee_at_a_time",
        "warnings_note",
        # Carries the balance and the amount refused: both are Banking Core
        # figures and must reach the customer unchanged.
        "insufficient_funds",
        "preflight_blocked",
        # Names the exact character position a checksum implicates, and repeats
        # that warning at the point of no return.
        "beneficiary_iban_typo",
        "unchecked_account_note",
        "alias_created",
        "alias_forgotten",
        "alias_not_found",
        # A policy message that ends the conversation: must stay firm and exact.
        "repeat_offense",
        # States what the assistant can and cannot do about a customer-service
        # topic; a re-worded capability claim would be a false promise.
        "topic_answer",
    }
)

CONVERSATIONAL_REPLIES: frozenset[str] = frozenset(
    {
        "ask_beneficiary_account",
        "ask_beneficiary_name",
        "balance_unavailable",
        "beneficiaries_unavailable",
        "beneficiary_add_invalid_account",
        "beneficiary_invalid_account",
        "beneficiary_not_found",
        "cancelled",
        "choose_action",
        "fallback",
        "greeting",
        "how_to_transact",
        "inappropriate",
        "no_beneficiaries",
        "opening",
        "out_of_scope",
        "resume_note",
        "slot_prompt",
        "small_talk",
    }
)


def tier_of(key: str) -> Tier:
    """Tier of a reply key.

    Keys may be namespaced (``"slot_prompt:amount"``, ``"small_talk:thanks"``) so a
    variant rotation is tracked per prompt; the tier is declared on the base name.
    """

    base = key.split(":", 1)[0]
    if base in CRITICAL_REPLIES:
        return Tier.CRITICAL
    if base in CONVERSATIONAL_REPLIES:
        return Tier.CONVERSATIONAL
    raise KeyError(f"reply {key!r} has no declared tier")


# ---------------------------------------------------------------- variation

# Last variant used per key, so an immediate repeat is avoided.
_LAST_INDEX: dict[str, int] = {}


def pick(key: str, variants: Sequence[str]) -> str:
    """Pick one phrasing, avoiding the one used last for this key."""

    if len(variants) == 1 or not settings.reply_variation_enabled:
        return variants[0]
    seed = settings.reply_variation_seed
    rng = random.Random(seed) if seed is not None else _RNG
    last = _LAST_INDEX.get(key)
    choices = [i for i in range(len(variants)) if i != last] or list(
        range(len(variants))
    )
    index = rng.choice(choices)
    _LAST_INDEX[key] = index
    return variants[index]


# ---------------------------------------------------------------------- guard

# Digit runs in either script, so "500" cannot become "50" or "٥٠٠".
_DIGIT_RUN = re.compile(r"[0-9\u0660-\u0669]+")
# Codes, currencies and masked accounts: STC, SAR, SA••7519, SADAD.
_CODE = re.compile(r"[A-Z][A-Z0-9•]{1,}")
_ARABIC = re.compile(r"[\u0600-\u06FF]")


def guard(template: str, candidate: str, language: Language) -> str | None:
    """Return ``candidate`` when it is a safe rewrite of ``template``, else ``None``.

    Conservative on purpose: anything unexpected falls back to the template, which
    costs a little fluency and never costs correctness.
    """

    text = candidate.strip()
    if not text or len(text) > max(160, 2 * len(template)):
        return None
    if "{" in text or "}" in text:  # an unfilled placeholder leaked through
        return None
    if sorted(_DIGIT_RUN.findall(text)) != sorted(_DIGIT_RUN.findall(template)):
        return None  # invented, dropped or altered a number
    if set(_CODE.findall(text)) != set(_CODE.findall(template)):
        return None  # invented or dropped a code / masked account
    if (language is Language.AR) is not bool(_ARABIC.search(text)):
        return None  # answered in the wrong script
    return text


# A decline must keep pointing the customer at the people who can answer.
_SERVICE_DESK: dict[Language, str] = {
    Language.EN: "customer service",
    Language.AR: "خدمة العملاء",
}


def guard_decline(candidate: str, language: Language) -> str | None:
    """Return ``candidate`` when it is a safe decline, else ``None``.

    Stricter than :func:`guard`, because this text is written from the customer's
    turn rather than from a template: a decline carries no figure at all, so any
    digit or code in it is an invented fee, rate or phone number.
    """

    text = candidate.strip()
    if not text or len(text) > 320:
        return None
    if "{" in text or "}" in text:
        return None
    if _DIGIT_RUN.search(text) or _CODE.search(text):
        return None  # a fee, a rate or a phone number we do not hold
    if (language is Language.AR) is not bool(_ARABIC.search(text)):
        return None
    if _SERVICE_DESK[language] not in text:
        return None  # dropped the one thing the customer can act on
    return text


# ------------------------------------------------------------------- rewrite

# Rewrites are cached: conversational replies repeat constantly, and a cache hit
# costs nothing while a model call costs the customer latency.
_CACHE: dict[tuple[str, str, Language], str] = {}
_CACHE_MAX = 512


def rewrite(key: str, text: str, language: Language) -> str:
    """Re-word a conversational reply, or return ``text`` unchanged.

    Raises when ``key`` is money-critical: the tier split is enforced here rather
    than trusted at every call site.
    """

    if tier_of(key) is Tier.CRITICAL:
        raise ValueError(f"{key!r} is money-critical and must never be rewritten")
    if not settings.reply_rewrite_enabled:
        return text
    cached = _CACHE.get((key, text, language))
    if cached is not None:
        return cached

    from app.llm import get_llm_handler

    handler = get_llm_handler()
    if handler is None:
        return text
    candidate = handler.rephrase(text, language.value, settings.reply_rewrite_timeout)
    safe = guard(text, candidate, language) if candidate else None
    if safe is None:
        logger.info("reply rewrite rejected for %s", key)
        return text
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[(key, text, language)] = safe
    return safe


def declined(key: str, turn: str, template: str, language: Language) -> str:
    """Word a decline from the customer's own turn, falling back to ``template``.

    The model may open on what the customer said ("يارب ما تشوف شر") before saying
    the information is not ours; :func:`guard_decline` drops anything that states
    a figure or forgets customer service.
    """

    if tier_of(key) is Tier.CRITICAL:
        raise ValueError(f"{key!r} is money-critical and must never be rewritten")
    if not settings.reply_rewrite_enabled:
        return template

    from app.llm import get_llm_handler

    handler = get_llm_handler()
    if handler is None:
        return template
    candidate = handler.decline(turn, language.value, settings.reply_rewrite_timeout)
    safe = guard_decline(candidate, language) if candidate else None
    if safe is None:
        logger.info("decline rewrite rejected for %s", key)
        return template
    return safe


def varied(
    key: str,
    variants: Mapping[Language, Sequence[str]],
    language: Language,
    **fields: str,
) -> str:
    """Render a conversational reply: pick a phrasing, fill it, then rewrite."""

    text = pick(key, variants[language])
    if fields:
        text = text.format(**fields)
    return rewrite(key, text, language)
