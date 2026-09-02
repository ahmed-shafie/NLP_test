"""LiteLLM-backed exception handler.

When the deterministic pipeline cannot resolve an utterance (intent falls back, or
a transfer is missing/garbling slots such as the Arabic word-amount "ألف"), this
module asks a local LLM (via LiteLLM → Ollama by default) to extract the slots as
strict JSON and propose a natural-language clarification. It degrades gracefully:
if the LLM server is unreachable or disabled, ``get_llm_handler`` returns ``None``
and the pipeline keeps its rule/vector behaviour unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache

from app.config import settings
from app.observability import signals
from app.schemas import TransferEntities

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are the fallback parser for a bilingual (English/Arabic) mobile-banking "
    "assistant that only handles MONEY TRANSFERS. You receive a user utterance and "
    "any slots already extracted. Return ONLY a JSON object with these keys:\n"
    '  "intent": "transfer_money" or "fallback"\n'
    '  "amount": number or null (resolve spelled-out amounts, e.g. Arabic '
    '"ألف"=1000, "مليون"=1000000)\n'
    '  "currency": ISO-4217 code (USD, EUR, GBP, EGP, SAR, AED, KWD, QAR) or null\n'
    '  "recipient": the beneficiary name as written, or null\n'
    '  "source_account": e.g. "savings"/"checking", or null\n'
    '  "clarification": one short sentence in the user\'s language: a follow-up '
    "question for any missing transfer slot, or a brief note if it is not a transfer.\n"
    "Do not invent values that are not implied by the utterance. Output JSON only."
)

_RESPONSE_SYSTEM_PROMPT = (
    "You are a bilingual (English/Arabic) mobile-banking assistant. A user asked to "
    "make a money transfer or a related query, but the destination ACCOUNT NUMBER they "
    "provided was NOT found in the bank's beneficiary records. Read their query and "
    "write ONE short, helpful reply in the SAME language as the user. Politely tell "
    "them the account could not be found and ask them to verify the account number or "
    "the beneficiary, and address any other part of their request. Reply with plain "
    "text only (no JSON, no preamble)."
)


_REPHRASE_SYSTEM_PROMPT = (
    "You rephrase replies for a bilingual (English/Arabic) Saudi banking assistant. "
    "Rewrite the reply so it sounds natural and warm, in the SAME language and the "
    "same colloquial register, keeping the SAME meaning and the same question. "
    "Rules: keep every number, code, currency and masked account EXACTLY as given; "
    "never add a number, an account or a promise that is not there; at most two "
    "short sentences; ask at most one question; no financial advice; plain text "
    "only, no preamble and no quotes around the reply."
)


_DECLINE_SYSTEM_PROMPT = (
    "You are a bilingual (English/Arabic) Saudi banking assistant. The customer "
    "asked for something you do NOT handle. Write ONE short reply in the SAME "
    "language and colloquial register as the customer, in three beats: (1) a brief "
    "human line acknowledging their situation — e.g. for an accident or illness "
    'wish them well ("يارب ما تشوف شر"); (2) say plainly that you do not have '
    "this information and customer service can help them with it; (3) name what you "
    "DO handle: transfers, bill payments, balance, beneficiaries. Rules: never state "
    "a fee, rate, policy, phone number, date or ANY number or code — not even an "
    "approximate one; never promise to do it or to pass the request on; no financial "
    "advice; at most three short sentences; plain text only, no preamble, no quotes."
)


def _parse_json_object(content: str) -> dict:
    """Extract the first JSON object from an LLM response (tolerates code fences)."""

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in LLM response: {content!r}")
    return json.loads(match.group(0))


def _coerce_amount(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return amount if amount > 0 else None


@dataclass(frozen=True)
class LLMResult:
    """Normalised output of an LLM exception-handling call."""

    intent: str | None
    amount: Decimal | None
    currency: str | None
    recipient: str | None
    source_account: str | None
    clarification: str | None


class LLMExceptionHandler:
    """Thin LiteLLM wrapper that turns a hard utterance into structured slots."""

    def __init__(
        self, model: str, api_base: str, timeout: float, temperature: float
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.timeout = timeout
        self.temperature = temperature

    def extract(
        self, text: str, language: str, known: TransferEntities
    ) -> LLMResult | None:
        """Call the LLM and return normalised slots, or ``None`` on any failure."""

        import litellm

        payload = json.dumps(
            {
                "utterance": text,
                "language": language,
                "known": {
                    "amount": str(known.amount) if known.amount is not None else None,
                    "currency": known.currency,
                    "recipient": known.recipient,
                    "source_account": known.source_account,
                },
            },
            ensure_ascii=False,
        )
        try:
            response = litellm.completion(
                model=self.model,
                api_base=self.api_base,
                timeout=self.timeout,
                temperature=self.temperature,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                response_format={"type": "json_object"},
            )
            content = response["choices"][0]["message"]["content"] or ""
            data = _parse_json_object(content)
        except Exception as exc:  # noqa: BLE001 - never let the LLM break the request
            logger.warning("LLM exception handler failed: %s", exc)
            signals.record_call(signals.LLM, ok=False)
            return None

        signals.record_call(signals.LLM, ok=True)
        intent = data.get("intent")
        currency = data.get("currency")
        recipient = data.get("recipient")
        source = data.get("source_account")
        clarification = data.get("clarification")
        return LLMResult(
            intent=intent if intent in {"transfer_money", "fallback"} else None,
            amount=_coerce_amount(data.get("amount")),
            currency=str(currency).upper() if currency else None,
            recipient=str(recipient) if recipient else None,
            source_account=str(source) if source else None,
            clarification=str(clarification) if clarification else None,
        )

    def respond_unresolved(
        self, text: str, language: str, account_number: str, known: TransferEntities
    ) -> str | None:
        """Generate a reply when the beneficiary account is not found in the database.

        Returns a short natural-language response in the user's language, or ``None``
        on any failure (so the caller can degrade gracefully).
        """

        import litellm

        payload = json.dumps(
            {
                "utterance": text,
                "language": language,
                "unknown_account_number": account_number,
                "known": {
                    "amount": str(known.amount) if known.amount is not None else None,
                    "currency": known.currency,
                    "recipient": known.recipient,
                },
            },
            ensure_ascii=False,
        )
        try:
            response = litellm.completion(
                model=self.model,
                api_base=self.api_base,
                timeout=self.timeout,
                temperature=self.temperature,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": _RESPONSE_SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
            )
            content = response["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001 - never let the LLM break the request
            logger.warning("LLM beneficiary delegation failed: %s", exc)
            signals.record_call(signals.LLM, ok=False)
            return None
        signals.record_call(signals.LLM, ok=True)
        content = content.strip()
        return content or None

    def decline(self, text: str, language: str, timeout: float) -> str | None:
        """Word a decline for a request the assistant does not handle.

        Unlike :meth:`rephrase` this writes the reply from the customer's own turn,
        so it can open with a line that fits what they said. It carries no money
        fact by construction, and the caller's guard drops any candidate that
        states a number, a code or the wrong script.
        """

        import litellm

        try:
            response = litellm.completion(
                model=self.model,
                api_base=self.api_base,
                timeout=timeout,
                temperature=0.7,
                max_tokens=160,
                messages=[
                    {"role": "system", "content": _DECLINE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"language={language}\ncustomer={text}",
                    },
                ],
            )
            content = response["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001 - never let the LLM break the request
            logger.warning("LLM decline failed: %s", exc)
            signals.record_call(signals.LLM, ok=False)
            return None
        signals.record_call(signals.LLM, ok=True)
        return content.strip() or None

    def rephrase(self, text: str, language: str, timeout: float) -> str | None:
        """Re-word an already-written reply, keeping its meaning and every value.

        Only ever called for ``Tier.CONVERSATIONAL`` replies (see
        ``app.conversation.phrasing``), and the caller discards anything the guard
        does not accept, so a bad rewrite costs a template fallback and nothing more.
        """

        import litellm

        try:
            response = litellm.completion(
                model=self.model,
                api_base=self.api_base,
                timeout=timeout,
                temperature=0.7,
                max_tokens=80,
                messages=[
                    {"role": "system", "content": _REPHRASE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"language={language}\nreply={text}",
                    },
                ],
            )
            content = response["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001 - never let the LLM break the request
            logger.warning("LLM rephrase failed: %s", exc)
            signals.record_call(signals.LLM, ok=False)
            return None
        # A rewrite the guard later rejects is a normal outcome and not counted
        # here: this records whether the model answered at all.
        signals.record_call(signals.LLM, ok=True)
        return content.strip() or None


def _server_reachable(api_base: str, timeout: float = 2.0) -> bool:
    """Cheap probe so a missing local LLM degrades instantly instead of hanging."""

    try:
        urllib.request.urlopen(f"{api_base}/api/tags", timeout=timeout)
        return True
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("LLM server %s unreachable: %s", api_base, exc)
        return False


@lru_cache(maxsize=1)
def get_llm_handler() -> LLMExceptionHandler | None:
    """Lazily build the LLM handler, or ``None`` if disabled/unreachable (cached)."""

    if not settings.llm_enabled:
        return None
    if settings.llm_api_base and not _server_reachable(settings.llm_api_base):
        return None
    try:
        import litellm

        litellm.telemetry = False
        litellm.suppress_debug_info = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("LiteLLM unavailable: %s; LLM exception handler disabled.", exc)
        return None
    return LLMExceptionHandler(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        timeout=settings.llm_timeout,
        temperature=settings.llm_temperature,
    )
