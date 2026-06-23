"""Haystack-orchestrated NLU pipeline.

The deterministic NLU steps (language detection, intent classification, entity
extraction, contact resolution) are wrapped as Haystack ``@component``s and wired
into a :class:`haystack.Pipeline`. A final LiteLLM exception-handler component runs
only when the deterministic path fails, filling missing slots and/or proposing a
clarification. Each component enriches a shared ``state`` dict that flows through
the graph, so the pipeline stays a simple, debuggable linear DAG.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from haystack import Pipeline, component

from app.config import DEFAULT_CURRENCY, settings
from app.db.beneficiary import get_beneficiary_repository
from app.llm import get_llm_handler
from app.nlu import arabic, english
from app.nlu.contacts import get_default_matcher
from app.nlu.intents import classify_intent
from app.nlu.lang import detect_language
from app.nlu.semantic_intents import get_semantic_classifier
from app.schemas import (
    Beneficiary,
    ContactMatch,
    Intent,
    Language,
    NLUResponse,
    TransferEntities,
)
from app.trace import BlockTracer

logger = logging.getLogger(__name__)


def _tracer(state: dict) -> BlockTracer:
    """Return the request-scoped tracer threaded through the pipeline state."""

    return state["tracer"]


@component
class LanguageDetector:
    """Resolve the utterance language, honouring an explicit hint when given."""

    @component.output_types(state=dict)
    def run(self, state: dict) -> dict:
        with _tracer(state).block("language_detection") as span:
            if state.get("language") is not None:
                span.annotate("hint")
            lang = state.get("language") or detect_language(state["text"])
            state["language"] = lang
        return {"state": state}


@component
class IntentClassifier:
    """Classify intent via the semantic classifier, falling back to keywords."""

    @component.output_types(state=dict)
    def run(self, state: dict) -> dict:
        with _tracer(state).block("intent_classification") as span:
            text, lang = state["text"], state["language"]
            classifier = get_semantic_classifier()
            if classifier is not None:
                intent, confidence = classifier.classify(text)
                source = "semantic"
            else:
                intent, confidence = classify_intent(text, lang)
                if confidence < settings.intent_threshold:
                    intent = Intent.FALLBACK
                source = "keyword"
            span.annotate(f"{source}:{intent.value}")
            state.update(intent=intent, confidence=confidence, intent_source=source)
        return {"state": state}


@component
class EntityExtractor:
    """Extract transfer slots for transfer utterances (spaCy EN / Stanza AR)."""

    @component.output_types(state=dict)
    def run(self, state: dict) -> dict:
        with _tracer(state).block("entity_extraction") as span:
            if state["intent"] is Intent.TRANSFER_MONEY:
                module = arabic if state["language"] is Language.AR else english
                entities = module.extract_entities(state["text"])
                if entities.amount is not None and entities.currency is None:
                    entities.currency = DEFAULT_CURRENCY
            else:
                span.skip("non-transfer intent")
                entities = TransferEntities()
            state["entities"] = entities
        return {"state": state}


@component
class ContactResolver:
    """Resolve the extracted recipient against the demo address book."""

    @component.output_types(state=dict)
    def run(self, state: dict) -> dict:
        with _tracer(state).block("contact_resolution") as span:
            entities: TransferEntities = state["entities"]
            resolved: ContactMatch | None = None
            if entities.recipient:
                resolved, _ = get_default_matcher().resolve(entities.recipient)
            else:
                span.skip("no recipient extracted")
            state["resolved"] = resolved
        return {"state": state}


@component
class BeneficiaryLookup:
    """Resolve the destination beneficiary by account number against the database.

    When the request supplies an ``account_number``, look it up via the configured
    provider. A hit populates ``resolved_beneficiary`` (authoritative) and fills the
    recipient/currency slots; a miss (or an unavailable database) marks the request as
    ``beneficiary_unresolved`` so the LLM handler can take over the response.
    """

    @component.output_types(state=dict)
    def run(self, state: dict) -> dict:
        state.setdefault("beneficiary", None)
        state.setdefault("beneficiary_source", None)
        state.setdefault("beneficiary_unresolved", False)

        with _tracer(state).block("beneficiary_lookup") as span:
            account = state.get("account_number")
            if not account:
                span.skip("no account number supplied")
                return {"state": state}

            repo = get_beneficiary_repository()
            beneficiary = repo.lookup(account) if repo is not None else None
            if beneficiary is not None:
                state["beneficiary"] = beneficiary
                state["beneficiary_source"] = "database"
                entities: TransferEntities = state["entities"]
                if not entities.recipient:
                    entities.recipient = beneficiary.name
                if beneficiary.currency and entities.currency is None:
                    entities.currency = beneficiary.currency
            else:
                # Account given but not found (or DB unavailable) -> delegate to LLM.
                span.annotate("account not found")
                state["beneficiary_unresolved"] = True
        return {"state": state}


def _needs_llm(state: dict) -> bool:
    """The LLM handler fires only when the deterministic path is incomplete."""

    if state["intent"] is Intent.PAY_BILL:
        # Bill slots are extracted by the conversation engine, not the LLM.
        return False
    if state["intent"] is Intent.SMALL_TALK:
        # Chit-chat is answered with canned warm replies, not the LLM.
        return False
    if state["intent"] is Intent.FALLBACK:
        return True
    entities: TransferEntities = state["entities"]
    return entities.amount is None or entities.recipient is None


@component
class LLMExceptionHandler:
    """LiteLLM safety net: fill missing slots / clarify when rules fall short."""

    @component.output_types(state=dict)
    def run(self, state: dict) -> dict:
        state.setdefault("llm_assisted", False)
        state.setdefault("clarification", None)

        with _tracer(state).block("llm_fallback") as span:
            lang = state["language"]
            lang_code = lang.value if isinstance(lang, Language) else str(lang)
            handler = get_llm_handler()

            # Beneficiary delegation takes priority: when an account number was
            # supplied but not found, let the LLM process the query and respond.
            if state.get("beneficiary_unresolved"):
                if handler is not None:
                    message = handler.respond_unresolved(
                        state["text"],
                        lang_code,
                        str(state.get("account_number") or ""),
                        state["entities"],
                    )
                    if message:
                        state["clarification"] = message
                        state["beneficiary_source"] = "llm"
                        state["llm_assisted"] = True
                else:
                    span.skip("llm disabled")
                return {"state": state}

            if not _needs_llm(state):
                span.skip("deterministic path complete")
                return {"state": state}
            if handler is None:
                span.skip("llm disabled")
                return {"state": state}

            result = handler.extract(state["text"], lang_code, state["entities"])
            if result is None:
                span.annotate("no extraction")
                return {"state": state}

            entities: TransferEntities = state["entities"]
            changed = False

            if state["intent"] is Intent.FALLBACK and result.intent == "transfer_money":
                state["intent"] = Intent.TRANSFER_MONEY
                changed = True

            for field in ("amount", "currency", "recipient", "source_account"):
                current = getattr(entities, field)
                new_value = getattr(result, field)
                if (current is None or current == "") and new_value is not None:
                    setattr(entities, field, new_value)
                    changed = True

            if entities.amount is not None and entities.currency is None:
                entities.currency = DEFAULT_CURRENCY

            # Re-resolve the contact if the LLM supplied a recipient the rules missed.
            if changed and entities.recipient and state.get("resolved") is None:
                state["resolved"], _ = get_default_matcher().resolve(entities.recipient)

            state["clarification"] = result.clarification
            state["llm_assisted"] = changed or bool(result.clarification)
        return {"state": state}


@lru_cache(maxsize=1)
def get_nlu_pipeline() -> Pipeline:
    """Build (once) the Haystack pipeline that orchestrates the NLU components."""

    pipe = Pipeline()
    pipe.add_component("detect", LanguageDetector())
    pipe.add_component("intent", IntentClassifier())
    pipe.add_component("entities", EntityExtractor())
    pipe.add_component("contacts", ContactResolver())
    pipe.add_component("beneficiary", BeneficiaryLookup())
    pipe.add_component("llm", LLMExceptionHandler())

    pipe.connect("detect.state", "intent.state")
    pipe.connect("intent.state", "entities.state")
    pipe.connect("entities.state", "contacts.state")
    pipe.connect("contacts.state", "beneficiary.state")
    pipe.connect("beneficiary.state", "llm.state")
    return pipe


def run_pipeline(
    text: str,
    language: Language | None = None,
    account_number: str | None = None,
) -> NLUResponse:
    """Run the Haystack NLU pipeline over a single utterance.

    Each block records a :class:`~app.trace.BlockTrace` entry (timing + status) on
    the shared tracer, and the Active Learning block runs passively at the end to
    log review-worthy cases. Both are surfaced on ``NLUResponse.block_trace``.
    """

    tracer = BlockTracer()
    initial = {
        "text": text,
        "language": language,
        "account_number": account_number,
        "tracer": tracer,
    }
    result = get_nlu_pipeline().run({"detect": {"state": initial}})
    state = result["llm"]["state"]
    # Haystack passes a copy of the state between components, so the tracer that
    # actually collected the per-block entries is the one on the final state.
    tracer = state["tracer"]
    beneficiary: Beneficiary | None = state.get("beneficiary")
    response = NLUResponse(
        text=text,
        language=state["language"],
        intent=state["intent"],
        confidence=state["confidence"],
        intent_source=state["intent_source"],
        entities=state["entities"],
        resolved_recipient=state.get("resolved"),
        resolved_beneficiary=beneficiary,
        beneficiary_source=state.get("beneficiary_source"),
        llm_assisted=state.get("llm_assisted", False),
        clarification=state.get("clarification"),
    )

    # Active Learning: passively log this case to the review queue (never fatal).
    with tracer.block("active_learning") as span:
        from app.active_learning.service import record_case

        logged = record_case(response, source="nlu.parse")
        if logged is None:
            span.skip("not a learning case")
        else:
            span.annotate(logged.status)

    response.block_trace = tracer.entries
    return response
