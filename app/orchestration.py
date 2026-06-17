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
from app.llm import get_llm_handler
from app.nlu import arabic, english
from app.nlu.contacts import get_default_matcher
from app.nlu.intents import classify_intent
from app.nlu.lang import detect_language
from app.nlu.semantic_intents import get_semantic_classifier
from app.schemas import (
    ContactMatch,
    Intent,
    Language,
    NLUResponse,
    TransferEntities,
)

logger = logging.getLogger(__name__)


@component
class LanguageDetector:
    """Resolve the utterance language, honouring an explicit hint when given."""

    @component.output_types(state=dict)
    def run(self, state: dict) -> dict:
        lang = state.get("language") or detect_language(state["text"])
        state["language"] = lang
        return {"state": state}


@component
class IntentClassifier:
    """Classify intent via the semantic classifier, falling back to keywords."""

    @component.output_types(state=dict)
    def run(self, state: dict) -> dict:
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
        state.update(intent=intent, confidence=confidence, intent_source=source)
        return {"state": state}


@component
class EntityExtractor:
    """Extract transfer slots for transfer utterances (spaCy EN / Stanza AR)."""

    @component.output_types(state=dict)
    def run(self, state: dict) -> dict:
        if state["intent"] is Intent.TRANSFER_MONEY:
            module = arabic if state["language"] is Language.AR else english
            entities = module.extract_entities(state["text"])
            if entities.amount is not None and entities.currency is None:
                entities.currency = DEFAULT_CURRENCY
        else:
            entities = TransferEntities()
        state["entities"] = entities
        return {"state": state}


@component
class ContactResolver:
    """Resolve the extracted recipient against the demo address book."""

    @component.output_types(state=dict)
    def run(self, state: dict) -> dict:
        entities: TransferEntities = state["entities"]
        resolved: ContactMatch | None = None
        if entities.recipient:
            resolved, _ = get_default_matcher().resolve(entities.recipient)
        state["resolved"] = resolved
        return {"state": state}


def _needs_llm(state: dict) -> bool:
    """The LLM handler fires only when the deterministic path is incomplete."""

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

        if not _needs_llm(state):
            return {"state": state}
        handler = get_llm_handler()
        if handler is None:
            return {"state": state}

        lang = state["language"]
        result = handler.extract(
            state["text"],
            lang.value if isinstance(lang, Language) else str(lang),
            state["entities"],
        )
        if result is None:
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
    pipe.add_component("llm", LLMExceptionHandler())

    pipe.connect("detect.state", "intent.state")
    pipe.connect("intent.state", "entities.state")
    pipe.connect("entities.state", "contacts.state")
    pipe.connect("contacts.state", "llm.state")
    return pipe


def run_pipeline(text: str, language: Language | None = None) -> NLUResponse:
    """Run the Haystack NLU pipeline over a single utterance."""

    initial = {"text": text, "language": language}
    result = get_nlu_pipeline().run({"detect": {"state": initial}})
    state = result["llm"]["state"]
    return NLUResponse(
        text=text,
        language=state["language"],
        intent=state["intent"],
        confidence=state["confidence"],
        intent_source=state["intent_source"],
        entities=state["entities"],
        resolved_recipient=state.get("resolved"),
        llm_assisted=state.get("llm_assisted", False),
        clarification=state.get("clarification"),
    )
