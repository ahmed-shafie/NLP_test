"""service_template — a headless, copy-paste starting point for a new NLU "case".

This package is a **self-contained skeleton** modelled on the production
conversation engine in ``app/conversation`` and the external validation service
in ``banking-core``. It turns a natural-language request (English or Arabic)
into a validated, structured JSON *action object* — exactly like the "transfer
money" and "pay bill" cases in the main app — but with **no GUI and no heavy ML
dependencies** (no spaCy / FAISS / LLM), so it runs anywhere with just FastAPI
and Pydantic.

Use it to stand up a brand-new "service"/case (e.g. *request money*, *open a
card*, *top up*, *schedule a payment*) quickly:

1. Read ``README.md`` (architecture + a step-by-step "add a new case" guide).
2. Copy this folder, rename it, and edit the clearly-marked
   ``# >>> EDIT PER CASE`` blocks.

The pipeline is a small, predictable finite-state machine:

    text ─▶ language detect ─▶ intent detect ─▶ slot extraction
         ─▶ (fill missing slots, one question at a time)
         ─▶ (optional) external pre-flight validation
         ─▶ confirmation (yes/no)
         ─▶ structured JSON action object

Nothing here moves real money or mutates a real system: the engine only
*validates* and *emits* an action object for a downstream system to execute.
"""

from __future__ import annotations

__version__ = "0.1.0"
