# Banking NLU Brain

An NLU (Natural Language Understanding) microservice for a mobile-banking AI assistant, focused on **money-transfer** intent.

## Features

- **Bilingual** — handles English (spaCy) and Arabic (Stanza) natively.
- **Semantic intent classification** — a FAISS vector index over multilingual sentence embeddings classifies intent by nearest labeled examples; falls back to keyword matching.
- **Cross-lingual contact matching** — resolves a spoken/typed recipient to an address-book contact across scripts (e.g. `Ahmed` → `أحمد حسن`); falls back to fuzzy string matching.
- **Entity/slot extraction** — extracts `amount`, `currency`, `recipient`, `source_account`, `note`.
- **Strict validation** — Pydantic schemas enforce business rules and produce human-friendly prompts for missing/invalid slots.
- **Graceful degradation** — runs in regex/fuzzy-only mode when NLP/embedding models are not downloaded.

## Tech Stack

| Layer | Tool |
|-------|------|
| API | FastAPI |
| Validation | Pydantic v2 |
| English NLU | spaCy |
| Arabic NLU | Stanza |
| Embeddings | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector DB | FAISS (in-process) |

## Quick Start

```bash
# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLP models (optional — service works without them via regex/fuzzy)
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('ar')"
# The multilingual embedding model auto-downloads on first use (~470MB).

# Run the server
uvicorn app.main:app --reload --port 8000
```

## Web simulator

Open <http://localhost:8000/> in a browser for a built-in simulation & testing
page. It lets you try utterances in English or Arabic (with one-click samples),
shows the parsed intent, `intent_source`, confidence, extracted entities and the
resolved contact, and has panels for `/nlu/similar` and `/contacts/resolve`. The
raw JSON for each call is available under a collapsible section. Interactive
OpenAPI docs remain at <http://localhost:8000/docs>.

## How semantic NLU works

```
utterance ── embed ──▶ FAISS index of labeled examples ──▶ top-k neighbours ──▶ aggregate per intent
recipient ── embed ──▶ FAISS index of contact names    ──▶ best match (cosine ≥ threshold)
```

Labeled intent examples live in `app/nlu/examples.py` (English + Arabic). The demo
address book lives in `app.config.DEMO_CONTACTS`. Because the embedding model is
multilingual, Arabic and English map into a shared vector space, so a transfer
phrased in either language lands near the same examples and `Ahmed`/`أحمد` resolve
to the same contact. Set `NLU_USE_SEMANTIC_INTENT=false` to force the keyword path.

## API Endpoints

### `POST /nlu/parse`

Parse a user utterance into an intent and extracted entities.

**Request:**
```json
{"text": "transfer 500 dollars to John"}
```

**Response:**
```json
{
  "text": "transfer 500 dollars to John",
  "language": "en",
  "intent": "transfer_money",
  "confidence": 0.94,
  "intent_source": "semantic",
  "entities": {
    "amount": "500",
    "currency": "USD",
    "recipient": "John",
    "source_account": null,
    "note": null
  },
  "resolved_recipient": {
    "contact": {"id": "c5", "name": "Sara Adel", "account": "EG1003"},
    "score": 0.71
  }
}
```

### `GET /nlu/similar`

Inspect the nearest labeled example utterances for a query (semantic debug/eval).

`GET /nlu/similar?text=send money to Sara&k=3` →
```json
[
  {"text": "send money to my friend Sara", "intent": "transfer_money", "score": 0.82},
  {"text": "can you send 50 to Mohamed", "intent": "transfer_money", "score": 0.74}
]
```

### `POST /contacts/resolve`

Resolve a recipient name to an address-book contact (cross-lingual). Pass your own
`contacts` or omit to use the demo address book.

**Request:**
```json
{"name": "Ahmed", "contacts": [{"id": "1", "name": "أحمد حسن", "account": "A1"}]}
```

**Response:**
```json
{"matched": {"contact": {"id": "1", "name": "أحمد حسن", "account": "A1"}, "score": 0.85}, "candidates": [...]}
```

### `POST /transfer/validate`

Validate gathered slots and return a ready transfer or follow-up prompts.

**Request:**
```json
{"amount": 500, "currency": "USD", "recipient": "John"}
```

**Response (success):**
```json
{"valid": true, "transfer": {"amount": "500", "currency": "USD", "recipient": "John"}, "missing": [], "errors": []}
```

**Response (missing slots):**
```json
{"valid": false, "transfer": null, "missing": ["recipient"], "errors": [...]}
```

### `GET /health`

Returns `{"status": "ok", "version": "0.1.0"}`.

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

## Project Structure

```
app/
  main.py         — FastAPI application
  config.py       — Settings, supported currencies
  schemas.py      — Pydantic models (request/response, validation)
  embeddings.py   — Multilingual sentence-embedding wrapper (lazy, cached)
  vectorstore.py  — Generic FAISS cosine-similarity index
  nlu/
    pipeline.py        — Orchestration (lang detect → intent → entities → contact)
    lang.py            — Arabic vs. English language detection
    intents.py         — Keyword-based intent classifier (fallback)
    semantic_intents.py— FAISS + embeddings intent classifier
    examples.py        — Labeled example utterances (EN + AR)
    contacts.py        — Semantic/fuzzy contact matcher
    entities.py        — Regex-based slot extraction
    english.py         — spaCy NER augmentation
    arabic.py          — Stanza NER augmentation
tests/
  test_lang.py
  test_intents.py
  test_entities.py
  test_pipeline.py
  test_api.py
  test_vectorstore.py
  test_semantic.py
  test_contacts_fuzzy.py
```

## License

MIT
