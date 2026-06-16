# Banking NLU Brain

An NLU (Natural Language Understanding) microservice for a mobile-banking AI assistant, focused on **money-transfer** intent.

## Features

- **Bilingual** — handles English (spaCy) and Arabic (Stanza) natively.
- **Intent classification** — detects `transfer_money` with keyword-pattern matching (extensible to ML classifiers).
- **Entity/slot extraction** — extracts `amount`, `currency`, `recipient`, `source_account`, `note`.
- **Strict validation** — Pydantic schemas enforce business rules and produce human-friendly prompts for missing/invalid slots.
- **Graceful degradation** — runs in regex-only mode when NLP models are not downloaded.

## Tech Stack

| Layer | Tool |
|-------|------|
| API | FastAPI |
| Validation | Pydantic v2 |
| English NLU | spaCy |
| Arabic NLU | Stanza |

## Quick Start

```bash
# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLP models (optional — service works without them via regex)
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('ar')"

# Run the server
uvicorn app.main:app --reload --port 8000
```

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
  "confidence": 0.6,
  "entities": {
    "amount": "500",
    "currency": "USD",
    "recipient": "John",
    "source_account": null,
    "note": null
  }
}
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
  nlu/
    pipeline.py   — Orchestration (lang detect → intent → entities)
    lang.py       — Arabic vs. English language detection
    intents.py    — Keyword-based intent classifier
    entities.py   — Regex-based slot extraction
    english.py    — spaCy NER augmentation
    arabic.py     — Stanza NER augmentation
tests/
  test_lang.py
  test_intents.py
  test_entities.py
  test_pipeline.py
  test_api.py
```

## License

MIT
