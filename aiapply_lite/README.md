# AIApply Lite

A personal, **local-first** job-search assistant inspired by aiapply.co — built to
run on your own PC with no per-use cost and full privacy.

## Features

- **My CV** — paste your resume once; it's parsed (skills, title, years) and saved locally.
- **Job Search + Fit Score** — search real listings and get a 0–100 match score per job
  (semantic similarity via embeddings + hard-skill keyword overlap), with a "what matched /
  what's missing" breakdown.
- **Company Deep-Dive** — what a company does, its history, products, tech stack and recent news.
- **Success Roadmap** — a personalized gap analysis + 30/60/90-day plan to land and succeed
  in a specific role at a specific company.
- **Tailored resume + cover letter** — generated per job posting.

Everything runs against a **local Ollama** model by default, so there are no API keys or costs.

> **Note on auto-apply:** this tool intentionally does **not** auto-submit applications to
> LinkedIn/Indeed etc. — that violates their terms of service. It's a "prepare & assist" tool.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with:
  ```bash
  ollama pull qwen2.5:3b          # generation
  ollama pull nomic-embed-text    # embeddings
  ```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp aiapply_lite/.env.example .env   # optional; sensible defaults otherwise
```

## Run

```bash
streamlit run aiapply_lite/app.py
# open http://localhost:8501
```

## Configuration

All optional — see `.env.example`. Highlights:

| Variable | Purpose | Default |
|---|---|---|
| `AIAPPLY_LLM_MODEL` | Ollama generation model | `qwen2.5:3b` |
| `AIAPPLY_EMBED_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `AIAPPLY_JOB_SOURCES` | `remotive,remoteok,adzuna,mock` | `remotive,remoteok` |
| `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` | enable the Adzuna source | _(unset)_ |
| `TAVILY_API_KEY` | live web context for company research | _(unset)_ |

Without keys the app still works: keyless job sources + the LLM's own knowledge for
company research. Every feature degrades gracefully if Ollama is offline.

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
mypy aiapply_lite
pytest -q
```
