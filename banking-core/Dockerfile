# syntax=docker/dockerfile:1
# Standalone Banking Core service (accounts, beneficiaries, billers).
# Deliberately slim: no NLP models, just FastAPI + SQLAlchemy + the Postgres driver.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

RUN pip install --upgrade pip \
    && pip install \
        "fastapi==0.115.6" \
        "uvicorn[standard]==0.34.0" \
        "pydantic==2.10.4" \
        "pydantic-settings==2.7.1" \
        "SQLAlchemy==2.0.36" \
        "psycopg[binary]==3.2.3"

COPY banking_core ./banking_core
COPY README.md ./

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8100/health || exit 1

CMD ["uvicorn", "banking_core.main:app", "--host", "0.0.0.0", "--port", "8100"]
