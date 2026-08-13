# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# System packages:
#  - build-essential: native wheels (faiss, etc.)
#  - curl: container healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Python dependencies.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# spaCy English model is a small pip package — bake it into the image so English
# NER works immediately. The larger models (Stanza Arabic and the multilingual
# sentence-transformer) download on first use into the mounted model cache
# (see docker-compose.yml) so they persist across restarts.
RUN python -m spacy download en_core_web_sm

COPY app ./app
COPY README.md ./

# Non-root user + writable model-cache locations (override with a volume mount).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.cache/huggingface /home/appuser/.cache/stanza \
    && chown -R appuser:appuser /app /home/appuser/.cache
ENV HF_HOME=/home/appuser/.cache/huggingface \
    STANZA_RESOURCES_DIR=/home/appuser/.cache/stanza
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single worker keeps the loaded NLP models in one process memory space.
# Scale out with replicas behind a load balancer rather than many workers here.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
