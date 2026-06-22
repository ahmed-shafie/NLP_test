# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# System packages:
#  - build-essential: native wheels (faiss, ctranslate2, etc.)
#  - ffmpeg + libsndfile1: audio decoding for the voice layer (faster-whisper)
#  - curl: container healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Python dependencies. INSTALL_VOICE=1 (default) adds the ASR/TTS stack so the
# /voice page works out of the box; set to 0 for a smaller text-only image.
ARG INSTALL_VOICE=1
COPY requirements.txt requirements-voice.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && if [ "$INSTALL_VOICE" = "1" ]; then pip install -r requirements-voice.txt; fi

# spaCy English model is a small pip package — bake it into the image so English
# NER works immediately. The larger models (Stanza Arabic, the multilingual
# sentence-transformer, and the Whisper ASR model) download on first use into the
# mounted model cache (see docker-compose.yml) so they persist across restarts.
RUN python -m spacy download en_core_web_sm

COPY app ./app
COPY README.md ./

# Non-root user + writable model-cache locations (override with a volume mount).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.cache/huggingface /home/appuser/.cache/stanza \
    && chown -R appuser:appuser /app /home/appuser/.cache
ENV HF_HOME=/home/appuser/.cache/huggingface \
    STANZA_RESOURCES_DIR=/home/appuser/.cache/stanza \
    NLU_VOICE_ENABLED=true
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single worker keeps the loaded NLP/voice models in one process memory space.
# Scale out with replicas behind a load balancer rather than many workers here.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
