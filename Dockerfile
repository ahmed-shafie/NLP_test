# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# System packages: build tools for native wheels; ffmpeg/libsndfile reserved for the
# voice layer (faster-whisper). curl is used by the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY README.md ./

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Multiple workers for production throughput; tune via $WEB_CONCURRENCY.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
