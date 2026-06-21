"""Speech-to-text via faster-whisper. Degrades gracefully when unavailable."""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import settings
from app.schemas import Language

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model():
    """Load the faster-whisper model once, returning None if it can't be loaded."""

    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415 - optional dependency
    except ImportError:
        logger.warning("faster-whisper not installed; ASR disabled")
        return None
    try:
        return WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    except Exception as exc:  # noqa: BLE001 - any load error disables ASR
        logger.warning(
            "Failed to load whisper model '%s': %s", settings.whisper_model, exc
        )
        return None


def asr_available() -> bool:
    return settings.voice_enabled and _get_model() is not None


def transcribe(audio_path: str, language: Language | None = None) -> str | None:
    """Transcribe an audio file to text, or None if ASR is unavailable/failed."""

    model = _get_model()
    if model is None:
        return None
    try:
        lang_code = language.value if language is not None else None
        segments, _info = model.transcribe(audio_path, language=lang_code)
        return "".join(segment.text for segment in segments).strip()
    except Exception as exc:  # noqa: BLE001 - transcription must not crash the request
        logger.warning("ASR transcription failed: %s", exc)
        return None
