"""Text-to-speech via edge-tts (preferred) or pyttsx3. Degrades gracefully."""

from __future__ import annotations

import asyncio
import logging

from app.config import settings
from app.schemas import Language

logger = logging.getLogger(__name__)


def _voice_for(language: Language) -> str:
    return settings.tts_voice_ar if language is Language.AR else settings.tts_voice_en


def _synthesize_edge(text: str, language: Language) -> bytes | None:
    try:
        import edge_tts  # noqa: PLC0415 - optional dependency
    except ImportError:
        return None

    async def _run() -> bytes:
        communicate = edge_tts.Communicate(text, _voice_for(language))
        chunks = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.extend(chunk["data"])
        return bytes(chunks)

    try:
        audio = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - network/runtime errors disable edge-tts
        logger.warning("edge-tts synthesis failed: %s", exc)
        return None
    return audio or None


def _synthesize_pyttsx3(text: str) -> bytes | None:
    try:
        import tempfile

        import pyttsx3  # noqa: PLC0415 - optional dependency
    except ImportError:
        return None
    try:
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            engine.save_to_file(text, tmp.name)
            engine.runAndWait()
            tmp.seek(0)
            return tmp.read() or None
    except Exception as exc:  # noqa: BLE001 - missing espeak etc. disables pyttsx3
        logger.warning("pyttsx3 synthesis failed: %s", exc)
        return None


def tts_available() -> bool:
    if not settings.voice_enabled:
        return False
    try:
        import edge_tts  # noqa: F401, PLC0415

        return True
    except ImportError:
        pass
    try:
        import pyttsx3  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def synthesize(text: str, language: Language) -> tuple[bytes, str] | None:
    """Return ``(audio_bytes, mime_type)`` for the reply, or None when TTS is off."""

    if not settings.voice_enabled or not text.strip():
        return None
    if settings.tts_engine == "edge-tts":
        audio = _synthesize_edge(text, language)
        if audio is not None:
            return audio, "audio/mpeg"
    audio = _synthesize_pyttsx3(text)
    if audio is not None:
        return audio, "audio/wav"
    return None
