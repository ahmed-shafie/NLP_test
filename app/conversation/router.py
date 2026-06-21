"""FastAPI routes for the multi-turn conversation and voice endpoints."""

from __future__ import annotations

import base64
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.conversation.engine import ConversationResult, get_engine
from app.conversation.schemas import (
    ConversationRequest,
    ConversationResponse,
    VoiceResponse,
)
from app.conversation.state import ConversationStatus
from app.schemas import Language
from app.security import require_api_key
from app.voice import asr, tts

router = APIRouter(tags=["conversation"])


def _to_response(result: ConversationResult) -> ConversationResponse:
    state = result.state
    return ConversationResponse(
        session_id=state.session_id,
        reply=result.reply,
        status=state.status,
        language=state.language,
        intent=state.intent,
        pending_slot=state.pending_slot,
        complete=state.status is ConversationStatus.COMPLETED,
        slots=state.slots,
        transfer=result.transfer,
    )


def _require_conversation() -> None:
    if not settings.conversation_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The conversation engine is disabled.",
        )


@router.post("/conversation/text", response_model=ConversationResponse)
def conversation_text(
    request: ConversationRequest, _: str = Depends(require_api_key)
) -> ConversationResponse:
    """Advance a multi-turn transfer dialogue with a single text message."""

    _require_conversation()
    result = get_engine().handle(request.text, request.session_id, request.language)
    return _to_response(result)


@router.post("/conversation/voice", response_model=VoiceResponse)
async def conversation_voice(
    audio: UploadFile = File(..., description="Audio clip of the user's message."),
    session_id: str | None = Form(default=None),
    language: Language | None = Form(default=None),
    _: str = Depends(require_api_key),
) -> VoiceResponse:
    """Transcribe an audio clip, advance the dialogue, and synthesize a spoken reply."""

    _require_conversation()
    if not asr.asr_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech recognition is unavailable on this deployment.",
        )

    data = await audio.read()
    tmp_path = None
    try:
        suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        # ASR is blocking (and TTS below runs its own event loop), so both run in a
        # worker thread to avoid blocking the request loop and the asyncio.run() clash.
        transcript = await run_in_threadpool(asr.transcribe, tmp_path, language)
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not transcribe the supplied audio.",
        )

    result = await run_in_threadpool(
        get_engine().handle, transcript, session_id, language
    )
    base_response = _to_response(result)

    audio_b64: str | None = None
    audio_mime: str | None = None
    synthesized = await run_in_threadpool(
        tts.synthesize, result.reply, result.state.language
    )
    if synthesized is not None:
        audio_b64 = base64.b64encode(synthesized[0]).decode("ascii")
        audio_mime = synthesized[1]

    return VoiceResponse(
        **base_response.model_dump(),
        transcript=transcript,
        audio_base64=audio_b64,
        audio_mime=audio_mime,
    )
