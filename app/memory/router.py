"""FastAPI routes for the Memory Brain (habits + shortcuts)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import settings
from app.memory.schemas import HabitsUpdate, Shortcut, UserMemory
from app.memory.service import get_memory_brain
from app.security import require_api_key

router = APIRouter(tags=["memory"], prefix="/memory")


def _require_memory() -> None:
    if not settings.memory_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The Memory Brain is disabled.",
        )


@router.get("/{user_id}", response_model=UserMemory)
def get_memory(user_id: str, _: str = Depends(require_api_key)) -> UserMemory:
    """Return a user's learned habits and saved shortcuts."""

    _require_memory()
    return get_memory_brain().get_memory(user_id)


@router.put("/{user_id}/habits", response_model=UserMemory)
def update_habits(
    user_id: str, update: HabitsUpdate, _: str = Depends(require_api_key)
) -> UserMemory:
    """Manually set habit defaults (currency, source account, favourite, language)."""

    _require_memory()
    return get_memory_brain().update_habits(user_id, update)


@router.put("/{user_id}/shortcuts", response_model=UserMemory)
def upsert_shortcut(
    user_id: str, shortcut: Shortcut, _: str = Depends(require_api_key)
) -> UserMemory:
    """Create or update a named transfer shortcut for the user."""

    _require_memory()
    return get_memory_brain().upsert_shortcut(user_id, shortcut)


@router.delete("/{user_id}/shortcuts/{name}", response_model=UserMemory)
def delete_shortcut(
    user_id: str, name: str, _: str = Depends(require_api_key)
) -> UserMemory:
    """Delete a shortcut by name."""

    _require_memory()
    brain = get_memory_brain()
    if not brain.delete_shortcut(user_id, name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No shortcut named '{name}' for this user.",
        )
    return brain.get_memory(user_id)
