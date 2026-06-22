"""Pydantic schemas for the Memory Brain (habits + shortcuts)."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field

from app.schemas import Language


class Shortcut(BaseModel):
    """A user-defined named transfer template, e.g. ``rent`` -> 5000 EGP to landlord."""

    name: str = Field(..., min_length=1, description="Trigger alias, e.g. 'rent'.")
    amount: Decimal | None = Field(default=None, gt=0)
    currency: str | None = None
    recipient: str | None = None
    source_account: str | None = None
    note: str | None = None


class Habits(BaseModel):
    """Habits learned from the user's completed transfers."""

    preferred_currency: str | None = None
    preferred_source_account: str | None = None
    preferred_language: Language | None = None
    favorite_recipient: str | None = None
    last_recipient: str | None = None
    last_currency: str | None = None
    total_transfers: int = 0
    # recipient name -> number of completed transfers (drives the favourite).
    recipient_counts: dict[str, int] = Field(default_factory=dict)
    # Most-recent distinct amounts (newest first), capped to a small window.
    common_amounts: list[Decimal] = Field(default_factory=list)


class UserMemory(BaseModel):
    """The full memory record for a single user."""

    user_id: str
    habits: Habits = Field(default_factory=Habits)
    shortcuts: list[Shortcut] = Field(default_factory=list)


class HabitsUpdate(BaseModel):
    """Editable habit fields (PUT ``/memory/{user_id}/habits``)."""

    preferred_currency: str | None = None
    preferred_source_account: str | None = None
    preferred_language: Language | None = None
    favorite_recipient: str | None = None
