"""Memory Brain service: learn habits, resolve shortcuts, and pre-fill slots."""

from __future__ import annotations

import logging
from decimal import Decimal

from app.config import settings
from app.memory.schemas import (
    Habits,
    HabitsUpdate,
    MemoryOverview,
    MemoryStats,
    RecipientCount,
    Shortcut,
    UserMemory,
    UserSummary,
)
from app.memory.store import get_memory_store
from app.schemas import TransferRequest

logger = logging.getLogger(__name__)

# Phrases that ask the assistant to reuse the user's favourite/last recipient.
_USUAL_RECIPIENT = {
    "usual",
    "favorite",
    "favourite",
    "same",
    "regular",
    "المعتاد",
    "المفضل",
    "نفس",
}

_MAX_COMMON_AMOUNTS = 5


def _tokens(text: str) -> list[str]:
    return [t.strip(".,!؟،:").lower() for t in text.split()]


class MemoryBrain:
    """Reads/writes per-user memory and applies it to the conversation engine."""

    def __init__(self) -> None:
        self._store = get_memory_store()

    @property
    def enabled(self) -> bool:
        return settings.memory_enabled

    # ------------------------------- reads -------------------------------- #

    def get_memory(self, user_id: str) -> UserMemory:
        return self._store.get(user_id)

    def overview(self, top_n: int = 10) -> MemoryOverview:
        """Aggregate stats + per-user summaries for the monitoring dashboard."""

        memories = self._store.list_memories()

        currency_distribution: dict[str, int] = {}
        recipient_totals: dict[str, int] = {}
        total_transfers = 0
        total_shortcuts = 0
        users_with_habits = 0
        users: list[UserSummary] = []

        for memory in memories:
            habits = memory.habits
            total_transfers += habits.total_transfers
            total_shortcuts += len(memory.shortcuts)
            if habits.total_transfers > 0:
                users_with_habits += 1
            if habits.preferred_currency:
                currency_distribution[habits.preferred_currency] = (
                    currency_distribution.get(habits.preferred_currency, 0) + 1
                )
            for recipient, count in habits.recipient_counts.items():
                recipient_totals[recipient] = recipient_totals.get(recipient, 0) + count
            users.append(
                UserSummary(
                    user_id=memory.user_id,
                    total_transfers=habits.total_transfers,
                    favorite_recipient=habits.favorite_recipient,
                    preferred_currency=habits.preferred_currency,
                    last_recipient=habits.last_recipient,
                    shortcut_count=len(memory.shortcuts),
                )
            )

        users.sort(key=lambda u: u.total_transfers, reverse=True)
        top_recipients = [
            RecipientCount(recipient=name, count=count)
            for name, count in sorted(
                recipient_totals.items(), key=lambda kv: kv[1], reverse=True
            )[:top_n]
        ]

        stats = MemoryStats(
            total_users=len(memories),
            users_with_habits=users_with_habits,
            total_transfers=total_transfers,
            total_shortcuts=total_shortcuts,
            currency_distribution=currency_distribution,
            top_recipients=top_recipients,
        )
        return MemoryOverview(stats=stats, users=users)

    # ------------------------------- habits ------------------------------- #

    def update_habits(self, user_id: str, update: HabitsUpdate) -> UserMemory:
        memory = self._store.get(user_id)
        habits = memory.habits
        data = update.model_dump(exclude_unset=True)
        for field, value in data.items():
            setattr(habits, field, value)
        self._store.save_habits(user_id, habits)
        return self._store.get(user_id)

    # ----------------------------- shortcuts ------------------------------ #

    def upsert_shortcut(self, user_id: str, shortcut: Shortcut) -> UserMemory:
        self._store.upsert_shortcut(user_id, shortcut)
        return self._store.get(user_id)

    def delete_shortcut(self, user_id: str, name: str) -> bool:
        return self._store.delete_shortcut(user_id, name)

    def resolve_shortcut(self, user_id: str, text: str) -> Shortcut | None:
        """Return a shortcut whose name appears as a word in ``text``."""

        memory = self._store.get(user_id)
        if not memory.shortcuts:
            return None
        tokens = set(_tokens(text))
        lowered = text.lower()
        for shortcut in memory.shortcuts:
            name = shortcut.name.lower()
            # Match a single-word name as a token, or a multi-word name as a phrase.
            if (" " in name and name in lowered) or name in tokens:
                return shortcut
        return None

    # --------------------------- apply to a turn -------------------------- #

    def wants_usual_recipient(self, text: str) -> bool:
        return bool(set(_tokens(text)) & _USUAL_RECIPIENT)

    def default_currency(self, user_id: str) -> str | None:
        return self._store.get(user_id).habits.preferred_currency

    def default_source_account(self, user_id: str) -> str | None:
        return self._store.get(user_id).habits.preferred_source_account

    def favorite_recipient(self, user_id: str) -> str | None:
        habits = self._store.get(user_id).habits
        return habits.favorite_recipient or habits.last_recipient

    # ------------------------------ learning ------------------------------ #

    def learn_from_transfer(self, user_id: str, transfer: TransferRequest) -> None:
        """Update habits from a completed transfer (favourite, currency, amounts)."""

        memory = self._store.get(user_id)
        habits = memory.habits
        habits.total_transfers += 1

        recipient = transfer.recipient
        habits.recipient_counts[recipient] = (
            habits.recipient_counts.get(recipient, 0) + 1
        )
        habits.last_recipient = recipient
        habits.favorite_recipient = self._pick_favorite(habits)

        habits.last_currency = transfer.currency
        habits.preferred_currency = self._pick_currency(habits, transfer.currency)

        if transfer.source_account:
            habits.preferred_source_account = transfer.source_account

        habits.common_amounts = self._update_amounts(
            habits.common_amounts, transfer.amount
        )

        self._store.save_habits(user_id, habits)

    def _pick_favorite(self, habits: Habits) -> str | None:
        if not habits.recipient_counts:
            return None
        recipient, count = max(habits.recipient_counts.items(), key=lambda kv: kv[1])
        if count >= settings.memory_favorite_min_count:
            return recipient
        return habits.favorite_recipient

    @staticmethod
    def _pick_currency(habits: Habits, currency: str) -> str:
        # Once the user has made a few transfers, adopt their latest currency as the
        # default so it pre-fills future turns; otherwise keep any existing preference.
        if habits.total_transfers >= settings.memory_favorite_min_count:
            return currency
        return habits.preferred_currency or currency

    @staticmethod
    def _update_amounts(amounts: list[Decimal], amount: Decimal) -> list[Decimal]:
        deduped = [a for a in amounts if a != amount]
        return [amount, *deduped][:_MAX_COMMON_AMOUNTS]


_brain: MemoryBrain | None = None


def get_memory_brain() -> MemoryBrain:
    global _brain
    if _brain is None:
        _brain = MemoryBrain()
    return _brain
