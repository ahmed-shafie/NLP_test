"""Durable SQLite/SQL store for the Active Learning review queue.

A small SQLAlchemy store (SQLite by default, any provider via
``NLU_ACTIVE_LEARNING_STORE_URL``) holding every logged case and its review
status. Approved (and auto-approved) cases become extra training examples for the
semantic intent index.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.active_learning.schemas import (
    ActiveLearningStats,
    CaseStatus,
    ReviewCase,
)
from app.config import settings
from app.schemas import Intent, Language

logger = logging.getLogger(__name__)

_APPROVED = (CaseStatus.APPROVED.value, CaseStatus.AUTO_APPROVED.value)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for the active-learning store."""


class ReviewCaseRow(Base):
    """A single logged case in the review queue."""

    __tablename__ = "active_learning_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, index=True, nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(8), default="en", nullable=False)
    predicted_intent: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    intent_source: Mapped[str] = mapped_column(String(16), default="semantic")
    llm_assisted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    clarification: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), default=CaseStatus.PENDING.value, index=True, nullable=False
    )
    corrected_intent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="nlu.parse")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (once) the configured store engine and ensure tables exist."""

    url = settings.active_learning_store_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def _to_case(row: ReviewCaseRow) -> ReviewCase:
    return ReviewCase(
        id=row.id,
        created_at=row.created_at,
        text=row.text,
        language=Language(row.language),
        predicted_intent=Intent(row.predicted_intent),
        confidence=row.confidence,
        intent_source=row.intent_source,
        llm_assisted=row.llm_assisted,
        clarification=row.clarification,
        status=CaseStatus(row.status),
        corrected_intent=(
            Intent(row.corrected_intent) if row.corrected_intent else None
        ),
        reviewer=row.reviewer,
        reviewed_at=row.reviewed_at,
        source=row.source,
    )


class ActiveLearningStore:
    """Persistence operations for the review queue."""

    def __init__(self) -> None:
        self._sessionmaker = get_sessionmaker()

    # ------------------------------- writes ------------------------------- #

    def log_case(
        self,
        *,
        text: str,
        language: Language,
        predicted_intent: Intent,
        confidence: float,
        intent_source: str,
        llm_assisted: bool,
        clarification: str | None,
        status: CaseStatus,
        source: str,
    ) -> ReviewCase:
        """Insert a new case and return it."""

        with self._sessionmaker() as session:
            row = ReviewCaseRow(
                text=text,
                language=language.value,
                predicted_intent=predicted_intent.value,
                confidence=confidence,
                intent_source=intent_source,
                llm_assisted=llm_assisted,
                clarification=clarification,
                status=status.value,
                source=source,
            )
            session.add(row)
            session.commit()
            return _to_case(row)

    def decide(
        self,
        case_id: int,
        status: CaseStatus,
        *,
        corrected_intent: Intent | None = None,
        reviewer: str | None = None,
    ) -> ReviewCase | None:
        """Apply a review decision (approve/reject) to a case."""

        with self._sessionmaker() as session:
            row = session.get(ReviewCaseRow, case_id)
            if row is None:
                return None
            row.status = status.value
            row.corrected_intent = corrected_intent.value if corrected_intent else None
            row.reviewer = reviewer
            row.reviewed_at = _utcnow()
            session.commit()
            return _to_case(row)

    # ------------------------------- reads -------------------------------- #

    def list_cases(
        self, status: CaseStatus | None = None, limit: int = 100
    ) -> list[ReviewCase]:
        """Return cases (newest first), optionally filtered by status."""

        stmt = select(ReviewCaseRow).order_by(ReviewCaseRow.created_at.desc())
        if status is not None:
            stmt = stmt.where(ReviewCaseRow.status == status.value)
        stmt = stmt.limit(limit)
        with self._sessionmaker() as session:
            rows = session.execute(stmt).scalars().all()
        return [_to_case(r) for r in rows]

    def get_case(self, case_id: int) -> ReviewCase | None:
        with self._sessionmaker() as session:
            row = session.get(ReviewCaseRow, case_id)
            return _to_case(row) if row is not None else None

    def approved_examples(self) -> list[tuple[str, Intent]]:
        """Return ``(text, intent)`` pairs for every approved/auto-approved case.

        The effective label is the reviewer's correction when present, otherwise
        the predicted intent.
        """

        stmt = select(ReviewCaseRow).where(ReviewCaseRow.status.in_(_APPROVED))
        with self._sessionmaker() as session:
            rows = session.execute(stmt).scalars().all()
        examples: list[tuple[str, Intent]] = []
        for row in rows:
            label = row.corrected_intent or row.predicted_intent
            try:
                examples.append((row.text, Intent(label)))
            except ValueError:
                logger.warning("Skipping case %s with unknown intent %r", row.id, label)
        return examples

    def stats(self) -> ActiveLearningStats:
        """Aggregate counts across the queue."""

        with self._sessionmaker() as session:
            rows = session.execute(
                select(ReviewCaseRow.status, func.count()).group_by(
                    ReviewCaseRow.status
                )
            ).all()
        by_status = {status: count for status, count in rows}
        approved = by_status.get(CaseStatus.APPROVED.value, 0)
        auto_approved = by_status.get(CaseStatus.AUTO_APPROVED.value, 0)
        return ActiveLearningStats(
            total=sum(by_status.values()),
            pending=by_status.get(CaseStatus.PENDING.value, 0),
            approved=approved,
            auto_approved=auto_approved,
            rejected=by_status.get(CaseStatus.REJECTED.value, 0),
            learned_examples=approved + auto_approved,
        )


@lru_cache(maxsize=1)
def get_store() -> ActiveLearningStore:
    """Return the process-wide active-learning store (created once)."""

    return ActiveLearningStore()
