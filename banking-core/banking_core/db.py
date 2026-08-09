"""SQLAlchemy models and engine/session helpers for the Banking Core database."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from functools import lru_cache

from sqlalchemy import Numeric, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from banking_core.config import settings


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    account_id: Mapped[str] = mapped_column(String, primary_key=True)
    owner_user: Mapped[str] = mapped_column(String, index=True)
    account_type: Mapped[str] = mapped_column(String)  # current / savings / credit
    number: Mapped[str] = mapped_column(String)
    currency: Mapped[str] = mapped_column(String)
    balance: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    status: Mapped[str] = mapped_column(String, default="active")


class Beneficiary(Base):
    __tablename__ = "beneficiaries"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    owner_user: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    name_ar: Mapped[str | None] = mapped_column(String, nullable=True)
    account: Mapped[str] = mapped_column(String)
    bank: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str] = mapped_column(String, default="SAR")
    status: Mapped[str] = mapped_column(String, default="active")
    is_favorite: Mapped[bool] = mapped_column(default=False)


class Biller(Base):
    __tablename__ = "billers"

    biller_code: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    category: Mapped[str | None] = mapped_column(String, nullable=True)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Engine for the configured database (SQLite locally, Postgres deployed)."""

    if settings.db_url.startswith("sqlite"):
        # SQLite is single-file and shared across the app's threads.
        return create_engine(
            settings.db_url,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )
    # Server-backed databases (Postgres): pool connections and recycle them so a
    # long-idle container doesn't hand out a connection the server already closed.
    return create_engine(
        settings.db_url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=1800,
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def init_db() -> None:
    """Create all tables if they do not yet exist."""

    Base.metadata.create_all(get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
