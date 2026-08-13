"""Background daemon that rebuilds and hot-swaps the intent index nightly.

The daemon sleeps until the next configured UTC hour/minute, rebuilds the index
(folding in approved review-queue examples), swaps it in atomically, and repeats —
all on a daemon thread so the service keeps serving without a restart.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

from app.active_learning.index_rebuilder import rebuild_index
from app.config import settings

logger = logging.getLogger(__name__)


def next_run_time(now: datetime | None = None) -> datetime:
    """Return the next UTC datetime the rebuild should run at."""

    now = now or datetime.now(UTC)
    target = now.replace(
        hour=settings.index_rebuild_hour_utc,
        minute=settings.index_rebuild_minute_utc,
        second=0,
        microsecond=0,
    )
    if target <= now:
        target += timedelta(days=1)
    return target


def seconds_until_next_run(now: datetime | None = None) -> float:
    """Seconds to wait until the next scheduled rebuild."""

    now = now or datetime.now(UTC)
    return (next_run_time(now) - now).total_seconds()


class IndexRebuildDaemon:
    """Daemon thread driving the nightly rebuild loop."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="index-rebuild-daemon", daemon=True
        )
        self._thread.start()
        logger.info(
            "Index rebuild daemon started; next rebuild at %s UTC.",
            next_run_time().isoformat(),
        )

    def stop(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            wait = seconds_until_next_run()
            # Wake at the scheduled time, or earlier if asked to stop.
            if self._stop.wait(timeout=wait):
                break
            try:
                rebuild_index()
            except Exception:  # noqa: BLE001 - the loop must survive a bad rebuild
                logger.exception("Nightly index rebuild failed")


_daemon: IndexRebuildDaemon | None = None


def get_daemon() -> IndexRebuildDaemon:
    global _daemon
    if _daemon is None:
        _daemon = IndexRebuildDaemon()
    return _daemon


def start_daemon() -> None:
    """Start the nightly rebuild daemon if it is enabled in settings."""

    if not (settings.active_learning_enabled and settings.index_rebuild_enabled):
        logger.info("Index rebuild daemon disabled by configuration.")
        return
    get_daemon().start()


def stop_daemon() -> None:
    """Stop the daemon (used on shutdown / in tests)."""

    if _daemon is not None:
        _daemon.stop()
