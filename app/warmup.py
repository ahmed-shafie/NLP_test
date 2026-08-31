"""Load the models and build the example index before the first customer arrives.

The first turn in a cold process pays for the embedding model, the FAISS index over
the example corpus and the spaCy/Stanza pipelines — measured at roughly three
minutes. Doing that work inside ``lifespan`` would block the liveness probe for as
long, so an orchestrator would kill the process before it ever served a request.

Warm-up therefore runs on a background thread: liveness answers immediately,
``/health/ready`` reports ``not_ready`` until the thread finishes, and traffic is
only routed once the index is built. A failure is recorded and left visible rather
than retried, because the lazy loaders still work — a failed warm-up is a slow
first turn, not an outage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock, Thread

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WarmupState:
    """A snapshot of warm-up progress, safe to serve from a probe."""

    status: str
    step: str | None
    duration_s: float | None
    error: str | None

    @property
    def ready(self) -> bool:
        return self.status in {"ready", "skipped"}


PENDING = "pending"
RUNNING = "running"
READY = "ready"
FAILED = "failed"
SKIPPED = "skipped"

_lock = Lock()
_status = PENDING
_step: str | None = None
_duration: float | None = None
_error: str | None = None
_thread: Thread | None = None


def state() -> WarmupState:
    """Return the current warm-up snapshot."""

    with _lock:
        return WarmupState(
            status=_status, step=_step, duration_s=_duration, error=_error
        )


def _steps() -> list[tuple[str, object]]:
    # Imported here so that importing this module does not pull in the models.
    from app.db.beneficiary import get_beneficiary_repository
    from app.embeddings import get_embedder
    from app.llm import get_llm_handler
    from app.nlu import arabic, english
    from app.nlu.semantic_intents import get_semantic_classifier
    from app.orchestration import get_nlu_pipeline

    return [
        ("english_model", english._load_model),
        ("arabic_model", arabic._load_model),
        ("embedder", get_embedder),
        # The dominant cost: encoding/loading the example corpus index.
        ("semantic_index", get_semantic_classifier),
        ("nlu_pipeline", get_nlu_pipeline),
        ("llm_handler", get_llm_handler),
        ("beneficiaries", get_beneficiary_repository),
    ]


def _run() -> None:
    global _status, _step, _duration, _error

    started = time.monotonic()
    for name, load in _steps():
        with _lock:
            _step = name
        try:
            load()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - a warm-up failure must not crash
            logger.warning("Warm-up step %r failed: %s", name, exc)
            with _lock:
                _status, _error = FAILED, f"{name}: {exc}"
                _duration = time.monotonic() - started
            return
    elapsed = time.monotonic() - started
    logger.info("Warm-up finished in %.1fs; ready to serve.", elapsed)
    with _lock:
        _status, _step, _duration = READY, None, elapsed


def start(*, enabled: bool = True) -> None:
    """Begin warm-up on a background thread. Idempotent."""

    global _status, _thread

    with _lock:
        if not enabled:
            _status = SKIPPED
            return
        if _thread is not None and _thread.is_alive():
            return
        _status = RUNNING
        _thread = Thread(target=_run, name="warmup", daemon=True)
        thread = _thread
    thread.start()


def wait(timeout: float | None = None) -> WarmupState:
    """Block until warm-up settles. Used by tests and one-shot commands."""

    with _lock:
        thread = _thread
    if thread is not None:
        thread.join(timeout)
    return state()


def reset() -> None:
    """Clear state so a test can run warm-up again."""

    global _status, _step, _duration, _error, _thread

    with _lock:
        _status, _step, _duration, _error, _thread = PENDING, None, None, None, None
