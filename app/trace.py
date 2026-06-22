"""Per-block execution tracing for pipeline observability.

Every block of the pipeline (language detection, intent classification, entity
extraction, the LLM safety net, active learning, ...) records how long it ran and
whether it succeeded, was skipped, or errored. The collected entries are surfaced
on the API response as ``block_trace`` so developers can see exactly what each
request did and where time went.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import BaseModel, Field


class BlockTrace(BaseModel):
    """Execution record for a single pipeline block."""

    block: str = Field(..., description="Block name, e.g. 'intent_classification'.")
    status: str = Field(
        default="ok",
        description="Outcome of the block: 'ok', 'skipped', or 'error'.",
    )
    duration_ms: float = Field(..., description="Wall-clock time spent in the block.")
    note: str | None = Field(
        default=None,
        description="Optional detail (why a block was skipped, or an error message).",
    )


class BlockSpan:
    """Mutable handle yielded while a block runs, used to annotate its outcome."""

    def __init__(self) -> None:
        self.status = "ok"
        self.note: str | None = None

    def skip(self, note: str | None = None) -> None:
        """Mark the block as skipped (e.g. a step that did not apply this request)."""

        self.status = "skipped"
        if note is not None:
            self.note = note

    def annotate(self, note: str) -> None:
        """Attach a human-readable note without changing the status."""

        self.note = note


class BlockTracer:
    """Collects :class:`BlockTrace` entries as blocks execute, in order."""

    def __init__(self) -> None:
        self.entries: list[BlockTrace] = []

    @contextmanager
    def block(self, name: str) -> Iterator[BlockSpan]:
        """Time the wrapped work and record a trace entry for ``name``.

        On an exception the entry is marked ``error`` (with the message as the
        note) and the exception is re-raised so callers keep their semantics.
        """

        span = BlockSpan()
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.note = span.note or f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            self.entries.append(
                BlockTrace(
                    block=name,
                    status=span.status,
                    duration_ms=elapsed_ms,
                    note=span.note,
                )
            )

    def add(
        self, name: str, status: str, duration_ms: float, note: str | None = None
    ) -> None:
        """Append a pre-computed trace entry (e.g. merged from a sub-pipeline)."""

        self.entries.append(
            BlockTrace(block=name, status=status, duration_ms=duration_ms, note=note)
        )

    def extend(self, traces: list[BlockTrace]) -> None:
        """Append a batch of existing trace entries (preserving order)."""

        self.entries.extend(traces)
