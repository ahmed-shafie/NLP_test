"""Structured JSON logging with request-id correlation.

Set ``NLU_LOG_JSON=true`` (the default) for machine-readable logs suitable for ELK /
Loki ingestion; set it to ``false`` for human-friendly logs during local development.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from app.config import settings
from app.request_context import get_request_id

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON, including the current request id."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        return json.dumps(payload, default=str)


class RequestIdFilter(logging.Filter):
    """Attach the current request id to records for non-JSON formatters."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


def configure_logging() -> None:
    """Install the configured handler/formatter on the root logger."""

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    handler = logging.StreamHandler()
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.addFilter(RequestIdFilter())
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
            )
        )
    root.handlers = [handler]
