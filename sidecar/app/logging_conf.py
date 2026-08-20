"""Logging setup shared by the sidecar entrypoint and the task worker."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.config import settings

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging() -> None:
    """Log to stderr (captured by Tauri) and to a rotating file in the data dir."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        settings.logs_dir / "sidecar.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers = [stream, file_handler]

    # These are chatty and say nothing useful in a single-user local app.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
