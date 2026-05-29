"""Structured logging via loguru with a rich-friendly format."""

from __future__ import annotations

import sys
from functools import lru_cache

from loguru import logger


@lru_cache(maxsize=1)
def configure_logging(level: str = "INFO") -> None:
    """Configure loguru once per process. Safe to call repeatedly."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


def get_logger():
    """Return the configured loguru logger. Configures on first call."""
    configure_logging()
    return logger
