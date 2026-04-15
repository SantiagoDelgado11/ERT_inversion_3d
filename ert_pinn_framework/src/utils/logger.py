"""Centralized logging helpers for experiments."""

from __future__ import annotations

import logging
from pathlib import Path

from .io import ensure_dir


def build_logger(
    name: str,
    log_dir: str | Path,
    level: int = logging.INFO,
    file_name: str = "run.log",
) -> logging.Logger:
    """Create a logger with both console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    target_dir = ensure_dir(log_dir)
    log_file = target_dir / file_name

    if logger.handlers:
        current_log_file = getattr(logger, "_copilot_log_file", None)
        if current_log_file == str(log_file):
            return logger

        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    setattr(logger, "_copilot_log_file", str(log_file))
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return an existing logger by name."""
    return logging.getLogger(name)
