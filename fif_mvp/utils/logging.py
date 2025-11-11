"""Lightweight experiment logging."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_file_logger(log_path: Path) -> logging.Logger:
    """Configure a file logger."""

    logger = logging.getLogger(f"fif_mvp.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
