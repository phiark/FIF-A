"""Simple timing utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Timer:
    """Context manager for wall-clock measurements."""

    start_time: float | None = None
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.stop()

    def stop(self) -> None:
        if self.start_time is not None:
            self.elapsed += time.time() - self.start_time
            self.start_time = None
