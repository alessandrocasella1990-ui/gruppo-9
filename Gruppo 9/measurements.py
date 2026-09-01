"""Buffer thread-safe per le misure temporali."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any


class MeasurementBuffer:
    """Conserva gli ultimi campioni acquisiti in ordine temporale."""

    def __init__(self, max_length: int) -> None:
        self._samples: deque[dict[str, Any]] = deque(maxlen=max_length)
        self._lock = Lock()

    def append(self, timestamp_ms: int, distance_normalized: float | None) -> None:
        sample = {
            "timestamp_ms": timestamp_ms,
            "distance_normalized": (
                round(distance_normalized, 6)
                if distance_normalized is not None
                else None
            ),
        }
        with self._lock:
            self._samples.append(sample)

    def latest(self) -> dict[str, Any]:
        with self._lock:
            if not self._samples:
                return {
                    "timestamp_ms": None,
                    "distance_normalized": None,
                    "hand_detected": False,
                }

            sample = dict(self._samples[-1])

        sample["hand_detected"] = sample["distance_normalized"] is not None
        return sample

    def series(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(sample) for sample in self._samples]
