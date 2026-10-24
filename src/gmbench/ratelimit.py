"""Share a request schedule across worker threads.

Each permit reserves the next available time slot. A throttle delays future
reservations on the shared schedule; workers with existing reservations keep
their slots. The client uses separate schedules for GoodMem requests and
proxied model calls. Ingestion has its own embedding budget.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Paces work to a fixed average rate across threads.

    Permits are handed out on a schedule rather than refilled into a burst
    allowance, which keeps a pool of workers from all arriving at once.
    """

    def __init__(self, per_second: float) -> None:
        """Creates a limiter.

        Args:
          per_second: Permitted requests per second. Zero or negative disables
            pacing entirely.
        """
        self._interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    @classmethod
    def per_minute(cls, per_minute: float) -> "RateLimiter":
        """Creates a limiter from a requests-per-minute budget."""
        return cls(per_minute / 60.0)

    @property
    def enabled(self) -> bool:
        return self._interval > 0.0

    def acquire(self, permits: int = 1) -> None:
        """Blocks until ``permits`` requests may be issued."""
        if self._interval <= 0.0 or permits <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_at)
            self._next_at = start + self._interval * permits
            delay = start - now
        if delay > 0:
            time.sleep(delay)

    def pause(self, seconds: float) -> None:
        """Postpone the next unreserved slot until at least ``seconds`` from now.

        Workers that have already reserved a slot keep their scheduled time.
        """
        if seconds <= 0:
            return
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic() + seconds)
