"""A shared token bucket for pacing outbound requests.

Two different ceilings bind this harness, for different reasons. The GoodMem
instance rate-limits per principal and answers 429 with ``Retry-After``. The
embedding provider counts one request per memory and caps requests per minute.
Retrying is not sufficient for either: several workers backing off on the same
hint retry in lockstep and re-saturate the bucket immediately, so the client has
to pace itself rather than rediscover the limit over and over.
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
        """Holds every waiter off for ``seconds`` after a server throttle.

        A 429 means the shared budget is already spent, so the whole pool should
        stand down together rather than each worker retrying on its own clock.
        """
        if seconds <= 0:
            return
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic() + seconds)
