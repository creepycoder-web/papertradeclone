"""
Small read-only probes that make rate-limit behaviour assertable without a
network.

Kept separate from the clients so a test can assert that a limit is reported as
a NAMED condition, without standing up an HTTP stack or a fake transport.
"""

from __future__ import annotations

from ..market.ratelimit import RateLimiter


class RateLimiterProbe:
    """
    Exercises the shared sliding-window limiter.

    Used to prove that a local rate cap raises a named error rather than
    silently dropping a request, which would leave the harness reporting a
    result it never actually fetched.
    """

    def __init__(self, max_per_minute: int = 2) -> None:
        self.limiter = RateLimiter(max_per_minute, name="probe")

    def rate_limited_is_named(self) -> bool:
        """
        Is the local cap reported as a distinct, recognisable condition?
        """
        try:
            for _ in range(self.limiter.max_per_minute + 2):
                self.limiter.acquire(1000.0)
        except Exception as exc:  # noqa: BLE001 - the message is the assertion
            return "rate cap" in str(exc)
        return False

    @property
    def reason(self) -> str:
        try:
            for _ in range(self.limiter.max_per_minute + 2):
                self.limiter.acquire(1000.0)
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        return "no rate limit was enforced"

    def note_retry_after(self, retry_after_s: float | None) -> None:
        """Record a provider-supplied backoff, as a client would."""
        self.limiter.note_retry_after(1000.0, retry_after_s)

    def allow_after_backoff(self, now: float) -> bool:
        allowed, _wait = self.limiter.allow(now)
        return allowed


__all__ = ["RateLimiterProbe"]