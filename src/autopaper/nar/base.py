"""
NAR / narrative source.

``.env.local`` as supplied configures exactly one Telegram chat (the alert
source) and contains **no** NAR endpoint, API key or bot token. That is a fact
about the configuration, not a failure, so this package:

* reads whatever NAR configuration exists (none, by default)
* returns a ``NarSnapshot`` whose fields are ALL ``None`` and ``available=False``
  when the source is not configured or does not answer
* never fabricates buy counts, sell counts, holder counts, wallet flow,
  liquidity, security flags, narrative or social activity
* exposes ``NAR_UNAVAILABLE`` so the decision engine can downgrade confidence
  and pick a route that does not depend on narrative confirmation

If the operator later adds ``NAR_BASE_URL`` / ``NAR_API_KEY`` to ``.env.local``,
the ``HttpNarSource`` below starts working with no code change.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ..clock import Clock, SystemClock
from ..config import NarConfig
from ..logging_setup import log
from ..models import NarSnapshot
from ..safety import guard_url


class NarSource(Protocol):
    name: str

    def fetch(self, mint: str) -> NarSnapshot:
        ...


class NullNarSource:
    """
    The honest default.

    Used when no NAR source is configured. It reports absence with provenance
    rather than guessing. ``available=False`` means "no NAR evidence", which the
    decision engine treats as reduced signal quality, not as a clean bill of
    health and not as a reason to invent numbers.
    """

    name = "nar:none"

    def __init__(self, clock: Clock | None = None, reason: str | None = None) -> None:
        self.clock = clock or SystemClock()
        self.reason = reason or "no NAR source configured in .env.local"
        self.calls = 0

    def fetch(self, mint: str) -> NarSnapshot:
        self.calls += 1
        return NarSnapshot(
            mint=mint,
            fetched_at=self.clock.now(),
            source=None,
            available=False,
            error=f"NAR_UNAVAILABLE: {self.reason}",
        )


class HttpNarSource:
    """
    Optional NAR HTTP source.

    Only constructed when ``NAR_BASE_URL`` and ``NAR_API_KEY`` exist. Reads are
    GET-only; a write/order-shaped URL is rejected by the safety guard.
    """

    name = "nar:http"

    def __init__(
        self,
        config: NarConfig,
        *,
        clock: Clock | None = None,
        timeout_s: float = 6.0,
    ) -> None:
        self.config = config
        self.clock = clock or SystemClock()
        self.timeout_s = timeout_s
        self.calls = 0

    def fetch(self, mint: str) -> NarSnapshot:
        import httpx

        self.calls += 1
        now = self.clock.now()
        if not self.config.available:
            return NarSnapshot(mint=mint, fetched_at=now, source=None,
                               available=False, error="NAR_UNAVAILABLE: not configured")
        url = f"{self.config.base_url.rstrip('/')}/token/{mint}"
        guard_url(url)
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.get(
                    url,
                    headers={"authorization": f"Bearer {self.config.api_key}"},
                )
            if resp.status_code >= 400:
                return NarSnapshot(mint=mint, fetched_at=now, source=self.config.source,
                                   available=False,
                                   error=f"NAR HTTP {resp.status_code}")
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            return NarSnapshot(mint=mint, fetched_at=now, source=self.config.source,
                               available=False, error=f"NAR error: {exc!r}")
        return self.parse(payload, mint, now)

    def parse(self, payload: Mapping[str, Any], mint: str, now: float) -> NarSnapshot:
        """Map only fields the upstream actually supplied. Nothing else."""
        d = payload.get("data") if isinstance(payload, Mapping) else None
        d = d if isinstance(d, Mapping) else {}
        flags = d.get("security_flags")
        return NarSnapshot(
            mint=mint,
            fetched_at=now,
            source=self.config.source,
            available=True,
            narrative=d.get("narrative"),
            narrative_score=_num(d.get("narrative_score")),
            social_activity=_num(d.get("social_activity")),
            holder_count=_int(d.get("holder_count")),
            top10_holders_pct=_num(d.get("top10_holders_pct")),
            dev_holdings_pct=_num(d.get("dev_holdings_pct")),
            bundler_count=_int(d.get("bundler_count")),
            sniper_count=_int(d.get("sniper_count")),
            insider_count=_int(d.get("insider_count")),
            security_flags=tuple(flags) if isinstance(flags, list) else None,
            wallet_flow_net=_num(d.get("wallet_flow_net")),
            dex_paid=d.get("dex_paid") if isinstance(d.get("dex_paid"), bool) else None,
            website=d.get("website"),
            twitter=d.get("twitter"),
            telegram=d.get("telegram"),
        )


class NarService:
    """
    Cache in front of a NarSource.

    Supports two kinds of source:

    * ``fetch(mint)``     - synchronous (HTTP, or the honest Null source)
    * ``fetch_async(mint)`` - awaits a network round trip (the Telegram bot)

    The engine awaits ``fetch_async`` when the source offers it, and otherwise
    falls back to ``fetch``. Either way an unavailable source is a normal
    outcome that degrades confidence, never an exception.
    """

    def __init__(
        self,
        config: NarConfig,
        *,
        clock: Clock | None = None,
        source: NarSource | None = None,
        cache_ttl_s: float = 20.0,
    ) -> None:
        self.config = config
        self.clock = clock or SystemClock()
        self.cache_ttl_s = cache_ttl_s
        self._source_name = "nar:none"
        if source is not None:
            self.source: NarSource = source
            self._source_name = getattr(source, "name", "nar:custom")
        elif (config.source or "").lower() in ("phanes_bot", "phanes", "telegram_bot"):
            from .phanes import PhanesNarSource

            # The configured budget is passed EXPLICITLY. Relying on the
            # source's own default left a 12s fallback reachable whenever a
            # caller forgot, which is exactly the inconsistency this fixes.
            self.source = PhanesNarSource(
                clock=self.clock,
                timeout_s=config.timeout_s,
                min_gap_s=config.min_query_gap_s,
                fallback_bot_username=config.fallback_bot_username,
                fallback_after_failures=config.fallback_after_failures,
            )
            self._source_name = self.source.name
        elif config.enabled and config.available:
            self.source = HttpNarSource(
                config, clock=self.clock, timeout_s=config.timeout_s
            )
            self._source_name = self.source.name
        else:
            self.source = NullNarSource(
                clock=self.clock,
                reason=(
                    "no NAR source configured in .env.local"
                    if config.enabled
                    else "NAR source disabled by configuration"
                ),
            )
            self._source_name = self.source.name
        self._cache: dict[str, tuple[float, NarSnapshot]] = {}
        #: Observable narrative health, so an outage is measurable.
        self.metrics = NarMetrics()
        self.hits = 0
        self.misses = 0
        self.unavailable = 0

    def attach(self, source: NarSource) -> None:
        """
        Swap in a source that needs a live connection (the Telegram bot).

        Called once the alert listener is connected, so the NAR source shares
        that connection rather than logging in a second time.
        """
        self.source = source
        self._source_name = getattr(source, "name", "nar:custom")
        self._cache.clear()

    @property
    def is_async(self) -> bool:
        return callable(getattr(self.source, "fetch_async", None))

    async def fetch_async(self, mint: str) -> NarSnapshot:
        now = self.clock.now()
        entry = self._cache.get(mint)
        if entry and (now - entry[0]) <= self.cache_ttl_s:
            self.hits += 1
            return entry[1]
        self.misses += 1
        started = time.monotonic()
        snap = await self.source.fetch_async(mint)  # type: ignore[attr-defined]
        # Measured so a narrative outage is a visible metric rather than a
        # silent 0.05 reduction in signal_quality that quietly moves the
        # candidate population.
        self.metrics.record(
            available=bool(snap.available),
            latency_ms=(time.monotonic() - started) * 1000.0,
        )
        if not snap.available and snap.error and "within" in str(snap.error):
            self.metrics.timeouts += 1
        self._record(snap)
        return snap

    def _record(self, snap: NarSnapshot) -> None:
        if not snap.available:
            self.unavailable += 1
            log("NAR", "%s NAR unavailable for %s: %s",
                self._source_name, snap.mint[:8], snap.error)
        self._cache[snap.mint] = (self.clock.now(), snap)

    @property
    def available(self) -> bool:
        if hasattr(self.source, "available"):
            return bool(self.source.available)  # type: ignore[attr-defined]
        return bool(
            getattr(self.source, "config", None)
            and self.source.config.available  # type: ignore[attr-defined]
        )

    @property
    def name(self) -> str:
        return self._source_name

    def fetch(self, mint: str) -> NarSnapshot:
        now = self.clock.now()
        entry = self._cache.get(mint)
        if entry and (now - entry[0]) <= self.cache_ttl_s:
            self.hits += 1
            return entry[1]
        self.misses += 1
        snap = self.source.fetch(mint)
        if not snap.available:
            self.unavailable += 1
            log("NAR", "%s NAR unavailable for %s: %s", self.name, mint[:8], snap.error)
        self._cache[mint] = (now, snap)
        return snap

    def health(self) -> dict[str, Any]:
        base = {
            "name": self.name,
            "available": self.available,
            "hits": self.hits,
            "misses": self.misses,
            "unavailable_responses": self.unavailable,
        }
        extra = getattr(self.source, "health", None)
        if callable(extra):
            try:
                base.update({k: v for k, v in extra().items() if k != "name"})
            except Exception:  # noqa: BLE001
                pass
        return base


@dataclass
class NarMetrics:
    """
    Observable narrative health.

    The failure this exists for: a narrative timeout used to be invisible. It
    produced ``available=False``, which quietly cost the candidate 0.05 of
    signal_quality, so a silent outage quietly changed the candidate POPULATION
    rather than showing up as an error. The behaviour is unchanged - L25 is
    preserved - but it is now measurable.
    """

    requested: int = 0
    available: int = 0
    unavailable: int = 0
    timeouts: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, *, available: bool, latency_ms: float | None) -> None:
        self.requested += 1
        if available:
            self.available += 1
        else:
            self.unavailable += 1
        if isinstance(latency_ms, (int, float)):
            self.latencies_ms.append(float(latency_ms))

    @property
    def unavailable_rate(self) -> float:
        return (self.unavailable / self.requested) if self.requested else 0.0

    def as_dict(self) -> dict[str, Any]:
        from ..latency import percentiles

        stats = percentiles(self.latencies_ms)
        return {
            "requested": self.requested,
            "available": self.available,
            "unavailable": self.unavailable,
            "timeouts": self.timeouts,
            "unavailable_rate": self.unavailable_rate,
            "latency_p50_ms": stats["p50"],
            "latency_p90_ms": stats["p90"],
            "latency_p95_ms": stats["p95"],
            "latency_p99_ms": stats["p99"],
            "latency_max_ms": stats["max"],
            "samples": stats["n"],
        }
