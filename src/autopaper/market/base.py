"""
MarketDataService - the single façade the engine talks to.

Responsibilities:
* route to the primary provider, fall back when it fails or looks implausible
* merge partial fields between providers (missing stays ``None``)
* record observations into the per-mint history (an ``ObservationSet``)
* track the per-mint "first valid price" the agent could actually have traded
* stamp every observation with ``observed_at`` / ``received_at`` and expose a
  ``DataState`` verdict
* serve the SAME observation to analysis, marking and fills, so nothing can see
  a price the rest of the system cannot
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Iterable

from ..clock import Clock, SystemClock
from ..config import MarketConfig
from ..logging_setup import log
from ..models import DataState, Observation, ObservationSet, QuoteResult
from .dexscreener import DexScreenerProvider
from .geckoterminal import GeckoTerminalProvider
from .ratelimit import MarketDataError, deadline_stats_snapshot, run_with_deadline

#: Below this, consulting the fallback cannot produce a usable answer inside the
#: budget and would only add latency, so it is skipped - explicitly and logged.
_MIN_FALLBACK_BUDGET_S = 0.25


class MarketDataService:
    def __init__(
        self,
        config: MarketConfig,
        *,
        clock: Clock | None = None,
        primary=None,
        fallback=None,
    ) -> None:
        self.config = config
        self.clock = clock or SystemClock()
        self.primary = primary or DexScreenerProvider(
            clock=self.clock,
            max_rpm=config.max_requests_per_minute,
            timeout_s=config.http_timeout_s,
            connect_timeout_s=config.connect_timeout_s,
            cache_ttl_s=config.cache_ttl_s,
            max_retries=config.max_retries,
        )
        self.fallback = fallback or GeckoTerminalProvider(
            clock=self.clock,
            max_rpm=max(20, config.max_requests_per_minute // 3),
            timeout_s=config.http_timeout_s,
            connect_timeout_s=config.connect_timeout_s,
            cache_ttl_s=config.cache_ttl_s * 2,
            max_retries=config.max_retries,
        )
        self._history: dict[str, ObservationSet] = {}
        self._first_valid_price: dict[str, float] = {}
        self._last_result: dict[str, QuoteResult] = {}
        self.request_count = 0
        self.error_count = 0
        self.rate_limit_count = 0
        self.last_error: str | None = None
        self.last_success_at: float | None = None
        self._fvp_probe_budget = 0

    # -- history -----------------------------------------------------------

    def history(self, mint: str) -> ObservationSet:
        return self._history.setdefault(mint, ObservationSet())

    def register_known_price(self, mint: str, price: float | None) -> None:
        """
        Record the first price the agent could legitimately have acted on.

        Called when the alert itself carries a price, or with the first market
        observation. This is the anchor for extension math.
        """
        if price and price > 0 and mint not in self._first_valid_price:
            self._first_valid_price[mint] = price

    def first_valid_price(self, mint: str) -> float | None:
        return self._first_valid_price.get(mint)

    def seed_first_valid_price(self, mint: str) -> None:
        """
        Optionally backfill the anchor from the venue's own oldest 1m candle.

        Rate-limited to a small number of probes per process so this stays a
        cheap context feature, not a scraping habit.

        ``first_valid_price`` is TWO sequential HTTP fetches from the fallback
        provider, so it is held to the same wall-clock budget as a quote. It used
        to be unbounded and sat squarely on the alert->decision path, which is
        how one stage could cost three provider reads.
        """
        if mint in self._first_valid_price or self._fvp_probe_budget <= 0:
            return
        self._fvp_probe_budget -= 1
        budget = max(0.1, float(self.config.http_timeout_s))
        self._shrink_to_budget(self.fallback, budget)
        try:
            price, opened = run_with_deadline(
                lambda: self.fallback.first_valid_price(mint),
                budget,
                label="geckoterminal:first_valid_price",
            )
        except MarketDataError as exc:
            log("MARKET", "first-valid-price probe skipped for %s: %s",
                mint[:6], exc)
            return
        if price and price > 0:
            self._first_valid_price[mint] = price
            log("MARKET", "first-valid-price anchored %s %.3e (candle %s)",
                mint[:6], price, opened)

    def adopt_first_valid_price(self, mint: str, price: float | None) -> None:
        """Testing / recovery hook to restore a persisted anchor."""
        if price and price > 0:
            self._first_valid_price.setdefault(mint, price)

    # -- fetching ----------------------------------------------------------

    def quote(self, mint: str, *, allow_fallback: bool = True) -> QuoteResult:
        """
        Fetch the current market view for ``mint``.

        Never invents a price. On total failure returns a QuoteResult with
        ``observation=None`` and an error string.

        ONE CALL, ONE BUDGET. Primary and fallback are consulted in sequence, so
        without a service-level deadline a dead primary plus a slow fallback
        cost ``2 x http_timeout_s`` - which is how a single "8 second" market
        read became a 16.6s stage. The remaining budget is what the fallback is
        allowed to spend, and a fallback that cannot be served inside it is
        skipped rather than allowed to overrun the caller.
        """
        self.request_count += 1
        budget = max(0.1, float(self.config.http_timeout_s))
        deadline = time.monotonic() + budget
        result = self._safe_quote(self.primary, mint)
        obs = result.observation

        if self._needs_fallback(obs, result) and allow_fallback:
            left = deadline - time.monotonic()
            if left > _MIN_FALLBACK_BUDGET_S:
                reason = self._fallback_reason(obs, result)
                # The fallback's own budget is what is LEFT, so the pair cannot
                # exceed one budget between them.
                self._shrink_to_budget(self.fallback, left)
                alt = self._safe_quote(self.fallback, mint)
                if alt.observation is not None:
                    obs = self._merge(obs, alt.observation, replace_primary=(
                        reason == "implausible_primary_liquidity"
                    ))
                    result = QuoteResult(
                        observation=obs,
                        error=result.error,
                        provider=f"{self.primary.name}+{self.fallback.name}",
                        latency_ms=result.latency_ms + alt.latency_ms,
                        from_cache=False,
                        rate_limited=result.rate_limited or alt.rate_limited,
                        retry_after_s=result.retry_after_s or alt.retry_after_s,
                    )
            else:
                # Recorded, not silent: "the fallback was never asked because
                # the primary had already spent the budget" is a different fact
                # from "the fallback answered and had nothing".
                self.error_count += 1
                self.last_error = (
                    f"{self.fallback.name}: skipped, the primary had already "
                    f"spent the {budget:.2f}s market budget"
                )
                log("MARKET", "%s", self.last_error)
                result = QuoteResult(
                    observation=obs,
                    error=(result.error or self.last_error),
                    provider=result.provider,
                    latency_ms=result.latency_ms,
                    rate_limited=result.rate_limited,
                    retry_after_s=result.retry_after_s,
                )

        self._last_result[mint] = result
        if result.observation is not None:
            self._record(mint, result.observation)
            self.last_success_at = self.clock.now()
        else:
            self.error_count += 1
            self.last_error = result.error
            log("MARKET", "no data for %s: %s", mint[:8], result.error)
        if result.rate_limited:
            self.rate_limit_count += 1
        return result

    @staticmethod
    def _shrink_to_budget(provider, budget_s: float) -> None:
        """
        Hand a provider the budget that is actually left.

        The provider's retry budget AND its per-call deadline move together, so
        the two bounds stay consistent instead of the retry loop being allowed
        to spend more than the caller can wait.
        """
        for attr, value in (("timeout_s", budget_s),):
            try:
                setattr(provider, attr, float(value))
            except Exception:  # noqa: BLE001 - a provider without the knob is fine
                return
        backoff = getattr(provider, "backoff", None)
        if backoff is not None:
            try:
                backoff.budget_s = float(budget_s)
            except Exception:  # noqa: BLE001
                pass

    async def quote_async(
        self, mint: str, *, allow_fallback: bool = True
    ) -> QuoteResult:
        """Event-loop-safe form of :meth:`quote`.

        Runs the ENTIRE existing synchronous quote (provider transport, retry
        loop, fallback merge, history recording) on a worker thread. Semantics
        are byte-for-byte the synchronous path; only the thread changes. This is
        the single call the engine makes, so no provider transport can block the
        asyncio loop.
        """
        return await asyncio.to_thread(
            self.quote, mint, allow_fallback=allow_fallback
        )

    def _safe_quote(self, provider, mint: str) -> QuoteResult:
        try:
            return provider.quote(mint)
        except MarketDataError as exc:
            return QuoteResult(
                observation=None,
                error=f"{provider.name}: {exc}",
                provider=provider.name,
                rate_limited=exc.rate_limited,
                retry_after_s=exc.retry_after_s,
            )
        except Exception as exc:  # noqa: BLE001 - never kill the loop on a provider bug
            return QuoteResult(
                observation=None, error=f"{provider.name}: unexpected {exc!r}",
                provider=provider.name,
            )

    def _fallback_reason(
        self, obs: Observation | None, result: QuoteResult
    ) -> str:
        if obs is None or obs.price_usd is None:
            return "primary_unavailable"
        if result.rate_limited:
            return "primary_rate_limited"
        if obs.liquidity_usd is not None and obs.liquidity_usd <= 0:
            return "implausible_primary_liquidity"
        return "none"

    def _needs_fallback(self, obs: Observation | None, result: QuoteResult) -> bool:
        return self._fallback_reason(obs, result) != "none"

    @staticmethod
    def _merge(
        primary: Observation | None, alt: Observation, *, replace_primary: bool = False
    ) -> Observation:
        """
        Combine two views of the same mint.

        Default is gap-filling: the primary's price wins wherever it exists,
        because silently substituting another venue's price would make the
        ledger's provenance ambiguous.

        ``replace_primary`` is set only when we consulted the fallback because
        the primary's pool was implausibly thin. In that case the primary pair
        is not a tradeable venue, so the fallback's price and size are the ones
        to believe.
        """
        if primary is None:
            return alt
        if replace_primary:
            merged = replace(
                alt,
                first_valid_price=primary.first_valid_price
                or alt.first_valid_price,
                pair_created_at=primary.pair_created_at or alt.pair_created_at,
                extra={**primary.extra, **{f"primary_{k}": v for k, v in primary.extra.items()}},
            )
            return merged
        patch: dict[str, object] = {}
        for field in (
            "price_usd", "price_native", "market_cap_usd", "fdv_usd", "liquidity_usd",
            "volume_5m_usd", "volume_1h_usd", "volume_24h_usd", "price_change_5m",
            "price_change_1h", "price_change_24h", "buys_5m", "sells_5m", "txns_5m",
            "txns_1h", "pair_created_at",
        ):
            if getattr(primary, field, None) is None:
                value = getattr(alt, field, None)
                if value is not None:
                    patch[field] = value
        if not patch:
            return primary
        merged = replace(primary, **patch)
        if "geckoterminal" in (primary.provider or ""):
            return merged
        extra = dict(primary.extra)
        extra.update({f"gt_{k}": v for k, v in alt.extra.items()})
        return replace(merged, extra=extra)

    def _record(self, mint: str, obs: Observation) -> None:
        if obs.first_valid_price is None:
            fvp = self._first_valid_price.get(mint)
            obs = replace(obs, first_valid_price=fvp)
        self.history(mint).append(obs)
        self.register_known_price(mint, obs.price_usd)

    # -- read helpers ------------------------------------------------------

    def latest(self, mint: str) -> Observation | None:
        return self.history(mint).latest()

    def latest_as_of(self, mint: str, t: float) -> Observation | None:
        return self.history(mint).latest(t)

    def state(self, mint: str, now: float | None = None) -> DataState:
        now = now if now is not None else self.clock.now()
        obs = self.latest(mint)
        if obs is None:
            return DataState.MISSING
        return obs.freshness(now, self.config.stale_after_ms, self.config.unusable_after_ms)

    def mark_price(self, mint: str, now: float | None = None) -> tuple[float | None, DataState]:
        """
        Mark-price convention.

        Use the newest observation with a usable price. If it is STALE we still
        mark at it (it is the last known good price) but report ``STALE`` so the
        caller can refuse to make aggressive decisions. If nothing usable is
        available, mark at ``None`` and report ``UNUSABLE``/``MISSING`` - the
        position is then valued by its entry cost basis and flagged
        ``DATA_UNCERTAIN`` rather than being marked at a made-up price.
        """
        now = now if now is not None else self.clock.now()
        hist = self.history(mint)
        for obs in reversed(list(hist)):
            if obs.price_usd is None or obs.price_usd <= 0:
                continue
            state = obs.freshness(now, self.config.stale_after_ms, self.config.unusable_after_ms)
            if state in (DataState.LIVE, DataState.STALE):
                return obs.price_usd, state
        return None, DataState.UNUSABLE if hist else DataState.MISSING

    def prices_as_of(self, mint: str, t: float) -> list[tuple[float, float]]:
        return self.history(mint).prices(t)

    def health(self) -> dict[str, object]:
        """
        Provider health, INCLUDING the budget the operator thinks is in force.

        ``state`` is derived here rather than at the call site. The dashboard
        read ``health().get("state", "?")`` and this dict had no such key, so the
        market row rendered a permanent "?" - a display path that could not
        report a market provider as anything but unknown.
        """
        primary = self.primary.name
        fallback = self.fallback.name
        state = (
            "OK" if self.last_success_at
            else ("DEGRADED" if self.error_count else "IDLE")
        )
        return {
            "state": state,
            "primary": primary,
            "fallback": fallback,
            "requests": self.request_count,
            "errors": self.error_count,
            "rate_limited": self.rate_limit_count,
            "last_error": self.last_error,
            "last_success_at": self.last_success_at,
            "primary_limiter": self.primary.limiter.stats(),
            "fallback_limiter": self.fallback.limiter.stats(),
            # Provider-failure visibility: a partial failure (retried, then
            # succeeded) is now reportable instead of invisible.
            "primary_retries": (
                self.primary.retry_stats()
                if callable(getattr(self.primary, "retry_stats", None)) else {}
            ),
            "fallback_retries": (
                self.fallback.retry_stats()
                if callable(getattr(self.fallback, "retry_stats", None)) else {}
            ),
            # The bound, stated as a number, plus the worst call actually
            # measured. A budget nobody can see is a budget nobody trusts.
            "budget_ms": round(self.config.http_timeout_s * 1000.0, 1),
            "connect_budget_ms": round(self.config.connect_timeout_s * 1000.0, 1),
            "max_call_ms": {
                primary: round(float(getattr(self.primary, "max_call_s", 0.0)) * 1000.0, 1),
                fallback: round(float(getattr(self.fallback, "max_call_s", 0.0)) * 1000.0, 1),
            },
            "deadlines": deadline_stats_snapshot(),
        }

    def all_stale(self, mints: Iterable[str], now: float | None = None) -> bool:
        now = now if now is not None else self.clock.now()
        mints = list(mints)
        if not mints:
            return False
        for mint in mints:
            if self.state(mint, now) is DataState.LIVE:
                return False
        return True
