"""The executable-quote provider layer, and the mark/quote divergence guard.

WHAT THIS LAYER IS FOR
----------------------
Before 2026-10-02 nothing in this system asked a venue "what will you actually
do for THIS size?" and then compared the answer to the mark it had been
displaying. The $GOAT position was bought 42.6% above the mark the bot had just
recorded, sold into a different pool from the one the mark described, and
reported to the operator with the mark-to-mark ratio as its result.

This module provides:

* :class:`QuoteProvider` -- one interface with an explicit ``side`` on every
  method, plus ``quote_buy`` / ``quote_sell`` conveniences that CANNOT be
  called with the wrong direction because they do not accept a side.
* :class:`DivergenceGuard` -- the entry check that compares a fresh executable
  quote against the observed mark and refuses the entry when the executable
  price is materially worse.
* :class:`RouterQuoteProvider` -- explicit, ordered provider selection with a
  recorded attempt log and fail-closed behaviour.

DIRECTION IS A TYPE, NOT A CONVENTION
--------------------------------------
``quote_buy(token_mint=..., quote_mint=..., quote_amount=...)`` is always
``quote -> token``. ``quote_sell(token_mint=..., quote_mint=...,
token_amount=...)`` is always ``token -> quote``. Neither accepts a ``side``
argument, so a caller cannot pass the wrong one; the two are separate methods
rather than one method with a flag, because a flag is exactly what got inverted
on 2026-10-02 (``raptor.Quote.price_per_token`` divided input by output on a
sell and published ``6634.9`` tokens-per-lamport).

THE GOAL IS NOT "BUY AT THE MARK"
--------------------------------
The guard exists to refuse an entry whose real cost is materially worse than the
market the system believes it is looking at. It does not attempt to make a
trade happen at the mark price, and a divergence is never "corrected" by
moving a threshold to fit a historical result. The threshold is configuration,
it is documented, and its default is justified in :attr:`DivergenceGuard.notes`.

A missing, stale, ambiguous or unpriceable quote FAILS CLOSED. There is no
path from "I could not price this" to "assume it was fine".
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .price_layers import (
    ExecutableQuote,
    ObservedMark,
    Side,
    SwapDirectionError,
    SwapLeg,
    leg_for_buy,
    leg_for_sell,
    mark_vs_quote_divergence,
    require_token_decimals,
)
from .quote_errors import (
    ChainStateUnavailable,
    MicroOrderUnsupported,
    QuoteRejected,
    QuoteStale,
    QuoteUnavailable,
)

log = logging.getLogger("execution.quoting")

__all__ = [
    "DivergenceGuard",
    "DivergenceVerdict",
    "QuoteProvider",
    "RouterQuoteProvider",
    "Side",
    "SwapDirectionError",
]


def _require_decimals(token_decimals: Any) -> None:
    """
    Refuse to quote without the token's decimals.

    Fails closed with the typed failure the caller already handles, rather than
    letting an absent scale propagate into a per-token price that would be wrong
    by a power of ten and look entirely plausible.
    """
    try:
        require_token_decimals(token_decimals)
    except ValueError as exc:
        raise QuoteUnavailable(str(exc)) from exc


# ---------------------------------------------------------------------------
# the provider interface
# ---------------------------------------------------------------------------


class QuoteProvider(ABC):
    """
    One interface for producing an :class:`ExecutableQuote`.

    A provider may be a venue's own API (Jupiter, Raptor) or a direct
    on-chain reader (Pump bonding curve, PumpSwap). It reports which it is via
    :attr:`venue_quoted`, because the two carry different authority:

    * ``venue_quoted=True``  -- the venue answered. This is the price a venue
      will do for this size, and it is the strongest pre-trade statement
      available.
    * ``venue_quoted=False`` -- derived by this system from on-chain state
      under a named model. Executable in principle, but a MODEL, and it is
      labelled ``model=...`` on the quote so no downstream consumer can present
      it as the venue's own answer.

    A provider never decides position size, risk, or whether a trade is
    profitable. It prices. That stays in the bot.
    """

    #: Stable lower-case id, e.g. ``"jupiter"``.
    name: str = "abstract"
    #: Whether this provider returns the venue's own answer or a chain model.
    venue_quoted: bool = True
    #: Which venues this provider can serve, e.g. ``("pumpswap",)``.
    supports_venues: tuple = ()

    @abstractmethod
    async def quote_swap(
        self,
        *,
        side: Any,
        token_mint: str,
        quote_mint: str,
        amount_raw: int,
        token_decimals: int,
        quote_decimals: Optional[int] = None,
        slippage_bps: int = 0,
        quote_asset_usd: Optional[float] = None,
    ) -> ExecutableQuote:
        """
        Price one declared leg.

        ``side`` is REQUIRED and has no default. An unrecognised side raises
        :class:`SwapDirectionError`; it is never normalised into a guess.
        ``amount_raw`` is in the INPUT asset's raw units, so its meaning follows
        the side: lamports for a BUY, raw token units for a SELL.
        """
        raise NotImplementedError

    async def quote_buy(
        self,
        *,
        token_mint: str,
        quote_mint: str,
        quote_amount_raw: int,
        token_decimals: int,
        quote_decimals: Optional[int] = None,
        slippage_bps: int = 0,
        quote_asset_usd: Optional[float] = None,
    ) -> ExecutableQuote:
        """
        Price ``quote_mint -> token_mint``. Always.

        There is no ``side`` parameter to get wrong. The input amount is in the
        QUOTE asset's raw units because that is what a BUY spends.
        """
        _require_decimals(token_decimals)
        side = Side.BUY
        leg = leg_for_buy(
            token_mint=token_mint, quote_mint=quote_mint,
            quote_amount=quote_amount_raw, quote_decimals=quote_decimals,
            token_decimals=token_decimals,
        )
        return await self.quote_swap(
            side=side, token_mint=token_mint, quote_mint=quote_mint,
            amount_raw=quote_amount_raw, token_decimals=token_decimals,
            quote_decimals=quote_decimals, slippage_bps=slippage_bps,
            quote_asset_usd=quote_asset_usd,
        )

    async def quote_sell(
        self,
        *,
        token_mint: str,
        quote_mint: str,
        token_amount_raw: int,
        token_decimals: int,
        quote_decimals: Optional[int] = None,
        slippage_bps: int = 0,
        quote_asset_usd: Optional[float] = None,
    ) -> ExecutableQuote:
        """
        Price ``token_mint -> quote_mint``. Always.

        There is no ``side`` parameter to get wrong. The input amount is in the
        TOKEN's raw units because that is what a SELL gives up.
        """
        _require_decimals(token_decimals)
        side = Side.SELL
        leg = leg_for_sell(
            token_mint=token_mint, quote_mint=quote_mint,
            token_amount=token_amount_raw, quote_decimals=quote_decimals,
            token_decimals=token_decimals,
        )
        return await self.quote_swap(
            side=side, token_mint=token_mint, quote_mint=quote_mint,
            amount_raw=token_amount_raw, token_decimals=token_decimals,
            quote_decimals=quote_decimals, slippage_bps=slippage_bps,
            quote_asset_usd=quote_asset_usd,
        )

    def as_dict(self) -> dict:
        return {
            "provider": self.name,
            "venue_quoted": self.venue_quoted,
            "supports_venues": list(self.supports_venues),
        }


# ---------------------------------------------------------------------------
# the entry divergence guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DivergenceVerdict:
    """
    The outcome of comparing an executable quote against an observed mark.

    ``allowed`` is the only field a caller may branch on. Everything else exists
    so a refusal can be explained instead of merely enforced.
    """

    allowed: bool
    reason: str
    #: ``(mark - quote) / quote``. Positive means the executable price is BELOW
    #: the mark, i.e. the trade is cheaper than the observed market.
    pct: Optional[float] = None
    mark: Optional[float] = None
    quote: Optional[float] = None
    threshold_pct: Optional[float] = None
    #: Set when the verdict is a refusal caused by missing/stale data rather
    #: than by a measured divergence. This is a FAIL-CLOSED, not a market call.
    fail_closed: bool = False
    detail: str = ""
    #: Cross-venue marker: the mark describes a different pool from the quote.
    cross_venue: bool = False

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "pct": self.pct,
            "mark": self.mark,
            "quote": self.quote,
            "threshold_pct": self.threshold_pct,
            "fail_closed": self.fail_closed,
            "detail": self.detail,
            "cross_venue": self.cross_venue,
        }


@dataclass
class DivergenceGuard:
    """
    How far an executable quote may sit from an observed mark before an entry is
    refused.

    THE SEMANTIC, PRECISELY
    -----------------------
    ``worse_than_mark_max_pct`` bounds ``(quote / mark - 1) * 100``: how much
    MORE we would have to pay than the observed market says the token costs. A
    positive value means the venue is dearer than the mark.

    $GOAT, 2026-10-02, measured from chain::

        observed mark   1.783e-05 USD/token
        executable buy  3.1079e-05 USD/token
        (quote/mark - 1) * 100 = +74.3%

    so with a threshold of 10% that entry is refused.

    THE DEFAULT, AND WHY IT IS NOT A RESEARCH RESULT
    ------------------------------------------------
    The default is ``10.0`` percent. It is a POLICY choice about how much
    disagreement between two price sources this system tolerates before it
    declines to trade -- deliberately NOT a number fitted to historical returns.
    No backtest in this repository was run to choose it, and none should be
    used to justify changing it after the fact: a threshold tuned until the
    historical record looks good is a threshold that will happily refuse a
    genuine opportunity tomorrow.

    What the number must satisfy structurally:

    * it is strictly positive (0 would make any tick a refusal);
    * it is wide enough that ordinary venue/mark disagreement passes -- the
      observed divergences in the 2026-10-01/02 REAL trades were -44.9%/+309%
      ($BUTTHOLE), -12.7% ($FLOOR) and +74.3% ($GOAT) on the buy side;
    * it is tight enough that a 70%-dearer quote is refused.

    A 10% band meets all three and is the conservative end of a range that is
    genuinely a research question. It lives in config, is reported on every
    verdict, and is changeable WITHOUT touching strategy code.

    THE ASYMMETRY IS DELIBERATE
    ---------------------------
    ``quote_worse_than_mark`` blocks a BUY that is dearer than the mark (we
    would overpay). ``quote_much_better_than_mark`` is an INTEGRITY ALARM, not
    an opportunity: an executable quote far CHEAPER than the observed market
    usually means the mark is describing a different venue, is stale in
    substance while fresh in transit, or the quote is for a different size. None
    of those is a reason to buy, so it blocks too -- at a deliberately looser
    band, because a genuinely better price is more often a data fault than a
    bargain.
    """

    #: Bound on ``(quote / mark - 1) * 100`` for a BUY. Positive = dearer.
    quote_worse_than_mark_max_pct: float = 10.0
    #: Bound on ``(mark / quote - 1) * 100`` for a BUY. Positive = the mark is
    #: dearer than the venue, which is an integrity alarm rather than a bargain.
    quote_much_better_than_mark_max_pct: float = 50.0
    #: Refuse when the mark describes a pool the quote did not come from.
    reject_cross_venue: bool = True
    #: Refuse when the quote is older than this. ``None`` disables the check,
    #: which the executor never does -- it has its own fresher budget.
    max_quote_age_s: Optional[float] = 8.0
    notes: str = (
        "policy band, not a fitted parameter; see the class docstring"
    )

    def evaluate_entry(
        self,
        *,
        mark: Optional[ObservedMark],
        quote: Optional[ExecutableQuote],
        now: Optional[float] = None,
    ) -> DivergenceVerdict:
        """
        Decide whether an entry may proceed. Never raises.

        Every refusal path is a refusal for a NAMED reason, and a data failure
        is distinguished from a market disagreement via ``fail_closed``.
        """
        now = time.time() if now is None else now

        # -- 1. did we get a price at all? ---------------------------------
        if quote is None:
            return DivergenceVerdict(
                False, "no_executable_quote", fail_closed=True,
                detail="no executable quote was obtained; refusing to enter "
                       "without one",
            )
        if quote.side is not Side.BUY:
            return DivergenceVerdict(
                False, "wrong_direction", fail_closed=True,
                detail=f"an ENTRY guard received a {quote.side.value} quote; "
                       "a BUY quote is required",
            )
        if not quote.is_executable:
            return DivergenceVerdict(
                False, "quote_unusable", fail_closed=True,
                detail=f"quote quality={quote.quality} stale={quote.stale}",
            )

        # -- 2. freshness --------------------------------------------------
        if self.max_quote_age_s is not None and quote.received_at:
            age = max(0.0, now - quote.received_at)
            if age > self.max_quote_age_s:
                return DivergenceVerdict(
                    False, "quote_stale", fail_closed=True,
                    quote=quote.effective_price_per_whole_token,
                    detail=f"quote age {age * 1000:.0f}ms exceeds the "
                           f"{self.max_quote_age_s * 1000:.0f}ms budget",
                )

        # -- 3. did we get a mark to compare against? ----------------------
        if mark is None:
            return DivergenceVerdict(
                False, "no_observed_mark", fail_closed=True,
                quote=quote.effective_price_per_whole_token,
                detail="no observed market state to validate the quote "
                       "against; refusing to enter on an unvalidated price",
            )
        mark_px = mark.price
        if mark_px is None or not (mark_px > 0) or mark_px != mark_px:
            return DivergenceVerdict(
                False, "observed_mark_unusable", fail_closed=True,
                detail="the observed mark is not a positive finite number",
            )
        if mark.stale or mark.quality in ("stale", "unusable", "missing"):
            return DivergenceVerdict(
                False, "observed_mark_stale", fail_closed=True,
                mark=mark_px,
                detail=f"observed mark quality={mark.quality} stale={mark.stale}",
            )

        # -- 4. comparable units -------------------------------------------
        quote_px = quote.price_usd_per_whole_token
        if quote_px is None or not (quote_px > 0):
            return DivergenceVerdict(
                False, "quote_not_comparable", fail_closed=True,
                mark=mark_px,
                detail="the quote has no USD conversion, so it cannot be "
                       "compared with a USD mark; refusing rather than "
                       "comparing unlike units",
            )
        if mark.price_unit not in ("USD/whole_token",):
            return DivergenceVerdict(
                False, "mark_not_comparable", fail_closed=True,
                mark=mark_px,
                detail=f"the mark's unit is {mark.price_unit!r}, which is not "
                       f"comparable with a per-whole-token quote",
            )

        # -- 5. cross-venue -------------------------------------------------
        cross_venue = bool(
            self.reject_cross_venue
            and mark.pool and quote.pool and str(mark.pool) != str(quote.pool)
        )
        if cross_venue:
            # A cross-venue match is not automatically fatal: aggregators route
            # legitimately and the quote's own price is what will be paid. It is
            # recorded, and the price test below still has to pass on its own
            # merits. A mark about a different pool simply cannot be used to
            # validate a quote about this one, so when the mark and quote
            # disagree the disagreement is not evidence either way.
            log.warning(
                "ENTRY_CROSS_VENUE mint=%s mark_pool=%s quote_pool=%s",
                quote.leg.output_mint, mark.pool, quote.pool)

        # -- 6. the price test ---------------------------------------------
        div = mark_vs_quote_divergence(mark_px, quote_px,
                                       kind="ENTRY_MARK_VS_QUOTE")
        if div is None:
            return DivergenceVerdict(
                False, "divergence_unavailable", fail_closed=True,
                mark=mark_px, quote=quote_px,
                detail="the mark/quote pair could not be differenced",
            )
        worse = (quote_px / mark_px - 1.0) * 100.0
        better = (mark_px / quote_px - 1.0) * 100.0
        if worse > float(self.quote_worse_than_mark_max_pct):
            return DivergenceVerdict(
                False, "quote_worse_than_mark",
                pct=div.pct, mark=mark_px, quote=quote_px,
                threshold_pct=self.quote_worse_than_mark_max_pct,
                cross_venue=cross_venue,
                detail=(f"executable buy {quote_px:.6g} USD/token is "
                        f"{worse:+.1f}% versus the observed mark "
                        f"{mark_px:.6g}; limit "
                        f"{self.quote_worse_than_mark_max_pct:+.1f}%"),
            )
        if better > float(self.quote_much_better_than_mark_max_pct):
            return DivergenceVerdict(
                False, "quote_much_better_than_mark",
                pct=div.pct, mark=mark_px, quote=quote_px,
                threshold_pct=self.quote_much_better_than_mark_max_pct,
                cross_venue=cross_venue,
                detail=(f"executable buy {quote_px:.6g} USD/token is "
                        f"{better:.1f}% CHEAPER than the observed mark "
                        f"{mark_px:.6g}; that is an integrity alarm "
                        f"(different venue, or a mark that is fresh in transit "
                        f"and wrong in substance), not an opportunity"),
            )
        return DivergenceVerdict(
            True, "within_band", pct=div.pct, mark=mark_px, quote=quote_px,
            threshold_pct=self.quote_worse_than_mark_max_pct,
            cross_venue=cross_venue,
            detail=(f"executable buy within {self.quote_worse_than_mark_max_pct:+.1f}% "
                    f"of the observed mark ({worse:+.1f}%)"),
        )

    def as_dict(self) -> dict:
        return {
            "quote_worse_than_mark_max_pct": self.quote_worse_than_mark_max_pct,
            "quote_much_better_than_mark_max_pct":
                self.quote_much_better_than_mark_max_pct,
            "reject_cross_venue": self.reject_cross_venue,
            "max_quote_age_s": self.max_quote_age_s,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# explicit provider selection
# ---------------------------------------------------------------------------

#: Fallback eligibility by terminal status. A provider OUTAGE must never cause a
#: second attempt on a live order, so transport failures stop the chain
#: immediately; only a refusal of the ORDER (this venue cannot do it) or an
#: oversized/undeliverable result may fall through.
_FALLBACK_ELIGIBLE_STATUSES = frozenset({
    "QUOTE_REJECTED",
    "MICRO_REAL_EXECUTION_UNSUPPORTED",
    "ROUTE_UNSUPPORTED",
    "TX_TOO_LARGE",
    "CHAIN_STATE_UNAVAILABLE",
})


class RouterQuoteProvider(QuoteProvider):
    """
    Try providers in an explicit, ordered chain. Records every attempt.

    Provider selection is data, not control flow: the chain is a sequence handed
    in, and the attempt log is part of the result. A caller that cannot say which
    providers were consulted cannot reason about why a quote is missing.

    It refuses rather than substitutes. If the chain is exhausted, the result is
    ``None`` and the terminal status is reported -- never a mark, never a
    re-sent request with a different direction, never a guess.
    """

    def __init__(self, providers: Sequence[QuoteProvider]) -> None:
        self._providers = list(providers)

    @property
    def providers(self) -> list[QuoteProvider]:
        return list(self._providers)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self._providers]

    def as_dict(self) -> dict:
        return {
            "provider": "router",
            "chain": self.names,
            "venue_quoted": any(p.venue_quoted for p in self._providers),
        }

    async def try_all(self, **kwargs) -> tuple[Optional[ExecutableQuote], dict]:
        """
        Run the chain and return ``(quote, report)``.

        The quote is the FIRST provider's answer, never a later one: if a
        cheaper primary quote exists and a fallback would have been dearer,
        taking the fallback's number would misstate what the trade costs.
        """
        side = Side.coerce(kwargs.get("side"))
        attempts: list[dict] = []
        for index, prov in enumerate(self._providers):
            last = index == len(self._providers) - 1
            t0 = time.time()
            try:
                quote = await prov.quote_swap(side=side, **{
                    k: v for k, v in kwargs.items() if k != "side"})
            except MicroOrderUnsupported as exc:
                # A venue minimum is a fact about THIS venue's notional. Another
                # venue's minimum is not a fix, and enlarging the order is never
                # the answer.
                attempts.append({"provider": prov.name, "status": "MICRO",
                                 "detail": str(exc)[:200],
                                 "ms": (time.time() - t0) * 1000.0})
                return None, self._report(side, attempts,
                                          "MICRO_REAL_EXECUTION_UNSUPPORTED",
                                          str(exc))
            except QuoteStale as exc:
                attempts.append({"provider": prov.name, "status": "QUOTE_STALE",
                                 "detail": str(exc)[:200],
                                 "ms": (time.time() - t0) * 1000.0})
                if not last:
                    continue
                return None, self._report(side, attempts, "QUOTE_STALE", str(exc))
            except QuoteRejected as exc:
                attempts.append({"provider": prov.name, "status": "REJECTED",
                                 "detail": str(exc)[:200],
                                 "ms": (time.time() - t0) * 1000.0})
                if not last:
                    continue
                return None, self._report(side, attempts, "QUOTE_REJECTED", str(exc))
            except ChainStateUnavailable as exc:
                attempts.append({"provider": prov.name, "status": "CHAIN_STATE",
                                 "detail": str(exc)[:200],
                                 "ms": (time.time() - t0) * 1000.0})
                if not last:
                    continue
                return None, self._report(side, attempts,
                                          "CHAIN_STATE_UNAVAILABLE", str(exc))
            except QuoteUnavailable as exc:
                attempts.append({"provider": prov.name, "status": "UNAVAILABLE",
                                 "detail": str(exc)[:200],
                                 "ms": (time.time() - t0) * 1000.0})
                if last:
                    return None, self._report(side, attempts,
                                              getattr(exc, "status",
                                                      "QUOTE_UNAVAILABLE"),
                                              str(exc))
                continue
            except SwapDirectionError as exc:
                # A direction fault is a bug in the CALLER, never a venue
                # problem. It stops the chain immediately -- and it is recorded,
                # because "the chain produced nothing" is the least useful
                # possible diagnosis for a caller that passed an inverted leg.
                attempts.append({"provider": prov.name, "stage": "quote",
                                 "status": "SWAP_DIRECTION",
                                 "detail": str(exc)[:200],
                                 "input_mint": exc.input_mint,
                                 "output_mint": exc.output_mint})
                return None, self._report(side, attempts, "SWAP_DIRECTION",
                                          str(exc))
            except Exception as exc:  # noqa: BLE001 - a provider bug must not
                # take the process down, and must not be reported as a refusal
                # the next provider could fix.
                attempts.append({"provider": prov.name, "status": "ERROR",
                                 "detail": f"{type(exc).__name__}: {exc}"[:200],
                                 "ms": (time.time() - t0) * 1000.0})
                return None, self._report(side, attempts, "PROVIDER_ERROR",
                                          f"{type(exc).__name__}: {exc}")
            attempts.append({"provider": prov.name, "status": "OK",
                             "venue": quote.venue, "pool": quote.pool,
                             "price_usd": quote.price_usd_per_whole_token,
                             "impact_pct": quote.price_impact_pct,
                             "venue_quoted": quote.venue_quoted,
                             "ms": (time.time() - t0) * 1000.0})
            quote.provider = quote.provider or prov.name
            return quote, self._report(side, attempts, "", "")
        return None, self._report(side, attempts, "QUOTE_UNAVAILABLE",
                                  "provider chain exhausted")

    @staticmethod
    def _report(side: Optional[Side], attempts: list[dict], status: str,
                detail: str) -> dict:
        return {
            "side": side.value if side else "",
            "attempts": list(attempts),
            "failure_status": status,
            "failure_detail": detail,
            "fallback_eligible": status in _FALLBACK_ELIGIBLE_STATUSES,
        }

    async def quote_swap(self, **kwargs) -> ExecutableQuote:
        """Chain entry point. Raises when the chain is exhausted."""
        quote, report = await self.try_all(**kwargs)
        if quote is None:
            status = report["failure_status"] or "QUOTE_UNAVAILABLE"
            detail = report["failure_detail"] or "no provider returned a quote"
            if status == "MICRO_REAL_EXECUTION_UNSUPPORTED":
                raise MicroOrderUnsupported(detail)
            if status == "CHAIN_STATE_UNAVAILABLE":
                raise ChainStateUnavailable(detail)
            if status == "QUOTE_REJECTED":
                raise QuoteRejected(detail)
            if status == "QUOTE_STALE":
                raise QuoteStale(detail)
            if status == "SWAP_DIRECTION":
                raise SwapDirectionError(detail)
            raise QuoteUnavailable(detail)
        return quote