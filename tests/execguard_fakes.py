"""Offline doubles for the executable-quote layer the REAL entry guard needs.

WHY THIS EXISTS
---------------
Since 2026-10-02 a REAL entry is refused unless a fresh EXECUTABLE quote can be
obtained and compared against the observed mark. That is the point of the guard:
before it, a $GOAT entry filled 74.3% above the mark the trade was justified by
and nothing had asked.

A test that drives ``_execute_real_buy`` therefore has to supply a quote, the
same way a real deployment supplies one over the network. This module is that
supply, entirely offline: no socket, no key, no transaction.

WHAT IT DOES NOT DO
-------------------
It never signs, builds or broadcasts. It returns
:class:`~autopaper.execution.price_layers.ExecutableQuote` objects and nothing
else. That is deliberate -- a test double that could reach a broadcaster would be
the exact hazard the guard is there to prevent.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from autopaper.execution.price_layers import ExecutableQuote, leg_for_buy
from autopaper.execution.solana import WSOL_MINT

LAMPORTS_PER_SOL = 1_000_000_000


class FakeAggregatorQuote:
    """
    A pre-built :class:`ExecutableQuote` for tests that need one on demand.

    Takes explicit amounts rather than a target price, so a test can state the
    leg it means without the double inferring anything.
    """

    provider = "fake"
    venue_quoted = True
    supports_venues = ("aggregated",)

    def __init__(self, *, side, token_mint, quote_mint, quote_amount=None,
                 token_amount=None, token_decimals, expected_token_amount=None,
                 expected_quote_amount=None, quote_asset_usd=None,
                 venue=None, pool=None, provider=None, received_at=0.0) -> None:
        from autopaper.execution.price_layers import (
            ExecutableQuote,
            Side,
            leg_for_buy,
            leg_for_sell,
        )
        s = Side.coerce(side)
        if s is Side.BUY:
            leg = leg_for_buy(
                token_mint=token_mint, quote_mint=quote_mint,
                quote_amount=int(quote_amount), token_decimals=token_decimals,
                expected_token_amount=int(expected_token_amount or 1))
        else:
            leg = leg_for_sell(
                token_mint=token_mint, quote_mint=quote_mint,
                token_amount=int(token_amount), token_decimals=token_decimals,
                expected_quote_amount=int(expected_quote_amount or 1))
        self.quote = ExecutableQuote(
            side=s, leg=leg, provider=provider or self.provider, venue=venue,
            pool=pool, price_impact_pct=0.1, slippage_bps=500,
            quoted_at=received_at, received_at=received_at,
            quote_asset_usd=quote_asset_usd, venue_quoted=True,
            provenance={"fake": True},
        )

    def __getattr__(self, name):
        return getattr(self.__dict__["quote"], name)

    def as_dict(self):
        return self.quote.as_dict()


class FakeQuoteProvider:
    """
    A venue that quotes at a chosen USD price per whole token.

    ``price_usd`` is the price it will charge. Set it near the observed mark to
    let an entry through and far above it to see the guard refuse; the refusal
    is the assertion the 2026-10-02 audit asked for and it must be reachable
    without a network.
    """

    venue_quoted = True

    def __init__(self, *, price_usd: float = 0.0, price_fn: Any = None,
                 sol_usd: float = 200.0,
                 decimals: int = 9, venue: str = "pumpswap",
                 pool: Optional[str] = "FakePool1111111111111111111111111111",
                 provider: str = "fake", fail_with: Optional[Exception] = None,
                 age_s: float = 0.0, now_fn: Any = None) -> None:
        self.price_usd = float(price_usd)
        #: An optional callable returning the USD price to quote at. Used by
        #: fixtures whose mark varies per test: the double then represents a
        #: venue that AGREES with the observed market, which is the case under
        #: test in every one of those tests.
        self._price_fn = price_fn
        self.sol_usd = float(sol_usd)
        self.decimals = int(decimals)
        self.venue = venue
        self.pool = pool
        self.provider = provider
        self.fail_with = fail_with
        self.age_s = float(age_s)
        #: The clock the quotes are stamped from. It MUST be the engine's clock:
        #: freshness is measured against it, and a double that stamps wall-clock
        #: time into a FakeClock-driven engine reports itself as billions of
        #: seconds stale, which measures the double rather than the guard.
        self.now_fn = now_fn
        self.calls: list[dict] = []

    # -- the QuoteProvider surface ---------------------------------------

    async def quote_buy(self, *, token_mint: str, quote_mint: str,
                        quote_amount_raw: int, token_decimals: int,
                        quote_decimals: Optional[int] = None,
                        slippage_bps: int = 500,
                        quote_asset_usd: Optional[float] = None) -> ExecutableQuote:
        self.calls.append({"side": "BUY", "token_mint": token_mint,
                           "quote_mint": quote_mint,
                           "amount": quote_amount_raw})
        if self.fail_with is not None:
            raise self.fail_with
        sol = self.sol_usd if quote_asset_usd is None else float(quote_asset_usd)
        dec = self.decimals if token_decimals is None else int(token_decimals)
        price = self._price_fn() if self._price_fn is not None else self.price_usd
        usd = (quote_amount_raw / LAMPORTS_PER_SOL) * sol
        out = int(usd / price * (10 ** dec)) if price and price > 0 else 0
        leg = leg_for_buy(
            token_mint=token_mint,
            quote_mint=quote_mint or WSOL_MINT,
            quote_amount=int(quote_amount_raw),
            quote_decimals=quote_decimals,
            token_decimals=dec,
            expected_token_amount=max(1, out),
        )
        now = (float(self.now_fn()) if self.now_fn is not None
               else time.time())
        received = now - self.age_s
        return ExecutableQuote(
            side=leg.side, leg=leg, provider=self.provider, venue=self.venue,
            pool=self.pool, price_impact_pct=0.1,
            slippage_bps=slippage_bps, min_output_amount=max(1, int(out * 0.99)),
            quoted_at=received, received_at=received, context_slot=1,
            quote_asset_usd=sol, venue_quoted=True, stale=self.age_s > 8.0,
            quality="live",
            provenance={"fake": True},
        )

    async def quote_swap(self, **kwargs: Any) -> ExecutableQuote:
        return await self.quote_buy(**{
            k: v for k, v in kwargs.items()
            if k in ("token_mint", "quote_mint", "quote_amount_raw",
                     "token_decimals", "quote_decimals", "slippage_bps",
                     "quote_asset_usd")})

    async def quote_sell(self, **kwargs: Any) -> ExecutableQuote:
        raise NotImplementedError(
            "FakeQuoteProvider only quotes buys; exits use the engine's own "
            "venue probe in these tests")


def install_fake_executable_quote(trader: Any, *, price_usd: float = 0.0,
                                  price_fn: Any = None,
                                  sol_usd: float = 200.0,
                                  decimals: int = 9,
                                  venue: str = "pumpswap",
                                  pool: Optional[str] = None,
                                  fail_with: Optional[Exception] = None,
                                  age_s: float = 0.0) -> FakeQuoteProvider:
    """
    Install a coherent offline quote source and decimals reader on a trader.

    Both seams are code-level and explicit. There is deliberately no environment
    variable that points at a local quote source, because a configurable quote
    source is a configurable price.
    """
    provider = FakeQuoteProvider(
        price_usd=price_usd, price_fn=price_fn,
        sol_usd=sol_usd, decimals=decimals,
        venue=venue, pool=pool, fail_with=fail_with, age_s=age_s,
        now_fn=getattr(getattr(trader, "clock", None), "now", None),
    )
    trader.set_quote_provider(provider)
    trader.set_token_decimals_reader(lambda mint: decimals)
    return provider

class FakeBuildProvider:
    """
    A BUILD-provider double: ``quote(input, output, amount, slippage, side=)``.

    This is the shape :class:`~autopaper.execution.pump.VenueScopedQuoteProvider`
    wraps -- a provider that quotes AND builds. It is quote-only in practice:
    there is no ``build``, no signer and no broadcaster.
    """

    provider_name = "fake-build"

    def __init__(self, *, price_usd: float = 1e-5, sol_usd: float = 200.0,
                 venue: str = "pumpswap", pool: Optional[str] = None,
                 program_id: str = "",
                 token_decimals: int = 6,
                 fail_with: Optional[Exception] = None,
                 now_fn: Any = None) -> None:
        self.price_usd = float(price_usd)
        self.sol_usd = float(sol_usd)
        self.venue = venue
        self.pool = pool
        self.program_id = program_id
        self.token_decimals = token_decimals
        self.fail_with = fail_with
        self.now_fn = now_fn
        self.calls: list[dict] = []

    async def quote(self, input_mint: str, output_mint: str, in_amount: int,
                    slippage_bps: int, *, side: Optional[str] = None):
        from autopaper.execution.price_layers import resolve_side_from_mints
        from autopaper.execution.raptor import RouteStep

        if self.fail_with is not None:
            raise self.fail_with
        resolved = resolve_side_from_mints(input_mint, output_mint,
                                           declared_side=side)
        self.calls.append({"side": resolved.value, "in": input_mint,
                           "out": output_mint, "amount": int(in_amount)})
        now = (float(self.now_fn()) if self.now_fn is not None else time.time())
        # One whole token is worth 1/price USD, so the raw output for a
        # lamport input follows from the SOL notional.
        sol = in_amount / LAMPORTS_PER_SOL
        usd = sol * self.sol_usd
        raw_token_value = 1.0 / self.price_usd / (10 ** self.token_decimals)
        out_amount = int(usd * raw_token_value)
        from autopaper.execution.raptor import Quote
        return Quote(
            input_mint=input_mint, output_mint=output_mint,
            in_amount=int(in_amount),
            expected_out_amount=max(1, out_amount),
            min_out_amount=max(1, int(out_amount * 0.98)),
            price_impact_pct=0.1, slippage_bps=int(slippage_bps),
            route_labels=[self.venue],
            route_steps=[RouteStep(dex=self.venue, program_id=self.program_id,
                                   pool=self.pool or "",
                                   input_mint=input_mint,
                                   output_mint=output_mint,
                                   amount_in=str(in_amount),
                                   amount_out=str(out_amount))],
            quote_created_at=now, quote_received_at=now, context_slot=1,
            side=resolved.value,
        )

    def provider_name(self) -> str:
        return "fake-build"
