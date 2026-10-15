"""
Shared offline doubles for the execution-aware PAPER tests.

Extracted as a module rather than imported test-to-test, matching the existing
``tests/pumpdirect_fakes.py`` convention. Everything here is a stand-in for a
NETWORK service: no RPC, no Telegram, no HTTP, no wallet.

The quote doubles are deliberately FAITHFUL rather than permissive. The real
``QuoteService`` always stamps ``quote_age_ms`` and ``max_quote_age_ms``, so a
double that omitted them would silently disable the staleness gate and make a
test assert that a stale quote was acceptable. A double that "repairs" a wrong
mint would likewise hide the exact defect the wrong-mint tests exist to catch,
which is why ``FakeQuoteService`` has a ``verbatim`` mode.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from decimal import Decimal

from autopaper.exits import ExitDecision
from autopaper.models import (
    AlertSnapshot,
    DataState,
    Decision,
    DecisionReport,
    ExitStyle,
    Regime,
    Route,
)
from autopaper.same_market.models import WSOL_MINT
from autopaper.same_market.numeric import effective_price

from autopaper.paper_live.arms import parse_arms
from autopaper.paper_live.bridge import TelegramPaperBridge
from autopaper.paper_live.fills import FillModel
from autopaper.paper_live.quotes import SolReference
from autopaper.paper_live.schema import (
    ExecutionNativeReference,
    QuotePoint,
)


MINT = "APxxh2tWCh95tbHitFJchcdbMsctSBgRS2Zzju9Bpump"
PAIR = "9LCvns4zZsYAinZcEWRdzEEz7P6V84K3c5ZYCLT6vzMb"
SOL_USD = 122.36
SOL_USD_D = Decimal("122.36")
BUY_LAMPORTS = 4_086_302
BUY_TOKENS_RAW = 9_118_750_271
SELL_TOKENS_RAW = 9_118_750_271
SELL_LAMPORTS = 3_989_025


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class FakeMarket:
    """DexScreener stand-in: observations only, never quotes."""

    def __init__(self, price=None, liquidity=60000.0, volume5=1000.0,
                 mcap=9000.0):
        # Defaults to the executable quote's own USD price, so the mark and the
        # quote are coherent BY CONSTRUCTION. The executable-vs-mark entry
        # guard is real, and a mark hardcoded next to a quote fixture drifts out
        # of agreement with it the moment either number changes -- which showed
        # up as a spurious +621% rejection that said nothing about the code.
        self.price = MARK_USD if price is None else price
        self.liquidity = liquidity
        self.volume5, self.mcap = volume5, mcap
        self.calls = 0

    async def quote_async(self, mint):
        self.calls += 1

        class O:
            pass

        class Q:
            pass

        o = O()
        o.price_usd = self.price
        o.liquidity_usd = self.liquidity
        o.volume_5m_usd = self.volume5
        o.market_cap_usd = self.mcap
        o.pair_address = PAIR
        o.dex_id = "pumpfun"
        o.observed_at = time.time()
        q = Q()
        q.observation = o
        return q


def buy_point(**over) -> QuotePoint:
    # COHERENT WITH THE MARKET MARK. The executable-vs-mark entry guard is
    # real, so a quote priced at an unrelated number would be refused for a
    # +600% gap that says nothing about the code under test. A few-percent
    # premium over the mark is what an executable quote actually looks like.
    price = Decimal("0.0000204") * Decimal("1.02") / SOL_USD_D
    base = dict(
        side="BUY", provider="pump_direct", market_type="PUMP_BONDING_CURVE",
        token_mint=MINT, quote_mint=WSOL_MINT,
        input_amount_raw=BUY_LAMPORTS, input_decimals=9,
        input_amount_decimal=Decimal("0.004086302"),
        output_amount_raw=BUY_TOKENS_RAW, output_decimals=6,
        output_amount_decimal=Decimal("9118.750271"),
        effective_price=price, effective_price_usd=price * SOL_USD_D,
        price_impact_pct=1.2, fee_bps=120, fee_amount=Decimal("0.000005"),
        venue="pump.fun", pool_or_curve=PAIR, route=("pump_direct",),
        # The venue PIN must name the exact book, not the family. "Pump.fun"
        # covers both a bonding curve and a graduated pool, and an exit held to
        # that label would not be held to any real market.
        venue_id=f"PUMPSWAP:{PAIR}",
        identity="PASS", available=True, missing_reason="",
        quote_timestamp=time.time(), quote_age_ms=0.0, max_quote_age_ms=30_000.0,
        # THE EXECUTION-NATIVE REFERENCE. The real quote path derives it from
        # the same account read, at the same slot, as the executable price --
        # there is no second data source. The double reproduces that: the spot
        # is the executable price divided by the same 2% premium the fixture
        # already encodes, so the guard sees a coherent +2% cost of transacting
        # this size rather than a fixture artefact.
        execution_native_reference=ExecutionNativeReference(
            market_id=f"PUMPSWAP:{PAIR}", mint=MINT, base_mint=MINT,
            quote_mint=WSOL_MINT, side="BUY",
            reference_price=float(price) / 1.02,
            numerator_raw=139_000_000, denominator_raw=6_800_000_000,
            calculation_method="PUMPSWAP_EFFECTIVE_RESERVES",
            venue="pump.fun", pool_or_curve=PAIR, provider="pump_direct",
            slot=452_922_948, observed_at=time.time(), state_age_ms=1.0,
            token_decimals=6,
        ),
    )
    base.update(over)
    return QuotePoint(**base)


def sell_point(**over) -> QuotePoint:
    # A sell executes slightly worse than the mark for the same reason.
    # A SELL pays OUT quote asset and spends IN tokens, so the price is
    # output/input. Calling the shared helper with (tokens, lamports) returns
    # tokens-per-SOL -- the inverse -- which is how this double carried an
    # absurd price (2.3e6 USD/token) for as long as nothing read it. The
    # executable-vs-mark guard reads it now, so it is computed correctly here.
    price = effective_price(SELL_LAMPORTS, SELL_TOKENS_RAW,
                            input_decimals=9, output_decimals=6)
    base = dict(
        side="SELL", provider="pump_direct", market_type="PUMP_BONDING_CURVE",
        token_mint=MINT, quote_mint=WSOL_MINT,
        input_amount_raw=SELL_TOKENS_RAW, input_decimals=6,
        input_amount_decimal=Decimal("9118.750271"),
        output_amount_raw=SELL_LAMPORTS, output_decimals=9,
        output_amount_decimal=Decimal("0.003989025"),
        effective_price=price, effective_price_usd=price * SOL_USD_D,
        price_impact_pct=-1.1, fee_bps=120, fee_amount=Decimal("0.0000048"),
        venue="pump.fun", pool_or_curve=PAIR, route=("pump_direct",),
        # The venue PIN must name the exact book, not the family. "Pump.fun"
        # covers both a bonding curve and a graduated pool, and an exit held to
        # that label would not be held to any real market.
        venue_id=f"PUMPSWAP:{PAIR}",
        identity="PASS", available=True, missing_reason="",
        quote_timestamp=time.time(), quote_age_ms=0.0, max_quote_age_ms=30_000.0,
        # THE EXECUTION-NATIVE REFERENCE. The real quote path derives it from
        # the same account read, at the same slot, as the executable price --
        # there is no second data source. The double reproduces that: the spot
        # is the executable price divided by the same 2% premium the fixture
        # already encodes, so the guard sees a coherent +2% cost of transacting
        # this size rather than a fixture artefact.
        execution_native_reference=ExecutionNativeReference(
            market_id=f"PUMPSWAP:{PAIR}", mint=MINT, base_mint=MINT,
            quote_mint=WSOL_MINT, side="SELL",
            reference_price=float(price) / 1.02,
            numerator_raw=139_000_000, denominator_raw=6_800_000_000,
            calculation_method="PUMPSWAP_EFFECTIVE_RESERVES",
            venue="pump.fun", pool_or_curve=PAIR, provider="pump_direct",
            slot=452_922_948, observed_at=time.time(), state_age_ms=1.0,
            token_decimals=6,
        ),
    )
    base.update(over)
    return QuotePoint(**base)


#: The executable BUY quote's own USD price, straight from the arithmetic
#: fixture. This is the QUOTE, not the mark.
QUOTE_USD = float(buy_point().effective_price_usd)

#: The OBSERVED MARK the fake market reports, set 2% BELOW the executable quote.
#:
#: Two facts have to hold at once for the guards to be meaningful: the mark and
#: the quote must be the same order of magnitude (so the entry guard is not
#: firing on a fixture artefact) and they must NOT be equal (so "the mark is
#: never the fill" remains demonstrable). Deriving one from the other keeps both
#: true when either number is edited.
MARK_USD = QUOTE_USD / 1.02


class FakeQuoteService:
    """Mint-aware quote double. A wrong mint can be requested explicitly."""

    def __init__(self, *, buy=None, sell=None, verbatim=False):
        self.buy = buy if buy is not None else buy_point()
        self.sell = sell if sell is not None else sell_point()
        self.buy_calls: list[dict] = []
        self.sell_calls: list[dict] = []
        #: A correct provider answers about the mint that was asked for, so the
        #: default double restamps. ``verbatim=True`` models a BROKEN provider
        #: that answers about whatever it was configured with, which is what the
        #: wrong-mint tests need -- restamping would silently repair the very
        #: defect under test.
        self.verbatim = verbatim

    def _maybe_restamp(self, point: QuotePoint, mint: str) -> QuotePoint:
        if self.verbatim or point.token_mint == mint:
            return point
        return replace(point, token_mint=mint)

    async def quote_buy(self, **kw):
        self.buy_calls.append(kw)
        return self._maybe_restamp(self.buy, kw["mint"])

    async def quote_sell(self, **kw):
        self.sell_calls.append(kw)
        return self._maybe_restamp(self.sell, kw["mint"])

    async def quote_sell_best(self, **kw):
        self.sell_calls.append(kw)
        return (self._maybe_restamp(self.sell, kw["mint"]), [])


class Gate:
    def __init__(self, status="AUTHORIZED", authorized=True):
        self.status, self.authorized = status, authorized


def real_report(*, decision=Decision.BUY, gate=None, vtl=0.75,
                mint=MINT, symbol="T") -> DecisionReport:
    """
    A genuine ``DecisionReport``.

    ``features`` is a plain dict with the real key names, exactly as
    ``DecisionEngine`` builds it -- that is the whole point.
    """
    return DecisionReport(
        candidate_id="cand_abc", mint=mint, symbol=symbol, decision=decision,
        regime=Regime.UNCERTAIN, route=Route.FAST_ENTRY, signal_quality=0.72,
        confidence=0.6, thesis="", risk_flags=(), entry_conditions=(),
        invalidation_conditions=(), exit_plan={},
        preferred_exit_style=ExitStyle.FAST_FAILURE, decided_at=time.time(),
        features={"volume_to_liquidity": vtl, "signal_quality": 0.72,
                  "liquidity_usd": 60000.0},
        data_state=DataState.LIVE, latency_ms=1.0,
        v3c=gate or Gate(), l25=gate or Gate(),
    )


class StubExit:
    """
    Stands in for the production ``ExitPolicy.assess``.

    The REAL policy is exercised by the production suite; what these tests need
    is control over WHEN an exit fires, so the shape matches the real signature
    exactly. A hold-until window in the future means HOLD, which is how the
    max-hold and precedence behaviours are pinned down.
    """

    def __init__(self, hold_until: float | None = None):
        self.hold_until = hold_until
        self.calls = 0
        self.seen: list = []

    def assess(self, position, *, now, mark_price, data_state,
               momentum_30s_pct=None, structure_30s_pct=None,
               liquidity_change_pct=None, data_invalid_since=None):
        self.calls += 1
        self.seen.append(position)
        if self.hold_until is not None and now < self.hold_until:
            return ExitDecision(
                "HOLD", "still inside the hold window",
                ExitStyle.FAST_FAILURE, trigger="min_hold",
            )
        return ExitDecision(
            "EXIT", "test exit", ExitStyle.FAST_FAILURE, trigger="test",
        )


def alert(*, mint=MINT, message_id=4242, chat="MooDengPresidentCallers",
          symbol="T", ambiguous=False) -> AlertSnapshot:
    return AlertSnapshot(
        candidate_id=f"cand_{message_id}", source="telegram", chat=chat,
        message_id=message_id, source_timestamp=time.time() - 1.2,
        received_timestamp=time.time(), raw_text=f"BUY ${mint}",
        mint=mint, symbol=symbol, name=symbol, fields={},
        provenance={"source": "telegram"}, parse_warnings=(),
    )


def bridge(**over) -> TelegramPaperBridge:
    quotes = over.pop("quotes", None)
    exits = over.pop("exits", None)
    return TelegramPaperBridge(
        arms=over.pop("arms", None) or parse_arms([]),
        decision_engine=over.pop("decision_engine", None)
        or _UnusedDecision(),
        exit_policy=exits or StubExit(hold_until=time.time() + 3600.0),
        quote_service=quotes or FakeQuoteService(),
        market=over.pop("market", None) or FakeMarket(),
        token_decimals_for=lambda mint: 6,
        sol_reference=SolReference(SOL_USD, time.time(), "test"),
        fill_model=FillModel(),
        starting_bankroll_usd=over.pop("bankroll", 5.0),
        root=over.pop("root", None),
        **over,
    )


class _UnusedDecision:
    """
    Reached ONLY if the bridge re-decides.

    The bridge must never call this: the Telegram runtime owns the decision.
    Raising makes that structural rather than a matter of trust.
    """

    def decide(self, **kw):
        raise AssertionError(
            "the bridge re-decided; the Telegram runtime's decision is the "
            "single source and must be reused verbatim"
        )


def run(coro):
    return asyncio.run(coro)
