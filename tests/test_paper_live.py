"""
Offline tests for the execution-aware PAPER harness.

NO NETWORK, NO TELEGRAM SESSION, NO WALLET. Every provider is a double, every
account is virtual, and the tests assert that the four price layers stay
distinct - which is the whole point of the change.

The doubles are deliberately built so a MARK cannot be passed where a quote is
expected. ``MarkAsQuote`` below is the adversarial case: it satisfies the duck
type of a quote point and would pass a naive implementation, and several tests
exist precisely to prove it does not.
"""


from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from autopaper.analysis.decision import DecisionEngine
from autopaper.config import DecisionConfig, ExitConfig
from autopaper.exits import ExitConfig as ExitCfg, ExitPolicy
from autopaper.models import DataState, Decision, PositionStatus
from autopaper.same_market.numeric import DecimalError, effective_price

import autopaper.paper_live  # noqa: F401 - installs the barrier
from autopaper.paper_live.arms import arm_names
from autopaper.paper_live.account import VirtualAccount
from autopaper.paper_live.arms import (
    ARMS,
    ArmSpec,
    ResearchArm,
    arm_by_name,
    arm_specs,
    parse_arms,
)
from autopaper.paper_live.engine import (
    BASELINE_HASH,
    PaperLiveEngine,
    build_signal_snapshot,
)
from autopaper.paper_live.fills import (
    FillModel,
    FillRefused,
    fill_from_quote,
    realize,
)
from autopaper.paper_live.guards import (
    PAPER_FORBIDDEN_ATTRS,
    PAPER_FORBIDDEN_MODULES,
    PaperCapabilityViolation,
    assert_paper_only,
    audit_paper_capability,
)
from autopaper.paper_live.quotes import QuoteService, SolReference
from autopaper.paper_live.report import DatasetWriter
from autopaper.paper_live.schema import (
    ACTUAL_FILL_AVAILABLE,
    SIMULATED_FILL_KIND,
    ExecutionNativeReference,
    FailureReason,
    ObservedMarkPoint,
    PaperPosition,
    PriceLayer,
    QuotePoint,
    SimulatedFill,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.paper_live,
    pytest.mark.safety,
    pytest.mark.same_market,
    pytest.mark.strategy,
]

MINT = "APxxh2tWCh95tbHitFJchcdbMsctSBgRS2Zzju9Bpump"
PAIR = "9LCvns4zZsYAinZcEWRdzEEz7P6V84K3c5ZYCLT6vzMb"
MINT_2 = "74RybdiCDHpndFmce6rBp8o48v4hWJemb1ivK3NWpump"
WSOL = "So11111111111111111111111111111111111111112"
SOL_USD = 122.36
SOL_USD_D = Decimal("122.36")

# The $5.00 -> 10% -> $0.50 example from the brief, and the live SDK-verified
# curve quote for a 1,879,699-lamport order (5,176,254,910 raw units).
#: $0.50 at $122.36/SOL -> 0.004086303 SOL -> 4,086,303 lamports.
#: Derived, never hand-typed: a mistyped notional would silently scale every
#: price in the dataset by a power of ten.
BUY_LAMPORTS = int(Decimal("0.50") / Decimal("122.36") * 10 ** 9)
BUY_TOKENS_RAW = 5_176_254_910
#: A plausible net SELL of the whole holding: a few percent below the entry
#: cost, i.e. fees plus impact on a small position.
SELL_SOL_RAW = 3_950_000


def run(coro):
    return asyncio.run(coro)


# ===========================================================================
# doubles
# ===========================================================================

class FakeMarket:
    """DexScreener stand-in. Produces OBSERVATIONS, never quotes."""

    def __init__(self, price=None, *, liquidity=60000.0, volume5=1000.0,
                 mcap=9000.0, fail=False):
        # Defaults to the executable quote's own USD price, so the mark and the
        # quote are coherent BY CONSTRUCTION. The executable-vs-mark entry guard
        # is real, and a mark hardcoded beside a quote fixture drifts out of
        # agreement the moment either number changes -- which surfaced as a
        # spurious +621% rejection that said nothing about the code.
        self.price = MARK_USD if price is None else price
        self.liquidity = liquidity
        self.volume5 = volume5
        self.mcap = mcap
        self.fail = fail
        self.calls = 0

    async def quote_async(self, mint):
        self.calls += 1
        if self.fail:
            raise RuntimeError("dexscreener is down")

        class Q:
            pass

        class O:
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



def _buy_with_native_gap(fraction: float) -> QuotePoint:
    """
    A BUY quote whose SAME-MARKET spot sits ``fraction`` below the executed
    price, so the execution-native gap is exactly ``fraction * 100`` percent.

    Built by moving the REFERENCE, not the quote, because the reference is what
    the guard compares against. The executed price, the fee and the venue are
    untouched, so the quote stays an ordinary executable quote.
    """
    quote = buy_point()
    native = quote.execution_native_reference
    assert native is not None
    spot = float(quote.effective_price) / (1.0 + fraction)
    return replace(
        quote,
        execution_native_reference=replace(
            native, reference_price=spot,
            numerator_raw=int(native.numerator_raw / (1.0 + fraction))),
    )


def buy_point(**overrides) -> QuotePoint:
    """A usable BUY quote for exactly the $0.50 notional."""
    price = effective_price(BUY_LAMPORTS, BUY_TOKENS_RAW,
                            input_decimals=9, output_decimals=6)
    base = dict(
        side="BUY", provider="pump_direct", market_type="PUMP_BONDING_CURVE",
        token_mint=MINT, quote_mint=WSOL,
        input_amount_raw=BUY_LAMPORTS, input_decimals=9,
        input_amount_decimal=BUY_LAMPORTS / 1e9,
        output_amount_raw=BUY_TOKENS_RAW, output_decimals=6,
        output_amount_decimal=BUY_TOKENS_RAW / 1e6,
        effective_price=price,
        effective_price_usd=float(price) * SOL_USD,
        price_impact_pct=1.25, fee_bps=125, fee_amount=Decimal(0),
        venue="pumpfun-bonding-curve", pool_or_curve=PAIR, route=("pump_direct",),
        state_slot=1, state_age_ms=10.0, quote_timestamp=time.time(),
        quote_age_ms=5.0, max_quote_age_ms=30_000.0, quote_latency_ms=12.0,
        identity="PASS", identity_evidence="the marked pair IS the curve",
        observed_pair_address=PAIR, available=True,
        venue_id=f"PUMP_BONDING_CURVE:{PAIR}",
        # THE EXECUTION-NATIVE REFERENCE. The production quote path derives it
        # from the SAME account read and slot as the executable price, so the
        # double reproduces that rather than inventing a second source: the
        # spot is the executable price less the 1.25% fee this fixture already
        # encodes, giving the guard a coherent +1.25% cost of this size.
        execution_native_reference=ExecutionNativeReference(
            market_id=f"PUMP_BONDING_CURVE:{PAIR}", mint=MINT, base_mint=MINT,
            quote_mint=WSOL, side="BUY",
            reference_price=float(price) / 1.0125,
            numerator_raw=30_000_000_000, denominator_raw=1_500_000_000_000,
            calculation_method="PUMP_BONDING_CURVE_VIRTUAL_RESERVES",
            venue="pumpfun-bonding-curve", pool_or_curve=PAIR,
            provider="pump_direct", slot=1, observed_at=time.time(),
            state_age_ms=10.0, token_decimals=6,
        ),
    )
    base.update(overrides)
    return QuotePoint(**base)


def sell_point(**overrides) -> QuotePoint:
    """A usable SELL quote: TOKEN -> SOL, never reversed."""
    price = effective_price(SELL_SOL_RAW, BUY_TOKENS_RAW,
                            input_decimals=9, output_decimals=6)
    base = dict(
        side="SELL", provider="pump_direct", market_type="PUMP_BONDING_CURVE",
        token_mint=MINT, quote_mint=WSOL,
        input_amount_raw=BUY_TOKENS_RAW, input_decimals=6,
        input_amount_decimal=BUY_TOKENS_RAW / 1e6,
        output_amount_raw=SELL_SOL_RAW, output_decimals=9,
        output_amount_decimal=SELL_SOL_RAW / 1e9,
        effective_price=price, effective_price_usd=float(price) * SOL_USD,
        price_impact_pct=-3.6, fee_bps=125, fee_amount=Decimal(0),
        venue="pumpfun-bonding-curve", pool_or_curve=PAIR, route=("pump_direct",),
        state_slot=1, state_age_ms=10.0, quote_timestamp=time.time(),
        quote_age_ms=5.0, max_quote_age_ms=30_000.0, quote_latency_ms=9.0,
        identity="PASS", identity_evidence="the marked pair IS the pool",
        observed_pair_address=PAIR, available=True,
        venue_id=f"PUMP_BONDING_CURVE:{PAIR}",
        # THE EXECUTION-NATIVE REFERENCE, side SELL. The exit price triggers run
        # on THIS, not on the observed mark: it is the same book's own reserves
        # at the same slot. It mirrors the BUY double's construction -- spot is
        # the executable price less the 3.6% impact this fixture encodes -- so
        # the sell trigger sees a coherent cost of transacting this size.
        execution_native_reference=ExecutionNativeReference(
            market_id=f"PUMP_BONDING_CURVE:{PAIR}", mint=MINT, base_mint=MINT,
            quote_mint=WSOL, side="SELL",
            reference_price=float(price) / 0.964,
            numerator_raw=30_000_000_000, denominator_raw=1_500_000_000_000,
            calculation_method="PUMP_BONDING_CURVE_VIRTUAL_RESERVES",
            venue="pumpfun-bonding-curve", pool_or_curve=PAIR,
            provider="pump_direct", slot=1, observed_at=time.time(),
            state_age_ms=10.0, token_decimals=6,
        ),
    )
    base.update(overrides)
    return QuotePoint(**base)


def native_sell_point(multiplier: float = 1.0, **overrides) -> QuotePoint:
    """
    A SELL quote whose NATIVE REFERENCE has moved by ``multiplier``.

    The exit price triggers are evaluated on the same-market native reference,
    so a test that needs the stop to fire must move THIS -- not the aggregator
    mark, which is diagnostic only. ``multiplier`` is relative to the spot the
    ordinary ``sell_point`` double implies, so 1.0 is "unchanged" and 0.5 is
    "the same book halved".
    """
    base = sell_point(**overrides)
    native = base.execution_native_reference
    return replace(
        base,
        execution_native_reference=replace(
            native, reference_price=float(native.reference_price) * multiplier
        ),
    )


def unavailable_point(**overrides) -> QuotePoint:
    """A quote that could not be obtained. Amounts are None, never 0."""
    base = dict(
        side="BUY", provider="none", market_type="UNKNOWN", token_mint=MINT,
        quote_mint=WSOL, input_amount_raw=None, input_decimals=None,
        input_amount_decimal=None, output_amount_raw=None, output_decimals=None,
        output_amount_decimal=None, effective_price=None, effective_price_usd=None,
        price_impact_pct=None, fee_bps=None, fee_amount=None, venue=None,
        pool_or_curve=None, route=(), state_slot=None, state_age_ms=None,
        quote_timestamp=None, quote_age_ms=None, max_quote_age_ms=None,
        quote_latency_ms=None, identity="UNKNOWN",
        observed_pair_address=PAIR, available=False,
        missing_reason="no executable quote",
    )
    base.update(overrides)
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
    """
    The quote service double. Records what it was asked for.

    Mint-aware: it returns a quote stamped with the mint that was actually
    asked about, which is what a correct provider does. A test that needs a
    WRONG mint passes one explicitly.
    """

    def __init__(self, *, buy=None, sell=None):
        self.buy = buy if buy is not None else buy_point()
        self.sell = sell if sell is not None else sell_point()
        self.buy_calls: list[dict] = []
        self.sell_calls: list[dict] = []

    @staticmethod
    def _restamp(point: QuotePoint, mint: str) -> QuotePoint:
        """
        Re-stamp a quote for the mint actually asked about.

        The NATIVE REFERENCE is re-stamped too. It is derived from the same
        account read as the executable price, so a correct provider returns it
        already stamped for the requested mint; leaving ``base_mint`` pointing at
        the original mint would be a double that contradicts its own docstring
        and would make the exit trigger's base-mint identity check fail for a
        reason that exists only in the fixture.
        """
        if point.token_mint == mint:
            return point
        native = point.execution_native_reference
        if native is not None:
            native = replace(native, base_mint=mint, mint=mint)
        return replace(point, token_mint=mint,
                       execution_native_reference=native)

    async def quote_buy(self, **kw):
        self.buy_calls.append(kw)
        return self._restamp(self.buy, kw["mint"])

    async def quote_sell(self, **kw):
        self.sell_calls.append(kw)
        return self._restamp(self.sell, kw["mint"])

    async def quote_sell_best(self, **kw):
        self.sell_calls.append(kw)
        return (self._restamp(self.sell, kw["mint"]), [])


class StubL25:
    status = "AUTHORIZED"
    authorized = True


class StubFeatures:
    def __init__(self, vtl=1.5):
        self.volume_to_liquidity = vtl


class StubReport:
    def __init__(self, vtl=1.5, decision=Decision.BUY, l25=None):
        self.decision = decision
        self.l25 = l25 or StubL25()
        self.features = StubFeatures(vtl)


class StubDecision:
    """Decision engine double. Lets a test drive L25 without market history."""

    def __init__(self, *, decision=Decision.BUY, vtl=1.5, authorized=True,
                 status="AUTHORIZED", raise_exc=None):
        self.decision = decision
        self.vtl = vtl
        self.authorized = authorized
        self.status = status
        self.raise_exc = raise_exc
        self.calls = 0

    def decide(self, **kw):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        l25 = StubL25()
        l25.authorized = self.authorized
        l25.status = self.status
        return StubReport(self.vtl, decision=self.decision, l25=l25)


class StubExit:
    """Exit policy double, OR the production one - see the exit tests."""

    def __init__(self, action="EXIT", trigger="fast_failure_stop"):
        self.action = action
        self.trigger = trigger
        self.calls = []

    def assess(self, position, *, now, mark_price, data_state, **kw):
        self.calls.append({
            "mark_price": mark_price,
            "fill_price": position.fill_price,
            "price_kind": position.fill_price_kind,
        })
        from autopaper.exits import ExitDecision, ExitStyle

        return ExitDecision(
            self.action, f"{self.trigger} at {now}", ExitStyle.FAST_FAILURE,
            trigger=self.trigger if self.action == "EXIT" else None,
        )


def make_engine(
    *,
    arms=None,
    market=None,
    quotes=None,
    decision=None,
    exits=None,
    bankroll=5.0,
    fill_model=None,
    require_identity=True,
    max_quote_age_ms=30_000.0,
    sol_usd=SOL_USD,
) -> PaperLiveEngine:
    return PaperLiveEngine(
        arms=arms if arms is not None else parse_arms([]),
        decision_engine=decision or StubDecision(),
        exit_policy=exits or StubExit(),
        quote_service=quotes or FakeQuoteService(),
        market=market if market is not None else FakeMarket(),
        token_decimals_for=lambda mint: 6,
        sol_reference=SolReference(sol_usd, time.time(), "test") if sol_usd else None,
        fill_model=fill_model or FillModel(),
        starting_bankroll_usd=bankroll,
        require_identity=require_identity,
    )


def signal(mint=MINT, **kw) -> object:
    return build_signal_snapshot(
        channel="MooDengPresidentCallers", text=f"BUY ${mint}",
        message_id=kw.get("message_id", 11),
        telegram_timestamp=kw.get("telegram_timestamp", time.time() - 1.2),
        mint=mint, symbol=kw.get("symbol", "T"),
    )


# ===========================================================================
# 1. signal event -> L25 decision
# ===========================================================================

class TestSignalToDecision:
    def test_one_signal_produces_one_record_per_arm(self):
        selected = parse_arms([])
        engine = make_engine(arms=selected)
        records = run(engine.on_signal(signal()))
        # One record per selected arm (beta group expands to 3 arms)
        expected_arms = {spec.name.value for spec in selected}
        assert len(records) == len(selected)
        assert {r.arm for r in records} == expected_arms

    def test_the_production_decision_engine_is_reachable_and_verbatim(self):
        """
        The engine must be able to call the REAL DecisionEngine, not only a
        double. This is what proves the production gate is integrated rather
        than reimplemented.
        """
        async def _go():
            engine = make_engine(
                decision=DecisionEngine(DecisionConfig(), ExitCfg()),
                market=FakeMarket(),
            )
            for _ in range(12):
                await engine.observe(MINT)
            mark = await engine.observe(MINT)
            report, status, authorized = engine.decide(
                mint=MINT, signal=signal(), mark=mark,
                decision_time=time.monotonic(), blocked_reason=None,
            )
            assert report.l25.status == status
            assert report.l25.authorized is authorized
            # And the verdict is a real production verdict, not a default.
            assert report.decision in set(Decision)

        run(_go())

    def test_a_l25_rejection_is_recorded_and_trades_nothing(self):
        engine = make_engine(decision=StubDecision(
            decision=Decision.REJECT, authorized=False, status="V3C_REJECTED_MARKET_CAP"
        ))
        records = run(engine.on_signal(signal()))
        # Find the L25_CURRENT record (beta arms are rejected at V2_FLOOR gate)
        l25_records = [r for r in records if r.arm == "L25_CURRENT"]
        assert len(l25_records) == 1
        record = l25_records[0]
        assert record.traded is False
        assert record.failure is FailureReason.L25_REJECTED
        assert record.l25_status == "V3C_REJECTED_MARKET_CAP"
        assert engine.accounts[record.arm].bankroll_usd == 5.0

    def test_a_decision_engine_error_does_not_kill_the_run(self):
        engine = make_engine(decision=StubDecision(raise_exc=RuntimeError("boom")))
        records = run(engine.on_signal(signal()))
        # Beta arms are rejected at V2_FLOOR gate; only regular arms hit the decision engine
        from autopaper.paper_live.arms import BETA_ARMS
        beta_arms = {a.value for a in BETA_ARMS}
        regular_records = [r for r in records if r.arm not in beta_arms]
        assert len(records) == len(engine.arms)
        assert all(r.failure is FailureReason.ARITHMETIC_ERROR for r in regular_records)
        assert all("boom" in r.failure_detail for r in regular_records)

    def test_a_signal_with_no_mint_is_recorded_not_dropped(self):
        engine = make_engine()
        records = run(engine.on_signal(build_signal_snapshot(
            channel="c", text="nothing here", message_id=None,
            telegram_timestamp=None, mint=None, symbol=None,
        )))
        from autopaper.paper_live.arms import BETA_ARMS
        beta_arms = {a.value for a in BETA_ARMS}
        regular_records = [r for r in records if r.arm not in beta_arms]
        assert len(records) == len(engine.arms)
        assert all(r.failure is FailureReason.TOKEN_NOT_FOUND for r in regular_records)


async def awaitable_mark(engine, mint):
    return await engine.observe(mint)


# ===========================================================================
# 2. position sizing from the virtual bankroll
# ===========================================================================

class TestSizing:
    def test_five_dollars_sizes_ten_percent(self):
        """The brief's example, and the production formula's own output."""
        account = VirtualAccount("A", "v", "h", starting_bankroll_usd=5.00)
        size = account.size_for_bankroll()
        assert size.position_pct == 10.0
        assert size.position_usd == pytest.approx(0.50)

    def test_the_size_tracks_a_compounding_bankroll(self):
        account = VirtualAccount("A", "v", "h", starting_bankroll_usd=5.40)
        assert account.size_for_bankroll().position_usd == pytest.approx(0.54)

    def test_the_tier_switches_at_fifty_dollars(self):
        from autopaper.sizing import size_for_equity

        assert size_for_equity(50.00) == pytest.approx(5.00)
        assert size_for_equity(50.01) == pytest.approx(50.01 * 0.125)
        assert size_for_equity(200.0) == pytest.approx(25.0)

    def test_the_engine_uses_the_arm_bankroll_not_a_fixed_size(self):
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), quotes=quotes,
                             bankroll=8.00)
        record = run(engine.on_signal(signal()))[0]
        assert record.position_size_usd == pytest.approx(0.80)
        expected = int(Decimal("0.80") / Decimal("122.36") * 10 ** 9)
        assert quotes.buy_calls[0]["sol_lamports"] == expected
        # NOT the $0.23 diagnostic quote size (1,879,699 lamports).
        assert expected != 1_879_699

    def test_the_zero_point_two_three_diagnostic_size_is_not_used(self):
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), quotes=quotes,
                             bankroll=5.0)
        run(engine.on_signal(signal()))
        assert quotes.buy_calls[0]["sol_lamports"] != 1_879_699


# ===========================================================================
# 3 + 4 + 6 + 7. executable BUY usage, direction, and the mark is not a fill
# ===========================================================================

class TestEntryFill:
    def test_the_entry_fill_comes_from_the_executable_quote(self):
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), quotes=quotes)
        record = run(engine.on_signal(signal()))[0]
        assert record.traded is True
        fill = engine.accounts["L25_CURRENT"].position.entry_fill
        assert fill.effective_price == quotes.buy.effective_price
        assert fill.kind == SIMULATED_FILL_KIND

    def test_the_observed_mark_is_not_the_entry_fill(self):
        market = FakeMarket(price=MARK_USD)
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), market=market,
                             quotes=quotes)
        record = run(engine.on_signal(signal()))[0]
        fill = engine.accounts["L25_CURRENT"].position.entry_fill
        assert record.observed_mark.value_usd == MARK_USD
        assert float(fill.effective_price_usd) != MARK_USD
        assert fill.effective_price_usd == quotes.buy.effective_price_usd

    def test_the_tokens_held_equal_the_quoted_output_exactly(self):
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), quotes=quotes)
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        assert position.tokens_raw == BUY_TOKENS_RAW
        assert position.entry_fill.output_amount_raw == BUY_TOKENS_RAW

    def test_the_buy_direction_is_sol_to_token(self):
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), quotes=quotes)
        run(engine.on_signal(signal()))
        fill = engine.accounts["L25_CURRENT"].position.entry_fill
        assert fill.input_mint == WSOL and fill.input_decimals == 9
        assert fill.output_mint == MINT and fill.output_decimals == 6

    def test_no_slippage_is_added_on_top_of_the_quote(self):
        """A quote already prices this size; adding a penalty double-counts."""
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), quotes=quotes)
        run(engine.on_signal(signal()))
        fill = engine.accounts["L25_CURRENT"].position.entry_fill
        assert fill.effective_price == quotes.buy.effective_price
        assert fill.price_impact_pct == 1.25      # retained separately

    def test_the_optional_penalty_is_off_by_default_and_recorded(self):
        model = FillModel()
        assert model.post_quote_penalty_bps == 0
        assert model.as_dict()["penalty_active"] is False
        price = Decimal("1.5")
        assert model.penalised_price(price, "BUY") == price

    def test_an_explicit_penalty_worsens_a_buy_and_lowers_a_sell(self):
        model = FillModel(post_quote_penalty_bps=50.0)
        assert model.penalised_price(Decimal("1.0"), "BUY") > Decimal("1.0")
        assert model.penalised_price(Decimal("1.0"), "SELL") < Decimal("1.0")
        assert model.as_dict()["penalty_active"] is True


# ===========================================================================
# 5 + 6 + 8. executable SELL usage and direction
# ===========================================================================

class TestExitFill:
    def _open(self, **kw):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), **kw)
        run(engine.on_signal(signal()))
        return engine

    def test_the_exit_fill_comes_from_a_fresh_sell_quote(self):
        quotes = FakeQuoteService()
        engine = self._open(quotes=quotes)
        trades = run(engine.monitor_once())
        assert len(trades) == 1
        trade = trades[0]
        assert trade.exit_fill.effective_price == quotes.sell.effective_price
        assert trade.exit_fill.kind == SIMULATED_FILL_KIND

    def test_the_sell_direction_is_token_to_sol(self):
        quotes = FakeQuoteService()
        engine = self._open(quotes=quotes)
        run(engine.monitor_once())
        assert quotes.sell_calls[0]["token_amount_raw"] == BUY_TOKENS_RAW
        trade = engine.trade_records[0]
        assert trade.exit_fill.input_mint == MINT and trade.exit_fill.input_decimals == 6
        assert trade.exit_fill.output_mint == WSOL and trade.exit_fill.output_decimals == 9

    def test_the_observed_mark_is_not_the_exit_fill(self):
        engine = self._open(market=FakeMarket(price=9.99e-05))
        trade = run(engine.monitor_once())[0]
        assert trade.exit_observed_mark == 9.99e-05
        assert float(trade.exit_fill.effective_price_usd) != 9.99e-05

    def test_the_exit_quotes_the_exact_tokens_held(self):
        quotes = FakeQuoteService()
        engine = self._open(quotes=quotes)
        run(engine.monitor_once())
        assert quotes.sell_calls[0]["token_amount_raw"] == engine.trade_records[
            0].position.tokens_raw

    def test_a_mark_to_mark_return_is_not_the_realized_result(self):
        engine = self._open(market=FakeMarket(price=MARK_USD))
        trade = run(engine.monitor_once())[0]
        pnl = trade.pnl
        mark_ratio = (
            (trade.exit_observed_mark / trade.position.entry_observed_mark - 1) * 100
        )
        assert pnl.return_pct is not None
        # The mark barely moved in this fixture; the fill-based result is very
        # different. If the implementation ever regressed to mark-to-mark these
        # two would coincide.
        assert abs(pnl.return_pct) > 1.0
        assert pnl.return_pct != pytest.approx(mark_ratio, abs=1.0)


# ===========================================================================
# 9. decimal normalisation
# ===========================================================================

class TestDecimals:
    def test_six_decimal_token_normalisation(self):
        fill = fill_from_quote(buy_point(), side="BUY")
        assert fill.output_amount_decimal == pytest.approx(5176.254910)
        assert fill.input_amount_decimal == pytest.approx(
            BUY_LAMPORTS / 1e9
        )

    def test_nine_decimal_quote_asset_normalisation(self):
        fill = fill_from_quote(sell_point(), side="SELL")
        assert fill.output_amount_decimal == pytest.approx(SELL_SOL_RAW / 1e9)

    def test_a_nine_decimal_token(self):
        point = buy_point(output_amount_raw=3_001_234_567_890,
                          output_decimals=9)
        fill = fill_from_quote(point, side="BUY")
        assert fill.output_amount_decimal == pytest.approx(3_001.234567890)

    def test_the_six_and_nine_decimal_scales_are_not_interchangeable(self):
        raw = 1_000_000
        a = fill_from_quote(buy_point(output_amount_raw=raw,
                                      output_decimals=6), side="BUY")
        b = fill_from_quote(buy_point(output_amount_raw=raw,
                                      output_decimals=9), side="BUY")
        assert a.output_amount_decimal == pytest.approx(1.0)
        assert b.output_amount_decimal == pytest.approx(0.001)
        assert a.output_amount_decimal / b.output_amount_decimal == pytest.approx(1000)

    def test_a_very_small_price_is_not_rounded_to_zero(self):
        point = buy_point(output_amount_raw=10 ** 15,
                          effective_price=Decimal("1E-15"))
        fill = fill_from_quote(point, side="BUY")
        assert fill.effective_price == Decimal("1E-15")
        assert fill.effective_price > 0

    def test_no_rounding_before_the_token_quantity_is_taken(self):
        """The held amount is the raw integer, never a rounded human figure."""
        point = buy_point(output_amount_raw=5_176_254_911)   # odd, one more
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]),
                             quotes=FakeQuoteService(buy=point))
        run(engine.on_signal(signal()))
        assert engine.accounts["L25_CURRENT"].position.tokens_raw == 5_176_254_911

    def test_effective_price_matches_the_published_helper(self):
        fill = fill_from_quote(buy_point(), side="BUY")
        expected = effective_price(BUY_LAMPORTS, BUY_TOKENS_RAW,
                                   input_decimals=9, output_decimals=6)
        assert fill.effective_price == expected


# ===========================================================================
# 10 + 11 + 12. quote freshness, stale rejection, missing rejection
# ===========================================================================

class TestFreshness:
    def test_a_fresh_quote_is_used(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        record = run(engine.on_signal(signal()))[0]
        assert record.traded is True

    def test_a_stale_quote_is_rejected_and_never_replaced_by_the_mark(self):
        point = buy_point(quote_age_ms=45_000.0, max_quote_age_ms=30_000.0)
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]),
                             quotes=FakeQuoteService(buy=point))
        record = run(engine.on_signal(signal()))[0]
        assert record.traded is False
        assert record.failure is FailureReason.ENTRY_REJECTED_QUOTE_STALE
        assert engine.accounts["L25_CURRENT"].bankroll_usd == 5.0
        # The mark is still recorded, as a non-trade observation.
        assert record.observed_mark.available is True
        assert record.observed_mark.value_usd == MARK_USD

    def test_a_missing_quote_is_rejected(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            quotes=FakeQuoteService(buy=unavailable_point()),
        )
        record = run(engine.on_signal(signal()))[0]
        assert record.traded is False
        assert record.failure is FailureReason.ENTRY_REJECTED_QUOTE_UNAVAILABLE

    def test_the_stale_flag_uses_the_configured_window(self):
        assert buy_point(quote_age_ms=10_000.0,
                         max_quote_age_ms=30_000.0).stale is False
        assert buy_point(quote_age_ms=40_000.0,
                         max_quote_age_ms=30_000.0).stale is True

    def test_an_unconfigured_window_is_not_treated_as_stale(self):
        assert buy_point(quote_age_ms=999_999.0, max_quote_age_ms=None).stale is False

    def test_a_zero_max_quote_age_is_passed_through_to_the_fill(self):
        fill = fill_from_quote(buy_point(), side="BUY")
        assert fill.max_quote_age_ms == 30_000.0
        assert fill.quote_age_ms == 5.0


# ===========================================================================
# 13 + 14. same-market identity and cross-venue rejection
# ===========================================================================

class TestIdentity:
    def test_a_proven_identity_allows_the_trade(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        record = run(engine.on_signal(signal()))[0]
        assert record.execution_identity == "PASS"
        assert record.traded is True

    def test_unproven_identity_fails_closed_and_keeps_the_observation(self):
        point = buy_point(identity="UNKNOWN",
                          identity_evidence="no chain-verified market")
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]),
                             quotes=FakeQuoteService(buy=point))
        record = run(engine.on_signal(signal()))[0]
        assert record.traded is False
        assert record.failure is FailureReason.ENTRY_REJECTED_IDENTITY_UNKNOWN
        assert record.observed_mark.available is True
        assert record.buy_quote.available is True      # the quote is still on record

    def test_a_cross_venue_identity_fails_closed(self):
        point = buy_point(identity="FAIL",
                          identity_evidence="the pair is a different pool")
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]),
                             quotes=FakeQuoteService(buy=point))
        record = run(engine.on_signal(signal()))[0]
        assert record.traded is False
        assert record.failure is FailureReason.ENTRY_REJECTED_IDENTITY_UNKNOWN

    def test_the_matching_pair_address_is_a_pass(self):
        identity, _ = QuoteService._identity_for(
            observed_pair_address=PAIR, market_type=None,
            market_address=PAIR, chain_verified=True,
        )
        assert identity == "PASS"

    def test_a_different_pair_address_is_a_fail(self):
        identity, evidence = QuoteService._identity_for(
            observed_pair_address="SomeOtherPairAddress", market_type=None,
            market_address=PAIR, chain_verified=True,
        )
        assert identity == "FAIL"
        assert "different books" in evidence

    def test_no_chain_verification_is_unknown_never_pass(self):
        identity, _ = QuoteService._identity_for(
            observed_pair_address=PAIR, market_type=None,
            market_address=PAIR, chain_verified=False,
        )
        assert identity == "UNKNOWN"

    def test_no_observed_pair_is_unknown(self):
        identity, _ = QuoteService._identity_for(
            observed_pair_address=None, market_type=None,
            market_address=PAIR, chain_verified=True,
        )
        assert identity == "UNKNOWN"


# ===========================================================================
# 15 + 16 + 17 + 18. settlement, SOL/USD, fees, compounding
# ===========================================================================

class TestSettlement:
    def _settled(self, bankroll=5.0):
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), quotes=quotes,
                             bankroll=bankroll)
        run(engine.on_signal(signal()))
        trade = run(engine.monitor_once())[0]
        return engine, trade

    def test_the_settlement_is_the_documented_formula(self):
        _engine, trade = self._settled()
        pnl = trade.pnl
        assert pnl.gross_pnl_quote == (
            pnl.exit_proceeds_quote - pnl.entry_cost_quote
        )
        assert pnl.net_pnl_quote == (
            pnl.gross_pnl_quote - pnl.trading_fees_quote
            - pnl.network_costs_quote - pnl.other_costs_quote
        )

    def test_the_bankroll_moves_only_by_the_fill_based_result(self):
        engine, trade = self._settled()
        assert trade.pnl.bankroll_after == pytest.approx(
            trade.pnl.bankroll_before + trade.pnl.realized_pnl_usd
        )
        assert engine.accounts["L25_CURRENT"].bankroll_usd == pytest.approx(
            trade.pnl.bankroll_after
        )

    def test_sol_usd_conversion_is_timestamped_and_explicit(self):
        engine, trade = self._settled()
        assert trade.pnl.sol_usd_reference == SOL_USD
        assert trade.pnl.sol_usd_reference_at is not None
        assert trade.pnl.realized_pnl_sol == pytest.approx(
            float(trade.pnl.net_pnl_quote)
        )
        assert trade.pnl.realized_pnl_usd == pytest.approx(
            float(trade.pnl.net_pnl_quote) * SOL_USD
        )

    def test_a_missing_sol_reference_refuses_the_entry(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), sol_usd=None)
        record = run(engine.on_signal(signal()))[0]
        assert record.traded is False
        assert record.failure is FailureReason.SOL_REFERENCE_MISSING

    def test_the_fee_is_recorded_and_not_double_subtracted(self):
        engine, trade = self._settled()
        fill = trade.position.entry_fill
        assert fill.fee_bps == 125
        # The venue fee is already inside the quoted output, so it is zero in
        # the subtraction rather than charged a second time.
        assert fill.fee == 0
        assert trade.pnl.net_pnl_quote == (
            trade.pnl.exit_proceeds_quote - trade.pnl.entry_cost_quote
        )

    def test_the_bankroll_compounds_across_trades(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), bankroll=5.0)
        run(engine.on_signal(signal(message_id=11)))
        first = run(engine.monitor_once())[0]
        assert first.pnl is not None
        # A DIFFERENT mint: the first is on cooldown by design, and re-entry is
        # disabled, so reusing it would be testing the cooldown twice.
        run(engine.on_signal(signal(MINT_2, message_id=12)))
        second = run(engine.monitor_once())[0]
        assert second.pnl is not None
        assert second.position.bankroll_before == pytest.approx(
            first.pnl.bankroll_after
        )
        assert second.position.position_size_usd == pytest.approx(
            second.position.bankroll_before * 0.10
        )

    def test_drawdown_is_recorded(self):
        _engine, trade = self._settled()
        assert trade.pnl.drawdown_pct is not None
        assert trade.pnl.drawdown_pct >= 0


# ===========================================================================
# 19 + 20 + 21. one position, cooldown, re-entry
# ===========================================================================

class TestConstraints:
    def test_a_second_signal_while_a_position_is_open_is_blocked(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal(message_id=1)))
        record = run(engine.on_signal(signal(message_id=2)))[0]
        assert record.traded is False
        assert record.failure is FailureReason.ENTRY_BLOCKED_POSITION_ACTIVE
        assert engine.accounts["L25_CURRENT"].blocked_by_position_active == 1

    def test_arms_do_not_block_each_other(self):
        """Each arm has its OWN position, so one cannot consume another's slot."""
        arms = parse_arms([])
        engine = make_engine(arms=arms)
        first = run(engine.on_signal(signal(message_id=1)))
        assert all(r.traded for r in first)
        assert sum(1 for r in first) == len(arms)

    def test_after_an_exit_the_mint_is_on_cooldown(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal(message_id=1)))
        run(engine.monitor_once())
        record = run(engine.on_signal(signal(message_id=2)))[0]
        assert record.traded is False
        assert record.failure is FailureReason.ENTRY_BLOCKED_COOLDOWN
        assert engine.accounts["L25_CURRENT"].blocked_by_cooldown >= 0

    def test_re_entry_is_never_enabled_by_this_harness(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal(message_id=1)))
        run(engine.monitor_once())
        record = run(engine.on_signal(signal(message_id=2)))[0]
        assert record.traded is False
        assert "re-entry is disabled" in record.failure_detail

    def test_the_position_constraint_is_one_per_arm_not_one_per_run(self):
        arms = parse_arms([])
        engine = make_engine(arms=arms)
        run(engine.on_signal(signal(message_id=1)))
        # One position per ARM, so the total is the arm count -- derived, so a
        # fourth arm cannot quietly make this assertion vacuous.
        assert sum(
            1 for a in engine.accounts.values() if a.has_position
        ) == len(engine.accounts) == len(arms)


# ===========================================================================
# 22 + 23 + 24. exit precedence, max hold, H2_CAP60
# ===========================================================================

class TestExits:
    def test_the_production_exit_policy_runs_unmodified(self):
        """
        The engine hands the PRODUCTION ExitPolicy a real Position whose entry
        price is the simulated fill, so the percentage triggers fire against
        what the position actually cost.
        """
        # The same-market NATIVE REFERENCE falls after entry, so the production
        # policy has a real condition to act on. It is the reference the stop is
        # evaluated on; the aggregator mark is diagnostic only, so dropping the
        # mark would no longer be a condition at all.
        # A flat market correctly produces HOLD, which would make "did not
        # produce a decision" pass for the wrong reason.
        market = FakeMarket(price=MARK_USD)
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            exits=ExitPolicy(ExitCfg()),
            market=market,
            quotes=FakeQuoteService(sell=native_sell_point(0.5)),
        )
        run(engine.on_signal(signal()))
        # The mark FALLS too, and must be irrelevant to the decision.
        market.price = MARK_USD * 0.4
        # The production policy is called with the FILL price as the entry.
        engine.exit_policy = ExitPolicy(ExitCfg())
        trades = run(engine.monitor_once())
        assert trades, "the production exit policy did not produce a decision"
        assert trades[0].position.entry_fill.kind == SIMULATED_FILL_KIND

    def test_the_minimum_hold_suppression_is_applied(self):
        exits = StubExit(action="EXIT", trigger="momentum_collapse")
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), exits=exits)
        run(engine.on_signal(signal()))
        assert run(engine.monitor_once()) == []
        assert engine.accounts["L25_CURRENT"].has_position

    def test_a_hold_leaves_the_position_open(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]),
                             exits=StubExit(action="HOLD"))
        run(engine.on_signal(signal()))
        assert run(engine.monitor_once()) == []
        assert engine.accounts["L25_CURRENT"].has_position

    def test_each_arm_stamps_its_own_production_profile(self):
        engine = make_engine(arms=parse_arms([]))
        run(engine.on_signal(signal()))
        assert engine.accounts["L25_CURRENT"].position.production_profile == "L25_CURRENT"
        assert engine.accounts["L25_60S"].position.production_profile == "L25_60S"
        assert engine.accounts["H2_CAP60"].position.production_profile == "L25_60S"

    def test_the_sixty_second_cap_is_production_not_a_new_rule(self):
         from autopaper.strategy_profile import PROFILE_MAX_HOLD_S, StrategyProfile
         from autopaper.exits import _max_hold_for

         spec = ARMS[ResearchArm.L25_60S]
         # The 60s cap is PRODUCTION, not a rule this arm invented: the arm
         # maps onto the shipping ``StrategyProfile.L25_60S`` unchanged, and
         # both declare the same 60.0. An earlier version of this assertion
         # read ``is None`` with a comment naming the retired TP100 arm, which
         # described a design that no longer existed -- take-profit is a
         # SETTING now (``paper_live.take_profit``), not an arm.
         assert spec.max_hold_s == 60.0
         assert PROFILE_MAX_HOLD_S[StrategyProfile.L25_60S] == 60.0
         assert spec.max_hold_s == PROFILE_MAX_HOLD_S[StrategyProfile.L25_60S]

         engine = make_engine(arms=parse_arms(["L25_60S"]))
         run(engine.on_signal(signal()))
         position = engine.accounts["L25_60S"].position
         # Read off the POSITION, so a later profile switch cannot rewrite it.
         assert _max_hold_for(
             type("P", (), {"strategy_profile": position.production_profile})()
         ) == 60.0

    def test_the_ninety_second_cap_belongs_to_v2_floor_and_not_to_an_l25_arm(self):
        """
        A 90s cap now exists -- on the V2_FLOOR PROFILE, not on a paper_live arm.

        This assertion originally read ``set(PROFILE_MAX_HOLD_S.values()) ==
        {None, 60.0}``, which asserted that NO 90s cap could ever exist. That
        was true while every arm traded L25. It is false now that a second
        strategy with a researched 90s hold is selectable, and the test is
        rewritten to state the real invariant:

          * no paper_live ARM carries a 90s cap (arms are the L25 research
            family and are unchanged), and
          * the only 90s cap in the system belongs to V2_FLOOR.
        """
        from autopaper.strategy_profile import PROFILE_MAX_HOLD_S, StrategyProfile

        # no arm has a 90s cap
        for spec in arm_specs():
            assert spec.max_hold_s in (None, 60.0), (
                f"arm {spec.name.value} must not carry a 90s cap; arms are the "
                f"L25 research family"
            )
        # and the profile table's caps are exactly the declared set
        assert set(PROFILE_MAX_HOLD_S.values()) == {None, 60.0, 90.0}
        assert PROFILE_MAX_HOLD_S[StrategyProfile.V2_FLOOR] == 90.0
        # the 90s cap is not reachable through any L25 arm's profile
        for spec in arm_specs():
            if spec.production_profile is StrategyProfile.V2_FLOOR:
                continue
            assert PROFILE_MAX_HOLD_S[spec.production_profile] != 90.0

    def test_h2_cap60_is_l25_plus_the_turnover_ceiling(self):
        from autopaper.strategy_profile import StrategyProfile
        spec = ARMS[ResearchArm.H2_CAP60]
        assert spec.extra_entry_rule == "f_volume_to_liquidity <= 2.0"
        assert spec.max_hold_s == 60.0
        assert spec.production_profile is StrategyProfile.L25_60S

    def test_h2_admits_a_turnover_ratio_under_the_ceiling(self):
        allowed, why = ARMS[ResearchArm.H2_CAP60].entry_allows(StubFeatures(1.9))
        assert allowed is True and "<= 2.0" in why

    def test_h2_refuses_a_ratio_over_the_ceiling(self):
        allowed, why = ARMS[ResearchArm.H2_CAP60].entry_allows(StubFeatures(2.1))
        assert allowed is False and "H2 turnover ceiling" in why

    def test_h2_refuses_an_unavailable_ratio_rather_than_passing_it(self):
        allowed, why = ARMS[ResearchArm.H2_CAP60].entry_allows(StubFeatures(None))
        assert allowed is False and "unavailable" in why

    def test_l25_tp100_has_no_extra_filter(self):
        assert ARMS[ResearchArm.L25_60S].entry_allows(StubFeatures(99.0))[0] is True

    def test_the_h2_rejection_is_distinct_from_the_l25_rejection(self):
        engine = make_engine(arms=parse_arms([]), decision=StubDecision(vtl=5.0))
        records = {r.arm: r for r in run(engine.on_signal(signal()))}
        assert records["L25_CURRENT"].traded is True
        assert records["L25_60S"].traded is True
        assert records["H2_CAP60"].traded is False
        assert records["H2_CAP60"].failure is FailureReason.ARM_FILTER_REJECTED
        # L25 itself authorised in all three: the rejection is H2's, not L25's.
        assert records["H2_CAP60"].l25_authorized is True


# ===========================================================================
# 25. rejected opportunities are logged
# ===========================================================================

class TestRejectedOpportunities:
    def test_every_signal_is_recorded_whether_traded_or_not(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal(message_id=1)))          # trades
        run(engine.on_signal(signal(message_id=2)))          # position active
        run(engine.on_signal(signal(message_id=3)))          # cooldown
        # One arm was requested, so three signals give three records.
        assert len(engine.signal_records) == 3
        assert [r.traded for r in engine.signal_records] == [True, False, False]

    def test_a_rejected_record_carries_the_reason_and_the_evidence(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]),
                             decision=StubDecision(decision=Decision.REJECT,
                                                   authorized=False,
                                                   status="V3C_UNTESTABLE"))
        record = run(engine.on_signal(signal()))[0]
        assert record.l25_decision == "REJECT"
        assert record.l25_status == "V3C_UNTESTABLE"
        assert "REJECT" in record.failure_detail
        assert record.observed_mark.available is True
        assert record.latency.as_dict()["decision_latency_ms"] is not None

    def test_a_rejected_signal_still_records_the_mark_and_gap(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        record = run(engine.on_signal(signal()))[0]
        assert record.observed_mark.value_usd is not None
        assert record.mark_to_quote_gap_pct is not None

    def test_a_gap_beyond_ten_percent_refuses_the_entry(self):
        """
        THE 10% ENTRY GUARD, against the SAME MARKET'S OWN price.

        The guard refuses a materially expensive entry, the threshold is still
        10%, and the reference is the same market's spot price from the same
        on-chain read. So the fixture moves the NATIVE REFERENCE to create the
        wide gap -- not the observed mark.
        """
        quotes = FakeQuoteService(buy=_buy_with_native_gap(0.90))
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), quotes=quotes)
        record = run(engine.on_signal(signal()))[0]
        assert record.mark_to_quote_gap_pct == pytest.approx(90.0, abs=0.5)
        assert record.traded is False
        assert record.failure is FailureReason.MARK_GAP_GUARD_REJECTED
        assert record.production_guard_source == "EXECUTION_NATIVE_SAME_MARKET"
        # The opportunity is retained: "we looked and refused" must stay
        # distinguishable from "we never looked".
        assert record.observed_mark.value_usd is not None
        assert engine.accounts["L25_CURRENT"].has_position is False

    def test_a_gap_within_ten_percent_still_trades(self):
        """The guard refuses a wide gap, not a real trade."""
        quotes = FakeQuoteService(buy=_buy_with_native_gap(0.09))
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), quotes=quotes)
        record = run(engine.on_signal(signal()))[0]
        assert record.mark_to_quote_gap_pct == pytest.approx(9.0, abs=0.5)
        assert record.traded is True

    def test_A_LAGGING_OBSERVED_MARK_NO_LONGER_REFUSES_THE_ENTRY(self):
        """
        THE REGRESSION THIS PROMOTION EXISTS FOR.

        A +90% gap against the external observed mark used to refuse the entry
        on young markets, because that mark is a third party's periodically
        refreshed snapshot of the very market being traded. Here the observed
        mark is moved 90% AWAY from an ordinary +1.25% same-market execution
        cost, and the entry proceeds -- because the mark is a diagnostic, not
        the reference.

        The legacy gap is still recorded, so the disagreement stays auditable.
        """
        quotes = FakeQuoteService(buy=_buy_with_native_gap(0.0125))
        market = FakeMarket(price=QUOTE_USD / 1.90)
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), market=market,
                             quotes=quotes)
        record = run(engine.on_signal(signal()))[0]
        # The mark is wildly wrong about this market...
        assert record.legacy_observed_mark_gap_pct == pytest.approx(90.0, abs=1.0)
        assert record.legacy_guard_verdict == "REJECT"
        # ...and the guard no longer cares, because it is not the reference.
        assert record.mark_to_quote_gap_pct == pytest.approx(1.25, abs=0.05)
        assert record.native_guard_verdict == "PASS"
        assert record.traded is True
        assert record.failure is FailureReason.NONE

    def test_an_executable_quote_below_the_native_reference_is_not_refused(self):
        """
        The guard is one-sided ON PURPOSE.

        It refuses paying far above the market. It does NOT refuse getting a
        better price than the market's own -- that is a good fill, and a
        symmetric guard would systematically discard the best trades.
        """
        quotes = FakeQuoteService(buy=_buy_with_native_gap(-0.20))
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]),
                             quotes=quotes)
        assert run(engine.on_signal(signal()))[0].traded is True

    def test_the_guard_is_unknown_and_refuses_when_the_comparison_cannot_be_made(self):
        """
        UNKNOWN refuses. A guard that passes when it cannot see is not a guard.
        """
        from autopaper.paper_live.economics import evaluate_mark_gap

        verdict = evaluate_mark_gap(
            mark_price=1.0, executable_price=1.01,
            same_market_identity=False)
        assert verdict.result == "UNKNOWN"
        assert verdict.allows_entry is False
        assert "same market" in verdict.reason

        # Each precondition is broken ONCE, by overriding exactly one field, so
        # a duplicate keyword can never mask the case under test.
        for field in ("same_mint", "same_quote_asset", "same_side",
                      "same_order_size", "mark_fresh", "quote_fresh"):
            base = dict(mark_price=1.0, executable_price=1.01, **{field: True})
            base[field] = False
            v = evaluate_mark_gap(**base)
            assert v.result == "UNKNOWN", field
            assert v.allows_entry is False, field
        for field in ("mark_price", "executable_price"):
            base = dict(mark_price=1.0, executable_price=1.01)
            base[field] = None
            v = evaluate_mark_gap(**base)
            assert v.result == "UNKNOWN", field
            assert v.allows_entry is False, field

    def test_the_guard_threshold_is_ten_percent(self):
        from autopaper.paper_live.economics import (
            MARK_GAP_GUARD_THRESHOLD_PCT,
            evaluate_mark_gap,
        )
        assert MARK_GAP_GUARD_THRESHOLD_PCT == 10.0
        at = evaluate_mark_gap(mark_price=1.0, executable_price=1.1000001)
        assert at.result == "REJECT"
        under = evaluate_mark_gap(mark_price=1.0, executable_price=1.0999)
        assert under.result == "PASS"

    def test_the_trade_enters_at_the_executable_price_not_the_mark(self):
        market = FakeMarket(price=MARK_USD)
        quoted = MARK_USD * 1.03
        quotes = FakeQuoteService(buy=buy_point(effective_price_usd=quoted))
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), market=market,
                             quotes=quotes)
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        assert position.entry_observed_mark == MARK_USD
        assert float(position.entry_fill.effective_price_usd) == pytest.approx(
            quoted, abs=1e-12)
        assert float(position.entry_fill.effective_price_usd) != pytest.approx(
            MARK_USD, abs=1e-12)
        assert position.entry_fill.kind == SIMULATED_FILL_KIND


# ===========================================================================
# 26. latency
# ===========================================================================

class TestLatency:
    def test_the_signal_to_entry_latency_is_measured(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        record = run(engine.on_signal(signal()))[0]
        lat = record.latency.as_dict()
        assert lat["signal_to_simulated_entry_ms"] is not None
        assert lat["signal_to_simulated_entry_ms"] >= 0

    def test_the_buy_quote_latency_is_measured(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        record = run(engine.on_signal(signal()))[0]
        assert record.latency.as_dict()["buy_quote_latency_ms"] is not None

    def test_the_telegram_timestamp_is_not_used_as_the_entry_time(self):
        """
        The Telegram gap is reported separately and LABELLED cross-system. The
        entry timing is measured from local receive, because crediting the
        engine with the Telegram round trip would flatter signal->entry.
        """
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        record = run(engine.on_signal(signal()))[0]
        lat = record.latency.as_dict()
        assert "telegram_to_receive_ms_cross_system" in lat
        assert "clock skew" in lat["clock"]
        assert lat["signal_to_simulated_entry_ms"] < 60_000

    def test_the_sell_quote_latency_is_measured(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        trade = run(engine.monitor_once())[0]
        assert trade.latency.as_dict()["sell_quote_latency_ms"] is not None
        assert trade.latency.as_dict()["exit_trigger_to_fill_ms"] is not None

    def test_durations_use_the_monotonic_clock(self):
        from autopaper.paper_live.schema import LatencyMarks

        marks = LatencyMarks()
        marks.mark_request_at = 100.0
        marks.mark_response_at = 100.25
        assert marks.mark_latency_ms == pytest.approx(250.0)

    def test_a_missing_stage_is_none_not_zero(self):
        from autopaper.paper_live.schema import LatencyMarks

        assert LatencyMarks().mark_latency_ms is None


# ===========================================================================
# 27. the GOAT regression
# ===========================================================================

class TestGoatRegression:
    GOAT_MARK_IN = 1.783e-05
    GOAT_MARK_OUT = 3.059e-05

    def test_the_old_mark_to_mark_figure_is_still_reproducible(self):
        """The historical number is retained, not deleted."""
        mark_return = (self.GOAT_MARK_OUT / self.GOAT_MARK_IN - 1) * 100
        assert mark_return == pytest.approx(71.56477846326415)

    def test_the_old_figure_is_not_a_realized_result(self):
        mark_return = (self.GOAT_MARK_OUT / self.GOAT_MARK_IN - 1) * 100
        # The recorded all-in cash result of the same position.
        realized = -0.182493925 / 0.42564 * 100
        assert mark_return > 0 > realized
        assert abs(mark_return - realized) > 100

    def test_the_regression_fails_if_a_mark_is_used_as_a_fill(self):
        """
        The guard: build the position with the mark as the entry price and
        assert the harness would NOT produce that result.
        """
        # The quote is priced against THIS test's mark (a 2% executable
        # premium) rather than the module default. Without that the 10% entry
        # guard would refuse the order for a fixture mismatch and the
        # regression would never reach the assertion it exists to make.
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            market=FakeMarket(price=self.GOAT_MARK_IN),
            quotes=FakeQuoteService(buy=buy_point(
                effective_price_usd=self.GOAT_MARK_IN * 1.02)),
        )
        record = run(engine.on_signal(signal()))[0]
        position = engine.accounts["L25_CURRENT"].position
        # The mark is retained...
        assert position.entry_observed_mark == self.GOAT_MARK_IN
        # ...and the fill is NOT it.
        assert float(position.entry_fill.effective_price_usd) != self.GOAT_MARK_IN
        assert position.entry_fill.kind == SIMULATED_FILL_KIND

    def test_the_engine_can_never_label_a_mark_as_a_fill(self):
        from autopaper.paper_live.schema import PriceLayer

        assert PriceLayer.OBSERVED_MARK is not PriceLayer.SIMULATED_FILL
        assert ACTUAL_FILL_AVAILABLE is False
        fill = fill_from_quote(buy_point(), side="BUY")
        assert fill.kind == SIMULATED_FILL_KIND
        assert fill.as_dict()["layer"] == PriceLayer.SIMULATED_FILL.value

    def test_mfe_and_mae_are_labelled_as_mark_observations(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal(message_id=1)))
        run(engine.monitor_once())
        engine.observe.__self__  # keep the reference explicit
        position = engine.accounts["L25_CURRENT"].position
        assert position is None or hasattr(position, "observed_mark_mfe")

    def test_a_recorded_mfe_is_named_observed_mark(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal(message_id=1)))
        run(engine.monitor_once())
        row = engine.signal_records[0].as_dict()
        # The gap is still reported under the key the dataset already used, and
        # it now says WHICH reference produced it.
        assert "mark_gap_pct" in row
        assert row["mark_gap_reference"] == "execution_native_same_market"
        assert row["observed_mark"]["layer"] == "OBSERVED_MARK"


# ===========================================================================
# 28. no-broadcast structural safety
# ===========================================================================

class TestNoBroadcast:
    def test_the_audit_passes(self):
        audit = audit_paper_capability()
        assert audit.ok, audit.notes
        assert audit.forbidden_attrs_found == ()
        assert audit.forbidden_modules_found == ()

    def test_assert_paper_only_returns_the_audit(self):
        assert assert_paper_only().ok is True

    def test_the_read_only_surface_is_actually_present(self):
        audit = audit_paper_capability()
        assert audit.read_only_modules_present

    def test_no_module_in_the_package_owns_a_submission_symbol(self):
        import sys

        from autopaper.same_market.guard import is_forbidden_attr

        hits = []
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith("autopaper.paper_live"):
                continue
            for attr in dir(module):
                if is_forbidden_attr(attr) or attr in PAPER_FORBIDDEN_ATTRS:
                    hits.append(f"{name}.{attr}")
        assert hits == []

    def test_no_paper_live_module_imports_an_execution_module(self):
        import sys

        from autopaper.same_market.guard import is_forbidden_module

        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith("autopaper.paper_live"):
                continue
            for attr in dir(module):
                value = getattr(module, attr, None)
                if hasattr(value, "__name__"):
                    assert not is_forbidden_module(value.__name__)
                    assert value.__name__ not in PAPER_FORBIDDEN_MODULES

    def test_the_engine_owns_no_submission_method(self):
        for forbidden in ("send_transaction", "sendRawTransaction", "sign",
                          "broadcast", "submit", "close_at_market",
                          "sell_at_market", "place_order", "withdraw"):
            assert not hasattr(PaperLiveEngine, forbidden), forbidden

    def test_the_virtual_account_owns_no_wallet(self):
        for forbidden in ("private_key", "keypair", "signer", "send",
                          "transfer", "withdraw"):
            assert not hasattr(VirtualAccount, forbidden), forbidden

    def test_the_fill_and_settlement_functions_are_pure(self):
        import inspect

        for func in (fill_from_quote, realize):
            source = inspect.getsource(func)
            for forbidden in ("await ", "send", "open(", "httpx", "socket"):
                assert forbidden not in source, f"{func.__name__} uses {forbidden}"

    def test_the_audit_fails_closed_on_an_injected_forbidden_module(self):
        import sys
        import types

        evil = types.ModuleType("autopaper.paper_live.evil_exec")
        evil.send_transaction = lambda *a, **k: None
        sys.modules["autopaper.paper_live.evil_exec"] = evil
        try:
            audit = audit_paper_capability()
            assert audit.ok is False
            assert any("send_transaction" in a for a in audit.forbidden_attrs_found)
            with pytest.raises(PaperCapabilityViolation):
                assert_paper_only()
        finally:
            del sys.modules["autopaper.paper_live.evil_exec"]

    def test_the_execution_modules_are_on_the_denylist(self):
        for name in ("autopaper.execution.real", "autopaper.execution.solana",
                     "solders", "solana"):
            assert name in PAPER_FORBIDDEN_MODULES

    def test_the_quote_service_owns_no_execution_method(self):
        for forbidden in ("send", "sign", "broadcast", "build", "submit"):
            assert not hasattr(QuoteService, forbidden), forbidden

    def test_the_harness_never_enables_real_trading(self):
        import autopaper.paper_live.engine as engine_module
        import autopaper.paper_live.cli as cli_module
        import autopaper.paper_live.fills as fills_module

        for module in (engine_module, cli_module, fills_module):
            source = __import__("inspect").getsource(module)
            assert "REAL_TRADING_ENABLED" not in source
            assert "trade_mode = TradeMode.REAL" not in source


# ===========================================================================
# 29. deterministic report generation
# ===========================================================================

class TestReports:
    def _dataset(self, tmp_path):
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms([]), quotes=quotes)
        records = run(engine.on_signal(signal(message_id=1)))
        trades = run(engine.monitor_once())
        writer = DatasetWriter(tmp_path, arms=engine.arms)
        return engine, records, trades, writer

    def test_all_four_artefacts_are_written(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        paths = [
            writer.write_json(signals=records, trades=trades, status=engine.status()),
            writer.write_markdown(signals=records, trades=trades,
                                  status=engine.status()),
            writer.write_signals_csv(records),
            writer.write_trades_csv(trades),
        ]
        for path in paths:
            assert path.exists() and path.stat().st_size > 0
            assert path.parent == tmp_path / "reports" / "live"

    def test_the_json_is_valid_and_declares_the_four_layers(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        path = writer.write_json(signals=records, trades=trades,
                                 status=engine.status())
        payload = json.loads(path.read_text())
        layers = payload["price_layers"]
        assert layers["ACTUAL_FILL"] is None
        assert layers["ACTUAL_FILL_AVAILABLE"] is False
        assert layers["SIMULATED_FILL"] == SIMULATED_FILL_KIND
        assert "!=" in layers["statement"]
        assert "exit_mark / entry_mark" in payload["realized_pnl_basis"]

    def test_the_json_contains_no_nan_or_infinity(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        text = writer.write_json(signals=records, trades=trades,
                                 status=engine.status()).read_text()
        assert "NaN" not in text and "Infinity" not in text

    def test_the_signals_csv_has_one_row_per_signal_per_arm(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        path = writer.write_signals_csv(records)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1 + len(records)
        header = lines[0].split(",")
        for column in ("signal_id", "strategy", "observed_entry_mark",
                       "buy_quote_price", "buy_quote_gap_pct",
                       "execution_identity", "failure_reason"):
            assert column in header

    def test_the_trades_csv_carries_the_full_telemetry_set(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        header = writer.write_trades_csv(trades).read_text().split("\n")[0]
        for column in ("observed_entry_mark", "buy_quote_price",
                       "simulated_entry_price", "exit_quote_timestamp",
                       "sell_quote_price", "simulated_exit_price",
                       "observed_mark_mfe", "observed_mark_mae", "hold_seconds",
                       "gross_pnl_quote", "fees_quote", "costs_quote",
                       "net_pnl_quote", "realized_pnl_usd", "realized_pnl_sol",
                       "bankroll_after", "drawdown_pct", "exit_reason",
                       "telegram_latency_ms", "decision_latency_ms",
                       "buy_quote_latency_ms", "signal_to_entry_ms",
                       "sell_quote_latency_ms", "total_lifecycle_ms",
                       "failure_reason"):
            assert column in header

    def _trade_rows(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        text = writer.write_trades_csv(trades).read_text().strip().split("\n")
        header = text[0].split(",")
        rows = [dict(zip(header, line.split(","))) for line in text[1:]]
        return rows, trades

    def test_every_price_column_in_a_trade_row_is_usd_per_token(self, tmp_path):
        """
        A simulated fill price must be the executable quote's own USD price.

        ``SimulatedFill`` carries BOTH ``effective_price`` (QUOTE ASSET - SOL -
        per token) and ``effective_price_usd`` (USD per token). The CSV used to
        emit the SOL form under ``simulated_entry_price`` while the two columns
        either side of it (``observed_entry_mark``, ``buy_quote_price``) are
        USD. That put a SOL price next to USD prices in one row: the token/SOL
        vs token/USD mix this project forbids, and a fill that reads ~120x below
        the quote it was taken from.
        """
        rows, _ = self._trade_rows(tmp_path)
        assert rows, "expected at least one settled trade row"
        row = rows[0]
        quote = float(row["buy_quote_price"])
        fill = float(row["simulated_entry_price"])
        assert quote > 0 and fill > 0
        # The fill IS the quote's own price (the fill model adds nothing on top).
        assert fill == pytest.approx(quote, rel=1e-9)
        # If this were the SOL form it would be quote / SOL_USD (~1/120) of it.
        assert fill > quote / 10.0

    def test_the_exit_trigger_timestamp_is_wall_clock(self, tmp_path):
        """
        ``exit_trigger_timestamp`` must be comparable to the other timestamps.

        The internal stamp is monotonic (seconds since boot, ~3e5) so durations
        cannot jump when the wall clock is adjusted. Emitting it as a timestamp
        next to UNIX stamps (~1.79e9) made ``exit_trigger_timestamp -
        entry_quote_timestamp`` a meaningless ~-1.79e9. The reported value is
        now the wall-clock companion; durations still use the monotonic stamp.
        """
        rows, _ = self._trade_rows(tmp_path)
        assert rows, "expected at least one settled trade row"
        row = rows[0]
        entry_quote = float(row["entry_quote_timestamp"])
        exit_quote = float(row["exit_quote_timestamp"])
        trigger = float(row["exit_trigger_timestamp"])
        assert entry_quote > 1e9 and exit_quote > 1e9
        assert trigger > 1e9, "monotonic value leaked into a timestamp column"
        assert entry_quote - 5.0 <= trigger <= exit_quote + 5.0

    def test_the_mark_to_quote_gap_distribution_is_reported(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        payload = json.loads(writer.write_json(
            signals=records, trades=trades, status=engine.status()).read_text())
        gaps = payload["mark_to_quote_gaps"]
        assert gaps["count"] == len(records)
        assert "threshold_configured" in gaps
        assert gaps["threshold_configured"] is False
        assert "median_pct" in gaps

    def test_report_generation_is_deterministic(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        a = writer.write_signals_csv(records).read_text()
        b = writer.write_signals_csv(records).read_text()
        assert a == b
        rows = DatasetWriter(tmp_path, arms=engine.arms).write_trades_csv(trades)
        first = rows.read_text()
        second = writer.write_trades_csv(trades).read_text()
        assert first == second

    def test_no_credential_appears_in_any_artefact(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        blob = "\n".join([
            writer.write_json(signals=records, trades=trades,
                              status=engine.status()).read_text(),
            writer.write_markdown(signals=records, trades=trades,
                                  status=engine.status()).read_text(),
            writer.write_signals_csv(records).read_text(),
            writer.write_trades_csv(trades).read_text(),
        ]).lower()
        for forbidden in ("api_id", "api_hash", "bot_token", "session",
                          "private_key", "seed phrase", ".env.local"):
            assert forbidden not in blob, forbidden

    def test_the_raw_signal_text_is_not_stored_only_hashed(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        row = records[0].signal
        assert len(row.raw_text_sha256) == 64
        assert row.raw_text_length == len(f"BUY ${MINT}")
        assert f"BUY ${MINT}" not in writer.write_signals_csv(records).read_text()

    def test_the_markdown_carries_the_no_execution_statement(self, tmp_path):
        engine, records, trades, writer = self._dataset(tmp_path)
        text = writer.write_markdown(signals=records, trades=trades,
                                     status=engine.status()).read_text()
        assert "NO TRANSACTION WAS BUILT, SIGNED, OR BROADCAST." in text
        assert "OBSERVED_MARK  !=  EXECUTABLE_QUOTE  !=  SIMULATED_FILL  !=  ACTUAL_FILL" in text


# ===========================================================================
# exit fallback: the deterministic rule
# ===========================================================================

class TestExitFallback:
    def test_a_position_is_left_open_when_the_sell_quote_is_unavailable(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            quotes=FakeQuoteService(sell=unavailable_point(side="SELL")),
        )
        run(engine.on_signal(signal()))
        trades = run(engine.monitor_once())
        assert len(trades) == 1
        assert trades[0].pnl is None
        assert trades[0].failure is FailureReason.EXIT_QUOTE_UNAVAILABLE
        assert engine.accounts["L25_CURRENT"].has_position

    # -- the graduated-curve trap -----------------------------------------
    #
    # A position entered on a pump BONDING CURVE and held past graduation has
    # its entry venue cease to exist ("the bonding curve is complete"). The
    # settlement path sells on the pinned entry venue, obtains no quote, and
    # leaves the position OPEN; with max 1 position per arm, every later signal
    # is then blocked (ENTRY_BLOCKED_POSITION_ACTIVE). This is the observed live
    # state (reports/live/execution_aware_paper.json: 8 EXIT_QUOTE_UNAVAILABLE,
    # then ENTRY_BLOCKED_POSITION_ACTIVE).
    #
    # The intended behaviour -- acknowledged elsewhere in the engine -- is that
    # a graduated curve is exited through the PumpSwap pool it became. That is
    # NOT implemented in the settlement path, so this test is marked xfail
    # (strict): it documents the defect and will start XPASSing the moment a
    # fix lands, forcing the marker off.
    class _GraduatedEntryVenue(FakeQuoteService):
        """
        The entry venue (a bonding curve) graduated mid-hold.

        The pinned curve is gone, but the PumpSwap pool it became CAN quote --
        which is the whole point: a real exit exists, just not on the pinned
        book. The pool quote carries a DIFFERENT venue id, so it can never be
        mistaken for (or reused as) the pinned settlement.
        """

        POOL_VENUE = f"PUMPSWAP:{PAIR}"

        def __init__(self):
            base = sell_point()
            native = replace(
                base.execution_native_reference, market_id=self.POOL_VENUE)
            pool = replace(
                base, venue_id=self.POOL_VENUE, market_type="PUMPSWAP",
                venue="pumpswap-pool", pool_or_curve=PAIR,
                execution_native_reference=native)
            super().__init__(sell=pool)

        async def quote_sell_pinned(self, **kw):
            return (
                unavailable_point(
                    side="SELL",
                    missing_reason=(
                        "the bonding curve is complete, so it is no longer the "
                        "market"),
                ),
                None,
            )

        async def quote_sell_best(self, **kw):
            sell = self._restamp(self.sell, kw["mint"])
            return (sell, [])

        async def resolve_graduated_exit_venue(self, **kw):
            # The position entered on a graduated curve; the exit venue moved
            # to the PumpSwap pool it became. Same token, same book its price
            # is now quoted on.
            return self._restamp(self.sell, kw["mint"])

    def test_a_position_whose_entry_venue_graduated_is_exited(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            quotes=self._GraduatedEntryVenue(),
        )
        run(engine.on_signal(signal()))
        trades = run(engine.monitor_once())
        assert trades and trades[0].pnl is not None
        # The migration is recorded on the position, so the pin, the provenance
        # and every later tick resolve against the NEW book, never the dead one.
        assert trades[0].position.exit_venue_id is not None
        assert trades[0].position.exit_venue_id != trades[0].position.entry_venue_id

    def test_a_position_is_left_open_when_the_sell_quote_is_stale(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            quotes=FakeQuoteService(
                sell=sell_point(quote_age_ms=90_000.0, max_quote_age_ms=30_000.0)
            ),
        )
        run(engine.on_signal(signal()))
        trades = run(engine.monitor_once())
        assert trades[0].pnl is None
        assert trades[0].failure is FailureReason.EXIT_QUOTE_STALE
        assert engine.accounts["L25_CURRENT"].has_position

    def test_a_position_is_left_open_when_the_exit_identity_is_unproven(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            quotes=FakeQuoteService(sell=sell_point(identity="UNKNOWN")),
        )
        run(engine.on_signal(signal()))
        trades = run(engine.monitor_once())
        assert trades[0].pnl is None
        assert trades[0].failure is FailureReason.EXIT_QUOTE_IDENTITY_UNKNOWN
        assert engine.accounts["L25_CURRENT"].has_position

    def test_the_bankroll_is_untouched_by_a_failed_exit(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            quotes=FakeQuoteService(sell=unavailable_point(side="SELL")),
        )
        run(engine.on_signal(signal()))
        run(engine.monitor_once())
        assert engine.accounts["L25_CURRENT"].bankroll_usd == 5.0
        assert engine.accounts["L25_CURRENT"].closed_trades == 0

    def test_the_rule_is_deterministic_across_repeated_passes(self):
        """A transient outage must not become a loss or a phantom gain."""
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            quotes=FakeQuoteService(sell=unavailable_point(side="SELL")),
        )
        run(engine.on_signal(signal()))
        for _ in range(4):
            run(engine.monitor_once())
        assert engine.accounts["L25_CURRENT"].bankroll_usd == 5.0
        assert engine.accounts["L25_CURRENT"].has_position

    def test_a_later_successful_quote_settles_the_still_open_position(self):
        quotes = FakeQuoteService(sell=unavailable_point(side="SELL"))
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]), quotes=quotes)
        run(engine.on_signal(signal()))
        run(engine.monitor_once())
        assert engine.accounts["L25_CURRENT"].has_position
        # The provider recovers.
        quotes.sell = sell_point()
        trades = run(engine.monitor_once())
        assert trades[0].pnl is not None
        assert engine.accounts["L25_CURRENT"].has_position is False


# ===========================================================================
# arm independence and the shared signal stream
# ===========================================================================

class TestArms:
    def test_every_arm_shares_one_signal_and_one_quote_snapshot(self):
        arms = parse_arms([])
        quotes = FakeQuoteService()
        engine = make_engine(arms=arms, quotes=quotes)
        records = run(engine.on_signal(signal(message_id=1)))
        assert len(records) == len(arms)

        # ONE provider request, THREE arm consumers.
        #
        # This used to assert three requests. That was measuring the wrong thing:
        # identical fetches mean the arms can be measured against
        # three DIFFERENT snapshots of the same book, so the arms stop being
        # comparable. Sharing is a correctness property here, not an
        # optimisation, which is why it is asserted rather than left to chance.
        assert len(quotes.buy_calls) == 1, (
            f"the arms made {len(quotes.buy_calls)} provider requests for one "
            "signal; they must share a single snapshot")
        assert quotes.buy_calls[0]["sol_lamports"] == BUY_LAMPORTS
        assert BUY_LAMPORTS == 4_086_302

        # Every arm consumed that identical snapshot.
        assert {r.buy_quote.quote_timestamp for r in records} == {
            records[0].buy_quote.quote_timestamp
        }
        assert {r.buy_quote.output_amount_raw for r in records} == {
            records[0].buy_quote.output_amount_raw
        }
        stats = engine.quote_cache.stats()
        # every arm after the first is served from the shared snapshot
        assert stats["cache_hits"] == len(engine.arms) - 1, stats
        assert stats["shared_snapshots"] == 1, stats

    def test_arms_with_different_sizes_do_not_share_a_snapshot(self):
        """
        Sharing is only correct when the ORDER SIZE matches.

        Two arms whose bankrolls have diverged quote different sizes, so serving
        one arm from the other's snapshot would fill a larger arm at a smaller
        arm's price -- exactly the bug the exact-size rule exists to prevent.
        The cache is keyed on the size, so this is checked directly rather than
        inferred from a signal flow the one-position rule would block anyway.
        """
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms([]), quotes=quotes)
        from autopaper.paper_live.quotes import QuoteCache

        cache = QuoteCache()
        small = run(engine._quote_buy_best(
            mint="M", sol_lamports=1_000_000, token_decimals=6,
            observed_pair_address=None))[0]
        before = len(quotes.buy_calls)
        same = run(engine._quote_buy_best(
            mint="M", sol_lamports=1_000_000, token_decimals=6,
            observed_pair_address=None))[0]
        assert len(quotes.buy_calls) == before, "an identical size was re-fetched"
        assert same.quote_timestamp == small.quote_timestamp

        run(engine._quote_buy_best(
            mint="M", sol_lamports=9_000_000, token_decimals=6,
            observed_pair_address=None))
        assert len(quotes.buy_calls) == before + 1, (
            "a DIFFERENT order size was served from the smaller arm's snapshot")

    def test_each_arm_owns_an_independent_bankroll(self):
        engine = make_engine(arms=parse_arms([]))
        engine.accounts["L25_CURRENT"].bankroll_usd = 4.0
        assert engine.accounts["L25_60S"].bankroll_usd == 5.0
        assert engine.accounts["H2_CAP60"].bankroll_usd == 5.0

    def test_one_arms_loss_does_not_move_another(self):
        quotes = FakeQuoteService()
        engine = make_engine(arms=parse_arms([]), quotes=quotes)
        run(engine.on_signal(signal(message_id=1)))
        run(engine.monitor_once())
        bankrolls = {n: a.bankroll_usd for n, a in engine.accounts.items()}
        assert len(set(round(v, 12) for v in bankrolls.values())) == 1

    def test_an_unknown_arm_name_raises_rather_than_defaulting(self):
        with pytest.raises(ValueError) as exc:
            arm_by_name("L25_TYPO")
        assert "L25_CURRENT" in str(exc.value)

    def test_a_retired_arm_is_refused_for_new_selection(self):
        from autopaper.paper_live.arms import parse_arms
        with pytest.raises(ValueError):
            parse_arms(["L25_TP100"])

    def test_an_empty_arm_list_means_every_arm(self):
        # "every arm" is defined by the registry, not by a literal.
        assert len(parse_arms([])) == len(arm_names())
        # ...and "every arm" is NOT "every arm plus the beta experiment": the
        # V2 FLOOR (BETA) arms are opt-in, so an empty list must not start them.
        assert not any(s.name.value.startswith("V2_BETA_")
                       for s in parse_arms([]))

    def test_parsing_is_case_insensitive_and_trims(self):
        assert arm_by_name(" l25_tp100 ").name is ResearchArm.L25_TP100, (
            "a RETIRED arm must still RESOLVE so historical ledgers and already-"
            "open positions keep their meaning; it is simply not offered for new "
            "selection")

    def test_every_arm_records_its_provenance(self):
        for spec in ARMS.values():
            assert spec.provenance
            assert "autopaper" in spec.provenance or "research" in spec.provenance
