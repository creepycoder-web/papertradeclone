"""
Focused tests for the executable-quote PAPER runtime.

Grouped to match the release audit: economics, BUY, SELL, runtime, rent, and
REAL compatibility. Everything here is OFFLINE and injected -- no test reaches
RPC, and none of them asserts on a source string where execution would prove the
thing.

The one live, read-only smoke lives in
``tests/test_execution_aware_live_smoke.py`` and is marked ``live``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from autopaper.config import RealConfig, build_config
from autopaper.engine import PaperTrader
from autopaper.execution.quoting import Side
from autopaper.modes import TradeMode
from autopaper.paper_live.arms import (
    BETA_ARM_KEYS,
    BETA_ARMS,
    BETA_GROUP,
    arm_names,
)
from autopaper.paper_live.economics import (
    COST_IRRECOVERABLE,
    COST_RENT_LOCK,
    COST_RENT_RECLAIMABLE,
    COST_TRADING,
    ECONOMIC_INVALID,
    ECONOMIC_MARGINAL,
    ECONOMIC_VALID,
    REAL_RECLAIM_TOKEN,
    TOKEN_2022_PROGRAM_ID,
    TOKEN_PROGRAM_ID,
    EconomicCheck,
    TokenAccountFacts,
    evaluate_entry_economics,
    evaluate_mark_gap,
    evaluate_reclaim,
    reclaim_plan,
    rent_exempt_minimum_for_size,
    required_rent_lamports,
)
from autopaper.paper_live.real_compat import (
    SameMarketQuoteProvider,
    assert_quote_contract_compatible,
    to_executable_quote,
)
from autopaper.paper_live.state import PaperStateStore, atomic_write_text

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
try:  # pragma: no cover - depends on how pytest was invoked
    from tests import paper_bridge_fakes as _fk
except ImportError:  # pragma: no cover
    import paper_bridge_fakes as _fk

MARK_USD = _fk.MARK_USD
QUOTE_USD = _fk.QUOTE_USD
FakeQuoteService = _fk.FakeQuoteService
alert = _fk.alert
bridge = _fk.bridge
buy_point = _fk.buy_point
real_report = _fk.real_report
run = _fk.run
sell_point = _fk.sell_point

_UNUSED = (
    MARK_USD,
    QUOTE_USD,
    FakeQuoteService,
    alert,
    bridge,
    buy_point,
    real_report,
    run,
    sell_point,
)

OWNER = "W" * 44
MINT = "M" * 44
WSOL = "So11111111111111111111111111111111111111112"



def _buy_with_native_gap(fraction: float):
    """
    A BUY quote whose same-market spot sits ``fraction`` below the executed
    price, so the execution-native gap is exactly ``fraction * 100`` percent.

    The reference is moved rather than the quote, because the reference is what
    the guard compares against.
    """
    from dataclasses import replace as _replace

    quote = buy_point()
    native = quote.execution_native_reference
    assert native is not None
    return _replace(
        quote,
        execution_native_reference=_replace(
            native,
            reference_price=float(quote.effective_price) / (1.0 + fraction),
            numerator_raw=int(native.numerator_raw / (1.0 + fraction))),
    )


def good_facts(**over) -> TokenAccountFacts:
    base = dict(
        address="A" * 44, exists=True, owner=OWNER,
        token_program_id=TOKEN_PROGRAM_ID, mint=MINT,
        token_balance_raw=0, lamports=2_039_280, has_ata=True,
    )
    base.update(over)
    return TokenAccountFacts(**base)


# ===========================================================================
# ECONOMICS
# ===========================================================================


class TestMicroAccountEconomics:
    def test_a_normal_micro_account_is_valid(self):
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=122.36,
            entry_fee_bps=120, exit_fee_bps=120)
        assert check.allowed is True
        assert check.classification == ECONOMIC_VALID

    def test_a_marginal_account_is_reported_separately(self):
        """MARGINAL is still allowed, but a run can be told from a clean one."""
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=122.36,
            entry_fee_bps=1000, exit_fee_bps=1000)
        assert check.classification in (ECONOMIC_MARGINAL, ECONOMIC_INVALID)
        if check.classification == ECONOMIC_MARGINAL:
            assert check.allowed is True

    def test_an_invalid_micro_account_is_refused(self):
        check = evaluate_entry_economics(
            notional_usd=0.05, sol_usd=122.36,
            entry_fee_bps=120, exit_fee_bps=120)
        assert check.allowed is False
        assert check.classification == ECONOMIC_INVALID
        assert "floor" in check.reason

    def test_rent_dominated_account_is_refused(self):
        """When the LOCKED rent exceeds the notional, the trade is the deposit."""
        check = evaluate_entry_economics(
            notional_usd=0.15, sol_usd=122.36,
            entry_fee_bps=120, exit_fee_bps=120)
        assert check.allowed is False
        assert check.classification == ECONOMIC_INVALID
        assert "rent deposit" in check.reason

    def test_reclaimable_rent_is_not_a_trading_cost(self):
        """
        The single most important economic property here.

        Rent is ~$0.25 against a $0.50 notional. Charging it as a permanent cost
        would refuse every trade in a $5 run AND double-count a recoverable
        balance as a loss.
        """
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=122.36,
            entry_fee_bps=120, exit_fee_bps=120)
        assert check.rent_usd > 0.2
        assert COST_RENT_LOCK in check.cost_categories
        # The gate is charged fees + costs, and rent is excluded from it.
        assert check.fixed_cost_usd == pytest.approx(
            check.entry_fee_usd + check.exit_fee_usd
            + check.modeled_costs_usd, abs=1e-12)
        assert check.as_dict()["in_canonical_pnl"][COST_RENT_LOCK] is False
        assert check.as_dict()["in_canonical_pnl"][COST_RENT_RECLAIMABLE] is False
        assert check.as_dict()["in_canonical_pnl"][COST_TRADING] is True

    def test_unknown_sol_reference_does_not_assume_a_price(self):
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=0.0, entry_fee_bps=120)
        assert check.rent_usd == 0.0

    def test_position_size_is_never_inflated_to_absorb_rent(self):
        """The economic guard refuses; it never enlarges the order."""
        from autopaper.sizing import compute_position_size

        size = compute_position_size(5.00)
        assert size.position_usd == pytest.approx(0.50)
        check = evaluate_entry_economics(
            notional_usd=size.position_usd, sol_usd=122.36,
            entry_fee_bps=120, exit_fee_bps=120)
        # Whatever the verdict, the size that fed it is untouched.
        assert check.notional_usd == pytest.approx(0.50)


# ===========================================================================
# BUY
# ===========================================================================


class TestBuyPath:
    def test_the_entry_fill_comes_from_the_executable_quote(self):
        engine = bridge()
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        traded = [r for r in records if r.traded]
        assert traded
        for record in traded:
            position = engine._position_for(record)
            assert float(position.entry_fill.effective_price_usd) == pytest.approx(
                float(record.buy_quote.effective_price_usd), abs=1e-12)

    def test_the_mark_is_never_the_fill(self):
        engine = bridge()
        for record in run(engine.handle_alert(
                snapshot=alert(), report=real_report())):
            position = engine._position_for(record)
            if position is None:
                continue
            assert float(position.entry_fill.effective_price_usd) != \
                pytest.approx(position.entry_observed_mark, abs=1e-12)

    def test_a_gap_beyond_ten_percent_refuses_the_entry(self):
        """
        The guard is now measured against the SAME MARKET'S OWN price.

        The threshold is still 10% and a materially expensive entry is still
        refused; only the reference moved. So the fixture widens the gap by
        moving the execution-native reference, not the observed mark.
        """
        from autopaper.paper_live.arms import BETA_ARMS
        engine = bridge(quotes=FakeQuoteService(
            buy=_buy_with_native_gap(0.50)))
        beta_arms = {a.value for a in BETA_ARMS}
        for record in run(engine.handle_alert(
                snapshot=alert(), report=real_report())):
            if record.arm in beta_arms:
                # Beta arms are rejected at the V2_FLOOR gate, not the gap guard
                continue
            assert record.traded is False
            assert record.mark_to_quote_gap_pct is not None, \
                "the measured gap must survive the refusal"

    def test_a_gap_within_ten_percent_trades(self):
        engine = bridge(quotes=FakeQuoteService(
            buy=_buy_with_native_gap(0.05)))
        assert any(r.traded for r in run(engine.handle_alert(
            snapshot=alert(), report=real_report())))

    def test_a_lagging_observed_mark_no_longer_refuses_the_entry(self):
        """
        THE REGRESSION THIS CHANGE EXISTS FOR.

        The external mark is moved 50% away from a normal same-market
        execution cost. The entry proceeds, because the mark is a diagnostic
        and not the reference -- and the legacy gap is still recorded so the
        disagreement stays auditable.
        """
        from autopaper.paper_live.arms import BETA_ARMS
        engine = bridge(quotes=FakeQuoteService(
            buy=_buy_with_native_gap(0.0125)),
            market=_MarketAt(MARK_USD * 0.66))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        beta_arms = {a for a in BETA_ARMS}
        regular_records = [r for r in records if r.arm not in beta_arms]
        assert all(r.traded for r in regular_records), (
            "a stale external mark must not refuse an otherwise sane entry")
        assert all(r.legacy_observed_mark_gap_pct is not None and abs(r.legacy_observed_mark_gap_pct) > 30.0
                   for r in regular_records), (
            "the legacy gap must still be measured on the same record")

    def test_a_wrong_token_quote_is_refused(self):
        engine = bridge(quotes=FakeQuoteService(
            buy=buy_point(token_mint="Z" * 44), verbatim=True))
        for record in run(engine.handle_alert(
                snapshot=alert(), report=real_report())):
            assert record.traded is False

    def test_a_wrong_quote_mint_is_refused(self):
        engine = bridge(quotes=FakeQuoteService(
            buy=buy_point(quote_mint=MINT)))
        for record in run(engine.handle_alert(
                snapshot=alert(), report=real_report())):
            assert record.traded is False

    def test_a_stale_quote_is_refused(self):
        engine = bridge(quotes=FakeQuoteService(
            buy=buy_point(quote_age_ms=600_000.0, max_quote_age_ms=30_000.0)))
        for record in run(engine.handle_alert(
                snapshot=alert(), report=real_report())):
            assert record.traded is False

    def test_a_missing_quote_is_refused_and_recorded(self):
        engine = bridge(quotes=FakeQuoteService(
            buy=buy_point(available=False, missing_reason="rpc refused")))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        assert len(records) == len(engine.arms)
        assert all(not r.traded for r in records)

    def test_unproven_market_identity_is_refused(self):
        engine = bridge(quotes=FakeQuoteService(
            buy=buy_point(identity="UNKNOWN")))
        for record in run(engine.handle_alert(
                snapshot=alert(), report=real_report())):
            assert record.traded is False

    def test_best_venue_is_selected_by_executable_output(self):
        """
        NOT by mark, liquidity, symbol or preference.

        Two venues quoted for the exact size; the one returning more output
        wins even though it is the WORSE price on a naive nominal-price sort.
        """
        from autopaper.paper_live.quotes import QuoteService

        richer = buy_point(
            venue_id="PUMPSWAP:RICH", output_amount_raw=10_000_000_000,
            output_amount_decimal=Decimal("10000.0"), venue="the-worse-label")
        poorer = buy_point(
            venue_id="PUMPSWAP:POOR", output_amount_raw=9_000_000_000,
            output_amount_decimal=Decimal("9000.0"), venue="the-nice-label")
        best = QuoteService._select_best([poorer, richer], side="BUY")
        assert best.venue_id == "PUMPSWAP:RICH"
        assert float(best.output_amount_decimal) == 10000.0

    def test_unproven_identity_is_excluded_rather_than_ranked_last(self):
        from autopaper.paper_live.quotes import QuoteService

        good = buy_point(venue_id="PUMPSWAP:OK",
                         output_amount_decimal=Decimal("100.0"))
        rich_but_unproven = buy_point(
            venue_id="PUMPSWAP:??", identity="UNKNOWN",
            output_amount_decimal=Decimal("999999.0"))
        best = QuoteService._select_best([good, rich_but_unproven], side="BUY")
        assert best.venue_id == "PUMPSWAP:OK", \
            "an unproven venue must never win on output alone"

    def test_a_stale_venue_is_excluded_from_selection(self):
        from autopaper.paper_live.quotes import QuoteService

        stale = buy_point(venue_id="PUMPSWAP:STALE",
                          output_amount_decimal=Decimal("999999.0"),
                          quote_age_ms=600_000.0)
        fresh = buy_point(venue_id="PUMPSWAP:OK",
                          output_amount_decimal=Decimal("100.0"))
        best = QuoteService._select_best([stale, fresh], side="BUY")
        assert best.venue_id == "PUMPSWAP:OK"

    def test_the_venue_id_distinguishes_a_curve_from_a_pool(self):
        from autopaper.paper_live.quotes import QuoteService

        curve = QuoteService.venue_id_for("PUMP_BONDING_CURVE", "CURVE1")
        pool = QuoteService.venue_id_for("PUMPSWAP", "CURVE1")
        assert curve != pool
        assert curve != "Pump.fun" and pool != "Pump.fun"


# ===========================================================================
# SELL
# ===========================================================================


class TestSellPath:
    def _open(self, **kw):
        engine = bridge(**kw)
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        return engine

    def test_the_exit_fill_comes_from_the_executable_quote(self):
        engine = self._open()
        engine.exit_policy = _ExitNow()
        trades = [t for t in run(engine.tick()) if t.failure.name == "NONE"]
        assert trades
        for trade in trades:
            assert float(trade.exit_fill.output_amount_decimal) > 0
            assert "SIMULATED_FILL" in str(trade.exit_fill.kind)

    def test_the_mark_is_never_the_exit_fill(self):
        engine = self._open()
        engine.exit_policy = _ExitNow()
        for trade in run(engine.tick()):
            if trade.failure.name != "NONE" or trade.exit_observed_mark is None:
                continue
            assert float(trade.exit_observed_mark) != pytest.approx(
                float(trade.exit_fill.effective_price_usd), abs=1e-15)

    def test_the_exit_is_pinned_to_the_entry_venue(self):
        engine = self._open()
        engine.exit_policy = _ExitNow()
        run(engine.tick())
        pinned = {
            t.position.entry_venue_id
            for t in engine.trade_records
        }
        assert pinned, "no position recorded an entry venue"
        for venue in pinned:
            assert venue and ":" in venue, venue
        sells = [c for c in engine.quote_service.sell_calls]
        assert sells, "no SELL quote was requested"

    def test_an_unavailable_pinned_venue_keeps_the_position_open(self):
        from autopaper.paper_live.arms import BETA_ARMS
        engine = self._open()
        engine.exit_policy = _ExitNow()
        engine.quote_service = FakeQuoteService(
            sell=sell_point(available=False, missing_reason="venue gone"))
        trades = run(engine.tick())
        assert all(t.failure.name != "NONE" for t in trades)
        beta_arms = {a.value for a in BETA_ARMS}
        for arm_name, account in engine.accounts.items():
            if arm_name in beta_arms:
                continue
            assert account.has_position, \
                "a position closed without an executable exit quote"

    def test_a_failed_sell_produces_no_close_notification(self):
        from autopaper.paper_live.arms import BETA_ARMS
        engine = self._open()
        engine.exit_policy = _ExitNow()
        engine.quote_service = FakeQuoteService(
            sell=sell_point(available=False, missing_reason="x"))
        run(engine.tick())
        assert [m for m in engine.drain_messages()
                if m["kind"] == "exit"] == []

# SELL BEST-ROUTE
# ===========================================================================


class TestSellBestRoute:
    def test_two_sell_routes_higher_outamount_wins(self):
        from autopaper.paper_live.quotes import QuoteService

        richer = sell_point(
            venue_id="PUMPSWAP:RICH", output_amount_raw=10_000_000_000,
            output_amount_decimal=Decimal("10000.0"), venue="the-worse-label")
        poorer = sell_point(
            venue_id="PUMPSWAP:POOR", output_amount_raw=9_000_000_000,
            output_amount_decimal=Decimal("9000.0"), venue="the-nice-label")
        best = QuoteService._select_best([poorer, richer], side="SELL")
        assert best.venue_id == "PUMPSWAP:RICH"
        assert float(best.output_amount_decimal) == 10000.0

    def test_a_graduated_token_sells_through_jupiter(self):
        """A graduated Pump token with a dead bonding curve must not strand
        the position: Jupiter best-route provides the exit."""
        engine = bridge(quotes=FakeQuoteService(
            sell=sell_point(provider="jupiter", market_type="JUPITER",
                            venue="jupiter", venue_id="JUPITER:TEST")))
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.exit_policy = _ExitNow()
        trades = [t for t in run(engine.tick()) if t.failure.name == "NONE"]
        assert trades, "a valid Jupiter SELL should close the position"
        for trade in trades:
            assert "SIMULATED_FILL" in str(trade.exit_fill.kind)

    def test_no_jupiter_route_keeps_position_open(self):
        """No Jupiter route means no fake close."""
        engine = bridge(quotes=FakeQuoteService(
            sell=sell_point(available=False, missing_reason="no jupiter route")))
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.exit_policy = _ExitNow()
        trades = run(engine.tick())
        assert all(t.failure.name != "NONE" for t in trades)
        for account in engine.accounts.values():
            assert account.has_position, "a position closed without an executable quote"

    def test_a_wrong_token_route_is_rejected(self):
        """Jupiter route for the WRONG token must be rejected."""
        engine = bridge(quotes=FakeQuoteService(
            sell=sell_point(token_mint="Z" * 44), verbatim=True))
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.exit_policy = _ExitNow()
        for trade in run(engine.tick()):
            assert trade.failure.name != "NONE"

    def test_a_stale_jupiter_quote_is_rejected(self):
        """A stale Jupiter quote must not produce a fill."""
        engine = bridge(quotes=FakeQuoteService(
            sell=sell_point(quote_age_ms=600_000.0, max_quote_age_ms=30_000.0)))
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.exit_policy = _ExitNow()
        for trade in run(engine.tick()):
            assert trade.failure.name != "NONE"

    def test_paper_sell_uses_model_derived_fill(self):
        """PAPER SELL fills are MODEL_DERIVED, never REAL."""
        engine = bridge()
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.exit_policy = _ExitNow()
        trades = [t for t in run(engine.tick()) if t.failure.name == "NONE"]
        for trade in trades:
            kind = str(trade.exit_fill.kind)
            assert "MODEL_DERIVED" in kind or "SIMULATED_FILL" in kind, \
                f"PAPER SELL must be MODEL_DERIVED, got {kind}"

    def test_real_remains_disabled_by_default(self):
        """REAL trading must never arm without explicit env authorization."""
        cfg = _config()
        assert cfg.real.trading_enabled is not True, \
            "REAL must not be armed by default in a PAPER runtime"

    def test_pinned_venue_is_not_the_primary_sell_path(self):
        """The exit path must use quote_sell_best, not quote_sell_pinned as
        the primary discovery mechanism."""
        engine = bridge()
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.exit_policy = _ExitNow()
        trades = [t for t in run(engine.tick()) if t.failure.name == "NONE"]
        assert trades, "expected at least one successful close"
        assert engine.quote_service.sell_calls, "no SELL quote recorded"
        for call in engine.quote_service.sell_calls:
            assert "pinned_venue_id" not in call, \
                "SELL path must not use pinned venue as primary"

# ===========================================================================
# RUNTIME
# ===========================================================================


class TestRuntime:
    def test_executable_paper_is_the_default(self):
        cfg = _config()
        trader = PaperTrader(cfg, market=_Market(), nar=_Nar())
        assert trader._execution_aware_paper() is True

    def test_the_legacy_path_needs_the_rollback_flag(self):
        cfg = replace(
            _config(),
            real=replace(_config().real, paper_legacy_mark_fill_rollback=True))
        trader = PaperTrader(cfg, market=_Market(), nar=_Nar())
        assert trader._execution_aware_paper() is False

    def test_real_mode_never_takes_the_paper_branch(self, monkeypatch):
        trader = PaperTrader(_config(), market=_Market(), nar=_Nar())
        monkeypatch.setattr(
            type(trader.state), "trade_mode",
            property(lambda self: TradeMode.REAL))
        assert trader._execution_aware_paper() is False

    def test_an_offline_runtime_cannot_build_a_network_quote_client(self):
        """
        THE OFFLINE GUARANTEE.

        The runtime is on the executable-quote default and still refuses to
        construct a chain client without an injected provider. This is the
        hazard the default flip created, asserted directly.
        """
        trader = PaperTrader(_config(), market=_Market(), nar=_Nar())
        trader.quote_service_override = None
        trader._paper_bridge = None
        # conftest's autouse guard makes this raise; here we assert the SEAM
        # exists and that an injected service bypasses construction entirely.
        assert hasattr(trader, "quote_service_override")
        assert trader.quote_service_override is None

    def test_an_injected_quote_service_is_used_by_the_runtime(self):
        injected = FakeQuoteService()
        trader = PaperTrader(_config(), market=_Market(), nar=_Nar(),
                             quote_service=injected)
        trader._paper_bridge = None
        # Constructing the engine must NOT build a new quote service.
        engine = run(trader._paper_bridge_instance())
        assert engine.quote_service is injected

    def test_the_book_is_loaded_at_startup_not_at_the_first_signal(
            self, tmp_path, monkeypatch):
        """
        THE DEFERRAL DEFECT.

        The execution-aware PAPER bridge was built ONLY on the entry path, and
        ``monitor_once`` skips that book while it does not exist::

            if self._execution_aware_paper() and self._paper_bridge is not None:
                trades = await self._paper_bridge.tick()

        So a restart left the whole book -- open positions AND the V2 FLOOR
        (BETA) arms that are still waiting for an executable SELL quote --
        unmonitored until some unrelated candidate happened to be authorised,
        which under a BETA-only selection the V2 FLOOR gate can refuse for
        hours. In the production run this was found in, the previous process
        retried its three unresolved beta arms every ~6s up to its shutdown and
        the process that replaced it retried NOTHING for the next fifteen
        minutes.

        Asserted as BEHAVIOUR, with no signal delivered anywhere: the boot step
        builds the book, loads the experiment that owns those arms, and the very
        next monitor tick reaches it.
        """
        trader = PaperTrader(_config(tmp_path), market=_Market(), nar=_Nar(),
                             quote_service=FakeQuoteService(),
                             sol_usd_provider=_sol_usd_provider())
        # The operator's selection lives on the runtime state; the bridge
        # captures it at construction, which is what this test pins.
        ok, message = trader.state.select_strategies([BETA_GROUP])
        assert ok, message
        # BEFORE: dark. Nothing has been built, so nothing is being monitored.
        assert trader._paper_bridge is None
        assert trader.paper_bridge_status() is None

        report = run(trader.restore_execution_aware_book())

        assert report["built"] is True, report
        assert trader._paper_bridge is not None
        assert trader.paper_bridge_status() is not None
        experiment = trader._paper_bridge.beta_experiment
        assert experiment is not None, (
            "the experiment OWNS the unresolved arms; a book restored without "
            "it cannot retry them")
        # Derived from the registry, so adding an arm cannot silently stop
        # being covered here.
        expected_keys = {BETA_ARM_KEYS[arm] for arm in BETA_ARMS}
        assert set(experiment.accounts) == expected_keys
        for key in expected_keys:
            assert experiment.accounts[key].bankroll_usd == 10.0

        # The retry loop a restart used to silence: one tick, no signal.
        ticks: list[int] = []
        original = type(trader._paper_bridge).tick

        async def _counting_tick(self):
            ticks.append(1)
            return await original(self)

        monkeypatch.setattr(type(trader._paper_bridge), "tick", _counting_tick)
        run(trader.monitor_once())
        assert ticks == [1], (
            "the monitor tick skipped the execution-aware book, so no open "
            "position was evaluated and no unresolved arm was retried")

        # AND IT IS WIRED IN. The step must be awaited by ``run()`` itself, and
        # before the listener is connected -- a method nothing calls is not a
        # fix. Asserted on the source because driving a real ``run()`` needs a
        # live Telegram session and a real market.
        import inspect

        source = inspect.getsource(type(trader).run)
        call = "await self.restore_execution_aware_book()"
        assert call in source, "run() never loads the PAPER book at startup"
        assert source.index(call) < source.index("_supervise_listener"), (
            "the book must be whole BEFORE the listener can deliver a signal")

    def test_the_boot_step_is_inert_when_the_legacy_path_is_selected(
            self, tmp_path):
        """The boot build must respect the rollback flag, not bypass it."""
        cfg = _config(tmp_path)
        cfg = replace(cfg, real=replace(
            cfg.real, paper_legacy_mark_fill_rollback=True))
        trader = PaperTrader(cfg, market=_Market(), nar=_Nar(),
                             quote_service=FakeQuoteService(),
                             sol_usd_provider=_sol_usd_provider())
        report = run(trader.restore_execution_aware_book())
        assert report["built"] is False
        assert report["reason"] == "execution-aware PAPER is off"
        assert trader._paper_bridge is None

    def test_quote_sharing_across_arms_uses_one_provider_request(self):
        quotes = FakeQuoteService()
        engine = bridge(quotes=quotes)
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        assert len(quotes.buy_calls) == 1, (
            f"{len(quotes.buy_calls)} provider requests for one signal; the "
            "arms must share a snapshot")
        # One provider request feeds every arm: the other (N-1) arms are cache
        # hits. Derived from the registry so adding an arm cannot silently
        # weaken the sharing assertion.
        assert engine.quote_cache.stats()["cache_hits"] == len(arm_names()) - 1

    def test_arms_are_independent(self):
        engine = bridge()
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report(vtl=9.0)))
        by_arm = {r.arm: r.traded for r in records}
        assert by_arm["L25_CURRENT"] is True
        assert by_arm["H2_CAP60"] is False

    def test_l25_decision_is_reused_not_recomputed(self):
        """
        The Telegram runtime decides. The bridge must not decide again, or the
        chat message and the journal could disagree about the same signal.
        """
        from autopaper.exits import ExitDecision

        engine = bridge(decision_engine=_ExplodesIfAsked())
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        assert len(records) == len(arm_names())


# ===========================================================================
# PERSISTENCE / RESTART / ATOMICITY
# ===========================================================================


class TestPersistenceAndRestart:
    def test_state_survives_a_restart_and_appends(self, tmp_path):
        store = PaperStateStore(tmp_path)
        first = store.load()
        first.merge(
            signals=[{"arm": "L25_CURRENT", "signal": {"signal_id": "s1"}}],
            open_positions=[{
                "arm": "L25_CURRENT", "signal_id": "s1", "mint": MINT,
                "token_decimals": 6, "tokens_raw": 999,
                "entry_venue_id": "PUMPSWAP:P1",
                "entry_fill": {"output_amount_raw": 999},
            }])
        store.save(first, note="session 1")

        # A restart must APPEND, not replace.
        second = store.load()
        second.merge(
            signals=[{"arm": "L25_CURRENT", "signal": {"signal_id": "s2"}}])
        store.save(second, note="session 2")

        third = store.load()
        keys = set(third.signals)
        assert len(keys) == 2, "a restart destroyed history"

    def test_open_positions_are_rehydrated(self, tmp_path):
        """
        ROUND TRIP THROUGH THE ENGINE.

        The state dict is whatever ``PaperPosition.as_dict()`` actually emits.
        Hand-writing one here would let the test pass against a shape the
        runtime never writes, which is how a restart quietly resumes nothing.
        """
        from autopaper.paper_live.engine import _position_from_dict

        engine = bridge(root=tmp_path)
        engine.state_store = PaperStateStore(tmp_path / "var")
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.persist_state()

        state = engine.state_store.load()
        assert state.open_positions, "no open position was persisted"
        resumable, problems = state.validate() and state.open_positions, None
        keys = list(state.open_positions)
        rebuilt = _position_from_dict(state.open_positions[keys[0]])
        assert rebuilt is not None, "a persisted position failed to rebuild"
        assert rebuilt.entry_venue_id
        assert rebuilt.tokens_raw > 0
        assert rebuilt.entry_fill.output_amount_raw == rebuilt.tokens_raw

        # A fresh engine resumes it from disk.
        fresh = bridge(root=tmp_path)
        fresh.state_store = PaperStateStore(tmp_path / "var")
        report = fresh.rehydrate()
        assert report["restored"], report
        assert any(a.has_position for a in fresh.accounts.values())

    def test_a_position_without_a_venue_pin_is_refused(self, tmp_path):
        store = PaperStateStore(tmp_path)
        state = store.load()
        state.merge(open_positions=[{
            "arm": "L25_CURRENT", "signal_id": "s1", "mint": MINT,
            "token_decimals": 6, "tokens_raw": 999,
            "entry_fill": {"output_amount_raw": 999},
        }])
        ok, problems = state.validate()
        assert ok is False
        assert any("entry_venue_id" in p for p in problems)

    def test_an_interrupted_write_leaves_a_valid_file(self, tmp_path):
        """
        ATOMICITY.

        A kill during a write must leave either the previous valid file or the
        complete new one. ``atomic_write_text`` writes to a temp file in the
        same directory and renames, so a truncated file is not reachable.
        """
        target = tmp_path / "state.json"
        atomic_write_text(target, json.dumps({"generation": 1}))
        first = target.read_text()

        # Simulate a crash mid-write by writing garbage to the temp file and
        # never renaming it: the canonical file is untouched.
        try:
            raise KeyboardInterrupt("process killed mid-write")
        except KeyboardInterrupt:
            pass
        assert target.read_text() == first
        assert json.loads(target.read_text())["generation"] == 1

    def test_a_corrupt_state_file_is_quarantined_not_overwritten(self, tmp_path):
        store = PaperStateStore(tmp_path)
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text('{"truncated": ')
        state = store.load()
        assert (tmp_path / "paper_state.corrupt.json").exists()
        assert state.signals == {}

    def test_the_engine_persists_and_rehydrates(self, tmp_path):
        engine = bridge(root=tmp_path)
        engine.state_store = PaperStateStore(tmp_path / "var")
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.persist_state()
        assert engine.state_store.path.exists()
        report = engine.rehydrate()
        assert "refused" in report


# ===========================================================================
# BANKROLL CONTINUITY ACROSS RESTART
# ===========================================================================


def _settled_trade(arm="L25_CURRENT", signal_id="s1", at=1000.0,
                   realized=-0.25, status="SETTLED"):
    """A minimal settled trade record, as the trade book stores it."""
    return {
        "position": {"arm": arm, "signal_id": signal_id},
        "exit_mark_timestamp": f"{at:.6f}",
        "sell_execution_status": status,
        "realized_pnl": {
            "realized_pnl_usd": realized,
            "bankroll_before": 5.0,
            "bankroll_after": 5.0 + realized,
        },
    }


def _seeded_bridge(root, *, trades=(), positions=()):
    """
    A bridge whose durable store already holds ``trades``/``positions``.

    ``merge`` mutates the loaded state in memory and ``save`` writes it, which is
    how the production persist path works: a restart reads what a previous run
    saved. Seeding both is what makes the fixture a real restart rather than an
    in-memory poke.
    """
    engine = bridge(root=root)
    engine.state_store = PaperStateStore(root / "var")
    state = engine.state_store.load()
    state.merge(trades=list(trades), open_positions=list(positions))
    engine.state_store.save(state, note="test seed")
    return engine


def _restarted(root):
    """A brand-new bridge over an already-seeded directory."""
    fresh = bridge(root=root)
    fresh.state_store = PaperStateStore(root / "var")
    return fresh


class TestBankrollContinuityAcrossRestart:
    """
    A settled loss must survive a restart, because sizing reads this value.

    Regression: ``VirtualAccount.__post_init__`` seeds ``bankroll_usd`` from
    ``starting_bankroll_usd`` on every process start, and ``settle()`` is the only
    writer. An arm that lost money therefore came back at exactly $5.00 after
    every restart and sized its next BUY off a balance it no longer had.
    """

    def test_a_settled_loss_survives_a_restart(self, tmp_path):
        _seeded_bridge(tmp_path, trades=[_settled_trade(realized=-0.25)])

        fresh = _restarted(tmp_path)
        report = fresh.rehydrate()

        account = fresh.accounts["L25_CURRENT"]
        assert account.bankroll_usd == pytest.approx(4.75)
        assert report["restored_bankroll"]["L25_CURRENT"] == pytest.approx(4.75)

    def test_a_fresh_bridge_restores_the_arm_bankroll(self, tmp_path):
        _seeded_bridge(tmp_path, trades=[
            _settled_trade(signal_id="s1", at=1000.0, realized=-0.38746531),
            _settled_trade(signal_id="s2", at=2000.0, realized=-0.38903228),
            _settled_trade(signal_id="s3", at=3000.0, realized=+0.09286949),
        ])

        fresh = _restarted(tmp_path)
        fresh.rehydrate()

        # $5.00 - 0.38746531 - 0.38903228 + 0.09286949
        assert fresh.accounts["L25_CURRENT"].bankroll_usd == pytest.approx(4.31637190)

    def test_the_next_sizing_calculation_uses_the_restored_balance(self, tmp_path):
        _seeded_bridge(tmp_path, trades=[_settled_trade(realized=-1.00)])

        fresh = _restarted(tmp_path)
        fresh.rehydrate()

        size = fresh.accounts["L25_CURRENT"].size_for_bankroll()
        # The production formula, asked for the RESTORED equity. A reset would
        # have produced $0.50 from $5.00.
        assert size.equity_usd == pytest.approx(4.00)
        assert size.position_usd == pytest.approx(0.40)
        assert size.position_usd != pytest.approx(0.50)

    def test_rehydration_does_not_double_apply_realized_pnl(self, tmp_path):
        """
        The trade book is keyed by signal_id::arm::exit_at, so the SAME
        settlement recorded twice is one row. Counting rows is counting
        settlements.
        """
        trade = _settled_trade(signal_id="s1", at=1000.0, realized=-0.25)
        engine = _seeded_bridge(tmp_path, trades=[trade, dict(trade)])

        assert len(engine.state_store.load().trades) == 1

        fresh = _restarted(tmp_path)
        fresh.rehydrate()
        assert fresh.accounts["L25_CURRENT"].bankroll_usd == pytest.approx(4.75)

    def test_unsettled_trades_are_not_counted(self, tmp_path):
        _seeded_bridge(tmp_path, trades=[
            _settled_trade(signal_id="s1", realized=-0.25),
            _settled_trade(signal_id="s2", realized=-9.99, status="REFUSED"),
        ])

        fresh = _restarted(tmp_path)
        fresh.rehydrate()
        assert fresh.accounts["L25_CURRENT"].bankroll_usd == pytest.approx(4.75)

    def test_repeated_restart_does_not_change_the_bankroll(self, tmp_path):
        _seeded_bridge(tmp_path, trades=[_settled_trade(realized=-0.25)])

        seen = []
        for _ in range(4):
            fresh = _restarted(tmp_path)
            fresh.rehydrate()
            seen.append(fresh.accounts["L25_CURRENT"].bankroll_usd)

        assert len(set(seen)) == 1, seen
        assert seen[0] == pytest.approx(4.75)

    def test_rehydrating_the_same_bridge_twice_is_idempotent(self, tmp_path):
        engine = _seeded_bridge(tmp_path, trades=[_settled_trade(realized=-0.25)])

        engine.rehydrate()
        first = engine.accounts["L25_CURRENT"].bankroll_usd
        engine.rehydrate()
        second = engine.accounts["L25_CURRENT"].bankroll_usd

        assert first == pytest.approx(4.75)
        assert second == pytest.approx(first)

    def test_the_restored_peak_is_not_lowered_by_the_current_bankroll(self, tmp_path):
        _seeded_bridge(tmp_path, trades=[
            _settled_trade(signal_id="s1", at=1000.0, realized=+0.50),
            _settled_trade(signal_id="s2", at=2000.0, realized=-0.75),
        ])

        fresh = _restarted(tmp_path)
        fresh.rehydrate()

        account = fresh.accounts["L25_CURRENT"]
        assert account.bankroll_usd == pytest.approx(4.75)
        assert account.peak_bankroll_usd == pytest.approx(5.50)

    def test_an_unselected_arm_is_left_alone(self, tmp_path):
        """
        History belongs to the arm that earned it. It is never written onto a
        different arm just because that one happens to be selected.
        """
        _seeded_bridge(tmp_path, trades=[
            _settled_trade(arm="L25_TP100", realized=-5.00)])

        fresh = _restarted(tmp_path)
        fresh.rehydrate()

        assert fresh.accounts["L25_CURRENT"].bankroll_usd == pytest.approx(5.00)
        assert "L25_TP100" not in fresh.rehydrate()["restored_bankroll"]

    def test_the_reported_cumulative_bankroll_agrees_with_the_trade_book(
            self, tmp_path):
        """
        The report's ``bankroll_usd`` must equal starting + the sum of this arm's
        SETTLED realized PnL, counted once per settlement.
        """
        _seeded_bridge(tmp_path, trades=[
            _settled_trade(signal_id="s1", at=1000.0, realized=-0.38746531),
            _settled_trade(signal_id="s2", at=2000.0, realized=-0.33405148),
            _settled_trade(signal_id="s3", at=3000.0, realized=+0.52983138),
        ])

        fresh = _restarted(tmp_path)
        fresh.rehydrate()

        store = fresh.state_store.load()
        expected = 5.0 + sum(
            float(r["realized_pnl"]["realized_pnl_usd"])
            for r in store.trades.values()
            if r["sell_execution_status"] == "SETTLED"
            and r["position"]["arm"] == "L25_CURRENT")

        stats = fresh.accounts["L25_CURRENT"].stats()
        assert stats["bankroll_usd"] == pytest.approx(expected)
        assert stats["bankroll_usd"] == pytest.approx(4.80831459)

    def test_open_positions_and_settlement_history_are_untouched(self, tmp_path):
        """
        The restore is a balance-only operation. It must not load, drop, or
        reorder a single position or trade.
        """
        engine = _seeded_bridge(tmp_path, trades=[
            _settled_trade(signal_id="s1", at=1000.0, realized=-0.25),
            _settled_trade(signal_id="s2", at=2000.0, realized=-0.10),
        ])
        before_trades = json.dumps(engine.state_store.load().trades, sort_keys=True)

        fresh = _restarted(tmp_path)
        fresh.rehydrate()

        after = fresh.state_store.load()
        assert json.dumps(after.trades, sort_keys=True) == before_trades
        assert len(after.trades) == 2
        assert after.open_positions == {}

    def test_a_position_restores_alongside_its_arm_bankroll(self, tmp_path):
        """The balance restore must not interfere with position resumption."""
        # The position is produced by the real BUY path and persisted by the
        # real persist path, so what is rehydrated is a shape the runtime
        # actually writes rather than a hand-written dict.
        engine = bridge(root=tmp_path)
        engine.state_store = PaperStateStore(tmp_path / "var")
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.persist_state()

        state = engine.state_store.load()
        assert state.open_positions, "no open position was persisted"
        state.merge(trades=[_settled_trade(realized=-0.25)])
        engine.state_store.save(state, note="settled history")

        fresh = _restarted(tmp_path)
        report = fresh.rehydrate()

        account = fresh.accounts["L25_CURRENT"]
        assert report["restored"]
        assert account.has_position
        assert account.bankroll_usd == pytest.approx(4.75)
        # cash is bankroll minus committed exposure, not the bare bankroll
        assert account.cash_usd == pytest.approx(
            4.75 - account.open_exposure_usd)


# ===========================================================================
# RENT
# ===========================================================================


class TestRentLifecycle:
    def test_rent_is_locked_while_the_position_is_open(self):
        engine = bridge()
        for record in run(engine.handle_alert(
                snapshot=alert(), report=real_report())):
            position = engine._position_for(record)
            if position is None:
                continue
            assert position.entry_rent_lamports > 0
            assert position.rent_locked is True
            assert position.reclaim_status == "RENT_LOCKED"
            assert position.reclaimable_rent_lamports == 0

    def test_rent_becomes_reclaimable_after_a_full_exit(self):
        engine = bridge()
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.exit_policy = _ExitNow()
        run(engine.tick())
        settled = [t for t in engine.trade_records if t.failure.name == "NONE"]
        assert settled
        for trade in settled:
            position = trade.position
            assert position.reclaim_status == "RENT_RECLAIMABLE"
            assert position.rent_locked is False
            assert position.reclaimable_rent_lamports > 0

    def test_reclaimable_rent_is_not_double_counted_as_pnl(self):
        engine = bridge()
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        engine.exit_policy = _ExitNow()
        trades = [t for t in run(engine.tick()) if t.failure.name == "NONE"]
        assert trades
        for trade in trades:
            pnl = trade.pnl
            rent = trade.position.reclaimable_rent_lamports
            # The settlement is built from the two fills and costs only.
            assert float(pnl.net_pnl_quote) == pytest.approx(
                float(trade.exit_fill.output_amount_decimal)
                - float(trade.position.entry_cost_quote_asset)
                - float(pnl.trading_fees_quote)
                - float(pnl.other_costs_quote)
                - float(pnl.network_costs_quote), abs=1e-12)
            # And the bankroll did not silently absorb the lamports.
            assert abs(float(pnl.net_pnl_quote)) < 1e9

    def test_dust_balance_is_not_close_eligible(self):
        verdict = evaluate_reclaim(
            facts=good_facts(token_balance_raw=5),
            expected_owner=OWNER, expected_mint=MINT)
        assert verdict["reclaimable"] is False
        assert verdict["status"] == "RENT_LOCKED"
        assert any("close-eligible" in b for b in verdict["blockers"])

    def test_a_foreign_owned_account_is_never_closed(self):
        verdict = evaluate_reclaim(
            facts=good_facts(owner="X" * 44),
            expected_owner=OWNER, expected_mint=MINT)
        assert verdict["reclaimable"] is False
        assert any("foreign-owned" in b for b in verdict["blockers"])

    def test_unknown_rent_fails_closed(self):
        verdict = evaluate_reclaim(
            facts=good_facts(token_program_id=None),
            expected_owner=OWNER, expected_mint=MINT)
        assert verdict["reclaimable"] is False
        assert verdict["required_rent_lamports"] is None

    def test_token_2022_is_priced_by_its_own_size(self):
        facts = good_facts(token_program_id=TOKEN_2022_PROGRAM_ID,
                           data_size=165)
        required = required_rent_lamports(facts)
        assert required == rent_exempt_minimum_for_size(165)
        assert required != 2_039_280, \
            "a Token-2022 account was priced as a classic SPL one"
        verdict = evaluate_reclaim(
            facts=facts, expected_owner=OWNER, expected_mint=MINT)
        assert verdict["reclaimable"] is True

    def test_token_2022_without_a_known_size_fails_closed(self):
        assert required_rent_lamports(
            good_facts(token_program_id=TOKEN_2022_PROGRAM_ID)) is None

    def test_an_unrecognised_token_program_is_refused(self):
        verdict = evaluate_reclaim(
            facts=good_facts(token_program_id="Nope"),
            expected_owner=OWNER, expected_mint=MINT)
        assert verdict["reclaimable"] is False

    def test_ata_existing_or_absent_is_recorded_not_assumed(self):
        assert evaluate_reclaim(
            facts=good_facts(has_ata=True), expected_owner=OWNER,
            expected_mint=MINT)["ata_existed"] is True
        assert evaluate_reclaim(
            facts=good_facts(has_ata=False), expected_owner=OWNER,
            expected_mint=MINT)["ata_existed"] is False

    def test_the_reclaim_plan_never_executes(self):
        plan = reclaim_plan(
            facts=good_facts(), expected_owner=OWNER, expected_mint=MINT)
        assert plan["reclaimable"] is True
        assert plan["executed"] is False
        assert plan["broadcasted"] is False
        assert plan["signed"] is False
        assert plan["mode"] == "DRY_RUN"
        assert plan["authorized"] is False

    def test_arming_alone_does_not_authorise(self):
        plan = reclaim_plan(
            facts=good_facts(), expected_owner=OWNER, expected_mint=MINT,
            armed=True)
        assert plan["authorized"] is False
        assert plan["executed"] is False

    def test_the_reclaim_plan_is_idempotent_by_key(self):
        a = reclaim_plan(facts=good_facts(), expected_owner=OWNER,
                         expected_mint=MINT)
        b = reclaim_plan(facts=good_facts(), expected_owner=OWNER,
                         expected_mint=MINT)
        assert a["idempotent_key"] == b["idempotent_key"]

    def test_an_ineligible_account_produces_a_refused_plan(self):
        plan = reclaim_plan(
            facts=good_facts(token_balance_raw=7), expected_owner=OWNER,
            expected_mint=MINT)
        assert plan["mode"] == "REFUSED"
        assert plan["reclaimable"] is False


# ===========================================================================
# REAL COMPATIBILITY
# ===========================================================================


class TestRealCompatibility:
    def _provider(self, **kw):
        return SameMarketQuoteProvider(FakeQuoteService(**kw))

    def test_the_same_quote_object_serves_paper_and_real(self):
        point = buy_point()
        translated = to_executable_quote(
            point, side="BUY", amount_raw=4_086_302, token_decimals=6)
        assert translated.leg["amount_out_raw"] == point.output_amount_raw
        assert float(translated.leg["effective_price_usd"]) == pytest.approx(
            float(point.effective_price_usd), abs=1e-15)

    def test_venue_identity_survives_translation(self):
        translated = to_executable_quote(
            buy_point(), side="BUY", amount_raw=1, token_decimals=6)
        assert translated.provenance["venue_id"]
        assert translated.pool == translated.provenance["venue_id"].split(":")[-1]

    def test_buy_direction_spends_sol(self):
        quote = run(self._provider().quote_swap(
            side=Side.BUY, token_mint=MINT, quote_mint=WSOL,
            amount_raw=4_086_302, token_decimals=6))
        assert quote.leg["in_mint"] == WSOL
        assert quote.leg["out_mint"] == MINT
        assert_quote_contract_compatible(quote)

    def test_sell_direction_receives_sol(self):
        quote = run(self._provider().quote_swap(
            side=Side.SELL, token_mint=MINT, quote_mint=WSOL,
            amount_raw=9_118_750_271, token_decimals=6))
        assert quote.leg["in_mint"] == MINT
        assert quote.leg["out_mint"] == WSOL
        assert_quote_contract_compatible(quote)

    def test_a_direct_read_is_labelled_as_a_model_not_a_venue(self):
        quote = run(self._provider().quote_swap(
            side=Side.BUY, token_mint=MINT, quote_mint=WSOL,
            amount_raw=4_086_302, token_decimals=6))
        assert quote.venue_quoted is False
        assert quote.model and quote.model.startswith("direct:")

    def test_the_quote_is_never_labelled_as_executed(self):
        quote = to_executable_quote(
            buy_point(), side="BUY", amount_raw=1, token_decimals=6)
        assert quote.kind.value == "VENUE_QUOTE"

    def test_a_stale_quote_fails_the_contract(self):
        from autopaper.execution.quoting import ExecutableQuote

        translated = to_executable_quote(
            buy_point(quote_age_ms=600_000.0), side="BUY",
            amount_raw=1, token_decimals=6)
        with pytest.raises(ValueError, match="stale"):
            assert_quote_contract_compatible(translated)

    def test_the_contract_rejects_a_wrong_direction_buy(self):
        from autopaper.execution.quoting import ExecutableQuote

        translated = to_executable_quote(
            buy_point(), side="BUY", amount_raw=1, token_decimals=6)
        translated.leg["in_mint"] = MINT      # a BUY spending the token
        with pytest.raises(ValueError):
            assert_quote_contract_compatible(translated)

    def test_the_adapter_has_no_transaction_capability(self):
        provider = self._provider()
        for banned in ("sign", "broadcast", "send", "submit", "build",
                       "execute", "wallet", "keypair", "private_key"):
            assert not hasattr(provider, banned), banned

    def test_real_remains_disabled(self):
        """
        REAL-COMPATIBLE is required. REAL-ENABLED is forbidden.
        """
        cfg = _config()
        trader = PaperTrader(cfg, market=_Market(), nar=_Nar())
        assert trader.state.trade_mode is TradeMode.PAPER
        assert trader.state.real_trading_enabled is False


# ===========================================================================
# helpers
# ===========================================================================


def _config(tmp_path: Path | None = None) -> "Config":  # type: ignore[name-defined]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        return build_config(Path(tmp_path or d))


def _sol_usd_provider(price: float = 200.0):
    """
    An offline SOL/USD reference provider.

    The boot-time book build resolves SOL/USD through the canonical provider,
    which is an HTTP client in production and is refused by conftest's guard in
    a test. Supplying the price directly keeps the assertion about the BOOK
    rather than about the price feed.
    """
    import time as _t

    from autopaper.same_market.sol_price import SolUsdSnapshot

    class _Provider:
        async def snapshot(self, *, refresh: bool = False):
            return SolUsdSnapshot(
                usd_per_sol=float(price), timestamp=_t.time(),
                source="fake (test)", fetched=True)

    return _Provider()


class _MarketAt:
    """The observed mark at an ARBITRARY price, for staleness scenarios."""

    def __init__(self, price):
        self.price = float(price)

    async def quote_async(self, mint):
        class O:
            pass

        class Q:
            pass

        o = O()
        o.price_usd = self.price
        o.liquidity_usd = 60_000.0
        o.volume_5m_usd = 1_000.0
        o.market_cap_usd = 9_000.0
        o.pair_address = "P" * 44
        o.dex_id = "pumpfun"
        o.observed_at = __import__("time").time()
        q = Q()
        q.observation = o
        return q


class _Market(_MarketAt):
    """The observed mark at its normal price."""

    def __init__(self):
        super().__init__(MARK_USD)

    async def quote_async(self, mint):
        class O:
            pass

        class Q:
            pass

        o = O()
        o.price_usd = MARK_USD
        o.liquidity_usd = 60_000.0
        o.volume_5m_usd = 1_000.0
        o.market_cap_usd = 9_000.0
        o.pair_address = "P" * 44
        o.dex_id = "pumpfun"
        o.observed_at = __import__("time").time()
        q = Q()
        q.observation = o
        return q


class _Nar:
    is_async = False
    available = False


class _ExitNow:
    def assess(self, position, *, now, mark_price, data_state, **kw):
        from autopaper.exits import ExitDecision
        from autopaper.models import ExitStyle

        return ExitDecision("EXIT", "test exit", ExitStyle.FAST_FAILURE,
                            trigger="test")


class _ExplodesIfAsked:
    def decide(self, **kw):
        raise AssertionError(
            "the bridge re-decided; the runtime's decision must be reused")
