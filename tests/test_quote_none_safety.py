"""Missing-quote (quote is None) safety for the PAPER exit paths.

The ONE invariant under test: a missing executable SELL quote never raises
AttributeError, never fabricates a fill, and never marks a position closed; the
position stays OPEN, the failure is classified, and the monitor continues with
the other positions.
"""

from __future__ import annotations

import time

import pytest

from autopaper.emergency_close import (
    EmergencyKillAll,
    PositionOutcome,
)
from autopaper.paper_live.arms import parse_arms
from autopaper.paper_live.fills import FillRefused, fill_from_quote
from autopaper.paper_live.schema import FailureReason

from tests.test_paper_live import (
    FakeQuoteService,
    make_engine,
    run,
    sell_point,
    signal,
    unavailable_point,
)


class _NoQuoteAtAll(FakeQuoteService):
    """Every SELL route answers None -- exactly the TITS failure shape."""

    async def quote_sell(self, **kw):
        return None

    async def quote_sell_pinned(self, **kw):
        return (None, None)

    async def resolve_graduated_exit_venue(self, **kw):
        return unavailable_point(
            side="SELL",
            missing_reason=(
                "graduation is proven but the PumpSwap pool is not executable: "
                "RPC error -32000, curve drained below minimum"),
        )


class _AvailableQuote(FakeQuoteService):
    """Every SELL route returns an executable quote so settlement succeeds."""

    def __init__(self):
        super().__init__(sell=sell_point())

    async def quote_sell(self, **kw):
        return self._restamp(self.sell, kw.get("mint", ""))

    async def quote_sell_pinned(self, **kw):
        return (self._restamp(self.sell, kw.get("mint", "")), None)

    async def resolve_graduated_exit_venue(self, **kw):
        return self._restamp(self.sell, kw.get("mint", ""))

    async def get_monitoring_identity(self, **kw):
        return None


class _OneFailingOneGood(FakeQuoteService):
    """Only ONE of the two arms' first SELL attempt fails with a raised
    exception; the other arm's quote succeeds immediately. Arm processing is
    deterministic (list order), so this isolates one position's failure while
    the other settles."""

    def __init__(self):
        super().__init__()
        self._failed_mint = None

    async def quote_sell(self, **kw):
        mint = kw.get("mint") or ""
        if self._failed_mint is None:
            self._failed_mint = mint
            raise RuntimeError("provider exploded: 503 routing")
        return self.sell

    async def quote_sell_pinned(self, **kw):
        mint = kw.get("mint") or ""
        if self._failed_mint is None:
            self._failed_mint = mint
            raise RuntimeError("provider exploded: 503 routing")
        return (self.sell, None)

    async def resolve_graduated_exit_venue(self, **kw):
        mint = kw.get("mint") or ""
        if self._failed_mint is None:
            self._failed_mint = mint
            raise RuntimeError("provider exploded: 503 routing")
        return self.sell

    async def get_monitoring_identity(self, **kw):
        return None

    async def quote_buy(self, **kw):
        return self._restamp(self.buy, kw["mint"])


class _FailingPinned(FakeQuoteService):
    """Every SELL route is unavailable for everyone."""

    async def quote_sell(self, **kw):
        return unavailable_point(
            side="SELL",
            missing_reason=(
                "no executable sell quote: bonding curve: the bonding curve is complete; "
                "pumpswap: RPC error -32000; jupiter: this client exposes no sell quote"))

    async def quote_sell_pinned(self, **kw):
        return (unavailable_point(
            side="SELL",
            missing_reason=(
                "no executable sell quote: bonding curve: the bonding curve is complete; "
                "pumpswap: RPC error -32000; jupiter: this client exposes no sell quote"),
        ), None)

    async def resolve_graduated_exit_venue(self, **kw):
        return unavailable_point(
            side="SELL", missing_reason="graduation proven; pumpswap read failed with RPC error -32000")


@pytest.mark.paper_live
class TestFillFromQuoteNoneSafety:
    def test_none_does_not_raise_attribute_error(self):
        with pytest.raises(FillRefused) as ei:
            fill_from_quote(None, side="SELL")
        # Structured failure with a concrete reason - NOT an AttributeError.
        assert ei.value.reason is FailureReason.EXIT_QUOTE_UNAVAILABLE
        assert "None" in ei.value.detail

    def test_none_buy_side_reason(self):
        with pytest.raises(FillRefused) as ei:
            fill_from_quote(None, side="BUY")
        assert ei.value.reason is FailureReason.ENTRY_REJECTED_QUOTE_UNAVAILABLE

    def test_available_quote_passes_the_none_guard(self):
        # The guard is for raise-on-None, not for availability classification.
        # An executable quote with a timestamp must pass the None check and
        # keep going toward the fill, not be classified as unavailable.
        quote = sell_point(quote_timestamp=time.time(), quote_age_ms=5.0)
        fill = fill_from_quote(quote, side="SELL")
        assert fill.side == "SELL"
        assert fill.effective_price == quote.effective_price
        assert fill.provider == quote.provider


@pytest.mark.paper_live
class TestNormalExitMissingQuote:
    def test_missing_quote_keeps_position_open(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]), quotes=_NoQuoteAtAll())
        run(engine.on_signal(signal()))
        trades = run(engine.monitor_once())
        assert len(trades) == 1
        assert trades[0].pnl is None
        assert trades[0].exit_fill is None
        assert trades[0].failure is FailureReason.EXIT_QUOTE_UNAVAILABLE
        assert trades[0].sell_quote is not None  # stub, not a bare None
        assert engine.accounts["L25_CURRENT"].has_position

    def test_monitor_survives_missing_quote(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]), quotes=_NoQuoteAtAll())
        run(engine.on_signal(signal()))
        trades1 = run(engine.monitor_once())
        trades2 = run(engine.monitor_once())
        assert len(trades1) == len(trades2) == 1
        assert all(t.pnl is None for t in trades1 + trades2)
        assert engine.accounts["L25_CURRENT"].has_position

    def test_one_position_failure_does_not_block_others(self):
        # L25_MAX_HOLD stands in for the retired L25_TP100 arm here: the test is
        # about ONE position's quote failure not blocking a SIBLING arm's exit,
        # which needs two arms on one signal and nothing more.
        e = make_engine(
            arms=parse_arms(["L25_CURRENT", "L25_MAX_HOLD"]),
            quotes=_OneFailingOneGood())
        run(e.on_signal(signal()))
        first = run(e.monitor_once())
        assert len(first) == 2
        assert any(t.pnl is not None for t in first)
        assert all(
            t.pnl is not None or t.failure_detail == "provider exploded: 503 routing"
            for t in first)

    def test_provider_error_preserved_as_concrete_reason(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]), quotes=_FailingPinned())
        run(engine.on_signal(signal()))
        trades = run(engine.monitor_once())
        assert trades[0].failure is FailureReason.EXIT_QUOTE_UNAVAILABLE
        assert "bonding curve is complete" in trades[0].failure_detail

    def test_identity_mismatch_still_fails_closed(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            quotes=FakeQuoteService(sell=sell_point(identity="FAIL")),
        )
        run(engine.on_signal(signal()))
        trades = run(engine.monitor_once())
        assert trades[0].pnl is None
        assert trades[0].exit_fill is None
        assert trades[0].failure in (
            FailureReason.EXIT_QUOTE_IDENTITY_UNKNOWN,
            FailureReason.EXIT_QUOTE_UNAVAILABLE,
        )
        assert engine.accounts["L25_CURRENT"].has_position

    def test_no_fake_fill_created(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]), quotes=_NoQuoteAtAll())
        run(engine.on_signal(signal()))
        trades = run(engine.monitor_once())
        assert trades[0].exit_fill is None
        assert trades[0].pnl is None
        assert trades[0].sell_execution_status == "NO_EXECUTABLE_SELL"

    def test_available_sell_quote_settles_normally(self):
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]), quotes=_AvailableQuote())
        run(engine.on_signal(signal()))
        trades = run(engine.monitor_once())
        assert len(trades) == 1
        trade = trades[0]
        assert trade.pnl is not None, trade.failure_detail
        assert trade.pnl.net_pnl_quote < 0
        assert trade.pnl.realized_pnl_usd < 0
        assert trade.exit_fill is not None
        assert trade.exit_fill.effective_price == trade.sell_quote.effective_price
        assert engine.accounts["L25_CURRENT"].has_position is False

    def test_real_sign_and_broadcast_are_not_invoked(self):
        # The PAPER harness is paper-only by construction; assert the audit
        # surface does not contain a signer/broadcaster, so no SELL attempt of
        # any kind reaches a network boundary.
        from autopaper.paper_live.guards import audit_paper_capability

        audit = audit_paper_capability()
        assert audit.ok


@pytest.mark.paper_live
class TestEmergencyCloseMissingQuote:
    def _killer(self, sell_quote):
        close_calls: list[dict] = []

        def _close_one(pos, **kw):
            close_calls.append(pos)
            raise AssertionError("close_one must never be called")

        killer = EmergencyKillAll(
            sell_quote=sell_quote,
            close_one=_close_one,
            list_open=lambda: [
                dict(position_id="p1", mint="MINT_A", tokens_raw=1000,
                     token_decimals=6, strategy_profile="L25_CURRENT",
                     entry_venue_id="VENUE_A", entry_pool_address="VENUE_A",
                     route=["pump_direct"])
            ],
            pause_entries=None, max_attempts=1, backoff_s=0.0,
            sleep=lambda _s: None, now=lambda: 1000.0, clock=lambda: 1000.0,
        )
        return killer, close_calls

    def test_emergency_close_none_quote_keeps_position_open(self):
        killer, close_calls = self._killer(lambda req: None)
        killer.request()
        op = killer._close_position(
            dict(position_id="p1", mint="MINT_A", tokens_raw=1000,
                 token_decimals=6, strategy_profile="L25_CURRENT",
                 entry_venue_id="VENUE_A", entry_pool_address="VENUE_A",
                 route=["pump_direct"]))
        assert close_calls == []
        assert op.outcome is PositionOutcome.NO_QUOTE
        assert not op.is_closed

    def test_emergency_close_unavailable_keeps_position_open(self):
        killer, close_calls = self._killer(
            lambda req: unavailable_point(side="SELL", missing_reason="curve complete"))
        killer.request()
        op = killer._close_position(
            dict(position_id="p1", mint="MINT_A", tokens_raw=1000,
                 token_decimals=6, strategy_profile="L25_CURRENT",
                 entry_venue_id="VENUE_A", entry_pool_address="VENUE_A",
                 route=["pump_direct"]))
        assert close_calls == []
        assert op.outcome is PositionOutcome.NO_QUOTE
        assert not op.is_closed
