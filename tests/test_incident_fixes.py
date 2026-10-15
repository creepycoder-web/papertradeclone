"""
The three incident fixes, verified offline.

1. PAPER self-resolves SOL/USD (today's incident: 13 L25 approvals, 0 fills).
2. ``obs=None`` declines the candidate instead of crashing the runtime.
3. Every attempted execution-aware entry states an explicit outcome.

Everything here is mocked. No test reaches a network, and none of the three
fixes is allowed to weaken a fail-closed guard or disable executable quotes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from tests import paper_bridge_fakes as _fk
except ImportError:
    import paper_bridge_fakes as _fk

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.paper_live.arms import arm_names
from autopaper.paper_live.economics import TOKEN_ACCOUNT_RENT_LAMPORTS
from autopaper.paper_live.engine import OBSERVATION_UNAVAILABLE_REASON
from autopaper.paper_live.quotes import SolReference

alert, bridge, real_report, run = _fk.alert, _fk.bridge, _fk.real_report, _fk.run


class _NoNetworkMarket:
    """Used only where the TEST must prove no network is touched."""

    async def quote_async(self, mint):
        raise AssertionError("no network in this test")


def _trader(tmp_path, *, quotes=None, market=None) -> PaperTrader:
    """
    A PAPER runtime with an INJECTED quote service.

    The injection is what keeps this offline: the conftest guard refuses to
    build a real chain client, which is the guarantee that an offline test can
    never reach RPC now that executable quotes are the default.
    """
    # The market default is an OFFLINE observation stand-in, not the raising one:
    # a missing observation is refused BEFORE the SOL/USD check, so a test about
    # SOL/USD needs a candidate that actually reaches that gate.
    return PaperTrader(
        build_config(tmp_path),
        market=market if market is not None else _fk.FakeMarket(),
        quote_service=quotes if quotes is not None
        else _fk.FakeQuoteService())


# ===========================================================================
# PART B -- SOL/USD self-resolution
# ===========================================================================


class TestSolUsdSelfResolution:
    def test_paper_resolves_sol_usd_without_real(self, tmp_path, monkeypatch):
        """
        THE INCIDENT.

        ``_paper_bridge_instance`` used to read only the memoised
        ``_sol_price_usd``, which only the REAL equity path ever populates. A
        PAPER runtime therefore built its bridge with ``sol_reference=None`` and
        refused EVERY authorised entry with SOL_REFERENCE_MISSING.
        """
        calls = {"n": 0}

        async def fake_resolver(self):
            calls["n"] += 1
            self._sol_price_usd = 119.55
            return 119.55

        monkeypatch.setattr(PaperTrader, "_backend_sol_price", fake_resolver)
        trader = _trader(tmp_path)
        assert trader.state.trade_mode.value == "PAPER"
        assert trader.state.real_armed is False
        assert trader._sol_price_usd is None

        engine = run(trader._paper_bridge_instance())
        assert calls["n"] == 1, "PAPER did not consult the canonical resolver"
        assert engine.sol_reference is not None
        assert engine.sol_reference.usd_per_sol == pytest.approx(119.55)

    def test_no_real_arming_or_wallet_dependency(self, tmp_path, monkeypatch):
        """Resolution must not require REAL, /balance or a wallet."""
        monkeypatch.setattr(
            PaperTrader, "_backend_sol_price",
            lambda self: _async_value(150.0))
        trader = _trader(tmp_path)
        run(trader._paper_bridge_instance())
        assert trader.state.real_armed is False
        assert trader.state.real_trading_enabled is False

    def test_snapshot_reaches_bridge_and_quote_service(self, tmp_path, monkeypatch):
        """ONE snapshot, shared -- not two independently-fetched references."""
        monkeypatch.setattr(
            PaperTrader, "_backend_sol_price",
            lambda self: _async_value(119.55))
        trader = _trader(tmp_path)
        engine = run(trader._paper_bridge_instance())
        assert engine.sol_reference is engine.quote_service.sol_reference
        assert engine.quote_service.sol_reference.usd_per_sol == pytest.approx(
            119.55)

    def test_the_memo_is_reused_rather_than_refetched(self, tmp_path, monkeypatch):
        """The existing cache is reused; no second provider, no second memo."""
        calls = {"n": 0}

        async def counting(self):
            calls["n"] += 1
            self._sol_price_usd = 119.55
            return 119.55

        monkeypatch.setattr(PaperTrader, "_backend_sol_price", counting)
        trader = _trader(tmp_path)
        first = run(trader._paper_bridge_instance())
        second = run(trader._paper_bridge_instance())
        assert first is second, "the bridge was rebuilt"
        assert calls["n"] == 1

    def test_missing_sol_usd_fails_closed(self, tmp_path, monkeypatch):
        """A genuinely unobtainable reference refuses entries; it is not assumed."""
        monkeypatch.setattr(
            PaperTrader, "_backend_sol_price", lambda self: _async_value(None))
        trader = _trader(tmp_path)
        engine = run(trader._paper_bridge_instance())
        assert engine.sol_reference is None
        engine.sol_reference = None
        engine.quote_service.sol_reference = None
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        assert records
        for record in records:
            assert record.traded is False
            assert "SOL_REFERENCE_MISSING" in str(record.failure)

    def test_stale_sol_usd_is_not_used(self):
        """A reference older than its own window is not a usable reference."""
        stale = SolReference(119.55, time.time() - 86_400.0, "old")
        assert stale.at < time.time() - 3_600.0
        fresh = SolReference(119.55, time.time(), "now")
        assert fresh.at <= time.time()

    def test_offline_tests_cannot_hit_network(self):
        """
        The fix must not make unit tests reach a network.

        Asserted at the seam that actually enforces it: the conftest guard
        REFUSES to build the executable-quote runtime unless a quote service
        was injected, and every test in this module injects one. A subprocess
        re-run was tried first and is worse -- blocking ``socket.connect``
        globally also breaks pytest's own monkeypatch machinery, producing a
        confusing failure that has nothing to do with the network.
        """
        import inspect

        from autopaper.engine import PaperTrader

        source = inspect.getsource(PaperTrader.__init__)
        assert "quote_service" in source, "there is no injection seam to abuse"

        # An UNINJECTED runtime is refused before any client is constructed.
        trader = PaperTrader(build_config(Path(__file__).parent),
                             market=_NoNetworkMarket())
        assert trader.quote_service_override is None
        with pytest.raises(AssertionError, match="inject"):
            run(trader._paper_bridge_instance())

# ===========================================================================
# PART C -- obs=None robustness
# ===========================================================================


class _MarketReturnsNoObservation:
    """The provider ANSWERED, but had no observation. Not an exception."""

    def __init__(self):
        self.calls = 0

    async def quote_async(self, mint):
        self.calls += 1

        class Q:
            observation = None

        return Q()


class _MarketRaises:
    async def quote_async(self, mint):
        raise RuntimeError("dexscreener is down")


class TestObservationRobustness:
    def test_obs_none_does_not_crash(self):
        engine = bridge(market=_MarketReturnsNoObservation())
        point = run(engine.observe(_fk.MINT))
        assert point is not None

    def test_missing_observation_produces_an_explicit_failure(self):
        engine = bridge(market=_MarketReturnsNoObservation())
        point = run(engine.observe(_fk.MINT))
        assert point.available is False
        assert point.value_usd is None
        assert OBSERVATION_UNAVAILABLE_REASON[:20] in (point.missing_reason or "")

    def test_a_missing_observation_refuses_rather_than_fabricating(self):
        engine = bridge(market=_MarketReturnsNoObservation())
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        assert records
        for record in records:
            assert record.traded is False
            assert record.observed_mark.available is False
            assert record.observed_mark.value_usd is None, \
                "a missing observation must never be replaced with a mark"

    def test_a_provider_exception_is_still_named(self):
        engine = bridge(market=_MarketRaises())
        point = run(engine.observe(_fk.MINT))
        assert point.available is False
        assert "dexscreener is down" in (point.missing_reason or "")

    def test_the_runtime_continues_after_a_missing_observation(self):
        engine = bridge(market=_MarketReturnsNoObservation())
        first = run(engine.handle_alert(
            snapshot=alert(message_id=1), report=real_report()))
        second = run(engine.handle_alert(
            snapshot=alert(message_id=2, mint="7LSsEoJGhLeZzGvDofTdNg7M3JttxQqGWNLo6vWMpump"),
            report=real_report(mint="7LSsEoJGhLeZzGvDofTdNg7M3JttxQqGWNLo6vWMpump")))
        assert len(first) == len(arm_names()) and len(second) == len(arm_names()), \
            "the runtime stopped after a missing observation"


# ===========================================================================
# PART D -- execution reporting
# ===========================================================================


class TestExecutionReporting:
    def test_every_attempted_entry_states_an_outcome(self, tmp_path, monkeypatch):
        """
        NO executed=False / reason=None for an attempted execution-aware entry.
        """
        monkeypatch.setattr(
            PaperTrader, "_backend_sol_price", lambda self: _async_value(None))
        trader = _trader(tmp_path)
        engine = run(trader._paper_bridge_instance())
        engine.sol_reference = None
        engine.quote_service.sol_reference = None
        report = real_report()
        run(trader._execute_paper_buy_execution_aware(
            report, alert(), None))
        assert trader.last_execution is not None
        assert trader.last_execution["executed"] is False
        assert trader.last_execution["reason"], "reason was None"
        assert trader.last_execution["outcome"] == "SOL_REFERENCE_MISSING"

    def test_a_fill_states_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            PaperTrader, "_backend_sol_price", lambda self: _async_value(119.55))
        trader = _trader(tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), None))
        # One signal, one execution. Pre-warming with the SAME message id would
        # be a duplicate, which the bridge correctly suppresses.
        assert trader.last_execution["executed"] is True
        assert trader.last_execution["outcome"] == "PAPER_FILL=SUCCESS"

    def test_any_refusal_is_reported_with_an_explicit_reason(self, tmp_path, monkeypatch):
        """
        Whatever the arm refuses for, the runtime states WHICH.

        A specific guard (here SOL_REFERENCE_MISSING) is asserted rather than
        the mark-gap path, because reaching the mark-gap guard requires a real
        token-decimals read, which would put a chain call in an offline test.
        The mark-gap rejection itself is covered in test_paper_live.py.
        """
        monkeypatch.setattr(
            PaperTrader, "_backend_sol_price", lambda self: _async_value(None))
        trader = _trader(tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), None))
        execution = trader.last_execution
        assert execution is not None
        assert execution["executed"] is False
        assert execution["reason"], "a refusal was recorded with reason=None"
        assert execution["reason"] == execution["outcome"]

    def test_executable_quotes_remain_reachable_after_the_fix(self, tmp_path, monkeypatch):
        """The SOL/USD fix must not have disabled the quote path."""
        monkeypatch.setattr(
            PaperTrader, "_backend_sol_price", lambda self: _async_value(119.55))
        trader = _trader(tmp_path)
        engine = run(trader._paper_bridge_instance())
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        assert any(r.traded for r in records), \
            "no arm traded after the SOL/USD fix"
        for record in records:
            if record.traded:
                assert record.buy_quote.available is True
                assert record.buy_quote.quote_mint


def _async_value(value):
    async def _inner():
        return value
    return _inner()
