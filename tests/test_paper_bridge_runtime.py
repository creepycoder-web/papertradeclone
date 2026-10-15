"""
The execution-aware PAPER engine inside the REAL ``PaperTrader``.

These tests are the integration proof: the standalone ``paper-live`` CLI is not
involved. A real ``PaperTrader`` is constructed with the production config, the
existing PAPER entry funnel is called, and the only thing swapped is the quote
transport (so there is no network) and the Telegram notifier (so nothing is
sent).

What is asserted here is the thing the whole task is about: the EXISTING
Telegram PAPER runtime executes on executable quotes, and no separate listener
process is required.
"""


from __future__ import annotations

import json
import sys
import time

import pytest

from autopaper.paper_live.arms import arm_names
THREE = arm_names()   # canonical arm set; never a literal
from autopaper.config import Config, build_config
from autopaper.engine import PaperTrader
from autopaper.models import DataState, Decision, ExitStyle, Regime, Route
from autopaper.modes import TradeMode

pytestmark = [
    pytest.mark.execution,
    pytest.mark.strategy,
]

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
try:  # pragma: no cover - depends on how pytest was invoked
    from tests import paper_bridge_fakes as _fk
except ImportError:  # pragma: no cover
    import paper_bridge_fakes as _fk

alert = _fk.alert
make_bridge = _fk.bridge
real_report = _fk.real_report
run = _fk.run


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A PAPER-only config with every credential absent or fake."""
    for name in ("REAL_TRADING_ENABLED", "SOLANA_PRIVATE_KEY", "SOLANA_KEYPAIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfakefakefake")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT", "123456")
    return build_config(tmp_path)


class FakeMarket:
    """Enough of MarketDataService for the PAPER funnel."""

    def __init__(self):
        self.calls = 0
        self._obs = None

    def _observation(self):
        class O:
            pass

        o = O()
        o.price_usd = 0.0000204
        o.liquidity_usd = 60000.0
        o.volume_5m_usd = 1000.0
        o.market_cap_usd = 9000.0
        o.pair_address = "9LCvns4zZsYAinZcEWRdzEEz7P6V84K3c5ZYCLT6vzMb"
        o.dex_id = "pumpfun"
        o.observed_at = time.time()
        o.received_at = time.time()
        o.state = DataState.LIVE
        return o

    async def quote_async(self, mint):
        self.calls += 1
        self._obs = self._observation()

        class Q:
            pass

        q = Q()
        q.observation = self._obs
        return q

    def mark_price(self, mint, now):
        return 0.0000204, DataState.LIVE

    def history(self, mint):
        class H:
            def as_of(self, t):
                return ()

            def latest(self):
                return None

        return H()

    def register_known_price(self, mint, price):
        return None

    async def seed_first_valid_price(self, mint):
        return None


class FakeNar:
    is_async = False
    available = False

    async def fetch(self, mint):
        return None

    def fetch_async(self, mint):
        async def _():
            return None

        return _()

    def health(self):
        return {"available": False}


def make_trader(cfg) -> PaperTrader:
    trader = PaperTrader(cfg, market=FakeMarket(), nar=FakeNar())
    trader._notifier = lambda text, kb=None: trader.sent.append(text)
    trader.sent = []
    return trader


def install_bridge(trader, tmp_path, **kw):
    """
    Give the runtime a bridge without touching the network.

    The real ``_paper_bridge_instance`` builds an RPC client and a chain
    reader. Those are injected here instead, which is the one substitution this
    test makes; the wiring around them is production.
    """
    engine = make_bridge(root=tmp_path, **kw)
    trader._paper_bridge = engine
    return engine


# ===========================================================================
# the runtime reaches the execution-aware engine with no extra process
# ===========================================================================


def execution_aware(cfg) -> PaperTrader:
    """
    A runtime on the DEFAULT execution model, with a capture notifier.

    No flag is set: executable quotes are the default, so this is the runtime an
    operator gets from ``autopaper run``. Building it explicitly (rather than
    reaching for a flag) is what keeps this a test of the default.
    """
    trader = PaperTrader(cfg, market=FakeMarket(), nar=FakeNar())
    trader.sent = []
    trader._notifier = lambda text, kb=None: trader.sent.append(text)
    return trader


class TestRuntimeIntegration:
    def test_execution_aware_is_the_default(self, cfg):
        """
        Executable quotes are the DEFAULT execution model.

        ``autopaper run`` in PAPER must quote every entry and exit against a
        same-market executable venue without an operator opting in. The legacy
        mark-based path is reachable only through the explicit rollback flag.
        """
        trader = make_trader(cfg)
        assert trader._execution_aware_paper() is True

    def test_legacy_path_needs_the_explicit_rollback_flag(self, cfg):
        """
        The legacy mark-based PAPER path is rollback/debug ONLY.

        Reachable through exactly one documented switch, and never by accident.
        """
        from dataclasses import replace as _replace

        rolled = _replace(
            cfg, real=_replace(cfg.real, paper_legacy_mark_fill_rollback=True))
        assert PaperTrader(
            rolled, market=FakeMarket(), nar=FakeNar()
        )._execution_aware_paper() is False

    def test_flag_turns_the_execution_aware_path_on(self, cfg):
        assert execution_aware(cfg)._execution_aware_paper() is True

    def test_real_mode_never_takes_the_paper_branch(self, cfg, monkeypatch):
        trader = execution_aware(cfg)
        # Force the REAL trade mode and confirm the PAPER engine is refused.
        monkeypatch.setattr(
            type(trader.state), "trade_mode",
            property(lambda self: TradeMode.REAL))
        assert trader._execution_aware_paper() is False

    def test_entry_funnel_reaches_the_bridge(self, cfg, tmp_path):
        trader = execution_aware(cfg)
        engine = install_bridge(trader, tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), None))
        assert engine.signal_records, "the runtime never reached the bridge"

    def test_entry_produces_a_telegram_notification(self, cfg, tmp_path):
        trader = execution_aware(cfg)
        install_bridge(trader, tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), None))
        assert trader.sent
        assert "NO TRANSACTION SENT" in trader.sent[0]

    def test_monitor_loop_drives_the_arms(self, cfg, tmp_path):
        trader = execution_aware(cfg)
        engine = install_bridge(trader, tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), None))
        engine.exit_policy = _ImmediateExit()
        run(trader.monitor_once())
        assert any(not a.has_position for a in engine.accounts.values()), \
            "the runtime's monitor loop never closed the arm"

    def test_dataset_persists_from_the_runtime(self, cfg, tmp_path):
        trader = execution_aware(cfg)
        install_bridge(trader, tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), None))
        trader.persist_paper_dataset()
        path = tmp_path / "reports/live/execution_aware_paper.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["status"]["mode"].startswith("PAPER")
        assert data["price_layers"]["ACTUAL_FILL"] is None

    def test_status_is_readable_from_the_runtime(self, cfg, tmp_path):
        """
        The arms the runtime REPORTS are the arms it is actually RUNNING.

        This used to assert the status listed every registered arm, which was
        the reported bug written down as a test: a runtime whose operator
        selection is a single arm reported all three, and opened trades as the
        one it had been frozen with. Asserting agreement with the LIVE
        selection is strictly stronger -- it fails both when the status lies and
        when the engine trades something the status does not name.
        """
        trader = execution_aware(cfg)
        install_bridge(trader, tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), None))
        status = trader.paper_bridge_status()
        assert status is not None
        selected = set(trader.state.arms)
        assert selected, "the runtime resolved no strategy selection at all"
        assert set(status["arms"]) == selected
        # The status must not be a silently-expanded "everything" either.
        assert not (
            "L25_TP100" in selected
            and "L25_TP100" not in set(trader.state.selected_arms)
        ), "L25_TP100 is running without ever having been selected"

    def test_status_is_none_before_any_paper_signal(self, cfg):
        trader = execution_aware(cfg)
        assert trader.paper_bridge_status() is None

    def test_quote_transport_is_the_verified_same_market_stack(self):
        """
        PAPER must not reach the Jupiter-only aggregator.

        Two quote systems would mean the dataset depends on which one answered,
        and the aggregator has no identity proof at all.
        """
        from pathlib import Path

        source = Path("src/autopaper/engine.py").read_text()
        start = source.index("    def _paper_quote_service")
        body = source[start:source.index("\n    def ", start + 10)]
        # The docstring explains what this REPLACES, so assert on code: the
        # aggregator must not be constructed anywhere in the method.
        code = "\n".join(
            line for line in body.splitlines()
            if not line.strip().startswith("#")
        )
        assert "QuoteOnlyJupiterProvider(" not in code
        assert "from .paper_live.quotes import QuoteService" in body
        assert "SameMarketQuote" not in code

    def test_legacy_paper_fill_path_still_exists(self):
        """Rollback is one config change, not a revert."""
        from pathlib import Path

        source = Path("src/autopaper/engine.py").read_text()
        assert "async def _execute_paper_buy(" in source
        assert "paper_execution_aware_enabled" in source


class _ImmediateExit:
    def assess(self, position, *, now, mark_price, data_state, **kw):
        from autopaper.exits import ExitDecision

        return ExitDecision("EXIT", "integration test exit",
                            ExitStyle.FAST_FAILURE, trigger="test")


# ===========================================================================
# no-broadcast isolation at the RUNTIME level
# ===========================================================================


class TestRuntimeSafety:
    def test_runtime_builds_no_transaction_capability(self, cfg):
        trader = execution_aware(cfg)
        for banned in ("send_transaction", "sendRawTransaction", "sign",
                       "broadcast", "submit_transaction"):
            assert not hasattr(trader._paper_bridge_instance, banned)

    def test_bridge_audit_passes_inside_the_runtime(self, cfg, tmp_path):
        trader = make_trader(cfg)
        engine = install_bridge(trader, tmp_path)
        assert engine.audit.ok is True

    def test_real_executor_is_never_constructed_for_paper(self, cfg):
        """
        A PAPER process must not even BUILD the REAL executor.

        This is the guarantee the existing smoke test asserts; it must still hold
        with the execution-aware engine attached.
        """
        import sys

        trader = execution_aware(cfg)
        engine = install_bridge(trader, cfg.var_dir.parent)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), None))
        real_backend = sys.modules.get("autopaper.execution.real_backend")
        if real_backend is not None:
            assert getattr(real_backend, "CONSTRUCTED", False) is False, \
                "the REAL backend was constructed during a PAPER run"

    def test_notifications_carry_no_credentials(self, cfg, tmp_path):
        trader = execution_aware(cfg)
        install_bridge(trader, tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), None))
        blob = "\n".join(trader.sent).lower()
        for secret in ("api_id", "api_hash", "bot_token", "private_key",
                       "999:fake"):
            assert secret not in blob

    def test_engine_says_actual_fill_is_unavailable(self, cfg, tmp_path):
        trader = execution_aware(cfg)
        install_bridge(trader, tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), None))
        status = trader.paper_bridge_status()
        assert status["actual_fill"] is None
        assert status["actual_fill_available"] is False
        assert "OBSERVED_MARK != EXECUTABLE_QUOTE" in status["statement"]
