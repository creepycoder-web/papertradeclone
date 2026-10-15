"""
Whole-runtime integration.

The unit suites each prove one property. These prove the properties that only
appear when the pieces are wired together, and the acceptance criteria from the
migration that span more than one module.
"""


from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from autopaper.config import build_config
from autopaper.journal import Journal, JournalError
from autopaper.modes import RunState, RuntimeState, TradeMode
from autopaper.real_account import RealAccount, RealFillRecord
from autopaper.sizing import compute_position_size
from autopaper.telegram import ui
from autopaper.telegram.actions import ButtonHandler, OperatorActions
from autopaper.v3c import V3C_HASH, V3C_MAX_MARKET_CAP_USD, V3C_MIN_SIGNAL_QUALITY

pytestmark = [
    pytest.mark.execution,
    pytest.mark.integrity,
    pytest.mark.strategy,
]

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# startup is safe
# ---------------------------------------------------------------------------


def test_a_built_config_reports_real_disabled(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    cfg = build_config(tmp_path)
    assert cfg.real.trading_enabled is False
    assert cfg.real.missing_capabilities(), "no wallet/RPC in a clean root"
    assert cfg.as_dict()["real"]["trading_enabled"] is False


def test_a_config_requesting_real_still_reports_disabled(tmp_path: Path,
                                                         monkeypatch):
    """
    The decisive test: a config file that says REAL_TRADING_ENABLED=true
    produces a process that cannot trade for real.
    """
    (tmp_path / ".env.local").write_text(
        "REAL_TRADING_ENABLED=true\n"
        "I_UNDERSTAND_REAL_TRADING_RISKS=YES\n"
        "WALLET_KEYPAIR_PATH=/tmp/id.json\n"
        "RPC_URL=https://rpc.example\n"
    )
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    cfg = build_config(tmp_path)
    assert cfg.real.env_trading_enabled_request is True
    assert cfg.real.risk_acknowledged is True
    assert cfg.real.wallet_configured is True
    # ...and the capability that matters is still False
    assert cfg.real.trading_enabled is False
    s = RuntimeState()
    s.start()
    assert s.trade_mode is TradeMode.PAPER
    assert not s.real_trading_enabled


def test_startup_does_not_arm_real(tmp_path: Path, monkeypatch):
    """The engine's startup asserts the invariant rather than assuming it."""
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    from autopaper.engine import PaperTrader

    cfg = build_config(tmp_path)
    trader = PaperTrader(cfg)
    trader.startup()
    assert trader.state.run_state is RunState.OFF
    assert trader.state.trade_mode is TradeMode.PAPER
    assert not trader.state.real_trading_enabled
    assert trader.running is False, "the loop does not start on its own"
    ok, why = trader.state.can_open_new_entries()
    assert not ok
    assert "press start" in why
    trader.shutdown_sync() if hasattr(trader, "shutdown_sync") else None


def test_a_second_process_is_refused(tmp_path: Path, monkeypatch):
    """
    Single-writer enforcement: a SECOND PROCESS fails closed.

    In-process re-acquisition is deliberately legal (an engine restart within
    one process is a new reference, not a new owner), so the refusal has to be
    proven across a process boundary or it proves nothing.
    """
    import subprocess
    import sys

    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    from autopaper.engine import PaperTrader

    cfg = build_config(tmp_path)
    first = PaperTrader(cfg)
    first.startup()
    assert first.owns_account is True

    # A separate interpreter pointed at the same account must be refused, and
    # must reach no alert / decision / sizing / exit path.
    code = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from autopaper.config import build_config\n"
        "from autopaper.engine import PaperTrader\n"
        f"cfg = build_config({str(tmp_path)!r})\n"
        "try:\n"
        "    PaperTrader(cfg).startup()\n"
        "    print('ACQUIRED')\n"
        "except Exception as exc:\n"
        "    print('REFUSED:' + type(exc).__name__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO,
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
    )
    assert "REFUSED:AccountAlreadyLocked" in out.stdout, out.stdout + out.stderr
    assert "ACQUIRED" not in out.stdout


# ---------------------------------------------------------------------------
# no automatic trade on start
# ---------------------------------------------------------------------------


def test_start_press_opens_no_position(tmp_path: Path, monkeypatch):
    """
    No startup -> immediate BUY. No backlog replay. No catch-up order.
    ``allow_backlog`` stays false, so an old alert cannot become a live order.
    """
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    from autopaper.engine import PaperTrader

    cfg = build_config(tmp_path)
    assert cfg.telegram.allow_backlog is False
    trader = PaperTrader(cfg)
    trader.startup()
    trader.state.start()
    assert trader.state.running
    assert trader.account.open_positions() == []
    assert trader.real_account.open_positions() == []
    assert trader.buys == 0
    assert trader.journal.all_trades() == []


def test_no_process_ever_holds_a_pending_order(tmp_path: Path, monkeypatch):
    """
    There is no order queue to drain. A refused order is recorded as an
    ``unfilled`` fact about the account, never as something to retry later.
    """
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    from autopaper.engine import PaperTrader

    cfg = build_config(tmp_path)
    trader = PaperTrader(cfg)
    trader.startup()
    trader.state.start()
    for attr in dir(trader):
        assert "pending_order" not in attr
        assert "order_queue" not in attr
        assert "catch_up" not in attr
        assert "backlog_replay" not in attr


# ---------------------------------------------------------------------------
# the operator action is the only path to REAL
# ---------------------------------------------------------------------------


async def _ok_pf():
    from tests.test_telegram_ui import _PF

    return _PF(True, "all checks passed")


def test_the_full_operator_arm_sequence(tmp_path: Path, monkeypatch):
    """OFF -> start -> REAL -> confirm -> preflight -> REAL_ARMED."""
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    journal = Journal(tmp_path / "j")
    s = RuntimeState()
    h = ButtonHandler(s, OperatorActions(preflight=_ok_pf, journal=journal))

    assert s.run_state is RunState.OFF
    asyncio.run(h.handle(ui.CB_START))
    assert s.run_state is RunState.PAPER

    r = asyncio.run(h.handle(ui.CB_REAL))
    assert not s.real_armed, "the REAL button alone arms nothing"
    token = r.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]

    asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert s.real_armed
    assert s.run_state is RunState.REAL_ARMED
    assert s.trade_mode is TradeMode.REAL

    # and every step is journaled
    actions = [e["event"] for e in journal.all_events()]
    assert "operator_start" in actions
    assert "operator_request_real" in actions
    assert "real_armed" in actions


def test_paper_disarms_and_the_old_token_is_dead():
    s = RuntimeState()
    h = ButtonHandler(s, OperatorActions(preflight=_ok_pf))
    asyncio.run(h.handle(ui.CB_START))
    r = asyncio.run(h.handle(ui.CB_REAL))
    token = r.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]
    asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert s.real_armed
    asyncio.run(h.handle(ui.CB_PAPER))
    assert not s.real_armed
    asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert not s.real_armed, "a pre-PAPER token cannot re-arm"


def test_emergency_stop_survives_a_new_runtime_state():
    """The latch is persisted and re-applied, so a restart cannot un-halt it."""
    s = RuntimeState()
    s.emergency_stop_now()
    fresh = RuntimeState()
    fresh.restore_emergency_stop(True)
    assert fresh.run_state is RunState.EMERGENCY_STOP
    ok, _ = fresh.can_open_new_entries()
    assert not ok


# ---------------------------------------------------------------------------
# account separation
# ---------------------------------------------------------------------------


def test_paper_and_real_accounts_cannot_see_each_other():
    from autopaper.account import PaperAccount
    from autopaper.ledger import PaperAccountState

    paper = PaperAccount(PaperAccountState(starting_bankroll_usd=1.55,
                                           cash_usd=1.55))
    real = RealAccount()
    for attr in ("wallet_snapshot", "open_exposure_usd", "realized_pnl_usd"):
        assert not hasattr(paper, attr), attr
    for attr in ("cash", "recently_exited", "unfilled_attempts"):
        assert not hasattr(real, attr), attr


def test_journal_partitions_paper_and_real(tmp_path):
    """
    Uses ``tmp_path`` rather than a hard-coded /tmp directory.

    A fixed shared path makes the counts depend on whatever a previous or
    interrupted run left behind, which showed up as an intermittent failure with
    no code change to explain it.
    """
    j = Journal(tmp_path / "journal")
    j.paper_entry(candidate_id="p1", mint="A", size_usd=1.0)
    j.paper_exit(candidate_id="p1", mint="A", pnl_usd=0.1)
    j.real_entry(signature="5SigA", candidate_id="r1", mint="B", size_usd=0.155)
    j.real_exit(signature="5SigB", candidate_id="r1", mint="B", pnl_usd=-0.01)

    s = j.summary()
    assert s["PAPER"]["entries"] == 1
    assert s["PAPER"]["exits"] == 1
    assert s["REAL"]["entries"] == 1
    assert s["REAL"]["exits"] == 1
    assert s["REAL"]["with_signature"] == 2
    assert j.verify_real_records_have_signatures() == []

    # the reads are partitioned, so a caller cannot accidentally sum across
    assert len(j.trades_for("PAPER")) == 2
    assert len(j.trades_for("REAL")) == 2
    assert {r["mode"] for r in j.trades_for("PAPER")} == {"PAPER"}
    assert {r["mode"] for r in j.trades_for("REAL")} == {"REAL"}


def test_a_real_record_without_a_signature_is_refused(tmp_path):
    j = Journal(tmp_path / "nosig")
    with pytest.raises(JournalError, match="signature"):
        j.real_entry(candidate_id="x", mint="B")
    with pytest.raises(JournalError, match="signature"):
        j.real_exit(candidate_id="x", mint="B")
    assert j.all_trades() == [], "nothing is written"


def test_a_real_position_needs_a_confirmed_fill():
    acct = RealAccount()
    unconfirmed = RealFillRecord(signature="5Sig", commitment="", side="ENTRY")
    with pytest.raises(ValueError, match="unconfirmed"):
        acct.register_entry(
            mint="M", symbol="T", fill=unconfirmed, token_amount=1,
            entry_price_usd=1.0, size_usd=0.155,
        )
    assert acct.open_positions() == []

    confirmed = RealFillRecord(
        signature="5Sig", commitment="confirmed", side="ENTRY",
        actual_output_amount=12_345,
    )
    pos = acct.register_entry(
        mint="M", symbol="T", fill=confirmed, token_amount=12_345,
        entry_price_usd=1.0, size_usd=0.155,
    )
    assert pos.is_open
    assert pos.confirmed
    assert acct.open_exposure_usd() == pytest.approx(0.155)


def test_a_stale_wallet_snapshot_refuses_to_produce_an_equity():
    """A real order sized off a stale balance is a guess with real money."""
    acct = RealAccount()
    assert acct.equity_usd() is None
    acct.observe_wallet(lamports=1_550_000_000, address="So1" + "x" * 42)
    assert acct.equity_usd() is None, "no SOL price yet"
    acct.observe_wallet(lamports=1_550_000_000, address="So1" + "x" * 42,
                        sol_price_usd=200.0, now=1_000.0)
    assert acct.equity_usd(now=1_000.0) == pytest.approx(310.0)
    assert acct.equity_usd(now=9_999.0) is None, "the snapshot is stale"


# ---------------------------------------------------------------------------
# the micro account end to end
# ---------------------------------------------------------------------------


def test_a_155_account_produces_a_155_order_and_nothing_else():
    """
    The whole sizing chain for the real account. Nothing enlarges it, and the
    venue is the only thing that may refuse it.
    """
    size = compute_position_size(1.55)
    assert size.position_usd == 0.155
    assert size.position_pct == 10.0
    # at $200/SOL that is 0.000775 SOL
    lamports = int(size.position_usd / 200.0 * 1_000_000_000)
    assert lamports == 775_000
    # and the executor's SOL guard demands that PLUS a reserve
    from autopaper.execution.real import MAX_OPEN_POSITIONS, RealExecutor

    class Cfg:
        wallet_keypair_path = None
        rpc_url = None
        slippage_bps = 500
        max_price_impact_pct = 5.0
        max_open_positions = MAX_OPEN_POSITIONS
        reentry_enabled = False
        priority_fee_lamports = 50_000

    reserve = RealExecutor(Cfg(), backend=object()).required_reserve_sol()
    assert reserve > 0
    assert lamports / 1e9 + reserve > lamports / 1e9


def test_the_155_order_is_never_enlarged_when_the_venue_refuses():
    """
    A venue that refuses $0.155 produces MICRO_REAL_EXECUTION_UNSUPPORTED and a
    refusal. There is no branch anywhere that raises the notional to a minimum.
    """
    from autopaper.execution.jupiter import MicroOrderUnsupported
    from autopaper.execution.real import LiveSolanaBackend, WSOL_MINT

    requested: list[int] = []

    class P:
        async def quote(self, i, o, amount, slip):
            requested.append(amount)
            raise MicroOrderUnsupported("venue rejected the notional as too small")

        async def build(self, *a, **k):
            raise AssertionError("must not build a transaction")

        async def health(self):
            return {"reachable": True}

        async def aclose(self):
            return None

    class R:
        async def simulate_transaction(self, b):
            raise AssertionError("must not simulate")

        async def send_transaction(self, b):
            raise AssertionError("MUST NOT BROADCAST")

        async def aclose(self):
            return None

    class W:
        pubkey = "So1" + "x" * 42
        masked = "So1…xxx"
        def sign_wire(self, raw):
            raise AssertionError("must not sign")

    b = LiveSolanaBackend(object(), wallet=W(), rpc=R(), provider=P())
    size = compute_position_size(1.55)
    lamports = int(size.position_usd / 200.0 * 1_000_000_000)
    fill = asyncio.run(
        b.execute_swap("ENTRY", WSOL_MINT, "MINT", lamports, 500)
    )
    assert requested == [lamports], "the amount must not change"
    assert fill.status == "MICRO_REAL_EXECUTION_UNSUPPORTED"
    assert not fill.filled
    assert fill.signature == ""


# ---------------------------------------------------------------------------
# the CLI and Telegram cannot disagree
# ---------------------------------------------------------------------------


def test_both_surfaces_render_from_the_same_view():
    """
    The CLI's `status` and the Telegram status message come from one view
    object, so a screenshot and a terminal cannot report different things.
    """
    import inspect

    from autopaper import cli

    src = inspect.getsource(cli.cmd_status)
    assert "render_status" in src, "the CLI renders the same message"
    assert "_offline_status_view" in src
    view_src = inspect.getsource(cli._offline_status_view)
    assert "StatusView(" in view_src, "the same view type as the Telegram UI"
    assert "restore_emergency_stop" in view_src, (
        "the CLI reads the same persisted emergency latch"
    )
    # and the Telegram side really does take a StatusView
    ui_src = inspect.getsource(ui.render_status)
    assert "view" in ui_src


def test_the_status_view_is_internally_consistent_for_both_accounts():
    for equity, expected_pct in ((1.55, 10.0), (50.0, 10.0),
                                 (50.01, 12.5), (100.0, 12.5)):
        s = RuntimeState()
        s.start()
        view = ui.StatusView(
            state=s, channel="Moo Deng President", chat="-1001",
            wallet_masked="So11…1112", balance_sol=0.0089, balance_usd=1.78,
            equity_usd=equity, cash_usd=equity, open_positions=0,
            executor_available=False, real_preflight_ok=None,
        )
        assert ui.consistency_check(view) == []
        text = ui.render_status(view)
        assert f"{expected_pct:g}%" in text
        assert f"${equity * expected_pct / 100.0:,.6f}" in text


def test_a_separated_stats_render_never_combines_the_accounts():
    paper = {"total_trades": 3, "wins": 2, "losses": 1, "win_rate": 0.667,
             "profit_factor": 2.0, "expectancy_usd": 0.01, "total_pnl_usd": 0.03,
             "max_drawdown_pct": -1.0, "avg_winner_usd": 0.02,
             "avg_loser_usd": -0.01, "runners": 0, "runner_capture_pct": 0.0}
    real = {"total_trades": 1, "wins": 0, "losses": 1, "win_rate": 0.0,
            "profit_factor": 0.0, "expectancy_usd": -0.01, "total_pnl_usd": -0.01,
            "max_drawdown_pct": -1.0, "avg_winner_usd": 0.0,
            "avg_loser_usd": -0.01, "runners": 0, "runner_capture_pct": 0.0}
    text = ui.render_stats(paper=paper, real=real)
    assert "Total: 3" in text and "Total: 1" in text
    # two separate PnL lines, no total of them
    assert text.count("Total PnL:") == 2
    assert "$+0.030000" in text and "$-0.010000" in text


# ---------------------------------------------------------------------------
# the migration audit, as a whole
# ---------------------------------------------------------------------------


def test_the_full_migration_audit_passes():
    from autopaper import audit

    r = audit()
    assert r.ok, r.render()
    names = {n for n, _, _ in r.checks}
    for required in (
        "no_forbidden_modules_loaded",
        "external_model_imports_zero",
        "model_calls_zero",
        "shadow_entries_zero",
        "shadow_capital_zero",
        "no_ml_symbols_in_package",
        "real_disabled_at_startup",
        "real_requires_operator",
        "real_arm_rejects_bare_call",
        "v3c_thresholds_frozen",
    ):
        assert required in names, required


def test_v3c_thresholds_are_the_migration_specification():
    assert V3C_MAX_MARKET_CAP_USD == 20_000.0
    assert V3C_MIN_SIGNAL_QUALITY == 0.55
    assert V3C_HASH[:8]


def test_no_removed_strategy_can_authorize_a_trade():
    """
    Structural, not a grep: the removed routes are gone from the enum, so no
    code in this process can name one.
    """
    from autopaper.models import HISTORICAL_ROUTES, Route

    assert {r.value for r in Route} == {"FAST_ENTRY", "NO_TRADE"}
    for name in HISTORICAL_ROUTES:
        assert not hasattr(Route, name)
    import autopaper.analysis.routes as routes
    for name in ("recovery_reclaim_conditions", "limits_min_size",
                 "classify_for_route"):
        assert not hasattr(routes, name)


def test_the_reference_repository_is_untouched_by_this_package():
    """
    AI-paper-trade stands alone. It must not import from, or depend on, the
    reference repository.
    """
    import ast
    import subprocess
    import sys

    # A FRESH interpreter, because this suite runs in one process alongside the
    # archived hybrid tests, which put the reference repository on sys.path.
    code = (
        "import sys, importlib; sys.path.insert(0, 'src')\n"
        "import autopaper.engine, autopaper.execution.real\n"
        "try:\n"
        "    importlib.import_module('bot'); print('LEAK')\n"
        "except ImportError:\n"
        "    print('CLEAN')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True
    )
    assert "CLEAN" in out.stdout, out.stdout + out.stderr


    for p in (REPO / "src").rglob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in ("bot", "memecoin_bot"), p.name
            elif isinstance(node, ast.ImportFrom):
                # A RELATIVE import (level > 0) is resolved inside this package
                # and cannot reach another repository, so only an ABSOLUTE
                # import can import the reference repo's `bot` package. This
                # distinction was previously ignored, which flagged legitimate
                # `from .botapi import ...` purely because of the word "bot".
                if node.level and node.level > 0:
                    continue
                root = (node.module or "").split(".")[0]
                assert root not in ("bot", "memecoin_bot"), p.name
        # the reference repo is named only in PROVENANCE prose, never imported
        text = p.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("#", "*", "-", '"', "'")) or "``" in stripped:
                continue
            assert "memecoin-bot" not in stripped, f"{p.name}: {stripped}"


def test_the_real_executor_was_ported_not_depended_on():
    """
    The executor is a port. Its module lives here, imports nothing from the
    reference repo, and carries its provenance in the docstring.
    """
    sol = (REPO / "src/autopaper/execution/solana.py").read_text()
    jup = (REPO / "src/autopaper/execution/jupiter.py").read_text()
    real = (REPO / "src/autopaper/execution/real.py").read_text()
    for text in (sol, jup, real):
        assert "PROVENANCE" in text
        assert "memecoin-bot" in text
    # the ported pieces are the proven ones
    assert "sign_wire" in sol, "wire-level v0/ALT signing"
    assert "lite-api.jup.ag" in jup or "DEFAULT_QUOTE_BASE" in jup
    assert "reconcile_transaction" in real
    assert "AMBIGUOUS" in real


def _pre_migration_ledgers(var: Path) -> list[Path]:
    """
    Every ledger file that can hold pre-migration rows.

    A documented account RESET ARCHIVES the pre-reset ledger under
    ``var/archive`` instead of truncating it, and the live ``ledger.jsonl``
    then continues in the new epoch. So the historical rows live in whichever
    of those files still carries them, and the preservation check has to span
    both -- reading only the live file would report a preserved history as
    lost the moment a reset happened.
    """
    out: list[Path] = []
    live = var / "ledger.jsonl"
    if live.exists():
        out.append(live)
    out.extend(sorted(var.glob("archive/*ledger*.jsonl")))
    out.extend(sorted(var.glob("archive/*/*ledger*.jsonl")))
    return out


def test_historical_records_are_preserved():
    """
    The pre-migration ledger, decisions and candidates files still exist and
    still parse. Nothing was rewritten.
    """
    var = REPO / "var"
    if not var.exists():
        pytest.skip("no runtime state present")
    ledgers = _pre_migration_ledgers(var)
    if not ledgers:
        pytest.skip("no pre-migration ledger present")
    names: set[str | None] = set()
    for ledger in ledgers:
        lines = [l for l in ledger.read_text().splitlines() if l.strip()]
        assert lines, f"the historical ledger {ledger} must not be emptied"
        for line in lines[:200]:
            json.loads(line)
        # and it still contains the OLD route names, which is correct: they are
        # historical facts, not live enum members
        names |= {json.loads(l).get("route") for l in lines}
    assert names & {"PRECISION_ENTRY", "CONFIRMATION_ENTRY", "RECOVERY_RECLAIM"}, (
        "historical rows must keep their original route strings"
    )
