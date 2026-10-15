"""Focused integration tests for the V2 FLOOR (BETA) wiring.

WHAT THESE PROVE (task section 8)
=================================
That the LIVE PAPER beta entry/exit path invokes the EXISTING executable-quote
collector, that the exact acquired quantity is reused at exit, that a missing
SELL quote can never settle off a mark, that completed beta trades reach the
EXISTING ``Closed Positions`` view and the ``Status`` statistics, and that the
persistence lifecycle is idempotent and restart-safe.

Everything is OFFLINE and uses isolated temporary state roots: no network, no
Telegram, no RPC, no real ``var/`` state, nothing signed or broadcast.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

#: pytest's rootdir conftest ends up putting ``tests/`` on sys.path at
#: collection time; depend on that same convention rather than rewriting it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataclasses import replace
from decimal import Decimal

from autopaper.engine import PaperTrader
from autopaper.paper_live.schema import QuotePoint
from autopaper.paper_live.v2_floor_beta import (
    BETA_SIZING_PCT,
    BETA_STARTING_BANKROLL_USD,
    EXECUTABLE_QUOTE_BASIS,
    SIZING_ALL_IN,
    SIZING_EXPLICIT,
    BetaExitFill,
    BetaExperiment,
    sizing_bucket,
)
from autopaper.telegram import ui

#: Shared offline doubles and beta helpers. These test modules are importable as
#: siblings because pytest puts ``tests/`` on ``sys.path`` at collection time.
from test_paper_live import (  # noqa: E402
    BUY_TOKENS_RAW,
    SOL_USD,
    FakeQuoteService,
    buy_point,
    run,
    sell_point,
    unavailable_point,
)
from test_v2_floor_beta import (  # noqa: E402
    T0,
    beta_engine,
    eligible_signal,
    new_exp,
    open_p1,
    quote_entry,
    quote_exit,
)

pytestmark = [pytest.mark.paper_live, pytest.mark.strategy]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bridge_with(experiment: BetaExperiment) -> SimpleNamespace:
    """A minimal stand-in for the runtime bridge, carrying only the experiment."""
    return SimpleNamespace(beta_experiment=experiment)


def _beta_closed_rows(experiment: BetaExperiment) -> list[dict]:
    """Call the engine's row builder without constructing the runtime engine."""
    return PaperTrader._beta_closed_rows(  # type: ignore[misc]
        SimpleNamespace(_paper_bridge=_bridge_with(experiment)))


def _beta_arm(experiment: BetaExperiment, pid: str = "p1", arm: str = "A"):
    return experiment.positions[pid][arm]


# ---------------------------------------------------------------------------
# 1-3. entries go through the executable BUY collector, shared across arms
# ---------------------------------------------------------------------------

def test_beta_entry_calls_the_existing_buy_quote_collector(tmp_path):
    qs = FakeQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    # The shared collector was actually invoked ...
    assert len(qs.buy_calls) >= 1
    # ... and the opened position carries its EXECUTABLE-QUOTE basis, not a mark.
    pid = next(iter(eng.beta_experiment.positions))
    pos = _beta_arm(eng.beta_experiment, pid, "A")
    assert pos.entry_basis == EXECUTABLE_QUOTE_BASIS
    # The acquired quantity is the one the accepted BUY quote reported.
    assert pos.entry_tokens_raw == int(qs.buy.output_amount_raw)
    assert pos.entry_cost_quote is not None and pos.entry_cost_quote > 0


def test_one_parent_signal_opens_abc_with_one_shared_buy_basis(tmp_path):
    qs = FakeQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    assert len(exp.positions) == 1
    arms = exp.positions[next(iter(exp.positions))]
    assert set(arms) == {"A", "B", "C"}
    fills = {(a.entry_fill_price, a.entry_cost_quote, a.entry_tokens_raw,
              a.entry_pool_or_curve) for a in arms.values()}
    assert len(fills) == 1, "the three arms must share ONE entry basis"
    # ONE buy quote, coalesced across the three arms.
    assert len(qs.buy_calls) == 1


# ---------------------------------------------------------------------------
# 4-5. exits reuse the exact quantity on the pinned market, with validation
# ---------------------------------------------------------------------------

def test_beta_exit_reuses_the_exact_entry_quantity(tmp_path):
    qs = FakeQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    pid = next(iter(eng.beta_experiment.positions))
    pos = _beta_arm(eng.beta_experiment, pid, "A")
    # The default fixture mark is far above every bracket, so the first monitor
    # tick triggers an exit for each arm.
    run(eng._monitor_beta_experiment())
    assert qs.sell_calls, "the SELL collector was never called"
    for call in qs.sell_calls:
        assert call["token_amount_raw"] == pos.entry_tokens_raw
        assert call["token_decimals"] == pos.entry_token_decimals
        # pinned to the market the ENTRY priced
        assert call["observed_pair_address"] == pos.entry_pool_or_curve


def test_beta_exit_identity_validation_is_enforced(tmp_path):
    # A SELL quote whose same-market identity is NOT PASS must be refused: the
    # exit stays unresolved rather than booking a fill on an unproven market.
    bad = sell_point(identity="UNKNOWN")
    eng = beta_engine(tmp_path, quotes=FakeQuoteService(sell=bad))
    run(eng.on_signal(eligible_signal()))
    run(eng._monitor_beta_experiment())
    pid = next(iter(eng.beta_experiment.positions))
    assert _beta_arm(eng.beta_experiment, pid, "A").status == "unresolved"


# ---------------------------------------------------------------------------
# 6-7. missing quote stays unresolved; a valid quote nets proceeds / cost - 1
# ---------------------------------------------------------------------------

def test_missing_sell_quote_never_settles_off_a_mark(tmp_path):
    qs = FakeQuoteService(sell=unavailable_point())
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    run(eng._monitor_beta_experiment())
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    for arm in "ABC":
        pos = _beta_arm(exp, pid, arm)
        assert pos.status == "unresolved"
        assert pos.exit_fill_price is None
        assert pos.net_pnl_usd is None
        # No fabricated P&L: the ledger did not move.
        assert exp.accounts[arm].bankroll_usd == BETA_STARTING_BANKROLL_USD
    assert exp.completed == []


def test_valid_sell_quote_uses_net_proceeds_over_total_buy_cost(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp, price=1.0, size=1.0)
    # observe the +30% target, then settle from an executable quote.
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    pos = exp.settle_arm("p1", "A", now=1_700_000_006.0,
                         exit_fill=quote_exit(exp, "p1", "A", price=1.30))
    assert pos.status == "closed"
    assert pos.exit_reason == "TP"
    # net = net SELL proceeds / total BUY cost - 1
    expected = pos.exit_cost_quote / pos.entry_cost_quote - 1.0
    assert pos.net_return_pct == pytest.approx(expected * 100.0)
    assert pos.net_pnl_usd == pytest.approx(1.0 * expected)


# ---------------------------------------------------------------------------
# 8. per-arm brackets and deadlines are independent
# ---------------------------------------------------------------------------

def test_each_arm_follows_its_own_bracket_and_deadline(tmp_path):
    from autopaper.paper_live.v2_floor_beta import arm_config
    # -- brackets: A's +30% target vs B/C's +50% --
    (tmp_path / "brackets").mkdir()
    exp = new_exp(tmp_path / "brackets")
    open_p1(exp)
    for arm in "ABC":
        exp.observe("p1", arm, mark_price=1.35, at=1_700_000_004.0)
    assert exp._exit_decision(_beta_arm(exp, "p1", "A"),
                              now=1_700_000_005.0)[0] == "TP"
    assert exp._exit_decision(_beta_arm(exp, "p1", "B"),
                              now=1_700_000_005.0)[0] == ""
    # -- deadlines: A/B at 90s, C at 120s --
    (tmp_path / "deadlines").mkdir()
    exp_dl = new_exp(tmp_path / "deadlines")
    open_p1(exp_dl)
    for arm in "ABC":
        exp_dl.observe("p1", arm, mark_price=1.0, at=1_700_000_004.0)
    assert arm_config("A").max_hold_s == 90.0
    assert arm_config("B").max_hold_s == 90.0
    assert arm_config("C").max_hold_s == 120.0
    # 91s: A and B are past their 90s cap; C (120s) is not yet due.
    assert exp_dl._exit_decision(_beta_arm(exp_dl, "p1", "A"),
                                 now=1_700_000_091.0)[0] == "TIME"
    assert exp_dl._exit_decision(_beta_arm(exp_dl, "p1", "C"),
                                 now=1_700_000_091.0)[0] == ""


# ---------------------------------------------------------------------------
# 9-10. failed quotes make accurate unresolved records, never fabricated fills
# ---------------------------------------------------------------------------

def test_a_failed_exit_is_an_unresolved_record_not_a_fill(tmp_path):
    qs = FakeQuoteService(sell=unavailable_point())
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    run(eng._monitor_beta_experiment())
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    # NOTHING was booked as a closed trade ...
    assert exp.completed == []
    # ... and the attempt is recorded with its reason.
    pos = _beta_arm(exp, pid, "A")
    assert pos.exit_blocked_reason is not None
    assert pos.exit_attempts >= 1
    # The stats view reports it as UNRESOLVED, excluded from closed results.
    stats = exp.arm_stats("A")
    assert stats["completed_trades"] == 0
    assert stats["unresolved"] == 1


# ---------------------------------------------------------------------------
# 11-13. Status and Closed Positions include beta, without double counting
# ---------------------------------------------------------------------------

def test_closed_positions_includes_a_completed_beta_trade(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp, price=1.0, size=1.0)
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    exp.settle_arm("p1", "A", now=1_700_000_006.0,
                   exit_fill=quote_exit(exp, "p1", "A", price=1.30))
    rows = _beta_closed_rows(exp)
    assert len(rows) == 1
    r = rows[0]
    assert r["strategy_profile"] == "V2 FLOOR (BETA)"
    assert r["parent_signal_id"] == "p1"
    assert r["arm"] == "A"
    assert r["exit_reason"] == "TP"
    assert r["realized_pnl_usd"] == pytest.approx(0.30, abs=1e-9)
    assert r["return_pct"] == pytest.approx(30.0, abs=1e-9)
    assert r["entry_tokens_raw"] == r["exit_tokens_raw"]
    # and it renders in the EXISTING closed-positions view.
    text = ui.render_closed_positions(rows, total=1)
    assert "CLOSED POSITIONS" in text
    assert "TP" in text


def test_open_and_unresolved_beta_positions_are_not_closed_rows(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp)
    # an OPEN arm ...
    assert _beta_closed_rows(exp) == []
    # ... and an arm that could not be quoted stays UNRESOLVED, still not closed.
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    # require_quote=True is the PRODUCTION contract: no quote, no settlement.
    exp.settle_arm("p1", "A", now=1_700_000_006.0, require_quote=True)
    assert _beta_arm(exp, "p1", "A").status == "unresolved"
    assert _beta_closed_rows(exp) == []


def test_status_renders_beta_without_mixing_into_the_legacy_totals():
    paper = {
        "total_trades": 3,
        "total_pnl_usd": 1.0,
        "execution_aware_paper": {
            "v2_floor_beta": {
                "session_id": "beta_test",
                "starting_bankroll_per_arm_usd": 10.0,
                "arms": {
                    "A": {
                        "take_profit_pct": 30.0, "stop_loss_pct": -30.0,
                        "max_hold_s": 90.0, "bankroll_usd": 10.3,
                        "realized_pnl_usd": 0.3, "unrealized_pnl_usd": 0.0,
                        "completed_trades": 1, "open_positions": 0,
                        "unresolved": 0, "win_rate_pct": 100.0,
                        "profit_factor": None,
                        "exits": {"TP": 1, "SL": 0, "TIME": 0},
                    },
                },
            },
        },
    }
    text = ui.render_stats(paper=paper, real={})
    assert "beta_test" in text
    assert "V2 FLOOR (BETA)" in text
    assert "Arm A" in text
    # The beta block is separate from the legacy PAPER totals line.
    assert text.count("PAPER") >= 1


# ---------------------------------------------------------------------------
# 14-15. restart-safe and idempotent persistence
# ---------------------------------------------------------------------------

def test_restart_restores_beta_records_without_duplicating(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp, price=1.0, size=1.0)
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    exp.settle_arm("p1", "A", now=1_700_000_006.0,
                   exit_fill=quote_exit(exp, "p1", "A", price=1.30))
    assert len(_beta_closed_rows(exp)) == 1
    # A restart reads the SAME durable state and must not double-book.
    revived = BetaExperiment(state_root=tmp_path)
    assert len(revived.completed) == 1
    rows = _beta_closed_rows(revived)
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "v2beta::A::p1"
    assert revived.accounts["A"].bankroll_usd == pytest.approx(10.30, abs=1e-9)


def test_duplicate_signal_delivery_does_not_duplicate_entries(tmp_path):
    qs = FakeQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    first = eligible_signal()
    run(eng.on_signal(first))
    run(eng.on_signal(first))
    run(eng.on_signal(eligible_signal()))  # same channel+mint+timestamp
    exp = eng.beta_experiment
    assert len(exp.positions) == 1
    assert sum(len(v) for v in exp.positions.values()) == 3
    assert exp.completed == []


# ---------------------------------------------------------------------------
# 16-17. scope: existing behaviour preserved; no live execution surface
# ---------------------------------------------------------------------------

def test_beta_module_has_no_signing_or_broadcast_surface():
    banned = {"sendTransaction", "signTransaction", "Keypair", "VersionedTransaction",
              "send_raw_transaction", "partialSign"}
    path = (Path(__file__).resolve().parents[1] / "src" / "autopaper"
            / "paper_live" / "v2_floor_beta.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                used.add(a.name)
                used.add(a.name.split(".")[0])
    assert not (used & banned), f"beta path exposes {used & banned}"


def test_exported_quantity_constant_is_the_quote_response(tmp_path):
    # Sanity lock: the fixture BUY acquires exactly BUY_TOKENS_RAW, which is what
    # the beta arms must carry. A silent change to the fixture would otherwise
    # make the quantity tests vacuous.
    assert BUY_TOKENS_RAW == 5_176_254_910


# ---------------------------------------------------------------------------
# 18-22. FAILURE RECOVERY: expiry, provider outage/failover, identity,
#        restart with open/unresolved arms, and retry-until-quoted
# ---------------------------------------------------------------------------

class _OutageSellQuotes(FakeQuoteService):
    """A provider whose SELL leg is DOWN (HTTP 429), and then recovers.

    Models the failover-exhausted case: every provider refused, so the collector
    has no executable SELL quote to hand back. The arm must stay unresolved --
    never priced from a mark -- and must still be retried on a later tick.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.outage = True

    async def quote_sell(self, **kw):
        self.sell_calls.append(kw)
        if self.outage:
            raise RuntimeError("HTTP 429 rate limited (all providers)")
        return self._restamp(self.sell, kw["mint"])


def test_expired_sell_quote_is_refused_and_never_becomes_a_fill(tmp_path):
    stale = sell_point(quote_age_ms=60_000.0, max_quote_age_ms=30_000.0)
    assert stale.stale is True, "fixture must actually be past its window"
    eng = beta_engine(tmp_path, quotes=FakeQuoteService(sell=stale))
    run(eng.on_signal(eligible_signal()))
    run(eng._monitor_beta_experiment())
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    pos = _beta_arm(exp, pid, "A")
    assert pos.status == "unresolved"
    assert pos.exit_fill_price is None and pos.net_pnl_usd is None
    assert exp.completed == []
    assert exp.accounts["A"].bankroll_usd == BETA_STARTING_BANKROLL_USD


def test_quote_outage_is_a_refusal_and_the_arm_recovers_on_a_later_tick(tmp_path):
    qs = _OutageSellQuotes()
    # The retry cooldown is driven to zero here so "a later tick" really is the
    # NEXT tick. That this is a deliberate override -- and not the production
    # policy -- is exactly what the cooldown tests in
    # ``tests/test_beta_exit_retry_cooldown.py`` pin down.
    eng = beta_engine(tmp_path, quotes=qs, exit_retry_base_s=0.0)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    pos = _beta_arm(exp, pid, "A")
    # 1) provider outage -> nothing booked, reason recorded, arm still outstanding
    run(eng._monitor_beta_experiment())
    assert pos.status == "unresolved"
    assert pos.exit_blocked_reason and pos.exit_attempts >= 1
    assert exp.completed == []
    # 2) the provider comes back -> the SAME arm settles from a real quote.
    qs.outage = False
    run(eng._monitor_beta_experiment())
    assert pos.status == "closed"
    assert pos.net_pnl_usd is not None
    assert len(exp.completed) == 3
    assert exp.accounts["A"].bankroll_usd != BETA_STARTING_BANKROLL_USD


def test_unproven_or_mismatched_market_identity_on_the_sell_leg_is_refused(tmp_path):
    for verdict in ("UNKNOWN", "MISMATCH"):
        qs = FakeQuoteService(sell=sell_point(identity=verdict))
        eng = beta_engine(tmp_path / verdict, quotes=qs)
        run(eng.on_signal(eligible_signal()))
        run(eng._monitor_beta_experiment())
        exp = eng.beta_experiment
        pid = next(iter(exp.positions))
        pos = _beta_arm(exp, pid, "A")
        assert pos.status == "unresolved", verdict
        assert pos.net_pnl_usd is None
        assert exp.completed == []


def test_restart_preserves_open_and_unresolved_arms(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp, price=1.0, size=1.0)
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    exp.settle_arm("p1", "A", now=1_700_000_006.0, require_quote=True)
    assert _beta_arm(exp, "p1", "A").status == "unresolved"

    revived = BetaExperiment(state_root=tmp_path)
    a = _beta_arm(revived, "p1", "A")
    # NOT silently dropped: the arm is still here, still unresolved, and it
    # still carries the evidence of the refused exit.
    assert a is not None and a.status == "unresolved"
    assert a.exit_attempts == 1
    assert a.exit_blocked_reason
    # ... and it is still NOT a closed trade.
    assert revived.completed == []
    assert _beta_closed_rows(revived) == []
    assert revived.accounts["A"].bankroll_usd == BETA_STARTING_BANKROLL_USD
    assert {_beta_arm(revived, "p1", x).status for x in ("B", "C")} == {"open"}


def test_an_unresolved_arm_settles_exactly_once_when_a_quote_arrives(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp, price=1.0, size=1.0)
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    exp.settle_arm("p1", "A", now=1_700_000_006.0, require_quote=True)
    pos = _beta_arm(exp, "p1", "A")
    # a real quote arrives later -> the arm settles ONCE and leaves unresolved.
    exp.settle_arm("p1", "A", now=1_700_000_007.0,
                   exit_fill=quote_exit(exp, "p1", "A", price=1.30),
                   require_quote=True)
    assert pos.status == "closed"
    assert len(exp.completed) == 1
    assert exp.accounts["A"].bankroll_usd == pytest.approx(10.30, abs=1e-9)
    # a restart must not resurrect it or re-book the same settlement.
    revived = BetaExperiment(state_root=tmp_path)
    assert len(revived.completed) == 1
    assert "A" not in revived.positions.get("p1", {})
    assert len(_beta_closed_rows(revived)) == 1


def test_flatten_records_an_unresolved_arm_instead_of_dropping_it(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp, price=1.0, size=1.0)
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    exp.settle_arm("p1", "A", now=1_700_000_006.0, require_quote=True)
    summary = exp.flatten_all(now=1_700_000_010.0)
    closed_arms = {r["arm"] for r in summary["closed"]}
    reported = {r["arm"] for r in summary["unresolved"]}
    # Every arm is ACCOUNTED FOR: closed, or explicitly reported as still open.
    assert closed_arms | reported == {"A", "B", "C"}
    # The unresolved arm was observed, so the flatten CAN close it -- and does,
    # rather than losing it.
    assert "A" in closed_arms


# ---------------------------------------------------------------------------
# 23-27. REPORTING: basis/session separation, no mixing, no double counting
# ---------------------------------------------------------------------------

def _mixed_basis_experiment(tmp_path):
    """One QUOTE-based settlement (A) and one MODELLED settlement (B), one session."""
    exp = new_exp(tmp_path)
    open_p1(exp, price=1.0, size=1.0)
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    exp.settle_arm("p1", "A", now=1_700_000_006.0,
                   exit_fill=quote_exit(exp, "p1", "A", price=1.30))
    exp.observe("p1", "B", mark_price=1.55, at=1_700_000_005.0)
    exp.settle_arm("p1", "B", now=1_700_000_006.0, require_quote=False)
    return exp


def test_provenance_separates_quote_based_from_historical_modelled(tmp_path):
    exp = _mixed_basis_experiment(tmp_path)
    prov = exp.stats()["provenance"]
    by_basis = prov["by_basis"]
    assert by_basis["quote"]["completed_trades"] == 1
    assert by_basis["modeled"]["completed_trades"] == 1
    assert prov["mixed"] is True
    quote_rec = next(r for r in exp.completed if r["arm"] == "A")
    model_rec = next(r for r in exp.completed if r["arm"] == "B")
    assert quote_rec["pnl_basis"].endswith("QUOTE_SIMULATED_FILL")
    assert model_rec["pnl_basis"].endswith("MODELED_BACKTEST")
    # Each bucket carries ONLY its own basis' realized dollars -- the historical
    # modelled P&L is never folded into the quote-based figure.
    assert by_basis["quote"]["realized_pnl_usd"] == pytest.approx(
        quote_rec["net_pnl_usd"], abs=1e-6)
    assert by_basis["modeled"]["realized_pnl_usd"] == pytest.approx(
        model_rec["net_pnl_usd"], abs=1e-6)
    assert by_basis["quote"]["realized_pnl_usd"] != by_basis["modeled"][
        "realized_pnl_usd"]
    # ... and both are attributed to the SESSION they happened in.
    assert prov["by_session"][exp.session_id]["completed_trades"] == 2


def test_arm_stats_split_realized_by_basis(tmp_path):
    exp = _mixed_basis_experiment(tmp_path)
    sa = exp.arm_stats("A")
    sb = exp.arm_stats("B")
    assert sa["realized_by_basis"]["quote"]["completed_trades"] == 1
    assert sa["realized_by_basis"]["modeled"]["completed_trades"] == 0
    assert sb["realized_by_basis"]["modeled"]["completed_trades"] == 1
    assert sb["realized_by_basis"]["quote"]["completed_trades"] == 0
    # the headline total is explicitly labelled as an all-basis number.
    assert "all-basis" in sa["realized_basis_note"]


def test_status_render_labels_the_modelled_history_apart_from_quote_results(tmp_path):
    exp = _mixed_basis_experiment(tmp_path)
    text = ui.render_stats(
        paper={"total_trades": 0,
               "execution_aware_paper": {"v2_floor_beta": exp.stats()}},
        real={})
    assert "quote-sim" in text
    assert "modelled-backtest" in text
    assert "NOT quote performance" in text
    assert "never summed" in text


def test_status_render_omits_the_mixed_basis_warning_when_only_one_basis(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp, price=1.0, size=1.0)
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    exp.settle_arm("p1", "A", now=1_700_000_006.0,
                   exit_fill=quote_exit(exp, "p1", "A", price=1.30))
    text = ui.render_stats(
        paper={"total_trades": 0,
               "execution_aware_paper": {"v2_floor_beta": exp.stats()}},
        real={})
    assert "quote-sim" in text
    assert "never summed" not in text


def test_beta_pnl_is_not_added_to_the_legacy_paper_total(tmp_path):
    exp = _mixed_basis_experiment(tmp_path)
    beta_total = sum(float(r["net_pnl_usd"]) for r in exp.completed)
    legacy_total = 1.0
    text = ui.render_stats(
        paper={"total_trades": 3, "total_pnl_usd": legacy_total,
               "execution_aware_paper": {"v2_floor_beta": exp.stats()}},
        real={})
    # The legacy block shows its OWN total, and it is NOT the sum of the two.
    assert f"Total PnL: <b>${legacy_total:+,.6f}</b>" in text
    assert f"${legacy_total + beta_total:+,.6f}" not in text


def test_stats_report_realized_unrealized_open_and_unresolved_separately(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp, price=1.0, size=1.0)
    # A: realized (quoted close).  B: open and priced.  C: open, never observed.
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    exp.settle_arm("p1", "A", now=1_700_000_006.0,
                   exit_fill=quote_exit(exp, "p1", "A", price=1.30))
    exp.observe("p1", "B", mark_price=1.20, at=1_700_000_005.0)
    sa, sb, sc = (exp.arm_stats(x) for x in "ABC")
    assert sa["completed_trades"] == 1 and sa["realized_pnl_usd"] > 0
    assert sa["open_positions"] == 0 and sa["unresolved"] == 0
    assert sb["realized_pnl_usd"] == 0.0 and sb["open_positions"] == 1
    assert sb["unrealized_priced_positions"] == 1
    assert sb["unrealized_pnl_usd"] != 0.0
    # NO OBSERVATION, NO PRICE: an unpriced open arm is not shown as $0 gain.
    assert sc["unrealized_unpriced_positions"] == 1
    assert sc["unrealized_pnl_usd"] == 0.0
    # no double counting: one closed settlement -> one closed row.
    assert len(_beta_closed_rows(exp)) == len(exp.completed) == 1


def test_unresolved_and_open_arms_are_excluded_from_closed_positions(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp, price=1.0, size=1.0)
    exp.observe("p1", "A", mark_price=1.30, at=1_700_000_005.0)
    exp.settle_arm("p1", "A", now=1_700_000_006.0, require_quote=True)
    assert _beta_arm(exp, "p1", "A").net_pnl_usd is None
    assert _beta_closed_rows(exp) == []
    stats = exp.stats()["provenance"]
    assert stats["by_basis"]["quote"]["completed_trades"] == 0
    assert stats["by_basis"]["modeled"]["completed_trades"] == 0


# ===========================================================================
# ALL-IN SIZING: 100% of each arm's OWN available capital
# ===========================================================================

def _sizing_buy_point(sol_lamports: int) -> QuotePoint:
    """A BUY quote that HONOURS the SOL amount it was asked for.

    The shared doubles answer with a fixed quote whatever size they are asked
    for, so they cannot show that an all-in notional was really spent -- a $10
    request and a $0.50 request produce the same amount. A venue prices the
    output from the input it is handed, so this one does too.
    """
    base = buy_point()
    decimals = int(base.output_decimals or 6)
    sol = Decimal(int(sol_lamports)).scaleb(-9)
    price_usd = float(base.effective_price_usd or 0.0)
    tokens = float(sol) * SOL_USD / price_usd
    tokens_raw = int(tokens * (10 ** decimals))
    price_quote = Decimal(str(price_usd)) / Decimal(str(SOL_USD))
    return replace(
        base,
        input_amount_raw=int(sol_lamports),
        input_amount_decimal=sol,
        output_amount_raw=tokens_raw,
        output_amount_decimal=Decimal(tokens_raw).scaleb(-decimals),
        effective_price=price_quote,
        effective_price_usd=price_usd,
        execution_native_reference=replace(
            base.execution_native_reference,
            reference_price=float(price_quote) / 1.02),
    )


class SizingQuoteService(FakeQuoteService):
    """A provider whose BUY output is priced from the requested SOL input."""

    async def quote_buy(self, **kw):
        self.buy_calls.append(kw)
        return self._restamp(_sizing_buy_point(int(kw["sol_lamports"])),
                             kw["mint"])


def _all_in_engine(tmp_path, **kw):
    """An engine whose quote provider honours the requested size."""
    return beta_engine(tmp_path, quotes=SizingQuoteService(**kw))


def _expected_outside_swap():
    """The outside-swap costs an ALL-IN arm must hold BESIDE its swap input.

    Built from the SAME module the engine uses, at the engine's own SOL/USD
    reference, so the assertions below restate the contract instead of
    hand-typing a second guess at it.
    """
    from autopaper.paper_live.execution_costs import outside_swap_costs
    return outside_swap_costs(sol_usd=SOL_USD, sol_usd_source="test")


def _exit_at_multiple(exp: BetaExperiment, pid: str, arm: str,
                      multiple: float):
    """A SELL quote whose SOL proceeds are ``multiple`` x the BUY's SOL cost.

    The experiment's return denominator is ``entry_cost_quote`` -- the SOL the
    venue actually took -- so an exit fixture has to be built in THAT unit. The
    synthetic ``quote_exit`` helper scales a USD price by a token count; against
    a SOL-denominated cost that books a ~SOL/USD windfall instead of the return
    under test, which is why this one exists.
    """
    bp = exp.positions[pid][arm]
    cost = float(bp.entry_cost_quote)
    return BetaExitFill(
        price_usd=float(bp.entry_fill_price) * float(multiple),
        tokens_raw=int(bp.entry_tokens_raw),
        token_decimals=int(bp.entry_token_decimals or 6),
        cost_quote=round(cost * float(multiple), 9),
        fee_quote=0.0, entry_cost_quote=cost,
        quote_timestamp=T0 + 5, identity="PASS",
    )


def test_every_new_entry_spends_the_whole_deployable_budget(tmp_path):
    """ALL-IN commits 100% of what the arm can DEPLOY, not of its cash.

    The arm's whole available cash is the SIZING decision. What actually goes
    into the swap input is that MINUS the costs that must be paid BESIDE the
    swap -- the Solana base fee, the priority fee and the token-account rent --
    because a wallet that has nothing left for those cannot execute at all.
    """
    qs = SizingQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    costs = _expected_outside_swap()
    assert costs.known and costs.total_reserve_usd > 0
    for arm in "ABC":
        pos = _beta_arm(exp, pid, arm)
        assert pos.sizing_rule == SIZING_ALL_IN
        assert pos.sizing_pct == BETA_SIZING_PCT == 100.0
        # 1) the SIZING decision is the arm's whole available cash ...
        assert pos.allocated_usd == pytest.approx(10.0, abs=1e-9)
        # 2) ... and the DEPLOYABLE part of it is the swap input, less only
        #    sub-lamport rounding dust (1 lamport ~= $1.2e-7 of SOL).
        assert pos.deployable_usd == pytest.approx(
            10.0 - costs.total_reserve_usd, abs=1e-6)
        assert pos.entry_cost_usd == pytest.approx(pos.deployable_usd, abs=1e-9)
        assert pos.entry_cost_usd <= pos.allocated_usd
        # 3) the reserve is RECORDED, not silently dropped: it is the fees +
        #    the rent, and it is what the arm still has but cannot deploy.
        assert pos.reserved_outside_swap_usd == pytest.approx(
            costs.total_reserve_usd, abs=1e-6)
        assert pos.remaining_cash_usd == pytest.approx(
            costs.total_reserve_usd, abs=1e-6)
        # the USD base every later P&L is measured against is the committed
        # cost, so a 10% return is 10% of what the arm actually spent.
        assert pos.position_size_usd == pos.entry_cost_usd
        # 4) ... and the arm's UNCOMMITTED cash is exactly that reserve: the
        #    capital is committed, but this part of it is not DEPLOYABLE.
        assert exp.available_cash_usd(arm) == pytest.approx(
            costs.total_reserve_usd, abs=1e-6)
        assert exp.committed_capital_usd(arm) == pytest.approx(
            pos.entry_cost_usd, abs=1e-9)
        # 5) the cost model is on the POSITION, with its provenance, so a
        #    reader can see the fees/rent split and that the rent is a deposit.
        assert pos.outside_swap_costs["model"] == "OUTSIDE_SWAP_COSTS_MODELED"
        assert pos.outside_swap_costs["known"] is True
        assert pos.outside_swap_costs[
            "recoverable_deposit_not_a_loss"] is True
        assert pos.outside_swap_costs["recoverable_lamports"] > 0
        assert pos.outside_swap_costs["non_recoverable_lamports"] > 0
        assert pos.outside_swap_cost_usd == pytest.approx(
            costs.charged_to_pnl_usd, abs=1e-12)
        # the fees are CHARGED, the rent is only RESERVED.
        assert pos.outside_swap_cost_usd < pos.reserved_outside_swap_usd
    # ONE quote was requested: the three arms held equal budgets, so the shared
    # collector coalesced them onto the same size.
    assert len({c["sol_lamports"] for c in qs.buy_calls}) == 1
    assert len(qs.buy_calls) == 1


def test_the_entry_is_quoted_at_the_arms_deployable_cash(tmp_path):
    """The lamports requested ARE the arm's deployable cash, net of the reserve."""
    qs = SizingQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    costs = _expected_outside_swap()
    gross = int(exp.starting_bankroll_usd / SOL_USD * 1_000_000_000)
    expected = gross - int(costs.total_reserve_lamports)
    assert qs.buy_calls[0]["sol_lamports"] == expected
    assert expected < gross, "the reserve must actually be held back"
    # the reserve is the protocol/configured components, stated as lamports.
    assert int(costs.base_fee_lamports) == 5_000 * 2
    assert int(costs.priority_fee_lamports) == 50_000 * 2
    assert int(costs.rent_lamports) > 2_000_000


def test_a_win_compounds_into_a_larger_next_all_in_budget(tmp_path):
    eng = _all_in_engine(tmp_path)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    a = _beta_arm(exp, pid, "A")
    spent = a.entry_cost_usd
    exp.observe(pid, "A", mark_price=a.entry_fill_price * 1.31, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6,
                   exit_fill=_exit_at_multiple(exp, pid, "A", 1.30),
                   require_quote=True)
    assert a.status == "closed"
    assert a.net_return_pct == pytest.approx(30.0, abs=1e-6)
    # THE WHOLE DEPLOYED budget earned 30%: ~+$2.92, not +$0.29 on a 10% stake.
    assert a.trading_pnl_usd == pytest.approx(spent * 0.30, abs=1e-6)
    # ... and the NET is that MINUS the fees paid outside the swap, so the
    # number that moves the bankroll is fully costed.
    assert a.net_pnl_usd == pytest.approx(
        spent * 0.30 - _expected_outside_swap().charged_to_pnl_usd, abs=1e-6)
    assert a.net_pnl_usd < a.trading_pnl_usd
    won = exp.accounts["A"].bankroll_usd
    assert won > 12.9
    # ... and the NEXT entry is sized on the GROWN bankroll.
    plan = exp.arm_entry_plan()
    assert plan["A"]["available_cash_usd"] == pytest.approx(won, abs=1e-9)
    assert plan["A"]["notional_usd"] == pytest.approx(won, abs=1e-9)
    assert plan["A"]["notional_usd"] > 12.9
    assert plan["A"]["sizing_pct"] == 100.0


def test_a_loss_shrinks_the_next_all_in_budget_and_never_goes_negative(tmp_path):
    eng = _all_in_engine(tmp_path)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    b = _beta_arm(exp, pid, "B")
    spent = b.entry_cost_usd
    # B's stop is -30%; the observed path troughs at -31%.
    exp.observe(pid, "B", mark_price=b.entry_fill_price * 0.69, at=T0 + 5)
    exp.settle_arm(pid, "B", now=T0 + 6,
                   exit_fill=_exit_at_multiple(exp, pid, "B", 0.69),
                   require_quote=True)
    assert b.status == "closed" and b.net_return_pct < 0
    assert b.trading_pnl_usd == pytest.approx(spent * (b.net_return_pct / 100.0),
                                              abs=1e-6)
    assert b.net_pnl_usd == pytest.approx(
        b.trading_pnl_usd - _expected_outside_swap().charged_to_pnl_usd,
        abs=1e-6)
    lost = exp.accounts["B"].bankroll_usd
    assert 0.0 < lost < spent
    plan = exp.arm_entry_plan()
    assert plan["B"]["notional_usd"] == pytest.approx(lost, abs=1e-9)
    assert plan["B"]["notional_usd"] < 10.0
    # the fully-costed loss is strictly worse than the quote-only one.
    assert lost < 10.0 + b.trading_pnl_usd


def test_all_in_arms_keep_independent_bankrolls(tmp_path):
    """One arm's all-in win never funds, or shrinks, another arm's entry."""
    eng = _all_in_engine(tmp_path)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    a, b, c = (_beta_arm(exp, pid, x) for x in "ABC")
    exp.observe(pid, "A", mark_price=a.entry_fill_price * 1.31, at=T0 + 5)
    exp.observe(pid, "B", mark_price=b.entry_fill_price * 0.69, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6,
                   exit_fill=_exit_at_multiple(exp, pid, "A", 1.30),
                   require_quote=True)
    exp.settle_arm(pid, "B", now=T0 + 6,
                   exit_fill=_exit_at_multiple(exp, pid, "B", 0.69),
                   require_quote=True)
    plan = exp.arm_entry_plan()
    # A's win enlarged only A's budget; B's loss shrank only B's; C, still
    # holding its position, is untouched and has no free capital.
    assert plan["A"]["notional_usd"] > BETA_STARTING_BANKROLL_USD
    assert plan["B"]["notional_usd"] < BETA_STARTING_BANKROLL_USD
    assert exp.accounts["C"].bankroll_usd == BETA_STARTING_BANKROLL_USD
    assert plan["C"]["blocked"] is True and c.status == "open"
    assert plan["C"]["committed_capital_usd"] == pytest.approx(
        c.entry_cost_usd, abs=1e-9)
    assert plan["C"]["notional_usd"] == 0.0
    # NO POOLING: the three budgets are the three ledgers, not one pot.
    assert exp.accounts["A"].bankroll_usd != exp.accounts["B"].bankroll_usd
    assert exp.accounts["C"].bankroll_usd == BETA_STARTING_BANKROLL_USD
    total = sum(exp.accounts[x].bankroll_usd for x in "ABC")
    assert total == pytest.approx(
        BETA_STARTING_BANKROLL_USD + a.net_pnl_usd
        + BETA_STARTING_BANKROLL_USD + b.net_pnl_usd
        + BETA_STARTING_BANKROLL_USD, abs=1e-6)


def test_the_all_in_entry_reserves_the_outside_swap_costs(tmp_path):
    """The venue fee is INSIDE the quoted input; the fees/rent are held BESIDE it.

    Two different things, and conflating them was the defect:

    * the SWAP fee is already inside the quoted input amount, and the fill model
      records it as zero so it is never charged twice;
    * the Solana base fee, the priority fee and the token-account rent are paid
      OUTSIDE the swap, so they are held back from the deployable notional.
    """
    eng = _all_in_engine(tmp_path)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    pos = _beta_arm(exp, pid, "A")
    costs = _expected_outside_swap()
    # the committed cost is the venue's gross input, converted at the same
    # SOL/USD reference the quote was requested with.
    assert pos.entry_cost_usd == pytest.approx(
        float(pos.entry_cost_quote) * SOL_USD, rel=1e-9)
    # the SWAP fee is zero BECAUSE it is already inside the quoted output;
    # charging it again would be double-counting.
    assert pos.entry_fee_quote == 0.0
    # ... but the OUTSIDE-swap costs are held back, and they are recorded with
    # the recoverable deposit separated from the spent fees.
    reserve = costs.total_reserve_usd
    assert reserve > 0
    assert exp.available_cash_usd("A") == pytest.approx(reserve, abs=1e-6)
    assert pos.allocated_usd - pos.entry_cost_usd == pytest.approx(
        reserve, abs=1e-6)
    assert pos.outside_swap_costs["non_recoverable_usd"] == pytest.approx(
        costs.charged_to_pnl_usd, abs=1e-12)
    assert pos.outside_swap_costs["recoverable_usd"] == pytest.approx(
        costs.recoverable_usd, abs=1e-12)
    assert (pos.outside_swap_costs["non_recoverable_lamports"]
            + pos.outside_swap_costs["recoverable_lamports"]
            == pos.outside_swap_costs["total_reserve_lamports"])


def test_an_arm_holding_a_position_takes_no_second_entry(tmp_path):
    """OVERLAP: an all-in arm already holding is blocked; the others are not."""
    qs = SizingQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    first = run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid1 = next(iter(exp.positions))
    a = _beta_arm(exp, pid1, "A")
    # free B and C by settling them, and leave A open.
    for arm in "BC":
        p = _beta_arm(exp, pid1, arm)
        exp.observe(pid1, arm, mark_price=p.entry_fill_price * 1.6, at=T0 + 5)
        exp.settle_arm(pid1, arm, now=T0 + 6,
                       exit_fill=quote_exit(exp, pid1, arm,
                                             price=p.entry_fill_price * 1.5),
                       require_quote=True)
        assert p.status == "closed"
    assert a.status == "open"
    buys_before = len(qs.buy_calls)
    records = run(eng.on_signal(eligible_signal(telegram_timestamp=T0 + 60)))
    pids = list(exp.positions)
    assert len(pids) == 2, "the second signal did not open its own parent"
    pid2 = [p for p in pids if p != pid1][0]
    assert set(exp.positions[pid2]) == {"B", "C"}   # A did NOT re-enter
    assert a.status == "open" and a.parent_signal_id == pid1
    # NO BUY QUOTE WAS SPENT ON THE BLOCKED ARM. The two free arms hold equal
    # (grown) budgets, so their identical notional coalesced onto ONE quote.
    assert len(qs.buy_calls) - buys_before == 1
    # ... and the new size is the arm's GROWN all-in budget, not the old $10.
    assert qs.buy_calls[-1]["sol_lamports"] > 81_726_054
    blocked = [r for r in records if r.arm == "V2_BETA_A"]
    assert blocked and blocked[0].traded is False
    assert "OVERLAP" in blocked[0].arm_filter_reason
    assert blocked[0].l25_status == "V2_FLOOR_BLOCKED"


def test_an_arm_without_capital_is_refused_and_the_others_still_trade(tmp_path):
    qs = SizingQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    exp = eng.beta_experiment
    exp.accounts["A"].bankroll_usd = 0.0
    records = run(eng.on_signal(eligible_signal()))
    pid = next(iter(exp.positions))
    assert set(exp.positions[pid]) == {"B", "C"}
    refused = [r for r in records if r.arm == "V2_BETA_A"]
    assert refused and refused[0].traded is False
    assert "INSUFFICIENT_CAPITAL" in refused[0].arm_filter_reason


def test_an_unresolved_arm_holds_its_capital_and_blocks_re_entry(tmp_path):
    qs = SizingQuoteService(sell=unavailable_point())
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    run(eng._monitor_beta_experiment())
    reserve = _expected_outside_swap().total_reserve_usd
    for arm in "ABC":
        assert _beta_arm(exp, pid, arm).status == "unresolved"
        # the deployed capital stays COMMITTED, and the only free cash the arm
        # has left is the fee/rent reserve -- which is NOT deployable, and the
        # arm is blocked on OVERLAP regardless.
        assert exp.available_cash_usd(arm) == pytest.approx(reserve, abs=1e-6)
        assert exp.available_cash_usd(arm) < 0.5
    plan = exp.arm_entry_plan()
    assert all(plan[x]["blocked"] for x in "ABC")
    assert all("OVERLAP" in plan[x]["reason"] for x in "ABC")
    # a fresh signal therefore opens NOTHING, and books a refusal per arm.
    records = run(eng.on_signal(eligible_signal(telegram_timestamp=T0 + 60)))
    assert len(exp.positions) == 1
    assert all(r.traded is False for r in records)


def test_an_all_in_arm_cannot_settle_twice(tmp_path):
    eng = _all_in_engine(tmp_path)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    a = _beta_arm(exp, pid, "A")
    exp.observe(pid, "A", mark_price=a.entry_fill_price * 1.31, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6,
                   exit_fill=_exit_at_multiple(exp, pid, "A", 1.30),
                   require_quote=True)
    bankroll = exp.accounts["A"].bankroll_usd
    # the same arm, the same parent, more attempts: no second booking.
    exp.settle_arm(pid, "A", now=T0 + 7,
                   exit_fill=_exit_at_multiple(exp, pid, "A", 1.30),
                   require_quote=True)
    exp.monitor_once(now=T0 + 8)
    assert exp.accounts["A"].bankroll_usd == bankroll
    assert len([r for r in exp.completed if r["arm"] == "A"]) == 1
    assert len(_beta_closed_rows(exp)) == 1


def test_restart_restores_the_all_in_commitment_and_blocks_re_entry(tmp_path):
    eng = _all_in_engine(tmp_path)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    cost = _beta_arm(exp, pid, "A").entry_cost_usd

    revived = BetaExperiment(state_root=tmp_path)
    pos = _beta_arm(revived, pid, "A")
    assert pos is not None and pos.status == "open"
    assert pos.sizing_rule == SIZING_ALL_IN
    assert pos.sizing_pct == 100.0
    assert pos.allocated_usd == pytest.approx(10.0, abs=1e-9)
    assert pos.entry_cost_usd == pytest.approx(cost, abs=1e-9)
    # the commitment survived, so the arm still has no free capital and is
    # still blocked from a second entry.
    reserve = _expected_outside_swap().total_reserve_usd
    assert revived.available_cash_usd("A") == pytest.approx(reserve, abs=1e-6)
    assert revived.available_cash_usd("A") < 0.5
    assert revived.arm_entry_plan()["A"]["blocked"] is True
    # the POSITION carries the exact restored numbers ...
    assert pos.entry_cost_usd == pytest.approx(cost, abs=1e-9)
    assert pos.reserved_outside_swap_usd == pytest.approx(reserve, abs=1e-6)
    # ... and the REPORT rounds them to 3dp for display, so its tolerance is the
    # report's own rounding rather than the stored precision.
    stats = revived.arm_stats("A")
    assert stats["capital_committed_usd"] == pytest.approx(cost, abs=1e-3)
    assert stats["available_bankroll_usd"] == pytest.approx(reserve, abs=1e-3)
    # the cost model survived the restart: the restored arm will be charged the
    # same outside-swap costs when it settles.
    assert pos.outside_swap_costs["model"] == "OUTSIDE_SWAP_COSTS_MODELED"
    assert pos.outside_swap_cost_usd == pytest.approx(
        _expected_outside_swap().charged_to_pnl_usd, abs=1e-12)
    assert pos.entry_provenance["side"] == "BUY"
    assert pos.entry_provenance["token_mint"], "entry provenance survived"


def test_restored_capital_prevents_a_negative_ledger(tmp_path):
    """A restart cannot revive an arm at a bankroll it no longer has."""
    eng = _all_in_engine(tmp_path)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    b = _beta_arm(exp, pid, "B")
    exp.observe(pid, "B", mark_price=b.entry_fill_price * 0.69, at=T0 + 5)
    exp.settle_arm(pid, "B", now=T0 + 6,
                   exit_fill=_exit_at_multiple(exp, pid, "B", 0.69),
                   require_quote=True)
    lost = exp.accounts["B"].bankroll_usd
    assert 0.0 < lost < 10.0
    revived = BetaExperiment(state_root=tmp_path)
    assert revived.accounts["B"].bankroll_usd == pytest.approx(lost, abs=1e-9)
    assert revived.accounts["B"].bankroll_usd > 0.0


# ---------------------------------------------------------------------------
# reporting: the sizing rule is stated, and old results are never relabelled
# ---------------------------------------------------------------------------

def test_sizing_is_reported_per_rule_and_legacy_never_reads_as_all_in(tmp_path):
    # a LEGACY record -- written before this change, by the removed 10% rule.
    legacy = {"status": "closed", "arm": "A", "net_pnl_usd": 0.30,
              "net_return_pct": 30.0, "position_size_usd": 1.0,
              "pnl_basis": "V2_FLOOR_BETA_MODELED_BACKTEST"}
    assert sizing_bucket(legacy) == "legacy"
    # an EXPLICIT-SIZE probe is not all-in either.
    explicit = {**legacy, "sizing_rule": SIZING_EXPLICIT, "sizing_pct": None}
    assert sizing_bucket(explicit) == "explicit"
    all_in = {**legacy, "sizing_rule": SIZING_ALL_IN, "sizing_pct": 100.0}
    assert sizing_bucket(all_in) == "all_in"
    # a record claiming a sub-100 percentage is legacy, not all-in.
    assert sizing_bucket({**legacy, "sizing_pct": 10.0}) == "legacy"

    exp = new_exp(tmp_path)
    open_p1(exp)                       # EXPLICIT_SIZE (fixture path)
    for arm in "ABC":
        p = _beta_arm(exp, "p1", arm)
        # +31% clears A's +30% target but not B/C's +50%; observe +60% so all
        # three trigger, then book each at +50%.
        exp.observe("p1", arm, mark_price=p.entry_fill_price * 1.60, at=T0 + 5)
        exp.settle_arm("p1", arm, now=T0 + 6,
                       exit_fill=quote_exit(exp, "p1", arm,
                                             price=p.entry_fill_price * 1.50),
                       require_quote=True)
        assert p.status == "closed"
    by_sizing = exp.stats()["provenance"]["by_sizing"]
    assert by_sizing["explicit"]["completed_trades"] == 3
    assert by_sizing["all_in"]["completed_trades"] == 0
    assert by_sizing["legacy"]["completed_trades"] == 0
    assert exp.arm_stats("A")["sizing_pct"] == 100.0
    assert exp.arm_stats("A")["sizing_rule"].startswith("ALL_IN")


def test_status_states_the_all_in_rule_and_splits_legacy_out(tmp_path):
    exp = new_exp(tmp_path)
    open_p1(exp)
    p = _beta_arm(exp, "p1", "A")
    exp.observe("p1", "A", mark_price=p.entry_fill_price * 1.31, at=T0 + 5)
    exp.settle_arm("p1", "A", now=T0 + 6,
                   exit_fill=quote_exit(exp, "p1", "A",
                                         price=p.entry_fill_price * 1.30),
                   require_quote=True)
    text = ui.render_stats(
        paper={"total_trades": 0,
               "execution_aware_paper": {"v2_floor_beta": exp.stats()}},
        real={})
    assert "ALL-IN" in text
    assert "100% of its own available" in text
    assert "sizing \u00b7 ALL-IN 100%" in text


def test_an_open_arm_reports_its_committed_capital_in_the_status(tmp_path):
    eng = _all_in_engine(tmp_path)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    cost = _beta_arm(exp, pid, "A").entry_cost_usd
    text = ui.render_stats(
        paper={"total_trades": 0,
               "execution_aware_paper": {"v2_floor_beta": exp.stats()}},
        real={})
    assert "committed" in text
    assert "entry cost" in text
    assert "sizing 100%" in text
    assert f"${cost:,.4f}" in text
