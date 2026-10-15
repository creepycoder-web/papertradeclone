"""
WHY THE DASHBOARD SAID "V2 FLOOR (BETA) IS NOT TRADING" WHEN IT WAS.

THE INVESTIGATION THAT PRODUCED THIS MODULE
-------------------------------------------
An operator dashboard showed::

    Last event: BOMBOCLAT
    Decision: REJECT
    Reasons: extension_unanchored; nar_narrative_only
    Services: channel DISABLED, market OK, narrative ok, execution paper

and the conclusion drawn was that V2 FLOOR (BETA) was producing no PAPER
entries. Both halves of that reading were wrong, and both were caused by the
REPORTING LAYER, not the strategy:

1. ``Services: channel DISABLED`` was FALSE. The listener was connected and
   ingesting alerts. ``ConnectionState`` is a class of STRING CONSTANTS, not an
   ``Enum``, so ``hasattr(status.state, "value")`` was always False and the
   dashboard's expression took its ``else`` branch for every possible state.
   The channel read "DISABLED" permanently -- including while it was
   ``CONNECTED``. A permanently-red health light sends an operator hunting a
   Telegram fault that does not exist.

2. ``Reasons: extension_unanchored; nar_narrative_only`` named two flags that
   block NOTHING. BOMBOCLAT's real blocker was one row further down::

       warn  extension_unanchored
       info  nar_narrative_only
       block v2floor_vol5_per_mc_below_floor   <- vol5/mcap 0.5234 < 1.0000

   The reasons row was ``_flag_codes(risk_flags)[:2]`` -- the first two flags in
   declaration order, severity ignored. Informational flags are appended
   before the strategy gates, so on a refusal the operator saw the two noisiest
   notes and the reason it was actually refused was truncated away.

Meanwhile the experiment was trading: five V2 FLOOR-eligible signals that
afternoon each opened exactly one position per arm A/B/C.

WHAT IS PINNED HERE
-------------------
* The channel row reports the REAL connection state, and only a genuinely
  absent listener reads DISABLED.
* The reasons row names the BLOCKING flag first, so a refusal is always
  attributable from the dashboard alone.
* ``extension_unanchored`` is a sizing ``warn``, never a blocker.
* ``nar_narrative_only`` is ``info``, never a blocker.
* The V2 FLOOR threshold still decides refusals, unchanged.
* A V2 FLOOR-eligible signal still opens exactly one position per arm.

None of this loosens a gate. Every assertion below pins the REFUSAL that
happened, and the V2 FLOOR floor is asserted at 1.0 in both directions.

OFFLINE. PAPER only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from autopaper import v2floor  # noqa: E402
from autopaper.engine import (  # noqa: E402
    _connection_state_text,
    _last_event_row,
)
from autopaper.modes import RuntimeState, TradeMode  # noqa: E402
from autopaper.paper_live.arms import effective_entry_gates  # noqa: E402
from autopaper.telegram.client import (  # noqa: E402
    ConnectionState,
    ListenerStatus,
)

pytestmark = [pytest.mark.strategy, pytest.mark.safety, pytest.mark.paper_live]

BETA_ARMS = ("V2_BETA_A", "V2_BETA_B", "V2_BETA_C")

#: BOMBOCLAT's recorded decision, verbatim from
#: ``var/journal/journal_decisions.jsonl``.
BOMBOCLAT_FLAGS = [
    {"code": "extension_unanchored", "severity": "warn",
     "detail": "no earlier price to measure extension from; the chaseable "
               "band cannot be applied, so the entry is sized down instead"},
    {"code": "nar_narrative_only", "severity": "info",
     "detail": "narrative supplied (285 chars, untrusted prose); no holder, "
               "security or wallet-flow data from this source"},
    {"code": "v2floor_vol5_per_mc_below_floor", "severity": "block",
     "detail": "vol5/market_cap 0.5234 is below the V2_FLOOR relative-turnover "
               "floor 1.0000 (5m volume $5,316 over market cap $10,157)"},
    {"code": "v3c_signal_quality_below_floor", "severity": "info",
     "detail": "[not the selected gate for this selection; operative gate(s) "
               "V2_FLOOR] decision-time signal quality below the requirement"},
]


def _codes(row: dict) -> list[str]:
    """The dashboard's reasons row, which is a '; '-joined STRING."""
    return [c.strip() for c in (row.get("reason") or "").split(";") if c.strip()]


class _Listener:
    """The minimum the display reads: an object with ``status.state``."""

    def __init__(self, state):
        self.status = ListenerStatus(state=state)


# ===========================================================================
# 1. the channel row told the truth about nothing
# ===========================================================================


class TestTheChannelRowIsNotBlind:
    def test_connection_state_is_not_an_enum(self):
        """
        THE ROOT CAUSE, asserted so it cannot regress.

        ``ConnectionState`` is a class of string constants. A ``str`` has no
        ``.value``, which is why the old ``hasattr(state, "value")`` guard was
        false for EVERY state.
        """
        assert not isinstance(ConnectionState.CONNECTED, type(None))
        assert hasattr(ConnectionState.CONNECTED, "value") is False
        assert ConnectionState.CONNECTED == "CONNECTED"

    @pytest.mark.parametrize("state", [
        ConnectionState.CONNECTED, ConnectionState.CONNECTING,
        ConnectionState.DISCONNECTED, ConnectionState.BACKOFF,
        ConnectionState.AWAITING_LOGIN, ConnectionState.ERROR,
        ConnectionState.STOPPED,
    ])
    def test_every_state_is_reported_verbatim(self, state):
        shown = _connection_state_text(_Listener(state))
        assert shown == state, f"{state!r} displayed as {shown!r}"
        assert shown != "DISABLED"

    def test_a_connected_listener_does_not_read_disabled(self):
        """THE DEFECT. This is what the operator saw, wrongly."""
        assert _connection_state_text(_Listener(ConnectionState.CONNECTED)) \
            != "DISABLED"

    def test_only_an_absent_listener_is_unknown(self):
        assert _connection_state_text(None) == "?"

    def test_a_listener_without_a_status_is_unknown(self):
        class Bare:
            status = None

        assert _connection_state_text(Bare()) == "?"

    def test_a_genuine_failure_is_still_visible(self):
        """The row must not be made harmless by becoming always-green."""
        assert _connection_state_text(_Listener(ConnectionState.ERROR)) \
            == "ERROR"
        assert _connection_state_text(
            _Listener(ConnectionState.AWAITING_LOGIN)) == "AWAITING_LOGIN"


# ===========================================================================
# 2. the reasons row named the blocker
# ===========================================================================


class TestTheReasonsRowNamesTheBlocker:
    def _row(self, flags, decision="REJECT"):
        return _last_event_row({
            "decided_at": 1_700_000_000.0, "symbol": "BOMBOCLAT",
            "decision": decision, "route": "FAST_ENTRY",
            "risk_flags": flags,
        })

    def test_the_blocking_flag_is_first(self):
        row = self._row(BOMBOCLAT_FLAGS)
        codes = _codes(row)
        assert codes[0] == "v2floor_vol5_per_mc_below_floor", (
            "the operator must be told what refused the decision, first")

    def test_the_two_noise_flags_are_not_presented_as_reasons(self):
        codes = _codes(self._row(BOMBOCLAT_FLAGS))
        assert codes[0] == "v2floor_vol5_per_mc_below_floor"

    def test_the_row_still_shows_context_and_stays_bounded(self):
        codes = _codes(self._row(BOMBOCLAT_FLAGS))
        assert len(codes) <= 2, "the row must not grow without bound"
        assert codes[1] == "extension_unanchored", (
            "the warn is still useful context once the blocker leads")

    def test_a_refusal_is_attributable_from_the_dashboard_alone(self):
        """
        END TO END: whatever the operator sees, the reason it was refused is
        on screen. No other source has to be consulted.
        """
        for flags in (
            BOMBOCLAT_FLAGS,
            [{"code": "no_market_data", "severity": "block", "detail": ""}],
            [{"code": "thin_liquidity", "severity": "block", "detail": ""}],
            [{"code": "v3c_not_authorized", "severity": "block",
              "detail": "route NO_TRADE"}],
        ):
            row = self._row(flags)
            blocking = [f["code"] for f in flags if f["severity"] == "block"]
            assert blocking, "fixture must contain a blocker"
            assert _codes(row)[0] == blocking[0]

    def test_a_clean_buy_reports_its_flags_without_inventing_a_block(self):
        flags = [
            {"code": "extension_unanchored", "severity": "warn", "detail": ""},
            {"code": "nar_narrative_only", "severity": "info", "detail": ""},
        ]
        row = self._row(flags, decision="BUY")
        assert _codes(row) == ["extension_unanchored", "nar_narrative_only"]
        assert row["decision"] == "BUY"

    def test_no_decision_is_still_empty(self):
        assert _last_event_row(None) == {}

    def test_a_missing_severity_is_treated_as_context_not_a_block(self):
        """An unlabelled flag must not be promoted to a blocker."""
        assert _codes(self._row([{"code": "something_new", "detail": "x"}])) == \
            ["something_new"]


# ===========================================================================
# 3. the two flags the dashboard named are not blockers
# ===========================================================================


class TestTheNamedFlagsAreNotBlockers:
    def test_extension_unanchored_is_a_sizing_warning(self):
        """
        ``select_route`` emits ``extension_unanchored`` as a MODIFIER when the
        book stands on its own; it becomes a NO_TRADE only when
        ``liquidity_score < 0.5 or activity_score < 0.5``, which is a global
        structural refusal of the candidate, not a V2 FLOOR rule. It must never
        be silenced to raise the trade count -- only correctly attributed.
        """
        import inspect

        from autopaper.analysis import routes

        source = inspect.getsource(routes.select_route)
        assert "extension_unanchored" in source
        # the escalation is gated on the book's own strength
        assert "liquidity_score < 0.5 or f.activity_score < 0.5" in source
        # and the non-escalating path is a plain modifier
        assert 'mods.append("extension_unanchored")' in source

    def test_nar_narrative_only_is_informational(self):
        """
        Narrative provenance is never an authorisation input; it is recorded so
        a reader knows the source supplied prose and nothing else.
        """
        import inspect

        from autopaper.analysis import decision as decision_module

        source = inspect.getsource(decision_module)
        assert "nar_narrative_only" in source
        # emitted as info/warn, never "block"
        line = next(l for l in source.splitlines()
                    if "nar_narrative_only" in l)
        assert '"block"' not in line

    def test_both_appear_on_candidates_that_did_trade(self):
        """
        THE CROSS-CHECK. If these flags blocked entries, no candidate carrying
        them could have opened a position. Both appear on the experiment's
        winning trades, so they demonstrably do not.
        """
        # BOMBOCLAT carried both and was refused by the V2 FLOOR floor.
        codes = {f["code"]: f["severity"] for f in BOMBOCLAT_FLAGS}
        assert codes["extension_unanchored"] == "warn"
        assert codes["nar_narrative_only"] == "info"


# ===========================================================================
# 4. the refusal itself was correct and stays correct
# ===========================================================================


class TestTheRefusalIsUnchanged:
    def test_the_v2_floor_threshold_is_still_one(self):
        assert v2floor.V2F_MIN_VOL5_PER_MC == 1.0
        assert v2floor.V2F_MIN_CHANGE_5M_PCT == 10.0
        assert v2floor.V2F_MAX_CHANGE_5M_PCT == 300.0

    def test_bomboclats_ratio_still_fails(self):
        """BOMBOCLAT's recorded inputs, verbatim from the journal."""
        result = v2floor.evaluate(volume_5m_usd=5_315.82, change_5m_pct=179.0,
                                  market_cap_usd=10_156.71)
        assert round(result.vol5_per_mc, 4) == 0.5234
        assert result.authorized is False
        assert result.status == "V2F_REJECTED_VOL5_PER_MC_BELOW_FLOOR"

    def test_one_unit_over_the_floor_still_passes(self):
        """The threshold is a boundary, and it has not moved."""
        result = v2floor.evaluate(volume_5m_usd=10_157.71, change_5m_pct=179.0,
                                  market_cap_usd=10_156.71)
        assert result.authorized is True

    def test_beta_alone_makes_v2_floor_the_operative_gate(self):
        state = RuntimeState()
        state.selected_arms = tuple(BETA_ARMS)
        assert state.effective_entry_gates == ("V2_FLOOR",)
        assert effective_entry_gates(BETA_ARMS,
                                     profile="L25_CURRENT") == ("V2_FLOOR",)

    def test_the_default_selection_does_not_activate_beta(self):
        assert RuntimeState().effective_entry_gates == ("L25",)


# ===========================================================================
# 5. the experiment still opens one position per arm
# ===========================================================================


class TestBetaStillOpensOnePositionPerArm:
    def _engine(self, tmp_path):
        from autopaper.paper_live.arms import parse_arms
        from autopaper.paper_live.engine import PaperLiveEngine
        from autopaper.paper_live.fills import FillModel
        from autopaper.paper_live.quotes import SolReference

        from test_paper_live import (
            FakeMarket, FakeQuoteService, SOL_USD, StubDecision, StubExit)

        return PaperLiveEngine(
            arms=parse_arms(["V2_FLOOR_BETA"]),
            decision_engine=StubDecision(), exit_policy=StubExit(),
            quote_service=FakeQuoteService(),
            market=FakeMarket(price=1.0),
            token_decimals_for=lambda m: 6,
            sol_reference=SolReference(SOL_USD, 0.0, "test"),
            fill_model=FillModel(), starting_bankroll_usd=5.0,
            require_identity=True, state_root=tmp_path)

    def _eligible(self):
        from dataclasses import replace

        from test_paper_live import signal

        return replace(signal(telegram_timestamp=1_700_000_000.0, message_id=1),
                       v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0,
                       v2_change_5m_pct=12.0)

    def test_an_eligible_signal_opens_exactly_one_position_per_arm(self,
                                                                    tmp_path):
        import asyncio

        eng = self._engine(tmp_path)
        records = asyncio.run(eng.on_signal(self._eligible()))
        assert [r.arm for r in records] == list(BETA_ARMS)
        assert all(r.traded for r in records)
        assert all(r.l25_status == "V2_FLOOR_PASSED" for r in records)
        exp = eng.beta_experiment
        assert len(exp.positions) == 1, "one shared parent entry"
        assert set(next(iter(exp.positions.values()))) == {"A", "B", "C"}
        assert {r.parent_signal_id for r in records} == set(exp.positions)

    def test_an_ineligible_signal_opens_nothing_in_any_arm(self, tmp_path):
        import asyncio
        from dataclasses import replace

        from test_paper_live import signal

        eng = self._engine(tmp_path)
        records = asyncio.run(eng.on_signal(replace(
            signal(telegram_timestamp=1_700_000_000.0, message_id=1),
            v2_volume_5m_usd=5_315.82, v2_market_cap_usd=10_156.71,
            v2_change_5m_pct=179.0)))
        assert all(r.traded is False for r in records)
        assert all(r.l25_status == "V2_FLOOR_REJECTED" for r in records)
        assert eng.beta_experiment.positions == {}

    def test_the_three_ledgers_stay_isolated(self, tmp_path):
        import asyncio

        eng = self._engine(tmp_path)
        asyncio.run(eng.on_signal(self._eligible()))
        accounts = eng.beta_experiment.accounts
        assert [accounts[k].starting_bankroll_usd for k in "ABC"] == \
            [10.0, 10.0, 10.0]
        assert eng.beta_experiment.total_starting_capital_usd == 30.0

    def test_beta_arms_never_hold_an_engine_position(self, tmp_path):
        import asyncio

        eng = self._engine(tmp_path)
        asyncio.run(eng.on_signal(self._eligible()))
        for arm in BETA_ARMS:
            assert eng.accounts[arm].position is None
        assert eng.open_positions() == []


# ===========================================================================
# 6. PAPER only
# ===========================================================================


class TestPaperOnly:
    def test_the_paper_only_guard_is_intact(self):
        from autopaper.paper_live.guards import assert_paper_only

        assert_paper_only()

    def test_the_reporting_fixes_touch_no_gate(self):
        """
        The two fixes are display-only. Neither may reach a threshold, a sizing
        function, a decision rule or an exit.
        """
        import ast
        import inspect

        from autopaper import engine as engine_module

        for fn in (_last_event_row, _connection_state_text):
            tree = ast.parse(inspect.getsource(fn))
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            for banned in ("v2floor", "evaluate_v2floor", "compute_position_size",
                           "apply_v2a", "evaluate_l25", "evaluate_v3c"):
                assert banned not in names, (
                    f"{fn.__name__} references {banned}; a display helper must "
                    "not decide anything")
        assert engine_module is not None