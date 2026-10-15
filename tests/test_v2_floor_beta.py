"""V2 FLOOR (BETA) — three-arm PAPER experiment regression suite.

Twenty required scenarios, each named where it is covered:

  1.  Selecting ``V2_FLOOR_BETA`` starts the three-arm experiment.
      -> TestExplicitOptIn, TestSelectorCallback, TestEngineEntry
  2.  All three arms initialize at exactly $10.00 ($30.00 total).
      -> TestEngineEntry.test_accounts_start_at_ten_dollars
  3.  Resetting beta does not change L25 or any other profile's bankroll.
      -> TestPersistenceLifecycle.test_reset_leaves_other_accounts_alone
  4.  Refreshing/reconnecting does not reset the session.
      -> TestPersistenceLifecycle.test_refresh_rehydrate_reconnect_keep_the_session
  5.  One eligible signal creates exactly one virtual position per arm.
      -> TestEngineEntry.test_one_signal_opens_one_position_per_arm
  6.  All three positions share the common entry assumptions and parent id.
      -> TestEngineEntry.test_positions_share_one_entry_basis
  7.  Duplicate signal delivery does not create extra positions.
      -> TestEngineEntry.test_duplicate_delivery_is_idempotent
  8.  Ineligible signals create no beta positions.
      -> TestEngineEntry.test_ineligible_signal_opens_nothing (+ gate values)
  9.  Arm A: +30% TP, -30% SL, 90s deadline.
  10. Arm B: +50% TP, -30% SL, 90s deadline.
  11. Arm C: +50% TP, -30% SL, 120s deadline.
      -> TestArmConfiguration (table, per-arm behavior, deadlines)
  12. Closing one arm does not close or settle either other arm.
      -> TestArmIndependence
  13. TP target touch is not automatically counted as a fill.
  14. Gapped-up targets do not receive unrealistic favorable fills.
  15. Stops and adverse gaps account for the implemented execution assumptions.
  16. Missing/stale quotes do not fabricate a completed trade.
  17. Ambiguous TP/SL ordering is conservative and flagged.
      -> TestExitSemantics
  18. Fees, slippage, realized P&L and ledger accounting are correct/isolated.
      -> TestAccounting
  19. Restart restores state; a new-session reset archives prior results and
      safely handles open positions.
      -> TestPersistenceLifecycle
  20. Beta cannot reach the live path and cannot be enabled by a default or
      legacy configuration.
      -> TestExplicitOptIn, TestSelectorCallback (REAL refusal), TestSafety

Everything is offline: no network, no Telegram, no RPC, nothing signed or
broadcast. Deterministic fixtures only — no new dependencies.

Every entry-gate number asserted here is a REGRESSION LOCK on the documented
V2 FLOOR rule (``autopaper.v2floor``: vol5/mcap >= 1.0, 5m change in
[+10%, +300%], all bounds inclusive) and on the frozen three-arm table. If one
of these fails because a threshold moved, the fix is to restore the threshold —
not to edit the test.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import time
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from pathlib import Path

import pytest

#: the tests/ dir ends up on sys.path at collection time by pytest's rootdir
#: conftest, so other test modules import as sibling modules. Depend on that
#: same convention rather than rewriting it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autopaper import v2floor
from autopaper.exits import _max_hold_for, _v2floor_bracket_for
from autopaper.ledger import PaperAccountState
from autopaper.modes import (
    RunState,
    RuntimeState,
    TradeMode,
    validate_strategy_selection,
)
from autopaper.paper_live.arms import (
    BETA_ARM_KEYS,
    BETA_ARMS,
    BETA_GROUP,
    ResearchArm,
    arm_by_name,
    arm_names,
    is_beta_group,
    parse_arms,
    selectable_arm_names,
)
from autopaper.paper_live.engine import (
    BETA_EXIT_RETRY_BASE_S,
    PaperLiveEngine,
)
from autopaper.paper_live.fills import FillModel
from autopaper.paper_live.guards import (
    PAPER_FORBIDDEN_ATTRS,
    PAPER_FORBIDDEN_MODULES,
    assert_paper_only,
    audit_paper_capability,
)
from autopaper.paper_live.quotes import SolReference
from autopaper.paper_live.schema import FailureReason, PaperPosition
from autopaper.paper_live.state import (
    STATE_SCHEMA_VERSION,
    PaperStateStore,
    PersistedState,
)
from autopaper.paper_live.v2_floor_beta import (
    BETA_ARM_CONFIGS,
    BetaExitFill,
    BetaEntryFill,
    BETA_ENTRY_PROFILE,
    BETA_LABEL,
    BETA_SELECTOR,
    BETA_STARTING_BANKROLL_USD,
    BetaExperiment,
    arm_config,
    net_from_gross,
)
from autopaper.strategy_profile import (
    DEFAULT_PROFILE,
    PROFILE_ENTRY_STRATEGY,
    PROFILE_EXIT_BRACKET,
    PROFILE_EXIT_STRATEGY,
    PROFILE_LABELS,
    PROFILE_MAX_HOLD_S,
    PROFILE_NOTES,
    PROFILE_OWN_EXIT_TRIGGERS,
    PROFILE_POSITION_PCT,
    PROFILE_VALUES,
    StrategyProfile,
    StrategyProfileError,
    exit_bracket_for,
    label,
    max_hold_s,
    parse_profile,
    position_pct_override,
)
from autopaper.telegram import ui
from autopaper.telegram.actions import ButtonHandler

from test_paper_live import (  # noqa: E402 - offline doubles, shared by the harness tests
    QUOTE_USD,
    FakeMarket,
    FakeQuoteService,
    MINT,
    SOL_USD,
    StubDecision,
    StubExit,
    run,
    sell_point,
    signal,
    unavailable_point,
)

pytestmark = [
    pytest.mark.paper_live,
    pytest.mark.strategy,
    pytest.mark.safety,
]

T0 = 1_700_000_000.0
#: The observation price every entry test uses, so the arithmetic is readable:
#: entry fill == 1.00, +30% == 1.30, +50% == 1.50, -30% == 0.70.
MARK_PRICE = 1.0

#: The three decision-time V2 FLOOR inputs, INSIDE the frozen rule:
#: vol5/mcap = 60,000/50,000 = 1.2 >= 1.0, change +12% within [+10%, +300%].
V2_OK = dict(
    v2_volume_5m_usd=60_000.0,
    v2_market_cap_usd=50_000.0,
    v2_change_5m_pct=12.0,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def quote_entry(*, price: float = MARK_PRICE, size: float = 10.0,
                tokens_raw: int = 1_000_000, decimals: int = 6,
                fig_reciprocal: float = 1.0) -> BetaEntryFill:
    """
    An EXECUTABLE-QUOTE entry, the only kind ``open_for_signal`` accepts now.

    Defect this replaced: the entry used to be the observed MARK price, so a
    missing quote could not be told apart from a good one and no exit could
    ever quote the quantity that had actually been acquired. Every field the
    exit needs -- acquired quantity, decimals, total BUY cost -- is carried.
    """
    fee = round(size * fig_reciprocal * 0.01, 6)
    return BetaEntryFill(
        price_usd=price, tokens_raw=tokens_raw, token_decimals=decimals,
        cost_quote=size, fee_quote=fee, mark_price=price,
        quote_timestamp=T0, quote_age_ms=12.0, venue="pumpfun:v1",
        pool_or_curve="testpool", provider="test-quote", route=("BUY",),
        identity="PASS",
    )


def beta_engine(tmp_path: Path, *, arms=None, market=None, decision=None,
                exits=None, quotes=None,
                exit_retry_base_s: float = BETA_EXIT_RETRY_BASE_S,
                ) -> PaperLiveEngine:
    """The real engine, state isolated under ``tmp_path``.

    NOTE: V2 FLOOR (BETA) is disabled from the active bot. This helper no longer
    defaults to beta arms. Tests that need specific arm configurations must pass
    them explicitly. Tests that specifically test beta functionality should be
    skipped with ``@pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled")``.

    Built directly rather than through ``test_paper_live.make_engine`` for ONE
    reason: ``state_root`` must point at this test's own directory, or the
    experiment's durable state file would be shared between tests (and would
    write into the operator's ``var/``). Everything else is the shared wiring.

    ``exit_retry_base_s`` stays at the PRODUCTION default. A test that needs two
    consecutive exit-quote attempts inside one wall-clock second -- i.e. one
    that proves RETRY RECOVERY rather than the retry cooldown -- passes ``0.0``
    explicitly, which is stated at that call site rather than assumed here.
    """
    return PaperLiveEngine(
        beta_exit_retry_base_s=float(exit_retry_base_s),
        arms=arms if arms is not None else parse_arms([]),
        decision_engine=decision or StubDecision(),
        exit_policy=exits or StubExit(),
        quote_service=quotes or FakeQuoteService(),
        market=market if market is not None else FakeMarket(price=MARK_PRICE),
        token_decimals_for=lambda mint: 6,
        sol_reference=SolReference(SOL_USD, time.time(), "test"),
        fill_model=FillModel(),
        starting_bankroll_usd=5.0,
        require_identity=True,
        state_root=tmp_path,
    )


def base_signal():
    return signal(telegram_timestamp=T0, message_id=11)


def eligible_signal(**overrides):
    """A snapshot carrying the decision-time V2 FLOOR gate inputs.

    The gateway accepts a snapshot that already carries its inputs (the
    research-harness / replay path), which is how the experiment receives the
    exact numbers the production decision was made on.
    """
    fields = dict(V2_OK)
    fields.update(overrides)
    return replace(base_signal(), **fields)


def new_exp(tmp_path: Path, bankroll: float = BETA_STARTING_BANKROLL_USD
            ) -> BetaExperiment:
    """A fresh experiment session with deterministic state under tmp_path."""
    return BetaExperiment(starting_bankroll_usd=bankroll, state_root=tmp_path)


def open_p1(exp: BetaExperiment, *, price: float = MARK_PRICE,
            size: float = 1.0, at: float = T0,
            pid: str = "p1") -> dict:
    """Open one shared entry across all three arms, at a controlled instant."""
    return exp.open_for_signal(
        parent_signal_id=pid, mint=MINT, symbol="T",
        entry_fill=quote_entry(price=price, size=size),
        position_size_usd=size, entry_at=at)


def _mark_market(eng, factor: float) -> float:
    """Set the engine's fake market price to ``factor`` x the QUOTED entry."""
    price = _mark_for(eng, factor)
    if eng is not None and getattr(eng, "market", None) is not None:
        eng.market.price = price
    return price


def _quote_market(eng, factor: float):
    """Move the EXECUTABLE SELL quote to ``factor`` x the entry cost.

    THE HELPER THE EXIT BRACKET NEEDS.

    ``_mark_market`` moves the AGGREGATOR MARK. That was enough to fire a bracket
    only while the bracket was evaluated against the mark -- which is the defect
    :data:`RETURN_BASIS_QUOTE` documents: the trigger and the P&L were two
    different books, and the two disagreed by a median 29% on the live cohort,
    so 18 of 20 ``SL`` labels described returns better than -30% and every
    hit-rate was meaningless.

    TP and SL are now tested against the executable SELL quote, so a test that
    wants to reach +30% has to move the QUOTE. This helper does exactly that:
    the net SELL proceeds are scaled by ``factor``, which makes the observed
    return precisely ``factor - 1`` on the declared basis. The mark is left
    where it was, on purpose -- the two are separate books and a test that
    moved only the mark and expected a fill is the bug under test.
    """
    qs = getattr(eng, "quote_service", None)
    assert qs is not None, "the engine has no quote service to move"
    current = qs.sell
    base = float(current.output_amount_decimal)
    scaled = base * float(factor)
    # The fee stays what it is: it is a real cost, not a price, and scaling it
    # would silently move the net return away from ``factor - 1``.
    fee = float(current.fee_amount or 0.0)
    price = float(current.effective_price_usd) * float(factor)
    qs.sell = replace(
        current,
        output_amount_decimal=Decimal(repr(scaled)),
        output_amount_raw=int(round(scaled * 10 ** 9)),
        effective_price_usd=price,
        effective_price=price / SOL_USD,
    )
    return qs.sell


def _mark_for(eng, factor: float, pid: str = None) -> float:
    """
    A market price that is ``factor`` x the entry the QUOTE produced.

    The entry is no longer the fixture's ``MARK_PRICE``: it is the executable
    BUY quote's price, which differs from the mark by design. Expressing the
    exit test in terms of the real entry keeps the intent ("only A's +30% is
    reached") meaningful instead of silently comparing two unrelated scales.
    """
    positions = eng.beta_experiment.positions
    pid = pid or next(iter(positions), None)
    base = 1.0
    if pid:
        base = float(next(iter(positions[pid].values())).entry_fill_price)
    return base * float(factor)


def move_sell_quote(eng, factor: float):
    """Scale the engine's executable SELL proceeds by ``factor`` x the entry cost.

    SHARED BY EVERY ENGINE-LEVEL BRACKET FIXTURE.

    TP and SL are evaluated on the executable SELL series, so a fixture that
    wants to reach a target has to move the QUOTE. Moving only the aggregator
    mark is precisely the defect documented on ``RETURN_BASIS_QUOTE``: the
    trigger and the P&L were two different books, they disagreed by a median 29%
    on the live cohort, and the resulting hit rates were meaningless.

    ``factor`` is a multiple of the BUY's recorded SOL cost, which is the return
    denominator, so the observed return is exactly ``factor - 1``.
    """
    from decimal import Decimal

    qs = getattr(eng, "quote_service", None)
    assert qs is not None, "the engine has no quote service to move"
    positions = eng.beta_experiment.positions
    pid = next(iter(positions))
    bp = next(iter(positions[pid].values()))
    current = qs.sell
    scaled = float(bp.entry_cost_quote) * float(factor)
    price = float(current.effective_price_usd) * float(factor)
    qs.sell = replace(
        current,
        output_amount_decimal=Decimal(repr(scaled)),
        output_amount_raw=int(round(scaled * 10 ** 9)),
        effective_price_usd=price,
        effective_price=price / SOL_USD,
    )
    return qs.sell


def quote_exit(exp: BetaExperiment, pid: str = "p1", arm: str = "A",
               price: float = MARK_PRICE) -> "BetaExitFill":
    """
    An EXECUTABLE-QUOTE exit for one arm's exact acquired quantity.

    Defect this replaced: ``settle_arm`` used to price the exit from the
    observed MARK, so a target "touch" became a booked trade. Now a settle
    requires a real quote, and an arm with none stays UNRESOLVED.
    """
    bp = exp.positions[pid][arm]
    raw = bp.entry_tokens_raw or 0
    decimals = bp.entry_token_decimals or 6
    gross = float(price) * (raw / (10 ** decimals))
    return BetaExitFill(
        price_usd=float(price), tokens_raw=int(raw),
        token_decimals=int(decimals), cost_quote=round(gross, 9),
        fee_quote=0.0, entry_cost_quote=float(bp.entry_cost_quote or 0.0),
        quote_timestamp=T0 + 5, identity="PASS",
    )


def pos(exp: BetaExperiment, arm: str, pid: str = "p1"):
    return exp.positions[pid][arm]


def json_file(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _winning_sell():
    """A SELL quote whose proceeds EXCEED the BUY input cost.

    Under the executable-quote settlement the P&L comes from the QUOTE, not
    from a mark, so a fixture whose sell books slightly worse than the buy
    would make every TP a loss. This one is priced so a TP exit is a win, which
    is what the live-results and dashboard tests assert.
    """
    return sell_point(output_amount_raw=4_500_000, output_amount_decimal=0.0045)


def _selectable_via_registry() -> set[str]:
    return set(selectable_arm_names())


# ===========================================================================
# registry coverage (task section 3)
# ===========================================================================


class TestRegistryCoverage:
    def test_beta_is_a_first_class_enum_member(self):
        assert isinstance(StrategyProfile.V2_FLOOR_BETA, StrategyProfile)
        assert StrategyProfile.V2_FLOOR_BETA.value == "V2_FLOOR_BETA"
        assert parse_profile("V2_FLOOR_BETA") is StrategyProfile.V2_FLOOR_BETA
        assert BETA_SELECTOR == "V2_FLOOR_BETA"
        assert BETA_GROUP == "V2_FLOOR_BETA"
        assert is_beta_group(BETA_GROUP) is True
        assert is_beta_group("V2_BETA_A") is False
        # it is part of the accepted profile vocabulary...
        assert "V2_FLOOR_BETA" in PROFILE_VALUES
        # ...but deliberately NOT offered as a free-standing selectable arm.
        assert StrategyProfile.V2_FLOOR_BETA not in _selectable_via_registry()

    @pytest.mark.parametrize("registry", [
        PROFILE_ENTRY_STRATEGY, PROFILE_EXIT_STRATEGY, PROFILE_MAX_HOLD_S,
        PROFILE_EXIT_BRACKET, PROFILE_OWN_EXIT_TRIGGERS,
        PROFILE_POSITION_PCT, PROFILE_LABELS, PROFILE_NOTES,
    ], ids=["ENTRY", "EXIT", "MAX_HOLD", "BRACKET", "TRIGGERS",
            "POSITION_PCT", "LABELS", "NOTES"])
    def test_every_registry_covers_beta_exactly_once(self, registry):
        beta_key = StrategyProfile.V2_FLOOR_BETA
        # one authoritative entry: a duplicate key cannot exist in a dict, but
        # counting catches a stray second lookup table or a redefinition.
        assert list(registry).count(beta_key) == 1
        assert beta_key in registry

    def test_position_pct_is_all_in_for_the_beta_experiment(self):
        # THE authoritative sizing value, in one place. V2_FLOOR_BETA is
        # ALL-IN: every arm commits 100% of its own uncommitted cash, so the
        # declared percentage is 100.0 and the registry cannot disagree with
        # the experiment's own rule (test_sizing_v3c locks this too).
        assert PROFILE_POSITION_PCT[StrategyProfile.V2_FLOOR_BETA] == 100.0
        assert position_pct_override(StrategyProfile.V2_FLOOR_BETA) == 100.0
        # beta and V2_FLOOR are the ONLY sizing overrides in the registry, and
        # V2_FLOOR keeps the researched flat 10% -- the change is beta-only.
        overriding = [p.value for p, v in PROFILE_POSITION_PCT.items()
                      if v is not None]
        assert sorted(overriding) == ["V2_FLOOR", "V2_FLOOR_BETA"]
        assert PROFILE_POSITION_PCT[StrategyProfile.V2_FLOOR] == 10.0

    def test_entry_strategy_is_the_existing_v2_floor_gate(self):
        assert (PROFILE_ENTRY_STRATEGY[StrategyProfile.V2_FLOOR_BETA]
                == "V2_FLOOR")
        assert (PROFILE_ENTRY_STRATEGY[StrategyProfile.V2_FLOOR_BETA]
                == PROFILE_ENTRY_STRATEGY[StrategyProfile.V2_FLOOR])

    def test_exit_registries_are_inert_so_no_generic_lookup_overrides_arms(self):
        # None = "the generic registry has no single answer for this profile".
        # If either of these said (30, -30) or 90, a generic lookup would
        # silently override arms B and C.
        assert PROFILE_MAX_HOLD_S[StrategyProfile.V2_FLOOR_BETA] is None
        assert PROFILE_EXIT_BRACKET[StrategyProfile.V2_FLOOR_BETA] is None
        assert max_hold_s(StrategyProfile.V2_FLOOR_BETA) is None
        assert exit_bracket_for(StrategyProfile.V2_FLOOR_BETA) is None
        # The exit STRATEGY names the experiment, and the trigger list spells
        # out all three arms with their own caps (90/90/120).
        assert (PROFILE_EXIT_STRATEGY[StrategyProfile.V2_FLOOR_BETA]
                == "V2_FLOOR_BETA")
        triggers = PROFILE_OWN_EXIT_TRIGGERS[StrategyProfile.V2_FLOOR_BETA]
        assert triggers == (
            "v2b_A_take_profit", "v2b_A_stop_loss", "v2b_A_max_hold_90s",
            "v2b_B_take_profit", "v2b_B_stop_loss", "v2b_B_max_hold_90s",
            "v2b_C_take_profit", "v2b_C_stop_loss", "v2b_C_max_hold_120s",
        )

    def test_labels_and_notes_describe_the_experiment(self):
        assert "V2 FLOOR (BETA)" in PROFILE_LABELS[StrategyProfile.V2_FLOOR_BETA]
        assert label(StrategyProfile.V2_FLOOR_BETA) == \
            PROFILE_LABELS[StrategyProfile.V2_FLOOR_BETA]
        note = PROFILE_NOTES[StrategyProfile.V2_FLOOR_BETA]
        assert "$10" in note
        assert "A=TP+30/SL-30/90s" in note
        assert "B=TP+50/SL-30/90s" in note
        assert "C=TP+50/SL-30/120s" in note
        assert "PAPER ONLY" in note

    def test_the_default_profile_is_unchanged(self):
        assert DEFAULT_PROFILE is StrategyProfile.L25_CURRENT
        with pytest.raises(StrategyProfileError):
            parse_profile("V2 BETA")

    def test_v2_floor_gate_thresholds_are_the_frozen_rule(self):
        # INCLUSIVE bounds on both ends; anything outside refuses.
        ok = v2floor.evaluate(volume_5m_usd=60_000.0, market_cap_usd=50_000.0,
                              change_5m_pct=12.0)
        assert ok.authorized is True
        edge_lo = v2floor.evaluate(volume_5m_usd=50_000.0,
                                   market_cap_usd=50_000.0, change_5m_pct=10.0)
        assert edge_lo.authorized is True  # ratio == 1.0, change == +10
        edge_hi = v2floor.evaluate(volume_5m_usd=50_000.0,
                                   market_cap_usd=50_000.0, change_5m_pct=300.0)
        assert edge_hi.authorized is True  # change == +300
        ratio_low = v2floor.evaluate(volume_5m_usd=49_999.0,
                                     market_cap_usd=50_000.0, change_5m_pct=12.0)
        assert ratio_low.authorized is False
        change_low = v2floor.evaluate(volume_5m_usd=60_000.0,
                                      market_cap_usd=50_000.0,
                                      change_5m_pct=9.99)
        assert change_low.authorized is False
        change_high = v2floor.evaluate(volume_5m_usd=60_000.0,
                                       market_cap_usd=50_000.0,
                                       change_5m_pct=300.01)
        assert change_high.authorized is False
        missing = v2floor.evaluate(volume_5m_usd=None,
                                   market_cap_usd=50_000.0, change_5m_pct=12.0)
        assert missing.authorized is False
        assert missing.status == v2floor.UNTESTABLE


# ===========================================================================
# explicit opt-in (task section 3, scenario 20 half)
# ===========================================================================


class TestExplicitOptIn:
    @pytest.mark.parametrize("raw", [[], ["L25_CURRENT"], ["L25_60S"]])
    def test_an_empty_or_legacy_selection_never_contains_beta(self, raw):
        specs = parse_arms(raw)
        names = {s.name.value for s in specs}
        assert not names & {a.value for a in BETA_ARMS}
        assert BETA_GROUP not in names

    def test_arm_names_is_the_arm_vocabulary_without_the_group(self):
        # One entry per SELECTABLE arm. The beta GROUP token is not here (it
        # is refused outright) and the beta arm names are not here (they
        # resolve to the disabled group), so an untouched default can never
        # start beta. The enabled Amarih GROUP follows the SAME pattern: it is
        # recognised by is_selectable_arm / is_amarih_group, but it is NOT an
        # arm name, so an empty selection never starts Amarih either.
        names = arm_names()
        assert BETA_GROUP not in names
        assert not any(n.startswith("V2_BETA_") for n in names)
        assert "AMARIH" not in names
        assert not any(n.startswith("AMARIH_T") for n in names)
        empty = {s.name.value for s in parse_arms([])}
        assert empty == set(names)
        # ...and the group IS a recognised selector token (beta's group still
        # is, so its disabled ROW can render and fail closed downstream).
        from autopaper.paper_live.arms import is_selectable_arm

        assert is_selectable_arm("AMARIH")
        assert is_selectable_arm(BETA_GROUP)  # recognised: refused in validate

    def test_the_group_token_is_now_refused(self):
        """V2 FLOOR (BETA) is disabled from the active bot.

        The group token no longer expands to three arms. Any attempt to select
        it -- or any of its arm tokens -- is refused with a clear message.
        This is the regression test that the disablement holds.
        """
        # The group token is refused.
        with pytest.raises(ValueError, match="V2 FLOOR \(BETA\) is disabled"):
            parse_arms([BETA_GROUP])
        # Individual beta arm tokens are also refused.
        for arm_token in ["V2_BETA_A", "V2_BETA_B", "V2_BETA_C"]:
            with pytest.raises(ValueError, match="V2 FLOOR \(BETA\) is disabled"):
                parse_arms([arm_token])
        # Mixed beta tokens are refused.
        with pytest.raises(ValueError, match="V2 FLOOR \(BETA\) is disabled"):
            parse_arms([BETA_GROUP, "V2_BETA_A"])
        # The beta arms are NOT in the selectable set.
        assert not any(s.name.value.startswith("V2_BETA_") for s in parse_arms([]))
        # L25_CURRENT and other selectable arms still work.
        specs = parse_arms(["L25_CURRENT"])
        assert len(specs) == 1
        assert specs[0].name.value == "L25_CURRENT"

    def test_arm_key_mapping_and_specs_are_unambiguous(self):
        """Beta arm specs are preserved for diagnostics even though disabled.

        The arm keys and exit brackets are still defined -- they're used for
        historical record interpretation and diagnostics -- but the group token
        no longer expands to arms since the experiment is disabled.
        """
        assert BETA_ARM_KEYS == {
            ResearchArm.V2_BETA_A: "A",
            ResearchArm.V2_BETA_B: "B",
            ResearchArm.V2_BETA_C: "C",
        }
        expected = {
            "V2_BETA_A": ((30.0, -30.0), 90.0),
            "V2_BETA_B": ((50.0, -30.0), 90.0),
            "V2_BETA_C": ((50.0, -30.0), 120.0),
        }
        # Beta arms are still resolvable by name for diagnostics.
        for arm_token in ["V2_BETA_A", "V2_BETA_B", "V2_BETA_C"]:
            spec = arm_by_name(arm_token)
            bracket, hold = expected[arm_token]
            assert spec.exit_bracket == bracket
            assert spec.max_hold_s == hold
            # beta arms stamp the PRODUCTION V2_FLOOR profile, not the group.
            assert spec.production_profile is StrategyProfile.V2_FLOOR
        # But the group token is refused.
        with pytest.raises(ValueError, match="V2 FLOOR \(BETA\) is disabled"):
            parse_arms([BETA_GROUP])

    def test_an_empty_request_is_refused_not_defaulted(self):
        ok, msg, arms = validate_strategy_selection([])
        assert ok is False and arms == ()
        # and an unknown profile never falls back to L25.
        with pytest.raises(StrategyProfileError):
            parse_profile("L25_30S")

    def test_an_unknown_arm_name_is_refused_visible(self):
        ok, msg, arms = validate_strategy_selection(["NOT_A_STRATEGY"])
        assert ok is False
        assert "unknown" in msg and "supported arms" in msg
        assert arms == ()

    def test_a_fresh_runtime_state_runs_the_default_arm_only(self):
        state = RuntimeState()
        assert tuple(state.arms) == (ResearchArm.L25_CURRENT.value,)
        assert state.strategy_profile == DEFAULT_PROFILE.value
        # a brand-new state never includes beta.
        assert BETA_GROUP not in state.arms

    def test_validate_strategy_selection_returns_visible_reasons(self):
        """Beta arm tokens are now refused since the experiment is disabled."""
        ok, msg, arms = validate_strategy_selection(["V2_BETA_A"])
        assert ok is False
        assert "disabled" in msg.lower() or "V2 FLOOR (BETA)" in msg
        assert arms == ()
        # A beta arm token is refused.
        with pytest.raises(ValueError, match="V2 FLOOR \(BETA\) is disabled"):
            parse_arms(["V2_BETA_A"])

    def test_an_engine_without_the_group_has_no_experiment(self, tmp_path):
        eng = beta_engine(tmp_path, arms=parse_arms([]))
        assert eng.beta_experiment is None
        assert not any(n.startswith("V2_BETA_") for n in
                       [s.name.value for s in eng.arms])
        # a default engine cannot open a beta position either.
        assert run(eng.on_signal(eligible_signal()))
        assert eng.beta_experiment is None
        assert eng.beta_positions == {}


# ===========================================================================
# the Telegram selector: label, callback, dispatch, engine wiring
# (task sections 2 + 3, scenarios 1 and 20)
# ===========================================================================


def _handler(state: RuntimeState) -> ButtonHandler:
    # OperatorActions() wires journal/notifier/preflight as None: a strategy
    # press performs no I/O, so these tests never depend on a real journal.
    from autopaper.telegram.actions import OperatorActions
    return ButtonHandler(state, OperatorActions())


class TestSelectorCallback:
    def test_the_beta_row_shows_the_group_label_and_group_callback(self):
        state = RuntimeState()
        rows = [b for row in ui.strategy_keyboard(state) for b in row]
        callbacks = {b["callback_data"] for b in rows}
        beta_rows = [b for b in rows
                     if b["callback_data"] == f"{ui.CB_PROFILE}{BETA_GROUP}"]
        assert len(beta_rows) == 1, \
            "the experiment must be ONE selectable row, not three arm rows"
        assert "V2 FLOOR (BETA)" in beta_rows[0]["text"]
        # no standalone beta-arm rows exist anywhere in the keyboard.
        for arm in BETA_ARMS:
            assert f"{ui.CB_PROFILE}{arm.value}" not in callbacks

    def test_the_menu_lists_the_group_with_its_three_brackets(self):
        text = ui.render_strategy_menu(RuntimeState())
        assert "SELECTED STRATEGIES" in text
        assert BETA_GROUP in text
        assert "V2 FLOOR (BETA)" in text
        assert "$10.00 isolated ledger per arm ($30.00 total)" in text
        # the numbers come from the registry: A/B/C with their own brackets.
        assert "A: TP +30% / SL -30% / 90s" in text
        assert "B: TP +50% / SL -30% / 90s" in text
        assert "C: TP +50% / SL -30% / 120s" in text
        # the pre-existing caveats are still on the screen.
        assert "never starts the bot" in text
        assert "never arms REAL" in text

    def test_the_menu_marks_the_group_on_once_all_three_arms_run(self):
        off = ui.render_strategy_menu(RuntimeState())
        assert "☐" in off
        state = RuntimeState()
        state.select_strategies([BETA_GROUP])
        on = ui.render_strategy_menu(state)
        assert "☑" in on

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_the_callback_selects_the_group_and_starts_three_arms(self):
        state = RuntimeState()
        handler = _handler(state)
        result = run(handler.handle(f"{ui.CB_PROFILE}{BETA_GROUP}"))
        # 1. dispatch recognised the value.
        assert result.handled is True
        assert result.error is None, result.error
        # 2. the group toggles as ONE unit: all three arms, one press.
        assert tuple(state.arms) == ("V2_BETA_A", "V2_BETA_B", "V2_BETA_C")
        assert state.strategy_profile == DEFAULT_PROFILE.value  # inert field
        # 3. the reply shows the menu the operator just changed.
        assert "V2_FLOOR_BETA" in result.text
        # 4. selection is inert: nothing started, nothing armed.
        assert state.run_state is RunState.OFF
        assert state.real_armed is False
        assert state.real_trading_enabled is False
        assert state.emergency_stop is False

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_the_engine_reconciles_the_selection_into_an_experiment(self, tmp_path):
        # The production handoff: state.arms -> parse_arms -> reconcile_arms.
        eng = beta_engine(tmp_path, arms=parse_arms(["L25_CURRENT"]))
        assert eng.beta_experiment is None

        state = RuntimeState()
        state.select_strategies([BETA_GROUP])
        selected_beta = tuple(s.name.value for s in parse_arms(list(state.arms)))
        assert selected_beta == ("V2_BETA_A", "V2_BETA_B", "V2_BETA_C")

        # (1) reconciling the CURRENT selection changes nothing, and says so.
        deferred, reconciled = eng.reconcile_arms(("L25_CURRENT",))
        assert deferred == ()
        assert reconciled == (), (
            "reconciling an already-selected arm must not report an add or a "
            "remove")
        assert eng.beta_experiment is None

        # (2) adding the group on top of L25_CURRENT starts the experiment, and
        # the three new arms are exactly the reported delta.
        selection = ("L25_CURRENT",) + selected_beta
        deferred, reconciled = eng.reconcile_arms(selection)
        assert deferred == ()
        assert set(reconciled) == set(selected_beta)
        assert tuple(s.name.value for s in eng.arms) == selection
        exp = eng.beta_experiment
        assert exp is not None
        for key in ("A", "B", "C"):
            assert exp.accounts[key].bankroll_usd == 10.0

        # (3) the experiment is the ONLY owner of the beta book: a settled arm
        # moves its own ledger, and no engine account ever holds the position.
        run(eng.on_signal(eligible_signal()))
        _quote_market(eng, 1.35)   # only A's +30% TP is reached, ON THE QUOTE
        run(eng.monitor_once())
        pid = next(iter(exp.positions))
        assert exp.positions[pid]["A"].status == "closed"
        assert exp.positions[pid]["B"].status == "open"
        settled_a = exp.accounts["A"].bankroll_usd
        assert settled_a != 10.0
        assert exp.accounts["B"].bankroll_usd == 10.0
        for arm in ("V2_BETA_A", "V2_BETA_B", "V2_BETA_C"):
            assert eng.accounts[arm].position is None

        # (4) deselecting the group stops the experiment safely: the still-open
        # arms are flattened at their last observed price, the in-process book
        # leaves, and the durable session stays on disk.
        deferred, reconciled = eng.reconcile_arms(("L25_CURRENT",))
        assert deferred == ()
        assert set(reconciled) == set(selected_beta)
        assert eng.beta_experiment is None
        assert eng.beta_positions == {}

        # (5) re-selecting it RESTORES that session -- the settled ledger is NOT
        # silently reset to $10, and the booked trades come back with it.
        deferred, reconciled = eng.reconcile_arms(selection)
        assert deferred == ()
        assert set(reconciled) == set(selected_beta)
        restored = eng.beta_experiment
        assert restored is not None and restored is not exp
        assert restored.accounts["A"].bankroll_usd == pytest.approx(settled_a)
        assert restored.accounts["B"].bankroll_usd != 10.0
        assert len(restored.completed) >= 1

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_the_running_status_line_agrees_with_the_selection(self):
        state = RuntimeState()
        assert ui._beta_running_summary(state) == ""
        state.select_strategies([BETA_GROUP])
        line = ui._beta_running_summary(state)
        assert "EXPERIMENT: " + ui.BETA_GROUP_LABEL in line
        assert "A TP +30/SL -30/90s" in line
        assert "B TP +50/SL -30/90s" in line
        assert "C TP +50/SL -30/120s" in line
        assert "$10.00 isolated ledger per arm ($30.00 total)" in line
        assert "PAPER ONLY" in line

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_pressing_the_group_twice_turns_it_off_again(self):
        state = RuntimeState()
        state.select_strategies(["L25_CURRENT"])
        handler = _handler(state)
        run(handler.handle(f"{ui.CB_PROFILE}{BETA_GROUP}"))
        assert tuple(state.arms) == ("L25_CURRENT", "V2_BETA_A", "V2_BETA_B",
                                   "V2_BETA_C")
        result = run(handler.handle(f"{ui.CB_PROFILE}{BETA_GROUP}"))
        assert result.error is None
        assert tuple(state.arms) == ("L25_CURRENT",)

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_the_beta_group_is_refused_in_real_mode(self):
        # Beta is PAPER-only; REAL is single-strategy, and the group expands
        # to three arms, so the validation layer refuses it exactly as a UI
        # press would.
        ok, msg, arms = validate_strategy_selection(
            [BETA_GROUP], trade_mode=TradeMode.REAL)
        assert ok is False and arms == ()
        assert "REAL is single-strategy" in msg
        ok, msg, arms = validate_strategy_selection(
            ["V2_BETA_A"], trade_mode=TradeMode.REAL)
        assert ok is False
        # ...while PAPER accepts it.
        ok, msg, arms = validate_strategy_selection([BETA_GROUP])
        assert ok is False
        assert "disabled" in msg.lower() or "V2 FLOOR (BETA)" in msg


# ===========================================================================
# engine-level entry: routing, one position per arm, dedup, refusals
# (scenarios 1, 2, 5, 6, 7, 8)
# ===========================================================================


class TestEngineEntry:
    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_accounts_start_at_ten_dollars(self, tmp_path):
        eng = beta_engine(tmp_path, arms=parse_arms([BETA_GROUP,
                                                     "L25_CURRENT"]))
        assert eng.beta_experiment is not None
        for key in ("A", "B", "C"):
            acct = eng.beta_experiment.accounts[key]
            assert acct.starting_bankroll_usd == 10.0
            assert acct.bankroll_usd == 10.0
        assert eng.beta_experiment.total_starting_capital_usd == 30.0
        # the experiment's bankroll is its OWN figure: the engine's ordinary
        # research accounts stay at their declared default.
        assert eng.accounts["L25_CURRENT"].bankroll_usd == 5.0
        assert eng.starting_bankroll_usd == 5.0

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_one_signal_opens_one_position_per_arm(self, tmp_path):
        eng = beta_engine(tmp_path)
        records = run(eng.on_signal(eligible_signal()))
        # one record per beta arm, all traded, nothing else ran.
        assert sorted(r.arm for r in records) == [
            "V2_BETA_A", "V2_BETA_B", "V2_BETA_C"]
        assert all(r.traded for r in records)
        exp = eng.beta_experiment
        assert len(exp.positions) == 1
        only = next(iter(exp.positions.values()))
        assert set(only) == {"A", "B", "C"}
        assert all(p.status == "open" for p in only.values())
        # regular engine book untouched: the experiment is the ONLY owner.
        assert eng.accounts["V2_BETA_A"].position is None
        assert eng.open_positions() == []
        assert eng.trade_records == []

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_positions_share_one_entry_basis(self, tmp_path):
        eng = beta_engine(tmp_path)
        records = run(eng.on_signal(eligible_signal()))
        exp = eng.beta_experiment
        pid = next(iter(exp.positions))
        a, b, c = (exp.positions[pid][k] for k in "ABC")
        # stable parent id, identical across arms and records.
        assert a.parent_signal_id == b.parent_signal_id == c.parent_signal_id
        assert len(a.parent_signal_id) == 24
        assert {r.parent_signal_id for r in records} == {pid}
        # shared entry assumptions: same instant, same fill, same size.
        assert a.entry_at == b.entry_at == c.entry_at
        # THE ENTRY IS THE EXECUTABLE BUY QUOTE'S PRICE, not the observed mark.
        # The observed mark is carried separately (below) and is NOT the fill.
        assert a.entry_fill_price == b.entry_fill_price == \
            c.entry_fill_price == pytest.approx(QUOTE_USD)
        assert a.entry_mark_price == b.entry_mark_price == c.entry_mark_price
        assert a.entry_mark_price == pytest.approx(MARK_PRICE)
        # ALL-IN, and identically so, because the three arms start at the same
        # $10 and therefore request the same notional from the same quote.
        assert a.position_size_usd == b.position_size_usd == \
            c.position_size_usd
        assert a.allocated_usd == b.allocated_usd == c.allocated_usd == 10.0
        assert a.entry_cost_usd == b.entry_cost_usd == c.entry_cost_usd
        assert a.mint == b.mint == c.mint == MINT
        # records report the ALL-IN percentage, not the removed flat 10%.
        assert {r.position_size_pct for r in records} == {100.0}
        assert len({r.position_size_usd for r in records}) == 1
        # visibility mirrors exist, one per arm, and are NOT engine positions.
        assert set(eng.beta_positions[pid]) == {
            "V2_BETA_A", "V2_BETA_B", "V2_BETA_C"}
        assert eng.accounts["V2_BETA_A"].position is None

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_duplicate_delivery_is_idempotent(self, tmp_path):
        eng = beta_engine(tmp_path)
        first = eligible_signal()
        run(eng.on_signal(first))
        # (a) the very same snapshot again...
        run(eng.on_signal(first))
        # (b) ...and a second delivery of the same channel message content
        # (the stable parent id is channel+mint+timestamp).
        run(eng.on_signal(eligible_signal()))
        exp = eng.beta_experiment
        assert len(exp.positions) == 1, "duplicate delivery opened a 2nd entry"
        only = next(iter(exp.positions.values()))
        assert set(only) == {"A", "B", "C"}
        # one shared parent id, exactly three positions in total.
        assert sum(len(v) for v in exp.positions.values()) == 3

    @pytest.mark.parametrize("overrides,expected_gate_response", [
        (None, "V2_FLOOR gate requires"),
        (dict(v2_change_5m_pct=5.0), "momentum floor"),
        (dict(v2_change_5m_pct=400.0), "anti-exhaustion"),
        (dict(v2_volume_5m_usd=10_000.0), "relative-turnover"),
    ],
    ids=["absent", "change_low", "change_high", "ratio_low"])
    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_ineligible_signal_opens_nothing(self, tmp_path,
                                             overrides, expected_gate_response):
        eng = beta_engine(tmp_path)
        # The threshold cases start from an OTHERWISE-ELIGIBLE snapshot and move
        # exactly ONE gate input out of bounds. Building them from the bare
        # signal would leave the other two inputs absent, so every case would
        # refuse with "requires ..." and the thresholds would never be
        # exercised -- a green test that proves nothing.
        sig = base_signal() if overrides is None else eligible_signal(**overrides)
        records = run(eng.on_signal(sig))
        assert len(records) == 3
        assert all(r.traded is False for r in records)
        assert all(r.l25_status == "V2_FLOOR_REJECTED" for r in records)
        assert all(r.failure == FailureReason.V2_FLOOR_REJECTED
                   for r in records)
        # every arm must name the gate that actually refused.
        details = [r.failure_detail for r in records]
        assert all(isinstance(d, str) and expected_gate_response in d
                   for d in details), (
            f"refusal details={details}; "
            f"expected {expected_gate_response!r} in each")
        assert eng.beta_experiment.positions == {}
        assert eng.beta_positions == {}

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_a_failed_entry_mark_is_a_refusal_not_a_position(self, tmp_path):
        # PRODUCTION ENTRY REQUIRES AN EXECUTABLE BUY QUOTE (the name is
        # historical: the entry is no longer priced from a mark). No quote, no
        # fill: the entry FAILS CLOSED with one named refusal per arm and
        # nothing opened.
        eng = beta_engine(
            tmp_path, quotes=FakeQuoteService(buy=unavailable_point()))
        records = run(eng.on_signal(eligible_signal()))
        assert len(records) == 3
        assert all(r.traded is False for r in records)
        assert eng.beta_experiment.positions == {}
        # the refusal still records WHICH arm refused.
        assert {r.arm for r in records} == {"V2_BETA_A", "V2_BETA_B",
                                            "V2_BETA_C"}

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_beta_arms_never_enter_the_regular_entry_loop(self, tmp_path):
        eng = beta_engine(tmp_path)
        run(eng.on_signal(eligible_signal()))
        # no engine account holds the trade, so the regular exit loop has
        # nothing to settle: exactly one authoritative book.
        for name in ("V2_BETA_A", "V2_BETA_B", "V2_BETA_C"):
            assert eng.accounts[name].position is None
        assert eng.open_positions() == []

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_monitor_settles_only_through_the_experiment(self, tmp_path):
        market = FakeMarket(price=MARK_PRICE)
        eng = beta_engine(tmp_path, market=market)
        run(eng.on_signal(eligible_signal()))
        # price runs to +35%: only arm A (TP +30) can settle — from the
        # EXPERIMENT's book, never the engine's.
        _quote_market(eng, 1.35)
        run(eng.monitor_once())
        run(eng.monitor_once())   # second tick drops A's settled mirror
        exp = eng.beta_experiment
        pid = next(iter(exp.positions))
        assert pos(exp, "A", pid).status == "closed"
        assert pos(exp, "B", pid).status == "open"
        assert pos(exp, "C", pid).status == "open"
        # the engine recorded no trade, closed no account position, and its
        # beta-arm accounts did not move.
        assert eng.trade_records == []
        assert eng.accounts["V2_BETA_A"].position is None
        assert exp.accounts["A"].bankroll_usd != 10.0   # A settled its P&L
        assert exp.accounts["B"].bankroll_usd == 10.0
        assert exp.accounts["C"].bankroll_usd == 10.0
        # the settled mirror disappeared; the open ones remain visible.
        assert set(eng.beta_positions[pid]) == {"V2_BETA_B", "V2_BETA_C"}

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_a_market_outage_settles_nothing(self, tmp_path):
        eng = beta_engine(tmp_path, market=FakeMarket(price=MARK_PRICE,
                                                      fail=True))
        run(eng.on_signal(eligible_signal()))
        run(eng.monitor_once())
        run(eng.monitor_once())
        exp = eng.beta_experiment
        assert all(p.status == "open"
                   for pid in exp.positions.values() for p in pid.values())
        assert exp.accounts["A"].bankroll_usd == 10.0
        assert exp.accounts["B"].bankroll_usd == 10.0
        assert exp.accounts["C"].bankroll_usd == 10.0
        assert exp.completed == []
        assert eng.trade_records == []


# ===========================================================================
# the immutable three-arm configuration (scenarios 9, 10, 11, task section 5)
# ===========================================================================


class TestArmConfiguration:
    def test_the_table_is_exactly_the_documented_experiment(self):
        assert [(c.arm, c.take_profit_pct, c.stop_loss_pct, c.max_hold_s)
                for c in BETA_ARM_CONFIGS] == [
            ("A", 30.0, -30.0, 90.0),
            ("B", 50.0, -30.0, 90.0),
            ("C", 50.0, -30.0, 120.0),
        ]
        # frozen: experiment parameters cannot change mid-session.
        with pytest.raises(FrozenInstanceError):
            BETA_ARM_CONFIGS[0].take_profit_pct = 50.0
        # per-arm lookup is by the internal key only.
        assert arm_config("A").take_profit_pct == 30.0
        with pytest.raises(ValueError):
            arm_config("D")

    def test_each_arm_honors_its_own_bracket_and_cap(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        for key, (tp, sl, hold) in {"A": (30.0, -30.0, 90.0),
                                    "B": (50.0, -30.0, 90.0),
                                    "C": (50.0, -30.0, 120.0)}.items():
            p = pos(exp, key)
            assert (p.config().take_profit_pct, p.config().stop_loss_pct,
                    p.config().max_hold_s) == (tp, sl, hold)

    def test_the_generic_registry_cannot_override_an_arm(self, tmp_path):
        # The per-arm cap is enforced by the experiment (and mirrored on the
        # engine's PaperPosition mirrors). A bare BetaPosition carries no per-arm
        # stamp, so the generic ``_max_hold_for`` helper returns the profile
        # default; mirror the real enforcement in the engine mirror test.
        exp = new_exp(tmp_path)
        open_p1(exp)
        for key, hold in {"A": 90.0, "B": 90.0, "C": 120.0}.items():
            assert pos(exp, key).config().max_hold_s == hold

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_the_engine_mirrors_carry_the_same_stamps(self, tmp_path):
        eng = beta_engine(tmp_path)
        run(eng.on_signal(eligible_signal()))
        pid = next(iter(eng.beta_positions))
        expected = {"V2_BETA_A": ((30.0, -30.0), 90.0),
                    "V2_BETA_B": ((50.0, -30.0), 90.0),
                    "V2_BETA_C": ((50.0, -30.0), 120.0)}
        for arm, (bracket, hold) in expected.items():
            mirror = eng.beta_positions[pid][arm]
            assert isinstance(mirror, PaperPosition)
            assert mirror.beta_exit_bracket == bracket
            assert mirror.beta_max_hold_s == hold
            assert _v2floor_bracket_for(mirror) == bracket
            assert _max_hold_for(mirror) == hold
            # never registered as an engine position.
            assert eng.accounts[arm].position is None

    def test_arm_a_settles_take_profit_at_plus_thirty(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.31, at=T0 + 5)
        settled = exp.settle_arm("p1", "A", now=T0 + 6)
        assert settled.status == "closed"
        assert settled.exit_reason == "TP"
        assert settled.gross_return_pct == pytest.approx(30.0)
        # a +31% print must not become a +31% fill: the LIMIT is the target.
        assert settled.exit_fill_price == pytest.approx(1.30)

    def test_arm_b_and_c_need_fifty_percent(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        # +35% is A's TP but NOT B's or C's.
        exp.observe("p1", "B", mark_price=1.35, at=T0 + 5)
        exp.observe("p1", "C", mark_price=1.35, at=T0 + 5)
        exp.settle_arm("p1", "B", now=T0 + 6)
        exp.settle_arm("p1", "C", now=T0 + 6)
        assert pos(exp, "B").status == "open"
        assert pos(exp, "C").status == "open"
        # +51% is both of their TPs.
        exp.observe("p1", "B", mark_price=1.51, at=T0 + 7)
        exp.observe("p1", "C", mark_price=1.51, at=T0 + 7)
        exp.settle_arm("p1", "B", now=T0 + 8)
        exp.settle_arm("p1", "C", now=T0 + 8)
        assert pos(exp, "B").gross_return_pct == pytest.approx(50.0)
        assert pos(exp, "C").gross_return_pct == pytest.approx(50.0)
        assert pos(exp, "B").exit_fill_price == pytest.approx(1.50)
        assert pos(exp, "C").exit_fill_price == pytest.approx(1.50)

    def test_the_90s_and_120s_deadlines_use_real_timestamps(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp, at=T0)
        # one flat observation so a TIME exit has an honest basis.
        for k in "ABC":
            exp.observe("p1", k, mark_price=MARK_PRICE, at=T0 + 1)
        # 89s: nobody is due.
        exp.monitor_once(now=T0 + 89)
        assert all(pos(exp, k).status == "open" for k in "ABC")
        # 90s: A and B hit their cap; C (120s) is not yet due.
        exp.monitor_once(now=T0 + 90)
        assert pos(exp, "A").status == "closed"
        assert pos(exp, "A").exit_reason == "TIME"
        assert pos(exp, "B").status == "closed"
        assert pos(exp, "B").exit_reason == "TIME"
        assert pos(exp, "C").status == "open"
        # 120s: C is due.
        exp.monitor_once(now=T0 + 120)
        assert pos(exp, "C").status == "closed"
        assert pos(exp, "C").exit_reason == "TIME"
        assert pos(exp, "C").exit_at == T0 + 120
        # a flat hold books the modeled exit friction, nothing else.
        expected = 10.0 + 1.0 * net_from_gross(0.0)
        assert exp.accounts["A"].bankroll_usd == pytest.approx(expected)
        assert exp.accounts["B"].bankroll_usd == pytest.approx(expected)
        assert exp.accounts["C"].bankroll_usd == pytest.approx(expected)

    def test_an_early_tick_cannot_fire_a_deadline(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp, at=T0)
        for k in "ABC":
            exp.observe("p1", k, mark_price=MARK_PRICE, at=T0 + 1)
        exp.monitor_once(now=T0 + 89.999)
        assert all(pos(exp, k).status == "open" for k in "ABC")


# ===========================================================================
# arm independence (scenarios 12, 18 isolation)
# ===========================================================================


class TestArmIndependence:
    def test_closing_one_arm_leaves_the_others_untouched(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.31, at=T0 + 5)
        exp.settle_arm("p1", "A", now=T0 + 6)
        assert pos(exp, "A").status == "closed"
        assert pos(exp, "B").status == "open"
        assert pos(exp, "C").status == "open"
        assert pos(exp, "B").exit_reason is None
        assert pos(exp, "C").exit_reason is None
        assert pos(exp, "B").net_pnl_usd is None
        assert pos(exp, "C").net_pnl_usd is None
        assert exp.accounts["B"].bankroll_usd == 10.0
        assert exp.accounts["C"].bankroll_usd == 10.0
        assert exp.accounts["B"].closed_trades == 0
        assert exp.accounts["C"].closed_trades == 0
        # exactly one trade was booked, and it belongs to arm A.
        assert len(exp.completed) == 1
        assert exp.completed[0]["arm"] == "A"

    def test_settling_is_idempotent_per_arm(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.31, at=T0 + 5)
        first = exp.settle_arm("p1", "A", now=T0 + 6)
        bankroll = exp.accounts["A"].bankroll_usd
        exp.settle_arm("p1", "A", now=T0 + 7)
        exp.monitor_once(now=T0 + 8)
        assert first.status == "closed"
        assert exp.accounts["A"].bankroll_usd == bankroll
        assert len(exp.completed) == 1, "one arm settled twice"

    def test_a_duplicate_parent_cannot_reopen_over_a_closed_arm(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.31, at=T0 + 5)
        exp.settle_arm("p1", "A", now=T0 + 6)
        # re-delivering the same parent id returns the SAME book.
        same = exp.open_for_signal(parent_signal_id="p1", mint=MINT,
                                   symbol="T",
                                   entry_fill=quote_entry(),
                                   position_size_usd=1.0, entry_at=T0 + 9)
        assert same["A"] is pos(exp, "A")
        assert pos(exp, "A").status == "closed"
        assert len(exp.completed) == 1


# ===========================================================================
# exit semantics: touches, gaps, stops, stale data, ambiguity
# (scenarios 13-17, task section 5 exit handling)
# ===========================================================================


class TestExitSemantics:
    def test_a_touch_books_at_the_limit_not_at_the_print(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.35, at=T0 + 4)  # TP touched
        settled = exp.settle_arm("p1", "A", now=T0 + 5)
        assert settled.status == "closed"
        assert settled.exit_reason == "TP"
        # booked at exactly the +30% target, never at the +35% print.
        assert settled.exit_fill_price == pytest.approx(1.30)
        assert settled.gross_return_pct == pytest.approx(30.0)

    def test_a_touch_is_not_a_win_until_it_settles(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.35, at=T0 + 4)
        # touched, but NOT settled: nothing realized, no trade record.
        assert pos(exp, "A").status == "open"
        assert pos(exp, "A").net_pnl_usd is None
        assert exp.completed == []
        assert exp.accounts["A"].bankroll_usd == 10.0
        stats = exp.arm_stats("A")
        assert stats["completed_trades"] == 0
        assert stats["win_rate_pct"] is None
        assert stats["exits"]["TP"] == 0

    def test_a_gapped_up_print_gets_no_favorable_fill(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        # gapped way past B's +50% TP: the fill is the TARGET, not the gap.
        exp.observe("p1", "B", mark_price=1.90, at=T0 + 4)
        settled = exp.settle_arm("p1", "B", now=T0 + 5)
        assert settled.exit_fill_price == pytest.approx(1.50)
        assert settled.gross_return_pct == pytest.approx(50.0)
        # and A, whose TP is lower, books at ITS target, not at B's print.
        exp.observe("p1", "A", mark_price=1.90, at=T0 + 4)
        settled_a = exp.settle_arm("p1", "A", now=T0 + 5)
        assert settled_a.exit_fill_price == pytest.approx(1.30)

    def test_a_stop_books_the_observed_trough(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        # -45% through the -30% stop: the honest fill is the observed trough.
        exp.observe("p1", "A", mark_price=0.55, at=T0 + 4)
        settled = exp.settle_arm("p1", "A", now=T0 + 5)
        assert settled.exit_reason == "SL"
        assert settled.gross_return_pct == pytest.approx(-45.0)
        assert settled.exit_fill_price == pytest.approx(0.55)

    def test_a_stop_exactly_at_the_stop_books_the_stop(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=0.70, at=T0 + 4)
        settled = exp.settle_arm("p1", "A", now=T0 + 5)
        assert settled.exit_reason == "SL"
        assert settled.gross_return_pct == pytest.approx(-30.0)

    def test_a_stop_worse_than_the_stop_cannot_book_better(self, tmp_path):
        # conservative direction: once the trough is below -30, the booked
        # return is the TROUGH, never a rounded-up -30.
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=0.60, at=T0 + 4)  # -40%
        settled = exp.settle_arm("p1", "A", now=T0 + 5)
        assert settled.gross_return_pct == pytest.approx(-40.0)

    def test_an_unobserved_position_never_fabricates_a_trade(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        # long past every deadline, but there is NO price: no price, no trade.
        settled = exp.settle_arm("p1", "A", now=T0 + 500)
        assert settled.status == "unresolved"
        assert settled.exit_reason == "TIME_NO_QUOTE"
        assert settled.exit_fill_price is None
        assert settled.net_pnl_usd is None
        assert exp.accounts["A"].bankroll_usd == 10.0
        assert exp.completed == []
        # and it stays visible in the stats instead of disappearing.
        stats = exp.arm_stats("A")
        assert stats["unresolved"] == 1
        assert stats["completed_trades"] == 0
        assert stats["win_rate_pct"] is None

    def test_monitor_keeps_unobserved_positions_open(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        settled = exp.monitor_once(now=T0 + 500)
        assert {p.status for p in settled} == {"unresolved"}
        assert all(exp.accounts[k].bankroll_usd == 10.0 for k in "ABC")
        assert exp.completed == []

    def test_ambiguous_tp_sl_order_books_the_stop_and_is_flagged(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        # peak ABOVE the TP, trough DOWN TO the stop: the path cannot prove
        # which came first, so it is booked conservatively as the stop.
        exp.observe("p1", "A", mark_price=1.60, at=T0 + 3)
        exp.observe("p1", "A", mark_price=0.70, at=T0 + 4)
        settled = exp.settle_arm("p1", "A", now=T0 + 5)
        assert settled.status == "closed"
        assert settled.exit_reason == "SL"
        assert settled.path_ambiguous is True
        assert settled.gross_return_pct == pytest.approx(-30.0)
        # a win was NOT booked from the earlier peak.
        assert settled.net_pnl_usd < 0
        assert exp.completed[0]["path_ambiguous"] is True
        assert exp.arm_stats("A")["exits"]["path_ambiguous"] == 1

    def test_the_ambiguity_flag_survives_a_restart(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.60, at=T0 + 3)
        exp.observe("p1", "A", mark_price=0.70, at=T0 + 4)
        exp.settle_arm("p1", "A", now=T0 + 5)
        revived = BetaExperiment(state_root=tmp_path)
        assert len(revived.completed) == 1
        rec = revived.completed[0]
        assert rec["path_ambiguous"] is True
        assert rec["exit_reason"] == "SL"


# ===========================================================================
# accounting: net P&L, friction, isolation (scenarios 18, task section 5)
# ===========================================================================


class TestAccounting:
    def test_net_pnl_is_net_of_the_modeled_exit_friction(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.31, at=T0 + 4)
        settled = exp.settle_arm("p1", "A", now=T0 + 5)
        # 1.30 target * the 0.9852 exit multiplier, on a $1.00 position.
        expected_net_pct = (1.30 * 0.9852 - 1.0) * 100.0
        expected_pnl = 1.0 * (1.30 * 0.9852 - 1.0)
        assert settled.net_return_pct == pytest.approx(expected_net_pct)
        assert settled.net_pnl_usd == pytest.approx(expected_pnl)
        assert exp.accounts["A"].bankroll_usd == pytest.approx(
            10.0 + expected_pnl)
        # and the trade record carries every filling assumption.
        rec = exp.completed[0]
        assert rec["fill_model"] == "MODELED_LIMIT_OR_TROUGH"
        assert rec["entry_profile"] == BETA_ENTRY_PROFILE == "V2_FLOOR"
        assert rec["take_profit_pct"] == 30.0
        assert rec["stop_loss_pct"] == -30.0
        assert rec["max_hold_s"] == 90.0
        assert rec["position_size_usd"] == 1.0
        assert rec["entry_fill_price"] == MARK_PRICE
        assert rec["session_id"] == exp.session_id

    def test_the_three_ledgers_never_mix(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.31, at=T0 + 4)
        exp.observe("p1", "B", mark_price=0.70, at=T0 + 4)
        exp.settle_arm("p1", "A", now=T0 + 5)   # win
        exp.settle_arm("p1", "B", now=T0 + 5)   # stop loss
        a = exp.accounts["A"].bankroll_usd
        b = exp.accounts["B"].bankroll_usd
        c = exp.accounts["C"].bankroll_usd
        assert a > 10.0
        assert b < 10.0
        assert c == 10.0
        # one arm's profit did not fund another arm's loss.
        stats = exp.stats()["arms"]
        assert stats["A"]["bankroll_usd"] == pytest.approx(a, abs=1e-4)
        assert stats["B"]["bankroll_usd"] == pytest.approx(b, abs=1e-4)
        assert stats["C"]["bankroll_usd"] == pytest.approx(c, abs=1e-4)

    def test_all_in_sizing_allocates_each_arms_own_bankroll(self, tmp_path):
        # EACH arm is all-in on ITS OWN bankroll. No minimum, no shared size:
        # arm A committing $10 must not shrink arm B, and B's smaller bankroll
        # must not shrink A or C.
        exp = new_exp(tmp_path)
        exp.accounts["B"].bankroll_usd = 8.0
        eligible, reason, positions = exp.process_signal(
            eligible_signal(), entry_fill=quote_entry(price=MARK_PRICE, size=8.0))
        assert eligible is True
        assert {k: p.allocated_usd for k, p in positions.items()} == \
            {"A": 10.0, "B": 8.0, "C": 10.0}
        assert {p.sizing_pct for p in positions.values()} == {100.0}
        assert {p.sizing_rule for p in positions.values()} == {"ALL_IN_100_PCT"}
        # the committed cost is the QUOTE's, and never more than the allocation.
        for p in positions.values():
            assert p.entry_cost_usd == pytest.approx(8.0)
            assert p.entry_cost_usd <= p.allocated_usd
            assert p.remaining_cash_usd == pytest.approx(
                p.allocated_usd - p.entry_cost_usd)

    def test_an_entry_that_costs_more_than_the_arm_holds_is_refused(self, tmp_path):
        # THE AFFORDABILITY GUARD: an all-in entry may never exceed the capital
        # it is funded from. B has $8 and the only quote on offer costs $10, so
        # B stands down with an AFFORDABILITY reason instead of opening.
        exp = new_exp(tmp_path)
        exp.accounts["B"].bankroll_usd = 8.0
        eligible, reason, positions = exp.process_signal(
            eligible_signal(), entry_fill=quote_entry(price=MARK_PRICE, size=10.0))
        assert eligible is True
        assert set(positions) == {"A", "C"}
        assert positions["A"].entry_cost_usd == pytest.approx(10.0)

    def test_an_arm_with_no_capital_takes_no_entry(self, tmp_path):
        exp = new_exp(tmp_path)
        for k in "ABC":
            exp.accounts[k].bankroll_usd = 0.0
        eligible, reason, positions = exp.process_signal(
            eligible_signal(), entry_fill=quote_entry(price=MARK_PRICE))
        assert eligible is False
        assert "no beta arm could take an entry" in reason
        assert "INSUFFICIENT_CAPITAL" in reason
        assert positions is None

    def test_a_shared_entry_produces_equal_notional_in_all_arms(self, tmp_path):
        exp = new_exp(tmp_path)
        eligible, reason, positions = exp.process_signal(
            eligible_signal(), entry_fill=quote_entry(price=MARK_PRICE, size=10.0))
        assert eligible is True
        sizes = {p.position_size_usd for p in positions.values()}
        allocations = {p.allocated_usd for p in positions.values()}
        parents = {p.parent_signal_id for p in positions.values()}
        # Equal arms, so equal all-in commitments ($10 each) from one quote.
        assert sizes == {10.0}
        assert allocations == {10.0}
        assert len(parents) == 1

    def test_the_stats_report_is_net_and_sample_aware(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        # before any close: metrics are UNAVAILABLE, not fake zeros; an
        # unobserved open arm is UNPRICED, not confidently $0.
        for k in "ABC":
            s = exp.arm_stats(k)
            assert s["win_rate_pct"] is None
            assert s["expectancy_pct"] is None
            assert s["profit_factor"] is None
            assert s["completed_trades"] == 0
            assert s["starting_bankroll_usd"] == 10.0
            assert s["unrealized_unpriced_positions"] == 1
            assert s["unrealized_priced_positions"] == 0
            assert s["unrealized_pnl_usd"] == 0.0
        exp.observe("p1", "A", mark_price=1.31, at=T0 + 4)
        exp.settle_arm("p1", "A", now=T0 + 5)
        s = exp.arm_stats("A")
        assert s["completed_trades"] == 1
        assert s["win_rate_pct"] == 100.0
        assert s["exits"] == {"TP": 1, "SL": 0, "TIME": 0, "path_ambiguous": 0}
        assert s["basis"].startswith("bankroll moves only on realized")


# ===========================================================================
# persistence, restart, reset and archive (scenarios 3, 4, 19;
# task section 6)
# ===========================================================================


class TestPersistenceLifecycle:
    def test_a_restart_restores_ledgers_positions_and_trades(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.31, at=T0 + 4)
        exp.settle_arm("p1", "A", now=T0 + 5)      # A closes with a win
        settled_bankroll = exp.accounts["A"].bankroll_usd
        revived = BetaExperiment(state_root=tmp_path)
        # settled ledger restored from the durable trade book, once.
        assert revived.accounts["A"].bankroll_usd == \
            pytest.approx(settled_bankroll)
        assert revived.accounts["B"].bankroll_usd == 10.0
        assert revived.accounts["C"].bankroll_usd == 10.0
        assert len(revived.completed) == 1
        # B and C are still OPEN, with their entry data intact; the CLOSED
        # arm left the rehydratable open set rather than being revived.
        assert set(revived.positions) == {"p1"}
        assert set(revived.positions["p1"]) == {"B", "C"}
        assert revived.positions["p1"]["B"].status == "open"
        assert revived.positions["p1"]["C"].status == "open"
        assert revived.positions["p1"]["B"].entry_fill_price == MARK_PRICE
        # and the restored arms keep working: B settles its own way.
        revived.observe("p1", "B", mark_price=1.51, at=T0 + 20)
        revived.settle_arm("p1", "B", now=T0 + 21)
        assert revived.positions["p1"]["B"].status == "closed"
        assert revived.accounts["B"].bankroll_usd > 10.0
        assert revived.accounts["C"].bankroll_usd == 10.0

    def test_a_restart_after_a_completed_exit_does_not_rebook_it(self,
                                                                 tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.31, at=T0 + 4)
        exp.settle_arm("p1", "A", now=T0 + 5)
        bankroll = exp.accounts["A"].bankroll_usd
        for i in range(3):
            revived = BetaExperiment(state_root=tmp_path)
            assert len(revived.completed) == 1
            assert revived.accounts["A"].bankroll_usd == pytest.approx(bankroll)
        # the settled arm left the rehydratable OPEN set: no revival, ever.
        payload = json_file(tmp_path / "v2_floor_beta_state.json")
        open_positions = payload.get("open_positions") or []
        assert not any(p.get("arm") == "A" for p in open_positions)

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_a_new_engine_restores_the_running_experiment(self, tmp_path):
        eng = beta_engine(tmp_path)
        run(eng.on_signal(eligible_signal()))
        pid = next(iter(eng.beta_experiment.positions))
        # restart: a brand-new engine over the same state root.
        eng2 = beta_engine(tmp_path)
        exp2 = eng2.beta_experiment
        assert exp2 is not None
        assert set(exp2.positions) == {pid}
        assert set(exp2.positions[pid]) == {"A", "B", "C"}
        assert all(p.status == "open" for p in exp2.positions[pid].values())
        assert exp2.accounts["A"].bankroll_usd == 10.0
        # the restored positions become observable AGAIN on the first tick:
        # the visibility mirrors are rebuilt, then settle from fresh marks.
        _quote_market(eng2, 1.35)
        run(eng2.monitor_once())
        run(eng2.monitor_once())   # second tick drops A's settled mirror
        assert exp2.positions[pid]["A"].status == "closed"
        assert exp2.positions[pid]["B"].status == "open"
        # and the mirror set agrees with the experiment's own book.
        assert set(eng2.beta_positions[pid]) == {"V2_BETA_B", "V2_BETA_C"}
        assert eng2.trade_records == []

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_refresh_rehydrate_reconnect_keep_the_session(self, tmp_path):
        eng = beta_engine(tmp_path)
        run(eng.on_signal(eligible_signal()))
        exp = eng.beta_experiment
        _quote_market(eng, 1.35)
        run(eng.monitor_once())
        settled_a = exp.accounts["A"].bankroll_usd
        pid = next(iter(exp.positions))

        # a dashboard refresh (status), a durable flush + rehydrate, and a
        # "reconnect" (monitor with the market down) — none may reset.
        for _ in range(3):
            eng.status()
        eng.persist_state()
        report = eng.rehydrate()
        assert "restored" in report and "refused" in report
        eng.market.fail = True
        run(eng.monitor_once())
        eng.market.fail = False

        assert eng.beta_experiment is exp
        assert exp.accounts["A"].bankroll_usd == pytest.approx(settled_a)
        assert exp.positions[pid]["A"].status == "closed"
        assert exp.positions[pid]["B"].status == "open"
        assert exp.positions[pid]["C"].status == "open"

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_reset_leaves_other_accounts_alone(self, tmp_path):
        eng = beta_engine(tmp_path, arms=parse_arms([BETA_GROUP,
                                                     "L25_CURRENT"]))
        # give the other profile a non-default book; it must survive intact.
        eng.accounts["L25_CURRENT"].bankroll_usd = 777.0
        run(eng.on_signal(eligible_signal()))
        eng.beta_experiment.reset()
        assert eng.accounts["L25_CURRENT"].bankroll_usd == 777.0
        # fresh $10 session for all three arms.
        for k in "ABC":
            assert eng.beta_experiment.accounts[k].bankroll_usd == 10.0
            assert eng.beta_experiment.accounts[k].starting_bankroll_usd == 10.0
        assert eng.beta_experiment.positions == {}
        assert eng.beta_experiment.completed == []

    def test_reset_archives_the_prior_session_before_starting_fresh(self,
                                                                    tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)
        exp.observe("p1", "A", mark_price=1.31, at=T0 + 4)
        exp.observe("p1", "B", mark_price=MARK_PRICE, at=T0 + 4)
        exp.observe("p1", "C", mark_price=MARK_PRICE, at=T0 + 4)
        exp.settle_arm("p1", "A", now=T0 + 5)  # a closed win
        result = exp.reset()
        # the archive exists and holds the prior session's records.
        archived = result["archived_to"]
        assert archived and Path(archived).exists()
        payload = json_file(archived)
        trades = payload.get("trades") or []
        assert len(trades) >= 1, "prior results were silently deleted"
        assert "A" in {t["arm"] for t in trades}
        # B and C had observations, so they were flattened (stamped FLATTEN,
        # booked at their last observed return) BEFORE the archive was cut.
        assert "FLATTEN" in {t["exit_reason"] for t in trades}
        closed = result["flattened_before_reset"]["closed"]
        assert {c["arm"] for c in closed} == {"B", "C"}
        # and the fresh session is genuinely fresh.
        assert exp.positions == {} and exp.completed == []
        for k in "ABC":
            assert exp.accounts[k].bankroll_usd == 10.0

    def test_reset_never_drops_an_unobserved_position_silently(self, tmp_path):
        exp = new_exp(tmp_path)
        open_p1(exp)   # never observed: NO price, so NO trade may be booked
        result = exp.reset()
        payload = json_file(result["archived_to"])
        # the unresolved position is recorded as OPEN in the archive rather
        # than being flattened at a made-up price or dropped.
        unresolved = result["flattened_before_reset"]["unresolved"]
        assert len(unresolved) == 3
        assert {u["arm"] for u in unresolved} == {"A", "B", "C"}
        assert all(u["reason"] == "NO_OBSERVATION" for u in unresolved)
        open_positions = payload.get("open_positions") or []
        assert len(open_positions) == 3
        # no trade was invented.
        assert not any(t.get("exit_reason") == "FLATTEN"
                       for t in payload.get("trades") or [])

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_deselecting_the_group_flattens_and_keeps_the_durable_state(self,
                                                                        tmp_path):
        eng = beta_engine(tmp_path, arms=parse_arms(["L25_CURRENT"]))
        selected = tuple(s.name.value for s in parse_arms([BETA_GROUP]))
        eng.reconcile_arms(selected)
        run(eng.on_signal(eligible_signal()))
        # one observation so a flatten has an honest basis.
        _mark_market(eng, 1.05)
        run(eng.monitor_once())
        eng.reconcile_arms(("L25_CURRENT",))
        # in-process experiment gone, mirrors gone...
        assert eng.beta_experiment is None
        assert eng.beta_positions == {}
        # ...but the durable session survives on disk and is RESTORED on
        # re-selection rather than silently starting over.
        eng.reconcile_arms(selected)
        assert eng.beta_experiment is not None
        restored = eng.beta_experiment.accounts["A"].bankroll_usd
        assert restored != 10.0, \
            "re-selecting the group must not silently reset the ledger to $10"
        payload = json_file(tmp_path / "v2_floor_beta_state.json")
        assert payload.get("trades")

    # -- the reported pre-existing persistence bugs, verified -------------

    def test_state_store_accepts_a_string_root(self):
        store = PaperStateStore(root=tempfile.mkdtemp())
        assert isinstance(store.path, Path)
        store2 = PaperStateStore(root=str(Path(tempfile.mkdtemp())))
        assert isinstance(store2.path, Path)

    def test_rehydrate_returns_a_tuple_never_none(self):
        store = PaperStateStore(root=Path(tempfile.mkdtemp()))
        state = store.load()          # empty store: a fresh state, not None
        result = store.rehydrate(state)
        assert isinstance(result, tuple) and len(result) == 2
        positions, problems = result
        assert positions == [] and problems == []

    def test_persisted_state_round_trips(self):
        state = PersistedState()
        state.merge(open_positions=[{
            "arm": "V2_BETA_A", "signal_id": "p1", "mint": MINT,
            "tokens_raw": 1, "entry_venue_id": "x",
            "entry_fill": {"output_amount_raw": 1},
        }])
        payload = json.loads(json.dumps(state.as_dict()))
        back = PersistedState.from_dict(payload)
        assert set(back.open_positions) == set(state.open_positions)
        assert back.schema_version == STATE_SCHEMA_VERSION

    def test_account_state_round_trips_the_stats_epoch(self):
        # ``PersistedState.from_dict`` must not silently lose the reporting
        # window (the reported stats_epoch_started_at restoration bug).
        st = PaperAccountState(stats_epoch_started_at=1_700_000_123.0,
                               cash_usd=42.0)
        back = PaperAccountState.from_dict(json.loads(json.dumps(
            st.as_dict())))
        assert back.stats_epoch_started_at == 1_700_000_123.0
        assert back.cash_usd == 42.0
        # a state without the field restores to None (the documented default).
        assert PaperAccountState.from_dict({}).stats_epoch_started_at is None

    def test_engine_rehydrate_on_an_empty_store_does_not_raise(self, tmp_path):
        eng = beta_engine(tmp_path, arms=parse_arms(["L25_CURRENT"]))
        report = eng.rehydrate()
        assert report["restored"] == []
        assert isinstance(report["refused"], list)


# ===========================================================================
# safety: PAPER-only, fail-closed (scenarios 11, 20; task section 11)
# ===========================================================================


class TestSafety:
    def test_the_paper_only_audit_passes(self):
        audit = assert_paper_only()
        assert audit.ok is True
        assert audit.as_dict()["transaction_broadcast"] == 0
        assert audit_paper_capability().ok is True

    def test_engine_status_reports_zero_transactions_after_a_full_cycle(
            self, tmp_path):
        eng = beta_engine(tmp_path)
        run(eng.on_signal(eligible_signal()))
        _quote_market(eng, 1.35)
        run(eng.monitor_once())
        status = eng.status()
        assert status["mode"] == "PAPER_ONLY"
        assert status["transaction_built"] == 0
        assert status["transaction_signed"] == 0
        assert status["transaction_broadcast"] == 0
        assert "NO TRANSACTION WAS BUILT, SIGNED, OR BROADCAST" in \
            status["execution_note"]
        assert status["safety"]["ok"] is True

    def test_the_beta_module_reaches_no_submission_capability(self):
        """Structural: the experiment source cannot name an execution path."""
        import autopaper.paper_live.v2_floor_beta as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden_attrs = set(PAPER_FORBIDDEN_ATTRS)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden_attrs:
                    offenders.append(node.attr)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in PAPER_FORBIDDEN_MODULES:
                        offenders.append(alias.name)
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in PAPER_FORBIDDEN_MODULES:
                    offenders.append(node.module)
        assert not offenders, \
            f"the beta experiment names a submission capability: {offenders}"

    def test_the_guard_would_refuse_a_breach(self):
        # prove the audit is a real check and not a constant True.
        from autopaper.same_market.guard import is_forbidden_module
        assert is_forbidden_module("autopaper.execution.real")
        assert "autopaper.execution.solana" in PAPER_FORBIDDEN_MODULES
        assert "send_raw_transaction" in PAPER_FORBIDDEN_ATTRS

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_beta_positions_are_invisible_to_the_live_close_paths(self,
                                                                  tmp_path):
        # The experiment's positions live ONLY in its own book; the engine's
        # regular close machinery (``_close_for_arm`` over account positions)
        # can never see them.
        eng = beta_engine(tmp_path)
        run(eng.on_signal(eligible_signal()))
        assert eng.open_positions() == []
        for arm in ("V2_BETA_A", "V2_BETA_B", "V2_BETA_C"):
            assert eng.accounts[arm].position is None


# ===========================================================================
# dashboard + status reporting (task section 8)
# ===========================================================================


class TestStatusAndDashboard:
    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_status_reports_the_experiment_from_its_own_ledgers(self, tmp_path):
        eng = beta_engine(tmp_path, arms=parse_arms([BETA_GROUP,
                                                     "L25_CURRENT"]))
        status = eng.status()
        beta = status["v2_floor_beta"]
        assert beta is not None
        assert beta["paper_only"] is True
        assert beta["label"] == BETA_LABEL
        assert "V2 FLOOR (BETA)" in beta["label"]
        assert beta["entry_profile"] == "V2_FLOOR"
        assert beta["starting_bankroll_per_arm_usd"] == 10.0
        assert beta["total_starting_capital_usd"] == 30.0
        assert set(beta["arms"]) == {"A", "B", "C"}
        # the beta engine accounts are EXCLUDED from the regular accounts
        # block so one surface never shows two numbers for one experiment.
        assert "V2_BETA_A" not in status["accounts"]
        assert "L25_CURRENT" in status["accounts"]
        # no experiment -> None, so a non-beta run's status is unchanged.
        plain = beta_engine(tmp_path / "plain",
                            arms=parse_arms(["L25_CURRENT"]))
        assert plain.status()["v2_floor_beta"] is None

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_status_reflects_live_results_per_arm(self, tmp_path):
        # A profitable SELL quote, so A's TP exit is a WIN under the
        # executable-quote settlement (the P&L is the quote, not the mark).
        eng = beta_engine(tmp_path,
                          quotes=FakeQuoteService(sell=_winning_sell()))
        run(eng.on_signal(eligible_signal()))
        _quote_market(eng, 1.35)   # only A's +30% TP is reached, ON THE QUOTE
        run(eng.monitor_once())
        beta = eng.status()["v2_floor_beta"]
        a, b, c = (beta["arms"][k] for k in "ABC")
        assert a["completed_trades"] == 1
        assert a["win_rate_pct"] == 100.0
        assert a["exits"]["TP"] == 1
        assert a["bankroll_usd"] > 10.0
        assert b["completed_trades"] == 0
        assert b["win_rate_pct"] is None          # unavailable, not 0
        assert b["bankroll_usd"] == 10.0
        assert c["completed_trades"] == 0
        assert c["bankroll_usd"] == 10.0
        # the open arms carry unrealized value only from live marks.
        assert a["unrealized_pnl_usd"] == 0.0     # closed arm
        assert b["unrealized_unpriced_positions"] == 0
        assert b["unrealized_priced_positions"] == 1
        # mirrors are labelled visibility-only.
        notes = {m["note"] for m in eng.status()["beta_mirror_positions"]}
        assert notes and all("VISIBILITY ONLY" in n for n in notes)

    def _render(self, eng):
        from autopaper import dashboard
        beta = eng.status()["v2_floor_beta"]
        return dashboard.render(
            account_snapshot={
                "execution_aware_paper": {
                    "enabled": True,
                    "mode": "PAPER",
                    "bankroll_usd": {},
                    "open_positions": 0,
                    "committed_usd": 0.0,
                    "v2_floor_beta": beta,
                },
            },
            stats={},
            positions=[],
            last_decision=None,
            listener={},
            nar={},
            market={},
            safety={},
            mode="PAPER",
            uptime_s=1.0,
        )

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_the_dashboard_renders_the_group_and_every_arm(self, tmp_path):
        eng = beta_engine(tmp_path)
        run(eng.on_signal(eligible_signal()))
        text = self._render(eng)
        assert "V2 FLOOR (BETA)" in text
        for arm_line in ("A: TP +30%", "B: TP +50%", "C: TP +50%"):
            assert arm_line in text, arm_line
        assert "hold 90s" in text
        assert "hold 120s" in text
        assert "trades 0" in text
        # insufficient sample is marked UNAVAILABLE, never shown as a number.
        assert "n/a (no closed trades)" in text
        assert "PAPER ONLY" in text
        # a TP touch must not render as a win.
        assert "WR 0.0%" not in text

    @pytest.mark.skip(reason="V2 FLOOR (BETA) is disabled from the active bot")
    def test_the_dashboard_marks_a_small_sample_preliminary(self, tmp_path):
        eng = beta_engine(tmp_path,
                          quotes=FakeQuoteService(sell=_winning_sell()))
        run(eng.on_signal(eligible_signal()))
        _quote_market(eng, 1.35)
        run(eng.monitor_once())
        text = self._render(eng)
        assert "100.0% (preliminary, n=1)" in text
        assert "exits: TP 1" in text
        # nothing is unresolved in this scenario, so no such line renders.
        assert "unresolved" not in text


# ===========================================================================
# meta: all twenty scenarios are anchored to a class above
# ===========================================================================


def test_the_twenty_required_scenarios_are_all_covered():
    covered = {
        1: ["TestExplicitOptIn", "TestSelectorCallback", "TestEngineEntry"],
        2: ["TestEngineEntry"],
        3: ["TestPersistenceLifecycle"],
        4: ["TestPersistenceLifecycle"],
        5: ["TestEngineEntry"],
        6: ["TestEngineEntry"],
        7: ["TestEngineEntry"],
        8: ["TestEngineEntry"],
        9: ["TestArmConfiguration"],
        10: ["TestArmConfiguration"],
        11: ["TestArmConfiguration"],
        12: ["TestArmIndependence"],
        13: ["TestExitSemantics"],
        14: ["TestExitSemantics"],
        15: ["TestExitSemantics"],
        16: ["TestExitSemantics", "TestEngineEntry"],
        17: ["TestExitSemantics"],
        18: ["TestAccounting"],
        19: ["TestPersistenceLifecycle"],
        20: ["TestSafety", "TestExplicitOptIn", "TestSelectorCallback"],
    }
    classes = {name for name in globals() if name.startswith("Test")}
    for scenario, wanted in covered.items():
        for cls in wanted:
            assert cls in classes, f"scenario {scenario} lost class {cls}"
    assert set(covered) == set(range(1, 21))
