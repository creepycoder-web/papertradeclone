"""
Position sizing: the frozen V3-C policy.

Every example in the migration specification is asserted exactly, the boundary
at $50.00 is asserted on both sides, and the absence of any override path is
asserted structurally.
"""


from __future__ import annotations

import inspect

import pytest

from autopaper import sizing
from autopaper.sizing import (
    SIZING_BOUNDARY_USD,
    SIZING_PCT_OVER_50,
    SIZING_PCT_UNDER_50,
    allocation_label,
    compute_position_size,
    describe_policy,
    next_position_label,
    position_pct_for_equity,
    size_for_equity,
    tradeable,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.strategy,
]


# ---------------------------------------------------------------------------
# the specification's worked examples, exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "equity,expected_pct,expected_usd",
    [
        (1.55, 10.0, 0.155),
        (5.00, 10.0, 0.50),
        (10.00, 10.0, 1.00),
        (25.00, 10.0, 2.50),
        (50.00, 10.0, 5.00),
        (50.01, 12.5, 6.25125),
        (100.00, 12.5, 12.50),
    ],
)
def test_specification_examples(equity, expected_pct, expected_usd):
    size = compute_position_size(equity)
    assert size.position_pct == expected_pct
    assert size.position_usd == pytest.approx(expected_usd, abs=1e-12)
    assert size_for_equity(equity) == pytest.approx(expected_usd, abs=1e-12)


def test_the_boundary_is_inclusive_at_50():
    """$50.00 -> 10%. One cent more -> 12.5%."""
    assert position_pct_for_equity(50.00) == SIZING_PCT_UNDER_50
    assert position_pct_for_equity(50.01) == SIZING_PCT_OVER_50
    assert position_pct_for_equity(49.999999) == 10.0
    assert position_pct_for_equity(50.000001) == 12.5


def test_boundary_constant():
    assert SIZING_BOUNDARY_USD == 50.0
    assert SIZING_PCT_UNDER_50 == 10.0
    assert SIZING_PCT_OVER_50 == 12.5


@pytest.mark.parametrize("equity", [0.01, 0.5, 1.0, 1.55, 10, 25, 49.9, 50.0])
def test_under_or_at_50_is_always_10_percent(equity):
    assert position_pct_for_equity(equity) == 10.0


@pytest.mark.parametrize("equity", [50.01, 75, 100, 250, 1_000, 10_000])
def test_over_50_is_always_12_5_percent(equity):
    assert position_pct_for_equity(equity) == 12.5


# ---------------------------------------------------------------------------
# the micro account
# ---------------------------------------------------------------------------


def test_the_155_account_sizes_to_155_cents():
    """The account this migration was built for."""
    size = compute_position_size(1.55)
    assert size.position_usd == 0.155
    assert size.branch == "equity<=50"


def test_the_size_is_not_rounded_up():
    """
    $0.155 stays $0.155. Rounding a micro position up by a third of itself is
    the same class of quiet breach as rounding it up to a venue minimum.
    """
    assert compute_position_size(1.55).position_usd == 0.155
    assert compute_position_size(1.55).position_usd != 0.16


def test_625125_is_not_rounded_to_625():
    assert compute_position_size(50.01).position_usd == 6.25125


def test_the_policy_has_no_venue_minimum_floor():
    """
    There is no $5 floor anywhere in the sizing module. The previous
    implementation had one, and on a $1.55 account it would have refused every
    trade and invited the exact "round up to $5" behaviour the migration
    forbids.
    """
    src = inspect.getsource(sizing)
    assert "min_position_notional" not in src
    assert "MIN_POSITION" not in src
    # Behaviour is the real test: a $5 floor would make the $1.55 account
    # unsizable. Proved by producing a size below $5 with no complaint.
    assert compute_position_size(1.55).position_usd < 5.0
    assert compute_position_size(10.0).position_usd == 1.0
    # ...and the smallest realistic account still sizes.
    assert compute_position_size(0.10).position_usd == 0.01


# ---------------------------------------------------------------------------
# no override path
# ---------------------------------------------------------------------------


def test_compute_position_size_takes_equity_and_an_optional_profile():
    """
    Two parameters, and the second one is a PROFILE, not a number.

    This test used to assert the signature was exactly ``[equity_usd]``, which
    expressed a real intent: there must be no way for a caller to move a
    position size off the policy. That intent is still correct, and it is why
    the override is a profile NAME and not a percentage:

      * the only lever is a declared strategy profile;
      * that profile's percentage lives in the frozen registry
        (``strategy_profile.PROFILE_POSITION_PCT``), so a call site cannot
        invent a number;
      * the parameter is keyword-only, so no positional call site silently
        starts meaning something else.

    ``test_a_caller_cannot_pass_a_number_to_move_the_size_off_policy`` below is
    the part that actually pins the intent.
    """
    params = inspect.signature(compute_position_size).parameters
    assert list(params) == ["equity_usd", "profile"]
    assert params["profile"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["profile"].default is None
    # and it is not annotated as a number, because it is not a number
    assert params["profile"].annotation not in (float, int, str)


def test_a_caller_cannot_pass_a_number_to_move_the_size_off_policy():
    """
    The override is resolved through the frozen registry, so it cannot be
    supplied as a value.

    A caller that tried to move the size off policy -- by passing a percentage,
    or a percentage-looking string -- gets the SHARED policy, and a logged
    error, not a bespoke size. This is what keeps ``test_compute_position_size_
    takes_equity_and_an_optional_profile`` honest.
    """
    baseline = compute_position_size(100.0)
    for bogus in (25.0, 99, "12.5", "25%"):
        got = compute_position_size(100.0, profile=bogus)
        assert got.position_pct == baseline.position_pct, bogus
        assert got.position_usd == baseline.position_usd, bogus
        assert got.branch == baseline.branch, bogus


def test_the_only_way_to_get_a_different_size_is_a_declared_profile():
    """
    Every profile that is not V2_FLOOR* gets the shared ladder. V2_FLOOR keeps
    the researched flat 10%; V2_FLOOR_BETA declares 100% because its arms are
    ALL-IN on their own bankroll. Nothing else in the registry has an override,
    so adding a strategy cannot silently change a size.
    """
    from autopaper.strategy_profile import PROFILE_POSITION_PCT, StrategyProfile

    overriding = [p.value for p, pct in PROFILE_POSITION_PCT.items()
                  if pct is not None]
    assert overriding == ["V2_FLOOR", "V2_FLOOR_BETA"], overriding
    assert PROFILE_POSITION_PCT[StrategyProfile.V2_FLOOR] == 10.0
    # V2_FLOOR_BETA is ALL-IN (100%), not the flat 10% it used to declare. The
    # registry value is a declaration, not the mechanism: the experiment sizes
    # each arm at 100% of that arm's own uncommitted cash.
    assert PROFILE_POSITION_PCT[StrategyProfile.V2_FLOOR_BETA] == 100.0
    for p in (StrategyProfile.L25_CURRENT, StrategyProfile.L25_60S,
              StrategyProfile.L25_MAX_HOLD):
        assert compute_position_size(
            100.0, profile=p.value).position_pct == 12.5


def test_size_for_equity_takes_equity_and_an_optional_profile():
    params = inspect.signature(size_for_equity).parameters
    assert list(params) == ["equity_usd", "profile"]
    assert params["profile"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["profile"].default is None


def test_there_is_exactly_one_sizing_entry_point():
    """
    One function turns equity into a notional. The old
    ``execution.sizing.compute_size`` (route target, band clamp, exposure cap,
    cash cap, floor) has been deleted, so there is no second policy to drift.
    """
    import importlib

    import autopaper.execution as execution

    assert not hasattr(execution, "compute_size")
    assert not hasattr(execution, "scale_target_for_route")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("autopaper.execution.sizing")


def test_no_martingale_no_size_increase_after_losses():
    """
    The policy is a pure function of equity. It has no memory of prior trades,
    so it cannot increase after a loss - the only input is the current equity.
    """
    size = compute_position_size(10.0)
    again = compute_position_size(10.0)
    assert size.position_usd == again.position_usd


def test_not_all_in_at_any_equity():
    for equity in (0.01, 1.55, 10, 50, 100, 1_000):
        assert compute_position_size(equity).position_usd < equity


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------


def test_allocation_label_switches_on_the_boundary():
    assert "10%" in allocation_label(50.0)
    assert "12.5%" in allocation_label(50.01)


def test_next_position_label_shows_the_unrounded_size():
    assert "$0.155000" in next_position_label(1.55)
    assert "$5.000000" in next_position_label(50.0)
    assert "$12.500000" in next_position_label(100.0)


def test_next_position_label_on_a_dry_account():
    assert next_position_label(0.0) == "—"


def test_describe_policy_states_both_branches():
    text = describe_policy()
    assert "10%" in text
    assert "12.5%" in text
    assert "$50" in text


# ---------------------------------------------------------------------------
# tradeable()
# ---------------------------------------------------------------------------


def test_a_micro_size_is_tradeable_in_principle():
    """$0.155 passes. The venue is what may refuse it, not this function."""
    ok, why = tradeable(compute_position_size(1.55))
    assert ok, why


def test_zero_equity_is_not_tradeable():
    ok, why = tradeable(compute_position_size(0.0))
    assert not ok
    assert "equity is zero" in why


def test_insufficient_cash_is_reported_with_both_numbers():
    size = compute_position_size(10.0)          # $1.00
    ok, why = tradeable(size, available_cash_usd=0.50)
    assert not ok
    assert "0.500000" in why
    assert "1.000000" in why


def test_tradeable_applies_no_venue_minimum():
    """
    A $0.155 size is not refused here even though no venue will accept it. The
    venue executor reports that with the venue's own reason, which is a far
    more honest message than a locally invented floor.
    """
    ok, _ = tradeable(compute_position_size(1.55), available_cash_usd=1.55)
    assert ok
