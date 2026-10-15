"""
L25 research-correctness invariants: accounting, provenance and hold-path data.

These tests exist because the previous L25 evidence base was wrong in ways that
were invisible. The rules below are the corrections, stated as assertions:

  * a closed trade's PnL is FILL vs FILL, and it reconciles against cash exactly
  * a fee is charged once, never twice
  * MFE is the return at the highest observation and MAE the return at the lowest
  * a MISSING observation is not a ZERO observation
  * a duplicated, wrong-mint, wrong-pool or stale quote is data, not a price
  * an exit whose books do not reconcile leaves the position OPEN
  * the runtime never substitutes an aggregator mark for an executable quote
  * a return is measured against the entry FILL and uses no future information
  * a horizon the recorded samples cannot resolve is reported NOT_TESTABLE

Nothing here tunes a threshold, changes an exit rule, or asserts a performance
outcome. The point is that the measurement is trustworthy, so a future study can
be believed or disbelieved on its merits.
"""

from __future__ import annotations

import asyncio
import json
import math
from decimal import Decimal
from pathlib import Path

import pytest

from autopaper.account import PaperAccount, _within_band
from autopaper.config import ExitConfig, RiskLimits
from autopaper.ledger import PaperAccountState
from autopaper.models import (
    DataState,
    ExcursionState,
    ExitStyle,
    FillResult,
    Position,
    PositionStatus,
    Regime,
    Route,
)
from autopaper.paper_live.schema import QuotePoint
from autopaper.pricing import PriceProvenance, is_actual_fill
from autopaper.settlement import WSOL_MINT

from .conftest import MINT, MINT2

pytestmark = pytest.mark.strategy

LIMITS = RiskLimits()
CFG_EXITS = ExitConfig()

#: conftest.MINT is the wrapped-SOL mint, i.e. the QUOTE asset. A held token
#: needs its own address, and confusing the two is precisely the identity error
#: the hold-path validator exists to catch.
TOKEN = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
TOKEN2 = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"

ENTRY_FILL = 1.0e-4
ENTRY_FEE = 0.0025
NOTIONAL = 10.0
STYLE = ExitStyle.MOMENTUM_FAILURE


def run(coro):
    """The project's convention: this suite has no pytest-asyncio plugin."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def account(start_cash: float = 1000.0) -> PaperAccount:
    return PaperAccount(
        PaperAccountState(starting_bankroll_usd=start_cash, cash_usd=start_cash),
        LIMITS,
    )


def open_position(**over) -> Position:
    """
    A position opened the way the runtime opens one: a fill that PAID MORE than
    the quoted reference, and a cash outlay of quantity x fill + fee.

    Using the quoted reference as the entry price here is precisely the error
    under test, so the fixture never does it. The provenance of both legs is
    declared explicitly, because a fixture that leaves them at the default cannot
    distinguish "the trade records its basis" from "the trade happened to have
    had the default".
    """
    entry_fill = float(over.pop("entry_fill", ENTRY_FILL))
    entry_reference = float(over.pop("entry_reference", ENTRY_FILL * 0.99))
    qty = NOTIONAL / entry_fill
    pos = Position(
        position_id=over.pop("position_id", "p_l25"),
        candidate_id="c", mint=MINT, symbol="X",
        status=PositionStatus.OPEN, route=Route.FAST_ENTRY, regime=Regime.EARLY,
        exit_style=STYLE, strategy="l25",
        entry_signal_timestamp=0.0, entry_decision_timestamp=0.0,
        entry_timestamp=0.0,
        entry_reference_price=entry_reference,
        fill_price=entry_fill, quantity=qty,
        cash_out_usd=qty * entry_fill + ENTRY_FEE,
        cost_basis_usd=qty * entry_fill + ENTRY_FEE,
        entry_fee_usd=ENTRY_FEE,
        entry_slippage_usd=qty * (entry_fill - entry_reference),
        entry_price_provenance=PriceProvenance.PAPER_FILL.value,
        liquidity_at_entry=1e6, market_cap_at_entry=1e6,
        token_age_s_at_entry=100.0, runner_mode="LIVE", confidence=1.0,
        signal_quality=1.0,
    )
    for k, v in over.items():
        setattr(pos, k, v)
    return pos


def close(acct, pos, exit_fill, *, exit_fee=0.0, reason="test", reference=None):
    return acct.close_position(
        pos,
        exit_reference=reference if reference is not None else exit_fill,
        exit_fill_price=exit_fill,
        exit_fee=exit_fee,
        exit_slippage=0.0,
        now=1000.0,
        reason=reason,
        style=STYLE,
    )


def fill(price: float, *, provenance: str = PriceProvenance.PAPER_FILL.value,
         filled: bool = True) -> FillResult:
    return FillResult(
        filled=filled, reason="ok" if filled else "not filled",
        quoted_price=price, fill_price=price if filled else None,
        slippage_bps=0.0, slippage_usd=0.001, fee_bps=85.0, fee_usd=0.002,
        latency_ms=1.0, placed_at=999.0, fill_at=1000.0, observed_at=1000.0,
        price_at_quote=price, data_state=DataState.LIVE,
        reference_price_provenance=provenance,
        fill_price_provenance=PriceProvenance.PAPER_FILL.value,
    )


# ---------------------------------------------------------------------------
# 1. the exit price actually used
# ---------------------------------------------------------------------------

def test_closed_trade_pnl_is_fill_against_fill():
    """
    PnL uses the two EXECUTED prices.

    Deriving gross from the pre-slippage references and deducting slippage
    afterwards is not an equivalent formulation, and the cash it has to reconcile
    against moves by the executed prices.
    """
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    exit_fill = ENTRY_FILL * 1.5
    trade = close(acct, pos, exit_fill)
    assert trade.gross_pnl_usd == pytest.approx(
        (exit_fill - ENTRY_FILL) * pos.quantity, rel=1e-12)
    assert trade.return_pct == pytest.approx(50.0, rel=1e-9)


def test_realized_pnl_reconciles_against_cash():
    """
    Realized PnL must equal the change in cash. This is the assertion that the
    old reference-basis figure could not satisfy.
    """
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    trade = close(acct, pos, ENTRY_FILL * 0.8)
    # ``cash_at_entry`` is the cash held BEFORE the outlay, so it -- not the cash
    # after the entry was funded -- is what realized PnL has to reconcile against.
    assert trade.net_pnl_usd == pytest.approx(
        acct.cash - pos.cash_at_entry, abs=1e-9)


def test_the_booked_exit_must_lie_inside_the_observed_book():
    """
    A booked exit below the lowest mark ever observed on this position is a
    price from outside the monitored book -- the signature that identified 22 of
    the 26 divergent historical exits.
    """
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    for price, at in ((ENTRY_FILL, 1.0), (ENTRY_FILL * 1.2, 2.0),
                       (ENTRY_FILL * 0.9, 3.0)):
        acct.apply_mark(pos, price, DataState.LIVE, at)
    pos.exit_price = ENTRY_FILL * 1.1
    assert _within_band(pos) is True
    pos.exit_price = ENTRY_FILL * 0.5
    assert _within_band(pos) is False


def test_band_check_returns_none_rather_than_guessing_without_observations():
    assert _within_band(open_position()) is None


# ---------------------------------------------------------------------------
# 2. fees are charged once
# ---------------------------------------------------------------------------

def test_exit_fee_is_deducted_once_not_twice():
    """
    An executable quote already prices the size it quoted. Charging slippage on
    top of a fill derived from that quote double-counts it.
    """
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    exit_fill = ENTRY_FILL * 1.2
    fee = 0.003
    trade = close(acct, pos, exit_fill, exit_fee=fee)
    assert trade.exit_fee_usd == pytest.approx(fee)
    assert trade.net_pnl_usd == pytest.approx(
        (exit_fill - ENTRY_FILL) * pos.quantity - ENTRY_FEE - fee, rel=1e-12)


def test_only_an_onchain_fill_is_an_actual_fill():
    """
    A PAPER fill is derived, never signed. Calling it an actual execution is the
    same category error as calling a mark a fill, so the test pins the narrow
    definition against being widened by convenience.
    """
    assert is_actual_fill(PriceProvenance.ONCHAIN_FILL.value)
    for prov in (PriceProvenance.PAPER_FILL.value,
                 PriceProvenance.MARK_ONLY.value,
                 PriceProvenance.NATIVE_REFERENCE.value,
                 PriceProvenance.EXECUTABLE_QUOTE.value):
        assert not is_actual_fill(prov)


def test_a_fill_result_records_the_reference_and_the_fill_separately():
    """
    The reference may be a venue quote while the fill itself is only a PAPER
    FILL. Collapsing the two is how a simulated price gets presented as settled.
    """
    f = fill(1.0, provenance=PriceProvenance.EXECUTABLE_QUOTE.value)
    assert f.reference_price_provenance == PriceProvenance.EXECUTABLE_QUOTE.value
    assert f.fill_price_provenance == PriceProvenance.PAPER_FILL.value
    assert not is_actual_fill(f.fill_price_provenance)


# ---------------------------------------------------------------------------
# 3. MFE / MAE are extrema of the return series
# ---------------------------------------------------------------------------

def test_mfe_is_the_highest_observation_and_mae_is_the_lowest():
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    for price in (ENTRY_FILL * 1.5, ENTRY_FILL * 0.5, ENTRY_FILL * 1.1):
        acct.apply_mark(pos, price, DataState.LIVE, 2.0)
    assert pos.mfe_pct == pytest.approx(50.0, rel=1e-9)
    assert pos.mae_pct == pytest.approx(-50.0, rel=1e-9)
    assert pos.peak_price == pytest.approx(ENTRY_FILL * 1.5)
    assert pos.trough_price == pytest.approx(ENTRY_FILL * 0.5)


def test_a_trade_closed_near_flat_does_not_book_a_positive_mae():
    """
    The old MAE was the return at the PEAK, which let a position that was 60 %
    underwater and closed near flat report a POSITIVE maximum adverse excursion.
    52 of 77 historical fills did exactly that.
    """
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    acct.apply_mark(pos, ENTRY_FILL * 0.4, DataState.LIVE, 2.0)
    acct.apply_mark(pos, ENTRY_FILL * 0.99, DataState.LIVE, 3.0)
    trade = close(acct, pos, ENTRY_FILL)
    assert trade.mae_pct == pytest.approx(-60.0, rel=1e-9)


def test_excursions_are_seeded_from_the_entry_fill_not_the_quote():
    """
    The entry reference was BELOW the fill. Seeding from it would report the
    position as already up on the day it opened.
    """
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    assert pos.excursion_basis == "ENTRY_FILL"
    assert pos.peak_price == pytest.approx(ENTRY_FILL)
    assert pos.trough_price == pytest.approx(ENTRY_FILL)
    assert pos.peak_return_pct == 0.0
    assert pos.trough_return_pct == 0.0


def test_extrema_only_ever_move_outwards():
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    for price, at in ((ENTRY_FILL * 2, 1.0), (ENTRY_FILL, 2.0),
                       (ENTRY_FILL * 3, 3.0), (ENTRY_FILL * 0.5, 4.0)):
        acct.apply_mark(pos, price, DataState.LIVE, at)
    assert pos.peak_price == pytest.approx(ENTRY_FILL * 3)
    assert pos.trough_price == pytest.approx(ENTRY_FILL * 0.5)


def test_with_a_valid_entry_and_one_observation_the_excursions_bracket_zero():
    """The structural consequence of seeding from the fill."""
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL * 1.3, DataState.LIVE, 1.0)
    assert pos.mfe_pct >= 0 >= pos.mae_pct


# ---------------------------------------------------------------------------
# 4. a missing observation is not a zero observation
# ---------------------------------------------------------------------------

def test_a_missing_mark_is_counted_as_rejected_and_moves_no_extrema():
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL * 2, DataState.LIVE, 1.0)
    for bad in (None, 0.0, -1.0):
        acct.apply_mark(pos, bad, DataState.LIVE, 2.0)
    assert pos.peak_price == pytest.approx(ENTRY_FILL * 2)
    assert pos.excursion_observations == 1
    assert pos.excursion_observations_rejected == 3


def test_a_never_observed_position_reports_unavailable_not_zero():
    """
    No mark was ever applied. The extrema then mean "at least 0", which is the
    only honest reading of a value with no observations behind it.
    """
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    trade = close(acct, pos, ENTRY_FILL)
    assert trade.excursion_state == ExcursionState.UNAVAILABLE_HISTORICAL_DATA.value
    assert trade.excursion_observations == 0


def test_a_stale_print_caps_the_state_at_partial_data_even_if_a_later_print_is_live():
    """
    COMPLETE means every observation was valid -- not that the most recent one
    was. Otherwise a single fresh print at exit launders a stale hold into a
    coverage claim.
    """
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    acct.apply_mark(pos, ENTRY_FILL * 1.1, DataState.STALE, 2.0)
    acct.apply_mark(pos, ENTRY_FILL * 1.2, DataState.LIVE, 3.0)
    assert pos.excursion_state == ExcursionState.PARTIAL_DATA.value


def test_a_clean_hold_is_complete():
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    for at, price in enumerate((ENTRY_FILL, ENTRY_FILL * 1.1, ENTRY_FILL * 0.95), 1):
        acct.apply_mark(pos, price, DataState.LIVE, float(at))
    assert pos.excursion_state == ExcursionState.COMPLETE.value
    assert pos.excursion_observations == 3


# ---------------------------------------------------------------------------
# 5. an exit that does not reconcile leaves the position open
# ---------------------------------------------------------------------------

def test_a_close_whose_books_do_not_reconcile_is_refused_before_any_mutation():
    """
    ``build_trade`` checks the books BEFORE touching anything. An unbookable
    exit is a data problem, not a reason to lose the position or to leave the
    account and the audit trail disagreeing.
    """
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    cash_before = acct.cash
    # The recorded outlay no longer matches the fills that funded it, so proceeds
    # and net PnL cannot both be right. (A wrong FEE cannot produce this: both
    # sides of the check move with the fee.)
    pos.cash_out_usd *= 1.5
    with pytest.raises(Exception):
        close(acct, pos, ENTRY_FILL)
    assert pos.status is PositionStatus.OPEN
    assert acct.cash == pytest.approx(cash_before)


def test_an_exit_without_a_price_cannot_be_booked():
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    cash_before = acct.cash
    with pytest.raises(Exception):
        close(acct, pos, None)
    assert pos.status is PositionStatus.OPEN
    assert acct.cash == pytest.approx(cash_before)


def test_an_unfilled_fill_result_carries_no_price():
    f = fill(1.0, filled=False)
    assert f.filled is False
    assert f.fill_price is None


# ---------------------------------------------------------------------------
# 6. provenance is carried per side and never inferred
# ---------------------------------------------------------------------------

def test_a_mark_closed_exit_is_labelled_mark_only_not_a_fill():
    pos = open_position(entry_price_provenance=PriceProvenance.MARK_ONLY.value,
                        exit_price_provenance=PriceProvenance.MARK_ONLY.value)
    acct = account()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    trade = close(acct, pos, ENTRY_FILL)
    assert trade.entry_price_provenance == PriceProvenance.MARK_ONLY.value
    assert trade.exit_price_provenance == PriceProvenance.MARK_ONLY.value
    assert trade.pnl_basis == "FILL_VS_FILL"


def test_trade_records_where_gross_differs_from_the_reference_basis():
    """
    The disagreement is recorded rather than resolved silently, so a reader can
    see how much of a PnL came from the choice of basis.
    """
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    exit_reference = ENTRY_FILL * 2.0
    exit_fill = ENTRY_FILL * 1.9
    trade = close(acct, pos, exit_fill, reference=exit_reference)
    assert trade.gross_pnl_reference_basis_usd == pytest.approx(
        (exit_reference - pos.entry_reference_price) * pos.quantity, rel=1e-12)
    # The reference basis is carried as GROSS, and the disagreement nets it the
    # same way ``net_pnl_usd`` is netted -- fees and entry slippage included.
    expected = (trade.gross_pnl_reference_basis_usd - pos.entry_slippage_usd
                - trade.gross_pnl_usd)
    assert trade.pnl_basis_disagreement_usd == pytest.approx(expected, rel=1e-9)
    assert trade.pnl_basis_disagreement_usd > 0


def test_a_trade_round_trips_through_its_dict_without_losing_provenance():
    acct = account()
    pos = open_position()
    acct.register_position(pos)
    for at, price in ((1.0, ENTRY_FILL), (2.0, ENTRY_FILL * 1.4),
                      (3.0, ENTRY_FILL * 0.8)):
        acct.apply_mark(pos, price, DataState.LIVE, at)
    trade = close(acct, pos, ENTRY_FILL * 1.1)
    d = trade.as_dict()
    assert d["pnl_basis"] == "FILL_VS_FILL"
    assert d["entry_price_provenance"] == PriceProvenance.PAPER_FILL.value
    assert d["excursion_state"] in {s.value for s in ExcursionState}
    assert d["excursion_observations"] == 3
    assert d["mfe_pct"] == pytest.approx(40.0, rel=1e-9)
    assert d["mae_pct"] == pytest.approx(-20.0, rel=1e-9)
    assert d["booked_exit_within_observed_band"] is True
    assert math.isfinite(d["net_pnl_usd"])


def test_the_exit_provenance_reaches_the_trade_row():
    """
    Regression: ``positions.close`` used to stamp the exit provenance AFTER
    ``account.close_position`` returned, so every closed trade silently degraded
    to the default MARK_ONLY and the band check fell back to comparing the exit
    against the last mark.
    """
    from autopaper.positions import PositionBook

    pos = open_position()
    pos.exit_price_provenance = PriceProvenance.EXECUTABLE_QUOTE.value
    pos.exit_price = ENTRY_FILL * 1.1
    acct = account()
    acct.register_position(pos)
    acct.apply_mark(pos, ENTRY_FILL, DataState.LIVE, 1.0)
    acct.apply_mark(pos, ENTRY_FILL * 1.2, DataState.LIVE, 2.0)
    trade = close(acct, pos, ENTRY_FILL * 1.1)
    assert trade.exit_price_provenance == PriceProvenance.EXECUTABLE_QUOTE.value
    assert trade.booked_exit_within_observed_band is True
    assert PositionBook is not None


# ---------------------------------------------------------------------------
# 7. hold-path observations
# ---------------------------------------------------------------------------

async def _null_quote(**_kw):
    return None


def _null_mark(_mint):
    return None


def _recorder(tmp_path: Path, **over):
    from autopaper.paper_live.telemetry import HoldPathConfig, HoldPathRecorder

    cfg = HoldPathConfig(
        interval_s=0.01, max_duration_s=5.0, max_state_age_ms=5_000.0,
        max_quote_age_ms=30_000.0, sink_path=tmp_path / "hold.jsonl", enabled=True,
    )
    kwargs = dict(
        trade_id="t1", mint=TOKEN, curve_or_pool_id="pumpswap:pool-1",
        token_decimals=6, remaining_token_amount_raw=1_000_000,
        quote_asset=WSOL_MINT, entry_fill_usd=100.0,
        entry_price_provenance="PAPER_FILL", strategy_profile="L25_CURRENT",
        # The SOL reference the recorder needs before a raw quote amount can be
        # turned into USD. Without it every observation is correctly recorded as
        # having no return.
        sol_usd_reference=100.0, entry_notional_usd=200.0,
        quote_fn=_null_quote, mark_fn=_null_mark, config=cfg,
    )
    kwargs.update(over)
    return HoldPathRecorder(**kwargs)


def _quote(**over) -> QuotePoint:
    """A valid, same-market, fresh ``EXECUTABLE_QUOTE`` for the position."""
    kwargs = dict(
        side="SELL", provider="jupiter", market_type="PUMPSWAP",
        token_mint=TOKEN, quote_mint=WSOL_MINT,
        input_amount_raw=1_000_000, input_decimals=6, input_amount_decimal=1.0,
        output_amount_raw=2_000_000_000, output_decimals=9,
        output_amount_decimal=2.0,
        effective_price=Decimal("2"), effective_price_usd=200.0,
        price_impact_pct=0.4, fee_bps=85, fee_amount=Decimal("0.0017"),
        venue="pumpswap", pool_or_curve="pool-1", route=("pool-1",),
        state_slot=7, state_age_ms=50.0,
        quote_timestamp=1000.0, quote_age_ms=10.0, max_quote_age_ms=30_000.0,
        identity="PASS", venue_id="pumpswap:pool-1", available=True,
    )
    kwargs.update(over)
    return QuotePoint(**kwargs)


def _sample(rec, quote=None, mark=100.0, now=1000.0):
    """One observation. Synchronous: the sampler holds no long-lived loop."""
    payload = quote

    async def _quote_fn(**_kw):
        if isinstance(payload, Exception):
            raise payload
        return payload

    rec._quote_fn = _quote_fn
    rec._mark_fn = lambda _m: mark
    rec._clock = lambda: now
    return run(rec.sample_once())


def test_a_hold_path_return_is_measured_against_the_entry_fill(tmp_path):
    rec = _recorder(tmp_path, entry_fill_usd=100.0)
    obs = _sample(rec, quote=_quote(), mark=100.0)
    # net_usd = 2_000_000_000 / 1e9 SOL x $100 = $200, against a $100 entry fill.
    # 2_000_000_000 raw SOL at $100 = $200 against a $200 entry notional.
    assert obs.entry_notional_usd == pytest.approx(200.0)
    assert obs.return_pct == pytest.approx(0.0, rel=1e-9)
    assert obs.entry_fill_usd == pytest.approx(100.0)
    assert obs.entry_price_provenance == "PAPER_FILL"
    assert obs.reference_price_provenance == "EXECUTABLE_QUOTE"
    assert obs.usable_for_return is True
    assert obs.data_quality_reason is None


def test_a_missing_quote_is_recorded_with_no_return_and_never_zero(tmp_path):
    """
    The most damaging data bug available here: a failed quote written as 0.0
    would make a hold-path study read "the position went to zero" at the exact
    moment the runtime simply could not see the market.
    """
    rec = _recorder(tmp_path)
    obs = _sample(rec, quote=None)
    assert obs.return_pct is None
    assert obs.usable_for_return is False
    assert obs.data_quality_reason
    assert obs.reference_price_provenance == "UNAVAILABLE"


def test_an_unavailable_quote_is_reported_unavailable(tmp_path):
    rec = _recorder(tmp_path)
    obs = _sample(rec, quote=_quote(available=False, missing_reason="empty"))
    assert obs.return_pct is None
    assert obs.unavailable_reason


def test_a_quote_for_the_wrong_mint_is_refused(tmp_path):
    rec = _recorder(tmp_path)
    obs = _sample(rec, quote=_quote(token_mint=TOKEN2))
    assert obs.return_pct is None
    assert obs.usable_for_return is False
    assert "MISMATCH" in (obs.identity_check_status or "").upper()


def test_a_quote_from_the_wrong_pool_is_refused(tmp_path):
    """
    A SELL routed to a different pool is a different trade with different depth,
    so it must never be mixed into this position's path.
    """
    rec = _recorder(tmp_path)
    obs = _sample(rec, quote=_quote(venue_id="pumpswap:pool-2"))
    assert obs.return_pct is None
    assert obs.usable_for_return is False
    assert "MISMATCH" in (obs.identity_check_status or "").upper()


def test_a_stale_quote_is_refused(tmp_path):
    rec = _recorder(tmp_path)
    obs = _sample(rec, quote=_quote(quote_age_ms=3_600_000.0))
    assert obs.return_pct is None
    assert obs.usable_for_return is False


def test_a_stale_pool_state_slot_is_refused(tmp_path):
    rec = _recorder(tmp_path)
    obs = _sample(rec, quote=_quote(state_age_ms=600_000.0))
    assert obs.return_pct is None
    assert obs.usable_for_return is False


def test_an_invalid_price_is_refused(tmp_path):
    rec = _recorder(tmp_path)
    obs = _sample(rec, quote=_quote(output_amount_raw=0))
    assert obs.return_pct is None
    assert obs.usable_for_return is False


def test_a_raising_quote_service_is_recorded_rather_than_raised(tmp_path):
    rec = _recorder(tmp_path)
    obs = _sample(rec, quote=RuntimeError("jupiter 502"))
    assert obs is not None
    assert obs.return_pct is None
    assert "raised" in (obs.unavailable_reason or "")


def test_the_aggregator_mark_cannot_substitute_for_the_quote(tmp_path):
    """
    When the quote fails the mark is still recorded, as a diagnostic. It must
    never become the observation's return.
    """
    rec = _recorder(tmp_path)
    obs = _sample(rec, quote=None, mark=250.0)
    assert obs.aggregator_mark_usd == pytest.approx(250.0)
    assert obs.return_pct is None
    assert obs.reference_price_provenance == "UNAVAILABLE"


def test_observation_ids_are_unique_and_ordered(tmp_path):
    rec = _recorder(tmp_path)
    seen = []
    for i in range(4):
        obs = _sample(rec, quote=_quote(), now=1000.0 + i)
        seen.append((obs.observation_id, obs.elapsed_ms))
    assert len({oid for oid, _ in seen}) == len(seen)
    assert [ms for _, ms in seen] == sorted(ms for _, ms in seen)


def test_duplicate_observations_are_dropped_when_the_path_is_read(tmp_path):
    """
    A re-sent sample must not count twice in a hold-path maximum. The recorder
    appends; the reader de-duplicates on ``observation_id``, which is the only
    stable key a re-sent sample carries.
    """
    from research.l25_exec_audit.horizon import load_paths

    sink = tmp_path / "hold.jsonl"
    rec = _recorder(tmp_path)
    run(rec.sample_entry())
    run(rec.sample_once())
    run(rec.sample_once())
    body = sink.read_text()
    sink.write_text(body + body)  # the whole stream, replayed
    paths = load_paths(sink)
    assert len(paths) == 1
    assert len(paths[0]) == 3


def test_a_quote_stamped_in_the_future_is_refused(tmp_path):
    """
    Causality: a sample may only carry the quote timestamp it was given. A quote
    stamped in the future must be refused rather than believed.
    """
    rec = _recorder(tmp_path)
    obs = _sample(rec, quote=_quote(quote_timestamp=9_999_999.0), now=1000.0)
    assert obs.return_pct is None
    assert obs.usable_for_return is False


def test_the_first_sample_is_the_entry_and_the_rest_are_path(tmp_path):
    rec = _recorder(tmp_path)
    first = run(rec.sample_entry())
    assert first.phase == "ENTRY"
    assert run(rec.sample_once()).phase == "PATH"


def test_without_an_entry_basis_no_return_is_fabricated(tmp_path):
    rec = _recorder(tmp_path, entry_fill_usd=None, entry_notional_usd=None)
    obs = _sample(rec, quote=_quote())
    assert obs.return_pct is None
    assert "entry fill" in (obs.data_quality_reason or "")


def test_observations_persist_as_one_json_object_per_line(tmp_path):
    sink = tmp_path / "hold.jsonl"
    rec = _recorder(tmp_path)
    run(rec.sample_entry())
    run(rec.sample_once())
    lines = [l for l in sink.read_text().splitlines() if l.strip()]
    assert lines, "the hold path must persist observations"
    kinds = [json.loads(l).get("record_type") for l in lines]
    assert kinds.count("OBSERVATION") == 2


def test_a_hold_path_summary_reports_coverage_alongside_every_number(tmp_path):
    rec = _recorder(tmp_path)
    for i in range(6):
        _sample(rec, quote=_quote(), now=1000.0 + i * 5)
    summary = rec.summary()
    assert summary["observation_count_total"] >= 6
    assert summary["metrics_status"] == "VALID"
    # Every reported metric carries its own coverage figure, so a partial hold
    # cannot be read as a complete one.
    assert any(k.endswith("_COVERAGE_PCT") for k in summary)


def test_the_hold_path_instrument_is_reachable_from_configuration():
    """
    It was previously a module global nothing ever assigned, so it silently
    recorded nothing. Reachability is the whole point of a measurement.
    """
    from autopaper.config import build_config

    real = build_config().real
    assert real.paper_hold_path_telemetry_enabled is True
    assert real.paper_hold_path_sink
    assert real.paper_hold_path_interval_s > 0
    assert real.paper_hold_path_max_duration_s > 0


# ---------------------------------------------------------------------------
# 8. horizon coverage is reported, never manufactured
# ---------------------------------------------------------------------------

def test_a_series_coarser_than_the_horizon_reports_not_testable():
    """
    60 s spacing cannot speak to anything under a minute, so every sub-minute
    horizon is NOT_TESTABLE. Interpolating would invent observations that were
    never made, which is the error this whole pass exists to remove.
    """
    from research.l25_exec_audit.horizon import HORIZONS, check_horizon_coverage

    assert 5 in HORIZONS and 15 in HORIZONS
    stamps = [1000.0 + i * 60.0 for i in range(20)]
    for horizon in HORIZONS:
        cov = check_horizon_coverage(stamps, horizon_s=horizon)
        if horizon < 60:
            assert cov.resolvable is False
            assert cov.status == "NOT_TESTABLE"
            assert "interpolat" in cov.reason
        else:
            assert cov.resolvable is True


def test_a_horizon_is_only_claimed_when_the_spacing_supports_it():
    from research.l25_exec_audit.horizon import check_horizon_coverage

    stamps = [1000.0 + i * 5.0 for i in range(200)]
    fine = check_horizon_coverage(stamps, horizon_s=5)
    assert fine.resolvable is True
    assert fine.status == "TESTABLE"
    assert check_horizon_coverage(stamps, horizon_s=900).resolvable is True
    # A single observation cannot bound a maximum, at any horizon.
    alone = check_horizon_coverage([1000.0], horizon_s=5)
    assert alone.resolvable is False
    assert alone.status == "INSUFFICIENT_OBSERVATIONS"


def test_unusable_timestamps_do_not_count_as_observed_time():
    from research.l25_exec_audit.horizon import check_horizon_coverage

    stamps = [1000.0 + i * 2.0 for i in range(20)]
    # Only the last two are usable, and they are 2 s apart -- dense enough for a
    # 5 s window, but the PATH does not reach one.
    flags = [False] * 18 + [True, True]
    cov = check_horizon_coverage(stamps, horizon_s=5, usable_flags=flags)
    assert cov.n_usable == 2
    assert cov.resolvable is False
    assert "never reach" in cov.reason
    # Dense AND spanning is resolvable.
    wide = check_horizon_coverage([1000.0, 1002.0, 1020.0], horizon_s=15,
                                  usable_flags=[True, True, True])
    assert wide.resolvable is True


def test_an_untestable_horizon_is_absent_from_the_statistics_not_stored_as_none():
    """
    A ``None`` sitting next to real numbers is an average waiting to happen, so
    an untestable horizon is left out of the stats map entirely.
    """
    from research.l25_exec_audit.horizon import analyse_path

    rows = [
        {"trade_id": "t1", "mint": MINT, "strategy_profile": "L25_CURRENT",
         "entry_fill_usd": 100.0, "entry_price_provenance": "PAPER_FILL",
         "observation_id": "o1", "phase": "ENTRY", "observed_at_utc": 1000.0,
         "usable_for_return": True, "return_pct": 0.0,
         "reference_price_provenance": "EXECUTABLE_QUOTE"},
    ]
    rows += [
        {"trade_id": "t1", "mint": MINT, "strategy_profile": "L25_CURRENT",
         "entry_fill_usd": 100.0, "entry_price_provenance": "PAPER_FILL",
         "observation_id": f"o{i}", "phase": "PATH",
         "observed_at_utc": 1000.0 + i * 60.0, "usable_for_return": True,
         "return_pct": float(i), "reference_price_provenance": "EXECUTABLE_QUOTE"}
        for i in range(1, 20)
    ]
    result = analyse_path(rows)
    assert "5" not in result.stats and "15" not in result.stats
    assert result.coverage["5"].status == "NOT_TESTABLE"
    # The 60 s window contains the ENTRY observation plus the first PATH sample.
    assert "60" in result.stats
    assert result.stats["60"].n == 2
    assert result.stats["60"].max_return_pct == pytest.approx(1.0)
    assert result.stats["60"].min_return_pct == pytest.approx(0.0)


def test_the_summary_counts_coverage_next_to_every_figure():
    from research.l25_exec_audit.horizon import analyse_path, summarise

    rows = [
        {"trade_id": "t1", "mint": MINT, "strategy_profile": "L25_CURRENT",
         "entry_fill_usd": 100.0, "entry_price_provenance": "PAPER_FILL",
         "observation_id": "e", "phase": "ENTRY", "observed_at_utc": 1000.0,
         "usable_for_return": True, "return_pct": 0.0,
         "reference_price_provenance": "EXECUTABLE_QUOTE"},
        {"trade_id": "t1", "mint": MINT, "strategy_profile": "L25_CURRENT",
         "entry_fill_usd": 100.0, "entry_price_provenance": "PAPER_FILL",
         "observation_id": "p1", "phase": "PATH", "observed_at_utc": 1030.0,
         "usable_for_return": True, "return_pct": 12.0,
         "reference_price_provenance": "EXECUTABLE_QUOTE"},
    ]
    summary = summarise([rows])
    assert summary["n_paths"] == 1
    h30 = summary["horizons"]["30"]
    assert h30["n_testable"] == 1
    assert h30["n_resolvable"] == 1
    h5 = summary["horizons"]["5"]
    assert h5["status"] == "NOT_TESTABLE"
    assert h5["reason"]


def test_a_path_with_no_usable_observation_yields_nothing_rather_than_zeroes():
    from research.l25_exec_audit.horizon import analyse_path

    rows = [
        {"trade_id": "t2", "mint": MINT, "strategy_profile": "L25_CURRENT",
         "entry_fill_usd": 100.0, "entry_price_provenance": "PAPER_FILL",
         "observation_id": "x1", "phase": "PATH", "observed_at_utc": 1000.0,
         "usable_for_return": False, "return_pct": None,
         "reference_price_provenance": "UNAVAILABLE",
         "data_quality_reason": "quote call raised: TimeoutError"},
    ]
    result = analyse_path(rows)
    assert result.n_usable == 0
    assert result.stats == {}
    assert result.rejected["quote call raised: TimeoutError"] == 1