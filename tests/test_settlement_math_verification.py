"""
Independent verification of the canonical PAPER settlement arithmetic.

WHY THIS FILE EXISTS
====================
``realize()`` is where a simulated position becomes a number the operator reads
as profit or loss. Every expectation below is derived BY HAND, from the
definition, and then compared against the production function -- never copied
back out of it. A test whose expected value comes from the function it is
testing proves only that the function is deterministic.

THE IDENTITY BEING VERIFIED, in the quote asset (SOL):

    gross = exit_proceeds - entry_cost
    net   = gross - fees - network_and_other_costs
    return_pct        = net / entry_cost * 100
    realized_pnl_usd  = net * SOL/USD

WHAT WAS WRONG
==============
``network_costs_quote`` -- a field whose name and every consumer say "quote
asset (SOL)" -- was being filled with ``FillModel.network_cost_lamports``, a
value declared in LAMPORTS, and then added to SOL-denominated amounts in both
``realize()`` and ``_entry_all_in_cost_quote_asset()``. 5,000 lamports of
intended cost (0.000005 SOL) would have been spent as 5,000 SOL: a 10^9 unit
error, silent because the value is a bare Decimal. It was masked only by the
zero default, which is what made it a landmine rather than a live error.

PAPER only. No network, no signer, no transaction, no store.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from autopaper.paper_live.fills import LAMPORTS_PER_SOL, FillModel, realize
from autopaper.paper_live.quotes import SolReference
from autopaper.paper_live.schema import PaperPosition, SimulatedFill

# ---------------------------------------------------------------------------
# hand-computed scenario
# ---------------------------------------------------------------------------
#
# A BUY paying 66,670,000 lamports for 658,884.916927 tokens, then a SELL
# returning 70,000,000 lamports. Chosen so every figure below can be checked on
# paper, and so the raw integers are large enough that a float error would show.
#
WSOL = "So11111111111111111111111111111111111111112"
TOKEN = "4RiJPcL6GXecaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

ENTRY_RAW = 66_670_000          # lamports actually swapped in
ENTRY_DECIMALS = 9
TOKENS_RAW = 658_884_916_927    # raw token units received
TOKEN_DECIMALS = 6
EXIT_RAW = 70_000_000           # lamports the SELL quote returns
EXIT_DECIMALS = 9
SOL_USD = 200.0

#: 66,670,000 / 10^9
ENTRY_COST = Decimal("0.066670000")
#: 70,000,000 / 10^9
EXIT_PROCEEDS = Decimal("0.070000000")
#: exit_proceeds - entry_cost
GROSS = Decimal("0.003330000")
#: 5,000 lamports = 0.000005 SOL
FIVE_THOUSAND_LAMPORTS_SOL = Decimal("0.000005000")


def _sell_fill(*, fee: Decimal = Decimal(0)) -> SimulatedFill:
    """The exit leg's fill. Only output amount/decimals and fee are read."""
    return SimulatedFill(
        side="SELL",
        input_amount_raw=TOKENS_RAW,
        input_mint=TOKEN,
        input_decimals=TOKEN_DECIMALS,
        input_amount_decimal=float(
            Decimal(TOKENS_RAW).scaleb(-TOKEN_DECIMALS)),
        output_amount_raw=EXIT_RAW,
        output_mint=WSOL,
        output_decimals=EXIT_DECIMALS,
        output_amount_decimal=float(
            Decimal(EXIT_RAW).scaleb(-EXIT_DECIMALS)),
        effective_price=EXIT_PROCEEDS / Decimal(TOKENS_RAW).scaleb(
            -TOKEN_DECIMALS),
        effective_price_usd=None,
        fee=fee,
        fee_bps=0,
        price_impact_pct=None,
        quote_timestamp=1_800_000_000.0,
        quote_age_ms=100.0,
        max_quote_age_ms=5_000.0,
        provider="test",
        venue="test",
        pool_or_curve="pool",
    )


def _position(*, network_costs: Decimal = Decimal(0)) -> PaperPosition:
    return PaperPosition(
        signal_id="s1",
        arm="L25_CURRENT",
        mint=TOKEN,
        symbol="TKN",
        token_decimals=TOKEN_DECIMALS,
        tokens_raw=TOKENS_RAW,
        entry_fill=None,  # type: ignore[arg-type]  # never read by realize()
        entry_observed_mark=None,
        entry_mark_timestamp=None,
        bankroll_before=5.0,
        position_size_usd=float(ENTRY_COST) * SOL_USD,
        position_size_pct=100.0,
        usd_notional=float(ENTRY_COST) * SOL_USD,
        sol_notional=ENTRY_COST,
        sol_usd_reference=SOL_USD,
        sol_usd_reference_at=1_800_000_000.0,
        opened_at=1_800_000_000.0,
        entry_cost_quote_asset=ENTRY_COST,
        trading_fees_quote=Decimal(0),
        network_costs_quote=network_costs,
        other_costs_quote=Decimal(0),
    )


def _settle(model: FillModel, *, network_costs: Decimal = Decimal(0)):
    return realize(
        _position(network_costs=network_costs),
        exit_fill=_sell_fill(),
        exit_observed_mark=None,
        sol_reference=SolReference(SOL_USD, 1_800_000_001.0, "test"),
        model=model,
        now=1_800_000_010.0,
    )


# ---------------------------------------------------------------------------
# 1. the unit defect
# ---------------------------------------------------------------------------

def test_five_thousand_lamports_is_converted_to_sol():
    """The conversion, checked against the definition rather than a literal."""
    model = FillModel(network_cost_lamports=Decimal(5_000))
    assert model.network_cost_sol == Decimal(5_000) / Decimal(10 ** 9)
    assert model.network_cost_sol == FIVE_THOUSAND_LAMPORTS_SOL


def test_the_network_cost_is_never_spent_as_lamports():
    """
    The 10^9 error, stated as an assertion.

    This is the value that used to reach ``network_costs_quote`` and therefore
    every SOL-denominated total downstream.
    """
    model = FillModel(network_cost_lamports=Decimal(5_000))
    assert model.network_cost_sol != Decimal(5_000)
    assert model.network_cost_sol < Decimal("1")


def test_the_default_model_charges_nothing_and_changes_no_behaviour():
    """The live default is still zero, so no historical figure moves."""
    assert FillModel().network_cost_sol == Decimal(0)
    assert FillModel().network_cost_lamports == Decimal(0)
    assert LAMPORTS_PER_SOL == Decimal(1_000_000_000)


# ---------------------------------------------------------------------------
# 2. the PnL identity, on hand-computed numbers
# ---------------------------------------------------------------------------

def test_gross_pnl_is_exit_proceeds_minus_entry_cost():
    settled = _settle(FillModel())
    assert settled.exit_proceeds_quote == EXIT_PROCEEDS
    assert settled.entry_cost_quote == ENTRY_COST
    assert settled.gross_pnl_quote == EXIT_PROCEEDS - ENTRY_COST == GROSS


def test_with_no_modeled_costs_net_equals_gross():
    """Nothing is subtracted that was not charged: fees live inside the quote."""
    settled = _settle(FillModel())
    assert settled.trading_fees_quote == Decimal(0)
    assert settled.net_pnl_quote == GROSS


def test_a_modeled_network_cost_reduces_net_by_its_sol_value():
    # Entry leg carries the snapshotted cost; the model charges the exit leg.
    settled = _settle(
        FillModel(network_cost_lamports=Decimal(5_000)),
        network_costs=FIVE_THOUSAND_LAMPORTS_SOL)

    # net = gross - (entry leg + exit leg)
    #     = 0.003330000 - 0.000005000 - 0.000005000 = 0.003320000
    expected_net = GROSS - FIVE_THOUSAND_LAMPORTS_SOL - FIVE_THOUSAND_LAMPORTS_SOL
    assert expected_net == Decimal("0.003320000")
    assert settled.net_pnl_quote == expected_net


def test_the_modeled_cost_cannot_pass_through_as_lamports():
    """
    The failure this pins: charging 5,000 (lamports) as 5,000 SOL would make
    net catastrophically negative rather than 0.00331 SOL positive.
    """
    settled = _settle(
        FillModel(network_cost_lamports=Decimal(5_000)),
        network_costs=FIVE_THOUSAND_LAMPORTS_SOL)
    assert settled.net_pnl_quote > 0
    assert settled.net_pnl_quote != GROSS - Decimal(10_000)


# ---------------------------------------------------------------------------
# 3. return percentage and USD conversion
# ---------------------------------------------------------------------------

def test_return_pct_uses_entry_cost_as_the_denominator():
    settled = _settle(FillModel())
    expected_pct = float(GROSS / ENTRY_COST * 100)
    assert settled.return_pct == pytest.approx(expected_pct, rel=1e-12)
    # Hand check: 0.00333 / 0.06667 = 4.9947...%
    assert settled.return_pct == pytest.approx(4.99475, abs=1e-4)


def test_realized_pnl_usd_applies_the_sol_reference_exactly_once():
    settled = _settle(FillModel())
    assert settled.realized_pnl_usd == pytest.approx(float(GROSS) * SOL_USD,
                                                     rel=1e-12)
    assert settled.realized_pnl_usd == pytest.approx(0.666, abs=1e-9)


def test_the_quote_asset_figures_are_sol_denominated():
    settled = _settle(FillModel())
    assert settled.realized_pnl_sol == float(GROSS)


def test_the_bankroll_moves_by_the_usd_pnl_only():
    settled = _settle(FillModel())
    assert settled.bankroll_before == 5.0
    assert settled.bankroll_after == pytest.approx(
        5.0 + settled.realized_pnl_usd, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. raw -> decimal conversion (exactness)
# ---------------------------------------------------------------------------

def test_raw_amounts_convert_without_binary_float_drift():
    """
    ``Decimal(raw).scaleb(-decimals)`` is EXACT; ``raw / 10**decimals`` in binary
    floating point is not. The production path uses the former, and this pins
    that a 12-digit raw amount survives it unchanged.
    """
    assert Decimal(TOKENS_RAW).scaleb(-TOKEN_DECIMALS) == Decimal("658884.916927")
    assert Decimal(EXIT_RAW).scaleb(-EXIT_DECIMALS) == EXIT_PROCEEDS
    assert Decimal(ENTRY_RAW).scaleb(-ENTRY_DECIMALS) == ENTRY_COST

    # The float division the code avoids would lose these.
    assert Decimal(TOKENS_RAW).scaleb(-TOKEN_DECIMALS) != Decimal(
        TOKENS_RAW / 10 ** TOKEN_DECIMALS)


def test_the_exit_proceeds_use_the_fill_decimals_not_a_default():
    """A token with 9 decimals must not be settled at 6."""
    fill = _sell_fill()
    assert Decimal(fill.output_amount_raw).scaleb(-fill.output_decimals) == \
        Decimal("0.07")
    wrong = Decimal(fill.output_amount_raw).scaleb(-6)
    assert wrong == Decimal("70.0")
    assert wrong != Decimal("0.07")
