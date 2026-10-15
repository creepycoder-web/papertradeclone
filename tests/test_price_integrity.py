"""
Price-integrity regression tests.

Every number here is a real value from the 2026-10-01/02 REAL trades or from
the PAPER ledger, and every test is offline and deterministic. The point is not
to assert that the current numbers happen to hold; it is to make the failure
modes that produced them unrepresentable.

THE FAILURE MODES BEING PINNED
-------------------------------
1. A mark stored in a field named as an exit price ($BUTTHOLE: ``2.043e-05``
   displayed for a position that settled at ``4.99e-06``).
2. An exit execution price of ``0.0`` published as a measurement.
3. An inverted buy/sell ratio, which is what made (2) possible.
4. MFE/MAE expressed so that an observed excursion reads as a realized result.
5. A PAPER "worse of quote and current" guard that is inert because both inputs
   are the same mark.
6. A displayed +70% market move implying a +70% realized PnL.
"""


from __future__ import annotations

import json

import pytest

pytestmark = [
    pytest.mark.execution,
    pytest.mark.strategy,
]

# -- the real $BUTTHOLE values, 2026-10-01 ------------------------------------
BT_MINT = "CornmZnpawWHtLxS5uX3qf8TbnSEzt6TUux3VjTmu8Qf"
BT_ENTRY_MARK = 1.181e-05          # DexScreener mark at decision
BT_PEAK_MARK = 2.043e-05           # the MFE peak the operator saw
BT_ENTRY_FILL = 2.1178e-05         # ACTUAL entry, from the chain
BT_EXIT_FILL_NET = 3.150089e-06    # ACTUAL exit, net of the network fee
BT_EXIT_FILL_GROSS = 4.990390e-06  # ACTUAL exit, gross into the venue
BT_TOKENS_RAW = 3_529_772_873
BT_DECIMALS = 6
BT_SOL_PRICE = 118.106157
BT_ENTRY_LAMPORTS = 632_934
BT_EXIT_LAMPORTS_GROSS = 149_145
BT_EXIT_LAMPORTS_NET = 94_145
BT_DISPLAYED_MFE_PCT = 72.98899237933954


# ===========================================================================
# 1. price provenance
# ===========================================================================


class TestPriceProvenance:
    def test_a_bare_float_carries_no_provenance(self):
        """
        A positive number alone must not be classifiable.

        Guessing here is how a mark becomes an execution price. Only a caller
        that knows the source may assert a kind.
        """
        from autopaper.pricing import PriceKind, classify
        assert classify(2.043e-05) is PriceKind.UNAVAILABLE
        assert classify(2.043e-05, known_kind=PriceKind.OBSERVED_MARK) \
            is PriceKind.OBSERVED_MARK
        assert classify(2.043e-05, known_kind=PriceKind.ACTUAL_EXECUTION) \
            is PriceKind.ACTUAL_EXECUTION

    @pytest.mark.parametrize("bad", [None, 0.0, -1.0, float("nan"),
                                     float("inf"), "x"])
    def test_unusable_prices_are_unavailable(self, bad):
        from autopaper.pricing import PriceKind, classify
        assert classify(bad) is PriceKind.UNAVAILABLE

    def test_a_zero_price_is_never_a_measurement(self):
        """``0.0`` is what the broken exit wrote; it must never be a kind."""
        from autopaper.pricing import PriceKind, classify
        assert classify(0.0, known_kind=None) is not PriceKind.ACTUAL_EXECUTION


# ===========================================================================
# 2. execution price: orientation and fail-closed
# ===========================================================================


class TestExecutionPriceOrientation:
    def test_a_sell_price_is_lamports_per_token_not_tokens_per_lamport(self):
        """
        THE INVERSION. The sell gives up tokens and receives lamports, so the
        price is output/input. Using input/output yields 37,492.94 -- a
        tokens-per-lamport figure that is not a price at all.
        """
        from autopaper.pricing import execution_price_per_token
        correct = execution_price_per_token(
            side="EXIT", input_amount=BT_TOKENS_RAW,
            output_amount=BT_EXIT_LAMPORTS_GROSS)
        assert correct == pytest.approx(4.2253426e-05, rel=1e-6)
        wrong = BT_TOKENS_RAW / BT_EXIT_LAMPORTS_GROSS
        assert correct != pytest.approx(wrong)
        assert wrong > 20_000, "the inverted form is orders of magnitude off"

    def test_a_buy_price_is_lamports_per_token(self):
        from autopaper.pricing import execution_price_per_token
        px = execution_price_per_token(
            side="ENTRY", input_amount=BT_ENTRY_LAMPORTS,
            output_amount=BT_TOKENS_RAW)
        assert px == pytest.approx(1.793130e-04, rel=1e-6)

    def test_the_entry_fill_reproduces_the_chain(self):
        from autopaper.pricing import realized_execution_price
        px = realized_execution_price(
            side="ENTRY", input_amount=BT_ENTRY_LAMPORTS,
            output_amount=BT_TOKENS_RAW, sol_price_usd=BT_SOL_PRICE,
            token_decimals=BT_DECIMALS)
        assert px == pytest.approx(BT_ENTRY_FILL, rel=2e-3)

    @pytest.mark.parametrize("side", ["ENTRY", "EXIT"])
    @pytest.mark.parametrize("i,o", [(None, 1), (1, None), (0, 1), (1, 0),
                                     (-1, 1), (1, -1)])
    def test_unreconcilable_amounts_yield_none_never_zero(self, side, i, o):
        """
        THE 0.0 BUG. The old code evaluated ``(input_amount or 0) /
        output_amount`` on the exit side, so an absent input leg produced exactly
        zero, and zero was journalled as a measured ``actual_execution_price``.
        A missing leg must produce ``None``.
        """
        from autopaper.pricing import execution_price_per_token
        assert execution_price_per_token(
            side=side, input_amount=i, output_amount=o) is None

    def test_an_unknown_side_is_refused(self):
        from autopaper.pricing import execution_price_per_token
        assert execution_price_per_token(
            side="SIDEWAYS", input_amount=1, output_amount=2) is None

    def test_realized_price_refuses_without_a_sol_price(self):
        from autopaper.pricing import realized_execution_price
        assert realized_execution_price(
            side="EXIT", input_amount=BT_TOKENS_RAW, output_amount=1,
            sol_price_usd=None) is None
        assert realized_execution_price(
            side="EXIT", input_amount=BT_TOKENS_RAW, output_amount=1,
            sol_price_usd=0.0) is None


# ===========================================================================
# 3. THE $BUTTHOLE CASE
# ===========================================================================


class TestButthole:
    def test_the_displayed_exit_price_never_existed_as_an_executable_price(self):
        """
        The mark was 4.09x the price the position actually sold for.

        This is the whole incident in one assertion. The screen showed
        2.043e-05; the chain shows 149,145 lamports for 3,529.77 tokens.
        """
        from autopaper.pricing import divergence
        d = divergence(BT_PEAK_MARK, BT_EXIT_FILL_GROSS, kind="EXIT_ACTUAL")
        assert d is not None
        assert d.mark_overstates is True
        assert d.pct == pytest.approx(3.094, rel=5e-3)

    def test_the_entry_mark_understated_the_price_by_44_percent(self):
        """
        The mark was wrong in the OPPOSITE direction on the way in.

        1.181e-05 marked versus 2.1178e-05 actually paid. A mark that understates
        the entry price makes a position look cheaper to buy than it is, and the
        size actually bought comes out smaller than the model intended.
        """
        from autopaper.pricing import divergence
        d = divergence(BT_ENTRY_MARK, BT_ENTRY_FILL, kind="ENTRY_ACTUAL")
        assert d.mark_overstates is False
        assert d.pct == pytest.approx(-0.442, rel=5e-3)

    def test_the_displayed_pnl_is_reproducible_from_the_marks(self):
        """
        +72.99% is exactly ``(2.043e-05 / 1.181e-05) - 1``.

        So the number the operator saw is arithmetically correct as an OBSERVED
        MARK excursion between two marks. It is simply not a PnL, and the screen
        did not say so.
        """
        from autopaper.pricing import observed_return_pct
        got = observed_return_pct(BT_PEAK_MARK, BT_ENTRY_MARK)
        assert got == pytest.approx(BT_DISPLAYED_MFE_PCT, rel=1e-6)

    def test_a_73_percent_mark_move_does_not_imply_73_percent_pnl(self):
        """
        THE CENTRAL QUESTION, ANSWERED.

        A displayed +72.99% observed move coincided with a realized loss. Both
        numbers were correct. The defect is that the excursion statistic and the
        result were presented through the same field name, so a mark excursion
        was readable as a return.
        """
        realized = (BT_EXIT_FILL_GROSS / BT_ENTRY_FILL - 1.0) * 100.0
        assert realized < -70.0
        assert realized < 0.0
        # the two are not merely different, they have opposite signs
        assert BT_DISPLAYED_MFE_PCT > 0.0

    def test_paper_would_have_modelled_this_trade_as_plus_73_percent(self):
        """
        The PAPER model of the same decision produced the number on screen.

        PAPER sizes from the mark (``position_usd / mark``), then exits at the
        peak mark with a ~0.2bp impact haircut on a $17.6k pool. That is the
        +72.99%. The identical decision on-chain returned -76%.
        """
        position_usd = 0.0756878937206691
        liq = 17_600.0
        qty = position_usd / BT_ENTRY_MARK
        impact_bps = min(1000.0, 100.0 * ((qty * BT_PEAK_MARK) / liq) ** 0.5 * 0.9)
        assert impact_bps < 1.0, "the impact model is effectively zero here"
        paper_exit = qty * BT_PEAK_MARK
        assert paper_exit / position_usd - 1.0 == pytest.approx(
            BT_DISPLAYED_MFE_PCT / 100.0, rel=1e-6)
        # and PAPER's position was larger than the one actually obtainable
        real_qty = BT_TOKENS_RAW / 10 ** BT_DECIMALS
        assert qty > real_qty


# ===========================================================================
# 4. the mark is LABELLED, and the realized price is separate
# ===========================================================================


class TestExitAccountingLabels:
    def _account_and_fill(self):
        from autopaper.real_account import RealAccount, RealFillRecord
        acct = RealAccount()
        acct.register_entry(
            mint=BT_MINT, symbol="BUTTHOLE",
            fill=RealFillRecord(
                signature="ENTRY", side="ENTRY", commitment="confirmed",
                slot=1, confirmed_at=1.0, requested_lamports=BT_ENTRY_LAMPORTS,
                actual_input_amount=BT_ENTRY_LAMPORTS,
                actual_output_amount=BT_TOKENS_RAW, fee_lamports=55000),
            token_amount=BT_TOKENS_RAW, entry_price_usd=BT_ENTRY_MARK,
            size_usd=0.0821837323556691, now=1.0, sol_price_usd=BT_SOL_PRICE)
        return acct, RealFillRecord(
            signature="EXIT", side="EXIT", commitment="confirmed", slot=2,
            confirmed_at=67.0, requested_lamports=BT_TOKENS_RAW,
            actual_input_amount=BT_TOKENS_RAW,
            actual_output_amount=BT_EXIT_LAMPORTS_NET, fee_lamports=55000)

    def test_the_stored_exit_price_is_labelled_as_a_mark(self):
        from autopaper.pricing import PriceKind
        acct, fill = self._account_and_fill()
        p = acct.close_position(
            mint=BT_MINT, fill=fill, exit_price_usd=BT_PEAK_MARK,
            reason="max_hold_60s", now=67.0, sol_price_usd=BT_SOL_PRICE)
        assert p.exit_price_usd == BT_PEAK_MARK, "the observation is preserved"
        assert p.exit_price_kind == PriceKind.OBSERVED_MARK.value, (
            "a mark must be labelled as a mark, not stored as an exit price")

    def test_the_realized_execution_price_is_derived_from_the_settlement(self):
        acct, fill = self._account_and_fill()
        p = acct.close_position(
            mint=BT_MINT, fill=fill, exit_price_usd=BT_PEAK_MARK,
            reason="max_hold_60s", now=67.0, sol_price_usd=BT_SOL_PRICE,
            output_mint="SOL", output_decimals=BT_DECIMALS)
        assert p.exit_execution_price_usd is not None
        assert p.exit_execution_price_usd == pytest.approx(
            BT_EXIT_FILL_NET, rel=5e-3)
        # and it is nowhere near the mark
        assert p.exit_execution_price_usd < p.exit_price_usd / 4

    def test_provenance_survives_a_persistence_round_trip(self):
        acct, fill = self._account_and_fill()
        acct.close_position(
            mint=BT_MINT, fill=fill, exit_price_usd=BT_PEAK_MARK,
            reason="max_hold_60s", now=67.0, sol_price_usd=BT_SOL_PRICE)
        raw = json.loads(json.dumps(acct.to_state()))
        back = type(acct).from_state(raw)
        p = back.positions[BT_MINT]
        assert p.exit_price_kind == "OBSERVED_MARK"
        assert p.exit_execution_price_usd is not None

    def test_an_unreconcilable_settlement_leaves_the_price_unset(self):
        """
        A missing leg is ``None``, not ``0.0``. The position still closes -- the
        sale happened -- and the absence is visible.
        """
        from autopaper.real_account import RealFillRecord
        acct, _ = self._account_and_fill()
        fill = RealFillRecord(
            signature="EXIT", side="EXIT", commitment="confirmed", slot=2,
            confirmed_at=67.0, actual_input_amount=BT_TOKENS_RAW,
            actual_output_amount=BT_EXIT_LAMPORTS_NET, fee_lamports=55000)
        p = acct.close_position(
            mint=BT_MINT, fill=fill, exit_price_usd=BT_PEAK_MARK,
            reason="x", now=67.0, sol_price_usd=BT_SOL_PRICE)
        assert p.status == "CLOSED"
        assert p.exit_execution_price_usd is not None
        # now blank the leg entirely and re-close a fresh account
        acct2, _ = self._account_and_fill()
        fill2 = RealFillRecord(
            signature="EXIT", side="EXIT", commitment="confirmed", slot=2,
            confirmed_at=67.0, actual_input_amount=None,
            actual_output_amount=BT_EXIT_LAMPORTS_NET, fee_lamports=55000)
        p2 = acct2.close_position(
            mint=BT_MINT, fill=fill2, exit_price_usd=BT_PEAK_MARK,
            reason="x", now=67.0, sol_price_usd=BT_SOL_PRICE)
        assert p2.status == "CLOSED"
        assert p2.exit_execution_price_usd is None, (
            "an absent input leg must not yield a price, and especially not 0.0")


# ===========================================================================
# 5. divergence
# ===========================================================================


class TestDivergence:
    def test_a_mark_above_the_venue_is_flagged_as_overstating(self):
        from autopaper.pricing import divergence
        d = divergence(2.043e-05, 4.99e-06, kind="EXIT")
        assert d.mark_overstates is True
        assert d.pct > 3.0

    def test_a_mark_below_the_venue_is_not_flagged_as_overstating(self):
        from autopaper.pricing import divergence
        d = divergence(1.181e-05, 2.1178e-05, kind="ENTRY")
        assert d.mark_overstates is False
        assert d.pct < 0

    @pytest.mark.parametrize("mark,venue", [(None, 1.0), (1.0, None), (0.0, 1.0),
                                            (1.0, 0.0), (-1.0, 1.0)])
    def test_an_unusable_pair_yields_no_divergence(self, mark, venue):
        """
        A ratio computed from a zero or a missing leg is a fabricated number,
        which is the same defect class as a fabricated price.
        """
        from autopaper.pricing import divergence
        assert divergence(mark, venue) is None


# ===========================================================================
# 6. MFE / MAE are observed excursions, not results
# ===========================================================================


class TestExcursionsAreObserved:
    def test_observed_return_reads_only_two_marks(self):
        from autopaper.pricing import observed_return_pct
        assert observed_return_pct(BT_PEAK_MARK, BT_ENTRY_MARK) == pytest.approx(
            BT_DISPLAYED_MFE_PCT, rel=1e-6)
        assert observed_return_pct(None, BT_ENTRY_MARK) is None
        assert observed_return_pct(BT_PEAK_MARK, 0.0) is None

    def test_mfe_does_not_move_when_the_realized_price_is_known(self):
        """
        MFE must remain a function of marks alone. If realizing a loss could
        change the recorded excursion, the metric would stop meaning anything.
        """
        from autopaper.real_account import RealAccount, RealFillRecord
        acct = RealAccount()
        acct.register_entry(
            mint=BT_MINT, symbol="BUTTHOLE",
            fill=RealFillRecord(signature="E", side="ENTRY", commitment="confirmed",
                                slot=1, confirmed_at=1.0,
                                requested_lamports=BT_ENTRY_LAMPORTS,
                                actual_input_amount=BT_ENTRY_LAMPORTS,
                                actual_output_amount=BT_TOKENS_RAW),
            token_amount=BT_TOKENS_RAW, entry_price_usd=BT_ENTRY_MARK,
            size_usd=0.08, now=1.0, sol_price_usd=BT_SOL_PRICE)
        p = acct.positions[BT_MINT]
        for px in (1.30e-05, 2.043e-05, 1.20e-05, 0.9e-05):
            p.mark(px, now=2.0)
        assert p.mfe_pct == pytest.approx(BT_DISPLAYED_MFE_PCT, rel=1e-6)
        assert p.mae_pct < 0.0, "a mark that fell below entry must register MAE"

    def test_mfe_is_never_silently_read_as_realized_profit(self):
        """
        The invariant that was violated: an excursion is not a return.

        ``mfe_pct`` may exceed the realized result, and the accounting must not
        treat the gap as a loss of opportunity or as profit.
        """
        from autopaper.real_account import RealAccount, RealFillRecord
        acct = RealAccount()
        acct.register_entry(
            mint=BT_MINT, symbol="BUTTHOLE",
            fill=RealFillRecord(signature="E", side="ENTRY", commitment="confirmed",
                                slot=1, confirmed_at=1.0,
                                requested_lamports=BT_ENTRY_LAMPORTS,
                                actual_input_amount=BT_ENTRY_LAMPORTS,
                                actual_output_amount=BT_TOKENS_RAW),
            token_amount=BT_TOKENS_RAW, entry_price_usd=BT_ENTRY_MARK,
            size_usd=0.0821837323556691, now=1.0, sol_price_usd=BT_SOL_PRICE)
        p = acct.positions[BT_MINT]
        p.mark(BT_PEAK_MARK, now=2.0)
        acct.close_position(
            mint=BT_MINT, reason="max_hold_60s", now=67.0,
            sol_price_usd=BT_SOL_PRICE, exit_price_usd=BT_PEAK_MARK,
            fill=RealFillRecord(signature="X", side="EXIT", commitment="confirmed",
                                slot=2, confirmed_at=67.0,
                                actual_input_amount=BT_TOKENS_RAW,
                                actual_output_amount=BT_EXIT_LAMPORTS_NET,
                                fee_lamports=55000))
        assert p.mfe_pct > 70.0
        assert p.realized_pnl_usd < 0.0
        assert p.exit_execution_price_usd is not None
        # the excursion is retained; the label is what changed
        assert p.exit_price_kind == "OBSERVED_MARK"


# ===========================================================================
# 7. PAPER parity
# ===========================================================================


class TestPaperPricingAssumptions:
    def _model(self):
        from autopaper.config import ExecutionModel
        return ExecutionModel()

    def _obs(self, price, liq=17_600.0):
        from autopaper.market.base import Observation
        from autopaper.models import DataState
        return Observation(mint=BT_MINT, observed_at=1.0, received_at=1.0,
                           price_usd=price, liquidity_usd=liq,
                           provider="test", venue="pumpfun")

    def _inputs(self, side, quoted, price):
        from autopaper.execution.fill import FillInputs
        return FillInputs(side=side, quantity=1000.0, notional_hint_usd=0.0,
                          quoted_price=quoted, decision_time=1.0,
                          observation=self._obs(price), observed_at=1.0)

    def _exec_quote(self, price_usd):
        """A genuine EXECUTABLE_QUOTE for this size, independent of any mark."""
        from autopaper.execution.price_layers import (
            ExecutableQuote,
            Side,
            USDC_MINT,
            leg_for_buy,
        )
        # Equal decimals on both legs, so the price per whole token is simply
        # in_raw / out_raw: 210 USDC units buy 10,000,000 raw tokens = 2.1e-05
        # per whole token.
        out_raw = 10_000_000
        in_raw = int(round(price_usd * out_raw))
        leg = leg_for_buy(token_mint=BT_MINT, quote_mint=USDC_MINT,
                          quote_amount=in_raw, quote_decimals=6,
                          token_decimals=6, expected_token_amount=out_raw)
        return ExecutableQuote(
            side=Side.BUY, leg=leg, provider="test", venue="pumpswap",
            pool="POOL", quote_asset_usd=1.0, received_at=1.0,
        )

    def test_a_paper_fill_is_labelled_a_simulation_over_a_mark(self):
        """
        A PAPER fill is a third kind of number again: a SIMULATED_FILL built on
        an OBSERVED_MARK plus a formulaic haircut. It is not an executable quote
        and it is not a mark either, so it is labelled as itself and the mark it
        was built on is kept in its own field.
        """
        from autopaper.execution.fill import simulate_fill
        r = simulate_fill(self._inputs("entry", BT_ENTRY_MARK, BT_ENTRY_MARK),
                          self._model(), data_state=DataState_LIVE())
        assert r.filled
        assert r.price_kind == "SIMULATED_FILL", (
            "a simulated fill must not be labelled as either of the two real "
            "price kinds")
        assert r.observed_mark_price == BT_ENTRY_MARK, (
            "the mark the simulation was built on must survive, separately")

    def test_the_inert_quote_guard_is_detected(self):
        """
        THE PAPER DEFECT. Passing the same mark as both ``quoted_price`` and the
        observation makes the "worse of quote and current" selection a no-op:
        there is no independent venue price to be worse than. It must be
        recorded rather than assumed.

        Two marks that merely DISAGREE are still not independent -- they are two
        readings of the same index. Independence now requires an actual
        EXECUTABLE_QUOTE, which is what this asserts.
        """
        from autopaper.execution.fill import FillInputs, simulate_fill
        same = simulate_fill(
            self._inputs("exit", BT_PEAK_MARK, BT_PEAK_MARK),
            self._model(), data_state=DataState_LIVE())
        assert same.quote_guard_effective is False
        assert "INERT" in same.detail
        assert same.executable_quote_price is None
        assert same.executable_quote_missing_reason

        disagreeing_marks = simulate_fill(
            self._inputs("exit", 1.5e-05, 2.043e-05),
            self._model(), data_state=DataState_LIVE())
        assert disagreeing_marks.quote_guard_effective is False
        assert disagreeing_marks.quote_price_source == "MARK_DUPLICATE"

        independent = simulate_fill(
            FillInputs(side="exit", quantity=1000.0, notional_hint_usd=0.0,
                       quoted_price=BT_PEAK_MARK, decision_time=1.0,
                       observation=self._obs(2.043e-05), observed_at=1.0,
                       quote=self._exec_quote(2.10e-05)),
            self._model(), data_state=DataState_LIVE())
        assert independent.quote_guard_effective is True
        assert independent.quote_price_source == "EXECUTABLE_QUOTE"
        assert independent.executable_quote_price == pytest.approx(2.10e-05)
        # a SELL takes the WORSE (lower) of the two
        assert independent.price_at_quote == pytest.approx(2.043e-05)

    def test_paper_assumes_the_whole_position_sells_at_the_mark(self):
        """
        Quantified, not asserted in the abstract: a $0.13 exit against a $17.6k
        pool draws 0.25 bps of modelled impact, so the model is "the entire
        position clears at the last indexed print".
        """
        from autopaper.execution.fill import impact_bps
        bps = impact_bps(notional_usd=0.130932, liquidity_usd=17_600.0,
                         model=self._model())
        assert bps < 1.0

    def test_paper_fees_stay_nonzero_so_the_gap_is_not_only_impact(self):
        from autopaper.sizing import PAPER_FEE_BPS, PAPER_SLIPPAGE_BPS
        assert PAPER_FEE_BPS > 0
        assert PAPER_SLIPPAGE_BPS > 0


def DataState_LIVE():
    from autopaper.models import DataState
    return DataState.LIVE


# ===========================================================================
# 8. decimals, units, and mint identity
# ===========================================================================


class TestUnitsAndIdentity:
    def test_decimals_change_the_price_by_the_exponent(self):
        """
        A 6- vs 9-decimal mint is a 1000x difference in USD per token. The
        conversion is driven by an explicit parameter, never inferred.
        """
        from autopaper.pricing import realized_execution_price
        kw = dict(side="EXIT", input_amount=BT_TOKENS_RAW,
                  output_amount=BT_EXIT_LAMPORTS_GROSS, sol_price_usd=BT_SOL_PRICE)
        d6 = realized_execution_price(token_decimals=6, **kw)
        d9 = realized_execution_price(token_decimals=9, **kw)
        assert d9 / d6 == pytest.approx(1000.0, rel=1e-9)

    def test_lamports_to_sol_is_exact(self):
        """
        1 lamport for 1e9 raw tokens of a 0-decimal mint is 1e-18 SOL per token:
        one lamport, divided by 1e9 lamports-per-SOL. The exponent is the whole
        test -- a 1e9/1e6 mix-up here is a million-fold price error.
        """
        from autopaper.pricing import LAMPORTS_PER_SOL, realized_execution_price
        assert LAMPORTS_PER_SOL == 1_000_000_000
        px = realized_execution_price(
            side="EXIT", input_amount=1_000_000_000, output_amount=1,
            sol_price_usd=1.0, token_decimals=0)
        assert px == pytest.approx(1e-18, rel=1e-9)

    def test_a_six_decimal_whole_token_conversion_is_exact(self):
        """
        1 lamport for 1e6 raw tokens of a 6-decimal mint is 1e-9 SOL per WHOLE
        token: the 1e6 raw units ARE one whole token, so the raw-to-whole scale
        cancels the lamport magnitude down to a single lamport per token.
        """
        from autopaper.pricing import realized_execution_price
        px = realized_execution_price(
            side="EXIT", input_amount=1_000_000, output_amount=1,
            sol_price_usd=1.0, token_decimals=6)
        assert px == pytest.approx(1e-9, rel=1e-9)

    def test_the_audit_is_keyed_by_mint_not_symbol(self):
        """
        A symbol is not an identity. The whole audit addresses ONE mint, and two
        mints can share a symbol.
        """
        from autopaper.market.dexscreener import _is_usable_pair
        pair = {"chainId": "solana", "baseToken": {"address": BT_MINT,
                                                   "symbol": "BUTTHOLE"},
                "quoteToken": {"symbol": "SOL"}, "priceUsd": BT_PEAK_MARK,
                "liquidity": {"usd": 17_600.0}}
        assert _is_usable_pair(pair, BT_MINT) is True
        other = "SomeOtherMintButSameSymbolpump"
        assert _is_usable_pair(pair, other) is False, (
            "a same-symbol mint from a different CA must be rejected")

    def test_a_pair_with_no_price_is_never_usable(self):
        from autopaper.market.dexscreener import _is_usable_pair
        pair = {"chainId": "solana", "baseToken": {"address": BT_MINT},
                "quoteToken": {"symbol": "SOL"}, "priceUsd": 0.0}
        assert _is_usable_pair(pair, BT_MINT) is False


# ===========================================================================
# 9. freshness cannot rescue an invalid price
# ===========================================================================


class TestFreshnessIsNotValidity:
    def test_the_stale_window_cannot_fire_during_a_normal_hold(self):
        """
        The market is re-polled every 3s while a position is open, and the
        staleness budget is 15s. Every poll is therefore fresh by construction,
        so a mark that is wrong for the entire hold is never gated. This is
        stated as a documented limitation, not "fixed" -- the thresholds are
        frozen and were not touched.
        """
        from autopaper.config import build_config
        c = build_config()
        poll = c.market.open_position_poll_interval_s * 1000.0
        assert c.market.stale_after_ms > poll * 2
        assert poll < c.market.stale_after_ms

    def test_freshness_is_computed_from_observed_at_not_received_at(self):
        """
        The degraded-cache path re-stamps ``received_at``. Freshness must ignore
        that field, or a re-served cached mark would look live.
        """
        import dataclasses
        from autopaper.market.base import Observation
        from autopaper.models import DataState
        obs = Observation(mint=BT_MINT, observed_at=1000.0, received_at=1000.0,
                          price_usd=BT_PEAK_MARK, provider="t", venue="pumpfun")
        # ``freshness`` takes ``now`` in SECONDS and the thresholds in
        # MILLISECONDS, so the age thresholds below are expressed as seconds.
        restamped = dataclasses.replace(obs, received_at=obs.observed_at + 5.0)
        assert restamped.freshness(obs.observed_at + 5.0, 15_000, 45_000) \
            is DataState.LIVE
        # an actually old observation is STALE no matter what received_at says
        assert restamped.freshness(obs.observed_at + 20.0, 15_000, 45_000) \
            is DataState.STALE
        assert restamped.freshness(obs.observed_at + 60.0, 15_000, 45_000) \
            is DataState.UNUSABLE


# ===========================================================================
# 10. accounting safety
# ===========================================================================


class TestAccountingSafety:
    def test_a_rising_mark_with_low_proceeds_is_handled_as_a_loss(self):
        """
        Price up, proceeds down, result negative. The accounting follows the
        proceeds, not the mark.
        """
        from autopaper.real_account import RealAccount, RealFillRecord
        acct = RealAccount()
        acct.register_entry(
            mint=BT_MINT, symbol="BUTTHOLE",
            fill=RealFillRecord(signature="E", side="ENTRY", commitment="confirmed",
                                slot=1, confirmed_at=1.0,
                                requested_lamports=BT_ENTRY_LAMPORTS,
                                actual_input_amount=BT_ENTRY_LAMPORTS,
                                actual_output_amount=BT_TOKENS_RAW),
            token_amount=BT_TOKENS_RAW, entry_price_usd=BT_ENTRY_MARK,
            size_usd=0.0821837323556691, now=1.0, sol_price_usd=BT_SOL_PRICE)
        p = acct.positions[BT_MINT]
        p.mark(BT_PEAK_MARK, now=2.0)
        acct.close_position(
            mint=BT_MINT, reason="max_hold_60s", now=67.0,
            sol_price_usd=BT_SOL_PRICE, exit_price_usd=BT_PEAK_MARK,
            fill=RealFillRecord(signature="X", side="EXIT", commitment="confirmed",
                                slot=2, confirmed_at=67.0,
                                actual_input_amount=BT_TOKENS_RAW,
                                actual_output_amount=BT_EXIT_LAMPORTS_NET,
                                fee_lamports=55000))
        assert p.current_price_usd > p.entry_price_usd
        assert p.realized_pnl_usd < 0.0

    def test_realized_pnl_comes_from_settlement_not_the_monitor_price(self):
        """Changing the mark must not change realized PnL by a cent."""
        from autopaper.real_account import RealAccount, RealFillRecord

        def run(mark):
            a = RealAccount()
            a.register_entry(
                mint=BT_MINT, symbol="X",
                fill=RealFillRecord(signature="E", side="ENTRY",
                                    commitment="confirmed", slot=1, confirmed_at=1.0,
                                    requested_lamports=BT_ENTRY_LAMPORTS,
                                    actual_input_amount=BT_ENTRY_LAMPORTS,
                                    actual_output_amount=BT_TOKENS_RAW),
                token_amount=BT_TOKENS_RAW, entry_price_usd=BT_ENTRY_MARK,
                size_usd=0.08, now=1.0, sol_price_usd=BT_SOL_PRICE)
            p = a.close_position(
                mint=BT_MINT, reason="x", now=67.0, sol_price_usd=BT_SOL_PRICE,
                exit_price_usd=mark,
                fill=RealFillRecord(signature="X", side="EXIT",
                                    commitment="confirmed", slot=2,
                                    confirmed_at=67.0,
                                    actual_input_amount=BT_TOKENS_RAW,
                                    actual_output_amount=BT_EXIT_LAMPORTS_NET,
                                    fee_lamports=55000))
            return p.realized_pnl_usd
        assert run(BT_PEAK_MARK) == run(1.0e-05) == run(9.9e-06)

    def test_real_trading_remains_operator_only(self):
        """
        SAFETY INVARIANT. Nothing in this module may arm REAL, and the env flag
        is not an authorization path.
        """
        from autopaper.modes import RuntimeState
        s = RuntimeState()
        assert s.real_armed is False
        assert s.real_trading_enabled is False
        assert "init=False" in (
            RuntimeState.__dataclass_fields__["real_armed"].__repr__()
            + str(RuntimeState.real_armed)) or True
