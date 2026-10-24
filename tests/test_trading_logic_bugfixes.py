"""
Regressions for defects found in the trading logic.

Each test below pins ONE defect. The header of each class says what the bug
was, what it did to a trade, and what the correct behaviour is, so a future
reader can tell a deliberate rule from a repaired accident.

Nothing here changes a threshold. Every assertion is about arithmetic, units,
provenance or the agreement of two code paths that measure the same quantity.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from autopaper.account import PaperAccount
from autopaper.analysis.features import FeatureSet, build_features
from autopaper.analysis.regime import classify
from autopaper.analysis.routes import select_route
from autopaper.clock import FakeClock
from autopaper.config import DecisionConfig, ExitConfig, RiskLimits
from autopaper.exits import held_seconds
from autopaper.ledger import PaperAccountState
from autopaper.models import (
    DataState,
    ExitStyle,
    Observation,
    ObservationSet,
    Position,
    PositionStatus,
    Regime,
    Route,
)
from autopaper.sizing import compute_position_size, tradeable

pytestmark = [pytest.mark.strategy, pytest.mark.integrity]

T0 = 1_700_000_000.0
MINT = "So11111111111111111111111111111111111111111"


def obs(at: float, price: float, **kw) -> Observation:
    base = dict(
        liquidity_usd=40_000.0,
        market_cap_usd=15_000.0,
        volume_5m_usd=20_000.0,
        txns_5m=120,
        buys_5m=80,
        sells_5m=40,
        pair_created_at=at - 180.0,
        provider="test",
    )
    base.update(kw)
    return Observation(
        mint=MINT, observed_at=at, received_at=at, price_usd=price, **base
    )


def features_for(prints: list[tuple[float, float]], **kw) -> FeatureSet:
    history = ObservationSet([obs(at, px) for at, px in prints])
    return build_features(
        candidate_id="cand",
        mint=MINT,
        decision_time=T0,
        history=history,
        alert=kw.pop("alert", None),
        nar=None,
        stale_after_ms=15_000.0,
        unusable_after_ms=45_000.0,
        **kw,
    )


# ---------------------------------------------------------------------------
# 1. the placeholder flag defaulted to a TUPLE
# ---------------------------------------------------------------------------


def test_narrative_placeholder_defaults_to_a_real_false():
    """
    ``narrative_placeholder: bool = False,`` -- a stray trailing comma in the
    dataclass body -- made the default the tuple ``(False,)``, which is TRUTHY.
    Every FeatureSet built without the field therefore read as "the narrative
    bot answered with a placeholder" and carried the -0.06 signal-quality
    discount it exists to apply.
    """
    f = FeatureSet(
        candidate_id="c", mint=MINT, decision_time=T0,
        data_state=DataState.LIVE, observation_count=0,
        history_span_s=None, newest_observed_at=None,
    )
    assert f.narrative_placeholder is False
    assert not isinstance(f.narrative_placeholder, tuple)


# ---------------------------------------------------------------------------
# 2. the 2s window measured the whole history
# ---------------------------------------------------------------------------


def test_change_2s_measures_two_seconds_and_not_the_whole_span():
    """
    ``change_2s_pct`` resolved the 2s window-start price and then divided by
    ``all_prices[0]`` -- the oldest print in the ENTIRE history -- so it was a
    duplicate of ``change_over_span_pct`` under a name that says two seconds.
    """
    f = features_for([(T0 - 60, 1.0e-4), (T0 - 1, 2.0e-4), (T0, 2.2e-4)])
    assert f.change_over_span_pct == pytest.approx(120.0)
    # from 2.0e-4 (inside the 2s window), not from 1.0e-4
    assert f.change_2s_pct == pytest.approx(10.0)
    assert f.change_2s_pct != pytest.approx(f.change_over_span_pct)


# ---------------------------------------------------------------------------
# 3. a fabricated 15m change outranked the momentum actually measured
# ---------------------------------------------------------------------------


def test_a_short_tape_reports_no_15m_change_rather_than_zero():
    """
    ``change_15m`` was computed without ``require_full_window``, so on a tape
    shorter than 15 minutes the window-start helper fell back to the newest
    print and the result was a fabricated ``0.0``.

    ``regime.classify`` ranks the 15m window ABOVE the 60s and 30s windows, so
    that zero became the regime's chosen ``move`` and suppressed the momentum
    the agent had actually observed.
    """
    f = features_for([(T0 - 20, 1.0e-4), (T0 - 10, 1.2e-4), (T0, 1.5e-4)])
    assert (f.history_span_s or 0.0) < 900.0
    assert f.change_15m is None, "a 20s tape has no 15m window"
    # the regime now reads the move it really has
    regime, reason = classify(f)
    assert regime is not Regime.STABILIZING
    assert f.change_over_span_pct == pytest.approx(50.0)


def test_a_long_tape_still_reports_its_15m_change():
    f = features_for([(T0 - 1200, 1.0e-4), (T0 - 600, 1.1e-4), (T0, 1.2e-4)])
    assert (f.history_span_s or 0.0) >= 900.0
    assert f.change_15m is not None


# ---------------------------------------------------------------------------
# 4. an anchor equal to the current price read as NO anchor
# ---------------------------------------------------------------------------


class TestExtensionAnchoring:
    """
    ``anchored`` was inferred from ``abs(price - fvp) > 0``, which cannot tell
    "there is no earlier price" from "there IS an earlier price and the token
    has not moved since". The second is extension 0% -- the LEAST extended
    entry the ladder can see -- and it was being reported as unanchored:
    extension ``None``, score 0.35 instead of 1.0, an ``extension_unanchored``
    warning, and the chaseable band removed in ``select_route`` so the setup
    had to carry itself on liquidity and activity alone.
    """

    def test_a_flat_token_is_anchored_at_zero_extension(self):
        f = features_for([(T0 - 20, 1.0e-4), (T0 - 10, 1.0e-4), (T0, 1.0e-4)])
        assert f.extension_anchored is True
        assert f.extension_from_first_valid_price == pytest.approx(0.0)
        assert select_route(f, DecisionConfig()).route is Route.FAST_ENTRY

    def test_a_single_print_is_still_unanchored(self):
        f = features_for([(T0, 1.0e-4)])
        assert f.extension_anchored is False
        assert f.extension_from_first_valid_price is None
        assert f.extension_score <= 0.6

    def test_a_moving_token_keeps_its_measured_extension(self):
        f = features_for([(T0 - 20, 1.0e-4), (T0, 1.04e-4)])
        assert f.extension_anchored is True
        assert f.extension_from_first_valid_price == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# 5. the cash guard ignored the fee the position consumes
# ---------------------------------------------------------------------------


class TestSizeFitsTheCashItConsumes:
    """
    ``tradeable`` compared available cash against the NOTIONAL, while
    ``PaperAccount.register_position`` debits ``notional + entry_fee``. A size
    equal to the whole remaining balance therefore cleared the guard and then
    drove cash negative by the fee, at which point ``register_position`` voided
    the position into ``PositionStatus.ERROR`` instead of this function
    refusing it with a readable reason.
    """

    def test_cash_equal_to_the_notional_is_refused(self):
        size = compute_position_size(10.0)          # $1.00 notional
        ok, why = tradeable(size, available_cash_usd=size.position_usd)
        assert not ok
        assert "entry fee" in why

    def test_cash_covering_notional_plus_fee_is_accepted(self):
        size = compute_position_size(10.0)
        cash = size.position_usd + size.estimated_fee_usd
        ok, why = tradeable(size, available_cash_usd=cash)
        assert ok, why

    def test_the_refusal_still_names_both_numbers(self):
        size = compute_position_size(10.0)
        ok, why = tradeable(size, available_cash_usd=0.50)
        assert not ok
        assert "0.500000" in why and "1.000000" in why

    def test_a_size_that_passes_never_voids_the_position(self):
        """The guard and the debit must agree, on the exact boundary."""
        account = PaperAccount(
            PaperAccountState(starting_bankroll_usd=10.0, cash_usd=10.0),
            RiskLimits(),
        )
        size = compute_position_size(10.0)
        ok, _why = tradeable(size, available_cash_usd=account.cash)
        assert ok
        position = _position(
            notional=size.position_usd, fee=size.estimated_fee_usd
        )
        account.positions[position.position_id] = position
        account.register_position(position)
        assert position.status is PositionStatus.OPEN
        assert account.cash >= 0.0


def _position(*, notional: float, fee: float, fill_price: float = 1.0e-4,
              position_id: str = "pos_test") -> Position:
    quantity = notional / fill_price
    return Position(
        position_id=position_id,
        candidate_id="cand",
        mint=MINT,
        symbol="TEST",
        status=PositionStatus.OPEN,
        strategy="L25",
        route=Route.FAST_ENTRY,
        regime=Regime.EARLY,
        exit_style=ExitStyle.FAST_FAILURE,
        entry_signal_timestamp=T0,
        entry_decision_timestamp=T0,
        entry_timestamp=T0,
        quoted_price=fill_price,
        entry_reference_price=fill_price,
        fill_price=fill_price,
        quantity=quantity,
        notional_usd=notional,
        entry_fee_usd=fee,
        entry_slippage_usd=0.0,
        cash_out_usd=notional + fee,
        cost_basis_usd=notional + fee,
    )


# ---------------------------------------------------------------------------
# 6. a cancelled position refunded the fee it said it had paid
# ---------------------------------------------------------------------------


class TestCancelKeepsTheFeeCharged:
    """
    ``cancel_position`` refunded the whole ``cash_out_usd`` -- notional AND
    entry fee -- while leaving the fee inside ``fees_paid_usd``. Cash returned
    to exactly where it started, so the account reported a fee that had never
    left it and no reconciliation of cash against the cost counters could
    close. The docstring already said the fee is the part a cancellation cannot
    undo; the arithmetic now says it too.
    """

    def _cancelled(self):
        account = PaperAccount(
            PaperAccountState(starting_bankroll_usd=100.0, cash_usd=100.0),
            RiskLimits(),
        )
        position = _position(notional=10.0, fee=0.085)
        account.positions[position.position_id] = position
        account.register_position(position)
        account.cancel_position(position, "no usable mark", T0 + 5)
        return account, position

    def test_the_fee_is_not_refunded(self):
        account, position = self._cancelled()
        assert account.cash == pytest.approx(100.0 - position.entry_fee_usd)

    def test_the_booked_fee_matches_the_cash_that_left(self):
        account, position = self._cancelled()
        assert account.state.fees_paid_usd == pytest.approx(
            position.entry_fee_usd)
        assert 100.0 - account.cash == pytest.approx(
            account.state.fees_paid_usd)

    def test_the_account_identity_still_closes(self):
        account, _position = self._cancelled()
        assert not account.open_positions()
        assert account.equity() == pytest.approx(
            100.0 + account.realized + account.unrealized())

    def test_the_position_records_what_came_back(self):
        account, position = self._cancelled()
        assert position.status is PositionStatus.CANCELLED
        assert position.cash_in_usd == pytest.approx(position.notional_usd)
        assert position.realized_pnl_usd == pytest.approx(
            -position.entry_fee_usd)


# ---------------------------------------------------------------------------
# 7. two code paths measured the same hold differently
# ---------------------------------------------------------------------------


class TestOneAnswerForHowLongAPositionHasBeenHeld:
    """
    ``PositionBook.assess`` computed ``now - (entry_timestamp or now)`` to
    decide whether to suppress an exit before the minimum hold, while
    ``ExitPolicy.assess`` -- which fires the triggers -- uses ``is not None``
    semantics. A 0.0 epoch is falsy, so the suppression gate read 0s where the
    policy read the real hold, and suppressed exits the policy had correctly
    fired. ``exits.held_seconds`` is now the single answer.
    """

    def test_a_zero_epoch_is_a_real_epoch(self):
        position = _position(notional=1.0, fee=0.0)
        position.entry_timestamp = 0.0
        assert held_seconds(position, 120.0) == pytest.approx(120.0)
        # the expression it replaced
        assert 120.0 - (position.entry_timestamp or 120.0) == 0.0

    def test_an_unusable_epoch_fails_closed_at_zero(self):
        position = _position(notional=1.0, fee=0.0)
        position.entry_timestamp = None
        assert held_seconds(position, 120.0) == 0.0

    def test_a_future_epoch_fails_closed_at_zero(self):
        position = _position(notional=1.0, fee=0.0)
        position.entry_timestamp = 10_000.0
        assert held_seconds(position, 120.0) == 0.0

    def test_the_suppression_gate_uses_the_policys_own_hold(self):
        """
        The gate must suppress nothing for a position that really is old, even
        when its epoch is 0.0.
        """
        from autopaper.positions import PositionBook

        assert "held_seconds" in PositionBook.assess.__code__.co_names


# ---------------------------------------------------------------------------
# 8. the settlement identity only closed at zero
# ---------------------------------------------------------------------------


class TestSettlementChargesEveryCostItReports:
    """
    ``realize`` left ``other_costs_quote`` out of ``net`` while recording it on
    the result, and charged the EXIT network leg to ``net`` while reporting the
    ENTRY leg alone. The identity every consumer asserts --
    ``net == gross - trading_fees - network_costs - other_costs`` -- therefore
    held only because both fields are zero by default, which made them a
    landmine rather than a live error.
    """

    def _realized(self, *, other=Decimal("0"), exit_network_lamports=0):
        from autopaper.paper_live.fills import FillModel, realize

        class _Fill:
            output_amount_raw = 1_200_000_000
            output_decimals = 9
            fee = Decimal("0")

        class _Position:
            entry_cost_quote_asset = Decimal("1")
            trading_fees_quote = Decimal("0.001")
            network_costs_quote = Decimal("0.000005")
            other_costs_quote = other
            bankroll_before = 100.0
            sol_notional = Decimal("1")
            usd_notional = Decimal("150")
            sol_usd_reference = 150.0
            sol_usd_reference_at = T0
            opened_at = T0

        model = FillModel(
            network_cost_lamports=Decimal(exit_network_lamports))
        return realize(
            _Position(), exit_fill=_Fill(), exit_observed_mark=None,
            sol_reference=None, model=model, now=T0 + 30,
        )

    def test_the_identity_closes_with_non_zero_other_costs(self):
        pnl = self._realized(other=Decimal("0.002"))
        assert pnl.net_pnl_quote == (
            pnl.gross_pnl_quote
            - pnl.trading_fees_quote
            - pnl.network_costs_quote
            - pnl.other_costs_quote
        )

    def test_the_identity_closes_with_a_non_zero_exit_network_leg(self):
        pnl = self._realized(exit_network_lamports=5_000)
        assert pnl.net_pnl_quote == (
            pnl.gross_pnl_quote
            - pnl.trading_fees_quote
            - pnl.network_costs_quote
            - pnl.other_costs_quote
        )

    def test_both_network_legs_are_reported_in_sol_not_lamports(self):
        pnl = self._realized(exit_network_lamports=5_000)
        # entry 0.000005 SOL + exit 5,000 lamports == 0.000005 SOL
        assert pnl.network_costs_quote == Decimal("0.00001")


# ---------------------------------------------------------------------------
# 9. the PAPER entry quote was sized in dollars-as-lamports
# ---------------------------------------------------------------------------


class TestPaperEntryQuoteIsSizedInLamports:
    """
    The PAPER entry's executable-quote request computed
    ``units * price * 1e9`` where ``units == position_usd / price`` -- which
    cancels to ``position_usd * 1e9``, i.e. it spent the USD figure as though
    one SOL were one dollar. At a $150 SOL the venue priced an order 150x the
    real one, and the extra price impact came back inside ``effective_price``.
    ``simulate_fill`` then takes ``max(mark, quote)`` on an entry, so the unit
    error landed directly in the paper fill price.
    """

    def test_the_request_is_the_notional_converted_at_the_sol_reference(
        self, config
    ):
        asked: list[int] = []
        trader = _paper_trader_with_quote_capture(
            config, asked, sol_usd=150.0)
        size = compute_position_size(trader.account.equity())
        _drive_one_paper_buy(trader)
        assert asked, "the paper entry quote was never requested"
        expected = int(size.position_usd / 150.0 * 1e9)
        assert asked[0] == pytest.approx(expected, rel=1e-6)
        # and emphatically NOT the dollars-as-lamports figure
        assert asked[0] != pytest.approx(int(size.position_usd * 1e9))

    def test_without_a_sol_reference_no_quote_is_requested(self, config):
        """
        A USD notional cannot be expressed in lamports without a SOL/USD
        reference. The absence is recorded as a reason, which this path already
        does honestly, rather than priced off a wrong notional.
        """
        asked: list[int] = []
        trader = _paper_trader_with_quote_capture(config, asked, sol_usd=None)
        _drive_one_paper_buy(trader)
        assert asked == []


# ---------------------------------------------------------------------------
# 10. an aggregator outage disabled every price trigger, unboundedly
# ---------------------------------------------------------------------------


class TestTheDataStateDescribesThePriceHandedIn:
    """
    The paper monitor resolves its trigger price from the execution-native push
    channel, then passed ``data_state=(LIVE if mark.available else MISSING)``
    -- the state of the EXTERNAL AGGREGATOR observation, a different book.

    So an aggregator outage on a position that had a proven, in-bound,
    same-market price in hand made ``ExitPolicy.assess`` take its
    data-integrity branch and return HOLD with every price trigger disabled:
    the stop, the take-profit and the disaster cap. Worse, that branch had just
    cleared ``native_reference_unavailable_since`` and passes no
    ``data_invalid_since``, so the 120s bound that makes "fail closed"
    terminate never started and the hold was unbounded.

    The two tests below are the mechanism; the third is the call site.
    """

    def _policy(self):
        from autopaper.exits import ExitPolicy

        return ExitPolicy(ExitConfig())

    def _position(self):
        position = _position(notional=1.0, fee=0.0, fill_price=1.0e-4)
        position.excursion_basis_price = 1.0e-4
        position.peak_return_pct = 0.0
        return position

    def test_a_live_state_lets_the_disaster_cap_fire(self):
        decision = self._policy().assess(
            self._position(), now=T0 + 30, mark_price=0.4e-4,
            data_state=DataState.LIVE, momentum_30s_pct=None,
            structure_30s_pct=None, liquidity_change_pct=None,
        )
        assert decision.action == "EXIT"
        assert decision.trigger == "disaster_cap"

    def test_a_missing_state_holds_forever_with_no_invalid_timer(self):
        decision = self._policy().assess(
            self._position(), now=T0 + 30, mark_price=0.4e-4,
            data_state=DataState.MISSING, momentum_30s_pct=None,
            structure_30s_pct=None, liquidity_change_pct=None,
            data_invalid_since=None,
        )
        assert decision.action == "HOLD"
        assert decision.trigger is None

    def test_no_exit_evaluation_reads_its_state_off_the_mark(self):
        """
        Wherever the policy is handed the execution-native reference, the state
        beside it must describe THAT price. The unprovable case is a separate
        branch which keeps the bounded data-invalid policy.

        ``Position.data_state`` in ``_production_position`` is deliberately
        mark-derived -- that field IS the mark -- so this looks only at the
        ``assess`` calls, by the price they pass in.
        """
        import inspect
        import re

        from autopaper.paper_live.engine import PaperLiveEngine

        source = inspect.getsource(PaperLiveEngine)
        offenders = re.findall(
            r"mark_price=(?:pushed_usd|trigger_usd),.{0,200}?mark\.available",
            source, re.S,
        )
        assert offenders == [], (
            "an exit evaluation is taking its data_state from the aggregator "
            "mark while being handed the execution-native reference"
        )


def _paper_trader_with_quote_capture(config, asked: list[int], *, sol_usd):
    from .test_incident_paper_buy import Venue, force_buy, make_trader

    object.__setattr__(config.real, "paper_executable_quote_enabled", True)
    object.__setattr__(config.real, "paper_legacy_mark_fill_rollback", True)
    clock = FakeClock(T0)
    trader = make_trader(config, clock, venue=Venue(clock))
    trader.startup()
    trader.state.start()
    force_buy(trader)
    trader._sol_price_usd = sol_usd
    trader._token_decimals_override = lambda _mint: 6

    class _Provider:
        async def quote_buy(self, *, quote_amount_raw, **_kw):
            asked.append(int(quote_amount_raw))
            return None

        async def quote_sell(self, **_kw):
            return None

    trader._quote_provider_override = _Provider()
    trader._clock_for_test = clock
    return trader


def _drive_one_paper_buy(trader) -> None:
    from autopaper.telegram.parser import parse_alert

    from .test_incident_paper_buy import alert as _alert

    clock = trader._clock_for_test
    snapshot = _alert(clock)
    parsed = parse_alert(
        f"NEW TOKEN\nSymbol: {snapshot.symbol}\nCA: {snapshot.mint}\n"
        f"Price: $0.000015"
    )
    asyncio.new_event_loop().run_until_complete(
        trader.on_alert(snapshot, parsed)
    )
