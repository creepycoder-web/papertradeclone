"""
Paper account accounting.

Exact accounting, enforced by construction:

    equity = cash + market_value(open positions)

``market_value`` uses the mark-price convention from
``MarketDataService.mark_price``: the newest usable price. When a position's
data is UNUSABLE it is marked at its entry cost basis and flagged
``DATA_UNCERTAIN`` - never at a price we do not have. That keeps the identity
true without inventing a number, and the flag is surfaced in the dashboard and
in the trade record so the reader knows the mark is a placeholder.

Per trade:

    gross_pnl  = (exit_price - entry_price) * quantity
    net_pnl    = gross_pnl - entry_fee - exit_fee - entry_slippage - exit_slippage

Cash is never allowed to go negative: the sizer only ever reduces a size, and
``apply_exit`` refuses to book a cash-out larger than the position's cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .config import RiskLimits
from .clock import iso
from .ledger import PaperAccountState
from .logging_setup import log
from .models import (DataState, ExcursionState, Position, PositionStatus,
                    Trade, new_id)

def _within_band(position: Position, tol: float = 0.005) -> bool | None:
    """
    Did the booked exit price lie inside the range the runtime actually observed?

    The check the forensic reconciliation had to reconstruct by hand for 26
    historical exits. Doing it at booking time means the next one does not need to:
    ``False`` here is a price from outside the monitored book, which is the
    signature that identified 22 of those 26 cases as a different reference source.

    ``None`` when the position has no recorded high or low, which is a statement
    about the data rather than about the price.
    """
    lo, hi = position.trough_price, position.peak_price
    booked = position.exit_price if position.exit_price is not None \
        else position.mark_price
    if lo is None or hi is None or booked is None or lo <= 0:
        return None
    return bool(lo * (1.0 - tol) <= booked <= hi * (1.0 + tol))



class AccountingError(RuntimeError):
    """
    Raised when a proposed close would not reconcile.

    A distinct type so the engine can record the refusal and keep the position
    open, rather than letting a half-applied close corrupt the ledger.
    """


def pytest_approx(value: float):
    """Tiny stand-in so the accounting assert does not need pytest at runtime."""
    class _Approx:
        def __eq__(self, other: float) -> bool:
            return abs(float(other) - value) <= max(1e-9, abs(value) * 1e-9)

        def __repr__(self) -> str:  # pragma: no cover
            return f"~{value}"

    return _Approx()


@dataclass
class MarkedPosition:
    position: Position
    mark_price: float | None
    mark_state: DataState

    @property
    def market_value(self) -> float:
        if self.mark_price is None or self.mark_price <= 0:
            # No usable mark. Value at cost so the identity holds, and the
            # DATA_UNCERTAIN flag tells the reader this is a placeholder.
            return self.position.cost_basis_usd
        return self.position.quantity * self.mark_price

    @property
    def unrealized(self) -> float:
        if self.mark_price is None or self.mark_price <= 0:
            return 0.0
        return self.position.unrealized_at(self.mark_price)


class PaperAccount:
    def __init__(
        self,
        state: PaperAccountState | None = None,
        limits: RiskLimits | None = None,
    ) -> None:
        self.state = state or PaperAccountState()
        self.limits = limits or RiskLimits()
        self.positions: dict[str, Position] = {}
        self.recently_exited: dict[str, float] = {}
        self.unfilled_attempts: int = 0

    # -- views -------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self.state.cash_usd

    @property
    def realized(self) -> float:
        return self.state.realized_pnl_usd

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.status is PositionStatus.OPEN]

    def open_exposure(self) -> float:
        return sum(p.cash_out_usd for p in self.open_positions())

    def market_value(self, marks: dict[str, MarkedPosition] | None = None) -> float:
        marks = marks or {}
        total = 0.0
        for p in self.open_positions():
            mp = marks.get(p.position_id)
            total += mp.market_value if mp else p.cost_basis_usd
        return total

    def unrealized(self, marks: dict[str, MarkedPosition] | None = None) -> float:
        marks = marks or {}
        total = 0.0
        for p in self.open_positions():
            mp = marks.get(p.position_id)
            total += mp.unrealized if mp else 0.0
        return total

    def equity(self, marks: dict[str, MarkedPosition] | None = None) -> float:
        return self.cash + self.market_value(marks)

    def has_mint(self, mint: str) -> bool:
        return any(p.mint == mint for p in self.open_positions())

    def cooldown_active(self, mint: str, now: float) -> bool:
        last = self.recently_exited.get(mint)
        if last is None:
            return False
        return (now - last) < self.limits.reentry_cooldown_s

    # -- mutations ---------------------------------------------------------

    def register_position(self, position: Position) -> None:
        self.positions[position.position_id] = position
        position.cash_at_entry = self.state.cash_usd
        self.state.cash_usd -= position.cash_out_usd
        # Costs are booked when they are incurred, not when the trade closes.
        # Otherwise an open position's fee and slippage are invisible on the
        # dashboard, and the running total under-reports what the account has
        # actually paid while a position is live.
        self.state.fees_paid_usd += position.entry_fee_usd
        self.state.slippage_paid_usd += position.entry_slippage_usd
        if self.state.cash_usd < -1e-9:
            # Should be unreachable because the sizer respects cash, but the
            # engine must not be able to go negative even if it is misused.
            self.state.cash_usd = 0.0
            position.status = PositionStatus.ERROR
            position.error = "cash would have gone negative; position voided"
            log("ERROR", "position %s voided: cash would go negative", position.position_id)

    def apply_mark(
        self, position: Position, mark_price: float | None, state: DataState, now: float
    ) -> None:
        """
        Fold one observation into the position's running state.

        THE EXCURSIONS ARE EXTREMA, NOT THE CURRENT VALUE.

        ``mfe_pct`` is the return at the highest valid observation and ``mae_pct``
        is the return at the lowest. They used to be the return at the CURRENT mark
        and the return at the PEAK respectively, which let a trade that was 60 %
        underwater and closed near flat book a POSITIVE "maximum adverse excursion" --
        52 of 77 historical fills did exactly that, and 25 of 37 positions carried a
        non-monotonic MAE series. Both extrema are now seeded from the entry fill
        and only ever moved outwards, so with a valid entry and at least one
        subsequent observation ``mfe_pct >= 0 >= mae_pct`` holds by construction.

        NOTHING IS INVENTED WHEN DATA IS MISSING. A mark that is absent, non-positive
        or not LIVE does not move the extrema; it increments the rejected counter and
        downgrades ``excursion_state``. When no valid observation was ever recorded
        the state is ``UNAVAILABLE_HISTORICAL_DATA`` and the extrema mean "at least",
        which is the only honest reading of a value with no observations behind it.

        The mark-based extrema are also explicitly recorded as ``mark_mfe_pct``,
        ``mark_mae_pct``, ``mark_peak_price``, ``mark_trough_price`` for clarity
        and to distinguish from executable-based extrema.
        """
        position.current_price = mark_price
        position.mark_price = mark_price
        position.data_state = state
        position.unrealized_pnl_usd = position.unrealized_at(mark_price)

        # Seed the extrema from the ENTRY BASIS the first time, so a fresh position
        # reports 0 % rather than an unexplained -100 %.
        if position.excursion_basis_price is None and position.fill_price:
            position.excursion_basis_price = position.fill_price
            position.excursion_basis = "ENTRY_FILL"
            position.peak_price = position.fill_price
            position.trough_price = position.fill_price
            position.mark_peak_price = position.fill_price
            position.mark_trough_price = position.fill_price
            position.peak_return_pct = 0.0
            position.trough_return_pct = 0.0
            position.mark_mfe_pct = 0.0
            position.mark_mae_pct = 0.0
            position.excursion_state = ExcursionState.UNAVAILABLE_HISTORICAL_DATA.value

        if not mark_price or mark_price <= 0:
            position.excursion_observations_rejected += 1
            position.last_observed_at = now
            return
        if state is not DataState.LIVE:
            # A STALE or UNUSABLE print is still a real observation of the last
            # tradable price, but it is not a fresh one. It is counted separately so
            # PARTIAL_DATA means what it says.
            position.excursion_observations_rejected += 1

        basis = position.excursion_basis_price
        if basis and basis > 0:
            ret = (mark_price / basis - 1.0) * 100.0
            if position.peak_price is None or mark_price > position.peak_price:
                position.peak_price = mark_price
                position.peak_return_pct = ret
            if position.trough_price is None or mark_price < position.trough_price:
                position.trough_price = mark_price
                position.trough_return_pct = ret
            position.mfe_pct = position.peak_return_pct
            position.mae_pct = position.trough_return_pct
            # Explicit mark-based tracking
            position.mark_mfe_pct = position.peak_return_pct
            position.mark_mae_pct = position.trough_return_pct
            position.mark_peak_price = position.peak_price
            position.mark_trough_price = position.trough_price
            position.current_return_pct = ret
            position.market_value_usd = position.quantity * mark_price
            position.excursion_observations += 1
            # COMPLETE MEANS EVERY OBSERVATION WAS VALID -- not "the most recent
            # one was". A position that printed stale for a minute and then went
            # live would otherwise be stamped COMPLETE by its last sample, which is
            # exactly the coverage claim this state exists to make. A rejected
            # observation anywhere in the hold caps the state at PARTIAL_DATA, and
            # PARTIAL_DATA means the extrema are "at least" these.
            position.excursion_state = (
                ExcursionState.COMPLETE.value
                if state is DataState.LIVE
                and position.excursion_observations_rejected == 0
                else ExcursionState.PARTIAL_DATA.value
            )
        position.last_observed_at = now

    def build_trade(self, position: Position, *, exit_reference: float,
                    exit_fill_price: float, exit_fee: float,
                    exit_slippage: float, now: float, reason: str,
                    style, liquidity_at_exit: float | None = None) -> Trade:
        """
        Book a closed paper trade.

        REALIZED PnL IS FILL vs FILL
        ------------------------------
        ``net_pnl_usd`` is ``(exit_fill_price - entry_fill_price) * qty`` minus the
        two fees. That is not a stylistic preference: cash already moves by the
        EXECUTED prices (``proceeds = qty * exit_fill_price - exit_fee`` and
        ``cash_out_usd`` was ``qty * entry_fill_price + entry_fee``), so the
        fill-based figure is the only one that can equal ``cash_after - cash_before``
        exactly. The historical formulation derived gross from the PRE-slippage
        reference prices and deducted slippage afterwards; that is algebraically the
        same number when the recorded slippage equals the price difference it
        implies, which is why both are carried and the gap between them is recorded
        rather than assumed away.

        Fees are subtracted exactly once, here, on top of a gross that already
        contains the executed prices. Slippage is not deducted at all in this
        function because it is already inside ``entry_fill_price`` and
        ``exit_fill_price``; subtracting it again would double-charge it.

        THE PRICE PROVENANCE TRAVELS WITH THE NUMBERS
        -------------------------------------------------
        ``entry_price_provenance`` and ``exit_price_provenance`` say what kind of
        number each price is. A ``MARK_ONLY`` exit may appear in a PnL, but the
        record then says in the same row that the PnL is mark-to-mark and not a
        result. Nothing in this system may present a mark-derived value as a fill.
        """
        entry_reference = position.entry_reference_price or position.fill_price or 0.0
        entry_fill = position.fill_price or entry_reference
        qty = position.quantity
        total_fees = position.entry_fee_usd + exit_fee

        # FILL vs FILL. This is the basis the cash movement already uses.
        gross = (exit_fill_price - entry_fill) * qty
        net = gross - total_fees

        # The pre-slippage reference basis, carried so the two can be compared.
        gross_reference = (exit_reference - entry_reference) * qty
        gross_reference_net = gross_reference - total_fees \
            - position.entry_slippage_usd - exit_slippage
        disagreement = gross_reference_net - net

        # PRE-FLIGHT CHECK, before a single field is mutated.
        #
        # This used to be asserted after the money had moved, which meant a
        # mismatch raised with cash already rebooked, the position already
        # CLOSED, and the trade never written to the ledger. The account and the
        # audit trail then disagreed, which is the one thing an experiment like
        # this cannot tolerate. A close either applies completely or not at all.
        if position.cash_at_entry is not None:
            expected = net
            actual = (qty * exit_fill_price - exit_fee) - position.cash_out_usd
            if abs(actual - expected) > max(1e-6, abs(expected) * 1e-6):
                raise AccountingError(
                    f"refusing to close {position.position_id}: cash would move "
                    f"{actual:.10f} but net_pnl is {expected:.10f} "
                    f"(entry_fill={entry_fill:.12g} exit_fill={exit_fill_price:.12g} "
                    f"qty={qty:.6g} fees={total_fees:.10f})"
                )

        from .memory import bucket_age, bucket_liq, bucket_mcap

        return Trade(
            trade_id=new_id("trade"),
            position_id=position.position_id,
            candidate_id=position.candidate_id,
            mint=position.mint,
            symbol=position.symbol,
            source="telegram",
            strategy=position.strategy,
            # Carried from the POSITION, not from the runtime's current
            # selection, for the same reason the position holds it: a profile
            # switch after the entry must not relabel this trade. Statistics
            # and reports read it to attribute a trade to a profile.
            strategy_profile=position.strategy_profile,
            signal_source=position.signal_source,
            route=position.route.value,
            regime=position.regime.value,
            entry_signal_timestamp=position.entry_signal_timestamp,
            entry_timestamp=position.entry_timestamp or position.entry_decision_timestamp,
            # ``entry_price`` keeps the QUOTED reference it always carried, and
            # ``fill_price`` is what was actually paid. Both are on the record so a
            # reader can see them differ rather than inferring it.
            entry_price=position.quoted_price or entry_reference,
            fill_price=entry_fill,
            exit_timestamp=now,
            exit_price=exit_reference,
            quantity=qty,
            entry_fee_usd=position.entry_fee_usd,
            exit_fee_usd=exit_fee,
            entry_slippage_usd=position.entry_slippage_usd,
            exit_slippage_usd=exit_slippage,
            gross_pnl_usd=gross,
            net_pnl_usd=net,
            # The return is quoted on the SAME basis as the PnL: the fills, after
            # fees. ``return_on_reference_pct`` is not recorded here because the
            # reference-basis gross is, and the reader can reconstruct it exactly.
            return_pct=(
                (exit_fill_price / entry_fill - 1.0) * 100.0
                if entry_fill else 0.0
            ),
            holding_seconds=max(0.0, now - (position.entry_timestamp or now)),
            mfe_pct=position.mfe_pct,
            mae_pct=position.mae_pct,
            exit_reason=reason,
            exit_style=style.value if hasattr(style, "value") else str(style),
            thesis=position.thesis,
            invalidation=tuple(position.invalidation),
            liquidity_at_entry=position.liquidity_at_entry,
            market_cap_at_entry=position.market_cap_at_entry,
            token_age_s_at_entry=position.token_age_s_at_entry,
            age_bucket=bucket_age(position.token_age_s_at_entry),
            mc_bucket=bucket_mcap(position.market_cap_at_entry),
            liq_bucket=bucket_liq(position.liquidity_at_entry),
            runner_mode=position.runner_mode,
            confidence=position.confidence,
            signal_quality=position.signal_quality,
            booked_at=now,
            # -- provenance, carried per side ------------------------------
            entry_price_provenance=position.entry_price_provenance,
            exit_price_provenance=position.exit_price_provenance,
            pnl_basis="FILL_VS_FILL",
            gross_pnl_reference_basis_usd=gross_reference,
            pnl_basis_disagreement_usd=disagreement,
            excursion_state=position.excursion_state,
            excursion_observations=position.excursion_observations,
            excursion_basis=position.excursion_basis,
            excursion_basis_price=position.excursion_basis_price,
            peak_price=position.peak_price,
            trough_price=position.trough_price,
            last_observed_mark=position.mark_price,
            booked_exit_within_observed_band=_within_band(position),
            # Entry execution-gap observability (copied from position)
            entry_observed_mark_price=position.observed_mark_price,
            entry_observed_mark_timestamp=position.observed_mark_timestamp,
            entry_executable_quote_price=position.executable_quote_price,
            entry_executable_quote_price_usd=position.executable_quote_price_usd,
            entry_executable_quote_kind=position.executable_quote_kind,
            entry_executable_quote_at=position.executable_quote_at,
            entry_executable_quote_provider=position.executable_quote_provider,
            entry_executable_quote_venue=position.executable_quote_venue,
            entry_executable_quote_pool=position.executable_quote_pool,
            entry_executable_quote_missing_reason=position.executable_quote_missing_reason,
            entry_native_reference_price=position.native_reference_price,
            entry_native_reference_price_usd=position.native_reference_price_usd,
            entry_native_reference_timestamp=position.native_reference_timestamp,
            entry_mark_to_executable_gap_pct=position.mark_to_executable_gap_pct,
            entry_native_to_executable_gap_pct=position.native_to_executable_gap_pct,
            entry_mark_age_ms=position.mark_age_ms,
            entry_quote_age_ms=position.quote_age_ms,
            # MFE/MAE labeling distinction
            mark_mfe_pct=position.mark_mfe_pct,
            mark_mae_pct=position.mark_mae_pct,
            executable_mfe_pct=position.executable_mfe_pct,
            executable_mae_pct=position.executable_mae_pct,
        )

    def close_position(
        self,
        position: Position,
        *,
        exit_reference: float,
        exit_fill_price: float,
        exit_fee: float,
        exit_slippage: float,
        now: float,
        reason: str,
        style,
    ) -> Trade:
        trade = self.build_trade(
            position,
            exit_reference=exit_reference,
            exit_fill_price=exit_fill_price,
            exit_fee=exit_fee,
            exit_slippage=exit_slippage,
            now=now,
            reason=reason,
            style=style,
        )
        # Cash moves by the EXECUTED price; the reference is only for reporting.
        #
        # No clamp on proceeds: the tokens are already held, so a sale always
        # realises its full value and can never make cash negative. An earlier
        # version capped proceeds at available cash, which silently under-recorded
        # winners whenever the account was drained - and broke the identity
        # between cash movement and net P&L. The invariant below now proves the
        # behaviour instead of masking it.
        proceeds = position.quantity * exit_fill_price - exit_fee
        self.state.cash_usd += proceeds
        self.state.realized_pnl_usd += trade.net_pnl_usd
        # entry-side costs were already booked in register_position
        self.state.fees_paid_usd += exit_fee
        self.state.slippage_paid_usd += exit_slippage
        self.state.last_exit_at = now

        position.status = PositionStatus.CLOSED
        position.exit_timestamp = now
        position.exit_price = exit_reference
        position.exit_fill_price = exit_fill_price
        position.exit_fee_usd = exit_fee
        position.exit_slippage_usd = exit_slippage
        position.cash_in_usd = proceeds
        position.realized_pnl_usd = trade.net_pnl_usd
        position.return_pct = trade.return_pct
        position.holding_seconds = trade.holding_seconds
        position.exit_reason = reason
        position.exit_style_used = style
        position.unrealized_pnl_usd = 0.0
        position.market_value_usd = 0.0

        # Post-condition, and deliberately POSITION-LOCAL.
        #
        # The whole-account form of this check - cash_now - cash_at_entry vs
        # net_pnl - is only true when one position is open at a time. With two
        # positions overlapping, the other one's cash flows sit in the same
        # delta and the check fires on a perfectly correct book. It did exactly
        # that in live trading, with the discrepancy equal to the second
        # position's cost basis. The invariant that actually holds is about
        # THIS position's own cash in and out.
        proceeds = position.quantity * exit_fill_price - exit_fee
        drift = proceeds - position.cash_out_usd - trade.net_pnl_usd
        if abs(drift) > 1e-6:
            raise AccountingError(
                f"position-local drift on {position.position_id}: proceeds minus "
                f"cost basis is {proceeds - position.cash_out_usd:.10f} but "
                f"net_pnl is {trade.net_pnl_usd:.10f}"
            )

        self.recently_exited[position.mint] = now
        return trade

    def mark_unfilled(self, position: Position, reason: str, now: float) -> None:
        position.status = PositionStatus.UNFILLED
        position.exit_reason = reason
        position.error = reason
        self.unfilled_attempts += 1

    def cancel_position(self, position: Position, reason: str, now: float) -> None:
        """
        Void a position that never produced a tradable outcome.

        Cash is returned, and the fee/slippage already incurred stays booked -
        a cancelled position really did cost the account that much, and quietly
        refunding it would understate what the simulation paid.
        """
        position.status = PositionStatus.CANCELLED
        position.exit_reason = reason
        self.state.cash_usd += position.cash_out_usd
        self.state.cash_usd = round(self.state.cash_usd, 8)

    def update_peak(self, equity: float) -> None:
        if equity > self.state.peak_equity_usd:
            self.state.peak_equity_usd = equity

    def current_drawdown(self, equity: float) -> float:
        peak = self.state.peak_equity_usd or self.state.starting_bankroll_usd
        if peak <= 0:
            return 0.0
        return (equity - peak) / peak * 100.0

    # -- persistence -------------------------------------------------------

    def to_state(self) -> PaperAccountState:
        self.state.updated_at = max(
            self.state.updated_at, self.state.last_heartbeat or 0.0
        )
        return self.state

    def snapshot(self, marks: dict[str, MarkedPosition] | None = None) -> dict:
        equity = self.equity(marks)
        unreal = self.unrealized(marks)
        return {
            "agent": "AUTONOMOUS_PAPER_AGENT_V1",
            "as_of": iso(max(self.state.updated_at, 0.0)) if self.state.updated_at else None,
            "starting_bankroll_usd": round(self.state.starting_bankroll_usd, 4),
            "cash_usd": round(self.cash, 4),
            "equity_usd": round(equity, 4),
            "realized_pnl_usd": round(self.realized, 4),
            "unrealized_pnl_usd": round(unreal, 4),
            "total_pnl_usd": round(self.realized + unreal, 4),
            "open_exposure_usd": round(self.open_exposure(), 4),
            "exposure_pct_equity": round(
                self.open_exposure() / equity * 100.0 if equity else 0.0, 2
            ),
            "fees_paid_usd": round(self.state.fees_paid_usd, 4),
            "slippage_paid_usd": round(self.state.slippage_paid_usd, 4),
            "peak_equity_usd": round(self.state.peak_equity_usd, 4),
            "current_drawdown_pct": round(self.current_drawdown(equity), 3),
            "open_positions": len(self.open_positions()),
            "unfilled_attempts": self.unfilled_attempts,
            "identity_check": round(
                equity - (self.cash + self.market_value(marks)), 8
            ),
        }
