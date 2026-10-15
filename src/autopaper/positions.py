"""
Position lifecycle.

Owns the open-position book: entry, per-observation monitoring (MFE/MAE), exit
bookkeeping and the DATA_UNCERTAIN bookkeeping. The engine owns *when* to call
these; this module owns *what* happens.

The DATA_UNCERTAIN rule lives here: a position whose market has gone unusable is
not marked at a made-up price and is not force-exited on a bad tick. It is
flagged, valued at cost for the equity identity, and given a bounded timer.
"""

from __future__ import annotations

from typing import Mapping

from .account import AccountingError, MarkedPosition, PaperAccount
from .clock import iso
from .config import ExitConfig, RiskLimits
from .execution.fill import fill_books_reconcile
from .exits import ExitDecision, ExitPolicy
from .ledger import EventType, Ledger
from .logging_setup import log
from .models import (
    AlertSnapshot,
    DataState,
    Decision,
    DecisionReport,
    ExcursionState,
    ExitStyle,
    Position,
    PositionStatus,
    Trade,
    new_id,
)


class PositionBook:
    def __init__(
        self,
        account: PaperAccount,
        ledger: Ledger,
        exit_cfg: ExitConfig,
        risk: RiskLimits,
    ) -> None:
        self.account = account
        self.ledger = ledger
        self.exit_cfg = exit_cfg
        self.risk = risk
        self.policy = ExitPolicy(exit_cfg, memory=None)
        self._data_invalid_since: dict[str, float] = {}
        self._last_hold_log: dict[str, float] = {}

    # -- entry -------------------------------------------------------------

    def open_from_decision(
        self,
        report: DecisionReport,
        *,
        alert: AlertSnapshot | None,
        fill,
        now: float,
        sizing_note: str,
        context: Mapping[str, object] | None = None,
        strategy_profile: str = "L25_CURRENT",
    ) -> Position:
        """
        Open (or record the refusal to open) a position.

        ``strategy_profile`` is STAMPED here and never re-read. It is the
        profile the runtime had selected at the moment of the entry, and it is
        the only thing that decides how this position will later exit. Passing
        it in explicitly rather than reading global state inside this function
        keeps that guarantee visible at the call site.
        """
        if not fill.filled or fill.fill_price is None:
            position = Position(
                position_id=new_id("pos"),
                candidate_id=report.candidate_id,
                mint=report.mint or "",
                symbol=(report.symbol or "?")[:16],
                status=PositionStatus.UNFILLED,
                route=report.route,
                regime=report.regime,
                exit_style=report.preferred_exit_style,
                strategy=report.route.value,
                strategy_profile=strategy_profile,
                entry_signal_timestamp=(
                    alert.received_timestamp if alert else report.decided_at
                ),
                entry_decision_timestamp=report.decided_at,
                alert_received_timestamp=alert.received_timestamp if alert else None,
                quoted_price=fill.quoted_price,
                # The three layers, at open. They are stored on the position so
                # an exit months later can be compared against the market the
                # ENTRY was actually judged on.
                observed_mark_price=fill.observed_mark_price,
                observed_mark_timestamp=fill.observed_at,
                executable_quote_price=fill.executable_quote_price,
                executable_quote_price_usd=fill.executable_quote_price_usd,
                executable_quote_kind=fill.executable_quote_kind,
                executable_quote_at=fill.executable_quote_at,
                executable_quote_provider=fill.executable_quote_provider,
                executable_quote_venue=fill.executable_quote_venue,
                executable_quote_pool=fill.executable_quote_pool,
                executable_quote_missing_reason=fill.executable_quote_missing_reason,
                native_reference_price=fill.executable_quote_price,
                native_reference_price_usd=fill.executable_quote_price_usd,
                native_reference_timestamp=fill.executable_quote_at,
                quote_price_source=fill.quote_price_source,
                quote_guard_effective=fill.quote_guard_effective,
                fill_price_kind=fill.price_kind,
                confidence=report.confidence,
                signal_quality=report.signal_quality,
                thesis=report.thesis,
                invalidation=report.invalidation_conditions,
                exit_plan=dict(report.exit_plan),
                error=fill.reason,
                notes=(fill.detail,) if fill.detail else (),
                token_decimals=fill.token_decimals,
                signal_source=alert.signal_source if alert else "MooDeng",
            )
            self.ledger.event(
                EventType.UNFILLED,
                {
                    "candidate_id": report.candidate_id,
                    "mint": report.mint,
                    "symbol": report.symbol,
                    "reason": fill.reason,
                    "detail": fill.detail,
                    "quoted_price": fill.quoted_price,
                    "decision": "BUY",
                    "route": report.route.value,
                },
            )
            log("ERROR", "UNFILLED %s: %s (%s)", (report.symbol or "?")[:10],
                fill.reason, fill.detail)
            return position

        quantity = fill.quantity
        notional = fill.notional_usd
        fee = fill.fee_usd
        cash_out = notional + fee
        slip_usd = fill.slippage_usd
        quoted = fill.quoted_price or fill.fill_price

        # Compute entry execution-gap observability
        mark_price = fill.observed_mark_price
        mark_ts = fill.observed_at
        exec_price = fill.executable_quote_price_usd
        exec_ts = fill.executable_quote_at
        native_price = fill.executable_quote_price  # This is the native (SOL/token) price from the quote
        native_ts = fill.executable_quote_at

        # Calculate gaps
        mark_to_exec_gap = None
        native_to_exec_gap = None
        mark_age = None
        quote_age = None

        if mark_price is not None and mark_price > 0 and exec_price is not None and exec_price > 0:
            mark_to_exec_gap = ((mark_price - exec_price) / exec_price) * 100.0
        if native_price is not None and native_price > 0 and exec_price is not None and exec_price > 0:
            native_to_exec_gap = ((native_price - exec_price) / exec_price) * 100.0
        if mark_ts is not None and mark_ts > 0:
            mark_age = max(0.0, (now - mark_ts) * 1000.0)
        if exec_ts is not None and exec_ts > 0:
            quote_age = max(0.0, (now - exec_ts) * 1000.0)

        position = Position(
            position_id=new_id("pos"),
            candidate_id=report.candidate_id,
            mint=report.mint or "",
            symbol=(report.symbol or "?")[:16],
            status=PositionStatus.OPEN,
            route=report.route,
            regime=report.regime,
            exit_style=report.preferred_exit_style,
            strategy=report.route.value,
            strategy_profile=strategy_profile,
            entry_signal_timestamp=(
                alert.received_timestamp if alert else report.decided_at
            ),
            entry_decision_timestamp=report.decided_at,
            entry_timestamp=fill.fill_at or now,
            alert_received_timestamp=alert.received_timestamp if alert else None,
            quoted_price=quoted,
            # The pre-slippage market reference. P&L is measured from the
            # references and slippage is then deducted, so the required
            # identity holds and no cost is counted twice. Omitting this would
            # make gross_pnl use the already-slipped fill price while net_pnl
            # still subtracted the slippage a second time.
            entry_reference_price=fill.price_at_quote or fill.fill_price,
            fill_price=fill.fill_price,
            # The excursion basis is the price ACTUALLY PAID, so MFE/MAE are
            # returns on the fill rather than on the quoted mark. Seeded at the fill
            # itself, which is what makes a fresh position report 0 % rather than an
            # unexplained -100 %, and what makes mfe >= 0 >= mae hold.
            excursion_basis_price=fill.fill_price,
            excursion_basis="ENTRY_FILL",
            excursion_state=ExcursionState.UNAVAILABLE_HISTORICAL_DATA.value,
            peak_price=fill.fill_price,
            trough_price=fill.fill_price,
            peak_return_pct=0.0,
            trough_return_pct=0.0,
            entry_price_provenance=fill.fill_price_provenance,
            quantity=quantity,
            notional_usd=notional,
            entry_fee_usd=fee,
            entry_slippage_usd=slip_usd,
            cash_out_usd=cash_out,
            cost_basis_usd=cash_out,
            thesis=report.thesis,
            invalidation=report.invalidation_conditions,
            entry_conditions=report.entry_conditions,
            exit_plan=dict(report.exit_plan),
            confidence=report.confidence,
            signal_quality=report.signal_quality,
            data_state=DataState.LIVE,
            liquidity_at_entry=float(context.get("liquidity_usd") or 0.0) or None
            if context else None,
            market_cap_at_entry=float(context.get("market_cap_usd") or 0.0) or None
            if context else None,
            token_age_s_at_entry=float(context.get("token_age_s") or 0.0) or None
            if context else None,
            alert_lag_ms=(context or {}).get("alert_lag_ms"),
            analysis_latency_ms=report.latency_ms,
            sizing_note=sizing_note,
            # Entry execution-gap observability
            observed_mark_price=mark_price,
            observed_mark_timestamp=mark_ts,
            executable_quote_price=fill.executable_quote_price,
            executable_quote_price_usd=exec_price,
            executable_quote_kind=fill.executable_quote_kind,
            executable_quote_at=exec_ts,
            executable_quote_provider=fill.executable_quote_provider,
            executable_quote_venue=fill.executable_quote_venue,
            executable_quote_pool=fill.executable_quote_pool,
            executable_quote_missing_reason=fill.executable_quote_missing_reason,
            native_reference_price=native_price,
            native_reference_price_usd=exec_price,  # Same as executable quote USD for now
            native_reference_timestamp=native_ts,
            quote_price_source=fill.quote_price_source,
            quote_guard_effective=fill.quote_guard_effective,
            mark_to_executable_gap_pct=mark_to_exec_gap,
            native_to_executable_gap_pct=native_to_exec_gap,
            mark_age_ms=mark_age,
            quote_age_ms=quote_age,
            token_decimals=fill.token_decimals,
            signal_source=alert.signal_source if alert else "MooDeng",
        )
        self.account.register_position(position)
        self.ledger.event(
            EventType.ENTRY,
            {
                "position_id": position.position_id,
                "candidate_id": position.candidate_id,
                "mint": position.mint,
                "symbol": position.symbol,
                "route": position.route.value,
                "regime": position.regime.value,
                "exit_style": position.exit_style.value,
                "strategy_profile": position.strategy_profile,
                "signal_source": position.signal_source,
                "alert_received_at": iso(position.alert_received_timestamp),
                "decision_at": iso(position.entry_decision_timestamp),
                "fill_at": iso(position.entry_timestamp),
                "signal_to_fill_ms": _span_ms(
                    position.entry_signal_timestamp, position.entry_timestamp
                ),
                "decision_to_fill_ms": _span_ms(
                    position.entry_decision_timestamp, position.entry_timestamp
                ),
                "quoted_price": position.quoted_price,
                "fill_price": position.fill_price,
                "quantity": position.quantity,
                "notional_usd": position.notional_usd,
                "entry_fee_usd": position.entry_fee_usd,
                "entry_slippage_usd": position.entry_slippage_usd,
                "cash_out_usd": position.cash_out_usd,
                "sizing": sizing_note,
                "fill_model": {
                    "slippage_bps": fill.slippage_bps,
                    "fee_bps": fill.fee_bps,
                    "latency_ms": fill.latency_ms,
                    "detail": fill.detail,
                },
                "invalidation": list(position.invalidation),
                "thesis": position.thesis,
                # Entry execution-gap observability
                "entry_observed_mark_price": position.observed_mark_price,
                "entry_observed_mark_timestamp": position.observed_mark_timestamp,
                "entry_executable_quote_price": position.executable_quote_price,
                "entry_executable_quote_price_usd": position.executable_quote_price_usd,
                "entry_executable_quote_kind": position.executable_quote_kind,
                "entry_executable_quote_at": position.executable_quote_at,
                "entry_executable_quote_provider": position.executable_quote_provider,
                "entry_executable_quote_venue": position.executable_quote_venue,
                "entry_executable_quote_pool": position.executable_quote_pool,
                "entry_executable_quote_missing_reason": position.executable_quote_missing_reason,
                "entry_native_reference_price": position.native_reference_price,
                "entry_native_reference_price_usd": position.native_reference_price_usd,
                "entry_native_reference_timestamp": position.native_reference_timestamp,
                "entry_mark_to_executable_gap_pct": position.mark_to_executable_gap_pct,
                "entry_native_to_executable_gap_pct": position.native_to_executable_gap_pct,
                "entry_mark_age_ms": position.mark_age_ms,
                "entry_quote_age_ms": position.quote_age_ms,
                "entry_quote_price_source": position.quote_price_source,
                "entry_quote_guard_effective": position.quote_guard_effective,
            },
        )
        lag = (position.entry_timestamp - position.entry_signal_timestamp) * 1000.0
        log(
            "BUY",
            "%s %s qty=%.6g notional=$%.2f quote=$%.3e fill=$%.3e (+%.0fbp) "
            "route=%s style=%s signal->fill=%.0fms mark->exec_gap=%s%%",
            position.position_id,
            position.symbol,
            position.quantity,
            position.notional_usd,
            position.quoted_price or 0.0,
            position.fill_price or 0.0,
            fill.slippage_bps,
            position.route.value,
            position.exit_style.value,
            lag,
            f"{position.mark_to_executable_gap_pct:+.2f}" if position.mark_to_executable_gap_pct is not None else "N/A",
        )
        return position

    # -- monitoring --------------------------------------------------------

    def mark(
        self,
        position: Position,
        *,
        mark_price: float | None,
        data_state: DataState,
        now: float,
    ) -> None:
        self.account.apply_mark(position, mark_price, data_state, now)
        # STALE starts the clock as well: the exit policy suppresses actions on
        # any non-LIVE mark, and this timer is what eventually forces the exit.
        if data_state is not DataState.LIVE:
            self._data_invalid_since.setdefault(position.position_id, now)
        else:
            self._data_invalid_since.pop(position.position_id, None)

    def assess(
        self,
        position: Position,
        *,
        now: float,
        mark_price: float | None,
        data_state: DataState,
        momentum_30s_pct: float | None,
        structure_pct: float | None,
        liquidity_change_pct: float | None,
        min_hold_s: float = 0.0,
    ) -> ExitDecision:
        since = self._data_invalid_since.get(position.position_id)
        decision = self.policy.assess(
            position,
            now=now,
            mark_price=mark_price,
            data_state=data_state,
            momentum_30s_pct=momentum_30s_pct,
            structure_30s_pct=structure_pct,
            liquidity_change_pct=liquidity_change_pct,
            data_invalid_since=since,
        )
        held = now - (position.entry_timestamp or now)
        if decision.action == "EXIT" and decision.trigger in (
            "momentum_collapse", "structure_break", "peak_giveback",
            "wide_giveback", "no_follow_through",
        ) and held < min_hold_s:
            return ExitDecision(
                "HOLD",
                f"exit signal '{decision.trigger}' suppressed before the "
                f"{min_hold_s:.0f}s minimum hold (this is a declared rule, not a "
                "post-hoc exception)",
                decision.style,
                trigger=None,
            )
        if decision.action == "HOLD":
            position.exit_hold_count += 1
            last = self._last_hold_log.get(position.position_id, 0.0)
            if now - last >= 15.0:
                self._last_hold_log[position.position_id] = now
                log("HOLD", "%s %s %s", position.position_id, position.symbol,
                    decision.reason)
        return decision

    def record_hold(self, position: Position, decision: ExitDecision, now: float) -> None:
        self.ledger.position_update(
            {
                "position_id": position.position_id,
                "mint": position.mint,
                "symbol": position.symbol,
                "action": "HOLD",
                "reason": decision.reason,
                "mark_price": position.mark_price,
                "mark_provenance": "MARK_ONLY",
                "return_pct": round(position.current_return_pct, 3),
                "peak_return_pct": round(position.peak_return_pct, 3),
                "trough_return_pct": round(position.trough_return_pct, 3),
                "mfe_pct": round(position.mfe_pct, 3),
                "mae_pct": round(position.mae_pct, 3),
                "mark_mfe_pct": round(position.mark_mfe_pct, 3),
                "mark_mae_pct": round(position.mark_mae_pct, 3),
                "executable_mfe_pct": round(position.executable_mfe_pct, 3) if position.executable_mfe_pct is not None else None,
                "executable_mae_pct": round(position.executable_mae_pct, 3) if position.executable_mae_pct is not None else None,
                "peak_price": position.peak_price,
                "trough_price": position.trough_price,
                "mark_peak_price": position.mark_peak_price,
                "mark_trough_price": position.mark_trough_price,
                "executable_peak_price": position.executable_peak_price,
                "executable_trough_price": position.executable_trough_price,
                "excursion_basis": position.excursion_basis,
                "excursion_basis_price": position.excursion_basis_price,
                "excursion_state": position.excursion_state,
                "excursion_observations": position.excursion_observations,
                "excursion_observations_rejected":
                    position.excursion_observations_rejected,
                "runner_mode": position.runner_mode,
                "data_state": position.data_state.value,
                "holding_s": round(now - (position.entry_timestamp or now), 1),
            }
        )

    # -- exit --------------------------------------------------------------

    def close(
        self,
        position: Position,
        *,
        decision: ExitDecision,
        fill,
        now: float,
        forced: bool = False,
    ) -> Trade | None:
        # THE FILL MUST RECONCILE ITS OWN BOOKS BEFORE IT IS TRUSTED WITH A CLOSE.
        #
        # Realized PnL is derived from the fills, which is the only basis that can
        # equal the cash movement exactly -- and which means a fill claiming one
        # reference price and a different executed price while charging no slippage
        # is now internally consistent with the cash movement. That lie is caught
        # here instead, on the only numbers that can expose it.
        books_ok, books_reason = (
            fill_books_reconcile(fill, quantity=position.quantity)
            if fill is not None else (False, "no fill supplied"))
        if fill is not None and not books_ok:
            self._data_invalid_since[position.position_id] = now
            log("ERROR", "exit REFUSED, position left open: %s", books_reason)
            self.ledger.position_update({
                "position_id": position.position_id,
                "action": "EXIT_REFUSED",
                "reason": books_reason[:400],
            })
            return None

        if fill is None or not fill.filled or fill.fill_price is None:
            log(
                "ERROR",
                "exit of %s NOT filled (%s); position left OPEN rather than "
                "pretending it closed",
                position.position_id,
                fill.reason if fill else "no quote",
            )
            self._data_invalid_since[position.position_id] = self._data_invalid_since.get(
                position.position_id, now
            )
            return None

        # STAMP THE EXIT LEGS ON THE POSITION BEFORE BOOKING.
        #
        # ``build_trade`` reads ``exit_price_provenance`` and ``_within_band`` reads
        # the exit price OFF THIS POSITION while constructing the row. Stamping them
        # afterwards -- which is what this did -- silently degraded every closed
        # trade to the default ``MARK_ONLY`` and made the band check fall back to the
        # last mark, i.e. the check compared the exit against itself. The trade is
        # the record of what the exit was, so the position has to carry it first.
        position.exit_price_provenance = getattr(
            fill, "reference_price_provenance", "MARK_ONLY")
        position.exit_price = fill.price_at_quote or fill.fill_price
        position.exit_fill_price = fill.fill_price
        try:
            trade = self.account.close_position(
                position,
                exit_reference=fill.price_at_quote or fill.fill_price,
                exit_fill_price=fill.fill_price,
                exit_fee=fill.fee_usd,
                exit_slippage=fill.slippage_usd,
                now=now,
                reason=("forced:" if forced else "")
                + (decision.trigger or decision.reason),
                style=decision.style if decision.style else position.exit_style,
            )
        except AccountingError as exc:
            # Nothing was mutated. Keep the position open, record the refusal,
            # and retry on the next cycle - an unbookable exit is a data problem,
            # not a reason to lose the position or the audit trail.
            self._data_invalid_since[position.position_id] = now
            log("ERROR", "exit REFUSED, position left open: %s", exc)
            self.ledger.position_update(
                {
                    "position_id": position.position_id,
                    "action": "EXIT_REFUSED",
                    "reason": str(exc)[:400],
                }
            )
            return None
        self._data_invalid_since.pop(position.position_id, None)
        self.ledger.trade(trade)
        self.ledger.position_update(
            {
                "position_id": position.position_id,
                "trade_id": trade.trade_id,
                "action": "EXIT",
                "reason": trade.exit_reason,
                "style": trade.exit_style,
                "exit_price": trade.exit_price,
                "exit_fill_price": position.exit_fill_price,
                "exit_price_provenance": trade.exit_price_provenance,
                "entry_price_provenance": trade.entry_price_provenance,
                "pnl_basis": trade.pnl_basis,
                "last_observed_mark": trade.last_observed_mark,
                "observed_low": trade.trough_price,
                "observed_high": trade.peak_price,
                "booked_exit_within_observed_band":
                    trade.booked_exit_within_observed_band,
                "excursion_state": trade.excursion_state,
                "excursion_observations": trade.excursion_observations,
                "excursion_basis": trade.excursion_basis,
                "excursion_basis_price": trade.excursion_basis_price,
                "net_pnl_usd": round(trade.net_pnl_usd, 4),
                "return_pct": round(trade.return_pct, 3),
                "mfe_pct": round(trade.mfe_pct, 3),
                "mae_pct": round(trade.mae_pct, 3),
                "holding_s": round(trade.holding_seconds, 1),
                "runner_mode": position.runner_mode,
                # -- THE THREE PRICE LAYERS, PER SIDE -------------------
                # A PAPER capture must keep the same distinction a REAL one
                # does. Absent layers stay ABSENT with a reason; they are never
                # written as zero, because a zero price is a missing value that
                # once reached a journal wearing a measurement's name.
                "entry_price_kind": getattr(
                    position, "fill_price_kind", "SIMULATED_FILL"),
                "entry_observed_mark": position.observed_mark_price,
                "entry_executable_quote": position.executable_quote_price,
                "entry_executable_quote_usd": position.executable_quote_price_usd,
                "entry_executable_quote_kind": position.executable_quote_kind,
                "entry_executable_quote_at": position.executable_quote_at,
                "entry_executable_quote_provider": position.executable_quote_provider,
                "entry_executable_quote_venue": position.executable_quote_venue,
                "entry_executable_quote_pool": position.executable_quote_pool,
                "entry_executable_quote_missing_reason":
                    position.executable_quote_missing_reason,
                # Read defensively: a legacy row or a minimal fill double may
                # not carry the layers, and an absent layer must stay absent
                # rather than becoming a fabricated zero.
                "exit_price_kind": getattr(fill, "price_kind", "SIMULATED_FILL"),
                "exit_observed_mark": getattr(fill, "observed_mark_price", None),
                "exit_executable_quote": getattr(
                    fill, "executable_quote_price", None),
                "exit_executable_quote_usd": getattr(
                    fill, "executable_quote_price_usd", None),
                "exit_executable_quote_kind": getattr(
                    fill, "executable_quote_kind", None),
                "exit_executable_quote_at": getattr(
                    fill, "executable_quote_at", None),
                "exit_executable_quote_provider": getattr(
                    fill, "executable_quote_provider", None),
                "exit_executable_quote_venue": getattr(
                    fill, "executable_quote_venue", None),
                "exit_executable_quote_pool": getattr(
                    fill, "executable_quote_pool", None),
                "exit_executable_quote_missing_reason": getattr(
                    fill, "executable_quote_missing_reason",
                    "fill carries no executable-quote layer"),
                "exit_quote_price_source": getattr(
                    fill, "quote_price_source", "UNKNOWN"),
                "exit_quote_guard_effective": bool(getattr(
                    fill, "quote_guard_effective", False)),
            }
        )
        log(
            "EXIT",
            "%s %s %s exit=$%.3e net=%+.2f (%+.1f%%) held=%.0fs mfe=%+.1f%% "
            "mae=%+.1f%% %s",
            trade.trade_id,
            position.symbol,
            decision.reason,
            trade.exit_price,
            trade.net_pnl_usd,
            trade.return_pct,
            trade.holding_seconds,
            trade.mfe_pct,
            trade.mae_pct,
            "RUNNER" if position.runner_mode else "",
        )
        log(
            "ACCOUNT",
            "cash=$%.2f realized=%+.2f fees=$%.4f slip=$%.4f open=%d",
            self.account.cash,
            self.account.realized,
            self.account.state.fees_paid_usd,
            self.account.state.slippage_paid_usd,
            len(self.account.open_positions()),
        )
        return trade

    def cancel(self, position: Position, reason: str, now: float) -> None:
        self.account.cancel_position(position, reason, now)
        self.ledger.position_update(
            {
                "position_id": position.position_id,
                "action": "CANCEL",
                "reason": reason,
            }
        )
        log("CONTROL", "position %s cancelled: %s", position.position_id, reason)


def _span_ms(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round((b - a) * 1000.0, 2)
