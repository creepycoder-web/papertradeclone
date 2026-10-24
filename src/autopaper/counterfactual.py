"""
Counterfactual tracking of the decisions the agent did NOT take.

WHAT QUESTION THIS ANSWERS
--------------------------
Every gate in the strategy trades false positives for false negatives, and a
ledger of taken trades can only show the first kind. "Which gate blocks
winners?" is unanswerable from the ledger alone, because the trades that would
have answered it were never opened. So each REJECT and WAIT is followed for
``window_s`` and the outcome it WOULD have had is recorded against the reasons
that blocked it.

WHY IT REUSES THE LIVE MODEL RATHER THAN APPROXIMATING IT
---------------------------------------------------------
A counterfactual computed with a cheaper model is not a comparison, it is two
different experiments. So the entry and the exit both run through the same code
the live path uses -- :func:`autopaper.execution.fill.simulate_fill` for both
legs and :class:`autopaper.exits.ExitPolicy` for the exit -- and the money is
booked by a real :class:`autopaper.account.PaperAccount`. The account is a
fresh, private one per counterfactual: that is what makes the isolation structural
rather than a convention, because there is no object through which a counterfactual's
cash movement could reach the live book.

THREE PROPERTIES THAT ARE NOT NEGOTIABLE
----------------------------------------
NO LOOKAHEAD. Every step reads ``ObservationSet.as_of(t)``. The tracker is
driven forwards through the window one step at a time and never sees a print
later than the step it is on, so a spike after the window cannot change the
outcome. A test appends a +400% print after the window and asserts the recorded
row is identical.

NOTHING IS WRITTEN UNTIL IT RESOLVES. An unresolved counterfactual is in memory only.
A half-finished counterfactual on disk would be indistinguishable from a
resolved one that happened to end flat.

NO PATH TO A LIVE DECISION. The tracker holds no market service, so it cannot
spend a request a live decision is waiting on -- it is handed the history the
live loop already fetched. It holds no ``ResearchMemory``, so a counterfactual outcome
cannot tilt route sizing. :meth:`CounterfactualTracker.needs` states both, and tests
assert them.

Counterfactuals are capped at ``max_tracked`` and every drop is counted, so a busy
channel degrades by tracking fewer candidates rather than by growing without
bound.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

from .account import AccountingError, PaperAccount
from .clock import Clock, SystemClock
from .execution.fill import FillInputs, simulate_fill
from .exits import ExitPolicy
from .ledger import PaperAccountState
from .logging_setup import log
from .models import (
    DataState,
    Decision,
    ExitStyle,
    ObservationSet,
    Position,
    PositionStatus,
    Regime,
    Route,
    new_id,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import ExecutionModel, ExitConfig, CounterfactualConfig

#: The append-only stream, relative to ``var/``.
COUNTERFACTUAL_FILE = "counterfactual.jsonl"

#: The decisions worth following. A BUY is already in the ledger; tracking it
#: would double-count the same opportunity.
TRACKED_DECISIONS = frozenset({Decision.REJECT, Decision.WAIT})

#: The notional every counterfactual is sized at, in USD.
#:
#: FIXED, and deliberately not the live sizing policy. A counterfactual asks "did the
#: price go the right way", and the live size is a function of equity at a
#: moment that never happened for this candidate. A constant keeps every
#: counterfactual's return comparable with every other one; the percentage is the
#: answer, and the dollar figure is only there so the fill model has a notional
#: to price impact against.
COUNTERFACTUAL_NOTIONAL_USD = 100.0


@dataclass
class _Counterfactual:
    """One candidate being followed. In memory only until it resolves."""

    candidate_id: str
    mint: str
    symbol: str
    decision: str
    route: str
    regime: str
    reject_reasons: tuple[str, ...]
    decision_price: Optional[float]
    liquidity_usd: Optional[float]
    opened_at: float
    deadline: float
    #: The simulated entry, filled lazily on the first resolve step so the
    #: entry is priced from an observation at or after the decision -- never
    #: from the decision-time print alone.
    position: Optional[Position] = None
    account: Optional[PaperAccount] = None
    entry_failed: str = ""
    last_step: float = 0.0


@dataclass
class CounterfactualTracker:
    """
    Follows REJECT/WAIT candidates and records what they would have done.

    ``path`` is the append-only stream. ``execution`` and ``exits`` are the
    LIVE configs, passed in rather than rebuilt, so a counterfactual is never evaluated
    under different costs or different stops than the real book.
    """

    path: Path
    config: "CounterfactualConfig"
    execution: "ExecutionModel"
    exits: "ExitConfig"
    clock: Clock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.clock = self.clock or SystemClock()
        self._open: dict[str, _Counterfactual] = {}
        self.policy = ExitPolicy(self.exits)
        #: Candidates refused because the tracker was already at its cap.
        self.dropped: int = 0
        #: Counterfactuals written. One row each, exactly once.
        self.resolved: int = 0
        #: Counterfactuals abandoned because their counterfactual could not be booked.
        self.failed: int = 0

    # -- what this object is allowed to touch ------------------------------

    def needs(self) -> tuple[str, ...]:
        """
        The collaborators this tracker holds, as names.

        Exists so the isolation can be asserted rather than reviewed: neither
        ``market`` nor ``memory`` may ever appear here. A counterfactual that could
        spend a market request would take rate-limit priority from a live
        decision, and one that could reach memory would let a counterfactual
        tilt live sizing.
        """
        return ("execution", "exits", "clock", "path")

    # -- tracking ----------------------------------------------------------

    def track(
        self, *, candidate_id: str, mint: str, symbol: str,
        decision: Decision | str, route: Route | str, regime: Regime | str,
        reject_reasons: Iterable[str], price: Optional[float],
        liquidity_usd: Optional[float], now: float,
    ) -> bool:
        """
        Start following a candidate. ``True`` when it was accepted.

        ``False`` for a decision that is not tracked, for a duplicate, when
        capture is off, and when the cap is reached -- the last of which
        increments :attr:`dropped` so a busy channel's lost coverage is a
        number rather than a silence.
        """
        if not self.config.enabled:
            return False
        value = decision if isinstance(decision, Decision) else None
        if value is None:
            try:
                value = Decision(str(decision))
            except ValueError:
                return False
        if value not in TRACKED_DECISIONS:
            return False
        if candidate_id in self._open:
            return False
        if len(self._open) >= int(self.config.max_tracked):
            self.dropped += 1
            return False
        window = float(self.config.window_s)
        self._open[candidate_id] = _Counterfactual(
            candidate_id=candidate_id,
            mint=mint,
            symbol=symbol,
            decision=value.value,
            route=str(getattr(route, "value", route)),
            regime=str(getattr(regime, "value", regime)),
            reject_reasons=tuple(str(r) for r in reject_reasons),
            decision_price=price,
            liquidity_usd=liquidity_usd,
            opened_at=now,
            deadline=now + window,
            last_step=now,
        )
        return True

    def open_tracked(self) -> list[str]:
        return sorted(self._open)

    def tracked_mints(self) -> set[str]:
        """The mints with an open counterfactual, so a caller knows which history to pass."""
        return {sh.mint for sh in self._open.values()}

    # -- resolving ---------------------------------------------------------

    def resolve(self, history: ObservationSet, now: float,
                *, mint: Optional[str] = None) -> int:
        """
        Advance every open counterfactual and write the ones that finished.

        ``history`` is whatever the live loop already holds for the mint. It is
        read exclusively through ``as_of``, so this call cannot see a print
        later than ``now`` however much history it is handed.

        Returns the number of rows written.
        """
        if not self._open:
            return 0
        written = 0
        for key in list(self._open):
            counterfactual = self._open[key]
            if mint is not None and counterfactual.mint != mint:
                continue
            try:
                done = self._advance(counterfactual, history, now)
            except AccountingError as exc:
                log("ERROR", "counterfactual %s refused to book: %s", key, exc)
                self._open.pop(key, None)
                self.failed += 1
                continue
            if done:
                self._open.pop(key, None)
                written += 1
        return written

    def _advance(self, counterfactual: _Counterfactual, history: ObservationSet,
                 now: float) -> bool:
        """
        Step one counterfactual. ``True`` when it resolved and was written.

        Every read is ``as_of``-bounded. The exit policy is driven on the
        observations between the last step and ``now``, in order, so a stop
        that was hit mid-window is recorded at the price that hit it rather
        than at the window's closing print.
        """
        if counterfactual.position is None and not counterfactual.entry_failed:
            self._open_entry(counterfactual, history)
        if counterfactual.entry_failed:
            if now < counterfactual.deadline:
                return False
            self._write_unfilled(counterfactual, now)
            return True

        position, account = counterfactual.position, counterfactual.account
        assert position is not None and account is not None

        steps = [
            o for o in history.as_of(now)
            if o.price_usd and o.price_usd > 0
            and o.observed_at > counterfactual.last_step
        ]
        for step in steps:
            counterfactual.last_step = step.observed_at
            state = step.freshness(
                step.observed_at,
                self.exits.data_invalid_exit_s * 1000.0,
                self.exits.data_invalid_exit_s * 2000.0,
            )
            account.apply_mark(position, step.price_usd, DataState.LIVE,
                               step.observed_at)
            decision = self.policy.assess(
                position,
                now=step.observed_at,
                mark_price=step.price_usd,
                data_state=DataState.LIVE,
                momentum_30s_pct=None,
                structure_30s_pct=None,
                liquidity_change_pct=None,
            )
            if decision.action == "EXIT":
                self._close(counterfactual, step, step.observed_at, decision)
                return True

        if now < counterfactual.deadline:
            return False

        # The window expired with no trigger. Close at the newest print we are
        # allowed to see, labelled as the window and not as a strategy exit.
        last = history.latest(now)
        if last is None or not last.price_usd or last.price_usd <= 0:
            self._write_unfilled(counterfactual, now, reason="no usable exit price")
            return True
        self._close(counterfactual, last, max(last.observed_at, counterfactual.deadline), None)
        return True

    def _open_entry(self, counterfactual: _Counterfactual, history: ObservationSet) -> None:
        """
        Simulate the entry the agent refused, at the decision instant.

        Priced from the observation available AT the simulated fill time --
        decision time plus the configured latency -- exactly as the live path
        does, so the counterfactual pays the same latency cost a real entry would.
        """
        latency_s = self.execution.decision_latency_ms / 1000.0
        fill_at = counterfactual.opened_at + latency_s
        entry_obs = history.latest(fill_at)
        price = counterfactual.decision_price or (
            entry_obs.price_usd if entry_obs is not None else None)
        if entry_obs is None or not price or price <= 0:
            counterfactual.entry_failed = "no usable price at the simulated fill time"
            return
        fill = simulate_fill(
            FillInputs(
                side="entry",
                quantity=COUNTERFACTUAL_NOTIONAL_USD / price,
                notional_hint_usd=COUNTERFACTUAL_NOTIONAL_USD,
                quoted_price=price,
                decision_time=counterfactual.opened_at,
                observation=entry_obs,
                observed_at=fill_at,
                quote=None,
                quote_missing_reason="counterfactual: no executable quote is requested",
            ),
            self.execution,
            data_state=DataState.LIVE,
        )
        if not fill.filled or fill.fill_price is None:
            counterfactual.entry_failed = f"{fill.reason}: {fill.detail}"[:200]
            return
        account = PaperAccount(
            PaperAccountState(
                starting_bankroll_usd=COUNTERFACTUAL_NOTIONAL_USD * 10.0,
                cash_usd=COUNTERFACTUAL_NOTIONAL_USD * 10.0,
            )
        )
        position = Position(
            position_id=new_id("counterfactual"),
            candidate_id=counterfactual.candidate_id,
            mint=counterfactual.mint,
            symbol=counterfactual.symbol,
            status=PositionStatus.OPEN,
            route=Route.FAST_ENTRY,
            regime=Regime.EARLY,
            exit_style=ExitStyle.FAST_FAILURE,
            strategy="COUNTERFACTUAL",
            entry_signal_timestamp=counterfactual.opened_at,
            entry_decision_timestamp=counterfactual.opened_at,
            entry_timestamp=fill_at,
            quoted_price=price,
            entry_reference_price=price,
            fill_price=fill.fill_price,
            quantity=fill.quantity,
            notional_usd=fill.notional_usd,
            entry_fee_usd=fill.fee_usd,
            entry_slippage_usd=fill.slippage_usd,
            cash_out_usd=fill.quantity * fill.fill_price + fill.fee_usd,
            cost_basis_usd=fill.quantity * fill.fill_price + fill.fee_usd,
            liquidity_at_entry=counterfactual.liquidity_usd,
        )
        account.positions[position.position_id] = position
        account.register_position(position)
        account.apply_mark(position, fill.fill_price, DataState.LIVE, fill_at)
        counterfactual.position = position
        counterfactual.account = account
        counterfactual.last_step = fill_at

    def _close(self, counterfactual: _Counterfactual, observation: Any, at: float,
               decision: Any) -> None:
        """
        Settle the counterfactual against ONE observation.

        The observation is passed, not just its price: ``simulate_fill`` prices
        the exit from it and refuses outright when it is absent. Handing it
        ``None`` and a bare float made every counterfactual resolve ``no_quote``, which
        looked like "the token had no market" rather than "the caller forgot
        the leg's observation".
        """
        position, account = counterfactual.position, counterfactual.account
        assert position is not None and account is not None
        price = float(observation.price_usd)
        exit_fill = simulate_fill(
            FillInputs(
                side="exit",
                quantity=position.quantity,
                notional_hint_usd=position.quantity * price,
                quoted_price=price,
                decision_time=at,
                observation=observation,
                observed_at=max(at, observation.observed_at),
                quote=None,
                quote_missing_reason="counterfactual: no executable quote is requested",
            ),
            self.execution,
            data_state=DataState.LIVE,
        )
        exit_price = exit_fill.fill_price if exit_fill.filled else None
        if exit_price is None:
            # No exit fill means no outcome. Recorded as unfilled rather than
            # settled at the mark, which is the same refusal the live path makes.
            self._write_unfilled(counterfactual, at, reason="exit could not be filled")
            return
        trade = account.close_position(
            position,
            exit_reference=price,
            exit_fill_price=exit_price,
            exit_fee=exit_fill.fee_usd,
            exit_slippage=exit_fill.slippage_usd,
            now=at,
            reason=(decision.reason if decision is not None
                    else "counterfactual window expired"),
            style=(decision.style if decision is not None
                   else ExitStyle.TIME_DECAY),
        )
        self._write(counterfactual, {
            "filled": True,
            "resolved_at": at,
            "entry_fill_price": position.fill_price,
            "exit_fill_price": exit_price,
            "exit_trigger": (decision.trigger if decision is not None
                             else "counterfactual_window"),
            "exit_reason": trade.exit_reason,
            "net_pnl_usd": trade.net_pnl_usd,
            "gross_pnl_usd": trade.gross_pnl_usd,
            "return_pct": trade.return_pct,
            "holding_seconds": trade.holding_seconds,
            "mfe_pct": trade.mfe_pct,
            "mae_pct": trade.mae_pct,
            "notional_usd": COUNTERFACTUAL_NOTIONAL_USD,
        })

    def _write_unfilled(self, counterfactual: _Counterfactual, at: float,
                        reason: str = "") -> None:
        self._write(counterfactual, {
            "filled": False,
            "resolved_at": at,
            "unfilled_reason": reason or counterfactual.entry_failed,
            "net_pnl_usd": None,
            "return_pct": None,
            "exit_trigger": None,
        })

    def _write(self, counterfactual: _Counterfactual, outcome: dict[str, Any]) -> None:
        row = {
            "counterfactual": True,
            "candidate_id": counterfactual.candidate_id,
            "mint": counterfactual.mint,
            "symbol": counterfactual.symbol,
            "decision": counterfactual.decision,
            "route": counterfactual.route,
            "regime": counterfactual.regime,
            "reject_reasons": list(counterfactual.reject_reasons),
            "decision_price": counterfactual.decision_price,
            "liquidity_at_decision_usd": counterfactual.liquidity_usd,
            "opened_at": counterfactual.opened_at,
            "window_s": float(self.config.window_s),
            **outcome,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, separators=(",", ":"),
                                    default=str) + "\n")
            self.resolved += 1
        except OSError as exc:
            self.failed += 1
            log("ERROR", "could not write counterfactual outcome: %s", exc)

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "open": len(self._open),
            "resolved": self.resolved,
            "dropped": self.dropped,
            "failed": self.failed,
            "max_tracked": int(self.config.max_tracked),
            "window_s": float(self.config.window_s),
            "path": str(self.path),
        }


def tracker_for(config: Any, var_dir: Path, *,
                clock: Optional[Clock] = None) -> Optional[CounterfactualTracker]:
    """The tracker for a runtime, or ``None`` when counterfactual tracking is off."""
    cf_cfg = getattr(config, "counterfactual", None)
    if cf_cfg is None or not getattr(cf_cfg, "enabled", False):
        return None
    return CounterfactualTracker(
        Path(var_dir) / COUNTERFACTUAL_FILE,
        cf_cfg,
        execution=config.execution,
        exits=config.exits,
        clock=clock or SystemClock(),
    )


__all__ = [
    "COUNTERFACTUAL_FILE",
    "COUNTERFACTUAL_NOTIONAL_USD",
    "TRACKED_DECISIONS",
    "CounterfactualTracker",
    "tracker_for",
]
