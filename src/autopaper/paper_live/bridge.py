"""
The execution-aware PAPER engine, hosted by the EXISTING Telegram runtime.

WHAT THIS IS
------------

The repository already had a Telegram-driven PAPER bot: it received a signal,
ran the production decision, sized a position, filled it at an OBSERVED MARK and
settled it against another mark. This module replaces only the EXECUTION of that
PAPER paper -- the fill and the settlement -- with an executable quote, and
leaves every other part of the runtime exactly where it was.

It is a :class:`~autopaper.paper_live.engine.PaperLiveEngine` subclass rather
than a second engine. That is the whole design:

* ONE decision. The Telegram runtime decides. This module never re-decides.
  ``decide`` is overridden to return the report the runtime already produced, so
  L25 is evaluated exactly once per signal and cannot drift between the chat
  message and the journal.
* ONE quote engine. The runtime's PAPER fill no longer goes through
  ``execution.aggregators.QuoteOnlyJupiterProvider``; it goes through
  ``paper_live.QuoteService``, which wraps the verified ``same_market`` stack.
  That is what makes the direct Pump curve, direct PumpSwap and Jupiter a
  routing decision inside one verified implementation instead of three
  separate notions of "a price".
* ONE account model. Each arm owns a :class:`VirtualAccount`, so the three arms
  cannot spend each other's money.
* ONE journal. The records are the existing ``SignalRecord``/``TradeRecord``
  types and land in the existing four ``reports/live`` artefacts.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------

It does not open its own position, print its own dashboard, or replace the
runtime's book. The runtime asks it to execute; the runtime's monitor loop asks
it to tick; everything else -- notifications, start/stop, status, mode -- stays
in the Telegram runtime. Two books would mean two positions per signal and two
sets of numbers, which is exactly the kind of ambiguity this dataset exists to
remove.

SAFETY
------

Construction calls :func:`assert_paper_only`. Nothing here can build, sign or
broadcast a transaction: the quote service is read-only by construction and
every collaborator is injected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from autopaper.models import AlertSnapshot, DecisionReport

from .arms import ARMS, BETA_ARMS, ArmSpec, ResearchArm, parse_arms
from .engine import PaperLiveEngine, build_signal_snapshot
from .guards import assert_paper_only
from .quote_diagnostics import breakdown_of, cost_block
from .report import DatasetWriter
from .schema import (
    FailureReason,
    PaperPosition,
    SignalRecord,
    SignalSnapshot,
    TradeRecord,
)


@dataclass(frozen=True)
class InboundDecision:
    """
    The decision the Telegram runtime already made for one signal.

    ``status`` and ``authorized`` are carried as plain values rather than
    re-read off the gate object, so a later mutation of the gate cannot change
    what this dataset claims the decision was.
    """

    report: DecisionReport
    status: str
    authorized: bool


@dataclass
class TelegramPaperBridge(PaperLiveEngine):
    """
    The execution-aware PAPER engine, driven by the existing Telegram runtime.

    The runtime owns the decision and the notification; this owns the fill. The
    methods the runtime calls are :meth:`handle_alert`, :meth:`tick`,
    :meth:`render_entries`, :meth:`render_exits` and :meth:`persist`.
    """

    #: Where the four ``reports/live`` artefacts are written. ``None`` means the
    #: process working directory, which is what the standalone harness used.
    root: Path | None = None
    #: The strategy profile this process runs. Used only to label the run; the
    #: per-arm profiles come from the arm specs themselves.
    runtime_profile: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        #: ``(channel, message_id, arm)`` triples already executed.
        self._seen: set[tuple[str, int | None, str]] = set()
        #: signal_id -> the decision the runtime made for it.
        self._inbound: dict[str, InboundDecision] = {}
        self._writer = DatasetWriter(self.root, arms=list(self.arms))
        #: Notification payload for the runtime to send, in order.
        self.pending_messages: list[dict[str, Any]] = []
        self.quote_failures = 0
        self.duplicate_signals = 0
        self.started_at = time.time()

    # ------------------------------------------------------------------
    # the decision is the runtime's, not ours
    # ------------------------------------------------------------------

    def decide(
        self,
        *,
        mint: str,
        signal: SignalSnapshot,
        mark: Any,
        decision_time: float,
        blocked_reason: str | None,
    ) -> tuple[Any, str, bool]:
        """
        Return the decision the Telegram runtime already made.

        When there is no inbound decision -- the standalone harness, or a
        fallback -- this defers to the base engine so the decision still comes
        from the production ``DecisionEngine`` either way. There is no branch
        here that can invent a decision.
        """
        inbound = self._inbound.get(signal.signal_id)
        if inbound is None:
            return super().decide(
                mint=mint, signal=signal, mark=mark,
                decision_time=decision_time, blocked_reason=blocked_reason,
            )
        return inbound.report, inbound.status, inbound.authorized

    # ------------------------------------------------------------------
    # entry
    # ------------------------------------------------------------------

    def _already_seen(self, channel: str, message_id: int | None) -> bool:
        """Has this exact Telegram message already been executed?"""
        if message_id is None:
            # No message id means no stable identity, so idempotency would have
            # to be guessed. Refusing to guess is the correct call: a
            # duplicated PAPER trade is worse than a missed one.
            return False
        return any(
            (channel, message_id, arm) in self._seen
            for arm in self.accounts
        )

    def _mark_seen(self, channel: str, message_id: int | None) -> None:
        if message_id is None:
            return
        for arm in self.accounts:
            self._seen.add((channel, message_id, arm))

    async def handle_alert(
        self,
        *,
        snapshot: AlertSnapshot,
        report: DecisionReport,
        features: Any = None,
        channel: str = "",
        arms: Sequence[ArmSpec] | None = None,
    ) -> list[SignalRecord]:
        """
        Execute one already-decided signal on every arm.

        The signal is recorded whether or not any arm traded, so a dataset of
        trades alone cannot be mistaken for a dataset of opportunities.
        """
        chat = channel or getattr(snapshot, "chat", "") or ""
        message_id = getattr(snapshot, "message_id", None)
        if self._already_seen(chat, message_id):
            self.duplicate_signals += 1
            return []

        signal = build_signal_snapshot(
            channel=chat,
            text=getattr(snapshot, "raw_text", "") or "",
            message_id=message_id,
            telegram_timestamp=getattr(snapshot, "source_timestamp", None),
            mint=getattr(snapshot, "mint", None) or report.mint,
            symbol=getattr(snapshot, "symbol", None) or report.symbol,
            provenance=getattr(snapshot, "provenance", None) or {},
            warnings=tuple(getattr(snapshot, "parse_warnings", ()) or ()),
            ambiguous=bool(getattr(snapshot, "ambiguous", False)),
            now=getattr(snapshot, "received_timestamp", None) or None,
        )
        gate = getattr(report, "l25", None)
        self._inbound[signal.signal_id] = InboundDecision(
            report=report,
            status=str(getattr(gate, "status", "UNKNOWN")),
            authorized=bool(getattr(gate, "authorized", False)),
        )
        try:
            if arms is not None:
                self.arms = list(arms)
            records = await self.on_signal(signal)
        finally:
            self._inbound.pop(signal.signal_id, None)
        self._mark_seen(chat, message_id)
        self.quote_failures += sum(
            1 for r in records if self._is_quote_failure(r)
        )
        self.pending_messages.extend(self.render_entries(records))
        # THE BETA NOTIFICATION. ``render_entries`` cannot see a beta position
        # (the experiment deliberately keeps it out of ``self.accounts``), so
        # without this line a successful V2 FLOOR (BETA) entry was silent.
        # ONE grouped message per parent signal, generated only after the
        # experiment actually holds open positions for it.
        self.pending_messages.extend(self.render_beta_entries(records))
        return records

    def _position_for(self, record: SignalRecord) -> PaperPosition | None:
        """
        The position this signal opened, if it opened one.

        ``SignalRecord`` deliberately does not carry the position -- it is the
        signal's own record and stays small -- so the lookup goes through the
        arm's account and matches on ``signal_id`` rather than assuming the
        account's open position is the one this signal made.
        """
        account = self.accounts.get(record.arm)
        position = getattr(account, "position", None) if account else None
        if position is None:
            return None
        if position.signal_id != record.signal.signal_id:
            return None
        return position

    @staticmethod
    def _is_quote_failure(record: SignalRecord) -> bool:
        return record.failure in (
            FailureReason.ENTRY_REJECTED_QUOTE_UNAVAILABLE,
            FailureReason.ENTRY_REJECTED_QUOTE_STALE,
            FailureReason.ENTRY_REJECTED_IDENTITY_UNKNOWN,
            FailureReason.TOKEN_MINT_MISMATCH,
        )

    # ------------------------------------------------------------------
    # exit
    # ------------------------------------------------------------------

    async def tick(self) -> list[TradeRecord]:
        """
        One monitor cycle for every arm's open position.

        The exit policy, its precedence and the max-hold profile are the
        production ones; this only supplies the observation. A position whose
        SELL quote cannot be obtained is left OPEN by the base engine and is
        retried on the next tick, which is the deterministic bounded retry.
        """
        trades = await self.monitor_once()
        if trades:
            self.pending_messages.extend(self.render_exits(trades))
        # The beta experiment settles inside ``monitor_once``, and its settled
        # positions are returned to it rather than added to ``trade_records``
        # (they are ``BetaPosition``s, not engine ``TradeRecord``s, and putting
        # them there would settle the same trade twice). So ``render_exits``
        # never saw them and a closed beta arm produced no notification.
        settled = self._monitor_beta_experiment_for_tick()
        if settled:
            self.pending_messages.extend(self.render_beta_exits(settled))
        # THE OTHER HALF OF THE EXIT. An arm whose SELL quote was refused stays
        # UNRESOLVED and is retried, and the operator must be TOLD that rather
        # than left to infer it from a silence. Read from the experiment's own
        # book rather than from this tick's list, so an arm whose first
        # notification was interrupted is still announced on a later tick; the
        # durable per-attempt marker is what keeps it from repeating.
        self.pending_messages.extend(self.render_beta_exit_blocked())
        return trades

    def beta_exit_block_notified(
        self, parent_signal_id: str, arm: str, attempt: int,
    ) -> bool:
        """
        Persist that an EXIT BLOCKED notification was SENT for one attempt.

        Called by the runtime ONLY after the notifier returned without raising,
        so a send that FAILED stays un-announced and the next tick retries it,
        while one that was delivered is never repeated -- across repeated
        monitor ticks and across process restarts, because the marker lives in
        the experiment's durable state.
        """
        exp = self.beta_experiment
        if exp is None:
            return False
        return bool(exp.mark_exit_block_notified(parent_signal_id, arm,
                                                 attempt))

    # ------------------------------------------------------------------
    # notifications
    # ------------------------------------------------------------------

    def _usd(self, value: float | None) -> str:
        return "-" if value is None else f"${value:,.2f}"

    def _price(self, value: float | None) -> str:
        return "-" if value is None else f"${value:.10f}".rstrip("0").rstrip(".")

    def _sol_usd(self) -> float | None:
        """The SOL reference this process logged, or ``None``. Never invented."""
        reference = getattr(self, "sol_reference", None)
        value = getattr(reference, "usd_per_sol", None)
        return float(value) if value and value > 0 else None

    def render_entries(self, records: Sequence[SignalRecord]) -> list[dict[str, Any]]:
        """
        The entry alerts, one per arm that actually opened a position.

        The simulated fill is the headline number and the observed mark is
        labelled a mark. A reader must never be able to take the mark for what
        was paid.

        Between them sits the COST BLOCK: the same-slot curve price, the raw
        integer legs, the fee, the size impact, the total quote cost and the
        premium of the executable price over the observed mark. Those are five
        different quantities that a single ``Gap: +1.26%`` line used to stand
        for, which is exactly how the mark and the fee came to be read as one
        number. See :mod:`autopaper.paper_live.quote_diagnostics`.
        """
        out: list[dict[str, Any]] = []
        for record in records:
            position = self._position_for(record)
            if not record.traded or position is None:
                continue
            fill = position.entry_fill
            mark = getattr(record.observed_mark, "value_usd", None)
            lines = [
                "\U0001f7e2 <b>PAPER BUY</b>",
                f"${position.symbol or position.mint or '?'}",
                "",
                f"Strategy: <b>{record.arm}</b>",
                f"Bankroll: {self._usd(record.bankroll_before)}",
                f"Position: {self._usd(position.position_size_usd)}"
                + (
                    f" ({float(position.position_size_pct):.1f}%)"
                    if position.position_size_pct is not None else ""
                ),
                "",
                f"Observed mark: {self._price(mark)}",
                f"Executable buy: {self._price(fill.effective_price_usd)}",
            ]
            lines += cost_block(
                breakdown_of(record.buy_quote, observed_mark_usd=mark),
                sol_usd=self._sol_usd(),
            )
            lines += [
                "",
                f"Simulated fill: <b>{self._price(fill.effective_price_usd)}</b>",
                f"Venue: {fill.venue or '-'}",
                f"Pool: {fill.pool_or_curve or '-'}",
                f"Quote age: {float(fill.quote_age_ms or 0.0):.0f}ms",
                "",
                "<i>PAPER ONLY \U0001f512</i>",
                "<i>NO TRANSACTION SENT</i>",
            ]
            out.append({
                "kind": "entry",
                "arm": record.arm,
                "signal_id": record.signal.signal_id,
                "text": "\n".join(lines),
            })
        return out

    def render_exits(self, trades: Sequence[TradeRecord]) -> list[dict[str, Any]]:
        """
        The close alerts, priced off the SELL fill.

        Realized PnL is the settlement between two simulated fills. The exit
        mark is shown for diagnosis only and is never the number the PnL came
        from.
        """
        out: list[dict[str, Any]] = []
        for trade in trades:
            if trade is None or trade.failure is not FailureReason.NONE:
                # A failed SELL produces NO close notification. There is no
                # position end to report, and printing one would be the exact
                # fabricated exit this whole design refuses to make.
                continue
            position = trade.position
            pnl = trade.pnl
            exit_fill = trade.exit_fill
            lines = [
                "\u2705 <b>PAPER CLOSED</b>",
                f"${position.symbol or position.mint or '?'}",
                "",
                f"Strategy: <b>{position.arm}</b>",
                f"Entry fill: {self._price(position.entry_fill.effective_price_usd)}",
                f"Exit fill:  <b>{self._price(exit_fill.effective_price_usd)}</b>",
                "",
                "<b>Realized PAPER PnL:</b>",
                f"{self._pnl_sign(pnl.realized_pnl_usd)}"
                f"{self._usd(abs(pnl.realized_pnl_usd) if pnl.realized_pnl_usd is not None else None)}",
            ]
            if pnl.return_pct is not None:
                lines.append(f"{float(pnl.return_pct):+.2f}%")
            if pnl.realized_pnl_sol is not None:
                lines.append(f"{float(pnl.realized_pnl_sol):+.6f} SOL")
            if trade.exit_observed_mark is not None:
                lines += ["", f"Observed exit mark: {self._price(trade.exit_observed_mark)}"]
                lines += cost_block(
                    breakdown_of(
                        trade.sell_quote,
                        observed_mark_usd=trade.exit_observed_mark,
                    ),
                    sol_usd=self._sol_usd(),
                )
            if trade.exit_reason:
                lines += ["", f"Exit reason: {trade.exit_reason}"]
            if pnl.hold_seconds is not None:
                lines.append(f"Hold time: {float(pnl.hold_seconds):.1f}s")
            lines += ["", "<i>PAPER ONLY \U0001f512</i>",
                      "<i>NO TRANSACTION SENT</i>"]
            out.append({
                "kind": "exit",
                "arm": position.arm,
                "signal_id": position.signal_id,
                "text": "\n".join(lines),
            })
        return out

    @staticmethod
    def _pnl_sign(value: float | None) -> str:
        if value is None:
            return ""
        return "+" if value >= 0 else "-"

    def drain_messages(self) -> list[dict[str, Any]]:
        """Take the notifications queued since the last call."""
        pending, self.pending_messages = self.pending_messages, []
        return pending

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """
        The operator-facing PAPER status.

        Everything here is virtual. There is no wallet balance, because there
        is no wallet.
        """
        base = super().status()
        arms: dict[str, Any] = {}
        beta_names = {a.value for a in BETA_ARMS}
        for name, account in self.accounts.items():
            if name in beta_names:
                # The beta arms' real ledgers are the experiment's own three
                # $10 accounts, reported under ``v2_floor_beta``. These engine
                # accounts are inert and would double-report the experiment.
                continue
            signals = [r for r in self.signal_records if r.arm == name]
            traded = [r for r in signals if r.traded]
            pnl = [t.pnl for t in self.trade_records
                   if t is not None and t.failure is FailureReason.NONE
                   and t.position.arm == name]
            realized = sum(float(p.realized_pnl_usd or 0.0) for p in pnl)
            arms[name] = {
                "bankroll_usd": round(account.bankroll_usd, 6),
                "starting_bankroll_usd": account.starting_bankroll_usd,
                "peak_bankroll_usd": round(account.peak_bankroll_usd, 6),
                "realized_pnl_usd": round(realized, 6),
                "realized_trades": len(pnl),
                "open_position": account.has_position,
                "open_position_mint": (
                    account.position.mint if account.position else None),
                "open_position_tokens_raw": (
                    account.position.tokens_raw if account.position else None),
                "signals": len(signals),
                "entries": len(traded),
                "exits": len(pnl),
                "rejected": sum(1 for r in signals if not r.traded),
                "quote_failures": sum(
                    1 for r in signals if self._is_quote_failure(r)),
                "entry_blocked_position_active":
                    account.blocked_by_position_active,
                "entry_blocked_cooldown": account.blocked_by_cooldown,
            }
        ages = [
            r.buy_quote.quote_age_ms for r in self.signal_records
            if getattr(r.buy_quote, "available", False)
            and r.buy_quote.quote_age_ms is not None
        ]
        providers = sorted({
            r.buy_quote.provider for r in self.signal_records
            if getattr(r.buy_quote, "available", False)
        })
        base.update({
            "mode": "PAPER (execution-aware, simulated fills)",
            "trade_mode": "PAPER",
            "actual_fill_available": False,
            "actual_fill": None,
            "statement": (
                "OBSERVED_MARK != EXECUTABLE_QUOTE != SIMULATED_FILL "
                "!= ACTUAL_FILL"
            ),
            "realized_pnl_basis": (
                "simulated settlement between two SIMULATED_FILLs; never "
                "exit_mark / entry_mark - 1"
            ),
            "quote_provider": ",".join(providers) or "none yet",
            "latest_quote_age_ms": (round(max(ages), 1) if ages else None),
            "signals_seen": len(self.signal_records),
            "signals_traded": sum(1 for r in self.signal_records if r.traded),
            "trades": len(self.trade_records),
            "rejected_opportunities": sum(
                1 for r in self.signal_records if not r.traded),
            "quote_failures": self.quote_failures,
            "duplicate_signals_suppressed": self.duplicate_signals,
            "uptime_s": round(time.time() - self.started_at, 1),
            "runtime_profile": self.runtime_profile or "default",
            "arms": arms,
        })
        return base

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def persist(self) -> dict[str, Path]:
        """
        Write the four ``reports/live`` artefacts.

        The runtime calls this on its own cadence so the dataset survives a
        kill: an interrupted run keeps everything already recorded, and a
        restart re-reads the same files rather than starting an empty one.
        """
        status = self.status()
        status["persisted_at"] = time.time()
        return {
            "json": self._writer.write_json(
                signals=self.signal_records, trades=self.trade_records,
                status=status,
                extra={
                    "runtime": {
                        "profile": self.runtime_profile,
                        "started_at": self.started_at,
                        "idempotency_keys": len(self._seen),
                    },
                },
            ),
            "markdown": self._writer.write_markdown(
                signals=self.signal_records, trades=self.trade_records,
                status=status,
            ),
            "signals_csv": self._writer.write_signals_csv(self.signal_records),
            "trades_csv": self._writer.write_trades_csv(self.trade_records),
        }


def build_telegram_bridge(
    *,
    arms: Sequence[ArmSpec] | None = None,
    root: Path | None = None,
    runtime_profile: str = "",
    **engine_kwargs: Any,
) -> TelegramPaperBridge:
    """
    Construct the bridge and assert PAPER-ONLY before anything else runs.

    ``assert_paper_only`` is called here as well as in the base constructor, so
    a failure stops construction before a single collaborator is touched.
    """
    audit = assert_paper_only()
    specs = list(arms) if arms else parse_arms([])
    bridge = TelegramPaperBridge(
        arms=specs, root=root, runtime_profile=runtime_profile, **engine_kwargs
    )
    bridge.audit = audit
    return bridge


__all__ = [
    "ARMS",
    "InboundDecision",
    "ResearchArm",
    "TelegramPaperBridge",
    "build_telegram_bridge",
    "parse_arms",
]
