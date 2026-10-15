"""
Append-only ledger and durable state.

Four JSONL streams, all append-only, never rewritten:

    candidates.jsonl         one record per alert seen
    decisions.jsonl          one record per decision
    ledger.jsonl             the full event log (seen/decision/entry/update/exit)
    position_updates.jsonl   one record per position tick (MFE/MAE path)

Plus ``paper_account.json``, the crash-recovery snapshot.

Discipline: nothing is ever deleted, no losing trade is removed, and unfilled
attempts are recorded with the same prominence as fills. A rejected candidate is
never quietly dropped.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from .clock import Clock, SystemClock, iso
from .logging_setup import log
from .models import DecisionReport, Trade


class EventType(str, Enum):
    CANDIDATE_SEEN = "candidate_seen"
    DECISION = "decision"
    ENTRY = "entry"
    POSITION_UPDATE = "position_update"
    EXIT = "exit"
    UNFILLED = "unfilled"
    REJECTED = "rejected"
    ACCOUNT = "account"
    RECOVERY = "recovery"
    NOTE = "note"
    #: A settlement of an execution-aware PAPER arm. Deliberately NOT ``EXIT``:
    #: an arm settles between two executable fills on its own bankroll, while a
    #: legacy EXIT settles against a mark on the shared account. They are different
    #: accounting bases and merging them is the error this separate event type
    #: exists to prevent. Both carry their price provenance on the record.
    ARM_EXIT = "arm_exit"


class Ledger:
    def __init__(
        self,
        ledger_path: Path,
        decisions_path: Path,
        candidates_path: Path,
        updates_path: Path,
        *,
        clock: Clock | None = None,
        forward: "ForwardRecorder | None" = None,
    ) -> None:
        """
        ``forward`` mirrors live PAPER activity into the prospective store.

        The ledger is the single live data path, so mirroring here captures
        every candidate, decision and closed trade without instrumenting each
        call site -- a site that is added later cannot be silently missed.
        ``None`` disables mirroring entirely, which is what the existing tests
        and any non-PAPER use rely on.
        """
        self.ledger_path = Path(ledger_path)
        self.decisions_path = Path(decisions_path)
        self.candidates_path = Path(candidates_path)
        self.updates_path = Path(updates_path)
        self.clock = clock or SystemClock()
        self.forward = forward
        for p in (self.ledger_path, self.decisions_path, self.candidates_path,
                  self.updates_path):
            p.parent.mkdir(parents=True, exist_ok=True)

    # -- writing -----------------------------------------------------------

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def candidate_seen(self, record: dict[str, Any]) -> None:
        rec = {
            "event": EventType.CANDIDATE_SEEN.value,
            "at": self.clock.now(),
            **record,
        }
        self._append(self.candidates_path, rec)
        self._append(self.ledger_path, rec)
        self._mirror_candidate(rec)

    def decision(self, report: DecisionReport) -> None:
        rec = {
            "event": EventType.DECISION.value,
            "at": report.decided_at,
            "candidate_id": report.candidate_id,
            "mint": report.mint,
            "symbol": report.symbol,
            **report.to_json_dict(),
        }
        self._append(self.decisions_path, rec)
        self._append(self.ledger_path, rec)
        self._mirror_decision(report, rec)

    def event(self, event: EventType, payload: dict[str, Any]) -> None:
        rec = {"event": event.value, "at": self.clock.now(), **payload}
        self._append(self.ledger_path, rec)

    def position_update(self, payload: dict[str, Any]) -> None:
        rec = {
            "event": EventType.POSITION_UPDATE.value,
            "at": self.clock.now(),
            **payload,
        }
        self._append(self.updates_path, rec)
        self._append(self.ledger_path, rec)

    def trade(self, trade: Trade) -> None:
        rec = {
            "event": EventType.EXIT.value,
            "at": trade.booked_at or trade.exit_timestamp,
            "trade": trade.as_dict(),
        }
        self._append(self.ledger_path, rec)
        self._mirror_trade(rec)

    # -- prospective mirroring -------------------------------------------
    #
    # Best-effort by design. A failure to write a validation record must NEVER
    # stop the trading loop or change a decision, so every mirror is guarded and
    # the error is logged. The forward store is evidence about the strategy, not
    # a precondition for taking a paper trade.

    def _mirror(self, call) -> None:
        """
        Run a forward-store write if one is wired.

        Takes a CALLABLE, not a bound method, on purpose: ``self.forward.trade``
        would be dereferenced at the call site and raise ``AttributeError`` on
        None before the guard below ever ran.
        """
        if self.forward is None:
            return
        try:
            call()
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "forward-validation capture failed (%s): %s",
                type(exc).__name__, exc)

    def _mirror_candidate(self, rec: dict[str, Any]) -> None:
        self._mirror(lambda: self.forward.candidate(
            candidate_id=rec.get("candidate_id") or "",
            mint=rec.get("mint"),
            route=rec.get("route") or "",
            features=rec.get("features") or {},
            at=rec.get("at"),
        ))

    def _mirror_decision(self, report: DecisionReport, rec: dict[str, Any]) -> None:
        self._mirror(lambda: self.forward.decision(report, at=rec.get("at")))

    def _mirror_trade(self, rec: dict[str, Any]) -> None:
        """
        Flatten a closed trade into the fields forward validation scores on.

        The names below are the real ``Trade.as_dict()`` keys, verified against
        the ledger rather than guessed. Fees and slippage are the SUM of the
        entry and exit legs, because a cost reported on one leg only reads as
        cheaper than it was. Anything absent stays None and is reported as a
        missing required field, never defaulted.
        """
        payload = dict(rec.get("trade") or {})

        def total(*keys: str) -> float | None:
            """Sum the present legs; None if none of them are present."""
            vals = [float(payload[k]) for k in keys
                    if isinstance(payload.get(k), (int, float))]
            return sum(vals) if vals else None

        flat = {
            "trade_id": payload.get("trade_id"),
            "at": rec.get("at"),
            "mint": payload.get("mint"),
            "symbol": payload.get("symbol"),
            # performance
            "return_pct": payload.get("return_pct"),
            "net_pnl_usd": payload.get("net_pnl_usd"),
            "gross_pnl_usd": payload.get("gross_pnl_usd"),
            "holding_s": payload.get("holding_seconds"),
            "mfe_pct": payload.get("mfe_pct"),
            "mae_pct": payload.get("mae_pct"),
            # entry / exit legs
            "entry_at": payload.get("entry_timestamp"),
            "entry_timestamp": payload.get("entry_timestamp"),
            "exit_timestamp": payload.get("exit_timestamp"),
            "entry_price": payload.get("entry_price"),
            "exit_price": payload.get("exit_price"),
            "quantity": payload.get("quantity"),
            "size_usd": total("entry_notional_usd", "notional_usd"),
            # costs, summed across legs
            "fees_paid_usd": total("entry_fee_usd", "exit_fee_usd"),
            "slippage_paid_usd": total("entry_slippage_usd", "exit_slippage_usd"),
            "exit_reason": payload.get("exit_reason") or payload.get("exit_style"),
            "exit_style": payload.get("exit_style"),
            "runner_mode": payload.get("runner_mode"),
            # provenance: what strategy, at what threshold, on what data
            "strategy": payload.get("strategy"),
            "route": payload.get("route"),
            "regime": payload.get("regime"),
            "signal_quality": payload.get("signal_quality"),
            "market_cap_at_entry": payload.get("market_cap_at_entry"),
            "liquidity_at_entry": payload.get("liquidity_at_entry"),
            "source": payload.get("source"),
            "candidate_id": payload.get("candidate_id"),
            "position_id": payload.get("position_id"),
        }
        self._mirror(lambda: self.forward.trade("exit", flat))

    def account(self, payload: dict[str, Any]) -> None:
        self.event(EventType.ACCOUNT, payload)

    def arm_exit(self, payload: dict[str, Any]) -> None:
        """
        Record an execution-aware PAPER settlement with its fill-level evidence.

        Appended to the ledger only -- NOT mirrored into the prospective forward
        store, because that store aggregates legacy ``exit`` trades and the two
        bases are not comparable.
        """
        rec = {"event": EventType.ARM_EXIT.value, "at": self.clock.now(), **payload}
        self._append(self.ledger_path, rec)

    def note(self, message: str, **payload: Any) -> None:
        self.event(EventType.NOTE, {"message": message, **payload})

    # -- reading -----------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log("ERROR", "corrupt ledger line in %s - skipped", path.name)
                    continue

    def events(self) -> list[dict[str, Any]]:
        return list(self._read(self.ledger_path))

    def all_decisions(self) -> list[dict[str, Any]]:
        return list(self._read(self.decisions_path))

    def all_candidates(self) -> list[dict[str, Any]]:
        return list(self._read(self.candidates_path))

    def trades(self) -> list[dict[str, Any]]:
        return [
            rec["trade"]
            for rec in self._read(self.ledger_path)
            if rec.get("event") == EventType.EXIT.value and isinstance(rec.get("trade"), dict)
        ]

    def arm_exits(self) -> list[dict[str, Any]]:
        """
        Execution-aware PAPER settlements, exactly as recorded.

        THE READER THAT WAS MISSING. :meth:`trades` filters on ``EXIT`` only,
        so before this existed every settlement an arm booked through
        :meth:`arm_exit` was durably on disk and structurally invisible to the
        CLOSED POSITIONS view -- the receipt said "PAPER CLOSED" while the
        ledger the operator was reading stopped days earlier.

        Same append-only file, same record, no reconstruction and no second
        store: "what closed" now has one durable answer for every PAPER path.
        Deliberately NOT merged into :meth:`trades`, because the two record
        types are different accounting bases (see ``ARM_EXIT``).
        """
        return [
            rec for rec in self._read(self.ledger_path)
            if rec.get("event") == EventType.ARM_EXIT.value
        ]

    def seen_message_keys(self) -> list[list[int]]:
        out: list[list[int]] = []
        for rec in self.all_candidates():
            chat_id = rec.get("chat_id")
            message_id = rec.get("message_id")
            if isinstance(chat_id, int) and isinstance(message_id, int):
                out.append([chat_id, message_id])
        return out

    def count(self, event: EventType) -> int:
        return sum(1 for rec in self.events() if rec.get("event") == event.value)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self.events():
            e = str(rec.get("event", "?"))
            counts[e] = counts.get(e, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# durable account state
# ---------------------------------------------------------------------------


@dataclass
class PaperAccountState:
    """
    The single continuous paper account.

    Persisted atomically. It is NEVER reset because of a loss, a restart, or a
    day boundary. ``starting_bankroll_usd`` is written once and is what the
    return calculation is measured against.
    """

    starting_bankroll_usd: float = 100.0
    cash_usd: float = 100.0
    realized_pnl_usd: float = 0.0
    fees_paid_usd: float = 0.0
    slippage_paid_usd: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    run_count: int = 0
    last_heartbeat: float | None = None
    last_exit_at: float | None = None
    peak_equity_usd: float = 100.0
    realized_at_peak: int = 0
    #: When the current STATISTICS EPOCH began, or None to count every trade
    #: ever recorded.
    #:
    #: The ledger is append-only and is also the frozen research universe, so a
    #: paper reset must NOT truncate it. But an operator who resets to $100
    #: expects the statistics card to describe the account they now hold, not
    #: the 418 trades of every strategy version that ran before it.
    #:
    #: So a reset starts a new epoch and the statistics count only trades at or
    #: after this instant. The history stays in the ledger, intact and auditable;
    #: only the REPORTING WINDOW moves. None preserves the original behaviour for
    #: an account that has never been reset.
    stats_epoch_started_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "starting_bankroll_usd": self.starting_bankroll_usd,
            "cash_usd": self.cash_usd,
            "realized_pnl_usd": self.realized_pnl_usd,
            "fees_paid_usd": self.fees_paid_usd,
            "slippage_paid_usd": self.slippage_paid_usd,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "run_count": self.run_count,
            "last_heartbeat": self.last_heartbeat,
            "last_exit_at": self.last_exit_at,
            "peak_equity_usd": self.peak_equity_usd,
            "realized_at_peak": self.realized_at_peak,
            # Must be serialised or the epoch is silently LOST on every save,
            # and the account quietly starts counting the whole ledger again.
            "stats_epoch_started_at": self.stats_epoch_started_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PaperAccountState":
        if not d:
            return cls()
        return cls(
            starting_bankroll_usd=float(d.get("starting_bankroll_usd", 100.0)),
            cash_usd=float(d.get("cash_usd", 100.0)),
            realized_pnl_usd=float(d.get("realized_pnl_usd", 0.0)),
            fees_paid_usd=float(d.get("fees_paid_usd", 0.0)),
            slippage_paid_usd=float(d.get("slippage_paid_usd", 0.0)),
            created_at=float(d.get("created_at") or 0.0),
            updated_at=float(d.get("updated_at") or 0.0),
            run_count=int(d.get("run_count", 0)),
            last_heartbeat=d.get("last_heartbeat"),
            last_exit_at=d.get("last_exit_at"),
            peak_equity_usd=float(d.get("peak_equity_usd", 100.0)),
            realized_at_peak=int(d.get("realized_at_peak", 0)),
            stats_epoch_started_at=(
                None if d.get("stats_epoch_started_at") is None
                else float(d["stats_epoch_started_at"])
            ),
        )


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log("ERROR", "cannot read %s: %s", path.name, exc)
        return None
