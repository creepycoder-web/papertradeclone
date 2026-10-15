"""
The unified trade journal.

ONE journal, TWO strictly separated accounts.

    PAPER  simulated, backed by :class:`autopaper.account.PaperAccount`
    REAL   on-chain, backed by the wallet and actual transaction metadata

Every journal record carries a ``mode`` field. Balances, PnL, fees, positions
and statistics are ALWAYS partitioned by it. There is no code path that sums
across the partition, and the partition is applied at read time as well as at
write time, so a record written before the partition existed is still attributed
correctly rather than being lumped into whichever bucket happens to be first.

WHY NOT TWO JOURNALS
--------------------
Because the operator needs one timeline. A split into two files makes it
possible to end up with two sources of truth; a single file with a mandatory
partition field cannot.

WHAT A REAL RECORD MUST CARRY
-----------------------------
A real entry is not written until the transaction is CONFIRMED, and the record
carries the chain's own numbers:

    signature, commitment, slot
    requested input amount (lamports)
    actual input amount
    actual output amount
    actual execution price
    actual fee (lamports and SOL)
    actual slippage
    confirmation timestamp

A REAL record without a signature is a bug. :meth:`Journal.write_real_entry`
refuses to write one, so an unconfirmed "real fill" cannot reach the journal at
all.

IMMUTABILITY
------------
Historical records are never rewritten. :func:`_partition` is a read-time view;
the underlying append-only files are opened in append mode and nothing here has
a delete or update path.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from .modes import TradeMode
from .safety import redact_mapping

#: The journal format version. Bumped only if the record shape changes; old
#: records keep their original version and are read as ``v0``-shaped.
JOURNAL_VERSION = "v3"

#: Modes a record may belong to. Nothing else is accepted.
VALID_MODES = ("PAPER", "REAL")


class JournalError(RuntimeError):
    """A journal record violated an invariant. Always means DO NOT TRADE."""


def _partition(record: dict[str, Any]) -> str:
    """
    Which account a record belongs to.

    A record with no ``mode`` predates the partition. Those are PAPER by
    construction: the runtime had no real execution path before V3, so
    defaulting a legacy record to PAPER is the only reading that can be
    correct, and it is stated here rather than left implicit.
    """
    mode = str(record.get("mode") or "").upper()
    if mode in VALID_MODES:
        return mode
    return "PAPER"


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


@dataclass
class JournalRecord:
    """A journal row, with its partition made explicit."""

    mode: str
    event: str
    at: float
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "journal_version": JOURNAL_VERSION,
            "mode": self.mode,
            "event": self.event,
            "at": self.at,
            **redact_mapping(self.payload),
        }


class Journal:
    """
    Append-only journal with a mandatory mode partition.

    Constructed with a directory; it writes one JSONL stream per event kind and
    exposes partitioned reads. Nothing is ever mutated or deleted.
    """

    def __init__(self, root: Path | str, *, clock=None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._clock = clock

    def _now(self) -> float:
        if self._clock is not None:
            return float(self._clock.now())
        return time.time()

    # -- paths -------------------------------------------------------------

    @property
    def trades_path(self) -> Path:
        return self.root / "trades.jsonl"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def decisions_path(self) -> Path:
        return self.root / "journal_decisions.jsonl"

    # -- writing -----------------------------------------------------------

    def _write(
        self,
        path: Path,
        event: str,
        mode: str | TradeMode,
        payload: dict[str, Any],
    ) -> JournalRecord:
        m = mode.value if isinstance(mode, TradeMode) else str(mode).upper()
        if m not in VALID_MODES:
            raise JournalError(
                f"refusing to journal an event with mode {m!r}; only "
                f"{VALID_MODES} are valid"
            )
        rec = JournalRecord(
            mode=m, event=event, at=self._now(), payload=dict(payload)
        )
        _append(path, rec.as_dict())
        return rec

    def event(
        self,
        event: str,
        mode: str | TradeMode,
        **payload: Any,
    ) -> JournalRecord:
        """A non-trade event (mode change, preflight, emergency stop, ...)."""
        return self._write(self.events_path, event, mode, payload)

    def decision(self, report, mode: str | TradeMode) -> JournalRecord:
        """A decision, with the V3-C gate result carried verbatim."""
        payload = report.as_dict() if hasattr(report, "as_dict") else dict(report)
        return self._write(self.decisions_path, "decision", mode, payload)

    def paper_entry(self, **payload: Any) -> JournalRecord:
        return self._write(self.trades_path, "entry", TradeMode.PAPER, payload)

    def real_entry(self, *, signature: str = "", **payload: Any) -> JournalRecord:
        """
        A REAL entry.

        Refuses to write without a signature. This is the single place where a
        simulated number could otherwise be labelled REAL, so the signature -
        the one thing a simulation cannot produce - is a write precondition
        rather than a convention.
        """
        if not signature or not str(signature).strip():
            raise JournalError(
                "refusing to journal a REAL entry without a transaction "
                "signature: a real trade must be on-chain confirmed first"
            )
        payload["tx_signature"] = signature
        return self._write(self.trades_path, "entry", TradeMode.REAL, payload)

    def paper_exit(self, **payload: Any) -> JournalRecord:
        return self._write(self.trades_path, "exit", TradeMode.PAPER, payload)

    def real_exit(self, *, signature: str = "", **payload: Any) -> JournalRecord:
        """A REAL exit. Same signature precondition as :meth:`real_entry`."""
        if not signature or not str(signature).strip():
            raise JournalError(
                "refusing to journal a REAL exit without a transaction "
                "signature"
            )
        payload["tx_signature"] = signature
        return self._write(self.trades_path, "exit", TradeMode.REAL, payload)

    def unfilled(self, mode: str | TradeMode, **payload: Any) -> JournalRecord:
        """
        An attempt that did not fill.

        Recorded with the same prominence as a fill. An unfilled real attempt is
        a real fact about the account's exposure and is never dropped.
        """
        return self._write(self.trades_path, "unfilled", mode, payload)

    # -- reading -----------------------------------------------------------

    def _read(self, path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return iter(())

        def _gen() -> Iterator[dict[str, Any]]:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        # A truncated final line from a hard kill. Historical
                        # records are never rewritten, so the partial line is
                        # skipped rather than repaired.
                        continue
                    if isinstance(row, dict):
                        row["mode"] = _partition(row)
                        yield row

        return _gen()

    def all_trades(self) -> list[dict[str, Any]]:
        return list(self._read(self.trades_path))

    def trades_for(self, mode: str | TradeMode) -> list[dict[str, Any]]:
        m = mode.value if isinstance(mode, TradeMode) else str(mode).upper()
        return [r for r in self.all_trades() if r["mode"] == m]

    def all_events(self) -> list[dict[str, Any]]:
        return list(self._read(self.events_path))

    def all_decisions(self) -> list[dict[str, Any]]:
        return list(self._read(self.decisions_path))

    def decisions_for(self, mode: str | TradeMode) -> list[dict[str, Any]]:
        m = mode.value if isinstance(mode, TradeMode) else str(mode).upper()
        return [r for r in self.all_decisions() if r["mode"] == m]

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Per-mode counts. Never a combined total, because a combined PnL across
        a simulated account and a real wallet is a meaningless number.
        """
        trades = self.all_trades()
        out: dict[str, Any] = {}
        for m in VALID_MODES:
            rows = [r for r in trades if r["mode"] == m]
            entries = [r for r in rows if r.get("event") == "entry"]
            exits = [r for r in rows if r.get("event") == "exit"]
            unfilled = [r for r in rows if r.get("event") == "unfilled"]
            signed = [r for r in rows if r.get("tx_signature")]
            out[m] = {
                "entries": len(entries),
                "exits": len(exits),
                "unfilled": len(unfilled),
                "closed_trades": len(exits),
                "with_signature": len(signed),
            }
        return out

    def verify_real_records_have_signatures(self) -> list[str]:
        """
        Audit: every REAL trade record must carry a signature.

        Returns a list of human-readable problems, empty when clean. Called by
        the preflight report.
        """
        problems: list[str] = []
        for row in self.all_trades():
            if row.get("mode") != "REAL":
                continue
            if not row.get("tx_signature"):
                problems.append(
                    f"REAL {row.get('event')} at {row.get('at')} has no "
                    f"tx_signature (trade_id={row.get('trade_id')})"
                )
        return problems
