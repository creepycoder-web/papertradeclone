"""
Durable execution-intent (idempotency) store for the REAL path.

WHY THIS EXISTS
---------------
A Solana ``sendTransaction`` is irreversible the moment it is broadcast, but the
process that broadcast it can crash, time out, or be restarted before it learns
the signature. Without a record written BEFORE the broadcast, a retry cannot
tell "this never happened" from "this already happened", and a duplicate order
against real funds becomes possible.

This module is that record. It is deliberately a small, append-only JSONL log of
state TRANSITIONS rather than a mutable JSON blob:

    CREATED -> READY -> SIGNING -> BROADCASTED -> CONFIRMED
                                             -> UNKNOWN
                                             -> FAILED

* A transition is written with ``flush`` + ``fsync`` BEFORE the network call it
  describes, so a crash between the write and the call leaves a record that says
  the attempt was in flight.
* The current state of an intent is the LAST transition for its ``execution_id``,
  reconstructed on every read. A truncated final line from a hard kill is
  skipped, never repaired, exactly like the journal.
* ``UNKNOWN`` is deliberately NON-terminal: it means "we may have broadcast".
  :meth:`ExecutionIntentStore.active` therefore still sees it, and the
  pre-trade checks refuse a new entry while any unresolved intent exists. That
  is what stops a retry from broadcasting a second time.

``FAILED`` is only used where it is provable that no broadcast occurred
(simulation failure, signing failure, a venue refusal). A broadcast rejection or
a confirmation timeout is UNKNOWN, because "the RPC said no" and "the network
never saw it" are not the same claim.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

#: Lifecycle states. Order is meaningful for reporting only; transitions are
#: enforced by :meth:`ExecutionIntentStore.transition`.
CREATED = "CREATED"
READY = "READY"
SIGNING = "SIGNING"
BROADCASTED = "BROADCASTED"
CONFIRMED = "CONFIRMED"
FAILED = "FAILED"
UNKNOWN = "UNKNOWN"

#: States that mean the logical order is finished and a new one may be reserved.
TERMINAL_STATES = (CONFIRMED, FAILED)

#: States that mean an attempt may be in flight on chain and MUST be reconciled
#: before any retry. ``UNKNOWN`` and ``BROADCASTED`` both qualify.
UNRESOLVED_STATES = (BROADCASTED, UNKNOWN)

#: Legal forward transitions. Anything not listed here is a programmer error.
_ALLOWED: dict[str, tuple[str, ...]] = {
    CREATED: (READY, FAILED, UNKNOWN),
    READY: (SIGNING, FAILED, UNKNOWN),
    SIGNING: (BROADCASTED, FAILED, UNKNOWN),
    BROADCASTED: (CONFIRMED, FAILED, UNKNOWN),
    UNKNOWN: (CONFIRMED, FAILED, UNKNOWN),
    CONFIRMED: (),
    FAILED: (),
}


class DuplicateExecutionError(RuntimeError):
    """An active intent already exists for this logical key. DO NOT REBROADCAST."""


class IntentStateError(RuntimeError):
    """A transition was requested that the state machine does not allow."""


@dataclass
class ExecutionIntent:
    """One logical order's durable execution record."""

    execution_id: str
    logical_key: str
    side: str
    mint: str
    requested_amount: Optional[int] = None
    state: str = CREATED
    signature: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    detail: str = ""
    attempts: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)

    # -- convenience -------------------------------------------------------

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def unresolved(self) -> bool:
        """True when an attempt may be on chain and a retry is unsafe."""
        return self.state in UNRESOLVED_STATES

    @property
    def broadcasted(self) -> bool:
        return self.state == BROADCASTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "logical_key": self.logical_key,
            "side": self.side,
            "mint": self.mint,
            "requested_amount": self.requested_amount,
            "state": self.state,
            "signature": self.signature,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "detail": self.detail,
            "attempts": self.attempts,
            "terminal": self.terminal,
            "unresolved": self.unresolved,
        }


def _now() -> float:
    return time.time()


class ExecutionIntentStore:
    """
    Append-only, restart-safe execution-intent log.

    One file, one line per transition. The store is safe to construct against a
    directory that does not exist yet; it creates it on first write.
    """

    def __init__(self, path: Path | str, *, clock: Any = None) -> None:
        self.path = Path(path)
        self._clock = clock
        self.last_error: Optional[str] = None

    def _now(self) -> float:
        if self._clock is not None:
            try:
                return float(self._clock.now())
            except Exception:  # noqa: BLE001
                pass
        return _now()

    # -- writing -----------------------------------------------------------

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, separators=(",", ":"))
        # fsync BEFORE the caller proceeds to the irreversible step. A buffered
        # write that is lost to a crash is precisely the hole this closes.
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def reserve(
        self,
        *,
        logical_key: str,
        side: str,
        mint: str,
        requested_amount: Optional[int] = None,
    ) -> ExecutionIntent:
        """
        Reserve a logical order. Raises :class:`DuplicateExecutionError` when an
        unresolved or terminal-but-active intent already exists for the key.

        The reservation is PERSISTED before this returns, so it precedes
        transaction construction, signing and broadcast.
        """
        existing = self.find(logical_key)
        if existing is not None and not existing.terminal:
            raise DuplicateExecutionError(
                f"an execution for {logical_key} is already {existing.state} "
                f"(execution_id={existing.execution_id}, "
                f"signature={existing.signature or 'unknown'}); refusing to "
                f"reserve a second attempt"
            )
        now = self._now()
        intent = ExecutionIntent(
            execution_id=secrets.token_hex(12),
            logical_key=str(logical_key),
            side=str(side),
            mint=str(mint),
            requested_amount=requested_amount,
            state=CREATED,
            created_at=now,
            updated_at=now,
        )
        self._write(intent, CREATED, detail="reserved")
        return intent

    def transition(
        self,
        execution_id: str,
        state: str,
        *,
        signature: Optional[str] = None,
        detail: str = "",
    ) -> ExecutionIntent:
        """Move an intent to ``state`` and persist it before returning."""
        intent = self.get(execution_id)
        if intent is None:
            raise IntentStateError(f"unknown execution_id {execution_id!r}")
        allowed = _ALLOWED.get(intent.state, ())
        if state != intent.state and state not in allowed:
            raise IntentStateError(
                f"illegal transition {intent.state} -> {state} for "
                f"{execution_id}"
            )
        if signature:
            intent.signature = str(signature)
        self._write(intent, state, detail=detail)
        return intent

    def _write(self, intent: ExecutionIntent, state: str, *, detail: str) -> None:
        now = self._now()
        if state == SIGNING:
            intent.attempts += 1
        intent.state = state
        intent.updated_at = now
        if detail:
            intent.detail = detail
        record = {
            "event": "intent_transition",
            "execution_id": intent.execution_id,
            "logical_key": intent.logical_key,
            "side": intent.side,
            "mint": intent.mint,
            "requested_amount": intent.requested_amount,
            "state": state,
            "signature": intent.signature,
            "at": now,
            "detail": detail,
            "attempts": intent.attempts,
        }
        intent.transitions.append(record)
        try:
            self._append(record)
        except Exception as exc:  # noqa: BLE001 - persistence must not be silent
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise

    # -- reading -----------------------------------------------------------

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                # A truncated final line from a hard kill. Skipped, not repaired.
                continue
            if isinstance(row, dict) and row.get("execution_id"):
                out.append(row)
        return out

    def _rebuild(self) -> dict[str, ExecutionIntent]:
        by_id: dict[str, ExecutionIntent] = {}
        for row in self._read_all():
            eid = str(row["execution_id"])
            intent = by_id.get(eid)
            if intent is None:
                intent = ExecutionIntent(
                    execution_id=eid,
                    logical_key=str(row.get("logical_key") or ""),
                    side=str(row.get("side") or ""),
                    mint=str(row.get("mint") or ""),
                    requested_amount=row.get("requested_amount"),
                    created_at=float(row.get("at") or 0.0),
                )
                by_id[eid] = intent
            intent.logical_key = str(row.get("logical_key") or intent.logical_key)
            intent.side = str(row.get("side") or intent.side)
            intent.mint = str(row.get("mint") or intent.mint)
            intent.state = str(row.get("state") or intent.state)
            if row.get("signature"):
                intent.signature = str(row["signature"])
            intent.updated_at = float(row.get("at") or intent.updated_at)
            intent.detail = str(row.get("detail") or intent.detail)
            intent.attempts = int(row.get("attempts") or intent.attempts)
            intent.transitions.append(row)
        return by_id

    def all_intents(self) -> list[ExecutionIntent]:
        return list(self._rebuild().values())

    def get(self, execution_id: str) -> Optional[ExecutionIntent]:
        return self._rebuild().get(execution_id)

    def find(self, logical_key: str) -> Optional[ExecutionIntent]:
        """The most recently created intent for a logical key, if any."""
        matches = [i for i in self._rebuild().values()
                   if i.logical_key == logical_key]
        if not matches:
            return None
        return max(matches, key=lambda i: i.created_at)

    def active(self) -> list[ExecutionIntent]:
        """Intents that are not terminal. Includes UNKNOWN."""
        return [i for i in self._rebuild().values() if not i.terminal]

    def unresolved(self) -> list[ExecutionIntent]:
        """Intents whose outcome may be on chain. Retry is unsafe for these."""
        return [i for i in self._rebuild().values() if i.unresolved]

    def has_unresolved(self, *, mint: Optional[str] = None) -> bool:
        for intent in self.unresolved():
            if mint is None or intent.mint == mint:
                return True
        return False

    def pending_summary(self) -> list[dict[str, Any]]:
        """Operator-facing summary of everything not yet definitively resolved."""
        return [i.as_dict() for i in self.active()]
