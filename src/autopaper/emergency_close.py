"""
EMERGENCY KILL-ALL — close every open PAPER position, right now, for any reason.

WHAT THIS IS
------------
An operational safety mechanism, NOT a trading strategy. It answers exactly one
question: *the normal exit path is stuck or wrong — get me flat.*

It therefore does NOT ask whether fast_failure fired, whether structure broke,
whether momentum collapsed, whether the max hold elapsed. Those are strategy
decisions and a bug in them is precisely the reason this exists. It DOES keep
every execution-safety property the normal path has: pinned market identity,
quote freshness, validated route, no invented fill price, no second slippage
haircut on top of a quote that already contains this order's impact.

WHY IT IS NEEDED
----------------
A real PAPER position sat at roughly -81% for about eight hours with the normal
exit path unable to close it. A stuck exit is not a market condition; it is a
defect, and the operator needs a control that does not route through the thing
that is broken.

THE `quote = None` TRAP
-----------------------
The historical failure mode was dereferencing quote fields before checking the
quote existed::

    if not quote.available:      # AttributeError when quote is None

Every quote result here goes through :func:`_read_quote`, which handles
``None`` and unconvertible quotes explicitly. That specific crash cannot
reproduce on this path.

NOTHING IS FAKED
----------------
If no validated executable SELL quote can be obtained, the position STAYS OPEN
and the reason is reported. It is never marked closed, never given a synthetic
price, never quietly dropped.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

#: The exit reason written to the ledger. Deliberately NOT the position's
#: strategy: an emergency close says why the trade ENDED, and the strategy that
#: OPENED it is a separate fact that must survive untouched.
EMERGENCY_EXIT_REASON = "EMERGENCY_KILL_ALL"


class EmergencyState(str, Enum):
    """Explicit, observable. Never implied by a boolean."""
    IDLE = "IDLE"
    EMERGENCY_REQUESTED = "EMERGENCY_REQUESTED"
    ENTRY_PAUSE = "ENTRY_PAUSE"
    EXITING_ALL = "EXITING_ALL"
    EXIT_RETRYING = "EXIT_RETRYING"
    VERIFIED_CLOSED = "VERIFIED_CLOSED"
    EMERGENCY_PARTIAL_FAILURE = "EMERGENCY_PARTIAL_FAILURE"


class PositionOutcome(str, Enum):
    CLOSED = "CLOSED"
    ALREADY_CLOSED = "ALREADY_CLOSED"      # idempotent re-run
    NO_QUOTE = "NO_EXECUTABLE_QUOTE"       # stays OPEN, will retry
    IDENTITY_UNSAFE = "IDENTITY_UNSAFE"     # refuses to sell the wrong market
    ERROR = "ERROR"                         # exception; other positions continue


#: Shown verbatim when a close is blocked. An operator must never have to guess
#: whether "no quote" means closed or not.
BLOCKED_NO_QUOTE = "EMERGENCY CLOSE BLOCKED — NO VALID EXECUTABLE QUOTE"


@dataclass
class EmergencyResult:
    """Outcome for one position. Never carries a fill price it did not earn."""
    position_id: str
    mint: str
    outcome: PositionOutcome
    strategy_profile: str | None
    reason: str
    close_intent_id: str
    fill_price: float | None = None
    exit_reference: float | None = None
    net_pnl_usd: float | None = None
    return_pct: float | None = None
    venue: str | None = None
    route: list[str] = field(default_factory=list)
    quote_provider: str | None = None
    quote_age_ms: float | None = None
    attempts: int = 1

    @property
    def is_closed(self) -> bool:
        return self.outcome in (PositionOutcome.CLOSED, PositionOutcome.ALREADY_CLOSED)


@dataclass
class EmergencyOperation:
    """One kill-all run. ``operation_id`` makes a duplicate request detectable."""
    operation_id: str
    state: EmergencyState = EmergencyState.IDLE
    started_at: float = 0.0
    finished_at: float | None = None
    entries_paused: bool = False
    results: list[EmergencyResult] = field(default_factory=list)
    error: str = ""

    # -- counts ---------------------------------------------------------
    @property
    def closed(self) -> int:
        """Positions this run actually CLOSED. An already-closed position that a
        duplicate press re-encountered is NOT counted again -- otherwise a second
        press would report the same single close twice."""
        return sum(1 for r in self.results if r.outcome is PositionOutcome.CLOSED)

    @property
    def already_closed(self) -> int:
        return sum(1 for r in self.results
                   if r.outcome is PositionOutcome.ALREADY_CLOSED)

    @property
    def settled(self) -> int:
        """Positions with no remaining exposure, however they got there."""
        return self.closed + self.already_closed

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.outcome is PositionOutcome.ERROR)

    @property
    def remaining(self) -> list[EmergencyResult]:
        return [r for r in self.results if not r.is_closed]

    def summary(self) -> dict[str, Any]:
        return dict(
            operation_id=self.operation_id,
            state=self.state.value,
            entries_paused=self.entries_paused,
            targeted=len(self.results),
            closed_count=self.closed,
            already_closed_count=self.already_closed,
            settled_count=self.settled,
            failed_count=self.failed,
            remaining_open_count=len(self.remaining),
            remaining=[dict(position_id=r.position_id, mint=r.mint,
                           outcome=r.outcome.value, reason=r.reason)
                      for r in self.remaining],
        )


@dataclass
class SellQuoteRequest:
    """What the orchestrator asks the provider for. Deliberately minimal."""
    position_id: str
    mint: str
    tokens_raw: int
    token_decimals: int
    venue_id: str | None
    pool_address: str | None


def _read_quote(quote: Any) -> tuple[float | None, str | None, str | None, float | None]:
    """
    Pull an executable SELL price out of whatever the provider returned.

    Returns ``(price_usd, provider, reason, quote_age_ms)``. NEVER raises and
    NEVER indexes into a possibly-None object.

    The historical defect was ``quote.available`` before ``quote is not None``.
    Here ``None`` is the FIRST case, checked before any attribute access.
    """
    if quote is None:
        return None, None, "no quote returned by any validated route", None
    # `available` is checked only once the object is known to exist.
    available = getattr(quote, "available", False)
    if not available:
        return (None, getattr(quote, "provider", None),
                str(getattr(quote, "unavailable_reason", None)
                    or "quote reported unavailable"), None)
    price = getattr(quote, "price_usd", None) or getattr(quote, "effective_price_usd", None)
    try:
        price_f = float(price) if price is not None else None
    except (TypeError, ValueError):
        price_f = None
    if price_f is None or price_f <= 0:
        return (None, getattr(quote, "provider", None),
                "quote carried no usable USD price", None)
    age = getattr(quote, "quote_age_ms", None)
    return (price_f, getattr(quote, "provider", None), None,
            float(age) if isinstance(age, (int, float)) else None)


class EmergencyKillAll:
    """
    Strategy-independent emergency close, as a thin orchestration layer.

    Every execution primitive is INJECTED:
      ``sell_quote``  -> the repository's validated SELL route
      ``close_one``   -> ``account.close_position`` (the canonical settlement)

    so this module owns sequencing, state, idempotency and reporting, and owns
    none of the market or settlement logic itself.
    """

    #: A close already in flight for a position is never duplicated.
    def __init__(
        self,
        *,
        sell_quote: Callable[[SellQuoteRequest], Any],
        close_one: Callable[..., Any],
        list_open: Callable[[], Sequence[dict[str, Any]]],
        pause_entries: Callable[[bool], None] | None = None,
        max_attempts: int = 3,
        backoff_s: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._sell_quote = sell_quote
        self._close_one = close_one
        self._list_open = list_open
        self._pause_entries = pause_entries
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_s = float(backoff_s)
        self._sleep = sleep
        self._now = now
        self._clock = clock
        self._in_flight: dict[str, str] = {}   # position_id -> intent id
        self._done: dict[str, str] = {}       # position_id -> intent id
        self.operation: EmergencyOperation | None = None

    # -- state ---------------------------------------------------------
    @property
    def state(self) -> EmergencyState:
        return self.operation.state if self.operation else EmergencyState.IDLE

    @property
    def entries_paused(self) -> bool:
        return bool(self.operation and self.operation.entries_paused)

    def request(self, *, operation_id: str | None = None) -> EmergencyOperation:
        """Operator pressed the button. Pause entries IMMEDIATELY.

        The pause happens here, before any quote work, so that a slow emergency
        cannot be defeated by a new signal arriving mid-close.
        """
        op = EmergencyOperation(
            operation_id=operation_id or f"emerg_{uuid.uuid4().hex[:12]}",
            started_at=self._now(),
            state=EmergencyState.EMERGENCY_REQUESTED,
        )
        self.operation = op
        self._set_pause(True)
        op.state = EmergencyState.ENTRY_PAUSE
        return op

    def _set_pause(self, paused: bool) -> None:
        if self.operation is not None:
            self.operation.entries_paused = paused
        if self._pause_entries is not None:
            try:
                self._pause_entries(paused)
            except Exception:  # noqa: BLE001
                # A pause failure must not abort the close; the operation state
                # still records the intent.
                pass

    def release_pause(self) -> None:
        """Explicit operator reset once the operator is satisfied."""
        self._set_pause(False)
        if self.operation is not None:
            self.operation.state = EmergencyState.IDLE

    # -- execution -----------------------------------------------------
    def _close_position(self, pos: dict[str, Any]) -> EmergencyResult:
        pid = str(pos.get("position_id") or pos.get("signal_id") or "")
        mint = str(pos.get("mint") or "")
        profile = pos.get("strategy_profile") or pos.get("production_profile")
        req = SellQuoteRequest(
            position_id=pid, mint=mint,
            tokens_raw=int(pos.get("tokens_raw") or 0),
            token_decimals=int(pos.get("token_decimals") or 0),
            venue_id=pos.get("entry_venue_id") or pos.get("venue_id"),
            pool_address=pos.get("entry_pool_address") or pos.get("pool_address"),
        )

        # IDEMPOTENCY. A second press, or a retry inside this run, must not
        # produce a second close intent for the same position.
        if pid in self._done:
            return EmergencyResult(
                position_id=pid, mint=mint, outcome=PositionOutcome.ALREADY_CLOSED,
                strategy_profile=profile, reason="already closed by this control",
                close_intent_id=self._done[pid], attempts=0)
        intent = self._in_flight.get(pid) or f"intent_{uuid.uuid4().hex[:10]}"
        self._in_flight[pid] = intent

        last_reason = "no attempt made"
        last_outcome = PositionOutcome.ERROR
        for attempt in range(1, self.max_attempts + 1):
            # An exception on ONE position must not abort the whole run.
            provider_error = ""
            try:
                quote = self._sell_quote(req)
            except Exception as exc:  # noqa: BLE001
                # The REASON must survive: a provider that raised and a provider
                # that returned nothing are different diagnoses, and losing the
                # exception text is how a real bug gets reported as "no quote".
                provider_error = (f"quote provider raised "
                                  f"{type(exc).__name__}: {exc}")
                quote = None

            price, provider, why, age = _read_quote(quote)

            if price is None:
                # Track the OUTCOME explicitly. Sniffing the reason string for
                # the word "quote" mislabelled a provider that returned
                # `available=False` as a generic ERROR, which is exactly the
                # distinction an operator needs when deciding whether to retry.
                last_outcome = PositionOutcome.NO_QUOTE
                last_reason = provider_error or why or BLOCKED_NO_QUOTE
                if attempt < self.max_attempts and self.backoff_s > 0:
                    self._sleep(self.backoff_s * attempt)
                continue

            # Identity gate: never sell a market we cannot prove is the one the
            # position was opened on. Checked by the caller-supplied closer via
            # the pinned venue/pool carried on the request.
            try:
                trade = self._close_one(
                    pos,
                    exit_reference=price,
                    exit_fill_price=price,
                    reason=EMERGENCY_EXIT_REASON,
                    quote_provider=provider,
                    quote_age_ms=age,
                    now=self._clock(),
                )
            except Exception as exc:  # noqa: BLE001
                last_outcome = PositionOutcome.ERROR
                last_reason = f"close raised {type(exc).__name__}: {exc}"
                if attempt < self.max_attempts and self.backoff_s > 0:
                    self._sleep(self.backoff_s * attempt)
                continue

            if trade is None:
                # The quote was fine; the SETTLEMENT layer refused. That is an
                # identity/accounting refusal, not a quote outage.
                last_outcome = PositionOutcome.IDENTITY_UNSAFE
                last_reason = ("closer refused the close (market identity or "
                               "accounting validation declined it)")
                if attempt < self.max_attempts and self.backoff_s > 0:
                    self._sleep(self.backoff_s * attempt)
                continue

            self._done[pid] = intent
            self._in_flight.pop(pid, None)
            return EmergencyResult(
                position_id=pid, mint=mint, outcome=PositionOutcome.CLOSED,
                strategy_profile=profile, reason=EMERGENCY_EXIT_REASON,
                close_intent_id=intent, fill_price=price, exit_reference=price,
                net_pnl_usd=getattr(trade, "net_pnl_usd", None),
                return_pct=getattr(trade, "return_pct", None),
                venue=pos.get("exit_venue_id") or req.venue_id,
                route=list(pos.get("route") or []),
                quote_provider=provider, quote_age_ms=age, attempts=attempt)

        self._in_flight.pop(pid, None)
        # Position is STILL OPEN. That is reported, never hidden.
        blocked = last_outcome is PositionOutcome.NO_QUOTE
        return EmergencyResult(
            position_id=pid, mint=mint, outcome=last_outcome,
            strategy_profile=profile,
            reason=(f"{BLOCKED_NO_QUOTE} ({last_reason})" if blocked
                    else last_reason),
            close_intent_id=intent, attempts=self.max_attempts)

    def run(self) -> EmergencyOperation:
        """Close everything. Never raises. Always reports what happened."""
        # A press after a FINISHED run starts a fresh operation, so the counts
        # on screen describe THIS press. The `_done` registry is deliberately NOT
        # reset: that is what makes a duplicate press skip, and skip, positions.
        if self.operation is None or self.operation.state in (
            EmergencyState.VERIFIED_CLOSED,
            EmergencyState.EMERGENCY_PARTIAL_FAILURE,
        ):
            self.request()
        op = self.operation
        assert op is not None
        op.state = EmergencyState.EXITING_ALL
        try:
            positions = list(self._list_open() or [])
        except Exception as exc:  # noqa: BLE001
            op.error = f"could not list open positions: {type(exc).__name__}: {exc}"
            op.state = EmergencyState.EMERGENCY_PARTIAL_FAILURE
            return op

        if not positions:
            op.state = EmergencyState.VERIFIED_CLOSED
            op.finished_at = self._now()
            return op

        for pos in positions:
            try:
                op.results.append(self._close_position(pos))
            except Exception as exc:  # noqa: BLE001
                # Absolute last resort: one bad position cannot stop the rest.
                op.results.append(EmergencyResult(
                    position_id=str(pos.get("position_id") or ""),
                    mint=str(pos.get("mint") or ""),
                    outcome=PositionOutcome.ERROR,
                    strategy_profile=pos.get("strategy_profile"),
                    reason=f"unexpected {type(exc).__name__}: {exc}",
                    close_intent_id=f"intent_{uuid.uuid4().hex[:10]}"))

        op.finished_at = self._now()
        op.state = (EmergencyState.EMERGENCY_PARTIAL_FAILURE if op.remaining
                    else EmergencyState.VERIFIED_CLOSED)
        # `remaining` excludes ALREADY_CLOSED, so a repeat press over an
        # already-flat book still terminates as VERIFIED_CLOSED.
        return op

    # -- presentation --------------------------------------------------
    def status_block(self) -> str:
        """Exactly what the dashboard shows while/after an emergency."""
        op = self.operation
        if op is None or op.state is EmergencyState.IDLE:
            return ""
        if op.state is EmergencyState.VERIFIED_CLOSED:
            return ("\U0001F6A8 EMERGENCY EXIT COMPLETE\n\n"
                    f"Closed:\n{op.closed}\n"
                    f"Remaining:\n{len(op.remaining)}\n")
        if op.state is EmergencyState.EMERGENCY_PARTIAL_FAILURE:
            reason = op.remaining[0].reason if op.remaining else op.error
            return ("\U0001F6A8 EMERGENCY EXIT PARTIAL\n\n"
                    f"Closed:\n{op.closed}\n"
                    f"Remaining:\n{len(op.remaining)}\n\n"
                    f"Reason:\n{reason}\n")
        closing = sum(1 for r in op.results if not r.is_closed) or len(self._list_open() or [])
        return ("\U0001F6A8 EMERGENCY EXIT ACTIVE\n\n"
                f"Positions:\n{len(op.results) or closing}\n\n"
                f"Closing:\n{closing}\n"
                f"Closed:\n{op.closed}\n"
                f"Failed:\n{op.failed}\n\n"
                "New entries:\nPAUSED\n")


# ---------------------------------------------------------------------------
# Persistence — so a restart mid-emergency recovers instead of forgetting.
# ---------------------------------------------------------------------------

_STATE_FILE = "emergency_operation.json"


def save_operation(op: EmergencyOperation, var_dir: Path) -> None:
    """Persist the minimum needed to recover: which run, and which positions."""
    payload = dict(
        operation_id=op.operation_id,
        state=op.state.value,
        started_at=op.started_at,
        finished_at=op.finished_at,
        entries_paused=op.entries_paused,
        error=op.error,
        results=[asdict(r) | {"outcome": r.outcome.value} for r in op.results],
    )
    (Path(var_dir) / _STATE_FILE).write_text(json.dumps(payload, indent=2, default=str))


def load_operation(var_dir: Path) -> EmergencyOperation | None:
    """
    Recover a previous emergency run.

    Returns the operation if one was INCOMPLETE. A finished operation returns
    None, so a completed kill-all never re-runs and never double-closes.
    """
    path = Path(var_dir) / _STATE_FILE
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    state = payload.get("state")
    if state in (EmergencyState.VERIFIED_CLOSED.value, EmergencyState.IDLE.value):
        return None
    op = EmergencyOperation(
        operation_id=str(payload.get("operation_id") or "recovered"),
        state=EmergencyState(state) if state in EmergencyState.__members__.values()
        else EmergencyState.EXITING_ALL,
        started_at=float(payload.get("started_at") or 0.0),
        finished_at=payload.get("finished_at"),
        entries_paused=bool(payload.get("entries_paused", True)),
        error=str(payload.get("error") or ""),
    )
    for raw in payload.get("results") or []:
        try:
            op.results.append(EmergencyResult(
                position_id=str(raw.get("position_id") or ""),
                mint=str(raw.get("mint") or ""),
                outcome=PositionOutcome(raw.get("outcome")),
                strategy_profile=raw.get("strategy_profile"),
                reason=str(raw.get("reason") or ""),
                close_intent_id=str(raw.get("close_intent_id") or ""),
                fill_price=raw.get("fill_price"),
                net_pnl_usd=raw.get("net_pnl_usd"),
                return_pct=raw.get("return_pct"),
                venue=raw.get("venue"), route=list(raw.get("route") or []),
                quote_provider=raw.get("quote_provider"),
                quote_age_ms=raw.get("quote_age_ms"),
                attempts=int(raw.get("attempts") or 1)))
        except (ValueError, TypeError):
            continue
    return op