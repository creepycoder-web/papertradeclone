"""
The AMARIH PAPER runtime driver -- ONE poller, ONE paired-entry book.

WHY THIS MODULE EXISTS
======================
:mod:`autopaper.paper_live.pumpview` knows how to *read* Amarih signals and
:mod:`autopaper.paper_live.amarih_experiment` knows how to *account for* three
paired arms. Neither is allowed to talk to the network, to the quote stack, to
the notifier or to the disk on its own -- deliberately, so both stay testable
offline. Something has to drive them, in the running process, on the existing
PAPER quote stack. This is that driver, and nothing else.

WHAT IT IS NOT
==============
It is not a second execution engine. It owns no quote implementation, no ledger
and no settlement maths: every entry and every exit quote arrives through the
two injected coroutines, and every realized number is produced by
:class:`AmarihExperiment`. It never opens a position it did not first price with
an executable quote, and it never substitutes a mark for a fill.

THE FOUR INVARIANTS THAT MATTER
===============================
1. **ONE entry observation per signal.** A signal is priced ONCE and the single
   observation is fanned out to T60/T120/T300. The three arms are paired
   outcomes, never three independent entries.
2. **Restart-safe, and restart-honest.** Accepted signal ids AND outstanding arms
   are on disk. A restart re-hydrates the arms so their scheduled deadlines still
   fire; a signal seen before the restart is never processed twice. The dedup
   store is capped and the cap is a *bound on the file*, not on safety -- the
   live PumpView poll is already ordered newest-first, so the ids that matter are
   the ones most recently seen.
3. **Fail closed.** No quote, an UNKNOWN identity, a MISMATCH or a stale quote
   produce an explicit recorded outcome. None of them produce a fill.
4. **Notifications follow state, never lead it.** A message is queued only after
   the transition it describes has actually happened, and each event carries a
   stable key so a retried tick cannot send it twice.
5. **The poll does not block the loop.** The adapter's transport is synchronous,
   so the fetch is run on a worker thread. A slow PumpView cannot stall the
   monitor tick that services every other PAPER position.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from .amarih_experiment import (
    AMARIH_ARMS,
    ARM_HORIZON,
    DEFAULT_ARM_NOTIONAL_USD,
    ENTRY_IDENTITY_UNKNOWN,
    ENTRY_QUOTE_UNAVAILABLE,
    ENTRY_VALID,
    EXIT_SETTLED,
    AmarihArm,
    AmarihExperiment,
    AmarihSignalRecord,
    EntryObservation,
)

#: Schema for the driver's own durable file. Bumped when its shape changes in a
#: way a reader must notice; separate from the experiment's telemetry schema.
RUNTIME_SCHEMA_VERSION = 1

#: Default poll cadence. PumpView fires server-side every ~2s and its own
#: front-end polls on load rather than continuously, so a 15s poll is well
#: inside the signal's useful life while spending almost no request budget.
DEFAULT_POLL_INTERVAL_S = 15.0

#: How many accepted signal ids to remember on disk. At tens of thousands of
#: Amarih signals over weeks an unbounded list would grow without limit; 5,000
#: covers far more than any restart window needs.
MAX_PERSISTED_SEEN_IDS = 5000

#: How many notification keys to remember, so a delivered message is not
#: repeated across ticks AND across restarts.
MAX_PERSISTED_NOTIFICATION_KEYS = 2000

#: Identity values that mean "this quote is not provably the requested market".
IDENTITY_UNKNOWN_VALUES = frozenset({"", "unknown", "unverified"})
IDENTITY_MISMATCH_VALUES = frozenset({"fail", "mismatch", "different", "other"})

#: Identity values that mean "this quote is PROVEN to be the requested market".
#:
#: THE CHECK IS AN ALLOWLIST AGAINST THIS SET, NOT A DENYLIST AGAINST the two
#: sets above. Denylisting meant that any token NOT on the deny lists fell
#: through to VALID -- and ``"fail"`` is a token the real quote stack emits
#: (``QuoteService`` sets ``identity="FAIL"`` when the executable market is not
#: the chain-verified market for this mint). A PROVEN MISMATCH was therefore
#: classified as a valid entry, gated only by the separate
#: ``fill_from_quote(require_identity=True)`` refusal further down. One line of
#: defence for a rule that must never have a second way to be reached.
#:
#: An allowlist also fails closed FORWARD: a provider that invents a new
#: identity token is treated as unproven until this vocabulary declares it,
#: which is the correct default for a same-market guarantee.
IDENTITY_PASS_VALUES = frozenset({"pass", "verified"})

#: The reason recorded when the Telegram channel tries to open a position while
#: AMARIH is the selected source. Named here because two modules and the
#: operator surfaces all have to agree on the spelling.
SOURCE_DISABLED_BY_AMARIH = "SOURCE_DISABLED_BY_AMARIH"


@dataclass
@dataclass
class ExitTiming:
    """
    When an exit actually happened, phase by phase, versus when it was DUE.

    Every field is a wall-clock Unix timestamp. ``target_exit_at`` is the only
    one that comes from the schedule; every other field is an OBSERVATION of a
    thing that actually occurred. None of them is ever back-dated to
    ``target_exit_at`` -- a late exit is recorded as late, because the whole
    point of the measurement is to distinguish a scheduler that is slow from a
    provider that is slow from a settlement that is slow.

    ``None`` means the phase never ran. It is never ``0``, which would place the
    phase in 1970 and be indistinguishable from "very late".
    """

    signal_id: str
    arm: str
    #: The scheduled deadline: ``entry_confirmed_at + horizon``. INVARIANT: it
    #: is never rewritten once written.
    target_exit_at: Optional[float] = None
    #: When the ENTRY was confirmed (the timestamp all horizons are measured
    #: from), so the target is auditable without re-deriving the horizon.
    entry_confirmed_at: Optional[float] = None
    #: When the scheduler first SAW this arm as due. This is the phase the
    #: dedicated deadline loop owns.
    scheduler_detected_due_at: Optional[float] = None
    exit_quote_requested_at: Optional[float] = None
    exit_quote_received_at: Optional[float] = None
    settlement_completed_at: Optional[float] = None
    #: How far past the target the scheduler noticed. ``None`` until the arm is
    #: detected as due.
    scheduler_lateness_ms: Optional[float] = None
    #: How far past the target settlement finished. ``None`` unless it settled.
    settlement_lateness_ms: Optional[float] = None
    #: Provider round-trip: ``exit_quote_received_at - exit_quote_requested_at``,
    #: in ms. Kept SEPARATE from scheduler lateness so the two never get
    #: blamed for each other.
    exit_quote_latency_ms: Optional[float] = None
    #: Terminal outcome: ``SETTLED`` or the refusal status. Always present once
    #: the arm has been processed, so a censored arm also carries its timing.
    outcome: Optional[str] = None
    failure_reason: Optional[str] = None

    def _lateness(self) -> None:
        if (self.target_exit_at is not None
                and self.scheduler_detected_due_at is not None):
            self.scheduler_lateness_ms = round(
                max(0.0, self.scheduler_detected_due_at - self.target_exit_at)
                * 1000.0, 3)
        if (self.target_exit_at is not None
                and self.settlement_completed_at is not None):
            self.settlement_lateness_ms = round(
                max(0.0, self.settlement_completed_at - self.target_exit_at)
                * 1000.0, 3)
        if (self.exit_quote_requested_at is not None
                and self.exit_quote_received_at is not None):
            self.exit_quote_latency_ms = round(
                max(0.0, self.exit_quote_received_at
                    - self.exit_quote_requested_at) * 1000.0, 3)

    def as_dict(self) -> dict[str, Any]:
        self._lateness()
        return {
            "signal_id": self.signal_id,
            "arm": self.arm,
            "entry_confirmed_at": self.entry_confirmed_at,
            "target_exit_at": self.target_exit_at,
            "scheduler_detected_due_at": self.scheduler_detected_due_at,
            "exit_quote_requested_at": self.exit_quote_requested_at,
            "exit_quote_received_at": self.exit_quote_received_at,
            "settlement_completed_at": self.settlement_completed_at,
            "scheduler_lateness_ms": self.scheduler_lateness_ms,
            "settlement_lateness_ms": self.settlement_lateness_ms,
            "exit_quote_latency_ms": self.exit_quote_latency_ms,
            "outcome": self.outcome,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExitTiming":
        return cls(
            signal_id=str(raw.get("signal_id") or ""),
            arm=str(raw.get("arm") or ""),
            target_exit_at=_num(raw.get("target_exit_at")),
            entry_confirmed_at=_num(raw.get("entry_confirmed_at")),
            scheduler_detected_due_at=_num(raw.get("scheduler_detected_due_at")),
            exit_quote_requested_at=_num(
                raw.get("exit_quote_requested_at")),
            exit_quote_received_at=_num(raw.get("exit_quote_received_at")),
            settlement_completed_at=_num(raw.get("settlement_completed_at")),
            outcome=(str(raw["outcome"]) if raw.get("outcome") else None),
            failure_reason=(str(raw["failure_reason"])
                            if raw.get("failure_reason") else None),
        )


class AmarihCounters:
    """Session counters. Deliberately NOT restored: they describe one process."""

    polls: int = 0
    events_seen: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    stale: int = 0
    entries_opened: int = 0
    entry_refused: int = 0
    arms_settled: int = 0
    arms_censored: int = 0
    #: Accepted by the adapter but refused by the EXPERIMENT's own eligibility
    #: policy (its signal-age limit or its de-duplication). Kept separate from
    #: ``rejected`` so an operator can tell a malformed feed from a signal that
    #: was well formed and simply too old to be this experiment's population.
    not_eligible: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class PendingNotification:
    """One message the runtime wants sent, with the key that de-duplicates it."""

    key: str
    text: str


@dataclass
class _OpenArm:
    """One outstanding research arm as it is written to disk and read back."""

    signal_id: str
    arm: str
    mint: str
    symbol: Optional[str]
    entry_ts: float
    scheduled_deadline: float
    token_quantity: float
    token_decimals: Optional[int]
    #: The EXACT integer token amount the entry quote produced. The SELL leg must
    #: quote this quantity, not a re-derived float, or the exit is priced for a
    #: slightly different size than the entry acquired.
    token_quantity_raw: Optional[int]
    entry_cost_usd: float
    entry_basis: str
    #: The market the entry quote traded in, so the exit can be quoted against
    #: the same book. ``None`` for a record written before this field existed,
    #: which then falls back to an unverified discovery route.
    entry_market: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "arm": self.arm,
            "mint": self.mint,
            "symbol": self.symbol,
            "entry_ts": self.entry_ts,
            "scheduled_deadline": self.scheduled_deadline,
            "token_quantity": self.token_quantity,
            "token_decimals": self.token_decimals,
            "token_quantity_raw": self.token_quantity_raw,
            "entry_cost_usd": self.entry_cost_usd,
            "entry_basis": self.entry_basis,
            "entry_market": self.entry_market,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "_OpenArm":
        return cls(
            signal_id=str(raw["signal_id"]),
            arm=str(raw["arm"]),
            mint=str(raw["mint"]),
            symbol=(str(raw["symbol"]) if raw.get("symbol") else None),
            entry_ts=float(raw["entry_ts"]),
            scheduled_deadline=float(raw["scheduled_deadline"]),
            token_quantity=float(raw["token_quantity"]),
            token_decimals=(
                int(raw["token_decimals"])
                if raw.get("token_decimals") is not None else None),
            token_quantity_raw=(
                int(raw["token_quantity_raw"])
                if raw.get("token_quantity_raw") is not None else None),
            entry_cost_usd=float(raw["entry_cost_usd"]),
            entry_basis=str(raw.get("entry_basis") or "executable_quote"),
            entry_market=(
                str(raw["entry_market"]) if raw.get("entry_market") else None),
        )


class AmarihRuntime:
    """
    Drive the Amarih research book on the existing PAPER quote stack.

    Transport and storage are injected. ``buy_quote`` returns the executable BUY
    for a USD notional, ``sell_quote`` the executable SELL for an exact token
    quantity, and ``notify`` receives already-rendered messages. The three
    coroutines are the ONLY ways this class can leave the process.
    """

    def __init__(
        self,
        *,
        source: Any,
        experiment: AmarihExperiment,
        state_path: Path,
        buy_quote: Callable[[str, float], Awaitable[Optional[Mapping[str, Any]]]],
        sell_quote: Callable[..., Awaitable[Optional[Mapping[str, Any]]]],
        notify: Optional[Callable[[str], None]] = None,
        arm_notional_usd: float = DEFAULT_ARM_NOTIONAL_USD,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        clock: Any = None,
    ) -> None:
        self.source = source
        self.experiment = experiment
        self.state_path = Path(state_path)
        self.buy_quote = buy_quote
        self.sell_quote = sell_quote
        self._notify_cb = notify
        self.arm_notional_usd = float(arm_notional_usd)
        self.poll_interval_s = float(poll_interval_s)
        self._now = clock.now if clock is not None else time.time

        self.counters = AmarihCounters()
        #: Whether the poll loop is running, for the status surface.
        self.running = False
        self.last_poll_at: Optional[float] = None
        self.last_error: Optional[str] = None
        #: Whether the CURRENT outage has already been reported. Deliberately
        #: in-memory: an outage is a fact about this process, and it is cleared
        #: by the first successful poll so the next outage pages again.
        self._poller_failure_reported = False

        #: Restored from disk in ``restore()``.
        self._seen_ids: list[str] = []
        self._open_arms: dict[tuple[str, str], _OpenArm] = {}
        self._notified: list[str] = []
        self._pending: list[PendingNotification] = []
        #: (signal_id, arm) -> exit phase timing. Restored from disk.
        self._exit_timings: dict[tuple[str, str], ExitTiming] = {}

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "seen_signal_ids": self._seen_ids[-MAX_PERSISTED_SEEN_IDS:],
            "open_arms": [a.as_dict() for a in self._open_arms.values()],
            "notified": self._notified[-MAX_PERSISTED_NOTIFICATION_KEYS:],
            "last_poll_at": self.last_poll_at,
            # Per-arm exit timing, so a restart keeps the measurements a live
            # process already made. Append-only within the run: an arm's record
            # is written once it has been processed and is never rewritten.
            "exit_timings": [t.as_dict() for t in self._exit_timings.values()],
        }

    def persist(self) -> None:
        """
        Write the resumable state. Best-effort: a failed write is logged, never
        raised into the polling path -- losing a telemetry line must not cost a
        trade, and the positions themselves are re-derived from the experiment.
        """
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._state_payload(), sort_keys=True), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            self.last_error = f"state write failed: {type(exc).__name__}"

    def restore(self) -> dict[str, Any]:
        """
        Re-hydrate the durable state and re-open every outstanding arm.

        An arm is restored into the EXPERIMENT (not just into this driver), so
        its scheduled deadline still fires and its own $10 ledger settles it
        exactly as it would have had the process never stopped. Restoring is
        idempotent: an arm already open in the experiment is not duplicated.

        The arms' BANKROLLS are re-hydrated here too, from the settled exit
        records, before any arm is re-opened. Order matters: restoring a
        position against a bankroll still sitting at its starting notional would
        let an arm that should have been unable to afford its next notional open
        one anyway.
        """
        restored_arms = 0
        raw: dict[str, Any] = {}
        restore = getattr(self.experiment, "restore_bankrolls", None)
        if callable(restore):
            try:
                restore()
            except Exception as exc:  # noqa: BLE001 - recovery is best-effort
                self.last_error = f"bankroll restore failed: {type(exc).__name__}"
        try:
            if self.state_path.exists():
                raw = json.loads(self.state_path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError) as exc:
            self.last_error = f"state read failed: {type(exc).__name__}"
            raw = {}

        self._seen_ids = [str(x) for x in (raw.get("seen_signal_ids") or [])]
        self._notified = [str(x) for x in (raw.get("notified") or [])]
        self.last_poll_at = raw.get("last_poll_at")

        # Restore the exit timing already OBSERVED for arms processed before the
        # restart. Without this a restart would lose the measurement of a late
        # exit -- which is exactly the number this feature exists to report.
        for entry in (raw.get("exit_timings") or []):
            if not isinstance(entry, Mapping):
                continue
            try:
                timing = ExitTiming.from_dict(entry)
            except (TypeError, ValueError):
                continue
            if timing.signal_id and timing.arm:
                self._exit_timings[(timing.signal_id, timing.arm)] = timing

        # The pumpview adapter de-duplicates process-locally; seeding its set
        # from the durable list is what makes that guarantee survive a restart.
        seed = getattr(self.source, "seen_ids", None)
        if isinstance(seed, set):
            seed.update(self._seen_ids)

        for entry in (raw.get("open_arms") or []):
            try:
                arm = _OpenArm.from_dict(entry)
                AmarihArm(arm.arm)  # refuse an arm name this build does not know
            except (KeyError, TypeError, ValueError):
                # An unreadable arm record is reported, not silently dropped and
                # not guessed at: a made-up position would be worse than a
                # missing one.
                self.last_error = "an outstanding arm record could not be read"
                continue
            self._open_arms[(arm.signal_id, arm.arm)] = arm
            restored = self.experiment.restore_position(
                arm=AmarihArm(arm.arm),
                signal_id=arm.signal_id,
                mint=arm.mint,
                symbol=arm.symbol,
                entry_ts=arm.entry_ts,
                scheduled_deadline=arm.scheduled_deadline,
                token_quantity=arm.token_quantity,
                entry_cost_usd=arm.entry_cost_usd,
                entry_basis=arm.entry_basis,
                entry_market=arm.entry_market,
            )
            if restored is not None:
                restored_arms += 1
        return {"restored_arms": restored_arms,
                "seen_signal_ids": len(self._seen_ids),
                "open_arms": len(self._open_arms)}

    # ------------------------------------------------------------------
    # notifications
    # ------------------------------------------------------------------

    def _queue(self, key: str, text: str) -> None:
        """
        Queue a message once, ever.

        ``key`` is stable across ticks and restarts, so a retried tick, a
        re-poll of the same signal, or a restart cannot deliver the same fact
        twice. That matters more than it sounds: a duplicated "3 arms opened"
        message reads as six arms and would be indistinguishable from a real
        double entry.
        """
        if key in self._notified:
            return
        self._notified.append(key)
        self._pending.append(PendingNotification(key=key, text=text))

    def drain_notifications(self) -> list[PendingNotification]:
        out = self._pending
        self._pending = []
        return out

    def flush_notifications(self) -> int:
        """
        Hand queued messages to the notifier and mark them delivered.

        Only AFTER a returned (non-raising) send is the key already recorded --
        it was recorded at queue time, which is what prevents a duplicate. A
        failing notifier therefore loses that message rather than risking a
        double send; the failure is recorded so the operator can see it.
        """
        if self._notify_cb is None:
            self._pending = []
            return 0
        sent = 0
        for item in self.drain_notifications():
            try:
                self._notify_cb(item.text)
                sent += 1
            except Exception as exc:  # noqa: BLE001 - a send must never break a tick
                self.last_error = f"notification failed: {type(exc).__name__}"
        return sent

    # ------------------------------------------------------------------
    # polling
    # ------------------------------------------------------------------

    async def poll_once(self, *, limit: int = 200) -> dict[str, Any]:
        """
        One poll: fetch, normalise, then price and fan out each fresh signal.

        THE FETCH GOES TO A WORKER THREAD. The adapter's transport is a
        synchronous ``httpx`` call with a 30s timeout, so awaiting it directly
        would block the event loop -- and with it the monitor tick, the exit
        scheduler and every other task -- for as long as PumpView takes to
        answer. This is the same defect the market stack already fixed by
        moving its provider transport off the loop; a new source must not
        reintroduce it just because its call site is ``async``.

        Everything after the fetch is normalisation and a quote await, neither
        of which blocks.
        """
        now = self._now()
        self.counters.polls += 1
        try:
            report = await asyncio.to_thread(self.source.poll_once, limit=limit)
        except Exception as exc:  # noqa: BLE001 - a poll failure is reported
            self.last_error = f"pumpview poll failed: {type(exc).__name__}"
            # ONE MESSAGE PER OUTAGE, NOT ONE PER CYCLE.
            #
            # The key used to carry the cycle's own timestamp, so at the 15s
            # cadence an outage lasting an hour sent ~240 identical pages. An
            # operator paged every 15 seconds stops reading the pages, which is
            # the exact failure deduplication exists to prevent -- and it buries
            # the first, informative one.
            #
            # The flag is cleared by a successful poll, so recovery re-arms the
            # alert; the timestamp still belongs in the KEY so a new outage (or a
            # restart mid-outage) is a fact the durable ``notified`` list has not
            # seen yet and is therefore delivered.
            if not self._poller_failure_reported:
                self._poller_failure_reported = True
                self._queue(
                    f"poller_error:{int(now)}",
                    f"⚠️ AMARIH poll FAILED: {type(exc).__name__}. "
                    f"No new signals this cycle; the poller is still running.",
                )
            self.persist()
            self.flush_notifications()
            return {"ok": False, "reason": self.last_error}

        # A poll that answered ends the outage, so the next failure pages again.
        self._poller_failure_reported = False
        self.last_poll_at = now
        self.counters.events_seen += getattr(report, "events_seen", 0)
        for _sid, reason in getattr(report, "refused", ()):
            if reason == "DUPLICATE":
                self.counters.duplicates += 1
            elif reason == "STALE":
                self.counters.stale += 1
            self.counters.rejected += 1

        opened = 0
        for signal in getattr(report, "accepted", ()):
            if signal.signal_id in self._seen_ids:
                self.counters.duplicates += 1
                continue
            self._seen_ids.append(signal.signal_id)
            self.counters.accepted += 1
            self._queue(
                f"signal:{signal.signal_id}",
                self._render_signal(signal),
            )
            opened += await self._enter(signal)

        # PERSIST FIRST, THEN SEND. The notification keys are part of the
        # durable state, so a crash between the two loses a message rather than
        # repeating one -- and a repeated "3 arms opened" reads as six arms.
        self.persist()
        self.flush_notifications()
        return {
            "ok": True,
            "events_seen": getattr(report, "events_seen", 0),
            "accepted": len(getattr(report, "accepted", ())),
            "arms_opened": opened,
        }

    # ------------------------------------------------------------------
    # entry
    # ------------------------------------------------------------------

    def _render_signal(self, signal: Any) -> str:
        return (
            f"🧪 AMARIH signal {signal.signal_id}\n"
            f"{signal.symbol or '?'} {signal.mint}\n"
            f"latency {signal.signal_to_receipt_latency_ms:.0f}ms "
            f"score {signal.buy_score if signal.buy_score is not None else '-'} "
            f"mc {signal.entry_market_cap_usd if signal.entry_market_cap_usd else '-'}"
        )

    async def _enter(self, signal: Any) -> int:
        """
        Price one accepted signal ONCE and fan it out to the three paired arms.

        Returns the number of arms actually opened (0 when the entry is refused).
        """
        now = self._now()
        record = AmarihSignalRecord(
            signal_id=signal.signal_id,
            mint=signal.mint or "",
            symbol=signal.symbol,
            strategy=signal.strategy,
            signal_ts=signal.fired_at,
            receipt_ts=signal.received_at,
            signal_to_receipt_latency_ms=signal.signal_to_receipt_latency_ms,
            market_cap_signal_usd=signal.entry_market_cap_usd,
            buy_score=signal.buy_score,
            owner_wallet=signal.owner_wallet,
            status="ACCEPTED",
            token_data=dict(signal.raw or {}),
            raw=dict(signal.raw or {}),
        )
        # THE EXPERIMENT'S OWN ELIGIBILITY VERDICT IS HONOURED.
        #
        # ``note_signal`` returns whether this is a NEW, ELIGIBLE signal: it
        # de-duplicates and enforces the experiment's declared
        # ``max_signal_age_s``. Its return value used to be discarded, so both
        # checks were dead on the live path. The runtime's own dedup masks the
        # duplicate case, but the age check did not: the adapter's
        # ``max_age_s`` is operator-tunable (``PUMPVIEW_AMARIH_MAX_AGE_S``)
        # while the experiment's is fixed, so widening the adapter's window
        # silently widened the experiment's signal population past the age its
        # own design declares.
        #
        # The signal is still RECORDED above -- a stale signal stays in the
        # denominator -- and it is simply not priced.
        if not self.experiment.note_signal(record):
            self.counters.not_eligible += 1
            return 0

        quote: Optional[Mapping[str, Any]] = None
        failure: Optional[str] = None
        try:
            quote = await self.buy_quote(signal.mint, self.arm_notional_usd)
        except Exception as exc:  # noqa: BLE001 - a quote failure is a reason
            failure = f"BUY quote raised {type(exc).__name__}"

        observation = self._entry_from_quote(signal, quote, failure=failure, now=now)
        experiment = self.experiment
        opened: dict[AmarihArm, Any] = experiment.open_from_entry(observation)
        if not opened:
            self.counters.entry_refused += 1
            reason = getattr(experiment, '_last_refusal_reason', None)
            if reason:
                # Clear it after use
                try:
                    experiment._last_refusal_reason = None
                except Exception:
                    pass
                msg_reason = reason
            else:
                msg_reason = observation.rejection_reason
            self._queue(
                f"entry_refused:{signal.signal_id}",
                f"⛔️ AMARIH signal {signal.signal_id} NOT traded: "
                f"{observation.validation}"
                f"{' - ' + msg_reason if msg_reason else ''}",
            )
            return 0

        for arm, position in opened.items():
            key = (position.signal_id, arm.value)
            self._open_arms[key] = _OpenArm(
                signal_id=position.signal_id,
                arm=arm.value,
                mint=position.mint,
                symbol=position.symbol,
                entry_ts=position.entry_ts,
                scheduled_deadline=position.scheduled_deadline,
                token_quantity=position.token_quantity,
                token_decimals=observation.token_decimals,
                token_quantity_raw=_num(quote.get("token_quantity_raw"))
                if quote else None,
                entry_cost_usd=position.entry_cost_usd,
                entry_basis=position.entry_basis,
                entry_market=(position.entry_market
                               or observation.market_id),
            )
        self.counters.entries_opened += 1
        self._queue(
            f"arms_opened:{signal.signal_id}",
            self._render_arms_opened(observation, list(opened)),
        )
        return len(opened)

    def _entry_from_quote(
        self, signal: Any, quote: Optional[Mapping[str, Any]],
        *, failure: Optional[str], now: float,
    ) -> EntryObservation:
        """
        Build THE canonical entry observation from one executable BUY quote.

        Validation is ordered so the FIRST thing wrong is the thing recorded:
        no quote, then identity, then a usable size. Anything other than
        ``ENTRY_VALID`` opens nothing on any arm.
        """
        base = dict(
            signal_id=signal.signal_id,
            mint=signal.mint,
            symbol=signal.symbol,
            decision_ts=now,
            entry_ts=now,
            signal_to_entry_latency_ms=round(
                (now - signal.fired_at) * 1000.0, 3),
            entry_basis="executable_quote",
            market_cap_usd=signal.entry_market_cap_usd,
            provider=None, venue=None, market_id=None,
            quote_ts=None, quote_age_ms=None,
            input_amount_usd=self.arm_notional_usd,
            token_quantity=None, token_decimals=None, fees_usd=None,
            quote_asset=None,
            liquidity_usd=None, volume_to_mcap=None,
            raw_quote=dict(quote or {}),
        )
        if not quote:
            return EntryObservation(
                validation=ENTRY_QUOTE_UNAVAILABLE,
                rejection_reason=failure or "no executable BUY quote",
                **base)

        identity = str(quote.get("identity") or "").strip().lower()
        if identity in IDENTITY_MISMATCH_VALUES:
            return EntryObservation(
                validation=ENTRY_IDENTITY_UNKNOWN,
                rejection_reason="IDENTITY_MISMATCH: the quoted market is not "
                                 "provably the requested token",
                **base)
        if identity not in IDENTITY_PASS_VALUES:
            # NOT AN ALLOWLIST-AND-NOTHING-ELSE: anything outside the declared
            # pass vocabulary is UNPROVEN, including the explicit "unknown"
            # tokens, the empty string and any token a provider has just
            # invented. Only a declared proof opens a position.
            return EntryObservation(
                validation=ENTRY_IDENTITY_UNKNOWN,
                rejection_reason=(
                    f"IDENTITY_UNKNOWN ({identity or 'empty'}): refusing to "
                    "open without a verified market identity"
                ),
                **base)

        quantity = _num(quote.get("token_quantity"))
        decimals = _num(quote.get("token_decimals"))
        if not quantity or quantity <= 0:
            return EntryObservation(
                validation=ENTRY_QUOTE_UNAVAILABLE,
                rejection_reason="the executable BUY quote produced no token "
                                 "quantity, so no arm can be opened",
                **base)

        base.update(
            token_quantity=float(quantity),
            token_decimals=(int(decimals) if decimals is not None else None),
            fees_usd=_num(quote.get("fees_usd")),
            provider=(str(quote["provider"]) if quote.get("provider") else None),
            venue=(str(quote["venue"]) if quote.get("venue") else None),
            market_id=(str(quote["market_id"]) if quote.get("market_id") else None),
            # THE COST BASIS IS WHAT WAS ACTUALLY DEPLOYED, NOT THE REQUESTED
            # NOTIONAL. ``base`` seeds ``input_amount_usd`` with the flat
            # ``arm_notional_usd``, and this update used to leave it there. The
            # quote already reports the SOL that was actually spent, which is
            # strictly LESS: the outside-swap reserve is withheld before the
            # order is built. Every closed arm therefore carried a cost basis
            # larger than the cost it paid, which biases net PnL negative and
            # net return low by that same margin on EVERY trade -- a systematic
            # error, not noise, and one that never announces itself because both
            # numbers stay plausible.
            #
            # Falls back to the notional only when the provider genuinely does
            # not report a deployed amount, which is recorded as an estimate
            # rather than passed off as an observation.
            input_amount_usd=(
                _num(quote.get("input_amount_usd"))
                or self.arm_notional_usd),
            quote_ts=_num(quote.get("quote_ts")),
            liquidity_usd=_num(quote.get("liquidity_usd")),
            quote_asset=(str(quote["quote_asset"]) if quote.get("quote_asset") else None),
        )
        return EntryObservation(validation=ENTRY_VALID, rejection_reason=None, **base)

    def _render_arms_opened(
        self, observation: EntryObservation, arms: Sequence[AmarihArm],
    ) -> str:
        per_arm = "\n".join(
            f"  • {arm.value}: exit at entry+{int(ARM_HORIZON[arm])}s"
            for arm in arms)
        return (
            f"✅ AMARIH {observation.symbol or '?'} {observation.mint}\n"
            f"1 entry observation → {len(arms)} paired PAPER arms "
            f"(${observation.input_amount_usd:,.2f} each)\n"
            f"qty {observation.token_quantity} "
            f"({observation.provider or '?'}/{observation.venue or '?'})\n"
            f"{per_arm}\n"
            f"These are PAIRED outcomes of ONE signal, not {len(arms)} signals."
        )

    # ------------------------------------------------------------------
    # scheduled exits
    # ------------------------------------------------------------------

    async def tick(self) -> dict[str, Any]:
        """
        Settle every arm whose scheduled deadline has arrived.

        ONE attempt per arm, and that attempt is TERMINAL. That is the
        experiment's own contract: an outcome observed at the scheduled
        deadline is the outcome, and a quote that was unavailable AT the
        deadline is recorded as such rather than silently replaced by a later
        (and better) quote. Retrying would quietly convert a censored arm into a
        settled one and bias every horizon in the same direction, which is
        precisely what a paired-exit comparison must not do.

        So a refused exit quote is RECORDED and the arm is retired -- never
        retried, never fabricated into a fill.

        TIMING IS OBSERVED, NOT DERIVED. ``scheduler_detected_due_at`` is stamped
        once, when the loop first sees the arm as due, and is never adjusted
        later. An arm detected five seconds after its deadline therefore keeps a
        five-second lateness, which is the fact the operator needs.

        FAULT ISOLATION. Each arm is processed inside its own guard, so one arm
        whose processing raises cannot abort the loop and starve the other due
        arms -- which is what a shared `try` around the whole body would have
        done.
        """
        now = self._now()
        due = [a for a in self._open_arms.values()
               if a.scheduled_deadline <= now]
        settled = 0
        censored = 0
        for arm_record in due:
            timing = self._begin_timing(arm_record, now)
            try:
                result = await self._settle_one(arm_record, now=now)
            except Exception as exc:  # noqa: BLE001 - one arm must not starve
                #: A processing failure is still an OUTCOME, and it is recorded
                #: with its reason rather than silently retiring the arm or
                #: looping on it forever.
                self.last_error = (
                    f"arm {arm_record.arm} exit processing failed: "
                    f"{type(exc).__name__}")
                self._finish_timing(arm_record, timing, outcome=None,
                                    failure_reason=(
                                        f"exit processing failed: "
                                        f"{type(exc).__name__}"))
                self._persist_timings()
                # Terminal: retire the arm so a broken record cannot spin the
                # loop, and record the outcome as a refusal.
                self._open_arms.pop(
                    (arm_record.signal_id, arm_record.arm), None)
                self.persist()
                censored += 1
                self.counters.arms_censored += 1
                self._queue(
                    f"censored:{arm_record.signal_id}:{arm_record.arm}",
                    self._render_censored_text(
                        arm_record,
                        "EXIT_PROCESSING_FAILED",
                        timing.failure_reason or "exit processing failed"))
                continue
            # Terminal either way: the deadline observation is the record.
            self._open_arms.pop((arm_record.signal_id, arm_record.arm), None)
            self._finish_timing(
                arm_record, timing,
                outcome=getattr(result, "settlement_status", None),
                failure_reason=getattr(result, "failure_reason", None))
            if result.settlement_status == EXIT_SETTLED:
                settled += 1
                self._queue(
                    f"settled:{arm_record.signal_id}:{arm_record.arm}",
                    self._render_settlement(arm_record, result))
            else:
                censored += 1
                self._queue(
                    f"censored:{arm_record.signal_id}:{arm_record.arm}",
                    self._render_censored(arm_record, result))
        self._persist_timings()
        if due:
            self.counters.arms_settled += settled
            self.counters.arms_censored += censored
            self.persist()
            self.flush_notifications()
        return {
            "due": len(due),
            "settled": settled,
            "censored": censored,
            #: Longest scheduler lateness across the arms processed this tick,
            #: so a caller can poll one number instead of every record.
            "max_scheduler_lateness_ms": (
                max((t.scheduler_lateness_ms or 0.0
                     for t in self._exit_timings.values()), default=0.0)
            ),
        }

    # -- exit timing -------------------------------------------------------

    def _begin_timing(self, arm_record: _OpenArm, now: float) -> ExitTiming:
        """
        Open (or resume) the timing record for a due arm.

        IDEMPOTENT BY (signal_id, arm): if the arm was already seen as due
        before a restart, the FIRST ``scheduler_detected_due_at`` is kept rather
        than overwritten with the later wake-up's. A restart that re-detects an
        overdue arm extends the measured lateness, which is true, and does not
        erase the original detection.
        """
        key = (arm_record.signal_id, arm_record.arm)
        existing = self._exit_timings.get(key)
        if existing is not None:
            if existing.scheduler_detected_due_at is None:
                existing.scheduler_detected_due_at = now
            return existing
        timing = ExitTiming(
            signal_id=arm_record.signal_id,
            arm=arm_record.arm,
            target_exit_at=arm_record.scheduled_deadline,
            entry_confirmed_at=arm_record.entry_ts,
            scheduler_detected_due_at=now,
        )
        self._exit_timings[key] = timing
        return timing

    def _finish_timing(self, arm_record: _OpenArm, timing: ExitTiming,
                       *, outcome: Optional[str],
                       failure_reason: Optional[str]) -> None:
        """Close the timing record with the phases actually completed."""
        now = self._now()
        # QUOTE LATENCY IS MEASURED, NOT ASSUMED. ``_settle_one`` records when it
        # asked and when the answer arrived; this only stamps settlement.
        timing.settlement_completed_at = now
        timing.outcome = outcome
        timing.failure_reason = failure_reason

    def _note_exit_quote_requested(self, arm_record: _OpenArm,
                                   timing: ExitTiming, now: float) -> None:
        if timing.exit_quote_requested_at is None:
            timing.exit_quote_requested_at = now

    def _note_exit_quote_received(self, arm_record: _OpenArm,
                                  timing: ExitTiming, now: float) -> None:
        if timing.exit_quote_received_at is None:
            timing.exit_quote_received_at = now

    def _timing_for(self, arm_record: _OpenArm) -> Optional[ExitTiming]:
        return self._exit_timings.get(
            (arm_record.signal_id, arm_record.arm))

    def _persist_timings(self) -> None:
        """Write the timing records. Idempotent and never raises."""
        self.persist()

    def _render_censored_text(self, arm_record: _OpenArm, status: str,
                              reason: str) -> str:
        """A censored outcome when no :class:`ExitResult` exists to render."""
        return (
            f"⛔️ AMARIH {arm_record.arm} exit NOT settled "
            f"({arm_record.symbol or arm_record.mint})\n"
            f"reason {status}"
            f"{' / ' + str(reason) if reason else ''}\n"
            f"Recorded as CENSORED for this deadline. No fabricated fill, and the "
            f"deadline outcome is not replaced by a later quote."
        )

    def exit_timings(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Every observed exit timing, as plain dictionaries."""
        return {k: v.as_dict() for k, v in self._exit_timings.items()}

    async def _request_sell_quote(self, arm_record: _OpenArm) -> Any:
        """
        Ask the injected transport for an executable SELL of the EXACT quantity
        the entry quote produced, pinned to the market that entry traded in.

        THE ENTRY MARKET IS PASSED. Same-market identity on the exit leg is
        proven by comparing the quoted market against a KNOWN address for this
        token; without one the identity resolves to ``UNKNOWN``, the fill is
        refused, and the arm is censored. That is not a market condition -- it
        is an unanswered question, asked 100% of the time.

        THE PIN IS OPTIONAL FOR THE CALLER, NOT FOR THE EXIT. A transport that
        does not accept the keyword is still called, with the legacy three
        positional arguments, so an injected double written against the older
        signature keeps working instead of raising ``TypeError`` into the
        classification path and being recorded as a quote outage.
        """
        quantity = (
            int(arm_record.token_quantity_raw)
            if arm_record.token_quantity_raw
            else int(round(arm_record.token_quantity))
        )
        decimals = int(arm_record.token_decimals or 6)
        if arm_record.entry_market and self._sell_quote_accepts_pin():
            return await self.sell_quote(
                arm_record.mint, quantity, decimals,
                entry_pool=arm_record.entry_market)
        return await self.sell_quote(arm_record.mint, quantity, decimals)

    def _sell_quote_accepts_pin(self) -> bool:
        """
        Does the injected ``sell_quote`` accept the ``entry_pool`` keyword?

        Resolved ONCE and cached. A transport that cannot be introspected (a
        C-implemented callable, a ``functools.partial`` over one, a mock) is
        assumed to accept it, because the production transport does and a
        false negative would silently reintroduce the unpinned exit.
        """
        cached = getattr(self, "_sell_pin_ok", None)
        if cached is not None:
            return cached
        try:
            params = inspect.signature(self.sell_quote).parameters
            accepts = (
                "entry_pool" in params
                or any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in params.values())
            )
        except (TypeError, ValueError):
            accepts = True
        self._sell_pin_ok = accepts
        return accepts

    async def _settle_one(self, arm_record: _OpenArm, *, now: float) -> Any:
        arm = AmarihArm(arm_record.arm)
        failure: Optional[str] = None
        quote: Optional[Mapping[str, Any]] = None
        timing = self._timing_for(arm_record)
        # PROVIDER LATENCY, MEASURED SEPARATELY FROM SCHEDULER LATENCY. The
        # request and arrival stamps are taken here, around the await, so a slow
        # quote is never folded into the scheduler's lateness.
        requested_at = self._now()
        if timing is not None:
            self._note_exit_quote_requested(arm_record, timing, requested_at)
        try:
            quote = await self._request_sell_quote(arm_record)
        except Exception as exc:  # noqa: BLE001 - a quote failure is a reason
            failure = f"SELL quote raised {type(exc).__name__}"
        if timing is not None:
            self._note_exit_quote_received(arm_record, timing, self._now())

        payload: Optional[Mapping[str, Any]] = None
        if quote:
            identity = str(quote.get("identity") or "").strip().lower()
            payload = {
                "proceeds_usd": _num(quote.get("proceeds_usd")),
                "quote_ts": _num(quote.get("quote_ts")),
                "market_id": quote.get("market_id"),
                "provider": quote.get("provider"),
                "venue": quote.get("venue"),
                "fees_usd": _num(quote.get("fees_usd")),
                # Same ALLOWLIST as the entry leg. An unproven exit market is
                # not a settlement: `settle_arm` reads this as False and records
                # EXIT_IDENTITY_MISMATCH, so no net PnL is ever booked from a
                # quote whose book could not be proven to be the entry's book.
                "valid_market": identity in IDENTITY_PASS_VALUES,
                "raw": dict(quote),
            }
        elif failure:
            self.last_error = failure

        return self.experiment.settle_arm(
            signal_id=arm_record.signal_id,
            arm=arm,
            sell_quote=payload,
            observed_ts=now,
        )

    def _render_censored(self, arm_record: _OpenArm, result: Any) -> str:
        return (
            f"⛔️ AMARIH {arm_record.arm} exit NOT settled at its deadline "
            f"({arm_record.symbol or arm_record.mint})\n"
            f"reason {result.settlement_status}"
            f"{' / ' + str(result.failure_reason) if result.failure_reason else ''}\n"
            f"Recorded as CENSORED for this deadline. No fabricated fill, and the "
            f"deadline outcome is not replaced by a later quote."
        )

    def _render_settlement(self, arm_record: _OpenArm, result: Any) -> str:
        pnl = result.net_pnl_usd
        pct = result.net_return_pct
        return (
            f"🧪 AMARIH {arm_record.arm} SETTLED "
            f"({arm_record.symbol or arm_record.mint})\n"
            f"entry ${result.total_buy_cost_usd:,.4f} → "
            f"proceeds ${result.net_sell_proceeds_usd:,.4f}\n"
            f"PnL {'%+.4f' % pnl if pnl is not None else '-'} USD "
            f"({'+%.2f' % pct if pct is not None else '-'}%)\n"
            f"class {result.classification} · "
            f"deadline deviation {result.deadline_deviation_ms:.0f}ms"
        )

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def open_arm_count(self) -> int:
        return len(self._open_arms)

    def status(self) -> dict[str, Any]:
        """Everything the operator's status block needs, and nothing invented."""
        return {
            "running": self.running,
            "last_poll_at": self.last_poll_at,
            "signals_seen": self.counters.events_seen,
            "signals_accepted": self.counters.accepted,
            "signals_rejected": self.counters.rejected,
            "open_arms": self.open_arm_count(),
            "last_error": self.last_error,
            "counters": self.counters.as_dict(),
        }


def _num(value: Any) -> Optional[float]:
    """Coerce to a finite float or ``None``. NaN/inf are never propagated."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


__all__ = [
    "AmarihRuntime",
    "AmarihCounters",
    "PendingNotification",
    "SOURCE_DISABLED_BY_AMARIH",
    "DEFAULT_POLL_INTERVAL_S",
    "MAX_PERSISTED_SEEN_IDS",
    "MAX_PERSISTED_NOTIFICATION_KEYS",
    "RUNTIME_SCHEMA_VERSION",
]
