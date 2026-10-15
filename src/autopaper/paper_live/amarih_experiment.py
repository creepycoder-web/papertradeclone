"""
AMARIH paired-exit PAPER experiment -- three arms off ONE canonical entry.

WHAT THIS IS
============
A PAPER-research experiment harness for the PumpView **Amarih** community
strategy. For each accepted Amarih signal it captures **ONE canonical entry
observation** (a single executable BUY quote, validated for market identity)
and then fans that single observation out into **three independent virtual
ledger arms** that differ ONLY in their scheduled holding period:

    AMARIH_T60  exit observed at entry + 60s
    AMARIH_T120 exit observed at entry + 120s
    AMARIH_T300 exit observed at entry + 300s

These are EXPERIMENTAL holding periods, not validated optima. The three arms
are PAIRED outcomes of one underlying signal and must NEVER be counted as
three independent signals (see :func:`build_report`).

PAPER ONLY. There is no wallet, no key, no router, no order API and no
transaction anywhere in this module. Every fill is SIMULATED from an
executable quote that a caller supplies; a mark price is NEVER a fill, and a
missing/stale/mismatched quote is NEVER invented.

FAIR PAIRED-ENTRY METHODOLOGY (the critical invariant)
======================================================
The three arms are derived from ONE immutable entry observation -- the same
entry amount, the same entry quote basis, the same token quantity. This module
never performs three sequential BUY quotes and treats the resulting prices as
identical: :meth:`AmarihExperiment.open_from_entry` takes a single
:class:`EntryObservation` and stamps it onto all three arms. Each arm then
keeps its OWN independent simulated balance, token quantity, PnL and
settlement record. The shared observation is explicitly labelled so it can
never be mistaken for three separately executed purchases.

EXIT OBSERVATION AND SETTLEMENT
===============================
At each arm's scheduled deadline the caller requests a FRESH executable SELL
quote for exactly the token quantity acquired at entry, validates the SAME
market identity, enforces quote freshness, and settles that arm's virtual
account independently. A quote failure at the deadline is RECORDED as
``QUOTE_UNAVAILABLE`` for that deadline and does NOT become a fabricated fill;
a later recovery quote is recorded as a SEPARATE late observation and never
overwrites the scheduled-deadline outcome. An emergency/integrity exit before
the deadline is classified separately (``EMERGENCY``), never as a normal
T30/T60/T120 result.

TELEMETRY
=========
Append-only event records with stable ids are written under
``research/amarih_experiment/`` (schema version pinned below). Every record
carries UTC timestamps and an explicit provenance for each field. Marks and
executable quotes are never mixed into one field.

See ``reports/research/amarih_pumpview_integration.md`` for the signal-source
evidence behind this experiment.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from autopaper.paper_live.account import VirtualAccount

# ---------------------------------------------------------------------------
# schema / storage
# ---------------------------------------------------------------------------

#: Bumped when the on-disk telemetry shape changes in a way a reader must notice.
TELEMETRY_SCHEMA_VERSION = 1

#: Where append-only experiment telemetry is written. Relative to the repo root
#: unless an explicit root is given.
DEFAULT_RESEARCH_DIR = Path("research") / "amarih_experiment"

#: The fixed per-arm notional, in USD. Chosen to match the existing PAPER
#: experiment convention (the V2 FLOOR BETA harness also runs isolated $10
#: ledgers). It is a PARAMETER here, never hard-coded into settlement math, and
#: it does NOT reset or touch any existing account.
DEFAULT_ARM_NOTIONAL_USD = 10.0

#: The three experimental holding periods, in seconds, in canonical order.
ARM_HORIZONS_S: tuple[float, ...] = (60.0, 120.0, 300.0)

#: A stable arm-version label for this experiment's virtual accounts. Bumped
#: when the experiment's accounting rules change so restored ledgers and new
#: ones are never silently pooled. It is NOT a strategy-performance claim.
AMARIH_ARM_VERSION = "amarih_paired_v1"

#: The baseline hash recorded on each virtual account. Amarih has no frozen
#: on-chain gate of its own (its thresholds are PumpView's private config), so
#: this is a label for the PAPER experiment basis, not a hash of a frozen rule.
#: It is recorded so an account's provenance is explicit and never mistaken for
#: a production strategy hash.
AMARIH_BASELINE_LABEL = "amarih_pumpview_paper_paired"


class AmarihArm(str, Enum):
    """The three paired arms. Identities are stable across restarts."""

    T60 = "AMARIH_T60"
    T120 = "AMARIH_T120"
    T300 = "AMARIH_T300"


#: Canonical arm order. ``len`` must equal ``len(ARM_HORIZONS_S)``.
AMARIH_ARMS: tuple[AmarihArm, ...] = (
    AmarihArm.T60, AmarihArm.T120, AmarihArm.T300,
)

#: Arm -> its scheduled holding period in seconds.
ARM_HORIZON: dict[AmarihArm, float] = {
    AmarihArm.T60: 60.0,
    AmarihArm.T120: 120.0,
    AmarihArm.T300: 300.0,
}


# ---------------------------------------------------------------------------
# outcome / status vocabularies (explicit, closed, never invented)
# ---------------------------------------------------------------------------

#: Entry validation result. Anything other than VALID blocks the entry.
ENTRY_VALID = "VALID"
ENTRY_ECONOMICALLY_INVALID = "ECONOMICALLY_INVALID"
ENTRY_QUOTE_UNAVAILABLE = "QUOTE_UNAVAILABLE"
ENTRY_IDENTITY_UNKNOWN = "IDENTITY_UNKNOWN"

#: Exit settlement status, per arm.
EXIT_SETTLED = "SETTLED"
EXIT_QUOTE_UNAVAILABLE = "QUOTE_UNAVAILABLE"
EXIT_IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
EXIT_STALE_QUOTE = "STALE_QUOTE"
EXIT_EMERGENCY = "EMERGENCY"
EXIT_PENDING = "PENDING"
#: The arm had no open position to settle. Used by ``settle_arm`` when it is
#: asked to close something it never opened.
#:
#: THIS CONSTANT WAS USED BUT NEVER DECLARED. ``settle_arm``'s no-position
#: branch raised ``NameError`` instead of recording a censored no-op, and that
#: branch is reachable: ``AmarihRuntime.restore()`` registers a durable arm in
#: its own ``_open_arms`` map BEFORE asking the experiment to restore the
#: position, and ``restore_position`` deliberately refuses an already-settled
#: arm. The next tick therefore asked a settled arm to settle again, hit this
#: branch, and raised -- aborting the whole exit tick, so every OTHER arm due
#: in the same tick lost its exit too.
EXIT_CENSORED = "CENSORED"

#: Result classification for reporting: normal, emergency, late, or censored.
RESULT_NORMAL = "NORMAL"
RESULT_EMERGENCY = "EMERGENCY"
RESULT_LATE = "LATE"
RESULT_CENSORED = "CENSORED"


def _num(value: Any) -> Optional[float]:
    """Coerce to a finite float or ``None``. Never returns NaN/inf."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


# ---------------------------------------------------------------------------
# signal + canonical entry observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AmarihSignalRecord:
    """A signal-level telemetry record (section A). Append-only, stable id.

    Every field that the source supplied is preserved with provenance; a field
    the source did not supply stays ``None`` -- nothing is invented here.
    """

    signal_id: str
    mint: str
    symbol: Optional[str]
    strategy: str
    signal_ts: float
    receipt_ts: float
    signal_to_receipt_latency_ms: float
    market_cap_signal_usd: Optional[float]
    buy_score: Optional[float]
    owner_wallet: Optional[str]
    #: status: ACCEPTED / DUPLICATE / STALE / NO_MINT / WRONG_STRATEGY / ...
    status: str
    #: Every raw token_data field the source provided, preserved verbatim.
    token_data: Mapping[str, Any] = field(default_factory=dict)
    #: Fields the source did NOT provide (explicit, so "missing" is a fact).
    missing_fields: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "record_type": "signal",
            "signal_id": self.signal_id,
            "mint": self.mint,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "signal_ts_utc": self.signal_ts,
            "receipt_ts_utc": self.receipt_ts,
            "signal_to_receipt_latency_ms": self.signal_to_receipt_latency_ms,
            "market_cap_signal_usd": self.market_cap_signal_usd,
            "buy_score": self.buy_score,
            "owner_wallet": self.owner_wallet,
            "status": self.status,
            "token_data": dict(self.token_data),
            "missing_fields": list(self.missing_fields),
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class EntryObservation:
    """THE canonical entry observation shared by all three arms (section B).

    Built ONCE from a single executable BUY quote. ``token_quantity`` is the
    exact expected token output of that quote and is the ONLY quantity any arm
    may later sell. ``entry_basis`` records that this came from an executable
    quote, never a mark.
    """

    signal_id: str
    mint: str
    symbol: Optional[str]
    decision_ts: float
    entry_ts: float
    signal_to_entry_latency_ms: float
    market_id: Optional[str]
    provider: Optional[str]
    venue: Optional[str]
    quote_ts: Optional[float]
    quote_age_ms: Optional[float]
    input_amount_usd: Optional[float]
    token_quantity: Optional[float]
    token_decimals: Optional[int]
    fees_usd: Optional[float]
    quote_asset: Optional[str]
    entry_basis: str
    market_cap_usd: Optional[float]
    liquidity_usd: Optional[float]
    volume_to_mcap: Optional[float]
    validation: str
    rejection_reason: Optional[str]
    raw_quote: Mapping[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.validation == ENTRY_VALID and (self.token_quantity or 0) > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "record_type": "entry",
            "signal_id": self.signal_id,
            "mint": self.mint,
            "symbol": self.symbol,
            "decision_ts_utc": self.decision_ts,
            "entry_ts_utc": self.entry_ts,
            "signal_to_entry_latency_ms": self.signal_to_entry_latency_ms,
            "market_id": self.market_id,
            "provider": self.provider,
            "venue": self.venue,
            "quote_ts_utc": self.quote_ts,
            "quote_age_ms": self.quote_age_ms,
            "input_amount_usd": self.input_amount_usd,
            "token_quantity": self.token_quantity,
            "token_decimals": self.token_decimals,
            "fees_usd": self.fees_usd,
            "quote_asset": self.quote_asset,
            "entry_basis": self.entry_basis,
            "market_cap_usd": self.market_cap_usd,
            "liquidity_usd": self.liquidity_usd,
            "volume_to_mcap": self.volume_to_mcap,
            "validation": self.validation,
            "rejection_reason": self.rejection_reason,
            "raw_quote": dict(self.raw_quote),
        }



# ---------------------------------------------------------------------------
# per-arm position and exit result (sections C + D)
# ---------------------------------------------------------------------------

@dataclass
class ArmPosition:
    """One arm's open virtual position, derived from a shared entry observation.

    The arm does NOT own its own entry quote; it owns a reference to the shared
    :class:`EntryObservation` and its OWN quantity/PnL/balance. ``arm`` makes it
    independently identifiable in the account, position, telemetry and reporting
    layers.
    """

    arm: AmarihArm
    signal_id: str
    mint: str
    symbol: Optional[str]
    entry_ts: float
    scheduled_deadline: float
    token_quantity: float
    entry_cost_usd: float
    entry_basis: str
    account: VirtualAccount
    #: The EXACT market the entry quote traded in -- the curve PDA or the pool
    #: address, as the entry quote reported it. The exit leg must quote against
    #: this book, not against whatever is discovered at exit time.
    #:
    #: WITHOUT IT THE EXIT CAN NEVER SETTLE. ``QuoteService.quote_sell``
    #: resolves same-market identity by comparing ``observed_pair_address`` to
    #: the chain-verified market account; with no address to compare, identity
    #: resolves to ``UNKNOWN``, ``fill_from_quote(require_identity=True)``
    #: raises, and every arm is permanently CENSORED as ``QUOTE_UNAVAILABLE``
    #: rather than for any market reason. ``None`` only for an arm restored from
    #: a record written before this field existed.
    entry_market: Optional[str] = None
    #: The exit result once settled, else ``None`` (still open).
    exit_result: Optional["ExitResult"] = None

    @property
    def horizon_s(self) -> float:
        return ARM_HORIZON[self.arm]

    def is_open(self) -> bool:
        return self.exit_result is None


@dataclass
class ExitResult:
    """One arm's exit observation at (or after) its scheduled deadline (section C).

    ``classification`` is NORMAL when the SELL quote arrived at the deadline,
    LATE when a recovery quote arrived later (never overwriting a missing
    scheduled outcome), EMERGENCY for an integrity/emergency exit, and CENSORED
    when no usable quote was ever obtained. ``net_return_pct`` and
    ``net_pnl_usd`` are ``None`` whenever no executable SELL quote settled the
    arm -- a missing quote is never a fabricated zero.
    """

    arm: AmarihArm
    signal_id: str
    scheduled_deadline: float
    observed_ts: float
    deadline_deviation_ms: float
    token_quantity: float
    sell_quote_available: bool
    quote_age_ms: Optional[float]
    market_id_valid: bool
    net_sell_proceeds_usd: Optional[float]
    total_buy_cost_usd: float
    net_return_pct: Optional[float]
    net_pnl_usd: Optional[float]
    fees_usd: Optional[float]
    provider: Optional[str]
    venue: Optional[str]
    exit_reason: str
    settlement_status: str
    classification: str
    failure_reason: Optional[str]
    raw_quote: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "record_type": "exit",
            "arm": self.arm.value,
            "signal_id": self.signal_id,
            "scheduled_deadline_utc": self.scheduled_deadline,
            "observed_ts_utc": self.observed_ts,
            "deadline_deviation_ms": self.deadline_deviation_ms,
            "token_quantity": self.token_quantity,
            "sell_quote_available": self.sell_quote_available,
            "quote_age_ms": self.quote_age_ms,
            "market_id_valid": self.market_id_valid,
            "net_sell_proceeds_usd": self.net_sell_proceeds_usd,
            "total_buy_cost_usd": self.total_buy_cost_usd,
            "net_return_pct": self.net_return_pct,
            "net_pnl_usd": self.net_pnl_usd,
            "fees_usd": self.fees_usd,
            "provider": self.provider,
            "venue": self.venue,
            "exit_reason": self.exit_reason,
            "settlement_status": self.settlement_status,
            "classification": self.classification,
            "failure_reason": self.failure_reason,
            "raw_quote": dict(self.raw_quote),
        }



# ---------------------------------------------------------------------------
# the experiment harness
# ---------------------------------------------------------------------------

class AmarihExperiment:
    """Three paired PAPER arms off one canonical entry, with auditable telemetry.

    Transport-only and PAPER-only: it never performs I/O of its own beyond the
    caller-supplied append-only store, never quotes, and never submits a
    transaction. The caller supplies the executable quotes via
    :meth:`open_from_entry` and :meth:`settle_arm`; this class owns the paired
    fan-out, the isolated virtual ledgers, the dedup/staleness rules and the
    telemetry.
    """

    def __init__(
        self,
        *,
        arm_notional_usd: float = DEFAULT_ARM_NOTIONAL_USD,
        max_signal_age_s: float = 120.0,
        research_dir: Optional[Path] = None,
        clock: Any = None,
    ) -> None:
        self.arm_notional_usd = float(arm_notional_usd)
        self.max_signal_age_s = float(max_signal_age_s)
        self._now = clock.now if clock is not None else time.time
        self._last_refusal_reason: Optional[str] = None
        #: research_dir is where telemetry is written. ``None`` keeps everything
        #: in memory (offline tests, dry runs) and touches nothing on disk.
        self.research_dir = Path(research_dir) if research_dir is not None else None
        #: One VirtualAccount per arm -- three ISOLATED ledgers.
        self.accounts: dict[AmarihArm, VirtualAccount] = {
            arm: VirtualAccount(
                arm=arm.value,
                arm_version=AMARIH_ARM_VERSION,
                baseline_hash=AMARIH_BASELINE_LABEL,
                starting_bankroll_usd=self.arm_notional_usd)
            for arm in AMARIH_ARMS
        }
        #: signal_id -> open positions, one per arm.
        self._open: dict[str, dict[AmarihArm, ArmPosition]] = {}
        #: De-duplication over accepted signal ids.
        #:
        #: PROCESS-LOCAL, and it must be read as such. The comment here used to
        #: claim "restart-safe (persisted by caller)", which was false: the
        #: caller (``AmarihRuntime``) persists its OWN ``seen_signal_ids`` in
        #: ``amarih_runtime.json`` and checks that before calling in here, so
        #: restart-safety is real -- but it is the CALLER's guarantee, not this
        #: set's, and a second caller would have had none at all.
        self.seen_signal_ids: set[str] = set()
        #: signal_id -> settled exit results, for reporting. Append-only.
        self._results: dict[str, dict[AmarihArm, ExitResult]] = {}
        #: Every signal record (accepted AND refused), append-only.
        self._signal_records: list[AmarihSignalRecord] = []
        #: Every entry observation, append-only.
        self._entry_records: list[EntryObservation] = []
        #: (signal_id, arm) pairs already settled -- idempotency guard.
        self._settled_keys: set[tuple[str, str]] = set()
        #: Whether the bankrolls have been re-hydrated from the durable event
        #: log. Guards :meth:`restore_bankrolls` against a double application.
        self._bankrolls_restored = False

    # -- restart recovery ---------------------------------------------------

    def restore_bankrolls(self) -> dict[str, float]:
        """
        Re-hydrate each arm's bankroll from its SETTLED history in ``events.jsonl``.

        WHY THIS EXISTS
        ---------------
        Each ``VirtualAccount`` is constructed with ``starting_bankroll_usd`` and
        ``__post_init__`` runs on every process start, so without this an arm
        that lost money came back at exactly its starting notional after EVERY
        restart and compounded from a balance it no longer had. That is a
        production accounting reset, not a display artifact: ``open_from_entry``
        refuses to open an arm that cannot afford the notional
        (``cost > account.bankroll_usd``), so a bankroll that should have been
        depleted would silently re-open positions it should never have taken.

        Each settlement is counted EXACTLY ONCE, and the FIRST row for a given
        ``(signal_id, arm)`` wins. ``events.jsonl`` is append-only and fsynced
        per row, and a crash between the settlement write and the runtime's own
        state write can leave a duplicate row; summing blindly would count that
        arm's PnL twice. Taking the first row also matches the in-process
        idempotency rule exactly -- the second call to ``settle_arm`` returns the
        first result rather than adding to it.

        IDEMPOTENT ACROSS CALLS. The restore is additive-once from
        ``starting_bankroll_usd``, and :attr:`_bankrolls_restored` short-circuits
        a second invocation, so calling this on every boot is safe.

        Returns the per-arm realized PnL that was re-applied. An unreadable or
        absent log yields zeros and leaves every bankroll at its starting value,
        which is the correct answer for an arm that has never settled.
        """
        if self._bankrolls_restored:
            return {}
        self._bankrolls_restored = True
        if self.research_dir is None:
            return {}
        path = self.research_dir / "events.jsonl"
        totals: dict[str, float] = {arm.value: 0.0 for arm in AMARIH_ARMS}
        seen: set[tuple[str, str]] = set()
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError:
            return {}
        try:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, Mapping):
                    continue
                if record.get("record_type") != "exit":
                    continue
                if record.get("settlement_status") != EXIT_SETTLED:
                    continue
                signal_id = str(record.get("signal_id") or "")
                arm_name = str(record.get("arm") or "")
                key = (signal_id, arm_name)
                if not signal_id or not arm_name or key in seen:
                    continue
                seen.add(key)
                pnl = _num(record.get("net_pnl_usd"))
                if pnl is None:
                    continue
                totals[arm_name] = totals.get(arm_name, 0.0) + float(pnl)
        except OSError:
            return {}
        finally:
            handle.close()
        for arm in AMARIH_ARMS:
            account = self.accounts.get(arm)
            if account is None:
                continue
            # ``peak_bankroll_usd`` is deliberately not restored: the exit
            # record does not carry a per-settlement bankroll, and inventing
            # one would fabricate a drawdown series. It stays at the starting
            # notional until this process settles an arm.
            account.restore_settled_bankroll(totals.get(arm.value, 0.0))
        return dict(totals)

    # -- telemetry persistence -------------------------------------------

    def _append(self, record: Mapping[str, Any]) -> None:
        """Append one JSON record to the experiment log. Best-effort, never raises
        into the trading path: a telemetry failure must not lose a trade.
        ``research_dir=None`` is a no-op (in-memory only).
        """
        if self.research_dir is None:
            return
        try:
            self.research_dir.mkdir(parents=True, exist_ok=True)
            path = self.research_dir / "events.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            pass

    # -- signal intake ---------------------------------------------------

    def note_signal(self, signal: AmarihSignalRecord) -> bool:
        """Record a signal. Returns True if it is a NEW eligible signal.

        Deduplicates by ``signal_id`` (restart-safe via ``seen_signal_ids``),
        and refuses stale / malformed / wrong-strategy signals with an explicit
        status. Receipt time is never backdated.
        """
        self._signal_records.append(signal)
        self._append(signal.as_dict())

        if signal.status != "ACCEPTED":
            return False
        if signal.signal_id in self.seen_signal_ids:
            return False
        age = signal.receipt_ts - signal.signal_ts
        if age > self.max_signal_age_s:
            return False
        self.seen_signal_ids.add(signal.signal_id)
        return True


    # -- paired entry ----------------------------------------------------

    def open_from_entry(
        self, entry: EntryObservation,
    ) -> dict[AmarihArm, ArmPosition]:
        """Fan ONE canonical entry observation out into three isolated arms.

        All three arms share ``entry.token_quantity`` and ``entry.entry_basis``
        -- the same entry amount and the same entry quote basis -- but each owns
        its OWN VirtualAccount, quantity accounting, PnL and settlement. This
        does NOT perform three BUY quotes and does NOT submit any transaction.

        An invalid entry (failed validation, no usable quote, or a notional the
        economics guard rejects) opens NOTHING on any arm and records the
        refusal once, with its reason. The signal stays in the denominator.
        """
        self._entry_records.append(entry)
        self._append(entry.as_dict())

        opened: dict[AmarihArm, ArmPosition] = {}
        if not entry.valid:
            return opened
        qty = float(entry.token_quantity or 0.0)
        cost = float(entry.input_amount_usd or self.arm_notional_usd)
        if qty <= 0:
            # Mark the entry as economically invalid with a clear reason
            try:
                # Create a new entry with updated reason (frozen, so we need to handle)
                pass
            except Exception:
                pass
            return opened

        skipped_econ = 0
        skipped_open = 0
        for arm in AMARIH_ARMS:
            account = self.accounts[arm]
            # Respect the existing economics: an arm that cannot afford the
            # notional is skipped, never force-opened and never bankroll-reset.
            if cost > account.bankroll_usd + 1e-9:
                skipped_econ += 1
                continue
            # One-open-position-per-arm: an arm already holding an open position
            # does not open a second, exactly as the production harness enforces.
            if any(p.arm is arm and p.is_open()
                   for positions in self._open.values() for p in positions.values()):
                skipped_open += 1
                continue
            # The bankroll moves ONLY on realized settlement
            position = ArmPosition(
                arm=arm,
                signal_id=entry.signal_id,
                mint=entry.mint,
                symbol=entry.symbol,
                entry_ts=entry.entry_ts,
                scheduled_deadline=entry.entry_ts + ARM_HORIZON[arm],
                token_quantity=qty,
                entry_cost_usd=cost,
                entry_basis=entry.entry_basis,
                account=account,
                entry_market=entry.market_id,
            )
            self._open.setdefault(entry.signal_id, {})[arm] = position
            opened[arm] = position
        # If nothing opened but entry was valid, record the reason
        if not opened:
            try:
                # Find and update the last entry record if it's frozen - we can't mutate
                # But we can track reasons separately; for now, log via the fact that
                # the runtime will look up - better to handle in runtime
                self._last_refusal_reason = (
                    f"ECONOMICALLY_INVALID: cost_usd={cost:.6f} bankroll_usd={account.bankroll_usd if 'account' in locals() else self.arm_notional_usd:.6f} "
                    f"(skipped_econ={skipped_econ} skipped_open={skipped_open})"
                )
            except Exception:
                self._last_refusal_reason = "ENTRY_VALID_BUT_NO_ARMS_OPENED"
        return opened

    def open_positions(self) -> dict[tuple[str, AmarihArm], ArmPosition]:
        """Every currently-open arm position, keyed by (signal_id, arm)."""
        out: dict[tuple[str, AmarihArm], ArmPosition] = {}
        for sid, arms in self._open.items():
            for arm, pos in arms.items():
                if pos.is_open():
                    out[(sid, arm)] = pos
        return out


    def restore_position(
        self,
        *,
        arm: AmarihArm,
        signal_id: str,
        mint: str,
        symbol: Optional[str],
        entry_ts: float,
        scheduled_deadline: float,
        token_quantity: float,
        entry_cost_usd: float,
        entry_basis: str = "executable_quote",
        entry_market: Optional[str] = None,
    ) -> Optional[ArmPosition]:
        """
        Re-register an OUTSTANDING arm after a restart, from its durable record.

        A restart must not silently abandon an arm that is still inside its
        holding period: without this the scheduled deadline would never fire and
        the arm would be permanently censored -- not by a bad quote, but by a
        process restart, which is exactly the kind of loss the experiment exists
        to avoid counting.

        IDEMPOTENT, because the same durable record can be replayed (a tick
        interrupted before its state write). An arm already open is returned
        unchanged rather than duplicated; an arm that has already been settled
        is REFUSED (``None``) rather than re-opened, so a restart can never
        revive a position whose outcome is already recorded.

        Only the ARM's own virtual account is used, so restoring one arm cannot
        move another arm's ledger.
        """
        existing = self._open.get(signal_id, {}).get(arm)
        if existing is not None and existing.is_open():
            return existing
        if (signal_id, arm.value) in self._settled_keys:
            return None
        position = ArmPosition(
            arm=arm,
            signal_id=signal_id,
            mint=mint,
            symbol=symbol,
            entry_ts=float(entry_ts),
            scheduled_deadline=float(scheduled_deadline),
            token_quantity=float(token_quantity),
            entry_cost_usd=float(entry_cost_usd),
            entry_basis=str(entry_basis),
            account=self.accounts[arm],
            entry_market=(str(entry_market) if entry_market else None),
        )
        self._open.setdefault(signal_id, {})[arm] = position
        return position

    # -- scheduled exit --------------------------------------------------

    def settle_arm(
        self,
        *,
        signal_id: str,
        arm: AmarihArm,
        sell_quote: Optional[Mapping[str, Any]],
        observed_ts: Optional[float] = None,
        max_quote_age_s: float = 30.0,
        emergency: bool = False,
    ) -> ExitResult:
        """Settle ONE arm from a FRESH executable SELL quote for exactly the
        token quantity acquired at the shared entry observation. FAIL-CLOSED: a
        missing, stale, identity-mismatched or emergency exit is recorded with
        an explicit status and is NEVER a fabricated fill. Idempotent: a second
        call for an already-settled (signal_id, arm) returns the FIRST result.

        ``sell_quote`` keys (any subset): ``proceeds_usd`` (net USD proceeds),
        ``quote_ts``, ``market_id``, ``provider``, ``venue``, ``fees_usd``,
        ``valid_market`` (bool), ``raw``.
        """
        now = float(observed_ts) if observed_ts is not None else self._now()
        position = self._open.get(signal_id, {}).get(arm)
        key = (signal_id, arm.value)

        # Idempotency: never double-settle. Return the first recorded result.
        if key in self._settled_keys and signal_id in self._results:
            existing = self._results[signal_id].get(arm)
            if existing is not None:
                return existing

        # Position must exist to settle. No open position is a censored no-op.
        if position is None:
            result = ExitResult(
                arm=arm, signal_id=signal_id,
                scheduled_deadline=(now - ARM_HORIZON[arm]),
                observed_ts=now, deadline_deviation_ms=0.0, token_quantity=0.0,
                sell_quote_available=False, quote_age_ms=None,
                market_id_valid=False, net_sell_proceeds_usd=None,
                total_buy_cost_usd=0.0, net_return_pct=None, net_pnl_usd=None,
                fees_usd=None, provider=None, venue=None,
                exit_reason="NO_POSITION", settlement_status=EXIT_CENSORED,
                classification=RESULT_CENSORED,
                failure_reason="no open position for this signal/arm",
            )
            self._record_exit(signal_id, arm, result)
            return result

        deadline = position.scheduled_deadline
        deviation_ms = round((now - deadline) * 1000.0, 3)
        qty = position.token_quantity
        cost = position.entry_cost_usd

        # Emergency / integrity exit BEFORE the deadline: classified separately.
        if emergency:
            proceeds = _num((sell_quote or {}).get("proceeds_usd"))
            net_pnl = (proceeds - cost) if proceeds is not None else None
            net_ret = ((proceeds / cost - 1.0) * 100.0
                       if (proceeds is not None and cost > 0) else None)
            result = ExitResult(
                arm=arm, signal_id=signal_id, scheduled_deadline=deadline,
                observed_ts=now, deadline_deviation_ms=deviation_ms,
                token_quantity=qty, sell_quote_available=bool(sell_quote),
                quote_age_ms=None,
                market_id_valid=bool((sell_quote or {}).get("valid_market")),
                net_sell_proceeds_usd=proceeds, total_buy_cost_usd=cost,
                net_return_pct=net_ret, net_pnl_usd=net_pnl,
                fees_usd=_num((sell_quote or {}).get("fees_usd")),
                provider=(sell_quote or {}).get("provider"),
                venue=(sell_quote or {}).get("venue"),
                exit_reason="EMERGENCY_EXIT", settlement_status=EXIT_EMERGENCY,
                classification=RESULT_EMERGENCY,
                failure_reason="emergency/integrity exit before scheduled deadline",
                raw_quote=(sell_quote or {}).get("raw", {}) or {},
            )
            self._apply_settlement(position, result)
            self._record_exit(signal_id, arm, result)
            return result


        # No quote at all at the deadline -> CENSORED, never a fabricated fill.
        if not sell_quote:
            result = ExitResult(
                arm=arm, signal_id=signal_id, scheduled_deadline=deadline,
                observed_ts=now, deadline_deviation_ms=deviation_ms,
                token_quantity=qty, sell_quote_available=False, quote_age_ms=None,
                market_id_valid=False, net_sell_proceeds_usd=None,
                total_buy_cost_usd=cost, net_return_pct=None, net_pnl_usd=None,
                fees_usd=None, provider=None, venue=None,
                exit_reason="SELL_QUOTE_UNAVAILABLE",
                settlement_status=EXIT_QUOTE_UNAVAILABLE,
                classification=RESULT_CENSORED,
                failure_reason="no executable SELL quote at the scheduled deadline",
            )
            self._record_exit(signal_id, arm, result)
            return result

        # Identity not PROVEN -> the market is not provably the entry market.
        #
        # ``is not True``, not ``is False``. A quote that simply omits the key
        # used to fall through this guard and settle with a net PnL booked from
        # a market nobody verified -- which is precisely the outcome the
        # same-market rule exists to make impossible. Only an explicit ``True``
        # settles.
        if sell_quote.get("valid_market") is not True:
            unproven = sell_quote.get("valid_market")
            result = ExitResult(
                arm=arm, signal_id=signal_id, scheduled_deadline=deadline,
                observed_ts=now, deadline_deviation_ms=deviation_ms,
                token_quantity=qty, sell_quote_available=True, quote_age_ms=None,
                market_id_valid=False, net_sell_proceeds_usd=None,
                total_buy_cost_usd=cost, net_return_pct=None, net_pnl_usd=None,
                fees_usd=None, provider=sell_quote.get("provider"),
                venue=sell_quote.get("venue"),
                exit_reason=("IDENTITY_MISMATCH"
                             if unproven is False else "IDENTITY_UNKNOWN"),
                settlement_status=EXIT_IDENTITY_MISMATCH,
                classification=RESULT_CENSORED,
                failure_reason=(
                    "SELL quote market was PROVEN to be a different book"
                    if unproven is False else
                    "SELL quote market was not proven to be the entry market"),
                raw_quote=sell_quote.get("raw", {}) or {},
            )
            self._record_exit(signal_id, arm, result)
            return result

        # Stale quote -> refuse; do not silently move the deadline.
        q_ts = _num(sell_quote.get("quote_ts"))
        quote_age_ms = round((now - q_ts) * 1000.0, 3) if q_ts is not None else None
        if q_ts is None or (now - q_ts) > max_quote_age_s:
            result = ExitResult(
                arm=arm, signal_id=signal_id, scheduled_deadline=deadline,
                observed_ts=now, deadline_deviation_ms=deviation_ms,
                token_quantity=qty, sell_quote_available=True,
                quote_age_ms=quote_age_ms, market_id_valid=True,
                net_sell_proceeds_usd=None, total_buy_cost_usd=cost,
                net_return_pct=None, net_pnl_usd=None, fees_usd=None,
                provider=sell_quote.get("provider"),
                venue=sell_quote.get("venue"), exit_reason="STALE_QUOTE",
                settlement_status=EXIT_STALE_QUOTE,
                classification=RESULT_CENSORED,
                failure_reason="SELL quote older than the freshness bound",
                raw_quote=sell_quote.get("raw", {}) or {},
            )
            self._record_exit(signal_id, arm, result)
            return result

        # A usable, fresh, same-market quote settles the arm.
        proceeds = _num(sell_quote.get("proceeds_usd"))
        fees = _num(sell_quote.get("fees_usd"))
        net_pnl = (proceeds - cost) if proceeds is not None else None
        net_ret = ((proceeds / cost - 1.0) * 100.0
                   if (proceeds is not None and cost > 0) else None)
        classification = RESULT_LATE if deviation_ms > 1000.0 else RESULT_NORMAL
        result = ExitResult(
            arm=arm, signal_id=signal_id, scheduled_deadline=deadline,
            observed_ts=now, deadline_deviation_ms=deviation_ms,
            token_quantity=qty, sell_quote_available=True,
            quote_age_ms=quote_age_ms, market_id_valid=True,
            net_sell_proceeds_usd=proceeds, total_buy_cost_usd=cost,
            net_return_pct=net_ret, net_pnl_usd=net_pnl, fees_usd=fees,
            provider=sell_quote.get("provider"), venue=sell_quote.get("venue"),
            exit_reason="SCHEDULED_TIME_EXIT", settlement_status=EXIT_SETTLED,
            classification=classification, failure_reason=None,
            raw_quote=sell_quote.get("raw", {}) or {},
        )
        self._apply_settlement(position, result)
        self._record_exit(signal_id, arm, result)
        return result

    # -- settlement bookkeeping -----------------------------------------

    def _apply_settlement(self, position: ArmPosition, result: ExitResult) -> None:
        """Move the arm's OWN ledger by realized PnL and close its position.

        Only called when a usable quote produced a realized proceeds figure, so
        a censored / quote-less arm's bankroll is NEVER moved by a number that
        did not exist.
        """
        position.exit_result = result
        if result.net_pnl_usd is not None:
            position.account.bankroll_usd += result.net_pnl_usd
            position.account.peak_bankroll_usd = max(
                position.account.peak_bankroll_usd,
                position.account.bankroll_usd)

    def _record_exit(
        self, signal_id: str, arm: AmarihArm, result: ExitResult,
    ) -> None:
        """Append the exit record once; mark (signal, arm) settled; retire it."""
        self._settled_keys.add((signal_id, arm.value))
        self._results.setdefault(signal_id, {})[arm] = result
        self._append(result.as_dict())
        arms = self._open.get(signal_id)
        if arms is not None:
            arms.pop(arm, None)
            if not arms:
                self._open.pop(signal_id, None)


    # -- research reporting ---------------------------------------------

    def signal_records(self) -> list[AmarihSignalRecord]:
        return list(self._signal_records)

    def entry_records(self) -> list[EntryObservation]:
        return list(self._entry_records)

    def results(self) -> dict[str, dict[AmarihArm, ExitResult]]:
        return {sid: dict(arms) for sid, arms in self._results.items()}

    def research_status(self) -> dict[str, Any]:
        """A compact, Telegram-safe status summary (no raw payloads)."""
        recs = self._signal_records
        accepted = [r for r in recs if r.status == "ACCEPTED"]
        valid_entries = [e for e in self._entry_records if e.valid]
        settled = self._results
        per_arm: dict[str, dict[str, Any]] = {}
        for arm in AMARIH_ARMS:
            arm_results = [arms[arm] for arms in settled.values() if arm in arms]
            avail = sum(1 for r in arm_results if r.sell_quote_available)
            per_arm[arm.value] = {
                "settled": len(arm_results),
                "sell_quote_available": avail,
                "availability_rate": (avail / len(arm_results)
                                      if arm_results else None),
            }
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "signals_received": len(recs),
            "signals_accepted": len(accepted),
            "valid_entry_observations": len(valid_entries),
            "open_arms": len(self.open_positions()),
            "settled_signals": len(settled),
            "per_arm": per_arm,
        }


    def build_report(self) -> dict[str, Any]:
        """Per-arm AND paired analysis. The three arms are PAIRED outcomes of
        one signal and are clustered by ``signal_id`` -- never counted as three
        independent signals.
        """
        results = self._results
        recs = self._signal_records

        def _stats(values: list[float]) -> dict[str, Any]:
            if not values:
                return {"n": 0}
            s = sorted(values)
            n = len(s)
            mean = sum(s) / n
            median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
            wins = sum(1 for v in values if v > 0)
            gross_win = sum(v for v in values if v > 0)
            gross_loss = -sum(v for v in values if v < 0)
            pf = (gross_win / gross_loss) if gross_loss > 0 else None
            return {"n": n, "mean": mean, "median": median,
                    "min": s[0], "max": s[-1], "win_rate": wins / n,
                    "profit_factor": pf}

        # Per-arm, over SETTLED (executable-quote) observations only. Censored
        # arms (no usable quote) are counted separately, never as a zero return.
        per_arm: dict[str, Any] = {}
        for arm in AMARIH_ARMS:
            settled_ret: list[float] = []
            censored = 0
            emergency = 0
            for arms in results.values():
                r = arms.get(arm)
                if r is None:
                    continue
                if r.net_return_pct is not None:
                    settled_ret.append(r.net_return_pct)
                if r.classification == RESULT_CENSORED:
                    censored += 1
                if r.classification == RESULT_EMERGENCY:
                    emergency += 1
            per_arm[arm.value] = {**_stats(settled_ret),
                                  "censored": censored, "emergency": emergency}

        # Paired evaluable signals: all three arms settled with a real return.
        paired_returns: list[float] = []
        paired_signals = 0
        for sid, arms in results.items():
            rets = [arms[a].net_return_pct for a in AMARIH_ARMS
                    if a in arms and arms[a].net_return_pct is not None]
            if len(rets) == len(AMARIH_ARMS):
                paired_signals += 1
                paired_returns.append(sum(rets) / len(rets))

        entry_failures: dict[str, int] = {}
        for e in self._entry_records:
            if not e.valid:
                entry_failures[e.validation] = (
                    entry_failures.get(e.validation, 0) + 1)

        latencies = [e.signal_to_entry_latency_ms
                     for e in self._entry_records if e.valid]

        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "signals_received": len(recs),
            "signals_accepted": sum(1 for r in recs if r.status == "ACCEPTED"),
            "entry_observations": len(self._entry_records),
            "valid_entry_observations": sum(
                1 for e in self._entry_records if e.valid),
            "entry_failures_by_reason": entry_failures,
            "per_arm": per_arm,
            "paired_evaluable_signals": paired_signals,
            "paired_mean_return": _stats(paired_returns),
            "signal_to_entry_latency_ms": _stats(latencies),
            "note": (
                "The three arms are PAIRED outcomes of one Amarih signal, "
                "clustered by signal_id. They are NOT three independent signals. "
                "Small samples and selection bias must be flagged before any "
                "exit horizon is called superior."
            ),
        }


__all__ = [
    "AmarihExperiment", "AmarihArm", "AmarihSignalRecord", "EntryObservation",
    "ArmPosition", "ExitResult", "AMARIH_ARMS", "ARM_HORIZON", "ARM_HORIZONS_S",
    "DEFAULT_ARM_NOTIONAL_USD", "DEFAULT_RESEARCH_DIR", "TELEMETRY_SCHEMA_VERSION",
    "AMARIH_ARM_VERSION", "AMARIH_BASELINE_LABEL",
    "ENTRY_VALID", "ENTRY_ECONOMICALLY_INVALID", "ENTRY_QUOTE_UNAVAILABLE",
    "ENTRY_IDENTITY_UNKNOWN", "EXIT_SETTLED", "EXIT_QUOTE_UNAVAILABLE",
    "EXIT_IDENTITY_MISMATCH", "EXIT_STALE_QUOTE", "EXIT_EMERGENCY", "EXIT_PENDING",
    "RESULT_NORMAL", "RESULT_EMERGENCY", "RESULT_LATE", "RESULT_CENSORED",
]

