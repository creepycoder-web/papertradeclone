"""
Entry-latency accounting and a fail-closed stale-entry gate.

WHY
---
A strategy validated on a ~2s median decision path degrades when the same
signal is acted on 34s later. Measured from the live ledger:

    analysis latency   p50 2,150ms   p90 6,653   p95 34,141   max 958,498
    signal -> fill     p50 3,354ms   p90 5,189   p95  9,195   max 304,916

The medians are fine, which is exactly why a median-only report hid this. The
tail is the problem, and the p95 clustering at ~34s points at bounded timeouts
stacking (market HTTP 8s + alert settle 2s + NAR 12s+2s), not at a smooth
distribution.

THE BUDGET IS NOT INVENTED HERE
------------------------------
``market.stale_after_ms`` (15s) already exists in the project and was only ever
used as a freshness LABEL on a quote. It is reused verbatim as the entry budget.
A new constant would be a second, conflicting definition of "fresh".

THE GATE
--------
Before an entry is accepted, the age of the signal is measured. If it exceeds
the budget the entry is REJECTED as a data-quality failure, not executed late.
Turning a latency problem into a strategy loss is the failure mode this prevents.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

#: Rejection reasons. Named for WHAT failed, so the operator can tell a stale
#: signal from a stale quote from a slow pipeline.
STALE_SIGNAL = "STALE_SIGNAL"
STALE_MARKET_DATA = "STALE_MARKET_DATA"
STALE_QUOTE = "STALE_QUOTE"
ENTRY_LATENCY_EXCEEDED = "ENTRY_LATENCY_EXCEEDED"

# ---------------------------------------------------------------------------
# execution outcomes
# ---------------------------------------------------------------------------
#
# A BUY DECISION IS NOT A FILL. The 2026-09-30 incident was un-diagnosable
# precisely because the two shared one word on one screen: the dashboard said
# `decision=BUY` next to `quote=-` and `fill=-`, which reads exactly like a
# broken executor. These names make the three states a reader must distinguish
# impossible to conflate, and every BUY path must end in exactly one of them.

#: The strategy authorized an entry. Says nothing about execution.
DECISION_BUY = "DECISION=BUY"
#: A paper fill happened and an open position exists.
PAPER_FILL_SUCCESS = "PAPER_FILL=SUCCESS"
#: A real fill happened, on-chain confirmed, with a signature.
REAL_FILL_SUCCESS = "REAL_FILL=SUCCESS"
#: Execution was refused or could not complete. ``reason`` says which.
BUY_BLOCKED = "BUY_BLOCKED"
#: A SELL decision that did not become a close. ``reason`` says which.
SELL_BLOCKED = "SELL_BLOCKED"
#: A real execution that was attempted and could not be completed.
#: Distinguishable from a pre-broadcast refusal.
REAL_EXECUTION_FAILED = "REAL_EXECUTION_FAILED"

#: Prefixes an operator can grep for. Deliberately exact, so a log search for
#: ``PAPER_FILL=SUCCESS`` cannot match a decision and vice versa.
OUTCOME_PREFIXES = (
    DECISION_BUY, PAPER_FILL_SUCCESS, REAL_FILL_SUCCESS, BUY_BLOCKED,
    "DECISION=SELL", SELL_BLOCKED, REAL_EXECUTION_FAILED,
)

#: Reasons a BUY can be refused. Every one of them must reach the operator, log
#: and journal; a refusal with no reason is the defect this replaces.
BLOCK_REASONS: dict[str, str] = {
    "mode_blocked": "the runtime refused new entries (PAUSED / KILLED / STOPPED)",
    "stale_signal": "the alert was older than the entry budget when the decision was taken",
    "stale_quote": "the quote aged out between decision and fill",
    "stale_market_data": "market data was unusable at fill time",
    "no_quote_at_fill": "no quote could be taken at fill time",
    "sizing_refused": "the sizer refused this size",
    "fill_refused": "the fill simulator refused the order",
    "execution_exception": "execution raised",
    "pretrade_refused": "a pre-trade gate refused the order (no execution attempted)",
    # Explicit execution-failure categories. These replace the single
    # ``execution_exception`` label at the REAL entry gate so the operator log
    # states the actual condition (e.g. ``BUY_BLOCKED=route_unavailable``).
    "route_unavailable": "no provider could construct a route for this token",
    "quote_invalid": "the venue quote was refused, malformed, stale or too small",
    "simulation_failed": "the transaction simulated and the chain refused it",
    "slippage_exceeded": "price impact or slippage breached the safety limit",
    "dex_program_error": "a DEX program emitted a custom error for this swap",
    "rpc_rejected": "the node rejected the transaction itself",
    "transaction_unknown": "the transaction's outcome is unknown; fail closed",
    "execution_failed": "the execution failed for a reason that did not classify",
    "cancelled": "the execution task was cancelled before it completed",
    "shutdown_during_execution": "shutdown began while the order was in flight",
}

#: Thai operator text. Kept beside the reasons so a new reason cannot ship
#: without one.
REJECT_REASON_TH: dict[str, str] = {
    STALE_SIGNAL: "สัญญาณเก่าเกินกำหนด (ข้อมูลล้าสมัย)",
    STALE_MARKET_DATA: "ข้อมูลตลาดเก่าเกินกำหนด",
    STALE_QUOTE: "ราคาคมชั่วคราวเก่าเกินกำหนด",
    ENTRY_LATENCY_EXCEEDED: "ความหน่วงเข้าสู่รายการเกินกำหนด",
}


# ---------------------------------------------------------------------------
# percentiles
# ---------------------------------------------------------------------------


def percentiles(samples: Sequence[Any]) -> dict[str, Any]:
    """
    p50/p90/p95/p99/max plus the sample count.

    The count is returned deliberately: a p95 over four samples is not a
    measurement, and a distribution with no max reported hides the outlier that
    actually matters here.
    """
    vals = sorted(float(v) for v in samples
                  if isinstance(v, (int, float)) and v == v)
    n = len(vals)
    if n == 0:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "p99": None,
                "max": None, "min": None, "mean": None}
    def at(q: float) -> float:
        return vals[min(n - 1, int(n * q / 100))]
    return {
        "n": n,
        "p50": at(50), "p90": at(90), "p95": at(95), "p99": at(99),
        "max": vals[-1], "min": vals[0],
        "mean": sum(vals) / n,
    }


def render_percentiles(title: str, stats: Mapping[str, Any]) -> str:
    if not stats.get("n"):
        return f"{title}: (no samples)"
    f = lambda k: ("n/a" if stats.get(k) is None else f"{stats[k]:,.0f}")
    return (f"{title}: n={stats['n']}  p50={f('p50')}  p90={f('p90')}  "
            f"p95={f('p95')}  p99={f('p99')}  max={f('max')}")


# ---------------------------------------------------------------------------
# stage timestamps
# ---------------------------------------------------------------------------


@dataclass
class LatencyTrace:
    """
    Wall-clock stamps for one candidate's journey, in milliseconds.

    Populated at each stage so a slow run can be attributed to a STAGE rather
    than guessed at. Any stamp may be None; ``as_stages`` then reports which
    stage is missing rather than inventing it.
    """

    signal_at: Optional[float] = None      # Telegram message received
    decision_at: Optional[float] = None    # L25 evaluation finished
    quote_at: Optional[float] = None       # quote taken
    submit_at: Optional[float] = None      # order construction started
    fill_at: Optional[float] = None        # fill confirmed
    budget_ms: float = 15_000.0

    # -- individual stages ------------------------------------------------

    @property
    def analysis_ms(self) -> Optional[float]:
        if self.signal_at is None or self.decision_at is None:
            return None
        return (self.decision_at - self.signal_at) * 1000.0

    @property
    def decision_to_quote_ms(self) -> Optional[float]:
        if self.decision_at is None or self.quote_at is None:
            return None
        return (self.quote_at - self.decision_at) * 1000.0

    @property
    def quote_age_at_submit_ms(self) -> Optional[float]:
        if self.quote_at is None or self.submit_at is None:
            return None
        return (self.submit_at - self.quote_at) * 1000.0

    @property
    def quote_age_at_fill_ms(self) -> Optional[float]:
        if self.quote_at is None or self.fill_at is None:
            return None
        return (self.fill_at - self.quote_at) * 1000.0

    @property
    def signal_to_fill_ms(self) -> Optional[float]:
        if self.signal_at is None or self.fill_at is None:
            return None
        return (self.fill_at - self.signal_at) * 1000.0

    # -- ordering ---------------------------------------------------------

    def ordering_violations(self) -> list[str]:
        """
        Stage stamps that run backwards.

        Reported rather than raised: a clock artefact must be visible, not fatal,
        but it must not be silently believed either.
        """
        order = [("signal_at", self.signal_at), ("decision_at", self.decision_at),
                 ("quote_at", self.quote_at), ("submit_at", self.submit_at),
                 ("fill_at", self.fill_at)]
        seen = [(n, v) for n, v in order if v is not None]
        bad = []
        for (n1, v1), (n2, v2) in zip(seen, seen[1:]):
            if v2 < v1:
                bad.append(f"{n2} ({v2}) precedes {n1} ({v1})")
        return bad

    # -- the gate ---------------------------------------------------------

    def stale_reason(self, now: Optional[float] = None) -> Optional[str]:
        """
        The rejection reason for this candidate, or ``None`` to proceed.

        Fail-CLOSED: if the signal timestamp is missing entirely, the entry is
        refused, because a candidate whose age cannot be established is exactly
        the case this gate exists for.

        Checked in order of specificity, so the reason names the thing that
        actually failed rather than a generic "too slow".
        """
        # 1. quote aged out between quote and submission/fill
        for age in (self.quote_age_at_submit_ms, self.quote_age_at_fill_ms):
            if age is not None and age > self.budget_ms:
                return STALE_QUOTE

        # 2. the market snapshot the decision was made on is no longer fresh
        if self.signal_at is not None and self.decision_at is not None:
            if (self.decision_at - self.signal_at) * 1000.0 > self.budget_ms:
                return STALE_SIGNAL

        # 3. the whole entry pipeline, measured to the fill
        s2f = self.signal_to_fill_ms
        if s2f is not None and s2f > self.budget_ms:
            return ENTRY_LATENCY_EXCEEDED

        # 4. the signal is older than the budget right NOW
        if self.signal_at is not None and now is not None:
            if (now - self.signal_at) * 1000.0 > self.budget_ms:
                return STALE_SIGNAL

        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis_ms": self.analysis_ms,
            "decision_to_quote_ms": self.decision_to_quote_ms,
            "quote_age_at_submit_ms": self.quote_age_at_submit_ms,
            "quote_age_at_fill_ms": self.quote_age_at_fill_ms,
            "signal_to_fill_ms": self.signal_to_fill_ms,
            "budget_ms": self.budget_ms,
            "ordering_violations": self.ordering_violations(),
            "stale_reason": self.stale_reason(),
        }


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


@dataclass
class StaleEntryGate:
    """
    Fail-closed entry gate over a candidate's age.

    Reuses the project's documented ``stale_after_ms`` budget rather than
    defining a second one. Configurable through the constructor because the
    architecture supports configuration, but it is NOT an L25 parameter: it
    governs data quality, and it can only ever REFUSE an entry, never authorise
    one. A stale candidate is refused regardless of how well it scored.
    """

    budget_ms: float = 15_000.0
    #: Refuse when the signal timestamp is missing. On by default: an
    #: unmeasurable age is not a fresh age.
    require_signal_timestamp: bool = True
    refused: list[str] = field(default_factory=list)

    def check(
        self,
        *,
        signal_at: Optional[float],
        now: float,
        decision_at: Optional[float] = None,
        quote_at: Optional[float] = None,
        submit_at: Optional[float] = None,
        fill_at: Optional[float] = None,
    ) -> Optional[str]:
        """``None`` to proceed, else the rejection reason."""
        if signal_at is None and self.require_signal_timestamp:
            self.refused.append(STALE_SIGNAL)
            return STALE_SIGNAL

        trace = LatencyTrace(
            signal_at=signal_at, decision_at=decision_at, quote_at=quote_at,
            submit_at=submit_at, fill_at=fill_at, budget_ms=self.budget_ms,
        )
        reason = trace.stale_reason(now=now)
        if reason is not None:
            self.refused.append(reason)
        return reason

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in self.refused:
            counts[r] = counts.get(r, 0) + 1
        return {"budget_ms": self.budget_ms, "refused_total": len(self.refused),
                "refused_by_reason": counts}


# ---------------------------------------------------------------------------
# trading vs notification latency
# ---------------------------------------------------------------------------

#: The critical-path stages. A trading latency is the span of these ONLY.
TRADING_STAGES = (
    "alert_received", "market_start", "market_end", "settle_start", "settle_end",
    "narrative_start", "narrative_end", "l25_start", "l25_end",
    "risk_start", "risk_end", "quote_start", "quote_end", "prepare_start",
    "prepare_end", "submit_start", "submit_end", "fill",
)

#: Presentation-only stages. NEVER folded into a trading latency.
NOTIFICATION_STAGES = ("notification_enqueue", "notification_sent")


class StageRecorder:
    """
    Per-candidate stage timestamps, split into trading and notification.

    The separation is the point. A previous measurement conflated the two, so a
    slow Telegram send could be read as slow trading - or, worse, a trade could
    be timed as if the notification were part of it. ``trading_ms`` is computed
    from TRADING_STAGES only; notification timing is reported separately and
    cannot enter it.

    WHY ``settle_*`` EXISTS
    -----------------------
    ``narrative_start`` used to be marked BEFORE the alert-settle sleep and before
    the second market read, and ``narrative_end`` after awaiting the narrative.
    The "narrative" window therefore contained ``alert_settle_s`` plus a whole
    extra provider fetch, which is how a 1.6s narrative was reported as 18.2s
    and the operator was sent looking for a narrative fault that did not exist.
    The settle window is now its own stage, so every number names the work it
    actually measures.

    ``blocked_reason`` records WHY a stage has no end stamp. A ``quote=-`` with no
    reason is indistinguishable from a display bug, which is the whole reason the
    2026-09-30 incident could not be told apart from a broken executor.
    """

    def __init__(self, budget_ms: float = 15_000.0) -> None:
        self.budget_ms = budget_ms
        self._stamps: dict[str, float] = {}
        self._order: list[str] = []
        self.blocked_reason: str | None = None
        #: The DETAIL behind the blocked reason (e.g. the per-source BUY
        #: candidate table). Persisted with the row so a recorded block is
        #: diagnosable without the log.
        self.blocked_detail: str = ""
        self.outcome: str | None = None

    def mark(self, stage: str, at: Optional[float] = None) -> None:
        if stage not in self._stamps:
            self._order.append(stage)
        self._stamps[stage] = float(at) if at is not None else 0.0

    def block(self, reason: str, *, outcome: str | None = None,
              detail: str = "") -> None:
        """
        Record WHY the pipeline stopped short of a stage.

        Called at the point execution is refused, so ``quote=-`` / ``fill=-``
        always carries the exact reason instead of rendering as a silent gap.
        ``detail`` carries the specifics (for a BUY, the per-source candidate
        failure table) so the operator surface never collapses to one token.
        """
        self.blocked_reason = str(reason)
        self.blocked_detail = str(detail or "")
        if outcome is not None:
            self.outcome = str(outcome)

    def finish(self, outcome: str) -> None:
        self.outcome = str(outcome)

    # -- trading ----------------------------------------------------------

    @property
    def trading_ms(self) -> Optional[float]:
        """alert_received -> fill. Trading only, by construction."""
        if "alert_received" not in self._stamps or "fill" not in self._stamps:
            return None
        span = (self._stamps["fill"] - self._stamps["alert_received"]) * 1000.0
        return max(0.0, span)

    @property
    def analysis_ms(self) -> Optional[float]:
        if "alert_received" not in self._stamps or "l25_end" not in self._stamps:
            return None
        return max(0.0, (self._stamps["l25_end"]
                         - self._stamps["alert_received"]) * 1000.0)

    @property
    def stamps(self) -> dict[str, float]:
        """The raw stage timestamps, for display and for the dashboard."""
        return dict(self._stamps)

    @property
    def signal_to_fill_ms(self) -> Optional[float]:
        """
        ALERT -> FILL: the number an operator means by "how slow was the trade".

        This deliberately starts at ``alert_received``, not ``submit_start``.
        The submit->fill span is the execution leg and is a different question;
        it is available as ``submit_to_fill_ms``.
        """
        start = self._stamps.get("alert_received")
        if start is None or "fill" not in self._stamps:
            return None
        return max(0.0, (self._stamps["fill"] - start) * 1000.0)

    @property
    def submit_to_fill_ms(self) -> Optional[float]:
        """The execution leg alone: submit_start -> fill."""
        if "submit_start" not in self._stamps or "fill" not in self._stamps:
            return None
        return max(0.0, (self._stamps["fill"]
                         - self._stamps["submit_start"]) * 1000.0)

    @property
    def narrative_ms(self) -> Optional[float]:
        """
        The NARRATIVE's own duration.

        Measured from the moment the narrative is awaited, so it cannot absorb
        the alert settle or the second market read. See the class docstring.
        """
        if "narrative_start" not in self._stamps or "narrative_end" not in self._stamps:
            return None
        return max(0.0, (self._stamps["narrative_end"]
                         - self._stamps["narrative_start"]) * 1000.0)

    @property
    def market_ms(self) -> Optional[float]:
        if "market_start" not in self._stamps or "market_end" not in self._stamps:
            return None
        return max(0.0, (self._stamps["market_end"]
                         - self._stamps["market_start"]) * 1000.0)

    @property
    def settle_ms(self) -> Optional[float]:
        """Alert settle plus the SECOND market read, as its own stage."""
        if "settle_start" not in self._stamps or "settle_end" not in self._stamps:
            return None
        return max(0.0, (self._stamps["settle_end"]
                         - self._stamps["settle_start"]) * 1000.0)

    # -- notification (reported, never merged) ----------------------------

    @property
    def notification_ms(self) -> Optional[float]:
        if "notification_enqueue" not in self._stamps \
                or "notification_sent" not in self._stamps:
            return None
        return max(0.0, (self._stamps["notification_sent"]
                         - self._stamps["notification_enqueue"]) * 1000.0)

    def ordering_violations(self) -> list[str]:
        seen = [(s, self._stamps[s]) for s in self._order]
        return [f"{b[0]} precedes {a[0]}"
                for a, b in zip(seen, seen[1:]) if b[1] < a[1]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "trading_ms": self.trading_ms,
            "analysis_ms": self.analysis_ms,
            "signal_to_fill_ms": self.signal_to_fill_ms,
            "narrative_ms": self.narrative_ms,
            "market_ms": self.market_ms,
            "settle_ms": self.settle_ms,
            "notification_ms": self.notification_ms,
            "stages_present": list(self._order),
            "blocked_reason": self.blocked_reason,
            "blocked_detail": self.blocked_detail,
            "outcome": self.outcome,
            "ordering_violations": self.ordering_violations(),
        }


#: The canonical stage order used for duration reconstruction. Not every stage
#: is present for every candidate; the store computes durations between
#: whatever consecutive stages WERE recorded, rather than inventing the gaps.
STAGE_ORDER: tuple[str, ...] = (
    "alert_received", "candidate_parsed", "market_start", "market_end",
    "settle_start", "settle_end",
    "narrative_start", "narrative_end", "l25_start", "l25_end",
    "risk_start", "risk_end", "quote_start", "quote_end",
    "prepare_start", "prepare_end", "submit_start", "submit_end", "fill",
)


def _finite_ms(value: Any) -> Optional[float]:
    """A finite float, or None. Rejects bool, NaN, inf and non-numbers."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def stage_durations(stamps: Mapping[str, Any]) -> dict[str, float]:
    """Milliseconds between consecutive present stages, in canonical order."""
    present = [
        (name, _finite_ms(stamps.get(name))) for name in STAGE_ORDER
        if _finite_ms(stamps.get(name)) is not None
    ]
    out: dict[str, float] = {}
    for (n1, v1), (n2, v2) in zip(present, present[1:]):
        out[f"{n1}->{n2}"] = max(0.0, (v2 - v1) * 1000.0)
    return out


class StageStore:
    """
    Append-only persistence for per-candidate stage timings.

    WHY
    ---
    ``StageRecorder`` is in-memory only, so its timings vanished on restart and
    the historical p50/p90/p95/p99/max could not be reconstructed. This writes
    one JSONL row per candidate into the project's existing ``var/`` structured
    store and can read them back.

    PROPERTIES
    ----------
    * **Backward compatible.** A row without a stage, a missing file and a
      legacy row are all fine; readers reconstruct only what is present.
    * **No fabrication.** No backfill and no historical rewrite; a stage that
      was not recorded is simply absent.
    * **Crash-safe on read.** A malformed line, a non-finite timestamp or a
      wrong-typed field is skipped rather than raising, so reporting can never
      be killed by one bad record.
    * **Off the network path.** A record is one small buffered append with no
      fsync - cheaper than the journal's own fsync-per-record - so it does not
      block the trading critical path. :meth:`flush` fsyncs on demand.
    * **Outcome alongside the timings.** ``outcome`` and ``blocked_reason`` are
      persisted next to the stages, so a candidate that never reached a quote is
      reconstructable afterwards rather than being a row with missing fields and
      no explanation.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.records_written = 0
        self.last_error: str | None = None

    def record(
        self,
        candidate_id: str | None,
        rec: "StageRecorder",
        *,
        mode: str | None = None,
        mint: str | None = None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        row = self._serialize(
            candidate_id, rec, mode=mode, mint=mint, symbol=symbol
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")
                fh.flush()
            self.records_written += 1
        except Exception as exc:  # noqa: BLE001 - persistence must not kill a trade
            self.last_error = f"{type(exc).__name__}: {exc}"
            from .logging_setup import log
            log("LATENCY", "stage persistence failed: %s", self.last_error)
        return row

    def _serialize(
        self,
        candidate_id: str | None,
        rec: "StageRecorder",
        *,
        mode: str | None,
        mint: str | None,
        symbol: str | None,
    ) -> dict[str, Any]:
        stamps: dict[str, float] = {}
        for name, value in rec.stamps.items():
            fv = _finite_ms(value)
            if fv is not None:
                stamps[name] = fv
        durations = stage_durations(stamps)
        start = stamps.get("alert_received")
        end = stamps.get("fill")
        trading_ms = (
            max(0.0, (end - start) * 1000.0)
            if start is not None and end is not None else None
        )
        return {
            "event": "stage_timing",
            "at": time.time(),
            "candidate_id": candidate_id,
            "mode": mode,
            "mint": mint,
            "symbol": symbol,
            "stages": stamps,
            "durations_ms": durations,
            "trading_ms": trading_ms,
            "outcome": rec.outcome,
            "blocked_reason": rec.blocked_reason,
            "blocked_detail": getattr(rec, "blocked_detail", ""),
            "stages_present": list(stamps),
        }

    @staticmethod
    def load(path: Path | str) -> list[dict[str, Any]]:
        """Read records back, tolerating anything malformed.

        A partial final line (hard kill), a non-dict row, a missing ``stages``
        map or a non-finite stamp are all skipped/cleaned here. This is the
        function that makes a restart-safe report possible without a crash on
        one bad write.
        """
        p = Path(path)
        out: list[dict[str, Any]] = []
        if not p.exists():
            return out
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            return out
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            raw = row.get("stages")
            if not isinstance(raw, Mapping):
                continue
            clean: dict[str, float] = {}
            for name, value in raw.items():
                fv = _finite_ms(value)
                if fv is not None:
                    clean[str(name)] = fv
            row["stages"] = clean
            row["durations_ms"] = stage_durations(clean)
            out.append(row)
        return out

    def flush(self) -> None:
        """Force any buffered bytes to disk. Best-effort."""
        try:
            if self.path.exists():
                with self.path.open("ab") as fh:
                    fh.flush()
                    import os
                    os.fsync(fh.fileno())
        except OSError:
            pass


class LatencyAccumulator:
    """Rolling percentiles, kept apart by pipeline so they cannot be summed."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {
            "analysis": [], "signal_to_fill": [], "narrative": [],
            "market": [], "settle": [], "trading": [], "notification": [],
        }
        #: Execution outcomes, keyed by the exact token the operator sees.
        self.outcomes: dict[str, int] = {}

    def record(self, rec: StageRecorder) -> None:
        for key, value in (
            ("analysis", rec.analysis_ms),
            ("signal_to_fill", rec.signal_to_fill_ms),
            ("narrative", rec.narrative_ms),
            ("market", rec.market_ms),
            ("settle", rec.settle_ms),
            ("trading", rec.trading_ms),
            # recorded separately; see notification_ms
            ("notification", rec.notification_ms),
        ):
            if isinstance(value, (int, float)):
                self.samples[key].append(float(value))

    def record_outcome(self, outcome: str) -> None:
        """One counter per execution outcome, so fills cannot hide among buys."""
        if outcome:
            self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1

    def stats(self) -> dict[str, Any]:
        return {k: percentiles(v) for k, v in self.samples.items() if v}

    def outcome_stats(self) -> dict[str, int]:
        return dict(self.outcomes)

    def trading_stats(self) -> dict[str, Any]:
        """Everything EXCEPT notification. The two are never combined."""
        out = {k: v for k, v in self.stats().items() if k != "notification"}
        return out
