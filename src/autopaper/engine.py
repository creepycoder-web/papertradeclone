"""
The trading loop. Moo Deng President -> V3-C -> PAPER or REAL.

    WAIT FOR ALERT
        -> PARSE
        -> T0 SNAPSHOT          (frozen, nothing can be backfilled)
        -> OBSERVE MARKET       (at decision time only)
        -> NAR LOOKUP           (null when unconfigured)
        -> ANALYZE
        -> V3-C GATE            (MC <= $20,000, SQ >= 0.55)
        -> BUY / WAIT / REJECT
        -> SIZE                 (one function: 10% at or below $50, else 12.5%)
        -> PRE-TRADE SAFETY     (REAL only; fail closed)
        -> FILL                 (paper simulator, or Solana/Jupiter)
        -> MONITOR
        -> HOLD / EXIT
        -> RECORD
        -> UPDATE ACCOUNT
        -> WAIT FOR NEXT ALERT

Two independent tasks run concurrently:

* the alert task (Telegram) - analysis and fills happen inline on the alert, so
  the latency between an alert and a decision is real rather than an artefact of
  batching
* the monitor task - a fixed-interval loop that marks open positions, runs the
  exit policy, and services the control channel (STOP / FLATTEN)

THE SINGLE MODE AUTHORITY
-------------------------
``self.state`` is a :class:`autopaper.modes.RuntimeState`. It is the only thing
that answers "is this bot allowed to open an entry, and against whose money".
The config cannot arm REAL, the environment cannot arm REAL, and nothing in this
module can arm REAL: only a confirmed Telegram operator action can, and that
arrives through :class:`autopaper.telegram.actions.ButtonHandler`.

Outbound calls when NOT armed for REAL are read-only market data and the
Telegram read stream, exactly as before. When the operator has armed REAL, the
only additional outbound call is the venue swap, and every one of them is
preceded by the full pre-trade safety sequence.
THE 2026-09-30 INCIDENT, IN ONE PLACE
--------------------------------------
A degraded candidate reported::

    data      = 16,604ms
    narrative = 18,269ms
    L25       =      2ms
    total     = 34,877ms

and the same candidate reported ``decision=BUY`` beside ``quote=-`` / ``fill=-``.

Three separate faults, none of them the others:

* **market 16.6s** - one provider call. ``httpx.Client(timeout=8.0)`` is not a
  wall-clock budget: the connect phase is charged ONCE PER RESOLVED ADDRESS, and
  both market hosts list two unroutable IPv6 addresses ahead of the reachable
  IPv4 ones. 2 x 8.0s + ~0.05s = 16.05s. See :mod:`autopaper.market.ratelimit`.
* **narrative "18.2s"** - a TELEMETRY fault, not a narrative fault. The bot
  answered in 1.6s (the ``NAR`` log line says so). ``narrative_start`` was marked
  above the alert settle and the SECOND market read, so the window contained
  ``alert_settle_s`` + a whole extra provider fetch. The stage is now its own
  window, and the narrative's own latency is measured inside ``NarService``.
* **BUY with no fill** - not a broken executor. The 34.9s analysis exceeded the
  15s entry budget, so :class:`autopaper.latency.StaleEntryGate` refused the entry
  as ``STALE_SIGNAL`` before an order was constructed. Correct behaviour,
  invisible presentation. Every BUY now ends in exactly one of
  ``DECISION=BUY`` / ``PAPER_FILL=SUCCESS`` / ``BUY_BLOCKED=<reason>``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import signal
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import dashboard, logging_setup
from .account import MarkedPosition, PaperAccount
from .analysis import DecisionEngine, build_features
from .clock import Clock, SystemClock, iso
from .config import Config, build_config
from .execution import FillInputs, simulate_fill
from .execution.intents import ExecutionIntentStore
from .execution.real import LiveSolanaBackend, RealExecutor
from .journal import Journal
from .exits import ExitDecision
from .latency import (
    BUY_BLOCKED,
    DECISION_BUY,
    PAPER_FILL_SUCCESS,
    REAL_EXECUTION_FAILED,
    REAL_FILL_SUCCESS,
    SELL_BLOCKED,
    STALE_MARKET_DATA,
    STALE_QUOTE,
    LatencyAccumulator,
    StageRecorder,
    StageStore,
    StaleEntryGate,
)
from .forward_validation import ForwardRecorder, load_or_create_deployment
from .ledger import EventType, Ledger, PaperAccountState, load_json, save_json_atomic
from .logging_setup import log
from .market import MarketDataService
from .memory import ResearchMemory
from .observations import store_for
from .models import (
    AlertSnapshot,
    DataState,
    Decision,
    DecisionReport,
    ExitStyle,
    Observation,
    Position,
    PositionStatus,
    Regime,
    Route,
    Trade,
    new_id,
)
from .modes import ModeError, RunState, RuntimeState, TradeMode
from .nar import NarService
from .real_account import RealAccount, RealFillRecord
from .safety import AGENT_ID, audit
from .sizing import (
    compute_position_size,
    describe_policy,
    tradeable as size_tradeable,
)
from .telegram.actions import ButtonHandler, OperatorActions
from .telegram.ui import StatusView, esc as _esc
from .stats import compute as compute_stats
from .telegram import ConnectionState, TelegramListener
from .telegram.signal_source import (
    FALLBACK, FALLBACK_ACTIVE, OPPORTUNITY_WINDOW_S, PRIMARY,
    PRIMARY_ACTIVE, SOURCE_AMARIH, SOURCE_AUTO, SOURCE_MOODENG,
    SOURCE_PAINAPE, SignalSourceState,
)
from . import l25, v3c


MODE_RUN = "RUN"
MODE_STOP = "STOP"


def _env_float(name: str, default: float) -> float:
    """Read a non-negative float from the environment, else the default.

    A malformed or negative value falls back rather than raising: an operator
    typo in an optional tuning variable must not stop the process from starting.
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if value >= 0 else float(default)
MODE_FLATTEN = "FLATTEN"

#: How many closed positions one CLOSED POSITIONS reply carries. A fixed recent
#: window, not an unbounded history: the ledger holds hundreds of rows and a
#: Telegram message cannot, and paging through them one button press per row
#: would be noise rather than inspection.
CLOSED_PAGE_SIZE = 8

#: The idempotency key for a real order that is already in flight. The engine
#: refuses a new entry while any unresolved intent exists, so a retry after an
#: ambiguous broadcast can never become a second transaction.
_BLOCKED_UNRESOLVED = "unresolved_execution"

#: How old a SOL/USD reference may be before BOTH PAPER sizing and REAL equity
#: valuation refuse to use it.
#:
#: The provider's own TTL governs its cache; this is the independent gate on the
#: CONSUMER side, so a value cannot be carried forward indefinitely just because
#: something still holds it. Two minutes is deliberately looser than the
#: provider's 60s cache TTL (which only decides when to re-fetch) and far tighter
#: than any drift that would matter for valuing a SOL balance. Fail-closed past
#: it: equity becomes unavailable and REAL entry sizing blocks, rather than a
#: stale number being presented as a measurement.
_SOL_USD_MAX_AGE_MS = 120_000.0

#: Status glyphs, duplicated here rather than imported from the UI module so the
#: engine does not depend on the rendering layer.
_OK = "\U0001F7E2"
_ERR = "\U0001F534"
_INFO = "\U0001F535"
#: Used for a state that is neither good nor fatal: it happened, it is not what
#: was expected, and the operator should look.
_WARN = "⚠️"

#: A blocked REAL exit notifies on the first occurrence, then every Nth. The
#: count shown in the message is always the true occurrence number, so the
#: cadence only controls how often the operator is interrupted, never what the
#: number means.
_SELL_BLOCK_REPEAT_EVERY = 10

#: Longest sanitized reason shown in a Telegram message.
_SANITIZED_REASON_MAX = 180


def _arm_close_failure_code(trade: Any) -> str:
    """
    The one classification code for a PAPER close that produced no fill.

    A trade whose ``failure`` is ``FailureReason.NONE`` -- or absent -- and which
    still carries no ``exit_fill`` is contradicting itself. It is NOT reported as
    ``NONE``: an operator reading "Errors: NONE" cannot act on it, and that is
    exactly what made this incident unreadable. It is reported as a fill that
    could not be constructed, which is the only honest description of a trade
    record with a success verdict and no fill on it.
    """
    from .paper_live.schema import (
        ARM_CLOSE_NO_FILL_CONTRADICTION as _NO_FILL,
    )
    from .paper_live.schema import FailureReason as _FR

    failure = getattr(trade, "failure", None)
    value = str(getattr(failure, "value", failure) or "").strip()
    if not value or value == _FR.NONE.value:
        return _NO_FILL
    return value


def _arm_close_failure_reason(trade: Any) -> str:
    """
    ``CODE: detail`` for a PAPER close that produced no fill.

    The code comes first and is never empty, so a reader who sees only the first
    field still knows the class of the failure. The detail is the provider's own
    refusal where one exists, and never the word ``NONE``.
    """
    code = _arm_close_failure_code(trade)
    detail = str(getattr(trade, "failure_detail", None) or "").strip()
    return f"{code}: {detail}" if detail else code


def _sanitize_reason(reason: Any) -> str:
    """
    Reduce a failure reason to one safe, single-line, bounded string.

    Reasons reach this code from RPC responses, venue errors and exception text,
    and those can contain a base64 transaction body, a signed URL, a key
    fragment, or a multi-line traceback. A status message must carry none of
    that, so: newlines collapse, the text is truncated, and anything that looks
    like a secret or an encoded payload is REDACTED rather than truncated --
    truncating a secret still leaks its prefix.
    """
    import re as _re

    text = "" if reason is None else str(reason)
    # Secrets and payloads go entirely, not partially.
    for pattern in (
        r"[A-Za-z0-9_\-]{40,}",            # keys, signatures, base64 blobs
        r"(?i)\b(?:api[_-]?key|token|secret|password|mnemonic|seed)\b\s*[:=]\s*\S+",
        r"(?i)https?://\S+",
    ):
        text = _re.sub(pattern, "[redacted]", text)
    text = " ".join(text.split())
    if len(text) > _SANITIZED_REASON_MAX:
        text = text[:_SANITIZED_REASON_MAX - 1].rstrip() + "…"
    return text or "unspecified"


def _close_text(symbol, return_pct, net_pnl_usd, *,
                mark_return_pct=None, show_mark_return: bool = False,
                entry_fill=None, exit_fill=None,
                mark_return_label: str = "Observed mark return (not executable)",
                ) -> str:
    """
    The close-order notification body.

    ``return_pct`` MUST be a REALIZED return derived from actual settlement on
    both legs. It is not optional knowledge, it is the contract of this
    parameter.

    WHY THIS FUNCTION IS NOT THE CODE IT WAS
    ---------------------------------------
    On 2026-10-02 a $GOAT REAL position was reported as::

        ✅ ปิดออเดอร์
        GOAT
        ผลลัพธ์: +71.56%  $+0.0016

    The dollar came from the chain. The percentage came from
    ``exit_mark / entry_mark - 1`` -- two DexScreener ``priceUsd`` observations,
    neither of which was executable. Two accounting systems on one line, and the
    flattering one was labelled "result".

    So: the percentage and the dollar are now required to come from the same
    settlement, and an OBSERVED MARK RETURN is never printed next to a realized
    figure unless an operator has explicitly asked for diagnostics -- and even
    then it is labelled as a mark observation, not as a result.

    ``mark_return_pct`` is accepted, not because it belongs on an operator's
    screen, but because an operator comparing the bot's output against an
    exchange's needs to see what the bot was looking at. It is hidden by default.
    """
    head = "✅ <b>ปิดออเดอร์</b>\n<b>" + _esc(str(symbol)[:16]) + "</b>"
    lines = [head]
    if isinstance(return_pct, (int, float)) and \
            isinstance(net_pnl_usd, (int, float)):
        lines.append(
            f"ผลลัพธ์: {return_pct:+.2f}%  ${net_pnl_usd:+.4f}")
    for label, value in (("Entry fill", entry_fill),
                         ("Exit fill", exit_fill)):
        if isinstance(value, str) and value:
            lines.append(f"{label}: {value}")
    if (show_mark_return and isinstance(mark_return_pct, (int, float))):
        lines.append(
            f"{mark_return_label}: {mark_return_pct:+.2f}%")
    if len(lines) == 1:
        return head
    return "\n".join(lines)


def _fmt_usd_price(value: Any) -> Optional[str]:
    """A per-whole-token USD price as an operator-readable string, or ``None``.

    ``None`` in, ``None`` out. A price that could not be established is never
    rendered as ``0`` -- a zero price is a missing value that once reached a
    journal wearing a measurement's name.
    """
    try:
        px = float(value)
    except (TypeError, ValueError):
        return None
    if not (px > 0) or px != px:
        return None
    return f"${px:.6g}"


def _connection_state_text(listener: Any) -> str:
    """
    The listener's connection state as operator-readable text.

    This used to be::

        listener.status.state.value if hasattr(listener.status.state, "value")
        else ("DISABLED" if listener else "?")

    ``ConnectionState`` is a class of STRING CONSTANTS, not an ``Enum``, so
    ``ListenerStatus.state`` is a ``str`` and ``hasattr(state, "value")`` is
    ALWAYS False. The expression therefore took the ``else`` branch for every
    possible state and the dashboard reported "channel DISABLED" permanently --
    including while the listener was ``CONNECTED`` and actively ingesting
    alerts. That is the worst kind of dashboard defect: it says a healthy service
    is down, so an operator debugging an empty-looking dashboard is sent looking
    for a Telegram fault that does not exist.

    The state is now read for what it is, and a genuinely absent listener is the
    only thing that reports DISABLED.
    """
    if not listener:
        return "?"
    state = getattr(getattr(listener, "status", None), "state", None)
    if state is None:
        return "?"
    return getattr(state, "value", state)   # Enum-tolerant, str-correct


def _last_event_row(decision: Any) -> dict[str, Any]:
    """
    Normalise a last-decision into a display row.

    Accepts either representation on purpose. The engine stores
    ``report.as_dict()``; the same report is available as an object at the call
    site, and the two must not be able to disagree about what the operator is
    shown. ``_field`` is the single place that knows the difference, so there is
    exactly one mapping to keep correct instead of one per call site.
    """
    if decision is None:
        return {}

    def _field(name: str, default: Any = None) -> Any:
        if isinstance(decision, Mapping):
            return decision.get(name, default)
        return getattr(decision, name, default)

    def _enum(value: Any) -> Any:
        return getattr(value, "value", value)

    def _flag_codes(raw: Any, severity: str | None = None) -> list[str]:
        out: list[str] = []
        for item in raw or ():
            code = _field_of(item, "code")
            sev = _field_of(item, "severity")
            if severity is not None and sev != severity:
                continue
            if code:
                out.append(str(code))
            elif isinstance(item, Mapping) and item.get("code"):
                out.append(str(item["code"]))
            else:
                out.append(str(item))
        return out

    decided_at = _field("decided_at")
    # THE BLOCKING FLAGS FIRST, THEN THE OTHERS.
    #
    # This was ``_flag_codes(...)[:2]`` -- the first two flags in declaration
    # order, severity ignored. Risk flags are appended in evaluation order, and
    # the informational ones (narrative provenance, an unanchored extension)
    # are written BEFORE the strategy gates. So on a refusal the operator was
    # shown the two noisiest notes and the reason it was actually refused was
    # truncated away. BOMBOCLAT read:
    #
    #     warn  extension_unanchored
    #     info  nar_narrative_only
    #     block v2floor_vol5_per_mc_below_floor   <- never shown
    #
    # and the dashboard reported "Reasons: extension_unanchored;
    # nar_narrative_only", naming two things that blocked nothing.
    #
    # The blocking codes are what an operator needs first; the rest are kept so
    # the context is still visible, and the count is unchanged at two so the
    # row cannot grow without bound.
    blocking = _flag_codes(_field("risk_flags"), severity="block")
    non_blocking = [c for c in _flag_codes(_field("risk_flags"))
                    if c not in blocking]
    reason = (blocking + non_blocking)[:2]

    # Format the timestamp only when it is a FINITE number. A NaN or infinity
    # is a real possibility in a timing field, and iso() raises on it -- which
    # is the same class of bug as the original crash: a display value that the
    # display cannot handle.
    at_iso: Any = decided_at
    if isinstance(decided_at, bool):
        at_iso = None
    elif isinstance(decided_at, (int, float)):
        at_iso = iso(decided_at) if math.isfinite(decided_at) else None

    return {
        "at_iso": at_iso,
        "symbol": _field("symbol") or _field("mint") or "",
        "decision": str(_enum(_field("decision")) or ""),
        "route": str(_enum(_field("route")) or ""),
        "confidence": _field("confidence"),
        "reason": "; ".join(reason),
        "latency_ms": _field("latency_ms"),
        "candidate_id": _field("candidate_id") or "",
    }


def _field_of(obj: Any, name: str) -> Any:
    """Read one field from either a mapping or an object."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _ms_text(end, start) -> str:
    """A short "123ms" for the pipeline row, or "-" when not reached."""
    if end is None or start is None:
        return "-"
    try:
        return f"{(end - start) * 1000.0:,.0f}ms"
    except TypeError:
        return "-"


def _equity_footer(equity_usd: float | None, cash_usd: float | None = None) -> str:
    """
    The equity/cash line appended to an order notification.

    Shows the account state at the moment the order was reported, so an
    operator reading a closed-order alert knows the balance it moved. An
    unavailable equity is stated as such and never rendered as $0.0000.
    """
    if equity_usd is None:
        return "\n\n<b>Equity: unavailable</b>"
    out = f"\n\n<b>Equity: ${equity_usd:,.4f}</b>"
    if cash_usd is not None:
        out += f"  |  Cash: ${cash_usd:,.4f}"
    return out


def _render_paper_buy(report, features, size) -> str:
    """The PAPER entry alert. Rendered by the same code the tests assert on."""
    from .telegram import ui

    liq = getattr(features, "liquidity_usd", None)
    mom = getattr(features, "momentum_30s_pct", None)
    return ui.render_paper_buy(
        symbol=report.symbol or "?",
        mint=report.mint or "",
        market_cap_usd=getattr(features, "market_cap_usd", None),
        signal_quality=float(getattr(features, "signal_quality", 0.0)),
        position_usd=size.position_usd,
        liquidity_usd=liq,
        momentum=("{:.1f}% / 30s".format(mom) if mom is not None else None),
        f_liq_to_mcp=l25.f_liq_to_mcap(
            liq, getattr(features, "market_cap_usd", None)
        ),
    )


def _render_strategy_reject(result, symbol: str) -> str:
    """
    The ACTIVE strategy's refusal alert, generated from the gate result itself.

    Rendered from the same result object the decision was made on, so the
    message cannot disagree with the journal.
    """
    from .telegram import ui

    return ui.render_strategy_reject(result, symbol=symbol)


def _render_real_buy(report, size, est_sol: float, sol_price) -> str:
    """The REAL entry alert, shown BEFORE the order is sent."""
    from .telegram import ui

    gate = getattr(report, "l25", None) or report.v3c
    return ui.render_real_buy(
        symbol=report.symbol or "?",
        mint=report.mint or "",
        market_cap_usd=(gate.market_cap_usd if gate else None),
        signal_quality=(float(gate.signal_quality) if gate else 0.0),
        position_usd=size.position_usd,
        est_cost_sol=est_sol,
        est_cost_usd=(size.position_usd if sol_price else None),
        f_liq_to_mcp=(
            float(gate.f_liq_to_mcap)
            if gate is not None and getattr(gate, "f_liq_to_mcap", None) is not None
            else None
        ),
    )


def _tp_sl_text(plan: Mapping[str, Any], entry: float | None,
                runner_mode: bool, runner_armed_at: float | None) -> str:
    """
    Derive the TP / SL LEVELS an open position is carrying, from the exit plan
    that was written BEFORE the trade existed.

    Display only. Nothing here can change an exit, tighten a stop or invent a
    target: the levels are read from ``position.exit_plan`` (the pre-declared
    plan) and from ``runner_mode``, and the exit policy still runs unchanged in
    :mod:`autopaper.exits`. The runner arm is shown because a runner's floor
    only exists once it has armed, and reporting a trail on an unarmed runner
    would describe a stop that is not there.
    """
    if not plan or not entry:
        return "not declared"
    parts: list[str] = []
    stop = plan.get("initial_stop_pct")
    if isinstance(stop, (int, float)):
        parts.append(f"SL {float(stop):+.1f}% @ {entry * (1 + float(stop) / 100.0):.3e}")
    target = plan.get("fixed_target_pct")
    if isinstance(target, (int, float)):
        parts.append(f"TP {float(target):+.1f}% @ {entry * (1 + float(target) / 100.0):.3e}")
    arm = plan.get("runner_arm_pct")
    trail = plan.get("runner_trail_pct")
    if isinstance(arm, (int, float)):
        parts.append(
            ("RUNNER armed" if runner_mode else f"runner arm {float(arm):+.1f}%")
            + (f" @ +{runner_armed_at - entry:+.3e} from entry"
               if runner_mode and runner_armed_at else "")
        )
    if isinstance(trail, (int, float)):
        parts.append(f"trail {float(trail):+.1f}% from peak")
    return "; ".join(parts) if parts else "not declared"


#: How long shutdown waits for an in-flight BUY to reach a definitive outcome.
#: Bounded: a stop must not hang on a wedged provider, and it must not silently
#: discard a successful fill either. Whatever is still pending after this is
#: cancelled and recorded as ``BUY_BLOCKED=shutdown_during_execution``.
EXECUTION_DRAIN_TIMEOUT_S = 5.0


class _BlockReport:
    """
    A minimal report-shaped carrier for ``_record_block``.

    Shutdown may need to record a refusal for an order whose ``DecisionReport``
    has already gone out of scope. Building this keeps ``_record_block`` on one
    ``candidate_id`` / ``mint`` / ``symbol`` contract instead of two.
    """

    __slots__ = ("candidate_id", "mint", "symbol", "decision")

    def __init__(self, candidate_id=None, mint=None, symbol=None):
        self.candidate_id = candidate_id
        self.mint = mint
        self.symbol = symbol
        self.decision = "BUY"


def _block_reason_for_stale(stale_reason: str | None) -> str:
    """
    Map a :mod:`autopaper.latency` gate reason onto the outcome vocabulary.

    The gate's reasons are precise (``STALE_SIGNAL`` vs ``STALE_QUOTE`` vs
    ``ENTRY_LATENCY_EXCEEDED``) and are preserved verbatim in the log, the
    ledger note and the journal payload. The outcome counter is coarser on
    purpose: an operator reading ``BUY_BLOCKED`` counts needs one bucket per
    KIND of refusal, not one per spelling.
    """
    if stale_reason == STALE_QUOTE:
        return "stale_quote"
    if stale_reason == STALE_MARKET_DATA:
        return "stale_market_data"
    if stale_reason:
        return "stale_signal"
    return "no_quote_at_fill"


def _block_reason_text(reason: str) -> str:
    """Human-readable text for a block reason. Never raises on a new reason."""
    from .latency import BLOCK_REASONS

    return BLOCK_REASONS.get(reason, reason)


def _is_beta_arm(arm: str) -> bool:
    """
    True when ``arm`` is one of the V2 FLOOR (BETA) experiment's three arms.

    Read from the arm registry rather than a string prefix, so a refusal can be
    attributed to the beta experiment without the reporting layer hard-coding
    the arm names a second time.
    """
    try:
        from .paper_live.arms import BETA_ARMS as _BETA, ResearchArm as _RA

        return _RA(str(arm)) in set(_BETA)
    except Exception:  # noqa: BLE001 - an unknown arm is simply not a beta arm
        return False


def ui_render_buy_blocked(*, symbol: str, mint: str, reason: str, detail: str,
                          mode, candidate_id: str | None = None,
                          strategy: str | None = None,
                          beta_arms=(), traded_arms=()) -> str:
    """
    The operator message for a BUY decision that did not become a position.

    Rendered from the same helpers the test suite asserts on, and it leads with
    ``BUY_BLOCKED=<reason>`` rather than a Thai phrase, because this is the
    message that has to be greppable in a log and unambiguous at a glance.

    ``beta_arms`` names the beta arms the V2 FLOOR experiment gate refused, and
    switches the message to a beta-gate rejection. Without it the renderer would
    fall back to the hardcoded ``ACTIVE_STRATEGY`` constant and report a beta
    refusal as a blocked L25 entry.
    """
    from .telegram import ui

    return ui.render_buy_blocked(
        symbol=symbol, mint=mint, reason=reason, detail=detail, mode=mode,
        reason_text=_block_reason_text(reason),
        candidate_id=candidate_id, strategy=strategy,
        beta_arms=beta_arms, traded_arms=traded_arms,
    )


@dataclass
class ControlCommand:
    command: str = ""
    issued_at: float = 0.0
    acked_at: float | None = None

    @property
    def pending(self) -> bool:
        return bool(self.command) and self.acked_at is None


# Monitor diagnostics counters
_MONITOR_DIAG = {
    "MONITOR_OK": 0,
    "MONITOR_RPC_ERROR": 0,
    "MONITOR_QUOTE_UNAVAILABLE": 0,
    "MONITOR_IDENTITY_UNKNOWN": 0,
    "MONITOR_IDENTITY_MISMATCH": 0,
    "MONITOR_SKIPPED_BUDGET": 0,
    "MONITOR_HTTP_ERROR": 0,
    "MONITOR_MARK_STALE": 0,
}


def _reset_monitor_diag():
    for k in _MONITOR_DIAG:
        _MONITOR_DIAG[k] = 0


def _get_monitor_diag() -> dict[str, int]:
    return dict(_MONITOR_DIAG)


def _inc_monitor_diag(key: str, n: int = 1):
    if key in _MONITOR_DIAG:
        _MONITOR_DIAG[key] += n


def _excursion_fields(trade: Mapping[str, Any],
                      position: Mapping[str, Any],
                      record: Optional[Mapping[str, Any]] = None
                      ) -> dict[str, Any]:
    """
    The sampled-quote-excursion block for one persisted settlement.

    WHY THIS LOOKS THE WAY IT DOES
    ==============================
    The two dicts in a ledger ``arm_exit`` record persist the SAME excursion
    under DIFFERENT key conventions, and that is the whole bug this exists
    to fix:

      * ``trade`` is ``TradeRecord.as_dict()`` and emits UPPERCASE keys
        (``QUOTE_MFE``, ``QUOTE_MFE_MAE_BASIS``, ...);
      * ``trade["position"]`` is ``PaperPosition.as_dict()`` and emits the
        SAME fields LOWERCASE (``quote_mfe_pct``, ``quote_mfe_mae_basis``,
        ...).

    The reader previously looked for the uppercase names on the position
    dict, where they never exist. Every lookup returned ``None``, every row
    came out ``NOT_MEASURED``, and the renderer printed its hardcoded
    fallback -- "no valid same-market SELL quote was obtained for this
    position" -- for trades that HAD been quoted, sampled and settled on a
    valid executable quote. The metric was not missing; the reader could not
    see it.

    So the trade dict is authoritative and the position dict is the
    fallback, and a record that carries neither is reported as carrying no
    telemetry at all rather than as a quote that never existed.
    """
    from .forensics.quote_excursions import (
        excursion_telemetry_state,
        excursion_unavailable_reason,
        settlement_quote_state,
    )

    def field(upper: str, lower: str) -> Any:
        if upper in trade:
            return trade.get(upper)
        return position.get(lower)

    basis = field("QUOTE_MFE_MAE_BASIS", "quote_mfe_mae_basis")
    observations = field("QUOTE_OBSERVATION_COUNT", "quote_observation_count")
    rejections = field("QUOTE_REJECTION_COUNT", "quote_rejection_count")
    # "Did this record persist telemetry at all?" is a question about the
    # RECORD, not about the value, so it is asked by key presence. A record
    # written before the field existed and a record that persisted
    # ``NOT_MEASURED`` are different facts and must not share a sentence.
    telemetry_persisted = any(
        key in trade or key in position for key in
        ("QUOTE_MFE_MAE_BASIS", "quote_mfe_mae_basis"))
    settlement = settlement_quote_state({
        # ``pnl_basis`` and ``sell_execution_status`` live on the TRADE;
        # the ledger RECORD carries its own ``pnl_basis``, which is the one
        # a legacy or modelled settlement is labelled with. Both are read,
        # because a settlement can only be called quote-based if the record
        # that claims it says so.
        "pnl_basis": (record or {}).get("pnl_basis")
                     or trade.get("pnl_basis") or "",
        "sell_execution_status": trade.get("sell_execution_status") or "",
        "exit_basis": trade.get("exit_basis") or "",
        "settlement_status": (record or {}).get("settlement_status")
                             or trade.get("settlement_status") or "",
    })
    state = excursion_telemetry_state(
        {},
        basis=basis,
        observation_count=observations,
        telemetry_persisted=telemetry_persisted,
        rejection_count=rejections,
        settlement=settlement,
    )
    block = {
        "quote_mfe_pct": field("QUOTE_MFE", "quote_mfe_pct"),
        "quote_mae_pct": field("QUOTE_MAE", "quote_mae_pct"),
        "quote_mfe_mae_basis": basis or "NOT_MEASURED",
        "quote_unavailable_reason": field(
            "QUOTE_MFE_UNAVAILABLE_REASON", "quote_unavailable_reason"),
        "quote_mfe_elapsed_ms": field(
            "QUOTE_MFE_ELAPSED_MS", "quote_mfe_elapsed_ms"),
        "quote_mae_elapsed_ms": field(
            "QUOTE_MAE_ELAPSED_MS", "quote_mae_elapsed_ms"),
        "quote_coverage_pct": field("QUOTE_COVERAGE_PCT", "quote_coverage_pct"),
        "quote_sampling_interval_s": field(
            "QUOTE_SAMPLING_INTERVAL_S", "quote_sampling_interval_s"),
        "quote_max_sampling_gap_ms": field(
            "QUOTE_MAX_SAMPLING_GAP_MS", "quote_max_sampling_gap_ms"),
        "quote_mfe_source": field("QUOTE_MFE_SOURCE", "quote_mfe_source"),
        "quote_mae_source": field("QUOTE_MAE_SOURCE", "quote_mae_source"),
        "quote_excursion_cost_basis": field(
            "QUOTE_EXCURSION_COST_BASIS", "quote_excursion_cost_basis"),
        # The evidence behind the pair, so a reader can see that a
        # single-sample excursion is a single observation and not a bound.
        "quote_observation_count": (
            None if observations is None else int(observations or 0)),
        "quote_rejection_count": (
            None if rejections is None else int(rejections or 0)),
        "quote_last_rejection_reason": field(
            "QUOTE_LAST_REJECTION_REASON", "quote_last_rejection_reason"),
        "quote_settlement_sample": bool(
            field("QUOTE_SETTLEMENT_SAMPLED", "quote_settlement_sample")),
        "quote_settlement_identity": field(
            "QUOTE_SETTLEMENT_IDENTITY", "quote_settlement_identity"),
        "quote_settlement_source": field(
            "QUOTE_SETTLEMENT_SOURCE", "quote_settlement_source"),
        "quote_settlement_age_ms": field(
            "QUOTE_SETTLEMENT_QUOTE_AGE_MS", "quote_settlement_age_ms"),
        "quote_settlement_status": settlement,
        "quote_telemetry_state": state,
    }
    if state != "MEASURED" and not block["quote_unavailable_reason"]:
        block["quote_unavailable_reason"] = excursion_unavailable_reason(
            {}, basis=basis, observation_count=observations,
            telemetry_persisted=telemetry_persisted,
            rejection_count=rejections, settlement=settlement)
    return block

def _beta_excursion_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """
    The excursion block for a V2 FLOOR (BETA) settlement, from ITS store.

    The beta experiment is deliberately a separate durable store with its
    own record shape, and that shape carries the two settlement quote
    receipts -- ``entry_basis``/``exit_basis`` and ``exit_identity`` -- but
    NO intratrade quote telemetry. So every beta row's excursion is
    unmeasured, and the only honest thing to report is WHICH KIND of
    unmeasured:

      * quote-based settlement (``exit_basis == EXECUTABLE_QUOTE_BASIS``
        with ``exit_identity == PASS``): a valid same-market SELL quote WAS
        accepted and the position really did sell on it; what is missing is
        the intratrade sampling. Reporting this as "no valid quote obtained"
        is simply false.
      * modelled settlement: booked off an observed mark by the legacy
        backtest path, so there was never an executable quote to observe.

    The invariant is untouched: a quote-based beta settlement still REQUIRES
    a valid accepted SELL quote to have been produced upstream in
    ``BetaExperiment``, and this only reads what that produced. Nothing here
    can turn a modelled settlement into a quote-based one, or grant an
    excursion to a record that never sampled one.
    """
    from .forensics.quote_excursions import (
        excursion_telemetry_state,
        excursion_unavailable_reason,
        settlement_quote_state,
    )
    from .paper_live.v2_floor_beta import EXECUTABLE_QUOTE_BASIS

    settlement = settlement_quote_state(record)
    # ``telemetry_persisted`` is left unset on purpose. A beta record does
    # carry no telemetry field, but it never could -- the experiment's store
    # has no such concept -- so calling it "absent" would describe a
    # historical ledger record rather than this one. Letting the settlement
    # decide produces the factually correct sentence instead.
    state = excursion_telemetry_state(
        {}, basis=None, settlement=settlement)
    exit_identity = str(record.get("exit_identity") or "")
    return {
        "quote_mfe_pct": None,
        "quote_mae_pct": None,
        "quote_mfe_mae_basis": "NOT_MEASURED",
        "quote_unavailable_reason": excursion_unavailable_reason(
            {}, settlement=settlement),
        "quote_mfe_elapsed_ms": None,
        "quote_mae_elapsed_ms": None,
        "quote_coverage_pct": None,
        "quote_sampling_interval_s": None,
        "quote_max_sampling_gap_ms": None,
        "quote_mfe_source": None,
        "quote_mae_source": None,
        "quote_excursion_cost_basis": "",
        "quote_observation_count": 0,
        "quote_rejection_count": 0,
        "quote_last_rejection_reason": None,
        # The settlement quote is real and IS recorded -- it is simply not
        # one of several intratrade samples, so it cannot bound an
        # excursion on its own. Claiming otherwise would turn one point
        # into a range.
        "quote_settlement_sample": False,
        "quote_settlement_identity": exit_identity or "",
        "quote_settlement_source": str(record.get("exit_provider") or ""),
        "quote_settlement_age_ms": None,
        "quote_settlement_status": settlement,
        "quote_telemetry_state": state,
    }


class PaperTrader:
    def __init__(
        self,
        config: Config,
        *,
        clock: Clock | None = None,
        market: MarketDataService | None = None,
        nar: NarService | None = None,
        listener: TelegramListener | None = None,
        quote_service: Any = None,
        paper_engine_factory: Any = None,
        sol_usd_provider: Any = None,
    ) -> None:
        #: Injected collaborators for the execution-aware PAPER engine.
        #:
        #: The offline-test guarantee is structural, not incidental: a test
        #: passes a quote service and never causes a network client to be
        #: built. Production passes nothing and gets the real read-only stack.
        #: Nothing here can sign or broadcast -- the injected object is a QUOTE
        #: provider, and every implementation of it is read-only by contract.
        self.quote_service_override = quote_service
        self.paper_engine_factory = paper_engine_factory
        #: The canonical SOL/USD REFERENCE provider, injected for the same
        #: reason. There is exactly one of these in the process and BOTH PAPER
        #: sizing and REAL equity valuation read it, so the two modes can never
        #: disagree about what a SOL is worth. A test injects a fake and the
        #: accounting layer never builds a network client of its own.
        self.sol_usd_provider = sol_usd_provider
        self.config = config
        self.clock = clock or SystemClock()
        self.started_at = self.clock.now()
        # Observation capture is attached to the market service, which is the
        # one point every observation passes through. A market service handed
        # in by a test or another runtime keeps whatever capture it was built
        # with; nothing here overrides it.
        self.observations = store_for(config, config.var_dir, clock=self.clock)
        self.market = market or MarketDataService(
            config.market, clock=self.clock, observations=self.observations)
        self.nar = nar or NarService(config.nar, clock=self.clock)
        # Prospective forward-validation capture. The boundary is created once
        # and never moved, so anything recorded here is genuinely post-
        # deployment and gating on the frozen L25. The recorder mirrors the
        # ledger and cannot block or alter a paper trade.
        self.forward_root = config.var_dir / "forward_validation"
        try:
            self.forward_deployment = load_or_create_deployment(
                self.forward_root,
                strategy=l25.ACTIVE_STRATEGY,
                l25_min_liq_to_mcap=config.decision.l25_min_liq_to_mcap,
                config_source=config.config_source,
                strategy_hash=getattr(l25, "L25_HASH", ""),
            )
            self.forward = ForwardRecorder(
                self.forward_root, self.forward_deployment
            )
        except Exception as exc:  # noqa: BLE001
            # A missing forward store must not stop PAPER trading. Validation
            # is evidence ABOUT the strategy, not a precondition for it.
            log("ERROR", "forward validation unavailable (%s): %s",
                type(exc).__name__, exc)
            self.forward_deployment = None
            self.forward = None

        self.ledger = Ledger(
            config.ledger_file,
            config.decisions_file,
            config.candidates_file,
            config.updates_file,
            clock=self.clock,
            forward=self.forward,
        )
        self.memory = ResearchMemory()
        self.listener = listener
        self.signal_source = SignalSourceState(
            threshold_s=config.telegram.fallback_threshold_s
        )
        self.signal_source_selection = SOURCE_AUTO
        self._signal_source_counters: dict[str, int] = {
            "primary_valid": 0, "fallback_valid": 0,
            "fallback_ignored_primary_active": 0,
            "cross_source_duplicates": 0,
        }
        self._seen_opportunities: dict[str, list[tuple[float, str]]] = {}
        self._wire_listener_sources()
        self.engine = DecisionEngine(
            config.decision, config.exits, memory=self.memory
        )
        self.account = PaperAccount(PaperAccountState(), config.risk)
        # Historical PAPER rows may predate the execution-aware accounting
        # book.  They are never silently merged into the active production
        # scope: startup records the boundary and the reconciliation result,
        # while leaving the unreconciled history untouched for investigation.
        self._historical_accounting_boundary: dict[str, Any] = {
            "scope": "ACTIVE_NEW_EXECUTION_AWARE_ONLY",
            "historical_unreconciled": False,
            "reason": None,
        }
        self._active_accounting_since: float | None = None
        self.open_book: dict[str, Any] = {}
        self.last_decision: dict[str, Any] | None = None
        self.last_trade: Trade | None = None
        self.control = ControlCommand()
        self.running = False
        self.recovered = False
        self.candidates_seen = 0
        # Second line of duplicate defence. The listener dedupes on
        # (chat_id, message_id); this dedupes on candidate identity, so a
        # duplicate that arrives through any other path still cannot open a
        # second position.
        self._seen_candidates: dict[str, float] = {}
        self._seen_messages: set[tuple[str, int]] = set()
        self.buys = 0
        self.rejects = 0
        self.waits = 0
        #: BUY decisions that did NOT become a position, and why. A separate
        #: counter from ``buys`` on purpose: ``buys`` counts fills, and a screen
        #: showing only ``buys`` cannot tell a quiet session from a session where
        #: every authorized entry was refused.
        self.blocks = 0
        #: One notification per MESSAGE IDENTITY -- keyed by the ``dedupe_key``
        #: the renderer stamps (kind + parent + the arms/attempts it reports).
        #: Keeps a re-entrant or duplicated flush from double-messaging the
        #: operator WITHOUT collapsing two genuinely different messages: one
        #: parent's three arms close at different times, so a key of
        #: ``(kind, parent)`` alone would swallow the second arm's receipt.
        #: Declared HERE rather than next to the bridge wiring, because tests
        #: (and the runtime) may install ``_paper_bridge`` directly, and a
        #: lazily-initialised attribute would simply not exist on those paths.
        self._beta_notified: dict[str, bool] = {}
        #: Structured record of every notification ATTEMPT, newest last.
        self.notification_log: list[dict[str, Any]] = []
        self.blocked_reasons: dict[str, int] = {}
        #: Strategy rejections, counted BY THE GATE THAT CAUSED THEM, for THIS
        #: process run. The exact counterpart of ``blocked_reasons``.
        #:
        #: ``rejects`` alone is a count with no distribution, so a shutdown
        #: reading "rejects=33" cannot be told apart from "33 candidates were
        #: rejected for 33 different reasons" and "33 were all the same reason".
        #: The per-candidate reason is already on record in the journal, but a
        #: reader of the shutdown line had no way to reach it. The operative
        #: gate's own status code is used, because that is the term the
        #: strategy, the log and the journal already use; when no gate refused
        #: the candidate, the block-level risk flag is named instead, and only
        #: when neither exists does it fall back to an explicit UNNAMED bucket
        #: rather than a blank that reads as "no reason".
        #: Candidate tasks currently between the alert and the journal write, so
        #: shutdown can drain them instead of cancelling an order into silence.
        self._inflight_candidates: set[asyncio.Task] = set()
        #: Strategy rejections, counted BY CAUSE for THIS run. ``rejects`` is the
        #: total; this is its distribution, so a shutdown reading "rejects=33"
        #: can also say "21 V3C_REJECTED_NOT_FAST_ENTRY, 6 ...". It is populated
        #: by :meth:`_record_reject_reason` on every REJECT decision and read by
        #: :meth:`shutdown` and :meth:`session_counters`. It is initialised here
        #: (not lazily) so a run with zero rejects reads ``{}`` -- an empty
        #: histogram -- rather than raising on an attribute that was never set.
        self.reject_reasons: dict[str, int] = {}
        #: True only while shutdown is actively cancelling in-flight work, so a
        #: cancellation can name its cause precisely.
        self._shutting_down = False
        #: The last BUY's execution outcome: ``DECISION=BUY``,
        #: ``PAPER_FILL=SUCCESS``, ``REAL_FILL=SUCCESS`` or
        #: ``BUY_BLOCKED=<reason>``. The authoritative answer to "did that buy
        #: happen", read by the dashboard and the operator UI alike.
        self.last_execution: dict[str, Any] | None = None
        self._marks: dict[str, MarkedPosition] = {}
        self._fvp_budget = 30
        self._safety_report = None
        self._flatten_requested = False
        self._listener_task: asyncio.Task | None = None
        self._last_listener_attempt = 0.0
        self._stop_event: asyncio.Event | None = None
        #: The Telegram Bot API poll task, if the channel is running.
        self._bot_task: asyncio.Task | None = None
        # Single-writer ownership.  Set by _acquire_ownership() before any
        # state or ledger mutation; consulted by _require_owner() on every
        # durable write.  One boundary, no second lock, so no deadlock.
        self._lock = None
        self._account_id: str | None = None
        self._lock_path: str | None = None
        self._legacy_report: dict[str, Any] | None = None
        self.adopt_legacy_account: bool = False
        # Injectable so tests can advance a fake clock instead of really waiting.
        # Everything that waits in this engine goes through here.
        self._sleep = asyncio.sleep

        # ------------------------------------------------------------------
        # V3 runtime: mode, accounts, executors
        # ------------------------------------------------------------------
        #
        # `self.state` is THE canonical runtime state. It is constructed OFF /
        # PAPER / not-armed and there is no argument, config value or
        # environment variable that can change that at construction time.
        self.state = RuntimeState()
        # THE CONFIGURATED PROFILE IS NOW ACTUALLY APPLIED.
        #
        # ``RuntimeState()`` used to be constructed BARE, so
        # ``DecisionConfig.strategy_profile`` -- validated at load time by
        # ``Config.__post_init__`` and then read by nothing except two
        # diagnostic prints -- never reached the runtime. ``STRATEGY_PROFILE``
        # was a dead setting: set it, and the bot ignored it without complaint.
        #
        # This applies it as the INITIAL value, before any operator selection.
        # An unset or invalid setting resolves to the declared default
        # (``L25_CURRENT``), so a process that never touches the setting behaves
        # exactly as it did before -- which is the point: the difference is only
        # for a process that DID set it and was being ignored.
        self.state.strategy_profile = str(
            getattr(config.decision, "strategy_profile", "") or
            self.state.strategy_profile)
        #: The unified journal, partitioned by mode.
        self.journal = Journal(config.var_dir / "journal", clock=self.clock)
        #: The real account. Holds real positions and a cached wallet snapshot;
        #: it is completely separate from ``self.account`` (the paper one).
        self.real_account = RealAccount()
        #: Durable pre-broadcast execution intents. Append-only, fsync'd BEFORE
        #: the irreversible broadcast, so a retry after an ambiguous outcome
        #: cannot broadcast a second time.
        self.execution_intents = ExecutionIntentStore(
            config.var_dir / "real_execution_intents.jsonl", clock=self.clock
        )
        #: Durable REAL account state: open/closed positions and realized PnL.
        #: ``RealAccount`` was previously in-memory only, so a restart orphaned
        #: an on-chain holding with no local trace. This is the file that makes
        #: restart reconciliation possible.
        self.real_state_file = config.var_dir / "real_account_state.json"
        # Authoritative multi-asset REAL equity, refreshed by the async
        # `real_equity()` and read by the sync status renderer. None until a
        # measurement exists, so the UI says "unavailable" rather than guessing.
        self._real_equity_cache = None
        # (mint, reason) -> occurrence count, so a repeatedly blocked exit
        # escalates visibly instead of being indistinguishable from one refusal.
        self._sell_block_counts: dict[tuple, int] = {}
        # Set by the exit path when the trigger mark and the realized price
        # disagree; applied to the position during booking.
        self._pending_mark_divergence_pct: float | None = None
        #: Result of the startup reconciliation, surfaced to the operator.
        self.real_reconcile_report: dict[str, Any] | None = None
        #: True once startup reconciliation has run and found no unresolved
        #: mismatch. New entries are refused while this is False and REAL is
        #: armed.
        self._real_state_consistent: bool = True
        #: The real executor. Constructed lazily, so importing and gate-testing
        #: never touches key material.
        self._real_executor: RealExecutor | None = None
        #: Populated by the preflight; drives the UI's "is REAL available" line.
        self.real_preflight: Any = None
        #: Cached SOL/USD, refreshed by the monitor loop.
        self._sol_price_usd: float | None = None
        #: The PROVENANCE of that number: the snapshot it came from. Held so the
        #: equity figure can record where the price was measured, rather than
        #: presenting a bare float as if it were self-describing.
        self._sol_usd_snapshot: Any = None
        #: When the executed REAL quote was taken. Used by the fill-time
        #: freshness gate so a quote older than REAL_QUOTE_MAX_AGE_MS cannot be
        #: executed.
        self._real_quote_at: float | None = None
        #: Per-mint timer for a REAL position whose market data went invalid,
        #: so the exit policy's DATA_INVALID path can still fire.
        self._real_data_invalid_since: dict[str, float] = {}
        #: The emergency-stop latch, persisted so it survives a restart.
        self._emergency_file = config.var_dir / "emergency_stop.json"
        self._restore_emergency_stop()
        self.buttons = ButtonHandler(self.state, self._operator_actions())
        # Restore the operator's PAPER strategy selection before anything can
        # trade, so the first signal after a restart uses the selected arms.
        self._restore_strategies()
        # The arm selection is now final; derive the entry gates from it
        # BEFORE the first candidate is decided.
        self._sync_effective_entry_gates()
        self._restore_take_profit()
        self._restore_signal_source()
        #: The notifier callable itself, before the guard is wrapped around it.
        self._notifier: Callable[[str, Any], None] = lambda text, kb=None: None
        self._notify: Callable[[str, Any], None] = self._guarded_notify
        #: The EXECUTION-AWARE PAPER engine, built lazily on the first PAPER
        #: signal. It is not constructed here because it pulls in the
        #: read-only chain stack, and a process that never trades PAPER should
        #: not pay for it or hold a second quote stack in its object graph.
        #: ``None`` means "not built yet"; see :meth:`_paper_bridge`.
        self._paper_bridge = None
        #: The arm names the LIVE bridge was built with.
        #:
        #: This is the fix for the reported "I changed the strategy and the bot
        #: kept trading the old one" defect. The bridge resolved
        #: ``self.state.arms`` ONCE, at first construction, and nothing
        #: invalidated it afterwards -- so the operator's selection changed the
        #: menu, the status line and the saved file while the running engine
        #: carried on opening trades as whatever it had captured at boot. The
        #: journal shows the selection moving at 16:13:07 and the removed arm
        #: still closing trades 47 minutes later.
        #:
        #: Recording what the bridge IS running turns "is the live runtime
        #: actually using the selection?" into a comparison, and
        #: :meth:`_paper_bridge_instance` reconciles whenever the two differ.
        self._paper_bridge_arms: tuple[str, ...] = ()
        #: Arms that are running but NO LONGER SELECTED, because they were
        #: deselected while holding a position and are being carried until that
        #: position closes. Tracked so the invariant below can tell a documented
        #: deferral from an unexplained disagreement -- they look identical in a
        #: set comparison, and treating a deferral as a mismatch would make the
        #: warning fire on correct behaviour.
        self._paper_bridge_deferred: tuple[str, ...] = ()
        #: The AMARIH paired-exit PAPER experiment's driver, built on first use
        #: and kept for the process while the experiment has anything to do.
        #: ``None`` means "never started in this process", which is different
        #: from "selected but idle": the driver is what owns the poller, the
        #: durable seen-id set and the outstanding arms.
        self._amarih_runtime: Any = None
        #: THE one PumpView polling task. Exactly one, for the life of the
        #: selection -- a second poller would double every entry, and the polled
        #: source has no server-side idempotency to fall back on.
        self._amarih_task: asyncio.Task | None = None
        #: The Amarih DEADLINE scheduler. Serviced once per second, separately
        #: from the poller and separately from the monitor loop, so an exit is
        #: detected on time regardless of what else the monitor tick is doing.
        #: Cancelled independently of the poller, because a deselected
        #: experiment still has open arms with deadlines to honour.
        self._amarih_scheduler_task: asyncio.Task | None = None
        #: ``arm::signal_id`` of every execution-aware PAPER settlement this
        #: process has committed to the ledger. THE WRITE-SIDE IDEMPOTENCY KEY.
        #: A settlement is one position ending once, so a repeated close event
        #: is a duplicate RECEIPT to be dropped, never a second ledger row.
        #: In-memory on purpose: the ledger is append-only and is read back on
        #: every CLOSED POSITIONS query, where :meth:`_arm_closed_rows` applies
        #: the same key so a duplicate already on disk still renders once.
        self._arm_settlement_keys: set[str] = set()
        #: Telegram Bot API transport. BUILT but NOT STARTED here: a transport
        #: exists in the object graph from construction, and only
        #: :meth:`start_bot_control` begins polling. Nothing in this block can
        #: change a state, and REAL arming is unreachable from it - the only
        #: path to REAL remains ButtonHandler's confirm-real branch, which needs
        #: a valid unexpired token AND a passing preflight.
        self.bot_transport: Any = None
        self.bot_notifier: Any = None
        self.bot_control: Any = None
        #: CLI-only: when True the Telegram BOT is never built. Declared before
        #: _build_bot_control() so that method can see it.
        self.cli_only: bool = bool(getattr(config, "cli_only", False))
        #: Rolling latency percentiles, split so a notification can never be
        #: timed as trading.
        self.latency = LatencyAccumulator()
        self._current_stages: StageRecorder | None = None
        #: Durable per-candidate stage timings. In-memory StageRecorder alone
        #: loses its timing on restart, so the historical p50/p90/p95/p99/max
        #: could not be reconstructed. One small buffered append per candidate,
        #: off the notification path.
        self.stage_store = StageStore(config.var_dir / "stage_timings.jsonl")
        #: Push each strategy rejection to Telegram. OFF by default: a rejection
        #: is the normal outcome for most candidates, so pushing every one is
        #: noise, not information. The reason remains in the log, the journal,
        #: the dashboard's LAST EVENT row and on demand via /status.
        self.notify_rejects: bool = bool(
            getattr(config, "notify_rejects", False)
        )
        self._last_error_text: str = ""
        self._build_bot_control()
        # Fail-closed entry gate. The budget is the project's EXISTING
        # market.stale_after_ms, which was previously only a freshness LABEL on
        # a quote and never an entry gate - so a 34s-old signal was traded as if
        # it were fresh. Reused verbatim; no second threshold is introduced.
        self.stale_gate = StaleEntryGate(
            budget_ms=config.market.stale_after_ms
        )

    # ------------------------------------------------------------------
    # V3 runtime: mode, operators, executors
    # ------------------------------------------------------------------

    # -- emergency stop latch ---------------------------------------------

    def _restore_emergency_stop(self) -> None:
        """
        Re-apply a persisted emergency stop BEFORE anything else.

        A restart must not quietly un-halt a halted bot, so this runs in
        ``__init__`` before the operator UI exists and long before the first
        candidate could be processed.
        """
        try:
            raw = load_json(self._emergency_file) or {}
        except Exception:  # noqa: BLE001 - a corrupt latch is a latch
            raw = {}
        self.state.restore_emergency_stop(bool(raw.get("emergency_stop")))

    def _persist_emergency_stop(self, latched: bool) -> None:
        try:
            save_json_atomic(
                self._emergency_file,
                {
                    "emergency_stop": bool(latched),
                    "at": self.clock.now(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "cannot persist the emergency-stop latch: %s", exc)

    # -- REAL wallet observation ------------------------------------------

    async def observe_real_wallet(self) -> tuple[bool, str]:
        """
        Load the AUTHORITATIVE wallet state into the REAL account.

        This is what makes the REAL BUY path reachable. Without an observation
        ``RealAccount.equity_usd()`` returns ``None``, ``_entry_equity_usd()``
        returns 0.0, and sizing refuses every order - the path is dead.

        It FAILS CLOSED: a wallet that cannot be read, or a SOL price that
        cannot be obtained, produces no observation. The caller then refuses,
        which is correct: a real order sized off a stale or absent balance is a
        guess with real money attached. No fake equity is ever synthesised.
        """
        try:
            state = await self.real_backend.wallet_state()
            address = self.real_backend.pubkey
        except Exception as exc:  # noqa: BLE001
            reason = f"wallet observation failed: {type(exc).__name__}"
            log("ERROR", "REAL %s", reason)
            return False, reason
        lamports = state.get("lamports")
        if lamports is None:
            return False, "wallet observation returned no native SOL balance"
        price = await self._backend_sol_price()
        if not price or price <= 0:
            return False, "SOL/USD valuation unavailable; cannot price wallet equity"
        self.real_account.observe_wallet(
            lamports=int(lamports),
            address=address,
            sol_price_usd=float(price),
            now=self.clock.now(),
        )
        return True, f"{state.get('sol', 0.0):.6f} SOL observed"

    def _real_logical_key(self, side: str, mint: str) -> str:
        """The stable identity of a logical REAL order, used for idempotency."""
        return f"{side}:{mint}"

    # -- REAL persistence --------------------------------------------------

    def save_real_state(self) -> bool:
        """Persist the REAL account atomically. Never touches the wallet key.

        Returns whether the durable write COMMITTED, so a caller that is about
        to tell the operator "closed" can tell whether it is entitled to.

        "The trade closed" and "the closed trade is persisted" are two
        different facts. The in-memory :class:`RealAccount` is mutated before
        this is called, so a failure here leaves the process believing the
        position is CLOSED while ``real_account_state.json`` still says OPEN --
        and a restart would then resurrect a position the chain has already
        sold. Returning the failure instead of swallowing it is what lets the
        caller say exactly that, rather than emitting a success receipt for a
        record that does not exist.
        """
        try:
            save_json_atomic(
                self.real_state_file,
                {
                    "saved_at": self.clock.now(),
                    "account": self.real_account.to_state(),
                    "pending_intents": self.execution_intents.pending_summary(),
                },
            )
            return True
        except Exception as exc:  # noqa: BLE001 - persistence must not kill a trade
            log("ERROR", "REAL state persistence FAILED (%s): %s",
                type(exc).__name__, exc)
            return False

    def load_real_state(self) -> bool:
        """Restore REAL positions from disk. Returns True when something loaded."""
        raw = load_json(self.real_state_file)
        if not raw or not raw.get("account"):
            return False
        try:
            self.real_account = RealAccount.from_state(raw["account"])
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "cannot restore REAL account state: %s", exc)
            return False
        return True

    async def reconcile_real_state(self) -> dict[str, Any]:
        """
        Reconcile local REAL state against the chain at startup.

        The hardest case is a broadcast whose signature the process never
        learned. For every UNRESOLVED intent we look up the chain:

        * a signature that the chain has CONFIRMED advances the intent to
          CONFIRMED (the transaction happened);
        * a signature the chain reports FAILED becomes FAILED;
        * no signature, or no chain answer, stays UNKNOWN and BLOCKS new entries.

        Restored open positions are checked against the wallet's actual token
        balance. A position the wallet no longer holds is a MISMATCH: the chain
        is authoritative and a mismatch fails closed for new entries rather than
        being silently overwritten with a local assumption.
        """
        report: dict[str, Any] = {
            "restored": False,
            "intents_checked": 0,
            "intents_confirmed": 0,
            "intents_failed": 0,
            "intents_unresolved": 0,
            "positions_checked": 0,
            "mismatches": [],
            # Per-position reconciliation findings, with an EXPLICIT state
            # (EXTERNALLY_CLOSED / SETTLED_NON_SOL / UNKNOWN_EXTERNAL_ACTIVITY
            # / ...). The ``mismatches`` strings above are kept unchanged so
            # every existing reader still works; this is the classifiable view
            # the operator needs to resolve a stuck state.
            "reconciliation": [],
            "consistent": True,
        }
        report["restored"] = self.load_real_state()
        rpc = None
        try:
            rpc = self.real_executor.backend.rpc
        except Exception:  # noqa: BLE001 - no configured wallet/RPC: nothing to do
            rpc = None

        # 1. Resolve execution intents that may or may not have landed.
        for intent in list(self.execution_intents.all_intents()):
            if intent.terminal:
                continue
            report["intents_checked"] += 1
            if not intent.signature or rpc is None:
                report["intents_unresolved"] += 1
                continue
            try:
                statuses = await rpc.get_signature_statuses(intent.signature)
            except Exception:  # noqa: BLE001 - an unanswered lookup is NOT a pass
                report["intents_unresolved"] += 1
                continue
            st = statuses[0] if statuses else None
            if st is None:
                # Not indexed: could still be in flight, or could have been
                # dropped. Ambiguous by definition - never assume either way.
                report["intents_unresolved"] += 1
                continue
            if st.get("err"):
                self.execution_intents.transition(
                    intent.execution_id, "FAILED", detail="chain reported failure",
                )
                report["intents_failed"] += 1
            elif st.get("confirmationStatus") in ("confirmed", "finalized"):
                self.execution_intents.transition(
                    intent.execution_id, "CONFIRMED", detail="confirmed at startup",
                )
                report["intents_confirmed"] += 1
            else:
                report["intents_unresolved"] += 1

        # 2. Every restored OPEN position must still be held by the wallet.
        from .reconciliation import classify_position_against_chain

        for pos in self.real_account.open_positions():
            report["positions_checked"] += 1

            def _record_finding(balance):
                report["reconciliation"].append(
                    classify_position_against_chain(
                        mint=pos.mint,
                        symbol=getattr(pos, "symbol", "") or "",
                        journal_status=getattr(pos, "status", "") or "",
                        chain_balance=balance,
                    ).as_dict()
                )

            if rpc is None:
                report["mismatches"].append(
                    f"{pos.mint}: RPC unavailable; cannot verify on-chain holding"
                )
                _record_finding(None)
                continue
            try:
                balance = await rpc.get_token_account_balance(
                    self.real_executor.backend.pubkey, pos.mint
                )
            except Exception as exc:  # noqa: BLE001
                report["mismatches"].append(
                    f"{pos.mint}: token balance unreadable ({type(exc).__name__})"
                )
                _record_finding(None)
                continue
            # A readable read that found no token account means the balance is
            # genuinely zero, not unreadable; the two must not be conflated.
            _record_finding(0 if balance is None else int(balance))
            if not balance:
                report["mismatches"].append(
                    f"{pos.mint}: local OPEN position but wallet holds no tokens"
                )

        unresolved = self.execution_intents.unresolved()
        report["consistent"] = not report["mismatches"] and not unresolved
        if unresolved:
            report["unresolved_execution_ids"] = [
                {"id": i.execution_id, "mint": i.mint, "state": i.state,
                 "signature": i.signature}
                for i in unresolved
            ]
        self.real_reconcile_report = report
        self._real_state_consistent = bool(report["consistent"])
        if report["restored"] or report["intents_checked"] or report["positions_checked"]:
            log(
                "RECOVERY",
                "REAL reconciliation: restored=%s intents=%d(confirmed=%d "
                "failed=%d unresolved=%d) positions=%d mismatches=%d",
                report["restored"], report["intents_checked"],
                report["intents_confirmed"], report["intents_failed"],
                report["intents_unresolved"], report["positions_checked"],
                len(report["mismatches"]),
            )
        return report

    # -- executors ---------------------------------------------------------

    @property
    def real_executor(self) -> RealExecutor:
        """The real executor, built on first use so no key is touched early."""
        if self._real_executor is None:
            # The RESOLVER is passed, not the provider instance: both are built
            # lazily and in either order, so binding the instance here would
            # capture "None" whenever the price is read before the executor is.
            # A resolver defers the question to the moment of use, which is what
            # keeps the preflight's ``sol_usd_valuation_available`` check and
            # :meth:`real_equity` on the ONE canonical reference.
            self._real_executor = RealExecutor(
                self.config, sol_usd_provider=self._sol_usd_reference)
        return self._real_executor

    @property
    def real_backend(self) -> LiveSolanaBackend:
        return self.real_executor.backend

    def real_capable(self) -> bool:
        """Are the real capabilities PRESENT? Not the same as armed."""
        real = self.config.real
        return real.wallet_configured and real.rpc_configured

    async def preflight_real(self):
        """
        Run the full live preflight and cache its result.

        Read-only. Proves signing offline, never broadcasts. The result drives
        both the REAL button's outcome and the status line's availability
        indicator, so the two cannot disagree.
        """
        try:
            result = await self.real_executor.preflight()
        except Exception as exc:  # noqa: BLE001 - a failed probe is a refusal
            from .execution.real import PreflightResult, SafetyCheck

            result = PreflightResult(
                False,
                [SafetyCheck("preflight_ran", False, type(exc).__name__)],
                "preflight raised",
            )
        self.real_preflight = result
        return result

    # -- operator surface --------------------------------------------------

    def _operator_actions(self) -> OperatorActions:
        return OperatorActions(
            preflight=self.preflight_real,
            journal=self.journal,
            on_state_change=self._on_operator_state_change,
            on_real_armed=self._on_real_armed,
            on_emergency_stop=self._on_emergency_stop,
            persist_emergency=self._persist_emergency_stop,
            persist_strategies=self._persist_strategies,
            persist_take_profit=self._persist_take_profit,
            persist_signal_source=self._persist_signal_source,
            current_signal_source=lambda: self.signal_source_selection,
            risk_acknowledged=lambda: bool(
                getattr(self.config.real, "risk_acknowledged", False)
            ),
            extras={
                "stats": self.statistics_by_mode,
                "open_orders": self.open_order_rows,
                # The two operator position views. Both read the SAME objects the
                # accounting layer reads - ``self.account`` /
                # ``self.real_account`` for open, the ledger's exit events for
                # closed - so /balance, /status, OPEN POSITIONS and CLOSED
                # POSITIONS cannot disagree because there is only one copy.
                "open_positions": self.open_positions_view,
                "closed_positions": self.closed_positions_view,
                # The armed-REAL message used to pass a literal "—" and no
                # equity at all, so it always rendered "$0.0000". This supplies
                # the measured values instead.
                "real_equity": self.real_equity,
                "recent_orders": self.recent_orders,
                # PAPER-only emergency close: closes all open PAPER positions
                "close_all_paper": self.close_all_paper,
            },
        )

    # -- strategy selection persistence ----------------------------------

    #: Where the selected PAPER arms live. Written atomically and read at
    #: startup, so a restart resumes the selection the operator actually made
    #: rather than silently reverting to a default.
    _strategy_file = "strategy_selection.json"

    #: Where the selectable TAKE PROFIT lives. Same mechanism as the strategy
    #: selection -- atomically written, validated on the way in and the way out,
    #: and re-read at startup -- because it is a setting, not a strategy.
    _take_profit_file = "take_profit_setting.json"

    def _strategy_path(self) -> Path:
        return self.config.var_dir / self._strategy_file

    def _take_profit_path(self) -> Path:
        return self.config.var_dir / self._take_profit_file

    def _persist_take_profit(self, pct: object) -> None:
        """
        Save the operator's take-profit selection atomically.

        Validated on the way OUT as well as on the way in, so a corrupted or
        tampered file cannot restore a level the vocabulary forbids.
        """
        from .paper_live.state import atomic_write_json
        from .paper_live.take_profit import TakeProfitError, normalise

        try:
            level = normalise(pct)
        except TakeProfitError as exc:
            log("ERROR", "refusing to persist an invalid take-profit: %s", exc)
            return
        self.state.take_profit_pct = level
        try:
            atomic_write_json(self._take_profit_path(), {
                "take_profit_pct": level,
                "at": time.time(),
            })
        except Exception as exc:  # noqa: BLE001 - never fail a button press
            log("ERROR", "cannot persist take-profit selection: %s: %s",
                type(exc).__name__, exc)

    def _restore_take_profit(self) -> None:
        """
        Restore the take-profit at startup, re-validating it against CURRENT
        rules -- exactly as the strategy selection is.

        A file written when a different vocabulary was live is not automatically
        valid now, so it is re-parsed here rather than trusted.
        """
        from .paper_live.take_profit import TakeProfitError, normalise

        path = self._take_profit_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a bad file is not fatal
            log("ERROR", "cannot read the saved take-profit selection: %s: %s",
                type(exc).__name__, exc)
            return
        try:
            level = normalise(payload.get("take_profit_pct"))
        except TakeProfitError as exc:
            log("ERROR", "saved take-profit selection is not valid here, using "
                "OFF: %s", exc)
            return
        self.state.take_profit_pct = level
        log("MODE", "restored take-profit selection: %s",
            "OFF" if level is None else "+%g%%" % level)

    def _signal_source_path(self) -> Path:
        return self.config.var_dir / "signal_source_selection.json"

    def _persist_signal_source(self, selection: str) -> None:
        if selection not in {SOURCE_AUTO, SOURCE_MOODENG, SOURCE_PAINAPE,
                             SOURCE_AMARIH}:
            log("ERROR", "refusing invalid signal source selection: %s", selection)
            return
        self.signal_source_selection = selection
        try:
            from .paper_live.state import atomic_write_json
            atomic_write_json(self._signal_source_path(), {
                "signal_source": selection,
                "at": time.time(),
            })
        except Exception as exc:  # noqa: BLE001 - UI selection must not crash runtime
            log("ERROR", "cannot persist signal source: %s: %s",
                type(exc).__name__, exc)

    def _restore_signal_source(self) -> None:
        path = self._signal_source_path()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            selection = str(raw.get("signal_source", SOURCE_AUTO))
            if selection in {SOURCE_AUTO, SOURCE_MOODENG, SOURCE_PAINAPE,
                             SOURCE_AMARIH}:
                self.signal_source_selection = selection
        except Exception as exc:  # noqa: BLE001 - corrupt preference falls back to AUTO
            log("ERROR", "cannot restore signal source selection: %s", type(exc).__name__)

    def _sync_effective_entry_gates(self) -> tuple[str, ...]:
        """
        Push the EFFECTIVE ENTRY GATE CONTRACT onto the decision layer.

        The decision engine's gate used to come from ``strategy_profile`` alone,
        so nothing connected it to the arms the operator actually selected. The
        two axes are written by different code -- the Telegram selector writes
        ``selected_arms`` and, in PAPER, never touches ``strategy_profile`` --
        which is how selecting the V2 FLOOR (BETA) experiment left L25 in front
        of it as the authorizing gate.

        This derives the operative gate set from the selection via
        ``paper_live.arms.effective_entry_gates`` and installs it. It is
        deliberately NOT a mutation of ``state.strategy_profile``: that field is
        the profile new POSITIONS are stamped with and several consumers read
        it (exit policy, sizing, the UI status line, the safety audit), and
        overwriting it from a selection would change all of them at once. The
        operative gate set is a separate, narrow fact, computed here and
        applied only to authorization.

        Returns the gates now operative, and logs a CHANGE -- so the log shows
        the moment a selection altered which gate authorizes entries.
        """
        from .paper_live.arms import effective_entry_gates

        try:
            gates = effective_entry_gates(
                list(self.state.arms), profile=self.state.strategy_profile)
        except Exception as exc:  # noqa: BLE001 - never fail a boot on this
            log("ERROR", "cannot derive the effective entry gates (%s); the "
                "profile's own gate stays operative", exc)
            return ()
        previous = getattr(self.engine, "entry_gates", ())
        try:
            applied = self.engine.set_entry_gates(gates)
        except AttributeError:
            # A decision layer that does not implement the contract -- a test
            # double, or a caller that supplied its own -- keeps whatever gate
            # it has. This must never take down an entry path, so it is
            # reported and skipped rather than raised.
            log("WARN", "the decision layer does not implement "
                "set_entry_gates; its own gate stays operative")
            return ()
        if applied is None:
            return ()
        if applied != previous:
            log("PAPER", "effective entry gate(s) %s -> %s for arms %s "
                "(profile %s)",
                ",".join(previous) or "<none>", ",".join(applied),
                ",".join(self.state.arms) or "<none>",
                self.state.strategy_profile)
        return applied

    def _persist_strategies(self, arms: list) -> None:
        """
        Save the operator's strategy selection atomically.

        Validated on the way OUT as well as on the way in, so a corrupted or
        tampered file cannot restore a selection the rules forbid.
        """
        from .modes import validate_strategy_selection
        from .paper_live.state import atomic_write_json

        ok, message, normalised = validate_strategy_selection(
            arms, trade_mode=self.state.trade_mode)
        if not ok:
            log("ERROR", "refusing to persist an invalid strategy selection: %s",
                message)
            return
        try:
            atomic_write_json(self._strategy_path(), {
                "strategies": list(normalised),
                "trade_mode": self.state.trade_mode.value,
                "at": time.time(),
            })
        except Exception as exc:  # noqa: BLE001 - never fail a button press
            log("ERROR", "cannot persist strategy selection: %s: %s",
                type(exc).__name__, exc)

    def _restore_strategies(self) -> None:
        """
        Restore the selection at startup, re-validating it against CURRENT rules.

        A file written when PAPER allowed several arms is not automatically
        valid for REAL, so the mode is re-checked here rather than trusted.
        """
        from .modes import validate_strategy_selection

        path = self._strategy_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a bad file is not fatal
            log("ERROR", "cannot read the saved strategy selection: %s",
                type(exc).__name__)
            return
        requested = payload.get("strategies") or []
        ok, message, normalised = validate_strategy_selection(
            requested, trade_mode=self.state.trade_mode)
        if not ok:
            log("ERROR", "saved strategy selection is not valid here, using the "
                "default: %s", message)
            return
        self.state.selected_arms = normalised
        log("MODE", "restored strategy selection: %s", ", ".join(normalised))

    def _on_operator_state_change(self, state: RuntimeState) -> None:
        # Every operator transition -- START, STOP, select PAPER, select REAL --
        # flushes the execution-aware PAPER dataset. STOP in particular must
        # not lose the run: a stopped bot that had already simulated trades
        # would otherwise leave the artefacts holding whatever was last written
        # by an entry, mid-trade.
        self.persist_paper_dataset()
        self.journal.event(
            "state_change",
            state.trade_mode,
            run_state=state.run_state.value,
            real_armed=state.real_armed,
            emergency_stop=state.emergency_stop,
        )
        log(
            "MODE",
            "%s -> %s (real_armed=%s emergency=%s)",
            state.run_state.value,
            state.trade_mode.value,
            state.real_armed,
            state.emergency_stop,
        )

    def _on_real_armed(self, state: RuntimeState) -> None:
        log("MODE", "REAL ARMED by operator; %s may now use real funds",
            l25.ACTIVE_STRATEGY)

    def _on_emergency_stop(self) -> None:
        log("MODE", "EMERGENCY STOP latched; new entries disabled")

    # -- Telegram Bot API transport ---------------------------------------

    def _build_bot_control(self) -> None:
        """
        Construct the Bot API transport, notifier and inbound controller.

        Safe to call when nothing is configured: with no token or no
        allow-listed chat the control channel stays INACTIVE rather than
        falling open, because "allow every chat" is never an acceptable default.

        The inbound controller is handed ``self.handle_button`` - the EXISTING
        entry point that already delegates to ``ButtonHandler``. This is the
        whole wiring: no second command path is introduced, and this method
        authors no state transition.
        """
        cfg = self.config.telegram
        try:
            from .telegram.botapi import BotApiTransport, BotControl
            from .telegram.notifier import TelegramNotifier

            if self.cli_only:
                # CLI-only: the bot is not built at all, so there is no notifier,
                # no poller, no queue and no outbound network call. The upstream
                # CHANNEL listener is built separately and is unaffected.
                log("INFO", "CLI-only mode: Telegram outbound disabled; "
                            "the channel listener still ingests alerts")
                return

            if not (cfg.bot_token and cfg.bot_chat):
                log("INFO", "telegram bot control inactive: %s",
                    "no bot token" if not cfg.bot_token else "no allow-listed chat")
                return

            self.bot_transport = BotApiTransport(
                cfg.bot_token,
                chat_id=cfg.bot_chat,
                request_timeout_s=cfg.bot_request_timeout_s,
                max_retries=cfg.bot_max_retries,
                retry_backoff_s=cfg.bot_retry_backoff_s,
                retry_backoff_max_s=cfg.bot_retry_backoff_max_s,
            )
            self.bot_notifier = TelegramNotifier(self.bot_transport)
            self.bot_control = BotControl(
                transport=self.bot_transport,
                handle_button=self.handle_button,
                allowed_chat=cfg.bot_chat,
                on_refused=self._on_bot_update_refused,
            )
            # Attach the notifier through the EXISTING hook.
            self.set_notifier(self.bot_notifier.send)
            log("INFO", "telegram bot control ready (chat allow-listed)")
        except Exception as exc:  # noqa: BLE001 - never block PAPER startup
            log("ERROR", "telegram bot control unavailable (%s): %s",
                type(exc).__name__, exc)
            self.bot_transport = None
            self.bot_notifier = None
            self.bot_control = None

    def _on_bot_update_refused(self, why: str, update: Mapping[str, Any]) -> None:
        """
        Record a refused inbound update.

        Journaled WITHOUT the payload: an unauthorized update is attacker-
        controlled, and writing its text into the journal would let a stranger
        write arbitrary content into an operator-facing record. Only our own
        reason and the update id are kept.
        """
        try:
            self.ledger.note(
                "telegram update refused", reason=why,
                update_id=update.get("update_id") if isinstance(update, Mapping) else None,
            )
        except Exception:  # noqa: BLE001
            pass

    def start_bot_control(self) -> None:
        """
        Begin polling Telegram. Called from the run loop, never from __init__.

        Returns the created task, or ``None`` when the channel is not
        configured. Starting the poller does NOT start the trading loop: that
        still requires the operator's START button, which is the existing
        ``ButtonHandler`` path.
        """
        if self.bot_control is None:
            return None
        self.bot_notifier.start()
        stop = self._stop_event or asyncio.Event()
        self._stop_event = stop
        self._bot_task = asyncio.ensure_future(
            self.bot_control.run(
                stop, poll_timeout_s=self.config.telegram.bot_poll_timeout_s)
        )
        return self._bot_task

    def bot_status(self) -> dict[str, Any]:
        """
        Sanitised transport diagnostics for status output and the dashboard.

        Booleans and counters only. The token and the chat id are never included;
        the chat appears as ``chat_authorized`` because knowing the allow-list
        would help an attacker target it, and it is not needed to operate.
        """
        out: dict[str, Any] = {
            "transport_initialized": self.bot_transport is not None,
            "control_ready": self.bot_control is not None,
            "notifier_attached": self.bot_notifier is not None,
            "token_configured": bool(self.config.telegram.bot_token),
            "chat_configured": bool(self.config.telegram.bot_chat),
        }
        # NESTED, not flattened. Flattening let BotStatus's own
        # ``notifier_attached`` field silently overwrite the engine's correct
        # value, because a dict update has no idea two layers meant different
        # things by the same key. Namespacing removes that whole class of bug.
        if self.bot_control is not None:
            out["control"] = self.bot_control.status_dict()
        if self.bot_notifier is not None:
            out["notifier"] = self.bot_notifier.status_dict()
        out["forward_validation_active"] = self.forward is not None
        # Narrative health, so a silent outage is measurable rather than a
        # quiet 0.05 reduction in signal_quality.
        out["narrative"] = self.nar.metrics.as_dict()
        out["narrative_timeout_s"] = self.config.nar.timeout_s
        out["latency_gate"] = self.stale_gate.stats()
        out["run_state"] = self.state.run_state.value
        out["trade_mode"] = self.state.trade_mode.value
        out["real_trading_enabled"] = self.state.real_trading_enabled
        return out

    def set_notifier(
        self, fn: Callable[[str, Any], None]
    ) -> None:
        """
        Attach the Telegram notifier.

        After this, mode changes, L25 decisions and trade alerts are pushed to
        the operator. Before it, the engine is fully functional and silent, which
        is what makes every one of its paths testable without a chat.

        The supplied callable is WRAPPED, so every internal call site is
        protected by construction rather than by remembering a try/except at
        each one. That matters now that a notifier can actually fail: before the
        Bot API existed ``_notify`` was a no-op lambda that could not raise, and
        only the button-reply path was guarded. A notification failure must
        never propagate into a decision, so the guard lives here.
        """
        self._notifier = fn
        self._notify = self._guarded_notify

    def _guarded_notify(self, text: str, keyboard: Any = None) -> None:
        """
        Deliver a notification. A failure is logged and swallowed.

        Notification timing is measured HERE and reported separately from
        trading, so a 5-second Telegram send can never be mistaken for a
        5-second trading decision. The enqueue returns immediately (the
        notifier owns a bounded queue), so this call is not on the trading path
        even though it is timed.
        """
        stages = self._current_stages
        if stages is not None:
            stages.mark("notification_enqueue", self.clock.now())
        try:
            self._notifier(text, keyboard)
        except Exception as exc:  # noqa: BLE001 - never fail trading on Telegram
            log("ERROR", "cannot deliver a notification (%s): %s",
                type(exc).__name__, exc)
        finally:
            if stages is not None:
                stages.mark("notification_sent", self.clock.now())

    async def handle_button(self, data: str):
        """Dispatch one operator button press. See ButtonHandler."""
        result = await self.buttons.handle(data)
        try:
            self._notify(result.text, result.keyboard)
        except Exception as exc:  # noqa: BLE001 - never fail a press on a send
            log("ERROR", "cannot deliver the button reply: %s", exc)
        return result

    # ------------------------------------------------------------------
    # AMARIH (PumpView) paired-exit PAPER experiment
    # ------------------------------------------------------------------
    #
    # THE ROUTING DECISION IS MADE HERE, AT THE SHARED ENTRY BOUNDARY.
    #
    # Amarih is a PumpView feed, not a Telegram one, so selecting it has to
    # change WHICH FEED may open a position -- not merely which label the menu
    # shows. Putting the check in the UI would leave the Telegram channel's
    # entry path intact and only hide it, and a stale operator surface, a
    # replayed callback or a direct call would all bypass the restriction. So
    # the rule lives in :meth:`on_alert`, the one function every inbound signal
    # passes through, and it is enforced by RETURNING before any candidate is
    # recorded, analysed or sized.

    def _amarih_selected(self) -> bool:
        """
        Is the Amarih paired-exit experiment currently selected in PAPER?

        PAPER ONLY, and answered from the SELECTION rather than from the bridge,
        because the selection is the operator's intent and the bridge's arm list
        lags it by one reconcile.
        """
        if self.state.trade_mode is not TradeMode.PAPER:
            return False
        from .paper_live.arms import AMARIH_ARMS

        return {a.value for a in AMARIH_ARMS} <= set(self.state.arms)

    def _amarih_running(self) -> bool:
        """Selected AND the runtime is actually in its PAPER run state."""
        return self._amarih_selected() and self.state.run_state is RunState.PAPER

    def _amarih_state_path(self) -> Path:
        return self.config.var_dir / "amarih_runtime.json"

    def _build_amarih_runtime(self) -> Any:
        """
        Construct the driver over the EXISTING PAPER quote stack.

        The two quote coroutines below are adapters onto ``_quote_buy_best`` and
        ``quote_sell`` -- the same collectors the L25 arms and the V2 FLOOR
        experiment use. No second execution engine, no second quote service and
        no second ledger is introduced; the experiment owns its own three $10
        virtual accounts and nothing else.
        """
        from .paper_live.amarih_experiment import (
            DEFAULT_ARM_NOTIONAL_USD,
            AmarihExperiment,
        )
        from .paper_live.amarih_runtime import (
            DEFAULT_POLL_INTERVAL_S,
            AmarihRuntime,
        )
        from .paper_live.pumpview import (
            DEFAULT_MAX_AGE_S,
            PumpViewAmarihSource,
        )

        research_dir = self.config.var_dir / "research" / "amarih_experiment"
        experiment = AmarihExperiment(
            arm_notional_usd=DEFAULT_ARM_NOTIONAL_USD,
            research_dir=research_dir,
        )
        base_url = os.environ.get("PUMPVIEW_BASE_URL") or None
        # The default is the ADAPTER'S declared default, not a second literal.
        #
        # This line carried the number 120.0 directly, while
        # ``pumpview.DEFAULT_MAX_AGE_S`` was the named source of the same
        # threshold. Two literals for one threshold is how they drifted apart,
        # and the operator's stated intent for the Amarih strategy is a 15 s
        # window. ``PUMPVIEW_AMARIH_MAX_AGE_S`` still overrides both.
        max_age = _env_float(
            "PUMPVIEW_AMARIH_MAX_AGE_S", DEFAULT_MAX_AGE_S)
        interval = _env_float(
            "PUMPVIEW_AMARIH_POLL_S", DEFAULT_POLL_INTERVAL_S)
        source = (
            PumpViewAmarihSource(max_age_s=max_age)
            if not base_url
            else PumpViewAmarihSource(base_url=base_url, max_age_s=max_age)
        )
        return AmarihRuntime(
            source=source,
            experiment=experiment,
            state_path=self._amarih_state_path(),
            buy_quote=self._amarih_quote_buy,
            sell_quote=self._amarih_quote_sell,
            notify=self._notify,
            poll_interval_s=interval,
        )

    def _sync_amarih_runtime(self) -> None:
        """
        Start or retire the ONE poller so it matches the current selection.

        Called from the monitor tick (inside the loop, so task creation is
        legal) and whenever the selection is re-derived. Deselecting stops new
        ingestion immediately by looping the poller out; the driver itself is
        KEPT, because an arm already inside its holding period still has a
        scheduled exit to honour and abandoning it would silently censor a
        position the operator never closed.
        """
        if not self._amarih_selected():
            self._stop_amarih_poller()
            # The SCHEDULER outlives the poller: an arm inside its holding
            # period still has a deadline to honour, and the runtime is kept
            # (see ``_stop_amarih_runtime_on_deselect``). Cancelling it here
            # would silently censor every open position on deselection.
            log("AMARIH", "experiment deselected: poller stopped")
            if self._amarih_runtime is not None:
                self._amarih_runtime.running = False
            self._stop_amarih_scheduler_if_nothing_open()
            return

        if self._amarih_runtime is None:
            try:
                self._amarih_runtime = self._build_amarih_runtime()
                restored = self._amarih_runtime.restore()
                log("AMARIH", "experiment initialized; restored %s arm(s), "
                    "%s seen signal id(s)",
                    restored.get("restored_arms"), restored.get("seen_signal_ids"))
            except Exception as exc:  # noqa: BLE001 - never break the runtime
                log("ERROR", "cannot initialize the Amarih experiment: %s: %s",
                    type(exc).__name__, exc)
                self._amarih_runtime = None
                return

        if self._stop_event is None or self._stop_event.is_set():
            return
        if self._amarih_task is not None and not self._amarih_task.done():
            return
        self._amarih_runtime.running = True
        self._amarih_task = asyncio.create_task(
            self._amarih_poller_loop(self._stop_event))
        # THE DEADLINE SCHEDULER, ON ITS OWN CADENCE.
        #
        # Due-exit detection previously happened only inside the general
        # monitor loop, whose interval is ``market.poll_interval_s`` (5 s) -- and
        # ``_monitor_interval`` selects its faster 3 s branch on
        # ``self.account.open_positions()`` / ``self.real_account``, which the
        # Amarih arms are NOT part of: they live in the experiment's own
        # VirtualAccounts. So an Amarih arm that was due was seen up to 5 s late
        # on average 2.5 s, plus whatever the rest of the tick cost -- the tick
        # also reconciles the bridge, monitors native positions and logs
        # capacity, none of which should hold up an exit.
        #
        # A 1 s cadence is chosen because it is well inside the shortest horizon
        # (60 s) and well inside a tolerable detection error, while making one
        # non-network call per second when nothing is due: ``tick()`` issues a
        # SELL quote only for an arm whose deadline has actually arrived, so an
        # empty book costs a timestamp comparison.
        self._amarih_scheduler_task = asyncio.create_task(
            self._amarih_deadline_loop(self._stop_event))
        log("AMARIH", "poller started (interval %.1fs, max signal age %.0fs)",
            self._amarih_runtime.poll_interval_s,
            getattr(self._amarih_runtime.source, "max_age_s", 0.0))
        log("AMARIH", "deadline scheduler started (cadence %.1fs)",
            self._amarih_scheduler_interval())

    #: Cadence of the Amarih DEADLINE scheduler, in seconds. Separate from every
    #: other poll interval in this file because it does not touch the network:
    #: it only looks for arms whose ``scheduled_deadline`` has passed. 1 s keeps
    #: scheduler-detection lateness inside the 2 s target on the shortest (60 s)
    #: horizon, and is bounded because a tick with nothing due does no I/O.
    AMARIH_DEADLINE_POLL_S = 1.0

    def _amarih_scheduler_interval(self) -> float:
        """The deadline scheduler cadence, env-overridable for a test."""
        return _env_float("AMARIH_DEADLINE_POLL_S", self.AMARIH_DEADLINE_POLL_S)

    async def _amarih_deadline_loop(self, stop: asyncio.Event) -> None:
        """
        Service the Amarih T60/T120/T300 deadlines on their OWN cadence.

        This loop exists because due-exit detection was coupled to the general
        monitor loop. It is now independent of:

        * incoming PumpView signals -- a due arm is settled whether or not
          another signal ever arrives;
        * ``self.account`` / ``self.real_account`` open positions, which do not
          contain Amarih arms, so the monitor's faster interval was never
          selected while the experiment was running;
        * the rest of the monitor tick, which reconciles the bridge and monitors
          native positions and can be arbitrarily slow.

        It does NOT poll the network. ``tick()`` fetches an executable SELL
        quote only for an arm that is already past its deadline, so a quiet book
        costs one comparison per second.
        """
        interval = self._amarih_scheduler_interval()
        mine = self._amarih_runtime
        while not stop.is_set():
            runtime = self._amarih_runtime
            if runtime is None:
                return
            # ``mine`` is the runtime this scheduler was created for. If the
            # selection rebuilt it, this scheduler is stale: it must exit rather
            # than keep servicing an experiment the runtime no longer holds --
            # otherwise two schedulers would share one book.
            if runtime is not mine:
                return
            # Only while the experiment is selected, OR while arms are still
            # outstanding. An arm already inside its holding period has a
            # scheduled exit to honour and abandoning it would silently censor a
            # position the operator never closed -- which is why the driver (and
            # this scheduler) outlive a deselection for as long as anything is
            # open.
            if not self._amarih_selected() and self._amarih_open_arms() == 0:
                return
            if self.state.run_state is RunState.PAPER:
                try:
                    await self._amarih_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - never kill the loop
                    runtime.last_error = (
                        f"deadline tick failed: {type(exc).__name__}")
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def _amarih_open_arms(self) -> int:
        """How many Amarih arms are outstanding, for the scheduler guard."""
        runtime = self._amarih_runtime
        if runtime is None:
            return 0
        try:
            return len(getattr(runtime, "_open_arms", None) or {})
        except Exception:  # noqa: BLE001 - a status guard must not raise
            return 0

    def _amarih_runtime_identity(self) -> Any:
        """
        The runtime object the currently-running scheduler belongs to.

        A selection change rebuilds the runtime; the scheduler holds a reference
        to the object it started with so it can notice it is no longer the live
        one instead of servicing a retired experiment.
        """
        return self._amarih_runtime

    def _amarih_scheduler_is_live(self) -> bool:
        task = getattr(self, "_amarih_scheduler_task", None)
        return task is not None and not task.done()

    def _stop_amarih_scheduler_if_nothing_open(self) -> None:
        """
        Stop the deadline scheduler ONLY when nothing can ever become due.

        The scheduler is kept alive while an arm is outstanding, so a deselected
        experiment still settles the positions it opened. Once the book is
        empty there is nothing left to schedule and the loop retires itself on
        its next pass; cancelling here is belt and braces.
        """
        if self._amarih_open_arms() == 0:
            self._stop_amarih_scheduler()

    def _stop_amarih_scheduler(self) -> None:
        """Cancel the deadline scheduler if it is running. Idempotent."""
        task = getattr(self, "_amarih_scheduler_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._amarih_scheduler_task = None

    def _stop_amarih_poller(self) -> None:
        """Cancel the PumpView poller if it is running. Idempotent."""
        task = getattr(self, "_amarih_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._amarih_task = None

    async def _amarih_poller_loop(self, stop: asyncio.Event) -> None:
        """
        Poll PumpView until deselected or stopped.

        Polling only happens while the runtime is in its PAPER run state, so
        starting the process does not begin ingesting signals before the
        operator has pressed START -- the same rule the Telegram feed follows.
        """
        runtime = self._amarih_runtime
        while (not stop.is_set() and runtime is not None
               and runtime is self._amarih_runtime and self._amarih_selected()):
            if self.state.run_state is RunState.PAPER:
                try:
                    await runtime.poll_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - one poll is not fatal
                    runtime.last_error = f"poll cycle failed: {type(exc).__name__}"
            try:
                await asyncio.wait_for(stop.wait(), timeout=runtime.poll_interval_s)
            except asyncio.TimeoutError:
                pass

    async def _amarih_tick(self) -> None:
        """Service the scheduled T60/T120/T300 exits. Never raises."""
        runtime = self._amarih_runtime
        if runtime is None:
            return
        try:
            await runtime.tick()
        except Exception as exc:  # noqa: BLE001 - a tick must not kill the loop
            runtime.last_error = f"exit tick failed: {type(exc).__name__}"

    async def _amarih_quote_buy(self, mint: str, usd: float) -> Any:
        """
        ONE executable BUY quote for a USD notional, on the existing stack.

        Returns ``None`` -- never a mark -- when no usable quote exists.
        """
        from .paper_live.execution_costs import (
            deployable_sol_lamports,
            outside_swap_costs,
        )
        from .paper_live.fills import FillModel, fill_from_quote

        bridge = await self._paper_bridge_instance()
        ref = getattr(bridge, "sol_reference", None)
        sol_usd = float(getattr(ref, "usd_per_sol", 0.0) or 0.0)
        if not sol_usd > 0:
            return None
        costs = outside_swap_costs(
            sol_usd=sol_usd,
            sol_usd_source=str(getattr(ref, "source", "") or ""),
            sol_usd_at=getattr(ref, "at", None),
        )
        lamports, _why = deployable_sol_lamports(
            available_usd=float(usd), sol_usd=sol_usd, costs=costs)
        if lamports <= 0:
            return None
        # ``token_decimals_for`` IS ASYNC in this runtime and the engine wires it
        # to ``DirectPumpReader.mint_decimals``. The beta path casts its result
        # with ``int(...)``; doing that here would raise on a coroutine, be
        # swallowed by the guard below, and silently price every token at 6
        # decimals -- a 10**decimals error in the entry size, invisible because
        # every individual number stays plausible. A reader may be sync or
        # async, so it is awaited only when it actually returns an awaitable.
        decimals = 6
        try:
            resolved = bridge.token_decimals_for(mint)
            if inspect.isawaitable(resolved):
                resolved = await resolved
            if resolved:
                decimals = int(resolved)
        except Exception as exc:  # noqa: BLE001 - unknown decimals falls back
            log("AMARIH", "could not read token decimals for %s: %s: %s",
                mint[:8], type(exc).__name__, exc)
        quote, _candidates = await bridge._quote_buy_best(
            mint=mint, sol_lamports=int(lamports), token_decimals=decimals,
            observed_pair_address=None)
        if quote is None or not getattr(quote, "available", False):
            return None
        if getattr(quote, "stale", False):
            return None
        fill = fill_from_quote(
            quote, side="BUY", model=FillModel(), sol_reference=ref,
            require_identity=bridge.require_identity)
        return {
            "identity": str(getattr(fill, "identity", "UNKNOWN")),
            "token_quantity": float(fill.output_amount_decimal),
            "token_quantity_raw": int(fill.output_amount_raw),
            "token_decimals": int(fill.output_decimals),
            "input_amount_usd": float(fill.input_amount_decimal) * sol_usd,
            "fees_usd": None,
            "provider": getattr(quote, "provider", None),
            "venue": getattr(quote, "venue", None),
            "market_id": getattr(quote, "pool_or_curve", None),
            "liquidity_usd": None,
            "quote_ts": float(fill.quote_timestamp),
            "quote_asset": "SOL",
            "raw": {},
        }

    async def _amarih_quote_sell(self, mint: str, tokens_raw: int,
                                 decimals: int, entry_pool: str | None = None
                                 ) -> Any:
        """
        A fresh executable SELL quote for the EXACT quantity acquired.

        ``entry_pool`` is the market the entry quote traded in. It is passed
        through as the quote's ``observed_pair_address`` so same-market identity
        is answered by comparing against a KNOWN address for this token.

        THIS ARGUMENT IS WHY THE EXPERIMENT SETTLES AT ALL. It used to be
        hard-coded ``observed_pair_address=None``, and
        ``QuoteService._identity_for`` returns ``UNKNOWN`` outright when there is
        no address to compare against. ``fill_from_quote`` then refuses the fill
        on ``require_identity``, and the refusal was swallowed by the caller's
        broad handler, so every arm was recorded as
        ``QUOTE_UNAVAILABLE``/``CENSORED`` forever -- an unsettled experiment
        that looked like a run of bad luck. The identical defect is documented,
        and fixed, on the beta exit path in ``paper_live.engine``.
        """
        from .paper_live.fills import FillModel, fill_from_quote

        bridge = await self._paper_bridge_instance()
        ref = getattr(bridge, "sol_reference", None)
        sol_usd = float(getattr(ref, "usd_per_sol", 0.0) or 0.0)
        if not sol_usd > 0:
            return None
        quote = None
        if entry_pool:
            # ``quote_sell_pinned`` RETURNS A PAIR: ``(chosen, counterfactual)``.
            # Reading the pair itself as the quote makes
            # ``getattr(quote, "available", False)`` False on every call, which
            # is the same class of bug this path already suffered once on beta.
            pinned = getattr(bridge.quote_service, "quote_sell_pinned", None)
            if pinned is not None:
                result = await pinned(
                    mint=mint, token_amount_raw=int(tokens_raw),
                    token_decimals=int(decimals),
                    pinned_venue_id=None, pinned_pool_address=entry_pool,
                    observed_pair_address=entry_pool)
                quote = (result[0] if isinstance(result, (tuple, list))
                         else result)
        if quote is None:
            quote = await bridge.quote_service.quote_sell(
                mint=mint, token_amount_raw=int(tokens_raw),
                token_decimals=int(decimals),
                observed_pair_address=entry_pool or None)
        if quote is None or not getattr(quote, "available", False):
            return None
        if getattr(quote, "stale", False):
            return None
        fill = fill_from_quote(
            quote, side="SELL", model=FillModel(), sol_reference=ref,
            require_identity=bridge.require_identity)
        return {
            "identity": str(getattr(fill, "identity", "UNKNOWN")),
            # On a SELL the token is the INPUT and SOL is the OUTPUT.
            "proceeds_usd": float(fill.output_amount_decimal) * sol_usd,
            "fees_usd": None,
            "provider": getattr(quote, "provider", None),
            "venue": getattr(quote, "venue", None),
            "market_id": getattr(quote, "pool_or_curve", None),
            "quote_ts": float(fill.quote_timestamp),
            "raw": {},
        }

    def _amarih_status_fields(self) -> dict[str, Any]:
        """
        The Amarih block of the status view.

        ``amarih_enabled`` follows the SELECTION, so the operator sees the
        experiment's state as soon as they select it -- a status block that only
        appears once a poll has succeeded would look like a broken selector.
        """
        selected = self._amarih_selected()
        runtime = self._amarih_runtime
        fields: dict[str, Any] = {
            "amarih_enabled": selected,
            "amarih_poller_running": bool(
                runtime is not None and runtime.status()["running"]
                and self._amarih_task is not None
                and not self._amarih_task.done()),
            # The deadline scheduler is reported SEPARATELY. It is the thing that
            # decides whether an exit fires on time, so "the poller is running"
            # is not a sufficient health signal -- an operator reading only that
            # would believe exits were being serviced when only entries were.
            "amarih_scheduler_running": self._amarih_scheduler_is_live(),
            "amarih_scheduler_cadence_s": self._amarih_scheduler_interval(),
            "amarih_signal_max_age_s": float(
                getattr(runtime.source, "max_age_s", 0.0) or 0.0)
            if runtime is not None else 0.0,
        }
        if runtime is None:
            return fields
        status = runtime.status()
        fields.update(
            amarih_last_poll_at=status["last_poll_at"],
            amarih_signals_seen=status["signals_seen"],
            amarih_signals_accepted=status["signals_accepted"],
            amarih_signals_rejected=status["signals_rejected"],
            amarih_open_arms=status["open_arms"],
        )
        return fields

    def status_view(self) -> StatusView:
        """
        Assemble the status view from the canonical state.

        Every field is read from ``self.state`` or from an account object, so
        this is the single place the UI's facts are produced and the CLI and
        Telegram necessarily agree.

        EQUITY IS MODE-AUTHORITATIVE. This used to read
        ``self.account.equity(...)`` unconditionally -- and ``self.account`` is
        the PAPER ledger -- so a REAL status screen displayed the PAPER
        account's number ($727.3455 on 2026-10-01, exactly the PAPER cash).
        That is not a stale read or a rounding difference: it is a different
        account's money, shown on the screen that says REAL.

        In REAL it now comes from :meth:`real_equity`, which reads the configured
        wallet across every supported asset. ``real_equity`` is async and this is
        sync, so the async path CACHES its authoritative result here and this
        reads the cache. An absent or unavailable cache yields ``None``, which
        renders as "unavailable" instead of inventing a balance.
        """
        snap = self.real_account.wallet_snapshot
        if self.state.trade_mode is TradeMode.REAL:
            equity = (self._real_equity_cache.equity_usd
                      if self._real_equity_cache is not None else None)
        else:
            equity = self.account.equity(self._refresh_marks())
        return StatusView(
            state=self.state,
            channel=self.config.telegram.CANDIDATE_SOURCE,
            chat=self.config.telegram.chat,
            wallet_masked=snap.masked_address if snap else "(not configured)",
            balance_sol=snap.sol if snap else None,
            balance_usd=snap.usd if snap else None,
            equity_usd=equity,
            cash_usd=self.account.cash,
            open_positions=(
                len(self.account.open_positions())
                + len(self.real_account.open_positions())
            ),
            executor_available=self.real_capable(),
            real_preflight_ok=(
                getattr(self.real_preflight, "ok", None)
                if self.real_preflight is not None
                else None
            ),
            emergency_stop=self.state.emergency_stop,
            signal_source=self.signal_source_selection,
            **self._amarih_status_fields(),
        )

    def open_order_rows(self) -> list[dict[str, Any]]:
        """
        Currently open positions, from BOTH accounts, each labelled with its own
        mode. Never summed.
        """
        now = self.clock.now()
        rows: list[dict[str, Any]] = []
        for p in self.account.open_positions():
            rows.append(
                {
                    "mode": TradeMode.PAPER.value,
                    "symbol": p.symbol or "?",
                    "mint": p.mint,
                    "entry_price": p.entry_reference_price or p.fill_price,
                    "current_price": p.current_price,
                    "market_cap_usd": p.market_cap_at_entry,
                    "size_usd": p.cash_out_usd,
                    "pnl_usd": p.unrealized_pnl_usd,
                    "pnl_pct": p.current_return_pct,
                    "mfe_pct": p.mfe_pct,
                    "mae_pct": p.mae_pct,
                    "hold_s": max(0.0, now - (p.entry_timestamp or now)),
                    "exit_state": "OPEN",
                    "signature": None,
                    "confirmed": None,
                }
            )
        for p in self.real_account.open_positions():
            rows.append(
                {
                    "mode": TradeMode.REAL.value,
                    "symbol": p.symbol,
                    "mint": p.mint,
                    "entry_price": p.entry_price_usd,
                    "current_price": p.current_price_usd,
                    "market_cap_usd": None,
                    "size_usd": p.size_usd,
                    "pnl_usd": p.unrealized_usd(),
                    "pnl_pct": p.return_pct(),
                    "mfe_pct": p.mfe_pct,
                    "mae_pct": p.mae_pct,
                    "hold_s": p.holding_seconds(now),
                    "exit_state": p.status,
                    "signature": p.entry_signature,
                    "confirmed": p.confirmed,
                }
            )
        return rows

    def statistics_by_mode(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Statistics, PAPER and REAL, kept apart.

        REAL statistics come from real closed positions only, so a real
        statistics block that is empty is an honest "no real trades yet" rather
        than a simulation's numbers wearing a real label.
        """
        # `ledger.trades()` yields raw dicts; the statistics need Trade objects.
        # Reconstructing them here keeps the ledger append-only and keeps
        # `compute` a pure function of a trade list.
        from .engine import _trade_from  # self-reference, resolved at call time

        closed = [
            tr for tr in (_trade_from(r) for r in self.ledger.trades())
            if tr is not None
        ]
        # Scope to the current statistics epoch when the account has one.
        #
        # The ledger is append-only and is also the frozen research universe, so
        # a reset must not truncate it. Filtering HERE moves only the reporting
        # window: a reset account reports its own trades, and the previous
        # account's history remains in the ledger, complete and auditable.
        epoch = self.account.state.stats_epoch_started_at
        if epoch is not None:
            closed = [
                tr for tr in closed
                if (tr.exit_timestamp or 0.0) >= epoch
            ]
        paper = compute_stats(
            closed,
            starting_equity=self.config.risk.starting_bankroll_usd,
            current_equity=self.account.equity(self._refresh_marks()),
        ).as_dict()
        paper["equity_usd"] = round(
            self.account.equity(self._refresh_marks()), 6
        )
        paper["cash_usd"] = round(self.account.cash, 6)
        # Canonical PnL key, so the stats renderer reads ONE name for both
        # accounts. Without this the PAPER block rendered a blank Total PnL
        # while the REAL block showed a number, which reads as "no paper
        # activity" rather than "two different key names".
        paper["total_pnl_usd"] = paper["realized_pnl_usd"]

        # The execution-aware PAPER block, appended rather than merged.
        #
        # It is kept SEPARATE from the block above on purpose. The block above
        # is marked-to-market accounting over the legacy PAPER book; this one is
        # simulated-settlement accounting over the research arms. Overwriting
        # one with the other would make an operator unable to tell which
        # accounting basis produced the number in front of them, and the whole
        # point of the four-layer model is that they never blur.
        execution_aware = self.paper_bridge_status()
        if execution_aware is not None:
            paper["execution_aware_paper"] = {
                "enabled": True,
                "mode": execution_aware["mode"],
                "arms": execution_aware["arms"],
                "signals": execution_aware["signals_seen"],
                "trades": execution_aware["trades"],
                "rejected_opportunities":
                    execution_aware["rejected_opportunities"],
                "quote_failures": execution_aware["quote_failures"],
                "quote_provider": execution_aware["quote_provider"],
                "latest_quote_age_ms": execution_aware["latest_quote_age_ms"],
                "actual_fill_available":
                    execution_aware["actual_fill_available"],
                "statement": execution_aware["statement"],
                "realized_pnl_basis": execution_aware["realized_pnl_basis"],
                # The V2 FLOOR (BETA) experiment, verbatim from its own three
                # ledgers (``None`` when no experiment is running). Kept inside
                # the exec-aware block because it IS exec-aware PAPER -- and
                # never merged into the legacy account above, which would blur
                # two accounting bases into one number.
                "v2_floor_beta": execution_aware.get("v2_floor_beta"),
            }
        else:
            paper["execution_aware_paper"] = {
                "enabled": bool(
                    getattr(self.config.real, "paper_execution_aware_enabled",
                            False)),
                "mode": "PAPER (execution-aware; awaiting the first signal)",
                "arms": {},
            }

        closed = [
            p for p in self.real_account.positions.values() if not p.is_open
        ]
        pnl = [p.realized_pnl_usd or 0.0 for p in closed]
        wins = [x for x in pnl if x > 0]
        losses = [x for x in pnl if x < 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        real: dict[str, Any] = {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(closed)) if closed else 0.0,
            "profit_factor": (
                (gross_win / gross_loss)
                if gross_loss > 0
                else (float("inf") if gross_win > 0 else 0.0)
            ),
            "expectancy_usd": (sum(pnl) / len(pnl)) if pnl else 0.0,
            "total_pnl_usd": sum(pnl),
            "max_drawdown_pct": self._real_max_drawdown(closed),
            "avg_winner_usd": (gross_win / len(wins)) if wins else 0.0,
            "avg_loser_usd": (-gross_loss / len(losses)) if losses else 0.0,
            "runners": sum(1 for p in closed if (p.mfe_pct or 0) >= 50),
            "runner_capture_pct": 0.0,
            "equity_usd": self.real_account.equity_usd(),
            "cash_usd": (
                self.real_account.wallet_snapshot.sol
                if self.real_account.wallet_snapshot
                else None
            ),
        }
        return paper, real

    @staticmethod
    def _real_max_drawdown(closed: list[Any]) -> float:
        if not closed:
            return 0.0
        equity = 0.0
        peak = 0.0
        worst = 0.0
        for p in sorted(closed, key=lambda x: x.exit_timestamp or 0.0):
            equity += p.realized_pnl_usd or 0.0
            peak = max(peak, equity)
            if peak > 0:
                worst = min(worst, (equity - peak) / peak * 100.0)
        return worst

    # ------------------------------------------------------------------
    # startup
    # ------------------------------------------------------------------

    def startup(self, *, adopt_legacy_account: bool = False) -> dict[str, Any]:
        """
        Arm the engine.

        Ownership is taken FIRST, before the safety audit, before ``_load_state``
        and before any ledger append.  ``_load_state`` writes a ``recovery``
        event and ``startup`` bumps ``run_count``, so anything that mutates state
        happens strictly after the lock is held.  A second process pointed at the
        same account is refused here and therefore never reaches the alert,
        decision, sizing or exit paths.

        STARTUP IS ALWAYS SAFE
        ---------------------
        This method does NOT start the trading loop and does NOT arm REAL. On
        return the process is:

            RUNNING = false
            MODE     = PAPER
            REAL     = false

        Candidate processing begins only when the operator presses start, and
        REAL only on a confirmed press plus a passing preflight.
        """
        from .account_lock import assert_no_legacy_writer

        # Withhold before taking the lock: a pre-lock process cannot be
        # arbitrated by flock, so arming while one may still be writing would
        # recreate the exact race this closes.
        self._legacy_report = assert_no_legacy_writer(
            self.config.state_file, allow_legacy_adopt=(
                adopt_legacy_account or self.adopt_legacy_account))
        self._acquire_ownership()
        try:
            report = audit()
        except BaseException:
            self._release_ownership()
            raise
        self._safety_report = report
        log("SAFETY", "migration audit: %s", "PASS" if report.ok else "FAIL")
        if not report.ok:
            self._release_ownership()
            raise RuntimeError("refusing to arm: " + report.render())

        # Belt and braces: prove the invariant the whole design rests on. A
        # fresh process must never find itself armed, whatever the environment
        # said.
        if self.state.real_trading_enabled:
            self._release_ownership()
            raise RuntimeError(
                "refusing to arm: the runtime state reported REAL at startup"
            )
        if self.state.run_state is not RunState.OFF:
            self._release_ownership()
            raise RuntimeError(
                "refusing to arm: the runtime state started in "
                + self.state.run_state.value
                + ", not OFF"
            )
        if self.config.real.env_trading_enabled_request:
            log(
                "SAFETY",
                "REAL_TRADING_ENABLED=true is present in the environment and "
                "is being IGNORED for authorization; REAL stays off until an "
                "operator arms it from Telegram",
            )

        log("ACCOUNT", "account %s owned by pid %d (%s)",
            self._account_id, os.getpid(), self._lock_path)
        log(
            "MODE",
            "startup complete: RUNNING=false MODE=PAPER REAL=false "
            "(awaiting the operator start button)",
        )
        self._load_state()
        self.account.state.run_count += 1
        # `self.running` is the ENGINE loop flag and stays False: the process is
        # initialized and listening, but not trading. `state.run_state` is the
        # canonical answer and is OFF.
        self.running = False
        return report.as_dict()

    # -- single-writer ownership -----------------------------------------

    def _acquire_ownership(self) -> None:
        from .account_lock import acquire

        if self._lock is not None and not self._lock.released:
            self._lock._refs += 1
            return
        self._lock = acquire(self.config.state_file, kind="paper-engine")
        self._account_id = self._lock.account_id
        self._lock_path = str(self._lock.identity.lock_path)

    def _release_ownership(self) -> None:
        from .account_lock import release

        if self._lock is not None:
            log("ACCOUNT", "account %s released by pid %d", self._account_id, os.getpid())
            release(self._lock)
            self._lock = None

    @property
    def account_id(self) -> str | None:
        return self._account_id

    @property
    def owns_account(self) -> bool:
        return self._lock is not None and not self._lock.released

    def _require_owner(self, what: str) -> None:
        """
        The single ownership boundary.

        Every durable mutation in this engine passes through here.  It is a
        cheap in-process check against a lock this module owns, not a second
        lock, so it cannot deadlock against anything.
        """
        if not self.owns_account:
            from .account_lock import AccountOwnershipError

            raise AccountOwnershipError(
                f"{what} refused: this process does not own the paper account "
                f"({self._account_id or 'not acquired'}). A second writer must "
                f"never mutate account state."
            )

    def _load_state(self) -> None:
        state_raw = load_json(self.config.state_file)
        if state_raw:
            self._assert_same_account(state_raw)
            state = PaperAccountState.from_dict(state_raw.get("account"))
            self.account = PaperAccount(state, self.config.risk)
            self.account.positions = _positions_from(
                state_raw.get("open_positions") or []
            )
            self.account.recently_exited = {
                k: float(v) for k, v in (state_raw.get("recently_exited") or {}).items()
            }
            self.account.unfilled_attempts = int(state_raw.get("unfilled_attempts", 0))
            for mint, price in (state_raw.get("first_valid_prices") or {}).items():
                self.market.adopt_first_valid_price(mint, price)
            for entry in (state_raw.get("seen_messages") or []):
                if len(entry) == 2:
                    self._seen_messages.add((str(entry[0]), int(entry[1])))
            for cid, ts in (state_raw.get("seen_candidates") or {}).items():
                self._seen_candidates[str(cid)] = float(ts)
            for mint, entries in (state_raw.get("seen_opportunities") or {}).items():
                self._seen_opportunities[str(mint)] = [
                    (float(item[0]), str(item[1])) for item in entries
                    if isinstance(item, (list, tuple)) and len(item) == 2
                ]
            self.signal_source = SignalSourceState.from_dict(
                state_raw.get("signal_source")
            )
            self.signal_source.refresh(self.clock.now())
            self._signal_source_counters.update(
                {str(k): int(v) for k, v in
                 (state_raw.get("signal_source_counters") or {}).items()}
            )
            for trade_raw in self.ledger.trades():
                trade = _trade_from(trade_raw)
                if trade is not None:
                    self.memory.remember(trade)
            self.recovered = True
            log(
                "RECOVERY",
                "restored bankroll: start=$%.2f cash=$%.2f realized=%+.2f "
                "open=%d closed=%d run#%d",
                state.starting_bankroll_usd, state.cash_usd, state.realized_pnl_usd,
                len(self.account.open_positions()), len(self.memory),
                state.run_count,
            )
        else:
            state = PaperAccountState(
                starting_bankroll_usd=self.config.risk.starting_bankroll_usd,
                cash_usd=self.config.risk.starting_bankroll_usd,
                created_at=self.clock.now(),
                updated_at=self.clock.now(),
                peak_equity_usd=self.config.risk.starting_bankroll_usd,
            )
            self.account = PaperAccount(state, self.config.risk)
            self.recovered = False
            log(
                "ACCOUNT",
                "new paper account %s: $%.2f simulated bankroll",
                self._account_id, state.starting_bankroll_usd,
            )
        self._build_book()
        try:
            rec = self.reconciliation()
            if rec.get("status") not in ("PASS", "OK"):
                boundary_at = self.clock.now()
                self._active_accounting_since = boundary_at
                # Preserve the historical ledger exactly, but make the active
                # account's statistics epoch begin at the migration boundary.
                # This is an in-memory boundary until the normal durable
                # checkpoint; it never edits or deletes a historical trade.
                self.account.state.stats_epoch_started_at = boundary_at
                self._historical_accounting_boundary = {
                    "scope": "ACTIVE_NEW_EXECUTION_AWARE_ONLY",
                    "historical_unreconciled": True,
                    "reason": rec.get("status"),
                    "drift_usd": rec.get("drift_usd"),
                    "boundary_at": boundary_at,
                    "historical_trade_count": len(self.memory),
                }
        except Exception as exc:  # noqa: BLE001
            self._historical_accounting_boundary = {
                "scope": "ACTIVE_NEW_EXECUTION_AWARE_ONLY",
                "historical_unreconciled": True,
                "reason": f"RECONCILIATION_UNAVAILABLE:{type(exc).__name__}",
            }
        self.ledger.event(
            EventType.RECOVERY,
            {
                "recovered": self.recovered,
                "run_count": self.account.state.run_count,
                "open_positions": len(self.account.open_positions()),
                "closed_trades": len(self.memory),
                "agent": AGENT_ID,
                "account_id": self._account_id,
                "owner_pid": os.getpid(),
            },
        )

    def _assert_same_account(self, state_raw: dict[str, Any]) -> None:
        """
        Refuse a state file that belongs to a different account identity.

        This catches the "same account reached through a different path or root"
        mistake at startup, before a single trade is booked.  A file with no
        recorded identity is legacy and is adopted silently.
        """
        recorded = state_raw.get("account_id")
        if recorded and recorded != self._account_id:
            self._release_ownership()
            raise RuntimeError(
                f"{self.config.state_file} is stamped for account {recorded} but "
                f"this process resolved it to {self._account_id}. Two different "
                f"identities for one state file means the account is being "
                f"reached by two paths; refusing rather than trading into it."
            )


    def _build_book(self) -> None:
        from .positions import PositionBook

        self.open_book = {
            "positions": PositionBook(
                self.account, self.ledger, self.config.exits, self.config.risk
            )
        }

    # ------------------------------------------------------------------
    # alert handling
    # ------------------------------------------------------------------

    async def on_alert(self, snapshot: AlertSnapshot, parsed) -> None:
        source = snapshot.signal_source or PRIMARY
        now = self.clock.now()
        selection = self.signal_source_selection

        # THE AMARIH SOURCE RESTRICTION, AT THE SHARED BOUNDARY.
        #
        # While AMARIH is selected the ONLY feed allowed to open a position is
        # PumpView. The Telegram channel still connects, still logs and is still
        # recorded as a candidate -- but it cannot enter, whatever the operator
        # did before, whichever callback is replayed, and whichever panel is
        # stale. The check is on the SOURCE of this snapshot, so it cannot be
        # bypassed by a UI that has not caught up.
        if source == SOURCE_AMARIH:
            # AN AMARIH SIGNAL IS NEVER TRADED THROUGH THIS BOUNDARY.
            #
            # The experiment owns its own ingestion: its poller prices one
            # entry observation and fans it into the three paired arms. Running
            # an Amarih signal through here instead would put it in front of
            # L25's entry filters and the per-arm sizing path, changing the
            # signal population the experiment is measuring -- which is the one
            # thing the Amarih gate must not inherit. So this returns, whether
            # or not AMARIH is selected, and the poller is the only way in.
            self._signal_source_counters["amarih_routed_to_experiment"] = (
                self._signal_source_counters.get(
                    "amarih_routed_to_experiment", 0) + 1)
            log("SIGNAL_SOURCE",
                "Amarih alert handled by the experiment poller, not this pipeline")
            return

        # KEYED ON THE ARM SELECTION, NOT ONLY ON THE SOURCE PREFERENCE.
        #
        # This used to test ``signal_source_selection == SOURCE_AMARIH`` alone.
        # That value lives in ``var/signal_source_selection.json`` and is set
        # only by the SIGNAL SOURCE submenu -- whose whitelist is
        # ``{MooDeng, PainApe, AUTO}``. "Amarih" is not selectable there, and no
        # code anywhere derives this field from the arm selection. So selecting
        # the Amarih experiment from the STRATEGY menu, which is exactly how an
        # operator selects it, left this guard disarmed and Telegram channel
        # alerts kept driving the L25 pipeline: the persisted selection said
        # AMARIH while the runtime traded L25, which is the exact disagreement
        # the restriction exists to prevent.
        #
        # Both conditions now hold it open. Selecting the ARMS is the operator's
        # statement of intent, so that is authoritative; the source preference
        # is kept so an operator who pinned the source explicitly still gets the
        # restriction. Neither can be bypassed by a replayed callback.
        if (selection == SOURCE_AMARIH or self._amarih_selected()) \
                and source != SOURCE_AMARIH:
            self._signal_source_counters["amarih_blocked_telegram"] = (
                self._signal_source_counters.get(
                    "amarih_blocked_telegram", 0) + 1)
            # The reason vocabulary has ONE home; importing it here keeps
            # the journal, the log and any future operator surface from
            # drifting apart over a spelling.
            from .paper_live.amarih_runtime import SOURCE_DISABLED_BY_AMARIH

            self.ledger.note(
                "signal blocked: AMARIH is the selected source",
                chat=snapshot.chat, message_id=snapshot.message_id,
                mint=snapshot.mint, signal_source=source,
                reason=SOURCE_DISABLED_BY_AMARIH)
            log("SIGNAL_SOURCE",
                "%s alert blocked: %s", source, SOURCE_DISABLED_BY_AMARIH)
            self._record_candidate(snapshot, parsed, traded=False)
            return
        # A valid primary parse is the AUTO heartbeat even when the operator is
        # temporarily forcing PainApe. Manual selection changes eligibility,
        # not the definition of a valid primary signal.
        if source == PRIMARY:
            self._signal_source_counters["primary_valid"] += 1
            transition = self.signal_source.observe_primary(
                at=snapshot.received_timestamp or now,
                message_id=snapshot.message_id,
            )
            if transition:
                log("SIGNAL_SOURCE", "transition -> %s", transition)
        if selection == SOURCE_MOODENG and source != PRIMARY:
            self._signal_source_counters["fallback_ignored_primary_active"] += 1
            log("SIGNAL_SOURCE", "PainApe alert ignored; manual source=MooDeng")
            return
        if selection == SOURCE_PAINAPE and source != FALLBACK:
            self._signal_source_counters["fallback_ignored_primary_active"] += 1
            log("SIGNAL_SOURCE", "MooDeng alert ignored; manual source=PainApe")
            return
        if source == PRIMARY:
            pass
        elif source == FALLBACK:
            self._signal_source_counters["fallback_valid"] += 1
            transition = self.signal_source.refresh(now)
            if transition:
                log("SIGNAL_SOURCE", "transition -> %s", transition)
            if selection == SOURCE_AUTO and not self.signal_source.accepts(FALLBACK, now):
                self._signal_source_counters["fallback_ignored_primary_active"] += 1
                log("SIGNAL_SOURCE", "fallback alert ignored while %s",
                    self.signal_source.active_source)
                return
        else:
            log("SIGNAL_SOURCE", "unknown source %r ignored", source)
            return

        # One underlying mint opportunity is owned by the earliest valid alert,
        # even when the two feeds report it independently.
        if snapshot.mint:
            prior = self._seen_opportunities.setdefault(snapshot.mint, [])
            prior[:] = [(ts, src) for ts, src in prior
                        if now - ts <= OPPORTUNITY_WINDOW_S]
            if any(src != source for _, src in prior):
                self._signal_source_counters["cross_source_duplicates"] += 1
                self.ledger.note("cross-source opportunity duplicate suppressed",
                                 mint=snapshot.mint, signal_source=source)
                return
            prior.append((now, source))
        if self.control.command == MODE_STOP:
            log("CONTROL", "STOP active: alert %s recorded but not traded",
                snapshot.message_id)
            self._record_candidate(snapshot, parsed, traded=False)
            return

        await self._handle_candidate(snapshot, parsed)

    def _record_candidate(self, snapshot: AlertSnapshot, parsed, traded: bool) -> None:
        self.candidates_seen += 1
        self._seen_candidates[snapshot.candidate_id] = snapshot.received_timestamp
        key = (snapshot.chat, snapshot.message_id)
        if key in self._seen_messages:
            self.ledger.note(
                "duplicate alert suppressed", chat=snapshot.chat,
                message_id=snapshot.message_id,
            )
            self.candidates_seen -= 1
            return
        self._seen_messages.add(key)
        self.ledger.candidate_seen(
            {
                "candidate_id": snapshot.candidate_id,
                "chat": snapshot.chat,
                "chat_id": self.listener.status.chat_id if self.listener else None,
                "signal_source": snapshot.signal_source,
                "message_id": snapshot.message_id,
                "source_timestamp": snapshot.source_timestamp,
                "source_time_iso": iso(snapshot.source_timestamp),
                "received_timestamp": snapshot.received_timestamp,
                "received_time_iso": iso(snapshot.received_timestamp),
                "transit_latency_ms": (
                    round((snapshot.received_timestamp - snapshot.source_timestamp) * 1000, 2)
                    if snapshot.source_timestamp else None
                ),
                "mint": snapshot.mint,
                "symbol": snapshot.symbol,
                "name": snapshot.name,
                "fields": dict(snapshot.fields),
                "provenance": dict(snapshot.provenance),
                "parse_warnings": list(snapshot.parse_warnings),
                "raw_text": snapshot.raw_text[:2000],
                "traded": traded,
            }
        )

    async def _handle_candidate(self, snapshot: AlertSnapshot, parsed) -> None:
        analysis_start = self.clock.now()
        if not snapshot.mint:
            return
        # Register the task so shutdown can DRAIN it. A BUY whose order is in
        # flight when the operator stops the bot must reach a definitive outcome
        # rather than being cancelled into a state file that still says
        # ``DECISION=BUY``. The drain is bounded; see ``shutdown``.
        task = asyncio.current_task()
        if task is not None:
            self._inflight_candidates.add(task)
        try:
            await self._handle_candidate_inner(snapshot, parsed, analysis_start)
        finally:
            if task is not None:
                self._inflight_candidates.discard(task)

    async def _handle_candidate_inner(
        self, snapshot: AlertSnapshot, parsed, analysis_start: float
    ) -> None:
        # One recorder per candidate. Every stage below marks it, and it is
        # handed to the accumulator in a finally so a mid-pipeline raise still
        # records how far the candidate actually got -- which is the case worth
        # measuring.
        stages = StageRecorder()
        stages.mark("alert_received", snapshot.received_timestamp or analysis_start)
        self._current_stages = stages
        if snapshot.candidate_id in self._seen_candidates:
            log("ALERT", "duplicate candidate %s suppressed",
                snapshot.candidate_id[-8:])
            return
        mint = snapshot.mint
        self._record_candidate(snapshot, parsed, traded=True)
        log(
            "ALERT",
            "candidate %s %s mint=%s fields=%d %s",
            snapshot.candidate_id[-6:],
            (snapshot.symbol or "?"),
            mint[:10],
            len(snapshot.fields),
            {k: v for k, v in list(snapshot.fields.items())[:6]},
        )

        # -- narrative lookup, fire-and-forget (non-blocking) ---------
        # Start at alert receipt so narrative work overlaps both market/data
        # and the settle window.  The callback consumes late exceptions; the
        # task is never allowed to become an unobserved background failure.
        def _consume_narrative_task(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            try:
                task.exception()
            except Exception:  # noqa: BLE001
                return

        async def _fetch_narrative() -> Any:
            stages.mark("narrative_start", self.clock.now())
            try:
                if self.nar.is_async:
                    return await self.nar.fetch_async(mint)
                return await asyncio.to_thread(self.nar.fetch, mint)
            finally:
                stages.mark("narrative_end", self.clock.now())

        nar_task = asyncio.ensure_future(_fetch_narrative())
        nar_task.add_done_callback(_consume_narrative_task)

        # -- observation at decision time ------------------------------
        stages.mark("market_start", self.clock.now())
        # quote_async runs the synchronous provider transport on a worker thread,
        # so an 8s HTTP timeout with 2 retries can no longer freeze the event
        # loop (and every other task with it) for ~16s.
        quote = await self.market.quote_async(mint)
        obs = quote.observation
        # The alert's own price is legitimate T0 information and, for a token we
        # have never seen, it is the first price we could have acted on.
        self.market.register_known_price(mint, snapshot.num("price_usd"))
        if self._fvp_budget > 0 and self.market.first_valid_price(mint) is None:
            # Also a synchronous provider read; keep it off the event loop.
            await asyncio.to_thread(self.market.seed_first_valid_price, mint)
            self._fvp_budget -= 1

        stages.mark("market_end", self.clock.now())

        # -- settle: a second print before we decide ---------------------
        # Analysing off a single tick means having no move to characterise, and
        # the agent would refuse essentially every live opportunity. Waiting a
        # couple of seconds costs almost nothing on a 5-minute chart and buys
        # the minimum structure needed to classify a regime.
        #
        # This is its OWN stage. It used to sit inside the narrative window,
        # because ``narrative_start`` was marked here - above - and
        # ``narrative_end`` only after the narrative was awaited. The narrative
        # was then reported as taking 18.2s when the bot had answered in 1.6s
        # and the 16.6s belonged to the second market read below. A stage that
        # measures more than its own work is a stage nobody can act on.
        if obs is not None and len(self.market.history(mint).as_of(self.clock.now())) < 2:
            stages.mark("settle_start", self.clock.now())
            await self._sleep(self.config.market.alert_settle_s)
            quote = await self.market.quote_async(mint)
            obs = quote.observation
            stages.mark("settle_end", self.clock.now())
        else:
            stages.mark("settle_start", self.clock.now())
            stages.mark("settle_end", self.clock.now())

        # Narrative is fire-and-forget. We do NOT wait for it.
        # If it completed during market/settle, use the legitimate snapshot so
        # the old signal-quality semantics are preserved.  Otherwise pass an
        # explicit pending snapshot: no narrative contribution is fabricated,
        # and the L25 critical path remains non-blocking.
        nar = None
        if nar_task.done():
            try:
                nar = nar_task.result()
            except Exception as exc:  # noqa: BLE001
                log("NAR", "background narrative failed for %s: %s", mint[:8], exc)
                nar = None
        else:
            from .models import NarSnapshot

            nar = NarSnapshot(
                mint=mint,
                fetched_at=self.clock.now(),
                source=self.nar.name,
                available=False,
                error="NARRATIVE_PENDING: not complete at L25 decision",
            )
        stages.mark("l25_start", self.clock.now())

        # -- features, gated at analysis time -------------------------
        stages.mark("candidate_parsed", self.clock.now())
        decision_time = self.clock.now()
        features = build_features(
            candidate_id=snapshot.candidate_id,
            mint=mint,
            decision_time=decision_time,
            history=self.market.history(mint),
            alert=snapshot,
            nar=nar,
            stale_after_ms=self.config.market.stale_after_ms,
            unusable_after_ms=self.config.market.unusable_after_ms,
        )

        # The decision layer's entry gates must describe the arms that are
        # ABOUT TO RUN, not the ones selected when the process booted. The
        # operator can change the selection from Telegram at any moment, and the
        # two syncs elsewhere in this class (bridge construction and the arm
        # reconcile) both run AFTER this decision -- so syncing only there left
        # a candidate decided by the previous selection's gate. One tuple
        # comparison and a dict lookup; it is not worth being wrong about.
        self._sync_effective_entry_gates()

        stages.mark("risk_start", self.clock.now())
        report = self.engine.decide(
            features=features,
            alert=snapshot,
            nar=nar,
            analysis_start=analysis_start,
            analysis_complete=self.clock.now(),
            account_blocked_reason=self._account_block(mint),
        )
        stages.mark("risk_end", self.clock.now())
        stages.mark("l25_end", self.clock.now())
        self.last_decision = report.as_dict()

        if report.decision is Decision.REJECT:
            self.rejects += 1
            # Count this rejection under its OWN cause. The per-candidate reason
            # is already on record in the journal; this is the in-memory
            # histogram the shutdown line and the session counters read, so
            # "rejects=33" is never left as a count with no distribution. It
            # classifies, it does not decide: it reads the gate/flag the
            # strategy already produced and cannot change the outcome.
            self._record_reject_reason(report)
            log(
                "REJECT",
                "%s %s | %s",
                (report.symbol or "?")[:12],
                " / ".join(f.code for f in report.risk_flags) or "no specific flag",
                report.thesis[:150],
            )
            self.journal.decision(report, self.state.trade_mode)
            # A refusal by the ACTIVE strategy gets an operator alert, because
            # "why didn't it trade" is the question an operator actually has.
            gate = getattr(report, "l25", None) or report.v3c
            if gate is not None and gate.rejects:
                self._log_strategy_reject(gate, report.symbol or "?")
                # A rejection is the expected outcome for most candidates, so
                # pushing each one turns a quiet strategy into a firehose. It
                # stays in the LOG, the journal, the dashboard LAST EVENT row
                # and on demand via /status; pushing it is opt-in.
                if self.notify_rejects:
                    self._notify(
                        _render_strategy_reject(gate, report.symbol or "?"), None
                    )
            self._record_latency(
                stages, candidate_id=snapshot.candidate_id, mint=mint,
                symbol=snapshot.symbol,
            )
            return
        if report.decision is Decision.WAIT:
            self.waits += 1
            stages.mark("l25_end", stages.stamps.get("l25_end", self.clock.now()))
            log(
                "WAIT",
                "%s %s | %s",
                (report.symbol or "?")[:12],
                report.data_state.value,
                report.thesis[:150],
            )
            self.journal.decision(report, self.state.trade_mode)
            self._record_latency(
                stages, candidate_id=snapshot.candidate_id, mint=mint,
                symbol=snapshot.symbol,
            )
            return

        self.journal.decision(report, self.state.trade_mode)
        # A BUY decision is recorded as a DECISION. It is explicitly NOT a fill,
        # and from here every path must end in exactly one execution outcome.
        stages.finish(DECISION_BUY)
        self.last_execution = {
            "candidate_id": report.candidate_id,
            "mint": report.mint,
            "symbol": report.symbol,
            "decision": "BUY",
            "outcome": DECISION_BUY,
            "executed": False,
            "reason": None,
            "at": self.clock.now(),
        }
        self.latency.record_outcome(DECISION_BUY)
        stages.mark("quote_start", self.clock.now())
        try:
            await self._execute_buy(report, snapshot, features, quote)
        except asyncio.CancelledError:
            # A cancelled BUY is an OUTCOME, not a silence. Recorded before it is
            # re-raised so the operator can see that an authorized entry died
            # mid-flight instead of vanishing.
            self._record_block(
                report,
                "shutdown_during_execution" if self._shutting_down else "cancelled",
                "shutdown began while the order was in flight; the task was "
                "cancelled and no position was opened"
                if self._shutting_down
                else "the execution task was cancelled before the order completed",
                stages=stages,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            # A fault inside execution must not read as a silent decision. The
            # outcome is recorded, then the exception is re-raised so a real bug
            # still reaches the operator's log and the supervisor.
            self._record_block(
                report, "execution_exception",
                f"{type(exc).__name__}: {exc}",
                stages=stages,
            )
            raise
        finally:
            # recorded on every path -- buy, refuse, or raise mid-execution
            self._record_latency(
                stages, candidate_id=snapshot.candidate_id, mint=mint,
                symbol=snapshot.symbol,
            )

    def _narrative_giveup(self, mint: str):
        """
        The honest result of a narrative that did not arrive in its budget.

        Never an exception: a narrative timeout must degrade the candidate's
        evidence, not kill the candidate. Recorded as unavailable WITH the reason,
        so "no narrative" and "narrative too slow" stay distinguishable in the
        decision's risk flags.
        """
        from .models import NarSnapshot

        budget = self.config.nar.timeout_s + self.config.nar.timeout_grace_s
        log("NAR", "narrative budget %.1fs exceeded for %s; continuing degraded",
            budget, mint[:8])
        self.nar.metrics.timeouts += 1
        return NarSnapshot(
            mint=mint, fetched_at=self.clock.now(), source=self.nar.name,
            available=False,
            error=f"NAR_UNAVAILABLE: no bot reply within {budget:.1f}s",
        )

    def _record_confirmed_unreconciled(
        self,
        report,
        fill,
    ) -> dict[str, Any]:
        """Record a swap the chain CONFIRMED whose amount could not be read.

        Deliberately a separate path from ``_record_block``. A confirmed fill
        that could not be reconciled did NOT become a tracked position, but it
        is emphatically not a "refused" order either: the swap executed and the
        wallet holds the output.

        This is the $Boy incident's actual lesson. Recording it as
        ``BUY_BLOCKED=fill_refused`` said "no order was placed" about an order
        that was placed, filled and confirmed -- so the runtime's position state
        and the wallet's real balances disagreed, and nothing in the operator
        surface flagged the gap.

        It counts as blocked (no position opened, so DECISION=BUY did not become
        PAPER_FILL=SUCCESS), but under a reason that cannot be misread as
        "nothing happened", and it carries the signature so recovery can find
        it. It must never trigger a re-broadcast: that would double-buy.
        """
        reason = "confirmed_but_unreconciled"
        detail = (
            f"{fill.status}: {fill.ambiguity_reason or '-'} "
            f"(signature={(fill.signature or '')[:16]}...) -- chain confirmed; "
            f"no re-broadcast; recovery from on-chain state required"
        )
        stages = self._current_stages
        if stages is not None:
            stages.block(reason, outcome=BUY_BLOCKED, detail=detail)
        self.latency.record_outcome(BUY_BLOCKED)
        self.blocks += 1
        self.blocked_reasons[reason] = self.blocked_reasons.get(reason, 0) + 1
        log(
            "EXECUTE",
            "%s=%s candidate=%s symbol=%s %s",
            BUY_BLOCKED,
            reason,
            getattr(report, "candidate_id", "?")[-8:],
            (getattr(report, "symbol", None) or "?")[:12],
            detail,
        )
        self.last_execution = {
            "candidate_id": getattr(report, "candidate_id", None),
            "mint": getattr(report, "mint", None),
            "symbol": getattr(report, "symbol", None),
            "decision": "BUY",
            "outcome": BUY_BLOCKED,
            "executed": False,
            "reason": reason,
            "detail": detail,
            "confirmed_on_chain": True,
            "signature": fill.signature,
            "requires_recovery": True,
            "at": self.clock.now(),
        }
        try:
            self.journal.unfilled(
                self.state.trade_mode,
                candidate_id=getattr(report, "candidate_id", None),
                mint=getattr(report, "mint", None),
                symbol=getattr(report, "symbol", None),
                status=fill.status,
                reason=reason,
                detail=detail,
                signature=fill.signature,
                confirmed_on_chain=True,
                requires_recovery=True,
            )
        except TypeError:
            # Older journal signature without the recovery fields. The
            # classification above still holds; only the extra detail is lost.
            self.journal.unfilled(
                self.state.trade_mode,
                candidate_id=getattr(report, "candidate_id", None),
                mint=getattr(report, "mint", None),
                symbol=getattr(report, "symbol", None),
                status=fill.status,
                reason=reason,
            )
        return self.last_execution

    def _record_block(
        self,
        report,
        reason: str,
        detail: str = "",
        *,
        stages: "StageRecorder | None" = None,
        beta_arms=(),
        traded_arms=(),
        notify: bool = True,
    ) -> dict[str, Any]:
        """
        Record that a BUY decision did NOT become a position.

        Every one of these lands in four places at once - the stage recorder, the
        latency accumulator, the journal and the operator log - because a
        refusal that exists in only one of them is exactly what made the
        2026-09-30 incident unreadable. The counter also increments, so
        ``DECISION=BUY`` and ``PAPER_FILL=SUCCESS`` can be compared directly.

        NOT for a swap the chain confirmed but that could not be reconciled --
        that is ``_record_confirmed_unreconciled``, because "no order was
        placed" is false in that case.

        ``beta_arms``/``traded_arms`` attribute the refusal to the gate that
        actually made it, so a V2 FLOOR (BETA) experiment refusal is reported as
        a beta-gate rejection rather than as an L25 entry that was blocked.
        ``notify=False`` records the outcome without a chat message; the
        execution-aware caller uses it to collapse the three per-arm beta
        refusals of one signal into ONE operator message.
        """
        stages = stages if stages is not None else self._current_stages
        if stages is not None:
            stages.block(reason, outcome=BUY_BLOCKED, detail=detail)
        self.latency.record_outcome(BUY_BLOCKED)
        self.blocks += 1
        self.blocked_reasons[reason] = self.blocked_reasons.get(reason, 0) + 1
        token = f"{BUY_BLOCKED}={reason}"
        log(
            "EXECUTE",
            "%s candidate=%s symbol=%s %s%s",
            token,
            getattr(report, "candidate_id", "?")[-8:],
            (getattr(report, "symbol", None) or "?")[:12],
            detail or "",
            (f" -- {detail}" if detail else ""),
        )
        self.last_execution = {
            "candidate_id": getattr(report, "candidate_id", None),
            "mint": getattr(report, "mint", None),
            "symbol": getattr(report, "symbol", None),
            "decision": "BUY",
            "outcome": BUY_BLOCKED,
            "executed": False,
            "reason": reason,
            "detail": detail,
            "at": self.clock.now(),
        }
        try:
            self.journal.unfilled(
                self.state.trade_mode,
                candidate_id=getattr(report, "candidate_id", None),
                mint=getattr(report, "mint", None),
                symbol=getattr(report, "symbol", None),
                decision="BUY",
                reason=reason,
                detail=detail,
            )
        except Exception as exc:  # noqa: BLE001 - journaling must not mask the reason
            log("ERROR", "cannot journal BUY_BLOCKED (%s): %s",
                type(exc).__name__, exc)
        # A BUY the strategy authorized and the executor did not fill is an
        # ANOMALY, not routine noise: it is the state an operator most needs to
        # see and the one the old dashboard rendered as a silent "quote=-". So
        # unlike a strategy rejection, which is opt-in, a blocked BUY always
        # reaches the chat -- ONCE per signal, however many arms refused.
        if notify:
            self._notify(
                ui_render_buy_blocked(
                    symbol=getattr(report, "symbol", None) or "?",
                    mint=getattr(report, "mint", None) or "",
                    reason=reason,
                    detail=detail,
                    mode=self.state.trade_mode,
                    candidate_id=getattr(report, "candidate_id", None),
                    beta_arms=beta_arms,
                    traded_arms=traded_arms,
                ),
                None,
            )
        return self.last_execution

    def _record_fill(self, report, position, *, mode_outcome: str) -> None:
        """Record that a BUY decision DID become a position."""
        stages = self._current_stages
        if stages is not None:
            stages.finish(mode_outcome)
        self.latency.record_outcome(mode_outcome)
        log(
            "EXECUTE",
            "%s candidate=%s symbol=%s position=%s",
            mode_outcome,
            getattr(report, "candidate_id", "?")[-8:],
            (getattr(report, "symbol", None) or "?")[:12],
            getattr(position, "position_id", "?"),
        )
        self.last_execution = {
            "candidate_id": getattr(report, "candidate_id", None),
            "mint": getattr(report, "mint", None),
            "symbol": getattr(report, "symbol", None),
            "decision": "BUY",
            "outcome": mode_outcome,
            "executed": True,
            "reason": None,
            "position_id": getattr(position, "position_id", None),
            "at": self.clock.now(),
        }

    def _record_latency(
        self,
        stages: "StageRecorder",
        *,
        candidate_id: str | None = None,
        mint: str | None = None,
        symbol: str | None = None,
    ) -> None:
        """Fold one candidate's stage timings into the rolling percentiles and
        PERSIST them, so the distribution survives a process restart.

        A partial recording is kept, not discarded: how far a candidate got is
        the interesting fact when it never reached a fill.
        """
        if self.cli_only or stages.trading_ms is None:
            # even in --cli, a candidate that was fully timed is still worth
            # recording -- only the notification half is meaningless there
            if not stages.stamps:
                return
        self.latency.record(stages)
        self._current_stages = stages
        # One small buffered append. No fsync, no network, and it happens after
        # the trading decision, so it never sits on the entry critical path.
        self.stage_store.record(
            candidate_id,
            stages,
            mode=self.state.trade_mode.value,
            mint=mint,
            symbol=symbol,
        )

    def _account_block(self, mint: str) -> str | None:
        if self.control.command:
            return f"control command {self.control.command} active"
        if self.account.has_mint(mint):
            return "already holding this mint"
        if self.account.cooldown_active(mint, self.clock.now()):
            return "re-entry into this mint is disabled"
        if len(self.account.open_positions()) >= self.config.risk.max_simultaneous_positions:
            return (
                f"{len(self.account.open_positions())} positions open "
                f"(max {self.config.risk.max_simultaneous_positions})"
            )
        return None

    def _entry_equity_usd(self) -> float:
        """
        Equity for sizing, from the account the current mode trades against.

        PAPER sizes off the simulated account; REAL sizes off the wallet. Mixing
        them is how a real order ends up sized off a simulated balance, so the
        two are read from different objects on purpose.
        """
        if self.state.trade_mode is TradeMode.REAL:
            eq = self.real_account.equity_usd(self.clock.now())
            if eq is None:
                # A stale or unpriced wallet cannot produce an honest size.
                # Zero means "refuse", not "size to zero and trade".
                return 0.0
            return eq
        return self.account.equity(self._refresh_marks())

    async def _execute_buy(
        self, report: DecisionReport, snapshot: AlertSnapshot, features, quote
    ) -> None:
        """
        Open a position. V3-C has already authorized; everything here is either
        sizing, pre-trade safety, or execution.

        The branch on mode happens FIRST. A PAPER decision never touches the
        real executor and a REAL decision never touches the paper simulator.

        EVERY RETURN IS EXPLICIT. Each refusal path calls ``_record_block`` with
        a reason, so a BUY decision that did not become a position is a recorded
        outcome rather than a silent ``return``. There is no path out of this
        method that leaves the operator unable to say what happened.
        """
        from .modes import validate_strategy_selection

        # The mode is re-read here, not assumed from the caller: a stop pressed
        # during the analysis window must take effect before the order.
        allowed, why = self.state.can_open_new_entries()
        if not allowed:
            log("REJECT", "%s entry blocked: %s",
                (report.symbol or "?")[:12], why)
            self._record_block(report, "mode_blocked", why)
            return

        # Data-quality gate. A candidate too old to act on is refused as stale,
        # so a latency problem cannot be executed as a strategy loss. Runs AFTER
        # the mode gate and BEFORE any order is constructed, so nothing is
        # prepared, signed or sent for a refused candidate.
        stale = self.stale_gate.check(
            signal_at=getattr(snapshot, "received_timestamp", None),
            now=self.clock.now(),
            decision_at=report.decided_at,
        )
        if stale is not None:
            age = ((self.clock.now()
                    - (getattr(snapshot, "received_timestamp", None)
                       or self.clock.now())) * 1000.0)
            self.ledger.note(
                "entry refused as stale", reason=stale, symbol=report.symbol,
                age_ms=round(age, 1), budget_ms=self.stale_gate.budget_ms,
            )
            log("REJECT", "%s entry refused: %s (age %.0fms > %.0fms)",
                (report.symbol or "?")[:12], stale, age,
                self.stale_gate.budget_ms)
            self.account.unfilled_attempts += 1
            # Named so the dashboard, the log and the journal agree. ``stale``
            # is already one of STALE_SIGNAL / STALE_QUOTE /
            # ENTRY_LATENCY_EXCEEDED, which is more precise than a generic
            # "stale"; it is normalised to the outcome vocabulary below.
            self._record_block(
                report, _block_reason_for_stale(stale),
                f"signal age {age:.0f}ms exceeds the "
                f"{self.stale_gate.budget_ms:.0f}ms entry budget",
            )
            return

        if self.state.trade_mode is TradeMode.REAL:
            # A single REAL signal must never fan out into several real
            # entries. The UI enforces one strategy for REAL, but the UI is not
            # a security boundary: a malformed config or state file could carry
            # several arms, so the entry path re-checks and REFUSES rather than
            # trading a configuration the rules forbid.
            ok, why, _arms = validate_strategy_selection(
                list(self.state.arms), trade_mode=TradeMode.REAL)
            if not ok:
                log("REJECT", "%s REAL entry refused: %s",
                    (report.symbol or "?")[:12], why)
                self._record_block(report, "REAL_MULTI_STRATEGY_REFUSED", why)
                return
            await self._execute_real_buy(report, snapshot, features)
            return
        if self._execution_aware_paper():
            # EXECUTION-AWARE PAPER. The decision above stands; only the
            # EXECUTION changes, and it changes to a real executable quote.
            await self._execute_paper_buy_execution_aware(
                report, snapshot, features)
            return
        await self._execute_paper_buy(report, snapshot, features, quote)

    # ------------------------------------------------------------------
    # execution-aware PAPER
    # ------------------------------------------------------------------

    def _execution_aware_paper(self) -> bool:
        """
        Should this PAPER fill go through the executable-quote engine?

        Requires the config flag AND PAPER mode. In REAL the branch is never
        even consulted: REAL keeps its own executor, and a PAPER-only engine can
        never become the path that builds a real transaction.
        """
        if getattr(self.config.real, "paper_legacy_mark_fill_rollback", False):
            return False
        return bool(
            getattr(self.config.real, "paper_execution_aware_enabled", True)
            and self.state.trade_mode is not TradeMode.REAL
        )

    async def _paper_bridge_instance(self) -> Any:
        """
        The execution-aware PAPER engine, built once per process.

        Constructed lazily and cached. It is a
        :class:`~autopaper.paper_live.bridge.TelegramPaperBridge`, which is a
        ``PaperLiveEngine``: it owns the per-arm virtual accounts, the quote
        service and the dataset writer, and its construction asserts PAPER-ONLY
        before any collaborator is touched.

        THE CACHE IS RECONCILED, NOT TRUSTED. Caching the bridge is correct --
        it owns per-arm virtual bankrolls, the quote cache and the durable
        writers, and rebuilding it on every signal would throw away an arm's
        open position and its account history. But the bridge captured the
        operator's arm selection at CONSTRUCTION, and a selection can change
        afterwards. So every call compares the selection against what the live
        bridge is actually running, and reconciles when they differ. Without that
        comparison this method silently returned a process-lifetime snapshot of a
        choice the operator had already changed.
        """
        if self._paper_bridge is not None:
            self._reconcile_paper_bridge_arms()
            self._reconcile_paper_bridge_take_profit()
            return self._paper_bridge
        from .paper_live.arms import parse_arms
        from .paper_live.bridge import build_telegram_bridge
        from .paper_live.fills import FillModel
        from .paper_live.quote_proximity import ProximityPolicy
        from .paper_live.quotes import SolReference
        from .same_market.reader import DirectPumpReader
        from .same_market.rpc import SolanaReadOnlyRpc
        from .execution.solana import WSOL_MINT as _WSOL

        rpc = SolanaReadOnlyRpc(
            # THE CHOKE POINT WAS ARTIFICIAL, AND IT CONTRADICTED OUR OWN
            # CONFIG.
            #
            # This read ``max_rpm = public_rpc_max_concurrency * 4`` = 16
            # requests/minute, while ``public_rpc_rate_per_s`` in the very same
            # config object declares 8/s = 480/min. So the wiring was ~30x
            # tighter than the capacity we had already decided we had, and
            # because the limiter RAISED instead of waiting, every request past
            # that ceiling was deleted and surfaced downstream as "no
            # executable quote" -- a capacity limit reported as a property of
            # the token.
            #
            # The declared rate is now honoured, bounded by the declared
            # concurrency, and the limiter WAITS inside a bounded budget before
            # it ever refuses.
            url=self.config.real.public_rpc_url or None,
            urls=tuple(
                u.strip() for u in (self.config.real.public_rpc_urls or "").split(",")
                if u.strip()),
            max_rpm=max(1, int(round(
                float(self.config.real.public_rpc_rate_per_s) * 60.0))),
max_concurrent=max(1, int(self.config.real.public_rpc_max_concurrency)),
            limiter_wait_s=float(self.config.real.public_rpc_limiter_wait_s),
            timeout_s=float(self.config.real.public_rpc_timeout_s),
            # THE TELEMETRY LANE'S RESERVED SHARE.
            #
            # The TOTAL is unchanged at ``max_rpm`` above -- this partitions the
            # existing budget rather than enlarging it, so the provider request
            # rate does not rise. What changes is that the background hold-path
            # sampler can now spend at most this much, leaving the remainder
            # unreachable by telemetry and always available to an open position
            # that needs to sell. Without this, a 1 Hz sampler and an exit shared
            # one FIFO budget and the exit lost.
            telemetry_max_per_minute=max(
                0, int(self.config.real.public_rpc_telemetry_max_per_minute)),
            # Account VALUES stay uncached. The slot and the AMM fee-config
            # caches inside the reader remove round trips that were never
            # pricing anything; caching the pool account itself would let a
            # quote describe a book that has moved, which is precisely the
            # confusion this engine exists to eliminate.
            cache_ttl_s=0.0,
        )
        reader = DirectPumpReader(rpc)
        # SOL/USD SELF-RESOLUTION.
        #
        # This used to read ONLY ``self._sol_price_usd``, a memo that nothing on
        # the PAPER path ever populates -- it is filled by the REAL equity path.
        # So a PAPER runtime built its bridge with ``sol_reference=None`` and
        # EVERY L25-approved entry was refused SOL_REFERENCE_MISSING. That is
        # exactly today's incident: 13 authorised, 0 filled, and nothing in the
        # journal said why.
        #
        # It now calls the CANONICAL resolver, which reads
        # ``same_market.sol_price.JupiterSolPriceProvider`` and applies its own
        # freshness gate. No second provider, no duplicate HTTP, and no
        # dependency on REAL arming, a wallet, or /balance.
        sol_usd = self._sol_price_usd
        if not sol_usd:
            sol_usd = await self._backend_sol_price()
        sol_usd = sol_usd or 0.0
        # APPLY THE CONFIGURED MAX-HOLD CAP before the bridge is built, so a
        # position opened under L25_MAX_HOLD carries the cap that was actually
        # configured at open time. PAPER research only; L25_CURRENT has no cap
        # and L25_60S is fixed at 60s, so neither is affected.
        try:
            from .strategy_profile import set_max_hold_s
            set_max_hold_s(float(self.config.decision.max_hold_seconds))
        except Exception:  # noqa: BLE001 - a bad cap must not stop PAPER boot
            log("ERROR",
                "max_hold_seconds=%r could not be applied; L25_MAX_HOLD will "
                "run its 60s default",
                getattr(self.config.decision, "max_hold_seconds", None))

        self._paper_bridge = build_telegram_bridge(
            # ONLY the operator's SELECTED arms. Each gets its own
            # VirtualAccount; the engine's shared quote cache is what keeps
            # several arms on one signal down to a single provider request.
            arms=parse_arms(list(self.state.arms)),
            root=self._dataset_root(),
            runtime_profile=str(getattr(self.state, "profile", "") or ""),
            decision_engine=self.engine,
            exit_policy=self._open_book_policy(),
            quote_service=(
                self.quote_service_override
                or self._paper_quote_service(rpc, reader)
            ),
            market=self.market,
            token_decimals_for=reader.mint_decimals,
            sol_reference=(SolReference(sol_usd, self.clock.now(), "runtime")
                           if sol_usd > 0 else None),
            fill_model=FillModel(),
            starting_bankroll_usd=5.00,
            require_identity=True,
            # -- THE CANONICAL DURABLE STATE ROOT -------------------------
            # ``Config.var_dir`` is the runtime's declared state directory:
            # built once, absolute, and created at config time. It is passed
            # EXPLICITLY so the bridge, the V2 FLOOR (BETA) experiment and the
            # state store all resolve the same directory instead of each
            # deriving one from the working directory. ``state_is_runtime`` is
            # the engine's permission to write there; a test or probe leaves it
            # False and is refused.
            state_root=self.config.var_dir,
            state_is_runtime=True,
            # -- EVENT-DRIVEN MARKET MONITORING --------------------------
            # Market state comes from pushed account updates and a local price
            # calculation; an executable SELL quote is spent only when an
            # existing exit condition is near or has fired. The polling path
            # remains reachable as the recovery fallback and through the
            # ``native_monitor_enabled`` flag.
            native_monitor_enabled=bool(
                getattr(self.config.market, "native_monitor_enabled", False)
                and getattr(self.config.market, "native_monitor_ws_url", "")
            ),
            native_monitor_ws_url=str(
                getattr(self.config.market, "native_monitor_ws_url", "") or ""),
            native_monitor_stale_after_ms=float(
                getattr(self.config.market, "native_monitor_stale_after_ms",
                        8000.0) or 8000.0),
            native_monitor_open_timeout_s=float(
                getattr(self.config.market, "native_monitor_open_timeout_s", 10.0)
                or 10.0),
            quote_proximity_policy=ProximityPolicy(
                enabled=bool(getattr(
                    self.config.market, "quote_proximity_enabled", True)),
                band_pct=float(getattr(
                    self.config.market, "quote_proximity_band_pct", 3.0) or 3.0),
            ),
            # -- THE OPERATOR'S TAKE-PROFIT SELECTION -----------------------
            # The GLOBAL value, so the first position opened in this process
            # carries the level the operator actually selected (restored from
            # ``var/take_profit_setting.json`` at boot). From then on
            # ``_reconcile_paper_bridge_take_profit`` keeps it current, and
            # each position holds its own snapshot of it.
            take_profit_pct=self.state.take_profit_pct,
        )
        # ONE SOL/USD SNAPSHOT, SHARED.
        #
        # The engine converts a USD notional to lamports and the quote service
        # converts lamports back to a USD price. If they hold different
        # references -- or if the quote service holds none, which it did while
        # this bot was still starting up -- every live quote arrives with
        # ``effective_price_usd = None`` and the executable-vs-mark entry guard
        # cannot be evaluated at all. Sharing one object removes the whole
        # class of disagreement.
        self._paper_bridge_arms = tuple(
            spec.name.value for spec in self._paper_bridge.arms)
        reference = self._paper_bridge.sol_reference
        quote_service = getattr(self._paper_bridge, "quote_service", None)
        if quote_service is not None:
            quote_service.sol_reference = reference
        log("PAPER", "execution-aware PAPER engine ready (arms=%s, "
            "sol_reference=%s)", ",".join(
                s.name.value for s in self._paper_bridge.arms),
            "shared" if reference is not None else "MISSING")
        # A restart that forgets its open positions would leave the operator's
        # book empty while the arms' own dataset still listed the trades. The
        # resume happens HERE, at construction, so the book is whole before the
        # first signal is analysed rather than after the first entry.
        self.rehydrate_paper_state()
        return self._paper_bridge

    async def restore_execution_aware_book(self) -> dict[str, Any]:
        """
        Load the execution-aware PAPER book during STARTUP, before any signal.

        THE DEFECT THIS CLOSES
        ----------------------
        ``_paper_bridge_instance`` is LAZY: it was built only on the entry path
        (:meth:`_execution_aware_entry`) and on the emergency-close path. But the
        bridge is what OWNS the arms' virtual bankrolls, their open positions,
        the quote service and -- through ``BetaExperiment`` -- the V2 FLOOR
        (BETA) book with its unresolved arms and its per-tick SELL retry. And
        ``monitor_once`` ticks that book only when ``self._paper_bridge is not
        None``::

            if self._execution_aware_paper() and self._paper_bridge is not None:
                trades = await self._paper_bridge.tick()

        So after a restart the whole book was DARK until some candidate was
        authorised -- not merely unrendered in the operator's views, but
        UNMONITORED. No open position had its exit bracket evaluated on any tick,
        and no unresolved BETA arm was retried, until an unrelated entry attempt
        happened to build the bridge. That is a long wait by construction: a
        BETA-only selection is gated by the same V2 FLOOR rule that can refuse
        every candidate for hours (in the run this was found in, every candidate
        after a restart was refused ``v2floor_vol5_per_mc_below_floor`` or
        ``v3c_market_cap_over_ceiling``).

        It is visible in the production log as a hard cliff. Process 51934
        emitted a ``BETA_STAGE`` retry line every ~6s up to its shutdown at
        21:04:17; the process that replaced it emitted NONE for the next fifteen
        minutes -- the exit retry of three unresolved arms simply did not run,
        because no signal had reached the entry path yet.

        ``rehydrate_paper_state`` already states the intent -- "the book is whole
        before the first signal is analysed rather than after the first entry" --
        but it is called at CONSTRUCTION, and construction is exactly what the
        first signal triggers. This method closes that gap by building the book
        once, during startup, before the listener can deliver anything.

        CALLED ONCE FROM ``run()``, RIGHT AFTER ``startup()``
        ----------------------------------------------------
        A boot-time build and nothing else in the ordinary path: it is idempotent
        (an already-built bridge is left alone), it is skipped entirely when the
        execution-aware PAPER path is not the active one (REAL mode, or the
        legacy mark-fill rollback flag), and on failure it DEGRADES to the old
        lazy behaviour rather than refusing to start. A book that cannot be built
        at boot -- no SOL/USD reference, no RPC, a corpus that cannot be read --
        is logged as an ERROR, because silence here would look exactly like
        "nothing to restore".
        """
        if self._paper_bridge is not None:
            return {"built": False, "reason": "already built"}
        if not self._execution_aware_paper():
            return {"built": False, "reason": "execution-aware PAPER is off"}
        try:
            bridge = await self._paper_bridge_instance()
        except Exception as exc:  # noqa: BLE001 - startup must not depend on it
            log("ERROR",
                "the execution-aware PAPER book could NOT be loaded at startup "
                "(%s: %s); it will be built on the next entry path, and until "
                "then NO open position is monitored and NO beta arm is retried",
                type(exc).__name__, exc)
            return {"built": False, "reason": f"{type(exc).__name__}: {exc}"}
        outstanding: list[str] = []
        experiment = getattr(bridge, "beta_experiment", None)
        if experiment is not None:
            try:
                outstanding = sorted(
                    f"{pos.parent_signal_id}:{arm}"
                    for arm, pos in experiment.outstanding_arms().items())
            except Exception:  # noqa: BLE001 - reporting must not break boot
                outstanding = []
        log("PAPER",
            "execution-aware PAPER book loaded at startup: arms=%s "
            "beta_outstanding=%s",
            ",".join(s.name.value for s in bridge.arms),
            ",".join(outstanding) if outstanding else "none")
        return {
            "built": True,
            "arms": [s.name.value for s in bridge.arms],
            "beta_outstanding": outstanding,
        }

    def _reconcile_paper_bridge_arms(self) -> None:
        """
        Make the LIVE bridge agree with the operator's CURRENT selection.

        Called on every entry path, and cheap when nothing changed (one tuple
        comparison). When it has changed, the bridge's arm set is reconciled in
        place rather than rebuilt, because rebuilding would discard each arm's
        virtual bankroll, its cooldown map and any open position.

        THE RULE ABOUT OPEN POSITIONS
        ----------------------------
        A position keeps the arm and profile that opened it. Deselecting an arm
        that currently holds a position therefore does NOT abandon it, does not
        re-stamp it, and does not quietly stop monitoring it -- the arm keeps
        running until its position closes, and the operator is told, loudly and
        once, that the deselection is deferred. The alternative, dropping the
        arm immediately, is what made a restart silently DISCARD open positions
        (``paper_live/engine.py::rehydrate`` skips an arm it no longer knows);
        keeping the arm until it is flat makes that unrepresentable.
        """
        bridge = self._paper_bridge
        if bridge is None:
            return
        from .paper_live.arms import parse_arms
        try:
            selected = tuple(spec.name.value
                             for spec in parse_arms(list(self.state.arms)))
        except Exception as exc:  # noqa: BLE001 - never fail an entry on this
            log("ERROR", "cannot resolve the current strategy selection (%s); "
                "the running arms stay as they are", exc)
            return
        # BEFORE the early return: the selection can be unchanged while the
        # profile has moved, and the decision layer must always be looking at
        # the gate that matches the arms that are actually running.
        self._sync_effective_entry_gates()
        if selected == self._paper_bridge_arms:
            return

        previous = tuple(self._paper_bridge_arms)
        blocked, reconciled = bridge.reconcile_arms(selected)
        self._paper_bridge_arms = tuple(
            spec.name.value for spec in bridge.arms)
        self._paper_bridge_deferred = tuple(blocked)
        if blocked:
            log("WARN",
                "strategy selection changed %s -> %s, but %s still holds an "
                "open position so %s keeps running until it closes. The "
                "position keeps the strategy it was opened under.",
                ",".join(previous) or "<none>", ",".join(selected),
                ",".join(blocked),
                ",".join(blocked))
        if reconciled:
            log("PAPER", "strategy selection applied: %s -> %s (running arms "
                "now %s)", ",".join(previous) or "<none>",
                ",".join(selected), ",".join(self._paper_bridge_arms))
        self._assert_strategy_invariant(previous)

    def _reconcile_paper_bridge_take_profit(self) -> None:
        """
        Push the operator's CURRENT take-profit selection onto the live bridge.

        WHAT THIS DOES AND DOES NOT DO, and the distinction is the whole point:
        the bridge holds the GLOBAL selection, which is the value a NEW position
        is stamped with at open. An ALREADY-OPEN position carries its own copy
        (``PaperPosition.take_profit_pct``) and every exit reads that copy, so
        changing this value here cannot rewrite the exit policy of something in
        the book. See ``exits._take_profit_for`` and
        ``paper_live.engine._take_profit_snapshot``.

        Called on every entry path, and cheap when nothing changed (one
        comparison). An unreadable selection leaves the last valid value in
        force rather than disabling a working take-profit.
        """
        bridge = self._paper_bridge
        if bridge is None:
            return
        from .paper_live.take_profit import TakeProfitError, normalise
        try:
            level = normalise(self.state.take_profit_pct)
        except TakeProfitError as exc:
            log("ERROR",
                "the current take-profit selection %r is not a valid level; the "
                "bridge keeps %r and NEW positions open with that: %s",
                self.state.take_profit_pct,
                getattr(bridge, "take_profit_pct", None), exc)
            return
        if level == getattr(bridge, "take_profit_pct", None):
            return
        previous = getattr(bridge, "take_profit_pct", None)
        bridge.take_profit_pct = level
        log("PAPER", "take-profit selection applied: %s -> %s "
            "(applies to NEW positions only)",
            "OFF" if previous is None else f"+{previous:g}%",
            "OFF" if level is None else f"+{level:g}%")

    def _assert_strategy_invariant(
            self, previous: tuple[str, ...] = ()) -> dict[str, Any]:
        """
        Log ``STRATEGY_MISMATCH`` when the layers disagree.

        Compares, in one place and on every entry path:

            selected -> the operator's saved selection,
            runtime  -> what the LIVE bridge is actually running,
            position -> the strategy the open position was stamped with.

        A disagreement here is the reported bug itself, so it is logged with
        every layer's value rather than summarised. It is a LOG, not an
        exception: refusing to open a PAPER position because a label disagrees
        would turn a reporting defect into a trading outage, and PAPER exists to
        produce the data that finds these.
        """
        bridge = self._paper_bridge
        if bridge is None:
            return {}
        selected = tuple(self.state.arms)
        runtime = tuple(spec.name.value for spec in bridge.arms)
        positions = {
            arm: getattr(account.position, "production_profile", "")
            for arm, account in (getattr(bridge, "accounts", {}) or {}).items()
            if getattr(account, "position", None) is not None
        }
        deferred = tuple(a for a in runtime if a not in selected)
        detail = {
            "selected": list(selected),
            "runtime": list(runtime),
            "deferred": list(deferred),
            "positions": positions,
            "previous_runtime": list(previous),
        }
        # A deferred arm is EXPECTED while it holds a position; it is not a
        # mismatch. Anything running that is neither selected nor deferred --
        # or deferred while holding nothing -- is unexplained, and that is what
        # this warning is for.
        unexplained = [a for a in runtime if a not in selected
                       and a not in self._paper_bridge_deferred]
        if unexplained:
            log("ERROR",
                "STRATEGY_MISMATCH selected=%s runtime=%s deferred=%s "
                "positions=%s",
                ",".join(selected), ",".join(runtime),
                ",".join(deferred), positions)
        detail["unexplained"] = unexplained
        return detail

    def _dataset_root(self) -> Any:
        """
        Where the ``reports/live`` artefacts go.

        The repository root, derived from ``var/``'s parent, so a live run
        writes next to the code rather than into whatever the cwd happened to
        be when the operator launched Telegram.
        """
        var = getattr(self.config, "var_dir", None)
        return var.parent if var is not None else None

    def _open_book_policy(self) -> Any:
        """
        The SAME ``ExitPolicy`` object the PAPER book uses.

        Reusing the live instance rather than building a second one is what
        keeps exit precedence identical between the execution-aware arms and the
        rest of the runtime. A second policy object would be a second set of
        thresholds, and the arms would silently drift from the strategy they are
        supposed to be measuring.
        """
        book = self.open_book.get("positions")
        policy = getattr(book, "policy", None)
        if policy is None:
            from .exits import ExitPolicy
            policy = ExitPolicy(self.config.exits, memory=self.memory)
        return policy

    def _paper_quote_service(self, rpc: Any, reader: Any) -> Any:
        """
        The ONE executable-quote source for PAPER.

        This is the verified ``same_market`` stack, NOT
        ``execution.aggregators.QuoteOnlyJupiterProvider``. Routing PAPER
        through the aggregator would mean two notions of "a price" -- one with
        identity proof and one without -- and the dataset would depend on which
        one happened to answer.

        ``JupiterQuoteClient`` here is the same_market quote-only client, not
        that aggregator: it returns the route plan whose single-hop pool key is
        checked against the observed pair, so a Jupiter candidate carries the
        same identity evidence a direct candidate does. It participates as ONE
        CANDIDATE among the three (Jupiter / pump direct / pumpswap direct) so
        best-route selection compares them all -- without it, "Jupiter is the
        primary market-discovery mechanism" in ``quote_candidates`` was dead
        code and a source that never answered could not be told apart from one
        that had no route.
        """
        from .paper_live.quotes import QuoteService
        sol_usd = self._sol_price_usd or 0.0
        from .paper_live.quotes import SolReference
        from .same_market.jupiter import JupiterQuoteClient
        market_cfg = getattr(self.config, "market", None)
        jupiter = JupiterQuoteClient(
            timeout_s=float(getattr(market_cfg, "http_timeout_s", 8.0)),
            cache_ttl_s=float(getattr(market_cfg, "cache_ttl_s", 2.0)),
        )
        return QuoteService(
            rpc=rpc,
            sol_reference=SolReference(sol_usd, self.clock.now(), "runtime"),
            jupiter=jupiter,
            max_quote_age_ms=30_000.0,
            allow_jupiter_fallback=True,
        )

    def _flush_paper_bridge_messages(self) -> None:
        """
        Send whatever the bridge queued, through the EXISTING notifier.

        Nothing here decides a message exists -- that is the bridge's job, and
        for a V2 FLOOR (BETA) entry it is
        ``TelegramPaperBridge.render_beta_entries``. This function only
        dispatches, and it records what it attempted.

        A notifier FAILURE IS NEVER ALLOWED TO EARN AN ALARM ABOUT THE TRADE.
        The position already exists and is already durable; a Telegram outage
        is an operator-visibility problem, not a trading one. So an exception is
        caught, logged and recorded, and the loop continues to the next message.
        The position is never rolled back and never re-sent here.
        """
        bridge = self._paper_bridge
        if bridge is None:
            return
        for message in bridge.drain_messages():
            kind = str(message.get("kind") or "entry")
            parent_id = message.get("parent_signal_id")
            # THE DEDUPE KEY IS THE MESSAGE'S OWN IDENTITY. The beta renderers
            # stamp it with the arms (and, for a blocked exit, the attempt
            # numbers) the message actually reports, so a re-entrant flush of
            # the SAME message is dropped while a genuinely NEW message -- an
            # arm that closed on a later tick, or a later refused attempt -- is
            # delivered. Older callers that send no key fall back to the
            # per-parent key they have always used.
            dedupe_key = message.get("dedupe_key")
            if dedupe_key is None and kind in ("beta_entry", "beta_exit"):
                dedupe_key = f"{kind}::{parent_id}"
            if dedupe_key is not None and self._beta_notified.get(dedupe_key):
                self._record_notification(
                    kind=kind, parent_signal_id=parent_id,
                    destination=self._notification_destination(),
                    result="deduplicated", error=None,
                    dedupe_reason="already sent for this message")
                continue
            try:
                self._notify(message["text"], None)
            except Exception as exc:  # noqa: BLE001 - never lose the position
                self._record_notification(
                    kind=kind, parent_signal_id=parent_id,
                    destination=self._notification_destination(),
                    result="error", error=f"{type(exc).__name__}",
                    dedupe_reason=None)
                log("ERROR", "beta %s notification for parent %s failed "
                    "(%s: %s); the PAPER position is unaffected",
                    kind, parent_id, type(exc).__name__, exc)
                continue
            if dedupe_key is not None:
                self._beta_notified[dedupe_key] = True
            # AN EXIT BLOCKED MESSAGE IS MARKED ONLY AFTER IT HAS BEEN SENT.
            # The experiment persists the attempt number it has announced, so a
            # repeated tick cannot re-send it and a restart cannot resurrect it.
            # NOTE what is NOT here: nothing in this function can settle, close
            # or re-open a position, so a notification failure can never cause a
            # duplicate settlement -- the settlement happened first, in
            # ``BetaExperiment.settle_arm``, and is idempotent on ``status``.
            if kind == "beta_exit_blocked":
                for pid, arm, attempt in message.get("blocked_arms") or ():
                    try:
                        bridge.beta_exit_block_notified(pid, arm, attempt)
                    except Exception as exc:  # noqa: BLE001 - never lose the position
                        log("ERROR", "cannot record the announced beta exit "
                            "block for %s/%s: %s", pid, arm,
                            type(exc).__name__)
            self._record_notification(
                kind=kind, parent_signal_id=parent_id,
                destination=self._notification_destination(),
                result="accepted", error=None, dedupe_reason=None)

    def _notification_destination(self) -> str:
        """
        The configured outbound destination, REDACTED.

        Never the raw chat id or token: this string goes to a log and a
        journal, and a destination identifier is exactly the kind of value that
        must not be copied out of them.
        """
        telegram = getattr(self.config, "telegram", None)
        # ``bot_chat`` is the OUTBOUND destination the notifier is attached to.
        # ``chat`` is the INBOUND candidate source channel and is a different
        # thing entirely -- naming it here would record the wrong destination.
        destination = str(getattr(telegram, "bot_chat", "") or "")
        if not destination:
            return "<unset>"
        return f"telegram:bot_chat:...{destination[-4:]}"

    def _record_notification(
        self, *, kind: str, parent_signal_id: Any, destination: str,
        result: str, error: str | None, dedupe_reason: str | None,
    ) -> None:
        """
        One structured record per notification ATTEMPT.

        Deliberately does NOT claim delivery. ``result="accepted"`` means the
        notifier returned without raising, which is what the transport can
        actually attest; whether a human read it is not knowable here, and a
        record that implied otherwise would be a fabricated receipt.
        """
        record = {
            "kind": kind,
            "parent_signal_id": parent_signal_id,
            "destination": destination,
            "attempted_at": self.clock.now(),
            "result": result,
            "error_class": error,
            "dedupe_reason": dedupe_reason,
            "delivery": "unknown",
        }
        self.notification_log.append(record)
        try:
            # ``event`` is the journal's own first parameter, so the record's
            # payload is passed as keywords WITHOUT a colliding "event" key.
            self.journal.event("notification", self.state.trade_mode,
                               **record)
        except Exception as exc:  # noqa: BLE001 - observability never blocks
            log("ERROR", "cannot journal a notification record: %s", exc)

    async def _execute_paper_buy_execution_aware(
        self, report: DecisionReport, snapshot: AlertSnapshot, features
    ) -> None:
        """
        Execute an already-authorized PAPER BUY on every research arm.

        The runtime's decision is passed in and never recomputed. Each arm then
        sizes from its OWN virtual bankroll with the production sizing function,
        obtains a same-market-verified executable quote, and fills from that
        quote. A missing, stale or unidentifiable quote produces NO fill and a
        recorded opportunity -- never a mark-based fill.

        ORDERING: durable state is written BEFORE the entry alerts are sent.
        An alert is the operator's receipt for a fill, so it must never be able
        to describe a position the canonical state has not accepted -- a crash
        between the two would leave a receipt for a trade that does not exist.
        """
        bridge = await self._paper_bridge_instance()
        try:
            records = await bridge.handle_alert(
                snapshot=snapshot, report=report, features=features,
                channel=getattr(snapshot, "chat", "") or "",
            )
        except Exception as exc:  # noqa: BLE001 - never kill the runtime
            log("PAPER", "execution-aware entry failed (%s): %s",
                type(exc).__name__, exc)
            self._record_block(
                report, "EXECUTION_AWARE_PAPER_ERROR",
                f"{type(exc).__name__}: {exc}"[:200],
            )
            return
        blocked = False
        # Group the per-arm refusals so ONE signal produces ONE operator
        # message. The V2 FLOOR (BETA) experiment writes one record per arm, and
        # three arms refusing the same gate used to raise THREE byte-identical
        # BUY_BLOCKED alerts -- which is what made this read as a repeating
        # malfunction rather than one threshold doing its job. The accounting is
        # unchanged: every refused arm is still recorded and still counted.
        #
        # Arms that refused DIFFERENTLY stay separate messages, so nothing is
        # merged across genuinely different outcomes.
        traded_arms = tuple(r.arm for r in records if r.traded)
        grouped: dict[tuple[str, bool], tuple[list[str], str]] = {}
        for record in records:
            if record.traded:
                self.buys += 1
                continue
            if record.failure is None:
                continue
            blocked = True
            reason = str(getattr(record.failure, "value", record.failure))
            detail = str(getattr(record, "failure_detail", "") or "")[:200]
            arm = str(getattr(record, "arm", "") or "")
            key = (reason, _is_beta_arm(arm))
            grouped.setdefault(key, ([], detail))[0].append(arm)
        for (reason, is_beta), (arms, detail) in grouped.items():
            self._record_block(
                report, reason, detail,
                beta_arms=tuple(arms) if is_beta else (),
                traded_arms=traded_arms,
            )
        # The canonical PAPER position set is the ARMS' positions while
        # execution-aware PAPER is the active book. Every operator-facing reader
        # goes through ``open_positions_view``, so a notified fill is visible
        # there instead of only in the bridge's own records.
        self.persist_paper_dataset()
        self._flush_paper_bridge_messages()
        traded = [r for r in records if r.traded]

        # EVERY attempted execution-aware entry states an outcome.
        #
        # This used to set ``last_execution`` ONLY on a fill, so an authorized
        # BUY that no arm could execute left the previous verdict standing:
        # ``executed=False, reason=None``, with nothing in the journal. That is
        # precisely how today's incident hid -- 13 L25 approvals, 0 fills, and
        # no recorded cause anywhere an operator would look.
        if traded:
            # A BUY that produced at least one real fill is a fill, whatever
            # the other arms did.
            self.last_execution = {
                **(self.last_execution or {}),
                "outcome": PAPER_FILL_SUCCESS,
                "executed": True,
                "reason": None,
                "arms": [r.arm for r in traded],
                "arm_notional_usd": self._arm_notionals(traded),
                "at": self.clock.now(),
            }
        elif records:
            # No arm filled: the outcome is the most specific refusal seen.
            reasons = [
                str(getattr(r.failure, "value", r.failure) or "")
                for r in records if r.failure is not None
            ]
            primary = (reasons[0] if reasons else "EXECUTION_AWARE_NO_RESULT")
            self.last_execution = {
                **(self.last_execution or {}),
                "outcome": primary,
                "executed": False,
                "reason": primary,
                "detail": str(
                    getattr(records[0], "failure_detail", "") or "")[:200],
                "arms": [],
                "at": self.clock.now(),
            }
            log("REJECT", "%s execution-aware entry produced no fill: %s",
                (report.symbol or "?")[:12], primary)
        else:
            # No record at all -- the bridge produced nothing to judge by.
            self.last_execution = {
                **(self.last_execution or {}),
                "outcome": "EXECUTION_AWARE_NO_RESULT",
                "executed": False,
                "reason": "EXECUTION_AWARE_NO_RESULT",
                "detail": "the execution-aware engine returned no record",
                "arms": [],
                "at": self.clock.now(),
            }

    def persist_paper_dataset(self) -> None:
        """
        Flush the ``reports/live`` artefacts AND the durable PAPER state.

        Two destinations, because they answer two different questions. The
        ``reports/live`` artefacts are the research dataset: every signal and
        trade, for analysis. ``var/paper_state.json`` is the resumable book:
        which arms hold which positions right now, so a restart can rehydrate
        them.

        The previous implementation wrote only the artefacts. That is why
        ``rehydrate`` existed, passed its own tests, and was never called: a
        restart began with every arm empty and the operator's positions simply
        gone.
        """
        if self._paper_bridge is None:
            return
        try:
            self._paper_bridge.persist()
        except Exception as exc:  # noqa: BLE001 - a disk problem is not a trade
            log("PAPER", "cannot persist the PAPER dataset: %s",
                type(exc).__name__, exc)
        try:
            self._paper_bridge.persist_state()
        except Exception as exc:  # noqa: BLE001
            log("PAPER", "cannot persist the execution-aware PAPER state: %s",
                type(exc).__name__, exc)

    def paper_state_path(self) -> Any:
        """
        Where the execution-aware PAPER book is persisted.

        Exposed so a test can assert the file exists and rehydrate from it,
        rather than trusting that a restart would work because the function
        that does it is present in the module.
        """
        bridge = self._paper_bridge
        if bridge is None:
            return None
        return bridge._store().path

    def paper_account_store_path(self) -> Any:
        """
        The canonical PAPER state file backing the operator's book.

        This is the ``PaperStateStore`` path the execution-aware arms persist
        into. Alias kept for readability at the call sites that assert on it.
        """
        return self.paper_state_path()

    def rehydrate_paper_state(self) -> dict[str, Any]:
        """
        Resume the execution-aware arms' open positions after a restart.

        Called once when the bridge is built. A position that cannot be exited
        coherently -- no venue pin, no known token balance -- is reported and
        NOT resumed, because resuming it would mean inventing the balance or the
        market to sell it on.
        """
        if self._paper_bridge is None:
            return {"restored": [], "refused": []}
        report = self._paper_bridge.rehydrate()
        if report.get("restored"):
            log("PAPER", "resumed execution-aware positions: %s",
                ", ".join(report["restored"]))
        if report.get("refused"):
            log("WARNING", "execution-aware positions NOT resumed: %s",
                "; ".join(str(p) for p in report["refused"])[:400])
        return report

    def paper_bridge_status(self) -> dict[str, Any] | None:
        """
        The execution-aware PAPER status, or ``None`` when it is not running.

        The Telegram status message reads this so the operator sees the same
        numbers the journal has, rather than a second set computed elsewhere.
        """
        if self._paper_bridge is None:
            return None
        return self._paper_bridge.status()

    async def _execute_paper_buy(
        self, report: DecisionReport, snapshot: AlertSnapshot, features, quote
    ) -> None:
        from .execution.solana import WSOL_MINT
        book = self.open_book["positions"]
        prep_stages = self._current_stages
        if prep_stages is not None:
            prep_stages.mark("prepare_start", self.clock.now())

        # -- sizing ----------------------------------------------------
        # ONE function, ONE policy. Not a route target, not a band, not a
        # multiplier: equity in, position size out.
        #
        # The SELECTED PROFILE is passed so a profile that declares its own
        # researched size gets it (V2_FLOOR: a flat 10% of current equity).
        # Profiles that declare nothing get the shared ladder, unchanged. The
        # profile is read from the runtime state rather than from the report, so
        # sizing and entry can never be resolved from two different profiles.
        size = compute_position_size(
            self._entry_equity_usd(), profile=self.state.strategy_profile
        )
        ok, why = size_tradeable(size, available_cash_usd=self.account.cash)
        if not ok:
            log("REJECT", "%s sizing refused: %s", (report.symbol or "?")[:12], why)
            self.journal.unfilled(
                self.state.trade_mode,
                candidate_id=report.candidate_id,
                mint=report.mint,
                symbol=report.symbol,
                reason=why,
            )
            self._record_block(report, "sizing_refused", why)
            return
        sizing_note = size.rationale
        if prep_stages is not None:
            prep_stages.mark("prepare_end", self.clock.now())

        # -- wait out the real decision latency, then fill -------------
        stages = self._current_stages
        if stages is not None:
            stages.mark("quote_start", self.clock.now())
        decision_time = self.clock.now()
        latency_s = self.config.execution.decision_latency_ms / 1000.0
        await self._sleep(latency_s)
        fill_time = self.clock.now()
        if stages is not None:
            stages.mark("quote_end", self.clock.now())
            stages.mark("submit_start", self.clock.now())

        fill_quote = await self.market.quote_async(snapshot.mint or "")
        if stages is not None:
            stages.mark("submit_end", self.clock.now())
        entry_obs = fill_quote.observation
        state = (
            entry_obs.freshness(fill_time, self.config.market.stale_after_ms,
                                self.config.market.unusable_after_ms)
            if entry_obs is not None
            else DataState.MISSING
        )
        # FAIL-CLOSED at the LAST moment before the order is constructed. The
        # pre-order gate in _execute_buy ran BEFORE the decision-latency sleep;
        # this re-runs it against the REAL fill time, so a pipeline that crossed
        # the 15s budget during that sleep is refused rather than filled. The
        # gate can only ever refuse - a None here still requires a usable quote.
        gate_reason = self.stale_gate.check(
            signal_at=getattr(snapshot, "received_timestamp", None),
            now=fill_time,
            decision_at=report.decided_at,
            quote_at=decision_time,
            submit_at=fill_time,
            fill_at=fill_time,
        )
        stale_at_fill = gate_reason
        if stale_at_fill is None and state is DataState.STALE:
            # The pipeline was fast, but the ONLY available quote is already
            # past the freshness budget. It cannot become an order.
            stale_at_fill = STALE_QUOTE
        if stale_at_fill is not None or state in (DataState.MISSING, DataState.UNUSABLE):
            reason = stale_at_fill or STALE_MARKET_DATA
            log(
                "REJECT",
                "%s no fillable market at +%.0fms (state=%s, %s); trade abandoned",
                (report.symbol or "?")[:12], latency_s * 1000, state.value, reason,
            )
            self.ledger.event(
                EventType.UNFILLED,
                {
                    "candidate_id": report.candidate_id,
                    "mint": report.mint,
                    "symbol": report.symbol,
                    "reason": reason if stale_at_fill else "no_quote_at_fill_time",
                    "data_state": state.value,
                    "stale_reason": stale_at_fill,
                },
            )
            self.account.unfilled_attempts += 1
            if gate_reason is None:
                # The gate itself saw nothing; the STALE quote state was the
                # refusal, so count it so the reason distribution stays honest.
                self.stale_gate.refused.append(reason)
            self._record_block(
                report, _block_reason_for_stale(reason),
                f"market state at fill was {state.value}"
                + (f" ({reason})" if stale_at_fill else ""),
            )
            return

        # The EXECUTABLE layer for a PAPER entry. PAPER does not settle, so this
        # is an OBSERVATION of what a venue would charge, never a price anything
        # was filled at -- and it is recorded in its own field, separately from
        # the mark and separately from the simulated fill.
        #
        # It is NOT a gate. A PAPER book exists to measure a strategy; refusing
        # its fills because a venue quote was unavailable would silently shrink
        # the sample. The absence is recorded with a reason instead, which is
        # what makes a PAPER sample auditable rather than merely smaller.
        paper_quote = None
        paper_quote_reason = ""
        if self.config.real.paper_executable_quote_enabled:
            try:
                dec = await self._token_decimals(report.mint or "")
                # THE QUOTE IS DENOMINATED IN LAMPORTS, AND THE SIZE IS IN USD,
                # SO A SOL/USD REFERENCE IS REQUIRED TO CROSS BETWEEN THEM.
                #
                # This used to compute ``units * price * 1e9`` where ``units``
                # was ``position_usd / price`` -- which cancels back to
                # ``position_usd * 1e9``, i.e. it spent the USD figure as though
                # one SOL were one dollar. At a $122 SOL that asked the venue to
                # price an order ~122x the real one, and the extra price impact
                # came back inside ``effective_price``. The entry then filled at
                # ``max(mark, quote)`` -- see ``simulate_fill`` -- so a sizing
                # unit error landed directly in the paper fill price.
                #
                # ``self._sol_price_usd`` is the memo the canonical resolver
                # fills; it is the same value handed to the provider below as
                # ``quote_asset_usd``. Nothing here touches ``real_backend``, so
                # the PAPER process still never constructs the REAL executor.
                # No reference means NO QUOTE, with the reason recorded: an
                # observation priced off a wrong notional is worse than its
                # absence, which this path already records honestly.
                sol_usd = self._sol_price_usd
                if dec is None:
                    paper_quote_reason = (
                        "token decimals unreadable from chain")
                elif not sol_usd or sol_usd <= 0:
                    paper_quote_reason = (
                        "no SOL/USD reference, so a USD notional cannot be "
                        "expressed in lamports")
                else:
                    lamports = int(
                        size.position_usd / float(sol_usd) * 1e9
                    ) or 1
                    paper_quote = await self._paper_quote_provider().quote_buy(
                        token_mint=report.mint or "",
                        quote_mint=WSOL_MINT,
                        quote_amount_raw=lamports,
                        token_decimals=dec,
                        slippage_bps=self.config.real.slippage_bps,
                        quote_asset_usd=sol_usd,
                    )
            except Exception as exc:  # noqa: BLE001 - never blocks a PAPER fill
                paper_quote = None
                paper_quote_reason = f"{type(exc).__name__}: {exc}"[:200]

        fill = simulate_fill(
            FillInputs(
                side="entry",
                quantity=(size.position_usd / features.price) if features.price else 0.0,
                notional_hint_usd=size.position_usd,
                quoted_price=features.price,
                decision_time=decision_time,
                observation=entry_obs,
                observed_at=fill_time,
                quote=paper_quote,
                quote_missing_reason=(paper_quote_reason
                                      or "executable quote disabled by config"),
            ),
            self.config.execution,
            data_state=state,
        )
        if not fill.filled or fill.fill_price is None:
            book.open_from_decision(
                report, alert=snapshot, fill=fill, now=fill_time,
                sizing_note=sizing_note,
            )
            self.account.unfilled_attempts += 1
            self._record_block(
                report, "fill_refused",
                f"{fill.reason}"
                + (f" -- {fill.detail}" if fill.detail else ""),
            )
            return

        position = book.open_from_decision(
            report,
            alert=snapshot,
            fill=fill,
            now=fill_time,
            # Stamped once, at open. This position's exit policy is fixed from
            # here on, whatever the operator selects later.
            strategy_profile=self.state.strategy_profile,
            sizing_note=(
                f"{sizing_note} -> ${size.position_usd:.6f} notional "
                f"(fee ~${size.estimated_fee_usd:.6f}, "
                f"slippage ~${size.estimated_slippage_usd:.6f})"
            ),
            context={
                "liquidity_usd": entry_obs.liquidity_usd,
                "market_cap_usd": entry_obs.market_cap_usd,
                "token_age_s": entry_obs.token_age_s,
                "alert_lag_ms": (
                    (snapshot.received_timestamp - snapshot.source_timestamp) * 1000.0
                    if snapshot.source_timestamp else None
                ),
            },
        )
        if position.status is not PositionStatus.OPEN:
            # The book refused the position. Accounted for exactly like any
            # other refusal, so the cash/exposure check downstream can never
            # see a BUY that quietly produced nothing.
            self._record_block(
                report, "fill_refused",
                f"position status is {getattr(position.status, 'value', '?')}",
            )
            return
        self.buys += 1
        self._record_fill(report, position, mode_outcome=PAPER_FILL_SUCCESS)
        self._mark(position, now=self.clock.now())
        self.journal.paper_entry(
            candidate_id=report.candidate_id,
            mint=report.mint,
            symbol=report.symbol,
            position_id=position.position_id,
            strategy=l25.ACTIVE_STRATEGY,
            v3c=dict(report.v3c.as_dict()) if report.v3c else None,
            l25=(
                dict(report.l25.as_dict())
                if getattr(report, "l25", None) is not None
                else None
            ),
            size_usd=size.position_usd,
            position_pct=size.position_pct,
            sizing=sizing_note,
            quoted_price=features.price,
            fill_price=fill.fill_price,
            quantity=fill.quantity,
            notional_usd=fill.notional_usd,
            entry_fee_usd=fill.fee_usd,
            entry_slippage_usd=fill.slippage_usd,
            fee_bps=fill.fee_bps,
            slippage_bps=fill.slippage_bps,
            data_state=fill.data_state.value,
        )
        stages = self._current_stages
        if stages is not None:
            stages.mark("fill", self.clock.now())
        self._notify(
            _render_paper_buy(report, features, size)
            + _equity_footer(self._paper_equity_now()),
            None,
        )

    # ------------------------------------------------------------------
    # REAL execution
    # ------------------------------------------------------------------

    async def _execute_real_buy(
        self, report: DecisionReport, snapshot: AlertSnapshot, features
    ) -> None:
        """
        Open a REAL position.

        The full pre-trade safety sequence runs BEFORE the order, and any
        failure means no order. The order is:

            armed  ->  wallet  ->  RPC  ->  quote provider  ->  sizing matches
            policy  ->  SOL for transaction costs  ->  venue quote valid
            ->  exposure  ->  duplicates  ->  no concurrent operation

        Then, and only then, the swap is built, signed, broadcast and confirmed.
        A trade is never recorded as filled before on-chain confirmation, and a
        simulated number is never substituted for an actual one.
        """
        mint = report.mint or ""
        symbol = report.symbol or "?"
        stages = self._current_stages

        def _stage(name: str) -> None:
            if stages is not None:
                stages.mark(name, self.clock.now())

        # -- 1. WALLET OBSERVATION (B2) --------------------------------
        # The authoritative wallet state MUST be loaded before sizing. Without
        # this, equity is always None and sizing refuses every order. It fails
        # CLOSED: no observation -> no order, with an explicit reason.
        _stage("real_wallet_observation_start")
        observed, why_wallet = await self.observe_real_wallet()
        _stage("real_wallet_observation_end")
        if not observed:
            self._refuse_real(symbol, why_wallet)
            self._record_block(report, "wallet_observation_failed", why_wallet)
            return

        # -- 2. SIZING off the ACTUAL wallet equity --------------------
        equity = self._entry_equity_usd()
        size = compute_position_size(
            equity, profile=self.state.strategy_profile
        )
        ok, why = size_tradeable(size)
        if not ok:
            self._refuse_real(symbol, "sizing: " + why)
            self._record_block(report, "sizing_refused", why)
            return

        position_usd = size.position_usd
        sol_price = self._sol_price_usd or await self._backend_sol_price()
        lamports = (
            int(position_usd / sol_price * 1_000_000_000) if sol_price else 0
        )

        # -- 3. PRE-TRADE SAFETY + FRESH VENUE QUOTE -------------------
        _stage("real_pretrade_start")
        checks = await self._real_pretrade_checks(
            mint=mint, size=size, lamports=lamports, sol_price=sol_price,
            v3c_authorized=bool(report.v3c_authorized),
        )
        _stage("real_pretrade_end")
        failures = [c for c in checks if not c.ok]
        if failures:
            detail = ", ".join(
                "{} ({})".format(c.name, c.detail) for c in failures
            )
            self._refuse_real(symbol, "pre-trade safety: " + detail)
            self._record_block(
                report, self._pretrade_block_reason(failures), detail
            )
            return

        # -- 4. FILL-TIME STALE GATE (B7) ------------------------------
        # The pre-order gate ran BEFORE the decision latency; this re-runs it
        # against the newest observation. It can only ever REFUSE. The quote
        # that will actually be executed has its own freshness budget.
        _stage("real_stale_gate_start")
        fill_now = self.clock.now()
        stale = self.stale_gate.check(
            signal_at=getattr(snapshot, "received_timestamp", None),
            now=fill_now,
            decision_at=report.decided_at,
            quote_at=self._real_quote_at,
            submit_at=fill_now,
        )
        quote_age_ms = (
            (fill_now - self._real_quote_at) * 1000.0
            if self._real_quote_at is not None else 0.0
        )
        _stage("real_stale_gate_end")
        if stale is not None or quote_age_ms > self.config.real.quote_max_age_ms:
            reason = stale or STALE_QUOTE
            self.account.unfilled_attempts += 1
            self._record_block(
                report, _block_reason_for_stale(reason),
                f"fill-time freshness: signal/staleness={stale or 'fresh'}, "
                f"quote age {quote_age_ms:.0f}ms "
                f"(budget {self.config.real.quote_max_age_ms:.0f}ms)",
            )
            return

        # -- 4b. ENTRY EXECUTABLE-QUOTE GUARD --------------------------
        #
        # The 2026-10-02 $GOAT entry was authorised at an observed mark of
        # 1.783e-05 USD/token and filled at 3.1079e-05 -- 74.3% dearer. Nothing
        # asked, because the venue quote that sizes the order was never compared
        # with the mark the order is justified by.
        #
        # This asks, and refuses. It does not try to make the trade happen at the
        # mark price: the goal is only to decline an entry whose real cost is
        # materially worse than the market the system believes it is looking at.
        # It FAILS CLOSED -- no quote, a stale quote, an unpriceable quote, or a
        # mark that cannot be compared are all refusals, never permissions.
        entry_quote = None
        entry_verdict = None
        entry_mark_obj = None
        if self.config.real.entry_quote_guard_enabled:
            entry_quote, entry_verdict, entry_mark_obj = \
                await self._entry_quote_guard(
                    mint=mint, lamports=lamports, sol_price=sol_price,
                    features=features)
            self.journal.event(
                "real_entry_quote_guard", TradeMode.REAL, mint=mint,
                symbol=symbol, allowed=bool(entry_verdict.allowed),
                reason=entry_verdict.reason, pct=entry_verdict.pct,
                mark=entry_verdict.mark, quote=entry_verdict.quote,
                threshold_pct=entry_verdict.threshold_pct,
                fail_closed=entry_verdict.fail_closed,
                cross_venue=entry_verdict.cross_venue,
                detail=entry_verdict.detail,
                executable_quote=(entry_quote.as_dict()
                                  if entry_quote is not None else None),
            )
            if not entry_verdict.allowed:
                self._refuse_real(
                    symbol,
                    f"mark/quote guard: {entry_verdict.reason} "
                    f"({entry_verdict.detail})",
                )
                self._record_block(
                    report,
                    f"entry_quote_guard_{entry_verdict.reason}",
                    entry_verdict.detail,
                )
                return

        self._notify(
            _render_real_buy(report, size, lamports / 1e9, sol_price), None
        )

        # -- 5. IRREVERSIBLE SEQUENCE --------------------------------
        # idempotency reservation -> build -> simulate -> sign -> broadcast ->
        # confirm -> reconcile, all inside the executor, all stage-marked.
        fill = await self.real_executor.open_position(
            mint=mint,
            position_lamports=lamports,
            slippage_bps=self.config.real.slippage_bps,
            prechecks=checks,
            intent_store=self.execution_intents,
            logical_key=self._real_logical_key("ENTRY", mint),
            stage=_stage,
        )
        if fill.requires_recovery:
            # The chain CONFIRMED this swap. It is NOT a refused entry and NOT
            # an ambiguous "we don't know": money moved, we just could not read
            # the amount back. Journalling it as unfilled/blocked is what made
            # the $Boy buy vanish while the wallet held the tokens.
            #
            # It is recorded as a distinct outcome, it is NOT retried (a second
            # broadcast would double-buy), and the pending intent is left
            # unresolved so the restarts/recovery path picks it up from
            # on-chain state rather than re-sending.
            self.journal.unfilled(
                TradeMode.REAL,
                candidate_id=report.candidate_id,
                mint=mint,
                symbol=symbol,
                status=fill.status,
                reason=fill.ambiguity_reason or fill.status,
                requested_usd=position_usd,
                requested_lamports=lamports,
            )
            self._notify(
                "{err} <b>REAL BUY ยืนยันแล้ว แต่ยังกระทบไม่ได้</b>\n"
                "<b>${sym}</b>\n"
                "Status: {st}\n"
                "เหตุผล: {why}\n\n"
                "<b>เงินอาจเข้าบัญชีแล้ว</b> — ไม่มีการส่งซ้ำ "
                "(ไม่ double-buy) และจะกู้คืนจำนวนจริงจาก on-chain".format(
                    err=_ERR, sym=symbol, st=fill.status,
                    why=fill.ambiguity_reason or "-",
                ),
                None,
            )
            # Deliberately NOT fill_refused: that reason means "no order was
            # placed", which is false here.
            self._record_confirmed_unreconciled(report, fill)
            return

        if not fill.filled:
            # MICRO_REAL_EXECUTION_UNSUPPORTED, QUOTE_REJECTED, SIGN_FAILED,
            # SIMULATION_FAILED, BROADCAST_REJECTED, AMBIGUOUS ... all land
            # here, and none of them silently becomes a fill.
            self.journal.unfilled(
                TradeMode.REAL,
                candidate_id=report.candidate_id,
                mint=mint,
                symbol=symbol,
                status=fill.status,
                reason=fill.ambiguity_reason or fill.status,
                requested_usd=position_usd,
                requested_lamports=lamports,
            )
            self._notify(
                "{err} <b>REAL BUY ไม่สำเร็จ</b>\n"
                "<b>${sym}</b>\n"
                "Status: {st}\n"
                "เหตุผล: {why}\n\n"
                "ไม่มีการเปิดออเดอร์".format(
                    err=_ERR, sym=symbol, st=fill.status,
                    why=fill.ambiguity_reason or "-",
                ),
                None,
            )
            self._record_block(
                report, "fill_refused",
                f"{fill.status}: {fill.ambiguity_reason or '-'}",
            )
            return

        record = RealFillRecord.from_onchain_fill(fill)
        from .execution.solana import WSOL_MINT
        # Decimals come from CHAIN, not from a caller-supplied guess. A wrong
        # scale here would make every per-token price on this position wrong by
        # a power of ten, silently.
        entry_decimals = await self._token_decimals(mint)
        try:
            pos = self.real_account.register_entry(
                mint=mint,
                symbol=symbol,
                fill=record,
                token_amount=int(fill.output_amount or 0),
                # The entry REFERENCE price, and its KIND. On 2026-10-02 this
                # was a DexScreener mark stored in a field every consumer read
                # as the price paid; it is now labelled, and the settlement-
                # derived fill price is a separate, authoritative field.
                entry_price_usd=float(features.price or 0.0),
                entry_price_kind="OBSERVED_MARK",
                entry_mark=entry_mark_obj,
                size_usd=position_usd,
                now=self.clock.now(),
                sol_price_usd=sol_price,
                exit_style=self._real_exit_style(report),
                token_decimals=entry_decimals,
                quote_mint=WSOL_MINT,
                provider=(fill.quote_provider or fill.route_metadata.get(
                    "provider") if isinstance(fill.route_metadata, dict)
                    else None),
                venue=entry_quote.venue if entry_quote is not None else None,
                pool=entry_quote.pool if entry_quote is not None else None,
                rent_lamports=fill.rent_lamports,
                wallet_debit_lamports=fill.wallet_debit_lamports,
                # Same stamping as PAPER, on the same registration path. The
                # 60s cap is a property OF THE POSITION, not a second REAL
                # executor: a max-hold exit goes through _execute_real_sell
                # like any other exit decision.
                strategy_profile=self.state.strategy_profile,
            )
        except ValueError as exc:
            # A confirmed fill we cannot book is an accounting failure, and it
            # is reported rather than swallowed: the chain moved, we know it
            # moved, and the record must exist.
            self.journal.event(
                "real_position_book_failed",
                TradeMode.REAL,
                candidate_id=report.candidate_id,
                mint=mint,
                signature=fill.signature,
                reason=str(exc),
            )
            log("ERROR", "REAL fill %s could not be booked: %s",
                fill.signature, exc)
            self._record_block(report, "execution_exception", str(exc))
            return

        self.journal.real_entry(
            signature=fill.signature,
            candidate_id=report.candidate_id,
            mint=mint,
            symbol=symbol,
            strategy=l25.ACTIVE_STRATEGY,
            v3c=dict(report.v3c.as_dict()) if report.v3c else None,
            l25=(
                dict(report.l25.as_dict())
                if getattr(report, "l25", None) is not None
                else None
            ),
            requested_usd=position_usd,
            requested_lamports=lamports,
            actual_input_amount=fill.input_amount,
            actual_output_amount=fill.output_amount,
            actual_execution_price=fill.execution_price,
            quote_price=fill.quote_price,
            actual_fee_lamports=fill.fee_lamports,
            actual_fee_sol=fill.fee_sol,
            actual_slippage=record.actual_slippage,
            price_impact_pct=fill.price_impact_pct,
            commitment=fill.commitment,
            slot=fill.slot,
            confirmed_at=fill.confirmed_at,
            route_labels=list(fill.route_labels),
            # -- the three layers, kept apart -----------------------------
            input_amount_kind=fill.input_amount_kind,
            swap_input_amount=fill.swap_input_amount,
            wallet_debit_lamports=fill.wallet_debit_lamports,
            rent_lamports=fill.rent_lamports,
            token_decimals=(pos.entry_token_decimals
                            if pos.entry_token_decimals is not None
                            else entry_decimals),
            entry_price_kind="OBSERVED_MARK",
            entry_observed_mark_usd=(pos.entry_price_usd or None),
            entry_execution_price_usd=pos.entry_execution_price_usd,
            entry_provider=pos.entry_provider,
            entry_venue=pos.entry_venue,
            entry_pool=pos.entry_pool,
            entry_cost_usd=pos.entry_cost_usd,
            entry_cost_usd_trading_only=pos.entry_cost_usd_trading_only,
            entry_cost_components=dict(pos.entry_cost_components),
            entry_cost_basis_reason=pos.entry_cost_basis_reason,
            cross_venue_detected=pos.cross_venue_detected,
            cross_venue_detail=pos.cross_venue_detail,
            entry_quote=(entry_quote.as_dict() if entry_quote is not None
                         else None),
        )
        self._record_fill(report, pos, mode_outcome=REAL_FILL_SUCCESS)
        # Persist the REAL position durably, BEFORE the notification, so a
        # crash between booking and notifying cannot orphan it.
        self.save_real_state()
        self._notify(
            "{ok} <b>BUY CONFIRMED</b>\n"
            "<b>${sym}</b>\n"
            "Actual size: <b>${size:,.6f}</b>\n"
            "Actual fill: {out}\n"
            "Actual fee: {fee:.6f} SOL\n"
            "Tx: <code>{sig}</code>".format(
                ok=_OK, sym=symbol, size=position_usd,
                out=fill.output_amount, fee=fill.fee_sol or 0.0,
                sig=fill.signature,
            ),
            None,
        )

    #: Fallback bucket for a rejection that no gate and no risk flag named.
    #: Explicit, because a blank key in a reason histogram is indistinguishable
    #: from "no rejects".
    UNNAMED_REJECT_REASON = "UNNAMED_REJECT"

    @staticmethod
    def _reject_reason_key(report: Any) -> str:
        """
        The ONE term that classifies this rejection, or ``UNNAMED_REJECT``.

        Preference order, and why:

        1. the ACTIVE strategy gate's own status code -- ``V3C_REJECTED_MARKET_CAP``,
           ``LIQUIDITY_MC_BELOW_THRESHOLD``. It is the gate that actually
           refused, and it is the token that appears in the log and the journal,
           so the histogram is directly comparable with them.
        2. the first BLOCK-severity risk flag, when a strategy declined the
           candidate for a condition rather than a gate verdict.
        3. the first risk flag of any severity, which is still a real cause.
        4. ``UNNAMED_REJECT``. Never a blank and never a guess.

        Deterministic by construction: it reads an ordered list, so the same
        report always yields the same key and the histogram stays a count of
        decisions rather than of iteration order.
        """
        gate = getattr(report, "l25", None) or getattr(report, "v3c", None)
        for name in ("status_code", "status"):
            value = getattr(gate, name, None)
            if value:
                return str(value)
        flags = tuple(getattr(report, "risk_flags", ()) or ())
        for severity in ("block", "warn", "info"):
            for flag in flags:
                if str(getattr(flag, "severity", "")) == severity:
                    code = getattr(flag, "code", None)
                    if code:
                        return str(code)
        # Class-qualified on purpose: this is a staticmethod, so the bare name
        # is not in scope at call time.
        return PaperTrader.UNNAMED_REJECT_REASON

    def _record_reject_reason(self, report: Any) -> str:
        """Count this rejection under its own cause and return that cause."""
        reason = self._reject_reason_key(report)
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1
        return reason

    def _log_strategy_reject(self, gate, symbol: str) -> None:
        """
        Log a refusal by the ACTIVE strategy, naming the operative gate.

        The shape the operator needs when a candidate passes V3-C but fails
        L25's liquidity gate, because that is the case where "V3-C said no" is
        actively misleading::

            L25_REJECT reason=LIQUIDITY_MC_BELOW_THRESHOLD
            f_liq_to_mcap=0.8412 threshold=0.9647

        Only the ratio and the threshold are printed - both are public market
        figures, no credential is involved. When the delegated V3-C leg refused
        instead, its own reason is logged so the two cases stay distinguishable.
        """
        code = getattr(gate, "status_code", None) or getattr(gate, "status", "?")
        ratio = getattr(gate, "f_liq_to_mcap", None)
        if code == "L25_REJECT":
            log(
                "STRATEGY",
                "%s_REJECT reason=%s f_liq_to_mcap=%.4f threshold=%.4f "
                "strategy=%s symbol=%s",
                l25.ACTIVE_STRATEGY,
                gate.status,
                ratio,
                float(getattr(gate, "min_liq_to_mcap", l25.L25_MIN_LIQ_TO_MCAP)),
                l25.ACTIVE_STRATEGY,
                symbol[:12],
            )
        elif code == "L25_UNTESTABLE":
            log(
                "STRATEGY",
                "%s_REJECT reason=UNTESTABLE strategy=%s symbol=%s detail=%s",
                l25.ACTIVE_STRATEGY, l25.ACTIVE_STRATEGY, symbol[:12],
                getattr(gate, "detail", ""),
            )
        elif getattr(gate, "status", None) == l25.REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD:
            log(
                "STRATEGY",
                "%s_REJECT reason=LIQUIDITY_MC_BELOW_THRESHOLD f_liq_to_mcap=%.4f "
                "threshold=%.4f strategy=%s symbol=%s",
                l25.ACTIVE_STRATEGY,
                ratio,
                float(getattr(gate, "min_liq_to_mcap", l25.L25_MIN_LIQ_TO_MCAP)),
                l25.ACTIVE_STRATEGY,
                symbol[:12],
            )
        else:
            log(
                "STRATEGY",
                "%s_REJECT reason=%s strategy=%s symbol=%s detail=%s",
                l25.ACTIVE_STRATEGY, code, l25.ACTIVE_STRATEGY, symbol[:12],
                getattr(gate, "detail", ""),
            )

    def _refuse_real(self, symbol: str, reason: str) -> None:
        """Refuse a REAL order and report the exact reason. Fail closed."""
        log("REJECT", "REAL %s refused: %s", symbol, reason)
        self.journal.event(
            "real_refused", TradeMode.REAL, symbol=symbol, reason=reason
        )
        self._notify(
            "{err} <b>REAL ปฏิเสธ</b>\n<b>${sym}</b>\n{why}".format(
                err=_ERR, sym=symbol, why=reason
            ),
            None,
        )

    async def real_equity(self):
        """
        The authoritative REAL equity, computed from the configured wallet.

        ONE implementation, consumed by the armed-REAL message, the preflight and
        the sizing guard, so those three cannot disagree. Native SOL from the
        live RPC, valued at the SOL/USD REFERENCE -- the same canonical provider
        PAPER sizing uses, not a swap quote; a failure at either step yields an
        explicit unavailable result rather than a silent zero.

        EVERY SUPPORTED ASSET IS INCLUDED. This previously valued native SOL and
        nothing else, so a wallet holding wrapped SOL or USDC reported an equity
        that silently excluded that money -- and REAL sizing would then size a
        position against a fraction of the account. WSOL is valued at the SOL
        price automatically (it is the same asset twice); a held asset whose
        price cannot be read makes the WHOLE figure unavailable rather than
        producing a plausible-looking partial sum.
        """
        from .real_equity import (
            AssetHolding,
            compute_real_equity_multi,
        )

        # The EXISTING, already-tested reader, so there is one balance
        # implementation and this cannot become a second one that disagrees.
        lamports = None
        masked = ""
        reason = ""
        try:
            state = await self.real_backend.wallet_state()
            lamports = state.get("lamports")
            masked = state.get("masked") or ""
        except Exception as exc:  # noqa: BLE001 - a failed read is NOT a zero
            log("ERROR", "native SOL balance unavailable: %s",
                type(exc).__name__)
            reason = f"native SOL balance unavailable ({type(exc).__name__})"

        price = await self._backend_sol_price()
        if price is None and not reason:
            reason = "SOL/USD valuation unavailable"

        holdings = await self._real_wallet_holdings(price)

        # PROVENANCE of the reference, recorded beside the figure so the equity
        # is auditable rather than a bare float. Read from the snapshot the
        # resolver cached, never re-fetched here.
        snap = self._sol_usd_snapshot
        provenance: dict[str, Any] = {}
        if price is not None and snap is not None:
            try:
                provenance = {
                    "sol_usd_source": str(getattr(snap, "source", "") or ""),
                    "sol_usd_timestamp": float(snap.timestamp),
                    "sol_usd_age_ms": float(snap.age_ms()),
                }
            except (AttributeError, TypeError, ValueError):
                provenance = {}

        result = compute_real_equity_multi(
            lamports, price, holdings, masked, **provenance)
        if not result.available and reason:
            # keep the more specific per-asset reason; the balance reason only
            # wins when it is the thing that actually failed
            if "no readable USD price" not in result.reason:
                result = replace(result, reason=reason)
        # The authoritative result is cached so the SYNC status renderer can show
        # the same number the sizing guard used. One measurement, one number.
        self._real_equity_cache = result
        return result

    async def _real_wallet_holdings(
        self, sol_price_usd: float | None
    ) -> list:
        """
        Every non-native balance in the configured REAL wallet, as holdings.

        Read best-effort: a wallet with no token accounts is the normal case, and
        an RPC that cannot list them must not be treated as "no holdings",
        because that would understate equity. So a failure here returns nothing
        and the caller still gets a valid SOL-only figure -- but only because
        an inability to LIST is not the same as an empty wallet, the log says so,
        and SOL-only equity is never presented as a full valuation.
        """
        from .execution.solana import WSOL_MINT
        from .real_equity import AssetHolding

        holdings: list = []
        try:
            accounts = await self.real_backend.rpc.token_balances(
                self.real_backend.pubkey)
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "REAL wallet token balances unreadable (%s); equity "
                "will be SOL-only and UNDERSTATED if the wallet holds any SPL "
                "asset", type(exc).__name__)
            return holdings

        for acc in accounts or ():
            try:
                amount = int(acc.get("amount"))
                decimals = int(acc.get("decimals"))
            except (TypeError, ValueError, AttributeError):
                continue
            if amount == 0:
                # nothing to value, and requiring a price for an empty balance
                # would make an idle wallet permanently unsizable
                continue
            mint = str(acc.get("mint") or "")
            symbol = str(acc.get("symbol") or "")
            px = sol_price_usd if (symbol.upper() in ("WSOL", "WRAPPED SOL")
                                   or mint == WSOL_MINT) else None
            if px is None:
                px = await self._real_asset_price_usd(mint, symbol)
            holdings.append(AssetHolding(mint, symbol, amount, decimals, px))
        return holdings

    async def _real_asset_price_usd(
        self, mint: str, symbol: str
    ) -> float | None:
        """
        USD price per WHOLE token for one held SPL asset, or None if unreadable.

        ASKED, NEVER ASSUMED. A stablecoin is worth its peg only when the peg is
        actually being fetched, so a hardcoded 1.0 is deliberately out of scope:
        it is a guess dressed as a measurement, and the fail-closed rule exists
        precisely to keep guesses out of REAL sizing. This asks the venue and
        reports failure.

        THE QUOTE IS asset -> WSOL, then multiplied by the SOL price. Quoting
        ``asset -> USDC`` would be the obvious-looking choice and is WRONG for
        USDC itself: a USDC -> USDC quote has no route, so the price of the most
        important settlement asset would be the one asset that could never be
        priced. WSOL is the leg that always has a route, and the SOL price is
        already measured by the same backend.

        The quote is for a MEANINGFUL notional, not one raw unit. Quoting 1 raw
        unit of a 6-decimal asset asks the venue to price 1e-6 of a token, which
        real routers refuse or answer with rounding noise. So it quotes one WHOLE
        token and the answer IS the per-token price.

        WSOL is never routed here: it is the same asset as SOL and the caller
        values it as SOL, because quoting the wrapper against itself would ask the
        venue a meaningless question.
        """
        from .execution.solana import WSOL_MINT

        if not mint or mint == WSOL_MINT:
            return None
        sol_px = await self._backend_sol_price()
        if sol_px is None:
            return None

        # One whole token, in the asset's own decimals. A wrong guess here only
        # changes the NOTIONAL asked for, never the resulting per-token price,
        # because both sides of the ratio scale together.
        try:
            supply = await self.real_backend.rpc.get_token_supply(mint)
            decimals = int((supply or {}).get("decimals", 6))
        except Exception:  # noqa: BLE001
            decimals = 6
        one_token = 10 ** max(0, decimals)

        try:
            q = await self.real_backend.provider.quote(
                mint, WSOL_MINT, one_token, 100)
        except Exception as exc:  # noqa: BLE001
            log("DEBUG", "no price for held asset %s (%s)",
                symbol or mint[:8], type(exc).__name__)
            return None
        if q is None:
            return None
        try:
            lamports_per_token = float(q.expected_out_amount)
        except (TypeError, ValueError, AttributeError):
            return None
        if not (lamports_per_token > 0) or lamports_per_token != lamports_per_token:
            return None
        # lamports -> SOL -> USD, for ONE whole token.
        return (lamports_per_token / 1_000_000_000) * float(sol_px)


    def _paper_equity_now(self) -> float | None:
        """
        PAPER equity right now, or None if it cannot be marked.

        Never raises into the notification path: an unmarkable account yields
        None, which renders as "unavailable" rather than a fake $0.0000.
        """
        try:
            return round(self.account.equity(self._refresh_marks()), 6)
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "paper equity unavailable: %s", type(exc).__name__)
            return None

    def _arm_notionals(self, records) -> dict[str, float]:
        """USD notional committed per arm, read from the arms' own positions."""
        out: dict[str, float] = {}
        bridge = self._paper_bridge
        for record in records:
            account = bridge.accounts.get(record.arm) if bridge else None
            position = getattr(account, "position", None) if account else None
            if position is not None:
                out[record.arm] = float(
                    getattr(position, "usd_notional", 0.0) or 0.0)
        return out

    def _record_arm_trades(self, trades) -> list:
        """
        Book the arms' settlements into the canonical PAPER record.

        The arms' realized PnL comes from two executable fills, which is a
        DIFFERENT accounting basis from the legacy book's marked-to-market
        trades. So the PnL itself is NOT merged into the legacy account's
        realized total -- mixing the two bases is exactly what the
        ``execution_aware_paper`` status block exists to prevent.

        What is recorded here is the operator-facing fact: a position was
        opened and closed, so the OPEN POSITIONS view must be able to stop
        showing it and the CLOSED POSITIONS view must be able to show it.

        RETURNS THE SETTLEMENTS THAT ARE NOW DURABLE, and nothing else. The
        close notification is driven from that return value, because "the
        trade closed" and "the closed trade is persisted" are two different
        facts and a receipt must not imply the second one it cannot prove.
        A write that fails is reported and its receipt is withheld.
        """
        durable: list = []
        for trade in trades:
            try:
                if not self._book_arm_trade(trade):
                    continue
            except Exception as exc:  # noqa: BLE001 - one bad record must
                # not cost the operator the close receipts for every OTHER
                # arm that settled in the same tick. Surfaced, never silent.
                symbol = "?"
                position = getattr(trade, "position", None)
                if position is not None:
                    symbol = getattr(position, "symbol", None) or "?"
                log("ERROR", "PAPER_CLOSE_PERSIST_FAILED arm=? symbol=%s "
                    "error=%s: %s", symbol, type(exc).__name__, exc)
                self.journal.event(
                    "paper_close_persist_failed", TradeMode.PAPER,
                    symbol=symbol, error=f"{type(exc).__name__}: {exc}")
                continue
            durable.append(trade)
        return durable

    def _book_arm_trade(self, trade) -> bool:
        """
        Record ONE arm settlement durably. True only if it committed.

        The append to the ledger is the commit point: it is fsync'd, and it is
        the same file the CLOSED POSITIONS view reads back, so a record that
        returns from here is a record the operator can see and a restart cannot
        lose. Everything the operator is told happens after this returns True.
        """
        failure = getattr(trade, "failure", None)
        pnl = getattr(trade, "pnl", None)
        position = getattr(trade, "position", None)
        symbol = getattr(position, "symbol", None) or "?"
        arm = getattr(position, "arm", "?")
        if failure is not None and str(
                getattr(failure, "value", failure)) != "NONE":
            log("EXECUTE", "arm exit did not settle (%s) %s: %s",
                arm, symbol,
                getattr(trade, "failure_detail", "") or "")
            return False
        realized = float(getattr(pnl, "realized_pnl_usd", 0.0) or 0.0)

        # THE STABLE IDENTITY OF THIS SETTLEMENT.
        #
        # One arm, one signal, one settlement: ``arm::signal_id`` can only ever
        # name one closed position, so it is the idempotency key for both the
        # write below and the reader in ``closed_positions_view``.
        signal_id = getattr(position, "signal_id", None) or ""
        settlement_key = f"{arm}::{signal_id}"

        log("EXECUTE", "PAPER_CLOSE_STARTED arm=%s symbol=%s key=%s",
            arm, symbol, settlement_key)
        log("EXECUTE", "PAPER_CLOSE_RESULT arm=%s symbol=%s key=%s "
            "realized_usd=%+.8f basis=FILL_VS_FILL trigger=%s",
            arm, symbol, settlement_key, realized,
            getattr(trade, "exit_trigger", "EXIT"))

        if settlement_key in self._arm_settlement_keys:
            # A CLOSE EVENT, NOT A CLOSE. The settlement is already durable;
            # appending again is the duplicate ledger row this guards against.
            log("EXECUTE", "PAPER_CLOSE_DUPLICATE_IGNORED arm=%s symbol=%s "
                "key=%s already_persisted=yes", arm, symbol, settlement_key)
            self.journal.event(
                "paper_close_duplicate_ignored", TradeMode.PAPER,
                arm=arm, symbol=symbol, settlement_key=settlement_key)
            return False

        # THE SETTLEMENT IS RECORDED WITH ITS PRICE PROVENANCE.
        #
        # An arm's realized PnL comes from two EXECUTABLE FILLS, so the record
        # says so on every leg rather than leaving a reader to infer it from a
        # mark-to-mark comparison. It is written as ``arm_exit``, not ``exit``:
        # the two accounting bases must never be aggregated into one total.
        try:
            as_dict = getattr(trade, "as_dict", None)
            payload = as_dict() if callable(as_dict) else dict(trade)
        except Exception:  # noqa: BLE001 - telemetry never blocks a settlement
            payload = {}
        log("EXECUTE", "PAPER_CLOSE_PERSIST_ATTEMPTED arm=%s symbol=%s "
            "key=%s", arm, symbol, settlement_key)
        try:
            self.ledger.arm_exit({
                "arm": arm,
                "settlement_key": settlement_key,
                "signal_id": signal_id or None,
                "mint": getattr(position, "mint", None),
                "symbol": symbol,
                "realized_pnl_usd": round(realized, 8),
                "pnl_basis": "FILL_VS_FILL",
                "price_provenance": {
                    "entry": "PAPER_FILL",
                    "exit": "PAPER_FILL",
                    "note": ("both legs are simulated fills derived from an "
                             "EXECUTABLE_QUOTE; neither is ONCHAIN_FILL"),
                },
                "trigger": getattr(trade, "exit_trigger", None),
                "trade": payload,
            })
        except Exception as exc:  # noqa: BLE001
            # The trade DID close, but the closed record is not durable. Say
            # so plainly instead of letting a PAPER CLOSED receipt imply a
            # ledger the operator cannot read back.
            log("ERROR", "PAPER_CLOSE_PERSIST_FAILED arm=%s symbol=%s key=%s "
                "error=%s: %s", arm, symbol, settlement_key,
                type(exc).__name__, exc)
            self.journal.event(
                "paper_close_persist_failed", TradeMode.PAPER, arm=arm,
                symbol=symbol, settlement_key=settlement_key,
                error=f"{type(exc).__name__}: {exc}")
            self._notify(
                f"{_ERR} <b>PAPER CLOSED - LEDGER WRITE FAILED</b>\n"
                f"<b>${_esc(symbol)}</b>\n\n"
                f"The position closed, but its closed-trade record was NOT "
                f"persisted, so it will not appear in CLOSED POSITIONS and a "
                f"restart cannot recover it.\n\n"
                f"Strategy: <b>{_esc(arm)}</b>\n"
                f"Realized PAPER PnL: "
                f"{self._pnl_usd_text(realized)}\n"
                f"Key: <code>{_esc(settlement_key)}</code>\n"
                f"Cause: <code>{_esc(f'{type(exc).__name__}: {exc}'[:160])}"
                f"</code>\n\n"
                f"<i>PAPER ONLY \U0001f512</i>\n"
                f"<i>NO TRANSACTION SENT</i>",
                None,
            )
            return False

        self._arm_settlement_keys.add(settlement_key)
        log("EXECUTE", "PAPER_CLOSE_PERSISTED arm=%s symbol=%s key=%s "
            "ledger=%s at=%s", arm, symbol, settlement_key,
            self.ledger.ledger_path.name, self.clock.now())
        self.ledger.note(
            "execution-aware paper settlement",
            arm=arm, symbol=symbol,
            mint=getattr(position, "mint", None),
            settlement_key=settlement_key,
            trigger=getattr(trade, "exit_trigger", None),
            realized_pnl_usd=round(realized, 6),
            bankroll_after=round(float(
                getattr(pnl, "bankroll_after", 0.0) or 0.0), 6),
            note=("simulated settlement between two executable fills; "
                  "the arms keep their own bankrolls"),
        )
        log("EXECUTE", "PAPER arm %s closed %s (%s) realized %+.6f",
            arm, symbol,
            getattr(trade, "exit_trigger", "EXIT"), realized)
        return True

    @staticmethod
    def _pnl_usd_text(value: float | None) -> str:
        """A PnL figure for a notification body, signed and labelled."""
        if value is None:
            return "n/a"
        return f"{'+' if float(value) >= 0 else '-'}${abs(float(value)):.4f}"

    def _retain_arm_exit_alerts(self, durable) -> None:
        """
        Drop the queued close receipts of settlements that are NOT durable.

        The bridge renders its alerts from the same trade list the ledger
        booking walks, so this is where the two are reconciled: a settlement
        that committed keeps its ``PAPER CLOSED`` receipt, and one that did
        not has already been reported as a failure and must not now also be
        announced as a clean close.

        MATCHED ON ``(arm, signal_id)``, NOT ON ``signal_id``. One alert opens a
        position on every arm, so all of them share one ``signal_id`` -- and a
        signal_id-only match would let one arm's failed write preserve that
        arm's own clean receipt, which is the exact false receipt this guards
        against. The pair is what identifies a settlement.
        """
        bridge = self._paper_bridge
        if bridge is None:
            return
        allowed = set()
        for t in (durable or ()):
            position = getattr(t, "position", None)
            allowed.add((
                getattr(position, "arm", None),
                getattr(position, "signal_id", None),
            ))
        pending = getattr(bridge, "pending_messages", None)
        if not pending:
            return
        bridge.pending_messages = [
            m for m in pending
            if m.get("kind") != "exit"
            or (m.get("arm"), m.get("signal_id")) in allowed
        ]

    def _arm_closed_rows(self) -> list[dict[str, Any]]:
        """
        Execution-aware PAPER settlements as CLOSED POSITIONS rows.

        Read from the ledger's own ``arm_exit`` records -- the same bytes the
        close receipt is a receipt FOR -- so this view cannot report a closed
        trade the ledger does not hold, and cannot miss one it does.

        ``pnl_basis`` is carried on every row because these are FILL_VS_FILL
        settlements on the arms' own bankrolls. They appear here so the
        operator can see them; they are deliberately NOT summed into the
        legacy marked-to-market statistics, which is why this is a separate
        reader and not a change to ``ledger.trades()``.
        """
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rec in self.ledger.arm_exits():
            trade = rec.get("trade")
            # Same shape check ``Ledger.trades`` applies: a torn or hand-
            # written record is not a closed trade, and must not be rendered
            # as one just because its event name is right.
            if not isinstance(trade, dict):
                continue
            if str(trade.get("failure_reason") or "NONE") != "NONE":
                continue
            position = trade.get("position") or {}
            realized = trade.get("realized_pnl") or {}
            arm = position.get("arm") or rec.get("arm") or "?"
            signal_id = position.get("signal_id") or rec.get("signal_id") or ""
            key = rec.get("settlement_key") or (f"{arm}::{signal_id}"
                                                if signal_id else None)
            # A settlement is one position ending once. If a duplicate row ever
            # reached the ledger, the FIRST one wins, because a settlement
            # cannot be revised -- so there is no later truth to prefer.
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            entry_fill = position.get("entry_fill") or {}
            exit_fill = trade.get("simulated_sell_fill") or {}
            exit_at = rec.get("at")
            excursion = _excursion_fields(trade, position, rec)
            rows.append({
                "source": "ledger",
                "mode": TradeMode.PAPER.value,
                "basis": "execution_aware_paper",
                "pnl_basis": rec.get("pnl_basis") or "FILL_VS_FILL",
                "trade_id": key or f"{arm}:{rec.get('mint') or '?'}",
                "position_id": key,
                "arm": arm,
                "signal_id": signal_id or None,
                "symbol": rec.get("symbol") or position.get("symbol") or "?",
                "mint": rec.get("mint") or position.get("mint"),
                "ca": rec.get("mint") or position.get("mint"),
                "entry_at": position.get("opened_at"),
                "entry_at_iso": iso(position["opened_at"]) if position.get(
                    "opened_at") else None,
                "entry_price": entry_fill.get("effective_price_usd"),
                "entry_price_kind": "SIMULATED_FILL",
                "exit_at": exit_at,
                "exit_at_iso": iso(exit_at) if exit_at else None,
                "exit_price": (exit_fill.get("effective_price_usd")
                               or trade.get("sell_quote_price")),
                "exit_price_kind": "SIMULATED_FILL",
                "observed_exit_mark": trade.get("observed_exit_mark"),
                "realized_pnl_usd": (rec.get("realized_pnl_usd")
                                     if rec.get("realized_pnl_usd") is not None
                                     else realized.get("realized_pnl_usd")),
                "return_pct": realized.get("return_pct"),
                "hold_s": (realized.get("hold_seconds")
                           if realized.get("hold_seconds") is not None
                           else trade.get("hold_time_seconds")),
                "exit_reason": trade.get("exit_reason"),
                "exit_reason_code": trade.get("exit_reason_code"),
                "exit_style": trade.get("exit_trigger"),
                # The mark excursion, under its own name. The fills above are
                # what the PnL came from; these are diagnosis only.
                "mfe_pct": position.get("OBSERVED_MARK_MFE"),
                "mae_pct": position.get("OBSERVED_MARK_mae"),
                "mfe_mae_basis": "OBSERVED_MARK",
                "route": (trade.get("sell_quote_source")
                          or exit_fill.get("venue")),
                "strategy": trade.get("strategy_profile") or arm,
                "price_provenance": rec.get("price_provenance"),
                # -- THE SAMPLED QUOTE EXCURSION ----------------------------
                # Read through ``_excursion_fields``, which knows that the
                # trade dict persists this block UPPERCASE and the position
                # dict persists the same fields LOWERCASE. ``None`` stays
                # ``None``: an unmeasured excursion is a fact about the
                # evidence, and turning it into 0.0% would report an
                # unmeasured position as one that never moved.
                "quote_mfe_pct": excursion["quote_mfe_pct"],
                "quote_mae_pct": excursion["quote_mae_pct"],
                "quote_mfe_mae_basis": excursion["quote_mfe_mae_basis"],
                "quote_unavailable_reason": excursion[
                    "quote_unavailable_reason"],
                "quote_mfe_elapsed_ms": excursion["quote_mfe_elapsed_ms"],
                "quote_mae_elapsed_ms": excursion["quote_mae_elapsed_ms"],
                "quote_coverage_pct": excursion["quote_coverage_pct"],
                "quote_sampling_interval_s": excursion[
                    "quote_sampling_interval_s"],
                "quote_max_sampling_gap_ms": excursion[
                    "quote_max_sampling_gap_ms"],
                "quote_mfe_source": excursion["quote_mfe_source"],
                "quote_mae_source": excursion["quote_mae_source"],
                "quote_excursion_cost_basis": excursion[
                    "quote_excursion_cost_basis"],
                "quote_observation_count": excursion[
                    "quote_observation_count"],
                "quote_rejection_count": excursion["quote_rejection_count"],
                "quote_last_rejection_reason": excursion[
                    "quote_last_rejection_reason"],
                "quote_settlement_sample": excursion["quote_settlement_sample"],
                "quote_settlement_identity": excursion[
                    "quote_settlement_identity"],
                "quote_settlement_source": excursion["quote_settlement_source"],
                "quote_settlement_age_ms": excursion[
                    "quote_settlement_age_ms"],
                "quote_settlement_status": excursion["quote_settlement_status"],
                "quote_telemetry_state": excursion["quote_telemetry_state"],
                "mark_age_ms": trade.get("mark_age_ms"),
                "entry_venue_id": position.get("entry_venue_id"),
                "entry_pool_address": position.get("entry_pool_address"),
                "entry_quote_source": position.get("entry_quote_source"),
                "sell_quote_source": trade.get("sell_quote_source"),
                "entry_price_provenance": "SIMULATED_FILL_FROM_EXECUTABLE_QUOTE",
                "exit_price_provenance": "SIMULATED_FILL_FROM_EXECUTABLE_QUOTE",
                "sell_quote_attempts": trade.get("sell_quote_attempts"),
                "strategy_profile": trade.get("strategy_profile"),
            })
        return rows

    def _beta_closed_rows(self) -> list[dict[str, Any]]:
        """
        V2 FLOOR (BETA) completed trades as CLOSED POSITIONS rows.

        THE AUTHORITATIVE SOURCE, READ AND NOT RE-SETTLED. A beta settlement is
        owned and persisted by ``BetaExperiment`` in its own durable store
        (``v2_floor_beta_state.json``); it is deliberately NOT written to the
        generic ``arm_exit`` ledger, because the experiment must be the ONE
        component that settles each beta arm. Reading that store here surfaces a
        completed beta trade to the operator WITHOUT a second ledger and
        WITHOUT a second settlement: this method only READS records the
        experiment has already booked as ``closed``.

        Only COMPLETED settlements appear. An OPEN or UNRESOLVED beta arm has no
        realized P&L and must never be shown as a closed position, so a record
        without a settled ``net_pnl_usd`` is skipped -- the same rule the ledger
        reader applies to a torn row, and the reason a TP "touch" cannot appear
        here as a profitable trade.

        Every row carries the fields the other closed-position rows carry, plus
        the beta identity (parent signal id and arm) so a beta trade is never
        mistaken for an L25 one, and the exact acquired quantity and both legs'
        costs so the return can be recomputed from the receipts. ``pnl_basis``
        and the per-side price provenance say whether the fills came from an
        executable quote or the legacy modelled mark path.
        """
        bridge = self._paper_bridge
        experiment = (getattr(bridge, "beta_experiment", None)
                      if bridge is not None else None)
        if experiment is None:
            return []
        from .paper_live.v2_floor_beta import (
            BETA_ENTRY_PROFILE,
            BETA_PROFILE_LABEL,
            EXECUTABLE_QUOTE_BASIS,
            record_pnl_basis,
        )

        def _provenance(rec: dict[str, Any], leg: str) -> str:
            basis = str(rec.get(f"{leg}_basis") or "")
            if basis == EXECUTABLE_QUOTE_BASIS:
                return "SIMULATED_FILL_FROM_EXECUTABLE_QUOTE"
            if basis:
                return basis
            # A record written before quote provenance existed was booked off
            # the observed MARK. Say so rather than leaving it unattributed.
            return "OBSERVED_MARK_LEGACY"

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rec in list(getattr(experiment, "completed", []) or []):
            if not isinstance(rec, dict):
                continue
            if str(rec.get("status") or "") != "closed":
                continue
            pnl = rec.get("net_pnl_usd")
            if pnl is None:
                # No realized P&L => not a completed settlement.
                continue
            arm = rec.get("arm") or "?"
            pid = rec.get("parent_signal_id") or rec.get("signal_id") or ""
            key = f"v2beta::{arm}::{pid}"
            if key in seen:
                continue
            seen.add(key)
            entry_at = rec.get("entry_at")
            exit_at = rec.get("exit_at")
            hold_s: float | None = None
            if entry_at is not None and exit_at is not None:
                try:
                    hold_s = max(0.0, float(exit_at) - float(entry_at))
                except (TypeError, ValueError):
                    hold_s = None
            entry_basis = str(rec.get("entry_basis") or "")
            exit_basis = str(rec.get("exit_basis") or "")
            # ONE derivation, shared with the experiment's own reporting, so a
            # row here and a row in Status cannot disagree about a trade's
            # accounting basis (quote-based vs historical modelled).
            pnl_basis = record_pnl_basis(rec)
            excursion = _beta_excursion_fields(rec)
            rows.append({
                "source": "v2_floor_beta",
                "mode": TradeMode.PAPER.value,
                "basis": "v2_floor_beta",
                "pnl_basis": pnl_basis,
                "trade_id": key,
                "position_id": f"{arm}::{pid}",
                "symbol": rec.get("symbol") or "?",
                "mint": rec.get("mint"),
                "ca": rec.get("mint"),
                "arm": arm,
                "beta_arm": arm,
                "parent_signal_id": pid,
                "strategy": BETA_ENTRY_PROFILE,
                "strategy_profile": BETA_PROFILE_LABEL,
                "entry_at": entry_at,
                "entry_at_iso": iso(entry_at) if entry_at else None,
                "entry_price": rec.get("entry_fill_price"),
                "entry_fill_price": rec.get("entry_fill_price"),
                "entry_price_kind": "SIMULATED_FILL",
                "exit_at": exit_at,
                "exit_at_iso": iso(exit_at) if exit_at else None,
                "exit_price": rec.get("exit_fill_price"),
                "exit_fill_price": rec.get("exit_fill_price"),
                "exit_price_kind": "SIMULATED_FILL",
                "realized_pnl_usd": float(pnl),
                # -- THE SIZING DECISION -----------------------------------
                # Which rule sized this trade, what it allocated, what it
                # actually committed and what was left over. Without these a
                # historical 10%-sized trade and an all-in one look identical.
                "sizing_rule": rec.get("sizing_rule"),
                "sizing_pct": rec.get("sizing_pct"),
                "allocated_usd": rec.get("allocated_usd"),
                "entry_cost_usd": (rec.get("entry_cost_usd")
                                   if rec.get("entry_cost_usd") is not None
                                   else rec.get("position_size_usd")),
                "remaining_cash_usd": rec.get("remaining_cash_usd"),
                "exit_proceeds_usd": rec.get("exit_proceeds_usd"),
                "return_pct": rec.get("net_return_pct"),
                "gross_return_pct": rec.get("gross_return_pct"),
                "hold_s": hold_s,
                "exit_reason": rec.get("exit_reason"),
                "exit_reason_code": rec.get("exit_reason"),
                "exit_style": rec.get("exit_reason"),
                # Quantity and cost, so the return can be recomputed from the
                # receipts and the exact acquired amount is visible.
                "entry_tokens_raw": rec.get("entry_tokens_raw"),
                "exit_tokens_raw": rec.get("exit_tokens_raw"),
                "entry_cost_quote": rec.get("entry_cost_quote"),
                "exit_cost_quote": rec.get("exit_cost_quote"),
                "entry_fee_quote": rec.get("entry_fee_quote"),
                "exit_fee_quote": rec.get("exit_fee_quote"),
                "entry_price_provenance": _provenance(rec, "entry"),
                "exit_price_provenance": _provenance(rec, "exit"),
                "fill_model": rec.get("fill_model"),
                "path_ambiguous": rec.get("path_ambiguous"),
                "exit_attempts": rec.get("exit_attempts"),
                "entry_venue": rec.get("entry_venue"),
                "entry_pool_or_curve": rec.get("entry_pool_or_curve"),
                "entry_provider": rec.get("entry_provider"),
                "entry_route": rec.get("entry_route"),
                "entry_identity": rec.get("entry_identity"),
                "exit_identity": rec.get("exit_identity"),
                "entry_basis": entry_basis or None,
                "exit_basis": exit_basis or None,
                "take_profit_pct": rec.get("take_profit_pct"),
                "stop_loss_pct": rec.get("stop_loss_pct"),
                "max_hold_s": rec.get("max_hold_s"),
                # No mark excursion is recorded for a beta trade. ``None`` is
                # NOT 0.0% -- an unmeasured excursion is a fact about the
                # evidence, and the renderer labels a missing one "N/A".
                "mfe_pct": None,
                "mae_pct": None,
                "mfe_mae_basis": "NOT_MEASURED",
                "route": (rec.get("entry_route")
                          or rec.get("entry_pool_or_curve")),
                # A beta trade has no sampled quote excursion, and saying so
                # precisely is the whole point of this block. A
                # ``V2_FLOOR_BETA_QUOTE_SIMULATED_FILL`` record DID accept a
                # same-market SELL quote at settlement (``exit_identity``
                # PASS) -- so reporting it as "no valid same-market SELL quote
                # was obtained" would be false -- while a
                # ``V2_FLOOR_BETA_MODELED_BACKTEST`` one never had an
                # executable quote at all. The store persists no intratrade
                # observation either way, and this names which of those two
                # situations the record is in instead of guessing.
                **excursion,
            })
        return rows

    def _execution_aware_open_rows(self, now: float) -> list[dict[str, Any]]:
        """
        OPEN POSITIONS rows for the execution-aware PAPER arms.

        WHILE EXECUTION-AWARE PAPER IS THE ACTIVE BOOK, the arms ARE the PAPER
        book. Their positions live in ``VirtualAccount.position`` -- one per
        arm, deliberately, so an arm cannot move another arm's bankroll -- and
        this is what puts them in front of the operator.

        The previous behaviour read only the legacy ``PaperAccount``, so every
        notified fill rendered as an empty OPEN POSITIONS with full cash and no
        exposure: a receipt for a trade the operator could not see anywhere.

        The fill price shown is the SIMULATED FILL's executable price and the
        mark is labelled separately, because a reader must never be able to take
        the mark for what was paid.
        """
        bridge = self._paper_bridge
        if bridge is None:
            return []
        rows: list[dict[str, Any]] = []
        for arm, account in getattr(bridge, "accounts", {}).items():
            position = getattr(account, "position", None)
            if position is None:
                continue
            fill = getattr(position, "entry_fill", None)
            fill_price = float(getattr(fill, "effective_price_usd", 0.0) or 0.0)
            # THE CURRENT MARK, not the entry-time mark. Previously the entry
            # mark was reused as the "current" one, which forced every open
            # position's PnL to exactly 0.00 -- the entry mark divided by
            # itself.
            live_mark = None
            observed = getattr(bridge, "observed_marks", None) or {}
            point = observed.get(getattr(position, "mint", None))
            if point is not None and getattr(point, "value_usd", None):
                live_mark = point.value_usd
            if live_mark is None:
                live_mark = getattr(position, "current_mark_usd", None)
            mark = live_mark
            mark_known = mark is not None and float(mark) > 0
            notional = float(getattr(position, "usd_notional", 0.0) or 0.0)
            # The token count is DERIVED from the two fields the position
            # actually holds -- the exact quoted raw output and the mint's
            # decimals. It is not read from a field that does not exist, which
            # is how this rendered as a fabricated "Qty: 0".
            decimals = getattr(position, "token_decimals", None)
            raw = getattr(position, "tokens_raw", None)
            tokens = (
                float(raw) / (10 ** int(decimals))
                if raw is not None and decimals is not None else None
            )
            opened = float(getattr(position, "opened_at", now) or now)
            rows.append({
                "source": "execution_aware_paper",
                "mode": TradeMode.PAPER.value,
                "arm": arm,
                "bankroll_usd": round(float(
                    getattr(account, "bankroll_usd", 0.0) or 0.0), 6),
                "position_id": (
                    f"paperlive_{getattr(position, 'signal_id', '?')}_{arm}"),
                "signal_id": getattr(position, "signal_id", None),
                "symbol": getattr(position, "symbol", None) or "?",
                "mint": getattr(position, "mint", None),
                "ca": getattr(position, "mint", None),
                "entry_at": opened,
                "entry_at_iso": iso(opened) if opened else None,
                # THE FILL, not the mark.
                "entry_price": fill_price or None,
                "fill_price": fill_price or None,
                "fill_price_kind": getattr(
                    fill, "kind", "SIMULATED_FILL_FROM_EXECUTABLE_QUOTE"),
                # The mark, kept distinct and named as a mark.
                "current_price": mark if mark_known else None,
                "observed_mark": mark if mark_known else None,
                "size_usd": notional,
                "notional_usd": notional,
                "quantity": tokens,
                "quantity_raw": raw,
                "token_decimals": decimals,
                "pnl_usd": self._arm_unrealized_usd(position, mark),
                "pnl_pct": self._arm_return_pct(position, mark),
                # The three canonical buckets, stated separately so an open
                # book can never render as "PnL $0" while it is actually down.
                "realized_pnl_usd": 0.0,
                "unrealized_pnl_usd": (
                    self._arm_unrealized_usd(position, mark)
                    if mark_known else None),
                "unrealized_pnl_pct": (
                    self._arm_return_pct(position, mark) if mark_known else None),
                "total_pnl_usd": (
                    self._arm_unrealized_usd(position, mark)
                    if mark_known else None),
                "status": getattr(position, "exit_status", None) or "OPEN_MONITORING",
                "exit_status": getattr(position, "exit_status", None) or "OPEN_MONITORING",
                "exit_attempts": int(getattr(position, "exit_attempts", 0) or 0),
                "last_exit_error": getattr(position, "last_exit_error", "") or "",
                "mark_available": mark_known,
                # Total equity for this arm: cash + marked value of the open
                # position. Never the position counted as realized cash, and
                # never dropped from the book just because it is unsellable.
                "open_mark_value_usd": (
                    round(notional * (float(mark) / fill_price), 6)
                    if (mark_known and fill_price > 0) else None),
                "total_equity_usd": (
                    round(float(getattr(account, "bankroll_usd", 0.0) or 0.0)
                          + self._arm_unrealized_usd(position, mark), 6)
                    if mark_known else None),
                "hold_s": max(0.0, now - opened),
                "mfe_pct": float(
                    getattr(position, "observed_mark_mfe", 0.0) or 0.0),
                "mae_pct": float(
                    getattr(position, "observed_mark_mae", 0.0) or 0.0),
                "peak_return_pct": None,
                "data_state": "LIVE" if mark_known else "MISSING",
                "exit_state": "OPEN",
                "exit_style": "FAST_FAILURE",
                "strategy_profile": getattr(
                    position, "production_profile", None),
                "venue": getattr(fill, "venue", None),
                "pool": getattr(fill, "pool_or_curve", None),
                "venue_id": getattr(position, "entry_venue_id", None),
                "quote_provider": getattr(fill, "provider", None),
                "quote_age_ms": getattr(fill, "quote_age_ms", None),
                "tp_sl": None,
                "thesis": None,
                "invalidation": [],
                "exit_monitor": f"execution-aware PAPER arm {arm}",
                "exit_hold_count": 0,
                "liquidity_usd": None,
                "market_cap_usd": None,
            })
        return rows

    @staticmethod
    def _arm_unrealized_usd(position, mark) -> float:
        """
        Unrealized PnL on the OBSERVED MARK.

        Stated as unrealized and mark-based on purpose: the arms' realized PnL
        comes from two executable fills, and an open position has only the mark
        until it sells. Reporting a mark excursion here does not make it a
        settlement, because the settlement path never reads this number.
        """
        notional = float(getattr(position, "usd_notional", 0.0) or 0.0)
        entry_price = float(getattr(
            getattr(position, "entry_fill", None), "effective_price_usd",
            0.0) or 0.0)
        if not mark or not entry_price:
            return 0.0
        return notional * (float(mark) / entry_price - 1.0)

    @staticmethod
    def _arm_return_pct(position, mark) -> float:
        entry_price = float(getattr(
            getattr(position, "entry_fill", None), "effective_price_usd",
            0.0) or 0.0)
        if not mark or not entry_price:
            return 0.0
        return (float(mark) / entry_price - 1.0) * 100.0

    def open_positions_view(self) -> list[dict[str, Any]]:
        """
        Currently open positions, for the Telegram OPEN POSITIONS button.

        THE AUTHORITATIVE SOURCE. Every field is read from the same
        :class:`~autopaper.account.PaperAccount` (and the same
        :class:`~autopaper.real_account.RealAccount`) the accounting layer reads,
        so this cannot disagree with ``/balance``, ``/status`` or the dashboard -
        there is no Telegram-side copy and nothing to fall out of sync.

        It now also reads the execution-aware PAPER arms. Those arms ARE the
        active PAPER book (the legacy mark-based book only runs behind the
        explicit rollback flag), so reading only the legacy account left every
        notified fill invisible. Each arm's position is one row, labelled with
        the arm it belongs to, because the arms are deliberately independent
        accounts and a reader must be able to tell them apart.

        Marked from the engine's own ``self._marks`` rather than refetching, so
        pressing the button costs no network call and can never sit on the
        trading path. Unrealized PnL is the mark the equity identity uses.

        On demand only: nothing here pushes, and there is no poller behind it.
        """
        now = self.clock.now()
        marks = self._marks or self._refresh_marks()
        rows: list[dict[str, Any]] = []
        for p in self.account.open_positions():
            mark = marks.get(p.position_id)
            plan = dict(p.exit_plan or {})
            entry = p.entry_reference_price or p.fill_price
            rows.append({
                "source": "paper_account",
                "mode": TradeMode.PAPER.value,
                "position_id": p.position_id,
                "symbol": p.symbol or "?",
                "mint": p.mint,
                "ca": p.mint,
                "entry_at": p.entry_timestamp,
                "entry_at_iso": iso(p.entry_timestamp) if p.entry_timestamp else None,
                "entry_price": entry,
                "current_price": p.current_price,
                "size_usd": p.cash_out_usd,
                "quantity": p.quantity,
                "notional_usd": p.notional_usd,
                "pnl_usd": (mark.unrealized if mark else p.unrealized_pnl_usd),
                "pnl_pct": p.current_return_pct,
                "hold_s": max(0.0, now - (p.entry_timestamp or now)),
                "mfe_pct": p.mfe_pct,
                "mae_pct": p.mae_pct,
                "peak_return_pct": p.peak_return_pct,
                "data_state": p.data_state.value,
                "exit_state": "OPEN",
                "exit_style": p.exit_style.value,
                # The profile THIS POSITION was opened under, not the currently
                # selected one. They differ after a mid-flight switch, and the
                # operator needs the one that governs this exit.
                "strategy_profile": p.strategy_profile,
                "tp_sl": _tp_sl_text(plan, entry, p.runner_mode, p.runner_armed_at),
                "thesis": p.thesis,
                "invalidation": list(p.invalidation),
                "exit_monitor": (
                    f"holding {p.exit_hold_count}x; style={p.exit_style.value}"
                    + ("; runner armed" if p.runner_mode else "")
                ),
                "exit_hold_count": p.exit_hold_count,
                "liquidity_usd": p.liquidity_at_entry,
                "market_cap_usd": p.market_cap_at_entry,
            })
        for p in self.real_account.open_positions():
            rows.append({
                "source": "real_account",
                "mode": TradeMode.REAL.value,
                "position_id": p.position_id,
                "symbol": p.symbol,
                "mint": p.mint,
                "ca": p.mint,
                "entry_at": p.entry_at,
                "entry_price": p.entry_price_usd,
                "current_price": p.current_price_usd,
                "size_usd": p.size_usd,
                "quantity": p.token_amount,
                "pnl_usd": p.unrealized_usd(),
                "pnl_pct": p.return_pct(),
                "hold_s": p.holding_seconds(now),
                "mfe_pct": p.mfe_pct,
                "mae_pct": p.mae_pct,
                "exit_state": p.status,
                "signature": p.entry_signature,
                "confirmed": p.confirmed,
                "strategy_profile": p.strategy_profile,
                "exit_monitor": f"on-chain; status={p.status}",
            })
        # The execution-aware PAPER arms, appended rather than merged. PAPER
        # rows first, then REAL, so a reader sees the simulated book and the
        # on-chain book as the two things they are.
        rows.extend(self._execution_aware_open_rows(now))
        return rows

    def closed_positions_view(self, offset: int = 0, *,
                              limit: int = CLOSED_PAGE_SIZE
                              ) -> tuple[list[dict[str, Any]], int]:
        """
        Recently closed positions, newest first, with a total count.

        THE AUTHORITATIVE SOURCE, and deliberately the same one
        :meth:`statistics_by_mode` and ``self.memory`` use: the ledger.
        Closed trades are not reconstructed from any Telegram-side list
        and not from a second position store, because a second store is
        exactly how two views of "closed" come to disagree.

        It reads BOTH of the ledger's closed-trade record types:

        * ``exit``       -- the legacy mark-to-market book (:meth:`Ledger.trades`)
        * ``arm_exit``   -- the execution-aware arms' FILL_VS_FILL settlements
          (:meth:`Ledger.arm_exits`)

        It used to read only the first. Since 2026-10-05 the arms have been the
        ACTIVE PAPER book, so every settlement they booked was durably recorded
        and invisible here: the ledger an operator was reading stopped on
        2026-10-02 while the receipts kept arriving. One file, one query, no
        cache -- so a close is visible on the next press of the button.

        The two record types stay SEPARATE ROWS carrying their own
        ``pnl_basis``. Merging them into one total is what the ``ARM_EXIT``
        event type exists to prevent, and ``statistics_by_mode`` is unchanged.

        Paginated rather than paged by button-spam: one reply carries a fixed
        recent window plus the totals, so the operator gets the answer without the
        chat receiving one message per price movement.
        """
        from .engine import _trade_from

        rows: list[dict[str, Any]] = []
        for raw in self.ledger.trades():
            tr = _trade_from(raw)
            if tr is None:
                continue
            rows.append({
                "source": "ledger",
                "mode": TradeMode.PAPER.value,
                "trade_id": tr.trade_id,
                "position_id": tr.position_id,
                "symbol": tr.symbol or "?",
                "mint": tr.mint,
                "ca": tr.mint,
                "entry_at": tr.entry_timestamp,
                "entry_at_iso": iso(tr.entry_timestamp) if tr.entry_timestamp else None,
                "entry_price": tr.entry_price,
                "exit_at": tr.exit_timestamp,
                "exit_at_iso": iso(tr.exit_timestamp) if tr.exit_timestamp else None,
                "exit_price": tr.exit_price,
                "realized_pnl_usd": tr.net_pnl_usd,
                "return_pct": tr.return_pct,
                "hold_s": tr.holding_seconds,
                "exit_reason": tr.exit_reason,
                "exit_style": tr.exit_style,
                "mfe_pct": tr.mfe_pct,
                "mae_pct": tr.mae_pct,
                "route": tr.route,
                "strategy": tr.strategy,
                # The profile the position was ACTUALLY OPENED under, not the
                # profile currently selected in the UI. A historical trade is
                # never relabelled to match what the operator happens to have
                # selected now.
                "strategy_profile": tr.strategy_profile,
                "entry_fill_price": tr.fill_price,
                # Price provenance, per side. A reader must never have to guess
                # whether a closed position's prices were marks, venue quotes or
                # executions.
                "entry_price_provenance": tr.entry_price_provenance,
                "exit_price_provenance": tr.exit_price_provenance,
                "pnl_basis": tr.pnl_basis,
                "mfe_basis": "OBSERVED_MARK",
                "thesis": tr.thesis,
            })
        # Execution-aware PAPER settlements. Appended, not merged: the arm rows
        # are their own accounting basis and are labelled with it.
        rows.extend(self._arm_closed_rows())
        # V2 FLOOR (BETA) completed settlements, read from the experiment's own
        # authoritative state. Also appended, never merged, and never summed
        # into the legacy totals -- the experiment keeps its own three ledgers.
        rows.extend(self._beta_closed_rows())
        # REAL closed positions. Same concepts as PAPER (entry, exit, quantity,
        # price, fees, PnL, hold time, MFE/MAE, status) but an explicit scope, so
        # a paper historical total and a current-session REAL counter can never
        # be read as one population.
        for p in self.real_account.positions.values():
            if p.is_open:
                continue
            rows.append({
                "source": "real_account",
                "mode": TradeMode.REAL.value,
                "trade_id": p.entry_signature or p.mint,
                "position_id": p.position_id,
                "symbol": p.symbol or "?",
                "mint": p.mint,
                "ca": p.mint,
                "entry_at": p.entry_at,
                "entry_at_iso": iso(p.entry_at) if p.entry_at else None,
                # The entry reference price, labelled with its kind. It is a
                # mark, and the settlement-derived fill price is beside it.
                "entry_price": p.entry_price_usd,
                "entry_price_kind": p.entry_price_kind,
                "entry_fill_price": p.entry_execution_price_usd,
                "exit_at": p.exit_timestamp,
                "exit_at_iso": iso(p.exit_timestamp) if p.exit_timestamp else None,
                "exit_price": p.exit_price_usd,
                "exit_price_kind": p.exit_price_kind,
                "exit_fill_price": p.exit_execution_price_usd,
                "realized_pnl_usd": p.realized_pnl_usd,
                # THE REALIZED RETURN, derived from settlement on both legs. It
                # used to be computed here as ``exit_mark / entry_mark - 1`` --
                # two DexScreener marks -- and published as "return_pct", which
                # is how the 2026-10-02 $GOAT position was reported as +71.56%
                # when its all-in cash result was -$0.1825.
                "return_pct": p.realized_return_pct,
                # The mark excursion, under its own name. Never "the result".
                "observed_mark_return_pct": p.observed_mark_return_pct,
                "cross_venue_detected": p.cross_venue_detected,
                "hold_s": p.holding_seconds(),
                "exit_reason": p.exit_reason,
                "mfe_pct": p.mfe_pct,
                "mae_pct": p.mae_pct,
                "signature": p.entry_signature,
                "exit_signature": (p.exit.signature if p.exit else None),
                "scope": "REAL (current session)",
            })
        rows.sort(key=lambda r: r.get("exit_at") or 0.0, reverse=True)
        total = len(rows)
        offset = max(0, int(offset))
        window = max(1, int(limit))
        return rows[offset:offset + window], total

    def recent_orders(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        The most recently CLOSED trades, newest first, for operator inspection.

        Reads the ledger, so it is a pure function of recorded history and
        cannot drift from what was actually traded. Read-only: it places no
        order and touches no wallet.
        """
        from .engine import _trade_from

        out: list[dict[str, Any]] = []
        for row in self.ledger.trades():
            tr = _trade_from(row)
            if tr is None:
                continue
            out.append({
                "at": tr.exit_timestamp,
                "mint": getattr(tr, "mint", None),
                "symbol": getattr(tr, "symbol", None),
                "return_pct": tr.return_pct,
                "net_pnl_usd": tr.net_pnl_usd,
                "holding_s": tr.holding_seconds,
                "exit_reason": getattr(tr, "exit_reason", None),
            })
        out.sort(key=lambda r: r.get("at") or 0.0, reverse=True)
        return out[: max(1, int(limit))]

    async def close_all_paper(self) -> dict[str, Any]:
        """
        Emergency close for ALL open PAPER positions.

        PAPER-ONLY: Never touches REAL positions.
        Uses the canonical execution-aware close path (bridge._close_for_arm).
        Legacy positions are reported as uncloseable rather than being closed
        at a mark. A mark is an observation, not an executable SELL quote, and
        treating it as a fill would make the operator kill switch lie about the
        price at which the position was closed.

        Returns a summary dict with closed/failed/remaining counts.
        """
        from .exits import ExitDecision, ExitStyle
        from .paper_live.schema import (
    ARM_CLOSE_NO_FILL_CONTRADICTION,
    ARM_CLOSE_SETTLEMENT_FAILED,
    ObservedMarkPoint,
)

        results: list[dict[str, Any]] = []
        closed = 0
        failed = 0
        remaining = 0
        errors: list[str] = []
        #: Trades this invocation SETTLED, awaiting the durable write below.
        settled: list[Any] = []

        # --- Execution-aware PAPER arms (the active book) ---
        try:
            bridge = await self._paper_bridge_instance()
        except Exception as exc:
            log("ERROR", "cannot build paper bridge for emergency close: %s", exc)
            # Do not return before inspecting the legacy book. A bridge
            # construction failure must not hide positions owned by this
            # process. They are reported below as an explicit refusal rather
            # than being closed at a mark.
            bridge = None
            errors.append(f"bridge unavailable: {type(exc).__name__}: {exc}")

        now = self.clock.now()
        # --- The V2 FLOOR (BETA) experiment book ---------------------------
        #
        # Its positions are MIRRORS in this bridge and are settled by
        # ``BetaExperiment``, not by ``_close_for_arm``. The kill switch still has
        # to reach them: a flatten that left a beta arm open and unaccounted
        # would defeat the point of an emergency stop. So the experiment is
        # flattened explicitly, and any arm it REFUSED to close (no observation
        # -> no price -> no trade) is reported as still open rather than counted
        # as closed.
        experiment = getattr(bridge, "beta_experiment", None)
        if experiment is not None:
            beta_summary = experiment.flatten_all(now=now)
            for rec in beta_summary.get("closed", []):
                closed += 1
                results.append({
                    "position_id": (
                        f"beta_{rec['parent_signal_id']}_{rec['arm']}"),
                    "mint": rec.get("mint", ""),
                    "outcome": "CLOSED",
                    "reason": "EMERGENCY_KILL_ALL",
                    "fill_price": rec.get("exit_fill_price"),
                    "net_pnl_usd": rec.get("net_pnl_usd"),
                })
            for rec in beta_summary.get("unresolved", []):
                failed += 1
                remaining += 1
                errors.append(
                    f"BETA_FLATTEN_NO_OBSERVATION: arm {rec['arm']} of signal "
                    f"{rec['parent_signal_id']} has no observed price and was "
                    "left OPEN")
                results.append({
                    "position_id": (
                        f"beta_{rec['parent_signal_id']}_{rec['arm']}"),
                    "mint": rec.get("mint", ""),
                    "outcome": "ERROR",
                    "reason": ("BETA_FLATTEN_NO_OBSERVATION: no observed "
                               "price; position left OPEN"),
                    "failure_reason": "BETA_FLATTEN_NO_OBSERVATION",
                })
            # The mirrors are now inert bookkeeping; drop them so the
            # open-positions view agrees with the experiment's own book.
            bridge.beta_positions = {}

        arms = getattr(bridge, "arms", ()) if bridge is not None else ()
        for spec in arms:
            account = getattr(bridge, "accounts", {}).get(spec.name.value)
            if account is None:
                continue
            position = getattr(account, "position", None)
            if position is None:
                continue
            arm = spec.name.value
            pid = f"paperlive_{getattr(position, 'signal_id', '?')}_{arm}"
            try:
                # Build a minimal mark from the bridge's observed marks cache.
                observed = getattr(bridge, "observed_marks", {}) or {}
                point = observed.get(getattr(position, "mint", None))
                if point is not None:
                    mark = point
                else:
                    mark = ObservedMarkPoint(
                        value_usd=None, pair_address=None, dex_id=None,
                        liquidity_usd=None, volume_5m_usd=None, market_cap_usd=None,
                        timestamp=now, available=False, missing_reason="emergency close: no observed mark available",
                    )
                decision = ExitDecision(
                    action="EXIT",
                    reason="EMERGENCY_KILL_ALL",
                    style=ExitStyle.FLATTEN,
                    trigger="EMERGENCY_KILL_ALL",
                )
                trade = await bridge._close_for_arm(
                    spec=spec, position=position, mark=mark, decision=decision,
                )
                # DID IT SETTLE?
                #
                # The test is the EXIT FILL, and only the exit fill.
                #
                # ``failure is not None`` was the test, and
                # ``FailureReason.NONE`` is an enum member that is NOT ``None``
                # -- so a successful settlement took the FAILURE branch on every
                # single close. The operator was told ``Failed: 1`` /
                # ``Errors: NONE`` for a position that had in fact been sold,
                # settled and settled exactly once. That is what turned one
                # mis-reported close into an incident where the position appeared
                # to be permanently uncloseable.
                #
                # The fill is the evidence. A trade with a fill settled; a trade
                # without one did not, whatever its reason field says.
                exit_fill = getattr(trade, "exit_fill", None)
                if exit_fill is None:
                    failed += 1
                    remaining += 1
                    reason = _arm_close_failure_reason(trade)
                    errors.append(reason)
                    results.append({
                        "position_id": pid,
                        "mint": getattr(position, "mint", ""),
                        "outcome": "ERROR",
                        "reason": reason,
                        "failure_reason": _arm_close_failure_code(trade),
                    })
                else:
                    closed += 1
                    # COLLECTED, NOT YET DURABLE. The settlement is committed to
                    # the ledger and to the resumable book below, once, after
                    # every arm has been attempted.
                    settled.append(trade)
                    results.append({
                        "position_id": pid,
                        "mint": getattr(position, "mint", ""),
                        "outcome": "CLOSED",
                        "reason": "EMERGENCY_KILL_ALL",
                        "fill_price": float(
                            getattr(exit_fill, "effective_price_usd", 0.0)
                            or 0.0),
                        "net_pnl_usd": (
                            float(trade.pnl.realized_pnl_usd)
                            if getattr(trade, "pnl", None) is not None else None
                        ),
                    })
            except Exception as exc:
                failed += 1
                remaining += 1
                err_msg = f"{type(exc).__name__}: {exc}"
                errors.append(f"EXIT_SETTLEMENT_FAILED: {err_msg}")
                results.append({
                    "position_id": pid,
                    "mint": getattr(position, "mint", ""),
                    "outcome": "ERROR",
                    "reason": err_msg,
                    "failure_reason": "EXIT_SETTLEMENT_FAILED",
                })
                # Mark the position as failed exit so it can be retried but is not silently closed
                position.exit_status = "EXIT_FAILED"
                position.last_exit_error = err_msg
                position.exit_attempts = int(getattr(position, "exit_attempts", 0) or 0) + 1

        # --- Legacy paper account positions (if any survive) ---
        #
        # There is intentionally NO mark-price fallback here. These positions
        # belong to the opt-in historical/rollback book and do not carry the
        # execution-aware arm identity required by the canonical SELL service.
        # Leaving them OPEN with a structured refusal is safer than recording a
        # fake fill. The operator can then migrate/reconcile them explicitly.
        for p in list(self.account.open_positions()):
            pid = p.position_id
            failed += 1
            remaining += 1
            err_msg = (
                "LEGACY_POSITION_REQUIRES_EXECUTION_AWARE_SELL: no canonical "
                "arm identity is available; position left OPEN"
            )
            errors.append(err_msg)
            results.append({
                "position_id": pid,
                "mint": p.mint,
                "outcome": "ERROR",
                "reason": err_msg,
            })

        # --- COMMIT THE SETTLEMENTS ------------------------------------------
        #
        # A settlement that is only in memory is a settlement that does not
        # exist. ``close_all_paper`` used to return straight after
        # ``_close_for_arm``, which books the arm's PnL into the account but
        # writes NOTHING durable: no ``arm_exit`` row on the ledger and no
        # ``paper_state.json`` rewrite. The operator was shown a close, the
        # account was debited in memory, and then the next restart re-read a
        # state file that still listed the position OPEN -- so it came back, kept
        # blocking the L25 slot, and the whole incident looked unfixable from
        # the outside. This is the same three-step the monitor path uses
        # (``_record_arm_trades`` -> ``persist_paper_dataset``), reached from
        # the kill switch for the first time.
        #
        # ``_record_arm_trades`` is idempotent on ``arm::signal_id``, so pressing
        # CLOSE ALL twice books one settlement, not two.
        if settled:
            durable = self._record_arm_trades(settled)
            # ``trade_records`` is what ``persist_state`` reads to decide which
            # persisted positions have closed. A trade closed outside
            # ``monitor_once`` has to be in it, or the resumable book keeps the
            # position and the restart problem returns exactly as it was.
            bridge.trade_records.extend(
                t for t in settled if t not in bridge.trade_records)
            self.persist_paper_dataset()
            self._retain_arm_exit_alerts(durable)
            # A settlement that could NOT be written is not a close. It is
            # reported as one, with its own code, because "the position is gone
            # from memory but no ledger row exists" is a persistence incident and
            # must never be presented to the operator as a completed sale.
            unbooked = [t for t in settled if t not in durable]
            if unbooked:
                closed -= len(unbooked)
                failed += len(unbooked)
                remaining += len(unbooked)
                for trade in unbooked:
                    position = getattr(trade, "position", None)
                    detail = (
                        "the position settled in memory but its closed-trade "
                        "record was NOT persisted; it will not appear in CLOSED "
                        "POSITIONS and a restart can revive it"
                    )
                    errors.append(f"EXIT_SETTLEMENT_FAILED: {detail}")
                    results.append({
                        "position_id": (
                            f"paperlive_{getattr(position, 'signal_id', '?')}_"
                            f"{getattr(position, 'arm', '?')}"),
                        "mint": getattr(position, "mint", ""),
                        "outcome": "ERROR",
                        "reason": detail,
                        "failure_reason": "EXIT_SETTLEMENT_FAILED",
                    })

        # ``remaining`` is RE-READ, not inferred. It was incremented alongside
        # every failure, which is the same accounting as the counts above and
        # drifts the moment a close settles, a booking fails, or an arm holds no
        # position at all. The number the operator is shown is the number of
        # positions the book still holds when this call returns.
        remaining = 0
        if bridge is not None:
            remaining += sum(
                1 for account in getattr(bridge, "accounts", {}).values()
                if getattr(account, "position", None) is not None
            )
        remaining += len(self.account.open_positions())

        # ``failed`` and ``remaining`` describe the same position: a failed
        # close is still open. Counting both here made the kill-switch summary
        # claim more targets than it actually attempted.
        total_targeted = len(results)
        return {
            "success": remaining == 0 and failed == 0,
            "targeted": total_targeted,
            "closed": closed,
            "failed": failed,
            "remaining": remaining,
            "details": results,
            "errors": errors,
        }

    def _sol_usd_reference(self) -> Any:
        """
        The canonical SOL/USD provider for this process.

        ONE provider, ONE cache, ONE reference -- shared by PAPER sizing and
        REAL equity valuation. It is injected (``sol_usd_provider``) and only
        built here, lazily, when production did not supply one. Building it in
        this method rather than in ``__init__`` is what keeps an offline test
        that never asks for a price from constructing an HTTP client at all;
        constructing it opens no socket, because the transport is created on
        first fetch.

        Deliberately NOT async: it performs no I/O, so making it awaitable would
        only invite an un-awaited call.

        The provider is a READ-ONLY price reference. It has no signer, no
        transaction builder and no broadcaster.
        """
        provider = self.sol_usd_provider
        if provider is None:
            from .same_market.sol_price import JupiterSolPriceProvider

            provider = JupiterSolPriceProvider()
            # Assign onto self so the cache is shared by every later consumer
            # rather than being rebuilt per call.
            self.sol_usd_provider = provider
        return provider

    async def _sol_usd_snapshot_or_none(self) -> Any:
        """
        A SOL/USD snapshot that is FRESH, or ``None``.

        Staleness is a refusal, not a warning. The provider's own TTL governs
        its cache; this is the second, independent gate: even a snapshot handed
        back from that cache is rejected here once it is older than
        ``_SOL_USD_MAX_AGE_MS``, because equity is an account balance and sizing
        against an old reference is a silent mis-size. There is deliberately no
        "carry the last value forward" branch -- that would be the guess this
        whole path exists to refuse.
        """
        from .same_market.sol_price import SolPriceUnavailable

        try:
            snap = await self._sol_usd_reference().snapshot()
        except SolPriceUnavailable as exc:
            log("ERROR", "SOL/USD reference unavailable: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "SOL/USD reference failed: %s", type(exc).__name__)
            return None

        try:
            age_ms = float(snap.age_ms())
        except Exception:  # noqa: BLE001 - a snapshot without an age is refused
            return None
        if age_ms > _SOL_USD_MAX_AGE_MS:
            log("ERROR", "SOL/USD reference is stale (%.0fms > %.0fms); "
                "refusing rather than valuing against it",
                age_ms, _SOL_USD_MAX_AGE_MS)
            return None
        return snap

    async def _backend_sol_price(self) -> float | None:
        """
        The SOL/USD reference price, or ``None`` when it cannot be obtained.

        Reads the canonical provider rather than the swap quoter. The value is
        memoised per run so a busy loop does not re-fetch against a shared
        keyless rate ceiling, but it is NOT carried forward indefinitely: the
        snapshot's own timestamp decides whether the cached value is still
        usable, which is what the ``_sol_usd_snapshot`` gate below checks.
        """
        snap = await self._sol_usd_snapshot_or_none()
        if snap is None:
            return None
        try:
            price = float(snap.usd_per_sol)
        except (AttributeError, TypeError, ValueError):
            return None
        if not (price > 0):
            return None
        self._sol_usd_snapshot = snap
        self._sol_price_usd = price
        return price

    def _pretrade_block_reason(self, failures) -> str:
        """
        Name a pre-trade refusal with the execution category it belongs to.

        A pre-trade failure used to be recorded as ``execution_exception``,
        which said nothing about WHY and collapsed a missing Raptor route into
        the same word as an exposure breach. The two families are separated:

        * a venue/quote failure (``quote_valid`` / ``exit_quote_valid``) carries
          the category the provider-chain probe already computed, so the
          operator log reads ``BUY_BLOCKED=route_unavailable``;
        * anything else (exposure, idempotency, armed state) is
          ``pretrade_refused`` -- no execution was ever attempted, so calling it
          an execution exception was wrong.
        """
        names = {getattr(c, "name", "") for c in failures}
        if "quote_valid" in names or "exit_quote_valid" in names:
            probe = getattr(self.real_executor, "last_quote_probe", None) or {}
            category = probe.get("category")
            if category:
                return category
            return "quote_invalid"
        return "pretrade_refused"

    async def _real_pretrade_checks(
        self,
        *,
        mint: str,
        size,
        lamports: int,
        sol_price: Optional[float],
        v3c_authorized: bool,
    ) -> list[Any]:
        """
        The complete pre-trade safety sequence. Every check returns
        ``(ok, detail)`` and ANY failure means DO NOT TRADE.
        """
        from .execution.real import SafetyCheck

        checks: list[Any] = [
            SafetyCheck(
                "real_mode_explicit",
                self.state.trade_mode is TradeMode.REAL,
                "trade mode is REAL",
            ),
            SafetyCheck(
                "no_paper_mark_fill_rollback",
                not bool(getattr(
                    self.config.real, "paper_legacy_mark_fill_rollback", False
                )),
                "legacy PAPER mark-fill rollback is disabled"
                if not bool(getattr(
                    self.config.real, "paper_legacy_mark_fill_rollback", False
                ))
                else "legacy PAPER mark-fill rollback is enabled",
            ),
            SafetyCheck(
                "real_armed_by_operator",
                self.state.real_trading_enabled,
                "armed by a confirmed Telegram operator action"
                if self.state.real_trading_enabled
                else "REAL is not armed",
            ),
            SafetyCheck(
                "no_emergency_stop",
                not self.state.emergency_stop,
                "not latched",
            ),
            SafetyCheck(
                "v3c_authorized", v3c_authorized,
                "V3-C authorized this candidate"
                if v3c_authorized
                else "V3-C did not authorize",
            ),
            SafetyCheck(
                "sizing_matches_policy",
                size.position_pct in (10.0, 12.5)
                and abs(
                    size.position_usd - size.equity_usd * size.position_pct / 100.0
                ) < 1e-12,
                size.rationale,
            ),
            SafetyCheck(
                "no_duplicate_position",
                not (self.real_account.has_mint(mint)
                     or self.account.has_mint(mint)),
                "no open position in this mint",
            ),
            SafetyCheck(
                "no_reentry",
                not self.account.cooldown_active(mint, self.clock.now()),
                "re-entry is disabled for every mint",
            ),
            SafetyCheck(
                "open_positions_within_limit",
                not self.real_account.open_positions()
                and not self.account.open_positions(),
                "max open positions is "
                + str(self.config.risk.max_simultaneous_positions),
            ),
            SafetyCheck(
                "lamports_positive", lamports > 0,
                "{} lamports".format(lamports)
                if lamports > 0
                else "the position size could not be converted to lamports "
                     "(SOL/USD unavailable)",
            ),
            SafetyCheck(
                "no_concurrent_operation",
                not self.real_executor.busy,
                "no real operation in flight",
            ),
        ]

        checks.extend(self.real_executor.configuration_gates())

        # The operator must have acknowledged the real-trading risk. This is a
        # REAL gate, not a UI decoration. A config that never set it refuses.
        acknowledged = bool(getattr(self.config.real, "risk_acknowledged", False))
        checks.append(
            SafetyCheck(
                "risk_acknowledged", acknowledged,
                "the operator acknowledged the real-trading risks"
                if acknowledged
                else "I_UNDERSTAND_REAL_TRADING_RISKS is not set",
            )
        )

        # IDEMPOTENCY. If any REAL execution may be in flight on chain, a new
        # entry is refused. This is what makes a retry after an ambiguous
        # broadcast impossible rather than merely discouraged.
        unresolved = self.execution_intents.unresolved()
        checks.append(
            SafetyCheck(
                "no_unresolved_execution",
                not unresolved,
                "no unresolved real execution"
                if not unresolved
                else (
                    "an unresolved REAL execution may be in flight "
                    f"({len(unresolved)}); refusing to add another"
                ),
            )
        )

        # Startup reconciliation must not have left an unresolved mismatch.
        checks.append(
            SafetyCheck(
                "recovery_consistent", self._real_state_consistent,
                "startup reconciliation found no unresolved mismatch"
                if self._real_state_consistent
                else "startup reconciliation has an unresolved mismatch",
            )
        )

        sol_check = await self.real_executor.sufficient_sol(
            position_lamports=lamports, sol_price_usd=sol_price
        )
        checks.append(sol_check)

        exposure_ok, exposure_detail = self._exposure_check(size)
        checks.append(
            SafetyCheck("exposure_within_limit", exposure_ok, exposure_detail)
        )

        if sol_check.ok and lamports > 0:
            self._real_quote_at = self.clock.now()
            quote_check, _ = await self.real_executor.quote_probe(
                mint=mint,
                position_lamports=lamports,
                slippage_bps=self.config.real.slippage_bps,
                # Required, explicit: WSOL -> token, WSOL lamports.
                side="ENTRY",
            )
            checks.append(quote_check)
            quote_age_ms = (
                (self.clock.now() - self._real_quote_at) * 1000.0
            )
            max_age = float(self.config.real.quote_max_age_ms)
            checks.append(
                SafetyCheck(
                    "quote_freshness", quote_age_ms <= max_age,
                    f"quote age {quote_age_ms:.0f}ms (max {max_age:.0f}ms)",
                )
            )
        return checks

    def _real_exit_style(self, report: Any) -> Any:
        """The exit style a REAL position is managed under.

        Uses the report's declared style when present, otherwise the same
        momentum-failure style the PAPER book defaults an entry to. The point is
        that a REAL position is governed by the SAME exit policy as a paper one.
        """
        style = getattr(report, "exit_style", None)
        if isinstance(style, ExitStyle):
            return style
        plan = getattr(report, "exit_plan", None)
        if isinstance(plan, Mapping):
            raw = plan.get("style")
            for candidate in ExitStyle:
                if raw == candidate.value or raw == candidate:
                    return candidate
        return ExitStyle.MOMENTUM_FAILURE

    async def _classify_real_settlement(self, fill: Any, *, mint: str) -> Any:
        """
        Read the confirmed exit transaction back and state what it returned.

        READ ONLY. Fetches the transaction, computes the signer's own native
        movement, and classifies the output asset. Any failure to read is
        reported as UNREADABLE rather than being resolved toward "it was SOL",
        because the two guesses have opposite consequences: assuming SOL books
        money that is not there, and assuming nothing blocks an exit that
        actually succeeded.
        """
        from .settlement import WSOL_MINT as SETTLE_WSOL
        from .settlement import classify_settlement

        if not getattr(fill, "signature", ""):
            return classify_settlement(None, wallet=self.real_backend.pubkey)

        try:
            tx = await self.real_backend.rpc.get_transaction(
                fill.signature, commitment="confirmed")
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "cannot read back exit %s: %s",
                fill.signature, type(exc).__name__)
            return classify_settlement(None, wallet=self.real_backend.pubkey)

        native_delta = 0
        try:
            meta = (tx or {}).get("meta") or {}
            keys = [str(k.get("pubkey") if isinstance(k, dict) else k)
                    for k in ((tx or {}).get("transaction", {}).get("message", {})
                              .get("accountKeys") or [])]
            pre, post = meta.get("preBalances") or [], meta.get("postBalances") or []
            idx = next((i for i, k in enumerate(keys)
                        if k == self.real_backend.pubkey), None)
            if idx is not None and idx < len(pre) and idx < len(post):
                native_delta = int(post[idx]) - int(pre[idx])
        except (StopIteration, TypeError, ValueError):
            native_delta = 0

        return classify_settlement(
            tx,
            wallet=self.real_backend.pubkey,
            native_lamports_delta=native_delta,
            expected_output_mint=SETTLE_WSOL,
        )

    async def _schedule_settlement_normalization(
        self, reading: Any, *, symbol: str, source_fill: Any
    ) -> None:
        """
        Queue the follow-up conversion when the exit settled in a convertible SPL.

        A CONVERSION IS A REAL ORDER, so it is DESCRIBED here and executed by the
        ordinary pipeline with its own intent key. It is not a second leg of the
        sell and it is not folded into the sell's accounting: it carries its own
        quote, size gate, simulation, signature, broadcast, confirmation and
        reconciliation, and it is refused outright for any asset outside the
        normalizable set so this code can never auto-swap something nobody
        approved.
        """
        from .settlement import WSOL_MINT as SETTLE_WSOL
        from .settlement import normalize_intent_for

        intent = normalize_intent_for(
            reading,
            source_mint=reading.output_mint,
            amount=reading.output_amount,
        )
        if intent is None:
            log("WARN", "settlement %s for %s is not auto-normalizable; "
                "leaving it to the operator", reading.output_mint[:12], symbol)
            return

        logical_key = self._real_logical_key(
            "NORMALIZE", f"{reading.output_mint}:{source_fill.signature[:16]}")
        self.journal.event(
            "real_settlement_normalization_queued", TradeMode.REAL,
            symbol=symbol, input_mint=intent["input_mint"],
            input_amount=intent["input_amount"],
            output_mint=intent["output_mint"], logical_key=logical_key,
        )
        log("INFO", "settlement normalization queued: %s %s -> %s",
            intent["input_amount"], intent["input_mint"][:12],
            SETTLE_WSOL[:12])

    def _exit_mark_vs_venue_pct(self, obs, fill, *, pos=None,
                                exit_quote=None) -> Any:
        """
        Compare the exit-triggering mark against what the position was REALLY
        worth, in the SELL direction.

        This is the comparison whose absence let $BUTTHOLE report +72.99% and
        settle at -76%, and which was STILL silent on $GOAT because its only
        input was ``fill.input_amount`` -- a field that is ``None`` for every
        native-SOL unwind, which is the normal case on an exit. The guard was
        therefore inert on exactly the trades it existed to catch.

        WHY IT IS NOT INERT NOW
        -----------------------
        The input leg of a SELL is the quantity the position held, and the
        position knows it. It is passed in and used when the fill does not carry
        it. Both legs then come from real amounts: the mark from the observation
        that triggered the exit, the other side from the settled lamports of the
        transaction that just confirmed, oriented ``output / input`` because we
        gave the tokens up.

        ``exit_quote``, when supplied, is the venue's own price for this size and
        is preferred over the derived settlement price, because it is a price
        the venue stated rather than one reconstructed here.

        Fail-closed on absence. A missing or unusable leg yields ``None`` rather
        than a ratio, because a fabricated divergence is the same class of
        defect as a fabricated price.
        """
        from .pricing import divergence, execution_price_per_token, normalize_side

        mark = getattr(obs, "price_usd", None) if obs is not None else None
        if not mark or mark <= 0:
            return None

        if exit_quote is not None:
            venue = exit_quote.price_usd_per_whole_token
            if venue is None or venue <= 0:
                return None
            return divergence(mark, venue, kind="EXIT_QUOTE")

        tokens_sold = getattr(pos, "token_amount", None)
        lamports_per_token = execution_price_per_token(
            side="EXIT",
            input_amount=(tokens_sold
                          if fill is not None
                          and getattr(fill, "input_amount", None) is None
                          else getattr(fill, "input_amount", None)),
            output_amount=getattr(fill, "output_amount", None),
        )
        if lamports_per_token is None or lamports_per_token <= 0:
            return None
        sol_px = getattr(self, "_sol_price_usd", None) or 0.0
        if not sol_px or sol_px <= 0:
            return None
        dec = (getattr(pos, "entry_token_decimals", None)
               if pos is not None else None)
        if not isinstance(dec, int) or dec < 0:
            dec = 6
        realized = (lamports_per_token / 1_000_000_000) * (10 ** dec) * sol_px
        return divergence(mark, realized, kind="EXIT_ACTUAL")

    # ------------------------------------------------------------------
    # executable quotes (OBSERVED_MARK != EXECUTABLE_QUOTE != ACTUAL_FILL)
    # ------------------------------------------------------------------

    def _public_rpc(self) -> Any:
        """The free public-RPC reader, built once and reused.

        Cached, deduplicated, rate-limited and retried. It is the DEFAULT on-chain
        read source and it needs no key: nothing in this runtime depends on a
        paid RPC or a paid data tier, and constructing one makes no network
        call.
        """
        rpc = getattr(self, "_public_rpc_client", None)
        if rpc is None:
            from .execution.chainstate import PublicRpcClient, RateBudget
            rc = self.config.real
            rpc = PublicRpcClient(
                url=rc.public_rpc_url,
                timeout=rc.public_rpc_timeout_s,
                cache_ttl_s=rc.public_rpc_cache_ttl_s,
                max_retries=rc.public_rpc_max_retries,
                budget=RateBudget(rate_per_s=rc.public_rpc_rate_per_s,
                                  max_concurrency=rc.public_rpc_max_concurrency),
            )
            self._public_rpc_client = rpc
        return rpc

    def _divergence_guard(self) -> Any:
        from .execution.quoting import DivergenceGuard
        rc = self.config.real
        return DivergenceGuard(
            quote_worse_than_mark_max_pct=rc.entry_quote_divergence_max_pct,
            quote_much_better_than_mark_max_pct=rc.entry_quote_alarm_max_pct,
            reject_cross_venue=rc.entry_quote_reject_cross_venue,
            max_quote_age_s=(rc.quote_max_age_ms / 1000.0
                             if rc.quote_max_age_ms else None),
        )

    def set_quote_provider(self, provider: Any) -> None:
        """
        Replace the executable-quote provider chain.

        THE ONLY WAY THE CHAIN IS EVER REPLACED. Tests inject a fake here; a
        production process never calls it. There is deliberately no environment
        variable that points at a local provider, because a configurable quote
        source is a configurable price.
        """
        self._quote_provider_override = provider

    def _paper_quote_provider(self) -> Any:
        """
        The executable-quote provider chain for PAPER.

        IT NEVER TOUCHES THE REAL EXECUTION BACKEND. A PAPER process must stay
        structurally incapable of building, signing or broadcasting -- and
        ``tests/test_paper_smoke.py`` asserts the REAL executor is not even
        constructed. So PAPER quotes through a QUOTE-ONLY client that has no
        build method, no signer and no broadcaster, rather than through the
        execution backend it must not reach.
        """
        override = getattr(self, "_quote_provider_override", None)
        if override is not None:
            return override
        cached = getattr(self, "_paper_quote_chain", None)
        if cached is None:
            from .execution.aggregators import QuoteOnlyJupiterProvider
            cached = QuoteOnlyJupiterProvider(
                timeout=self.config.real.public_rpc_timeout_s)
            self._paper_quote_chain = cached
        return cached

    def _native_reference_service(self) -> Any:
        """
        A quote service for native reference queries during hold-path monitoring.

        This service can query Pump.fun bonding curve, PumpSwap pool, and Jupiter
        for the SAME-MARKET native reference price (the venue's own spot/marginal
        price). It is used to evaluate exit triggers against the market the
        position was actually opened on, rather than an aggregator mark.

        The service is built lazily and cached. It uses the same read-only RPC
        client as the execution-aware PAPER bridge, so it respects the telemetry
        lane budget and does not starve execution capacity.
        """
        cached = getattr(self, "_native_reference_chain", None)
        if cached is not None:
            return cached

        # Use the same RPC client as the paper bridge if available, otherwise
        # build a minimal read-only client.
        rpc = self._paper_bridge_rpc()
        if rpc is None:
            from .same_market.rpc import SolanaReadOnlyRpc
            rpc = SolanaReadOnlyRpc(
                url=self.config.real.public_rpc_url or None,
                urls=tuple(
                    u.strip() for u in (self.config.real.public_rpc_urls or "").split(",")
                    if u.strip()),
                max_rpm=max(1, int(round(
                    float(self.config.real.public_rpc_rate_per_s) * 60.0))),
                max_concurrent=max(1, int(self.config.real.public_rpc_max_concurrency)),
                limiter_wait_s=float(self.config.real.public_rpc_limiter_wait_s),
                timeout_s=float(self.config.real.public_rpc_timeout_s),
            )

        from .paper_live.quotes import QuoteService, SolReference
        from .same_market.jupiter import JupiterQuoteClient
        market_cfg = getattr(self.config, "market", None)
        jupiter = JupiterQuoteClient(
            timeout_s=float(getattr(market_cfg, "http_timeout_s", 8.0)),
            cache_ttl_s=float(getattr(market_cfg, "cache_ttl_s", 2.0)),
        )
        sol_usd = self._sol_price_usd or 0.0
        cached = QuoteService(
            rpc=rpc,
            sol_reference=SolReference(sol_usd, self.clock.now(), "runtime"),
            jupiter=jupiter,
            max_quote_age_ms=30_000.0,
            allow_jupiter_fallback=True,
        )
        self._native_reference_chain = cached
        return cached

    def _quote_provider(self) -> Any:
        """
        The executable-quote provider chain, built EXPLICITLY from config.

        Provider selection is data. An unknown key is refused rather than
        skipped, so a typo cannot quietly remove a venue from the chain and make
        a refusal look like an absence of routes.
        """
        from .execution.aggregators import AggregatorQuoteProvider
        from .execution.chainstate import PublicRpcClient
        from .execution.price_layers import SwapDirectionError
        from .execution.pump import (
            PumpBondingCurveQuoteProvider,
            PumpSwapQuoteProvider,
        )
        from .execution.quoting import RouterQuoteProvider

        override = getattr(self, "_quote_provider_override", None)
        if override is not None:
            return override
        rc = self.config.real
        chain: list[Any] = []
        for key in [k.strip().lower()
                    for k in (rc.quote_provider_chain or "").split(",")]:
            if not key:
                continue
            try:
                if key in ("pumpswap", "pump_swap"):
                    chain.append(PumpSwapQuoteProvider(
                        self.real_backend.provider,
                        chain=self._public_rpc(),
                        require_program_attestation=rc.pool_attestation_required,
                    ))
                elif key in ("pump_bonding_curve", "pumpfun", "pump_amm"):
                    chain.append(PumpBondingCurveQuoteProvider(
                        self.real_backend.provider,
                        chain=self._public_rpc(),
                        require_program_attestation=rc.pool_attestation_required,
                    ))
                elif key == "jupiter":
                    prov = getattr(self.real_backend, "fallback_provider", None)
                    chain.append(AggregatorQuoteProvider(
                        prov() if callable(prov) else prov,
                        name="jupiter"))
                elif key == "raptor":
                    chain.append(AggregatorQuoteProvider(
                        self.real_backend.provider, name="raptor"))
                else:
                    raise ValueError(
                        f"unknown quote provider {key!r} in "
                        f"REAL_QUOTE_PROVIDER_CHAIN; refusing to start a chain "
                        f"that silently omits a venue")
            except SwapDirectionError:
                raise
        if not chain:
            raise ValueError("REAL_QUOTE_PROVIDER_CHAIN resolved to no providers")
        return RouterQuoteProvider(chain)

    def set_token_decimals_reader(self, reader: Any) -> None:
        """
        Replace the on-chain decimals reader.

        Same rule as :meth:`set_quote_provider`: an explicit, code-level seam for
        tests, never an environment variable. Decimals are the one number that
        must never be guessed, which is exactly why the production reader is the
        only one and this override has to be deliberate.
        """
        self._token_decimals_override = reader

    async def _token_decimals(self, mint: str) -> Optional[int]:
        """The mint's decimals, READ FROM CHAIN. ``None`` when unreadable.

        Never guessed. A per-token price derived from wrong decimals is wrong by
        a power of ten, and on 2026-10-02 the 6-decimal scale of a Token-2022
        mint had to be reconstructed by hand because nothing in the runtime
        recorded it.
        """
        override = getattr(self, "_token_decimals_override", None)
        if override is not None:
            value = override(mint)
            if hasattr(value, "__await__"):
                value = await value
            return int(value) if value is not None else None
        read = await self._public_rpc().get_mint(mint)
        if not read.ok or not read.value:
            return None
        dec = read.value.get("decimals")
        return int(dec) if dec is not None else None

    async def _entry_quote_guard(self, *, mint: str, lamports: int,
                                 sol_price: Optional[float],
                                 features: Any) -> tuple[Any, Any]:
        """
        Ask the venue what this entry would actually cost, and refuse if the
        answer is materially worse than the mark the entry is justified by.

        Returns ``(quote, verdict, mark)``. The verdict is always returned,
        including on every failure, so a refusal is always explainable and never
        silent. The mark is returned so the position can record which POOL it
        described, which is what makes a cross-venue round trip visible later.
        """
        from .execution.price_layers import ObservedMark
        from .execution.quoting import DivergenceVerdict
        from .execution.solana import WSOL_MINT

        guard = self._divergence_guard()
        now = self.clock.now()

        # -- the observed side -------------------------------------------
        mark: Optional[ObservedMark] = None
        try:
            mark_price, mark_state = self.market.mark_price(mint, now)
            obs = self.market.latest(mint)
        except Exception as exc:  # noqa: BLE001 - a market read must not crash
            log("WARN", "entry mark read failed for %s: %s", mint, exc)
            obs = None
            mark_price = None
            mark_state = None
        if obs is not None and mark_price:
            # The mark keeps its POOL IDENTITY, which is what makes a
            # cross-venue mismatch detectable later: a mark about a different
            # pool is not wrong, it is about somewhere else.
            base = ObservedMark.from_observation(
                obs, now=now,
                source=(getattr(obs, "provider", None) or "dexscreener"))
            mark = ObservedMark(
                mint=base.mint, price=float(mark_price),
                quote_mint=base.quote_mint, source=base.source,
                venue=base.venue, pool=base.pool,
                observed_at=base.observed_at,
                source_timestamp=base.source_timestamp,
                received_at=base.received_at,
                liquidity_usd=base.liquidity_usd,
                market_cap_usd=base.market_cap_usd,
                provenance={**base.provenance, "data_state": str(mark_state)},
                stale=bool(mark_state and "STALE" in str(mark_state).upper()),
                quality=("live" if mark_state and "LIVE" in str(mark_state).upper()
                         else "stale"),
            )
        elif features is not None and getattr(features, "price", None):
            # No live observation, but the decision-time mark exists. It is used
            # as a MARK, labelled as one, and never as a price.
            mark = ObservedMark(
                mint=mint, price=float(features.price),
                source="decision_features", observed_at=now,
                provenance={"note": "decision-time feature price; no live "
                                    "observation was available"})

        # -- the executable side ----------------------------------------
        dec = await self._token_decimals(mint)
        if dec is None:
            return None, DivergenceVerdict(
                False, "token_decimals_unreadable", fail_closed=True,
                detail="the mint's decimals could not be read from chain, so no "
                       "per-token executable price can be stated; refusing"), \
                mark
        try:
            provider = self._quote_provider()
            quote = await provider.quote_buy(
                token_mint=mint,
                quote_mint=WSOL_MINT,
                quote_amount_raw=int(lamports),
                token_decimals=dec,
                slippage_bps=self.config.real.slippage_bps,
                quote_asset_usd=sol_price,
            )
        except Exception as exc:  # noqa: BLE001 - every failure fails closed
            log("WARN", "entry quote unavailable for %s: %s", mint, exc)
            return None, DivergenceVerdict(
                False, "executable_quote_unavailable", fail_closed=True,
                detail=f"{type(exc).__name__}: {exc}"[:220]), mark
        return quote, guard.evaluate_entry(
            mark=mark, quote=quote, now=now), mark

    def _exposure_check(self, size) -> tuple[bool, str]:
        """Total exposure must stay within the position-size policy."""
        if size.equity_usd <= 0:
            return False, "no equity available to size against"
        room = size.equity_usd * size.position_pct / 100.0
        open_exp = self.real_account.open_exposure_usd()
        if open_exp >= room:
            return (
                False,
                "open exposure ${:,.6f} already meets the ${:,.6f} limit "
                "({:g}% of equity)".format(open_exp, room, size.position_pct),
            )
        return (
            True,
            "exposure ${:,.6f} + ${:,.6f} = ${:,.6f} within ${:,.6f}".format(
                open_exp, size.position_usd,
                open_exp + size.position_usd, room,
            ),
        )

    # ------------------------------------------------------------------
    # monitoring
    # ------------------------------------------------------------------

    def _mark(self, position: Position, now: float) -> None:
        book = self.open_book["positions"]
        price, state = self.market.mark_price(position.mint, now)
        book.mark(position, mark_price=price, data_state=state, now=now)
        self._marks[position.position_id] = MarkedPosition(position, price, state)

    def _refresh_marks(self) -> dict[str, MarkedPosition]:
        now = self.clock.now()
        for p in self.account.open_positions():
            self._mark(p, now)
        return dict(self._marks)

    def _market_derivatives(self, mint: str, now: float) -> dict[str, float | None]:
        history = self.market.history(mint)
        latest = history.latest()
        window_m = self.config.exits.momentum_collapse_lookback_s
        window_s = self.config.exits.structure_break_lookback_s
        liq_change = None
        if latest is not None and len(history) >= 2:
            older = history.last_n(6)[0]
            if older.liquidity_usd:
                liq_change = (
                    (latest.liquidity_usd - older.liquidity_usd) / older.liquidity_usd * 100.0
                )
        return {
            "momentum_30s_pct": _window_change(history, now, window_m),
            "structure_pct": _window_change(history, now, window_s),
            "liquidity_change_pct": liq_change,
        }

    def _paper_bridge_rpc(self):
        """
        The read-only RPC client behind the PAPER bridge, if there is one.

        Resolved by walking the real collaborator graph rather than by keeping a
        parallel reference, so it cannot drift out of sync with the client the
        quote service is actually using.
        """
        try:
            service = getattr(self._paper_bridge, "quote_service", None)
            reader = getattr(service, "_reader", None)
            return getattr(reader, "rpc", None)
        except Exception:  # noqa: BLE001
            return None

    def _log_rpc_capacity_events(self) -> None:
        """
        Log upstream RPC throttle events once each, with the facts that matter.

        Deliberately reads the READ-ONLY client's event ledger rather than
        pattern-matching error strings: wording can change, and a reworded
        message must not silently stop being reported. Never raises -- this runs
        at the top of every monitor tick and observability must not be able to
        break monitoring.
        """
        try:
            client = self._paper_bridge_rpc()
            if client is None:
                return
            from .same_market.rpc import upstream_events

            events = upstream_events()
            seen = getattr(self, "_rpc_events_logged", 0)
            for ev in events[seen:]:
                log(
                    "SYSTEM",
                    "RPC UPSTREAM THROTTLE provider=%s op=%s retry_after=%s "
                    "cooldown=%.1fs blocked_until=%.0f cooldown_entered=%s",
                    ev.get("provider"), ev.get("operation"),
                    ev.get("retry_after_s"),
                    float(ev.get("cooldown_remaining_s") or 0.0),
                    float(ev.get("blocked_until") or 0.0),
                    ev.get("entered_cooldown"),
                )
            self._rpc_events_logged = len(events)
        except Exception:  # noqa: BLE001 - observability never breaks monitoring
            return

    def _log_rpc_capacity_summary(self) -> None:
        """One line naming the reservation, so the split is inspectable."""
        try:
            limiter = getattr(self._paper_bridge_rpc(), "limiter", None)
            if limiter is None:
                return
            s = limiter.stats()
            log(
                "SYSTEM",
                "RPC budget total=%s/min telemetry=%s/min reserved_for_execution=%s"
                " | local_window_refusals=%s upstream_blocked_refusals=%s "
                "telemetry_budget_refusals=%s",
                s.get("max_per_minute"), s.get("telemetry_max_per_minute"),
                s.get("reserved_for_execution"),
                s.get("rejections_local_window"),
                s.get("rejections_upstream_blocked"),
                s.get("rejections_telemetry_budget"),
            )
        except Exception:  # noqa: BLE001
            return

    async def monitor_once(self) -> None:
        now = self.clock.now()
        # Per-tick diagnostics
        rpc_calls_this_tick = 0
        http_calls_this_tick = 0
        tick_start = time.monotonic()

        # DRAIN UPSTREAM RPC THROTTLE EVENTS INTO THE OPERATOR LOG.
        self._log_rpc_capacity_events()

        # THE AMARIH EXPERIMENT IS MANUFACTURED HERE.
        #
        # The POLLER is started or retired here, because this is a point in the
        # loop where nothing else is mid-transition. Its scheduled exits are NOT
        # settled here: the dedicated deadline scheduler
        # (``_amarih_deadline_loop``) now owns exit timing on its own 1-second
        # cadence, because servicing them on this tick pinned every exit to
        # ``market.poll_interval_s`` (5 s) -- the Amarih arms are not in
        # ``self.account``/``self.real_account``, so the monitor never selected
        # its faster interval while the experiment was running.
        if self.state.trade_mode is TradeMode.PAPER:
            self._sync_amarih_runtime()

        if self._execution_aware_paper() and self._paper_bridge is not None:
            # Reconcile BEFORE monitoring, so an arm that was deselected while
            # it held a position is still monitored on this tick. Without this
            # the deferral would not take effect until the next entry, and the
            # position would sit unmonitored in between -- which is precisely
            # the outcome the deferral exists to prevent.
            self._reconcile_paper_bridge_arms()
            try:
                trades = await self._paper_bridge.tick()
            except Exception as exc:  # noqa: BLE001 - never kill the runtime
                log("ERROR", "execution-aware monitor cycle failed: %s: %s",
                    type(exc).__name__, exc)
                trades = []
            if trades:
                durable = self._record_arm_trades(trades)
                self.persist_paper_dataset()
                self._retain_arm_exit_alerts(durable)
                rows, total = self.closed_positions_view()
                log("SYSTEM", "PAPER ledger query after close: "
                    "visible=%d total=%d newest=%s",
                    len(rows), total,
                    (rows[0].get("symbol") if rows else "none"))
            # ALWAYS, AND NOT GATED ON ``trades``.
            #
            # ``tick`` also settles the V2 FLOOR (BETA) experiment and queues
            # ITS exit / exit-blocked notifications, and those are
            # ``BetaPosition``s, NOT engine ``TradeRecord``s -- so they never
            # appear in ``trades``. Gating the flush on ``trades`` therefore left
            # every beta receipt sitting in the queue until an unrelated L25 arm
            # happened to close, and under a BETA-ONLY selection no regular arm
            # exists at all: a beta CLOSE, and a beta EXIT BLOCKED, were queued
            # and NEVER delivered. The settlement itself is already durable at
            # this point (``BetaExperiment.settle_arm`` persists before it
            # returns), so flushing here cannot announce a trade the state has
            # not accepted.
            self._flush_paper_bridge_messages()

        open_positions = self.account.open_positions()
        if not open_positions:
            return

        # Get native reference service for hold-path identity monitoring
        native_ref_service = None
        try:
            native_ref_service = self._native_reference_service()
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "native reference service unavailable: %s", exc)

        # Phase 1: Fetch aggregator marks for ALL positions (batched HTTP)
        mark_results: dict[str, tuple[Any, DataState]] = {}
        for position in open_positions:
            try:
                q = await self.market.quote_async(position.mint)
                http_calls_this_tick += 1
                obs = getattr(q, "observation", None)
                if obs is not None:
                    state = obs.freshness(now, self.config.market.stale_after_ms,
                                          self.config.market.unusable_after_ms)
                else:
                    state = DataState.MISSING
                mark_results[position.mint] = (obs, state)
                if state != DataState.LIVE:
                    _inc_monitor_diag("MONITOR_MARK_STALE")
            except Exception as exc:
                _inc_monitor_diag("MONITOR_HTTP_ERROR")
                mark_results[position.mint] = (None, DataState.MISSING)
                log("ERROR", "mark quote failed for %s: %s",
                    position.mint[:8], type(exc).__name__)

        # Phase 2: Native reference queries (RPC, telemetry lane)
        native_ref_service = None
        try:
            native_ref_service = self._native_reference_service()
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "native reference service unavailable: %s", exc)

        for position in open_positions:
            obs, state = mark_results.get(position.mint, (None, DataState.MISSING))
            self._mark(position, now)
            # Determine trigger price/state
            trigger_price = position.mark_price
            trigger_state = state

            # Query native reference for positions with pinned market identity
            if (native_ref_service is not None and
                    position.executable_quote_pool is not None and
                    position.executable_quote_pool != ""):
                try:
                    from .same_market.rpc import rpc_lane, telemetry_sample, LANE_TELEMETRY
                    from .same_market.rpc import current_rpc_lane
                    decimals = position.token_decimals if position.token_decimals is not None else 6
                    token_amount_raw = int(position.quantity * (10 ** decimals))
                    # Count RPC calls made during this native reference query
                    # by temporarily patching the RPC client
                    rpc_client = self._paper_bridge_rpc()
                    rpc_before = rpc_client.calls if rpc_client else 0

                    with telemetry_sample():
                        with rpc_lane(LANE_TELEMETRY):
                            quote = await native_ref_service.quote_sell(
                                mint=position.mint,
                                token_amount_raw=int(position.quantity * (10 ** decimals)),
                                token_decimals=decimals,
                                observed_pair_address=position.executable_quote_pool,
                            )

                    rpc_after = rpc_client.calls if rpc_client else 0
                    rpc_calls_this_tick += max(0, rpc_after - rpc_before)

                    if getattr(quote, "available", False):
                        native_ref_price = quote.effective_price_usd
                        native_ref_state = DataState.LIVE
                        # Check identity
                        identity = getattr(quote, "identity", "UNKNOWN")
                        if identity == "PASS":
                            pass  # MONITOR_OK counted below
                        elif identity == "FAIL":
                            _inc_monitor_diag("MONITOR_IDENTITY_MISMATCH")
                        else:
                            _inc_monitor_diag("MONITOR_IDENTITY_UNKNOWN")
                        trigger_price = quote.effective_price_usd
                        trigger_state = DataState.LIVE
                        _inc_monitor_diag("MONITOR_OK")
                    else:
                        missing = getattr(quote, "missing_reason", "") or ""
                        if "TELEMETRY_BUDGET" in missing or "telemetry" in missing.lower():
                            _inc_monitor_diag("MONITOR_SKIPPED_BUDGET")
                        elif "IDENTITY" in missing.upper():
                            _inc_monitor_diag("MONITOR_IDENTITY_UNKNOWN")
                        else:
                            _inc_monitor_diag("MONITOR_QUOTE_UNAVAILABLE")
                except Exception as exc:  # noqa: BLE001
                    from .same_market.rpc import RpcRateLimited, TELEMETRY_BUDGET
                    if isinstance(exc, RpcRateLimited) and getattr(exc, "reason", None) == TELEMETRY_BUDGET:
                        _inc_monitor_diag("MONITOR_SKIPPED_BUDGET")
                    else:
                        _inc_monitor_diag("MONITOR_RPC_ERROR")
                    log("ERROR", "native reference query failed for %s: %s",
                        position.mint[:8], exc)

            # Use native reference for exit triggers if available, else aggregator mark
            trigger_price = trigger_price if 'trigger_price' in locals() and trigger_price is not None else position.mark_price
            trigger_state = trigger_state if 'trigger_state' in locals() and trigger_state is not None else state

            deriv = self._market_derivatives(position.mint, now)
            book = self.open_book["positions"]
            min_hold = (
                0.0
                if position.exit_style in (ExitStyle.DISASTER_CAP, ExitStyle.DATA_INVALID)
                else 5.0
            )
            decision = book.assess(
                position,
                now=now,
                mark_price=trigger_price,
                data_state=trigger_state,
                min_hold_s=min_hold,
                **deriv,
            )
            if decision.action == "HOLD":
                book.record_hold(position, decision, now)
                continue
            await self._do_exit(position, decision, now)

        # Log per-tick diagnostics
        elapsed_ms = (time.monotonic() - tick_start) * 1000
        log("MONITOR_TICK", "positions=%d rpc=%d http=%d elapsed=%.1fms "
            "OK=%d SKIPPED=%d QUOTE_UNAV=%d ID_UNK=%d ID_MIS=%d RPC_ERR=%d HTTP_ERR=%d MARK_STALE=%d",
            len(open_positions), rpc_calls_this_tick, http_calls_this_tick, elapsed_ms,
            _MONITOR_DIAG["MONITOR_OK"], _MONITOR_DIAG["MONITOR_SKIPPED_BUDGET"],
            _MONITOR_DIAG["MONITOR_QUOTE_UNAVAILABLE"], _MONITOR_DIAG["MONITOR_IDENTITY_UNKNOWN"],
            _MONITOR_DIAG["MONITOR_IDENTITY_MISMATCH"], _MONITOR_DIAG["MONITOR_RPC_ERROR"],
            _MONITOR_DIAG["MONITOR_HTTP_ERROR"], _MONITOR_DIAG["MONITOR_MARK_STALE"])

        await self._monitor_real_positions(now)
        # REAL positions are monitored by the SAME loop and the SAME exit
        # policy. Before this, a REAL position could be opened and never sold:
        # TP, SL, momentum failure and structure invalidation were all evaluated
        # for PAPER positions only.
        await self._monitor_real_positions(now)

    # ------------------------------------------------------------------
    # REAL exit
    # ------------------------------------------------------------------

    async def _monitor_real_positions(self, now: float) -> None:
        """Mark and assess every OPEN REAL position, then exit when the policy
        says to. Runs regardless of the entry gate, because KILL/PAUSE must stop
        NEW entries without orphaning a position that is already on chain."""
        positions = self.real_account.open_positions()
        if not positions:
            return
        book = self.open_book["positions"]
        for pos in positions:
            try:
                q = await self.market.quote_async(pos.mint)
            except Exception as exc:  # noqa: BLE001 - a failed quote is not a crash
                log("ERROR", "REAL mark quote failed for %s: %s",
                    pos.mint[:8], type(exc).__name__)
                q = None
            obs = getattr(q, "observation", None) if q is not None else None
            price = obs.price_usd if obs is not None else None
            state = (
                obs.freshness(now, self.config.market.stale_after_ms,
                              self.config.market.unusable_after_ms)
                if obs is not None else DataState.MISSING
            )
            self.real_account.mark(pos.mint, price, now)
            if state is not DataState.LIVE:
                self._real_data_invalid_since.setdefault(pos.mint, now)
            else:
                self._real_data_invalid_since.pop(pos.mint, None)
            deriv = self._market_derivatives(pos.mint, now)
            min_hold = (
                0.0
                if pos.exit_style in (ExitStyle.DISASTER_CAP, ExitStyle.DATA_INVALID)
                else 5.0
            )
            # The SAME ExitPolicy object the PAPER book uses. ``RealPosition``
            # exposes the fields the policy reads, so both modes share one
            # definition of "when to exit".
            decision = book.policy.assess(
                pos,
                now=now,
                mark_price=price,
                data_state=state,
                momentum_30s_pct=deriv["momentum_30s_pct"],
                structure_30s_pct=deriv["structure_pct"],
                liquidity_change_pct=deriv["liquidity_change_pct"],
                data_invalid_since=self._real_data_invalid_since.get(pos.mint),
            )
            if decision.action == "EXIT" and decision.trigger in (
                "momentum_collapse", "structure_break", "peak_giveback",
                "wide_giveback", "no_follow_through",
            ) and (now - pos.opened_at) < min_hold:
                continue
            if decision.action == "HOLD":
                pos.exit_hold_count += 1
                continue
            await self._execute_real_sell(pos, decision, now)

    async def _execute_real_sell(
        self, pos, decision, now: float
    ) -> None:
        """
        The complete production REAL exit path:

            open position -> exit authorization -> fresh quote -> fill-time
            stale validation -> risk validation -> idempotency reservation ->
            transaction construction -> simulation -> signing -> broadcast ->
            confirmation -> reconciliation -> realized PnL -> journal ->
            operator notification

        Nothing here is paper-only. Every failure is recorded as
        ``SELL_BLOCKED=<reason>`` or ``REAL_EXECUTION_FAILED=<reason>``, never as
        a silent no-op.
        """
        symbol = pos.symbol or "?"
        mint = pos.mint
        stages = self._current_stages

        def _stage(name: str) -> None:
            if stages is not None:
                stages.mark(name, self.clock.now())

        # -- exit authorization (fail closed) --------------------------
        if not pos.confirmed:
            self._record_sell_block(symbol, mint, "position_not_confirmed",
                                    "the entry fill is not confirmed on chain")
            return
        if pos.token_amount <= 0:
            self._record_sell_block(symbol, mint, "zero_token_amount",
                                    "the position holds no token quantity")
            return
        if self.real_executor.busy:
            self._record_sell_block(symbol, mint, "concurrent_operation",
                                    "another real operation is in flight")
            return
        if self.execution_intents.has_unresolved(mint=mint):
            self._record_sell_block(symbol, mint, _BLOCKED_UNRESOLVED,
                                    "a real execution for this mint may be in "
                                    "flight; refusing to broadcast another")
            return

        # -- fresh market / quote --------------------------------------
        _stage("real_exit_quote_start")
        try:
            q = await self.market.quote_async(mint)
        except Exception as exc:  # noqa: BLE001
            self._record_sell_block(
                symbol, mint, "quote_unavailable",
                f"market quote failed ({type(exc).__name__})")
            return
        obs = getattr(q, "observation", None)
        _stage("real_exit_quote_end")
        state = (
            obs.freshness(now, self.config.market.stale_after_ms,
                          self.config.market.unusable_after_ms)
            if obs is not None else DataState.MISSING
        )
        if obs is None or obs.price_usd is None or state is not DataState.LIVE:
            self._record_sell_block(
                symbol, mint, STALE_QUOTE if state is DataState.STALE else "stale_market_data",
                f"market state at exit was {state.value}")
            return
        # -- FILL-TIME STALE GATE (protection for the SELL) -------------
        age_ms = max(0.0, (now - obs.observed_at) * 1000.0)
        if age_ms > self.stale_gate.budget_ms:
            self._record_sell_block(
                symbol, mint, STALE_QUOTE,
                f"exit quote age {age_ms:.0f}ms exceeds the "
                f"{self.stale_gate.budget_ms:.0f}ms budget")
            return

        sol_price = await self._backend_sol_price()
        if not sol_price or sol_price <= 0:
            self._record_sell_block(
                symbol, mint, "sol_usd_unavailable",
                "SOL/USD unavailable; realized PnL cannot be stated")
            return

        # -- risk validation -------------------------------------------
        _stage("real_exit_checks_start")
        checks = await self._real_exit_checks(pos, sol_price)
        _stage("real_exit_checks_end")
        failures = [c for c in checks if not c.ok]
        if failures:
            detail = ", ".join(f"{c.name} ({c.detail})" for c in failures)
            self._record_sell_block(symbol, mint, "risk_refused", detail)
            return

        # -- the executable SELL quote ----------------------------------
        #
        # Captured BEFORE the irreversible sequence so the exit can be measured
        # against what the venue said it would pay, and so the venue that will
        # actually be used is KNOWN rather than discovered afterwards. A $GOAT
        # exit filled on a PumpSwap pool while the mark being acted on described
        # a different pool entirely, and nothing recorded either fact.
        #
        # A missing exit quote is NOT a refusal: a confirmed exit must never be
        # blocked, because refusing would leave a position open while its tokens
        # are gone. It is recorded as missing, with a reason.
        from .execution.solana import WSOL_MINT
        exit_quote = None
        exit_quote_reason = ""
        _stage("real_exit_executable_quote_start")
        try:
            dec = pos.entry_token_decimals
            if dec is None:
                dec = await self._token_decimals(mint)
            if dec is None:
                exit_quote_reason = (
                    "token decimals unreadable, so no per-token exit quote")
            else:
                provider = self._quote_provider()
                exit_quote = await provider.quote_sell(
                    token_mint=mint,
                    quote_mint=WSOL_MINT,
                    token_amount_raw=int(pos.token_amount),
                    token_decimals=dec,
                    slippage_bps=self.config.real.slippage_bps,
                    quote_asset_usd=sol_price,
                )
        except Exception as exc:  # noqa: BLE001 - never blocks a confirmed exit
            exit_quote = None
            exit_quote_reason = f"{type(exc).__name__}: {exc}"[:200]
        if exit_quote is None:
            self.journal.event(
                "real_exit_quote_unavailable", TradeMode.REAL, mint=mint,
                symbol=symbol, reason=exit_quote_reason or "not captured",
                token_amount=int(pos.token_amount),
            )
        _stage("real_exit_executable_quote_end")

        # -- irreversible sequence -------------------------------------
        fill = await self.real_executor.close_position(
            mint=mint,
            token_amount=int(pos.token_amount),
            slippage_bps=self.config.real.slippage_bps,
            prechecks=checks,
            intent_store=self.execution_intents,
            logical_key=self._real_logical_key("EXIT", mint),
            stage=_stage,
        )
        if not fill.filled:
            # The position stays OPEN. A sell in flight is not a closed
            # position, and reporting it as one would be a lie.
            pos.ambiguity_reason = (
                fill.ambiguity_reason or fill.status or "exit did not confirm"
            )
            self.journal.unfilled(
                TradeMode.REAL,
                candidate_id=None, mint=mint, symbol=symbol,
                status=fill.status, reason=pos.ambiguity_reason,
                side="EXIT",
            )
            self._notify(
                "{err} <b>REAL SELL ไม่สำเร็จ</b>\n<b>${sym}</b>\n"
                "Status: {st}\nเหตุผล: {why}\n\n"
                "ตำแหน่งยังเปิดอยู่".format(
                    err=_ERR, sym=symbol, st=fill.status,
                    why=fill.ambiguity_reason or "-"),
                None,
            )
            self._record_sell_block(
                symbol, mint, "execution_failed",
                f"{fill.status}: {fill.ambiguity_reason or '-'}")
            self.save_real_state()
            return

        # -- WHAT DID THIS ACTUALLY RETURN? ------------------------------
        # A confirmed signature is not the same as "we got paid in SOL". The
        # venue may have returned USDC, or a token credited to a different
        # account, or nothing. Booking the sale from a bare signature is what let
        # a non-SOL settlement be marked received while the money was not
        # spendable, so the chain is read back and the output asset classified.
        #
        # BOOKED FIRST, ACTED ON SECOND. The sale is confirmed, so the position
        # closes: refusing to book it would leave the position OPEN forever while
        # its tokens are gone, which is a worse lie than a flagged settlement.
        # The classification then decides the PROCEEDS treatment, and any problem
        # is reported to the operator after the accounting is correct.
        settlement = await self._classify_real_settlement(fill, mint=mint)
        fill.output_mint = settlement.output_mint

        # -- WAS THE MARK THE MARKET WE COULD HAVE ACTUALLY SOLD INTO? -----
        # A venue quote was fetched moments ago by the exit quote probe, and the
        # mark that triggered this exit is ``obs.price_usd``. On $BUTTHOLE those
        # two differed by 4.09x: the position was sold at 4.99e-06 while the
        # screen read 2.043e-05, and MFE recorded +72.99% off the mark with
        # MAE at exactly 0. Nothing compared them, because nothing required the
        # mark to resemble an executable price.
        #
        # This SURFACES the divergence; it does not refuse the trade. The exit
        # gate has already passed on the venue's own quote, and silently
        # blocking a confirmed exit would repeat the $SI failure. The divergence
        # is journalled and shown so the number exists for review.
        mark_vs_venue = self._exit_mark_vs_venue_pct(
            obs, fill, pos=pos, exit_quote=exit_quote)
        if mark_vs_venue is not None:
            log("WARN", "REAL_EXIT_MARK_DIVERGENCE symbol=%s mark=%.9g "
                "venue=%.9g pct=%+.1f%%", symbol, mark_vs_venue.mark,
                mark_vs_venue.venue, mark_vs_venue.pct * 100.0)
            self.journal.event(
                "real_exit_mark_divergence", TradeMode.REAL, mint=mint,
                symbol=symbol, signature=fill.signature,
                mark=mark_vs_venue.mark, venue=mark_vs_venue.venue,
                pct=mark_vs_venue.pct * 100.0,
                mark_overstates=mark_vs_venue.mark_overstates,
            )
            # Persisted on the position so the divergence travels with the
            # closed trade instead of existing only in a log line.
            self._pending_mark_divergence_pct = mark_vs_venue.pct * 100.0

        out_price = None
        if settlement.needs_normalization:
            out_price = await self._real_asset_price_usd(
                settlement.output_mint, settlement.output_mint[:8])

        record = RealFillRecord.from_onchain_fill(fill)
        reason = decision.trigger or decision.reason
        try:
            from .execution.price_layers import ObservedMark
            closed = self.real_account.close_position(
                mint=mint,
                fill=record,
                # The OBSERVED MARK the exit decision was taken on. It is
                # preserved and LABELLED; it is not the price we sold at.
                exit_price_usd=float(obs.price_usd),
                reason=reason,
                now=self.clock.now(),
                sol_price_usd=float(sol_price),
                output_mint=settlement.output_mint,
                output_decimals=9 if settlement.is_settled_sol else 6,
                output_usd_price=out_price,
                # The quantity actually held and sold. This is the EXIT leg the
                # reconciler cannot see: a native-SOL unwind records the token
                # movement as a balance change, not as an SPL transfer out of
                # the fee payer, so the fill's input leg is absent by
                # construction. Without it the exit execution price is
                # unmeasurable and the mark/venue divergence guard is silent.
                token_amount_sold=int(pos.token_amount),
                token_decimals=(pos.entry_token_decimals
                                if pos.entry_token_decimals is not None
                                else await self._token_decimals(mint)),
                exit_mark=ObservedMark(
                    mint=mint, price=float(obs.price_usd),
                    source=(getattr(obs, "provider", None) or "dexscreener"),
                    venue=getattr(obs, "venue", None),
                    pool=(dict(getattr(obs, "extra", None) or {}).get(
                        "pairAddress")),
                    observed_at=getattr(obs, "observed_at", 0.0) or 0.0,
                ) if obs is not None else None,
                exit_quote=exit_quote,
                provider=fill.quote_provider,
                venue=(exit_quote.venue if exit_quote is not None
                       else fill.route_metadata.get("provider")
                       if isinstance(fill.route_metadata, dict) else None),
                pool=(exit_quote.pool if exit_quote is not None else None),
                rent_lamports=fill.rent_lamports,
            )
            if self._pending_mark_divergence_pct is not None:
                closed.exit_mark_vs_venue_pct = self._pending_mark_divergence_pct
            self._pending_mark_divergence_pct = None
        except (ValueError, KeyError) as exc:
            # A confirmed sell that cannot be booked is an accounting failure.
            # It is reported, never silently swallowed, and the position is left
            # for reconciliation rather than being faked closed.
            self.journal.event(
                "real_exit_book_failed", TradeMode.REAL, mint=mint,
                symbol=symbol, signature=fill.signature, reason=str(exc))
            log("ERROR", "REAL exit %s could not be booked: %s",
                fill.signature, exc)
            self._record_sell_block(symbol, mint, "accounting_failed", str(exc))
            self.save_real_state()
            return

        if not settlement.is_settled_sol:
            log("WARN", "REAL_SELL_SETTLEMENT symbol=%s class=%s out_mint=%s "
                "amount=%s", symbol, settlement.settlement,
                settlement.output_mint[:12] or "-", settlement.output_amount)
            self.journal.event(
                "real_exit_settlement", TradeMode.REAL, mint=mint,
                symbol=symbol, signature=fill.signature,
                settlement=settlement.settlement,
                output_mint=settlement.output_mint,
                output_amount=settlement.output_amount,
                controlled=settlement.controlled,
                pnl_stated=closed.realized_pnl_usd is not None,
                reason=settlement.reason,
            )
            self._record_sell_block(
                symbol, mint, "settlement_" + settlement.settlement.lower(),
                settlement.reason)
            self._notify(
                "{warn} <b>REAL SELL SETTLEMENT</b>\n<b>${sym}</b>\n"
                "{what}\n{why}".format(
                    warn=_WARN if not settlement.is_problem else _ERR,
                    sym=_esc(symbol or "?"),
                    what=(
                        f"Proceeds asset: "
                        f"{_esc(settlement.output_mint or 'none')} "
                        f"({settlement.output_amount:,})"
                        + ("" if closed.realized_pnl_usd is not None
                           else "\nPnL is NOT stated: "
                                "proceeds could not be valued.")),
                    why=_esc(_sanitize_reason(settlement.reason))),
                None,
            )
            if settlement.needs_normalization:
                await self._schedule_settlement_normalization(
                    settlement, symbol=symbol, source_fill=fill)

        self.journal.real_exit(
            signature=fill.signature,
            mint=mint,
            symbol=symbol,
            position_id=closed.position_id,
            reason=reason,
            strategy=l25.ACTIVE_STRATEGY,
            requested_token_amount=int(pos.token_amount),
            actual_input_amount=fill.input_amount,
            actual_output_amount=fill.output_amount,
            actual_execution_price=fill.execution_price,
            actual_fee_lamports=fill.fee_lamports,
            actual_fee_sol=fill.fee_sol,
            realized_pnl_usd=closed.realized_pnl_usd,
            exit_price_usd=closed.exit_price_usd,
            exit_price_kind=closed.exit_price_kind,
            exit_execution_price_usd=closed.exit_execution_price_usd,
            exit_mark_vs_venue_pct=closed.exit_mark_vs_venue_pct,
            entry_cost_usd=closed.entry_cost_usd,
            sol_price_usd=float(sol_price),
            commitment=fill.commitment,
            slot=fill.slot,
            holding_seconds=closed.holding_seconds(self.clock.now()),
            mfe_pct=closed.mfe_pct,
            mae_pct=closed.mae_pct,
            # -- the three returns, kept apart ----------------------------
            realized_return_pct=closed.realized_return_pct,
            observed_mark_return_pct=closed.observed_mark_return_pct,
            executable_return_pct=closed.executable_return_pct,
            entry_execution_price_usd=closed.entry_execution_price_usd,
            entry_price_kind=closed.entry_price_kind,
            entry_cost_usd_trading_only=closed.entry_cost_usd_trading_only,
            entry_cost_components=dict(closed.entry_cost_components),
            exit_proceeds_usd=closed.exit_proceeds_usd,
            exit_cost_components=dict(closed.exit_cost_components),
            exit_venue=closed.exit_venue,
            exit_pool=closed.exit_pool,
            cross_venue_detected=closed.cross_venue_detected,
            cross_venue_detail=closed.cross_venue_detail,
            exit_quote=(exit_quote.as_dict() if exit_quote is not None
                        else None),
            exit_quote_reason=(exit_quote_reason or None),
        )
        # THE RECEIPT IS ENTITLED TO WHAT THE DISK ACTUALLY HOLDS.
        #
        # The sell settled on chain -- that is not in doubt and not re-litigated
        # here -- but ``real_account_state.json`` is the only thing a restart
        # reads. If that write did not commit, the success receipt below would
        # be a claim about a closed trade the canonical state does not contain,
        # and the next start would show the position OPEN again for tokens the
        # wallet has already sold. So the two facts are stated separately: the
        # close, then its durability.
        durable = self.save_real_state()
        # THE PERCENTAGE AN OPERATOR SEES IS DERIVED FROM THE SETTLEMENT.
        #
        # It used to be ``obs.price_usd / closed.entry_price_usd - 1``: two
        # DexScreener marks. On 2026-10-02 that printed "+71.56%" on a trade
        # whose executable price moved +6.38% and whose all-in cash result was
        # -$0.1825, immediately above a correct "+$0.0016" from the chain. The
        # mark return is still recorded, under its own name, and is not shown
        # unless an operator has enabled diagnostics.
        mark_ret = closed.observed_mark_return_pct
        if not durable:
            self._notify(
                f"{_ERR} <b>REAL SELL SETTLED - STATE NOT PERSISTED</b>\n"
                f"<b>${_esc(symbol or '?')}</b>\n\n"
                f"The transaction is final: signature "
                f"<code>{_esc(str(fill.signature)[:64])}</code>\n"
                f"Realized: {self._pnl_usd_text(closed.realized_pnl_usd)}\n\n"
                f"The closed REAL position could NOT be written to durable "
                f"state, so it is NOT in CLOSED POSITIONS and a restart will "
                f"show this position open again. Reconcile against the chain "
                f"before arming REAL again.",
                None,
            )
        else:
            self._notify(
                _close_text(
                    symbol,
                    closed.realized_return_pct,
                    closed.realized_pnl_usd,
                    mark_return_pct=mark_ret,
                    show_mark_return=bool(
                        self.config.real.show_mark_return_in_operator_output),
                    entry_fill=_fmt_usd_price(closed.entry_execution_price_usd),
                    exit_fill=_fmt_usd_price(closed.exit_execution_price_usd),
                ),
                None,
            )
        log(
            "EXECUTE",
            "REAL_SELL=SUCCESS symbol=%s pnl_usd=%s sig=%s persisted=%s",
            symbol, closed.realized_pnl_usd, fill.signature, durable,
        )

    async def _real_exit_checks(self, pos, sol_price: float) -> list[Any]:
        """Risk validation for a REAL exit. Any failure means DO NOT SELL."""
        from .execution.real import SafetyCheck

        checks: list[Any] = [
            SafetyCheck(
                "real_capability_available", self.real_capable(),
                "wallet and RPC configured" if self.real_capable()
                else "wallet/RPC not configured",
            ),
            SafetyCheck(
                "exit_position_confirmed", pos.confirmed,
                "entry transaction confirmed on chain",
            ),
            SafetyCheck(
                "exit_token_amount_positive", pos.token_amount > 0,
                f"token amount {pos.token_amount}",
            ),
            SafetyCheck(
                "no_concurrent_operation", not self.real_executor.busy,
                "no real operation in flight",
            ),
            SafetyCheck(
                "recovery_consistent", self._real_state_consistent,
                "no unresolved reconciliation mismatch"
                if self._real_state_consistent
                else "unresolved reconciliation mismatch",
            ),
            SafetyCheck(
                "no_unresolved_execution",
                not self.execution_intents.has_unresolved(mint=pos.mint),
                "no unresolved real execution for this mint"
                if not self.execution_intents.has_unresolved(mint=pos.mint)
                else "an execution for this mint may be in flight",
            ),
            SafetyCheck(
                "sol_usd_valuation_available", bool(sol_price and sol_price > 0),
                f"${sol_price:,.2f}" if sol_price else "SOL/USD unavailable",
            ),
        ]
        # A fresh venue quote proves a sell route exists before broadcast.
        #
        # DIRECTION AND UNITS (the $SI incident, 2026-10-01): this probe used to
        # price WSOL -> token, i.e. a BUY, while the trade actually about to be
        # broadcast is token -> WSOL. It also passed a WSOL lamport notional,
        # while ``close_position`` broadcasts ``int(pos.token_amount)`` raw
        # token units. Both halves were wrong, so the gate priced a phantom buy
        # of a far larger notional than the position, reported its price impact
        # as the sell's, and refused a sell it was never pricing -- leaving an
        # open REAL position that could never be exited.
        #
        # The probe must therefore mirror ``close_position`` exactly: token ->
        # WSOL, raw token amount, no lamport conversion. The safety envelope is
        # unchanged; a genuinely expensive sell is still refused.
        try:
            check, _ = await self.real_executor.quote_probe(
                mint=pos.mint,
                # RAW TOKEN UNITS. Deliberately NOT converted to SOL lamports.
                position_lamports=max(1, int(pos.token_amount)),
                slippage_bps=self.config.real.slippage_bps,
                side="EXIT",
            )
            if check.name == "quote_valid":
                check.name = "exit_quote_valid"
            checks.append(check)
        except Exception as exc:  # noqa: BLE001
            checks.append(
                SafetyCheck("exit_quote_valid", False,
                            f"exit quote probe failed ({type(exc).__name__})")
            )
        return checks

    def _record_sell_block(
        self, symbol: str, mint: str, reason: str, detail: str = ""
    ) -> None:
        """
        Record a SELL that did not become a close. Never a silent no-op.

        THE OPERATOR IS TOLD, WITH THE REPEAT COUNT. This previously only wrote
        to the log, the journal and the latency table, so a REAL exit that was
        refused every single loop looked identical to a REAL exit that was
        refused once and then the bot went quiet: nothing distinguished "one
        transient refusal" from "stuck forever, losing money". The occurrence
        number is what makes a stuck exit visible.

        ESCALATION, NOT SPAM. The first occurrence notifies, and after that every
        Nth does, so a permanently blocked position announces itself repeatedly
        without filling the operator's chat with identical lines. The number in
        the message is always the true count, not the notification count.

        NO INTENT IS RESERVED HERE. A blocked exit has broadcast nothing, so it
        must not create an execution intent: an intent that exists without a
        transaction is exactly what makes the next run believe something is
        already in flight and refuse to try again.
        """
        stages = self._current_stages
        if stages is not None:
            stages.block(reason, outcome=SELL_BLOCKED)
        self.latency.record_outcome(SELL_BLOCKED)
        log(
            "EXECUTE",
            "%s=%s symbol=%s%s",
            SELL_BLOCKED, reason, (symbol or "?")[:12],
            f" -- {detail}" if detail else "",
        )
        self.last_execution = {
            "mint": mint,
            "symbol": symbol,
            "decision": "SELL",
            "outcome": SELL_BLOCKED if reason != "execution_failed" else REAL_EXECUTION_FAILED,
            "executed": False,
            "reason": reason,
            "detail": detail,
            "at": self.clock.now(),
        }
        try:
            self.journal.unfilled(
                TradeMode.REAL, mint=mint, symbol=symbol, decision="SELL",
                reason=reason, detail=detail,
            )
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "cannot journal REAL SELL_BLOCKED (%s): %s",
                type(exc).__name__, exc)
        self._notify_sell_block(symbol, mint, reason, detail)

    def _notify_sell_block(
        self, symbol: str, mint: str, reason: str, detail: str
    ) -> None:
        """
        Notify that a REAL exit is blocked, escalating on repeats.

        THE REASON IS SANITIZED before it is shown. Detail strings here are
        assembled from RPC and venue error text, which can contain a base64
        transaction, a URL with a key, or an internal address. A status message
        is not the place for any of that, so the reason is taken from a
        sanitized, truncated, single-line form and the raw detail stays in the
        log where it belongs.
        """
        key = (mint, reason)
        self._sell_block_counts[key] = self._sell_block_counts.get(key, 0) + 1
        occurrence = self._sell_block_counts[key]

        # first occurrence, then every Nth. A stuck exit keeps announcing itself
        # and an operator watching sees the count climb.
        if occurrence != 1 and occurrence % _SELL_BLOCK_REPEAT_EVERY != 0:
            return

        self.journal.event(
            "real_exit_blocked", TradeMode.REAL, mint=mint,
            symbol=symbol, reason=reason, occurrence=occurrence,
        )
        self._notify(
            "{err} <b>POSITION EXIT BLOCKED</b>\n"
            "<b>${sym}</b> {mint}\n"
            "Reason: {why}\n"
            "Occurrence: {n}{esc_count}\n"
            "<i>The position is still OPEN. No order was sent.</i>".format(
                err=_ERR, sym=_esc(str(symbol or "?")[:16]),
                mint=_esc(mint[:12]), why=_esc(_sanitize_reason(reason)),
                n=occurrence,
                esc_count=("" if occurrence == 1
                           else f" (notified every "
                                f"{_SELL_BLOCK_REPEAT_EVERY})"),
            ),
            None,
        )

    async def _do_exit(
        self, position: Position, decision: ExitDecision, now: float
    ) -> None:
        fill_quote = await self.market.quote_async(position.mint)
        # Providers and test doubles may return None on a transport failure.
        # A missing quote is a structured unavailable-data result, never a
        # reason for the monitor task to crash before the fill model can refuse
        # the operation.
        obs = getattr(fill_quote, "observation", None) if fill_quote is not None else None
        state = (
            obs.freshness(now, self.config.market.stale_after_ms,
                          self.config.market.unusable_after_ms)
            if obs is not None else DataState.MISSING
        )
        # The EXECUTABLE layer for a PAPER exit. As with the entry it is an
        # observation of a venue price, never a settlement, and its absence is
        # recorded with a reason instead of blocking the simulated fill.
        paper_quote = None
        paper_quote_reason = ""
        if self.config.real.paper_executable_quote_enabled:
            try:
                from .execution.solana import WSOL_MINT as _W
                dec = getattr(position, "token_decimals", None)
                if dec is None:
                    dec = await self._token_decimals(position.mint)
                if dec is None:
                    paper_quote_reason = "token decimals unreadable from chain"
                else:
                    # A SELL leg needs no USD conversion, so nothing here reads
                    # the REAL backend -- see the entry note above.
                    paper_quote = await self._paper_quote_provider().quote_sell(
                        token_mint=position.mint,
                        quote_mint=_W,
                        token_amount_raw=int(position.quantity * (10 ** dec)),
                        token_decimals=dec,
                        slippage_bps=self.config.real.slippage_bps,
                        quote_asset_usd=self._sol_price_usd,
                    )
            except Exception as exc:  # noqa: BLE001 - never blocks a PAPER fill
                paper_quote = None
                paper_quote_reason = f"{type(exc).__name__}: {exc}"[:200]

        fill = simulate_fill(
            FillInputs(
                side="exit",
                quantity=position.quantity,
                notional_hint_usd=position.quantity * (position.mark_price or 0.0),
                # ``position.mark_price`` is an OBSERVED MARK. It is no longer
                # offered to the fill model as if it were a quote: the "worse of
                # quote and current" selection is a real guard only when the
                # quote leg is a venue's own price. Before 2026-10-02 it was the
                # same DexScreener mark, so the guard was inert on every PAPER
                # exit while appearing to work.
                quoted_price=None,
                decision_time=now,
                observation=obs,
                observed_at=now,
                quote=paper_quote,
                quote_missing_reason=(paper_quote_reason
                                      or "executable quote disabled by config"),
            ),
            self.config.execution,
            data_state=state,
        )
        book = self.open_book["positions"]
        trade = book.close(position, decision=decision, fill=fill, now=now)
        if trade is not None:
            self.last_trade = trade
            # A close is reported deliberately, and carries the equity it moved.
            # There was no close notification before, so an operator watching
            # only the chat had no way to see a position end.
            if trade is not None and self.state.trade_mode is TradeMode.PAPER:
                ret = getattr(trade, "return_pct", None)
                pnl = getattr(trade, "net_pnl_usd", None)
                sym = getattr(trade, "symbol", None) or \
                    getattr(trade, "mint", None) or "?"
                self._notify(
                    _close_text(
                        sym, ret, pnl,
                        mark_return_pct=getattr(trade, "peak_return_pct", None),
                        # PAPER never settles, so its percentage is a SIMULATED
                        # result. It is labelled here rather than left to look
                        # like a realized one -- the $GOAT incident was an
                        # operator unable to tell the two apart on one line.
                        show_mark_return=False,
                    ) + "\n<i>PAPER simulation · "
                         "ผลลัพธ์จำลอง ไม่ใช่การซื้อขายจริง</i>"
                    + _equity_footer(self._paper_equity_now()),
                    None,
                )
            self.memory.remember(trade)
            book.policy.memory = self.memory
        equity = self.account.equity(self._refresh_marks())
        self.account.update_peak(equity)

    # ------------------------------------------------------------------
    # control
    # ------------------------------------------------------------------

    def _service_control(self) -> None:
        """
        Apply the cross-process control file to the CANONICAL runtime state.

        The control file is an operator action arriving from another terminal,
        not a second source of truth. It is translated into a
        :class:`RuntimeState` transition and the state is what everything else
        reads, so a stop from the CLI and a stop from Telegram are the same
        event with one representation.

        Note what a control-file ``RUN`` does: nothing. Starting the trading
        loop is the operator's Telegram start button and nothing else, so a
        stray "RUN" in a file cannot start trading.
        """
        raw = load_json(self.config.control_file)
        if not raw:
            return
        cmd = str(raw.get("command", "")).strip().upper()
        if not cmd or cmd == self.control.command:
            return
        self.control = ControlCommand(command=cmd, issued_at=float(raw.get("at", 0.0)))
        log("CONTROL", "command received: %s", cmd)
        if cmd == MODE_STOP:
            try:
                self.state.stop()
            except ModeError:
                pass
            self.running = False
        elif cmd == MODE_FLATTEN:
            self._flatten_requested = True
        elif cmd == MODE_RUN:
            log(
                "CONTROL",
                "control-file RUN ignored: the trading loop starts only from "
                "the operator start button, so a stray file cannot start it",
            )

    def _drain_control_writes(self) -> None:
        if self.control.pending:
            payload = dict(load_json(self.config.control_file) or {})
            payload["acked_at"] = self.clock.now()
            save_json_atomic(self.config.control_file, payload)
            self.control.acked_at = self.clock.now()

    # ------------------------------------------------------------------
    # main loops
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self.startup()
        # Reconcile REAL state against the chain BEFORE any market processing.
        # A process that died after a broadcast must learn that here, not by
        # trading on top of it. Failure fails closed (``_real_state_consistent``
        # False) rather than blocking PAPER startup.
        try:
            await self.reconcile_real_state()
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "REAL startup reconciliation failed: %s",
                type(exc).__name__)
            self._real_state_consistent = False
        # THE EXECUTION-AWARE PAPER BOOK IS LOADED HERE, NOT AT THE FIRST SIGNAL.
        #
        # Restarting used to leave that book UNMONITORED until some candidate
        # was authorised, because the bridge is built lazily on the entry path
        # and the monitor tick skips the book while it does not exist. This runs
        # before the listener is connected, so a restart resumes every open
        # position -- and every unresolved beta arm -- before the first message
        # can be delivered. It never raises: a failure is reported and the lazy
        # path still applies.
        await self.restore_execution_aware_book()
        self._print_ready()

        stop = asyncio.Event()
        self._stop_event = stop
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):  # pragma: no cover
                pass

        await self._supervise_listener(start_stream=True)

        # The Amarih poller is NOT started here. It is started by the monitor
        # tick once AMARIH is actually selected, so exactly one poller exists
        # for the life of that selection and none exists for any other.
        monitor = asyncio.create_task(self._monitor_loop(stop))
        dash = asyncio.create_task(self._dashboard_loop(stop))
        supervisor = asyncio.create_task(self._listener_supervisor(stop))

        tasks = [monitor, dash, supervisor]
        # The Bot API poller is supervised exactly like the others. Starting it
        # does not start trading: the loop still waits for the operator START.
        bot_task = self.start_bot_control()
        if bot_task is not None:
            tasks.append(bot_task)
        if self._listener_task is not None:
            tasks.append(self._listener_task)
        try:
            await stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.shutdown()

    def _refresh_telegram_config(self) -> bool:
        """
        Re-read ``.env.local`` before each connection attempt.

        The engine reads its configuration once at construction, so a session
        added by ``cli login`` while the process is running would otherwise be
        invisible until a restart - which is exactly the wrong behaviour for the
        one situation the retry loop exists to handle.
        """
        try:
            from .config import build_config

            fresh = build_config(self.config.root)
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "could not re-read config: %r", exc)
            return False
        if not fresh.telegram.has_session:
            return False
        if fresh.telegram.session == self.config.telegram.session:
            return False
        log("TELEGRAM", "new TELEGRAM_SESSION detected in %s - adopting it",
            self.config.root / ".env.local")
        self.config = replace(self.config, telegram=fresh.telegram)
        if self.listener is not None:
            self.listener.config = fresh.telegram
        return True

    def _wire_listener_sources(self) -> None:
        """
        Point the listener at the configured candidate feeds.

        This wiring used to happen ONLY in ``__init__``, but on the production
        entrypoint the listener is attached AFTER construction: ``cli._trader``
        builds ``PaperTrader(cfg, nar=nar)`` and only then assigns
        ``trader.listener``. ``self.listener`` was therefore ``None`` when the
        block ran, so ``source_chats`` stayed unset and ``TelegramListener.run``
        took the single-chat branch: it polled the primary MooDeng channel only
        and never even FETCHED the fallback PainApe feed. A PainApe message
        could not "fail to parse" because it never arrived at the listener.

        Idempotent, and called both at construction and again immediately
        before the listener starts, so the wiring no longer depends on the
        order in which the two halves are assembled.
        """
        if self.listener is None:
            return
        fallback = getattr(self.config.telegram, "fallback_chat", None)
        if not fallback:
            self.listener.source_chats = None
            return
        self.listener.source_chats = {
            PRIMARY: self.config.telegram.chat or "",
            FALLBACK: fallback,
        }

    async def _supervise_listener(self, *, start_stream: bool = True) -> bool:
        """
        Connect the listener and start reading, immediately.

        Returns True when the read stream is live.

        A missing or rejected session is a recoverable state, not a fatal one:
        the engine stays up and retries, re-reading ``.env.local`` each time so
        that completing ``cli login`` is picked up with no restart.
        """
        if self.listener is None:
            return False
        self._last_listener_attempt = self.clock.now()
        self._refresh_telegram_config()
        # Wire the feeds HERE as well as in __init__: the production entrypoint
        # attaches the listener after construction, so __init__ saw None and the
        # fallback feed was never polled.
        self._wire_listener_sources()
        await self.listener.start()
        if self.listener.status.state != ConnectionState.CONNECTED:
            return False
        self._attach_nar_source()
        log("TELEGRAM", "listener armed")
        if start_stream and not (
            self._listener_task is not None and not self._listener_task.done()
        ):
            stop = self._stop_event
            if stop is None:
                stop = asyncio.Event()
                self._stop_event = stop
            self._listener_task = asyncio.create_task(self.listener.run(stop))
        return True

    def _attach_nar_source(self) -> None:
        """
        Give the NAR service the live Telethon client.

        One session, one connection: the narrative bot is queried over the same
        client the alert listener already uses, so no second login and no second
        socket. If the source is not the Telegram bot, nothing changes.
        """
        if not self.config.nar.is_bot or self.listener is None:
            return
        if self.listener._client is None:
            return
        from .nar.phanes import PhanesNarSource

        source = PhanesNarSource(
            clock=self.clock,
            client=self.listener._client,
            bot_username=self.config.nar.bot_username,
            command=self.config.nar.bot_command,
            timeout_s=self.config.nar.timeout_s,
            min_gap_s=self.config.nar.min_query_gap_s,
            fallback_bot_username=self.config.nar.fallback_bot_username,
            fallback_after_failures=self.config.nar.fallback_after_failures,
        )
        self.nar.attach(source)
        log(
            "NAR",
            "narrative source attached: @%s (%s <CA>) over the existing "
            "connection; fallback @%s after %d consecutive operational failures",
            self.config.nar.bot_username, self.config.nar.bot_command,
            self.config.nar.fallback_bot_username or "(none)",
            self.config.nar.fallback_after_failures,
        )

    async def _listener_supervisor(self, stop: asyncio.Event) -> None:
        """
        Keep the read stream alive.

        It only re-arms when the stream has actually FINISHED. A healthy stream
        is left strictly alone: re-anchoring on a live connection would take the
        newest message id at that moment and silently skip anything that arrived
        while the stream was down, which is exactly the alert the agent must not
        miss.
        """
        if self.listener is None:
            return
        backoff = self.config.telegram.reconnect_backoff_s
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=20.0)
                return
            except asyncio.TimeoutError:
                pass
            if self._listener_task is not None and not self._listener_task.done():
                continue
            if self._listener_task is not None:
                exc = self._listener_task.exception()
                log(
                    "TELEGRAM",
                    "read stream finished (error=%s); resuming from the last "
                    "message id we actually processed, not the newest in the chat",
                    repr(exc) if exc else "clean end",
                )
            if self.listener.status.state != ConnectionState.CONNECTED:
                log("TELEGRAM", "reconnecting (state=%s)", self.listener.status.state)
            else:
                self.listener.status.state = ConnectionState.DISCONNECTED
            # The anchor must not move forward on reconnect: resume from the
            # highest id already processed so nothing in the gap is lost.
            resume_from = self._highest_seen_message_id()
            if resume_from is not None:
                self.listener.status.since_message_id = resume_from
                log("TELEGRAM", "resuming after message id %s", resume_from)
            await self._sleep(backoff)
            if await self._supervise_listener(start_stream=True):
                backoff = self.config.telegram.reconnect_backoff_s
            else:
                backoff = min(
                    self.config.telegram.reconnect_backoff_max_s, backoff * 2
                )

    def _highest_seen_message_id(self) -> int | None:
        highest = None
        for chat, message_id in self._seen_messages:
            try:
                value = int(message_id)
            except (TypeError, ValueError):
                continue
            if highest is None or value > highest:
                highest = value
        return highest

    async def _monitor_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                self._service_control()
                await self.monitor_once()
            except Exception as exc:  # noqa: BLE001
                log("ERROR", "monitor cycle failed: %r", exc)
            if self._flatten_requested:
                await self.flatten()
                self._flatten_requested = False
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._monitor_interval())
            except asyncio.TimeoutError:
                pass

    def _monitor_interval(self) -> float:
        return (
            self.config.market.open_position_poll_interval_s
            if (self.account.open_positions()
                or self.real_account.open_positions())
            else self.config.market.poll_interval_s
        )

    async def _dashboard_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                self.render_dashboard()
            except Exception as exc:  # noqa: BLE001
                log("ERROR", "dashboard failed: %r", exc)
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self.config.dashboard_interval_s
                )
            except asyncio.TimeoutError:
                pass

    async def flatten(self) -> dict[str, Any]:
        """
        Explicit FLATTEN: close every open paper position now.

        Never happens implicitly on shutdown - a stopped process must not
        silently close positions.
        """
        # Keep the operator command and the Telegram kill switch on precisely
        # one implementation in production. The rollback flag is retained
        # solely for historical/offline reproducibility tests; startup safety
        # and the normal operator path leave it false.
        if getattr(self.config.real, "paper_legacy_mark_fill_rollback", False):
            book = self.open_book["positions"]
            now = self.clock.now()
            for position in list(self.account.open_positions()):
                price, _state = self.market.mark_price(position.mint, now)
                if price is None:
                    book.cancel(
                        position,
                        "legacy rollback flatten with no usable mark; "
                        "position left unvalued",
                        now,
                    )
                    continue
                decision = ExitDecision(
                    "EXIT", "FLATTEN command issued by operator", ExitStyle.FLATTEN,
                    trigger="flatten-legacy-rollback",
                )
                await self._do_exit(position, decision, now)
            return {
                "success": not self.account.open_positions(),
                "targeted": 0,
                "closed": 0,
                "failed": 0,
                "remaining": len(self.account.open_positions()),
                "details": [],
                "errors": [],
                "mode": "explicit_legacy_rollback",
            }
        result = await self.close_all_paper()
        if not result.get("success", False):
            log(
                "CONTROL",
                "FLATTEN completed with failures: closed=%d failed=%d remaining=%d",
                int(result.get("closed", 0) or 0),
                int(result.get("failed", 0) or 0),
                int(result.get("remaining", 0) or 0),
            )
        return result

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------

    def _last_event_for_display(self) -> dict:
        """
        The LAST EVENT row, from the authoritative last decision.

        ``self.last_decision`` is a DICT (``report.as_dict()``), not a
        ``DecisionReport``. Reading it with attribute access is what produced
        the live ``'dict' object has no attribute 'decided_at'`` crash: the
        dashboard is a display surface, so it must not care which of the two
        representations it was handed.

        Every field is read defensively. A dashboard that raises is caught by
        the dashboard loop and logged, but a display surface should degrade to
        a blank row rather than throw at all.
        """
        return _last_event_row(self.last_decision)

    def _pipeline_for_display(self) -> dict:
        """
        The CURRENT PIPELINE row: every stage's own duration, plus the reason a
        stage is empty.

        A stage with no end stamp used to render as a bare ``-``, which on
        2026-09-30 read exactly like a broken executor: ``quote=-``, ``fill=-``
        with nothing else. A ``-`` is now followed by the outcome, so the row
        says ``quote:- BUY_BLOCKED=stale_signal`` and the operator is told the
        executor was never reached rather than having to infer it.
        """
        rec = self._current_stages
        if rec is None:
            return {}
        blocked = rec.blocked_reason
        blocked_detail = " ".join(
            str(getattr(rec, "blocked_detail", "") or "").split())
        outcome = rec.outcome

        def cell(start: str, end: str) -> str:
            text = _ms_text(rec.stamps.get(end), rec.stamps.get(start))
            if text == "-" and blocked:
                base = f"{text} BUY_BLOCKED={blocked}"
                if blocked_detail:
                    # The per-source candidate table travels WITH the token,
                    # so the row never collapses to one undiagnosable code.
                    return f"{base}: {blocked_detail[:400]}"
                return base
            return text

        return {
            "alert": "ok",
            "data": _ms_text(rec.stamps.get("market_end"), rec.stamps.get("market_start")),
            "settle": _ms_text(rec.stamps.get("settle_end"), rec.stamps.get("settle_start")),
            "narrative": cell("narrative_start", "narrative_end"),
            "L25": _ms_text(rec.stamps.get("l25_end"), rec.stamps.get("l25_start")),
            "quote": cell("quote_start", "quote_end"),
            "fill": cell("submit_start", "fill"),
            "outcome": outcome or "-",
            "executed": str(self.last_execution.get("executed")) if self.last_execution else "-",
        }

    def _services_for_display(self) -> dict:
        listener = self.listener
        market_health = self.market.health()
        return {
            "channel": _connection_state_text(listener),
            # Read from the health dict's own `state`, which now exists. It did
            # not before, so `.get("state", "?")` returned "?" on every frame
            # and the market row could never report anything at all.
            "market": market_health.get("state", "?"),
            "market_detail": market_health,
            "narrative": ("disabled" if not self.nar.available
                          else ("ok" if self.nar.metrics.unavailable_rate < 0.2
                                else "degraded")),
            # The narrative's OWN measured latency, taken inside NarService where
            # only the narrative is running. A stage-recorder number can include
            # neighbouring work; this one cannot. The CONFIGURED budget travels
            # with it, so "measured 1.6s against a 2.5s budget" is one line and
            # neither number has to be looked up anywhere else.
            "narrative_source": {
                **self.nar.metrics.as_dict(),
                "timeout_s": self.config.nar.timeout_s,
                "timeout_grace_s": self.config.nar.timeout_grace_s,
                "worst_case_s": (
                    self.config.nar.timeout_s + self.config.nar.timeout_grace_s
                ),
            },
            "execution": "paper" if self.state.trade_mode is TradeMode.PAPER
                         else "REAL",
        }

    def _last_error_for_display(self) -> str:
        return str(self._last_error_text or "")

    def latency_stats(self) -> dict:
        """Trading and notification percentiles, reported SEPARATELY."""
        return self.latency.stats()

    def _merge_arm_snapshot(
        self, snapshot: dict, arm_rows: list[dict], arm_exposure: float
    ) -> dict:
        """
        Fold the execution-aware arms' open capital into the ACCOUNT block.

        The dashboard said "cash on the book; no open exposure" while three
        $0.50 simulated BUYs were open, because exposure was read from the
        legacy account the arms do not write to. The arms hold REAL committed
        capital, so that line has to include it.

        The arms keep their OWN bankrolls -- three independent $5 accounts, by
        design, so one arm cannot move another's money -- so the committed
        amount is reported as a separate, named figure rather than folded into
        the legacy cash balance. Merging two account bases into one number is
        what this codebase's accounting notes exist to prevent.
        """
        merged = dict(snapshot)
        committed = round(arm_exposure, 4)
        accounts = getattr(self._paper_bridge, "accounts", {}) or {}
        bankrolls = {
            arm: float(getattr(account, "bankroll_usd", 0.0) or 0.0)
            for arm, account in accounts.items()
        }
        arm_equity = sum(bankrolls.values())
        legacy_equity = float(snapshot.get("equity_usd") or 0.0)
        # BOTH books hold capital, so the denominator is BOTH books' equity.
        # Taking a total exposure over one book's equity would print a
        # percentage that no account could produce.
        total_equity = legacy_equity + arm_equity
        merged["execution_aware_paper"] = {
            "enabled": True,
            "open_positions": len(arm_rows),
            "committed_usd": committed,
            "arms_open": [r.get("arm") for r in arm_rows],
            "bankroll_usd": {k: round(v, 6) for k, v in bankrolls.items()},
            # Cash each arm can actually spend. Reported per arm because the
            # arms are independent accounts; a single summed cash figure would
            # imply they share a purse, which by design they do not.
            "cash_usd": {
                arm: round(float(getattr(account, "cash_usd", 0.0) or 0.0), 6)
                for arm, account in accounts.items()
            },
            "arm_equity_usd": round(arm_equity, 6),
            "legacy_equity_usd": round(legacy_equity, 6),
            "total_equity_usd": round(total_equity, 6),
        }
        # TOTAL exposure across BOTH PAPER books, so "no exposure" cannot be
        # printed over an open position.
        merged["open_exposure_usd"] = round(
            float(snapshot.get("open_exposure_usd") or 0.0) + committed, 4)
        merged["exposure_pct_equity"] = round(
            merged["open_exposure_usd"] / total_equity * 100.0
            if total_equity else 0.0, 2)
        merged["open_positions"] = int(
            snapshot.get("open_positions") or 0) + len(arm_rows)
        return merged

    def _dashboard_snapshot_for_test(self) -> dict:
        """
        The exact ACCOUNT snapshot ``render_dashboard`` builds.

        Exposed so the numbers on that screen can be asserted directly. The
        dashboard composes its snapshot inline, which is why "no exposure over
        an open position" was a rendering bug nobody could reproduce outside a
        terminal.
        """
        return self._dashboard_snapshot()

    def _dashboard_snapshot(self) -> dict:
        marks = self._marks if self._marks else self._refresh_marks()
        snapshot = self.account.snapshot(marks)
        arm_rows = self._execution_aware_open_rows(self.clock.now())
        if arm_rows:
            arm_exposure = sum(
                float(r.get("size_usd") or 0.0) for r in arm_rows)
            snapshot = self._merge_arm_snapshot(
                snapshot, arm_rows, arm_exposure)
        return snapshot

    def render_dashboard(self) -> None:
        marks = self._marks if self._marks else self._refresh_marks()
        snapshot = self.account.snapshot(marks)
        equity = snapshot["equity_usd"]
        self.account.update_peak(equity)
        stats = self._stats()
        # The execution-aware PAPER arms are the ACTIVE PAPER book, so their
        # positions and their committed capital belong on this screen. Reading
        # only the legacy account is what let a notified fill show as
        # "OPEN POSITIONS none / cash on the book / no exposure".
        arm_rows = self._execution_aware_open_rows(self.clock.now())
        if arm_rows:
            arm_exposure = sum(float(r.get("size_usd") or 0.0) for r in arm_rows)
            snapshot = self._merge_arm_snapshot(
                snapshot, arm_rows, arm_exposure)
        positions = [
            {
                "symbol": p.symbol,
                "return_pct": p.current_return_pct,
                "peak_return_pct": p.peak_return_pct,
                "mfe_pct": p.mfe_pct,
                "mae_pct": p.mae_pct,
                "runner_mode": p.runner_mode,
                "route": p.route.value,
                "exit_style": p.exit_style.value,
                "fill_price": p.fill_price,
                "quantity": p.quantity,
                "cash_out_usd": p.cash_out_usd,
                "mark_price": p.mark_price,
                "data_state": p.data_state.value,
                "holding_s": (self.clock.now() - p.entry_timestamp)
                if p.entry_timestamp else 0.0,
            }
            for p in self.account.open_positions()
        ]
        positions.extend(
            {
                "symbol": r.get("symbol"),
                "return_pct": r.get("pnl_pct"),
                "peak_return_pct": r.get("peak_return_pct"),
                "mfe_pct": r.get("mfe_pct"),
                "mae_pct": r.get("mae_pct"),
                "runner_mode": False,
                "route": "FAST_ENTRY",
                "exit_style": "FAST_FAILURE",
                "fill_price": r.get("fill_price"),
                "quantity": r.get("quantity"),
                "cash_out_usd": r.get("size_usd"),
                "mark_price": r.get("observed_mark"),
                "data_state": "LIVE",
                "holding_s": r.get("hold_s"),
                "arm": r.get("arm"),
            }
            for r in arm_rows
        )
        # NOTE: this used to build a full `dashboard.render(...)` string here
        # and assign it to `text`, which was then never read. It was dead code
        # that ran on every redraw, did real formatting work, and could throw
        # -- a latent crash in a display path. Removed deliberately.
        #
        # Redraw IN PLACE. The old behaviour appended a whole dashboard every
        # few seconds, so a long run scrolled unboundedly and buried whatever
        # was on screen. Logs go to the log FILE; the terminal is a status
        # surface, not a history.
        cols, _rows = dashboard.terminal_size()
        dashboard.write_live(
            dashboard.build_frame(
                strategy=l25.ACTIVE_STRATEGY,
                mode=self.state.trade_mode.value,
                run_state=self.state.run_state.value,
                uptime_s=self.clock.now() - self.started_at,
                account=snapshot,
                performance=self._stats(),
                reconciliation=self.reconciliation(),
                strategy_filter=(getattr(self.config.decision,
                                         "strategy_profile", None)
                                 or l25.ACTIVE_STRATEGY),
                last_event=self._last_event_for_display(),
                execution=self.last_execution or {},
                pipeline=self._pipeline_for_display(),
                latency=self.latency_stats(),
                services=self._services_for_display(),
                positions=positions,
                last_error=self._last_error_for_display(),
                cols=cols,
            )
        )
        # NOTE: the persist() that used to live here was display-only and ran on
        # every redraw, i.e. an fsync on the account state several times a minute
        # for no safety benefit. State is persisted on checkpoints and at
        # shutdown, which is where it belongs.

    def performance_by_scope(self) -> dict[str, Any]:
        """
        Performance for the CURRENT statistics epoch AND for the whole ledger.

        Two numbers, both true, previously collapsed into one. The paper account
        is a single continuous accumulator whose statistics epoch started at
        2026-09-30T06:39:38Z; the ledger holds 420 closed trades in total but
        only 2 of them fall inside that epoch. The dashboard showed 420 next to
        ``buys=0`` and ``open=0``, which reads as "420 trades happened and none
        of them are mine" rather than "418 are a previous account's history and
        this session has 2".

        Both are reported, each labelled with what it counts. NO historical
        record is touched, filtered away or rewritten: the epoch only chooses
        which rows a REPORT includes, and the all-time figure is always present
        beside it so nothing is hidden.
        """
        all_trades = list(self.memory.trades)
        epoch = self.account.state.stats_epoch_started_at
        scoped = all_trades
        if epoch is not None:
            scoped = [t for t in all_trades if (t.exit_timestamp or 0.0) >= epoch]
        active_trades = scoped
        if self._active_accounting_since is not None:
            active_trades = [
                t for t in all_trades
                if (t.exit_timestamp or 0.0) >= self._active_accounting_since
            ]
        now = self.clock.now()
        equity = self.account.equity(self._marks)
        bankroll = self.account.state.starting_bankroll_usd

        def block(trades, *, w_l: bool) -> dict[str, Any]:
            s = compute_stats(
                trades, starting_equity=bankroll, current_equity=equity
            ).as_dict()
            out = {"total_trades": int(s.get("total_trades", 0)),
                   "wins": int(s.get("wins", 0)),
                   "losses": int(s.get("losses", 0)),
                   "win_rate": float(s.get("win_rate", 0.0)),
                   "profit_factor": s.get("profit_factor"),
                   "expectancy_usd": float(s.get("expectancy_usd", 0.0)),
                   "realized_pnl_usd": float(s.get("realized_pnl_usd", 0.0)),
                   "runners": int(s.get("runners", 0))}
            if w_l:
                out["by_route"] = dict(s.get("by_route") or {})
                out["max_drawdown_pct"] = float(s.get("max_drawdown_pct", 0.0))
                out["max_drawdown_scope"] = str(s.get("max_drawdown_scope",
                                                      "realized_trade_to_trade"))
                out["avg_holding_s"] = float(s.get("avg_holding_s", 0.0))
                # Renamed: this is runners' dollars / winner dollars. It is NOT
                # MFE capture and is no longer displayed under that name.
                out["runner_pnl_share_pct"] = s.get("runner_pnl_share_pct")
                out["runner_capture_pct"] = s.get("runner_capture_pct")  # deprecated alias
                # True MFE capture, mark-based, None when not computable.
                out["mfe_capture_pct"] = s.get("mfe_capture_pct")
                out["mfe_capture_population"] = int(s.get("mfe_capture_population", 0))
                out["mfe_capture_basis"] = s.get("mfe_capture_basis", "")
                out["mixed_strategy_population"] = bool(
                    s.get("mixed_strategy_population", False))
                out["strategy_profiles_present"] = list(
                    s.get("strategy_profiles_present") or [])
                out["by_strategy_profile"] = dict(s.get("by_strategy_profile") or {})
                out["avg_winner_usd"] = float(s.get("avg_winner_usd", 0.0))
                out["avg_loser_usd"] = float(s.get("avg_loser_usd", 0.0))
                out["avg_mfe_pct"] = float(s.get("avg_mfe_pct", 0.0))
                out["avg_mae_pct"] = float(s.get("avg_mae_pct", 0.0))
                out["avg_giveback_pct"] = float(s.get("avg_giveback_pct", 0.0))
                out["current_losing_streak"] = int(
                    s.get("current_losing_streak", 0)
                )
            return out

        return {
            "stats_epoch_started_at": epoch,
            "stats_epoch_iso": iso(epoch) if epoch else None,
            "epoch": block(active_trades, w_l=False),
            "active_detail": block(active_trades, w_l=True),
            "all_time": block(all_trades, w_l=False),
            "all_time_detail": block(all_trades, w_l=True),
            "historical_unreconciled": bool(
                self._historical_accounting_boundary.get("historical_unreconciled")
            ),
            "historical_trade_count": len(all_trades) - len(active_trades),
            "session": self.session_counters(),
            "as_of": now,
        }

    def session_counters(self) -> dict[str, Any]:
        """
        Counters for THIS PROCESS RUN only.

        Explicitly separate from every historical figure. ``buys`` counts paper
        fills performed by this process; the closed-trade totals count the
        ledger. Keeping them in one dict with one name each is what stops a
        screen from implying that "420 trades" and "buys=0" describe the same
        population.
        """
        return {
            "candidates_seen": self.candidates_seen,
            "buys": self.buys,
            "blocked": self.blocks,
            "blocked_by_reason": dict(self.blocked_reasons),
            "rejects": self.rejects,
            "rejects_by_reason": dict(self.reject_reasons),
            "waits": self.waits,
            "unfilled_attempts": self.account.unfilled_attempts,
            "open_positions": len(self.account.open_positions()),
            "started_at": self.started_at,
            "uptime_s": max(0.0, self.clock.now() - self.started_at),
            "run_count": int(self.account.state.run_count),
            "signal_source": self.signal_source.as_dict(),
            "signal_source_counters": dict(self._signal_source_counters),
        }

    def reconciliation(self) -> dict[str, Any]:
        """
        The canonical sequential PAPER equity curve, reconciled against the
        account book.

        This exists so the screen can never show a Total PnL and an Equity that
        contradict each other without saying so. The ledger figure is derived
        from the same closed-trade records the performance block reads, so when
        the account book disagrees the divergence is a real defect in the
        settlement path and is reported as one.
        """
        from .paper_accounting import reconcile
        trades = list(self.memory.trades)
        if self._active_accounting_since is not None:
            trades = [
                trade for trade in trades
                if (trade.exit_timestamp or 0.0) >= self._active_accounting_since
            ]
        rec = reconcile(
            trades,
            starting_equity=self.account.state.starting_bankroll_usd,
            account_book=self.account.snapshot(self._marks),
        )
        out = rec.as_dict()
        out["scope"] = (
            "ACTIVE_PRODUCTION"
            if self._active_accounting_since is not None
            else "ALL_PERSISTED_TRADES"
        )
        out["historical_unreconciled"] = bool(
            self._historical_accounting_boundary.get("historical_unreconciled")
        )
        out["historical_trade_count"] = (
            len(self.memory) - len(trades)
            if self._active_accounting_since is not None else 0
        )
        out["realized_pnl_usd"] = rec.realized_pnl_usd
        out["equity_curve_tail"] = [
            {"trade_id": p.trade_id, "equity_after_usd": round(p.equity_after_usd, 6),
             "drawdown_pct": round(p.drawdown_pct, 4),
             "strategy_profile": p.strategy_profile}
            for p in rec.curve[-10:]
        ]
        return out

    def _stats(self) -> dict[str, Any]:
        """
        Performance for the dashboard.

        Scoped to the current statistics epoch, with the all-time totals carried
        alongside under ``all_time``. It previously used the WHOLE ledger with no
        epoch filter, while ``/status`` used the epoch - so the terminal and the
        chat reported different numbers for the same account. Both now read the
        same scope and both show the epoch and the all-time figure.
        """
        perf = self.performance_by_scope()
        current_detail = perf.get("active_detail") or perf["all_time_detail"]
        return {**current_detail, **{
            "total_trades": perf["epoch"]["total_trades"],
            "wins": perf["epoch"]["wins"],
            "losses": perf["epoch"]["losses"],
            "win_rate": perf["epoch"]["win_rate"],
            "profit_factor": perf["epoch"]["profit_factor"],
            "expectancy_usd": perf["epoch"]["expectancy_usd"],
            "runners": perf["epoch"]["runners"],
        },
            "by_route": current_detail.get("by_route", {}),
            "stats_epoch_started_at": perf["stats_epoch_started_at"],
            "all_time": perf["all_time"],
            "session": perf["session"],
            "accounting_boundary": dict(self._historical_accounting_boundary),
        }

    def _print_ready(self) -> None:
        tl = self.listener.status.as_dict() if self.listener else {"state": "DISABLED"}
        nar = self.nar.health()
        equity = self.account.equity(self._marks)
        size = compute_position_size(
            equity, profile=self.state.strategy_profile
        )
        real = self.config.real
        lines = [
            "",
            "=" * 62,
            "                 MEMEBOT V3 READY",
            "=" * 62,
            f"AGENT ID            : {AGENT_ID}",
            f"STRATEGY            : {l25.describe()}",
            f"  (active)          : {l25.ACTIVE_STRATEGY}  "
            f"f_liq_to_mcap >= {l25.L25_MIN_LIQ_TO_MCAP:.4f}",
            f"  research status   : {l25.L25_RESEARCH_CLASSIFICATION} "
            f"({l25.L25_POST_DERIVATION_VALIDATION})",
            f"CANDIDATE SOURCE    : {self.config.telegram.CANDIDATE_SOURCE}  (only)",
            f"BANKROLL            : ${self.account.state.starting_bankroll_usd:.2f}"
            f"   (equity ${equity:.2f})",
            f"POSITION SIZING     : {describe_policy()}",
            f"NEXT POSITION       : ${size.position_usd:,.6f}",
            f"MAX OPEN POSITIONS  : {self.config.risk.max_simultaneous_positions}",
            f"RE-ENTRY            : {'ENABLED' if self.config.risk.reentry_enabled else 'DISABLED'}",
            "-" * 62,
            "REAL TRADING IS OFF.",
            f"  armed now          : {self.state.real_trading_enabled}",
            f"  env request        : {real.env_trading_enabled_request}  (IGNORED)",
            f"  wallet configured  : {real.wallet_configured}",
            f"  rpc configured     : {real.rpc_configured}",
            f"  missing            : {', '.join(real.missing_capabilities()) or 'none'}",
            "  arming requires a CONFIRMED TELEGRAM OPERATOR ACTION plus a",
            "  passing preflight. No env var, config value or CLI command can",
            "  do it. Press the REAL button yourself after reading the preflight.",
            "-" * 62,
            f"RUNNING             : {self.state.running}  ({self.state.run_state.value})",
            f"TRADE MODE          : {self.state.trade_mode.value}",
            f"EMERGENCY STOP      : {self.state.emergency_stop}",
            f"TELEGRAM            : {tl.get('state')}  chat={tl.get('chat')}",
            f"NAR SOURCE          : {'CONNECTED' if nar['available'] else 'NOT CONFIGURED'}"
            f"  ({nar['name']}; all NAR fields -> null)",
            f"MARKET DATA         : {self.config.market.primary}"
            f" / {self.config.market.fallback}",
            # The banner must name the execution model that will ACTUALLY run.
            # It previously advertised the legacy mark/slippage model while the
            # runtime filled from executable quotes, which is precisely the
            # "display says one thing, engine does another" defect an operator
            # cannot detect from the chat.
            f"PAPER EXECUTION     : "
            + ("EXECUTABLE QUOTES (same-market, simulated fills)"
               if self._execution_aware_paper()
               else "ENABLED (legacy mark-based simulation)"),
            f"RECOVERED           : {'yes' if self.recovered else 'no (new account)'}",
            f"STARTING CAPITAL    : ${self.account.state.starting_bankroll_usd:.2f}"
            " - never reset",
            "-" * 62,
            "NO TRADE HAPPENS YET. Press the Telegram start button to begin",
            "candidate processing; it runs in PAPER unless REAL is armed.",
            "-" * 62,
            ("PAPER EXECUTION MODEL (same-market executable quotes):"
             if self._execution_aware_paper()
             else "PAPER EXECUTION MODEL (LEGACY mark-based simulation -- ROLLBACK MODE):"),
            *([
                "  entry fill        : executable BUY quote on the exact order size",
                "  exit fill         : executable SELL quote on the PINNED entry venue",
                "  venue selection   : best executable output across candidate pools",
                "  entry guard       : refuse a BUY more than 10% above the observed mark",
                "  missing/stale/unidentified quote -> NO FILL, never the mark",
                "  realized PnL      : settlement between the two simulated fills",
            ] if self._execution_aware_paper() else [
                f"  entry fee         : {self.config.execution.entry_fee_bps:.0f} bps",
                f"  exit fee          : {self.config.execution.exit_fee_bps:.0f} bps",
                f"  base slippage     : {self.config.execution.entry_slippage_bps:.0f} bps"
                " + sqrt(notional/liquidity) impact + staleness penalty",
                f"  decision latency  : {self.config.execution.decision_latency_ms:.0f} ms"
                " (fill uses the price available AFTER this delay, at the worse of"
                " quote/print)",
                f"  quote validity    : {self.config.execution.max_quote_age_ms:.0f} ms"
                " max; otherwise the trade is recorded UNFILLED",
            ]),
            "=" * 62,
            "",
        ]
        if tl.get("state") == "AWAITING_LOGIN":
            lines += [
                "  ACTION REQUIRED: TELEGRAM_SESSION is empty in .env.local.",
                "  Run this once in another terminal:",
                "      python -m autopaper.cli login",
                "  It prints a session string; put it in .env.local as",
                "  TELEGRAM_SESSION=<string>. This engine is running and will",
                "  pick it up automatically on its next retry.",
                "",
            ]
        lines.append("  Waiting for the operator to press the start button.")
        lines.append("")
        print("\n".join(lines), flush=True)

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------

    async def _drain_inflight_executions(self) -> None:
        """
        Give an in-flight BUY a bounded chance to FINISH before shutdown persists.

        A shutdown that cancelled an order mid-flight and then wrote the state
        file would leave ``last_execution = DECISION=BUY`` with no explanation -
        exactly the shape that made 2026-09-30 unreadable, except now it would
        be permanent and on disk. So:

        1. wait up to ``execution_drain_timeout_s`` for every candidate task;
        2. cancel whatever is still pending;
        3. record ``BUY_BLOCKED=shutdown_during_execution`` for anything that was
           cancelled while an order was in flight, so the outcome is named.

        Bounded on purpose: a shutdown must not hang on a wedged provider. And it
        never FLATTENs - a stop does not close positions, which is unchanged.
        """
        pending = [t for t in self._inflight_candidates if not t.done()]
        if not pending:
            return
        log("SYSTEM", "shutdown draining %d in-flight candidate task(s), "
                      "budget %.1fs", len(pending), EXECUTION_DRAIN_TIMEOUT_S)
        # Set for the WHOLE drain, wait + cancel + gather, so a task cancelled by
        # THIS drain records the accurate reason
        # (`shutdown_during_execution`) rather than a bare `cancelled`. A
        # cancellation from anywhere else is still `cancelled`.
        self._shutting_down = True
        try:
            _done, still_pending = await asyncio.wait(
                pending, timeout=EXECUTION_DRAIN_TIMEOUT_S
            )
            if not still_pending:
                log("SYSTEM",
                    "every in-flight execution completed before shutdown")
                return
            for t in still_pending:
                t.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)
        finally:
            self._shutting_down = False
        # Safety net: if the cancelled task recorded nothing, record it here, so
        # the drain can never end with an unexplained in-flight BUY.
        execu = self.last_execution or {}
        if execu.get("decision") == "BUY" and not execu.get("executed"):
            if execu.get("reason") == "shutdown_during_execution":
                # The cancelled task already recorded it; counting it twice would
                # make the reason distribution lie.
                return
            self._record_block(
                _BlockReport(
                    candidate_id=execu.get("candidate_id"),
                    mint=execu.get("mint"),
                    symbol=execu.get("symbol"),
                ),
                "shutdown_during_execution",
                "shutdown began while the order was in flight; the task was "
                "cancelled and no position was opened",
            )
            log("CONTROL",
                "an in-flight BUY was cancelled by shutdown and is recorded as "
                "BUY_BLOCKED=shutdown_during_execution; no position was opened")

    def persist(self) -> None:
        """
        The one durable account-state write path.

        Guarded by the single-writer boundary: an engine that does not own the
        account cannot rewrite it.  ``render_dashboard`` and ``shutdown`` are the
        only callers in the running engine, so this covers every checkpoint.
        """
        self._require_owner("persist()")
        payload = {
            "agent": AGENT_ID,
            "account_id": self._account_id,
            "state_path": str(self.config.state_file),
            "owner_pid": os.getpid(),
            "saved_at": self.clock.now(),
            "account": self.account.to_state().__dict__,
            # Persist the same reconciliation object the dashboard renders.
            # This makes an account/ledger divergence survive a restart as an
            # explicit finding instead of being reduced to two contradictory
            # numbers in separate files.
            "reconciliation": self.reconciliation(),
            "accounting_boundary": dict(self._historical_accounting_boundary),
            "telegram_listener": (
                self.listener.status.as_dict() if self.listener is not None else None
            ),
            "open_positions": [p.as_dict() for p in self.account.open_positions()],
            "recently_exited": self.account.recently_exited,
            "unfilled_attempts": self.account.unfilled_attempts,
            "first_valid_prices": self.market._first_valid_price,
            "seen_messages": [list(k) for k in sorted(self._seen_messages)],
            "seen_candidates": {k: v for k, v in self._seen_candidates.items()},
            "seen_opportunities": {
                mint: [[ts, source] for ts, source in entries]
                for mint, entries in self._seen_opportunities.items()
            },
            "signal_source": self.signal_source.as_dict(),
            "signal_source_counters": dict(self._signal_source_counters),
            "counters": {
                "candidates_seen": self.candidates_seen,
                "buys": self.buys,
                "blocked": self.blocks,
                "blocked_reasons": dict(self.blocked_reasons),
                "rejects": self.rejects,
                "waits": self.waits,
            },
            # Named explicitly so a reader of the state file cannot mistake the
            # session counters for account totals. They are deliberately NOT
            # restored on load: a counter describes one process run, and
            # resuming a "buys" number from a previous process would be a lie
            # about what this process did.
            "counters_scope": "this process run only; never restored on load",
            # The last BUY's outcome. Persisted so a BUY decision can never
            # survive a restart as an unexplained one: if the process died with
            # an order in flight, the state file says so in the same vocabulary
            # the operator sees on screen.
            "last_execution": self.last_execution,
            # REAL state is scoped separately and never summed with PAPER.
            "real": {
                "scope": "REAL on-chain account; never mixed with PAPER totals",
                "summary": self.real_account.as_dict(),
                "positions": [
                    p.to_state() for p in self.real_account.positions.values()
                ],
            },
        }
        save_json_atomic(self.config.state_file, payload)
        # The REAL account also gets its OWN file so reconciliation at startup
        # does not depend on the paper state file being parsed.
        self.save_real_state()

    async def shutdown(self) -> None:
        self._require_owner("shutdown()")
        self.running = False
        self._drain_control_writes()
        await self._drain_inflight_executions()
        marks = self._marks or self._refresh_marks()
        equity = self.account.equity(marks)
        self.account.update_peak(equity)
        self.account.state.last_heartbeat = self.clock.now()
        self.persist()
        if self.listener is not None:
            await self.listener.stop()
        if self._bot_task is not None and not self._bot_task.done():
            self._bot_task.cancel()
            try:
                await self._bot_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._bot_task = None
        # BOTH Amarih loops, or the process hangs on the scheduler: it is a
        # 1-second loop that only retires when ``stop`` is set or the runtime is
        # gone, neither of which shutdown would otherwise cause.
        self._stop_amarih_poller()
        self._stop_amarih_scheduler()
        if self.bot_notifier is not None:
            await self.bot_notifier.aclose()
        if self.bot_transport is not None:
            await self.bot_transport.aclose()
        open_n = len(self.account.open_positions())
        self.ledger.account(
            {
                **self.account.snapshot(marks),
                "shutdown_at": iso(self.clock.now()),
                "reason": self.control.command or "signal",
                # Session counters and historical totals travel together in one
                # record, so a reader of the ledger cannot mistake one for the
                # other later.
                "session": self.session_counters(),
                "closed_trades_total": len(self.memory),
                "closed_trades_since_epoch": (
                    self.performance_by_scope()["epoch"]["total_trades"]
                ),
                "stats_epoch_started_at": (
                    self.account.state.stats_epoch_started_at
                ),
                "last_execution": self.last_execution,
            }
        )
        perf = self.performance_by_scope()
        epoch = perf["stats_epoch_started_at"]
        # EXPLICIT LABELS. This line previously printed `closed=<len(memory)>`
        # beside `candidates=/buys=/rejects=/waits=` from this run only, so a
        # shutdown read "closed=420 ... buys=0" as one population and looked like
        # the executor had lost 420 trades. They are different populations and
        # are now named as such.
        log(
            "SYSTEM",
            "shutdown: equity=$%.2f cash=$%.2f realized=%+.2f open=%d | "
            "THIS RUN: candidates=%d buys=%d blocked=%d rejects=%d waits=%d | "
            "LEDGER: closed_total=%d closed_since_epoch(%s)=%d",
            equity, self.account.cash, self.account.realized, open_n,
            self.candidates_seen, self.buys, self.blocks, self.rejects,
            self.waits,
            perf["all_time"]["total_trades"],
            epoch and iso(epoch) or "all",
            perf["epoch"]["total_trades"],
        )
        if self.blocks:
            log("SYSTEM", "THIS RUN blocked BUYs by reason: %s",
                self.blocked_reasons or "(none)")
        if self.rejects:
            # The counterpart of the line above. A run that evaluated 45
            # candidates and bought none must be able to say, on the shutdown
            # screen, how many were refused by each gate -- without an
            # operator having to go and count journal rows by hand. It reads
            # the SAME per-candidate reasons the strategy already produced; it
            # does not introduce a new classification and it cannot change a
            # decision.
            log("SYSTEM", "THIS RUN rejected candidates by reason: %s",
                self.reject_reasons or "(none)")
        if open_n:
            log(
                "CONTROL",
                "%d paper position(s) left OPEN deliberately - a stop does not "
                "silently flatten. Use FLATTEN to close them.",
                open_n,
            )
        log("SYSTEM", "state persisted to %s", self.config.state_file)
        # Ownership is given up only after the final state write and the final
        # ledger append, so a restart can never observe a half-finished save.
        self._release_ownership()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _as_coroutine(value):
    return value


def _window_change(history, now: float, window_s: float) -> float | None:
    prices = [
        (ts, px) for ts, px in history.prices(now) if px and px > 0 and ts >= now - window_s
    ]
    if len(prices) < 2:
        # a single print inside the window tells us nothing about change
        return None
    first, last = prices[0][1], prices[-1][1]
    if first <= 0:
        return None
    return (last - first) / first * 100.0


def _positions_from(rows: list[dict[str, Any]]) -> dict[str, Position]:
    from dataclasses import fields as dc_fields

    out: dict[str, Position] = {}
    valid = {f.name: f for f in dc_fields(Position)}
    for row in rows:
        clean = {k: v for k, v in row.items() if k in valid}
        for key, typ in (
            ("status", PositionStatus),
            ("route", Route),
            ("regime", Regime),
            ("exit_style", ExitStyle),
            ("data_state", DataState),
        ):
            if isinstance(clean.get(key), str):
                clean[key] = typ(clean[key])
        for key in ("invalidation", "entry_conditions", "notes"):
            if isinstance(clean.get(key), list):
                clean[key] = tuple(clean[key])
        try:
            p = Position(**clean)
        except TypeError as exc:
            log("ERROR", "cannot restore position from snapshot: %s", exc)
            continue
        p.status = PositionStatus.OPEN
        out[p.position_id] = p
    return out


def _trade_from(row: dict[str, Any]) -> Trade | None:
    from dataclasses import fields as dc_fields

    valid = {f.name for f in dc_fields(Trade)}
    clean = {k: v for k, v in row.items() if k in valid}
    if "invalidations" in clean and "invalidation" not in clean:
        clean["invalidation"] = tuple(clean.pop("invalidations"))
    if isinstance(clean.get("invalidation"), list):
        clean["invalidation"] = tuple(clean["invalidation"])
    try:
        return Trade(**clean)
    except TypeError as exc:
        log("ERROR", "cannot restore trade from ledger: %s", exc)
        return None
