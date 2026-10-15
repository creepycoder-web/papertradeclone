"""
Terminal dashboard.

Renders a compact, readable status block to stderr. Non-TTY output drops the
redraw so logs stay readable in a file or a pipe.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any

from .clock import hms, iso
from .models import DataState, PositionStatus

CLEAR = "\033[H\033[2J" if sys.stderr.isatty() else ""
DIM = "\033[90m" if sys.stderr.isatty() else ""
BOLD = "\033[1m" if sys.stderr.isatty() else ""
RESET = "\033[0m" if sys.stderr.isatty() else ""

GREEN = "\033[32m" if sys.stderr.isatty() else ""
RED = "\033[31m" if sys.stderr.isatty() else ""
YELLOW = "\033[33m" if sys.stderr.isatty() else ""
CYAN = "\033[36m" if sys.stderr.isatty() else ""


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if colour else text


def _pct(value: float | None, width: int = 8) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:+{width}.1f}%"


def _usd(value: float | None) -> str:
    if value is None:
        return "$-"
    return f"${value:,.2f}"


def render(
    *,
    account_snapshot: dict[str, Any],
    stats: dict[str, Any],
    positions: list[dict[str, Any]],
    last_decision: dict[str, Any] | None,
    listener: dict[str, Any],
    nar: dict[str, Any],
    market: dict[str, Any],
    safety: dict[str, Any],
    mode: str,
    uptime_s: float,
    lessons: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    width = min(shutil.get_terminal_size((88, 24)).columns, 96)
    out: list[str] = []
    add = out.append

    add("")
    add(_c("=" * width, DIM))
    add(
        f"{_c('AUTONOMOUS PAPER AGENT V1', BOLD)}  "
        f"{_c('REAL EXECUTION: BLOCKED', RED)}  "
        f"{_c('PAPER ONLY', YELLOW)}   mode={mode}  up={_dur(uptime_s)}  {hms(_now())}"
    )
    add(_c("=" * width, DIM))

    # -- account ---------------------------------------------------------
    add("")
    add(_c("ACCOUNT", BOLD))
    eq = account_snapshot.get("equity_usd", 0.0)
    total = account_snapshot.get("total_pnl_usd", 0.0)
    realised = account_snapshot.get("realized_pnl_usd", 0.0)
    unreal = account_snapshot.get("unrealized_pnl_usd", 0.0)
    colour = GREEN if total >= 0 else RED
    add(
        f"  bankroll {_usd(account_snapshot.get('starting_bankroll_usd'))}"
        f"   equity {_c(_usd(eq), colour)}"
        f"   cash {_usd(account_snapshot.get('cash_usd'))}"
        f"   exposure {_usd(account_snapshot.get('open_exposure_usd'))}"
        f" ({account_snapshot.get('exposure_pct_equity', 0):.0f}%)"
    )
    add(
        f"  realized {_c(f'{realised:+.2f}', GREEN if realised >= 0 else RED)}"
        f"   unrealized {_c(f'{unreal:+.2f}', GREEN if unreal >= 0 else RED)}"
        f"   total {_c(f'{total:+.2f}', colour)}"
        f"   fees ${account_snapshot.get('fees_paid_usd', 0):.2f}"
        f"   slip ${account_snapshot.get('slippage_paid_usd', 0):.2f}"
    )
    add(
        f"  peak {_usd(account_snapshot.get('peak_equity_usd'))}"
        f"   drawdown {account_snapshot.get('current_drawdown_pct', 0):+.2f}%"
        f"   open={account_snapshot.get('open_positions', 0)}"
        f"   unfilled={account_snapshot.get('unfilled_attempts', 0)}"
        f"   identity_check={account_snapshot.get('identity_check', 0):.9f}"
    )
    # The execution-aware PAPER arms are the ACTIVE PAPER book. Their committed
    # capital is reported separately, because the arms are independent accounts
    # with their own bankrolls and folding them into one cash figure is exactly
    # the accounting blur the two bases exist to avoid.
    ea = account_snapshot.get("execution_aware_paper")
    if ea:
        arms = ", ".join(
            f"{arm} {_usd(value)}"
            for arm, value in (ea.get("bankroll_usd") or {}).items()
        )
        add(_c(
            f"  exec-aware PAPER  open={ea.get('open_positions', 0)}"
            f"   committed {_usd(ea.get('committed_usd'))}"
            + (f"   [{arms}]" if arms else ""), DIM))

        # -- V2 FLOOR (BETA): three isolated $10 ledgers, one gate ---------
        # Rendered ONLY from the experiment's own persisted records (the same
        # dict ``status()["v2_floor_beta"]`` carries). Internal identities stay
        # A/B/C; the group is the one selectable experiment. Win rate and
        # expectancy are NET of the modeled exit friction, and a sample under
        # five closed trades is labelled preliminary rather than presented as a
        # measurement.
        beta = ea.get("v2_floor_beta")
        if beta:
            add(_c(f"  {beta.get('label') or 'V2 FLOOR (BETA)'}", BOLD))
            for arm_name in ("A", "B", "C"):
                st = (beta.get("arms") or {}).get(arm_name) or {}
                n = int(st.get("completed_trades") or 0)
                wr = st.get("win_rate_pct")
                if wr is None:
                    wr_s = "n/a (no closed trades)"
                elif n < 5:
                    wr_s = f"{wr:.1f}% (preliminary, n={n})"
                else:
                    wr_s = f"{wr:.1f}%"
                exp = st.get("expectancy_pct")
                exp_s = "n/a" if exp is None else f"{exp:+.3f}%"
                pf = st.get("profit_factor")
                pf_s = "n/a" if pf is None else f"{pf:.2f}"
                add(
                    f"    {arm_name}: TP +{st.get('take_profit_pct'):g}% "
                    f"SL {st.get('stop_loss_pct'):g}% "
                    f"hold {st.get('max_hold_s'):g}s"
                    f"   bank {_usd(st.get('starting_bankroll_usd'))}"
                    f" -> {_usd(st.get('bankroll_usd'))}"
                    f"   realized {(st.get('realized_pnl_usd') or 0.0):+.4f}"
                    f"   unreal {(st.get('unrealized_pnl_usd') or 0.0):+.4f}"
                )
                add(
                    f"      trades {n}  WR {wr_s}  expectancy {exp_s}"
                    f"  PF {pf_s}  DD {st.get('max_drawdown_pct') if st.get('max_drawdown_pct') is not None else 'n/a'}%"
                    f"  open {st.get('open_positions', 0)}"
                    + (f"  unresolved {st.get('unresolved')}"
                       if st.get("unresolved") else "")
                )
                exits = st.get("exits") or {}
                add(_c(
                    f"      exits: TP {exits.get('TP', 0)}  "
                    f"SL {exits.get('SL', 0)}  "
                    f"TIME {exits.get('TIME', 0)}  "
                    f"ambiguous {exits.get('path_ambiguous', 0)}", DIM))
            add(_c(
                "    PAPER ONLY -- $10.00 isolated ledger per arm; nothing "
                "broadcast. Metrics are preliminary until n>=5.", DIM))

    # -- performance -----------------------------------------------------
    add("")
    add(_c("PAPER PERFORMANCE", BOLD))
    pf = stats.get("profit_factor")
    pf_s = "inf" if pf is None or pf == float("inf") else f"{pf:.2f}"
    add(
        f"  trades {stats.get('total_trades', 0)}"
        f"   W/L {stats.get('wins', 0)}/{stats.get('losses', 0)}"
        f"   win rate {stats.get('win_rate', 0) * 100:.1f}%"
        f"   PF {pf_s}"
        f"   expectancy {_c(f'{stats.get('expectancy_usd', 0.0):+.3f}', GREEN if stats.get('expectancy_usd', 0.0) >= 0 else RED)}"
    )
    # "maxDD" is REALISED, trade-to-trade drawdown: it cannot see an open
    # position that is underwater. It is labelled rather than left to be read as
    # an all-in risk number.
    add(
        f"  realized {stats.get('realized_pnl_usd', 0.0):+.2f}"
        f"   realizedDD {stats.get('max_drawdown_pct', 0):.1f}%"
        f"   avg hold {stats.get('avg_holding_s', 0):.0f}s"
        f"   runners {stats.get('runners', 0)}"
        f"   losing streak {stats.get('current_losing_streak', 0)}"
    )
    # Runner PnL SHARE is what the old, mislabelled "capture" number always was:
    # runners' dollars over total winner dollars. It says nothing about MFE.
    # True MFE capture is a separate, mark-based figure and may legitimately be
    # None, which is rendered as "n/a" and never as 0.0.
    _share = stats.get("runner_pnl_share_pct")
    _cap = stats.get("mfe_capture_pct")
    add(f"  runner PnL share {_pct_or_na(_share)}"
        f"   MFE capture (mark) {_pct_or_na(_cap)}")
    if stats.get("mixed_strategy_population"):
        add(_c(f"  MIXED STRATEGIES: {' + '.join(stats.get('strategy_profiles_present') or [])}"
               " -- pooled stats describe no single strategy", YELLOW))
    if stats.get("total_trades"):
        add(
            f"  avg winner {stats.get('avg_winner_usd', 0):+.2f}"
            f"   avg loser {stats.get('avg_loser_usd', 0):+.2f}"
            f"   avg MFE {stats.get('avg_mfe_pct', 0):+.1f}%"
            f"   avg MAE {stats.get('avg_mae_pct', 0):+.1f}%"
            f"   giveback {stats.get('avg_giveback_pct', 0):+.1f}%"
        )
        routes = stats.get("by_route") or {}
        if routes:
            line = "  by route: " + "  ".join(
                f"{k} n={v['n']} wr={v['win_rate'] * 100:.0f}% pnl={v['net_pnl_usd']:+.2f}"
                for k, v in list(routes.items())[:5]
            )
            add(line)
    else:
        add(_c("  no closed paper trades yet", DIM))

    # -- positions -------------------------------------------------------
    add("")
    add(_c("OPEN POSITIONS", BOLD))
    if not positions:
        add(_c("  none", DIM))
    for p in positions:
        tag = "RUNNER" if p.get("runner_mode") else (
            "UNCERTAIN" if p.get("data_state") == DataState.UNUSABLE.value else "WATCH"
        )
        tag_colour = CYAN if tag == "RUNNER" else (
            YELLOW if tag in ("WATCH", "UNCERTAIN") else DIM
        )
        add(
            f"  {p.get('symbol', '?')[:12]:<12} "
            f"{_pct(p.get('return_pct'), 7)}  "
            f"peak {_pct(p.get('peak_return_pct'), 7)}  "
            f"mfe {_pct(p.get('mfe_pct'), 7)}  mae {_pct(p.get('mae_pct'), 7)}  "
            f"{_c(f'{tag:<9}', tag_colour)} "
            f"{_c(p.get('route', '?') + '/' + p.get('exit_style', '?'), DIM)}"
        )
        mark = p.get("mark_price")
        detail = (
            f"fill ${p.get('fill_price', 0):.3e}  qty {p.get('quantity', 0):.6g}  "
            f"cost ${p.get('cash_out_usd', 0):.2f}  held {p.get('holding_s', 0):.0f}s  "
            f"mark {('$' + format(mark, '.3e')) if mark else 'n/a'} "
            f"({p.get('data_state', '?')})"
        )
        add(f"    {_c(detail, DIM)}")

    # -- last decision ---------------------------------------------------
    add("")
    add(_c("LAST DECISION", BOLD))
    if not last_decision:
        add(_c("  none yet - waiting for the first alert", DIM))
    else:
        d = last_decision
        colour = {"BUY": GREEN, "WAIT": YELLOW, "REJECT": RED}.get(d.get("decision", ""), "")
        add(
            f"  {(d.get('symbol') or d.get('mint') or '?')[:16]} "
            f"{_c(str(d.get('decision', '?')), colour + BOLD)}"
            f"   regime={d.get('regime')}  route={d.get('route')}"
            f"   sq={d.get('signal_quality', 0):.2f}"
            f"   conf={d.get('confidence', 0):.2f}  (decision strength, not P(profit))"
        )
        thesis = str(d.get("thesis", ""))
        add(f"    {_c(thesis[:width - 6], DIM)}")
        flags = d.get("risk_flags") or []
        if flags:
            add(
                "    "
                + _c(
                    "flags: "
                    + " ".join(
                        f"{f.get('code')}({f.get('severity', '?')})" for f in flags[:8]
                    ),
                    DIM,
                )
            )

    # -- services --------------------------------------------------------
    add("")
    add(_c("SERVICES", BOLD))
    tl = listener.get("state", "?")
    tl_colour = GREEN if tl == "CONNECTED" else (
        YELLOW if tl in ("AWAITING_LOGIN", "BACKOFF", "CONNECTING") else RED
    )
    add(
        f"  Telegram: {_c(tl, tl_colour)}"
        f"  chat={listener.get('chat')}"
        f"  msgs={listener.get('messages_accepted', 0)}"
        f"  dupes={listener.get('duplicates_dropped', 0)}"
        f"  backlog={listener.get('backlog_mode')}"
    )
    if listener.get("detail"):
        add(f"    {_c(str(listener['detail'])[:width - 6], DIM)}")
    nar_state = "CONNECTED" if nar.get("available") else "NOT CONFIGURED"
    nar_colour = GREEN if nar.get("available") else YELLOW
    add(
        f"  NAR source: {_c(nar_state, nar_colour)}  name={nar.get('name')}"
        f"  unavailable_responses={nar.get('unavailable_responses', 0)}"
    )
    mstate = "OK" if market.get("last_success_at") else (
        "DEGRADED" if market.get("errors") else "IDLE"
    )
    m_colour = GREEN if mstate == "OK" else (YELLOW if mstate == "DEGRADED" else DIM)
    add(
        f"  Market data: {_c(mstate, m_colour)}  {market.get('primary')}"
        f"/{market.get('fallback')}"
        f"  requests={market.get('requests', 0)}"
        f"  errors={market.get('errors', 0)}"
        f"  429s={market.get('rate_limited', 0)}"
    )
    add(
        f"  Paper execution: {_c('ENABLED', GREEN)}   "
        f"Real execution: {_c('BLOCKED', RED + BOLD)}"
    )
    add(
        f"  Wallet: NONE   keys: NONE   RPC writes: NONE   "
        f"swap/order routers: NONE   import guard: "
        f"{_c('ACTIVE' if safety.get('import_guard_installed') else 'INACTIVE', GREEN)}"
    )

    if lessons:
        add("")
        add(_c("PAPER MEMORY (time-correct)", BOLD))
        for line in lessons[:6]:
            add(f"  {_c(line, DIM)}")

    if extra:
        add("")
        for k, v in extra.items():
            add(f"  {k}: {v}")

    add(_c("=" * width, DIM))
    return CLEAR + "\n".join(out)


def _pct_or_na(value) -> str:
    """A percentage that may legitimately be UNKNOWN.

    ``None`` means "not computable from the data held", which is a different
    fact from 0%. Rendering the two identically is how a missing MFE becomes a
    confident-looking "0.0%" on a screen.
    """
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num_or_na(value, suffix: str = "%", fmt: str = "+.2f") -> str:
    """Same rule as :func:`_pct_or_na` for a plain number with a unit."""
    if value is None:
        return "n/a"
    try:
        return format(float(value), fmt) + suffix
    except (TypeError, ValueError):
        return "n/a"


def _dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def _now() -> float:
    import time

    return time.time()


def write(text: str) -> None:
    """
    Append a dashboard.

    Retained for tests and for the non-TTY case, where an in-place redraw would
    produce no output at all. The live loop uses :func:`write_live`.
    """
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


# --- ANSI, no external UI dependency -------------------------------------

_ANSI_ESC = "\x1b["
_CLEAR_SCREEN = _ANSI_ESC + "2J"
_CURSOR_HOME = _ANSI_ESC + "H"
_HIDE_CURSOR = _ANSI_ESC + "?25l"
_SHOW_CURSOR = _ANSI_ESC + "?25h"


def clear() -> str:
    """Clear the screen and home the cursor."""
    return _CLEAR_SCREEN + _CURSOR_HOME


def terminal_size(default_cols: int = 78, default_rows: int = 40) -> tuple[int, int]:
    """
    Current terminal size, or a sane default.

    shutil is stdlib, so this is a graceful degradation rather than a
    dependency. Never raises: a non-TTY (piped output, CI) falls back.
    """
    try:
        import shutil

        size = shutil.get_terminal_size(fallback=(default_cols, default_rows))
        return max(20, int(size.columns)), max(10, int(size.lines))
    except Exception:  # noqa: BLE001
        return default_cols, default_rows


def is_tty(stream=None) -> bool:
    """Whether we can safely redraw in place."""
    try:
        return bool((stream or sys.stderr).isatty())
    except Exception:  # noqa: BLE001
        return False


def write_live(text: str, stream=None) -> None:
    """
    Redraw a single screen in place.

    The terminal is cleared and the frame is rewritten, so repeated updates
    REPLACE rather than append and the scrollback does not grow without bound.
    Falls back to appending when the stream is not a TTY, so piped output is
    still readable.
    """
    stream = stream or sys.stderr
    if not is_tty(stream):
        stream.write(text + "\n")
        stream.flush()
        return
    stream.write(clear() + text.rstrip("\n") + "\n")
    stream.flush()


def hide_cursor(stream=None) -> None:
    if is_tty(stream):
        (stream or sys.stderr).write(_HIDE_CURSOR)
        (stream or sys.stderr).flush()


def show_cursor(stream=None) -> None:
    if is_tty(stream):
        (stream or sys.stderr).write(_SHOW_CURSOR)
        (stream or sys.stderr).flush()


def fit(text: str, cols: int) -> str:
    """
    Truncate every line to the terminal width.

    Without this a long token or error would wrap and push the frame past the
    screen bottom, which on some terminals re-wraps and corrupts the redraw.
    """
    cols = max(20, int(cols))
    out = []
    for line in text.splitlines():
        # count display width only as characters; good enough and never wider
        out.append(line if len(line) <= cols else line[: cols - 1] + "\u2026")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the live frame
# ---------------------------------------------------------------------------

#: One entry per pipeline stage, in pipeline order. ``settle`` is its own stage
#: because the alert settle plus the second market read used to be measured
#: inside the narrative window and were reported as narrative latency.
STAGES = ("alert", "data", "settle", "narrative", "L25", "quote", "fill")


def _pf_text(value) -> str:
    if value is None or value == float("inf"):
        return "inf"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _row(label: str, value: object, width: int = 26) -> str:
    return f"  {label:<{width}} {value}"


def _kv(label: str, value: str, cols: int, width: int = 26) -> list[str]:
    """
    A key/value row that WRAPS instead of being cut off.

    The budget lines are the ones an operator reads when something is slow, and a
    truncated one - ending in an ellipsis just before "worst call observed" -
    is worse than a wrapped one.
    """
    head = f"  {label:<{width}} "
    text = head + str(value)
    if len(text) <= cols:
        return [text]
    return _wrap(head + str(value), cols)


def _wrap(text: str, cols: int) -> list[str]:
    """
    Wrap a long line to the terminal width, breaking on spaces where possible.

    ``fit()`` TRUNCATES with an ellipsis, which is right for a token name and
    wrong for a pipeline row: the pipeline row's most important part - the
    outcome and the blocked reason - is at the END, so truncation removed
    exactly the content an operator needed during the 2026-09-30 incident.
    Nothing that must be read is dropped; it is moved to the next line.
    """
    import textwrap

    width = max(20, int(cols))
    if len(text) <= width:
        return [text]
    indent = " " * (len(text) - len(text.lstrip()))
    return textwrap.wrap(
        text, width=width, subsequent_indent=indent + "  ",
        break_long_words=False, break_on_hyphens=False,
    ) or [text]


def _pct_lines(label: str, stats: dict, cols: int) -> list[str]:
    """
    One latency line: p50/p95/p99/max plus the sample count.

    Wrapped rather than truncated. `fit()` cuts a long line with an ellipsis,
    and `max=` is at the END - so at 78 columns the worst observed latency, which
    is the number an operator is looking for during a tail-latency incident, was
    the part that disappeared.
    """
    if not stats or not stats.get("n"):
        return [_row(label, "(no samples)")]

    def f(k):
        v = stats.get(k)
        return "n/a" if v is None else f"{v:,.0f}"

    return _kv(label, f"p50={f('p50')}  p95={f('p95')}  p99={f('p99')}  "
                      f"max={f('max')}  n={stats.get('n')}", cols)


def _pct_line(label: str, stats: dict, cols: int) -> str:
    """Backwards-compatible single-line form. See :func:`_pct_lines`."""
    return "\n".join(_pct_lines(label, stats, cols))


def build_frame(
    *,
    strategy: str = "L25",
    mode: str = "PAPER",
    run_state: str = "OFF",
    uptime_s: float = 0.0,
    account: dict | None = None,
    performance: dict | None = None,
    reconciliation: dict | None = None,
    strategy_filter: str | None = None,
    last_event: dict | None = None,
    execution: dict | None = None,
    pipeline: dict | None = None,
    latency: dict | None = None,
    services: dict | None = None,
    positions: list | None = None,
    last_error: str = "",
    cols: int = 78,
) -> str:
    """
    The live screen: current state only.

    No historical log lines. Detailed chronology lives in the log FILE; this is
    a status surface, and mixing the two is what made the old append-only output
    unusable.
    """
    account = account or {}
    performance = performance or {}
    reconciliation = reconciliation or {}
    last_event = last_event or {}
    execution = execution or {}
    pipeline = pipeline or {}
    latency = latency or {}
    services = services or {}
    positions = positions or []

    rule = "=" * cols
    thin = "-" * cols
    out: list[str] = [rule, f" MEMEBOT {strategy}", thin]
    if strategy_filter and strategy_filter != "ALL":
        out.append(_row("SHOWING", strategy_filter))
        out.append(_c("  stats below are THIS profile only; not pooled with others", YELLOW))
    if reconciliation.get("drift_status") == "UNSETTLED_ACCOUNT":
        out.append(_c(
            f"  ACCOUNT NOT SETTLED: ledger ${float(reconciliation.get('realized_pnl_usd') or 0.0):+,.2f}"
            f" vs account book ${float((reconciliation.get('account_book_reported') or {}).get('realized_pnl_usd') or 0.0):+,.2f}"
            " -- showing the ledger figure", YELLOW))
    out.append(_row("MODE", mode))
    out.append(_row("STATUS", run_state))
    out.append(_row("UPTIME", _fmt_duration(uptime_s)))

    out += ["", " ACCOUNT", thin]
    for label, key, fmt in (
        ("equity", "equity_usd", "usd"),
        ("cash", "cash_usd", "usd"),
        ("exposure", "open_exposure_usd", "usd"),
        ("realized", "realized_pnl_usd", "usd"),
        ("unrealized", "unrealized_pnl_usd", "usd"),
    ):
        v = account.get(key)
        out.append(_row(label, _fmt_num(v, fmt)))
    dd = account.get("current_drawdown_pct")
    if dd is not None:
        out.append(_row("drawdown", f"{dd:+.2f}%"))

    out += ["", " PERFORMANCE", thin]
    # SCOPE IS STATED, NOT ASSUMED. `total_trades` counts the CURRENT statistics
    # epoch; `all_time` counts the whole ledger. The two were previously the same
    # number on one line, which is how a screen showed 420 closed trades beside
    # buys=0 / open=0 and read as a broken executor rather than as history from a
    # previous account.
    epoch = performance.get("stats_epoch_started_at")
    all_time = performance.get("all_time") or {}
    n = int(performance.get("total_trades") or 0)
    label = "since " + (iso(float(epoch)) if epoch else "account start")
    out.append(_row("trades", f"{n}   ({label})"))
    if n:
        out.append(_row("W/L", f"{performance.get('wins', 0)}/"
                               f"{performance.get('losses', 0)}"))
        out.append(_row("win rate", f"{float(performance.get('win_rate') or 0)*100:.1f}%"))
        pf = performance.get("profit_factor")
        out.append(_row("PF", "inf" if pf in (None, float("inf"))
                        else f"{float(pf):.2f}"))
        out.append(_row("expectancy", _fmt_num(performance.get("expectancy_usd"), "usd")))
        out.append(_row("runners", performance.get("runners", 0)))
        out.append(_row("runner PnL share", _pct_or_na(
            performance.get("runner_pnl_share_pct"))))
        out.append(_row("MFE capture (mark)", _pct_or_na(
            performance.get("mfe_capture_pct"))))
        out.append(_row("realized DD", f"{float(performance.get('max_drawdown_pct') or 0):.2f}%"))
    if performance.get("mixed_strategy_population"):
        out.append(_row("mixed strategies", " + ".join(
            performance.get("strategy_profiles_present") or [])))
    if all_time:
        out.append(_row("all-time", (
            f"trades={all_time.get('total_trades', 0)} "
            f"W/L={all_time.get('wins', 0)}/{all_time.get('losses', 0)} "
            f"PF={_pf_text(all_time.get('profit_factor'))} "
            f"pnl={_fmt_num(all_time.get('realized_pnl_usd'), 'usd')}")))
    sess = performance.get("session") or {}
    if sess:
        out.append(_row("this run", (
            f"candidates={sess.get('candidates_seen', 0)} "
            f"buys={sess.get('buys', 0)} "
            f"blocked={sess.get('blocked', 0)} "
            f"rejects={sess.get('rejects', 0)} "
            f"waits={sess.get('waits', 0)}")))

    out += ["", " LAST EVENT", thin]
    if last_event:
        out.append(_row("time", last_event.get("at_iso") or last_event.get("at") or "-"))
        out.append(_row("token", last_event.get("symbol") or last_event.get("mint") or "-"))
        out.append(_row("decision", last_event.get("decision") or "-"))
        out.append(_row("reason", last_event.get("reason") or "-"))
        lat = last_event.get("latency_ms")
        if isinstance(lat, (int, float)):
            out.append(_row("latency", f"{lat:,.0f} ms"))
    else:
        out.append(_row("(none yet)", "-"))

    # DECISION vs FILL. `decision=BUY` alone is not a trade, and this row is the
    # answer to "did it actually happen".
    if execution:
        out.append(_row("execution", str(execution.get("outcome") or "-")))
        if execution.get("executed") is not None:
            out.append(_row("executed", "yes" if execution["executed"] else "no"))
        if execution.get("reason"):
            out.append(_row("blocked_reason", str(execution["reason"])))
        if execution.get("position_id"):
            out.append(_row("position", str(execution["position_id"])))

    out += ["", " CURRENT PIPELINE", thin]
    # WRAPPED, NOT TRUNCATED. One long line was cut off by `fit()` with an
    # ellipsis, which silently removed the very thing the incident report exists
    # to surface: the outcome and the blocked reason at the end of the row.
    stage_line = "  " + " -> ".join(
        f"{s}:" + (str(pipeline.get(s)) if pipeline.get(s) is not None else "-")
        for s in STAGES)
    out.extend(_wrap(stage_line, cols))
    extra = [(k, v) for k, v in pipeline.items()
             if k not in STAGES and v not in (None, "-", "")]
    if extra:
        out.extend(_wrap("  " + "   ".join(f"{k}={v}" for k, v in extra), cols))

    out += ["", " LATENCY", thin]
    out.extend(_pct_lines("analysis    ", latency.get("analysis") or {}, cols))
    out.extend(_pct_lines("signal->fill", latency.get("signal_to_fill") or {}, cols))
    out.extend(_pct_lines("market      ", latency.get("market") or {}, cols))
    out.extend(_pct_lines("settle      ", latency.get("settle") or {}, cols))
    out.extend(_pct_lines("narrative   ", latency.get("narrative") or {}, cols))
    nar_src = services.get("narrative_source")
    if isinstance(nar_src, dict):
        worst = nar_src.get("worst_case_s")
        out.extend(_kv("narrative budget", (
            f"{_s(worst)} worst case (timeout {_s(nar_src.get('timeout_s'))} +"
            f" grace {_s(nar_src.get('timeout_grace_s'))})"
            if worst else "n/a"), cols))
        out.extend(_kv("narrative measured", (
            f"p50 {_ms(nar_src.get('latency_p50_ms'))}"
            f"  p95 {_ms(nar_src.get('latency_p95_ms'))}"
            f"  max {_ms(nar_src.get('latency_max_ms'))}"
            f"  n={nar_src.get('samples', 0)}"), cols))
        if nar_src.get("timeouts"):
            out.append(_row("narrative timeouts", nar_src["timeouts"]))
    mkt = services.get("market_detail")
    if isinstance(mkt, dict):
        worst_call = max((mkt.get("max_call_ms") or {}).values() or [0])
        out.extend(_kv("market budget", (
            f"{_ms(mkt.get('budget_ms'))} wall-clock"
            f" (connect {_ms(mkt.get('connect_budget_ms'))});"
            f" worst call observed {_ms(worst_call)}"), cols))
        dl = mkt.get("deadlines") or {}
        if dl.get("overruns"):
            out.append(_row("market overruns", (
                f"{dl['overruns']} of {dl.get('calls', 0)} call(s) hit the "
                "wall-clock deadline")))

    out += ["", " SERVICES", thin]
    for name in ("channel", "market", "narrative", "execution"):
        st = services.get(name)
        out.append(_row(name, "n/a" if st is None else st))

    if positions:
        out += ["", " OPEN POSITIONS", thin]
        for pos in positions[:6]:
            # `or 0` used to turn a MISSING MFE/MAE into a real-looking 0.0%.
            # A position that has never printed a mark, or whose telemetry did
            # not carry an extreme, must read "n/a" -- not "0.0".
            out.append(_row(
                str(pos.get("symbol") or "?")[:14],
                f"{_num_or_na(pos.get('return_pct'))}  "
                f"MFE {_num_or_na(pos.get('mfe_pct'), fmt='+.1f')}  "
                f"MAE {_num_or_na(pos.get('mae_pct'), fmt='+.1f')}"))

    if last_error:
        out += ["", " LAST ERROR", thin]
        out.append("  " + str(last_error)[: cols - 4])

    out.append(rule)
    return fit("\n".join(out), cols)


def _fmt_num(v, kind: str = "usd"):
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"${f:,.4f}" if kind == "usd" else f"{f:,.4f}"


def _ms(v) -> str:
    """A millisecond quantity, or an explicit dash. Never a fake precision."""
    if v is None or isinstance(v, bool):
        return "-"
    try:
        return f"{float(v):,.0f}ms"
    except (TypeError, ValueError):
        return "-"


def _s(v) -> str:
    """A second quantity, or an explicit dash."""
    if v is None or isinstance(v, bool):
        return "-"
    try:
        return f"{float(v):,.1f}s"
    except (TypeError, ValueError):
        return "-"


def _fmt_duration(seconds: float) -> str:
    try:
        s = int(max(0.0, float(seconds)))
    except (TypeError, ValueError):
        return "n/a"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"
