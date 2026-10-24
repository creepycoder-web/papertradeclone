"""
Heartbeat and health, so an unattended run can be watched by cron or systemd.

WHY A FILE AND NOT A PORT
-------------------------
The paper bot must not open a listening socket: that is new attack surface on a
machine whose whole value is that it cannot reach real money. A heartbeat file
is readable by a timer unit, a cron job and a human with ``cat``, needs no
network, and survives the process dying -- which is exactly the case that has
to be detectable.

WRITTEN ATOMICALLY, BECAUSE THE READER IS A WATCHDOG
----------------------------------------------------
``write_heartbeat`` writes a temp file, fsyncs it, then ``os.replace``s it.
Without the fsync a crash can leave a zero-length file that parses as "no
heartbeat"; without the replace a reader can see a half-written one. Either
would make a watchdog restart a healthy bot, which is worse than not watching
at all.

EXIT CODES ARE THE INTERFACE
----------------------------
``0`` healthy, ``1`` degraded, ``2`` stale or missing. A watchdog restarts on
``2`` only: ``1`` means "still trading, but something is wrong", and restarting
on that would turn a provider wobble into a restart loop.

THE PESSIMISTIC READING
-----------------------
An absent, unreadable or unparseable heartbeat is ``2``, never ``0``. A missing
file is indistinguishable from a bot that died before writing one, so it is
reported as the worse of the two.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

HEARTBEAT_FILE = "heartbeat.json"

#: Exit codes. The watchdog restarts on STALE only.
OK = 0
DEGRADED = 1
STALE = 2

_MB = 1024 * 1024


def write_heartbeat(path: Path, payload: dict[str, Any]) -> None:
    """
    Write the heartbeat atomically. Never raises.

    Called from the monitor tick, so it is on the same cadence as position
    management: it must not be able to interrupt it. A failure to write a
    heartbeat is a monitoring problem, and a watchdog will see it as staleness,
    which is the correct outcome.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".heartbeat-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"), default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
    except Exception:  # noqa: BLE001 - monitoring must not raise into trading
        return


def read_heartbeat(path: Path) -> Optional[dict[str, Any]]:
    """The heartbeat, or ``None`` when it is absent or unreadable."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class Verdict:
    code: int
    lines: tuple[str, ...]

    @property
    def label(self) -> str:
        return {OK: "OK", DEGRADED: "DEGRADED", STALE: "STALE"}[self.code]


def assess(heartbeat: Optional[dict[str, Any]], *, now: float,
           degraded_s: float, stale_s: float,
           error_rate_degraded: float) -> Verdict:
    """
    Turn a heartbeat into an exit code and the lines a human needs.

    Age is judged first, because a stale heartbeat makes every other field in
    it stale too -- reporting "provider errors are fine" off a six-hour-old file
    would be reporting on a bot that is not running.
    """
    if heartbeat is None:
        return Verdict(STALE, (
            "heartbeat: MISSING or unreadable",
            "  the bot is not running, or has never written one",
            "  fix: systemctl status autopaper, then journalctl -u autopaper -n 50",
        ))
    try:
        written = float(heartbeat.get("at") or 0.0)
    except (TypeError, ValueError):
        written = 0.0
    age = max(0.0, float(now) - written)
    lines = [
        f"heartbeat age        {age:.0f}s",
        f"state                {heartbeat.get('state') or 'unknown'}",
        f"open positions       {heartbeat.get('open_positions')}",
        f"last alert           {_ago(heartbeat.get('last_alert_at'), now)}",
        f"last market fetch    {_ago(heartbeat.get('last_market_fetch_at'), now)}",
    ]
    if age >= stale_s:
        return Verdict(STALE, (*lines,
                               f"  STALE: older than {stale_s:.0f}s",
                               "  fix: systemctl restart autopaper"))
    if age >= degraded_s:
        return Verdict(DEGRADED, (*lines,
                                  f"  DEGRADED: older than {degraded_s:.0f}s"))
    rate = heartbeat.get("provider_error_rate")
    try:
        rate = float(rate) if rate is not None else None
    except (TypeError, ValueError):
        rate = None
    if rate is not None and rate >= error_rate_degraded:
        return Verdict(DEGRADED, (
            *lines,
            f"  DEGRADED: provider error rate {rate:.0%} >= "
            f"{error_rate_degraded:.0%}",
            "  fix: check network and the provider's status; the bot keeps "
            "running and marks data stale rather than guessing",
        ))
    return Verdict(OK, tuple(lines))


def _ago(value: Any, now: float) -> str:
    try:
        at = float(value)
    except (TypeError, ValueError):
        return "never"
    if at <= 0:
        return "never"
    return f"{max(0.0, now - at):.0f}s ago"


# ---------------------------------------------------------------------------
# startup checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""


def startup_checks(config: Any) -> list[Check]:
    """
    Everything that must be true before the bot is left alone.

    NAMES ONLY, NEVER VALUES. The Telegram check reports whether the session
    and chat KEYS are set; printing either would put a credential in a terminal
    and, via systemd, in the journal.
    """
    out: list[Check] = []
    root = Path(config.root)

    tg = config.telegram
    out.append(Check(
        "telegram.session", bool(getattr(tg, "has_session", False)),
        "TELEGRAM_SESSION is set" if getattr(tg, "has_session", False)
        else "TELEGRAM_SESSION is empty",
        "run `python -m autopaper.cli login` once, then put the printed "
        "TELEGRAM_SESSION=<value> into .env.local",
    ))
    out.append(Check(
        "telegram.chat", bool(getattr(tg, "chat", None)),
        "TELEGRAM_CHAT is set" if getattr(tg, "chat", None)
        else "TELEGRAM_CHAT is empty",
        "set TELEGRAM_CHAT=<channel name or id> in .env.local",
    ))

    try:
        from .safety import audit
        report = audit()
        out.append(Check("safety.verify", bool(report.ok),
                         "verify passes" if report.ok else "verify FAILED",
                         "run `python -m autopaper.cli verify` and read the "
                         "failing lines; do not start the bot until it passes"))
    except Exception as exc:  # noqa: BLE001
        out.append(Check("safety.verify", False, f"{type(exc).__name__}",
                         "run `python -m autopaper.cli verify`"))

    try:
        from .run_isolation import RunPathViolation, assert_run_path_isolated
        try:
            assert_run_path_isolated()
            out.append(Check("safety.isolation", True,
                             "no real-execution capability is installed"))
        except RunPathViolation as exc:
            out.append(Check("safety.isolation", False, str(exc)[:160],
                             "uninstall the capability distributions named "
                             "above, or run the bot in a virtualenv without "
                             "them; `run` refuses to start until then"))
    except Exception as exc:  # noqa: BLE001
        out.append(Check("safety.isolation", False, f"{type(exc).__name__}",
                         "investigate before starting the bot"))

    for label, directory in (("var", config.var_dir), ("logs", config.log_dir)):
        writable = _writable(Path(directory))
        out.append(Check(
            f"writable.{label}", writable,
            "writable" if writable else "NOT writable",
            f"mkdir -p {label} && chown $USER {label}",
        ))

    margin = float(getattr(config.observations, "max_mb", 0.0)) + \
        float(getattr(config.observations, "disk_margin_mb", 500.0))
    free_mb = _free_mb(Path(config.var_dir))
    enough = free_mb is not None and free_mb >= margin
    out.append(Check(
        "disk.free", bool(enough),
        f"{free_mb:.0f} MB free, {margin:.0f} MB needed"
        if free_mb is not None else "could not measure free space",
        f"free up disk, or lower OBS_MAX_MB (currently "
        f"{getattr(config.observations, 'max_mb', '?')} MB)",
    ))

    env_path = root / ".env.local"
    mode_ok, mode_detail = _env_mode(env_path)
    out.append(Check("secrets.env_local_mode", mode_ok, mode_detail,
                     f"chmod 600 {env_path.name}"))
    return out


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".health-probe"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _free_mb(directory: Path) -> Optional[float]:
    try:
        return shutil.disk_usage(str(directory)).free / _MB
    except OSError:
        return None


def _env_mode(path: Path) -> tuple[bool, str]:
    """
    Is ``.env.local`` unreadable by group and others?

    An ABSENT file is not a pass. The bot cannot run without it, so "missing"
    is reported as a failure with the same fix path rather than as "nothing to
    check" -- the pessimistic reading.
    """
    try:
        mode = path.stat().st_mode
    except OSError:
        return False, ".env.local is missing"
    leaky = bool(mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH
                         | stat.S_IWOTH))
    octal = oct(stat.S_IMODE(mode))[-3:]
    if leaky:
        return False, f"mode {octal} is readable beyond the owner"
    return True, f"mode {octal}"


def render(checks: list[Check]) -> str:
    lines = ["STARTUP CHECKS", "=" * 60]
    for check in checks:
        mark = "ok  " if check.ok else "FAIL"
        lines.append(f"[{mark}] {check.name:<26} {check.detail}")
        if not check.ok and check.fix:
            lines.append(f"         fix: {check.fix}")
    failed = [c for c in checks if not c.ok]
    lines.append("-" * 60)
    lines.append(f"{len(checks) - len(failed)}/{len(checks)} passed")
    return "\n".join(lines)


__all__ = [
    "DEGRADED", "HEARTBEAT_FILE", "OK", "STALE", "Check", "Verdict",
    "assess", "read_heartbeat", "render", "startup_checks", "write_heartbeat",
]
