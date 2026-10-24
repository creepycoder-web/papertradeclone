"""
D3: a watchdog can tell a healthy bot from a dead one.

The exit code IS the interface, so these tests are mostly about codes. The
important asymmetry: a watchdog restarts on 2 only, so anything that cannot be
established must land on 2 (restart) or 1 (warn) and never on 0.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from autopaper.health import (
    DEGRADED,
    OK,
    STALE,
    assess,
    read_heartbeat,
    render,
    startup_checks,
    write_heartbeat,
)

pytestmark = pytest.mark.integrity

T0 = 1_700_000_000.0
THRESH = dict(degraded_s=90.0, stale_s=300.0, error_rate_degraded=0.25)


def beat(**kw):
    base = {
        "at": T0, "state": "PAPER", "open_positions": 1,
        "last_alert_at": T0 - 30.0, "last_market_fetch_at": T0 - 2.0,
        "provider_error_rate": 0.0,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def test_the_heartbeat_is_written_and_read_back(tmp_path):
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, beat())
    assert read_heartbeat(path) == beat()


def test_the_write_leaves_no_temp_files_behind(tmp_path):
    path = tmp_path / "heartbeat.json"
    for _ in range(3):
        write_heartbeat(path, beat())
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "heartbeat.json"]
    assert leftovers == [], leftovers


def test_a_write_failure_never_raises(tmp_path):
    """
    Called from the monitor tick. It must not be able to interrupt position
    management; a watchdog sees the resulting staleness, which is correct.
    """
    blocked = tmp_path / "ro"
    blocked.mkdir()
    os.chmod(blocked, 0o500)
    try:
        write_heartbeat(blocked / "heartbeat.json", beat())   # must not raise
    finally:
        os.chmod(blocked, 0o700)


def test_an_unparseable_heartbeat_reads_as_absent(tmp_path):
    path = tmp_path / "heartbeat.json"
    path.write_text("{not json")
    assert read_heartbeat(path) is None


# ---------------------------------------------------------------------------
# the codes
# ---------------------------------------------------------------------------


def test_a_fresh_heartbeat_is_ok():
    assert assess(beat(), now=T0 + 10, **THRESH).code == OK


def test_an_aging_heartbeat_is_degraded():
    assert assess(beat(), now=T0 + 120, **THRESH).code == DEGRADED


def test_an_old_heartbeat_is_stale():
    assert assess(beat(), now=T0 + 400, **THRESH).code == STALE


def test_a_missing_heartbeat_is_stale_not_ok():
    """The pessimistic reading: absent is indistinguishable from dead."""
    verdict = assess(None, now=T0, **THRESH)
    assert verdict.code == STALE
    assert any("MISSING" in line for line in verdict.lines)
    assert any("systemctl" in line for line in verdict.lines)


def test_a_high_provider_error_rate_is_degraded_not_stale():
    """
    Restarting on a provider wobble would turn it into a restart loop, so this
    must warn rather than ask for a restart.
    """
    verdict = assess(beat(provider_error_rate=0.6), now=T0 + 5, **THRESH)
    assert verdict.code == DEGRADED
    assert any("error rate" in line for line in verdict.lines)


def test_age_is_judged_before_the_error_rate():
    """
    A stale heartbeat makes every other field in it stale too, so reporting on
    them would be reporting on a bot that is not running.
    """
    verdict = assess(beat(provider_error_rate=0.9), now=T0 + 400, **THRESH)
    assert verdict.code == STALE


def test_a_heartbeat_with_no_timestamp_is_stale():
    assert assess(beat(at=None), now=T0, **THRESH).code == STALE


def test_the_verdict_reports_what_a_human_needs():
    lines = "\n".join(assess(beat(), now=T0 + 5, **THRESH).lines)
    for expected in ("heartbeat age", "state", "open positions",
                     "last alert", "last market fetch"):
        assert expected in lines


# ---------------------------------------------------------------------------
# startup checks
# ---------------------------------------------------------------------------


def test_startup_checks_cover_the_declared_set(config):
    names = {c.name for c in startup_checks(config)}
    assert names >= {
        "telegram.session", "telegram.chat", "safety.verify",
        "safety.isolation", "writable.var", "writable.logs",
        "disk.free", "secrets.env_local_mode",
    }


def test_every_failure_prints_an_exact_fix(config):
    for check in startup_checks(config):
        if not check.ok:
            assert check.fix, f"{check.name} failed with no fix"


def test_startup_checks_report_names_not_values(config):
    """
    A credential printed here would reach the terminal and, through systemd,
    the journal. The session check may say whether the KEY is set and nothing
    more.
    """
    rendered = render(startup_checks(config))
    assert "TELEGRAM_SESSION" in rendered or "TELEGRAM_SESSION is empty" in rendered
    assert "h" * 8 not in rendered          # the fixture's api_hash value
    session = getattr(config.telegram, "session", None)
    if session:
        assert session not in rendered


def test_a_group_readable_env_local_fails(config, tmp_path):
    env = config.root / ".env.local"
    env.write_text("TELEGRAM_CHAT=x\n")
    os.chmod(env, 0o644)
    check = _named(startup_checks(config), "secrets.env_local_mode")
    assert check.ok is False
    assert "chmod 600" in check.fix


def test_an_owner_only_env_local_passes(config):
    env = config.root / ".env.local"
    env.write_text("TELEGRAM_CHAT=x\n")
    os.chmod(env, stat.S_IRUSR | stat.S_IWUSR)
    assert _named(startup_checks(config), "secrets.env_local_mode").ok is True


def test_a_missing_env_local_is_a_failure_not_a_pass(config):
    """Absent is not safe: the bot cannot run without it."""
    env = config.root / ".env.local"
    if env.exists():
        env.unlink()
    check = _named(startup_checks(config), "secrets.env_local_mode")
    assert check.ok is False
    assert "missing" in check.detail


def test_the_isolation_check_is_reused_not_reimplemented(config):
    check = _named(startup_checks(config), "safety.isolation")
    assert check.ok is True, check.detail


def _named(checks, name):
    for check in checks:
        if check.name == name:
            return check
    raise AssertionError(f"no check named {name}")
