"""
The integrity check must cover the trading tree, and must not cover secrets.

This is a test about the *checker*, not about the tree. Its purpose is to stop
the specific failure that produced a stale integrity list in the first place: a
hand-maintained file list silently omitting new trading modules, while reporting
a phantom failure for a file that no longer exists.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFY = REPO / "research/same_alert/verify_integrity.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_integrity", VERIFY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load()


def test_every_discovered_file_actually_exists():
    """No phantom entries -- the old list named a deleted file forever."""
    for rel in V._discover_ai_trading_files():
        assert (REPO / rel).exists(), rel


def test_every_production_module_is_covered():
    """
    A new trading file cannot go untracked.

    Compared against the filesystem, not against a list, so adding a module
    without updating anything still passes -- because discovery finds it.
    """
    discovered = set(V._discover_ai_trading_files())
    pkg = REPO / "src/autopaper"
    on_disk = {
        str(p.relative_to(REPO)).replace(os.sep, "/")
        for p in pkg.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert on_disk <= discovered, f"untracked: {sorted(on_disk - discovered)}"


def test_the_safety_critical_modules_are_all_covered():
    uncovered = [r for r in V.AI_REQUIRED_FILES if r not in V._discover_ai_trading_files()]
    assert not uncovered, uncovered


def test_the_required_floor_is_not_vacuous():
    """Guards the guard: a discovery bug must not silently pass."""
    assert len(V.AI_REQUIRED_FILES) >= 15
    for r in V.AI_REQUIRED_FILES:
        assert (REPO / r).exists(), f"required file does not exist: {r}"


@pytest.mark.parametrize("secret", [
    ".env.local", ".env", ".env.production",
    "id.json", "wallet.json", "keypair.json",
])
def test_secrets_are_never_hashed_into_the_report(secret):
    """
    A hash of a secret is a secret oracle, and these files change constantly.

    The integrity report is committed; a secret's hash must not be in it.
    """
    assert secret not in V._discover_ai_trading_files()
    src = VERIFY.read_text()
    # and the env file is never sha()'d anywhere in the checker
    for i, line in enumerate(src.splitlines(), 1):
        if "sha(" in line:
            assert ".env.local" not in line, f"line {i} hashes the env file"


def test_the_check_never_opens_the_env_file_for_hashing():
    """Reading .env.local for a NAME check is fine; hashing it is not."""
    src = VERIFY.read_text()
    assert 'open(ai_env).read()' in src or 'open(ai_env)' in src, (
        "the by-name secret check should still exist"
    )
    # the only sha() calls take a path variable, never the env path
    for line in src.splitlines():
        if "sha(" in line:
            assert "ai_env" not in line


def test_discovery_does_not_walk_pycache():
    files = V._discover_ai_trading_files()
    assert not any("__pycache__" in f for f in files)


def test_the_file_set_is_deterministic():
    a = V._discover_ai_trading_files()
    b = V._discover_ai_trading_files()
    assert a == b
    assert list(a) == sorted(a), "must be sorted for a stable report"


def test_the_old_stale_entries_are_gone():
    """Named explicitly so the regression is obvious in the diff."""
    src = VERIFY.read_text()
    assert '"src/autopaper/execution/sizing.py"' not in src
    assert '".env.local",' not in src.split("MB_TRADING_FILES")[0]
    # the file that actually holds sizing
    assert "src/autopaper/sizing.py" in V.AI_REQUIRED_FILES


def test_the_strategy_gate_is_in_the_integrity_set():
    """The frozen strategy itself must be under integrity control."""
    files = V._discover_ai_trading_files()
    for critical in ("src/autopaper/l25.py", "src/autopaper/v3c.py",
                     "src/autopaper/v2a.py", "src/autopaper/modes.py",
                     "src/autopaper/safety.py"):
        assert critical in files, critical
