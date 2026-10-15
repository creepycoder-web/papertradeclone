"""
The narrative budget must be 2.5s, and it must not silently become 12s.

THE INCONSISTENCY
-----------------
``NarConfig.timeout_s`` has a dataclass default of 2.5, but the config LOADER's
environment fallback was ``12.0``:

    timeout_s=_as_float(get("NAR_TIMEOUT_S"), 12.0)

Any deployment that did not set NAR_TIMEOUT_S in ``.env.local`` therefore ran
with a 12s narrative budget while the dataclass, the docs and every report said
2.5. The observed runtime used 12s.

THE FIX
-------
The fallback is now the same value as the dataclass default, and the configured
budget is passed explicitly into the narrative source so no source-level default
can reintroduce a 12s wait. The engine already adds only the configured grace.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from autopaper.config import NarConfig, build_config
from autopaper.nar.base import NarService
from autopaper.nar.phanes import PhanesNarSource

REPO = Path(__file__).resolve().parents[1]

TOTAL_BUDGET_ALLOWED_S = 3.0


@pytest.fixture
def no_env_file(monkeypatch):
    """Make the loader see NO .env.local and no NAR_TIMEOUT_S in the process."""
    import autopaper.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_env_file", lambda path: {})
    monkeypatch.delenv("NAR_TIMEOUT_S", raising=False)
    return monkeypatch


def test_dataclass_default_is_2_5():
    assert NarConfig().timeout_s == 2.5


def test_environment_fallback_is_now_2_5_not_12(no_env_file):
    """
    With no env file and no process variable, the fallback must be 2.5.

    This is the exact regression: the fallback used to be 12.0.
    """
    cfg = build_config(REPO)
    assert cfg.nar.timeout_s == 2.5


def test_env_local_remains_the_source_of_truth(no_env_file):
    """
    A value in ``.env.local`` still wins over the fallback.

    Precedence is unchanged: .env.local > process environment > default.
    """
    import autopaper.config as cfgmod

    no_env_file.setattr(
        cfgmod, "load_env_file", lambda path: {"NAR_TIMEOUT_S": "1.75"}
    )
    cfg = build_config(REPO)
    assert cfg.nar.timeout_s == 1.75


def test_total_narrative_wait_is_bounded():
    cfg = build_config(REPO)
    total = cfg.nar.timeout_s + cfg.nar.timeout_grace_s
    assert total <= TOTAL_BUDGET_ALLOWED_S, (
        f"narrative worst case is {total}s; the intended budget is 2.5s"
    )


def test_phanes_source_default_cannot_be_12_seconds():
    """No source-level default may reintroduce the 12s wait."""
    sig = inspect.signature(PhanesNarSource.__init__)
    assert sig.parameters["timeout_s"].default == 2.5


def test_nar_service_passes_the_configured_budget_to_the_bot():
    """
    A NarService with the bot source must install a source whose timeout is the
    CONFIG's, not the source class default.
    """
    cfg = NarConfig(source="phanes_bot", timeout_s=2.5, min_query_gap_s=4.0)
    svc = NarService(cfg)
    assert isinstance(svc.source, PhanesNarSource)
    assert svc.source.timeout_s == 2.5


def test_the_engine_does_not_wait_for_narrative_on_the_critical_path():
    from autopaper.engine import PaperTrader

    src = inspect.getsource(PaperTrader)
    assert "nar = await asyncio.wait_for(" not in src
    assert "NARRATIVE_PENDING: not complete at L25 decision" in src
    assert "nar_task.done()" in src


def test_no_12s_nar_fallback_remains_in_the_loader_source():
    """Structural: the literal cannot come back through another fallback."""
    src = (REPO / "src/autopaper/config.py").read_text()
    line = next(
        ln for ln in src.splitlines() if 'get("NAR_TIMEOUT_S")' in ln
    )
    assert "12.0" not in line
    assert "NarConfig.timeout_s" in line or "2.5" in line
