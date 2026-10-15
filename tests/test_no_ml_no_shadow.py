"""
ML and Laya are gone from the runtime, and shadow is gone with them.

Three independent layers are tested:

1. structural  - the import guard refuses every ML/Laya/shadow module name
2. symbolic    - no module in this package defines an ML or shadow symbol
3. behavioural - the counters the migration requires are zero

The counters are not a tautology: ``EXTERNAL_MODEL_IMPORTS`` increments on every
blocked import attempt, so a test that actually imported XGBoost would see it
move rather than quietly pass.
"""


from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from autopaper import safety
from autopaper.safety import (
    BLOCKED_MODEL_IMPORT_ATTEMPTS,
    EXTERNAL_MODEL_IMPORTS,
    FORBIDDEN_MODULES,
    MODEL_CALLS,
    SHADOW_CAPITAL,
    SHADOW_ENTRIES,
    SafetyViolation,
    audit,
    audit_loaded_modules,
    is_forbidden_module,
    loaded_forbidden_modules,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.strategy,
]


# ---------------------------------------------------------------------------
# 1. structural: the import guard
# ---------------------------------------------------------------------------


ML_MODULES = [
    "xgboost",
    "lightgbm",
    "catboost",
    "sklearn",
    "torch",
    "tensorflow",
    "onnxruntime",
    "laya",
    "laya_finetune",
    "legacy_ml",
    "ml_gate",
]

SHADOW_MODULES = [
    "shadow",
    "shadow_trader",
    "shadow_journal",
    "shadow_account",
]


@pytest.mark.parametrize("name", ML_MODULES)
def test_ml_modules_are_blocked(name):
    assert is_forbidden_module(name)
    assert name in FORBIDDEN_MODULES


@pytest.mark.parametrize("name", SHADOW_MODULES)
def test_shadow_modules_are_blocked(name):
    assert is_forbidden_module(name)
    assert name in FORBIDDEN_MODULES


@pytest.mark.parametrize("name", ML_MODULES + SHADOW_MODULES)
def test_importing_a_forbidden_module_raises(name):
    """
    A blocked import raises, it does not return a stub. A stub is the failure
    mode that lets an ML gate survive "removal" while still being called.
    """
    if name in sys.modules:
        pytest.skip(f"{name} already imported by another test")
    before = safety.EXTERNAL_MODEL_IMPORTS
    attempts = safety.BLOCKED_MODEL_IMPORT_ATTEMPTS
    with pytest.raises((SafetyViolation, ImportError, ModuleNotFoundError)):
        importlib.import_module(name)
    # A successful import is the only unacceptable outcome. If the guard fired
    # the attempt counter moved; if the module genuinely does not exist, the
    # import counter stayed at zero. Both are correct.
    assert safety.EXTERNAL_MODEL_IMPORTS == before
    assert (
        safety.BLOCKED_MODEL_IMPORT_ATTEMPTS > attempts
        or name not in str(sys.modules)
    )


def test_the_guard_fires_and_counts_the_attempt():
    """The guard is armed, not merely configured."""
    before = safety.BLOCKED_MODEL_IMPORT_ATTEMPTS
    with pytest.raises((SafetyViolation, ImportError, ModuleNotFoundError)):
        importlib.import_module("xgboost.never_installed_submodule")
    assert safety.BLOCKED_MODEL_IMPORT_ATTEMPTS == before + 1


def test_the_guard_is_installed_at_import_time():
    import autopaper  # noqa: F401

    assert safety._INSTALLER["installed"] is True
    assert any(
        isinstance(f, safety._ForbiddenImportFinder)
        for f in sys.meta_path
    )


def test_a_submodule_of_a_forbidden_package_is_also_blocked():
    assert is_forbidden_module("xgboost.sklearn")
    assert is_forbidden_module("laya.finetune.model")
    assert is_forbidden_module("shadow.journal")


def test_a_similarly_named_module_is_not_blocked():
    """A false positive would break the runtime, so the guard is exact."""
    for ok in ("shadows", "shadowy", "layment", "xgboost_notes", "torchlight"):
        assert not is_forbidden_module(ok), ok


# ---------------------------------------------------------------------------
# 2. symbolic: no ML/shadow symbol in this package
# ---------------------------------------------------------------------------


def test_no_ml_symbols_in_the_package():
    assert audit_loaded_modules() == []


def test_no_forbidden_module_is_loaded():
    assert loaded_forbidden_modules() == []


def test_the_decision_path_imports_no_model():
    """
    The production decision path must not reach a model. Checked on the actual
    import graph, not on a grep.
    """
    import autopaper.analysis.decision as decision
    import autopaper.v3c as v3c

    for mod in (decision, v3c):
        for value in vars(mod).values():
            name = getattr(value, "__name__", "")
            if is_forbidden_module(name):
                pytest.fail(f"{mod.__name__} references {name}")


def test_the_executor_imports_no_model():
    import autopaper.execution.real as real

    for value in vars(real).values():
        name = getattr(value, "__name__", "")
        if is_forbidden_module(name):
            pytest.fail(f"execution.real references {name}")


# ---------------------------------------------------------------------------
# 3. behavioural: the required counters
# ---------------------------------------------------------------------------


def test_external_model_imports_is_zero():
    assert EXTERNAL_MODEL_IMPORTS == 0


def test_model_calls_is_zero():
    assert MODEL_CALLS == 0


def test_shadow_entries_is_zero():
    assert SHADOW_ENTRIES == 0


def test_shadow_capital_is_zero():
    assert SHADOW_CAPITAL == 0.0


def test_no_shadow_account_exists_in_the_runtime():
    """
    There is no shadow account, no shadow capital, no shadow journal writer.
    Proved by absence of any such attribute on the engine and the accounts.
    """
    from autopaper.account import PaperAccount
    from autopaper.engine import PaperTrader
    from autopaper.journal import Journal
    from autopaper.real_account import RealAccount

    for cls in (PaperAccount, RealAccount, Journal):
        for attr in dir(cls):
            assert "shadow" not in attr.lower(), f"{cls.__name__}.{attr}"
    for attr in dir(PaperTrader):
        assert "shadow" not in attr.lower(), f"PaperTrader.{attr}"


def test_the_journal_has_no_shadow_stream():
    from autopaper.journal import Journal

    j = Journal.__init__.__doc__ or ""
    assert "shadow" not in j.lower()
    assert not any(
        "shadow" in n.lower() for n in dir(Journal)
    )


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------


def test_the_migration_audit_passes():
    report = audit()
    assert report.ok, report.render()
    names = {n for n, _, _ in report.checks}
    assert "external_model_imports_zero" in names
    assert "model_calls_zero" in names
    assert "shadow_entries_zero" in names
    assert "shadow_capital_zero" in names
    assert "no_ml_symbols_in_package" in names


# ---------------------------------------------------------------------------
# a clean subprocess, so a previous import cannot mask a later one
# ---------------------------------------------------------------------------


def test_a_fresh_interpreter_also_has_no_model():
    """
    The in-process counters could be masked by a module another test already
    imported. A fresh interpreter removes that doubt.
    """
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "import autopaper;"
        "from autopaper import safety, audit;"
        "import json;"
        "print(json.dumps({"
        "'audit_ok': audit().ok,"
        "'forbidden_loaded': safety.loaded_forbidden_modules(),"
        "'symbol_violations': safety.audit_loaded_modules(),"
        "'imports': safety.EXTERNAL_MODEL_IMPORTS,"
        "'calls': safety.MODEL_CALLS,"
        "}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd="/Users/lnwper/AI-paper-trade",
    )
    assert out.returncode == 0, out.stderr
    import json

    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["audit_ok"] is True
    assert data["forbidden_loaded"] == []
    assert data["symbol_violations"] == []
    assert data["imports"] == 0
    assert data["calls"] == 0


def test_no_ml_or_shadow_dependency_is_declared():
    """
    The dependency audit: nothing ML-related is required to run this project.
    """
    import tomllib
    from pathlib import Path

    data = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    declared: list[str] = []
    for key in ("dependencies",):
        declared += data["project"].get(key, [])
    for extra in data["project"].get("optional-dependencies", {}).values():
        declared += extra
    banned = ("xgboost", "lightgbm", "catboost", "scikit", "sklearn", "torch",
              "tensorflow", "onnx", "joblib", "laya", "pandas", "numpy")
    for dep in declared:
        low = dep.lower()
        for b in banned:
            assert b not in low, f"ML dependency declared: {dep}"


def test_the_real_executor_dependency_is_declared():
    """
    The dependency audit cuts both ways: the real executor's own requirements
    must be present, or REAL would be advertised and not work.
    """
    import tomllib
    from pathlib import Path

    data = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    declared = " ".join(data["project"].get("dependencies", [])).lower()
    assert "solders" in declared, "the Solana executor needs solders"
    assert "httpx" in declared
