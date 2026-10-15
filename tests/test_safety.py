"""
The safety barrier for the V3 runtime.

The barrier has INVERTED since the V2/PAPER-only design. The thing that is now
structurally impossible is ML / Laya / shadow. The thing that is now
procedurally guarded is REAL arming, because the runtime is supposed to be able
to trade for real - just never by itself.

These tests cover both halves, plus the secret-hygiene rules that were
previously the only concern.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from autopaper import safety
from autopaper.safety import (
    AGENT_ID,
    FORBIDDEN_MODULES,
    SECRET_KEY_NAMES,
    SECRET_PATTERNS,
    STRATEGY,
    SafetyViolation,
    assert_real_trading_disabled_at_startup,
    audit,
    audit_loaded_modules,
    find_secrets_in_text,
    guard_url,
    is_forbidden_module,
    loaded_forbidden_modules,
    mask_address,
    redact_mapping,
    redact_value_container,
    real_env_requests,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.integrity,
    pytest.mark.safety,
    pytest.mark.strategy,
]

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "autopaper"


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_agent_identity():
    assert AGENT_ID == "MEMEBOT_V3"
    # The ACTIVE strategy. Migrated from V3-C to the frozen L25 candidate; see
    # tests/test_l25.py for the full gate behaviour. This is an identity
    # assertion, not a strategy-behaviour assertion.
    assert STRATEGY == "L25"


def test_the_audit_passes():
    report = audit()
    assert report.ok, report.render()


# ---------------------------------------------------------------------------
# layer 1: ML / Laya / shadow are unimportable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", ["xgboost", "laya", "laya_finetune", "sklearn", "shadow", "torch"]
)
def test_forbidden_module_import_is_blocked(module):
    with pytest.raises((SafetyViolation, ImportError, ModuleNotFoundError)):
        importlib.import_module(module)


def test_no_forbidden_symbols_in_loaded_modules():
    assert audit_loaded_modules() == []


def test_no_forbidden_module_is_loaded():
    assert loaded_forbidden_modules() == []


def test_forbidden_module_namespaces_are_blocked():
    """A submodule of a forbidden package is blocked too."""
    assert is_forbidden_module("xgboost.dask")
    assert is_forbidden_module("laya.finetune")
    assert is_forbidden_module("shadow.journal")


def test_the_guard_sits_at_the_front_of_meta_path():
    import autopaper  # noqa: F401

    assert isinstance(sys.meta_path[0], safety._ForbiddenImportFinder)


def test_import_guard_install_is_idempotent():
    safety.install_import_guard()
    safety.install_import_guard()
    assert sum(
        1 for f in sys.meta_path if isinstance(f, safety._ForbiddenImportFinder)
    ) == 1


# ---------------------------------------------------------------------------
# layer 2: the URL guard for read-only providers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://api.binance.com/api/v3/order",
        "https://api.exchange.coinbase.com/orders",
        "https://api.kraken.com/0/private/AddOrder",
        "https://quote-api.jup.ag/swap/v1/swap",
        "https://rpc.ankr.com/solana/sendTransaction",
        "https://api.binance.com/wallet/withdraw",
    ],
)
def test_guard_url_blocks_order_surfaces(url):
    with pytest.raises(SafetyViolation):
        guard_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.dexscreener.com/latest/dex/tokens/So111",
        "https://api.geckoterminal.com/api/v2/networks/solana/pools",
        "https://quote-api.jup.ag/swap/v1/quote?inputMint=So111",
    ],
)
def test_guard_url_allows_read_only_data_endpoints(url):
    guard_url(url)


def test_guard_url_allows_a_read_only_rpc_string_in_a_data_provider():
    """
    The data providers must not be able to reach an order surface, but a
    read-only URL of any kind is fine. The chain endpoints themselves are NOT
    blocked here - they belong to the execution layer, which builds its own
    clients and never calls this function.
    """
    guard_url("https://api.dexscreener.com/token-pairs/v1/solana/X")


# ---------------------------------------------------------------------------
# layer 3: REAL arming
# ---------------------------------------------------------------------------


def test_assert_real_trading_disabled_at_startup_passes():
    assert assert_real_trading_disabled_at_startup() is None


def test_a_fresh_runtime_state_is_never_armed():
    from autopaper.modes import RunState, RuntimeState

    s = RuntimeState()
    assert s.run_state is RunState.OFF
    assert s.real_trading_enabled is False


def test_a_non_authorizing_env_var_is_reported_but_ignored(monkeypatch):
    monkeypatch.setenv("REAL_TRADING_ENABLED", "true")
    reqs = real_env_requests()
    assert "REAL_TRADING_ENABLED" in reqs
    assert "ignored" in reqs["REAL_TRADING_ENABLED"].lower()


def test_a_non_authorizing_env_var_cannot_arm_real(monkeypatch):
    """
    The whole design in one test: set every non-authorizing variable the old
    design might have honoured, and REAL is still off.
    """
    from autopaper.modes import RunState, RuntimeState

    for name in safety.NON_AUTHORIZING_ENV:
        monkeypatch.setenv(name, "true")
    s = RuntimeState()
    s.start()
    assert s.real_trading_enabled is False
    assert s.run_state is RunState.PAPER
    assert s.trade_mode.value == "PAPER"


def test_real_trading_enabled_is_not_readable_from_config():
    """
    ``RealConfig.trading_enabled`` has no config or environment input. Reading
    it on a fresh config is always False, whatever the environment says.
    """
    from autopaper.config import RealConfig

    assert RealConfig().trading_enabled is False
    assert RealConfig(env_trading_enabled_request=True).trading_enabled is False


def test_the_env_request_is_kept_separately_from_the_capability():
    from autopaper.config import RealConfig

    cfg = RealConfig(env_trading_enabled_request=True, risk_acknowledged=True)
    d = cfg.as_dict()
    assert d["env_trading_enabled_request"] is True
    assert d["trading_enabled"] is False


# ---------------------------------------------------------------------------
# layer 4: secret hygiene
# ---------------------------------------------------------------------------


def test_mask_address_hides_the_middle():
    masked = mask_address("So11111111111111111111111111111111111111112")
    assert masked == "So11…1112"
    assert mask_address(masked) == masked, "masking must be idempotent"


@pytest.mark.parametrize(
    "key", ["api_hash", "bot_token", "private_key", "seed_phrase", "session"]
)
def test_secret_key_names_are_covered(key):
    assert key in SECRET_KEY_NAMES


def test_redact_mapping_masks_secrets_and_keeps_public_values():
    out = redact_mapping({
        "api_hash": "deadbeef",
        "seed": "words here",
        "public_address": "So11111111111111111111111111111111111111112",
        "lamports": 155_000,
    })
    assert out["api_hash"] == safety.REDACTED
    assert out["seed"] == safety.REDACTED
    assert out["public_address"].startswith("So11")
    assert out["lamports"] == 155_000


def test_redaction_is_decided_by_name_not_by_blinding_every_value():
    """
    ``redact_value(value)`` was removed.

    It took no NAME, so it could not do what its docstring claimed ("when its
    NAME looks like a secret") -- it took no name at all, and returned
    ``REDACTED`` for every non-``None`` value. It had no production caller.

    The contract that replaces it: a value is redacted because of the KEY it
    sits under, and an operational value is never blanked for being a value.
    """
    out = redact_mapping({
        "api_hash": "deadbeef",
        "public_address": "So11111111111111111111111111111111111111112",
        "lamports": 155_000,
        "notes": "",
        "flag": False,
        "count": 0,
    })
    assert out["api_hash"] == safety.REDACTED
    # Operational identifiers survive -- a journal that redacts these is a
    # journal that cannot be debugged.
    assert out["public_address"].startswith("So11")
    assert out["lamports"] == 155_000
    assert out["notes"] == ""
    assert out["flag"] is False
    assert out["count"] == 0


def test_redaction_reaches_into_lists_of_mappings():
    """
    ``redact_mapping`` recursed on ``dict`` only, so a secret inside a list of
    mappings was written verbatim. ``risk_flags`` is exactly that shape.
    """
    out = redact_mapping({
        "risk_flags": [
            {"code": "v3c_market_cap_over_ceiling", "severity": "block"},
            {"credential": "LEAKED_IN_A_LIST"},
        ],
    })
    flags = out["risk_flags"]
    assert flags[0]["code"] == "v3c_market_cap_over_ceiling"
    assert flags[1]["credential"] == safety.REDACTED
    assert "LEAKED_IN_A_LIST" not in json.dumps(out)


def test_redaction_reaches_into_tuples_and_keeps_their_type():
    assert redact_value_container(("a", "b")) == ("a", "b")
    assert isinstance(redact_value_container(("a",)), tuple)
    nested = redact_value_container({"rows": [{"bot_token": "LEAK"}]})
    assert nested["rows"][0]["bot_token"] == safety.REDACTED


def test_secret_patterns_detect_shapes_without_returning_values():
    samples = {
        "mnemonic": "alpha " * 12,
        "solana_private_key": "[" + ", ".join(["12"] * 64) + "]",
        "bearer": "Bearer " + "a" * 40,
    }
    for name, text in samples.items():
        hits = find_secrets_in_text(text)
        assert name in hits, name
        # the hits are NAMES, so no value can leak through this API
        for h in hits:
            assert h in {n for n, _ in SECRET_PATTERNS}


def test_a_clean_source_file_reports_no_secrets():
    src = (SRC / "sizing.py").read_text()
    assert find_secrets_in_text(src) == []


# ---------------------------------------------------------------------------
# the source tree
# ---------------------------------------------------------------------------


def _py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_source_tree_contains_no_ml_or_shadow_capability():
    """A grep-level backstop behind the runtime checks."""
    banned = re.compile(
        r"\b(xgboost|XGBClassifier|laya|LayaModel|shadow_trader|"
        r"shadow_journal|shadow_account|predict_proba)\b"
    )
    for f in _py_files():
        text = f.read_text()
        hits = banned.findall(text)
        # this file legitimately names them in FORBIDDEN_MODULES / docstrings
        if f.name in ("safety.py",):
            continue
        assert not hits, f"{f.name} references {hits}"


def test_the_only_model_or_shadow_names_are_in_the_guard():
    """Every ML/shadow name lives in safety.py, i.e. only in the thing that bans them."""
    for f in _py_files():
        if f.name == "safety.py":
            continue
        text = f.read_text()
        for word in ("xgboost", "laya", "shadow_journal", "shadow_account"):
            assert word not in text, f"{f.name} mentions {word}"


def test_package_exposes_no_ml_symbols():
    import autopaper
    import autopaper.analysis
    import autopaper.engine
    import autopaper.execution.real
    import autopaper.sizing
    import autopaper.telegram

    for mod in (autopaper, autopaper.engine, autopaper.sizing):
        for key in dir(mod):
            low = key.lower()
            banned = ("xgboost", "laya", "shadow")
            assert not any(b in low for b in banned), f"{mod.__name__}.{key}"


def test_no_env_file_contains_a_private_key():
    """
    The env file may hold credentials, but never key MATERIAL. The keypair is a
    separate file referenced by path.
    """
    env = REPO / ".env.local"
    if not env.exists():
        pytest.skip("no .env.local")
    text = env.read_text()
    assert find_secrets_in_text(text) == []
    for banned in ("PRIVATE_KEY", "SOLANA_PRIVATE_KEY", "SEED_PHRASE", "MNEMONIC"):
        for line in text.splitlines():
            if line.strip().startswith(banned):
                pytest.fail(f"{banned} is present in .env.local")


def test_config_refuses_a_private_key_in_the_env_file(tmp_path: Path, monkeypatch):
    """
    Key material in an environment file is refused. The wallet is loaded from a
    keypair FILE, and an env-var key would bypass the file permission boundary
    the executor depends on.
    """
    from autopaper.config import RealExecutionBlocked, build_config

    (tmp_path / ".env.local").write_text("PRIVATE_KEY=abc123\n")
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    with pytest.raises(RealExecutionBlocked, match="PRIVATE_KEY"):
        build_config(tmp_path)


def test_key_material_in_env_refuses_to_arm(monkeypatch):
    """The same rule applies to the process environment."""
    from autopaper.config import RealExecutionBlocked, build_config

    monkeypatch.setenv("SOLANA_PRIVATE_KEY", "abc")
    with pytest.raises(RealExecutionBlocked, match="SOLANA_PRIVATE_KEY"):
        build_config(REPO)


def test_env_local_is_gitignored():
    """Ask git, not the file: the pattern is what actually matters."""
    out = subprocess.run(
        ["git", "check-ignore", "-q", ".env.local"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, ".env.local is not ignored by git"


def test_the_env_patterns_cover_the_secret_file_types():
    ignore = (REPO / ".gitignore").read_text()
    for pattern in (".env", ".env.*", "*.session", "*keypair*.json"):
        assert pattern in ignore, pattern


def test_git_does_not_track_the_env_file():
    out = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env.local"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode != 0, ".env.local must not be tracked by git"


# ---------------------------------------------------------------------------
# the executor's capability boundary
# ---------------------------------------------------------------------------


def test_only_the_execution_layer_imports_a_chain_library():
    """
    The Solana library is imported lazily, inside the executor's functions, and
    nowhere else. A PAPER-only process therefore never needs it, and no data
    provider can reach it.
    """
    for f in _py_files():
        if f.parts[-2:] == ("execution", "solana.py"):
            continue
        text = f.read_text()
        # lazy `from solders...` inside a function is fine; a top-level one is not
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import solders", "from solders")):
                assert f.parts[-2] == "execution", f"{f.name}: {stripped}"


def test_the_read_only_providers_cannot_import_the_executor():
    for name in ("dexscreener.py", "geckoterminal.py", "base.py"):
        f = SRC / "market" / name
        text = f.read_text()
        assert "solders" not in text
        assert "execution" not in text.replace("execution/fill", "")


def test_no_secret_is_printed_by_the_preflight():
    """
    A gate reports a name and a reason, never a path, a URL or a value.
    """
    from autopaper.execution.real import RealExecutor

    class Cfg:
        wallet_keypair_path = "/home/operator/.config/solana/id.json"
        rpc_url = "https://secret-rpc.example/very-private-path"
        slippage_bps = 500
        max_price_impact_pct = 5.0
        max_open_positions = 1
        reentry_enabled = False

    ex = RealExecutor(Cfg(), backend=object())
    for c in ex.configuration_gates():
        assert "/home/operator" not in c.detail
        assert "secret-rpc" not in c.detail
        assert "id.json" not in c.detail


# ---------------------------------------------------------------------------
# a clean subprocess
# ---------------------------------------------------------------------------


def test_a_fresh_interpreter_audits_clean():
    code = (
        "import sys; sys.path.insert(0, 'src'); import autopaper;"
        "from autopaper import audit;"
        "print('OK' if audit().ok else 'FAIL')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO,
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("OK")


def test_the_package_imports_without_the_chain_library_installed():
    """
    A PAPER-only install must work without `solders`. The import is lazy, so the
    failure would be an ImportError at module import time if that regressed.
    """
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'solders':\n"
        "            raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.path.insert(0, 'src')\n"
        "import autopaper, autopaper.engine, autopaper.execution.real\n"
        "print('OK')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO,
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_the_environment_has_no_stray_real_credentials():
    """
    Nothing in the process environment should look like a credential the runtime
    would consume. Reported by NAME only.
    """
    suspicious = [
        k for k in os.environ
        if any(b in k.upper() for b in ("PRIVATE_KEY", "SEED_PHRASE", "MNEMONIC"))
    ]
    assert not suspicious, f"credential-shaped env vars present: {suspicious}"
