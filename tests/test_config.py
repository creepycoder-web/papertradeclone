"""
Configuration architecture: source of truth, precedence, and PAPER-safe defaults.

Every test here is deterministic and depends on NO secret. ``.env.local`` is
never required: each test builds its own root in ``tmp_path``, so the suite
passes on a machine that has never had a Telegram session.
"""


from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autopaper.config import ENV_FILENAME, build_config
from autopaper.safety import STRATEGY, RealExecutionBlocked

pytestmark = pytest.mark.safety

REPO = Path(__file__).resolve().parents[1]


def _root(tmp_path: Path, body: str) -> Path:
    """A project root with a .env.local. No secrets, ever."""
    (tmp_path / ENV_FILENAME).write_text(body)
    return tmp_path


# ---------------------------------------------------------------------------
# .env.local is the source of truth
# ---------------------------------------------------------------------------


def test_env_local_is_read(tmp_path: Path):
    _root(tmp_path, "PAPER_START_BANKROLL_USD=250.5\n")
    cfg = build_config(tmp_path)
    assert cfg.risk.starting_bankroll_usd == 250.5


def test_config_source_reports_env_local(tmp_path: Path):
    _root(tmp_path, "PAPER_START_BANKROLL_USD=100\n")
    assert build_config(tmp_path).config_source == ENV_FILENAME


def test_config_source_reports_defaults_without_a_file(tmp_path: Path):
    assert build_config(tmp_path).config_source == "defaults"


def test_env_local_beats_the_process_environment(tmp_path: Path, monkeypatch):
    """
    The operator's local file is the source of truth.

    A stale exported variable must not silently shadow a value the operator set
    deliberately. The loader's own docstring always claimed the file was the
    source of truth; the implementation now matches it.
    """
    _root(tmp_path, "PAPER_START_BANKROLL_USD=123.45\nTELEGRAM_CHAT=-100LOCAL\n")
    monkeypatch.setenv("PAPER_START_BANKROLL_USD", "999.99")
    monkeypatch.setenv("TELEGRAM_CHAT", "-100FROMENV")
    cfg = build_config(tmp_path)
    assert cfg.risk.starting_bankroll_usd == 123.45
    assert cfg.telegram.chat == "-100LOCAL"


def test_environment_supplies_keys_the_file_omits(tmp_path: Path, monkeypatch):
    """Env is a FALLBACK, so CI and tests can still inject a value."""
    _root(tmp_path, "PAPER_START_BANKROLL_USD=50\n")
    monkeypatch.setenv("MARKET_MAX_RPM", "42")
    cfg = build_config(tmp_path)
    assert cfg.risk.starting_bankroll_usd == 50          # from the file
    assert cfg.market.max_requests_per_minute == 42      # from the environment


def test_built_in_default_is_the_last_resort(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MARKET_MAX_RPM", raising=False)
    cfg = build_config(tmp_path)
    assert cfg.market.max_requests_per_minute == 110


def test_origin_is_recorded_per_key(tmp_path: Path, monkeypatch):
    _root(tmp_path, "PAPER_START_BANKROLL_USD=10\n")
    monkeypatch.setenv("MARKET_MAX_RPM", "7")
    cfg = build_config(tmp_path)
    assert cfg.env_origin["PAPER_START_BANKROLL_USD"] == ENV_FILENAME
    assert cfg.env_origin["MARKET_MAX_RPM"] == "environment"
    assert cfg.env_origin["NAR_BOT"] == "default"


def test_dot_env_is_not_read(tmp_path: Path, monkeypatch):
    """
    There is no `.env` fallback.

    The architecture has exactly one file plus the environment. A stray `.env`
    must not become a hidden second source of truth.
    """
    (tmp_path / ".env").write_text("PAPER_START_BANKROLL_USD=777\n")
    assert build_config(tmp_path).risk.starting_bankroll_usd == 100.0


def test_no_runtime_code_hardcodes_the_env_filename():
    """
    The loader reads the constant; nothing else OPENS the file by name.

    Checked on executable code only. A docstring that tells an operator to edit
    ``.env.local`` is documentation, and a test that forbids it would force the
    code to stop explaining itself - the same distinction applied to the
    research-tree import test.
    """
    import tokenize

    for p in (REPO / "src/autopaper").rglob("*.py"):
        if p.name == "config.py":
            continue
        code: dict[int, list[str]] = {}
        with p.open("rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                if tok.type in (tokenize.NL, tokenize.NEWLINE,
                                tokenize.INDENT, tokenize.DEDENT):
                    continue
                code.setdefault(tok.start[0], []).append(tok.string)
        for lineno, toks in code.items():
            assert ENV_FILENAME not in " ".join(toks), (
                f"{p.name}:{lineno} opens the env file by name instead of using "
                f"config.ENV_FILENAME"
            )


# ---------------------------------------------------------------------------
# git hygiene
# ---------------------------------------------------------------------------


def test_env_local_is_git_ignored():
    out = subprocess.run(["git", "check-ignore", "-q", ENV_FILENAME],
                         cwd=REPO, capture_output=True)
    assert out.returncode == 0, f"{ENV_FILENAME} is not git-ignored"


def test_env_local_is_not_tracked():
    out = subprocess.run(["git", "ls-files", "--error-unmatch", ENV_FILENAME],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode != 0, f"{ENV_FILENAME} must not be tracked"


def test_env_example_exists_and_is_secret_free():
    """
    The safe template must exist, must be complete enough to copy, and must not
    contain a credential value.
    """
    ex = REPO / ".env.example"
    assert ex.exists(), ".env.example is missing"
    text = ex.read_text()
    from autopaper.safety import find_secrets_in_text

    assert find_secrets_in_text(text) == [], "template contains secret material"
    for required in ("TELEGRAM_API_ID", "TELEGRAM_CHAT", "WALLET_KEYPAIR_PATH",
                     "RPC_URL", "REAL_TRADING_ENABLED=false",
                     "STRATEGY_PROFILE=L25_CURRENT"):
        assert required in text, f"{required} missing from the template"
    # every assignment must be empty or a non-secret default
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        _k, v = line.split("=", 1)
        v = v.strip()
        assert v == "" or v in (
            "false", "true", "Phanes_bot", "/nar", "5", "3", "2",
            "15000", "45000", "110", "100.0", "85", "120", "1200", "500",
            "5.0", "50000", "90", "8000",
            # Raptor's hosted endpoint is public / no API key; the URL is not a
            # secret and is the safe default the operator copies.
            "https://raptor.solanatracker.io",
            # The default strategy profile. Not a secret; it is the value the
            # operator is meant to see, and it must be the DEFAULT so a fresh
            # copy of the template trades exactly what shipped.
            "L25_CURRENT",
        ), f"unexpected value in template: {line}"


# ---------------------------------------------------------------------------
# PAPER-safe defaults and REAL disabled
# ---------------------------------------------------------------------------


def test_paper_is_the_default_mode(tmp_path: Path):
    from autopaper.modes import RunState, RuntimeState

    cfg = build_config(tmp_path)
    s = RuntimeState()
    assert s.run_state is RunState.OFF
    assert s.trade_mode.value == "PAPER"
    s.start()
    assert s.trade_mode.value == "PAPER"
    assert cfg.real.trading_enabled is False


def test_real_trading_enabled_is_false_in_a_clean_root(tmp_path: Path):
    assert build_config(tmp_path).real.trading_enabled is False


def test_env_cannot_arm_real(tmp_path: Path, monkeypatch):
    """
    Even with every non-authorizing variable set, REAL stays off.

    The values in .env.local and the environment are recorded for visibility
    and authorize nothing.
    """
    (tmp_path / ENV_FILENAME).write_text(
        "REAL_TRADING_ENABLED=true\n"
        "I_UNDERSTAND_REAL_TRADING_RISKS=YES\n"
        "WALLET_KEYPAIR_PATH=/tmp/id.json\n"
        "RPC_URL=https://rpc.example\n"
    )
    for name in ("REAL_TRADING_ENABLED", "I_UNDERSTAND_REAL_TRADING_RISKS",
                 "MODE", "TRADING_MODE", "AUTO_ARM_REAL", "ENABLE_LIVE"):
        monkeypatch.setenv(name, "true")
    cfg = build_config(tmp_path)
    assert cfg.real.env_trading_enabled_request is True
    assert cfg.real.risk_acknowledged is True
    assert cfg.real.wallet_configured is True
    # the capability that matters is still False
    assert cfg.real.trading_enabled is False
    from autopaper.modes import RuntimeState

    s = RuntimeState()
    s.start()
    assert s.trade_mode.value == "PAPER"
    assert not s.real_trading_enabled


def test_paper_risk_defaults_are_safe(tmp_path: Path):
    r = build_config(tmp_path).risk
    assert r.reentry_enabled is False
    assert r.max_simultaneous_positions == 1
    assert r.max_total_exposure_pct_equity == 12.5
    assert r.max_position_pct_equity == 12.5


def test_reentry_has_no_configuration_override(tmp_path: Path, monkeypatch):
    """There is no switch. Setting one must not change anything."""
    (tmp_path / ENV_FILENAME).write_text("PAPER_REENTRY_ENABLED=true\n")
    monkeypatch.setenv("PAPER_REENTRY_ENABLED", "true")
    assert build_config(tmp_path).risk.reentry_enabled is False


def test_no_real_wallet_is_required_for_paper_startup(tmp_path: Path):
    """
    PAPER must start with no wallet and no RPC.

    A missing capability is reported, never required, on the PAPER path.
    """
    cfg = build_config(tmp_path)
    assert cfg.real.missing_capabilities()
    assert cfg.real.trading_enabled is False
    assert cfg.is_live_allowed is False or cfg.is_live_allowed is True
    from autopaper.engine import PaperTrader

    trader = PaperTrader(cfg)
    report = trader.startup()
    assert report["ok"] is True
    assert trader.state.run_state.value == "OFF"
    assert not trader.state.real_trading_enabled


# ---------------------------------------------------------------------------
# key material is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["PRIVATE_KEY", "SOLANA_PRIVATE_KEY",
                                 "SEED_PHRASE", "MNEMONIC"])
def test_key_material_in_env_file_refuses_to_start(tmp_path: Path, key,
                                                   monkeypatch):
    (tmp_path / ENV_FILENAME).write_text(f"{key}=not-a-real-key\n")
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(RealExecutionBlocked, match=key):
        build_config(tmp_path)


def test_key_material_in_environment_refuses_to_start(tmp_path: Path,
                                                     monkeypatch):
    monkeypatch.setenv("SOLANA_PRIVATE_KEY", "not-a-real-key")
    with pytest.raises(RealExecutionBlocked, match="SOLANA_PRIVATE_KEY"):
        build_config(tmp_path)


# ---------------------------------------------------------------------------
# the safe diagnostic
# ---------------------------------------------------------------------------


def test_config_source_command_reports_without_leaking(tmp_path: Path):
    """
    The diagnostic must be useful AND safe.

    Asserted by planting a recognisable fake secret in a temp root and proving
    the value never appears in the output.
    """
    root = _root(tmp_path, "TELEGRAM_API_HASH=" + "F" * 32 + "\n"
                             "TELEGRAM_SESSION=" + "S" * 40 + "\n")
    out = subprocess.run(
        [sys.executable, "-m", "autopaper.cli", "--root", str(root),
         "config-source", "--json"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "F" * 32 not in out.stdout
    assert "S" * 40 not in out.stdout
    assert ENV_FILENAME in out.stdout
    # key NAMES are reported
    assert "TELEGRAM_API_HASH" in out.stdout
    # and the safety booleans
    assert '"real_trading_enabled": false' in out.stdout


def test_config_source_json_shape(tmp_path: Path):
    _root(tmp_path, "PAPER_START_BANKROLL_USD=10\n")
    out = subprocess.run(
        [sys.executable, "-m", "autopaper.cli", "--root", str(tmp_path),
         "config-source", "--json"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["config_source"] == ENV_FILENAME
    assert data["precedence"] == [ENV_FILENAME, "environment", "default"]
    assert data["safety"]["real_trading_enabled"] is False
    assert data["safety"]["reentry_enabled"] is False
    assert data["strategy"]["active"] == STRATEGY
    assert data["strategy"]["l25_min_liq_to_mcap"] == 0.9647


# ---------------------------------------------------------------------------
# strategy thresholds are config-frozen
# ---------------------------------------------------------------------------


def test_strategy_thresholds_come_from_config_and_are_frozen(tmp_path: Path):
    cfg = build_config(tmp_path)
    assert cfg.decision.l25_min_liq_to_mcap == 0.9647
    assert cfg.decision.v3c_max_market_cap_usd == 20_000.0
    assert cfg.decision.v3c_min_signal_quality == 0.55
    assert cfg.decision.v2a_ceiling_usd == 25_372.71


@pytest.mark.parametrize("field,value", [
    ("l25_min_liq_to_mcap", 0.96),
    ("l25_min_liq_to_mcap", 0.97),
    ("v3c_max_market_cap_usd", 25_000.0),
    ("v3c_min_signal_quality", 0.5),
    ("v2a_ceiling_usd", 30_000.0),
])
def test_a_threshold_cannot_be_configured_away(field, value):
    from autopaper.config import DecisionConfig

    with pytest.raises(ValueError):
        DecisionConfig(**{field: value})


def test_thresholds_are_not_read_from_the_environment(tmp_path: Path,
                                                     monkeypatch):
    """The env file must not be able to move a frozen threshold."""
    (tmp_path / ENV_FILENAME).write_text(
        "L25_MIN_LIQ_TO_MCAP=0.5\nV3C_MAX_MARKET_CAP_USD=999999\n"
    )
    monkeypatch.setenv("L25_MIN_LIQ_TO_MCAP", "0.5")
    cfg = build_config(tmp_path)
    assert cfg.decision.l25_min_liq_to_mcap == 0.9647
    assert cfg.decision.v3c_max_market_cap_usd == 20_000.0


# ---------------------------------------------------------------------------
# no secret dependency
# ---------------------------------------------------------------------------


def test_tests_do_not_require_a_real_env_local(tmp_path):
    """
    The suite must pass on a machine that has never had a session.

    Asserted by building config from a root that has no .env.local at all and
    requiring only the PAPER-safe subset.
    """
    cfg = build_config(tmp_path)
    assert not (tmp_path / ENV_FILENAME).exists()
    assert cfg.real.trading_enabled is False
    assert cfg.telegram.has_session is False


def test_no_credential_value_is_ever_stored_on_the_config(tmp_path):
    """
    Config must not retain a resolved credential where a repr could leak it.

    ``env`` is retained for diagnostics, so the repr of every section must be
    redacted - that is what the telegram/nar sections already do.
    """
    root = _root(tmp_path, "TELEGRAM_API_HASH=" + "F" * 32 + "\n")
    cfg = build_config(root)
    rendered = repr(cfg.telegram.as_dict()) + repr(cfg.nar.as_dict())
    assert "F" * 32 not in rendered
