"""
REAL safety architecture, verified against the SOURCE rather than a report.

The properties asserted here are the ones that must hold even if a future
refactor is careless. Each is stated as a property of the code, and several are
checked by calling the real code paths rather than by reading them.
"""


from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from autopaper import modes
from autopaper.modes import (
    REAL_CONFIRMATION_TTL_S,
    ModeError,
    RunState,
    RuntimeState,
    TradeMode,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.safety,
]

REPO = Path(__file__).resolve().parents[1]


class _Clock:
    """Deterministic monotonic clock, so expiry is testable without sleeping."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _state(clock: _Clock | None = None) -> RuntimeState:
    s = RuntimeState()
    if clock is not None:
        s._clock = clock
    return s


def _armed(state: RuntimeState) -> str:
    state.start()
    token = state.request_real_confirmation()
    state.arm_real(confirmation=token, preflight_ok=True)
    assert state.real_armed
    return token


# ---------------------------------------------------------------------------
# 1. ordinary startup cannot arm REAL
# ---------------------------------------------------------------------------


def test_a_fresh_state_is_never_armed():
    s = RuntimeState()
    assert s.real_armed is False
    assert s.real_trading_enabled is False
    assert s.trade_mode is TradeMode.PAPER


def test_startup_does_not_arm_real():
    s = RuntimeState()
    s.start()
    assert s.real_armed is False
    assert s.trade_mode is TradeMode.PAPER


def test_construction_cannot_request_an_armed_state():
    """
    No constructor argument can produce an armed runtime.

    ``real_armed`` is ``init=False``, so the attempt raises ``TypeError``
    instead of silently building an armed state -- a loud failure rather than a
    latent footgun. Found by this audit, not assumed.
    """
    with pytest.raises(TypeError):
        RuntimeState(real_armed=True)  # type: ignore[call-arg]

    # the one inconsistent construction that remains is downgraded, not honoured
    s = RuntimeState(run_state=RunState.REAL_ARMED)
    assert s.real_armed is False
    assert s.real_trading_enabled is False
    assert s.trade_mode is TradeMode.PAPER
    assert s.run_state is RunState.PAPER


def test_the_armed_flag_is_not_a_constructor_parameter():
    """
    Signature-agnostic: rather than pinning the exact parameter list (which
    changes whenever a field is added), try every parameter with a truthy value
    and assert the result is never armed. This fails if ANY constructor
    argument, present or future, can set the flag.
    """
    import itertools

    params = [p for p in inspect.signature(RuntimeState).parameters]
    assert "real_armed" not in params
    for combo in itertools.product([True, False], repeat=len(params)):
        kwargs = dict(zip(params, combo))
        s = RuntimeState(**kwargs)
        assert s.real_armed is False, f"{kwargs} produced an armed state"
        assert s.real_trading_enabled is False, kwargs


def test_engine_startup_asserts_the_interlock(tmp_path: Path, monkeypatch):
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    trader = PaperTrader(build_config(tmp_path))
    trader.startup()
    assert trader.state.real_trading_enabled is False
    assert trader.state.run_state is RunState.OFF


# ---------------------------------------------------------------------------
# 2. configuration alone cannot bypass the gate
# ---------------------------------------------------------------------------


def test_no_configuration_input_produces_trading_enabled():
    """
    ``RealConfig.trading_enabled`` is a property with no setter and no source.

    Asserted structurally: the body is a bare ``return False``, so there is no
    branch through which a value could arrive.
    """
    from autopaper.config import RealConfig

    src = inspect.getsource(RealConfig.trading_enabled.fget)
    assert "return False" in src
    assert "self." not in src, "trading_enabled must read no field at all"
    assert RealConfig(env_trading_enabled_request=True).trading_enabled is False
    assert RealConfig(risk_acknowledged=True).trading_enabled is False


def test_env_file_cannot_arm_real(tmp_path: Path, monkeypatch):
    from autopaper.config import build_config

    (tmp_path / ".env.local").write_text(
        "REAL_TRADING_ENABLED=true\nI_UNDERSTAND_REAL_TRADING_RISKS=YES\n"
        "WALLET_KEYPAIR_PATH=/tmp/id.json\nRPC_URL=https://rpc.example\n"
        "MODE=REAL\n"
    )
    for name in ("REAL_TRADING_ENABLED", "I_UNDERSTAND_REAL_TRADING_RISKS",
                 "MODE", "AUTO_ARM_REAL", "ENABLE_LIVE"):
        monkeypatch.setenv(name, "true")
    cfg = build_config(tmp_path)
    assert cfg.real.trading_enabled is False
    s = RuntimeState()
    s.start()
    assert s.trade_mode is TradeMode.PAPER


def test_arm_real_has_no_override_parameter():
    """
    There is no third parameter a caller could use to skip a step.

    Only the token, the preflight verdict and a detail string exist.
    """
    params = set(inspect.signature(RuntimeState.arm_real).parameters)
    assert params == {"self", "confirmation", "preflight_ok", "preflight_detail"}


# ---------------------------------------------------------------------------
# 3-4. confirmation tokens: single use, invalid, expired
# ---------------------------------------------------------------------------


def test_requesting_a_confirmation_arms_nothing():
    s = _state()
    s.start()
    s.request_real_confirmation()
    assert s.real_armed is False
    assert s.trade_mode is TradeMode.PAPER


def test_a_token_is_single_use():
    s = _state()
    token = _armed(s)
    s.turn_off_real()
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=True)
    assert s.real_armed is False


def test_a_token_is_consumed_by_a_failed_preflight():
    s = _state()
    s.start()
    token = s.request_real_confirmation()
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=False)
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=True)
    assert s.real_armed is False


def test_a_token_is_consumed_by_a_cancelled_confirmation():
    s = _state()
    s.start()
    token = s.request_real_confirmation()
    s.cancel_real_confirmation()
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=True)


def test_a_forged_token_is_refused():
    s = _state()
    s.start()
    s.request_real_confirmation()
    for forged in ("", "x", "forged-token", None):
        with pytest.raises(ModeError):
            s.arm_real(confirmation=forged, preflight_ok=True)
    assert s.real_armed is False


def test_a_token_is_refused_without_a_prior_request():
    s = _state()
    s.start()
    with pytest.raises(ModeError):
        s.arm_real(confirmation="anything", preflight_ok=True)
    assert s.real_armed is False


def test_tokens_are_unpredictable():
    """Distinct tokens per request, and long enough not to be guessable."""
    s = _state()
    tokens = {s.request_real_confirmation() for _ in range(50)}
    assert len(tokens) == 50
    assert all(len(t) >= 16 for t in tokens)


def test_an_expired_token_fails_closed():
    """
    A confirmation left open must not arm REAL later.

    The token has a deadline, measured on a monotonic clock, and an expired one
    is consumed rather than retried.
    """
    clock = _Clock()
    s = _state(clock)
    s.start()
    token = s.request_real_confirmation()
    assert s.confirmation_expired() is False

    clock.advance(REAL_CONFIRMATION_TTL_S + 1)
    assert s.confirmation_expired() is True
    with pytest.raises(ModeError, match="expired"):
        s.arm_real(confirmation=token, preflight_ok=True)
    assert s.real_armed is False
    # consumed, so even a later clock change cannot revive it
    assert s.confirmation_age_s() is None
    clock.advance(1)
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=True)


def test_a_token_just_inside_the_deadline_still_works():
    clock = _Clock()
    s = _state(clock)
    s.start()
    token = s.request_real_confirmation()
    clock.advance(REAL_CONFIRMATION_TTL_S - 1)
    assert s.confirmation_expired() is False
    s.arm_real(confirmation=token, preflight_ok=True)
    assert s.real_armed is True


def test_expiry_uses_a_monotonic_clock():
    """
    A wall-clock jump cannot extend or collapse a token's life.

    Two structural facts plus one behavioural one: the default clock really is
    ``time.monotonic`` (not ``time.time``), the age is derived only from the
    injected clock, and the TTL is a short bounded constant.
    """
    # exact discriminator: time.monotonic() yields ('time', 'monotonic'),
    # whereas wall-clock time.time() would yield ('time', 'time')
    default = RuntimeState.__dataclass_fields__["_clock"].default
    assert default.__code__.co_names == ("time", "monotonic"), (
        "the default token clock must be time.monotonic, not wall time"
    )

    # the age derives from the injected clock alone
    age_src = inspect.getsource(RuntimeState.confirmation_age_s)
    assert "self._clock()" in age_src
    assert "time." not in age_src, "age must not read a clock directly"

    # bounded, and short enough that a stale button is useless
    assert isinstance(REAL_CONFIRMATION_TTL_S, float)
    assert 0 < REAL_CONFIRMATION_TTL_S <= 900

    # behaviourally: age tracks the injected clock exactly
    clock = _Clock()
    s = _state(clock)
    s.request_real_confirmation()
    clock.advance(30)
    assert s.confirmation_age_s() == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 5. preflight is required before arming
# ---------------------------------------------------------------------------


def test_preflight_false_refuses_arming():
    s = _state()
    s.start()
    token = s.request_real_confirmation()
    with pytest.raises(ModeError, match="preflight"):
        s.arm_real(confirmation=token, preflight_ok=False, preflight_detail="x")
    assert s.real_armed is False


def test_arming_records_the_preflight_detail():
    s = _state()
    s.start()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True, preflight_detail="all 14 checks")
    arm = [h for h in s.history if h["action"] == "real_armed"]
    assert arm and "all 14 checks" in arm[-1]["detail"]


# ---------------------------------------------------------------------------
# 6-7. executor and PAPER/REAL separation
# ---------------------------------------------------------------------------


def test_paper_mode_never_routes_to_the_real_executor():
    """
    The branch is on the canonical state, and the PAPER branch cannot reach the
    real executor.
    """
    from autopaper.engine import PaperTrader

    src = inspect.getsource(PaperTrader._execute_buy)
    assert "self.state.trade_mode is TradeMode.REAL" in src
    assert "_execute_real_buy" in src
    # the paper branch is a separate method that never touches real_executor
    paper = inspect.getsource(PaperTrader._execute_paper_buy)
    assert "real_executor" not in paper
    assert "self.real_account" not in paper


def test_the_real_executor_is_not_even_constructed_in_paper(tmp_path):
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    trader = PaperTrader(build_config(tmp_path))
    trader.startup()
    trader.state.start()
    assert trader._real_executor is None, (
        "a PAPER process must not acquire the real execution capability at all"
    )


def test_every_pretrade_check_names_real_armed():
    from autopaper.execution.real import PRE_TRADE_CHECKS

    assert "real_armed_by_operator" in PRE_TRADE_CHECKS
    assert "real_mode_explicit" in PRE_TRADE_CHECKS
    assert "no_emergency_stop" in PRE_TRADE_CHECKS


# ---------------------------------------------------------------------------
# 8. test mode cannot broadcast
# ---------------------------------------------------------------------------


def test_the_preflight_never_broadcasts():
    """
    The preflight is read-only: it never calls a sending method.

    The source's own ``no_broadcast_in_preflight`` check TRUSTS the backend's
    ``broadcast_attempted`` report, so asserting on that check alone would only
    be testing the backend's honesty. This instead runs the real preflight
    against a backend that raises on any send, so a broadcast fails the test
    outright.
    """
    import asyncio

    from autopaper.execution.real import RealExecutor

    class Tripwire:
        """Any attempt to send raises, and is recorded."""

        def __init__(self):
            self.attempts: list[str] = []

        async def verify(self):
            return {"wallet_loaded": True, "wallet_masked": "Abc1...Wxyz",
                    "balance_sol": 1.0, "slot": 1, "rpc_ok": True,
                    "jupiter_ok": True, "broadcast_attempted": False}

        def _trip(self, name):
            async def boom(*a, **k):
                self.attempts.append(name)
                raise AssertionError(f"preflight called a sending method: {name}")
            return boom

        def __getattr__(self, name):
            if name in {"send", "send_transaction", "execute_swap", "broadcast",
                        "sign", "sign_transaction", "sign_and_send"}:
                return self._trip(name)
            raise AttributeError(name)

    class Cfg:
        wallet_keypair_path = "/tmp/does-not-exist.json"
        rpc_url = "https://rpc.invalid"
        slippage_bps = 500
        max_price_impact_pct = 5.0
        max_open_positions = 1
        reentry_enabled = False

    backend = Tripwire()
    ex = RealExecutor(Cfg(), backend=backend)
    result = asyncio.run(ex.preflight())
    assert backend.attempts == [], f"preflight tried to send: {backend.attempts}"
    # the read-only claim is reported, and the preflight still reached a verdict
    names = {c.name for c in result.checks}
    assert "no_broadcast_in_preflight" in names
    assert isinstance(result.ok, bool)


def test_a_failed_pretrade_check_prevents_the_order():
    from autopaper.execution.real import RealExecutor, SafetyCheck

    class Cfg:
        wallet_keypair_path = None
        rpc_url = None
        slippage_bps = 500
        max_price_impact_pct = 5.0
        max_open_positions = 1
        reentry_enabled = False

    class Backend:
        async def execute_swap(self, *a, **k):
            raise AssertionError("must not trade when a precheck failed")

    ex = RealExecutor(Cfg(), backend=Backend())

    import asyncio

    fill = asyncio.run(
        ex.open_position(
            mint="M", position_lamports=155_000, slippage_bps=500,
            prechecks=[SafetyCheck("real_armed_by_operator", True, "armed"),
                       SafetyCheck("no_emergency_stop", False, "latched")],
        )
    )
    assert fill.status == "PRE_TRADE_CHECK_FAILED"
    assert not fill.filled
    assert "no_emergency_stop" in fill.ambiguity_reason


# ---------------------------------------------------------------------------
# 9. restart does not restore armed REAL
# ---------------------------------------------------------------------------


def test_armed_state_is_not_persisted():
    """
    No serialisation of the armed flag anywhere in the runtime.

    Checked for the actual persistence primitives. ``_record(...)`` writes to an
    in-memory bounded list -- an audit trail for the operator, not a file -- so
    logging a transition is allowed and only a write is not.
    """
    WRITERS = ("json", "save", "write_text", "write_bytes", "dump(",
               "pickle", "marshal", "to_json", "sqlite", "insert")
    offenders = []
    for p in (REPO / "src/autopaper").rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "real_armed" not in line:
                continue
            if any(w in line.lower() for w in WRITERS):
                offenders.append(f"{p.name}:{i}: {line.strip()}")
    assert not offenders, "real_armed appears to be persisted:\n" + "\n".join(offenders)


def test_the_armed_flag_is_never_written_to_the_journal(tmp_path: Path):
    """The trade journal must not carry an armed flag either."""
    from autopaper.journal import Journal
    from autopaper.config import build_config

    j = Journal(tmp_path / "var" / "journal")
    j.event("PROBE", mode="PAPER", note="x")
    j.paper_entry(mint="M", sol=1.0, price=1.0, signal_quality=0.9)
    j.paper_exit(mint="M", reason="probe", pnl_pct=1.0)
    for p in (tmp_path / "var").rglob("*"):
        if p.is_file():
            blob = p.read_text(errors="replace")
            assert '"real_armed"' not in blob, p
            assert '"real_trading_enabled": true' not in blob.lower(), p


def test_a_new_state_after_a_restart_is_never_armed(tmp_path: Path,
                                                   monkeypatch):
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    first = PaperTrader(build_config(tmp_path))
    first.startup()
    first.state.start()
    _armed(first.state)                      # arm it
    assert first.state.real_trading_enabled

    # a brand-new process over the same account
    second = PaperTrader(build_config(tmp_path))
    assert second.state.real_trading_enabled is False
    second.startup()
    assert second.state.real_trading_enabled is False
    assert second.state.run_state is RunState.OFF


def test_the_emergency_stop_latch_does_survive_a_restart(tmp_path: Path,
                                                         monkeypatch):
    """
    The inverse property, and the reason the latch is persisted at all.

    A restart must not quietly un-halt a halted bot.
    """
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    first = PaperTrader(build_config(tmp_path))
    first.startup()
    first.state.emergency_stop_now()
    first._persist_emergency_stop(True)

    second = PaperTrader(build_config(tmp_path))
    assert second.state.emergency_stop is True
    assert second.state.run_state is RunState.EMERGENCY_STOP
    ok, _why = second.state.can_open_new_entries()
    assert ok is False


def test_arming_while_stopped_does_not_start_trading():
    s = _state()
    s.stop()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    assert s.real_armed is True
    assert s.run_state is RunState.STOPPED
    ok, _why = s.can_open_new_entries()
    assert ok is False, "arming a capability must not start the loop"


def test_emergency_stop_refuses_a_fresh_arming():
    s = _state()
    s.start()
    s.emergency_stop_now()
    token = s.request_real_confirmation()
    with pytest.raises(ModeError, match="EMERGENCY STOP"):
        s.arm_real(confirmation=token, preflight_ok=True)
    assert s.real_armed is False


# ---------------------------------------------------------------------------
# cross-cutting
# ---------------------------------------------------------------------------


def test_no_transition_sets_armed_without_arm_real():
    """
    ``real_armed = True`` appears in exactly one place.

    If a second assignment ever appears, a new path to armed REAL exists and
    this test should fail rather than let it pass unnoticed.
    """
    offenders = []
    for p in (REPO / "src/autopaper").rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("self.real_armed = True"):
                if p.name == "modes.py" and "self.real_armed = True" in s:
                    continue
                offenders.append(f"{p.name}:{i}")
    # exactly one, inside arm_real
    arm_src = inspect.getsource(RuntimeState.arm_real)
    assert arm_src.count("self.real_armed = True") == 1
    assert not offenders, offenders


def test_audit_reports_real_disabled_at_startup():
    from autopaper import audit

    r = audit()
    assert r.ok, r.render()
    names = {n for n, _, _ in r.checks}
    assert "real_disabled_at_startup" in names
    assert "real_requires_operator" in names
    assert "real_arm_rejects_bare_call" in names
    assert "active_strategy_is_l25" in names


def test_the_ttl_default_is_not_configurable(tmp_path: Path, monkeypatch):
    """The deadline is a safety constant, not an operator dial."""
    (tmp_path / ".env.local").write_text("REAL_CONFIRMATION_TTL_S=99999\n")
    monkeypatch.setenv("REAL_CONFIRMATION_TTL_S", "99999")
    from autopaper.config import build_config

    build_config(tmp_path)   # must not raise
    assert REAL_CONFIRMATION_TTL_S == 300.0
    cfg = build_config(tmp_path)
    assert not hasattr(cfg, "confirmation_ttl")
    assert modes.REAL_CONFIRMATION_TTL_S == 300.0
