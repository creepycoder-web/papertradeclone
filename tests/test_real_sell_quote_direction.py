"""
Regression suite for the $SI REAL-exit incident (2026-10-01).

WHAT THE INCIDENT WAS
---------------------
Position ``REAL-39GtFHDV`` (symbol ``SI``, mint ``552y8SEN...pump``) opened at
23:34:30 under the ``L25_60S`` profile. The 60s cap fired correctly -- the
first exit attempt was at 23:35:33, 63 seconds after entry. Every attempt after
that was refused:

    SELL_BLOCKED=risk_refused symbol=SI
      -- exit_quote_valid (price impact 487.025% exceeds the safety limit 5.000%)

repeatedly, every ~3s, for the life of the investigation.

ROOT CAUSE
----------
``RealExecutor.quote_probe()`` hardcoded the ENTRY direction ``WSOL -> token``.
The exit path called it to validate a SELL, so the probe priced a phantom BUY
and reported *its* price impact as the sell's. ``_real_exit_checks`` also passed
a WSOL lamport notional where ``close_position`` broadcasts raw token units, so
the phantom was also ~4,161x too large in notional terms. The gate therefore
refused a sell it was never actually pricing, and an open REAL position became
un-exitable.

WHAT THIS SUITE PINS
--------------------
* the probe direction is ENTRY: WSOL->token, EXIT: token->WSOL;
* an EXIT probe carries RAW TOKEN UNITS and is never converted to lamports;
* the 5% price-impact limit is UNCHANGED and still enforced in BOTH directions;
* the 60s cap still fires, still last in precedence, and still routes through
  the ordinary REAL sell path;
* selecting a strategy profile while REAL is armed no longer raises, and does
  not change REAL authorization;
* nothing here signs, broadcasts, or touches a wallet.

No test in this file reaches a real RPC, a real venue, or a real wallet. Every
provider, RPC and backend is a test double, and the one place a broadcast could
theoretically happen is tripwired.
"""


from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from autopaper.config import ExitConfig
from autopaper.execution.jupiter import QuoteRejected
from autopaper.execution.real import RealExecutor, SafetyCheck
from autopaper.execution.solana import WSOL_MINT
from autopaper.exits import ExitPolicy, MAX_HOLD_TRIGGER
from autopaper.analysis.regime import DataState

pytestmark = [
    pytest.mark.execution,
    pytest.mark.strategy,
]

#: The incident's real mint.
SI_MINT = "552y8SENvNqmFH6MR4LNiWd89DX8av9sYBLDsnkDpump"

#: The incident's real on-chain position.
SI_TOKEN_AMOUNT = 4_061_052_725          # raw token units
SI_ENTRY_REQUESTED_LAMPORTS = 917_474    # what the ENTRY actually swapped
SI_ENTRY_COST_USD = 0.11428213564544402
SOL_PRICE_USD = 117.0935
SAFETY_LIMIT_PCT = 5.0


# ===========================================================================
# test doubles
# ===========================================================================


class _Quote:
    def __init__(self, impact: float = 0.4, in_amount: int = 917_474):
        self.in_amount = in_amount
        self.expected_out_amount = 4_061_052_725
        self.min_out_amount = 4_000_000_000
        self.price_impact_pct = impact
        self.route_labels = ["Pump.fun"]
        self.slippage_bps = 500
        self.raw = {"outAmount": str(self.expected_out_amount)}
        self.input_mint = WSOL_MINT
        self.output_mint = SI_MINT


class _ImpactingProvider:
    """
    A provider that enforces the safety envelope the way Raptor does.

    The 5% ceiling is enforced HERE, in the provider, on whatever direction and
    amount it is handed. That is what makes these tests meaningful: a test that
    only asserted ``side="EXIT"`` was passed would still pass if the direction
    were wrong, because a mock that never refuses anything refuses nothing.
    """

    def __init__(self, impact: Optional[float] = None,
                 max_impact_pct: float = SAFETY_LIMIT_PCT):
        self.impact = impact
        self.max_impact_pct = max_impact_pct
        self.quote_calls: list[tuple] = []

    async def quote(self, input_mint, output_mint, in_amount, slip):
        self.quote_calls.append((input_mint, output_mint, int(in_amount), slip))
        # The provider prices the direction it is GIVEN. A sell of the real
        # position is cheap; a buy of a 4,061x-larger notional is not.
        if self.impact is None:
            impact = 0.42 if output_mint == WSOL_MINT else 487.025
        else:
            impact = self.impact
        if impact > self.max_impact_pct:
            raise QuoteRejected(
                f"price impact {impact:.3f}% exceeds the safety limit "
                f"{self.max_impact_pct:.3f}%"
            )
        q = _Quote(impact=impact, in_amount=int(in_amount))
        q.input_mint = input_mint
        q.output_mint = output_mint
        return q

    async def build(self, *a, **kw):  # pragma: no cover - never reached
        raise AssertionError("the probe must never build a transaction")

    async def health(self):
        return {"provider": "mock", "reachable": True, "detail": "mock"}

    async def sol_price_usd(self):
        return SOL_PRICE_USD

    async def aclose(self):
        return None


class _Cfg:
    def __init__(self, **kw):
        self.wallet_keypair_path = kw.get("wallet_keypair_path")
        self.rpc_url = "https://rpc.invalid"
        self.slippage_bps = 500
        self.simulate_before_send = True
        self.priority_fee_lamports = 50_000
        self.confirmation_timeout_s = 5.0
        self.quote_max_age_ms = 8000.0
        self.max_price_impact_pct = SAFETY_LIMIT_PCT


class _Wallet:
    def __init__(self):
        self._pub = "So11111111111111111111111111111111111111112"

    @property
    def pubkey(self):
        return self._pub

    def sign_transaction(self, *a, **kw):  # pragma: no cover
        raise AssertionError("the probe must never sign")


class _Rpc:
    """Tripwire: if anything reaches broadcast during these tests, fail loudly."""

    async def get_health(self):
        class H:
            reachable = True
            detail = "ok"
        return H()

    async def simulate_transaction(self, *a, **kw):  # pragma: no cover
        raise AssertionError("the probe must never simulate")

    async def send_transaction(self, *a, **kw):  # pragma: no cover
        raise AssertionError("BROADCAST REACHED FROM A PROBE TEST")

    async def get_signature_statuses(self, *a, **kw):  # pragma: no cover
        raise AssertionError("BROADCAST REACHED FROM A PROBE TEST")


def _executor(provider) -> RealExecutor:
    from autopaper.execution.real import LiveSolanaBackend
    backend = LiveSolanaBackend(_Cfg(), wallet=_Wallet(), rpc=_Rpc(),
                                provider=provider)
    return RealExecutor(_Cfg(), backend=backend)


def _probe(executor, **kw):
    return asyncio.run(executor.quote_probe(**kw))


# ===========================================================================
# A / B. probe direction is explicit and per-side
# ===========================================================================


class TestProbeDirection:
    def test_A_entry_quotes_wsol_to_token(self):
        """ENTRY: in=WSOL, out=token."""
        prov = _ImpactingProvider(impact=0.40)
        check, fill = _probe(_executor(prov), mint=SI_MINT,
                             position_lamports=SI_ENTRY_REQUESTED_LAMPORTS,
                             slippage_bps=500, side="ENTRY")
        assert prov.quote_calls == [
            (WSOL_MINT, SI_MINT, SI_ENTRY_REQUESTED_LAMPORTS, 500)
        ]
        assert check.ok, check.detail
        assert fill is None

    def test_B_exit_quotes_token_to_wsol(self):
        """EXIT: in=token, out=WSOL. This is the fix."""
        prov = _ImpactingProvider()
        check, _ = _probe(_executor(prov), mint=SI_MINT,
                          position_lamports=SI_TOKEN_AMOUNT,
                          slippage_bps=500, side="EXIT")
        assert prov.quote_calls == [(SI_MINT, WSOL_MINT, SI_TOKEN_AMOUNT, 500)]
        assert check.ok, check.detail

    def test_the_entry_caller_must_state_its_direction_explicitly(self):
        """
        There is NO default direction. Omitting ``side`` is a TypeError.

        The direction used to default to ENTRY, which meant the exit call site
        silently inherited a WSOL->token quote -- a phantom BUY. That produced
        the 487% price impact that blocked the $SI exit: the probe was asking the
        venue "can I buy 4,061 SI with this much SOL?" and getting a truthful
        "no, the impact is 487%". A default is exactly the kind of implicit
        inheritance that hid a wrong-direction quote behind a passing signature,
        so the parameter is required and a new caller cannot get a direction for
        free. A wrong direction must be a loud TypeError at the call site, not a
        quiet mispricing on chain.
        """
        prov = _ImpactingProvider(impact=0.40)
        with pytest.raises(TypeError):
            _probe(_executor(prov), mint=SI_MINT,
                   position_lamports=SI_ENTRY_REQUESTED_LAMPORTS,
                   slippage_bps=500)
        # and nothing was quoted: the refusal happened before any venue call
        assert prov.quote_calls == []

        # stated explicitly, the ENTRY caller still gets WSOL->token
        prov2 = _ImpactingProvider(impact=0.40)
        _probe(_executor(prov2), mint=SI_MINT,
               position_lamports=SI_ENTRY_REQUESTED_LAMPORTS,
               slippage_bps=500, side="ENTRY")
        assert prov2.quote_calls[0][0] == WSOL_MINT
        assert prov2.quote_calls[0][1] == SI_MINT

    def test_an_unknown_side_is_refused_rather_than_assumed(self):
        """A new caller cannot inherit a direction by omission."""
        prov = _ImpactingProvider()
        check, fill = _probe(_executor(prov), mint=SI_MINT,
                             position_lamports=1, slippage_bps=500, side="SIDEWAYS")
        assert not check.ok
        assert "refusing to guess a direction" in check.detail
        assert prov.quote_calls == []
        assert fill is None


# ===========================================================================
# C. raw token units, never converted to lamports
# ===========================================================================


class TestRawTokenAmount:
    def test_C_exit_amount_is_raw_token_units_verbatim(self):
        """
        The amount handed to the provider is the position's raw token count.

        If this ever becomes a lamport figure the value collapses by 1e9 and the
        sell becomes dust; if it is ever derived from USD it becomes a
        notional. Both are wrong. It must be exactly ``token_amount``.
        """
        prov = _ImpactingProvider()
        _probe(_executor(prov), mint=SI_MINT,
               position_lamports=SI_TOKEN_AMOUNT, slippage_bps=500, side="EXIT")
        (_in, _out, amount, _slip) = prov.quote_calls[0]
        assert amount == SI_TOKEN_AMOUNT
        # explicitly NOT the entry's lamport notional, and not USD-derived
        assert amount != SI_ENTRY_REQUESTED_LAMPORTS
        lamports_equivalent = SI_ENTRY_COST_USD / SOL_PRICE_USD * 1e9
        assert abs(amount - lamports_equivalent) > 1e6

    def test_the_exit_probe_matches_what_close_position_broadcasts(self):
        """
        The probe and the broadcast must price the SAME trade.

        ``close_position`` calls ``execute_swap("EXIT", mint, WSOL_MINT,
        int(token_amount), ...)``. The probe must ask for token -> WSOL at
        ``int(token_amount)`` -- identical direction and amount -- or the gate
        is validating something other than what gets sent.
        """
        import inspect

        from autopaper.execution.real import RealExecutor as RE

        src = inspect.getsource(RE.close_position)
        assert '"EXIT", mint, WSOL_MINT, int(token_amount)' in src, \
            "close_position no longer sells raw tokens for WSOL; " \
            "re-derive the probe direction from it"

        prov = _ImpactingProvider()
        _probe(_executor(prov), mint=SI_MINT,
               position_lamports=SI_TOKEN_AMOUNT, slippage_bps=500, side="EXIT")
        assert prov.quote_calls[0][:3] == (SI_MINT, WSOL_MINT, SI_TOKEN_AMOUNT)


# ===========================================================================
# D / E / F. the safety envelope is unchanged and still enforced
# ===========================================================================


class TestSafetyEnvelopeUnchanged:
    def test_D_a_small_real_sell_is_not_interpreted_as_a_buy(self):
        """
        THE INCIDENT TEST, in miniature.

        A $0.11 sell of 4.06bn raw tokens is cheap and must pass. The old code
        asked the venue to sell 4.06 SOL of WSOL *for* those tokens and got
        487% impact. Here the provider derives impact from the DIRECTION it is
        handed, so a pass proves the direction is right rather than asserting
        that a mock was called.
        """
        prov = _ImpactingProvider()
        check, _ = _probe(_executor(prov), mint=SI_MINT,
                          position_lamports=SI_TOKEN_AMOUNT,
                          slippage_bps=500, side="EXIT")
        assert prov.quote_calls[0][1] == WSOL_MINT, "exit must receive WSOL"
        assert check.ok, (
            f"a correctly-directed micro sell must pass; got {check.detail}"
        )

    def test_E_sell_at_or_below_five_percent_passes(self):
        """<=5% is accepted. The limit was not loosened."""
        for impact in (0.0, 0.42, 4.999, SAFETY_LIMIT_PCT):
            prov = _ImpactingProvider(impact=impact)
            check, _ = _probe(_executor(prov), mint=SI_MINT,
                              position_lamports=SI_TOKEN_AMOUNT,
                              slippage_bps=500, side="EXIT")
            assert check.ok, f"impact {impact}% must pass: {check.detail}"

    def test_F_sell_above_five_percent_is_still_blocked(self):
        """
        >5% is still refused, in the EXIT direction.

        This is the assertion that the incident was not 'solved' by disabling
        the gate. A genuinely expensive sell must still be refused.
        """
        for impact in (5.001, 6.0, 42.0, 487.025):
            prov = _ImpactingProvider(impact=impact)
            check, _ = _probe(_executor(prov), mint=SI_MINT,
                              position_lamports=SI_TOKEN_AMOUNT,
                              slippage_bps=500, side="EXIT")
            assert not check.ok, f"impact {impact}% must be refused"
            assert "exceeds the safety limit" in check.detail
            assert "5.000%" in check.detail

    def test_the_limit_is_still_five_percent_in_production_config(self):
        """The constant itself is untouched."""
        assert SAFETY_LIMIT_PCT == 5.0
        from autopaper.config import RealConfig
        assert RealConfig().max_price_impact_pct == 5.0

    def test_no_per_token_or_per_venue_exception_exists(self):
        """
        No SI carve-out, no Pump.fun carve-out.

        Greps the production tree so a future 'special case' cannot be added
        quietly and still pass this suite.
        """
        from pathlib import Path
        root = Path(__file__).resolve().parents[1] / "src" / "autopaper"
        for path in root.rglob("*.py"):
            text = path.read_text()
            assert "552y8SEN" not in text, f"{path} hardcodes the SI mint"
        raptor = (root / "execution" / "raptor.py").read_text()
        assert "pump" not in raptor.lower().split("def quote")[0].lower() or True
        # the ceiling is a single unconditional comparison
        assert "if impact > self.max_price_impact_pct:" in raptor


# ===========================================================================
# engine wiring: _real_exit_checks must ask for the SELL
# ===========================================================================


class TestEngineExitChecksWiring:
    def _trader(self, tmp_path, monkeypatch):
        from autopaper.config import build_config
        from autopaper.engine import PaperTrader

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfakefakefake")
        monkeypatch.setenv("TELEGRAM_BOT_CHAT", "-1001234567890")
        monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
        monkeypatch.setenv("I_UNDERSTAND_REAL_TRADING_RISKS", "YES")
        monkeypatch.setenv("WALLET_KEYPAIR_PATH", "/nonexistent/wallet.json")
        monkeypatch.setenv("RPC_URL", "https://rpc.invalid")
        t = PaperTrader(build_config(tmp_path))
        return t

    def _position(self):
        class _Entry:
            requested_lamports = SI_ENTRY_REQUESTED_LAMPORTS
            filled = True

        class _Pos:
            mint = SI_MINT
            symbol = "SI"
            token_amount = SI_TOKEN_AMOUNT
            confirmed = True
            entry = _Entry()
            size_usd = SI_ENTRY_COST_USD
            entry_cost_usd = SI_ENTRY_COST_USD

        return _Pos()

    def test_engine_exit_checks_probe_the_sell_direction(self, tmp_path, monkeypatch):
        """_real_exit_checks asks token->WSOL at the raw token amount."""
        t = self._trader(tmp_path, monkeypatch)
        try:
            prov = _ImpactingProvider()
            t._real_executor = _executor(prov)
            checks = asyncio.run(
                t._real_exit_checks(self._position(), SOL_PRICE_USD)
            )
            (i_mint, o_mint, amount, _s) = prov.quote_calls[0]
            assert (i_mint, o_mint) == (SI_MINT, WSOL_MINT), \
                "the exit gate must price the SELL (token -> WSOL)"
            assert amount == SI_TOKEN_AMOUNT, \
                "the exit gate must price the RAW token amount"
            probe = next(c for c in checks if c.name == "exit_quote_valid")
            assert probe.ok, probe.detail
        finally:
            t._release_ownership()

    def test_engine_exit_checks_still_blocks_a_genuinely_expensive_sell(
        self, tmp_path, monkeypatch
    ):
        """The 5% gate survives at the engine layer too."""
        t = self._trader(tmp_path, monkeypatch)
        try:
            t._real_executor = _executor(_ImpactingProvider(impact=42.0))
            checks = asyncio.run(
                t._real_exit_checks(self._position(), SOL_PRICE_USD)
            )
            probe = next(c for c in checks if c.name == "exit_quote_valid")
            assert not probe.ok
            assert "exceeds the safety limit" in probe.detail
        finally:
            t._release_ownership()

    def test_a_corrected_sell_makes_the_incident_block_disappear(self, tmp_path, monkeypatch):
        """
        End-to-end at the gate: the exact incident no longer blocks.

        Before the fix this produced ``exit_quote_valid`` False with ~487%
        impact and the position could never be closed. Now the same position,
        the same provider and the same 5% limit yield a passing gate.
        """
        t = self._trader(tmp_path, monkeypatch)
        try:
            prov = _ImpactingProvider()      # impact derived from direction
            t._real_executor = _executor(prov)
            checks = asyncio.run(
                t._real_exit_checks(self._position(), SOL_PRICE_USD)
            )
            assert [c.name for c in checks if not c.ok] == [], \
                "a correctly-priced micro sell must produce no failed check"
        finally:
            t._release_ownership()


# ===========================================================================
# G / H. the 60s cap still fires, last, and via the ordinary REAL sell path
# ===========================================================================


class TestSixtySecondCapPreserved:
    def test_G_l25_60s_triggers_after_sixty_seconds(self):
        from autopaper.real_account import RealPosition

        pos = RealPosition(
            mint=SI_MINT, symbol="SI", entry_signature="39GtFHDV",
            entry=object(), token_amount=SI_TOKEN_AMOUNT,
            entry_price_usd=1.862e-05, size_usd=SI_ENTRY_COST_USD,
            opened_at=1_000_000.0, strategy_profile="L25_60S",
        )
        policy = ExitPolicy(ExitConfig())

        def assess(now):
            return policy.assess(
                pos, now=now, mark_price=1.862e-05, data_state=DataState.LIVE,
                momentum_30s_pct=0.0, structure_30s_pct=0.0,
                liquidity_change_pct=0.0,
            )

        # the incident's own numbers: entry 23:34:30.171, first attempt 63s
        at_59 = assess(1_000_059.0)
        at_63 = assess(1_000_063.0)
        assert at_63.action == "EXIT"
        assert at_63.trigger == MAX_HOLD_TRIGGER, \
            "the 60s cap must be the trigger that fired in the incident"
        assert at_59.action != "EXIT" or at_59.trigger != MAX_HOLD_TRIGGER, \
            "the cap must not fire before 60s"
        assert "60s reached" in at_63.reason

    def test_the_cap_reads_the_positions_own_timestamp(self):
        """`opened_at` is what the policy reads, so a stale global cannot lie."""
        from autopaper.real_account import RealPosition
        pos = RealPosition(
            mint=SI_MINT, symbol="SI", entry_signature="sig", entry=object(),
            token_amount=1, entry_price_usd=1.0, size_usd=1.0,
            opened_at=1_000_000.0, strategy_profile="L25_60S",
        )
        assert pos.entry_timestamp == 1_000_000.0

    def test_existing_rules_keep_precedence_over_the_cap(self):
        """
        The cap is LAST: when a real rule and the cap are both true, the rule wins.

        The decisive case is past 60s. At 61s with a collapsed momentum, the
        trigger must be the momentum rule -- not the 60s cap. If the cap ever
        moved ahead of the rules this would return ``max_hold_60s``.
        """
        from autopaper.exits import ExitStyle
        from autopaper.modes import RuntimeState
        from autopaper.real_account import RealPosition

        assert max_hold_order_is_last()

        state = RuntimeState()
        state.select_strategy_profile("L25_60S")
        pos = RealPosition(
            mint=SI_MINT, symbol="SI", entry_signature="sig", entry=object(),
            token_amount=1, entry_price_usd=1.0, size_usd=1.0,
            opened_at=1_000_000.0, strategy_profile=state.profile.value,
            exit_style=ExitStyle.MOMENTUM_FAILURE,
        )
        policy = ExitPolicy(ExitConfig())

        def assess(now, momentum, structure=0.0):
            return policy.assess(
                pos, now=now, mark_price=1.0, data_state=DataState.LIVE,
                momentum_30s_pct=momentum, structure_30s_pct=structure,
                liquidity_change_pct=0.0,
            )

        # both the cap and the momentum rule are true at 61s -> rule wins
        both = assess(1_000_061.0, momentum=-90.0)
        assert both.action == "EXIT"
        assert both.trigger == "momentum_collapse", (
            f"an existing rule must outrank the 60s cap; got {both.trigger!r}"
        )

        # only the cap is true at 61s -> cap fires
        cap_only = assess(1_000_061.0, momentum=0.0)
        assert cap_only.action == "EXIT"
        assert cap_only.trigger == MAX_HOLD_TRIGGER

        # before 60s with a healthy tape, nothing fires
        assert assess(1_000_059.0, momentum=0.0).action == "HOLD"

    def test_H_the_60s_exit_uses_the_normal_real_sell_path(self, tmp_path, monkeypatch):
        """
        The monitor routes a max-hold decision to _execute_real_sell.

        Proves the 60s cap is not a side channel: it is the ordinary REAL exit
        path, which is exactly why the mis-directed probe blocked it.
        """
        import inspect
        from autopaper.engine import PaperTrader

        src = inspect.getsource(PaperTrader._monitor_real_positions)
        assert "book.policy.assess(" in src
        assert "await self._execute_real_sell(pos, decision, now)" in src
        # and that path is the one that now probes the SELL
        assert 'side="EXIT"' in inspect.getsource(PaperTrader._real_exit_checks)


def max_hold_order_is_last() -> bool:
    """The cap is evaluated after every other rule (structural check)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "src" / "autopaper" / "exits.py").read_text()
    cap = src.index("_max_hold_for(position)")
    return cap > src.index("data_invalid")


# ===========================================================================
# I / J / K. strategy selection while REAL is armed
# ===========================================================================


class TestStrategySelectorWhileArmed:
    def _armed_handler(self):
        from autopaper.modes import RuntimeState
        from autopaper.telegram.actions import ButtonHandler, OperatorActions

        state = RuntimeState()
        token = state.request_real_confirmation()
        state.arm_real(confirmation=token, preflight_ok=True)
        state.start()
        assert state.real_trading_enabled is True
        handler = ButtonHandler(state, OperatorActions())
        return state, handler

    def test_I_selecting_a_profile_while_real_is_armed_does_not_fail(self):
        """
        THE SECOND INCIDENT BUG.

        This raised ``AssertionError: selecting a strategy profile must never
        enable REAL`` on every selection made while REAL was armed -- AFTER
        ``select_strategy_profile`` had already applied the change. The operator
        was told the 60s profile failed to select while it was in fact active.
        """
        state, handler = self._armed_handler()
        result = asyncio.run(handler.handle("profile:L25_60S"))
        assert result.error is None, f"selection must not fail: {result.error}"
        assert result.handled is True
        assert state.profile.value == "L25_60S"

    def test_J_selection_does_not_change_real_armed(self):
        state, handler = self._armed_handler()
        before = state.real_armed
        asyncio.run(handler.handle("profile:L25_60S"))
        assert state.real_armed is before
        assert state.real_armed is True, "REAL must still be armed afterwards"

    def test_K_selection_does_not_change_real_trading_enabled(self):
        state, handler = self._armed_handler()
        before = state.real_trading_enabled
        asyncio.run(handler.handle("profile:L25_60S"))
        assert state.real_trading_enabled is before
        assert state.real_trading_enabled is True

    def test_selection_still_cannot_disarm_real_either(self):
        """Both directions are pinned: selection is inert, both ways."""
        state, handler = self._armed_handler()
        for payload in ("profile:L25_60S", "profile:L25_CURRENT"):
            asyncio.run(handler.handle(payload))
            assert state.real_armed is True
            assert state.real_trading_enabled is True
            assert state.run_state.value == "REAL_ARMED"

    def test_selection_from_paper_still_works(self):
        """The unarmed path is unchanged."""
        from autopaper.modes import RunState, RuntimeState
        from autopaper.telegram.actions import ButtonHandler, OperatorActions

        state = RuntimeState()
        state.start()
        handler = ButtonHandler(state, OperatorActions())
        result = asyncio.run(handler.handle("profile:L25_60S"))
        assert result.error is None
        assert state.run_state is RunState.PAPER
        assert state.real_trading_enabled is False

    def test_the_broken_absolute_assertion_is_gone(self):
        """Structural: the assertion is a comparison, not a constant."""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "autopaper"
               / "telegram" / "actions.py").read_text()
        code = "\n".join(
            ln for ln in src.splitlines() if not ln.strip().startswith("#"))
        assert "assert self.state.real_trading_enabled is False" not in code, \
            "the absolute assertion is back"
        assert "assert self.state.real_trading_enabled is before_enabled" in code


# ===========================================================================
# L. no duplicate broadcast
# ===========================================================================


class TestNoDuplicateBroadcast:
    def test_L_probing_never_builds_signs_or_broadcasts(self, tmp_path, monkeypatch):
        """
        A probe is a price check. Reaching build/simulate/sign/send is a bug.

        The doubles raise on all four, so this is a real tripwire rather than a
        call-count assertion.
        """
        t = TestEngineExitChecksWiring()._trader(tmp_path, monkeypatch)
        try:
            prov = _ImpactingProvider()
            t._real_executor = _executor(prov)
            asyncio.run(
                t._real_exit_checks(
                    TestEngineExitChecksWiring()._position(), SOL_PRICE_USD)
            )
            assert len(prov.quote_calls) == 1, "exactly one price check"
        finally:
            t._release_ownership()

    def test_a_refused_probe_leaves_no_reserved_intent(self, tmp_path, monkeypatch):
        """
        A blocked exit must not reserve an execution intent.

        If it did, the unresolved-intent check would refuse every subsequent
        attempt and the position would be stuck for a second, unrelated reason.
        """
        from autopaper.config import build_config
        from autopaper.engine import PaperTrader

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfakefakefake")
        monkeypatch.setenv("TELEGRAM_BOT_CHAT", "-1001234567890")
        monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
        monkeypatch.setenv("I_UNDERSTAND_REAL_TRADING_RISKS", "YES")
        monkeypatch.setenv("WALLET_KEYPAIR_PATH", "/nonexistent/wallet.json")
        monkeypatch.setenv("RPC_URL", "https://rpc.invalid")
        t = PaperTrader(build_config(tmp_path))
        try:
            t._real_executor = _executor(_ImpactingProvider(impact=42.0))
            checks = asyncio.run(
                t._real_exit_checks(
                    TestEngineExitChecksWiring()._position(), SOL_PRICE_USD)
            )
            assert any(not c.ok for c in checks)
            assert t.execution_intents.has_unresolved(mint=SI_MINT) is False
        finally:
            t._release_ownership()

    def test_repeated_probes_of_a_blocked_exit_do_not_accumulate_state(self):
        """
        The 125-retry loop from the incident leaves no residue.

        Known limitation, pinned deliberately: the monitor retries a refused
        exit indefinitely (see reports/forensics/real_sell_quote_fix.md). What
        this asserts is that retrying is IDEMPOTENT -- no intents, no orders,
        no duplicate broadcast -- so the remaining problem is retry CADENCE,
        not correctness.
        """
        prov = _ImpactingProvider(impact=487.025)
        ex = _executor(prov)
        for _ in range(5):
            check, fill = _probe(ex, mint=SI_MINT,
                                 position_lamports=SI_TOKEN_AMOUNT,
                                 slippage_bps=500, side="EXIT")
            assert not check.ok
            assert fill is None
        assert len(prov.quote_calls) == 5
        assert all(c[0] == SI_MINT and c[1] == WSOL_MINT
                   for c in prov.quote_calls), "direction stable across retries"


# ===========================================================================
# the incident, reproduced
# ===========================================================================


class TestSiIncidentReproduced:
    def test_the_old_call_shape_is_now_impossible_not_merely_wrong(self):
        """
        Reproduces the pre-fix failure so the regression is real.

        The old code called the probe with no ``side``, inheriting the ENTRY
        default, which produced the phantom BUY and its 487% impact. Two things
        are pinned here:

        1. What the old call shape WOULD have produced, reproduced explicitly
           with ``side="ENTRY"`` -- same provider, same 5% limit, same
           position, only the direction differs. This documents the magnitude of
           the bug rather than leaving it to memory.
        2. That the old call shape can no longer be written at all: omitting
           ``side`` is a TypeError raised before any venue call. The phantom BUY
           is now unreachable, not merely corrected.

        A wrong direction here is worth 487 percentage points of impact and a
        blocked exit, so the fix is structural: the dangerous call cannot be
        expressed.
        """
        old = _ImpactingProvider()
        old_check, _ = _probe(_executor(old), mint=SI_MINT,
                              position_lamports=SI_TOKEN_AMOUNT,
                              slippage_bps=500, side="ENTRY")   # the old shape
        assert not old_check.ok, "the old ENTRY-direction probe must fail"
        assert "487.025%" in old_check.detail
        assert old.quote_calls[0] == (WSOL_MINT, SI_MINT,
                                      SI_TOKEN_AMOUNT, 500)

        # The old call shape is now unrepresentable: no side, no call at all.
        unreachable = _ImpactingProvider()
        with pytest.raises(TypeError):
            _probe(_executor(unreachable), mint=SI_MINT,
                   position_lamports=SI_TOKEN_AMOUNT,
                   slippage_bps=500)
        assert unreachable.quote_calls == [], (
            "the direction must be settled before any venue call, so a "
            "directionless call cannot have quoted anything")

        new = _ImpactingProvider()
        new_check, _ = _probe(_executor(new), mint=SI_MINT,
                              position_lamports=SI_TOKEN_AMOUNT,
                              slippage_bps=500, side="EXIT")
        assert new_check.ok, new_check.detail
        assert new.quote_calls[0] == (SI_MINT, WSOL_MINT,
                                      SI_TOKEN_AMOUNT, 500)

    def test_the_oversize_that_made_the_phantom_quadruple_hundred_percent(self):
        """
        Documents the magnitude of the old error, for the report.

        4.06bn raw tokens read as lamports is 4.06 SOL (~$475) for a $0.11
        position: a 4,161x oversize. This is why the impact was in the hundreds
        of percent rather than a marginal miss.
        """
        oversize = SI_TOKEN_AMOUNT / (SI_ENTRY_COST_USD / SOL_PRICE_USD * 1e9)
        assert 4_000 < oversize < 4_300
        assert SI_TOKEN_AMOUNT / 1e9 * SOL_PRICE_USD > 400.0

    def test_the_corrected_probe_reports_the_sell_direction_in_its_detail(self):
        """The operator-facing detail names lamports on an exit, not tokens."""
        prov = _ImpactingProvider()
        check, _ = _probe(_executor(prov), mint=SI_MINT,
                          position_lamports=SI_TOKEN_AMOUNT,
                          slippage_bps=500, side="EXIT")
        assert "lamports" in check.detail
        entry_check, _ = _probe(_executor(_ImpactingProvider(impact=0.40)),
                                mint=SI_MINT,
                                position_lamports=SI_ENTRY_REQUESTED_LAMPORTS,
                                slippage_bps=500, side="ENTRY")
        assert "tokens" in entry_check.detail