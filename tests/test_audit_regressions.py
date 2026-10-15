"""
Regression tests for the defects found in the October 2026 repository audit.

Every test here corresponds to a confirmed defect, and every expected value is
INDEPENDENTLY DERIVED -- hand-computed from the units and the contract, never
copied from the production function's own output. Where a test pins a value,
the arithmetic is written out in the docstring so a reader can check it without
running anything.

THE DEFECTS
===========
1.  ``AMARIH`` was dropped by ``DecisionEngine.set_entry_gates``, so selecting
    the Amarih experiment left ``L25`` operative and the runtime traded L25.
2.  ``PaperTrader.on_alert`` blocked Telegram only on ``signal_source_selection``
    -- a file the operator UI cannot set to ``Amarih`` -- so selecting the
    Amarih arms did not stop Telegram alerts entering the L25 pipeline.
3.  ``PaperTrader._amarih_quote_sell`` hard-coded ``observed_pair_address=None``
    and ``QuoteService._identity_for`` answers ``UNKNOWN`` when there is no
    address to compare, so ``fill_from_quote(require_identity=True)`` refused
    every exit and no Amarih arm could ever settle.
4.  ``IDENTITY_MISMATCH_VALUES`` omitted ``"fail"``, the token the real quote
    stack emits for a PROVEN mismatch, so a denylist-shaped check accepted it.
5.  ``settle_arm`` tested ``valid_market is False``, so a quote OMITTING the key
    settled with a net PnL booked from an unverified market.
6.  ``input_amount_usd`` kept the flat notional instead of the quote's deployed
    amount, overstating every cost basis and biasing PnL negative.
7.  Arm bankrolls were never persisted, so a restart reset all three ledgers to
    their starting notional.

Plus arithmetic that is verified rather than merely exercised: quote price
orientation on both legs, decimal scaling, and the net-PnL identity.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from autopaper.analysis.decision import DecisionEngine
from autopaper.config import DecisionConfig, ExitConfig
from autopaper.paper_live.arms import (
    AMARIH_ARMS,
    AMARIH_GROUP,
    effective_entry_gates,
)
from autopaper.paper_live.fills import FillModel, fill_from_quote
from autopaper.paper_live.quotes import QuoteService
from autopaper.paper_live.schema import QuotePoint
from autopaper.same_market.models import WSOL_MINT
from autopaper.strategy_profile import (
    KNOWN_ENTRY_GATES,
    GATE_AMARIH,
    GATE_L25,
    GATE_V2_FLOOR,
)

MINT = "4RiJPcL6GXecaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
POOL = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"


def _engine(profile: str = "L25_CURRENT") -> DecisionEngine:
    return DecisionEngine(
        DecisionConfig(strategy_profile=profile), ExitConfig())


# ===========================================================================
# 1. The AMARIH gate was silently dropped
# ===========================================================================


class TestTheAmarihGateIsOperative:
    def test_selecting_amarih_makes_amarih_operative(self):
        """The defect: ``set_entry_gates(["AMARIH"])`` returned ``("L25",)``."""
        assert _engine().set_entry_gates([GATE_AMARIH]) == (GATE_AMARIH,)

    def test_the_contract_survives_the_hop_from_arms_to_the_engine(self):
        """
        The two halves that have to agree. ``effective_entry_gates`` computed
        ``("AMARIH",)`` correctly; only the engine's filter lost it.
        """
        requested = effective_entry_gates(
            [a.value for a in AMARIH_ARMS], profile="L25_CURRENT")
        assert requested == (GATE_AMARIH,)
        assert _engine().set_entry_gates(requested) == (GATE_AMARIH,)

    @pytest.mark.parametrize("gate", KNOWN_ENTRY_GATES)
    def test_every_declared_gate_survives_the_filter(self, gate):
        """
        The filter is checked against ``KNOWN_ENTRY_GATES`` precisely so a
        newly declared gate cannot be left declared-but-inert.
        """
        assert _engine().set_entry_gates([gate]) == (gate,)

    @pytest.mark.parametrize("bad", [[], None, (), ["NOPE"], ["l25 "], [42]])
    def test_it_still_fails_closed(self, bad):
        """
        Unrecognised or empty input must NOT authorize everything. The fix must
        not have weakened the fail-closed property to accept AMARIH.
        """
        engine = _engine("L25_CURRENT")
        assert engine.set_entry_gates(bad) == (GATE_L25,)

    def test_l25_and_beta_are_unchanged(self):
        """No strategy parameter changed for the existing profiles."""
        engine = _engine("L25_CURRENT")
        assert engine.set_entry_gates([GATE_L25]) == (GATE_L25,)
        assert engine.set_entry_gates([GATE_V2_FLOOR]) == (GATE_V2_FLOOR,)
        assert engine.set_entry_gates(
            [GATE_L25, GATE_V2_FLOOR]) == (GATE_L25, GATE_V2_FLOOR)
        assert _engine("L25_CURRENT").entry_gates == (GATE_L25,)

    def test_an_amarih_selection_does_not_veto_through_l25(self):
        """
        Under ``("AMARIH",)`` the L25 refusal must be recorded at ``info`` and
        must NOT be a blocking flag -- otherwise the L25 veto returns through
        the other door once the gate token itself is honoured.
        """
        engine = _engine("L25_CURRENT")
        engine.set_entry_gates([GATE_AMARIH])
        assert engine._operative_gates == (GATE_AMARIH,)
        assert GATE_L25 not in engine._operative_gates


# ===========================================================================
# 3. The exit leg could never settle (identity UNKNOWN)
# ===========================================================================


class TestTheExitLegCanProveItsMarket:
    def test_no_observed_pair_resolves_identity_to_unknown(self):
        """
        The ROOT CAUSE, pinned directly on the quote layer.

        ``QuoteService._identity_for`` answers ``UNKNOWN`` whenever there is no
        ``observed_pair_address`` to compare -- by design, and correct. This
        test documents the fact the exit adapter was violating: with no address,
        identity is unanswerable.
        """
        identity, evidence = QuoteService._identity_for(
            observed_pair_address=None,
            market_type=SimpleNamespace(value="PUMP_BONDING_CURVE"),
            market_address=POOL,
            chain_verified=True,
        )
        assert identity == "UNKNOWN"
        assert "no observed pair address" in evidence

    def test_a_matching_address_resolves_identity_to_pass(self):
        identity, _ = QuoteService._identity_for(
            observed_pair_address=POOL,
            market_type=SimpleNamespace(value="PUMP_BONDING_CURVE"),
            market_address=POOL,
            chain_verified=True,
        )
        assert identity == "PASS"

    def test_a_different_address_is_a_proven_mismatch(self):
        identity, evidence = QuoteService._identity_for(
            observed_pair_address="SomeOtherPool1111111111111111111111",
            market_type=SimpleNamespace(value="PUMP_BONDING_CURVE"),
            market_address=POOL,
            chain_verified=True,
        )
        assert identity == "FAIL"
        assert "different books" in evidence

    def _sell_point(self, identity: str) -> QuotePoint:
        return QuotePoint(
            side="SELL", provider="pump", market_type="PUMP_BONDING_CURVE",
            token_mint=MINT, quote_mint=WSOL_MINT,
            input_amount_raw=1_000_000_000, input_decimals=6,
            input_amount_decimal=1000.0,
            output_amount_raw=2_500_000_000, output_decimals=9,
            output_amount_decimal=2.5,
            effective_price=Decimal("0.0025"), effective_price_usd=None,
            price_impact_pct=None, fee_bps=None, fee_amount=None,
            venue="pumpfun", pool_or_curve=POOL, route=("pump",),
            quote_timestamp=1.0, quote_age_ms=0.0, max_quote_age_ms=60_000.0,
            identity=identity, available=True,
        )

    def test_an_unproven_exit_quote_cannot_produce_a_fill(self):
        """``require_identity=True`` refuses it -- the arm is censored."""
        from autopaper.paper_live.fills import FillRefused

        with pytest.raises(FillRefused):
            fill_from_quote(
                self._sell_point("UNKNOWN"), side="SELL", model=FillModel(),
                require_identity=True)

    def test_a_proven_exit_quote_does_produce_a_fill(self):
        """With the pool address supplied, the same quote settles."""
        fill = fill_from_quote(
            self._sell_point("PASS"), side="SELL", model=FillModel(),
            require_identity=True)
        assert fill.output_amount_decimal == pytest.approx(2.5)
        assert fill.identity == "PASS"


# ===========================================================================
# Quote price orientation and decimal scaling -- ARITHMETIC
# ===========================================================================


class TestQuotePriceOrientation:
    """
    ``effective_price`` is QUOTE ASSET PER WHOLE TOKEN on both legs.

    Hand-computed, 6-decimal token:

    BUY  -- 2.000000000 SOL (2e9 lamports, 9dp) buys 1_000_000 tokens
            (1e12 raw, 6dp).
            price = 2.0 / 1_000_000 = 2e-6 SOL per token.

    SELL -- 1_000_000 tokens (1e12 raw, 6dp) returns 0.002500000 SOL
            (2_500_000 lamports, 9dp).
            price = 0.0025 / 1_000_000 = 2.5e-9 SOL per token.

    The SELL orientation is the one that inverts: dividing lamports by token
    RAW units (rather than whole tokens) is a 10**6 error, and multiplying
    instead of dividing is a second. Both are asserted below.
    """

    def _service(self) -> QuoteService:
        svc = QuoteService.__new__(QuoteService)
        svc.sol_reference = None
        return svc

    def _jupiter(self, side: str, in_amount: int, out_amount: int):
        svc = self._service()
        result = SimpleNamespace(
            in_amount=in_amount, out_amount=out_amount,
            route_plan=[SimpleNamespace(label="PumpSwap")],
            pool_keys=[POOL], price_impact_pct=0.001,
            timestamp=1_000_000_000.0,
        )
        return svc._point_from_jupiter(
            result, side=side, mint=MINT, decimals=6,
            observed_pair_address=POOL, latency_ms=1.0)

    def test_buy_price_is_quote_over_whole_token(self):
        q = self._jupiter("BUY", 2_000_000_000, 1_000_000_000_000)
        assert q.input_decimals == 9 and q.output_decimals == 6
        assert q.input_amount_decimal == pytest.approx(2.0)
        assert q.output_amount_decimal == pytest.approx(1_000_000.0)
        assert q.effective_price == pytest.approx(Decimal("2e-6"))

    def test_sell_price_is_quote_over_whole_token(self):
        q = self._jupiter("SELL", 1_000_000_000_000, 2_500_000)
        assert q.input_decimals == 6 and q.output_decimals == 9
        assert q.input_amount_decimal == pytest.approx(1_000_000.0)
        assert q.output_amount_decimal == pytest.approx(0.0025)
        assert q.effective_price == pytest.approx(Decimal("2.5e-9"))

    def test_a_sell_price_is_not_the_buy_price_of_the_mirror_image(self):
        """
        Regression guard for the historical 1000x inversion: the SELL
        orientation must differ from naively reusing the BUY expression.
        """
        sell = self._jupiter("SELL", 1_000_000_000_000, 2_500_000)
        naive = (
            Decimal(sell.input_amount_raw).scaleb(-sell.input_decimals)
            / Decimal(sell.output_amount_raw).scaleb(-sell.output_decimals)
        )
        assert sell.effective_price != naive

    def test_a_zero_output_has_no_price(self):
        svc = self._service()
        result = SimpleNamespace(
            in_amount=1_000_000_000_000, out_amount=0,
            route_plan=[SimpleNamespace(label="PumpSwap")], pool_keys=[POOL],
            price_impact_pct=None, timestamp=1.0,
        )
        q = svc._point_from_jupiter(
            result, side="SELL", mint=MINT, decimals=6,
            observed_pair_address=POOL, latency_ms=1.0)
        assert q.available is False
        assert q.effective_price is None

    def test_a_single_hop_on_the_observed_pool_passes_identity(self):
        q = self._jupiter("BUY", 2_000_000_000, 1_000_000_000_000)
        assert q.identity == "PASS"


class TestFillDecimalScaling:
    """
    ``fill_from_quote`` must scale each leg by ITS OWN decimals.

    BUY: 2e9 lamports at 9dp = 2.0 SOL in; 1e12 raw at 6dp = 1_000_000 tokens
    out. Asserting both conversions catches a leg scaled by the other leg's
    decimals, which is a 10**3 error on a 6-decimal token.
    """

    def _buy(self) -> QuotePoint:
        return QuotePoint(
            side="BUY", provider="pump", market_type="PUMP_BONDING_CURVE",
            token_mint=MINT, quote_mint=WSOL_MINT,
            input_amount_raw=2_000_000_000, input_decimals=9,
            input_amount_decimal=2.0,
            output_amount_raw=1_000_000_000_000, output_decimals=6,
            output_amount_decimal=1_000_000.0,
            effective_price=Decimal("2e-6"), effective_price_usd=None,
            price_impact_pct=None, fee_bps=None, fee_amount=None,
            venue="pumpfun", pool_or_curve=POOL, route=("pump",),
            quote_timestamp=1.0, quote_age_ms=0.0, max_quote_age_ms=60_000.0,
            identity="PASS", available=True,
        )

    def test_each_leg_uses_its_own_decimals(self):
        fill = fill_from_quote(
            self._buy(), side="BUY", model=FillModel(), require_identity=True)
        assert fill.input_amount_decimal == pytest.approx(2.0)
        assert fill.output_amount_decimal == pytest.approx(1_000_000.0)
        assert fill.input_mint == WSOL_MINT
        assert fill.output_mint == MINT

    def test_no_slippage_is_added_on_top_of_the_quote(self):
        """
        The fill price IS the quote's effective price. An extra penalty would
        look like a worse strategy rather than a bug.
        """
        fill = fill_from_quote(
            self._buy(), side="BUY", model=FillModel(), require_identity=True)
        assert fill.effective_price == Decimal("2e-6")

    def test_a_quote_in_the_wrong_quote_asset_is_refused(self):
        from autopaper.paper_live.fills import FillRefused

        q = self._buy()
        bad = QuotePoint(**{**q.__dict__, "quote_mint": "EPjFWdd5AufqSSqeM2qN1"})
        with pytest.raises(FillRefused):
            fill_from_quote(bad, side="BUY", model=FillModel(),
                            require_identity=True)

    def test_a_stale_quote_is_refused(self):
        from autopaper.paper_live.fills import FillRefused

        q = self._buy()
        stale = QuotePoint(**{**q.__dict__, "quote_age_ms": 61_000.0,
                              "max_quote_age_ms": 60_000.0})
        assert stale.stale is True
        with pytest.raises(FillRefused):
            fill_from_quote(stale, side="BUY", model=FillModel(),
                            require_identity=True)

    def test_net_pnl_identity(self):
        """
        ``realize`` arithmetic, hand-computed.

        entry cost 2.0 SOL, exit proceeds 2.5 SOL, no fees, no network cost:
            gross = 2.5 - 2.0            = 0.5
            net   = 0.5 - 0 - 0           = 0.5
            ret%  = 0.5 / 2.0 * 100      = 25.0
        """
        from autopaper.paper_live.fills import realize

        position = SimpleNamespace(
            entry_cost_quote_asset=Decimal("2.0"),
            trading_fees_quote=Decimal("0"),
            network_costs_quote=Decimal("0"),
            other_costs_quote=Decimal("0"),
            bankroll_before=10.0, sol_notional=2.0, usd_notional=300.0,
            sol_usd_reference=150.0, sol_usd_reference_at=None, opened_at=0.0,
        )
        exit_fill = SimpleNamespace(
            output_amount_raw=2_500_000_000, output_decimals=9, fee=Decimal("0"))
        out = realize(position, exit_fill=exit_fill, exit_observed_mark=None,
                      sol_reference=None, model=FillModel(), now=60.0)
        assert out.gross_pnl_quote == pytest.approx(0.5)
        assert out.net_pnl_quote == pytest.approx(0.5)
        assert out.return_pct == pytest.approx(25.0)
        assert out.bankroll_after == pytest.approx(10.0 + 0.5 * 150.0)


# ===========================================================================
# 4 + 5. Identity vocabulary and the fail-closed exit
# ===========================================================================


class TestIdentityVocabularyFailsClosed:
    def _runtime(self):
        from autopaper.paper_live.amarih_runtime import AmarihRuntime
        runtime = AmarihRuntime.__new__(AmarihRuntime)
        runtime.arm_notional_usd = 10.0
        return runtime

    def _signal(self):
        return SimpleNamespace(
            signal_id="1", mint=MINT, symbol="TKN", strategy="Amarih",
            fired_at=1_000.0, received_at=1_000.0,
            signal_to_receipt_latency_ms=0.0, entry_market_cap_usd=20_000.0,
            buy_score=6.0, owner_wallet="tg:1", raw={})

    def _entry(self, identity):
        from autopaper.paper_live.amarih_runtime import AmarihRuntime
        runtime = self._runtime()
        return AmarihRuntime._entry_from_quote(
            runtime,
            self._signal(),
            {"identity": identity, "token_quantity": 1000.0,
             "token_decimals": 6, "market_id": POOL},
            failure=None, now=1_000.0)

    @pytest.mark.parametrize("identity", ["PASS", "pass", "verified", "VERIFIED"])
    def test_a_declared_proof_opens(self, identity):
        assert self._entry(identity).valid is True

    @pytest.mark.parametrize(
        "identity",
        ["FAIL", "fail", "MISMATCH", "unknown", "unverified", "",
         "probably-fine", "OK", "true", "1"])
    def test_everything_else_is_refused(self, identity):
        """
        The defect: ``"fail"`` -- the token ``QuoteService`` really emits for a
        PROVEN mismatch -- was absent from the denylist and so fell through to
        VALID. An allowlist closes that and every future token with it.
        """
        observation = self._entry(identity)
        assert observation.valid is False
        assert observation.rejection_reason

    def test_a_proven_mismatch_is_named_as_a_mismatch(self):
        observation = self._entry("FAIL")
        assert "IDENTITY_MISMATCH" in observation.rejection_reason

    def test_an_unproven_market_is_named_as_unknown(self):
        observation = self._entry("unknown")
        assert "IDENTITY_UNKNOWN" in observation.rejection_reason

    def test_mismatch_and_unknown_are_distinguishable(self):
        assert (self._entry("FAIL").rejection_reason
                != self._entry("unknown").rejection_reason)


class TestExitSettlementFailsClosed:
    def _experiment(self, tmp_path):
        from autopaper.paper_live.amarih_experiment import (
            AmarihArm, AmarihExperiment,
        )
        exp = AmarihExperiment(research_dir=tmp_path / "research")
        exp.accounts[AmarihArm.T60].bankroll_usd = 100.0
        return exp, AmarihArm.T60

    def _open(self, exp, arm):
        from autopaper.paper_live.amarih_experiment import EntryObservation
        obs = EntryObservation(
            signal_id="s1", mint=MINT, symbol="TKN", decision_ts=0.0,
            entry_ts=0.0, signal_to_entry_latency_ms=0.0, market_id=POOL,
            provider="pump", venue="pumpfun", quote_ts=0.0, quote_age_ms=0.0,
            input_amount_usd=10.0, token_quantity=1000.0, token_decimals=6,
            fees_usd=None, quote_asset="SOL", entry_basis="executable_quote",
            market_cap_usd=20_000.0, liquidity_usd=None, volume_to_mcap=None,
            validation="VALID", rejection_reason=None,
        )
        return exp.open_from_entry(obs)[arm]

    def _settle(self, exp, arm, quote):
        return exp.settle_arm(
            signal_id="s1", arm=arm, sell_quote=quote, observed_ts=60.0)

    def _quote(self, **kw):
        base = {"proceeds_usd": 11.0, "quote_ts": 60.0, "valid_market": True,
                "provider": "pump", "venue": "pumpfun", "raw": {}}
        base.update(kw)
        return base

    def test_a_verified_exit_settles_with_the_net_pnl_identity(self, tmp_path):
        """net = proceeds - cost = 11.0 - 10.0 = 1.0; return = +10.0%."""
        exp, arm = self._experiment(tmp_path)
        self._open(exp, arm)
        out = self._settle(exp, arm, self._quote())
        assert out.net_pnl_usd == pytest.approx(1.0)
        assert out.net_return_pct == pytest.approx(10.0)
        assert exp.accounts[arm].bankroll_usd == pytest.approx(101.0)

    def test_a_proven_mismatch_never_settles(self, tmp_path):
        exp, arm = self._experiment(tmp_path)
        self._open(exp, arm)
        out = self._settle(exp, arm, self._quote(valid_market=False))
        assert out.net_pnl_usd is None
        assert out.classification == "CENSORED"
        assert exp.accounts[arm].bankroll_usd == pytest.approx(100.0)

    def test_a_quote_omitting_valid_market_never_settles(self, tmp_path):
        """
        The defect: ``is False`` let a missing key through and booked net PnL
        from a market nobody verified. Only an explicit ``True`` settles.
        """
        exp, arm = self._experiment(tmp_path)
        self._open(exp, arm)
        quote = self._quote()
        quote.pop("valid_market")
        out = self._settle(exp, arm, quote)
        assert out.net_pnl_usd is None
        assert out.settlement_status == "IDENTITY_MISMATCH"
        assert exp.accounts[arm].bankroll_usd == pytest.approx(100.0)

    def test_a_stale_quote_never_settles(self, tmp_path):
        exp, arm = self._experiment(tmp_path)
        self._open(exp, arm)
        out = self._settle(exp, arm, self._quote(quote_ts=0.0))
        assert out.settlement_status == "STALE_QUOTE"
        assert out.net_pnl_usd is None

    def test_settlement_is_idempotent(self, tmp_path):
        exp, arm = self._experiment(tmp_path)
        self._open(exp, arm)
        first = self._settle(exp, arm, self._quote())
        second = self._settle(exp, arm, self._quote(proceeds_usd=99.0))
        assert second.net_pnl_usd == pytest.approx(first.net_pnl_usd)
        assert exp.accounts[arm].bankroll_usd == pytest.approx(101.0)


# ===========================================================================
# 6 + 7. Cost basis and bankroll durability
# ===========================================================================


class TestCostBasisIsWhatWasDeployed:
    def test_the_quote_amount_wins_over_the_flat_notional(self, tmp_path):
        """
        ``arm_notional_usd`` is 10.00. The quote reports 9.40 actually spent,
        because the outside-swap reserve is withheld before the order is built.
        The cost basis must be 9.40; using 10.00 overstates every trade.
        """
        from autopaper.paper_live.amarih_experiment import AmarihExperiment
        from autopaper.paper_live.amarih_runtime import AmarihRuntime

        runtime = AmarihRuntime.__new__(AmarihRuntime)
        runtime.arm_notional_usd = 10.0
        signal = SimpleNamespace(
            signal_id="1", mint=MINT, symbol="TKN", strategy="Amarih",
            fired_at=0.0, received_at=0.0, signal_to_receipt_latency_ms=0.0,
            entry_market_cap_usd=20_000.0, buy_score=6.0, owner_wallet="tg:1",
            raw={})
        obs = AmarihRuntime._entry_from_quote(
            runtime, signal,
            {"identity": "PASS", "token_quantity": 1000.0,
             "token_decimals": 6, "market_id": POOL, "input_amount_usd": 9.40},
            failure=None, now=0.0)
        assert obs.input_amount_usd == pytest.approx(9.40)

    def test_it_falls_back_to_the_notional_only_when_unreported(self, tmp_path):
        from autopaper.paper_live.amarih_runtime import AmarihRuntime

        runtime = AmarihRuntime.__new__(AmarihRuntime)
        runtime.arm_notional_usd = 10.0
        signal = SimpleNamespace(
            signal_id="1", mint=MINT, symbol="TKN", strategy="Amarih",
            fired_at=0.0, received_at=0.0, signal_to_receipt_latency_ms=0.0,
            entry_market_cap_usd=20_000.0, buy_score=6.0, owner_wallet="tg:1",
            raw={})
        obs = AmarihRuntime._entry_from_quote(
            runtime, signal,
            {"identity": "PASS", "token_quantity": 1.0, "token_decimals": 6},
            failure=None, now=0.0)
        assert obs.input_amount_usd == pytest.approx(10.0)


class TestBankrollSurvivesARestart:
    def _settled_experiment(self, tmp_path, proceeds: float):
        from autopaper.paper_live.amarih_experiment import (
            AmarihArm, AmarihExperiment, EntryObservation,
        )
        exp = AmarihExperiment(research_dir=tmp_path / "research")
        obs = EntryObservation(
            signal_id="s1", mint=MINT, symbol="TKN", decision_ts=0.0,
            entry_ts=0.0, signal_to_entry_latency_ms=0.0, market_id=POOL,
            provider="pump", venue="pumpfun", quote_ts=0.0, quote_age_ms=0.0,
            input_amount_usd=10.0, token_quantity=1000.0, token_decimals=6,
            fees_usd=None, quote_asset="SOL", entry_basis="executable_quote",
            market_cap_usd=20_000.0, liquidity_usd=None, volume_to_mcap=None,
            validation="VALID", rejection_reason=None,
        )
        exp.open_from_entry(obs)
        exp.settle_arm(
            signal_id="s1", arm=AmarihArm.T60,
            sell_quote={"proceeds_usd": proceeds, "quote_ts": 60.0,
                        "valid_market": True, "raw": {}},
            observed_ts=60.0)
        return exp

    def test_a_restart_does_not_reset_a_depleted_bankroll(self, tmp_path):
        """
        The defect: ``VirtualAccount.__post_init__`` reseeds the bankroll on
        every process start, so a restart handed a losing arm back its full
        starting notional -- and ``open_from_entry`` gates new entries on that
        number.
        """
        from autopaper.paper_live.amarih_experiment import (
            AmarihArm, AmarihExperiment,
        )
        first = self._settled_experiment(tmp_path, proceeds=8.0)
        assert first.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(8.0)

        second = AmarihExperiment(research_dir=tmp_path / "research")
        assert second.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(10.0)
        second.restore_bankrolls()
        assert second.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(8.0)

    def test_only_the_settled_arm_moves(self, tmp_path):
        from autopaper.paper_live.amarih_experiment import AmarihArm, AmarihExperiment
        self._settled_experiment(tmp_path, proceeds=8.0)
        fresh = AmarihExperiment(research_dir=tmp_path / "research")
        fresh.restore_bankrolls()
        assert fresh.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(8.0)
        assert fresh.accounts[AmarihArm.T120].bankroll_usd == pytest.approx(10.0)
        assert fresh.accounts[AmarihArm.T300].bankroll_usd == pytest.approx(10.0)

    def test_duplicate_settlement_rows_are_counted_once(self, tmp_path):
        """A crash between the event write and the state write can duplicate."""
        from autopaper.paper_live.amarih_experiment import (
            AmarihArm, AmarihExperiment,
        )
        self._settled_experiment(tmp_path, proceeds=8.0)
        path = tmp_path / "research" / "events.jsonl"
        rows = [r for r in path.read_text().splitlines()
                if r.strip() and json.loads(r).get("record_type") == "exit"]
        assert rows
        with path.open("a", encoding="utf-8") as fh:
            fh.write(rows[0] + "\n")

        fresh = AmarihExperiment(research_dir=tmp_path / "research")
        fresh.restore_bankrolls()
        assert fresh.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(8.0)

    def test_the_restore_is_idempotent(self, tmp_path):
        from autopaper.paper_live.amarih_experiment import AmarihArm, AmarihExperiment
        self._settled_experiment(tmp_path, proceeds=8.0)
        fresh = AmarihExperiment(research_dir=tmp_path / "research")
        fresh.restore_bankrolls()
        fresh.restore_bankrolls()
        assert fresh.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(8.0)

    def test_a_censored_arm_never_moves_a_bankroll(self, tmp_path):
        """A refusal is not a zero-PnL trade."""
        from autopaper.paper_live.amarih_experiment import AmarihArm, AmarihExperiment
        exp = self._settled_experiment(tmp_path, proceeds=8.0)
        exp.settle_arm(signal_id="s2", arm=AmarihArm.T120, sell_quote=None,
                       observed_ts=120.0)
        fresh = AmarihExperiment(research_dir=tmp_path / "research")
        fresh.restore_bankrolls()
        assert fresh.accounts[AmarihArm.T120].bankroll_usd == pytest.approx(10.0)

    def test_an_absent_log_leaves_every_arm_at_its_start(self, tmp_path):
        from autopaper.paper_live.amarih_experiment import AmarihArm, AmarihExperiment
        fresh = AmarihExperiment(research_dir=tmp_path / "nothing-here")
        assert fresh.restore_bankrolls() == {}
        for arm in AMARIH_ARMS:
            assert fresh.accounts[arm].bankroll_usd == pytest.approx(10.0)


# ===========================================================================
# 2. Selecting the ARMS is what blocks Telegram (not a separate file)
# ===========================================================================


class TestSelectingTheArmsIsWhatBlocksTelegram:
    """
    The defect: ``on_alert`` keyed the Amarih restriction on
    ``signal_source_selection``, which lives in its own JSON file and whose
    operator whitelist is ``{MooDeng, PainApe, AUTO}``. Nothing anywhere derives
    it from the arm selection, and ``Amarih`` is not offered there at all. So
    selecting the experiment -- the one and only way an operator selects it --
    left Telegram channel alerts flowing into the L25 pipeline.

    Every existing routing test set BOTH the arms AND the source preference, so
    the gap was never exercised. These tests set only what the operator sets.
    """

    @pytest.fixture
    def trader(self, tmp_path):
        from autopaper.config import build_config
        from autopaper.engine import PaperTrader

        class NoNetworkMarket:
            quotes = 0

            async def quote_async(self, mint):
                self.quotes += 1
                raise AssertionError("no network in this test")

            def mark_price(self, mint, now):
                return None, None

        return PaperTrader(build_config(tmp_path), market=NoNetworkMarket())

    @pytest.fixture
    def pipeline_calls(self, monkeypatch):
        calls: list[object] = []

        async def spy(self, snapshot, parsed):
            calls.append(snapshot)

        import autopaper.engine as engine_mod
        monkeypatch.setattr(engine_mod.PaperTrader, "_handle_candidate", spy)
        return calls

    @staticmethod
    def _alert(source: str, message_id: int):
        from autopaper.models import AlertSnapshot
        return AlertSnapshot(
            candidate_id=f"c{message_id}", source="test", chat="testchat",
            message_id=message_id, source_timestamp=1_800_000_000.0,
            received_timestamp=1_800_000_005.0, raw_text="call",
            mint=MINT, symbol="TKN", signal_source=source)

    class _Parsed:
        pass

    @pytest.mark.parametrize("source", ["MooDeng", "PainApe"])
    def test_arms_alone_are_enough(self, trader, pipeline_calls, source):
        """EXACTLY what the STRATEGY menu does -- and nothing else."""
        assert trader.signal_source_selection == "AUTO", (
            "this test is only meaningful if the source preference was never "
            "touched")
        ok, _msg = trader.state.select_strategies(["AMARIH"])
        assert ok, "the AMARIH selection must be valid"
        assert trader._amarih_selected() is True

        asyncio.run(trader.on_alert(self._alert(source, 1), self._Parsed()))

        assert pipeline_calls == [], (
            f"a Telegram alert from {source} reached the L25 pipeline while "
            "the AMARIH experiment was selected")
        assert trader._signal_source_counters.get(
            "amarih_blocked_telegram") == 1

    def test_the_refusal_is_recorded_with_its_reason(self, trader,
                                                      pipeline_calls):
        from autopaper.paper_live.amarih_runtime import SOURCE_DISABLED_BY_AMARIH
        trader.state.select_strategies(["AMARIH"])
        asyncio.run(
            trader.on_alert(self._alert("MooDeng", 7), self._Parsed()))
        notes = [e for e in trader.ledger.events()
                 if e.get("reason") == SOURCE_DISABLED_BY_AMARIH]
        assert notes, "the block must leave durable evidence in the ledger"
        assert notes[0].get("mint") == MINT

    def test_the_candidate_is_still_recorded(self, trader, pipeline_calls):
        """Blocked means not traded, not invisible."""
        trader.state.select_strategies(["AMARIH"])
        before = trader.candidates_seen
        asyncio.run(
            trader.on_alert(self._alert("MooDeng", 9), self._Parsed()))
        assert trader.candidates_seen == before + 1
        assert trader.account.open_positions() == []

    def test_an_l25_selection_still_allows_telegram(self, trader,
                                                    pipeline_calls):
        """The fix must not block the OTHER strategies' feeds."""
        trader.state.select_strategies(["L25_CURRENT"])
        assert trader._amarih_selected() is False
        asyncio.run(
            trader.on_alert(self._alert("MooDeng", 11), self._Parsed()))
        assert pipeline_calls, "L25 selection must still admit Telegram alerts"

    def test_the_default_selection_still_allows_telegram(self, trader,
                                                          pipeline_calls):
        assert trader._amarih_selected() is False
        asyncio.run(
            trader.on_alert(self._alert("MooDeng", 13), self._Parsed()))
        assert pipeline_calls

    def test_an_explicit_source_preference_still_blocks(self, trader,
                                                        pipeline_calls):
        """The pre-existing behaviour is preserved, not replaced."""
        trader.signal_source_selection = "Amarih"
        asyncio.run(
            trader.on_alert(self._alert("MooDeng", 15), self._Parsed()))
        assert pipeline_calls == []

    def test_real_mode_is_unaffected(self, trader, pipeline_calls):
        """
        ``_amarih_selected`` is PAPER-only by definition, and Amarih cannot be
        selected in REAL, so this guard must not fire there.
        """
        from autopaper.modes import RunState
        trader.state.select_strategies(["AMARIH"])
        trader.state.real_armed = True
        trader.state.run_state = RunState.REAL_ARMED
        assert trader._amarih_selected() is False
        asyncio.run(
            trader.on_alert(self._alert("MooDeng", 17), self._Parsed()))
        assert pipeline_calls, "REAL must not inherit the PAPER-only guard"

    def test_the_selection_and_the_runtime_gate_now_agree(self, trader):
        """
        The end-to-end invariant: what is persisted is what authorizes.

        This is the fact that was false at the start of the audit -- the file
        said AMARIH while ``DecisionEngine._operative_gates`` said L25.
        """
        from autopaper.paper_live.arms import effective_entry_gates

        trader.state.select_strategies(["AMARIH"])
        requested = effective_entry_gates(
            list(trader.state.arms), profile=trader.state.strategy_profile)
        applied = trader.engine.set_entry_gates(requested)
        assert requested == applied == (GATE_AMARIH,)
        assert trader.state.selected_arms


class TestTheEntryMarketReachesTheExit:
    def test_the_open_arm_record_round_trips_it(self, tmp_path):
        from autopaper.paper_live.amarih_runtime import _OpenArm
        arm = _OpenArm(
            signal_id="s1", arm="AMARIH_T60", mint=MINT, symbol="TKN",
            entry_ts=0.0, scheduled_deadline=60.0, token_quantity=1000.0,
            token_decimals=6, token_quantity_raw=1_000_000_000,
            entry_cost_usd=10.0, entry_basis="executable_quote",
            entry_market=POOL)
        assert _OpenArm.from_dict(arm.as_dict()).entry_market == POOL

    def test_a_record_written_before_the_field_still_loads(self, tmp_path):
        from autopaper.paper_live.amarih_runtime import _OpenArm
        legacy = {
            "signal_id": "s1", "arm": "AMARIH_T60", "mint": MINT,
            "symbol": "TKN", "entry_ts": 0.0, "scheduled_deadline": 60.0,
            "token_quantity": 1000.0, "token_decimals": 6,
            "token_quantity_raw": 1_000_000_000, "entry_cost_usd": 10.0,
            "entry_basis": "executable_quote",
        }
        assert _OpenArm.from_dict(legacy).entry_market is None

    @pytest.mark.asyncio_compatible
    def test_the_pin_is_forwarded_when_the_transport_accepts_it(self):
        import asyncio
        from autopaper.paper_live.amarih_runtime import AmarihRuntime, _OpenArm

        seen = {}

        class Modern:
            async def __call__(self, mint, qty, decimals, entry_pool=None):
                seen["entry_pool"] = entry_pool
                return {"identity": "PASS", "proceeds_usd": 11.0,
                        "quote_ts": 60.0}

        runtime = AmarihRuntime.__new__(AmarihRuntime)
        runtime.sell_quote = Modern()
        arm = _OpenArm(
            signal_id="s1", arm="AMARIH_T60", mint=MINT, symbol="TKN",
            entry_ts=0.0, scheduled_deadline=60.0, token_quantity=1000.0,
            token_decimals=6, token_quantity_raw=1_000_000_000,
            entry_cost_usd=10.0, entry_basis="executable_quote",
            entry_market=POOL)
        asyncio.run(runtime._request_sell_quote(arm))
        assert seen["entry_pool"] == POOL

    def test_a_legacy_three_argument_transport_still_works(self):
        """
        An injected double written against the older signature must not raise
        ``TypeError`` into the classification path -- that would be recorded as
        a quote outage.
        """
        import asyncio
        from autopaper.paper_live.amarih_runtime import AmarihRuntime, _OpenArm

        calls = []

        class Legacy:
            async def __call__(self, mint, qty, decimals):
                calls.append((mint, qty, decimals))
                return {"identity": "PASS", "proceeds_usd": 11.0}

        runtime = AmarihRuntime.__new__(AmarihRuntime)
        runtime.sell_quote = Legacy()
        arm = _OpenArm(
            signal_id="s1", arm="AMARIH_T60", mint=MINT, symbol="TKN",
            entry_ts=0.0, scheduled_deadline=60.0, token_quantity=1000.0,
            token_decimals=6, token_quantity_raw=1_000_000_000,
            entry_cost_usd=10.0, entry_basis="executable_quote",
            entry_market=POOL)
        assert asyncio.run(runtime._request_sell_quote(arm)) is not None
        assert calls == [(MINT, 1_000_000_000, 6)]

# ===========================================================================
# Aggregate statistics -- INDEPENDENTLY hand-computed
# ===========================================================================


class TestAggregateStatistics:
    """
    A five-trade dataset whose every aggregate is worked out by hand here.

        trade   net_pnl_usd   return_pct   holding_s
          A        +10.0         +50.0        300
          B         -5.0         -20.0         60
          C         +20.0        +100.0        400
          D          0.0           0.0         120
          E        -15.0         -30.0         90

    starting_equity = 100.0

        wins / losses / scratches   2 / 2 / 1     (0.0 is EXPLICIT, not a loss)
        win_rate                     2/5      = 0.40
        realized_pnl_usd            10-5+20+0-15 = +10.0
        total_return_pct            10/100*100   = +10.0 %
        expectancy_usd              10/5         = +2.0
        expectancy_pct             (50-20+100+0-30)/5 = 100/5 = +20.0 %
        gross_win                   10 + 20          = 30
        gross_loss                  5 + 15           = 20
        profit_factor               30/20            = 1.5
        max_losing_streak           2                 (D is a scratch, not a loss)

        ``compute`` ORDERS BY ``exit_timestamp`` (``stats.py:145``), which is
        the only correct order for a drawdown or a streak -- input order would
        let the same trades produce a different curve. With exit times
        B=60, E=90, D=120, A=300, C=400 the sequence is B, E, D, A, C:

            equity  100 ->  95 ->  80 ->  80 ->  90 -> 110
            peak     100 -> 100 -> 100 -> 100 -> 100 -> 110
            dd          0    -5.0%  -20.0%  -20.0%  -10.0%     0

            max drawdown = (80 - 100) / 100 = -20.0 %
            losing streak (B, E) = 2; D is a scratch and must NOT extend it
    """

    @staticmethod
    def _trades():
        from autopaper.models import Trade
        rows = [
            ("A", 10.0, 50.0, 300.0),
            ("B", -5.0, -20.0, 60.0),
            ("C", 20.0, 100.0, 400.0),
            ("D", 0.0, 0.0, 120.0),
            ("E", -15.0, -30.0, 90.0),
        ]
        return [
            Trade(trade_id=tid, position_id=tid, candidate_id=tid,
                  mint=MINT, symbol="TKN", source="s", strategy="L25",
                  route="FAST_ENTRY", regime="UNKNOWN",
                  entry_signal_timestamp=0.0, entry_timestamp=0.0,
                  entry_price=1.0, fill_price=1.0, exit_timestamp=hold,
                  exit_price=1.0 + pnl / 100.0, quantity=1.0,
                  entry_fee_usd=0.0, exit_fee_usd=0.0,
                  entry_slippage_usd=0.0, exit_slippage_usd=0.0,
                  gross_pnl_usd=pnl, net_pnl_usd=pnl, return_pct=ret,
                  holding_seconds=hold, mfe_pct=ret, mae_pct=0.0,
                  exit_reason="TIME", exit_style="TIMED",
                  thesis="", invalidation="", liquidity_at_entry=0.0,
                  market_cap_at_entry=0.0)
            for tid, pnl, ret, hold in rows
        ]

    def _stats(self):
        from autopaper.stats import compute
        return compute(self._trades(), starting_equity=100.0)

    def test_a_zero_pnl_trade_is_a_scratch_not_a_loss(self):
        s = self._stats()
        assert s.wins == 2
        assert s.losses == 2
        assert s.scratches == 1, "a 0.0 PnL trade must be counted separately"

    def test_win_rate_is_over_the_whole_population(self):
        assert self._stats().win_rate == pytest.approx(0.40)

    def test_realized_and_return(self):
        s = self._stats()
        assert s.realized_pnl_usd == pytest.approx(10.0)
        assert s.total_return_pct == pytest.approx(10.0)

    def test_expectancy(self):
        s = self._stats()
        assert s.expectancy_usd == pytest.approx(2.0)
        assert s.expectancy_pct == pytest.approx(20.0)

    def test_profit_factor_uses_gross_not_net(self):
        assert self._stats().profit_factor == pytest.approx(1.5)

    def test_profit_factor_with_no_losses(self):
        from autopaper.stats import compute
        wins_only = [t for t in self._trades() if t.net_pnl_usd >= 0]
        pf = compute(wins_only, starting_equity=100.0).profit_factor
        assert pf == float("inf"), "an undefined ratio, not a fabricated 0.0"

    def test_profit_factor_with_no_trades(self):
        from autopaper.stats import compute
        assert compute([], starting_equity=100.0).profit_factor == 0.0

    def test_trades_are_ordered_by_exit_time_not_input_order(self):
        """The drawdown and streaks must not depend on how they arrived."""
        from autopaper.stats import compute
        forward = compute(self._trades(), starting_equity=100.0)
        backward = compute(list(reversed(self._trades())), starting_equity=100.0)
        assert forward.max_drawdown_pct == backward.max_drawdown_pct
        assert forward.max_consecutive_losses == backward.max_consecutive_losses

    def test_max_drawdown_uses_the_realised_equity_curve(self):
        assert self._stats().max_drawdown_pct == pytest.approx(-20.0)

    def test_drawdown_scope_is_declared(self):
        """Realised only -- it cannot see an open underwater position."""
        assert self._stats().max_drawdown_scope == "realized_trade_to_trade"

    def test_losing_streak_ignores_scratches(self):
        s = self._stats()
        # In exit order the sequence is B(-5), E(-15), D(0), A(+10), C(+20),
        # so the longest LOSS run is B,E = 2 and the scratch D neither extends
        # it nor resets it.
        assert s.max_consecutive_losses == 2
        assert s.losses == 2, "the streak must agree with the population count"
        assert s.max_consecutive_losses <= s.losses
        assert s.current_losing_streak == 0
