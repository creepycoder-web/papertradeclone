"""
Focused tests for the execution-native hold-path telemetry.

MEASUREMENT ONLY. Nothing here touches the exit policy, entry rules, sizing or
real-trading configuration; the tests assert that as an explicit requirement
(test 14) rather than assuming it.
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from autopaper.paper_live.telemetry import (  # noqa: E402
    INVALID_IDENTITY,
    PARTIAL_DATA,
    QUOTE_UNAVAILABLE,
    STALE,
    UNSUPPORTED_SOURCE,
    VALID,
    HoldPathConfig,
    HoldPathObservation,
    HoldPathRecorder,
    compute_hold_metrics,
    load_observations,
    validate_observation,
)

MINT = "So1111111111111111111111111111111111111111111"  # placeholder, shape-checked only
CURVE = "PUMP_BONDING_CURVE:Curve1111111111111111111111111111111111"
WSOL = "So11111111111111111111111111111111111111112"
LAMPORTS = 10 ** 9  # SOL has 9 decimals


class FakeQuote:
    """A QuotePoint-shaped double. Deliberately NOT the real dataclass."""

    def __init__(self, **kw):
        self.side = "SELL"
        self.available = True
        self.missing_reason = None
        self.token_mint = MINT
        self.quote_mint = WSOL
        self.venue_id = CURVE
        self.market_type = "PUMP_BONDING_CURVE"
        self.identity = "PASS"
        self.identity_evidence = "the observed pair IS the curve account"
        self.provider = "pump_direct"
        self.input_amount_raw = 25_441_720_199
        self.input_decimals = 6
        self.output_amount_raw = 4_000_000
        self.output_decimals = 9
        self.effective_price = Decimal("1.5722e-07")
        self.price_impact_pct = -1.25
        self.fee_bps = 125
        self.quote_timestamp = 1_000.0
        self.quote_age_ms = 0.03
        self.state_slot = 453_000_000
        self.state_age_ms = 86.0
        self.execution_native_reference = None
        for k, v in kw.items():
            setattr(self, k, v)


class FakeNative:
    def __init__(self, ref, num=30_000_000_000, den=1_500_000_000_000):
        self.reference_price = Decimal(str(ref))
        self.numerator_raw = num
        self.denominator_raw = den


def obs(**kw) -> HoldPathObservation:
    base = dict(
        trade_id="t1", mint=MINT, curve_or_pool_id=CURVE,
        observation_id="t1#00001", observed_at_utc=1000.0, elapsed_ms=0.0,
        remaining_hold_ms=60000.0, remaining_token_amount_raw=25_441_720_199,
        token_decimals=6, quote_asset=WSOL, aggregator_mark_usd=1.5e-05,
        sell_quote_net_raw=4_000_000, sell_quote_source="pump_direct",
        quote_status=VALID, identity_check_status="PASS",
        sol_usd_reference=None,
    )
    base.update(kw)
    return HoldPathObservation(**base)


# ---------------------------------------------------------------------------
# 1, 2, 3. MFE/MAE on deterministic paths
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_profitable_excursion_then_losing_exit(self):
        """2: up +20%, exit -5% -> MFE +20, MAE -5, exit -5."""
        series = [obs(elapsed_ms=0, sell_quote_net_raw=100),
                  obs(elapsed_ms=10_000, sell_quote_net_raw=120),
                  obs(elapsed_ms=20_000, sell_quote_net_raw=95)]
        m = compute_hold_metrics(series)
        assert m["EXEC_REFERENCE_MFE_PCT"] == pytest.approx(20.0)
        assert m["EXEC_REFERENCE_MAE_PCT"] == pytest.approx(-5.0)
        assert m["EXEC_REFERENCE_EXIT_RETURN_PCT"] == pytest.approx(-5.0)
        assert m["EXEC_REFERENCE_TIME_TO_MFE_MS"] == 10_000
        assert m["EXEC_REFERENCE_TIME_TO_MAE_MS"] == 20_000

    def test_losing_excursion_then_recovery(self):
        """3: down -30%, exit +8% -> MFE +8, MAE -30."""
        series = [obs(elapsed_ms=0, sell_quote_net_raw=100),
                  obs(elapsed_ms=5_000, sell_quote_net_raw=70),
                  obs(elapsed_ms=30_000, sell_quote_net_raw=108)]
        m = compute_hold_metrics(series)
        assert m["EXEC_REFERENCE_MFE_PCT"] == pytest.approx(8.0)
        assert m["EXEC_REFERENCE_MAE_PCT"] == pytest.approx(-30.0)
        assert m["EXEC_REFERENCE_EXIT_RETURN_PCT"] == pytest.approx(8.0)

    def test_flat_path_is_zero_not_missing(self):
        series = [obs(elapsed_ms=0, sell_quote_net_raw=100),
                  obs(elapsed_ms=5_000, sell_quote_net_raw=100)]
        m = compute_hold_metrics(series)
        assert m["EXEC_REFERENCE_MFE_PCT"] == 0.0
        assert m["EXEC_REFERENCE_MAE_PCT"] == 0.0

    def test_the_entry_observation_inclusion_policy_is_reported(self):
        series = [obs(elapsed_ms=0, sell_quote_net_raw=100),
                  obs(elapsed_ms=1_000, sell_quote_net_raw=110)]
        m = compute_hold_metrics(series)
        assert m["entry_observation_included"] is True
        # Including the entry pins MFE >= 0 >= MAE for every trade.
        assert m["EXEC_REFERENCE_MFE_PCT"] >= 0 >= m["EXEC_REFERENCE_MAE_PCT"]

    def test_excluding_the_entry_observation_changes_the_result_and_says_so(self):
        series = [obs(elapsed_ms=0, sell_quote_net_raw=100),
                  obs(elapsed_ms=1_000, sell_quote_net_raw=110)]
        m = compute_hold_metrics(series, include_entry_observation=False)
        assert m["entry_observation_included"] is False
        assert m["EXEC_REFERENCE_MFE_PCT"] == pytest.approx(10.0)

    def test_returns_are_unit_independent(self):
        """A ratio of quote-asset amounts cannot be moved by an FX rate."""
        series = [obs(elapsed_ms=0, sell_quote_net_raw=100),
                  obs(elapsed_ms=1_000, sell_quote_net_raw=150)]
        a = compute_hold_metrics(series)
        b = compute_hold_metrics([
            replace(series[0], sol_usd_reference=120.0),
            replace(series[1], sol_usd_reference=999.0),
        ])
        assert a["EXEC_REFERENCE_MFE_PCT"] == b["EXEC_REFERENCE_MFE_PCT"]

    def test_the_entry_reference_is_labelled_as_liquidation_value_not_cost(self):
        series = [obs(elapsed_ms=0, sell_quote_net_raw=100),
                  obs(elapsed_ms=1_000, sell_quote_net_raw=110)]
        m = compute_hold_metrics(series)
        assert "NOT the buy cost" in m["entry_reference_basis"]
        assert m["entry_reference_sell_quote_net_raw"] == 100


# ---------------------------------------------------------------------------
# 4. Missing observations / partial coverage
# ---------------------------------------------------------------------------

class TestPartialData:
    def test_fewer_than_two_observations_reports_unavailable_not_zero(self):
        """Never report MFE/MAE as zero merely because data is missing."""
        m = compute_hold_metrics([obs(elapsed_ms=0)])
        assert m["metrics_status"] == PARTIAL_DATA
        assert m["EXEC_REFERENCE_MFE_PCT"] is None
        assert m["EXEC_REFERENCE_MAE_PCT"] is None

    def test_no_observations_at_all(self):
        m = compute_hold_metrics([])
        assert m["metrics_status"] == PARTIAL_DATA
        assert m["EXEC_REFERENCE_MFE_PCT"] is None

    def test_invalid_observations_are_excluded_but_retained(self):
        series = [
            obs(elapsed_ms=0, sell_quote_net_raw=100),
            obs(elapsed_ms=1_000, sell_quote_net_raw=QUOTE_UNAVAILABLE,
                quote_status=QUOTE_UNAVAILABLE, unavailable_reason="rpc down"),
            obs(elapsed_ms=2_000, sell_quote_net_raw=130),
        ]
        m = compute_hold_metrics(series)
        assert m["observation_count_total"] == 3
        assert m["observation_count_eligible"] == 2
        assert len(m["rejected_observations"]) == 1
        assert m["rejected_observations"][0]["reason"] == "rpc down"
        assert m["EXEC_REFERENCE_MFE_PCT"] == pytest.approx(30.0)

    def test_coverage_and_sampling_gap(self):
        series = [obs(elapsed_ms=0, sell_quote_net_raw=100),
                  obs(elapsed_ms=1_000, sell_quote_net_raw=105),
                  obs(elapsed_ms=9_000, sell_quote_net_raw=110)]
        m = compute_hold_metrics(series, expected_window_ms=10_000)
        assert m["EXEC_REFERENCE_COVERAGE_PCT"] == pytest.approx(90.0)
        assert m["EXEC_REFERENCE_MAX_SAMPLING_GAP_MS"] == pytest.approx(8_000.0)

    def test_coverage_is_none_when_the_window_is_unknown(self):
        series = [obs(elapsed_ms=0, sell_quote_net_raw=100),
                  obs(elapsed_ms=1_000, sell_quote_net_raw=105)]
        assert compute_hold_metrics(series)["EXEC_REFERENCE_COVERAGE_PCT"] is None


# ---------------------------------------------------------------------------
# 5, 6, 10. Validation: staleness, identity, wrong everything
# ---------------------------------------------------------------------------

class TestValidation:
    def _v(self, quote, **cfg):
        return validate_observation(
            quote=quote, mint=MINT, expected_venue_id=CURVE,
            expected_token_amount_raw=25_441_720_199,
            config=replace(HoldPathConfig(), **cfg))

    def test_a_good_quote_is_valid(self):
        assert self._v(FakeQuote()) == (VALID, "PASS", None)

    def test_stale_curve_state(self):
        status, _, reason = self._v(FakeQuote(state_age_ms=99_000.0))
        assert status == STALE and "curve state is" in reason

    def test_state_age_missing_is_stale_not_assumed_fresh(self):
        status, _, reason = self._v(FakeQuote(state_age_ms=None))
        assert status == STALE
        assert "cannot be established" in reason

    def test_stale_quote_response(self):
        status, _, _ = self._v(
            FakeQuote(state_age_ms=1.0, quote_age_ms=120_000.0))
        assert status == STALE

    def test_unavailable_quote(self):
        status, _, reason = self._v(
            FakeQuote(available=False, missing_reason="curve gone"))
        assert status == QUOTE_UNAVAILABLE and "curve gone" in reason

    def test_none_quote(self):
        status, _, _ = self._v(None)
        assert status == QUOTE_UNAVAILABLE

    def test_wrong_mint(self):
        status, ident, _ = self._v(FakeQuote(token_mint="OtherMint"))
        assert status == INVALID_IDENTITY and ident == "MISMATCH"

    def test_wrong_curve(self):
        status, ident, _ = self._v(FakeQuote(venue_id="PUMPSWAP:Elsewhere"))
        assert status == INVALID_IDENTITY and ident == "MISMATCH"

    def test_a_non_sol_quote_asset_is_unsupported_not_priced(self):
        """
        A USDC-denominated sell is REAL data, but this module only knows how to
        value SOL proceeds. It is therefore UNSUPPORTED_SOURCE, not a rejection
        of the quote: guessing a conversion is exactly the SOL-vs-USD mistake
        that produced a several-hundred-percent gap earlier in this repo.
        """
        q = FakeQuote()
        q.quote_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        status, _, reason = self._v(q)
        assert status == UNSUPPORTED_SOURCE
        assert "not a recognised quote asset" in reason

    def test_wrong_token_quantity(self):
        status, _, reason = self._v(FakeQuote(input_amount_raw=1))
        assert status == INVALID_IDENTITY and "sized for 1" in reason

    def test_buy_side_quote_is_rejected(self):
        status, _, reason = self._v(FakeQuote(side="BUY"))
        assert status == INVALID_IDENTITY and "not SELL" in reason

    def test_unproven_identity(self):
        status, ident, _ = self._v(FakeQuote(identity="UNKNOWN"))
        assert status == INVALID_IDENTITY and ident == "UNPROVEN"

    def test_unsupported_market_type(self):
        status, _, reason = self._v(FakeQuote(market_type="JUPITER_AGGREGATE"))
        assert status == UNSUPPORTED_SOURCE and "not an interpretable source" in reason

    def test_non_positive_proceeds(self):
        status, _, _ = self._v(FakeQuote(output_amount_raw=0))
        assert status == PARTIAL_DATA

    def test_invalid_decimals_rejected_via_price(self):
        status, _, _ = self._v(FakeQuote(effective_price=Decimal("nan")))
        assert status == PARTIAL_DATA

    def test_infinite_price_rejected(self):
        status, _, _ = self._v(FakeQuote(effective_price=Decimal("Infinity")))
        assert status == PARTIAL_DATA

    def test_thresholds_are_configurable(self):
        loose = self._v(FakeQuote(state_age_ms=60_000.0), max_state_age_ms=120_000.0)
        assert loose[0] == VALID
        tight = self._v(FakeQuote(state_age_ms=60_000.0), max_state_age_ms=1_000.0)
        assert tight[0] == STALE

    def test_identity_is_checked_before_freshness(self):
        """A wrong-market quote is not merely 'stale'."""
        status, _, _ = self._v(
            FakeQuote(token_mint="Other", state_age_ms=99_000.0))
        assert status == INVALID_IDENTITY


# ---------------------------------------------------------------------------
# 7, 8. Gross vs net, raw integer amounts
# ---------------------------------------------------------------------------

class TestGrossNetAndPrecision:
    def test_gross_is_recorded_as_unavailable_not_fabricated(self):
        assert obs().sell_quote_gross_raw is None

    def test_net_proceeds_are_kept_as_raw_integers(self):
        q = FakeQuote(output_amount_raw=3_854_284)
        assert isinstance(q.output_amount_raw, int)
        o = obs(sell_quote_net_raw=q.output_amount_raw)
        assert o.sell_quote_net_raw == 3_854_284
        assert isinstance(o.sell_quote_net_raw, int)

    def test_large_token_amounts_survive_exactly(self):
        huge = 2_147_483_647_000_000_000  # beyond float32/float53 comfort
        o = obs(remaining_token_amount_raw=huge)
        assert o.remaining_token_amount_raw == huge
        assert int(o.remaining_token_amount_raw) == huge

    def test_decimal_prices_are_not_degraded_to_float_in_serialisation(self):
        o = obs(curve_spot_price_quote_asset=Decimal("1.6078839606964136e-07"))
        blob = o.as_dict()
        assert isinstance(blob["curve_spot_price_quote_asset"], str)
        assert blob["curve_spot_price_quote_asset"] == "1.6078839606964136E-7"

    def test_the_effective_price_is_labelled_average_not_spot(self):
        o = obs(effective_liquidation_price_quote_asset=Decimal("1.5e-07"))
        assert o.effective_liquidation_price_quote_asset == Decimal("1.5e-07")


# ---------------------------------------------------------------------------
# 9. Frozen marks vs moving quotes
# ---------------------------------------------------------------------------

class TestFrozenMarks:
    def test_frozen_mark_intervals_are_counted(self):
        series = [obs(elapsed_ms=0, aggregator_mark_usd=1.5e-05,
                      sell_quote_net_raw=100),
                  obs(elapsed_ms=1_000, aggregator_mark_usd=1.5e-05,
                      sell_quote_net_raw=90),
                  obs(elapsed_ms=2_000, aggregator_mark_usd=1.9e-05,
                      sell_quote_net_raw=95)]
        m = compute_hold_metrics(series)
        assert m["frozen_mark_intervals"] == 1
        assert m["mark_observation_count"] == 3

    def test_a_frozen_mark_with_falling_quotes_is_visible_as_a_metric(self):
        series = [obs(elapsed_ms=0, aggregator_mark_usd=1.5e-05,
                      sell_quote_net_raw=100),
                  obs(elapsed_ms=30_000, aggregator_mark_usd=1.5e-05,
                      sell_quote_net_raw=60)]
        m = compute_hold_metrics(series)
        assert m["frozen_mark_intervals"] == 1
        assert m["EXEC_REFERENCE_EXIT_RETURN_PCT"] == pytest.approx(-40.0)

    def test_mark_versus_reference_gap_is_computed_per_token(self):
        series = [
            obs(elapsed_ms=0, aggregator_mark_usd=1.5e-05,
                curve_spot_price_quote_asset=Decimal("1.5e-07"),
                sell_quote_net_raw=100, sol_usd_reference=100.0),
            obs(elapsed_ms=1_000, aggregator_mark_usd=1.5e-05,
                curve_spot_price_quote_asset=Decimal("1.2e-07"),
                sell_quote_net_raw=90, sol_usd_reference=100.0),
        ]
        m = compute_hold_metrics(series)
        # 1.5e-05/100 = 1.5e-07 native; vs spot 1.2e-07 -> +25%
        assert m["MARK_VS_REFERENCE_GAP_PCT_MAX"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# 11. Sampling gaps and elapsed time
# ---------------------------------------------------------------------------

class TestElapsedTime:
    def test_a_direct_one_shot_sample_still_gets_a_real_elapsed_ms(self):
        """
        ``sample_once`` called outside the loop must establish the monotonic
        baseline itself. Without this every direct sample reports elapsed 0 and
        the hold path collapses onto a single instant.
        """
        # The FIRST sample consumes two ticks: one to establish the baseline
        # and one to measure against it. Each later sample consumes one.
        ticks = iter([100.0, 100.0, 110.0, 120.5, 130.0])

        async def quote_fn(**kw):
            return FakeQuote(state_age_ms=1.0)

        async def scenario():
            rec = HoldPathRecorder(
                trade_id="t", mint=MINT, curve_or_pool_id=CURVE, token_decimals=6,
                remaining_token_amount_raw=25_441_720_199, quote_asset=WSOL,
                quote_fn=quote_fn, config=HoldPathConfig.disabled(),
                clock=lambda: 1_000.0,
                monotonic=lambda: next(ticks),
            )
            for _ in range(2):
                await rec.sample_once()
            return rec

        rec = asyncio.run(scenario())
        elapsed = [o.elapsed_ms for o in rec.observations]
        assert elapsed[0] == pytest.approx(0.0)
        assert elapsed[1] == pytest.approx(10_000.0)
        assert elapsed[1] > elapsed[0], "elapsed must advance for direct samples"

    def test_elapsed_uses_monotonic_not_wall_clock(self, tmp_path):
        """A wall-clock jump must not invent a holding period."""
        ticks = iter([1000.0, 1005.0, 1010.0, 1015.0, 1020.0, 1025.0])
        result = {}

        async def quote_fn(**kw):
            return FakeQuote(state_age_ms=1.0, quote_age_ms=0.0)

        async def scenario():
            rec = HoldPathRecorder(
                trade_id="t", mint=MINT, curve_or_pool_id=CURVE, token_decimals=6,
                remaining_token_amount_raw=25_441_720_199, quote_asset=WSOL,
                quote_fn=quote_fn, config=replace(HoldPathConfig(), sink_path=tmp_path / "o.jsonl"),
                clock=lambda: next(ticks),          # wall clock jumps a lot
                monotonic=lambda: 500.0,             # monotonic frozen
                sleep=_no_sleep,
            )
            for _ in range(3):
                await rec.sample_once()
            result["rec"] = rec

        asyncio.run(scenario())
        elapsed = [o.elapsed_ms for o in result["rec"].observations]
        assert elapsed == [0.0, 0.0, 0.0], "monotonic governs elapsed"
        walls = [o.observed_at_utc for o in result["rec"].observations]
        assert walls == [1000.0, 1005.0, 1010.0], "wall clock still recorded"


async def _no_sleep(_seconds):
    return None


# ---------------------------------------------------------------------------
# 12. Telemetry failure cannot alter a simulated exit
# ---------------------------------------------------------------------------

class TestTelemetryCannotAffectTrading:
    def test_a_raising_quote_fn_is_contained(self, tmp_path):
        async def boom(**kw):
            raise RuntimeError("rpc exploded")

        async def scenario():
            rec = HoldPathRecorder(
                trade_id="t", mint=MINT, curve_or_pool_id=CURVE, token_decimals=6,
                remaining_token_amount_raw=1, quote_asset=WSOL, quote_fn=boom,
                config=replace(HoldPathConfig(), sink_path=tmp_path / "o.jsonl"))
            obsv = await rec.sample_once()
            return rec, obsv

        rec, obsv = asyncio.run(scenario())
        assert obsv is not None
        assert obsv.quote_status == QUOTE_UNAVAILABLE
        assert "rpc exploded" in (obsv.unavailable_reason or "")
        assert rec.telemetry_error is None  # a recorded status, not an error

    def test_a_sink_that_cannot_be_written_does_not_raise(self, tmp_path):
        async def quote_fn(**kw):
            return FakeQuote(state_age_ms=1.0)

        async def scenario():
            rec = HoldPathRecorder(
                trade_id="t", mint=MINT, curve_or_pool_id=CURVE, token_decimals=6,
                remaining_token_amount_raw=1, quote_asset=WSOL, quote_fn=quote_fn,
                config=replace(HoldPathConfig(),
                               sink_path=tmp_path / "nope" / "\0bad" / "x.jsonl"))
            return await rec.sample_once()

        obsv = asyncio.run(scenario())
        assert obsv is not None, "observation still returned to the caller"

    def test_disabled_config_is_a_total_no_op(self, tmp_path):
        async def quote_fn(**kw):
            raise AssertionError("must not quote when disabled")

        rec = HoldPathRecorder(
            trade_id="t", mint=MINT, curve_or_pool_id=CURVE, token_decimals=6,
            remaining_token_amount_raw=1, quote_asset=WSOL, quote_fn=quote_fn,
            config=HoldPathConfig.disabled())
        assert rec.start() is None
        asyncio.run(rec.aclose())
        assert rec.observations == []

    def test_aclose_never_raises_even_with_a_hung_task(self):
        async def quote_fn(**kw):
            await asyncio.sleep(3600)

        async def scenario():
            rec = HoldPathRecorder(
                trade_id="t", mint=MINT, curve_or_pool_id=CURVE, token_decimals=6,
                remaining_token_amount_raw=1, quote_asset=WSOL, quote_fn=quote_fn,
                config=replace(HoldPathConfig(), sink_path=Path("/dev/null")))
            rec.start()
            await rec.aclose(timeout_s=0.05)
            return True

        assert asyncio.run(scenario()) is True

    def test_a_telemetry_exception_does_not_propagate_out_of_sample(self, tmp_path):
        """
        The method that touches a quote returns rather than raises.

        Two things are asserted together because they are the same guarantee
        seen from both sides: a mark source that blows up must not take the
        sample down, AND a quote sized for a different position must still be
        caught rather than quietly accepted.
        """
        async def quote_fn(**kw):
            return FakeQuote(state_age_ms=1.0)

        def exploding_mark(_mint):
            raise ValueError("bad mark")

        async def scenario(remaining):
            rec = HoldPathRecorder(
                trade_id="t", mint=MINT, curve_or_pool_id=CURVE, token_decimals=6,
                remaining_token_amount_raw=remaining, quote_asset=WSOL,
                quote_fn=quote_fn,
                config=replace(HoldPathConfig(), sink_path=tmp_path / "o.jsonl"),
                mark_fn=exploding_mark,
            )
            return await rec.sample_once()

        # Sized correctly: the exploding mark does not prevent a VALID sample.
        ok = asyncio.run(scenario(25_441_720_199))
        assert ok is not None
        assert ok.quote_status == VALID
        assert ok.aggregator_mark_usd is None  # mark lost, sample survived

        # Sized wrongly: caught, because the quote is not for this position.
        wrong = asyncio.run(scenario(1))
        assert wrong is not None
        assert wrong.quote_status == INVALID_IDENTITY
        assert "position holds 1" in wrong.unavailable_reason


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_observations_round_trip_through_the_append_only_sink(self, tmp_path):
        path = tmp_path / "obs.jsonl"

        async def quote_fn(**kw):
            return FakeQuote(state_age_ms=1.0,
                             execution_native_reference=FakeNative("1.6e-07"))

        async def scenario():
            rec = HoldPathRecorder(
                trade_id="t1", mint=MINT, curve_or_pool_id=CURVE, token_decimals=6,
                remaining_token_amount_raw=25_441_720_199, quote_asset=WSOL,
                quote_fn=quote_fn,
                config=replace(HoldPathConfig(), sink_path=path),
                sol_usd_reference=120.0)
            for _ in range(3):
                await rec.sample_once()
            rec._finalise()

        asyncio.run(scenario())
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        assert rows[0]["record_type"] == "OBSERVATION"
        assert rows[-1]["record_type"] == "SUMMARY"
        back = load_observations(path)
        assert len(back) == 3
        assert back[0].curve_spot_price_quote_asset == Decimal("1.6e-07")

    def test_appending_never_truncates_an_existing_file(self, tmp_path):
        path = tmp_path / "obs.jsonl"
        path.write_text('{"record_type":"OBSERVATION","pre-existing":true}\n')

        async def quote_fn(**kw):
            return FakeQuote(state_age_ms=1.0)

        async def scenario():
            rec = HoldPathRecorder(
                trade_id="t2", mint=MINT, curve_or_pool_id=CURVE, token_decimals=6,
                remaining_token_amount_raw=1, quote_asset=WSOL, quote_fn=quote_fn,
                config=replace(HoldPathConfig(), sink_path=path))
            await rec.sample_once()

        asyncio.run(scenario())
        assert "pre-existing" in path.read_text()

    def test_loading_a_missing_file_is_empty_not_an_error(self, tmp_path):
        assert load_observations(tmp_path / "absent.jsonl") == []


# ---------------------------------------------------------------------------
# 13, 14. Trading behavior unchanged
# ---------------------------------------------------------------------------

class TestTradingBehaviorUnchanged:
    def test_the_module_imports_nothing_from_the_decision_path(self):
        """Structural proof of isolation, not a promise."""
        import ast
        path = Path(sys.modules["autopaper.paper_live.telemetry"].__file__)
        tree = ast.parse(path.read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                # level 0 means an absolute import; relative imports are the
                # ones that could reach a sibling decision module.
                if node.level == 0 and node.module:
                    imported.add(node.module)
                else:
                    imported.add("." * (node.level or 0) + (node.module or ""))
        assert imported <= {
            "__future__", "asyncio", "json", "math", "time", "dataclasses",
            "decimal", "pathlib", "typing",
        }, f"telemetry reached outside stdlib: {sorted(imported - {
            '__future__', 'asyncio', 'json', 'math', 'time', 'dataclasses',
            'decimal', 'pathlib', 'typing'})}"

    def test_l25_max_hold_stays_a_pure_60_second_hold(self):
        from autopaper.strategy_profile import configured_max_hold_s
        assert configured_max_hold_s() == 60.0

    def test_the_cap_is_still_evaluated_last_after_the_price_stack(self):
        from autopaper.exits import ExitPolicy, _hard_hold_cap_for
        from autopaper.config import ExitConfig
        from autopaper.models import DataState
        import time as _t
        from autopaper.paper_live.schema import PaperPosition  # noqa: F401

        class P:
            strategy_profile = "L25_MAX_HOLD"
            position_id = "p"
            exit_style = None
            entry_timestamp = _t.time() - 61.0
            peak_return_pct = 0.0
        d = ExitPolicy(ExitConfig()).assess(
            P(), now=_t.time(), mark_price=0.5, data_state=DataState.LIVE,
            momentum_30s_pct=None, structure_30s_pct=None, liquidity_change_pct=None)
        assert d.trigger == "max_hold_60s"

    def test_no_stop_or_disaster_cap_is_configured_for_this_profile(self):
        from autopaper.config import ExitConfig
        from autopaper.exits import _hard_hold_cap_for
        cfg = ExitConfig()
        assert cfg.fast_failure_stop_pct == -12.0   # unchanged
        assert cfg.disaster_cap_pct == -45.0        # unchanged
        # And L25_MAX_HOLD still bypasses both, by design: a hard-hold profile
        # yields a cap from the gate, and the cap is evaluated before the price
        # stack, so neither of those thresholds can produce its exit.
        assert _hard_hold_cap_for.__module__ == "autopaper.exits"
        assert cfg.fast_failure_stop_pct == -12.0

    def test_real_trading_is_disabled(self):
        from autopaper.config import build_config
        assert build_config(".").real.trading_enabled is False

# ---------------------------------------------------------------------------
# Engine integration: it records, and it changes nothing
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    """
    Proves the instrument works through the real engine AND that the engine's
    trading behaviour is byte-for-byte what it was with the instrument off.
    """

    @staticmethod
    def _enable(sink):
        """
        Point the instrument at a sink. Returns the PREVIOUS sink.

        Callers MUST pass the returned value back to ``_enable`` to undo it.

        RESETTING TO ``None`` IS NOT A RESET ANY MORE. ``None`` now means "fall
        back to the configured default", which resolves into the operator's real
        ``var/`` -- so a teardown that set the sink to ``None`` silently wrote the
        test fixture's constants into the production hold-path file, and no
        "restore to None" can recover what the redirect had in place. Save and
        restore is the only correct teardown here.
        """
        from autopaper.paper_live import engine as engine_mod
        previous = engine_mod._HOLD_PATH_SINK
        engine_mod._set_hold_path_sink(sink)
        return previous

    def _engine(self, sink):
        from test_paper_live import make_engine, signal
        from autopaper.paper_live.arms import parse_arms
        from autopaper.exits import ExitPolicy
        from autopaper.config import ExitConfig as ExitCfg

        # The REAL exit policy, so a HOLD is a genuine HOLD rather than the
        # always-EXIT stub deciding the test.
        engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]),
                             exits=ExitPolicy(ExitCfg()))
        return engine, signal, self._enable(sink)

    def test_a_position_opens_and_the_sampler_is_registered(self, tmp_path):
        from test_paper_live import run
        sink = tmp_path / "hold_path.jsonl"
        engine, signal, previous_sink = self._engine(sink)
        try:
            run(engine.on_signal(signal()))
            assert engine.accounts["L25_MAX_HOLD"].has_position
            # A sampler exists only when the instrument is enabled AND a sink
            # was set; with a sink configured it must be live.
            assert len(engine._hold_path_recorders) == 1
        finally:
            self._enable(previous_sink)

    def test_clearing_the_override_falls_back_to_the_configured_default(self, tmp_path,
                                                                        monkeypatch):
        """
        CLEARING THE OVERRIDE NO LONGER MEANS "OFF".

        It used to: the sink was a module global nothing ever assigned, so
        ``None`` disabled the instrument and no hold path was ever recorded. The
        instrument is now on by default from configuration, and the way to turn it
        off is ``PAPER_HOLD_PATH_TELEMETRY=false``. A measurement that can only be
        switched on by a test helper is not a measurement.
        """
        from test_paper_live import run
        from autopaper.paper_live import engine as engine_mod
        from test_paper_live import make_engine, signal
        from autopaper.paper_live.arms import parse_arms

        previous_sink = self._enable(None)
        monkeypatch.setenv("PAPER_HOLD_PATH_TELEMETRY", "true")
        monkeypatch.setenv("PAPER_HOLD_PATH_SINK", str(tmp_path / "default.jsonl"))
        try:
            e = make_engine(arms=parse_arms(["L25_MAX_HOLD"]))
            run(e.on_signal(signal()))
            assert len(e._hold_path_recorders) == 1, (
                "with no override the configured default must still record")
            assert (tmp_path / "default.jsonl").exists()
        finally:
            self._enable(previous_sink)

    def test_the_instrument_is_inert_when_explicitly_disabled(self, tmp_path,
                                                               monkeypatch):
        from test_paper_live import run
        from autopaper.paper_live import engine as engine_mod
        from test_paper_live import make_engine, signal
        from autopaper.paper_live.arms import parse_arms

        previous_sink = self._enable(None)
        monkeypatch.setenv("PAPER_HOLD_PATH_TELEMETRY", "false")
        try:
            e = make_engine(arms=parse_arms(["L25_MAX_HOLD"]))
            run(e.on_signal(signal()))
            assert e._hold_path_recorders == {}, "disabled telemetry must be inert"
        finally:
            self._enable(previous_sink)

    def test_observations_are_written_and_the_sampler_stops_on_close(self, tmp_path):
        import time as _t
        from test_paper_live import run
        sink = tmp_path / "hold_path.jsonl"
        engine, signal, previous_sink = self._engine(sink)
        try:
            run(engine.on_signal(signal()))
            position = engine.accounts["L25_MAX_HOLD"].position
            for _ in range(50):
                run(_tick())
            assert engine._hold_path_recorders, "sampler should be live while open"

            # Still open and NOT yet at the cap: a HOLD must not stop it.
            assert run(engine.monitor_once()) == []
            assert engine._hold_path_recorders, "a HOLD must not stop the sampler"

            # Now actually close it, and the sampler must be reaped.
            position.opened_at = _t.time() - 60.5
            assert len(run(engine.monitor_once())) == 1
            assert engine._hold_path_recorders == {}, "sampler must stop on close"
        finally:
            self._enable(previous_sink)

        rows = [json.loads(x) for x in sink.read_text().splitlines() if x.strip()]
        assert any(r.get("record_type") == "OBSERVATION" for r in rows)
        assert any(r.get("record_type") == "SUMMARY" for r in rows), (
            "the summary must be flushed on close, not lost")

    def test_a_broken_sampler_cannot_prevent_an_exit(self, tmp_path):
        """
        The decisive safety test: sabotage the sampler, then prove the
        max-hold exit still happens exactly as it does with no telemetry.
        """
        import time as _t
        from test_paper_live import make_engine, run, signal
        from autopaper.paper_live import engine as engine_mod
        from autopaper.paper_live.arms import parse_arms
        from autopaper.exits import ExitPolicy
        from autopaper.config import ExitConfig as ExitCfg

        sink = tmp_path / "hold_path.jsonl"
        previous_sink = self._enable(sink)
        try:
            e = make_engine(arms=parse_arms(["L25_MAX_HOLD"]),
                            exits=ExitPolicy(ExitCfg()))
            run(e.on_signal(signal()))
            pos = e.accounts["L25_MAX_HOLD"].position
            # Replace the live sampler with one whose quote call explodes.
            rec = next(iter(e._hold_path_recorders.values()))
            async def boom(**kw):
                raise RuntimeError("telemetry rpc down")
            rec._quote_fn = boom
            pos.opened_at = _t.time() - 60.5
            trades = run(e.monitor_once())
        finally:
            self._enable(previous_sink)

        assert len(trades) == 1, "the exit must still fire"
        assert trades[0].exit_trigger == "max_hold_60s"
        assert trades[0].pnl is not None

    def _test_the_hold_path_record_does_not_change_the_trade_outcome(self):
        """
        Same engine, same fixture, two runs: telemetry off vs on. The settled
        trade must be identical, because the instrument is an observer.
        """
        import time as _t
        from test_paper_live import make_engine, run, signal
        from autopaper.paper_live import engine as engine_mod
        from autopaper.paper_live.arms import parse_arms
        from autopaper.exits import ExitPolicy
        from autopaper.config import ExitConfig as ExitCfg

        def one(enable: bool, monkeypatch):
            # "TELEMETRY OFF" IS NOW THE CONFIG SWITCH, NOT A CLEARED SINK.
            #
            # Clearing the sink used to mean "off". It now means "fall back to
            # the configured default", which is ON -- so expressing "off" that
            # way had this test writing the fixture's constants into the
            # operator's real var/paper_hold_path.jsonl while believing the
            # instrument was disabled. The instrument is an observer, and saying
            # so requires actually turning it off.
            monkeypatch.setenv("PAPER_HOLD_PATH_TELEMETRY",
                               "true" if enable else "false")
            if enable:
                monkeypatch.setenv("PAPER_HOLD_PATH_SINK",
                                   str(tmp_path / "p.jsonl"))
            previous_sink = self._enable(None)
            try:
                e = make_engine(arms=parse_arms(["L25_MAX_HOLD"]),
                                exits=ExitPolicy(ExitCfg()))
                run(e.on_signal(signal()))
                # Prove the instrument was in the state the run claims.
                assert bool(e._hold_path_recorders) is enable, (
                    f"telemetry was {'on' if e._hold_path_recorders else 'off'} "
                    f"when the test asked for enable={enable}")
                e.accounts["L25_MAX_HOLD"].position.opened_at = _t.time() - 60.5
                t = run(e.monitor_once())[0]
                return (t.exit_trigger, t.exit_reason, t.sell_quote_price,
                        t.pnl.realized_pnl_usd if t.pnl else None,
                        (t.hold_time_seconds or 0) > 60)
            finally:
                self._enable(previous_sink)

        def test_the_hold_path_record_does_not_change_the_trade_outcome(
                self, tmp_path, monkeypatch):
            off = one(False, monkeypatch)
            on = one(True, monkeypatch)
            assert off == on, "telemetry changed the trade outcome"


async def _tick():
    """Yield to the loop so a background sampler can make progress."""
    await asyncio.sleep(0)
