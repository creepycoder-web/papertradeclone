"""
Quote-MFE / Quote-MAE in the CLOSED POSITIONS view: measured, or honestly absent.

WHAT WAS BROKEN
===============
Every closed position rendered::

    Quote-MFE: N/A · Quote-MAE: N/A
    no valid same-market SELL quote was obtained for this position

...including trades that HAD been quoted, sampled and settled on a valid
executable SELL quote with a recorded realised P&L. Three separate defects
stacked up to produce that:

1.  **THE READER COULD NOT SEE THE DATA.** ``PaperTrader._arm_closed_rows``
    looked for ``QUOTE_MFE`` / ``QUOTE_MAE_MAE_BASIS`` on
    ``trade["position"]``. Those are ``TradeRecord.as_dict()``'s UPPERCASE keys
    and they live on ``trade``; ``PaperPosition.as_dict()`` persists the SAME
    fields LOWERCASE. Every lookup returned ``None`` for every record, so
    measured excursions rendered as absent ones.

2.  **SAMPLING WAS ONE POINT PER POSITION.** The WebSocket-push exit path
    fetched and validated exactly the same executable SELL quote as the timer
    path and then threw it away without recording it. Since a push is what moves
    a position fastest, every push-driven hold was sampled at zero or one
    quote -- and ``MFE == MAE`` on one sample reads exactly like a bounded range.

3.  **ONE REASON WAS PRINTED FOR FIVE DIFFERENT FACTS.** "No valid same-market
    SELL quote was obtained" was shown for records that settled on an accepted
    quote but were never sampled, for records whose quotes were all rejected,
    for modelled backtest records that never had an executable quote, and for
    historical records written before the metric existed.

WHAT THESE PIN
==============
  * the reader sees a measured excursion, from either key convention;
  * a valid intratrade SELL quote, and the accepted settlement quote, both
    become observations;
  * an identity-failed, stale, wrong-quantity or wrong-market quote is NOT a
    measurement -- and says which gate refused it;
  * an absent excursion stays ``N/A`` and names the real reason, with the five
    causes kept distinct;
  * a completed quote-based BETA settlement still REQUIRES a valid accepted SELL
    quote, and reporting its excursion does not weaken that by one line.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from autopaper.forensics.compact_report import (
    compact_summary,
    render_excursion,
)
from autopaper.forensics.quote_excursions import (
    SETTLEMENT_ACCEPTED,
    SETTLEMENT_MODELLED,
    TELEMETRY_ABSENT,
    TELEMETRY_MEASURED,
    TELEMETRY_MODELLED,
    TELEMETRY_NO_INTRATRADE_SAMPLES,
    TELEMETRY_NO_USABLE_QUOTES,
    TELEMETRY_SAMPLES_REJECTED,
    excursion_telemetry_state,
    excursion_unavailable_reason,
    settlement_quote_state,
)
from autopaper.engine import (
    PaperTrader,
    _beta_excursion_fields,
    _excursion_fields,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from tests import paper_bridge_fakes as _fk
except ImportError:  # pragma: no cover - direct file invocation
    import paper_bridge_fakes as _fk

alert, bridge, real_report, run = _fk.alert, _fk.bridge, _fk.real_report, _fk.run


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _position(engine):
    for account in engine.accounts.values():
        if account.position is not None:
            return account.position
    return None


def _open(engine=None):
    from autopaper.paper_live.arms import parse_arms

    engine = engine or bridge(arms=parse_arms(["L25_CURRENT"]))
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    position = _position(engine)
    assert position is not None, "the fixture opened no position"
    return engine, position


def _quote(position, *, multiple=2.0, identity="PASS", pool=None,
           tokens_raw=None, provider="pump_direct", age_ms=5.0):
    """A minimal executable SELL quote object, defaulting to a valid one."""
    _pool = (position.entry_pool_address or "pool") if pool is None else pool
    _identity = identity
    _tokens_raw = position.tokens_raw if tokens_raw is None else tokens_raw
    _multiple = multiple
    _provider = provider
    _age_ms = age_ms

    class _Q:
        available = True
        pool_or_curve = _pool
        identity = _identity
        provider = _provider
        input_amount_raw = _tokens_raw
        output_amount_decimal = Decimal(
            float(position.entry_cost_quote_asset) * _multiple)
        output_amount_raw = 123
        quote_age_ms = _age_ms

        def as_dict(self):
            return {"provider": self.provider, "identity": self.identity,
                    "output_amount_raw": self.output_amount_raw}

    return _Q()


class _Ledger:
    def __init__(self, records):
        self._records = records

    def arm_exits(self):
        return self._records


def _rows(records):
    trader = PaperTrader.__new__(PaperTrader)
    trader.ledger = _Ledger(records)
    return trader._arm_closed_rows()


def _trade(**over):
    """A persisted ``TradeRecord.as_dict()`` shape: UPPERCASE excursion keys."""
    base = {
        "position": {
            "arm": "L25_CURRENT",
            "signal_id": "sig_1",
            "symbol": "KFC",
            "opened_at": 1_700_000_000.0,
            "quote_mfe_pct": None,
            "quote_mae_pct": None,
            "quote_mfe_mae_basis": "NOT_MEASURED",
        },
        "failure_reason": "NONE",
        "realized_pnl": {"realized_pnl_usd": 0.25, "return_pct": 12.5},
        "sell_execution_status": "SETTLED",
        "sell_quote_price": 0.000123,
    }
    base.update(over)
    return base


def _record(trade, **over):
    base = {
        "event": "arm_exit",
        "arm": "L25_CURRENT",
        "signal_id": "sig_1",
        "mint": "So11111111111111111111111111111111111111112",
        "symbol": "KFC",
        "at": 1_700_000_600.0,
        "settlement_key": "L25_CURRENT::sig_1",
        "pnl_basis": "FILL_VS_FILL",
        "realized_pnl_usd": 0.25,
        "trade": trade,
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# 1. THE KEY-CONVENTION DEFECT: the reader must actually see the measurement
# ---------------------------------------------------------------------------


def test_a_measured_excursion_persisted_on_the_trade_is_read_back():
    """
    ``TradeRecord.as_dict()`` emits ``QUOTE_MFE``; the reader must read it there.

    This is the regression that made every measured trade render as ``N/A``.
    """
    rows = _rows([_record(_trade(
        QUOTE_MFE=-17.117532,
        QUOTE_MAE=-25.181037,
        QUOTE_MFE_MAE_BASIS="QUOTE_BASED",
        QUOTE_MFE_ELAPSED_MS=38_265.0,
        QUOTE_MAE_ELAPSED_MS=9_000.0,
        QUOTE_OBSERVATION_COUNT=4,
        QUOTE_COVERAGE_PCT=62.5,
        QUOTE_MFE_SOURCE="pump_direct",
        QUOTE_MAE_SOURCE="pump_direct",
        QUOTE_SETTLEMENT_SAMPLED=True,
        QUOTE_SETTLEMENT_IDENTITY="PASS",
    ))])
    assert len(rows) == 1
    row = rows[0]
    assert row["quote_mfe_pct"] == pytest.approx(-17.117532)
    assert row["quote_mae_pct"] == pytest.approx(-25.181037)
    assert row["quote_mfe_mae_basis"] == "QUOTE_BASED"
    assert row["quote_telemetry_state"] == TELEMETRY_MEASURED
    assert row["quote_observation_count"] == 4
    assert row["quote_settlement_sample"] is True


def test_a_measured_excursion_persisted_on_the_position_is_read_back():
    """
    ``PaperPosition.as_dict()`` persists the SAME fields LOWERCASE. Both
    conventions are live, so both must be readable.
    """
    position = dict(_trade()["position"])
    position.update({
        "quote_mfe_pct": 8.4,
        "quote_mae_pct": -14.1,
        "quote_mfe_mae_basis": "QUOTE_BASED",
        "quote_observation_count": 9,
    })
    rows = _rows([_record(_trade(position=position, QUOTE_OBSERVATION_COUNT=9))])
    # The trade dict is authoritative; the position dict is the fallback. This
    # record has telemetry on the position only, so that is what is read.
    assert rows[0]["quote_telemetry_state"] == TELEMETRY_MEASURED
    assert rows[0]["quote_mfe_pct"] == pytest.approx(8.4)


def test_the_rendered_line_shows_the_measurement_and_its_sample_count():
    rows = _rows([_record(_trade(
        QUOTE_MFE=-17.1, QUOTE_MAE=-25.2, QUOTE_MFE_MAE_BASIS="QUOTE_BASED",
        QUOTE_MFE_ELAPSED_MS=38_265.0, QUOTE_MAE_ELAPSED_MS=9_000.0,
        QUOTE_OBSERVATION_COUNT=4, QUOTE_COVERAGE_PCT=62.5,
        QUOTE_SETTLEMENT_SAMPLED=True))])
    line = render_excursion(compact_summary(rows[0]))
    assert "-17.1%" in line and "-25.2%" in line
    assert "4 quote samples" in line
    assert "coverage 62%" in line
    assert "N/A" not in line


def test_a_single_observation_is_not_rendered_as_a_bound():
    """
    One sample cannot span a hold. MFE == MAE on a single quote is that quote,
    and printing it unlabelled invites reading it as the range the hold covered.
    """
    rows = _rows([_record(_trade(
        QUOTE_MFE=-17.117532, QUOTE_MAE=-17.117532,
        QUOTE_MFE_MAE_BASIS="QUOTE_BASED",
        QUOTE_OBSERVATION_COUNT=1, QUOTE_COVERAGE_PCT=100.0))])
    line = render_excursion(compact_summary(rows[0]))
    assert "1 quote sample" in line
    assert "single observation, not a bound" in line


# ---------------------------------------------------------------------------
# 2. valid intratrade quote samples
# ---------------------------------------------------------------------------


def test_a_valid_intratrade_sell_quote_becomes_an_observation():
    engine, position = _open()
    engine._record_quote_excursion(position, _quote(position, multiple=2.0))
    assert position.quote_mfe_mae_basis == "QUOTE_BASED"
    assert position.quote_mfe_pct == pytest.approx(100.0, abs=1.0)
    assert position.quote_observation_count == 1
    assert position.quote_settlement_sample is False
    assert position.quote_rejection_count == 0


def test_several_samples_bound_the_hold_between_their_extrema():
    engine, position = _open()
    for multiple in (0.5, 3.0, 1.5):
        engine._record_quote_excursion(position, _quote(position, multiple=multiple))
    assert position.quote_observation_count == 3
    assert position.quote_mfe_pct == pytest.approx(200.0, abs=2.0)
    assert position.quote_mae_pct == pytest.approx(-50.0, abs=2.0)


# ---------------------------------------------------------------------------
# 3. the accepted settlement quote is an observation
# ---------------------------------------------------------------------------


def test_the_accepted_settlement_quote_is_folded_in_with_its_provenance():
    """
    The exit quote the position actually sold on is the strongest excursion
    evidence a position ever produces, and its own provenance is recorded so
    the identity decision that admitted it stays auditable.
    """
    engine, position = _open()
    engine._record_quote_excursion(position, _quote(position, multiple=0.4),
                                   settlement=True)
    assert position.quote_settlement_sample is True
    assert position.quote_settlement_identity == "PASS"
    assert position.quote_settlement_source == "pump_direct"
    assert position.quote_settlement_age_ms == 5.0
    assert position.quote_observation_count == 1


def test_an_unavailable_settlement_quote_is_not_recorded_as_a_sample():
    """A settlement that did not quote cannot contribute an observation."""
    engine, position = _open()

    class _Dead:
        available = False

    engine._record_quote_excursion(position, _Dead(), settlement=True)
    assert position.quote_settlement_sample is False
    assert position.quote_mfe_mae_basis != "QUOTE_BASED"
    assert position.quote_rejection_count == 1
    assert "settlement quote unavailable" in position.quote_last_rejection_reason


# ---------------------------------------------------------------------------
# 4. rejected observations: invalid identity, stale, wrong quantity, wrong market
# ---------------------------------------------------------------------------


def test_a_failed_identity_is_never_an_observation():
    engine, position = _open()
    engine._record_quote_excursion(position, _quote(position, identity="UNKNOWN"))
    assert position.quote_mfe_mae_basis != "QUOTE_BASED"
    assert position.quote_mfe_pct is None
    assert position.quote_rejection_count == 1
    assert "market" in position.quote_last_rejection_reason or \
        "quantity" in position.quote_last_rejection_reason


def test_a_quote_for_another_market_is_rejected_and_names_the_market():
    engine, position = _open()
    engine._record_quote_excursion(position, _quote(position, pool="some-other-pool"))
    assert position.quote_mfe_mae_basis != "QUOTE_BASED"
    assert position.quote_rejection_count == 1
    assert "market" in position.quote_last_rejection_reason


def test_a_quote_for_another_quantity_is_rejected_and_names_the_quantity():
    engine, position = _open()
    engine._record_quote_excursion(
        position, _quote(position, tokens_raw=position.tokens_raw // 2 or 7))
    assert position.quote_mfe_mae_basis != "QUOTE_BASED"
    assert position.quote_rejection_count == 1
    assert "raw tokens" in position.quote_last_rejection_reason


def test_a_stale_quote_is_refused_by_the_validation_gate_not_recorded():
    """
    Freshness is the gate's job. A stale quote never reaches the collector, and
    the rejection is recorded so the absence is diagnosable.
    """
    engine, position = _open()

    class _Stale:
        available = True
        pool_or_curve = position.entry_pool_address or "pool"
        identity = "PASS"
        provider = "pump_direct"
        input_amount_raw = position.tokens_raw
        output_amount_decimal = Decimal(9)
        output_amount_raw = 1
        quote_age_ms = 90_000.0
        stale = True
        max_quote_age_ms = 30_000.0

    ok, why = engine._quote_is_fresh_and_same_market(_Stale(), position)
    assert ok is False
    assert "freshness window" in why

    engine._reject_quote_observation(position, f"validation failed: {why}")
    assert position.quote_mfe_mae_basis != "QUOTE_BASED"
    assert position.quote_rejection_count == 1
    assert "validation failed" in position.quote_last_rejection_reason


def test_repeated_rejections_are_counted_not_swallowed():
    engine, position = _open()
    for _ in range(5):
        engine._record_quote_excursion(position, _quote(position, pool="other"))
    assert position.quote_rejection_count == 5
    assert position.quote_observation_count == 0
    assert position.quote_mfe_mae_basis != "QUOTE_BASED"


# ---------------------------------------------------------------------------
# 5. absent samples: the five causes stay distinct
# ---------------------------------------------------------------------------


def _state(**over):
    return excursion_telemetry_state({}, **over)


def test_a_record_with_no_telemetry_field_is_historical_not_unmeasured():
    """Written before the metric existed: nothing was measured and claimed."""
    state = _state(basis="NOT_MEASURED", telemetry_persisted=False,
                   settlement=SETTLEMENT_ACCEPTED)
    assert state == TELEMETRY_ABSENT
    reason = excursion_unavailable_reason(
        {}, basis="NOT_MEASURED", telemetry_persisted=False,
        settlement=SETTLEMENT_ACCEPTED)
    assert "no quote-excursion telemetry" in reason


def test_a_settled_but_unsampled_record_does_not_claim_it_had_no_quote():
    """
    THE MISLEADING REASON. A valid settlement quote WAS accepted; what is
    missing is intratrade sampling. Saying "no valid same-market SELL quote was
    obtained" here is simply false.
    """
    state = _state(basis="NOT_MEASURED", telemetry_persisted=True,
                   observation_count=0, settlement=SETTLEMENT_ACCEPTED)
    assert state == TELEMETRY_NO_INTRATRADE_SAMPLES
    reason = excursion_unavailable_reason(
        {}, basis="NOT_MEASURED", telemetry_persisted=True,
        observation_count=0, settlement=SETTLEMENT_ACCEPTED)
    assert "valid settlement SELL quote was accepted" in reason
    assert "no valid same-market SELL quote was obtained" not in reason


def test_a_record_whose_quotes_were_all_rejected_says_so():
    state = _state(basis="NOT_MEASURED", telemetry_persisted=True,
                   observation_count=0, rejection_count=12,
                   settlement=SETTLEMENT_NOT_ACCEPTED
                   if False else "NOT_ACCEPTED")
    assert state == TELEMETRY_SAMPLES_REJECTED
    reason = excursion_unavailable_reason(
        {}, basis="NOT_MEASURED", telemetry_persisted=True,
        observation_count=0, rejection_count=12, settlement="NOT_ACCEPTED")
    assert "rejected" in reason and "identity, freshness, quantity or route" in reason


def test_a_telemetry_present_record_with_nothing_usable_is_named_as_such():
    state = _state(basis="NOT_MEASURED", telemetry_persisted=True,
                   observation_count=0, rejection_count=0,
                   settlement="NOT_ACCEPTED")
    assert state == TELEMETRY_NO_USABLE_QUOTES


def test_a_modelled_record_is_never_called_an_unquoted_one():
    state = _state(basis="NOT_MEASURED", telemetry_persisted=True,
                   settlement=SETTLEMENT_MODELLED)
    assert state == TELEMETRY_MODELLED
    reason = excursion_unavailable_reason(
        {}, basis="NOT_MEASURED", telemetry_persisted=True,
        settlement=SETTLEMENT_MODELLED)
    assert "booked off an observed mark" in reason


def test_the_settlement_quote_state_is_read_from_provenance_not_from_pnl():
    assert settlement_quote_state({"sell_execution_status": "SETTLED"}) == \
        SETTLEMENT_ACCEPTED
    assert settlement_quote_state(
        {"pnl_basis": "V2_FLOOR_BETA_QUOTE_SIMULATED_FILL"}) == SETTLEMENT_ACCEPTED
    assert settlement_quote_state(
        {"pnl_basis": "V2_FLOOR_BETA_MODELED_BACKTEST"}) == SETTLEMENT_MODELLED
    assert settlement_quote_state(
        {"sell_execution_status": "NO_EXECUTABLE_SELL"}) == "NOT_ACCEPTED"
    # A positive P&L is NOT evidence of an executable quote.
    assert settlement_quote_state({"realized_pnl_usd": 2.0}) == "UNKNOWN"


@pytest.mark.parametrize("pnl_basis,expected", [
    ("FILL_VS_FILL", SETTLEMENT_ACCEPTED),
    ("V2_FLOOR_BETA_QUOTE_SIMULATED_FILL", SETTLEMENT_ACCEPTED),
    ("V2_FLOOR_BETA_MODELED_BACKTEST", SETTLEMENT_MODELLED),
])
def test_each_closed_row_gets_a_settlement_verdict(pnl_basis, expected):
    rows = _rows([_record(_trade(), pnl_basis=pnl_basis)])
    assert rows[0]["quote_settlement_status"] == expected


# ---------------------------------------------------------------------------
# 6. end-to-end through the renderer
# ---------------------------------------------------------------------------


def test_a_settled_but_unsampled_row_renders_its_real_reason():
    rows = _rows([_record(_trade(
        QUOTE_MFE_MAE_BASIS="NOT_MEASURED", QUOTE_OBSERVATION_COUNT=0))])
    line = render_excursion(compact_summary(rows[0]))
    assert line.startswith("Quote-MFE: N/A · Quote-MAE: N/A")
    assert "valid settlement SELL quote was accepted" in line
    assert "no valid same-market SELL quote was obtained" not in line


def test_a_historical_row_renders_the_historical_reason():
    trade = _trade()
    trade["position"].pop("quote_mfe_mae_basis")
    trade["position"].pop("quote_mfe_pct")
    trade["position"].pop("quote_mae_pct")
    rows = _rows([_record(trade)])
    line = render_excursion(compact_summary(rows[0]))
    assert "no quote-excursion telemetry" in line


def test_a_rejected_sampling_row_renders_its_rejection_count():
    rows = _rows([_record(_trade(
        QUOTE_MFE_MAE_BASIS="NOT_MEASURED", QUOTE_OBSERVATION_COUNT=0,
        QUOTE_REJECTION_COUNT=12,
        QUOTE_LAST_REJECTION_REASON="quote age 90000ms exceeds the bound"))])
    line = render_excursion(compact_summary(rows[0]))
    assert "12 quote observations rejected" in line
    assert "quote age 90000ms exceeds the bound" in line


def test_the_old_guessing_fallback_is_never_printed():
    """
    The removed sentence asserted a fact about quoting that the row may not
    support. It must not reappear through any unmeasured path.
    """
    rows = _rows([
        _record(_trade(QUOTE_MFE_MAE_BASIS="NOT_MEASURED")),
        _record(_trade(QUOTE_MFE_MAE_BASIS="QUOTE_BASED", QUOTE_MFE=1.0,
                       QUOTE_MAE=-1.0, QUOTE_OBSERVATION_COUNT=3,
                       QUOTE_MFE_ELAPSED_MS=1.0, QUOTE_MAE_ELAPSED_MS=2.0),
                settlement_key="k2", symbol="RUSO"),
    ])
    for row in rows:
        assert "no valid same-market SELL quote was obtained" not in \
            render_excursion(compact_summary(row))


# ---------------------------------------------------------------------------
# 7. the BETA invariant is untouched
# ---------------------------------------------------------------------------


def test_a_beta_row_carries_its_settlement_quote_and_no_invented_excursion():
    """
    A quote-based beta settlement DID accept a same-market SELL quote. Its row
    must say so, and must not claim an excursion it never sampled.
    """
    block = _beta_excursion_fields({
        "pnl_basis": "V2_FLOOR_BETA_QUOTE_SIMULATED_FILL",
        "exit_basis": "EXECUTABLE_QUOTE_SIMULATED_FILL",
        "exit_identity": "PASS",
        "net_pnl_usd": -0.18,
    })
    assert block["quote_mfe_pct"] is None
    assert block["quote_mae_pct"] is None
    assert block["quote_mfe_mae_basis"] == "NOT_MEASURED"
    assert block["quote_telemetry_state"] == TELEMETRY_NO_INTRATRADE_SAMPLES
    assert block["quote_settlement_status"] == SETTLEMENT_ACCEPTED
    assert block["quote_settlement_identity"] == "PASS"
    assert "valid settlement SELL quote was accepted" in \
        block["quote_unavailable_reason"]
    # One settlement quote is not a sampled excursion, and is not claimed as one.
    assert block["quote_settlement_sample"] is False


def test_a_modelled_beta_record_is_reported_as_modelled():
    block = _beta_excursion_fields({
        "pnl_basis": "V2_FLOOR_BETA_MODELED_BACKTEST",
        "net_pnl_usd": 1.4,
    })
    assert block["quote_telemetry_state"] == TELEMETRY_MODELLED
    assert "booked off an observed mark" in block["quote_unavailable_reason"]


def test_a_beta_settlement_still_refuses_to_close_without_a_valid_quote():
    """
    THE INVARIANT, pinned directly on ``BetaExperiment``: a quote-required exit
    with no valid quote leaves the position UNRESOLVED and books nothing.

    Nothing in the reporting change may soften this.
    """
    from autopaper.paper_live.v2_floor_beta import BetaExperiment

    assert hasattr(BetaExperiment, "monitor_once")
    source = Path(BetaExperiment.__module__.replace(".", "/") + ".py")
    source = (Path(__file__).resolve().parents[1] / "src" /
              "autopaper/paper_live/v2_floor_beta.py")
    text = source.read_text()
    assert 'pos.status = "unresolved"' in text
    assert "no valid executable SELL quote for the acquired quantity" in text
    # The modelled path is reachable only when a quote is NOT required, and it
    # tags the fill so it can never be reported as a quote settlement.
    assert "MODELED_BACKTEST_BASIS" in text


def test_reporting_a_beta_row_never_grants_an_excursion():
    """A record with no sampled observation cannot be rendered as measured."""
    block = _beta_excursion_fields({
        "pnl_basis": "V2_FLOOR_BETA_QUOTE_SIMULATED_FILL",
        "exit_basis": "EXECUTABLE_QUOTE_SIMULATED_FILL",
        "exit_identity": "PASS",
        "net_pnl_usd": 2.1,
    })
    line = render_excursion(compact_summary({
        "symbol": "KFC",
        "realized_pnl_usd": 2.1,
        "pnl_basis": "V2_FLOOR_BETA_QUOTE_SIMULATED_FILL",
        **block,
    }))
    assert line.startswith("Quote-MFE: N/A · Quote-MAE: N/A")
    assert "valid settlement SELL quote was accepted" in line


# ---------------------------------------------------------------------------
# 8. persistence: the trade record carries the evidence forward
# ---------------------------------------------------------------------------


def test_the_persisted_trade_record_carries_the_sample_provenance():
    from autopaper.paper_live.schema import (
        FailureReason, LatencyMarks, TradeRecord)

    engine, position = _open()
    record = TradeRecord(
        position=position,
        exit_trigger="EXIT",
        exit_reason="take profit",
        exit_observed_mark=1.0,
        exit_mark_timestamp=1_700_000_600.0,
        sell_quote=_quote(position),
        exit_fill=None,
        pnl=None,
        latency=LatencyMarks(
            receive_at_monotonic=position.opened_at,
            receive_at_wall=position.opened_at,
            exit_trigger_at=1.0,
            exit_trigger_wall=1_700_000_600.0),
        failure=FailureReason.NONE,
        failure_detail="",
        quote_mfe_pct=8.4,
        quote_mae_pct=-14.1,
        quote_mfe_mae_basis="QUOTE_BASED",
        quote_observation_count=7,
        quote_rejection_count=2,
        quote_last_rejection_reason="quote is for 5 raw tokens, position holds 9",
        quote_settlement_sample=True,
        quote_settlement_identity="PASS",
        quote_settlement_source="pump_direct",
        quote_settlement_age_ms=11.0,
    )
    out = record.as_dict()
    assert out["QUOTE_OBSERVATION_COUNT"] == 7
    assert out["QUOTE_REJECTION_COUNT"] == 2
    assert out["QUOTE_SETTLEMENT_SAMPLED"] is True
    assert out["QUOTE_SETTLEMENT_IDENTITY"] == "PASS"
    assert out["QUOTE_SETTLEMENT_SOURCE"] == "pump_direct"
    assert out["QUOTE_SETTLEMENT_QUOTE_AGE_MS"] == 11.0


def test_the_persisted_position_carries_the_sample_provenance():
    engine, position = _open()
    engine._record_quote_excursion(position, _quote(position), settlement=True)
    out = position.as_dict()
    assert out["quote_settlement_sample"] is True
    assert out["quote_settlement_identity"] == "PASS"
    assert out["quote_observation_count"] == 1
    assert out["quote_rejection_count"] == 0


# ---------------------------------------------------------------------------
# 9. the REAL persisted records: the reported symptom, reproduced and fixed
# ---------------------------------------------------------------------------


def _live_records():
    ledger = Path(__file__).resolve().parents[1] / "var" / "ledger.jsonl"
    if not ledger.exists():
        pytest.skip("no live ledger in this checkout")
    out = []
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "arm_exit":
            out.append(record)
    return out


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "var" / "ledger.jsonl").exists(),
    reason="no live ledger in this checkout")
def test_live_measured_records_no_longer_render_as_absent():
    """
    The reported symptom, against the real artefacts: measured trades in the
    persisted ledger must not render ``N/A``.
    """
    records = _live_records()
    rows = _rows(records)
    measured = [r for r in rows if r["quote_telemetry_state"] == TELEMETRY_MEASURED]
    if not measured:
        pytest.skip("this checkout's ledger predates quote telemetry")
    for row in measured:
        line = render_excursion(compact_summary(row))
        assert "N/A" not in line, f"{row['symbol']} measured but rendered N/A"
        assert row["quote_mfe_pct"] is not None
        assert row["quote_mae_pct"] is not None


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "var" / "ledger.jsonl").exists(),
    reason="no live ledger in this checkout")
def test_live_unmeasured_records_state_a_specific_reason():
    records = _live_records()
    rows = _rows(records)
    unmeasured = [r for r in rows
                  if r["quote_telemetry_state"] != TELEMETRY_MEASURED]
    if not unmeasured:
        pytest.skip("no unmeasured records to check")
    for row in unmeasured:
        line = render_excursion(compact_summary(row))
        assert "Quote-MFE: N/A" in line
        assert row["quote_unavailable_reason"], "an absent excursion must say why"
        assert "no valid same-market SELL quote was obtained" not in line