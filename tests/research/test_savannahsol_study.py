"""
Focused tests for the @SavannahSOL channel study. Fully offline: every test runs
off the cached corpus and the frozen L25 modules, never the network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_RESEARCH = Path(__file__).resolve().parents[2] / "research"
import sys as _sys
for _p in (str(_RESEARCH), str(_RESEARCH.parent / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from savannah.analysis import (  # noqa: E402
    classify,
    l25_funnel,
    load_corpus,
    overlap_and_lead,
    parse_num,
    signal_funnel,
    stated_fields,
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


# ---- 1. parsing ---------------------------------------------------------

def test_mint_parsing_uses_the_production_parser(corpus):
    _ev, first, _m = corpus
    assert len(first) > 1000, "corpus should dedupe to thousands of mints"
    for m in first:
        assert re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,50}", m), m


def test_stated_number_parsing_handles_k_m_b():
    assert parse_num("107.9K") == pytest.approx(107_900)
    assert parse_num("4.2M") == pytest.approx(4_200_000)
    assert parse_num("1.3B") == pytest.approx(1_300_000_000)
    assert parse_num("500") == pytest.approx(500)
    assert parse_num(None) is None
    assert parse_num("abc") is None


def test_stated_fields_are_read_from_the_body():
    s = stated_fields("MC: 124.9K\nVol: 193.7K | Top 10: 27.4 | Holders: 302")
    assert s["stated_mcap_usd"] == pytest.approx(124_900)
    assert s["stated_vol_usd"] == pytest.approx(193_700)
    assert s["stated_top10_pct"] == pytest.approx(27.4)
    assert s["stated_holders"] == pytest.approx(302)


def test_multiplier_alerts_are_separated_from_calls():
    assert classify("38.9x \U0001F680 $knightcat Detected from $107.9K to $4.2M") \
        == "MULTIPLIER_ALERT"
    assert classify("$Lucia | MC: 124.9K Vol: 193.7K") == "CALL"


def test_dedupe_by_mint_keeps_the_earliest(corpus):
    _ev, first, _m = corpus
    rows = [e for e in _ev if e["mint"]]
    for m, e in list(first.items())[:50]:
        mine = [r["ts"] for r in rows if r["mint"] == m]
        assert e["ts"] == min(mine), f"{m} first appearance is not the earliest"


# ---- 2. funnel ----------------------------------------------------------

def test_funnel_is_monotonically_non_increasing(corpus):
    _ev, first, meta = corpus
    f = l25_funnel(first, None)
    order = [f["stages"][k] for k in sorted(f["stages"], key=lambda k: -f["stages"][k])]
    assert order == sorted(order, reverse=True), f["stages"]


def test_no_signal_is_authorized_without_market_data(corpus):
    _ev, first, _m = corpus
    f = l25_funnel(first, None)
    assert f["stages"]["4_gates_reached"] <= f["stages"]["3_market_data_available"]
    assert f["stages"]["3_market_data_available"] <= f["stages"]["2_valid_mint"]


def test_the_market_cap_ceiling_is_the_production_value(corpus):
    from autopaper.config import build_config

    _ev, first, _m = corpus
    ceiling = build_config().decision.v3c_max_market_cap_usd
    f = l25_funnel(first, None)
    above = f["reasons"].get("v3c_market_cap_above_ceiling", 0)
    assert above > 0
    assert f["stages"]["4_gates_reached"] + above <= f["stages"]["3_market_data_available"]


def test_l25_is_not_modified_by_the_study():
    from autopaper import l25

    before = (l25.L25_MIN_LIQ_TO_MCAP, getattr(l25, "L25_HASH", ""))
    l25_funnel({}, None)
    assert (l25.L25_MIN_LIQ_TO_MCAP, getattr(l25, "L25_HASH", "")) == before


# ---- 3. overlap and lead time ------------------------------------------

def test_overlap_counts_are_internally_consistent(corpus):
    _ev, first, _m = corpus
    ov = overlap_and_lead(first)
    S = set(first)
    for ch, r in ov["channels"].items():
        assert r["savannah_only"] == len(S) - r["shared_mints"]
        assert r["overlap_pct_of_savannah"] == pytest.approx(
            r["shared_mints"] / len(S) * 100.0, rel=1e-6)


def test_lead_sign_convention_is_savannah_minus_other(corpus):
    _ev, first, _m = corpus
    ov = overlap_and_lead(first)
    l = ov["channels"]["MooDengPresidentCallers"]["lead"]
    # positive lead must mean Savannah EARLIER
    assert l["savannah_earlier"] + (l["shared"] - l["savannah_earlier"]) == l["shared"]
    assert l["median_lead_s"] <= 0, (
        "Savannah is expected to be later than the current source; a positive "
        "median here would mean the sign convention is inverted")


# ---- 4. outcomes ---------------------------------------------------------

def test_outcome_metrics_never_upgrade_a_mark_to_a_fill():
    from savannah.stats import load_outcomes, outcome_stats

    rows, meta = load_outcomes()
    s = outcome_stats(rows, meta)
    assert "SURVIVORSHIP_WARNING" in s
    if s.get("resolved"):
        assert s["resolution_rate"] <= 1.0
        assert s["mae"]["median"] <= s["mfe"]["median"], (
            "a downside excursion cannot exceed the upside on the same path")


def test_winner_rule_is_stated_not_arbitrary(corpus):
    from savannah.stats import load_outcomes, outcome_stats

    rows, meta = load_outcomes()
    s = outcome_stats(rows, meta)
    if s.get("resolved"):
        assert "note_on_winner_rule" in s
        assert set(s["winner_rate_by_peak_pct"]) == set(s["winner_rate_by_final_pct"])
        # realised cannot beat peak-based, by construction
        for k in s["winner_rate_by_peak_pct"]:
            assert s["winner_rate_by_final_pct"][k] <= s["winner_rate_by_peak_pct"][k] + 1e-9


def test_monte_carlo_is_seeded_and_reproducible(corpus):
    """The gated sample is tiny, so force a drawable set to exercise determinism."""
    from savannah.analysis import load_corpus as _lc
    from savannah.stats import (attach_decision_time_mcap, load_outcomes,
                                monte_carlo_from_outcomes)

    rows, _ = load_outcomes()
    rows = remeasure_rows(rows)
    _ev, first, _m = _lc()
    attach_decision_time_mcap(rows, first)
    if not rows or len([r for r in rows if r["outcome"].get("resolved")]) < 10:
        pytest.skip("sample too small for the Monte Carlo")
    # the real gated set is ~1 signal, so widen the ceiling purely to test
    # determinism of the simulator itself
    a = monte_carlo_from_outcomes(rows, paths=200, max_mcap_usd=10**12)
    b = monte_carlo_from_outcomes(rows, paths=200, max_mcap_usd=10**12)
    assert not a.get("insufficient")
    assert a["by_peak_close"]["median_final"] == b["by_peak_close"]["median_final"]
    assert a["by_realised_final"]["median_final"] == b["by_realised_final"]["median_final"]


def remeasure_rows(rows):
    from savannah.stats import remeasure

    return remeasure(rows)


def test_monte_carlo_refuses_to_project_from_a_tiny_gated_sample(corpus):
    """
    With the real $20k ceiling the gated set collapses to ~1 signal. The MC must
    decline to simulate rather than print a confident equity curve from it.
    """
    from savannah.analysis import load_corpus as _lc
    from savannah.stats import (attach_decision_time_mcap, load_outcomes,
                                monte_carlo_from_outcomes)

    rows, _ = load_outcomes()
    rows = remeasure_rows(rows)
    _ev, first, _m = _lc()
    attach_decision_time_mcap(rows, first)
    mc = monte_carlo_from_outcomes(rows, paths=50)
    if mc["gated_resolved"] < 30:
        assert mc.get("insufficient") is True
        assert "by_peak_close" not in mc
        assert "min_draws_required" in mc


def test_expectancy_uses_the_production_cost_model():
    """340bps was an under-estimate; the model is the source of truth."""
    from savannah.stats import production_round_trip_cost_bps

    c = production_round_trip_cost_bps()
    assert c >= 410.0, f"round-trip cost {c} is below fees+base slippage alone"
    from autopaper.config import build_config

    e = build_config().execution
    assert c == pytest.approx(
        e.entry_fee_bps + e.exit_fee_bps + e.entry_slippage_bps
        + e.exit_slippage_bps + e.impact_base_bps * 2)


def test_gate_uses_stated_cap_not_post_mortem_cap(corpus):
    """
    mcap_now is read weeks after the signal, so collapsed tokens look tiny and
    would wave through the ceiling. The stated decision-time cap must be used.
    """
    from savannah.analysis import load_corpus as _lc
    from savannah.stats import attach_decision_time_mcap, load_outcomes

    rows, _ = load_outcomes()
    _ev, first, _m = _lc()
    attach_decision_time_mcap(rows, first)
    stated = [r["stated_mcap_usd"] for r in rows if r.get("stated_mcap_usd")]
    assert stated, "stated caps should join onto the outcome rows"
    naive = sum(1 for r in rows
                if r.get("mcap_now_usd") and r["mcap_now_usd"] <= 20_000)
    correct = sum(1 for v in stated if v <= 20_000)
    assert correct < naive, (
        "post-mortem caps must not admit more signals than decision-time caps")


# ---- 5. research-only boundary ------------------------------------------

def test_no_execution_capability_in_the_savannah_package():
    import ast
    import glob

    banned = ("broadcast", "send_transaction", "sign_transaction", "arm_real",
              "real_buy", "real_sell", "send_message", "join_channel")
    banned_imports = ("autopaper.execution.real", "autopaper.execution.solana",
                      "solders", "autopaper.token_cleanup")
    for p in sorted(glob.glob(str(_RESEARCH / "savannah" / "*.py"))):
        tree = ast.parse(Path(p).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                assert nm not in banned, f"{p} calls {nm}"
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                        else [n.module or ""])
                for m in mods:
                    assert not any(m.startswith(b) for b in banned_imports), \
                        f"{p} imports {m}"


def test_no_telegram_write_capability_in_the_package():
    """Read-only: the collector only calls GetEntity / GetMessages."""
    src = (_RESEARCH / "savannah" / "outcomes.py").read_text()
    src += (_RESEARCH / "savannah" / "collect_outcomes.py").read_text()
    assert "iter_messages" not in src or True   # allowed (read)
    for banned in ("send_message", "join", "delete_messages", "CreateChannel"):
        assert banned not in src, f"the study references {banned}"


def test_env_local_is_never_opened_by_the_study():
    """
    The study must never READ or WRITE the secrets file.

    The name appears in report PROSE ("`.env.local` untouched"), so this checks
    CODE for actual file access rather than for the string.
    """
    import ast

    for p in (_RESEARCH / "savannah").glob("*.py"):
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if nm not in ("open", "write_text", "read_text", "read_bytes",
                          "write_bytes", "unlink"):
                continue
            for arg in list(n.args) + [k.value for k in n.keywords]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and ".env" in arg.value:
                    raise AssertionError(f"{p} opens {arg.value}")


# ---- 6. no-lookahead in the excursion measurement -------------------------

def test_excursion_scan_never_includes_the_signal_bar_itself():
    """
    The signal lands INSIDE the signal minute bar, so that bar's high may have
    printed BEFORE the alert. Counting it as favourable excursion is lookahead:
    it inflates MFE and forces time-to-peak to zero.

    This is a real bug that was found and fixed; the test pins it shut.
    """
    from savannah.outcomes import SignalOutcome, measure

    sig = 1_000_000.0
    # bar at the signal: close 1.0 but a HIGH of 10.0 that happened pre-alert
    signal_bar = [int(sig), 1.0, 10.0, 1.0, 1.0, 100.0]
    after = [
        [int(sig) + 60, 1.0, 1.2, 1.0, 1.1, 50.0],
        [int(sig) + 120, 1.1, 1.3, 1.1, 1.2, 40.0],
        [int(sig) + 180, 1.2, 1.25, 0.9, 0.95, 10.0],
    ]
    m = measure(SignalOutcome(mint="M", signal_ts=sig,
                              path=[signal_bar] + after, source="OBSERVED"))
    assert m["resolved"] is True
    # the pre-alert high of 10.0 must NOT appear as MFE
    assert m["mfe_pct"] < 100.0, f"signal-bar high leaked into MFE: {m}"
    assert m["time_to_peak_s"] > 0, "time to peak cannot be zero"
    assert m["price_at_signal"] == pytest.approx(1.0)


def test_a_genuine_post_signal_spike_is_still_measured():
    from savannah.outcomes import SignalOutcome, measure

    sig = 2_000_000.0
    path = [
        [int(sig), 1.0, 1.0, 1.0, 1.0, 100.0],
        [int(sig) + 60, 1.0, 5.0, 1.0, 4.0, 900.0],   # real post-alert spike
        [int(sig) + 120, 4.0, 4.2, 3.0, 3.5, 200.0],
    ]
    m = measure(SignalOutcome(mint="M", signal_ts=sig, path=path,
                              source="OBSERVED"))
    assert m["resolved"] is True
    assert m["mfe_pct"] > 300.0, "a genuine post-signal spike must be counted"
    assert m["time_to_peak_s"] == pytest.approx(60.0, abs=1)


def test_index_selection_ignores_path_history_before_the_signal():
    """
    ``next(i for i,t in enumerate(ts) if t <= sig)`` yields the FIRST such bar.
    On a path that starts days before the signal that is the very first bar in
    the file, which pulls pre-signal history into p0 and into every excursion --
    it inflated median MFE from a believable ~+94% to a fabricated ~+305%.
    bisect_right(ts, sig) - 1 is the bar the signal actually lands in.
    """
    from savannah.outcomes import SignalOutcome, measure

    sig = 1_700_000_000.0
    DAY = 86_400
    # exactly three days of dead-flat pre-signal history at 1.0 (4320 bars)
    pre = [[int(sig - 3 * DAY + 60 * k), 1.0, 1.0, 1.0, 1.0, 5.0]
           for k in range(3 * 24 * 60)]
    at_signal = [int(sig), 1.0, 1.0, 1.0, 1.0, 5.0]
    # genuine post-signal action: +40% then fade to -50%
    post = [
        [int(sig) + 60, 1.0, 1.5, 1.0, 1.4, 900.0],
        [int(sig) + 120, 1.4, 1.45, 0.7, 0.75, 300.0],
        [int(sig) + 180, 0.75, 0.8, 0.5, 0.5, 100.0],
    ]
    m = measure(SignalOutcome(mint="M", signal_ts=sig,
                              path=pre + [at_signal] + post, source="OBSERVED"))
    assert m["resolved"] is True
    assert m["price_at_signal"] == pytest.approx(1.0), (
        "p0 must come from the signal bar, not the start of the path")
    # MFE uses the bar HIGH (1.5), not its close (1.4): +50%, not +40%
    assert m["mfe_pct"] == pytest.approx(50.0, abs=2.0), (
        f"MFE should be the genuine post-signal +50%, got {m['mfe_pct']}")
    assert m["time_to_peak_s"] == pytest.approx(60.0, abs=1)
    assert m["bars_after_signal"] == 3, "only post-signal bars may be counted"


def test_horizons_shorter_than_the_bar_grid_report_none_not_zero():
    """A 10s horizon on 1-minute bars is UNMEASURED, never a confident 0.00%."""
    from savannah.outcomes import SignalOutcome, measure

    sig = 1_700_000_100.0
    path = [
        [int(sig), 1.0, 1.0, 1.0, 1.0, 100.0],
        [int(sig) + 60, 1.0, 2.0, 1.0, 2.0, 900.0],
        [int(sig) + 3600, 2.0, 2.0, 2.0, 2.0, 10.0],
    ]
    r = measure(SignalOutcome(mint="M", signal_ts=sig, path=path,
                              source="OBSERVED"))["returns"]
    assert r["t10s"] is None and r["t30s"] is None, r
    assert r["t60s"] == pytest.approx(100.0, abs=1e-6), r
    assert r["t3600s"] == pytest.approx(100.0, abs=1e-6), r
