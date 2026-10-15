"""
Regression tests for the KOL study.

Each test corresponds to a real bug found while building this study, or to a
guarantee that a later edit could silently break. The bugs here were not
hypothetical: the epoch-unit and pool-selection bugs each caused 4/4 resolvable
signals to be rejected for fabricated reasons, which would have turned a
"nothing to study" result into a falsely quantified one.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

AI = Path(__file__).resolve().parents[2]
_PKG = AI / "research" / "kol"
for _p in (str(AI / "research"), str(AI / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kol import (  # noqa: E402
    SUB_MINUTE_DELAYS_NOT_TESTABLE, VERDICTS, EVIDENCE_TIER_AVAILABILITY)
from kol import costs as K  # noqa: E402
from kol import prices as P  # noqa: E402
from kol.analysis import (  # noqa: E402
    MIN_OOS_FOR_AN_EDGE, summarise_outcomes, tp_median_trap_flag,
    verdict_for)
from kol.collect import _Msg  # noqa: E402
from kol.identity import (  # noqa: E402
    CHANNELS, OPERATOR_CORRECTION, WALLETS, identity_table, verdict_for as vf)
from kol.signals import extract_mints, scan_messages  # noqa: E402

SEC = 1_700_000_000.0


def _vbars(start, closes, hi=None, lo=None, step=60.0):
    """VALID OHLCV bars derived from a close path."""
    out = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        h = max(hi[i] if hi else max(o, c), o, c)
        l = min(lo[i] if lo else min(o, c), o, c)
        out.append([start + step * i, o, h, l, c, 1000.0])
    return out


def _event(bars, *, ts=None, mint="MINT", stated_mc=None, stated_liq=None):
    ts = ts if ts is not None else bars[0][0]
    return P.Event(kol="X", username="chan", entity_id=1, message_id=1,
                   signal_ts=ts, mint=mint, symbol="S",
                   extraction="explicit_mint_field", raw_text="CA: " + mint,
                   stated_mcap_usd=stated_mc, stated_liquidity_usd=stated_liq)


# =================================================== 1. EPOCH UNIT HANDLING

def test_epoch_normalises_milliseconds_to_seconds():
    """
    DexScreener returns pairCreatedAt in MILLISECONDS; Telegram and OHLCV are in
    SECONDS. Comparing them directly made every pool look ~55 years too new, so
    all 4 recovered signals were rejected as 'pool created after the signal'.
    """
    assert P._epoch_s(1_700_000_000_000) == 1_700_000_000
    assert P._epoch_s(1_700_000_000) == 1_700_000_000
    assert P._epoch_s(1_700_000_000_000_000) == 1_700_000_000


def test_epoch_rejects_implausible_values_instead_of_guessing():
    assert P._epoch_s(None) is None
    assert P._epoch_s(0) is None
    assert P._epoch_s(-5) is None
    assert P._epoch_s(12345) is None          # not a plausible epoch


def test_pool_created_before_signal_is_not_rejected():
    """The direct regression for the unit bug, at the decision layer."""
    snap = P.PoolSnapshot(mint="M", pool="POOL", dex="pumpswap",
                          symbol="S",
                          liquidity_usd=0, mcap_usd=0, price_usd=0,
                          pair_created_at=SEC - 3600, resolved=True,
                          all_pairs=[{"pairAddress": "POOL",
                                      "pairCreatedAt": int((SEC - 3600) * 1000),
                                      "liquidity": {"usd": 50000},
                                      "baseToken": {"symbol": "S"},
                                      "marketCap": 100000, "priceUsd": "1e-5"}])

    def fake_snapshot(mint):
        return snap

    orig = P.snapshot
    P.snapshot = fake_snapshot
    orig_fetch = P._ohlcv_back_to
    P._ohlcv_back_to = lambda pool, ts: _vbars(ts - 300, [1.0, 1.0, 1.0, 1.0,
                                                         1.0, 1.1])
    try:
        e = _event(_vbars(SEC - 300, [1.0] * 6), ts=SEC)
        P.attach_price(e)
        assert e.resolved is True, e.unresolved_reason
        assert e.pool.pool == "POOL"
    finally:
        P.snapshot = orig
        P._ohlcv_back_to = orig_fetch


# ======================================== 2. POOL SELECTION / SURVIVORSHIP

def test_pool_selection_prefers_the_venue_that_existed_at_signal_time():
    """
    A graduated token migrates to a NEW, DEEPER pool. Selecting by depth alone
    picks the post-migration venue and then wrongly concludes the market did
    not exist when the KOL called it.
    """
    old = {"pairAddress": "OLD_POOL", "pairCreatedAt": int((SEC - 7200) * 1000),
           "liquidity": {"usd": 20000}, "baseToken": {"symbol": "S"},
           "marketCap": 50000, "priceUsd": "1e-5"}
    new = {"pairAddress": "NEW_POOL", "pairCreatedAt": int((SEC + 600) * 1000),
           "liquidity": {"usd": 900000}, "baseToken": {"symbol": "S"},
           "marketCap": 900000, "priceUsd": "1e-4"}
    snap = P.PoolSnapshot(mint="M", pool=new["pairAddress"], dex="pumpswap",
                          symbol="S", liquidity_usd=900000, mcap_usd=900000,
                          price_usd=1e-4, pair_created_at=SEC + 600,
                          resolved=True, all_pairs=[old, new])
    orig, orig_fetch = P.snapshot, P._ohlcv_back_to
    P.snapshot = lambda m: snap
    P._ohlcv_back_to = lambda pool, ts: _vbars(ts - 300, [1.0] * 6)
    try:
        e = _event(_vbars(SEC - 300, [1.0] * 6), ts=SEC)
        P.attach_price(e)
        assert e.resolved is True
        assert e.pool.pool == "OLD_POOL", (
            "chose the deeper post-signal pool instead of the signal-time venue")
    finally:
        P.snapshot, P._ohlcv_back_to = orig, orig_fetch


def test_event_is_rejected_when_no_pool_existed_at_signal_time():
    late = {"pairAddress": "LATE", "pairCreatedAt": int((SEC + 900) * 1000),
            "liquidity": {"usd": 5000}, "baseToken": {"symbol": "S"},
            "marketCap": 5000, "priceUsd": "1e-6"}
    snap = P.PoolSnapshot(mint="M", pool="LATE", dex="d", symbol="S",
                          liquidity_usd=5000, mcap_usd=5000, price_usd=1e-6,
                          pair_created_at=SEC + 900, resolved=True,
                          all_pairs=[late])
    orig = P.snapshot
    P.snapshot = lambda m: snap
    try:
        e = _event(_vbars(SEC - 300, [1.0] * 6), ts=SEC)
        P.attach_price(e)
        assert e.resolved is False
        assert "no pool existed at signal time" in e.unresolved_reason
    finally:
        P.snapshot = orig


# ==================================================== 3. DECISION-TIME FIELDS

def test_decision_time_mcap_is_never_backfilled_from_post_mortem():
    """
    A token that pumped after the signal would otherwise look like a low-cap
    buyable at decision time. The stated value is INFERRED; absent a stated
    value the field stays None rather than borrowing the analysis-time figure.
    """
    e = _event(_vbars(SEC, [1.0]), stated_mc=None)
    e.pool = P.PoolSnapshot(mint="M", pool="p", dex="d", symbol="S",
                            liquidity_usd=1000, mcap_usd=999_999_999,
                            price_usd=1, pair_created_at=SEC, resolved=True)
    assert e.decision_time_mcap_usd is None
    d = e.as_event()
    assert d["decision_time_mcap_usd"] is None
    assert "UNRESOLVED" in d["mcap_evidence"]
    assert d["post_mortem_pool"]["mcap_usd"] == 999_999_999  # kept for audit only


def test_stated_mcap_is_used_and_labelled_inferred():
    e = _event(_vbars(SEC, [1.0]), stated_mc=50_000.0)
    assert e.decision_time_mcap_usd == 50_000.0
    assert e.as_event()["mcap_evidence"] == "INFERRED_channel_stated"


# ==================================================== 4. NO SIGNAL-BAR LOOKAHEAD

def test_entry_price_is_the_decision_bar_close_not_its_high():
    sig = SEC
    bars = _vbars(sig - 120, [1.0, 1.0, 1.0, 1.0, 1.0],
                  hi=[1.0, 1.0, 1.0, 1.0, 9.0])   # 9x spike in signal bar
    e = _event(bars, ts=sig)
    e.bars = [tuple(b) for b in bars]
    e.decision_index = 4
    assert e.entry_ref_px == 1.0, "used the signal bar's HIGH as entry"


def test_excursions_never_include_the_signal_bar():
    sig = SEC
    bars = _vbars(sig - 120, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  hi=[1.0, 1.0, 1.0, 1.0, 1.0, 5.0])
    e = _event(bars, ts=sig)
    e.bars = [tuple(b) for b in bars]
    e.decision_index = 4
    ex = e.excursions()
    # the 5x high is on the bar AFTER entry, so it is legitimately reachable
    assert ex["mfe_pct"] > 100


def test_a_spike_in_the_decision_bar_is_never_counted_as_reachable():
    """
    Entry is at the CLOSE of the decision bar, so that bar's own high may have
    printed before the fill. Only bars starting AFTER it are reachable.
    """
    sig = SEC
    bars = _vbars(sig - 120, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  hi=[1.0, 1.0, 1.0, 1.0, 50.0, 1.05])   # spike in index 4
    e = _event(bars, ts=sig)
    e.bars = [tuple(b) for b in bars]
    e.decision_index = 4
    assert e.excursions()["mfe_pct"] < 10, (
        "a pre-entry spike inside the decision bar was counted as reachable")


# =========================================== 5. NO SUB-MINUTE FABRICATION

def test_sub_minute_horizons_return_none_not_zero():
    """
    On a 1-minute grid a 5s horizon has no bar. Returning 0.00% would print a
    confident number for something never measured.
    """
    bars = _vbars(SEC, [1.0, 1.0, 1.0, 1.0])
    e = _event(bars, ts=SEC)
    e.bars = [tuple(b) for b in bars]
    e.decision_index = 0
    for h in SUB_MINUTE_DELAYS_NOT_TESTABLE:
        assert e.return_at(h) is None, f"{h}s returned a fabricated value"


def test_first_new_bar_makes_the_return_measurable():
    bars = _vbars(SEC, [1.0, 2.0, 2.0])
    e = _event(bars, ts=SEC)
    e.bars = [tuple(b) for b in bars]
    e.decision_index = 0
    assert e.return_at(30.0) is None      # no new bar yet
    assert e.return_at(60.0) == pytest.approx(100.0)


def test_sub_minute_delays_are_declared_not_tested():
    ids = {n["id"] for n in __import__("kol.analysis", fromlist=["x"])
           .not_testable_register()}
    assert "sub_minute_entry_delay" in ids


# ================================================= 6. TAIL REMOVAL CORRECTNESS

def test_without_top_decile_removes_ten_percent_sorted_by_pnl():
    """
    Two bugs lived here once: slicing in SIGNAL order, and removing n//20 while
    labelling it 10%. Both flatter the strategies that most need flattering.
    """
    nets = [100.0] + [1.0] * 90 + [-50.0] * 9     # n=100
    events = []
    for i, n in enumerate(nets):
        events.append({
            "kol": "X", "mint": f"M{i}", "resolved": True,
            "returns": {"t300s": n}, "excursions": {"resolved": True},
            "decision_time_liquidity_usd": 10_000.0,
        })
    s = summarise_outcomes(events)
    assert s["n"] == 100
    assert s["without_top_decile"]["removed"] == 10     # 100//10, not //20
    assert s["without_top_decile"]["n"] == 90
    # the +100 winner must be gone
    assert s["without_top_decile"]["mean_net_pct"] < s["mean_net_pct"]


def test_signal_order_does_not_affect_tail_metrics():
    """The same outcomes shuffled must give the same tail statistics."""
    import random
    base = [{"kol": "X", "mint": f"M{i}", "resolved": True,
             "returns": {"t300s": (i % 7) - 3.0},
             "excursions": {"resolved": True},
             "decision_time_liquidity_usd": 10_000.0} for i in range(40)]
    a = summarise_outcomes(base)
    shuf = base[:]
    random.Random(7).shuffle(shuf)
    b = summarise_outcomes(shuf)
    assert a["mean_net_pct"] == pytest.approx(b["mean_net_pct"])
    assert a["without_top_decile"]["mean_net_pct"] == pytest.approx(
        b["without_top_decile"]["mean_net_pct"])


def test_tp_median_trap_is_flagged_not_hidden():
    """Positive median on negative expectancy must be called out."""
    events = [{"kol": "X", "mint": f"M{i}", "resolved": True,
               "returns": {"t300s": 15.0 if i < 80 else -95.0},
               "excursions": {"resolved": True},
               "decision_time_liquidity_usd": 10_000.0} for i in range(100)]
    s = summarise_outcomes(events)
    assert (s["median_net_pct"] or 0) > 0
    assert (s["mean_net_pct"] or 0) < 0
    assert tp_median_trap_flag(s) is True


# ================================================== 7. COSTS FROM PRODUCTION

def test_costs_come_from_production_config():
    from autopaper.config import build_config
    m = build_config().execution
    c = K.round_trip_bps(liquidity_usd=10_000.0)
    assert c.entry_fee_bps == m.entry_fee_bps
    assert c.exit_fee_bps == m.exit_fee_bps
    assert c.total_bps > 400


def test_impact_is_notional_and_liquidity_dependent():
    small = K.round_trip_bps(liquidity_usd=1_000.0).total_bps
    large = K.round_trip_bps(liquidity_usd=1_000_000.0).total_bps
    assert small > large, "cost must fall as the pool gets deeper"
    big_notional = K.round_trip_bps(notional_usd=100.0,
                                    liquidity_usd=10_000.0).total_bps
    assert big_notional > K.round_trip_bps(liquidity_usd=10_000.0).total_bps


def test_impact_is_not_charged_at_base_maximum():
    """The flat-530bps error: impact was charged as if always maximal."""
    from autopaper.config import build_config
    m = build_config().execution
    c = K.round_trip_bps(liquidity_usd=10_000.0)
    assert c.impact_bps < 10.0, c.impact_bps
    naive = (m.entry_fee_bps + m.exit_fee_bps + m.entry_slippage_bps
             + m.exit_slippage_bps + 2 * m.impact_base_bps)
    assert c.total_bps < naive


def test_unknown_liquidity_substitutes_the_production_assumption():
    """
    production's ``impact_bps`` replaces missing liquidity with
    ``model.assume_sol_usd``. Asserting the substitution rather than guessing a
    direction -- the earlier draft of this test assumed missing liquidity was
    always worse, and it is not.
    """
    from autopaper.config import build_config
    m = build_config().execution
    none_liq = K.round_trip_bps(liquidity_usd=None)
    at_assumed = K.round_trip_bps(liquidity_usd=m.assume_sol_usd)
    assert none_liq.total_bps == pytest.approx(at_assumed.total_bps)
    # a genuinely tiny pool is strictly more expensive than the assumption
    assert K.round_trip_bps(liquidity_usd=1.0).total_bps > none_liq.total_bps


# ================================================ 8. MINT EXTRACTION SAFETY

def test_mint_requires_an_explicit_field_or_a_dex_permalink():
    assert extract_mints("CA: " + "A" * 44) == [("A" * 44, "explicit_mint_field")]
    assert extract_mints("mint: " + "B" * 43)
    assert extract_mints("https://dexscreener.com/solana/" + "C" * 44)


def test_bare_base58_is_not_accepted_as_a_mint():
    """Chat text and wallet addresses share the 32-44 char pattern."""
    assert extract_mints("gm everyone, wallet is " + "D" * 44) == []
    assert extract_mints("that looks like a pump") == []


def test_promo_context_mints_are_recorded_but_excluded():
    msgs = [
        _Msg({"message_id": 1, "ts": SEC, "has_media": False,
              "text": "join our telegram group for guaranteed profit CA: "
                      + "E" * 44}),
    ]
    scan = scan_messages("K", 1, "chan", "T", msgs)
    assert len(scan.signals) == 0, "advertising was counted as a signal"
    assert scan.promo_messages == 1


def test_genuine_call_with_mint_is_accepted():
    msgs = [_Msg({"message_id": 2, "ts": SEC, "has_media": False,
                  "text": "CA: " + "F" * 44 + " $GOOD"})]
    scan = scan_messages("K", 1, "chan", "T", msgs)
    assert len(scan.signals) == 1
    assert scan.signals[0].mint == "F" * 44


def test_duplicate_mints_across_messages_are_counted_once_per_channel():
    m = "G" * 44
    msgs = [_Msg({"message_id": i, "ts": SEC + 60 * i, "has_media": False,
                  "text": f"CA: {m}"}) for i in range(4)]
    scan = scan_messages("K", 1, "chan", "T", msgs)
    assert len(scan.signals) == 4           # every message is a real event
    assert len(scan.unique_mints) == 1      # but one unique token


# ================================================== 9. IDENTITY INTEGRITY

def test_no_wallet_is_shared_between_two_kols():
    by_addr: dict[str, set[str]] = {}
    for w in WALLETS:
        by_addr.setdefault(w.address, set()).add(w.kol)
    shared = {a: ks for a, ks in by_addr.items() if len(ks) > 1}
    assert not shared, f"wallets silently merged across KOLs: {shared}"


def test_trunoest_wallet_is_not_the_decu_wallet():
    """The specification error must stay visible, not be quietly merged."""
    d = {w.kol: w.address for w in WALLETS}
    assert d["Decu"] != d["Trunoest"]
    oc = OPERATOR_CORRECTION
    assert oc["original_value"] == d["Decu"], (
        "the collision is no longer recorded, so the correction is unauditable")
    assert oc["corrected_value"] == d["Trunoest"]
    assert oc["independent_verification"] is False


def test_every_wallet_carries_confidence_and_evidence():
    for w in WALLETS:
        assert w.confidence in ("CONFIRMED", "LIKELY", "UNVERIFIED")
        assert len(w.evidence) > 40, f"{w.kol} has no recorded evidence"
        assert w.source


def test_rejected_channels_are_retained_not_dropped():
    tbl_ = identity_table()
    assert len(tbl_["channels"]) == len(CHANNELS)
    assert any(c["verdict"].startswith("REJECTED") for c in tbl_["channels"])


def test_verdict_uses_only_the_closed_vocabulary():
    for v in (verdict_for("X", has_signal=False, has_oos_edge=False, n_measured=0),
              verdict_for("X", has_signal=True, has_oos_edge=False,
                          n_measured=100),
              verdict_for("X", has_signal=True, has_oos_edge=True,
                          n_measured=100)):
        assert v in VERDICTS


def test_wallet_quality_cannot_rescue_an_unobservable_signal():
    """A wallet cannot substitute for a signal: no signal -> DATA INSUFFICIENT."""
    assert verdict_for("X", has_signal=False, has_oos_edge=False,
                       n_measured=0) == "DATA INSUFFICIENT"


def test_a_handful_of_measured_returns_is_insufficient_not_a_failed_test():
    """
    1 measured return must not be reported as "RESEARCH ONLY - NO VALIDATED
    EDGE". That wording asserts the KOL was evaluated on evidence and rejected;
    one trade is not evidence.
    """
    assert verdict_for("X", has_signal=True, has_oos_edge=False,
                       n_measured=1, min_measured=20) == "DATA INSUFFICIENT"
    assert verdict_for("X", has_signal=True, has_oos_edge=False,
                       n_measured=25, min_measured=20) == \
        "RESEARCH ONLY - NO VALIDATED EDGE"


def test_small_sample_is_insufficient_evidence_not_an_edge():
    events = [{"kol": "X", "mint": f"M{i}", "resolved": True,
               "returns": {"t300s": 50.0}, "excursions": {"resolved": True},
               "decision_time_liquidity_usd": 10_000.0} for i in range(5)]
    s = summarise_outcomes(events)
    assert s["n"] == 5 < MIN_OOS_FOR_AN_EDGE
    assert s["significant_at_95pct"] is False


# ============================================ 10. UNRESOLVED HANDLING

def test_unresolved_event_is_explicit_and_carries_a_reason():
    e = _event(_vbars(SEC, [1.0]), ts=SEC)
    e.resolved = False
    e.unresolved_reason = "aggregator returned no OHLCV"
    d = e.as_event()
    assert d["resolved"] is False
    assert d["unresolved_reason"]
    assert d["n_bars"] == 0


def test_evidence_tier_is_mark_only_and_flags_never_executable():
    e = _event(_vbars(SEC, [1.0]), ts=SEC)
    d = e.as_event()
    assert d["evidence_tier"] == "OBSERVED_MARK"
    assert d["is_executable"] is False


def test_all_non_mark_tiers_are_declared_unavailable():
    for tier in ("EXECUTABLE_QUOTE", "SIMULATED_FILL", "ACTUAL_FILL",
                 "EXECUTION_NATIVE_REFERENCE"):
        assert tier in EVIDENCE_TIER_AVAILABILITY
        assert "UNAVAILABLE" in EVIDENCE_TIER_AVAILABILITY[tier] \
            or "NEVER AVAILABLE" in EVIDENCE_TIER_AVAILABILITY[tier]


# ============================================= 11. RESEARCH-ONLY BOUNDARY

def _code_only(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)


def test_package_has_no_signing_or_broadcast_capability():
    banned_calls = {"broadcast", "sign_transaction", "send_transaction",
                    "send_message", "join_channel", "arm_real", "real_buy",
                    "real_sell", "transfer", "close_account"}
    banned_roots = {"solders", "solana", "base58", "nacl"}
    banned_modules = ("autopaper.execution.real", "autopaper.wallet",
                      "autopaper.token_cleanup", "autopaper.safety",
                      "autopaper.engine", "autopaper.l25")
    for p in sorted(_PKG.glob("*.py")):
        tree = ast.parse(p.read_text())
        # imports must be checked structurally: a URL containing "solana" is not
        # an import, and a substring scan would both false-positive on those and
        # miss a `from x import y` form.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for al in node.names:
                    root = al.name.split(".")[0]
                    assert root not in banned_roots, f"{p.name} imports {al.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert mod.split(".")[0] not in banned_roots, \
                    f"{p.name} imports from {mod}"
                assert mod not in banned_modules, f"{p.name} imports {mod}"
        code = _code_only(p)
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Call):
                fn = node.func
                nm = (fn.attr if isinstance(fn, ast.Attribute)
                      else getattr(fn, "id", ""))
                assert nm not in banned_calls, f"{p.name} calls {nm}"


def test_package_never_touches_env_or_wallet_files():
    for p in sorted(_PKG.glob("*.py")):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            nm = (fn.attr if isinstance(fn, ast.Attribute)
                  else getattr(fn, "id", ""))
            if nm not in ("open", "read_text", "write_text", "read_bytes",
                          "write_bytes", "unlink"):
                continue
            for arg in list(node.args) + [k.value for k in node.keywords]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    low = arg.value.lower()
                    assert ".env" not in low, f"{p.name} opens {arg.value}"
                    assert "wallet" not in low or "wallet_probe" in arg.value, \
                        f"{p.name} opens {arg.value}"


def test_no_kol_profile_was_added_to_production():
    from autopaper import strategy_profile as sp
    values = {v.value for v in sp.StrategyProfile}
    assert not any("KOL" in v.upper() for v in values)
    for d in (sp.PROFILE_MAX_HOLD_S, sp.PROFILE_EXIT_STRATEGY,
              sp.PROFILE_LABELS, sp.PROFILE_NOTES):
        assert not any("KOL" in str(getattr(k, "value", k)).upper() for k in d)


def test_package_does_not_import_production_l25_or_strategy_rules():
    for p in sorted(_PKG.glob("*.py")):
        code = _code_only(p)
        assert "autopaper.l25" not in code, f"{p.name} imports production L25"
        assert "autopaper.strategy_profile" not in code
        assert "autopaper.engine" not in code


def test_write_roots_are_confined():
    from kol import ALLOWED_WRITE_ROOTS
    assert ALLOWED_WRITE_ROOTS == ("reports/research/kol", "research/kol/cache")


def test_cost_module_reads_production_and_hardcodes_no_rates():
    """
    No fee/slippage/impact rate may appear as a NUMERIC LITERAL. Prose that
    mentions "530bps" while explaining why that number is wrong is fine -- a
    substring scan would flag the explanation as the offence.
    """
    tree = ast.parse((_PKG / "costs.py").read_text())
    banned = {85.0, 120.0, 60.0, 90.0, 530.0, 530}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            assert node.value not in banned, (
                f"costs.py hardcodes a cost rate literal: {node.value}")
    # and the module must actually import the production primitives
    src = (_PKG / "costs.py").read_text()
    assert "build_config" in src and "impact_bps" in src

def test_summary_survives_resolved_events_with_no_measurable_horizon():
    """
    A resolved event (bars exist around the signal) can still lack a bar at the
    evaluation horizon. The denominator must be the count of MEASURED returns,
    not of resolved events -- otherwise the win rate divides by zero.
    """
    events = [{"kol": "X", "mint": f"M{i}", "resolved": True,
               "returns": {"t300s": None},          # tape too short
               "excursions": {"resolved": False},
               "decision_time_liquidity_usd": 10_000.0} for i in range(4)]
    s = summarise_outcomes(events)
    assert s["insufficient"] is True
    assert s["n"] == 0
    assert "300s evaluation horizon" in s["reason"]


def test_summary_handles_a_single_measured_return():
    events = [{"kol": "X", "mint": "M0", "resolved": True,
               "returns": {"t300s": 12.0}, "excursions": {"resolved": True},
               "decision_time_liquidity_usd": 10_000.0}]
    s = summarise_outcomes(events)
    assert s["n"] == 1
    assert s["significant_at_95pct"] is False, "one trade is never significant"
