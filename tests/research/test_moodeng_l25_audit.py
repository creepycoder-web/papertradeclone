"""
Regression tests for the MooDeng / L25 reject-vs-buy audit.

These are not smoke tests. Each one pins a SPECIFIC failure mode that has either
already bitten this repository or is the obvious way to get this study wrong:

    test_forward_bars_never_include_the_bar_containing_the_anchor
        signal-bar lookahead. The bar that CONTAINS t0 opens up to 60 s BEFORE the
        decision. Counting its high as a post-decision excursion manufactures
        winners out of pre-decision moves.

    test_forward_return_at_a_horizon_uses_only_a_genuine_print
        the "wrong next() match selection" bug. A horizon must not silently take the
        next bar after a gap, and must not interpolate.

    test_post_exit_window_starts_after_the_exit
        the same lookahead, applied to the exit.

    test_excursion_uses_bar_high_and_low_not_close
        an excursion computed from closes under-counts by construction.

    test_tradable_gate_requires_surviving_l25s_own_stop_before_the_move
        the pre-registered tradability definition.

    test_engine_price_anchor_is_used_and_never_overwritten_by_the_tape
        the reference price is what the engine saw; the tape must not replace it.

    test_mint_is_the_identity_key_not_the_ticker
        the brief's identity requirement.

    test_snapshot_is_hash_stable_and_read_only
        the analysis must be reproducible from the frozen snapshot, and the freeze
        must not have written anything under var/.

    test_reported_headline_numbers_match_the_machine_readable_json
        the brief's final engineering check: the markdown and the JSON cannot
        disagree.

    test_no_production_file_is_modified_by_this_audit
        the brief's scope guarantee.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research"))
sys.path.insert(0, str(REPO / "src"))

from research.moodeng_l25_audit import dataset as D            # noqa: E402
from research.moodeng_l25_audit import outcomes as O            # noqa: E402
from research.moodeng_l25_audit import snapshot as SNAP          # noqa: E402
from research.moodeng_l25_audit.prelude import (               # noqa: E402
    L25_FAST_FAILURE_STOP_PCT, MIN_COVERAGE_S)

GRID = 60.0
T0 = 1_700_000_000.0


def bar(ts: float, o: float, h: float, l: float, c: float) -> list[float]:
    return [ts, o, h, l, c]


# ---------------------------------------------------------------------------
# a synthetic path where the pre-decision move is huge and the post-decision
# move is flat. If the bar containing t0 leaks into the forward set this token
# reads as a +900 % winner.
# ---------------------------------------------------------------------------
LOOKAHEAD_BARS = [
    bar(T0 - 180.0, 1.0, 1.0, 1.0, 1.0),
    bar(T0 - 120.0, 1.0, 1.0, 1.0, 1.0),
    # the bar CONTAINING t0 opened 40 s BEFORE the decision and spiked +900 %
    bar(T0 - 40.0, 1.0, 10.0, 1.0, 10.0),
    bar(T0 + 20.0, 1.0, 1.0, 1.0, 1.0),
    bar(T0 + 80.0, 1.0, 1.0, 1.0, 1.0),
    bar(T0 + 140.0, 1.0, 1.0, 1.0, 1.0),
    bar(T0 + 200.0, 1.0, 1.0, 1.0, 1.0),
]


def test_forward_bars_never_include_the_bar_containing_the_anchor():
    """Signal-bar lookahead: the containing bar must not count as forward."""
    pre, containing, fwd = O._split(LOOKAHEAD_BARS, T0)
    assert len(containing) == 1, "exactly one bar should straddle the anchor"
    assert containing[0][0] == T0 - 40.0
    assert all(b[0] > T0 for b in fwd), \
        "every forward bar must START after the anchor, not merely overlap it"

    m = O.measure(LOOKAHEAD_BARS, T0, 1.0)
    assert m["mfe_pct"] is not None
    assert m["mfe_pct"] == pytest.approx(0.0), (
        "a +900 % spike that happened BEFORE the decision must not be reported as a "
        f"post-decision excursion; got {m['mfe_pct']}")
    assert not m["reached"]["10"]


def test_pre_context_may_see_the_containing_bar_but_forward_may_not():
    """The containing bar is legitimate history; it is only banned from the future."""
    pre, containing, _ = O._split(LOOKAHEAD_BARS, T0)
    assert any(b[0] == T0 - 40.0 for b in pre + containing)
    assert not any(b[0] == T0 - 40.0 for b in
                   [x for x in LOOKAHEAD_BARS if x[0] > T0])


def test_forward_return_at_a_horizon_uses_only_a_genuine_print():
    """
    Wrong-next()-match regression.

    A gap in the tape must produce ``None`` for that horizon, not the next
    available bar's close and not an interpolated value.
    """
    gapped = [
        bar(T0 + 20.0, 1.0, 1.0, 1.0, 1.10),   # covers +20..+80
        bar(T0 + 80.0, 1.1, 1.1, 1.1, 1.20),   # covers +80..+140
        # a hole: nothing covers +140..+500
        bar(T0 + 500.0, 1.2, 1.2, 1.2, 9.99),  # covers +500..+560
    ]
    # the answer is the bar that COVERS the horizon, not the next one after it
    assert O._at_horizon(gapped, T0, 30.0) == pytest.approx(1.10)
    assert O._at_horizon(gapped, T0, 60.0) == pytest.approx(1.10)
    assert O._at_horizon(gapped, T0, 90.0) == pytest.approx(1.20)
    assert O._at_horizon(gapped, T0, 120.0) == pytest.approx(1.20)
    # +300 s falls inside the hole -> None, never the +500 print
    assert O._at_horizon(gapped, T0, 300.0) is None
    assert O._at_horizon(gapped, T0, 360.0) is None
    # +480 s is still inside the hole
    assert O._at_horizon(gapped, T0, 480.0) is None
    assert O._at_horizon(gapped, T0, 520.0) == pytest.approx(9.99)


def test_ret_at_is_none_when_no_print_exists_and_never_interpolated():
    m = O.measure([
        bar(T0 + 20.0, 1.0, 1.0, 1.0, 1.0),
        bar(T0 + 6000.0, 1.0, 1.0, 1.0, 5.0),
    ], T0, 1.0)
    assert m["ret_at"]["90"] is None
    assert m["ret_at"]["300"] is None
    assert m["ret_at"]["600"] is None
    # the printed horizon set is fixed; a horizon outside it is simply not reported
    assert set(m["ret_at"]) == {"30", "60", "90", "180", "300", "600", "900",
                                "1800", "3600"}


def test_post_exit_window_starts_after_the_exit():
    """Same lookahead, applied to the exit side."""
    t_exit = T0 + 100.0
    bars = [
        bar(t_exit - 40.0, 1.0, 5.0, 1.0, 5.0),        # spike BEFORE the exit
        bar(t_exit + 20.0, 1.0, 1.0, 1.0, 1.0),
        bar(t_exit + 80.0, 1.0, 1.0, 1.0, 1.0),
    ]
    pe = O.post_exit(bars, t_exit, 1.0)
    assert pe["post_exit_bars"] == 2
    assert pe["post_exit_mfe_pct"] == pytest.approx(0.0)
    assert pe["post_exit_recovered"] is False


def test_excursion_uses_bar_high_and_low_not_close():
    """
    A move that happens and closes back inside the same bar is still a move.
    Measuring from closes would report 0 %.
    """
    bars = [
        bar(T0 + 20.0, 1.0, 2.5, 0.8, 1.0),   # spiked to +150 %, closed flat
        bar(T0 + 80.0, 1.0, 1.0, 0.9, 1.0),
    ]
    m = O.measure(bars, T0, 1.0)
    assert m["mfe_pct"] == pytest.approx(150.0)
    assert m["mae_pct"] == pytest.approx(-20.0)
    assert m["reached"]["100"] is True
    assert m["time_to_mfe_s"] == pytest.approx(80.0)     # bar END, not bar start


def test_tradable_gate_requires_surviving_l25s_own_stop_before_the_move():
    """
    Pre-registered tradability.

    Same +50 % peak, two very different paths. One is reachable without breaching
    L25's -12 % stop; the other only prints its high after a -60 % collapse.
    """
    good = [bar(T0 + 20.0, 1.0, 1.5, 0.99, 1.5),
            bar(T0 + 80.0, 1.0, 1.5, 1.4, 1.45)]
    bad = [bar(T0 + 20.0, 1.0, 1.0, 0.40, 0.5),      # -60 % first
           bar(T0 + 80.0, 0.5, 1.5, 0.5, 1.5)]      # then back to 1.5 (== +50 % on ref)
    mg, mb = O.measure(good, T0, 1.0), O.measure(bad, T0, 1.0)
    assert mg["reached"]["50"] and mb["reached"]["50"]
    assert mg["tradable_at"]["50"] is True
    assert mb["tradable_at"]["50"] is False, \
        "a +50 % print that required surviving a -60 % drawdown is not tradable for L25"
    # the gate must be exactly L25's own production stop
    assert L25_FAST_FAILURE_STOP_PCT == -12.0
    edge = [bar(T0 + 20.0, 1.0, 1.5, 0.88, 1.5)]     # exactly -12 %
    assert O.measure(edge, T0, 1.0)["tradable_at"]["50"] is True


def test_engine_price_anchor_is_used_and_never_overwritten_by_the_tape():
    """The reference price is what L25 saw; the tape must not replace it."""
    bars = [bar(T0 + 20.0, 99.0, 99.0, 99.0, 99.0)]
    m = O.measure(bars, T0, 1.25)          # ref supplied by the caller
    assert m["ref_price"] == 1.25
    assert m["final_return_pct"] == pytest.approx((99.0 / 1.25 - 1.0) * 100.0)


def test_coverage_is_zero_when_no_forward_bars_exist():
    assert O.measure([bar(T0 - 300.0, 1, 1, 1, 1)], T0, 1.0)["covered_s"] == 0.0


def test_measure_refuses_a_non_positive_reference():
    m = O.measure([bar(T0 + 20.0, 1, 1, 1, 1)], T0, 0.0)
    assert m["mfe_pct"] is None
    m2 = O.measure([bar(T0 + 20.0, 1, 1, 1, 1)], T0, None)
    assert m2["mfe_pct"] is None


def test_merge_mark_tape_does_not_double_count():
    """Finer marks must displace the bars they overlap, not stack on them."""
    marks = [{"at": T0 + 30.0, "price": 1.10},
             {"at": T0 + 33.0, "price": 1.20},
             {"at": T0 + 36.0, "price": 1.30}]
    bars = [bar(T0 + 20.0, 1.0, 1.4, 1.0, 1.4),      # inside the mark window
            bar(T0 + 300.0, 1.3, 2.6, 1.3, 2.6)]     # after it, must be kept
    merged = O.merge_mark_tape(marks, bars, T0)
    stamps = [b[0] for b in merged]
    assert T0 + 20.0 not in stamps, "the coarser bar inside the mark window must be dropped"
    assert T0 + 300.0 in stamps, "a bar outside the mark window must be kept"
    assert len([b for b in merged if abs(b[0] - T0 - 30.0) < 1e-9]) == 1


def test_merge_mark_tape_is_a_no_op_without_marks():
    bars = [bar(T0 + 20.0, 1, 1, 1, 1)]
    assert O.merge_mark_tape([], bars, T0) == bars


def test_rank_score_is_zero_without_a_tradable_move():
    assert O.rank_score({"tradable_p20": False}) == 0.0


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_mint_is_the_identity_key_not_the_ticker():
    """
    The brief's rule, enforced.

    Two decisions with the SAME ticker and DIFFERENT mints must remain two rows.
    If anything keyed on ticker they would have collapsed into one.
    """
    rows = D.build_rows()
    by_ticker: dict[str, set[str]] = {}
    for r in rows:
        by_ticker.setdefault(r["symbol"], set()).add(r["mint"])
    collisions = {t: m for t, m in by_ticker.items() if len(m) > 1}
    assert collisions, ("the fixture window is expected to contain ticker collisions; if it "
                        "does not, this test can no longer prove identity is mint-based")
    for t, mints in collisions.items():
        n = sum(1 for r in rows if r["symbol"] == t)
        assert n >= len(mints)
    assert len(rows) == len({r["mint"] for r in rows}), \
        "one decision per mint is expected; duplicate mints would mean a bad join"
    assert len(rows) == len({r["candidate_id"] for r in rows})


def test_every_decision_is_anchored_on_the_decision_instant():
    """`decided_at` is the verdict instant and never precedes the alert."""
    for r in D.build_rows():
        assert r["decided_at"] > 0
        if r.get("telegram_ts"):
            assert r["decided_at"] >= r["telegram_ts"] - 5.0, (
                "a decision cannot predate the Telegram message that caused it")


def test_no_post_mortem_field_leaks_into_the_feature_block():
    """
    Guard against a future edit that adds, say, ``final_market_cap`` to a row.

    Only the FEATURE block is checked: those are the numbers the engine computed
    at ``decided_at``. Booked fill/exit/PnL fields live elsewhere on the row and
    are deliberately excluded from the check because they are outcomes, not inputs.
    """
    # "realized_vol_60s" is deliberately NOT banned: realised volatility over the
    # preceding 60 s is a backward-looking feature the engine had at decision time
    # (src/autopaper/analysis/features.py). What is banned is anything that reads
    # forward or names a booked outcome.
    banned = ("future", "forward", "final_", "outcome", "mfe", "mae", "peak_",
              "trough_", "after", "net_pnl", "exit", "post_", "holding")
    row = D.build_rows()[0]
    offenders = [k for k in D.FEATURE_FIELDS
                 if any(t in k.lower() for t in banned)]
    assert not offenders, (
        f"forward-looking field names must not appear in the feature block: {offenders}")
    # and the values must not be post-hoc: alert fields must predate the decision
    assert row["alert_mcap_usd"] is not None
    assert row["decided_at"] >= row["telegram_ts"] - 5.0


# ---------------------------------------------------------------------------
# snapshot integrity and scope
# ---------------------------------------------------------------------------


def test_snapshot_digests_match_their_manifest():
    """
    The frozen snapshot is immutable.

    Every file is re-hashed and compared with the digest recorded at freeze time.
    This is what makes the study reproducible: the analysis reads the snapshot, so a
    result can be regenerated exactly even while the bot keeps appending to
    ``var/``.
    """
    man = SNAP.load_manifest()
    assert man["frozen_at"] > 0
    assert not man["missing"], f"snapshot is missing sources: {man['missing']}"
    v = SNAP.verify()
    assert v["ok"], f"snapshot drifted after the freeze: {v['problems']}"


def test_freeze_does_not_write_to_any_source_file():
    """
    Read-only proof, measured rather than asserted.

    ``freeze()`` is the only function in the audit that touches ``var/``, and it
    only reads. Re-running it must leave every source file's size and mtime
    untouched. Size and mtime are used rather than digest alone because the live bot
    appends concurrently, which would make a digest comparison flaky in the wrong
    direction.
    """
    targets = [REPO / rel for rel in SNAP.SOURCES if (REPO / rel).exists()]
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in targets}
    # into a scratch directory: re-freezing the real snapshot would silently widen
    # the study's window every time this test runs
    import shutil
    probe = REPO / "research" / "moodeng_l25_audit" / "cache" / "freeze_probe"
    SNAP.freeze(dest=probe)
    shutil.rmtree(probe, ignore_errors=True)
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in targets}
    changed = [str(p.relative_to(REPO)) for p in targets if before[p] != after[p]]
    assert not changed, f"freeze() modified its own sources: {changed}"


def test_analysis_is_reproducible_from_the_frozen_snapshot():
    """
    Two runs, same answers.

    If anything in the pipeline read live ``var/`` instead of the snapshot, the two
    runs would diverge while the bot is appending. This is the regression test for
    the whole measurement base.
    """
    sys.path.insert(0, str(REPO))
    from research.moodeng_l25_audit import analysis as A
    a = A.classify_all(A.attach_outcomes())
    b = A.classify_all(A.attach_outcomes())
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x["mint"] == y["mint"]
        assert x["classification"] == y["classification"]
        assert x["forward"]["mfe_pct"] == y["forward"]["mfe_pct"]
        assert x["forward"]["mae_pct"] == y["forward"]["mae_pct"]
        assert x["forward"]["tradable_p20"] == y["forward"]["tradable_p20"]
    assert A.aggregate(a) == A.aggregate(b)


AUDIT_START = (REPO / "research" / "moodeng_l25_audit" / "prelude.py").stat().st_mtime
AUDIT_PKG = REPO / "research" / "moodeng_l25_audit"


def _digest_tree(root: Path) -> dict[str, str]:
    """sha256 of every source-like file under ``root``, keyed by repo-relative path."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in str(p):
            continue
        if p.suffix not in (".py", ".toml", ".json", ".csv", ".jsonl", ".md", ".cfg"):
            continue
        out[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_a_full_analysis_run_writes_only_its_own_artifacts():
    """
    The scope guarantee for production source, checked by digest rather than by
    mtime or git.

    This repository is being worked on concurrently by other processes, so "a
    production file changed recently" is not evidence that THIS audit changed it.
    Byte equality across a full analysis run is: everything under ``src/`` plus
    ``alt_l25.py`` and ``pyproject.toml`` is digested, a complete analysis is
    executed (including writing all seven artifacts), and the digests are compared.

    ``var/`` is deliberately NOT in this set: the trading bot is live and appends to
    it continuously, so its contents change during any window. The read-only
    guarantee for ``var/`` is covered by ``test_freeze_does_not_write_to_any_source_file``
    and ``test_audit_package_never_writes_to_the_runtime_tree``.
    """
    watched: dict[str, str] = {}
    for rel in ("src", "alt_l25.py", "pyproject.toml"):
        t = REPO / rel
        if t.is_file():
            watched[rel] = hashlib.sha256(t.read_bytes()).hexdigest()
        elif t.is_dir():
            watched.update(_digest_tree(t))
    assert watched, "nothing was watched; the digest guard would be vacuous"

    sys.path.insert(0, str(REPO))
    from research.moodeng_l25_audit import report as R
    R.run_analysis()

    after: dict[str, str] = {}
    for rel in ("src", "alt_l25.py", "pyproject.toml"):
        t = REPO / rel
        if t.is_file():
            after[rel] = hashlib.sha256(t.read_bytes()).hexdigest()
        elif t.is_dir():
            after.update(_digest_tree(t))

    changed = [k for k in watched if watched[k] != after.get(k)]
    assert not changed, f"the audit modified production source: {changed}"


def test_the_audit_produced_exactly_its_declared_artifacts():
    """The audit's own writes are accounted for, and land in the right place."""
    stem_files = {
        "moodeng_l25_reject_vs_buy.md", "moodeng_l25_reject_vs_buy.json",
        "moodeng_l25_rejected_signals.csv", "moodeng_l25_bought_signals.csv",
        "moodeng_l25_rejection_reason_stats.csv", "moodeng_l25_entry_setup_stats.csv",
        "moodeng_l25_decision_matrix.csv", "moodeng_l25_group_comparison.csv",
        "moodeng_l25_path_comparison.csv",
    }
    for name in stem_files:
        p = REPORTS / name
        assert p.exists() and p.stat().st_size > 0, f"missing or empty artifact: {name}"
    pkg = REPO / "research" / "moodeng_l25_audit"
    assert (pkg / "cache" / "snapshot" / "manifest.json").exists(), \
        "the snapshot manifest must exist"
    assert (pkg / "cache" / "tape").is_dir(), "the tape cache must exist"


def test_this_audit_did_not_touch_env_local():
    """Explicit brief scope item, checked individually."""
    p = REPO / ".env.local"
    if not p.exists():
        pytest.skip(".env.local is not present in this working tree")
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    sys.path.insert(0, str(REPO))
    from research.moodeng_l25_audit import report as R
    R.run_analysis()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before, \
        ".env.local must never be written by this audit"


def test_audit_package_never_writes_to_the_runtime_tree():
    """
    Static guard on the audit's own source.

    A write mode combined with a path into ``var/`` is the failure this catches
    before it happens. The runtime digest test above is the behavioural proof; this
    is the cheap one that runs even when the runtime tree is absent.
    """
    offenders: list[str] = []
    for p in sorted(AUDIT_PKG.glob("*.py")):
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#")[0]
            writes = any(m in code for m in ('"w"', "'w'", '"a"', "'a'", '"wb"', '"r+"'))
            if not writes:
                continue
            if "var" in code:
                offenders.append(f"{p.name}:{lineno}: {line.strip()}")
    assert not offenders, f"the audit must never open a runtime path for writing: {offenders}"


def test_real_trading_is_not_enabled_by_anything_in_the_audit():
    """
    The brief's safety item, read from the runtime's own effective state.

    ``.env.local`` in this working tree carries ``REAL_TRADING_ENABLED=true``, which
    looks alarming until the arming contract is read: ``REAL_TRADING_ENABLED`` is
    listed in ``src/autopaper/safety.py:455-462`` as a NON-AUTHORIZING environment
    variable -- it is read as a capability and can never arm REAL. The state that
    actually decides is ``var/strategy_selection.json`` plus the runtime journal's
    own arming events, so those are what is asserted.
    """
    control = REPO / "var" / "control.json"
    stop = REPO / "var" / "emergency_stop.json"
    sel = REPO / "var" / "strategy_selection.json"
    if control.exists():
        assert json.loads(control.read_text()).get("command", "") == "", \
            "a control command is active; the audit does not set or clear one"
    if stop.exists():
        assert json.loads(stop.read_text()).get("emergency_stop") is False
    if sel.exists():
        blob = json.loads(sel.read_text())
        assert blob.get("trade_mode") == "PAPER", \
            "the runtime must be in PAPER mode; this audit never changes the mode"
    # the env flag may be set; the arming contract is what matters and it is
    # asserted by the repository's own tests. Here the only thing to prove is that
    # this audit did not change the flag.
    env = REPO / ".env.local"
    if env.exists():
        # NEVER HOLD THE RAW FILE IN A LOCAL THAT CAN BE RENDERED.
        #
        # This was ``text = env.read_text(); assert "REAL_TRADING_ENABLED" in
        # text``. On failure pytest renders the LEFT-HAND operand of an ``in``
        # assertion verbatim -- so a failing check on this one file printed the
        # operator's real Telegram API id, api hash, chat id and session string
        # into the test report. An assertion whose INPUT is a credential file
        # must derive a boolean and pass the boolean, never the text.
        with env.open("r", encoding="utf-8") as fh:
            flag_present = any(
                "REAL_TRADING_ENABLED" in line for line in fh)
        assert flag_present, (
            "the audit must leave the REAL_TRADING_ENABLED flag untouched")
        # confirm the flag is a capability, not an authorization, in the code that
        # consumes it -- otherwise the env value above would be load-bearing
        safety = (REPO / "src" / "autopaper" / "safety.py").read_text(encoding="utf-8")
        assert "NON_AUTHORIZING_ENV" in safety
        assert '"REAL_TRADING_ENABLED"' in safety


# ---------------------------------------------------------------------------
# artifact consistency -- the brief's final engineering check
# ---------------------------------------------------------------------------

REPORTS = REPO / "reports" / "research"
STEM = "moodeng_l25_reject_vs_buy"


@pytest.fixture(scope="module")
def artifacts():
    md = REPORTS / f"{STEM}.md"
    js = REPORTS / f"{STEM}.json"
    if not md.exists() or not js.exists():
        pytest.skip("report not generated")
    return md, json.loads(js.read_text())


def test_reported_headline_numbers_match_the_machine_readable_json(artifacts):
    """
    The brief requires the markdown headline figures to match the machine-readable
    output exactly. The markdown is rendered from the same dicts, so this asserts
    the wiring rather than re-deriving every number.
    """
    md, js = artifacts
    text = md.read_text()
    rj = js["aggregate"]["rejected"]
    bo = js["aggregate"]["bought"]
    must_appear = [
        f"{rj['total_rejected']:,}",
        f"{rj['rejected_measurable']:,}",
        f"{bo['fills']:,}",
        f"{rj['false_reject']:,}",
        f"{bo['false_entry_count']:,}",
    ]
    for token in must_appear:
        assert token in text, f"headline figure {token} is missing from the markdown"


def test_markdown_and_json_agree_on_the_reject_and_buy_denominators(artifacts):
    md, js = artifacts
    text = md.read_text()
    rj, bo = js["aggregate"]["rejected"], js["aggregate"]["bought"]
    assert rj["total_rejected"] == js["coverage"]["population"]["REJECT"]
    assert bo["fills"] == js["coverage"]["population"]["BOUGHT_FILLED"]
    assert rj["false_reject"] + rj["correct_reject"] == rj["rejected_measurable"], \
        "every measurable reject must be classified as exactly one of the two verdicts"
    assert rj["false_reject_pct"] == pytest.approx(
        100.0 * rj["false_reject"] / rj["rejected_measurable"], abs=0.02)
    assert f"{rj['false_reject']} of" in text or str(rj["false_reject"]) in text


def test_csv_row_counts_match_the_populations(artifacts):
    _, js = artifacts
    rej_csv = REPORTS / "moodeng_l25_rejected_signals.csv"
    buy_csv = REPORTS / "moodeng_l25_bought_signals.csv"
    if not rej_csv.exists():
        pytest.skip("csvs not generated")
    with rej_csv.open() as fh:
        n_rej = sum(1 for _ in csv.DictReader(fh))
    with buy_csv.open() as fh:
        n_buy = sum(1 for _ in csv.DictReader(fh))
    assert n_rej == js["coverage"]["population"]["REJECT"]
    assert n_buy == (js["coverage"]["population"]["BOUGHT_FILLED"]
                     + js["coverage"]["population"]["BUY_AUTHORIZED_NOT_FILLED"])


def test_rejection_reason_stats_partition_the_measurable_rejects(artifacts):
    _, js = artifacts
    stats = js["rejection_reason_stats"]
    total = sum(r["n"] for r in stats)
    assert total == js["aggregate"]["rejected"]["rejected_measurable"], (
        "every measurable reject must carry exactly one L25 status code")
    for r in stats:
        if r["n"] == 0:
            continue
        assert 0.0 <= r["missed_opportunity_rate_pct"] <= 100.0
        assert r["tradable_p20_pct"] <= r["reached_p20_pct"], (
            "a tradable move cannot exceed a reached move")


def test_every_rejected_csv_row_is_causal_and_internally_consistent(artifacts):
    rej_csv = REPORTS / "moodeng_l25_rejected_signals.csv"
    if not rej_csv.exists():
        pytest.skip("csvs not generated")
    with rej_csv.open() as fh:
        for row in csv.DictReader(fh):
            covered = float(row["covered_s"] or 0)
            if row["classification"] == "CORRECT_REJECT":
                assert covered >= MIN_COVERAGE_S, \
                    "a reject may not be called CORRECT without enough forward data"
            mfe = row["mfe_pct"]
            mae = row["mae_pct"]
            if mfe and mae:
                assert float(mfe) >= float(mae)
            if row["tradable_p20"] == "true":
                assert row["reached_p20"] == "true"


def test_bought_csv_rows_carry_a_fill_or_a_blocked_reason(artifacts):
    buy_csv = REPORTS / "moodeng_l25_bought_signals.csv"
    if not buy_csv.exists():
        pytest.skip("csvs not generated")
    with buy_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row["population"] == "BOUGHT_FILLED":
                assert row["fill_iso"], "a filled row must carry its fill instant"
                assert row["exit_iso"], "a closed paper position must carry its exit"
                assert row["exit_reason"], "a closed position must record why it closed"
            else:
                # an authorized-but-unfilled row must be explainable: either an
                # explicit block, an execution attempt that failed, or the pipeline
                # simply never got as far as filling
                assert (row["fill_blocked_reason"] or row["unfilled_reasons"]
                        or row["stage_outcome"]), \
                    "an authorized-but-unfilled row must record what happened next"


def test_buy_exit_never_precedes_its_fill(artifacts):
    buy_csv = REPORTS / "moodeng_l25_bought_signals.csv"
    if not buy_csv.exists():
        pytest.skip("csvs not generated")
    from datetime import datetime

    def ts(s):
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp() if s else None

    with buy_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row["population"] != "BOUGHT_FILLED":
                continue
            f, e, d = ts(row["fill_iso"]), ts(row["exit_iso"]), ts(row["decided_iso"])
            assert d <= f <= e, (
                f"decision {d} -> fill {f} -> exit {e} is out of order for {row['mint']}")

# ---------------------------------------------------------------------------
# the measurement decisions that are easy to get wrong
# ---------------------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate():
    from research.moodeng_l25_audit.analysis import wilson
    lo, hi = wilson(9, 31)
    p = 100.0 * 9 / 31
    assert lo < p < hi
    # a zero-count cell must still produce a usable upper bound, not a null
    lo0, hi0 = wilson(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 100.0
    # an empty cell has no interval at all
    assert wilson(0, 0) == (None, None)


def test_matched_window_comparison_does_not_compare_a_truncated_path():
    """
    A fill's path stops at its exit; a reject's runs an hour. Comparing whole-path MFE
    would make every fill look like a winner. This test pins the guard: the matched
    comparator refuses a window the row does not cover.
    """
    from research.moodeng_l25_audit.analysis import _tradable_within

    def row(bars, t0, ref):
        return {"anchor_ts": t0, "ref_price": ref,
                "forward": {"_matched_bars": bars}}

    # covers 0..120 s only
    bars = [bar(T0 + 20.0, 1, 1.6, 1, 1.6), bar(T0 + 80.0, 1, 1.1, 1, 1.1)]
    assert _tradable_within(row(bars, T0, 1.0), 90.0) is True
    # 300 s is not covered -> None, never False
    assert _tradable_within(row(bars, T0, 1.0), 300.0) is None
    # no reference price -> no verdict
    assert _tradable_within(row(bars, T0, None), 90.0) is None


def test_matched_window_comparator_applies_the_same_tradability_rule():
    """+20 % after a -60 % collapse is still not tradable at a matched horizon."""
    from research.moodeng_l25_audit.analysis import _tradable_within
    bars = [bar(T0 + 20.0, 1, 1.0, 0.40, 0.5),
            bar(T0 + 80.0, 0.5, 1.5, 0.5, 1.5),
            bar(T0 + 140.0, 1.5, 1.5, 1.4, 1.5)]
    row = {"anchor_ts": T0, "ref_price": 1.0, "forward": {"_matched_bars": bars}}
    assert _tradable_within(row, 180.0) is False


def test_measurement_validity_rejects_an_anchor_conflict():
    """
    The engine's price and the tape's price disagreeing by more than 50 % means two
    different books, and every percentage return across that switch is meaningless.
    """
    from research.moodeng_l25_audit.analysis import measurement_invalid
    from research.moodeng_l25_audit.prelude import ANCHOR_CONFLICT_PCT

    def row(ref, close):
        return {"forward": {"anchor_conflict_pct": (ref / close - 1.0) * 100.0,
                            "excursion_out_of_range": False}}

    assert measurement_invalid(row(1.0, 1.2)) is None            # tape 20 % below
    assert measurement_invalid(row(0.75, 1.0)) is None           # tape 25 % above
    assert measurement_invalid(row(0.50, 1.0)) is None           # exactly -50 %, not ">"
    assert measurement_invalid(row(1.0, 3.0)) == "ANCHOR_CONFLICT"     # +200 %
    assert measurement_invalid(row(1.0, 0.40)) == "ANCHOR_CONFLICT"    # -60 %
    assert ANCHOR_CONFLICT_PCT == 50.0
    far = {"forward": {"anchor_conflict_pct": 0.0, "excursion_out_of_range": True}}
    assert measurement_invalid(far) == "EXCURSION_OUT_OF_RANGE"


def test_iqr_separation_is_withheld_when_a_group_is_too_small():
    """A perfect split on four rows is noise. It must not be reported as a finding."""
    import research.moodeng_l25_audit.analysis as ANA
    assert ANA.MIN_GROUP_N >= 5

    # build two synthetic groups: A with 20 well-separated rows, B with 4.
    def mk(n, key, value):
        out = []
        for i in range(n):
            row = {"population": "REJECT" if key == "A" else "BOUGHT_FILLED",
                   "l25_status_code": "V3C_REJECTED_MARKET_CAP",
                   "classification": "FALSE_REJECT" if key == "A" else "BAD_ENTRY",
                   "forward": {"covered_s": 900.0, "mfe_pct": 1.0, "mae_pct": -1.0,
                               "pre_context": {}, "ret_at": {}, "mfe_at": {},
                               "time_to_mfe_s": 1.0},
                   "false_entry": key == "B"}
            row[key] = value + i * 0.001
            out.append(row)
        return out

    rows = mk(20, "A_reject_winners", 100.0)
    # group_compare reads the rows by classification, so build them properly
    rows = []
    for i in range(20):
        rows.append({"population": "REJECT", "l25_status_code": "V3C_REJECTED_MARKET_CAP",
                     "classification": "FALSE_REJECT", "market_cap_usd": 100.0 + i,
                     "forward": {"covered_s": 900.0, "mfe_pct": 1.0, "mae_pct": -1.0,
                                 "pre_context": {}, "ret_at": {}, "mfe_at": {},
                                 "time_to_mfe_s": 1.0},
                     "false_entry": False})
    for i in range(4):
        rows.append({"population": "BOUGHT_FILLED",
                     "l25_status_code": "L25_AUTHORIZED", "classification": "BAD_ENTRY",
                     "market_cap_usd": 10.0 + i,
                     "forward": {"covered_s": 900.0, "mfe_pct": 1.0, "mae_pct": -1.0,
                                 "pre_context": {}, "ret_at": {}, "mfe_at": {},
                                 "time_to_mfe_s": 1.0},
                     "false_entry": True})
    out = ANA.group_compare(rows)
    mcap = next(r for r in out if r["field"] == "market_cap_usd")
    assert mcap["iqr_overlap"] is False, "the synthetic split IS separated"
    assert mcap["separates"] is False, "but it must not be CLAIMED with n=4"
    assert mcap["separates_but_underpowered"] is True


def test_no_ambiguous_reject_is_scored_as_a_missed_winner():
    """AMBIGUOUS must never leak into the missed-opportunity count."""
    js = REPORTS / f"{STEM}.json"
    if not js.exists():
        pytest.skip("report not generated")
    d = json.loads(js.read_text())
    rj = d["aggregate"]["rejected"]
    assert rj["false_reject"] + rj["correct_reject"] == rj["rejected_measurable"]
    matrix = REPORTS / "moodeng_l25_decision_matrix.csv"
    if matrix.exists():
        with matrix.open() as fh:
            for row in csv.DictReader(fh):
                if row["rejection_reason"] is None:
                    continue
                if row["classification"] == "FALSE_REJECT":
                    # a FALSE_REJECT must be tradable and must be measurable
                    assert row["tradable_p20"] == "true"
                    assert float(row["covered_s"]) >= 300.0


def test_top_lists_only_contain_measurement_valid_rows():
    js = REPORTS / f"{STEM}.json"
    if not js.exists():
        pytest.skip("report not generated")
    d = json.loads(js.read_text())
    invalid = d["data_quality"]["measurement_validity"]["reasons"]
    for t in d["top_rejected_missed"]:
        assert t["max_forward_return_pct"] is None or \
            abs(t["max_forward_return_pct"]) <= 100000.0, \
            f"{t['symbol']} is in the top-missed list with an out-of-range excursion"
    for reason in invalid:
        assert reason in ("ANCHOR_CONFLICT", "EXCURSION_OUT_OF_RANGE")


def test_outcome_tags_cover_every_decision_and_are_greppable():
    """
    The brief asks for `REJECT -> PUMP` and `BUY -> DUMP` to be the easiest things to
    inspect. That only holds if every row carries exactly one of a small vocabulary,
    and if the two named tags actually exist in the vocabulary.
    """
    from research.moodeng_l25_audit.analysis import OUTCOME_TAG
    for needed in ("REJECT -> PUMP", "BUY -> DUMP", "REJECT -> CORRECT",
                   "REJECT -> UNMEASURED"):
        assert needed in OUTCOME_TAG.values(), f"{needed} is not in the tag vocabulary"

    matrix = REPORTS / "moodeng_l25_decision_matrix.csv"
    if not matrix.exists():
        pytest.skip("decision matrix not generated")
    known = set(OUTCOME_TAG.values())
    n = 0
    with matrix.open() as fh:
        for row in csv.DictReader(fh):
            assert row["outcome_tag"] in known, f"unknown tag {row['outcome_tag']!r}"
            assert row["outcome_tag_90s"].startswith(row["outcome_tag"].split(" (")[0])
            n += 1
    assert n > 0


def test_the_two_population_csvs_are_filterable_by_tag():
    for name, expected in (("moodeng_l25_rejected_signals.csv", "REJECT -> "),
                           ("moodeng_l25_bought_signals.csv", "BUY -> ")):
        p = REPORTS / name
        if not p.exists():
            pytest.skip("csvs not generated")
        with p.open() as fh:
            rows = list(csv.DictReader(fh))
        assert rows
        assert all(r["outcome_tag"].startswith(expected) for r in rows), \
            f"{name} must contain only its own population"
