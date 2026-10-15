"""
The statistics epoch: a reset moves the reporting window, not the history.

The point of this feature is that an operator who resets to $100 sees a clean
account, while the append-only ledger - which is ALSO the frozen research
universe the L25 replay validates against - stays complete.

So these tests pin both halves. A change that "fixes" the display by truncating
the ledger must fail here.
"""


from __future__ import annotations

import json
from pathlib import Path

import pytest

from autopaper.ledger import PaperAccountState

pytestmark = pytest.mark.integrity

REPO = Path(__file__).resolve().parents[1]
RESET = REPO / "scripts/reset_paper_account.py"


# ---------------------------------------------------------------------------
# the field itself
# ---------------------------------------------------------------------------


def test_the_epoch_defaults_to_counting_everything():
    """A never-reset account must behave exactly as it did before."""
    assert PaperAccountState().stats_epoch_started_at is None


def test_the_epoch_round_trips_through_json():
    assert PaperAccountState.from_dict({}).stats_epoch_started_at is None
    got = PaperAccountState.from_dict({"stats_epoch_started_at": 1234.5})
    assert got.stats_epoch_started_at == 1234.5
    state = PaperAccountState(stats_epoch_started_at=99.0)
    assert PaperAccountState.from_dict(state.as_dict()).stats_epoch_started_at \
        == 99.0


def test_a_null_epoch_stays_null_rather_than_becoming_zero():
    """
    A missing key must not become 0.0, which would silently scope the window to
    "everything since 1970" and read as if a reset had happened.
    """
    assert PaperAccountState.from_dict(
        {"stats_epoch_started_at": None}).stats_epoch_started_at is None


# ---------------------------------------------------------------------------
# the reporting window
# ---------------------------------------------------------------------------


def _trade(exit_at: float, pnl: float = 1.0):
    """
    A minimal stand-in, not a real ``Trade``.

    ``compute`` reads only these attributes, and building a real Trade needs 23
    required fields that are irrelevant to a statistics-window test. Binding to
    the attributes actually consumed keeps the test honest about what it needs.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        exit_timestamp=exit_at, net_pnl_usd=pnl, gross_pnl_usd=pnl,
        return_pct=1.0, holding_seconds=1.0, mfe_pct=1.0, mae_pct=-1.0,
        fees_paid_usd=0.0, entry_fee_usd=0.0, exit_fee_usd=0.0,
        entry_slippage_usd=0.0, exit_slippage_usd=0.0,
        # the remaining attributes `compute` groups by
        route="FAST_ENTRY", regime="EARLY", exit_style="TARGET",
        age_bucket="b", liq_bucket="b", mc_bucket="b",
    )


def test_no_epoch_counts_every_trade():
    """The un-reset behaviour: the whole ledger."""
    from autopaper.stats import compute

    trades = [_trade(100.0), _trade(200.0), _trade(300.0)]
    assert compute(trades, starting_equity=100.0).total_trades == 3


def test_an_epoch_excludes_earlier_trades():
    from autopaper.stats import compute

    trades = [_trade(100.0), _trade(200.0), _trade(300.0)]
    epoch = 200.0
    scoped = [t for t in trades if (t.exit_timestamp or 0.0) >= epoch]
    assert len(scoped) == 2
    assert compute(scoped, starting_equity=100.0).total_trades == 2


def test_the_boundary_trade_is_included():
    """A trade closing exactly at the epoch belongs to the new account."""
    from autopaper.stats import compute

    trades = [_trade(199.999), _trade(200.0)]
    scoped = [t for t in trades if (t.exit_timestamp or 0.0) >= 200.0]
    assert len(scoped) == 1
    assert compute(scoped, starting_equity=100.0).total_trades == 1


def test_a_fresh_epoch_yields_a_clean_account():
    from autopaper.stats import compute

    trades = [_trade(100.0, 5.0), _trade(200.0, -2.0), _trade(300.0, 9.0)]
    epoch = 1_000.0            # later than every trade
    scoped = [t for t in trades if (t.exit_timestamp or 0.0) >= epoch]
    s = compute(scoped, starting_equity=100.0, current_equity=100.0)
    assert s.total_trades == 0
    assert s.wins == 0 and s.losses == 0
    assert s.realized_pnl_usd == 0.0
    assert s.max_drawdown_pct == 0.0
    assert s.profit_factor == 0.0


# ---------------------------------------------------------------------------
# the reset must NOT truncate the ledger
# ---------------------------------------------------------------------------


def test_the_reset_script_never_truncates_the_ledger():
    """
    The regression that motivated this whole design.

    Asserted on the source: the append-only files must not be written, emptied
    or unlinked anywhere in the script.
    """
    src = RESET.read_text()
    for name in ("ledger.jsonl", "decisions.jsonl", "candidates.jsonl"):
        for write in (f'write_text("")', "unlink()", "open(\"w\")",
                      "'w', encoding", 'truncate('):
            assert write not in src or name not in src.split(write)[0][-200:], (
                f"the reset script appears to write {name}"
            )
    # and it must not import anything that empties a file
    assert "shutil.rmtree" not in src
    assert "os.remove" not in src


def test_the_reset_script_says_the_ledger_is_untouched():
    src = RESET.read_text()
    assert "UNTOUCHED" in src
    assert "append-only" in src
    # it must name the three tests that a truncation would break, so the next
    # reader knows why the constraint exists
    for t in ("test_replay_reproduces_the_frozen_universe",
              "test_snapshot_is_a_prefix_of_the_live_ledger",
              "test_historical_records_are_preserved"):
        assert t in src, t


def test_the_reset_script_does_not_touch_the_forward_validation_store():
    src = RESET.read_text()
    assert "forward_validation" in src          # named, to be left alone
    # no write or delete targeting the deployment boundary
    for bad in ("deployment.json", "forward_validation/deployment"):
        for line in src.splitlines():
            if bad in line:
                low = line.lower()
                assert not any(w in low for w in
                               ("write", "unlink", "rmtree", "remove")), line


def test_the_reset_script_does_not_clear_the_emergency_latch():
    src = RESET.read_text()
    for line in src.splitlines():
        if "emergency_stop" in line:
            low = line.lower()
            assert not any(w in low for w in ("unlink", "rmtree", "remove",
                                              "write_text")), line


def test_the_reset_script_sets_a_new_epoch():
    src = RESET.read_text()
    assert "stats_epoch_started_at=now" in src
    assert "cash_usd=100.0" in src
    assert "peak_equity_usd=100.0" in src
    assert '"open_positions": []' in src


def test_the_reset_script_archives_before_writing():
    """Reversibility: the old state must be copied out before it is replaced."""
    src = RESET.read_text()
    assert "shutil.copy2(state, archived)" in src
    assert src.index("shutil.copy2(state, archived)") < src.index(
        "save_json_atomic(state, payload)"), (
        "the archive copy must happen BEFORE the state is overwritten"
    )


def test_the_reset_script_refuses_while_an_engine_runs():
    src = RESET.read_text()
    assert "live_locks" in src
    assert "REFUSED" in src
    i_copy = src.index("shutil.copy2(state, archived)")
    assert src.index("if locks:") < i_copy, (
        "the liveness check must precede any write"
    )


# ---------------------------------------------------------------------------
# the real ledger on disk
# ---------------------------------------------------------------------------


def test_the_live_ledger_still_holds_its_history():
    """Belt and braces: the research universe is still there."""
    ledger = REPO / "var/ledger.jsonl"
    if not ledger.exists():
        pytest.skip("no live ledger in this checkout")
    n = 0
    for line in ledger.read_text().splitlines():
        if line.strip() and json.loads(line).get("event") == "exit":
            n += 1
    assert n > 0, "the historical ledger must not be empty"


# ---------------------------------------------------------------------------
# the engine actually applies the window
# ---------------------------------------------------------------------------


def _exit_record(template: dict, exit_at: float, pnl: float) -> dict:
    """A ledger exit record built from a REAL archived record's shape.

    Reusing a real record's field names means this test cannot pass by inventing
    a shape the production reader does not understand.
    """
    trade = dict(template["trade"])
    trade["exit_timestamp"] = exit_at
    trade["booked_at"] = exit_at
    trade["entry_timestamp"] = exit_at - 30.0
    trade["net_pnl_usd"] = pnl
    trade["gross_pnl_usd"] = pnl
    trade["return_pct"] = pnl / 0.155 * 100.0
    trade["holding_seconds"] = 30.0
    return {"event": "exit", "at": exit_at, "trade": trade}


def _real_template() -> dict:
    archive = sorted((REPO / "var/archive").glob("ledger-pre-reset-*.jsonl"))
    if not archive:
        pytest.skip("no archived ledger in this checkout")
    for line in archive[0].read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "exit" and isinstance(rec.get("trade"), dict):
            return rec
    pytest.skip("no exit record in the archive")


def _engine_with_ledger(tmp_path: Path, records: list[dict]):
    """A real engine over a temp root whose ledger holds `records`."""
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    var = tmp_path / "var"
    var.mkdir(parents=True, exist_ok=True)
    (var / "ledger.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    trader = PaperTrader(build_config(tmp_path))
    return trader


def test_the_engine_counts_the_whole_ledger_without_an_epoch(tmp_path):
    """No epoch set: unchanged behaviour, every trade counted."""
    template = _real_template()
    recs = [_exit_record(template, 100.0, 1.0),
            _exit_record(template, 200.0, 2.0),
            _exit_record(template, 300.0, -1.0)]
    t = _engine_with_ledger(tmp_path, recs)
    try:
        paper, _real = t.statistics_by_mode()
        assert paper["total_trades"] == 3
    finally:
        t._release_ownership()


def test_the_engine_excludes_trades_before_the_epoch(tmp_path):
    """
    THE test the mutation harness needed.

    An epoch after every recorded trade must report a clean account, even though
    the ledger still holds the trades. Without this, "the engine ignores the
    epoch" is invisible.
    """
    template = _real_template()
    recs = [_exit_record(template, 100.0, 5.0),
            _exit_record(template, 200.0, 5.0),
            _exit_record(template, 300.0, 5.0)]
    t = _engine_with_ledger(tmp_path, recs)
    try:
        # set an epoch AFTER all three trades
        state = t.account.state
        object.__setattr__(state, "stats_epoch_started_at", 1_000.0)
        paper, _real = t.statistics_by_mode()
        assert paper["total_trades"] == 0
        assert paper["wins"] == 0
        assert paper["realized_pnl_usd"] == pytest.approx(0.0)
        # the ledger is untouched: the history is still there
        lines = (tmp_path / "var/ledger.jsonl").read_text().splitlines()
        assert len([l for l in lines if l.strip()]) == 3
    finally:
        t._release_ownership()


def test_the_engine_counts_trades_after_the_epoch(tmp_path):
    template = _real_template()
    recs = [_exit_record(template, 100.0, 5.0),   # before
            _exit_record(template, 500.0, -2.0),  # after
            _exit_record(template, 600.0, 3.0)]   # after
    t = _engine_with_ledger(tmp_path, recs)
    try:
        state = t.account.state
        object.__setattr__(state, "stats_epoch_started_at", 200.0)
        paper, _real = t.statistics_by_mode()
        assert paper["total_trades"] == 2
        assert paper["realized_pnl_usd"] == pytest.approx(1.0)
    finally:
        t._release_ownership()


def test_the_epoch_survives_a_save_and_reload(tmp_path):
    """
    The epoch must survive a persist cycle, or a restart would silently widen
    the window back to the whole ledger.
    """
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    t = PaperTrader(build_config(tmp_path))
    try:
        # persist() refuses without ownership, which is the correct guard, so
        # this takes the lock the way the runtime does
        t.startup()
        object.__setattr__(t.account.state, "stats_epoch_started_at", 4242.0)
        t.persist()
    finally:
        t._release_ownership()

    # A RESTART: _load_state() runs in startup(), so the reload must go through
    # it exactly as a real restart would.
    again = PaperTrader(build_config(tmp_path))
    try:
        again.startup()
        assert again.account.state.stats_epoch_started_at == 4242.0
    finally:
        again._release_ownership()
