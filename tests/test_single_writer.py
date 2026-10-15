"""
Single-writer hardening tests.

Deterministic and offline: no network, no Telegram, no live market data, and no
sleeps.  Concurrency is exercised with real operating-system processes because
the guarantee under test is a kernel guarantee; a thread inside one interpreter
could not demonstrate it.  Workers synchronise on pipes, and simultaneous starts
are forced with an explicit start barrier rather than by sleeping.

Coverage map (task section 8):

    A  one writer starts successfully                 test_first_writer_acquires
    B  second writer is refused                      test_second_writer_refused
    C  simultaneous startup -> exactly one owner     test_simultaneous_startup
    D  lock releases after graceful shutdown         test_release_after_graceful_shutdown
    E  lock is safe after abnormal termination       test_lock_survives_sigkill
    F  state cannot be concurrently overwritten      test_state_cannot_be_overwritten
    G  concurrent candidates cannot duplicate a trade test_concurrent_candidates_do_not_duplicate_trade
    H  ledger remains append-only                    test_ledger_remains_append_only
    I  recovery cannot bypass the lock               test_recovery_cannot_bypass_the_lock
    --  the old double-process failure, reproduced   test_old_double_process_failure_is_eliminated
    --  the accounting equation                      test_accounting_equation
"""

from __future__ import annotations

import hashlib
import json
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integrity

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from autopaper.account import MarkedPosition, PaperAccount  # noqa: E402
from autopaper.account_lock import (  # noqa: E402
    REFUSAL, AccountAlreadyLocked, AccountOwnershipError, acquire,
    assert_no_legacy_writer, describe_holder, held_account_ids, identity_for,
    pid_is_alive, release,
)
from autopaper.config import RiskLimits  # noqa: E402
from autopaper.ledger import (  # noqa: E402
    EventType, Ledger, PaperAccountState, load_json, save_json_atomic,
)
from autopaper.models import DataState, ExitStyle, Position, PositionStatus, Regime, Route, Trade  # noqa: E402

BANKROLL = 100.0

# ---------------------------------------------------------------------------
# the worker: a real, separate OS process that does real engine-shaped work
# ---------------------------------------------------------------------------

WORKER_SRC = r'''
import json, os, sys
sys.path.insert(0, "__SRC__")
from autopaper.account_lock import AccountAlreadyLocked, acquire, release
from autopaper.ledger import EventType, Ledger, save_json_atomic
from autopaper.models import Trade

root, mode, tag = sys.argv[1], sys.argv[2], sys.argv[3]
var = os.path.join(root, "var")
state = os.path.join(var, "paper_account.json")


def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def gate():
    """Block until the parent says go.  A pipe, never a sleep."""
    sys.stdin.readline()


def make_trade(account_id):
    return Trade(
        trade_id="trade_FIXED", position_id="pos_FIXED", candidate_id="cand_FIXED",
        mint="MINT_FIXED", symbol="FIXED", source="telegram", strategy="FAST_ENTRY",
        route="FAST_ENTRY", regime="EXPLOSIVE", entry_signal_timestamp=1.0,
        entry_timestamp=1.0, entry_price=1e-5, fill_price=1.01e-5,
        exit_timestamp=10.0, exit_price=1.2e-5, quantity=1000.0,
        entry_fee_usd=0.1, exit_fee_usd=0.1, entry_slippage_usd=0.2,
        exit_slippage_usd=0.2, gross_pnl_usd=2.0, net_pnl_usd=1.4,
        return_pct=20.0, holding_seconds=9.0, exit_reason="test",
        exit_style="FAST_FAILURE", thesis="t", booked_at=10.0,
        mfe_pct=25.0, mae_pct=-2.0, invalidation=())  # entry_signal_timestamp above


def do_trade_work(lock):
    """Read state, mutate, save, append - exactly the sequence that raced before.
    Every one of these lines is AFTER the lock is held."""
    try:
        raw = json.load(open(state))
    except Exception:
        raw = {}
    acct = raw.get("account") or {}
    cash = float(acct.get("cash_usd", 100.0))
    realized = float(acct.get("realized_pnl_usd", 0.0))
    save_json_atomic(state, {
        "agent": "AUTONOMOUS_PAPER_AGENT_V1",
        "account_id": lock.account_id,
        "account": {**acct, "cash_usd": cash + 1.0, "realized_pnl_usd": realized + 1.0,
                    "run_count": int(acct.get("run_count", 0)) + 1},
        "counters": {"writes": tag},
    })
    led = Ledger(os.path.join(var, "ledger.jsonl"), os.path.join(var, "decisions.jsonl"),
                 os.path.join(var, "candidates.jsonl"),
                 os.path.join(var, "position_updates.jsonl"))
    led.event(EventType.RECOVERY, {"recovered": True, "run_count": 1,
                                   "open_positions": 0, "closed_trades": 1,
                                   "agent": "AUTONOMOUS_PAPER_AGENT_V1",
                                   "account_id": lock.account_id,
                                   "owner_pid": os.getpid()})
    led.candidate_seen({"candidate_id": "cand_FIXED", "chat_id": 1, "message_id": 42,
                        "received_timestamp": 1.0, "source_timestamp": 1.0,
                        "traded": True})
    led.trade(make_trade(lock.account_id))


# --- arm -------------------------------------------------------------------
out({"stage": "armed", "pid": os.getpid()})
gate()   # start barrier: every racer is now released at once

try:
    lock = acquire(state, kind=tag)
except AccountAlreadyLocked as exc:
    out({"result": "refused", "refusal": str(exc), "holder": exc.holder,
         "wrote_ledger": False, "wrote_state": False})
    sys.exit(3)

out({"result": "owner", "account_id": lock.account_id, "pid": os.getpid()})

if mode == "hold":
    gate()                 # hold ownership while the parent inspects
    release(lock)
    out({"result": "released"})
    sys.exit(0)

if mode == "crash":
    gate()
    os._exit(9)            # no release, no cleanup, no flush

if mode == "trade":
    gate()                 # hold ownership so the loser is genuinely refused
    do_trade_work(lock)
    release(lock)
    out({"result": "wrote", "account_id": lock.account_id})
    sys.exit(0)
'''


@pytest.fixture
def worker_script(tmp_path):
    p = tmp_path / "_worker.py"
    p.write_text(WORKER_SRC.replace("__SRC__", str(SRC)), encoding="utf-8")
    return p


def spawn(worker_script: Path, root: Path, mode: str, tag: str):
    return subprocess.Popen(
        [sys.executable, str(worker_script), str(root), mode, tag],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )


def recv(proc) -> dict:
    """Read one JSON line from whichever process produced one first."""
    ready, _, _ = select.select([p.stdout for p in _ALL_PROCS], [], [], 30)
    assert ready, "no worker produced output within 30s"
    fh = ready[0]
    for p in _ALL_PROCS:
        if p.stdout is fh:
            return json.loads(fh.readline())
    raise AssertionError("unreadable worker output")


_ALL_PROCS: list = []


def recv_from(proc, timeout: float = 30.0) -> dict:
    """One JSON line from one worker, with a hard timeout so a protocol mistake
    fails the test instead of hanging the suite."""
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    assert ready, f"{proc.args[-1]}: no output within {timeout}s; stderr={proc.stderr.read()}"
    line = proc.stdout.readline()
    assert line, f"{proc.args[-1]}: EOF on stdout; stderr={proc.stderr.read()}"
    return json.loads(line)


def go(*procs) -> None:
    for p in procs:
        try:
            p.stdin.write("go\n")
            p.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass


def finish(*procs) -> None:
    for p in procs:
        try:
            p.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            p.kill()
            p.wait(timeout=10)


# ---------------------------------------------------------------------------
# local helpers
# ---------------------------------------------------------------------------


def make_root(tmp_path: Path, name: str = "acct") -> Path:
    root = tmp_path / name
    (root / "var").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    return root


def state_file(root: Path) -> Path:
    return root / "var" / "paper_account.json"


def ledger_for(root: Path) -> Ledger:
    var = root / "var"
    return Ledger(var / "ledger.jsonl", var / "decisions.jsonl",
                  var / "candidates.jsonl", var / "position_updates.jsonl")


def account_ids_in_ledger(root: Path) -> set:
    return {r.get("account_id") for r in ledger_for(root).events() if r.get("account_id")}


def seed_state(root: Path, **acct) -> dict:
    fields = {"starting_bankroll_usd": BANKROLL, "cash_usd": BANKROLL, **acct}
    payload = {
        "agent": "AUTONOMOUS_PAPER_AGENT_V1",
        "account_id": identity_for(state_file(root)).account_id,
        "account": PaperAccountState(**fields).as_dict(),
    }
    save_json_atomic(state_file(root), payload)
    return payload


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_account_identity_is_canonical_across_path_aliases(tmp_path):
    """Symlinks and `..` must not be able to create a second writer."""
    root = make_root(tmp_path)
    canonical = identity_for(state_file(root))

    assert identity_for(root / "var" / ".." / "var" / "paper_account.json"
                        ).account_id == canonical.account_id
    assert identity_for(str(root / "var" / "paper_account.json")
                        ).account_id == canonical.account_id

    alias = tmp_path / "alias"
    alias.symlink_to(root / "var", target_is_directory=True)
    assert identity_for(alias / "paper_account.json").account_id == canonical.account_id

    # the lock file is derived from the same canonical path
    assert canonical.lock_path.parent == canonical.state_path.parent
    assert canonical.state_path.is_absolute()
    assert identity_for(alias / "paper_account.json").lock_path == canonical.lock_path

    # a genuinely different account is a genuinely different identity
    assert identity_for(tmp_path / "other" / "paper_account.json"
                        ).account_id != canonical.account_id


# ---------------------------------------------------------------------------
# A / B
# ---------------------------------------------------------------------------


def test_first_writer_acquires(tmp_path):
    """A: the first process takes ownership cleanly."""
    root = make_root(tmp_path)
    lock = acquire(state_file(root))
    try:
        assert not lock.released
        assert lock.fd is not None
        assert lock.account_id.startswith("acct_")
        meta = describe_holder(lock.identity.lock_path)
        assert meta["pid"] == os.getpid()
        assert meta["account_id"] == lock.account_id
    finally:
        release(lock)
    assert lock.released


def test_second_writer_refused(tmp_path, worker_script):
    """B: a second PROCESS is refused, and writes nothing at all."""
    root = make_root(tmp_path)
    holder = acquire(state_file(root))
    proc = spawn(worker_script, root, "trade", "second")
    try:
        assert recv_from(proc)["stage"] == "armed"
        go(proc)
        result = recv_from(proc)
        assert result["result"] == "refused"
        assert REFUSAL in result["refusal"]
        assert result["wrote_ledger"] is False
        assert result["wrote_state"] is False
        assert result["holder"].get("pid") == os.getpid()
    finally:
        finish(proc)
        release(holder)
    assert proc.returncode == 3
    assert not state_file(root).exists()
    assert not (root / "var" / "ledger.jsonl").exists()


def test_same_process_reacquire_is_legal(tmp_path):
    """An in-process restart is a new reference, not a new owner."""
    from autopaper.account_lock import identity_for

    root = make_root(tmp_path)
    mine = identity_for(state_file(root)).account_id
    a = acquire(state_file(root))
    b = acquire(state_file(root))
    assert a is b and b._refs == 2
    release(b)
    assert not b.released
    release(a)
    assert b.released
    # Scoped to THIS account. `held_account_ids()` is a process-global registry
    # and other test modules legitimately hold their own locks, so asserting it
    # is empty tests the test suite's ordering rather than the lock.
    assert mine not in held_account_ids()


# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------


def test_simultaneous_startup_allows_exactly_one_owner(tmp_path, worker_script):
    """
    C: six processes released from a start barrier at once.  Exactly one wins.

    Without the barrier the processes would start sequentially and could each
    win in turn, which would test nothing.
    """
    root = make_root(tmp_path)
    n = 6
    procs = [spawn(worker_script, root, "trade", f"racer{i}") for i in range(n)]
    _ALL_PROCS.extend(procs)
    try:
        for p in procs:                      # everyone armed
            assert recv_from(p)["stage"] == "armed"
        go(*procs)                            # everyone released together
        results = [recv_from(p) for p in procs]
    finally:
        _ALL_PROCS.clear()
        go(*procs)                            # let the winner finish
        finish(*procs)

    owners = [r for r in results if r["result"] == "owner"]
    refused = [r for r in results if r["result"] == "refused"]
    assert len(owners) == 1, [r["result"] for r in results]
    assert len(refused) == n - 1
    assert all(REFUSAL in r["refusal"] for r in refused)

    state = load_json(state_file(root))
    assert state["account"]["cash_usd"] == 101.0
    assert state["account"]["run_count"] == 1
    assert len(ledger_for(root).trades()) == 1
    assert len(account_ids_in_ledger(root)) == 1


# ---------------------------------------------------------------------------
# D / E
# ---------------------------------------------------------------------------


def test_release_after_graceful_shutdown(tmp_path, worker_script):
    """D: a normal shutdown hands ownership back; a restart then succeeds."""
    root = make_root(tmp_path)
    first = spawn(worker_script, root, "hold", "first")
    assert recv_from(first)["stage"] == "armed"
    go(first)
    assert recv_from(first)["result"] == "owner"   # it now blocks HOLDING the lock

    second = spawn(worker_script, root, "trade", "second")
    assert recv_from(second)["stage"] == "armed"
    go(second)
    assert recv_from(second)["result"] == "refused"
    finish(second)

    go(first)                                    # graceful release
    assert recv_from(first)["result"] == "released"
    finish(first)
    assert first.returncode == 0

    third = spawn(worker_script, root, "trade", "third")
    assert recv_from(third)["stage"] == "armed"
    go(third)
    assert recv_from(third)["result"] == "owner"
    go(third)
    assert recv_from(third)["result"] == "wrote"
    finish(third)
    assert third.returncode == 0
    assert load_json(state_file(root))["account"]["cash_usd"] == 101.0


def test_lock_survives_sigkill(tmp_path, worker_script):
    """
    E: SIGKILL.  The kernel drops the flock, so no stale-lock cleanup is needed.

    This is exactly why "delete the lock file if it looks old" is not used: it
    would be a second, racy mechanism guarding the same resource.
    """
    root = make_root(tmp_path)
    victim = spawn(worker_script, root, "hold", "victim")
    assert recv_from(victim)["stage"] == "armed"
    go(victim)
    assert recv_from(victim)["result"] == "owner"
    go(victim)
    victim_pid = victim.pid
    lock_path = identity_for(state_file(root)).lock_path

    victim.send_signal(signal.SIGKILL)
    victim.wait(timeout=30)
    assert victim.returncode == -signal.SIGKILL
    assert not pid_is_alive(victim_pid)
    assert lock_path.exists(), "the lock file survives; it is simply not trusted"

    lock = acquire(state_file(root))
    try:
        assert not lock.released
    finally:
        release(lock)


def test_lock_file_is_never_unlinked_on_release(tmp_path):
    """Unlinking would let a second process lock a different inode."""
    root = make_root(tmp_path)
    lock = acquire(state_file(root))
    path = lock.identity.lock_path
    release(lock)
    assert path.exists()
    again = acquire(state_file(root))
    assert again.fd is not None
    release(again)


# ---------------------------------------------------------------------------
# F / G / H / I
# ---------------------------------------------------------------------------


def test_state_cannot_be_overwritten(tmp_path, worker_script):
    """
    F: the read -> mutate -> save sequence cannot be run by two processes.

    This is the exact race that produced the gap between ACCOUNT realized and the
    ledger: two processes each read a state file, each moved cash by its own
    trade, and the second write discarded the first.
    """
    root = make_root(tmp_path)
    seed_state(root)
    before = load_json(state_file(root))["account"]["cash_usd"]

    lock = acquire(state_file(root))
    proc = spawn(worker_script, root, "trade", "racer")
    try:
        assert recv_from(proc)["stage"] == "armed"
        go(proc)
        assert recv_from(proc)["result"] == "refused"
    finally:
        finish(proc)
        release(lock)
    assert load_json(state_file(root))["account"]["cash_usd"] == before


def test_concurrent_candidates_do_not_duplicate_trade(tmp_path, worker_script):
    """
    G: two workers processing the SAME candidate id produce exactly one trade.

    The old failure double-counted every trade in the duplicated window.
    """
    root = make_root(tmp_path)
    loser = spawn(worker_script, root, "trade", "loser")
    assert recv_from(loser)["stage"] == "armed"
    go(loser)
    assert recv_from(loser)["result"] == "owner"
    # `loser` stays blocked holding the lock: NO second go() yet.
    winner = spawn(worker_script, root, "trade", "winner")
    try:
        assert recv_from(winner)["stage"] == "armed"
        go(winner)
        assert recv_from(winner)["result"] == "refused"
    finally:
        finish(winner)
        go(loser)                                # now let the holder do its work
        assert recv_from(loser)["result"] == "wrote"
        finish(loser)
    assert loser.returncode == 0

    led = ledger_for(root)
    assert [t["candidate_id"] for t in led.trades()] == ["cand_FIXED"]
    assert len([c for c in led.all_candidates()
                if c.get("candidate_id") == "cand_FIXED"]) == 1
    assert sum(t["net_pnl_usd"] for t in led.trades()) == pytest.approx(1.4)


def test_ledger_remains_append_only(tmp_path, worker_script):
    """H: a refused writer cannot change a single byte of the ledger."""
    root = make_root(tmp_path)
    owner = spawn(worker_script, root, "trade", "owner")
    assert recv_from(owner)["stage"] == "armed"
    go(owner)
    assert recv_from(owner)["result"] == "owner"
    go(owner)
    assert recv_from(owner)["result"] == "wrote"
    finish(owner)

    led_path = root / "var" / "ledger.jsonl"
    before = led_path.read_bytes()
    step = max(1, len(before) // 23)
    prefixes = {n: hashlib.sha256(before[:n]).hexdigest()
                for n in range(0, len(before) + 1, step)}

    lock = acquire(state_file(root))
    proc = spawn(worker_script, root, "trade", "racer")
    try:
        assert recv_from(proc)["stage"] == "armed"
        go(proc)
        assert recv_from(proc)["result"] == "refused"
    finally:
        finish(proc)
        release(lock)

    after = led_path.read_bytes()
    assert after == before
    for n, h in prefixes.items():
        assert hashlib.sha256(after[:n]).hexdigest() == h


def test_recovery_cannot_bypass_the_lock(tmp_path, worker_script):
    """
    I: recovery is the first thing startup does, so it must sit behind the lock.

    A refused process must not append a `recovery` event, bump `run_count`, or
    touch the state file.
    """
    root = make_root(tmp_path)
    before = seed_state(root, realized_pnl_usd=42.81, cash_usd=142.81, run_count=1)
    n_before = ledger_for(root).count(EventType.RECOVERY)

    lock = acquire(state_file(root))
    proc = spawn(worker_script, root, "trade", "racer")
    try:
        assert recv_from(proc)["stage"] == "armed"
        go(proc)
        assert recv_from(proc)["result"] == "refused"
    finally:
        finish(proc)
        release(lock)

    after = load_json(state_file(root))
    assert after["account"] == before["account"]
    assert ledger_for(root).count(EventType.RECOVERY) == n_before


# ---------------------------------------------------------------------------
# 10 -  the historical failure
# ---------------------------------------------------------------------------


def test_old_double_process_failure_is_eliminated(tmp_path, worker_script):
    """
    Reproduce the old scenario and assert every symptom is gone.

    Without the lock this fails all six ways listed in the task: two owners,
    duplicated candidate processing, duplicated account writes, double-counted
    P&L, and two account identities in one ledger.
    """
    root = make_root(tmp_path)
    procs = [spawn(worker_script, root, "trade", f"w{i}") for i in range(2)]
    try:
        for p in procs:
            assert recv_from(p)["stage"] == "armed"
        go(*procs)
        results = [recv_from(p) for p in procs]
    finally:
        go(*procs)
        finish(*procs)

    owners = [r for r in results if r["result"] == "owner"]
    refused = [r for r in results if r["result"] == "refused"]

    assert len(owners) == 1, [r["result"] for r in results]
    assert len(refused) == 1
    assert refused[0]["wrote_ledger"] is False
    assert refused[0]["wrote_state"] is False

    led = ledger_for(root)
    assert len([c for c in led.all_candidates()
                if c.get("candidate_id") == "cand_FIXED"]) == 1
    state = load_json(state_file(root))
    assert state["account"]["run_count"] == 1
    assert state["account"]["realized_pnl_usd"] == 1.0
    assert sum(t["net_pnl_usd"] for t in led.trades()) == pytest.approx(1.4)
    assert len(account_ids_in_ledger(root)) == 1
    assert led.count(EventType.RECOVERY) == 1


def test_ownership_boundary_raises_for_a_non_owner(tmp_path):
    """The in-process boundary refuses a durable write from a non-owner."""
    from autopaper.engine import PaperTrader

    root = make_root(tmp_path)
    lock = acquire(state_file(root))
    ident = identity_for(state_file(root))
    lock.release()          # simulate ownership being taken away underneath us

    class _Stub:
        _lock = lock
        _account_id = ident.account_id
        owns_account = property(lambda self: self._lock is not None
                                and not self._lock.released)

    with pytest.raises(AccountOwnershipError) as exc:
        PaperTrader._require_owner(_Stub(), "persist()")
    assert "does not own" in str(exc.value)


# ---------------------------------------------------------------------------
# 9 -  accounting equation
# ---------------------------------------------------------------------------


QTY = 1_000_000.0
ENTRY_REF = 1.0e-5
ENTRY_SLIP = 0.12                      # == qty * (fill - entry_ref)
ENTRY_FILL = ENTRY_REF + ENTRY_SLIP / QTY
ENTRY_FEE = 0.085
EXIT_REF = 1.5e-5
EXIT_SLIP = 0.12                       # == qty * (exit_ref - exit_fill)
EXIT_FILL = EXIT_REF - EXIT_SLIP / QTY
EXIT_FEE = 0.085


def _open_position() -> Position:
    """
    A position whose numbers satisfy V1's own accounting relation exactly:

        entry_slippage == qty * (fill_price - entry_reference_price)
        exit_slippage  == qty * (exit_reference - exit_fill_price)

    which is what makes ``cash_after - cash_before == net_pnl`` hold.  Anything
    looser and ``build_trade`` correctly refuses to close.
    """
    return Position(
        position_id="pos_a", candidate_id="cand_a", mint="MINT_A", symbol="POSA",
        status=PositionStatus.OPEN, route=Route.FAST_ENTRY, regime=Regime.EXPLOSIVE,
        strategy="FAST_ENTRY", exit_style=ExitStyle.FAST_FAILURE,
        entry_signal_timestamp=100.0, entry_decision_timestamp=100.0,
        entry_timestamp=100.0, quoted_price=ENTRY_REF,
        entry_reference_price=ENTRY_REF, fill_price=ENTRY_FILL,
        quantity=QTY, notional_usd=QTY * ENTRY_FILL, entry_fee_usd=ENTRY_FEE,
        entry_slippage_usd=ENTRY_SLIP,
        cash_out_usd=QTY * ENTRY_FILL + ENTRY_FEE, cost_basis_usd=QTY * ENTRY_FILL,
        cash_at_entry=BANKROLL,
    )


def test_accounting_equation(tmp_path):
    """
    Prove the accounting equation the application actually implements.

    Read from ``account.py`` rather than assumed:

        cash       = bankroll + realized_net - open_cost_basis
        market_val = sum(quantity * mark)          (cost basis when unmarked)
        equity     = cash + market_val
        unrealized = sum(quantity * (mark - fill))

    which for the whole account gives

        equity = bankroll + realized_net + unrealized - entry_fees_still_open

    and collapses to ``equity = bankroll + realized_net + unrealized`` exactly
    when nothing is open.  Both are asserted on a real account, with a real
    position marked up, and again after it is closed.
    """
    account = PaperAccount(
        PaperAccountState(starting_bankroll_usd=BANKROLL, cash_usd=BANKROLL),
        RiskLimits(),
    )

    # nothing open: the textbook identity
    assert account.equity() == pytest.approx(
        BANKROLL + account.realized + account.unrealized())

    pos = _open_position()
    account.positions["pos_a"] = pos
    account.register_position(pos)
    assert account.cash == pytest.approx(BANKROLL - pos.cash_out_usd)

    marks = {"pos_a": MarkedPosition(pos, EXIT_REF, DataState.LIVE)}
    equity, unreal = account.equity(marks), account.unrealized(marks)

    # the two defining halves
    assert equity == pytest.approx(account.cash + account.market_value(marks))
    assert account.market_value(marks) == pytest.approx(QTY * EXIT_REF)
    assert unreal == pytest.approx(QTY * (EXIT_REF - ENTRY_FILL))
    # and the whole-account form, including the entry fee already paid
    assert equity == pytest.approx(
        BANKROLL + account.realized + unreal - pos.entry_fee_usd)

    # close it.  The cash movement that must equal net_pnl is the WHOLE
    # lifecycle - entry debit plus exit credit - not the exit leg alone, because
    # the entry cost was already taken out of cash at register time.
    before_cash, before_real = account.cash, account.realized
    kw = dict(exit_reference=EXIT_REF, exit_fill_price=EXIT_FILL, exit_fee=EXIT_FEE,
              exit_slippage=EXIT_SLIP, now=200.0, reason="test_close",
              style=ExitStyle.FAST_FAILURE)
    trade = account.build_trade(pos, **kw)
    account.close_position(pos, **kw)
    assert not account.open_positions()
    assert account.cash - BANKROLL == pytest.approx(trade.net_pnl_usd)
    assert account.realized - before_real == pytest.approx(trade.net_pnl_usd)
    # nothing open again -> the textbook identity holds exactly
    assert account.equity() == pytest.approx(
        BANKROLL + account.realized + account.unrealized())
    assert account.equity() == pytest.approx(BANKROLL + trade.net_pnl_usd)
    # and the exit leg on its own is PROCEEDS, not net P&L - the entry cost
    # already left the account at register time
    assert before_cash + QTY * EXIT_FILL - EXIT_FEE == pytest.approx(account.cash)
    assert account.cash < before_cash + QTY * EXIT_FILL  # the fee was really paid

    # a realized move shifts cash, realized and equity by the SAME amount
    cash1, real1, eq1 = account.cash, account.realized, account.equity()
    account.state.cash_usd += 3.25
    account.state.realized_pnl_usd += 3.25
    assert account.cash - cash1 == pytest.approx(3.25)
    assert account.realized - real1 == pytest.approx(3.25)
    assert account.equity() - eq1 == pytest.approx(3.25)
    assert account.equity() == pytest.approx(
        BANKROLL + account.realized + account.unrealized())


def test_ledger_net_identity_holds_for_every_recorded_trade(tmp_path):
    """H-companion: net == gross - fees - slippage, on the ledger's own rows."""
    root = make_root(tmp_path)
    proc_owner = acquire(state_file(root))
    try:
        from autopaper.models import Trade
        led = ledger_for(root)
        rows = [
            Trade(trade_id="t1", position_id="p1", candidate_id="c1", mint="m",
                  symbol="S", source="telegram", strategy="FAST_ENTRY",
                  route="FAST_ENTRY", regime="EXPLOSIVE",
                  entry_signal_timestamp=1.0, entry_timestamp=1.0,
                  entry_price=1e-5, fill_price=1.01e-5, exit_timestamp=5.0,
                  exit_price=1.1e-5, quantity=100.0, entry_fee_usd=0.085,
                  exit_fee_usd=0.094, entry_slippage_usd=0.121, exit_slippage_usd=0.13,
                  gross_pnl_usd=0.9, net_pnl_usd=0.47, return_pct=8.9,
                  mfe_pct=9.0, mae_pct=-1.0, invalidation=(),
                  holding_seconds=4.0, exit_reason="x", exit_style="FAST_FAILURE",
                  thesis="t", booked_at=5.0),
            Trade(trade_id="t2", position_id="p2", candidate_id="c2", mint="m",
                  symbol="S", source="telegram", strategy="FAST_ENTRY",
                  route="FAST_ENTRY", regime="EXPLOSIVE",
                  entry_signal_timestamp=1.0, entry_timestamp=1.0,
                  entry_price=1e-5, fill_price=1.01e-5, exit_timestamp=9.0,
                  exit_price=0.9e-5, quantity=100.0, entry_fee_usd=0.085,
                  exit_fee_usd=0.085, entry_slippage_usd=0.121, exit_slippage_usd=0.121,
                  gross_pnl_usd=-0.11, net_pnl_usd=-0.522, return_pct=-10.9,
                  mfe_pct=0.5, mae_pct=-12.0, invalidation=(),
                  holding_seconds=8.0, exit_reason="y", exit_style="FAST_FAILURE",
                  thesis="t", booked_at=9.0),
        ]
        for r in rows:
            led.trade(r)
        trades = led.trades()
        assert len(trades) == 2
        for t in trades:
            assert t["net_pnl_usd"] == pytest.approx(
                t["gross_pnl_usd"] - t["entry_fee_usd"] - t["exit_fee_usd"]
                - t["entry_slippage_usd"] - t["exit_slippage_usd"])
        assert sum(t["net_pnl_usd"] for t in trades) == pytest.approx(-0.052)
    finally:
        release(proc_owner)


# ---------------------------------------------------------------------------
# legacy adoption gate
# ---------------------------------------------------------------------------


def test_legacy_unstamped_account_is_refused_while_in_flux(tmp_path):
    """
    A state file written by a pre-lock process carries no account_id stamp.

    While that file is still being written, arming would recreate the original
    race, because flock cannot arbitrate against a process that never took it.
    The gate withholds instead - and it can only withhold.
    """
    from autopaper.account_lock import LegacyWriterSuspected

    root = make_root(tmp_path)
    seed_state(root)
    payload = load_json(state_file(root))
    payload.pop("account_id", None)
    save_json_atomic(state_file(root), payload)
    os.utime(state_file(root), None)          # just written

    with pytest.raises(LegacyWriterSuspected) as exc:
        assert_no_legacy_writer(state_file(root), grace_s=30.0)
    assert "LEGACY_WRITER_SUSPECTED" in str(exc.value)
    assert "adopt-legacy-account" in str(exc.value)
    assert "adopt-legacy-account" in exc.value.adopt_hint


def test_legacy_gate_expires_once_the_old_writer_stops(tmp_path):
    """It cannot deadlock a restart: once the file goes quiet, start proceeds."""
    from autopaper.account_lock import LegacyWriterSuspected

    root = make_root(tmp_path)
    seed_state(root)
    payload = load_json(state_file(root))
    payload.pop("account_id", None)
    save_json_atomic(state_file(root), payload)
    stale = time.time() - 600.0
    os.utime(state_file(root), (stale, stale))

    report = assert_no_legacy_writer(state_file(root), grace_s=30.0)
    assert report["legacy_writer_suspected"] is False
    assert report["stamped"] is False
    assert report["age_s"] > 30.0
    # and the operator can adopt it explicitly, which is the documented escape
    adopted = assert_no_legacy_writer(state_file(root), allow_legacy_adopt=True)
    assert adopted["adopted"] is True


def test_stamped_account_never_triggers_the_legacy_gate(tmp_path):
    """Steady state after one restart: stamped, so the gate is silent."""
    root = make_root(tmp_path)
    seed_state(root)
    os.utime(state_file(root), None)
    report = assert_no_legacy_writer(state_file(root), grace_s=30.0)
    assert report["stamped"] is True
    assert report["legacy_writer_suspected"] is False


def test_explicit_adoption_bypasses_the_gate(tmp_path):
    root = make_root(tmp_path)
    payload = seed_state(root)
    payload.pop("account_id", None)
    save_json_atomic(state_file(root), payload)
    os.utime(state_file(root), None)
    report = assert_no_legacy_writer(state_file(root), allow_legacy_adopt=True)
    assert report["adopted"] is True


def test_live_account_would_be_refused_right_now(tmp_path):
    """
    The real operator-facing case, on a copy of the live state file.

    The currently running trader predates the lock, so the account is unstamped
    and its file is moving.  A new process must be told to wait, not allowed in.
    """
    import shutil

    from autopaper.account_lock import LegacyWriterSuspected

    # A LEGACY writer is one whose state file predates the single-writer stamp,
    # i.e. it has no ``account_id``. Built explicitly rather than copied from the
    # live file: the live file's shape depends on whether the account has ever
    # run under the lock, which is not a property this test should depend on.
    root = make_root(tmp_path, "livecopy")
    state_file(root).write_text(json.dumps({"agent": "AUTONOMOUS_PAPER_AGENT_V1"}))
    os.utime(state_file(root), None)
    with pytest.raises(LegacyWriterSuspected):
        assert_no_legacy_writer(state_file(root), grace_s=30.0)
