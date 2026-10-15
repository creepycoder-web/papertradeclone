"""
A test must never write into the operator's real ``var/`` state.

THE INCIDENT
------------
``tests/test_strategy_selection_propagation.py`` drove the PAPER entry path with
an execution-aware bridge built by ``tests/paper_bridge_fakes.bridge()``, which
defaults ``root`` to ``None``. ``PaperLiveEngine._store`` resolves ``None`` to
``PaperStateStore(None)``, and ``PaperStateStore`` resolves *its* ``None`` to
``Path("var") / "paper_state.json"`` -- RELATIVE TO THE CWD. Under pytest the
CWD is the repository, so the fixture position (mint ``APxxh2tWCh…``, symbol
``T``) was written into the operator's live resumable book on every run.

Twenty-four such rows accumulated. They are shaped like real ``PaperPosition``
records, so the runtime could not distinguish them from trades: they held the
L25 slot and rendered in OPEN POSITIONS as if they were open positions.

WHAT IS ENFORCED HERE
---------------------
The property is not "this one test passes an argument". It is: **no test may
write to the repository's production state path.** That is asserted structurally,
for the whole suite, at the boundary where a write would actually happen -- and
it is proved by running the offending test's code path and observing the file.

Everything offline. No network, no runtime process, no wallet.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from autopaper.paper_live.state import (  # noqa: E402
    PaperStateStore,
    StateRootError,
    is_runtime_state_path,
)
from tests import paper_bridge_fakes as _fk  # noqa: E402


#: The production paths a test must never touch. ``paper_state.json`` is the
#: resumable PAPER book; ``ledger.jsonl`` is the append-only settlement history.
#: Both are the operator's, not the suite's.
PRODUCTION_STATE = (
    ROOT / "var" / "paper_state.json",
    ROOT / "var" / "ledger.jsonl",
)


def _digest(path: Path) -> str | None:
    """Content digest, or ``None`` when the file does not exist."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot() -> dict[Path, str | None]:
    return {p: _digest(p) for p in PRODUCTION_STATE}


def _open_counts(state_path: Path) -> tuple[int, int, int]:
    if not state_path.exists():
        return (0, 0, 0)
    payload = json.loads(state_path.read_text())
    return (
        len(payload.get("open_positions") or []),
        len(payload.get("trades") or []),
        len(payload.get("signals") or []),
    )


# ===========================================================================
# 1. the defect itself, reproduced and then closed
# ===========================================================================


class TestTheOffendingPathIsIsolated:
    def test_the_strategy_propagation_test_passes_an_isolated_root(self, tmp_path):
        """
        The fix, asserted at the source.

        Reading the test's own source is deliberate: the failure mode was an
        omitted keyword argument, and an omitted argument cannot be caught by
        asserting on behaviour the test itself no longer has. This pins the
        argument so removing it is a visible diff rather than a silent
        regression that only shows up as corrupted operator state.
        """
        source = (
            ROOT / "tests" / "test_strategy_selection_propagation.py"
        ).read_text()
        offending = source[
            source.index(
                "def test_the_runtime_reconciles_the_running_arms_on_the_entry_path"
            ):source.index(
                "def test_restart_preserves_the_configured_strategy"
            )
        ]
        assert "root=tmp_path" in offending, (
            "the runtime-driven strategy-propagation test must build its bridge "
            "with root=tmp_path; without it the bridge's state store resolves to "
            "the CWD-relative Path('var') and writes into the repository"
        )

    def test_a_bridge_without_a_root_is_REFUSED_not_pointed_at_the_repository(
            self, tmp_path, monkeypatch):
        """
        THE HAZARD IS NOW UNREACHABLE.

        This test used to assert that a rootless bridge resolved to the
        operator's state file -- documenting the defect so it could not be
        "simplified" away. The defect is fixed at the source now
        (``PaperStateStore`` requires a root and refuses the runtime directory
        unless told it is the runtime), so the correct thing to assert is that
        the same construction RAISES. The fix above still matters: a raise is
        caught here instead of a corrupted dataset.
        """
        monkeypatch.chdir(ROOT)
        unrooted = _fk.bridge()
        with pytest.raises(StateRootError):
            unrooted._store()
        # and the operator's file is still exactly where it was
        assert not is_runtime_state_path(
            _fk.bridge(root=tmp_path)._store().path)

    def test_a_bridge_with_a_root_never_touches_the_repository(self, tmp_path,
                                                                monkeypatch):
        """The same bridge, given a root, is confined to ``tmp_path``."""
        monkeypatch.chdir(ROOT)
        rooted = _fk.bridge(root=tmp_path)
        resolved = rooted._store().path.resolve()
        assert resolved.is_relative_to(tmp_path.resolve()), resolved
        assert not resolved.is_relative_to(ROOT / "var"), resolved


# ===========================================================================
# 2. the behavioural proof: run the real code path, observe the real file
# ===========================================================================


class TestTheRealEntryPathWritesNothingToProduction:
    """
    The strongest form, because it asserts on the FILE rather than on intent.

    This reproduces the exact sequence the offending test performs -- build the
    execution-aware bridge, drive ``_execute_paper_buy_execution_aware``, which
    ends in ``persist_paper_dataset()`` -- and then checks the operator's
    ``var/paper_state.json`` for a byte-level change.
    """

    def test_the_offending_entry_path_does_not_modify_production_state(
        self, tmp_path, monkeypatch
    ):
        """
        THE REGRESSION TEST.

        Runs the polluting sequence with the fixed construction and asserts the
        production files are bit-identical afterwards.
        """
        from autopaper.config import build_config
        from autopaper.engine import PaperTrader
        from autopaper.paper_live.arms import parse_arms

        class _NoNetworkMarket:
            async def quote_async(self, mint):
                raise AssertionError("no network in this test")

        monkeypatch.chdir(ROOT)
        before = _snapshot()
        before_counts = _open_counts(PRODUCTION_STATE[0])

        (tmp_path / "var").mkdir(parents=True, exist_ok=True)
        cfg = build_config(tmp_path)
        trader = PaperTrader(cfg, market=_NoNetworkMarket())

        # The arm under test only has to be a REAL-L25 arm the runtime can hold;
        # the isolation property being checked is that the entry path wrote
        # nowhere but the fixture, which does not depend on which L25 arm it is.
        engine = _fk.bridge(
            arms=parse_arms(["L25_MAX_HOLD"]), root=tmp_path)
        trader._paper_bridge = engine
        trader._paper_bridge_arms = ("L25_MAX_HOLD",)
        trader.state.select_strategies(["L25_CURRENT"])

        _fk.run(trader._execute_paper_buy_execution_aware(
            _fk.real_report(), _fk.alert(), None))

        # The test's own assertions: the entry really happened, so a silent
        # no-op cannot make the isolation check below pass vacuously.
        assert engine._opened() if hasattr(engine, "_opened") else True
        assert any(
            getattr(account, "position", None) is not None
            for account in engine.accounts.values()
        ), "the entry opened nothing, so the persistence check would be vacuous"

        # THE FIXTURE LANDED SOMEWHERE -- AND IT IS NOT THE REPOSITORY.
        assert (tmp_path / "var" / "paper_state.json").exists(), (
            "the position was not persisted anywhere; the entry path under test "
            "is no longer the one that was fixed")
        tmp_state = json.loads(
            (tmp_path / "var" / "paper_state.json").read_text())
        assert tmp_state["open_positions"], (
            "the isolated state file records no open position")
        assert all(
            p["mint"] == _fk.MINT for p in tmp_state["open_positions"]
        ), "the isolated file does not hold the fixture position"

        # AND PRODUCTION IS UNTOUCHED.
        after = _snapshot()
        after_counts = _open_counts(PRODUCTION_STATE[0])
        assert after == before, (
            "running this test modified production state: "
            f"{[(p.name, before[p], after[p]) for p in PRODUCTION_STATE if before[p] != after[p]]}"
        )
        assert after_counts == before_counts, (
            f"production open/trade/signal counts moved {before_counts} -> "
            f"{after_counts}")

    def test_production_state_carries_no_fixture_positions(self):
        """
        The repository's live book must hold no fixture row.

        Read-only. This is the assertion that would have caught the original
        defect on the very next test run, from any test in the suite, rather
        than through a manual audit weeks later.
        """
        if not PRODUCTION_STATE[0].exists():
            pytest.skip("no production PAPER state file present")

        payload = json.loads(PRODUCTION_STATE[0].read_text())
        offenders = [
            {
                "signal_id": p.get("signal_id"),
                "mint": p.get("mint"),
                "symbol": p.get("symbol"),
            }
            for p in (payload.get("open_positions") or [])
            if p.get("mint") == _fk.MINT
            or (p.get("entry_fill") or {}).get("pool_or_curve") == _fk.PAIR
        ]
        assert not offenders, (
            "the operator's var/paper_state.json contains offline test-fixture "
            f"positions: {offenders}. A test wrote them; see "
            "test_the_offending_entry_path_does_not_modify_production_state"
        )


# ===========================================================================
# 3. the store itself: the ambiguity that caused all of this
# ===========================================================================


class TestPaperStateStoreRefusesAnAmbiguousRoot:
    @pytest.mark.parametrize("bad_root", [None, "", "   ", 0, 3.5, object()])
    def test_a_rootless_or_ambiguous_store_is_REFUSED(self, bad_root, tmp_path,
                                                      monkeypatch):
        """
        ``PaperStateStore`` used to resolve a missing root to
        ``<cwd>/var/paper_state.json`` -- the whole defect. It now refuses, so
        the omission cannot reach the operator's book.

        ``""`` and whitespace are included because ``Path("")`` is ``Path(".")``,
        the working directory: the same ambiguity in a different spelling.
        """
        monkeypatch.chdir(tmp_path)
        with pytest.raises(StateRootError):
            PaperStateStore(bad_root)
        assert not list(Path(tmp_path).glob("*.json"))

    def test_a_store_rooted_at_the_working_directory_is_refused(
            self, tmp_path, monkeypatch):
        """``Path(".")`` means "here" -- also refused, for the same reason."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(StateRootError):
            PaperStateStore(".")
        with pytest.raises(StateRootError):
            PaperStateStore(tmp_path)

    def test_a_rooted_store_is_confined_to_that_root(self, tmp_path):
        store = PaperStateStore(tmp_path / "var")
        assert store.path.resolve().is_relative_to(tmp_path.resolve())
        assert store.path.name == "paper_state.json"

    def test_the_fixture_bridge_factory_defaults_root_to_none(self):
        """
        The default is the hazard, so it is pinned rather than left implicit.

        If a future change makes ``bridge()`` default to an isolated root, this
        fails and the reviewer learns the safety net moved -- rather than
        discovering later that every test silently changed where it writes.
        """
        bridge = _fk.bridge()
        assert bridge.root is None, (
            "tests/paper_bridge_fakes.bridge() no longer defaults root to None; "
            "if that changed deliberately, update this test and re-audit every "
            "caller, because the default is what made an omitted root= argument "
            "write into the repository"
        )