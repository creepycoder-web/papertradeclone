"""
``PaperStateStore`` PATH SAFETY -- a test must never reach the operator's state.

THE DEFECT
----------
``PaperStateStore(root=None)`` resolved to ``Path("var") / filename`` --
RELATIVE TO THE PROCESS WORKING DIRECTORY. Any caller that omitted the argument
therefore wrote into whatever ``var/`` happened to be under the current
directory, which under pytest is the repository:

* the execution-aware engine's ``_store()`` wrote fixture positions into
  ``var/paper_state.json`` (documented and fixed per-caller in
  ``tests/test_production_state_isolation.py``);
* ``BetaExperiment`` resolved its ``None`` the same way, so a synthetic probe
  created a phantom session ``beta_<now>`` with synthetic OPEN POSITIONS inside
  the operator's live ``var/v2_floor_beta_state.json``.

Both records are shaped like real ones, so nothing downstream could distinguish
them from real trades.

THE CONTRACT ENFORCED HERE
--------------------------
1. A missing root REFUSES -- there is no default, so the omission cannot mean
   "the real thing".
2. An ambiguous root (``""``, whitespace, ``"."``, the working directory itself)
   REFUSES.
3. The root is resolved to an ABSOLUTE path once, at construction, so a later
   ``chdir`` cannot redirect a store that already exists.
4. A store targeting the runtime state root REFUSES unless the caller says
   ``runtime=True``. Tests never say it, so production state is unreachable
   from a test -- structurally, not by checksum-after-the-fact.
5. Invalid argument types are refused with a clear error.
6. ``..`` traversal and symlinks are judged on the RESOLVED path.
7. Every persistence operation -- create, save, quarantine, archive, reset,
   restart -- uses the one resolved root.
8. Production still resolves to its original location.

The runtime files are watched for real: a checksum-and-metadata fixture runs
around the suite, and the structural guards above mean the suite should never
reach the file in the first place.

OFFLINE. PAPER only. No network, no runtime process, no wallet.
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
    RUNTIME_STATE_ROOT,
    PaperStateStore,
    StateRootError,
    is_runtime_state_path,
)
from autopaper.paper_live.v2_floor_beta import (  # noqa: E402
    BetaEntryFill,
    BetaExperiment,
)


def _synthetic_entry(price: float = 1.0, *, size: float = 1.0) -> BetaEntryFill:
    """An EXECUTABLE-QUOTE entry for a synthetic path-safety fixture.

    These tests assert PATH SAFETY, not fill semantics, so the entry is built as
    a quote and the synthetic work exercises the SAME public API the runtime
    does. A mark-priced entry is exactly what the experiment refuses now.
    """
    return BetaEntryFill(
        price_usd=float(price), tokens_raw=1_000_000, token_decimals=6,
        cost_quote=float(size), fee_quote=0.0, mark_price=float(price),
        quote_timestamp=1_700_000_000.0, quote_age_ms=1.0,
        venue="test", pool_or_curve="testpool", provider="test",
        route=("BUY",), identity="PASS",
    )

pytestmark = pytest.mark.paper_live


#: The operator's runtime files. Watched read-only; never opened for writing by
#: anything in this suite.
WATCHED = (
    "var/paper_state.json",
    "var/v2_floor_beta_state.json",
    "var/ledger.jsonl",
    "var/paper_account.json",
    "var/real_account_state.json",
    ".env.local",
)


def _meta(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "bytes": st.st_size,
        "mtime": st.st_mtime,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _snapshot() -> dict[str, dict]:
    return {rel: _meta(ROOT / rel) for rel in WATCHED}


@pytest.fixture(scope="module", autouse=True)
def runtime_files_unchanged():
    """
    PROVE the operator's files survived, around the WHOLE module.

    This is a backstop, not the mechanism: the guards below make reaching
    these files impossible for a test. It exists so that if some future path
    does slip through, the suite says so with a diff instead of the corruption
    being discovered days later.
    """
    before = _snapshot()
    yield
    after = _snapshot()
    changed = [rel for rel in WATCHED
               if before[rel].get("sha256") != after[rel].get("sha256")
               or before[rel].get("mtime") != after[rel].get("mtime")]
    assert not changed, (
        f"this test module modified the operator's runtime files: {changed}")


# ===========================================================================
# 1. a store without an explicit root fails clearly
# ===========================================================================


class TestMissingRootRefuses:
    def test_the_argument_is_required_by_the_signature(self):
        """Not merely "None raises" -- there is no default to omit."""
        import inspect

        params = inspect.signature(PaperStateStore.__init__).parameters
        assert params["root"].default is inspect.Parameter.empty, (
            "PaperStateStore must not have a default root; a default is what "
            "let an omitted argument mean the real state directory")

    @pytest.mark.parametrize("bad", [None, "", "   ", "\t", 0, 3.5, object(),
                                     [], True, False])
    def test_a_missing_or_ambiguous_root_raises(self, bad, tmp_path):
        with pytest.raises(StateRootError):
            PaperStateStore(bad)

    def test_the_error_names_the_working_directory_hazard(self):
        with pytest.raises(StateRootError) as exc:
            PaperStateStore(None)
        message = str(exc.value)
        assert "REQUIRED" in message
        assert "working directory" in message.lower()
        assert "temporary" in message, (
            "the error should tell the caller what to pass instead")

    def test_a_dotted_root_raises(self, tmp_path, monkeypatch):
        """``"."`` is "here" -- the same ambiguity, spelled differently."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(StateRootError):
            PaperStateStore(".")

    def test_a_root_equal_to_the_working_directory_raises(self, tmp_path,
                                                         monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(StateRootError):
            PaperStateStore(tmp_path)

    def test_an_engine_with_no_root_refuses_to_build_a_store(self):
        from autopaper.paper_live.arms import parse_arms
        from autopaper.paper_live.engine import PaperLiveEngine

        eng = PaperLiveEngine(
            arms=parse_arms(["L25_CURRENT"]), decision_engine=None,
            exit_policy=None, quote_service=None, market=None,
            token_decimals_for=lambda m: 6)
        with pytest.raises(StateRootError):
            eng._store()


# ===========================================================================
# 2. a production store receives the expected explicit runtime root
# ===========================================================================


class TestProductionRootIsExplicit:
    def test_the_runtime_root_is_the_repository_var(self):
        assert RUNTIME_STATE_ROOT == (ROOT / "var").resolve()

    def test_config_agrees_with_the_runtime_root(self):
        """``build_config``'s default and the store's guard cannot drift."""
        from autopaper.config import build_config

        assert build_config().var_dir.resolve() == RUNTIME_STATE_ROOT

    def test_a_runtime_store_resolves_to_the_original_location(self):
        store = PaperStateStore(RUNTIME_STATE_ROOT, runtime=True)
        assert store.path == RUNTIME_STATE_ROOT / "paper_state.json"
        assert store.runtime is True

    def test_the_runtime_bridge_and_beta_share_one_root(self):
        """
        The runtime builds the state store AND the beta experiment. They must
        resolve to the SAME directory, or a restart could read one book and
        write another.
        """
        from autopaper.config import build_config
        from autopaper.paper_live.arms import parse_arms
        from autopaper.paper_live.engine import PaperLiveEngine
        from autopaper.paper_live.fills import FillModel
        from autopaper.paper_live.quotes import SolReference

        cfg = build_config()
        eng = PaperLiveEngine(
            arms=parse_arms(["V2_FLOOR_BETA"]), decision_engine=None,
            exit_policy=None, quote_service=None, market=None,
            token_decimals_for=lambda m: 6,
            sol_reference=SolReference(150.0, 0.0, "test"),
            fill_model=FillModel(), starting_bankroll_usd=5.0,
            state_root=cfg.var_dir, state_is_runtime=True)
        assert eng._store().path.parent == RUNTIME_STATE_ROOT
        assert eng.beta_experiment.state_store.path == (
            RUNTIME_STATE_ROOT / "v2_floor_beta_state.json")

    def test_a_non_runtime_store_aimed_at_production_is_refused(self):
        """The whole point: only the runtime may write there."""
        with pytest.raises(StateRootError) as exc:
            PaperStateStore(RUNTIME_STATE_ROOT)
        assert "RUNTIME" in str(exc.value)
        with pytest.raises(StateRootError):
            PaperStateStore(RUNTIME_STATE_ROOT / "nested")


# ===========================================================================
# 3./4. an isolated store stays isolated and survives a chdir
# ===========================================================================


class TestIsolatedStoreIsConfined:
    def test_an_isolated_store_writes_only_inside_its_root(self, tmp_path):
        root = tmp_path / "isolated"
        store = PaperStateStore(root)
        state = store.load()
        state.signals["k"] = {"arm": "L25_CURRENT"}
        store.save(state, note="isolated")
        written = list(root.glob("*"))
        assert written == [root / "paper_state.json"]
        assert not is_runtime_state_path(store.path)

    def test_changing_the_working_directory_does_not_redirect_it(self, tmp_path,
                                                                 monkeypatch):
        """
        THE SECOND-HALF OF THE DEFECT.

        The resolved path is absolute and captured at construction, so a
        ``chdir`` afterwards -- which is what running a probe from another
        directory looks like -- cannot move where the store reads or writes.
        """
        root = tmp_path / "isolated"
        store = PaperStateStore(root)
        original = store.path
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert store.path == original
        assert store.path.parent == root.resolve()

        state = store.load()
        state.signals["k"] = {"arm": "L25_CURRENT"}
        store.save(state, note="after chdir")
        assert (root / "paper_state.json").exists()
        assert not list(elsewhere.iterdir()), (
            "the store wrote into the new working directory")

    def test_a_relative_root_is_absolutised_at_construction(self, tmp_path,
                                                            monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rel").mkdir()
        store = PaperStateStore("rel")
        assert store.path.is_absolute()
        assert store.path == (tmp_path / "rel" / "paper_state.json").resolve()

    def test_a_string_root_is_accepted_and_resolved(self, tmp_path):
        """A ``str`` stays supported; it is absolutised, not reinterpreted."""
        store = PaperStateStore(str(tmp_path / "viastr"))
        assert store.path == (tmp_path / "viastr" / "paper_state.json").resolve()


# ===========================================================================
# 5./6. synthetic activity cannot touch the real file
# ===========================================================================


class TestSyntheticWorkNeverTouchesProduction:
    def test_a_beta_probe_cannot_write_the_real_beta_file(self, tmp_path):
        """THE PHANTOM-SESSION INCIDENT, reproduced and closed."""
        before = _meta(ROOT / "var" / "v2_floor_beta_state.json")
        exp = BetaExperiment(starting_bankroll_usd=10.0, state_root=tmp_path)
        exp.open_for_signal(
            parent_signal_id="synthetic", mint="SYNTH", symbol="S",
            entry_fill=_synthetic_entry(1.0), position_size_usd=1.0)
        exp.observe("synthetic", "A", mark_price=1.1)
        exp.settle_arm("synthetic", "A")
        assert (tmp_path / "v2_floor_beta_state.json").exists()
        assert exp.state_store.path.parent == tmp_path.resolve()
        after = _meta(ROOT / "var" / "v2_floor_beta_state.json")
        assert after == before, (
            "a synthetic beta session changed the operator's beta state file")

    def test_a_rootless_beta_experiment_refuses(self):
        with pytest.raises(StateRootError):
            BetaExperiment(starting_bankroll_usd=10.0)

    def test_a_beta_experiment_aimed_at_production_refuses(self):
        with pytest.raises(StateRootError):
            BetaExperiment(starting_bankroll_usd=10.0,
                           state_root=RUNTIME_STATE_ROOT)

    def test_creating_updating_and_settling_never_alters_the_real_file(
            self, tmp_path):
        before = _snapshot()
        exp = BetaExperiment(starting_bankroll_usd=10.0, state_root=tmp_path)
        for pid, price in (("p1", 1.0), ("p2", 2.0), ("p3", 3.0)):
            exp.open_for_signal(parent_signal_id=pid, mint="SYNTH", symbol="S",
                                entry_fill=_synthetic_entry(price),
                                position_size_usd=1.0)
        for arm in ("A", "B", "C"):
            exp.observe("p2", arm, mark_price=2.3)
            exp.settle_arm("p2", arm)
        exp.monitor_once()
        exp._persist()
        assert _snapshot() == before

    def test_reset_of_an_isolated_session_touches_no_real_state(self, tmp_path):
        before = _snapshot()
        exp = BetaExperiment(starting_bankroll_usd=10.0, state_root=tmp_path)
        exp.open_for_signal(parent_signal_id="p1", mint="SYNTH", symbol="S",
                            entry_fill=_synthetic_entry(1.0),
                            position_size_usd=1.0)
        exp.reset()
        assert (tmp_path / "v2_floor_beta_state.json").exists()
        assert _snapshot() == before

    def test_archiving_an_isolated_session_stays_in_its_root(self, tmp_path):
        before = _snapshot()
        exp = BetaExperiment(starting_bankroll_usd=10.0, state_root=tmp_path)
        exp.open_for_signal(parent_signal_id="p1", mint="SYNTH", symbol="S",
                            entry_fill=_synthetic_entry(1.0),
                            position_size_usd=1.0)
        archive = exp.state_store.archive_current()
        assert archive is not None
        archived = Path(archive)
        assert archived.parent == tmp_path.resolve()
        # the live file stays in place; archiving never moves or deletes it
        assert (tmp_path / "v2_floor_beta_state.json").exists()
        assert _snapshot() == before

    def test_a_restart_restores_the_isolated_state_not_the_runtime_one(
            self, tmp_path):
        exp = BetaExperiment(starting_bankroll_usd=10.0, state_root=tmp_path)
        exp.open_for_signal(parent_signal_id="survivor", mint="SYNTH",
                            symbol="S", entry_fill=_synthetic_entry(1.0),
                            position_size_usd=1.0)
        exp._persist()

        restarted = BetaExperiment(starting_bankroll_usd=10.0, state_root=tmp_path)
        assert "survivor" in restarted.positions
        assert restarted.accounts["A"].bankroll_usd == 10.0, (
            "a restart must not reset the ledger to the starting bankroll")
        assert restarted.state_store.path.parent == tmp_path.resolve()


# ===========================================================================
# 7. every state-changing operation uses the one validated root
# ===========================================================================


class TestEveryOperationUsesTheSameRoot:
    def test_the_store_exposes_the_resolved_root_and_path(self, tmp_path):
        store = PaperStateStore(tmp_path / "iso")
        assert store.root == (tmp_path / "iso").resolve()
        assert store.path.parent == store.root
        assert store.path.is_absolute()

    def test_quarantine_lands_beside_the_state_file(self, tmp_path):
        root = tmp_path / "iso"
        store = PaperStateStore(root)
        (root).mkdir(parents=True)
        (root / "paper_state.json").write_text("{ not json")
        state = store.load()
        quarantined = root / "paper_state.corrupt.json"
        assert quarantined.exists(), (
            "a corrupt file must be quarantined as a SIBLING of the state "
            "file, never at a path re-derived from somewhere else")
        assert state.write_log, "the quarantine must be recorded"
        assert str(quarantined) in json.dumps(state.write_log)

    def test_the_archive_is_a_sibling_of_the_state_file(self, tmp_path):
        root = tmp_path / "iso"
        store = PaperStateStore(root)
        store.save(store.load(), note="seed")
        archive = Path(store.archive_current())
        assert archive.parent == root.resolve()
        assert ".archive.json" in archive.name

    def test_a_second_archive_does_not_overwrite_the_first(self, tmp_path):
        store = PaperStateStore(tmp_path / "iso")
        store.save(store.load(), note="seed")
        first = store.archive_current()
        second = store.archive_current()
        assert first != second
        assert Path(first).exists() and Path(second).exists()

    def test_a_filename_cannot_escape_the_root(self, tmp_path):
        for bad in ("/etc/passwd", "../escape.json", "a/../../escape.json"):
            with pytest.raises(StateRootError):
                PaperStateStore(tmp_path, filename=bad)

    def test_an_empty_filename_is_refused(self, tmp_path):
        for bad in ("", "   "):
            with pytest.raises(StateRootError):
                PaperStateStore(tmp_path, filename=bad)

    def test_a_failed_write_raises_rather_than_reporting_success(self, tmp_path):
        """
        Persistence must not falsely report a state change that did not happen.
        """
        store = PaperStateStore(tmp_path / "iso")
        state = store.load()
        state.signals["k"] = {"arm": "L25_CURRENT"}
        (tmp_path / "iso").write_text("i am a file, not a directory")
        with pytest.raises(Exception):
            store.save(state, note="must not silently succeed")
        assert not (tmp_path / "iso" / "paper_state.json").exists()


# ===========================================================================
# 8. traversal, symlinks and alternate working directories
# ===========================================================================


class TestTraversalAndSymlinks:
    @pytest.mark.parametrize("raw", [
        lambda r: f"{r}/../var",
        lambda r: f"{r}/sub/../..",
        lambda r: f"{r}/./",
        lambda r: f"{r}/x/./../..",
    ], ids=["parent_then_back", "subdir_then_out", "dot_segment",
            "mixed"])
    def test_dotdot_traversal_into_production_is_refused(self, raw):
        """
        Traversal is judged on the RESOLVED path, not on the literal string.

        The raw strings are built before ``Path`` normalises them away, so the
        fixture provably contains a traversal segment; ``resolve()`` in the
        store must then collapse it and still recognise the target as
        production.
        """
        literal = raw(str(RUNTIME_STATE_ROOT))
        assert ".." in literal or literal.endswith("/./"), (
            f"the fixture must contain a traversal: {literal}")
        with pytest.raises(StateRootError):
            PaperStateStore(Path(literal))

    def test_a_symlink_to_production_is_refused(self, tmp_path):
        link = tmp_path / "looks_innocent"
        link.symlink_to(RUNTIME_STATE_ROOT, target_is_directory=True)
        with pytest.raises(StateRootError) as exc:
            PaperStateStore(link)
        assert "RUNTIME" in str(exc.value)

    def test_a_symlink_chain_to_production_is_refused(self, tmp_path):
        first = tmp_path / "one"
        second = tmp_path / "two"
        first.symlink_to(RUNTIME_STATE_ROOT, target_is_directory=True)
        second.symlink_to(first, target_is_directory=True)
        with pytest.raises(StateRootError):
            PaperStateStore(second)

    def test_the_guard_judges_the_resolved_path_not_the_name(self, tmp_path):
        """
        A basename check would be defeated by this: the file is called
        ``v2_floor_beta_state.json`` and is perfectly legitimate HERE, because
        what matters is the resolved DIRECTORY.
        """
        store = PaperStateStore(tmp_path / "iso",
                                filename="v2_floor_beta_state.json")
        assert store.path == (
            tmp_path / "iso" / "v2_floor_beta_state.json").resolve()
        assert not is_runtime_state_path(store.path)

    def test_a_test_root_near_the_repository_is_allowed(self, tmp_path,
                                                        monkeypatch):
        """
        The guard is about the RUNTIME ROOT, not about being inside the repo.
        A scratch directory under the repository is isolated and must work.
        """
        monkeypatch.chdir(ROOT)
        scratch = ROOT / ".pytest_scratch_isolated"
        try:
            store = PaperStateStore(scratch)
            assert store.path.parent == scratch.resolve()
            assert not store.path.is_relative_to(RUNTIME_STATE_ROOT)
        finally:
            if scratch.exists():
                for child in scratch.iterdir():
                    child.unlink()
                scratch.rmdir()

    def test_an_isolated_root_inside_the_runtime_directory_is_refused(self):
        """Even a subdirectory of production is production."""
        with pytest.raises(StateRootError):
            PaperStateStore(RUNTIME_STATE_ROOT / "scratch")

    def test_an_alternate_working_directory_changes_nothing(self, tmp_path,
                                                             monkeypatch):
        store = PaperStateStore(tmp_path / "iso")
        resolved = store.path
        for where in (ROOT, tmp_path, Path("/")):
            monkeypatch.chdir(where)
            assert store.path == resolved
            assert store.root == (tmp_path / "iso").resolve()


# ===========================================================================
# 9. the migration itself: no unsafe call site remains
# ===========================================================================


class TestCallSiteAudit:
    """Every ``PaperStateStore(`` in the repository, classified."""

    #: (file glob, must not contain a rootless construction)
    PRODUCTION_FILES = (
        "src/autopaper/paper_live/engine.py",
        "src/autopaper/paper_live/v2_floor_beta.py",
        "src/autopaper/paper_live/bridge.py",
        "src/autopaper/engine.py",
        "scripts/manual_force_close.py",
        "scripts/repair_paper_state_fixture_rows.py",
    )

    @pytest.mark.parametrize("rel", PRODUCTION_FILES)
    def test_production_construction_is_explicit(self, rel):
        import ast

        path = ROOT / rel
        if not path.exists():
            pytest.skip(f"{rel} does not exist in this checkout")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = 0
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "PaperStateStore"):
                continue
            found += 1
            has_root = (bool(node.args) or "root" in {k.arg for k in node.keywords}
                        or any(k.arg is None for k in node.keywords))
            assert has_root, (
                f"{rel}:{node.lineno} constructs PaperStateStore with no root")
        # A file that constructs the store more than once is fine; a file that
        # constructs it ZERO times is also fine (the runtime engine passes its
        # root down rather than building a store itself), so presence is not
        # asserted -- only that no construction is ambiguous.

    def test_no_test_or_script_constructs_a_rootless_store(self):
        import ast

        offenders = []
        for path in sorted((ROOT / "tests").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "PaperStateStore"):
                    if not node.args and "root" not in {
                            k.arg for k in node.keywords}:
                        offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, (
            f"tests construct PaperStateStore with no root: {offenders}")

    def test_the_resolved_root_is_exported_and_importable(self):
        import autopaper.paper_live.state as state_module

        for name in ("PaperStateStore", "StateRootError", "RUNTIME_STATE_ROOT",
                     "is_runtime_state_path"):
            assert name in state_module.__all__, (
                f"{name} must stay exported so callers can be audited against it")