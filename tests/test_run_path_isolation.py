"""
G3a: nothing `autopaper.cli run` can reach may exercise real execution.

THE HONEST SHAPE OF THIS GUARANTEE
----------------------------------
G1 established that `solders` IS reachable in the source graph from `run`:
`cli.py:87 -> engine.py:90 -> execution/real.py:63 -> execution/solana.py:327`,
the last two being lazy in-function imports. No test can claim that graph is
clean, so these tests do not. They assert the narrower, checkable fact that
makes the paper deployment safe:

* nothing on `safety.py`'s own forbidden lists is reachable at all (that
  barrier is structural);
* no capability-bearing distribution is INSTALLED, so the reachable code cannot
  be executed;
* `run` REFUSES TO START the moment either of those stops holding.

The walker is also tested against a fixture package, because a walker that
missed a lazy import would make the whole check worthless while passing.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from autopaper import run_isolation
from autopaper.run_isolation import (
    CAPABILITY_DISTRIBUTIONS,
    RUN_ENTRY_MODULE,
    ImportGraph,
    RunPathViolation,
    assert_run_path_isolated,
    capability_report,
    walk_imports,
)
from autopaper.safety import FORBIDDEN_MODULES

pytestmark = pytest.mark.integrity


@pytest.fixture(scope="module")
def graph() -> ImportGraph:
    """One walk for the whole module: it is the expensive part."""
    return walk_imports()


# ---------------------------------------------------------------------------
# the walker is trustworthy
# ---------------------------------------------------------------------------


def _fixture_pkg(tmp_path, body: str, name: str):
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "entry.py").write_text(textwrap.dedent(body))
    sys.path.insert(0, str(tmp_path))
    try:
        yield_graph = walk_imports(f"{name}.entry", within=name)
    finally:
        sys.path.remove(str(tmp_path))
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            sys.modules.pop(mod, None)
    return yield_graph


def test_the_walker_catches_a_lazy_forbidden_import(tmp_path):
    """
    The case that matters. A forbidden import inside a function body is exactly
    as capable as one at the top of the file, and a walker that only read
    module level would pass this while the capability was live.
    """
    g = _fixture_pkg(tmp_path, """
        def innocent():
            return 1

        def not_innocent():
            from solders.keypair import Keypair   # lazy, inside a function
            return Keypair
    """, "lazypkg")
    assert "solders.keypair" in g.names
    assert "solders.keypair" in g.lazy_names, "the lazy path was not marked lazy"
    assert g.capability_names(), "a capability import was not classified"


def test_the_walker_catches_a_lazy_safety_forbidden_import(tmp_path):
    """The same, for a name on safety.py's own structural list."""
    forbidden = sorted(FORBIDDEN_MODULES)[0]
    g = _fixture_pkg(tmp_path, f"""
        def later():
            import {forbidden}
            return {forbidden}
    """, "mlpkg")
    assert g.forbidden_names(), f"{forbidden} was not caught"


def test_the_walker_catches_a_literal_importlib_call(tmp_path):
    g = _fixture_pkg(tmp_path, """
        import importlib

        def dynamic():
            return importlib.import_module("solders")
    """, "dynpkg")
    assert "solders" in g.names


def test_a_computed_importlib_name_is_reported_not_ignored(tmp_path):
    """
    A computed name cannot be resolved statically. Recording a sentinel keeps
    the gap visible instead of letting it read as "nothing found".
    """
    g = _fixture_pkg(tmp_path, """
        import importlib

        def dynamic(which):
            return importlib.import_module(which)
    """, "computedpkg")
    assert "<computed>" in g.names


def test_the_walker_follows_module_level_imports(tmp_path):
    g = _fixture_pkg(tmp_path, """
        import solders
    """, "eagerpkg")
    assert "solders" in g.names
    assert "solders" not in g.lazy_names


# ---------------------------------------------------------------------------
# the real code
# ---------------------------------------------------------------------------


def test_the_walk_starts_at_the_module_that_implements_run(graph):
    assert graph.entry == RUN_ENTRY_MODULE
    assert "autopaper.engine" in graph.walked, "the engine was never walked"
    assert len(graph.walked) > 20, f"only {len(graph.walked)} modules walked"


def test_nothing_on_the_safety_barriers_forbidden_list_is_reachable(graph):
    """
    This barrier IS structural: ML, Laya and shadow modules must not appear in
    the graph at all. Reaching one would be a construction bug.
    """
    assert graph.forbidden_names() == set(), (
        f"forbidden modules reachable from run: {sorted(graph.forbidden_names())}"
    )


def test_no_real_execution_capability_is_installed(graph):
    """
    The capability IS in the source graph -- G1 proved it and this test does
    not pretend otherwise. What must hold is that none of it can be executed
    here, because the distributions are absent.
    """
    report = capability_report(graph)
    assert report["capability_distributions_installed"] == [], (
        "real-execution capability is INSTALLED: "
        f"{report['capability_distributions_installed']}"
    )
    assert report["isolated"] is True


def test_the_capability_is_reachable_in_source_and_that_is_recorded(graph):
    """
    The pessimistic reading, pinned as a fact rather than left implicit. If a
    future refactor genuinely removes real execution from the graph, this test
    fails and the RUNBOOK's security note must be rewritten to match.
    """
    reachable = graph.capability_names()
    assert reachable, (
        "no capability name is reachable any more -- if real execution was "
        "moved out of the package, update docs/RUNBOOK.md and var/audit.md"
    )
    assert any(n.split(".", 1)[0] == "solders" for n in reachable)


def test_run_is_allowed_to_start_on_this_machine(graph):
    report = assert_run_path_isolated(graph)
    assert report["isolated"] is True


# ---------------------------------------------------------------------------
# the refusal
# ---------------------------------------------------------------------------


def test_run_refuses_when_a_capability_distribution_is_installed(
    graph, monkeypatch
):
    """
    Containment B, and the message that has to be actionable at 3am. Installing
    solders into this environment must stop the paper bot from starting, not
    silently hand it the ability to sign.

    The cached graph is passed in so this re-checks installation without
    re-walking the tree.
    """
    real_find_spec = run_isolation.importlib.util.find_spec

    def fake(name, *a, **k):
        if name == "solders":
            return object()          # pretend it is installed
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(run_isolation.importlib.util, "find_spec", fake)
    with pytest.raises(RunPathViolation) as exc:
        assert_run_path_isolated(graph)
    message = str(exc.value)
    assert "solders" in message
    assert "refusing to start" in message
    # the remedy, not just the diagnosis
    assert "Uninstall" in message or "virtualenv" in message


def test_run_refuses_when_a_structurally_forbidden_module_is_reachable():
    forbidden = sorted(FORBIDDEN_MODULES)[0]
    graph = ImportGraph(entry="fake", names={forbidden}, walked={"fake"})
    with pytest.raises(RunPathViolation) as exc:
        assert_run_path_isolated(graph)
    assert "construction bug" in str(exc.value)


def test_capability_names_are_declared_not_copied_from_safety():
    """
    safety.py's barrier was inverted and no longer carries these names, so they
    cannot be imported from there. What it DOES carry must still be imported
    rather than duplicated, so the static source scan never sees two copies.
    """
    assert not (CAPABILITY_DISTRIBUTIONS & FORBIDDEN_MODULES)
    assert run_isolation.FORBIDDEN_MODULES is FORBIDDEN_MODULES


# ---------------------------------------------------------------------------
# the refusal is actually wired into run
# ---------------------------------------------------------------------------


def test_cmd_run_checks_isolation_before_building_anything(tmp_path, monkeypatch,
                                                           config):
    """
    The check must sit BEFORE the engine is constructed, so a machine that has
    gained the capability never reaches the alert, decision, sizing or exit
    paths. Proven by making the check raise and asserting `cmd_run` returns the
    refusal code without ever calling the trader factory.

    This never starts the engine and never touches the network: the raise
    happens before `_trader` would be called.
    """
    import argparse

    from autopaper import cli

    built = []
    monkeypatch.setattr(cli, "_trader", lambda *a, **k: built.append(1))
    monkeypatch.setattr(
        cli, "assert_run_path_isolated",
        lambda *a, **k: (_ for _ in ()).throw(
            RunPathViolation("solders is INSTALLED and reachable")),
    )
    args = argparse.Namespace(log_level="ERROR", cli=False, reset_control=False)
    assert cli.cmd_run(config, args) == 6
    assert built == [], "the engine was constructed despite the refusal"
