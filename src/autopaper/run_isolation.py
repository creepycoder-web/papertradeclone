"""
Containment B: ``run`` refuses to start when real-execution capability is live.

WHAT G1 FOUND, AND WHY A SOURCE-LEVEL TEST IS NOT ENOUGH
--------------------------------------------------------
``autopaper.cli.cmd_run`` imports ``autopaper.engine``, which at module level
does ``from .execution.real import LiveSolanaBackend, RealExecutor``
(``engine.py:90``), and ``execution/real.py`` imports ``.solana`` at module
level in turn (``real.py:63``). ``execution/solana.py`` then loads a Solana
keypair with ``from solders.keypair import Keypair`` (``solana.py:327``), and
``execution/real.py`` builds and signs transactions with
``solders.transaction.VersionedTransaction`` (``real.py:554``, ``:733``).

Those imports are inside function bodies, so the module tree imports cleanly
without ``solders`` present. That is the whole point of this module: the REAL
capability is **in the import graph reachable from ``run``** and cannot be
removed from it without a refactor nobody asked for. What makes the paper
deployment safe is a different fact -- the capability-bearing distributions are
NOT INSTALLED -- and that is an environment fact, not a property of the code.

So this module makes the environment fact enforced rather than assumed:

* :func:`walk_imports` is the honest measurement. It reports every module name
  reachable from an entry module, following module-level imports, imports inside
  function bodies, and ``importlib.import_module("literal")`` calls. It does
  NOT pretend the real-execution modules are absent from the graph.
* :func:`assert_run_path_isolated` is the enforcement. It refuses to let ``run``
  start if any capability-bearing distribution is actually importable, or if
  anything on ``safety.py``'s forbidden lists is reachable at all.

THE PESSIMISTIC READING, RECORDED
---------------------------------
``solders`` IS reachable in the source graph from ``run``. This module does not
claim otherwise. It claims the narrower, checkable thing: on a machine where the
capability is not installed, it cannot be exercised, and the moment someone
installs it ``run`` stops starting instead of quietly gaining the ability to
sign. A guarantee that depends on a package being absent is weaker than one that
depends on the code being absent, and the RUNBOOK says so.

WHY THE NAMES BELOW ARE DECLARED HERE
-------------------------------------
``safety.py`` used to block ``solders``, ``solana`` and every swap router from
being importable. That barrier was deliberately INVERTED (``safety.py:29-36``):
ML/Laya/shadow became structurally impossible, and REAL arming became
procedurally guarded, so ``FORBIDDEN_MODULES`` no longer carries any of these
names. They therefore cannot be imported from there, and they are declared here
instead, next to the check that uses them. ``safety.py`` is not edited and its
own lists are imported, never copied.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .safety import FORBIDDEN_MODULES, SafetyViolation, is_forbidden_module

#: Distributions that CARRY the real-execution capability: Solana key handling,
#: transaction construction and signing, address codecs, and the swap routers an
#: order would be placed through. Present in the source graph (see the module
#: docstring); the paper deployment is safe because they are not installed.
#:
#: Declared here, not in ``safety.py``, because that module's barrier was
#: inverted and no longer carries them -- see the docstring.
CAPABILITY_DISTRIBUTIONS: frozenset[str] = frozenset(
    {
        "solders",
        "solana",
        "anchorpy",
        "base58",
        "based58",
        "jupiter",
        "raydium",
        "pumpswap",
        "ccxt",
    }
)

#: Research distributions the test suite cross-checks the frozen strategy
#: against. Not capabilities; listed so the walker's report distinguishes
#: "missing research tool" from "missing capability".
RESEARCH_DISTRIBUTIONS: frozenset[str] = frozenset(
    {"l25x", "l25r", "savannah", "randorder", "r2search", "research"}
)

#: The module that implements ``run``. The walk starts here.
RUN_ENTRY_MODULE = "autopaper.cli"


class RunPathViolation(SafetyViolation):
    """``run`` must not start: real-execution capability is live."""


@dataclass
class ImportGraph:
    """What a walk found, as data rather than as a verdict."""

    entry: str
    #: Every module name referenced anywhere in the walked sources.
    names: set[str] = field(default_factory=set)
    #: Names reached from inside a function body rather than at module level.
    lazy_names: set[str] = field(default_factory=set)
    #: Files actually parsed.
    walked: set[str] = field(default_factory=set)

    def roots(self) -> set[str]:
        return {n.split(".", 1)[0] for n in self.names}

    def capability_names(self) -> set[str]:
        return {n for n in self.names
                if n.split(".", 1)[0] in CAPABILITY_DISTRIBUTIONS}

    def forbidden_names(self) -> set[str]:
        return {n for n in self.names if is_forbidden_module(n)}


def _module_file(name: str) -> Optional[Path]:
    """
    The source file for ``name``, WITHOUT importing it.

    ``find_spec`` is import-free for a package that is already on the path, and
    returning ``None`` for anything it cannot locate is what keeps the walk from
    needing the capability distributions to be installed in order to audit them.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError, AttributeError):
        return None
    if spec is None or not spec.origin or spec.origin == "built-in":
        return None
    path = Path(spec.origin)
    return path if path.suffix == ".py" and path.exists() else None


def _resolve(node: ast.ImportFrom, module: str) -> str:
    """Absolute name for a relative ``from . import x``."""
    if not node.level:
        return node.module or ""
    parts = module.split(".")
    # A module's own package is parts[:-1]; each extra level climbs one more.
    base = parts[: max(0, len(parts) - node.level)]
    return ".".join([*base, node.module]) if node.module else ".".join(base)


_WALK_CACHE: dict[tuple[str, str, int], "ImportGraph"] = {}


def walk_imports(entry: str = RUN_ENTRY_MODULE, *,
                 within: str = "autopaper",
                 max_modules: int = 400,
                 refresh: bool = False) -> ImportGraph:
    """
    Every module name reachable from ``entry``, by static analysis only.

    Follows three forms, because a guarantee that only covers module-level
    imports is not a guarantee: a lazy ``from solders.keypair import Keypair``
    inside a function is exactly as capable as one at the top of the file.

    * ``import a.b`` / ``from a.b import c`` at module level;
    * the same inside any function, method or class body;
    * ``importlib.import_module("a.b")`` with a LITERAL argument. A computed
      name cannot be resolved statically and is reported in ``names`` as the
      sentinel ``"<computed>"`` so it is visible rather than silently ignored.

    Recursion is bounded to modules under ``within`` -- walking the standard
    library would add nothing and cost seconds. Names outside it are RECORDED
    but not expanded, which is what lets the capability check see ``solders``
    without needing it installed.
    """
    # CACHED PER PROCESS. The walk AST-parses every module under ``within``,
    # which for this package includes two 8,000-line engines, so it costs about
    # a second. A live run calls it once at startup, but ``health --startup``
    # and the tests call it repeatedly. The source cannot change inside one
    # process, so the result cannot go stale; ``refresh`` exists for a test that
    # writes a fixture package and needs a fresh walk.
    key = (entry, within, max_modules)
    if not refresh and key in _WALK_CACHE:
        return _WALK_CACHE[key]

    graph = ImportGraph(entry=entry)
    queue: list[str] = [entry]
    seen: set[str] = set()

    while queue and len(graph.walked) < max_modules:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_file(name)
        if path is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        graph.walked.add(name)

        # ONE traversal, with the lazy flag threaded down.
        #
        # The obvious implementation walks the tree twice -- once for every
        # name, once more under each FunctionDef to find the lazy ones -- which
        # doubles the parse-adjacent work on this package's two 8,000-line
        # engines and pushed this test past its one-second budget. Threading a
        # flag through a single recursive descent costs one traversal and
        # answers both questions.
        for found, lazy in _descend(tree, name, inside_function=False):
            if not found:
                continue
            graph.names.add(found)
            if lazy:
                graph.lazy_names.add(found)
            if found == within or found.startswith(within + "."):
                queue.append(found)
    _WALK_CACHE[key] = graph
    return graph


def _descend(node: ast.AST, module: str, *, inside_function: bool):
    """Yield ``(module_name, was_lazy)`` for one subtree, depth first."""
    for found in _names_of(node, module):
        yield found, inside_function
    in_func = inside_function or isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef)
    )
    for child in ast.iter_child_nodes(node):
        yield from _descend(child, module, inside_function=in_func)


def _names_of(node: ast.AST, module: str) -> Iterable[str]:
    """The module names one AST node references, if any."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        resolved = _resolve(node, module)
        if not resolved:
            return []
        out = [resolved]
        # ``from pkg import sub`` may name a submodule, so record both forms.
        out += [f"{resolved}.{a.name}" for a in node.names if a.name != "*"]
        return out
    if isinstance(node, ast.Call):
        func = node.func
        is_import_module = (
            (isinstance(func, ast.Attribute) and func.attr == "import_module")
            or (isinstance(func, ast.Name) and func.id == "import_module")
        )
        if is_import_module and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return [first.value]
            return ["<computed>"]
    return []


def capability_report(graph: Optional[ImportGraph] = None) -> dict[str, object]:
    """
    What is reachable, and what of it is actually installed.

    Returns data, not a pass/fail, so ``health`` can print the same facts the
    assertion refuses on.
    """
    graph = graph or walk_imports()
    reachable = sorted(graph.capability_names())
    installed = sorted(
        {n.split(".", 1)[0] for n in reachable
         if importlib.util.find_spec(n.split(".", 1)[0]) is not None}
    )
    return {
        "entry": graph.entry,
        "modules_walked": len(graph.walked),
        "capability_names_reachable": reachable,
        "capability_distributions_installed": installed,
        "forbidden_names_reachable": sorted(graph.forbidden_names()),
        "research_distributions_reachable": sorted(
            {n for n in graph.names
             if n.split(".", 1)[0] in RESEARCH_DISTRIBUTIONS}
        ),
        "isolated": not installed and not graph.forbidden_names(),
    }


def assert_run_path_isolated(graph: Optional[ImportGraph] = None) -> dict[str, object]:
    """
    Refuse to start ``run`` if real-execution capability is live.

    Two refusals, and they are different in kind:

    * a module on ``safety.py``'s own forbidden lists is reachable at all --
      that barrier IS structural, so reaching one is a construction bug;
    * a capability-bearing distribution is INSTALLED. The source graph reaches
      it either way (see the module docstring); what this refuses is the
      machine being able to execute it.

    Returns the report on success so the caller can log what it checked.
    """
    report = capability_report(graph)
    forbidden = report["forbidden_names_reachable"]
    if forbidden:
        raise RunPathViolation(
            "refusing to start: modules on the safety barrier's forbidden list "
            f"are reachable from {report['entry']}: {', '.join(forbidden)}. "
            "That barrier is structural, so this is a construction bug, not a "
            "configuration problem."
        )
    installed = report["capability_distributions_installed"]
    if installed:
        raise RunPathViolation(
            "refusing to start: real-execution capability is INSTALLED and "
            f"reachable from {report['entry']}: {', '.join(installed)}. "
            "The paper runtime's isolation depends on these distributions being "
            "absent (see docs/RUNBOOK.md, Security notes). Uninstall them from "
            "this environment, or run the paper bot in a virtualenv that does "
            "not have them."
        )
    return report


__all__ = [
    "CAPABILITY_DISTRIBUTIONS",
    "RESEARCH_DISTRIBUTIONS",
    "RUN_ENTRY_MODULE",
    "FORBIDDEN_MODULES",
    "ImportGraph",
    "RunPathViolation",
    "assert_run_path_isolated",
    "capability_report",
    "walk_imports",
]
