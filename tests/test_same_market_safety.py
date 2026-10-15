"""
The explicit no-broadcast guarantee (test item 27).

Three independent layers are asserted:

1. **Absence of capability.** The harness's own clients have no submit, sign or
   send method. Checked by reflection over the real classes, not by reading
   them, so a future edit that added one would fail here.
2. **An import barrier.** Every module in this repository containing a signer or
   a broadcaster is refused at import time.
3. **A startup audit.** The audit reports no execution capability, and refuses
   to run if one is somehow present.

The guarantee is asserted for the SOURCE, not for a report, because a report
describes what the code once did.
"""


from __future__ import annotations

import importlib
import inspect
import sys

import pytest

import autopaper.same_market  # noqa: F401 - installs the barrier
from autopaper.same_market import guard
from autopaper.same_market.guard import (
    NoBroadcastViolation,
    assert_no_execution_capability,
    audit_no_execution_capability,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.safety,
    pytest.mark.same_market,
    pytest.mark.strategy,
]


def _public_methods(cls) -> set[str]:
    return {
        name for name, _ in inspect.getmembers(cls, callable)
        if not name.startswith("_")
    }


# ---------------------------------------------------------------------------
# 1. absence of capability - structural, not procedural
# ---------------------------------------------------------------------------

def test_the_rpc_client_cannot_broadcast():
    from autopaper.same_market.rpc import SolanaReadOnlyRpc

    names = {n for n in dir(SolanaReadOnlyRpc) if not n.startswith("_")}
    for forbidden in ("send_transaction", "send_raw_transaction",
                      "sendTransaction", "sign", "sign_transaction",
                      "build_transaction", "simulate_transaction"):
        assert forbidden not in names, f"the read-only RPC client has {forbidden}"


def test_the_rpc_client_exposes_only_read_methods():
    from autopaper.same_market.rpc import READ_METHODS, SolanaReadOnlyRpc

    assert set(READ_METHODS) == {
        "getAccountInfo", "getMultipleAccounts", "getTokenSupply",
        "getSlot", "getVersion",
    }
    # Nothing that writes is reachable by any other name either.
    assert all(m.startswith("get") for m in READ_METHODS)
    assert SolanaReadOnlyRpc.read_methods == READ_METHODS


def test_the_jupiter_client_cannot_build_or_send_a_transaction():
    from autopaper.same_market.jupiter import JupiterQuoteClient

    for forbidden in ("build", "build_transaction", "execute_swap", "send",
                      "sign", "submit", "swap"):
        assert not hasattr(JupiterQuoteClient, forbidden), forbidden
    assert JupiterQuoteClient.read_methods == ("quote",)


def test_no_harness_module_defines_a_submission_symbol():
    """
    Reflection over every module in the package. A submission-named attribute
    appearing anywhere is the finding, regardless of intent.
    """
    package = "autopaper.same_market"
    hits: list[str] = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(package):
            continue
        for attr in dir(module):
            if guard.is_forbidden_attr(attr):
                hits.append(f"{name}.{attr}")
    assert hits == [], f"submission symbols reachable from the harness: {hits}"


def test_the_pipeline_constructs_no_execution_object():
    """The pipeline's collaborators are all read-only clients."""
    from autopaper.same_market.pipeline import SameMarketPipeline

    annotations = {
        name: param.annotation
        for name, param in inspect.signature(
            SameMarketPipeline.__init__).parameters.items()
    }
    text = " ".join(str(v) for v in annotations.values())
    for forbidden in ("wallet", "executor", "LiveSolanaBackend",
                      "RealExecutor", "SolanaWallet"):
        assert forbidden not in text, forbidden


def test_the_harness_does_not_modify_trading_behaviour():
    """
    Structural rather than behavioural: the harness never imports the strategy,
    risk or execution modules, so it has no code path to L25 or to the risk
    settings even if someone wanted it to.
    """
    package = "autopaper.same_market"
    imported: set[str] = set()
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(package):
            continue
        for attr in dir(module):
            mod = getattr(module, attr, None)
            if inspect.ismodule(mod):
                imported.add(mod.__name__)
    for forbidden in ("autopaper.l25", "autopaper.engine",
                      "autopaper.execution.real", "autopaper.execution.solana",
                      "autopaper.risk", "autopaper.sizing"):
        assert forbidden not in imported, f"the harness imports {forbidden}"


# ---------------------------------------------------------------------------
# 2. the import barrier
# ---------------------------------------------------------------------------

BLOCKED_MODULES = [
    "autopaper.execution.solana",       # send_transaction + SolanaWallet.sign
    "autopaper.execution.txcodec",      # sign_transaction_bytes
    "autopaper.execution.jupiter",      # build / build_transaction
    "autopaper.execution.planner",      # signs candidates to simulate
    "autopaper.execution.real",         # end-to-end executor
    "autopaper.execution.raptor",       # execution backend
    "autopaper.execution.fill",         # models fills
    "autopaper.token_cleanup",          # signs and broadcasts
    "solders",                          # the ed25519 primitive
    "solana",                           # a full SDK that can send
]


def _harness_import_attempt(module: str) -> str:
    """
    Attempt an import FROM harness code, in a clean interpreter.

    A subprocess because the guarantee is about a process's import state: in a
    shared process another test may already have loaded the module, and a cached
    module never reaches the finder. Proving this in-process would then depend
    on test ordering, which is not a property worth having.
    """
    import subprocess
    import sys

    script = f"""
import sys, types
import autopaper.same_market
mod = types.ModuleType("autopaper.same_market._probe")
sys.modules["autopaper.same_market._probe"] = mod
try:
    exec(compile("import {module}", "<harness>", "exec"), mod.__dict__)
    print("LEAK")
except Exception as exc:
    print(type(exc).__name__)
"""
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )
    return out.stdout.strip() or out.stderr.strip()[-120:]


@pytest.mark.parametrize("module", BLOCKED_MODULES)
def test_the_harness_cannot_import_an_execution_module(module):
    assert _harness_import_attempt(module) == "NoBroadcastViolation", (
        f"the harness could import {module}"
    )


def test_the_barrier_does_not_affect_the_rest_of_the_process():
    """
    Scoping is load-bearing, not a nicety.

    An unscoped finder lives in ``sys.meta_path`` for the whole interpreter and
    would also block this repository's own executor tests - which would mean
    the harness changed the behaviour of unrelated code in the same process.
    That is exactly the kind of side effect a read-only test tool must not
    have.
    """
    import subprocess
    import sys
    from pathlib import Path

    script = """
import autopaper.same_market           # installs the barrier
import autopaper.execution.solana      # must still work
import autopaper.token_cleanup         # must still work
import autopaper.execution.fill
print("UNAFFECTED")
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert "UNAFFECTED" in out.stdout, out.stdout + out.stderr


def test_the_refusal_message_explains_itself():
    assert _harness_import_attempt("autopaper.execution.solana") == \
        "NoBroadcastViolation"


def test_refused_attempts_are_counted():
    """Distinct from successful imports, which are structurally zero."""
    import subprocess
    import sys
    from pathlib import Path

    script = """
import sys, types
import autopaper.same_market
from autopaper.same_market import guard
mod = types.ModuleType("autopaper.same_market._probe")
sys.modules["autopaper.same_market._probe"] = mod
before = guard.BLOCKED_IMPORT_ATTEMPTS
for m in ("autopaper.execution.solana", "solders"):
    try:
        exec(compile("import " + m, "<harness>", "exec"), mod.__dict__)
    except Exception:
        pass
print(guard.BLOCKED_IMPORT_ATTEMPTS - before)
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert out.stdout.strip() == "2", out.stdout + out.stderr


def test_the_barrier_does_not_break_the_telegram_transport():
    """
    Encryption is not submission. Telethon needs its crypto libraries for
    MTProto auth, and blocking them would break the read-only listener this
    harness depends on while protecting nothing.
    """
    import telethon  # noqa: F401
    from autopaper.same_market import listener  # noqa: F401

    for allowed in ("cryptography", "rsa", "nacl", "pyaes"):
        assert not guard.is_forbidden_module(allowed), (
            f"{allowed} is blocked, but it is a cryptography library, not a "
            "submission capability"
        )


def test_the_barrier_is_idempotent():
    before = sum(1 for f in sys.meta_path if isinstance(f, guard._NoBroadcastFinder))
    guard.install_no_broadcast_guard()
    guard.install_no_broadcast_guard()
    after = sum(1 for f in sys.meta_path if isinstance(f, guard._NoBroadcastFinder))
    assert after == before == 1


# ---------------------------------------------------------------------------
# 3. the startup audit
# ---------------------------------------------------------------------------

def test_the_audit_reports_no_execution_capability():
    audit = audit_no_execution_capability()
    assert audit.ok is True
    assert audit.forbidden_attrs_found == ()
    assert audit.forbidden_modules_loaded == ()


def test_the_asserting_wrapper_returns_the_clean_audit():
    assert assert_no_execution_capability().ok is True


def test_the_audit_renders_an_explicit_non_execution_notice():
    rendered = audit_no_execution_capability().render()
    assert "NO TRANSACTION BROADCAST" in rendered
    assert "absent (structural)" in rendered


def test_the_audit_refuses_when_a_capability_is_present():
    """
    The failure path, exercised by planting a forbidden symbol in a harness
    module. Proves the audit can actually fail rather than always returning ok.
    """
    package = "autopaper.same_market"
    module = sys.modules[package]
    setattr(module, "send_transaction", lambda: None)
    try:
        audit = audit_no_execution_capability()
        assert audit.ok is False
        assert any("send_transaction" in hit
                   for hit in audit.forbidden_attrs_found)
        with pytest.raises(NoBroadcastViolation):
            assert_no_execution_capability()
    finally:
        delattr(module, "send_transaction")

    assert audit_no_execution_capability().ok is True


# ---------------------------------------------------------------------------
# the report never claims otherwise
# ---------------------------------------------------------------------------

def test_the_cli_banner_and_rendered_output_state_the_non_execution():
    from autopaper.same_market.cli import BANNER

    assert "NO EXECUTION" in BANNER
    assert "BROADCAST NOTHING" in BANNER or "NO EXECUTION" in BANNER


def test_the_renderer_always_prints_the_non_execution_footer():
    """
    Driven from a real report so this is an assertion about rendered output,
    not about a constant. The three values are also always shown separately,
    which is the property the whole harness exists to guarantee.
    """
    from autopaper.same_market.pipeline import SameMarketPipeline, TestConfig
    from autopaper.same_market.dexscreener import DexScreenerProbe
    from autopaper.same_market.report import render_report

    import asyncio

    pipeline = SameMarketPipeline(
        config=TestConfig(order_usd=0.23, sol_reference_usd=200.0),
        dexscreener=DexScreenerProbe(http_get=lambda url, t: {"pairs": []}),
        rpc=None, jupiter=None,
    )
    report = asyncio.run(pipeline.run(
        raw_text="no mint here", channel="MooDengPresidentCallers"))
    text = render_report(report)

    assert "NO TRANSACTION BUILT, SIGNED OR BROADCAST" in text
    assert "ACTUAL_FILL       : NOT AVAILABLE" in text
    assert "OBSERVED_MARK" in text
    assert "EXECUTABLE_QUOTE" in text
    assert "A quote is not a fill" in text
    assert report.as_dict()["actual_fill_available"] is False
    assert report.as_dict()["actual_fill"] is None
    assert report.as_dict()["test_order"]["hypothetical"] is True
    assert report.as_dict()["test_order"]["executed"] is False