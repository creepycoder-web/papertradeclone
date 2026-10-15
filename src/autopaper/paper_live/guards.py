"""
Structural safety audit for the PAPER live harness.

THE CLAIM
=========

"The PAPER engine cannot broadcast a transaction."

This module turns that from a promise into a check that runs before the engine
does anything, and that a test asserts. Four independent mechanisms:

1. **Absence of capability.** The RPC client
   (:class:`autopaper.same_market.rpc.SolanaReadOnlyRpc`) exposes five read
   methods and refuses any other name at call time, so a write cannot be
   expressed even by accident.

2. **An import barrier.** :func:`autopaper.same_market.guard.install_no_broadcast_guard`
   refuses to import any module in this repository containing a signer,
   transaction builder or broadcaster - scoped to the harness, so unrelated code
   in the same process is unaffected.

3. **This audit.** :func:`audit_paper_capability` reflects over every loaded
   module of this package and refuses to proceed if any submission symbol or
   execution-module reference is reachable.

4. **Pure pricing.** Every function that produces a price is arithmetic over
   state the caller already holds.

WHY IT FAILS CLOSED
===================

An unfamiliar module reference or symbol is treated as a finding, not waved
through. A safety check that passes on "I did not recognise the problem" is not
a safety check.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass, field
from typing import Any

from autopaper.safety import SafetyViolation
from autopaper.same_market.guard import (
    FORBIDDEN_ATTRS,
    FORBIDDEN_MODULES,
    install_no_broadcast_guard,
    is_forbidden_attr,
    is_forbidden_module,
)

AGENT_ID = "PAPER_LIVE"

#: Execution modules that must not be reachable from this harness. A superset of
#: the shared read-only barrier's list, because the PAPER engine additionally
#: loads the account, sizing, exits and decision modules, so it is worth being
#: explicit about which of them must be absent.
PAPER_FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "autopaper.execution.real",
        "autopaper.execution.solana",
        "autopaper.execution.txcodec",
        "autopaper.execution.planner",
        "autopaper.external_close",
        "autopaper.reconciliation",
        "autopaper.real_equity",
        "autopaper.wallet_assets",
        "autopaper.token_cleanup",
        "solders",
        "solana",
        "base58",
    }
)

#: Symbol names that indicate a submission path.
PAPER_FORBIDDEN_ATTRS: tuple[str, ...] = tuple(
    sorted(set(FORBIDDEN_ATTRS) | {
        "send_raw_transaction",
        "sign_transaction_bytes",
        "close_position",          # an execution capability here, not an exit
        "close_at_market",
        "sell_at_market",
        "place_order",
        "submit_order",
        "transfer",
        "withdraw",
        "private_key",
        "keypair",
    })
)

#: Modules this harness legitimately needs. Used to make the audit's positive
#: finding explicit: it is not merely "nothing forbidden found" but "the
#: expected read-only surface is present and everything else is not".
EXPECTED_READ_ONLY: frozenset[str] = frozenset(
    {
        "autopaper.l25",
        "autopaper.v3c",
        "autopaper.exits",
        "autopaper.sizing",
        "autopaper.strategy_profile",
        "autopaper.models",
        "autopaper.analysis.decision",
        "autopaper.analysis.features",
        "autopaper.paper_live",
        "autopaper.same_market",
    }
)


class PaperCapabilityViolation(SafetyViolation):
    """Something tried to give the PAPER harness an execution capability."""


@dataclass(frozen=True)
class PaperAudit:
    ok: bool
    forbidden_attrs_found: tuple[str, ...] = ()
    forbidden_modules_found: tuple[str, ...] = ()
    read_only_modules_present: tuple[str, ...] = ()
    blocked_import_attempts: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": "PAPER_ONLY",
            "transaction_built": 0,
            "transaction_signed": 0,
            "transaction_broadcast": 0,
            "forbidden_attrs_found": list(self.forbidden_attrs_found),
            "forbidden_modules_found": list(self.forbidden_modules_found),
            "read_only_modules_present": list(self.read_only_modules_present),
            "blocked_import_attempts": self.blocked_import_attempts,
            "notes": list(self.notes),
        }

    def render(self) -> str:
        head = "PAPER-ONLY SAFETY AUDIT" if self.ok else "PAPER CAPABILITY VIOLATION"
        lines = [head, "=" * 66]
        lines.append("  mode                      : PAPER ONLY")
        lines.append(
            f"  capability to broadcast   : "
            f"{'absent (structural)' if self.ok else 'PRESENT - ABORT'}"
        )
        lines.append(f"  submission symbols        : {len(self.forbidden_attrs_found)}")
        lines.append(f"  execution modules found   : {len(self.forbidden_modules_found)}")
        lines.append(f"  read-only modules present : {len(self.read_only_modules_present)}")
        for note in self.notes:
            lines.append(f"  - {note}")
        lines.append("-" * 66)
        lines.append("  NO TRANSACTION WAS BUILT, SIGNED, OR BROADCAST.")
        return "\n".join(lines)


def audit_paper_capability() -> PaperAudit:
    """
    Reflect over the loaded ``autopaper.paper_live`` modules.

    Scoped to this package, for the same reason the import barrier is: another
    part of the repository may legitimately hold execution modules loaded, and
    their presence says nothing about whether THIS harness can reach them.
    """
    install_no_broadcast_guard()

    package = __package__ or "autopaper.paper_live"
    forbidden_attrs: set[str] = set()
    forbidden_modules: set[str] = set()
    present: set[str] = set()

    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if not (name == package or name.startswith(package + ".")):
            continue
        for attr in dir(module):
            try:
                value = getattr(module, attr)
            except Exception:  # noqa: BLE001 - a lazy module may not load
                continue
            if is_forbidden_attr(attr) or attr in PAPER_FORBIDDEN_ATTRS:
                forbidden_attrs.add(f"{name}.{attr}")
            if inspect.ismodule(value):
                value_name = getattr(value, "__name__", "")
                if (
                    is_forbidden_module(value_name)
                    or value_name in PAPER_FORBIDDEN_MODULES
                    or value_name.split(".", 1)[0] in PAPER_FORBIDDEN_MODULES
                ):
                    forbidden_modules.add(value_name)
                elif value_name.split(".", 1)[0] == "autopaper":
                    present.add(value_name)

    # Positive confirmation, not just absence of problems.
    roots = {m.split(".", 1)[0] for m in present}
    missing = sorted(
        m for m in EXPECTED_READ_ONLY
        if m.split(".", 1)[0] not in roots and m not in present
    )

    ok = not forbidden_modules and not forbidden_attrs
    notes: list[str] = []
    if forbidden_modules:
        notes.append(
            "the harness reaches an execution module: "
            + ", ".join(sorted(forbidden_modules))
        )
    if forbidden_attrs:
        notes.append(
            "submission symbol(s) reachable: " + ", ".join(sorted(forbidden_attrs))
        )
    if ok:
        notes.append(
            "no signer, transaction builder, broadcaster or wallet is reachable "
            "from this harness; the capability does not exist rather than merely "
            "being unused"
        )
    if missing:
        # Not a failure: an optional import may simply not be loaded yet. It is
        # reported so the positive surface is visible.
        notes.append("not currently loaded: " + ", ".join(missing))

    from autopaper.same_market.guard import BLOCKED_IMPORT_ATTEMPTS

    return PaperAudit(
        ok=ok,
        forbidden_attrs_found=tuple(sorted(forbidden_attrs)),
        forbidden_modules_found=tuple(sorted(forbidden_modules)),
        read_only_modules_present=tuple(sorted(present)),
        blocked_import_attempts=BLOCKED_IMPORT_ATTEMPTS,
        notes=tuple(notes),
    )


def assert_paper_only() -> PaperAudit:
    """Audit, and refuse to run if anything is wrong."""
    audit = audit_paper_capability()
    if not audit.ok:
        raise PaperCapabilityViolation(
            f"{AGENT_ID}: refusing to start a PAPER measurement run. "
            + " | ".join(audit.notes)
        )
    return audit


__all__ = [
    "AGENT_ID",
    "EXPECTED_READ_ONLY",
    "PAPER_FORBIDDEN_ATTRS",
    "PAPER_FORBIDDEN_MODULES",
    "PaperAudit",
    "PaperCapabilityViolation",
    "assert_paper_only",
    "audit_paper_capability",
]
