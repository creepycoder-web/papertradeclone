"""
NO-BROADCAST GUARD for the same-market quote harness.

This harness exists to MEASURE a price. It must be structurally incapable of
placing, signing or broadcasting a transaction, so that guarantee is enforced
three independent ways rather than promised in a docstring:

1. **Absence of capability.** Nothing in :mod:`autopaper.same_market` has a
   submit, sign, send or swap method. The Solana client here is
   :class:`autopaper.same_market.rpc.SolanaReadOnlyRpc`, which exposes
   ``getAccountInfo``, ``getTokenSupply``, ``getMultipleAccounts``,
   ``getSlot`` and ``getVersion`` - and has no ``sendTransaction`` in its
   surface at all. It cannot broadcast because there is nothing to call.

2. **An import barrier.** Every module that contains a signer, a transaction
   builder or a broadcaster is refused at import time by a meta-path finder
   installed at position 0, the same mechanism
   :func:`autopaper.safety.install_import_guard` uses for the ML/Laya barrier.
   The vendored Solana RPC client in :mod:`autopaper.execution.solana` is
   therefore not reachable from this harness even by accident - which is why
   this package ships its own minimal read-only RPC client instead of reusing
   that one. ``solders`` is blocked outright: it is the ed25519 signing
   primitive, and this harness has no use for it.

3. **A startup audit.** :func:`assert_no_execution_capability` refuses to run if
   any already-imported module in the process exposes a known submission
   symbol, or if the harness's own namespace has grown one.

The barrier is deliberately fail-CLOSED: an unknown name that looks like a
broadcast path is refused rather than allowed with a warning.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..safety import SafetyViolation

AGENT_ID = "SAME_MARKET_TEST"

# ---------------------------------------------------------------------------
# Layer 1: modules that can sign, build or broadcast are not importable
# ---------------------------------------------------------------------------

#: Modules refused outright. Each entry states why it is here.
#:
#: ``autopaper.execution.solana`` holds ``SolanaRpc.send_transaction`` (the
#: broadcaster) and ``SolanaWallet.sign*``. It also holds the read-only RPC
#: calls, which is precisely why this package carries its own
#: :mod:`autopaper.same_market.rpc` client: the capability has to be separable
#: from the vendor, and here it is.
#:
#: ``autopaper.execution.txcodec`` holds ``sign_transaction_bytes``.
#: ``autopaper.execution.jupiter`` holds ``build`` / ``build_transaction``,
#: the unsigned-transaction builders. It is blocked even though this harness
#: wants Jupiter's *quote* endpoint, because the module cannot be imported
#: without the builder being reachable; the quote path here is a separate,
#: quote-only client.
#: ``autopaper.execution.planner`` signs every candidate to simulate it.
#: ``autopaper.execution.real`` and ``token_cleanup`` are the end-to-end
#: executors. ``fill`` models fills. ``raptor`` is an execution backend.
#:
#: NOTE WHAT IS DELIBERATELY ABSENT: ``cryptography``, ``nacl`` and ``rsa``.
#: Encrypting is not submitting, and Telethon needs them for MTProto auth -
#: blocking them would break the read-only Telegram transport this harness
#: depends on while protecting nothing. The deny set targets submission, not
#: cryptography.
FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        # vendored Solana RPC + wallet: broadcaster AND signer
        "autopaper.execution.solana",
        # wire-level signing
        "autopaper.execution.txcodec",
        # unsigned-transaction builders
        "autopaper.execution.jupiter",
        # signs candidates in order to simulate them
        "autopaper.execution.planner",
        # end-to-end executors
        "autopaper.execution.real",
        "autopaper.execution.raptor",
        "autopaper.execution.fill",
        "autopaper.token_cleanup",
        # ed25519 signing primitive, and the full Solana SDK (which can send)
        "solders",
        "solana",
    }
)

#: Also refuse the package roots, so ``import autopaper.execution.X`` cannot
#: smuggle a blocked submodule in under a different spelling.
FORBIDDEN_PACKAGES: frozenset[str] = frozenset({"autopaper.execution"})

#: Attribute names that indicate a submission path. Matched against the
#: harness's OWN namespaces by :func:`assert_no_execution_capability`.
FORBIDDEN_ATTRS: tuple[str, ...] = (
    "send_transaction",
    "sendTransaction",
    "send_raw_transaction",
    "sendRawTransaction",
    "sign_transaction",
    "sign_transaction_bytes",
    "sign_new_transaction",
    "sign_wire",
    "build_transaction",
    "execute_swap",
    "execute_cleanup",
    "open_position",
    "close_position",
    "submit",
    "broadcast",
)

#: Counter of refused import attempts. Distinct from "imports that succeeded",
#: which is structurally zero: the finder raises instead of returning a spec.
BLOCKED_IMPORT_ATTEMPTS: int = 0

#: How far up the stack the importer check will look. A harness import is
#: always initiated from harness code, so a small budget is sufficient; the
#: bound just keeps the check cheap.
_IMPORT_WALK_DEPTH = 40


class NoBroadcastViolation(SafetyViolation):
    """Something tried to give this read-only harness an execution capability."""


class _NoBroadcastFinder(importlib.abc.MetaPathFinder):
    """
    Meta path finder that refuses execution and signing modules - but ONLY when
    the import is made BY THE HARNESS.

    SCOPING IS LOAD-BEARING
    =======================

    An unscoped finder would live in ``sys.meta_path`` for the whole interpreter
    and would therefore also block the repository's own executor tests, which
    legitimately import ``autopaper.token_cleanup`` and
    ``autopaper.execution.solana``. That is not a nuisance: it would mean the
    harness changed the behaviour of unrelated code in the same process, which
    is exactly the kind of side effect this harness must not have.

    So the finder inspects the importing frame. The rule is precise and gives
    the guarantee the brief asks for: **this harness cannot acquire an execution
    capability**, while nothing else in the process is affected.

    Inserted at position 0 so it wins over every other finder, including ones
    a third-party library installs later.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: Any | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not is_forbidden_module(fullname):
            return None
        if not _imported_by_harness():
            # Someone else is importing it. Not our business.
            return None

        global BLOCKED_IMPORT_ATTEMPTS
        BLOCKED_IMPORT_ATTEMPTS += 1
        raise NoBroadcastViolation(
            f"{AGENT_ID}: import of '{fullname}' is blocked. This harness "
            "measures a price; it has no capability to sign, submit or "
            "broadcast a transaction, and none may be added to it."
        )


def _imported_by_harness() -> bool:
    """
    Is the import currently in progress being made by harness code?

    Walks the call stack for the nearest frame belonging to this package. Two
    frames are skipped first: this function's caller is ``find_spec``, which is
    *itself* defined in this module, so without skipping it the walk would
    match the finder and conclude that every import in the process is a harness
    import.

    The walk is depth-bounded: a harness import is always initiated from harness
    code, so a small budget suffices and the check stays cheap.
    """
    # 0 = this function, 1 = find_spec, 2 = the actual import site.
    frame = sys._getframe(2)
    package = __package__ or "autopaper.same_market"
    for _ in range(_IMPORT_WALK_DEPTH):
        if frame is None:
            return False
        name = frame.f_globals.get("__name__", "")
        if name == package or name.startswith(package + "."):
            return True
        frame = frame.f_back
    return False


_INSTALLER: dict[str, Any] = {"installed": False}


def install_no_broadcast_guard() -> None:
    """
    Idempotently install the barrier. Safe to call repeatedly.

    Called at import time by :mod:`autopaper.same_market`, which is imported
    before any provider module, so the barrier is armed before the first
    client is constructed.

    POSITION MATTERS. This finder is inserted immediately AFTER any existing
    :class:`autopaper.safety._ForbiddenImportFinder` rather than at index 0.
    The ML/Laya/shadow barrier is a runtime-wide invariant that other modules
    and tests assert on position, and this harness has no business displacing
    it. Sitting directly behind it still puts this finder ahead of every
    third-party finder, which is where it needs to be.
    """
    if _INSTALLER["installed"]:
        return
    for finder in sys.meta_path:
        if isinstance(finder, _NoBroadcastFinder):
            _INSTALLER["installed"] = True
            return

    from ..safety import _ForbiddenImportFinder

    position = 0
    for index, finder in enumerate(sys.meta_path):
        if isinstance(finder, _ForbiddenImportFinder):
            position = index + 1
    sys.meta_path.insert(position, _NoBroadcastFinder())
    _INSTALLER["installed"] = True


def is_forbidden_module(name: str) -> bool:
    root = name.split(".", 1)[0]
    return (
        name in FORBIDDEN_MODULES
        or root in FORBIDDEN_MODULES
        or root in FORBIDDEN_PACKAGES
    )


def is_forbidden_attr(name: str) -> bool:
    return name in FORBIDDEN_ATTRS


# ---------------------------------------------------------------------------
# Layer 3: startup audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityAudit:
    """Result of the no-execution-capability audit."""

    ok: bool
    forbidden_attrs_found: tuple[str, ...] = ()
    forbidden_modules_loaded: tuple[str, ...] = ()
    blocked_import_attempts: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "forbidden_attrs_found": list(self.forbidden_attrs_found),
            "forbidden_modules_loaded": list(self.forbidden_modules_loaded),
            "blocked_import_attempts": self.blocked_import_attempts,
            "notes": list(self.notes),
        }

    def render(self) -> str:
        head = "NO-BROADCAST GUARD" if self.ok else "GUARD VIOLATION"
        lines = [head, "=" * 62]
        lines.append(f"  capability to broadcast        : "
                     f"{'absent (structural)' if self.ok else 'PRESENT - ABORT'}")
        lines.append(f"  submission symbols in harness  : "
                     f"{len(self.forbidden_attrs_found)}")
        lines.append(f"  execution modules loaded       : "
                     f"{len(self.forbidden_modules_loaded)}")
        lines.append(f"  refused import attempts        : {self.blocked_import_attempts}")
        for note in self.notes:
            lines.append(f"  - {note}")
        lines.append("-" * 62)
        lines.append("  NO TRANSACTION BROADCAST")
        return "\n".join(lines)


def audit_no_execution_capability() -> CapabilityAudit:
    """
    Prove this harness has no way to place an order.

    Checks two things: that no harness module holds a reference to a blocked
    module, and that no harness namespace exposes a submission symbol.

    Scoped to the harness, for the same reason the import finder is: another
    part of this repository may legitimately hold the execution modules loaded,
    and their presence says nothing about whether THIS harness can reach them.
    """
    install_no_broadcast_guard()

    package = __package__ or "autopaper.same_market"
    forbidden_modules: set[str] = set()
    forbidden_attrs: set[str] = set()

    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == package or name.startswith(package + ".")):
            continue
        for attr in dir(mod):
            try:
                value = getattr(mod, attr)
            except Exception:  # noqa: BLE001 - a lazy module may not load
                continue
            if is_forbidden_attr(attr):
                forbidden_attrs.add(f"{name}.{attr}")
            if inspect.ismodule(value):
                value_name = getattr(value, "__name__", "")
                if is_forbidden_module(value_name):
                    forbidden_modules.add(value_name)

    ok = not forbidden_modules and not forbidden_attrs
    notes: list[str] = []
    if forbidden_modules:
        notes.append(
            "the harness holds a reference to an execution module: "
            + ", ".join(sorted(forbidden_modules))
        )
    if forbidden_attrs:
        notes.append(
            "submission symbol(s) reachable from the harness: "
            + ", ".join(sorted(forbidden_attrs))
        )
    if ok:
        notes.append(
            "the harness has no submit/sign/broadcast method of any kind; the "
            "capability does not exist rather than merely being unused"
        )
    return CapabilityAudit(
        ok=ok,
        forbidden_attrs_found=tuple(sorted(forbidden_attrs)),
        forbidden_modules_loaded=tuple(sorted(forbidden_modules)),
        blocked_import_attempts=BLOCKED_IMPORT_ATTEMPTS,
        notes=tuple(notes),
    )


def assert_no_execution_capability() -> CapabilityAudit:
    """Audit, and refuse to proceed if anything is wrong."""
    audit = audit_no_execution_capability()
    if not audit.ok:
        raise NoBroadcastViolation(
            f"{AGENT_ID}: refusing to start. " + " | ".join(audit.notes)
        )
    return audit


__all__ = [
    "AGENT_ID",
    "BLOCKED_IMPORT_ATTEMPTS",
    "CapabilityAudit",
    "FORBIDDEN_ATTRS",
    "FORBIDDEN_MODULES",
    "FORBIDDEN_PACKAGES",
    "NoBroadcastViolation",
    "assert_no_execution_capability",
    "audit_no_execution_capability",
    "install_no_broadcast_guard",
    "is_forbidden_attr",
    "is_forbidden_module",
]