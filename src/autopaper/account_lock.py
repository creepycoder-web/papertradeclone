"""
Single-writer ownership for a paper account.

WHY THIS EXISTS
---------------
``var/paper_account.json`` is written with ``save_json_atomic`` - a temp file plus
``os.replace``.  That makes each individual write atomic, and it makes the file
correct for whoever wrote it last.  It does nothing about two processes doing
read -> mutate -> save against the SAME account at the SAME time, which is
exactly what happened: the account and the audit trail disagreed by the whole P&L
of the second process, while the append-only ledger recorded both.

An atomic *write* is not an atomic *read-modify-write*.  This module supplies the
missing half.

THE LOCK
--------
``fcntl.flock(LOCK_EX | LOCK_NB)`` on a file whose name is derived from the
CANONICAL path of the account's state file.

* It is an operating-system exclusive lock, not a flag file.  Two processes
  starting at the same instant are arbitrated by the kernel; there is no
  check-then-write window to lose.
* It is released by the kernel when the process dies, however it dies - SIGKILL,
  power loss, panic.  A crash therefore cannot leave a lock that blocks a
  restart, and stale-lock cleanup is not needed and is deliberately NOT
  implemented, because "delete the lock file if it looks old" is precisely the
  unsafe pattern this replaces.
* The file is opened ``O_RDWR | O_CREAT`` and never unlinked while held, so the
  inode the kernel arbitrates on is stable.

The JSON written inside the lock file is DIAGNOSTIC ONLY.  Correctness comes
from ``flock`` alone; the contents are never trusted to decide anything.

FAIL-CLOSED
-----------
If ``fcntl`` is unavailable the engine refuses to start.  Running unlocked is
worse than not running.

SIGNAL HANDLERS
---------------
Deliberately none.  The engine installs its own ``SIGINT``/``SIGTERM`` handlers
via ``loop.add_signal_handler`` in ``PaperTrader.run``; a second registration
would silently replace them.  Release is handled by ``atexit`` for the normal
exit, and by the kernel for every abnormal exit, which is the case that matters.

ACCOUNT IDENTITY
----------------
The identity is ``sha256`` of ``os.path.realpath(state_file)``.  ``realpath``
resolves symlinks and ``..`` before hashing, so two path spellings of the same
account cannot produce two identities and therefore cannot produce two writers.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # POSIX
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - this project is POSIX-only
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False

#: Refusal string, used verbatim by the CLI and asserted by the tests.
REFUSAL = "REFUSED: ACCOUNT_ALREADY_LOCKED"

__all__ = [
    "REFUSAL",
    "AccountAlreadyLocked",
    "AccountLockUnavailable",
    "AccountOwnershipError",
    "AccountIdentity",
    "AccountLock",
    "acquire",
    "release",
    "is_held",
    "identity_for",
    "describe_holder",
    "held_account_ids",
]


class AccountAlreadyLocked(RuntimeError):
    """Another process already owns this account. Raised BEFORE any mutation."""

    def __init__(self, message: str, *, identity: "AccountIdentity | None" = None,
                 holder: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.identity = identity
        self.holder = holder or {}


class AccountLockUnavailable(RuntimeError):
    """The platform cannot provide an OS-level lock. Refuse rather than run unlocked."""


class AccountOwnershipError(RuntimeError):
    """An engine tried to mutate an account it does not own."""


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountIdentity:
    """The protected resource, named canonically."""

    account_id: str
    label: str
    state_path: Path
    lock_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "label": self.label,
            "state_path": str(self.state_path),
            "lock_path": str(self.lock_path),
        }


def identity_for(state_file: os.PathLike | str) -> AccountIdentity:
    """
    Canonical identity for a state file.

    The lock file sits beside the state file and is named from the same hash, so
    any alias of the account - symlink, ``..`` segment, trailing slash, relative
    path - resolves to the same lock inode.
    """
    state = Path(os.path.realpath(str(state_file)))
    digest = hashlib.sha256(str(state).encode("utf-8")).hexdigest()[:16]
    return AccountIdentity(
        account_id=f"acct_{digest}",
        label=f"paper:{state.parent.name}/{state.name}",
        state_path=state,
        lock_path=state.with_name(state.name + f".{digest}.lock"),
    )


# ---------------------------------------------------------------------------
# the lock
# ---------------------------------------------------------------------------


@dataclass
class AccountLock:
    account_id: str
    identity: AccountIdentity
    fd: int | None = None
    released: bool = False
    _refs: int = 1
    acquired_at: float = field(default_factory=time.time)

    def release(self) -> None:
        """Idempotent.  Never deletes the lock file: the inode must stay put."""
        if self.released or self.fd is None:
            self.released = True
            return
        try:
            os.fsync(self.fd)
        except OSError:
            pass
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.fd = None
        self.released = True

    def describe(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "label": self.identity.label,
            "state_path": str(self.identity.state_path),
            "lock_path": str(self.identity.lock_path),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": self.acquired_at,
            "refs": self._refs,
        }


# Per-process registry.  Keyed by account_id: one PROCESS owns an account, and
# may open several engines against it (that is what a restart-in-process looks
# like, and what the existing test suite does).
_HELD: dict[str, AccountLock] = {}


def held_account_ids() -> list[str]:
    return sorted(_HELD)


def acquire(state_file: os.PathLike | str, *, kind: str = "engine",
            require: bool = True) -> AccountLock | None:
    """
    Take exclusive ownership of the account backing ``state_file``.

    Returns the lock.  If this process already owns the account the existing
    lock is returned with its refcount raised, so an in-process restart is
    legal and a second PROCESS is not.

    Raises ``AccountAlreadyLocked`` when another process holds it and
    ``AccountLockUnavailable`` when the platform has no ``flock``.
    """
    if not _HAVE_FCNTL:  # fail closed
        raise AccountLockUnavailable(
            "fcntl is unavailable on this platform; refusing to run without an "
            "OS-level single-writer lock. This agent is PAPER ONLY and will not "
            "trade unlocked."
        )

    ident = identity_for(state_file)
    existing = _HELD.get(ident.account_id)
    if existing is not None and not existing.released:
        existing._refs += 1
        return existing

    ident.lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(ident.lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
            holder = describe_holder(ident.lock_path)
            who = f" pid={holder.get('pid')}" if holder.get("pid") else ""
            raise AccountAlreadyLocked(
                f"{REFUSAL} account {ident.account_id} "
                f"({ident.label}) is already owned by another process{who}",
                identity=ident, holder=holder,
            ) from exc
        raise

    lock = AccountLock(account_id=ident.account_id, identity=ident, fd=fd)
    payload = json.dumps({**lock.describe(), "kind": kind}, sort_keys=True)
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload.encode("utf-8") + b"\n")
        os.fsync(fd)
    except OSError:
        # The lock is already held; failing to annotate it is not fatal.
        pass
    _HELD[ident.account_id] = lock
    return lock


def release(lock: AccountLock | None) -> None:
    """Drop one reference; the last one actually unlocks."""
    if lock is None or lock.released:
        return
    lock._refs -= 1
    if lock._refs > 0:
        return
    lock.release()
    _HELD.pop(lock.account_id, None)


def is_held(state_file: os.PathLike | str) -> bool:
    """True when THIS process owns the account."""
    return not _HELD.get(identity_for(state_file).account_id, AccountLock("", identity_for(state_file))).released


def describe_holder(lock_path: os.PathLike | str) -> dict[str, Any]:
    """
    Read the diagnostic metadata.  Never decides anything - it is only ever used
    to make a refusal message useful.  An unreadable or partial file yields {}.
    """
    try:
        raw = Path(str(lock_path)).read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw.strip() or "{}")
    except json.JSONDecodeError:
        return {"unparsed": raw[:200]}
    return data if isinstance(data, dict) else {}


def pid_is_alive(pid: int | None) -> bool:
    """Diagnostic only.  Used to render a refusal, never to grant ownership."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# legacy adoption gate
# ---------------------------------------------------------------------------


class LegacyWriterSuspected(RuntimeError):
    """
    A pre-lock process may still be writing this account.

    The flock can only arbitrate between processes that take it.  A process
    started before this module existed never took one, so against the kernel the
    account looks free while something else is still writing it.  There is no
    non-heuristic way to detect that from outside - no heartbeat, no PID file the
    legacy process maintains.

    So this gate WITHHOLDS ownership when the state file is demonstrably in
    flux, and the operator can adopt it once.  It is deliberately asymmetric:

      * it can only keep a process OUT, never let one in, so it cannot become a
        second synchronisation mechanism racing the flock;
      * it expires on its own once the old writer stops touching the file, so
        it cannot deadlock a restart;
      * it is advisory in the sense that it never grants ownership, which only
        ``flock`` does.
    """

    def __init__(self, message: str, *, age_s: float, adopt_hint: str) -> None:
        super().__init__(message)
        self.age_s = age_s
        self.adopt_hint = adopt_hint


def assert_no_legacy_writer(state_file: os.PathLike | str, *, grace_s: float = 30.0,
                            allow_legacy_adopt: bool = False) -> dict[str, Any]:
    """
    Refuse to arm an account that a pre-lock process may still be writing.

    Returns a small report describing what was checked.  Silent when the account
    is already stamped by a lock-holding process, which is the steady state
    after the first restart.
    """
    path = Path(os.path.realpath(str(state_file)))
    report = {
        "state_path": str(path),
        "stamped": False,
        "age_s": None,
        "legacy_writer_suspected": False,
        "adopt_hint": "--adopt-legacy-account",
    }
    if allow_legacy_adopt:
        report["adopted"] = True
        return report
    if not path.exists():
        return report

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        report["stamped"] = bool(isinstance(raw, dict) and raw.get("account_id"))
    except (OSError, json.JSONDecodeError):
        # unreadable state is a separate problem; do not conflate it with this
        return report
    if report["stamped"]:
        return report

    try:
        age = max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return report
    report["age_s"] = round(age, 3)
    if age <= grace_s:
        report["legacy_writer_suspected"] = True
        raise LegacyWriterSuspected(
            f"REFUSED: LEGACY_WRITER_SUSPECTED for {path}\n"
            f"  the state file was last written {age:.1f}s ago and carries no "
            f"account_id stamp, which means it has never been written by a "
            f"process holding the single-writer lock.\n"
            f"  a process started before the lock existed cannot be arbitrated "
            f"by flock, so starting now would recreate the original race.\n"
            f"  wait {grace_s:.0f}s and retry once the old writer has stopped, or "
            f"stop it and start again, or start with {report['adopt_hint']} to "
            f"adopt this account into the locked regime explicitly.",
            age_s=age,
            adopt_hint="--adopt-legacy-account",
        )
    return report


@atexit.register
def _release_all_at_exit() -> None:  # pragma: no cover - process teardown
    for lock in list(_HELD.values()):
        try:
            lock.release()
        except Exception:
            pass
    _HELD.clear()
