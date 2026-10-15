"""
Durable PAPER state: written atomically, and rehydrated on restart.

TWO FAILURES THIS PREVENTS
--------------------------

**1. A kill mid-write leaving a corrupt canonical file.**
``open(path, "w")`` truncates first and writes second. A process killed in
between leaves a truncated file, and the next restart reads a half-written
dataset as if it were whole. Every canonical write here goes to a temp file in
the SAME directory and is moved into place with :func:`os.replace`, which is
atomic on POSIX. A reader therefore sees either the old file or the new one,
never a spliced one.

**2. A restart silently replacing history with an empty dataset.**
The in-memory records are the run's working set, not its history. Writing them
back over the top of the artefacts would erase every signal and trade from
before the restart -- the dataset would begin again at zero on each restart,
and a long capture would retain only its final session. So the store is
APPEND-AND-RECONCILE: records are keyed by a stable identity, existing keys are
kept, and only genuinely new records are added.

OPEN POSITIONS
--------------
A PAPER position that was open when the process stopped is rehydrated, because
otherwise its entry venue pin, its entry fill and its token balance would be
lost and the exit could no longer be quoted on the market it was opened
against. Rehydration is validated: a record that does not satisfy the schema is
refused with a reason rather than loaded half-formed.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Bumped when the on-disk shape changes in a way rehydration must notice.
STATE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# STATE ROOT RESOLUTION -- fail closed, never CWD-relative
# ---------------------------------------------------------------------------
#
# THE FAILURE THIS PREVENTS
# -------------------------
# ``PaperStateStore`` used to accept ``root=None`` and resolve it to
# ``Path("var") / filename`` -- RELATIVE TO THE PROCESS WORKING DIRECTORY. Any
# test, smoke run, research probe or synthetic experiment built without an
# explicit root therefore wrote into whatever ``var/`` happened to be under the
# current directory. Under pytest that is the repository, so a fixture position
# (mint ``APxxh2tWCh...``, symbol ``T``) landed in the operator's live resumable
# book; in the V2 FLOOR (BETA) experiment it created a phantom session
# ``beta_<now>`` holding synthetic open positions inside
# ``var/v2_floor_beta_state.json``.
#
# Those rows are shaped like real records, so nothing downstream could tell them
# apart from real trades. The repository already asserted the *specific* caller
# was fixed (tests/test_production_state_isolation.py); this makes the
# CLASS of mistake unrepresentable instead.
#
# THE CONTRACT
# ------------
# 1. ``root`` is REQUIRED. ``None`` raises :class:`StateRootError`. There is no
#    default, so omitting the argument cannot silently mean "the real thing".
# 2. ``""`` and whitespace raise. ``Path("")`` is ``Path(".")`` -- the working
#    directory -- which is the same ambiguity wearing a different hat.
# 3. The root is resolved to an ABSOLUTE path ONCE, at construction. Changing
#    the working directory afterwards cannot redirect a store that already
#    exists; every later read, write, quarantine and archive uses the resolved
#    target, not a re-resolved relative path.
# 4. A store whose target lands inside the canonical RUNTIME state root refuses
#    to construct unless the caller passes ``runtime=True``. Tests and probes
#    never pass it, so they cannot reach production state even by accident,
#    even through a symlink or a ``..`` traversal.

#: The canonical production PAPER state root: ``<repository>/var``. This is the
#: same base :func:`autopaper.config.build_config` uses when it is given no
#: root, so the two cannot drift.
RUNTIME_STATE_ROOT: Path = (
    Path(__file__).resolve().parents[3] / "var").resolve()


class StateRootError(RuntimeError):
    """
    A durable-state root was omitted, ambiguous, or aimed at production.

    A subclass of ``RuntimeError`` so a caller that only guards against store
    failures still catches it, and distinct so the message can say WHICH of the
    three rules was broken.
    """


def is_runtime_state_path(path: Path | str) -> bool:
    """
    Does ``path`` resolve INSIDE the canonical runtime state root?

    Resolution is done on BOTH sides, so a symlink or a ``..`` traversal that
    points at production state is recognised as production rather than sliding
    past the check as an innocent-looking path. A relative ``path`` is resolved
    against the current working directory first, which is exactly the ambiguity
    the resolution rules exist to remove -- so it is judged, not ignored.
    """
    try:
        candidate = Path(path).expanduser().resolve()
        root = RUNTIME_STATE_ROOT
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return candidate == root or candidate.is_relative_to(root)


def _resolve_state_root(root: object, *, runtime: bool) -> Path:
    """
    Validate and absolutise a state root. See the module contract above.

    Returns the resolved ABSOLUTE directory. Raises :class:`StateRootError`
    with a message that names the broken rule.
    """
    if root is None:
        raise StateRootError(
            "a durable PAPER state root is REQUIRED. Omitting it used to "
            f"resolve to {Path('var')}/ -- RELATIVE TO THE PROCESS WORKING "
            "DIRECTORY -- which is how a test or probe wrote synthetic "
            "sessions into the operator's real var/. Pass an explicit root: "
            "Config.var_dir for the runtime, or an isolated temporary "
            "directory for a test."
        )
    if isinstance(root, bool) or not isinstance(root, (str, Path)):
        raise StateRootError(
            f"state root must be a pathlib.Path or str, got "
            f"{type(root).__name__}. Refusing to guess a state directory "
            "from an arbitrary object."
        )
    text = str(root)
    if not text.strip():
        raise StateRootError(
            "an empty state root is ambiguous: Path('') is the process "
            "working directory, which is the exact hazard this check exists "
            "to remove. Pass an explicit directory."
        )
    try:
        resolved = Path(root).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise StateRootError(
            f"state root {root!r} could not be resolved: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if is_runtime_state_path(resolved) and not runtime:
        raise StateRootError(
            f"{resolved} is inside the RUNTIME PAPER state root "
            f"({RUNTIME_STATE_ROOT}). Only the runtime may write there, and it "
            "must say so explicitly (runtime=True). A test, smoke run, research "
            "probe or synthetic experiment must pass an isolated directory."
        )
    # ``Path(".")`` -- and anything else that lands on the working directory --
    # means "here", which is the same CWD dependence as the omitted-root case
    # wearing a different hat. It must be named, not inherited.
    try:
        if resolved == Path.cwd().resolve():
            raise StateRootError(
                f"state root {root!r} resolves to the process working "
                f"directory ({resolved}), so where the state lands would depend "
                "on where the process was started. Name the directory "
                "explicitly."
            )
    except OSError:
        pass
    return resolved


def atomic_write_text(path: Path, text: str) -> Path:
    """
    Write ``text`` to ``path`` atomically.

    The temp file is created in the destination directory so the final move is a
    same-filesystem rename. ``os.replace`` is atomic, so a concurrent reader --
    or a restart -- never observes a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent),
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    tmp = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave the temp file behind on a failure or a cancellation.
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path: Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, default=str))


@dataclass
class PersistedState:
    """
    The durable PAPER dataset.

    ``signals`` and ``trades`` are keyed lists: order is preserved for reading,
    but identity is what prevents a duplicate from becoming a second row.
    """

    schema_version: int = STATE_SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)
    updated_at: float = 0.0
    #: signal key -> signal record dict
    signals: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: trade key -> trade record dict
    trades: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: position key -> open position dict, for rehydration
    open_positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: trade key -> settlement that is EXCLUDED from the clean forward cohort
    #: because it was DELAYED (the arm was already unresolved when the process
    #: that settled it loaded it) rather than produced on the exit rule's
    #: schedule. A separate section, not a flag on ``trades``: a record in
    #: ``trades`` is in the cohort, and nothing that counts a cohort should have
    #: to remember to filter. Written only by the V2 FLOOR (BETA) forward mirror;
    #: empty for every other state file.
    incident_recovery: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Append-only note of what each write did, for the operator's audit trail.
    write_log: list[dict[str, Any]] = field(default_factory=list)

    # -- keys -------------------------------------------------------------

    @staticmethod
    def signal_key(record: Any) -> str:
        """Identity of a signal record: the signal plus the arm."""
        if isinstance(record, dict):
            arm = record.get("arm", "")
            sid = ((record.get("signal") or {}).get("signal_id")
                   or record.get("signal_id") or "")
        else:
            arm = record.arm
            sid = record.signal.signal_id
        return f"{sid}::{arm}"

    @staticmethod
    def trade_key(record: Any) -> str:
        """Identity of a trade: the position plus the arm plus the exit time.

        TWO RECORD SHAPES, ONE KEY SPACE
        --------------------------------
        The engine's own records nest the position under ``position`` and date
        it with ``exit_mark_timestamp``. The V2 FLOOR (BETA) experiment writes
        a FLAT record: ``arm`` and ``parent_signal_id`` at the top level,
        ``exit_at`` as its timestamp. Reading only the nested shape would give
        EVERY beta trade the key ``"::"``, so storing two beta trades would
        OVERWRITE the first and the restored bankroll would be wrong by one
        trade. Both shapes are accepted, and each produces a distinct key.
        """
        if isinstance(record, dict):
            position = record.get("position") or {}
            if position:
                arm = position.get("arm", "")
                sid = position.get("signal_id", "")
                exit_at = record.get("exit_mark_timestamp") or ""
            else:
                arm = record.get("arm", "")
                sid = (record.get("signal_id") or record.get("parent_signal_id")
                       or "")
                exit_at = (record.get("exit_mark_timestamp")
                           or record.get("exit_at") or "")
        else:
            arm = record.position.arm
            sid = record.position.signal_id
            exit_at = record.exit_mark_timestamp or ""
        return f"{sid}::{arm}::{exit_at}"

    # -- merge ------------------------------------------------------------

    def merge(
        self,
        *,
        signals: Iterable[Any] = (),
        trades: Iterable[Any] = (),
        open_positions: Iterable[Any] = (),
    ) -> dict[str, int]:
        """
        Add records that are new, keep everything already stored.

        Returns counts so the caller can report what a write actually changed,
        rather than assuming the in-memory set is the whole history.
        """
        added = {"signals": 0, "trades": 0, "positions": 0,
                 "signals_updated": 0, "trades_updated": 0}
        for record in signals:
            key = self.signal_key(record)
            payload = _as_dict(record)
            if key in self.signals:
                # A signal's verdict can legitimately change while it is fresh
                # (a quote arrives after the first refusal), so an existing key
                # is UPDATED, never duplicated.
                if self.signals[key] != payload:
                    self.signals[key] = payload
                    added["signals_updated"] += 1
            else:
                self.signals[key] = payload
                added["signals"] += 1
        for record in trades:
            key = self.trade_key(record)
            payload = _as_dict(record)
            if key in self.trades:
                if self.trades[key] != payload:
                    self.trades[key] = payload
                    added["trades_updated"] += 1
            else:
                self.trades[key] = payload
                added["trades"] += 1
        for position in open_positions:
            if isinstance(position, dict):
                key = (f"{position.get('arm', '')}::"
                       f"{position.get('signal_id', '')}")
            else:
                key = f"{position.arm}::{position.signal_id}"
            if key not in self.open_positions:
                self.open_positions[key] = _as_dict(position)
                added["positions"] += 1
        return added

    def close_positions(self, closed_keys: Iterable[str]) -> int:
        """Drop positions that have since closed, so restarts do not revive them."""
        removed = 0
        for key in list(closed_keys):
            if self.open_positions.pop(key, None) is not None:
                removed += 1
        return removed

    # -- serialisation ----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at or self.created_at,
            "counts": {
                "signals": len(self.signals),
                "trades": len(self.trades),
                "open_positions": len(self.open_positions),
                "incident_recovery": len(self.incident_recovery),
            },
            "signals": list(self.signals.values()),
            "trades": list(self.trades.values()),
            "open_positions": list(self.open_positions.values()),
            "incident_recovery": list(self.incident_recovery.values()),
            "write_log": self.write_log[-200:],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersistedState":
        state = cls(
            schema_version=int(payload.get("schema_version", 0) or 0),
            created_at=float(payload.get("created_at") or time.time()),
            updated_at=float(payload.get("updated_at") or 0.0),
        )
        for record in payload.get("signals") or ():
            key = state.signal_key(record)
            state.signals[key] = record
        for record in payload.get("trades") or ():
            key = state.trade_key(record)
            state.trades[key] = record
        for record in payload.get("open_positions") or ():
            arm = (record or {}).get("arm", "")
            sid = (record or {}).get("signal_id", "")
            state.open_positions[f"{arm}::{sid}"] = record
        for record in payload.get("incident_recovery") or ():
            state.incident_recovery[state.trade_key(record)] = record
        state.write_log = list(payload.get("write_log") or ())
        return state

    # -- validation -------------------------------------------------------

    def validate(self) -> tuple[bool, list[str]]:
        """
        Is this state usable, and if not, why.

        A restart refuses to half-load. Naming the problem is better than
        starting with half a position and discovering it at the exit.
        """
        problems: list[str] = []
        if self.schema_version != STATE_SCHEMA_VERSION:
            problems.append(
                f"schema_version {self.schema_version} does not match "
                f"{STATE_SCHEMA_VERSION}")
        for key, record in self.open_positions.items():
            if not isinstance(record, dict):
                problems.append(f"open position {key} is not an object")
                continue
            for field in ("arm", "signal_id", "mint", "tokens_raw"):
                if record.get(field) in (None, ""):
                    problems.append(
                        f"open position {key} is missing {field!r}")
            # The venue pin is what makes a rehydrated exit coherent. Without
            # it, an exit would have to guess, and a guessed venue is a
            # different trade.
            if not record.get("entry_venue_id"):
                problems.append(
                    f"open position {key} has no entry_venue_id, so its exit "
                    "cannot be pinned to the market it was opened on")
            entry_fill = record.get("entry_fill") or {}
            if not isinstance(entry_fill, dict) or not entry_fill.get(
                    "output_amount_raw"):
                problems.append(
                    f"open position {key} has no usable entry fill, so its "
                    "token balance is unknown")
        return (not problems), problems


class PaperStateStore:
    """
    The on-disk home of :class:`PersistedState`.

    ``root`` is REQUIRED and ``runtime`` defaults to ``False``. See the module
    docstring for why: an omitted root used to resolve to ``Path("var")``
    relative to the process working directory, which let any caller that forgot
    the argument write into the operator's live state. There is no default now,
    so that mistake raises instead of corrupting a dataset.

    Every path this store touches -- the state file, the ``.corrupt.json``
    quarantine written by :meth:`load`, and the ``.archive.json`` written by
    :meth:`archive_current` -- is derived from the ONE resolved root captured
    here, so reset, archival and restart recovery cannot disagree about which
    directory they are operating on.
    """

    def __init__(self, root: Path | str, *, filename: str = "paper_state.json",
                 runtime: bool = False):
        # Resolved ONCE, to an absolute path, and validated at the boundary.
        # Everything below uses ``self.root`` / ``self.path``, so a later
        # ``os.chdir`` cannot redirect a store that already exists.
        self.root: Path = _resolve_state_root(root, runtime=runtime)
        self.runtime = bool(runtime)
        if not isinstance(filename, str) or not filename.strip():
            raise StateRootError(
                f"state filename must be a non-empty str, got {filename!r}")
        # A filename must not be able to climb out of the resolved root.
        candidate = (self.root / filename)
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise StateRootError(
                f"state filename {filename!r} must be a plain name inside the "
                "state root; an absolute path or a '..' segment would let the "
                "store write outside the directory it was given")
        self.path: Path = candidate.resolve()

    def load(self) -> PersistedState:
        """
        Read the durable state, or start empty if there is none.

        A CORRUPT file is not silently replaced. It is moved aside and a fresh
        state begins, because overwriting the only copy of a dataset is the one
        unrecoverable outcome available here.
        """
        if not self.path.exists():
            return PersistedState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # Same root as the state file: the quarantine is a sibling, never
            # a path re-derived from somewhere else.
            quarantine = self.path.with_suffix(".corrupt.json")
            try:
                os.replace(self.path, quarantine)
            except OSError:
                pass
            state = PersistedState()
            state.write_log.append({
                "at": time.time(), "event": "load_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "quarantined_to": str(quarantine),
            })
            return state
        if not isinstance(payload, dict):
            return PersistedState()
        return PersistedState.from_dict(payload)

    def save(self, state: PersistedState, *, note: str = "") -> dict[str, Any]:
        """Write atomically and record what the write did."""
        state.updated_at = time.time()
        state.write_log.append({
            "at": state.updated_at, "event": "save", "note": note,
            "counts": {
                "signals": len(state.signals), "trades": len(state.trades),
                "open_positions": len(state.open_positions),
            },
        })
        atomic_write_json(self.path, state.as_dict())
        return {"path": str(self.path), "note": note,
                "signals": len(state.signals), "trades": len(state.trades),
                "open_positions": len(state.open_positions)}

    def rehydrate(
        self, state: PersistedState,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        The open positions a restart should resume, and anything refused.

        A position that cannot be validated is NOT loaded. Loading it would
        mean inventing a token balance or an exit venue, and both of those
        quietly become a trade that never happened.
        """
        ok, problems = state.validate()
        if not ok and any("entry_venue_id" in p or "entry fill" in p
                          for p in problems):
            return [], problems
        return [dict(p) for p in state.open_positions.values()], problems

    def archive_current(self) -> str | None:
        """
        Copy the current state file aside with a timestamped name, keeping the
        live file in place. Returns the archive path, or ``None`` if there was
        nothing to archive.

        Used by an explicit session reset so prior experiment history is NEVER
        silently deleted: the reset starts a fresh dataset, and the previous one
        is preserved under a timestamped name the operator can still read.
        """
        if not self.path.exists():
            return None
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        archive = self.path.with_name(f"{self.path.stem}.{stamp}.archive.json")
        n = 1
        while archive.exists():
            archive = self.path.with_name(
                f"{self.path.stem}.{stamp}.{n}.archive.json")
            n += 1
        try:
            archive.write_bytes(self.path.read_bytes())
        except OSError:
            return None
        return str(archive)


def _as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    as_dict = getattr(record, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    import dataclasses
    if dataclasses.is_dataclass(record):
        return dataclasses.asdict(record)
    return {"repr": repr(record)}


__all__ = [
    "PaperStateStore",
    "StateRootError",
    "RUNTIME_STATE_ROOT",
    "is_runtime_state_path",
    "PersistedState",
    "STATE_SCHEMA_VERSION",
    "atomic_write_json",
    "atomic_write_text",
]
