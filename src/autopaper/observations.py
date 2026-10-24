"""
Append-only capture of every market observation, for replay.

WHY THIS EXISTS
---------------
``replay`` can only be as honest as its input. If the recorded stream is a
summary, a sample, or a set of observations filtered by what the agent happened
to act on, then every replayed decision is conditioned on the live run's
choices and the comparison is worthless. So this records EVERY observation the
agent received -- for candidates it rejected as well as for positions it held --
at the one point they all pass through, ``MarketDataService._record``.

WHAT IT IS NOT
--------------
It is TELEMETRY. It never decides anything, and it is wired so it cannot:
every entry point is wrapped by the caller, a failure is counted and logged
rather than raised, and the trading loop keeps the observation regardless. A
capture layer that can refuse an observation the agent needs in order to decide
would be a worse bug than no capture at all.

DISK IS BOUNDED, AND ONLY OUR OWN FILES ARE PRUNED
--------------------------------------------------
The active file rotates at ``rotate_mb``. Total use is capped at ``max_mb``,
enforced by deleting the OLDEST ROTATED file first. Two properties are
load-bearing and both are tested:

* the ACTIVE file is never pruned, even when the cap alone is smaller than one
  file -- losing the observations currently being written is the one loss that
  cannot be recovered from the next rotation;
* only paths matching this module's own glob, inside its own directory, are
  ever considered. ``var/`` also holds the ledgers and the account state, and a
  disk-pressure cleanup that could reach those would be able to destroy the
  audit trail to save space. It cannot.

Rotated names carry a zero-padded millisecond stamp, so lexical order IS age
order and "oldest first" needs no stat call or parsing.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

from .clock import Clock, SystemClock
from .logging_setup import log

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import ObservationsConfig
    from .models import Observation

#: The file currently being appended to. Stable name so a tail or a replay can
#: find it without guessing.
ACTIVE_NAME = "observations.jsonl"

#: Stamped on every row. A future reader must be able to tell which writer
#: produced a row WITHOUT guessing from which keys happen to be present, so the
#: version travels with the data rather than with the code that reads it. Bump
#: it whenever a field's MEANING changes; adding a field does not need a bump,
#: because an older reader simply ignores it.
OBS_SCHEMA_VERSION = 1

#: Rotated files. The stamp is zero-padded so lexical order is age order, and
#: they are gzipped: a JSONL observation stream compresses about 10x, and the
#: cap counts the COMPRESSED size, so the same disk holds ~10x the history.
ROTATED_PREFIX = "observations-"
ROTATED_SUFFIX = ".jsonl.gz"

#: Every file this module is allowed to delete. Narrow on purpose: see the
#: module docstring.
ROTATED_GLOB = f"{ROTATED_PREFIX}*{ROTATED_SUFFIX}"

_BYTES_PER_MB = 1024 * 1024


def _fields_of(observation: "Observation") -> dict[str, Any]:
    """
    Every DECLARED field of an Observation, by name.

    Deliberately not ``Observation.as_dict()``: that is a reporting view and it
    omits ``volume_24h_usd``, so capturing through it dropped that field from
    every recorded row. Reading the dataclass fields keeps capture lossless and
    keeps a future field captured without anyone updating this module.
    """
    return {f.name: getattr(observation, f.name)
            for f in dataclasses.fields(observation)}


@dataclass
class ObservationStore:
    """
    One append-only observation stream, bounded on disk.

    Construct it with the directory it owns. It creates that directory lazily,
    on the first write, so a disabled store leaves no trace at all.
    """

    directory: Path
    config: "ObservationsConfig"
    clock: Clock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.clock = self.clock or SystemClock()
        #: Observations written, and observations lost to an error. Both are
        #: reported; a silent drop would make the stream look complete.
        self.written: int = 0
        self.failed: int = 0
        #: Rows written to the OS but not yet flushed to disk. Reported so a
        #: reader can tell "nothing recorded" from "recorded, not yet durable".
        self.buffered: int = 0
        self._last_flush: float = float(self.clock.now())
        #: Bytes handed to the active file since it was opened, counted in
        #: process. Rotation reads THIS rather than ``stat()``: a stat only
        #: sees flushed bytes, so checking it would have forced a flush on
        #: every row and defeated the buffering entirely.
        self._active_bytes: int = 0
        #: The candidate every row is attributed to while one is being handled.
        #: Set by the engine around a candidate's analysis; ``None`` for
        #: position monitoring, which legitimately has no candidate.
        self.current_candidate: Optional[str] = None
        #: Rotated files deleted to stay under the cap.
        self.pruned_files: int = 0
        #: True once the cap has actually bitten. The notifier warns on this.
        self.cap_reached: bool = False
        self._handle = None

    # -- paths -------------------------------------------------------------

    @property
    def active_path(self) -> Path:
        return self.directory / ACTIVE_NAME

    def rotated_paths(self) -> list[Path]:
        """Rotated files, OLDEST FIRST. Lexical order is age order by design."""
        if not self.directory.exists():
            return []
        return sorted(self.directory.glob(ROTATED_GLOB))

    def total_bytes(self) -> int:
        paths: Iterable[Path] = [*self.rotated_paths(), self.active_path]
        total = 0
        for path in paths:
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    # -- writing -----------------------------------------------------------

    def record(self, mint: str, observation: "Observation",
               **context: Any) -> None:
        """
        Append one observation. Never raises.

        ``candidate_id`` is what makes the stream replay-ready: it is the key a
        replay groups observations under, so a row without one cannot be
        attributed to a decision. Position monitoring legitimately has no
        candidate, and that case is written as an explicit ``null`` rather than
        by omitting the key -- every row keeps the same shape, and "no
        candidate" stays distinguishable from "the writer forgot".

        Any other ``context`` is merged in for fields the observation itself
        does not carry. Reserved keys are not overwritten: the observation's own
        values win, so a caller cannot relabel a recorded price.
        """
        if not self.config.enabled:
            return
        try:
            candidate_id = context.pop("candidate_id", self.current_candidate)
            row: dict[str, Any] = {"schema_version": OBS_SCHEMA_VERSION}
            row.update(context)
            # SERIALISED FROM THE DATACLASS FIELDS, NOT FROM ``as_dict()``.
            #
            # ``Observation.as_dict()`` is a REPORTING view and it omits
            # ``volume_24h_usd``. Capturing through it lost that field on every
            # row, so a replay would have read a different observation from the
            # one the live agent decided on -- silently, because the key was
            # simply absent rather than wrong. Reading the fields directly also
            # means a field added to ``Observation`` later is captured without
            # anyone remembering to update this function.
            row.update(_fields_of(observation))
            row["mint"] = mint
            row["candidate_id"] = candidate_id
            self._write(json.dumps(row, separators=(",", ":"), default=str))
            self.written += 1
        except Exception as exc:  # noqa: BLE001 - telemetry must not raise
            self.failed += 1
            log("ERROR", "observation capture failed (%s): %s",
                type(exc).__name__, exc)

    def _write(self, line: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self._handle is None:
            self._handle = self.active_path.open("a", encoding="utf-8")
        payload = line + "\n"
        self._handle.write(payload)
        self.buffered += 1
        self._active_bytes += len(payload.encode("utf-8"))
        # FLUSHED ON AN INTERVAL, NOT PER ROW.
        #
        # An unattended run records thousands of observations an hour, and
        # fsync-per-row on a small server's disk is latency the trading loop
        # waits behind for no gain: the stream is telemetry, and losing the
        # last few seconds of it to a kill -9 costs nothing a replay needs.
        # ``close()`` flushes, and so does the shutdown path, so an orderly
        # stop loses nothing at all.
        now = float(self.clock.now())
        if now - self._last_flush >= float(self.config.flush_s):
            self._flush()
            self._last_flush = now
        # THE CAP IS ENFORCED ON ROTATION, NOT ON EVERY WRITE.
        #
        # ``total_bytes`` stats every file in the stream, so checking it per
        # observation is O(files) per observation -- on a long unattended run
        # with hundreds of rotated files that is real overhead on the path a
        # trading decision waits behind. The total can only jump by a whole
        # file at a rotation, and between rotations it can grow by at most
        # ``rotate_mb``, so rotation is the only moment the answer can change.
        if self._active_bytes >= self._rotate_bytes:
            self._rotate()
            self._enforce_cap()

    @property
    def _rotate_bytes(self) -> int:
        return max(1, int(float(self.config.rotate_mb) * _BYTES_PER_MB))

    @property
    def _cap_bytes(self) -> int:
        return max(1, int(float(self.config.max_mb) * _BYTES_PER_MB))

    def _rotate(self) -> None:
        """
        Close the active file and gzip it under a stamped name.

        Compress-then-replace, never compress-in-place: the plain file is only
        unlinked once the ``.gz`` is complete, so a crash mid-rotation leaves
        either the original or the finished archive, never a half-written one
        that a reader would silently truncate.
        """
        self.close()
        stamp = f"{int(self.clock.now() * 1000):020d}"
        target = self.directory / f"{ROTATED_PREFIX}{stamp}{ROTATED_SUFFIX}"
        suffix = 0
        while target.exists():
            suffix += 1
            target = (self.directory
                      / f"{ROTATED_PREFIX}{stamp}-{suffix:03d}{ROTATED_SUFFIX}")
        partial = target.with_suffix(target.suffix + ".partial")
        try:
            with self.active_path.open("rb") as src, \
                    gzip.open(partial, "wb", compresslevel=6) as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            partial.replace(target)
            self.active_path.unlink()
        except OSError as exc:
            log("ERROR", "could not gzip %s: %s", self.active_path.name, exc)
            partial.unlink(missing_ok=True)
            # Fall back to an uncompressed rotation rather than losing the data
            # or wedging the active file at its rotation size forever.
            self.active_path.rename(target.with_suffix(""))
        finally:
            self._active_bytes = 0
            # Reopen immediately so the active file always exists after a
            # record. "The active file is never deleted" is an invariant the
            # prune logic depends on, and leaving a gap between rotations made
            # it true only by luck of where the last write landed.
            self.directory.mkdir(parents=True, exist_ok=True)
            self._handle = self.active_path.open("a", encoding="utf-8")

    def _enforce_cap(self) -> None:
        """
        Delete the OLDEST ROTATED files until the total fits the cap.

        The active file is excluded from the candidates, not from the total: it
        still counts against the cap, so a small cap simply prunes everything
        rotated rather than silently allowing unbounded growth.
        """
        cap = self._cap_bytes
        if self.total_bytes() <= cap:
            return
        self.cap_reached = True
        for path in self.rotated_paths():            # oldest first
            try:
                path.unlink()
                self.pruned_files += 1
            except OSError as exc:
                log("ERROR", "could not prune %s: %s", path.name, exc)
            if self.total_bytes() <= cap:
                return
        log("SYSTEM",
            "observation capture is at its %.0f MB cap with nothing left to "
            "prune; the active file is kept and capture continues",
            float(self.config.max_mb))

    def _flush(self) -> None:
        """Push buffered rows to the OS. Never raises; capture is telemetry."""
        if self._handle is None:
            return
        try:
            self._handle.flush()
            self.buffered = 0
        except OSError as exc:
            log("ERROR", "observation flush failed: %s", exc)

    def close(self) -> None:
        """Flush and release the handle. Called on shutdown, and on rotation."""
        if self._handle is not None:
            try:
                self._handle.flush()
                self._handle.close()
            except OSError:
                pass
            self._handle = None
        self.buffered = 0

    def attribute_to(self, candidate_id: Optional[str]) -> None:
        """
        Attribute subsequent rows to ``candidate_id``, or to nothing.

        The attribution lives here rather than being threaded through
        ``MarketDataService``: the service is the one choke point every
        observation passes through and it has no idea which candidate is being
        analysed, while the engine knows exactly. Passing ``None`` is how the
        engine says "back to position monitoring" -- an explicit state, not a
        forgotten one.
        """
        self.current_candidate = candidate_id

    def all_paths(self) -> list[Path]:
        """Every file in the stream, OLDEST FIRST, for a replay to read in order."""
        paths = self.rotated_paths()
        # A failed gzip rotation leaves a plain ``.jsonl`` sibling; include it.
        if self.directory.exists():
            paths += sorted(
                p for p in self.directory.glob(f"{ROTATED_PREFIX}*.jsonl")
                if p not in paths
            )
            paths = sorted(paths, key=lambda p: p.name)
        if self.active_path.exists():
            paths.append(self.active_path)
        return paths

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "directory": str(self.directory),
            "written": self.written,
            "failed": self.failed,
            "rotated_files": len(self.rotated_paths()),
            "pruned_files": self.pruned_files,
            "buffered": self.buffered,
            "schema_version": OBS_SCHEMA_VERSION,
            "total_mb": round(self.total_bytes() / _BYTES_PER_MB, 4),
            "max_mb": float(self.config.max_mb),
            "cap_reached": self.cap_reached,
        }


def store_for(config: Any, var_dir: Path, *,
              clock: Optional[Clock] = None) -> Optional[ObservationStore]:
    """
    The store for a runtime, or ``None`` when capture is off.

    ``None`` rather than a disabled store so the wiring reads the same in both
    cases and a caller cannot accidentally depend on the object existing.
    """
    obs_cfg = getattr(config, "observations", None)
    if obs_cfg is None or not getattr(obs_cfg, "enabled", False):
        return None
    return ObservationStore(Path(var_dir) / "observations", obs_cfg,
                            clock=clock or SystemClock())


def _open_text(path: Path):
    """Open a stream file, gzipped or not, as text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_observations(paths: Iterable[Path], *, report: bool = False):
    """
    Rebuild ``{candidate_id: ObservationSet}`` from recorded stream files.

    This is the function a replay reads history through, so its failure mode
    matters more than its speed.

    A MALFORMED LINE IS SKIPPED, NOT RAISED ON. The writer can be killed
    mid-row at any moment, so a truncated final line is the EXPECTED shape of a
    crash-interrupted session, not an exceptional one. A loader that raised
    would make every such session unreadable -- the opposite of durable. Skips
    are COUNTED and returned with ``report=True`` so a silent skip can never be
    mistaken for a clean read; ``datacheck`` surfaces the count.

    A missing path is skipped the same way: a pruned rotated file is a normal
    consequence of the disk cap, not an error.

    Rows are fed to ``ObservationSet`` which owns the ``as_of`` gate, so a
    replay reads recorded history through exactly the same no-lookahead path
    the live agent read live history through.
    """
    from .models import Observation, ObservationSet

    fields = {f.name for f in dataclasses.fields(Observation)}
    grouped: dict[Any, list[Observation]] = {}
    stats = {"files": 0, "rows": 0, "skipped_lines": 0, "missing_files": 0}

    for path in paths:
        path = Path(path)
        if not path.exists():
            stats["missing_files"] += 1
            continue
        stats["files"] += 1
        try:
            handle = _open_text(path)
        except OSError:
            stats["missing_files"] += 1
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, EOFError):
                    stats["skipped_lines"] += 1
                    continue
                if not isinstance(row, dict) or "observed_at" not in row:
                    stats["skipped_lines"] += 1
                    continue
                candidate = row.get("candidate_id")
                kwargs = {k: v for k, v in row.items() if k in fields}
                try:
                    obs = Observation(**kwargs)
                except (TypeError, ValueError):
                    stats["skipped_lines"] += 1
                    continue
                grouped.setdefault(candidate, []).append(obs)
                stats["rows"] += 1

    out = {
        key: ObservationSet(sorted(items, key=lambda o: o.observed_at))
        for key, items in grouped.items()
    }
    return (out, stats) if report else out


__all__ = [
    "ACTIVE_NAME",
    "OBS_SCHEMA_VERSION",
    "load_observations",
    "ROTATED_GLOB",
    "ROTATED_PREFIX",
    "ROTATED_SUFFIX",
    "ObservationStore",
    "store_for",
]
