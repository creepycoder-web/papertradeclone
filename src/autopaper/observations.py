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

#: Rotated files. The stamp is zero-padded so lexical order is age order.
ROTATED_PREFIX = "observations-"
ROTATED_SUFFIX = ".jsonl"

#: Every file this module is allowed to delete. Narrow on purpose: see the
#: module docstring.
ROTATED_GLOB = f"{ROTATED_PREFIX}*{ROTATED_SUFFIX}"

_BYTES_PER_MB = 1024 * 1024


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

        ``context`` is merged into the row for fields the observation itself
        does not carry (which phase of the loop asked for it, for instance).
        Reserved keys are not overwritten: the observation's own values win, so
        a caller cannot relabel a recorded price.
        """
        if not self.config.enabled:
            return
        try:
            row = dict(context)
            row.update(observation.as_dict())
            row["mint"] = mint
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
        self._handle.write(line + "\n")
        self._handle.flush()
        # THE CAP IS ENFORCED ON ROTATION, NOT ON EVERY WRITE.
        #
        # ``total_bytes`` stats every file in the stream, so checking it per
        # observation is O(files) per observation -- on a long unattended run
        # with hundreds of rotated files that is real overhead on the path a
        # trading decision waits behind. The total can only jump by a whole
        # file at a rotation, and between rotations it can grow by at most
        # ``rotate_mb``, so rotation is the only moment the answer can change.
        if self.active_path.stat().st_size >= self._rotate_bytes:
            self._rotate()
            self._enforce_cap()

    @property
    def _rotate_bytes(self) -> int:
        return max(1, int(float(self.config.rotate_mb) * _BYTES_PER_MB))

    @property
    def _cap_bytes(self) -> int:
        return max(1, int(float(self.config.max_mb) * _BYTES_PER_MB))

    def _rotate(self) -> None:
        self.close()
        stamp = f"{int(self.clock.now() * 1000):020d}"
        target = self.directory / f"{ROTATED_PREFIX}{stamp}{ROTATED_SUFFIX}"
        suffix = 0
        while target.exists():
            suffix += 1
            target = (self.directory
                      / f"{ROTATED_PREFIX}{stamp}-{suffix:03d}{ROTATED_SUFFIX}")
        self.active_path.rename(target)

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

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
                self._handle.close()
            except OSError:
                pass
            self._handle = None

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "directory": str(self.directory),
            "written": self.written,
            "failed": self.failed,
            "rotated_files": len(self.rotated_paths()),
            "pruned_files": self.pruned_files,
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


__all__ = [
    "ACTIVE_NAME",
    "ROTATED_GLOB",
    "ROTATED_PREFIX",
    "ROTATED_SUFFIX",
    "ObservationStore",
    "store_for",
]
