"""
Forward PAPER validation: capture, separation, and honest status.

WHY THIS EXISTS
---------------
L25's 0.9647 threshold was derived from a 276-event research universe. Those
events are IN-SAMPLE by construction and can never validate the strategy. The
only data that can is data the runtime collected prospectively, after L25 was
frozen and deployed, while actually gating on it.

The three datasets are kept in separate files and are never summed:

``IN_SAMPLE``
    The research universe the threshold was derived from. Never prospective.

``POST_DERIVATION_RECONSTRUCTED``
    Pre-L25 runtime journal rows, re-scored offline against the frozen L25 gate.
    Genuinely out-of-sample with respect to the *threshold*, but the runtime did
    not gate on L25 when they were collected, so the execution path they exercise
    is not the one now deployed. Useful as a sanity check; never as validation.

``PROSPECTIVE_PAPER``
    Written only by :class:`ForwardRecorder`, only after a deployment boundary
    exists, only from the live decision path. This is the only dataset that
    validates the deployed pipeline.

WHY THE BOUNDARY IS IMMUTABLE
-----------------------------
Moving a deployment boundary is how historical data gets laundered into a
prospective record. :func:`load_or_create_deployment` refuses to rewrite an
existing boundary, and every record carries the deployment id it was written
under, so a report can detect records that predate or postdate it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# dataset classes
# ---------------------------------------------------------------------------


class Dataset(str, Enum):
    """Which universe a record belongs to. Never merged."""

    IN_SAMPLE = "IN_SAMPLE"
    """The research universe the threshold was derived from."""

    POST_DERIVATION_RECONSTRUCTED = "POST_DERIVATION_RECONSTRUCTED"
    """Pre-L25 journal rows re-scored offline against the frozen gate."""

    PROSPECTIVE_PAPER = "PROSPECTIVE_PAPER"
    """Collected by the live runtime, after deployment, gating on L25."""


#: Only this dataset may be used to claim validation.
VALIDATING = Dataset.PROSPECTIVE_PAPER

DATASETS = tuple(Dataset)


# ---------------------------------------------------------------------------
# status vocabulary
# ---------------------------------------------------------------------------


class Status(str, Enum):
    """
    The only statuses a forward report may emit.

    Chosen a priori and never tuned against results. ``CONSISTENT`` is
    deliberately hard to reach: it needs a sample large enough for the
    confidence interval to exclude zero, which at paper-trade dispersion takes
    far more trades than the early gates.
    """

    NO_DEPLOYMENT = "NO_DEPLOYMENT"
    """No deployment boundary exists. Nothing prospective can exist."""

    NOT_ENOUGH_DATA = "NOT_ENOUGH_DATA"
    """Too few closed prospective trades to say anything."""

    DATA_QUALITY_PROBLEM = "DATA_QUALITY_PROBLEM"
    """The prospective data itself is untrustworthy. Read nothing from it."""

    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    """Enough to look, not enough to conclude."""

    EARLY_SIGNAL = "EARLY_SIGNAL"
    """Directionally informative and not yet conclusive."""

    CONSISTENT = "CONSISTENT"
    """The pre-declared criteria are met on a clean, adequately sized sample."""

    DEGRADED = "DEGRADED"
    """Adequately sized and clean, but the criteria are not met."""


#: No status other than these may claim validation.
NON_VALIDATING_STATUSES = frozenset({
    Status.NO_DEPLOYMENT, Status.NOT_ENOUGH_DATA, Status.DATA_QUALITY_PROBLEM,
    Status.INSUFFICIENT_SAMPLE, Status.EARLY_SIGNAL, Status.DEGRADED,
})


# ---------------------------------------------------------------------------
# pre-declared validation policy
# ---------------------------------------------------------------------------

#: Closed prospective trades below which no performance statement is made at all.
MIN_CLOSED_FOR_ANY_READ = 30

#: Closed prospective trades below which a read is reported but no conclusion.
MIN_CLOSED_FOR_CONCLUSION = 100

#: Deterministic bootstrap draws. A fixed seed makes the report reproducible.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20240917

#: Confidence level for the expectancy interval.
CI_LEVEL = 0.95

#: Maximum acceptable peak-to-trough drawdown, as a fraction of equity.
MAX_DRAWDOWN_FRACTION = 0.15


@dataclass(frozen=True)
class ValidationPolicy:
    """
    The pre-declared rules, in one place.

    These are REPORTING gates, not strategy parameters: they decide what may be
    said about a result, never whether a trade is taken. Changing one is a
    change to the evaluation standard and is recorded in the report.
    """

    min_closed_for_any_read: int = MIN_CLOSED_FOR_ANY_READ
    min_closed_for_conclusion: int = MIN_CLOSED_FOR_CONCLUSION
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    ci_level: float = CI_LEVEL
    max_drawdown_fraction: float = MAX_DRAWDOWN_FRACTION
    require_ci_excludes_zero: bool = True
    require_positive_expectancy: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_closed_for_any_read": self.min_closed_for_any_read,
            "min_closed_for_conclusion": self.min_closed_for_conclusion,
            "bootstrap_draws": self.bootstrap_draws,
            "bootstrap_seed": self.bootstrap_seed,
            "ci_level": self.ci_level,
            "max_drawdown_fraction": self.max_drawdown_fraction,
            "require_ci_excludes_zero": self.require_ci_excludes_zero,
            "require_positive_expectancy": self.require_positive_expectancy,
        }


DEFAULT_POLICY = ValidationPolicy()


# ---------------------------------------------------------------------------
# required fields
# ---------------------------------------------------------------------------

#: Fields every prospective candidate must carry, or the record is flagged.
REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id", "at", "mint", "route", "market_cap_usd",
    "liquidity_usd", "liq_to_mcap", "signal_quality",
)

#: Fields every prospective decision must carry.
REQUIRED_DECISION_FIELDS = (
    "candidate_id", "at", "decision", "route", "signal_quality", "l25",
)

#: Fields every prospective trade must carry.
REQUIRED_TRADE_FIELDS = (
    "trade_id", "at", "mint", "return_pct", "net_pnl_usd", "holding_s",
)


class DataQualityError(RuntimeError):
    """A prospective record is missing something a verdict would depend on."""


# ---------------------------------------------------------------------------
# deployment boundary
# ---------------------------------------------------------------------------


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class Deployment:
    """
    The instant prospective collection began. Written once.

    ``started_at`` is the boundary: records at or after it are prospective, and
    anything earlier is not, regardless of which file it sits in.
    """

    deployment_id: str
    started_at: float
    strategy: str
    l25_min_liq_to_mcap: float
    config_source: str
    #: Hash of the FROZEN STRATEGY DEFINITION (L25_HASH), not of the source
    #: tree: what matters for validation is which rule produced the data.
    #: Named for what it holds -- it was previously "source_tree_sha256",
    #: which claimed to be the source tree while holding the definition hash.
    strategy_hash: str
    created_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "started_at": self.started_at,
            "strategy": self.strategy,
            "l25_min_liq_to_mcap": self.l25_min_liq_to_mcap,
            "config_source": self.config_source,
            "strategy_hash": self.strategy_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Deployment":
        return cls(
            deployment_id=d["deployment_id"],
            started_at=float(d["started_at"]),
            strategy=d["strategy"],
            l25_min_liq_to_mcap=float(d["l25_min_liq_to_mcap"]),
            config_source=d.get("config_source", "unknown"),
            strategy_hash=d.get("strategy_hash", d.get("source_tree_sha256", "")),
            created_at=float(d.get("created_at", 0.0)),
        )


def _deployment_id(strategy: str, threshold: float, strategy_hash: str) -> str:
    raw = f"{strategy}|{threshold!r}|{strategy_hash}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_or_create_deployment(
    root: Path | str,
    *,
    strategy: str,
    l25_min_liq_to_mcap: float,
    config_source: str = "unknown",
    strategy_hash: str = "",
    clock=_now,
    force_start_at: Optional[float] = None,
) -> Deployment:
    """
    Return the deployment boundary, creating it once if absent.

    If one already exists it is returned UNCHANGED, even if the arguments differ.
    Re-arming a boundary is the mechanism by which pre-deployment data would be
    relabelled as prospective, so it is refused rather than offered.

    ``force_start_at`` exists only for tests that need a boundary in the past.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "deployment.json"
    if path.exists():
        return Deployment.from_dict(json.loads(path.read_text()))

    started = float(force_start_at) if force_start_at is not None else float(clock())
    dep = Deployment(
        deployment_id=_deployment_id(strategy, l25_min_liq_to_mcap, strategy_hash),
        started_at=started,
        strategy=strategy,
        l25_min_liq_to_mcap=float(l25_min_liq_to_mcap),
        config_source=config_source,
        strategy_hash=strategy_hash,
        created_at=float(clock()),
    )
    _atomic_write_json(path, dep.as_dict())
    return dep


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record. Durable: flushed and fsynced before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, default=str) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# prospective capture
# ---------------------------------------------------------------------------


def _check_fields(record: Mapping[str, Any], required: Sequence[str]) -> list[str]:
    """
    Names of required fields that are absent or null.

    Absent and null are treated the same: a null is a hole in the evidence, not
    a value. Nothing is defaulted, because a defaulted required field silently
    becomes a plausible-looking number in a report.
    """
    return [f for f in required if record.get(f) is None]


class ForwardRecorder:
    """
    Writes PROSPECTIVE records only, stamped with the deployment boundary.

    Used by the live engine. It will not write before a deployment exists, and
    will not write a record whose timestamp precedes the boundary, so a
    mis-wired call site produces a refusal rather than a mislabelled record.
    """

    def __init__(self, root: Path | str, deployment: Deployment) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.deployment = deployment

    # -- paths -------------------------------------------------------------

    @property
    def candidates_path(self) -> Path:
        return self.root / "candidates.jsonl"

    @property
    def decisions_path(self) -> Path:
        return self.root / "decisions.jsonl"

    @property
    def trades_path(self) -> Path:
        return self.root / "trades.jsonl"

    # -- stamping ----------------------------------------------------------

    def _stamp(self, record: Mapping[str, Any], kind: str) -> dict[str, Any]:
        out = dict(record)
        out["dataset"] = Dataset.PROSPECTIVE_PAPER.value
        out["kind"] = kind
        out["deployment_id"] = self.deployment.deployment_id
        out["strategy"] = self.deployment.strategy
        out["l25_min_liq_to_mcap"] = self.deployment.l25_min_liq_to_mcap
        ts = out.get("at")
        if ts is not None and float(ts) < self.deployment.started_at:
            raise DataQualityError(
                f"refusing to write a prospective {kind} stamped "
                f"{float(ts):.3f}, which precedes the deployment boundary "
                f"{self.deployment.started_at:.3f}: a pre-deployment record "
                f"must not be labelled prospective"
            )
        return out

    # -- writers -----------------------------------------------------------

    def candidate(
        self,
        *,
        candidate_id: str,
        mint: Optional[str],
        route: str,
        features: Mapping[str, Any],
        at: Optional[float] = None,
        **extra: Any,
    ) -> dict[str, Any]:
        rec = {
            "candidate_id": candidate_id,
            "at": float(at if at is not None else _now()),
            "mint": mint,
            "route": str(getattr(route, "value", route)),
            # copied from the live feature snapshot, never recomputed here
            "market_cap_usd": features.get("market_cap_usd"),
            "liquidity_usd": features.get("liquidity_usd"),
            "liq_to_mcap": features.get("liq_to_mcap"),
            "signal_quality": features.get("signal_quality"),
            **extra,
        }
        rec = self._stamp(rec, "candidate")
        missing = _check_fields(rec, REQUIRED_CANDIDATE_FIELDS)
        if missing:
            rec["missing_fields"] = missing
        append_jsonl(self.candidates_path, rec)
        return rec

    def decision(self, report: Any, *, at: Optional[float] = None) -> dict[str, Any]:
        """
        Record a live DecisionReport. ``l25`` is carried verbatim.

        The report is serialised through its own accessor. ``dict(report)`` is
        NOT used as a fallback: on a dataclass it raises a bare TypeError, which
        would be swallowed by the ledger's guard and silently drop every
        prospective decision.
        """
        payload: Optional[Mapping[str, Any]] = None
        for accessor in ("as_dict", "to_json_dict"):
            fn = getattr(report, accessor, None)
            if callable(fn):
                payload = fn()
                break
        if payload is None:
            raise DataQualityError(
                f"cannot serialise a decision report of type "
                f"{type(report).__name__}: no as_dict()/to_json_dict()"
            )
        rec = dict(payload)
        rec.setdefault("at", float(at if at is not None else _now()))
        rec = self._stamp(rec, "decision")
        missing = _check_fields(rec, REQUIRED_DECISION_FIELDS)
        if missing:
            rec["missing_fields"] = missing
        append_jsonl(self.decisions_path, rec)
        return rec

    def trade(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if kind not in ("entry", "exit", "unfilled"):
            raise ValueError(f"unexpected trade record kind: {kind!r}")
        rec = self._stamp(dict(payload), kind)
        missing = _check_fields(rec, REQUIRED_TRADE_FIELDS if kind == "exit"
                               else ("trade_id", "at", "mint"))
        if missing:
            rec["missing_fields"] = missing
        append_jsonl(self.trades_path, rec)
        return rec

    # -- reading -----------------------------------------------------------

    def all_records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in (self.candidates_path, self.decisions_path, self.trades_path):
            out.extend(read_jsonl(p))
        return out


# ---------------------------------------------------------------------------
# offline reconstruction (POST_DERIVATION only)
# ---------------------------------------------------------------------------


def reconstruct_dataset(
    journal_decisions: Iterable[Mapping[str, Any]],
    *,
    l25_min_liq_to_mcap: float,
    strategy: str = "L25",
) -> list[dict[str, Any]]:
    """
    Re-score pre-L25 journal rows against the frozen gate.

    Produces ``POST_DERIVATION_RECONSTRUCTED`` rows ONLY. The dataset label is
    hard-coded, and the result is a sanity check on the gate's behaviour on
    data it was not fitted to -- never a validation of the deployed pipeline,
    because the runtime that wrote these rows was not gating on L25.
    """
    from autopaper.l25 import evaluate, verify_definition

    # refuse to reconstruct against a threshold other than the frozen one,
    # rather than quietly scoring history under a different rule
    verify_definition(l25_min_liq_to_mcap)

    out: list[dict[str, Any]] = []
    for row in journal_decisions:
        feats = row.get("features") or {}
        try:
            ev = evaluate(
                route=str(row.get("route") or ""),
                market_cap_usd=feats.get("market_cap_usd"),
                liquidity_usd=feats.get("liquidity_usd"),
                signal_quality=row.get("signal_quality"),
                min_liq_to_mcap=l25_min_liq_to_mcap,
            )
        except Exception as exc:  # noqa: BLE001
            out.append({
                "dataset": Dataset.POST_DERIVATION_RECONSTRUCTED.value,
                "candidate_id": row.get("candidate_id"),
                "at": row.get("at"),
                "error": f"{type(exc).__name__}: {exc}",
                "l25_passes": None,
            })
            continue
        out.append({
            "dataset": Dataset.POST_DERIVATION_RECONSTRUCTED.value,
            "candidate_id": row.get("candidate_id"),
            "mint": row.get("mint"),
            "at": row.get("at"),
            "strategy": strategy,
            "l25_passes": bool(ev.authorized),
            "l25_status": ev.status,
            "l25_detail": ev.detail,
            "liq_to_mcap": feats.get("liq_to_mcap"),
            "market_cap_usd": feats.get("market_cap_usd"),
            "liquidity_usd": feats.get("liquidity_usd"),
            "route": row.get("route"),
            "signal_quality": row.get("signal_quality"),
            "original_decision": row.get("decision"),
        })
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class Metrics:
    n: int
    wins: int
    losses: int
    win_rate: Optional[float]
    expectancy_pct: Optional[float]
    total_return_pct: Optional[float]
    profit_factor: Optional[float]
    max_drawdown_pct: Optional[float]
    median_holding_s: Optional[float]
    mean_mfe_pct: Optional[float]
    mean_mae_pct: Optional[float]
    expectancy_ci_low: Optional[float]
    expectancy_ci_high: Optional[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "expectancy_pct": self.expectancy_pct,
            "total_return_pct": self.total_return_pct,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "median_holding_s": self.median_holding_s,
            "mean_mfe_pct": self.mean_mfe_pct,
            "mean_mae_pct": self.mean_mae_pct,
            "expectancy_ci_low_pct": self.expectancy_ci_low,
            "expectancy_ci_high_pct": self.expectancy_ci_high,
        }


def expectancy_ci(
    returns: Sequence[float],
    *,
    policy: ValidationPolicy = DEFAULT_POLICY,
) -> tuple[Optional[float], Optional[float]]:
    """
    Percentile bootstrap interval for mean return. Deterministic.

    Seeded from the policy, so the same data always yields the same interval and
    a report can be diffed across runs.
    """
    n = len(returns)
    if n < 2:
        return (None, None)
    rng = random.Random(policy.bootstrap_seed)
    means: list[float] = []
    for _ in range(policy.bootstrap_draws):
        total = 0.0
        for _ in range(n):
            total += returns[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1.0 - policy.ci_level) / 2.0
    lo = means[max(0, int(math.floor(alpha * len(means))))]
    hi = means[min(len(means) - 1, int(math.ceil((1.0 - alpha) * len(means))) - 1)]
    return (lo, hi)


def compute_metrics(
    closed_trades: Sequence[Mapping[str, Any]],
    *,
    policy: ValidationPolicy = DEFAULT_POLICY,
) -> Metrics:
    """Metrics over CLOSED trades. Every field is null rather than guessed."""
    returns: list[float] = []
    for t in closed_trades:
        r = _finite(t.get("return_pct"))
        if r is not None:
            returns.append(r)
    n = len(returns)
    if n == 0:
        return Metrics(0, 0, 0, None, None, None, None, None, None,
                       None, None, None, None)

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    # equity curve for drawdown, in the order the trades closed
    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    for r in returns:
        equity *= (1.0 + r / 100.0)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)

    holds = [h for h in (_finite(t.get("holding_s")) for t in closed_trades)
             if h is not None]
    mfe = [m for m in (_finite(t.get("mfe_pct")) for t in closed_trades)
           if m is not None]
    mae = [m for m in (_finite(t.get("mae_pct")) for t in closed_trades)
           if m is not None]
    holds_sorted = sorted(holds)
    if not holds_sorted:
        median_hold = None
    else:
        mid = len(holds_sorted) // 2
        median_hold = (holds_sorted[mid] if len(holds_sorted) % 2
                       else (holds_sorted[mid - 1] + holds_sorted[mid]) / 2.0)

    lo, hi = expectancy_ci(returns, policy=policy)
    return Metrics(
        n=n,
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / n,
        expectancy_pct=sum(returns) / n,
        total_return_pct=equity - 100.0,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else None,
        max_drawdown_pct=max_dd,
        median_holding_s=median_hold if holds else None,
        mean_mfe_pct=(sum(mfe) / len(mfe)) if mfe else None,
        mean_mae_pct=(sum(mae) / len(mae)) if mae else None,
        expectancy_ci_low=lo,
        expectancy_ci_high=hi,
    )


# ---------------------------------------------------------------------------
# data quality
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    n_records: int
    records_with_missing_fields: int
    missing_field_counts: dict[str, int]
    records_before_boundary: int
    deployment_id_mismatches: int
    unlinked_trades: int
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_records": self.n_records,
            "records_with_missing_fields": self.records_with_missing_fields,
            "missing_field_counts": self.missing_field_counts,
            "records_before_boundary": self.records_before_boundary,
            "deployment_id_mismatches": self.deployment_id_mismatches,
            "unlinked_trades": self.unlinked_trades,
            "problems": self.problems,
        }


def check_quality(
    records: Sequence[Mapping[str, Any]],
    deployment: Optional[Deployment],
) -> QualityReport:
    """
    Decide whether the prospective data may be read at all.

    A verdict computed over data that is missing required fields, predates the
    boundary, or carries a foreign deployment id is worse than no verdict,
    because it looks like evidence.
    """
    if deployment is None:
        return QualityReport(0, 0, {}, 0, 0, 0,
                             ["no deployment boundary exists"])

    problems: list[str] = []
    missing_counts: dict[str, int] = {}
    n_missing = 0
    before = 0
    mismatched = 0
    entry_ids: set[str] = set()
    exit_ids: list[str] = []

    for rec in records:
        if rec.get("dataset") != Dataset.PROSPECTIVE_PAPER.value:
            problems.append(
                f"a non-prospective record ({rec.get('dataset')!r}) is present "
                f"in the prospective store"
            )
        if rec.get("deployment_id") != deployment.deployment_id:
            mismatched += 1
        ts = _finite(rec.get("at"))
        if ts is not None and ts < deployment.started_at:
            before += 1
        miss = rec.get("missing_fields")
        if isinstance(miss, list) and miss:
            n_missing += 1
            for f in miss:
                missing_counts[f] = missing_counts.get(f, 0) + 1
        kind = rec.get("kind")
        tid = rec.get("trade_id")
        if kind == "entry" and tid is not None:
            # the entry defines the trade universe
            entry_ids.add(str(tid))
        elif kind == "exit" and tid is not None:
            exit_ids.append(str(tid))

    # An exit with no matching entry cannot be scored: there is no size, no
    # entry price and no fill to attribute the return to. Compared against
    # ENTRIES only -- counting the exit as its own entry would make this
    # check vacuous, which is the bug this replaces.
    unlinked = sum(1 for tid in exit_ids if tid not in entry_ids)

    if n_missing:
        problems.append(
            f"{n_missing} prospective record(s) are missing required fields: "
            f"{sorted(missing_counts)}"
        )
    if before:
        problems.append(
            f"{before} prospective record(s) predate the deployment boundary"
        )
    if mismatched:
        problems.append(
            f"{mismatched} record(s) carry a foreign deployment id"
        )
    if unlinked:
        problems.append(f"{unlinked} exit(s) have no matching entry")

    return QualityReport(
        n_records=len(records),
        records_with_missing_fields=n_missing,
        missing_field_counts=missing_counts,
        records_before_boundary=before,
        deployment_id_mismatches=mismatched,
        unlinked_trades=unlinked,
        problems=problems,
    )


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


@dataclass
class ForwardReport:
    status: Status
    dataset: Dataset
    metrics: Metrics
    quality: QualityReport
    policy: ValidationPolicy
    criteria: dict[str, Optional[bool]]
    n_candidates: int
    n_decisions: int
    n_entered: int
    n_closed: int
    n_unfilled: int
    deployment: Optional[Deployment]
    notes: list[str] = field(default_factory=list)

    @property
    def is_validating(self) -> bool:
        """True only when this report may be cited as validating the strategy."""
        return self.status is Status.CONSISTENT and self.dataset is VALIDATING

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "dataset": self.dataset.value,
            "is_validating": self.is_validating,
            "counts": {
                "candidates": self.n_candidates,
                "decisions": self.n_decisions,
                "entered": self.n_entered,
                "closed": self.n_closed,
                "unfilled": self.n_unfilled,
            },
            "metrics": self.metrics.as_dict(),
            "data_quality": self.quality.as_dict(),
            "criteria": self.criteria,
            "policy": self.policy.as_dict(),
            "deployment": self.deployment.as_dict() if self.deployment else None,
            "notes": self.notes,
        }

    def render(self) -> str:
        d = self.as_dict()
        lines = [
            "FORWARD PAPER VALIDATION",
            "=" * 62,
            f"status            : {d['status']}",
            f"dataset           : {d['dataset']}",
            f"validating        : {d['is_validating']}",
            f"candidates        : {self.n_candidates}",
            f"decisions         : {self.n_decisions}",
            f"entries           : {self.n_entered}",
            f"closed trades     : {self.n_closed}",
            f"unfilled          : {self.n_unfilled}",
            "-" * 62,
        ]
        m = d["metrics"]
        if self.n_closed:
            lines += [
                f"win rate          : {_pct(m['win_rate'])}",
                f"expectancy/trade  : {_pct(m['expectancy_pct'])}",
                f"expectancy 95% CI : {_pct(m['expectancy_ci_low_pct'])} .. "
                f"{_pct(m['expectancy_ci_high_pct'])}",
                f"total return      : {_pct(m['total_return_pct'])}",
                f"profit factor     : {_num(m['profit_factor'])}",
                f"max drawdown      : {_pct(m['max_drawdown_pct'])}",
                f"median hold       : {_num(m['median_holding_s'])}s",
                f"mean MFE / MAE    : {_pct(m['mean_mfe_pct'])} / "
                f"{_pct(m['mean_mae_pct'])}",
            ]
        else:
            lines.append("(no closed trades yet)")
        lines.append("-" * 62)
        lines.append("data quality      : "
                     + ("OK" if d["data_quality"]["ok"] else "PROBLEMS"))
        for p in d["data_quality"]["problems"]:
            lines.append(f"  ! {p}")
        lines.append("-" * 62)
        for k, v in d["criteria"].items():
            mark = "n/a" if v is None else ("pass" if v else "FAIL")
            lines.append(f"  {k:<34} {mark}")
        if self.notes:
            lines.append("-" * 62)
            for n in self.notes:
                lines.append(f"  {n}")
        if not self.is_validating:
            lines.append("-" * 62)
            lines.append(
                "This report does NOT validate L25. Only PROSPECTIVE_PAPER data "
                "collected by the deployed runtime, past the pre-declared sample "
                "size, can do that."
            )
        return "\n".join(lines)


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:+.2f}%"


def _num(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def build_report(
    records: Sequence[Mapping[str, Any]],
    *,
    deployment: Optional[Deployment],
    policy: ValidationPolicy = DEFAULT_POLICY,
) -> ForwardReport:
    """
    Build the forward report. Deterministic given the same records.

    The status ladder is ordered by severity of what it prevents you believing,
    and data quality is checked BEFORE performance, so dirty data can never
    produce an encouraging number.
    """
    quality = check_quality(records, deployment)
    records = list(records)

    cands = [r for r in records if r.get("kind") == "candidate"]
    decs = [r for r in records if r.get("kind") == "decision"]
    entries = [r for r in records if r.get("kind") == "entry"]
    exits = [r for r in records if r.get("kind") == "exit"]
    unfilled = [r for r in records if r.get("kind") == "unfilled"]
    exits.sort(key=lambda r: _finite(r.get("at")) or 0.0)
    metrics = compute_metrics(exits, policy=policy)

    notes: list[str] = []
    criteria: dict[str, Optional[bool]] = {}

    if deployment is None:
        return ForwardReport(
            status=Status.NO_DEPLOYMENT, dataset=Dataset.PROSPECTIVE_PAPER,
            metrics=metrics, quality=quality, policy=policy, criteria={},
            n_candidates=len(cands), n_decisions=len(decs),
            n_entered=len(entries), n_closed=len(exits),
            n_unfilled=len(unfilled), deployment=None,
            notes=["no deployment boundary: nothing can be prospective"],
        )

    if not quality.ok:
        return ForwardReport(
            status=Status.DATA_QUALITY_PROBLEM, dataset=Dataset.PROSPECTIVE_PAPER,
            metrics=metrics, quality=quality, policy=policy, criteria={},
            n_candidates=len(cands), n_decisions=len(decs),
            n_entered=len(entries), n_closed=len(exits),
            n_unfilled=len(unfilled), deployment=deployment,
            notes=["performance is not reported: the data cannot be trusted"],
        )

    m = metrics
    criteria = {
        "sample >= min_for_read":
            m.n >= policy.min_closed_for_any_read,
        "sample >= min_for_conclusion":
            m.n >= policy.min_closed_for_conclusion,
        "expectancy > 0": (m.expectancy_pct > 0.0
                           if policy.require_positive_expectancy
                           and m.expectancy_pct is not None else None),
        "95% CI excludes zero": (
            (m.expectancy_ci_low is not None and m.expectancy_ci_low > 0.0)
            if policy.require_ci_excludes_zero and m.n >= 2 else None),
        "max drawdown within limit": (
            m.max_drawdown_pct is not None
            and m.max_drawdown_pct <= policy.max_drawdown_fraction * 100.0
        ) if m.n else None,
    }

    if m.n < policy.min_closed_for_any_read:
        status = Status.NOT_ENOUGH_DATA
        notes.append(
            f"{m.n} closed trade(s); {policy.min_closed_for_any_read} required "
            f"before any performance statement"
        )
    elif m.n < policy.min_closed_for_conclusion:
        status = Status.INSUFFICIENT_SAMPLE
        notes.append(
            f"{m.n} closed trade(s); {policy.min_closed_for_conclusion} required "
            f"before a conclusion"
        )
    else:
        # every criterion must hold, and the sample must be adequate
        all_ok = all(v for v in criteria.values() if v is not None)
        status = Status.CONSISTENT if all_ok else Status.DEGRADED
        if all_ok:
            notes.append(
                "all pre-declared criteria met on prospective PAPER data; this "
                "still validates the PAPER pipeline, not real-money behaviour"
            )
        else:
            failed = [k for k, v in criteria.items() if v is False]
            notes.append(f"criteria not met: {failed}")

    return ForwardReport(
        status=status, dataset=Dataset.PROSPECTIVE_PAPER, metrics=metrics,
        quality=quality, policy=policy, criteria=criteria,
        n_candidates=len(cands), n_decisions=len(decs), n_entered=len(entries),
        n_closed=len(exits), n_unfilled=len(unfilled), deployment=deployment,
        notes=notes,
    )
