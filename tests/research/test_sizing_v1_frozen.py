"""
Differential equivalence: research/frozen/sizing_v1.py vs the deleted V1 module.

The oracle is the byte-compiled cache of the module that V3 deleted::

    src/autopaper/execution/__pycache__/sizing.cpython-314.pyc

Every case below compares the recovered implementation against that oracle field
by field, so a silent drift in any branch, any refusal reason, any constraint
binding, or any float fails loudly.

SCOPE, STATED PLAINLY: this proves SEMANTIC equivalence. It cannot prove
byte-for-byte identity with the original source, because comments and
formatting do not survive in bytecode. See the module docstring.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import marshal
import math
import pathlib
import random
import sys
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "research"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autopaper.config import RiskLimits                    # noqa: E402
from autopaper.models import (                              # noqa: E402
    ExitStyle, Position, PositionStatus, Regime, Route,
)
from frozen import sizing_v1 as recovered                  # noqa: E402

ORACLE_PYC = (ROOT / "src/autopaper/execution/__pycache__"
              / "sizing.cpython-314.pyc")

# The orphan cache is the only surviving copy of V1's sizer. If it is ever
# pruned there is nothing left to verify against, so the tests skip loudly
# rather than passing vacuously.
pytestmark = pytest.mark.skipif(
    not ORACLE_PYC.exists(),
    reason="the deleted V1 module's bytecode cache is gone; nothing to verify against",
)


def _load_oracle():
    """Import the deleted module straight out of its orphaned bytecode.

    It must be registered as ``autopaper.execution.sizing`` for its own
    relative imports (``from ..config import RiskLimits``) to resolve, but it is
    removed from ``sys.modules`` immediately afterwards on purpose: leaving a
    working ``autopaper.execution.sizing`` importable would mask exactly the
    bug this whole exercise exists to document, and any other test that
    accidentally imported it later would silently start passing.
    """
    import autopaper.execution
    spec = importlib.util.spec_from_file_location(
        "autopaper.execution.sizing", ORACLE_PYC)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autopaper.execution.sizing"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        del sys.modules["autopaper.execution.sizing"]
    return mod


@pytest.fixture(scope="module")
def oracle():
    return _load_oracle()


def _pos(mint: str, *, status=PositionStatus.OPEN, cash_out=10.0) -> Position:
    """A minimal OPEN/CLOSED Position. Only mint/status/cash_out_usd are read
    by the sizer; the rest satisfy the model's required fields."""
    _r = Route.FAST_ENTRY
    return Position(
        position_id=f"p_{mint}", candidate_id="c", mint=mint, symbol="TOK",
        status=status, route=_r, regime=Regime.EARLY, exit_style=ExitStyle.RUNNER,
        strategy=_r.value, entry_signal_timestamp=0.0,
        entry_decision_timestamp=0.0, entry_timestamp=0.0, quoted_price=1.0,
        entry_reference_price=1.0, fill_price=1.0, quantity=1.0,
        cash_out_usd=cash_out, cost_basis_usd=cash_out, peak_price=1.0,
        trough_price=1.0,
    )


def _fields(d) -> dict[str, Any]:
    return dataclasses.asdict(d)


def _assert_same(o, r, ctx: str) -> None:
    assert _fields(r) == _fields(o), (
        f"{ctx}\n  oracle:    {_fields(o)}\n  recovered: {_fields(r)}")
    # as_dict() rounds, so it can hide float drift the raw fields would catch;
    # compare it too, since it is the study's reporting surface.
    assert r.as_dict() == o.as_dict(), f"{ctx}: as_dict differs"


# --------------------------------------------------------------------------
# scale_target_for_route
# --------------------------------------------------------------------------

_SCALE_EDGE = [
    0.0, 0.1, 0.29, 0.2999999, 0.30, 0.3000001, 0.39, 0.3999999,
    0.40, 0.4000001, 0.54, 0.5499999, 0.55, 0.5500001, 0.70, 1.0, 1.5,
]


def test_scale_target_for_route_matches_oracle_on_every_branch_boundary(oracle):
    """Each penalty, each pair, and the clamps -- exhaustively, not sampled."""
    cases = itertools.product(
        (5.0, 10.0, 12.5, 18.0, 0.0, 40.0),          # target_pct
        _SCALE_EDGE,                                   # liquidity_score
        (0.45, 0.5499999, 0.55, 0.7, 0.0),            # signal_quality
        (1.0, 0.99, 0.5, 0.0, 1.2),                   # route_multiplier
    )
    checked = 0
    for target_pct, liq, sig, mult in cases:
        for max_pct in (12.5, 18.0, 5.0):
            limits = dataclasses.replace(RiskLimits(), max_position_pct_equity=max_pct)
            o = oracle.scale_target_for_route(
                target_pct, liquidity_score=liq, signal_quality=sig,
                route_multiplier=mult, limits=limits)
            r = recovered.scale_target_for_route(
                target_pct, liquidity_score=liq, signal_quality=sig,
                route_multiplier=mult, limits=limits)
            assert o == r, (
                f"target={target_pct} liq={liq} sig={sig} mult={mult} "
                f"max={max_pct}\n  oracle:    {o}\n  recovered: {r}")
            checked += 1
    assert checked > 1000, checked


# --------------------------------------------------------------------------
# compute_size
# --------------------------------------------------------------------------

_REFUSAL_CASES = [
    # duplicate mint on an OPEN position
    dict(equity_usd=100.0, cash_usd=100.0, open_positions=[_pos("M")],
         target_pct=12.5, price=1.0, mint="M"),
    # NOT a refusal: a CLOSED position with the same mint is not a duplicate,
    # so this must be APPROVED. Filed here deliberately -- it is the branch
    # most easily "fixed" into a false duplicate by a careless port.
    dict(equity_usd=100.0, cash_usd=100.0,
         open_positions=[_pos("M", status=PositionStatus.CLOSED)],
         target_pct=12.5, price=1.0, mint="M", approved=True),
    # simultaneous position cap, and the exact boundary
    dict(equity_usd=100.0, cash_usd=100.0,
         open_positions=[_pos("A"), _pos("B")],
         target_pct=12.5, price=1.0, mint="M", max_positions=2),
    dict(equity_usd=100.0, cash_usd=100.0,
         open_positions=[_pos("A")],
         target_pct=12.5, price=1.0, mint="M", max_positions=1),
    # re-entry refused while recently exited (and only when re-entry is off)
    dict(equity_usd=100.0, cash_usd=100.0, open_positions=[],
         target_pct=12.5, price=1.0, mint="M",
         recently_exited={"M": 1_000.0}, now=1_030.0),
    # NOT a refusal: with re-entry enabled the same recent exit must APPROVE.
    dict(equity_usd=100.0, cash_usd=100.0, open_positions=[],
         target_pct=12.5, price=1.0, mint="M",
         recently_exited={"M": 1_000.0}, now=1_030.0, reentry_enabled=True,
         approved=True),
    dict(equity_usd=100.0, cash_usd=100.0, open_positions=[],
         target_pct=12.5, price=1.0, mint="M",
         recently_exited={"M": 1_000.0}, now=0),
    # empty account, either side
    dict(equity_usd=0.0, cash_usd=100.0, open_positions=[],
         target_pct=12.5, price=1.0, mint="M"),
    dict(equity_usd=100.0, cash_usd=0.0, open_positions=[],
         target_pct=12.5, price=1.0, mint="M"),
    # missing price
    dict(equity_usd=100.0, cash_usd=100.0, open_positions=[],
         target_pct=12.5, price=0.0, mint="M"),
    dict(equity_usd=100.0, cash_usd=100.0, open_positions=[],
         target_pct=12.5, price=-1.0, mint="M"),
    # exposure already full
    dict(equity_usd=100.0, cash_usd=100.0,
         open_positions=[_pos("A", cash_out=100.0)],
         target_pct=12.5, price=1.0, mint="M", max_exposure=12.5),
    # room below the minimum
    dict(equity_usd=1.0, cash_usd=1.0, open_positions=[],
         target_pct=12.5, price=1.0, mint="M", min_notional=5.0),
    # degenerate quantity from a vanishing notional
    dict(equity_usd=1e12, cash_usd=1e12, open_positions=[],
         target_pct=12.5, price=1e-300, mint="M", max_pct=100.0,
         min_notional=0.0, max_exposure=1e12),
]


@pytest.mark.parametrize("case", _REFUSAL_CASES,
                         ids=range(len(_REFUSAL_CASES)))
def test_compute_size_matches_oracle_on_every_boundary(oracle, case):
    """One case per reachable refusal, plus the two approval branches that sit
    next to them, including each boundary."""
    case = dict(case)
    kwargs = dict(
        equity_usd=100.0, cash_usd=100.0, open_positions=[], target_pct=12.5,
        price=1.0, mint="M", recently_exited={},
        now=0.0, entry_fee_bps=100.0,
    )
    kwargs.update(
        open_positions=case.pop("open_positions"),
        equity_usd=case.pop("equity_usd"),
        cash_usd=case.pop("cash_usd"),
        target_pct=case.pop("target_pct"),
        price=case.pop("price"),
        mint=case.pop("mint"),
        recently_exited=case.pop("recently_exited", {}),
        now=case.pop("now", 0.0),
    )
    limits = dataclasses.replace(
        RiskLimits(),
        max_simultaneous_positions=case.pop("max_positions",
                                            RiskLimits.max_simultaneous_positions),
        reentry_enabled=case.pop("reentry_enabled", RiskLimits.reentry_enabled),
        min_position_notional_usd=case.pop("min_notional", 0.0),
        max_position_pct_equity=case.pop("max_pct", 12.5),
        max_total_exposure_pct_equity=case.pop("max_exposure", 12.5),
        default_position_pct_low=case.pop("low", 10.0),
        default_position_pct_high=case.pop("high", 12.5),
    )
    expect_approved = case.pop("approved", False)
    assert not case, f"unconsumed case keys: {case}"
    o = oracle.compute_size(limits=limits, **kwargs)
    r = recovered.compute_size(limits=limits, **kwargs)
    _assert_same(o, r, f"boundary case {kwargs}")
    assert r.approved is expect_approved, (
        f"expected approved={expect_approved}, got {r}")
    if not expect_approved:
        # every refusal must name the rule that refused it
        assert r.binding_constraint, r


def test_compute_size_matches_oracle_on_a_randomised_grid(oracle):
    """Randomised sweep across the clamp interactions and the cents flooring."""
    rng = random.Random(20261001)
    checked = 0
    for _ in range(4000):
        equity = rng.choice([
            0.0, 1.55, rng.uniform(0.01, 1e4), rng.uniform(0, 1e6),
        ])
        cash = rng.choice([0.0, rng.uniform(0, equity * 1.2), equity])
        target_pct = rng.uniform(0.0, 40.0)
        price = rng.choice([0.0, 1e-9, rng.uniform(1e-9, 1e6)])
        fee_bps = rng.choice([0.0, 100.0, 2500.0, rng.uniform(0, 5000)])
        n_open = rng.randint(0, 3)
        open_positions = [
            _pos(rng.choice(["M", "A", "B", "C"]),
                 status=rng.choice(list(PositionStatus)),
                 cash_out=rng.uniform(0.0, equity))
            for _ in range(n_open)
        ]
        limits = dataclasses.replace(
            RiskLimits(),
            max_simultaneous_positions=rng.choice([1, 2, 5]),
            reentry_enabled=rng.random() < 0.3,
            min_position_notional_usd=rng.choice([0.0, 1.0, 5.0]),
            max_position_pct_equity=rng.choice([5.0, 12.5, 18.0, 100.0]),
            max_total_exposure_pct_equity=rng.choice([5.0, 12.5, 50.0]),
            default_position_pct_low=rng.choice([0.0, 10.0]),
            default_position_pct_high=rng.choice([12.5, 18.0]),
        )
        kwargs = dict(
            equity_usd=equity, cash_usd=cash, open_positions=open_positions,
            target_pct=target_pct, price=price, limits=limits, mint="M",
            recently_exited={"M": rng.uniform(0, 4000)} if rng.random() < 0.4 else {},
            now=rng.uniform(0, 4000) if rng.random() < 0.5 else 0.0,
            entry_fee_bps=fee_bps,
        )
        o = oracle.compute_size(**kwargs)
        r = recovered.compute_size(**kwargs)
        _assert_same(o, r, f"grid case {kwargs}")
        checked += 1
    assert checked == 4000


def test_compute_size_approvals_bind_to_the_last_constraint_that_bit(oracle):
    """The binding is not incidental: it must name the FINAL clamp applied.

    This is what lets the study attribute a trade to a rule instead of
    guessing, so it is asserted positively and not only via the oracle diff.
    """
    limits = dataclasses.replace(
        RiskLimits(), max_position_pct_equity=12.5,
        max_total_exposure_pct_equity=12.5,
        default_position_pct_low=10.0, default_position_pct_high=12.5,
        min_position_notional_usd=0.0, max_simultaneous_positions=5)

    # route target above the band ceiling -> clamped by the band, binding stays
    # "route_target" because the band clamp is not a constraint binding
    d = recovered.compute_size(
        equity_usd=100.0, cash_usd=100.0, open_positions=[], target_pct=40.0,
        price=1.0, limits=limits, mint="M", recently_exited={}, now=0.0,
        entry_fee_bps=0.0)
    assert d.approved and d.binding_constraint == "route_target"
    assert d.cash_required_usd == 12.5

    # cash below the route target -> "cash" wins
    d = recovered.compute_size(
        equity_usd=100.0, cash_usd=5.0, open_positions=[], target_pct=12.5,
        price=1.0, limits=limits, mint="M", recently_exited={}, now=0.0,
        entry_fee_bps=0.0)
    assert d.approved and d.binding_constraint == "cash"
    assert d.cash_required_usd == 5.0

    # Exposure room below the route target but still positive -> the target is
    # clamped to the room, and "exposure_cap" is the binding. Room is
    # 12.5 - 5.0 = 7.5, so cash (100) and the position ceiling (12.5) do not
    # bite and "exposure_cap" is genuinely the LAST clamp applied.
    d = recovered.compute_size(
        equity_usd=100.0, cash_usd=100.0,
        open_positions=[_pos("A", cash_out=5.0)], target_pct=12.5, price=1.0,
        limits=limits, mint="M", recently_exited={}, now=0.0,
        entry_fee_bps=0.0)
    assert d.approved and d.binding_constraint == "exposure_cap", d
    assert d.cash_required_usd == pytest.approx(7.5), d

    # ...and when that room is exhausted entirely it refuses rather than
    # sizing to zero, which is the branch the positive case must not blur.
    d = recovered.compute_size(
        equity_usd=100.0, cash_usd=100.0,
        open_positions=[_pos("A", cash_out=95.0)], target_pct=12.5, price=1.0,
        limits=limits, mint="M", recently_exited={}, now=0.0,
        entry_fee_bps=0.0)
    assert d.approved is False and d.binding_constraint == "exposure_cap", d


def test_the_fee_is_reserved_not_deducted_from_the_notional(oracle):
    """The sizing identity the study depends on: notional + fee == cash.

    If the recovered sizer had been written to shrink the notional to make a
    fee fit, this would break -- which is the exact bug the V3 policy comment
    says the old implementation had.
    """
    limits = dataclasses.replace(
        RiskLimits(), max_position_pct_equity=12.5,
        max_total_exposure_pct_equity=12.5, min_position_notional_usd=0.0)
    fee_bps = 2500.0
    d = recovered.compute_size(
        equity_usd=100.0, cash_usd=100.0, open_positions=[], target_pct=12.5,
        price=2.0, limits=limits, mint="M", recently_exited={}, now=0.0,
        entry_fee_bps=fee_bps)
    assert d.approved
    # cash = floor to cents of 12.5; notional = floor to cents of cash/fee_mult
    assert d.cash_required_usd == 12.5
    assert d.notional_usd == pytest.approx(
        math.floor((12.5 / (1 + fee_bps / 10000.0)) * 100.0) / 100.0)
    assert d.fee_reserved_usd == pytest.approx(
        d.cash_required_usd - d.notional_usd, abs=1e-4)
    _assert_same(
        oracle.compute_size(
            equity_usd=100.0, cash_usd=100.0, open_positions=[],
            target_pct=12.5, price=2.0, limits=limits, mint="M",
            recently_exited={}, now=0.0, entry_fee_bps=fee_bps),
        d, "fee-reservation identity")


def test__fmt_ago_matches_oracle_across_every_unit_boundary(oracle):
    for s in [0, 1, 59, 60, 61, 599, 600, 5399, 5400, 5401, 3600, 7200,
              86399, 86400, 100000.5, -5.0]:
        assert recovered._fmt_ago(s) == oracle._fmt_ago(s), s


def test_the_recovered_module_is_not_reachable_from_the_runtime():
    """The frozen copy must stay research-only.

    This mirrors the invariant tests/research/test_v1lab.py already enforces
    for the runtime's imports; asserted here too because this module is a new
    research file that a future change could wrongly start importing.
    """
    for p in (ROOT / "src").rglob("*.py"):
        tree = __import__("ast").parse(p.read_text())
        for node in __import__("ast").walk(tree):
            mods: list[str] = []
            if isinstance(node, __import__("ast").Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, __import__("ast").ImportFrom):
                mods = [node.module or ""]
            for m in mods:
                assert m.split(".")[0] not in {"frozen", "bankroll_study"}, (
                    f"{p} imports the research module {m!r}")
