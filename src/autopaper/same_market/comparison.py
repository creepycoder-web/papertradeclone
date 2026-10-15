"""
Four-way price comparison, pairwise gaps, and the disagreement diagnostics.

THE FOUR THINGS BEING COMPARED ARE NOT INTERCHANGEABLE
=====================================================

============  ===========================================================
label         what it is
============  ===========================================================
``OBSERVED_MARK``    DexScreener's ``priceUsd``. An observation about a pool.
``DIRECT_EXECUTABLE_QUOTE``  priced from on-chain state, for THIS size.
``JUPITER_QUOTE``    an aggregator's executable route.
``ACTUAL_FILL``      never exists here. No transaction is ever sent.
============  ===========================================================

DexScreener's mark is USD per token. A direct quote or a Jupiter quote taken
with a SOL-denominated input is SOL per token. Comparing them without
converting produces a gap of a few hundred percent that looks like an extreme
market, and it is silent because both numbers are positive and finite. So the
USD conversion happens first, using the SAME logged ``--sol-usd`` reference that
sized the order, and the reference travels with the result.

NO THRESHOLD IS INVENTED
========================

``CLOSE`` and ``DIVERGENT`` need a tolerance, and the default is *no
tolerance*. With none configured every pair is ``UNKNOWN`` with the raw
percentage gap printed, because manufacturing a threshold is the exact
behaviour this harness exists to catch. Supply ``--divergence-pct`` and the
classification becomes available; the default verdict stays ``INFORMATIVE``
either way, because a gap is a diagnostic and not a trading decision.

AN UNAVAILABLE SOURCE IS NOT A ZERO
===================================

A source that could not be read produces ``None`` and a named reason. It is
never treated as ``0.0``, so it can never produce a 100% gap that reads as a
finding.

THE TWO NAMED INVESTIGATIONS
=============================

``MARK_DIVERGENCE``
    Two executable quotes agree with each other but the mark does not. The
    executable prices are computed from state on chain, so this points at the
    mark describing a different market, a stale pool, or a different pool than
    the one quoted. Recorded with everything needed to reproduce it.

``EXECUTION_QUOTE_DIVERGENCE``
    Two executable quotes disagree. Both are executable; something about the
    routing, the pool, or the state snapshot differs. Also recorded, also
    without deciding which is right - this harness has no basis for that
    judgement and pretending otherwise would be the bug.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Sequence

from .numeric import DecimalError, to_decimal, to_float


class DivergenceClass(str, Enum):
    """How two prices relate. ``UNKNOWN`` is the default, not a failure."""

    CLOSE = "CLOSE"
    DIVERGENT = "DIVERGENT"
    UNKNOWN = "UNKNOWN"


class PriceLabel(str, Enum):
    """The four kinds of number, kept apart."""

    OBSERVED_MARK = "OBSERVED_MARK"
    DIRECT_EXECUTABLE_QUOTE = "DIRECT_EXECUTABLE_QUOTE"
    JUPITER_QUOTE = "JUPITER_QUOTE"
    ACTUAL_FILL = "ACTUAL_FILL"


@dataclass(frozen=True)
class PricePoint:
    """One comparable price, with its own provenance."""

    label: PriceLabel
    provider: str
    price_usd_per_token: float | None
    price_sol_per_token: float | None = None
    venue: str | None = None
    pool_or_curve: str | None = None
    token_mint: str | None = None
    quote_mint: str | None = None
    route: tuple[str, ...] = ()
    raw_input: str | None = None
    raw_output: str | None = None
    input_decimals: int | None = None
    output_decimals: int | None = None
    state_slot: int | None = None
    state_timestamp: float | None = None
    state_age_ms: float | None = None
    quote_timestamp: float | None = None
    quote_latency_ms: float | None = None
    usd_reference_price: float | None = None
    missing_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.price_usd_per_token is not None and self.price_usd_per_token > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "provider": self.provider,
            "price_usd_per_token": _f(self.price_usd_per_token),
            "price_sol_per_token": _f(self.price_sol_per_token),
            "venue": self.venue,
            "pool_or_curve": self.pool_or_curve,
            "token_mint": self.token_mint,
            "quote_mint": self.quote_mint,
            "route": list(self.route),
            "raw_input": self.raw_input,
            "raw_output": self.raw_output,
            "input_decimals": self.input_decimals,
            "output_decimals": self.output_decimals,
            "state_slot": self.state_slot,
            "state_timestamp": _f(self.state_timestamp),
            "state_age_ms": _f(self.state_age_ms),
            "quote_timestamp": _f(self.quote_timestamp),
            "quote_latency_ms": _f(self.quote_latency_ms),
            "usd_reference_price": _f(self.usd_reference_price),
            "available": self.available,
            "missing_reason": self.missing_reason,
        }


def _f(value: float | None) -> float | None:
    return None if value is None else to_float(value)


@dataclass(frozen=True)
class PairGap:
    """
    The gap between two prices.

    ``gap_pct`` is ``(b / a - 1) * 100`` where ``b`` is ``to_label`` and ``a``
    is ``from_label``, so a positive gap means the second source is HIGHER than
    the first. For a BUY that matters in both directions and the sign is kept
    because a mark that is 74% above the executable price is a different
    problem from one that is 74% below it.
    """

    name: str
    from_label: str
    to_label: str
    from_price: float | None
    to_price: float | None
    gap_pct: float | None
    classification: DivergenceClass
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "from": self.from_label,
            "to": self.to_label,
            "from_price_usd_per_token": _f(self.from_price),
            "to_price_usd_per_token": _f(self.to_price),
            "gap_pct": _f(self.gap_pct),
            "classification": self.classification.value,
            "reason": self.reason,
        }


def _gap(name: str, a: PricePoint, b: PricePoint,
         divergence_pct: float | None) -> PairGap:
    """
    Compare two points, or explain why it cannot be done.

    Every branch returns something printable. A comparison that could not be
    made is a result, not an omission.
    """
    for point in (a, b):
        if not point.available:
            return PairGap(
                name, a.label.value, b.label.value,
                a.price_usd_per_token, b.price_usd_per_token, None,
                DivergenceClass.UNKNOWN,
                (
                    f"{point.label.value} is unavailable"
                    + (f": {point.missing_reason}" if point.missing_reason else "")
                    + ". An absent source is not a zero and produces no gap"
                ),
            )
    try:
        gap = (to_decimal(b.price_usd_per_token) / to_decimal(a.price_usd_per_token) - 1) * 100
    except (DecimalError, ZeroDivisionError) as exc:
        return PairGap(
            name, a.label.value, b.label.value,
            a.price_usd_per_token, b.price_usd_per_token, None,
            DivergenceClass.UNKNOWN, f"the gap is undefined: {exc}",
        )
    absolute = abs(gap)
    if divergence_pct is None:
        return PairGap(
            name, a.label.value, b.label.value,
            a.price_usd_per_token, b.price_usd_per_token, to_float(gap),
            DivergenceClass.UNKNOWN,
            (
                "no --divergence-pct was configured, so the raw gap is reported "
                "and NO close/distant verdict is claimed"
            ),
        )
    close = absolute <= Decimal(str(divergence_pct))
    return PairGap(
        name, a.label.value, b.label.value,
        a.price_usd_per_token, b.price_usd_per_token, to_float(gap),
        DivergenceClass.CLOSE if close else DivergenceClass.DIVERGENT,
        (
            f"|gap| {to_float(absolute):.4f}% is "
            + ("within" if close else "outside")
            + f" the configured {divergence_pct:g}%"
        ),
    )


#: The five diagnostics the brief names, in the order it names them.
DIAGNOSTIC_PAIRS: tuple[tuple[str, PriceLabel, PriceLabel], ...] = (
    ("MARK_VS_PUMP_DIRECT", PriceLabel.OBSERVED_MARK, PriceLabel.DIRECT_EXECUTABLE_QUOTE),
    ("MARK_VS_JUPITER", PriceLabel.OBSERVED_MARK, PriceLabel.JUPITER_QUOTE),
    ("PUMP_DIRECT_VS_JUPITER", PriceLabel.DIRECT_EXECUTABLE_QUOTE, PriceLabel.JUPITER_QUOTE),
    ("MARK_VS_PUMPSWAP", PriceLabel.OBSERVED_MARK, PriceLabel.DIRECT_EXECUTABLE_QUOTE),
    ("PUMPSWAP_VS_JUPITER", PriceLabel.DIRECT_EXECUTABLE_QUOTE, PriceLabel.JUPITER_QUOTE),
)


@dataclass(frozen=True)
class CrossCheck:
    """Every pairwise gap plus the two named investigations."""

    gaps: tuple[PairGap, ...]
    findings: tuple[str, ...]
    actionable: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairwise_gaps": [g.as_dict() for g in self.gaps],
            "findings": list(self.findings),
            "triggers_diagnostic_investigation": self.actionable,
            "note": self.note,
        }

    def render(self) -> str:
        lines = ["CROSS-CHECK"]
        for gap in self.gaps:
            if gap.gap_pct is None:
                lines.append(f"  {gap.name:<26} UNKNOWN   {gap.reason}")
            else:
                lines.append(
                    f"  {gap.name:<26} {gap.gap_pct:+9.4f}%  "
                    f"[{gap.classification.value}]"
                )
        for finding in self.findings:
            lines.append(f"  ! {finding}")
        lines.append(f"  {self.note}")
        return "\n".join(lines)


def cross_check(
    mark: PricePoint,
    pump_direct: PricePoint | None,
    pumpswap_direct: PricePoint | None,
    jupiter: PricePoint | None,
    *,
    divergence_pct: float | None = None,
) -> CrossCheck:
    """
    Build every pairwise gap that is meaningful, and name the two findings.

    Pairs are only formed between sources that describe a comparable market.
    A curve price and a pool price are different books, so they are never
    compared to each other; each is compared to the mark and to Jupiter, and
    Jupiter is the one source that can be compared with both.
    """
    # Absent sources are normalised into EXPLICIT unavailable points rather
    # than being passed around as ``None``. Two reasons: the report can then
    # name which source was missing, and no comparison branch has to guard
    # against a null object - which is where an earlier version of this
    # function crashed, on a signal with no quotes at all.
    def absent(label: PriceLabel, why: str) -> PricePoint:
        return PricePoint(
            label=label, provider="none", price_usd_per_token=None,
            missing_reason=why,
        )

    mark_point = mark if mark is not None else absent(
        PriceLabel.OBSERVED_MARK, "no observed mark was collected")
    pump_point = pump_direct if pump_direct is not None else absent(
        PriceLabel.DIRECT_EXECUTABLE_QUOTE,
        "no direct bonding-curve quote was collected for this signal")
    swap_point = pumpswap_direct if pumpswap_direct is not None else absent(
        PriceLabel.DIRECT_EXECUTABLE_QUOTE,
        "no direct PumpSwap quote was collected for this signal")
    jup_point = jupiter if jupiter is not None else absent(
        PriceLabel.JUPITER_QUOTE, "no Jupiter quote was collected for this signal")

    gaps: list[PairGap] = []

    def add(name: str, a: PricePoint, b: PricePoint) -> None:
        if not a.available or not b.available:
            absent_labels = ", ".join(
                p.label.value for p in (a, b) if not p.available
            )
            reasons = "; ".join(
                f"{p.label.value}: {p.missing_reason}"
                for p in (a, b) if not p.available and p.missing_reason
            )
            gaps.append(PairGap(
                name, a.label.value, b.label.value,
                a.price_usd_per_token, b.price_usd_per_token, None,
                DivergenceClass.UNKNOWN,
                f"unavailable source ({absent_labels}). An absent source is "
                f"not a zero and produces no gap. {reasons}".strip(),
            ))
            return
        gaps.append(_gap(name, a, b, divergence_pct))

    add("MARK_VS_PUMP_DIRECT", mark_point, pump_point)
    add("MARK_VS_PUMPSWAP", mark_point, swap_point)
    add("MARK_VS_JUPITER", mark_point, jup_point)
    add("PUMP_DIRECT_VS_JUPITER", pump_point, jup_point)
    add("PUMPSWAP_VS_JUPITER", swap_point, jup_point)

    findings: list[str] = []
    actionable = False

    pump_side = pump_point if pump_point.available else (
        swap_point if swap_point.available else None
    )
    pump_name = (
        "Pump direct" if pump_point.available else "PumpSwap direct"
    )

    # EXECUTION_QUOTE_DIVERGENCE: two executable sources disagree.
    if pump_side is not None and jup_point.available:
        gap = _gap("exec-vs-jupiter", pump_side, jup_point, divergence_pct)
        if gap.gap_pct is not None and divergence_pct is not None and abs(gap.gap_pct) > divergence_pct:
            findings.append(
                f"EXECUTION_QUOTE_DIVERGENCE: {pump_name} and Jupiter are "
                f"{gap.gap_pct:+.4f}% apart. Both are executable quotes, so this "
                "is a difference of routing, pool selection, or state snapshot - "
                "not an observation error. NOT resolved automatically; the "
                "snapshot below makes it reproducible."
            )
            actionable = True
        elif gap.gap_pct is not None:
            findings.append(
                f"{pump_name} and Jupiter agree to {gap.gap_pct:+.4f}% on an "
                "executable basis"
            )

    # MARK_DIVERGENCE: the executable sources agree, the mark does not.
    if (
        pump_side is not None and pump_side.available
        and jupiter is not None and jupiter.available
        and mark.available
        and divergence_pct is not None
    ):
        exec_vs_exec = _gap("e", pump_side, jupiter, divergence_pct)
        mark_vs_pump = _gap("m", mark, pump_side, divergence_pct)
        mark_vs_jup = _gap("mj", mark, jupiter, divergence_pct)
        if (
            exec_vs_exec.gap_pct is not None
            and abs(exec_vs_exec.gap_pct) <= divergence_pct
            and (
                (mark_vs_pump.gap_pct is not None and abs(mark_vs_pump.gap_pct) > divergence_pct)
                or (mark_vs_jup.gap_pct is not None and abs(mark_vs_jup.gap_pct) > divergence_pct)
            )
        ):
            findings.append(
                "MARK_DIVERGENCE: the executable quotes agree with each other "
                f"({exec_vs_exec.gap_pct:+.4f}%) but the DexScreener mark does "
                f"not ({mark_vs_pump.gap_pct:+.4f}% vs {pump_name}, "
                f"{mark_vs_jup.gap_pct:+.4f}% vs Jupiter). The mark most likely "
                "describes a different pool or a stale one. Reproduce with the "
                "state slot, order size and addresses below."
            )
            actionable = True

    if not actionable and not findings:
        note = (
            "raw percentage gaps are reported; with no --divergence-pct "
            "configured nothing is classified, which is the point"
        )
    else:
        note = (
            "INFORMATIVE: a price gap is a diagnostic finding, not a trading "
            "decision. No threshold was invented and no source was chosen as "
            "correct"
        )

    return CrossCheck(
        gaps=tuple(gaps), findings=tuple(findings), actionable=actionable, note=note
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityFinding:
    """
    One provider's claim about which market it describes.

    ``kind`` distinguishes the object types deliberately: a bonding curve and
    an AMM pool are different accounts with different programs, and treating
    one address as interchangeable with the other is how a "same market" check
    silently becomes meaningless.
    """

    provider: str
    kind: str  # BONDING_CURVE | AMM_POOL | OBSERVED_PAIR | ROUTE_HOP | NONE
    address: str | None
    token_mint: str | None
    quote_mint: str | None
    venue: str | None
    verified_by_chain: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "kind": self.kind,
            "address": self.address,
            "token_mint": self.token_mint,
            "quote_mint": self.quote_mint,
            "venue": self.venue,
            "verified_by_chain": self.verified_by_chain,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class IdentityReport:
    """
    Whether the marked pair and the quoted markets are demonstrably the same.

    The aggregate is deliberately tri-state. ``UNKNOWN`` is a real answer here,
    not a soft pass: the harness cannot prove that two addresses describe the
    same market, and saying so is the finding.
    """

    findings: tuple[IdentityFinding, ...]
    pool_match: str
    market_match: str
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.as_dict() for f in self.findings],
            "POOL_MATCH": self.pool_match,
            "MARKET_MATCH": self.market_match,
            "notes": list(self.notes),
        }

    def render(self) -> str:
        lines = ["MARKET IDENTITY"]
        for f in self.findings:
            addr = f.address[:10] + "…" if f.address else "-"
            lines.append(
                f"  {f.provider:<16} {f.kind:<14} {addr:<11} "
                f"chain-verified={str(f.verified_by_chain).lower()}"
            )
        lines.append(f"  POOL_MATCH   = {self.pool_match}")
        lines.append(f"  MARKET_MATCH = {self.market_match}")
        return "\n".join(lines)


def build_identity_report(
    *,
    mint: str,
    observed_pair_address: str | None,
    observed_dex_id: str | None,
    curve_address: str | None = None,
    curve_chain_verified: bool = False,
    pool_address: str | None = None,
    pool_chain_verified: bool = False,
    jupiter_pool_keys: Sequence[str] = (),
    jupiter_labels: Sequence[str] = (),
) -> IdentityReport:
    """
    Map every provider onto the market and compare the ADDRESSES.

    The mapping is provider-specific because "the pair address" means something
    different per provider, and this was verified live:

    * a DexScreener ``pairAddress`` on a ``pumpfun`` pair **is the bonding-curve
      PDA**;
    * a DexScreener ``pairAddress`` on a ``pumpswap`` pair **is the AMM pool
      address**;
    * a Jupiter ``routePlan[].swapInfo.ammKey`` is the curve PDA for a curve
      route (``label`` ``Pump.fun``) and the pool address for an AMM route
      (``label`` ``Pump.fun Amm``). Branching on the label is NOT relied upon -
      ``ammKey`` is compared and the account type comes from the chain.

    So an exact address comparison is meaningful here, which it would not be if
    one side were a curve and the other a pool.
    """
    findings: list[IdentityFinding] = []
    notes: list[str] = []

    if observed_pair_address:
        findings.append(IdentityFinding(
            provider="dexscreener", kind="OBSERVED_PAIR",
            address=observed_pair_address, token_mint=mint,
            quote_mint=None, venue=observed_dex_id, verified_by_chain=False,
            detail=(
                f"aggregator label {observed_dex_id!r}; on pump.fun the pair "
                "address is the market account itself (the curve PDA before "
                "graduation, the pool address after), which is why it can be "
                "compared to a derived address directly"
            ),
        ))
    else:
        notes.append("no observed pair address, so nothing can be mapped onto it")

    if curve_address:
        findings.append(IdentityFinding(
            provider="pump_direct", kind="BONDING_CURVE", address=curve_address,
            token_mint=mint, quote_mint=None, venue="pumpfun",
            verified_by_chain=curve_chain_verified,
            detail=(
                "derived from seeds ['bonding-curve', mint] under the pump.fun "
                "program and confirmed by the account's owner and discriminator"
            ),
        ))

    if pool_address:
        findings.append(IdentityFinding(
            provider="pumpswap_direct", kind="AMM_POOL", address=pool_address,
            token_mint=mint, quote_mint=None, venue="pumpswap",
            verified_by_chain=pool_chain_verified,
            detail=(
                "account owned by the PumpSwap program carrying the Pool "
                "discriminator, with base mint equal to the token under test"
            ),
        ))

    for index, key in enumerate(jupiter_pool_keys):
        label = jupiter_labels[index] if index < len(jupiter_labels) else None
        findings.append(IdentityFinding(
            provider=f"jupiter[{index}]", kind="ROUTE_HOP", address=key,
            token_mint=mint, quote_mint=None, venue=label, verified_by_chain=False,
            detail=(
                "routePlan[].swapInfo.ammKey; for a pump venue this is the curve "
                "PDA or the pool address, verified on chain rather than assumed "
                "from the label"
            ),
        ))

    # -- the comparison ----------------------------------------------------
    direct_addresses = [a for a in (curve_address, pool_address) if a]
    pool_match = "UNKNOWN"
    if observed_pair_address and direct_addresses:
        if observed_pair_address in direct_addresses:
            pool_match = "PASS"
            notes.append(
                f"the marked pair {observed_pair_address[:8]}… IS the "
                "chain-verified market account, so the mark and the executable "
                "quote describe the same book"
            )
        else:
            pool_match = "FAIL"
            notes.append(
                f"the marked pair {observed_pair_address[:8]}… is not any of the "
                f"chain-verified market accounts {direct_addresses}; the gap "
                "between their prices is a difference of books, not a "
                "discrepancy in one book"
            )
    elif not observed_pair_address:
        pool_match = "UNKNOWN"
    else:
        notes.append(
            "no chain-verified market account was obtained, so the marked pair "
            "cannot be compared against anything; POOL_MATCH stays UNKNOWN "
            "rather than PASS"
        )

    jup_exact = [k for k in jupiter_pool_keys if k in direct_addresses]
    if jupiter_pool_keys and direct_addresses and not jup_exact:
        notes.append(
            "the Jupiter route names pool keys that are not the chain-verified "
            "market accounts, so the aggregator routed through a different "
            "venue than the direct quote priced"
        )

    market_match = "PASS" if pool_match == "PASS" else "UNKNOWN"
    if pool_match == "FAIL":
        market_match = "FAIL"

    return IdentityReport(
        findings=tuple(findings), pool_match=pool_match,
        market_match=market_match, notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


@dataclass
class LatencyMarks:
    """
    Stage timings for one signal.

    Durations come from :func:`time.monotonic`, which cannot jump backwards when
    the wall clock is adjusted mid-run. Wall-clock timestamps are kept
    separately and used only for CORRELATION - pairing a Telegram message with a
    report - never for a duration.
    """

    #: Wall-clock stamp of the Telegram message, as recorded by the channel.
    #: Only the CROSS-SYSTEM gap from it is meaningful, so it is used for that
    #: and for correlation - never for a stage duration.
    telegram_message_at: float | None = None
    #: Monotonic stamp when this process received the message.
    local_receive_at: float | None = None
    #: Wall-clock stamp of the same receive event. Needed because a Telegram
    #: timestamp and a local monotonic reading live on different clocks, and
    #: subtracting one from the other would be meaningless.
    receive_wall_at: float | None = None
    stages: dict[str, tuple[float, float]] = field(default_factory=dict)
    wall: dict[str, float] = field(default_factory=dict)
    final_decision_at: float | None = None
    #: A provider that reports its own request duration (Jupiter does).
    provider_latencies_ms: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str, moment: float | None = None,
             wall: float | None = None) -> None:
        now = moment if moment is not None else time.monotonic()
        start, _ = self.stages.get(stage, (now, now))
        self.stages[stage] = (start, now)
        self.wall[stage] = wall if wall is not None else time.time()

    def duration_ms(self, start: str, end: str) -> float | None:
        if start not in self.stages or end not in self.stages:
            return None
        return max(0.0, (self.stages[end][1] - self.stages[start][0]) * 1000.0)

    def as_dict(self) -> dict[str, Any]:
        telegram_to_receive = None
        if (
            self.telegram_message_at is not None
            and self.receive_wall_at is not None
        ):
            # Cross-system: the message timestamp came from Telegram's server
            # clock and this one is local, so the difference includes clock skew
            # between the two. Reported because it is operationally useful, and
            # labelled as cross-system so it is not read as a stage duration.
            telegram_to_receive = max(
                0.0, (self.receive_wall_at - self.telegram_message_at) * 1000.0
            )
        total = None
        if self.local_receive_at is not None and self.final_decision_at is not None:
            total = max(0.0, (self.final_decision_at - self.local_receive_at) * 1000.0)
        return {
            "clock": (
                "monotonic for all stage durations; wall-clock only for "
                "cross-system correlation (telegram_to_receive_ms includes "
                "clock skew between Telegram and this host)"
            ),
            "telegram_to_receive_ms_cross_system": telegram_to_receive,
            "telegram_to_receive_ms": telegram_to_receive,
            "receive_to_dexscreener_ms": self.duration_ms("receive", "dexscreener"),
            "dexscreener_to_rpc_ms": self.duration_ms("dexscreener", "rpc"),
            "rpc_to_pump_direct_ms": self.duration_ms("rpc", "pump_direct"),
            "rpc_to_pumpswap_direct_ms": self.duration_ms("rpc", "pumpswap_direct"),
            "jupiter_quote_ms": self.provider_latencies_ms.get(
                "jupiter", self.duration_ms("jupiter_start", "jupiter_end")
            ),
            "provider_reported_latencies_ms": dict(sorted(self.provider_latencies_ms.items())),
            "identity_comparison_ms": self.duration_ms("rpc", "identity"),
            "total_signal_to_final_decision_ms": total,
            "stages_ms": {
                name: max(0.0, (end - start) * 1000.0)
                for name, (start, end) in sorted(self.stages.items())
            },
            "wall_clock_timestamps": dict(sorted(self.wall.items())),
        }


__all__ = [
    "CrossCheck",
    "DivergenceClass",
    "DIAGNOSTIC_PAIRS",
    "IdentityFinding",
    "IdentityReport",
    "LatencyMarks",
    "PairGap",
    "PriceLabel",
    "PricePoint",
    "build_identity_report",
    "cross_check",
]
