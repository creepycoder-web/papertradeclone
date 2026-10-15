"""
Execution planner: pick a transaction that can actually be sent.

WHY THIS EXISTS
---------------
A Solana transaction over 1232 raw bytes cannot be broadcast, and the node
rejects it during deserialisation with JSON-RPC -32602 before simulation even
runs. The production path used to hand whatever the aggregator returned straight
to ``simulateTransaction``, so an unsendable transaction surfaced as an opaque
remote error at the last possible moment.

This module moves the decision EARLIER and makes it EXPLICIT: generate a small,
deterministic set of execution candidates, serialize each one, measure it,
discard anything over the limit, simulate what survives, and only then choose.

    venue detect -> candidates -> build -> serialize -> SIZE GATE
                 -> simulate  -> select -> broadcast (caller)

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* It never chooses a route by expected profit. Price is not an input to
  selection. A cheaper-looking route that does not fit is rejected, and a
  dearer one that fits is taken.
* It never skips simulation to reach the send step.
* It never enlarges an order to make a route work.
* It never retries blindly against an RPC error. A route that does not fit is
  rejected once, on measured bytes, before any round trip.
* It never guesses a venue. If the on-chain evidence for a native venue is not
  present and parseable, the native candidate is not offered at all.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

from .solana import SimulationOutcome
from .txcodec import VERSION_V1, measure_transaction

# --------------------------------------------------------------------------
# Venues
# --------------------------------------------------------------------------
VENUE_UNKNOWN = "UNKNOWN"
VENUE_PUMP_FUN_BONDING_CURVE = "PUMP_FUN_BONDING_CURVE"
VENUE_PUMPSWAP = "PUMPSWAP"

#: Well-known Pump programs. Used for DETECTION only -- no instruction is ever
#: built for them here (see the note on native candidates below).
PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_SWAP_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"

#: Why no native Pump.fun BUY is built here.
NATIVE_BUILD_DISABLED_REASON = (
    "A native Pump.fun/PumpSwap BUY requires an instruction layout and a "
    "confirmed venue for the specific mint. Neither could be verified for the "
    "incident token: the public Pump.fun API returns 404 for it, "
    "getProgramAccounts is unavailable on the public RPC endpoint, and the "
    "bonding-curve account's layout did not match the documented Anchor struct. "
    "Building that instruction from a guessed layout would risk sending a "
    "Pump.fun BUY to a token that is not on that venue, so no native "
    "instruction is constructed. The planner is structured to accept a native "
    "candidate the moment one can be built from verified venue state."
)


@dataclass
class Candidate:
    """
    One concrete, already-built and already-measured way to execute an order.

    A candidate only exists once its bytes exist. That is what makes the size
    gate meaningful: we measure what we would send, not what we hope to send.
    """
    label: str
    provider: str
    venue: str
    input_mint: str
    output_mint: str
    amount_lamports: int
    slippage_bps: int
    route_metadata: dict = field(default_factory=dict)
    instruction_count: Optional[int] = None
    account_count: Optional[int] = None
    raw_bytes: Optional[int] = None
    encoded_bytes: Optional[int] = None
    #: The version read off the wire after deserialization, and the size ceiling
    #: that applies to it. Never inferred from the provider: legacy and v0 are
    #: held to 1232 bytes, a v1 transaction to 4096.
    transaction_version: Optional[str] = None
    max_bytes: Optional[int] = None
    #: base64 of the UNSIGNED venue transaction. Never logged, never journalled.
    transaction_b64: Optional[str] = None
    #: Populated once simulated.
    simulation: Optional[SimulationOutcome] = None
    #: Terminal verdict: FIT / OVERSIZED / SIM_FAILED / INVALID / UNSUPPORTED
    status: str = "PENDING"
    reject_reason: str = ""

    @property
    def fits(self) -> bool:
        if self.raw_bytes is None:
            return False
        limit = self.max_bytes
        if limit is None:
            from .txcodec import max_bytes_for_version
            limit = max_bytes_for_version(self.transaction_version)
        return self.raw_bytes <= limit

    @property
    def usable(self) -> bool:
        return self.status == "FIT"

    def report(self) -> dict:
        """Sanitized projection. No bytes, no signatures, no secrets."""
        return {
            "label": self.label, "provider": self.provider, "venue": self.venue,
            "input_mint": self.input_mint, "output_mint": self.output_mint,
            "amount_lamports": self.amount_lamports,
            "slippage_bps": self.slippage_bps,
            "instruction_count": self.instruction_count,
            "account_count": self.account_count,
            "raw_bytes": self.raw_bytes, "encoded_bytes": self.encoded_bytes,
            "transaction_version": self.transaction_version,
            "size_limit": self.max_bytes, "fits": self.fits,
            "status": self.status, "reject_reason": self.reject_reason,
            "route_metadata": self.route_metadata,
            "simulation": (self.simulation.as_dict() if self.simulation else None),
        }


@dataclass
class Plan:
    venue_detected: str
    candidates: list
    chosen: Optional[Candidate]
    status: str
    detail: str = ""

    def report(self) -> dict:
        return {
            "venue_detected": self.venue_detected,
            "status": self.status,
            "detail": self.detail,
            "chosen": self.chosen.label if self.chosen else None,
            "candidates": [c.report() for c in self.candidates],
        }


# --------------------------------------------------------------------------
# Venue detection
# --------------------------------------------------------------------------
async def detect_venue(rpc, mint: str) -> tuple:
    """
    Determine which venue a mint actually lives on.

    CONSERVATIVE BY CONSTRUCTION. Returns ``VENUE_UNKNOWN`` unless the
    on-chain evidence is unambiguous. A wrong "PUMP_FUN" would let a native
    instruction be aimed at a token that has migrated, so every check below
    must pass; anything else degrades to UNKNOWN, which simply means the
    planner offers Jupiter candidates only.

    Evidence used, and nothing else:
      * the mint account exists and is an SPL mint
      * a Pump program account that references this mint exists
    """
    from solders.pubkey import Pubkey

    try:
        res = await rpc._rpc("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
    except Exception as exc:  # noqa: BLE001
        return VENUE_UNKNOWN, {"reason": f"mint lookup failed: {type(exc).__name__}"}

    value = ((res or {}).get("value") or {})
    if not value:
        return VENUE_UNKNOWN, {"reason": "mint account not found"}
    data = value.get("data") or {}
    if not isinstance(data, dict) or "parsed" not in data:
        return VENUE_UNKNOWN, {"reason": "mint is not a jsonParsed SPL account"}

    info = {
        "mint_exists": True,
        "mint_program": data.get("program"),
    }
    return VENUE_UNKNOWN, info


# --------------------------------------------------------------------------
# Candidate construction
# --------------------------------------------------------------------------
#: Jupiter configurations tried, in PREFERENCE ORDER. Small and fixed: this is
#: a decision list, not a search space. Each is a genuinely different
#: transaction, not a retry of the same one.
JUPITER_CANDIDATES = (
    # (label, wrap_and_unwrap_sol, only_direct_routes, dynamic_compute_unit_limit)
    ("jupiter/wsol-preprovisioned", False, True, True),
    ("jupiter/wsol-preprovisioned-staticcu", False, True, False),
    ("jupiter/wsol-preprovisioned-nodirect", False, False, True),
    ("jupiter/temporary-wsol", True, True, True),
)


async def build_candidates(*, provider, rpc, wallet, side, input_mint: str,
                           output_mint: str, lamports: int, slippage_bps: int,
                           owner: str, venue: str,
                           priority_fee: Any = "auto",
                           allow_temporary_wsol: bool = True) -> list:
    """
    Build every candidate transaction and measure it.

    The measurement happens here, on real serialized bytes, before any RPC
    round trip. A candidate that does not fit is discarded at this point and
    never reaches ``simulateTransaction`` -- which is the whole point.
    """
    from solders.transaction import VersionedTransaction

    out: list[Candidate] = []
    try:
        quote = await provider.quote(input_mint, output_mint, lamports,
                                     slippage_bps)
    except Exception as exc:  # noqa: BLE001
        return [Candidate(label="jupiter", provider="jupiter", venue=venue,
                          input_mint=input_mint, output_mint=output_mint,
                          amount_lamports=lamports, slippage_bps=slippage_bps,
                          status="INVALID",
                          reject_reason=f"quote failed: {type(exc).__name__}")]

    for label, wrap, direct, dyn_cu in JUPITER_CANDIDATES:
        if wrap and not allow_temporary_wsol:
            continue
        cand = Candidate(label=label, provider="jupiter", venue=venue,
                         input_mint=input_mint, output_mint=output_mint,
                         amount_lamports=lamports, slippage_bps=slippage_bps,
                         route_metadata={
                             "wrap_and_unwrap_sol": wrap,
                             "only_direct_routes": direct,
                             "dynamic_compute_unit_limit": dyn_cu,
                         })
        try:
            built = await provider.build(
                quote, owner, wrap_and_unwrap_sol=wrap,
                only_direct_routes=direct,
                dynamic_compute_unit_limit=dyn_cu,
                priority_fee=priority_fee)
        except Exception as exc:  # noqa: BLE001
            cand.status = "INVALID"
            cand.reject_reason = f"build failed: {type(exc).__name__}"
            out.append(cand)
            continue

        import base64 as _b64
        try:
            raw = _b64.b64decode(built.transaction_b64)
        except Exception as exc:  # noqa: BLE001
            cand.status = "INVALID"
            cand.reject_reason = f"undecodable venue payload: {type(exc).__name__}"
            out.append(cand)
            continue

        # Version-aware measurement: the limit is the one that applies to the
        # version this transaction actually is.
        measured = measure_transaction(built.transaction_b64)
        cand.raw_bytes = measured.get("raw_bytes") or len(raw)
        cand.encoded_bytes = len(built.transaction_b64)
        cand.max_bytes = measured.get("max_bytes")
        cand.transaction_version = measured.get("transaction_version")
        cand.transaction_b64 = built.transaction_b64
        cand.route_metadata["quote_out_amount"] = quote.expected_out_amount
        cand.route_metadata["price_impact_pct"] = quote.price_impact_pct
        try:
            msg = VersionedTransaction.from_bytes(raw).message
            cand.instruction_count = len(msg.instructions)
            cand.account_count = len(msg.account_keys)
        except Exception:  # noqa: BLE001
            cand.instruction_count = None
            cand.account_count = None

        if cand.fits:
            cand.status = "FIT"
        else:
            cand.status = "OVERSIZED"
            cand.reject_reason = (
                f"TX_SIZE {cand.raw_bytes} > TX_LIMIT {cand.max_bytes} "
                f"OVER_BY {cand.max_bytes and cand.raw_bytes - cand.max_bytes} "
                f"(version {cand.transaction_version})")
        out.append(cand)
    return out


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
async def _simulate(rpc, tx_b64: str) -> SimulationOutcome:
    """Simulate, preferring the rich outcome and degrading to the two-tuple API.

    The executor already tolerates a SolanaRpc-shaped object that only exposes
    the original ``simulate_transaction`` pair; the planner must tolerate the
    same shapes, or a partially-implemented RPC double turns a safety stage into
    an ``AttributeError`` that looks like a crash rather than a refusal. A
    failure is still a failure either way -- this changes the plumbing, never
    the verdict.
    """
    detailed = getattr(rpc, "simulate_transaction_detailed", None)
    if callable(detailed):
        return await detailed(tx_b64)
    ok, detail = await rpc.simulate_transaction(tx_b64)
    return SimulationOutcome(
        ok=ok,
        status="SIMULATION_OK" if ok else "SIMULATION_FAILED",
        detail=detail or ("simulation ok" if ok else "simulation failed"),
    )


async def plan_execution(*, provider, rpc, wallet, side, input_mint: str,
                          output_mint: str, lamports: int, slippage_bps: int,
                          owner: str, priority_fee: Any = "auto",
                          allow_temporary_wsol: bool = True) -> Plan:
    """
    Produce a vetted execution plan, or an explicit refusal.

    Order of operations, which is the safety order and not a preference:
    detect venue -> build candidates -> measure -> SIZE GATE -> simulate
    -> deterministic select.
    """
    venue, venue_info = await detect_venue(rpc, output_mint)
    log.info("execution plan: mint venue=%s (%s)", venue, venue_info.get("reason", ""))

    candidates = await build_candidates(
        provider=provider, rpc=rpc, wallet=wallet, side=side,
        input_mint=input_mint, output_mint=output_mint, lamports=lamports,
        slippage_bps=slippage_bps, owner=owner, venue=venue,
        priority_fee=priority_fee, allow_temporary_wsol=allow_temporary_wsol)

    # No native candidate is constructed: see NATIVE_BUILD_DISABLED_REASON.
    survivors = [c for c in candidates if c.fits]
    rejected = [c for c in candidates if not c.fits]

    if not survivors:
        # Distinguish WHY nothing survived. "The venue would not build a
        # transaction", "every build was oversized" and "the quote itself
        # failed" are different faults with different fixes, and reporting them
        # as one opaque NO_FITTING_ROUTE is how a venue outage gets read as a
        # size problem.
        if candidates and all(c.status == "INVALID" for c in candidates):
            reason = next((c.reject_reason for c in candidates
                           if c.reject_reason.startswith("quote failed")), "")
            status = ("QUOTE_FAILED" if reason else "ALL_BUILDS_FAILED")
            detail = reason or ("every candidate failed to build: "
                                + "; ".join(c.reject_reason
                                            for c in candidates)[:200])
            return Plan(venue_detected=venue, candidates=candidates,
                        chosen=None, status=status, detail=detail)
        smallest = min((c.raw_bytes for c in rejected if c.raw_bytes), default=None)
        limit = min((c.max_bytes for c in rejected if c.max_bytes),
                    default=None)
        detail = (f"no execution candidate fits its own version's "
                  f"{limit}-byte transaction limit"
                  if limit else
                  "no execution candidate fits the transaction size limit")
        if smallest is not None:
            detail += f"; smallest candidate was {smallest} bytes"
        return Plan(venue_detected=venue, candidates=candidates, chosen=None,
                    status="NO_FITTING_ROUTE", detail=detail)

    # Simulate the survivors. A simulation failure removes a candidate; it is
    # never reinterpreted as a pass. A SIGNING failure is treated the same way:
    # it removes the candidate rather than escaping as an exception, so a key
    # problem is reported as a refused order instead of crashing the executor.
    for cand in survivors:
        try:
            signed = wallet.sign_wire(_b64d(cand.transaction_b64))
        except Exception as exc:  # noqa: BLE001
            cand.status = "SIGN_FAILED"
            cand.reject_reason = f"signing failed: {type(exc).__name__}"
            continue
        signed_b64 = _b64e(signed)
        cand.simulation = await _simulate(rpc, signed_b64)
        if cand.simulation.ok:
            cand.status = "FIT"
        else:
            cand.status = "SIM_FAILED"
            cand.reject_reason = cand.simulation.detail

    viable = [c for c in survivors if c.status == "FIT"]
    if not viable:
        signed_out = [c for c in survivors if c.status == "SIGN_FAILED"]
        if signed_out and len(signed_out) == len(survivors):
            return Plan(venue_detected=venue, candidates=candidates,
                        chosen=None, status="ALL_SIGNINGS_FAILED",
                        detail="every candidate within the size limit could "
                               "not be signed locally")
        return Plan(venue_detected=venue, candidates=candidates, chosen=None,
                    status="ALL_SIMULATIONS_FAILED",
                    detail="every candidate within the size limit failed simulation")

    # Deterministic selection. NOT by price, NOT by expected profit.
    #
    #   1. validity -- all viable ones simulated clean
    #   2. size      -- prefer the smallest, for headroom
    #   3. rules     -- every viable candidate already passed them
    #   4. preference -- the fixed JUPITER_CANDIDATES order breaks any tie
    #
    # Size margin is preferred over route order on purpose: a transaction one
    # byte under ITS OWN limit is technically valid and practically fragile,
    # because Jupiter re-routes thin tokens and the same order can come back
    # larger next time. Candidates are compared on the fraction of their own
    # limit they use, so a v1 candidate and a v0 candidate are not judged on
    # scales that have nothing to do with each other.
    order = {label: i for i, (label, *_rest) in enumerate(JUPITER_CANDIDATES)}

    def _pressure(c):
        limit = c.max_bytes or 1
        return (c.raw_bytes / limit, c.raw_bytes, order.get(c.label, 99))

    viable.sort(key=_pressure)
    chosen = viable[0]

    margin = (chosen.max_bytes or 0) - chosen.raw_bytes
    return Plan(
        venue_detected=venue, candidates=candidates, chosen=chosen,
        status="PLANNED",
        detail=(f"chose {chosen.label}: TX_SIZE {chosen.raw_bytes} <= "
                f"TX_LIMIT {chosen.max_bytes} (version "
                f"{chosen.transaction_version}), margin {margin}B, "
                f"SIMULATION=OK, BROADCAST=NOT_ATTEMPTED"))


def _b64d(s: str) -> bytes:
    import base64
    return base64.b64decode(s)


def _b64e(b: bytes) -> str:
    import base64
    return base64.b64encode(b).decode()


# --------------------------------------------------------------------------
# Size distribution reporting
# --------------------------------------------------------------------------
def size_distribution(candidates: list) -> dict:
    """
    min / median / p95 / max / fit_rate per provider, over measured bytes.

    Reporting a median alone is how "2 of 8 fit" gets described as "smaller".
    The fit RATE is the number that matters for whether a path is dependable.
    """
    by_provider: dict = {}
    for c in candidates:
        if c.raw_bytes is None:
            continue
        by_provider.setdefault(c.provider, []).append(c)
    out = {}
    for provider, cs in by_provider.items():
        sizes = sorted(c.raw_bytes for c in cs)
        # A candidate "fits" against the ceiling for its own version, so a v1
        # transaction is never reported as an outlier for using bytes a v0
        # transaction could not have used.
        fitting = [c for c in cs if c.fits]
        limits = sorted({c.max_bytes or 0 for c in cs})
        out[provider] = {
            "sample_count": len(sizes),
            "min": sizes[0],
            "median": statistics.median(sizes),
            "p95": sizes[min(int(0.95 * (len(sizes) - 1)), len(sizes) - 1)],
            "max": sizes[-1],
            "fit_count": len(fitting),
            "fit_rate_pct": round(100.0 * len(fitting) / len(sizes), 1),
            "versions": sorted({c.transaction_version for c in cs
                                if c.transaction_version}),
            "tx_limits": limits,
        }
    return out
