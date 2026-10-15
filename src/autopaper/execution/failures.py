"""
Execution-failure taxonomy and on-chain error parsing.

WHY THIS EXISTS
---------------
Every REAL execution failure used to reach the operator as
``BUY_BLOCKED=execution_exception`` with the real reason buried in the detail
text. That collapses genuinely different failures into one word:

* a route that does not exist anywhere (nothing to do),
* a quote that came back malformed or stale (nothing to do),
* a simulation the chain rejected (investigate),
* a DEX program refusing the swap with a ``Custom`` error (venue/pool state),
* the node refusing the transaction (``rpc_rejected``),
* a broadcast whose outcome is genuinely unknown (STOP, do not retry).

The operator action differs in each case, so the category must be explicit and
machine-readable, and the raw evidence (program id, instruction index, custom
code, logs) must be carried alongside it.

CUSTOM ERRORS ARE PROGRAM-SCOPED
--------------------------------
``{"InstructionError": [6, {"Custom": 6001}]}`` is NOT a universal code. Anchor
offsets a program's ``#[error_code]`` enum by 6000, so ``6001`` is the SECOND
custom error *of whichever program emitted it* -- and the name is defined by
that program's own IDL. This module therefore classifies it as
``DEX_PROGRAM_ERROR`` and NAMES the program and instruction index; it never
asserts a meaning the emitting program has not told us.
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# explicit failure categories
# ---------------------------------------------------------------------------

#: No provider can construct a route for this mint/notional.
ROUTE_UNAVAILABLE = "route_unavailable"
#: The quote itself was refused, malformed, stale, or below the venue's minimum.
QUOTE_INVALID = "quote_invalid"
#: The transaction simulated and the program refused it.
SIMULATION_FAILED = "simulation_failed"
#: Price impact / slippage bound breached.
SLIPPAGE_EXCEEDED = "slippage_exceeded"
#: A DEX program emitted a ``Custom`` error. Pool/venue state refused the trade.
DEX_PROGRAM_ERROR = "dex_program_error"
#: The node (or our local size gate) rejected the transaction itself.
RPC_REJECTED = "rpc_rejected"
#: We do not know whether the transaction happened. Fail closed; never retry.
TRANSACTION_UNKNOWN = "transaction_unknown"
#: Last resort. Deliberately a DIFFERENT name from ``execution_exception`` so a
#: reader can tell a categorized failure from an uncategorized one.
EXECUTION_FAILED = "execution_failed"

#: The categories an operator surface may report as ``BUY_BLOCKED=<category>``.
FAILURE_CATEGORIES = (
    ROUTE_UNAVAILABLE,
    QUOTE_INVALID,
    SIMULATION_FAILED,
    SLIPPAGE_EXCEEDED,
    DEX_PROGRAM_ERROR,
    RPC_REJECTED,
    TRANSACTION_UNKNOWN,
    EXECUTION_FAILED,
)

#: Explicit status -> category. Anything not listed falls through to the
#: heuristics below and then to EXECUTION_FAILED.
_STATUS_CATEGORY = {
    # -- route ------------------------------------------------------------
    "RAPTOR_UNSUPPORTED_ROUTE": ROUTE_UNAVAILABLE,
    "JUPITER_UNSUPPORTED_ROUTE": ROUTE_UNAVAILABLE,
    "REAL_BUY_BLOCKED_NO_VALID_EXECUTION_ROUTE": ROUTE_UNAVAILABLE,
    "REAL_BUY_BLOCKED": ROUTE_UNAVAILABLE,
    "ROUTE_REQUIREMENTS_UNSATISFIED": ROUTE_UNAVAILABLE,
    # -- quote ------------------------------------------------------------
    "RAPTOR_QUOTE_FAILED": QUOTE_INVALID,
    "RAPTOR_INVALID_RESPONSE": QUOTE_INVALID,
    "RAPTOR_NETWORK_ERROR": QUOTE_INVALID,
    "RAPTOR_STALE_QUOTE": QUOTE_INVALID,
    "JUPITER_QUOTE_FAILED": QUOTE_INVALID,
    "JUPITER_STALE_QUOTE": QUOTE_INVALID,
    "QUOTE_REJECTED": QUOTE_INVALID,
    "QUOTE_UNAVAILABLE": QUOTE_INVALID,
    "MICRO_REAL_EXECUTION_UNSUPPORTED": QUOTE_INVALID,
    # -- simulation -------------------------------------------------------
    "RAPTOR_SIMULATION_FAILED": SIMULATION_FAILED,
    "JUPITER_SIMULATION_FAILED": SIMULATION_FAILED,
    "SIMULATION_FAILED_ALL_CANDIDATES": SIMULATION_FAILED,
    # -- node / size ------------------------------------------------------
    "RAPTOR_TX_TOO_LARGE": RPC_REJECTED,
    "BROADCAST_REJECTED": RPC_REJECTED,
    "SIGN_FAILED": RPC_REJECTED,
    # -- unknown ----------------------------------------------------------
    "AMBIGUOUS": TRANSACTION_UNKNOWN,
    "CHAIN_TIMEOUT": TRANSACTION_UNKNOWN,
}

#: Public program ids, for NAMING the program that emitted an error. These are
#: stable identifiers; their error-code MEANINGS are NOT asserted here.
KNOWN_PROGRAMS = {
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P": "Pump.fun",
    "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA": "PumpSwap",
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium AMM v4",
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": "Raydium CLMM",
    "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C": "Raydium CPMM",
    "LanMV9sAd7wArD4vJFi2qDdfnVhFxYSUg6eADduJ3uj": "Raydium LaunchLab",
    "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo": "Meteora DLMM",
    "Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB":
        "Meteora Dynamic AMM",
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "Jupiter v6",
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": "SPL Token",
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb": "Token-2022",
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL":
        "Associated Token Account",
    "ComputeBudget111111111111111111111111111111": "ComputeBudget",
    "11111111111111111111111111111111": "System",
}


def program_label(program_id: str) -> str:
    """A human name for a program id, or ``UNKNOWN(...)``. Never guesses."""
    if not program_id:
        return "(unresolved)"
    return KNOWN_PROGRAMS.get(program_id, f"UNKNOWN({program_id[:12]}…)")


# ---------------------------------------------------------------------------
# instruction-error parsing
# ---------------------------------------------------------------------------


def parse_instruction_error(err: Any) -> tuple[Optional[int], str, Optional[int]]:
    """
    Parse an ``err`` from ``simulateTransaction`` / ``getTransaction``.

    Returns ``(instruction_index, error_kind, custom_code)``.

    ``error_kind`` is ``"Custom"`` when the program emitted a numeric custom
    error, the name of a built-in error (e.g. ``"NotEnoughAccountKeys"``)
    otherwise, or ``""`` when nothing could be parsed. Every field is optional
    because an unrecognised shape must not be coerced into a meaning.
    """
    if isinstance(err, dict):
        payload = err.get("InstructionError")
        if isinstance(payload, (list, tuple)) and payload:
            idx = payload[0] if isinstance(payload[0], int) else None
            body = payload[1] if len(payload) > 1 else None
            if isinstance(body, dict) and "Custom" in body:
                try:
                    return idx, "Custom", int(body["Custom"])
                except (TypeError, ValueError):
                    return idx, "Custom", None
            if body is None:
                return idx, "", None
            return idx, str(body), None
        # A non-InstructionError (e.g. a blockhash expiry) is not an ix error.
        return None, "", None
    if isinstance(err, str):
        return None, err, None
    return None, "", None


def looks_like_custom_error(detail: str) -> bool:
    text = (detail or "").lower()
    return "instructionerror" in text and "custom" in text


def classify_execution_failure(
    status: str,
    *,
    detail: str = "",
    custom_code: Optional[int] = None,
    program_id: str = "",
    route_metadata: Optional[dict] = None,
) -> str:
    """
    Map a fill status (plus its evidence) to one explicit category.

    The order matters: an unknown/ambiguous outcome and a custom program error
    are the two most consequential and are checked first, because classifying
    either as a generic "simulation failed" would hide the fact that a DEX
    program refused the trade, or that we do not know whether it happened.
    """
    s = (status or "").upper()
    d = detail or ""

    # 1. a custom program error is the most specific fact available
    if custom_code is not None or looks_like_custom_error(d):
        return DEX_PROGRAM_ERROR

    # 2. an unknown outcome must never be reported as a plain refusal
    if s in ("AMBIGUOUS", "CHAIN_TIMEOUT") or s.startswith("CHAIN_UNKNOWN"):
        return TRANSACTION_UNKNOWN
    if "unknown" in s and "SIMULATION" not in s:
        return TRANSACTION_UNKNOWN

    # 3. slippage / impact
    if s.endswith("SLIPPAGE_EXCEEDED") or "price impact" in d.lower() \
            or "slippage" in d.lower():
        return SLIPPAGE_EXCEEDED

    # 4. explicit table
    if s in _STATUS_CATEGORY:
        return _STATUS_CATEGORY[s]

    # 5. heuristics for prefixes without an exact entry
    if "NO_VALID_EXECUTION_ROUTE" in s or "UNSUPPORTED_ROUTE" in s:
        return ROUTE_UNAVAILABLE
    if "SIMULATION" in s:
        return SIMULATION_FAILED
    if "BROADCAST" in s or "TX_TOO_LARGE" in s or "RPC" in s:
        return RPC_REJECTED
    if "QUOTE" in s or "MICRO" in s or "STALE" in s:
        return QUOTE_INVALID

    # 6. honest last resort
    return EXECUTION_FAILED


def failure_telemetry(
    *,
    status: str,
    detail: str = "",
    provider: str = "",
    category: str = "",
    http_status: Optional[int] = None,
    rpc_code: Optional[int] = None,
    program_id: str = "",
    instruction_index: Optional[int] = None,
    custom_code: Optional[int] = None,
    quote_at: Optional[float] = None,
    simulation_at: Optional[float] = None,
    price_impact_pct: Optional[float] = None,
    slippage_bps: Optional[int] = None,
    fallback_attempted: bool = False,
    fallback_result: str = "",
) -> dict:
    """
    The audit record for one failed REAL entry.

    Contains no secrets: no key material, no signed bytes, no transaction
    payload. ``detail`` is the caller's already-sanitized reason.
    """
    return {
        "status": status,
        "category": category or classify_execution_failure(
            status, detail=detail, custom_code=custom_code,
            program_id=program_id),
        "provider": provider,
        "http_status": http_status,
        "rpc_code": rpc_code,
        "program_id": program_id,
        "program_label": program_label(program_id),
        "instruction_index": instruction_index,
        "custom_code": custom_code,
        "quote_at": quote_at,
        "simulation_at": simulation_at,
        "price_impact_pct": price_impact_pct,
        "slippage_bps": slippage_bps,
        "fallback_attempted": bool(fallback_attempted),
        "fallback_result": fallback_result,
        "detail": (detail or "")[:400],
    }
