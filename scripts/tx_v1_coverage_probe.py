"""Live READ-ONLY coverage probe: Raptor -> Jupiter -> V1 reconstruction.

WHAT THIS MEASURES
------------------
For a sample of the bot's OWN alert mints (taken from ``var/candidates.jsonl``,
the candidate log the bot writes), at the EXACT production micro notional, this
script measures the decision the executor would make:

    Raptor quote -> Raptor build -> detect version -> size gate for that version
        -> if oversized v0/legacy: resolve ALT, rebuild as v1, re-gate
        -> simulate
        -> (only if Raptor cannot serve) Jupiter quote -> build -> same treatment

and classifies the outcome into the fixed vocabulary below.

WHAT IT DOES NOT DO
-------------------
* It never signs, and the private key is never USED for anything. The
  operator keypair FILE is opened, and only to derive the public address the
  transaction is built for -- building for a real address is what makes the
  measurement representative. The key itself is never used to sign anything,
  never leaves the process, and is never logged, journalled or transmitted.
  With no keypair configured, a throwaway public key is used instead and the
  report says so.
* It never broadcasts. ``sendTransaction`` is not called on any provider or on
  the RPC; only ``simulateTransaction`` is.
* It never shares key material with anything. A provider only ever receives the
  public address, exactly as in production.
* It never creates a wSOL account, an ATA, or any other on-chain infrastructure.
* It never resizes the order. The notional is fixed at 1 071 843 lamports, and a
  provider that refuses it is recorded as a refusal, not retried larger.

Usage
-----
    .venv/bin/python scripts/tx_v1_coverage_probe.py --limit 100 --out PATH
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solders.keypair import Keypair  # noqa: E402
from solders.transaction import VersionedTransaction  # noqa: E402

from autopaper.execution.jupiter import (  # noqa: E402
    JupiterProvider,
    MicroOrderUnsupported,
    QuoteRejected,
    QuoteUnavailable,
)
from autopaper.execution.raptor import (  # noqa: E402
    RaptorError,
    RaptorProvider,
    transaction_size_report,
)
from autopaper.execution.solana import WSOL_MINT, SolanaRpc  # noqa: E402
from autopaper.execution.txcodec import (  # noqa: E402
    MAX_BYTES_V1,
    MAX_BYTES_V0,
    is_alt_load_refusal,
    reconstruct_as_v1,
)

#: The production notional. NOT changed by this probe, and never increased to
#: satisfy a provider minimum -- a venue that will not take this size is a
#: MICRO_REAL_EXECUTION_UNSUPPORTED answer.
POSITION_LAMPORTS = 1_071_843
SLIPPAGE_BPS = 500
PRIORITY_FEE_LAMPORTS = 50_000

#: The classification vocabulary. Every sample lands in exactly one of these,
#: and they are kept DISTINCT on purpose: "the route does not exist" and "the
#: node would not answer" and "the transaction is too big" have different fixes,
#: and collapsing them into one "failed" bucket is how a real capability gap
#: gets mistaken for a flaky afternoon.
#
# SUCCESSES
RAPTOR_SUCCESS = "RAPTOR_SUCCESS"
JUPITER_SUCCESS = "JUPITER_SUCCESS"
V1_RECONSTRUCTION_SUCCESS = "V1_RECONSTRUCTION_SUCCESS"
#: The rebuild ran, produced a fitting v1 transaction, and it simulated clean.
#: Distinct from the two above because the executable path is DIFFERENT: the
#: transaction the bot would send is one the provider never built.
# FAILURES, each with its own reason
BOTH_UNSUPPORTED = "BOTH_UNSUPPORTED"
TX_TOO_LARGE = "TX_TOO_LARGE"
SIMULATION_FAILED = "SIMULATION_FAILED"
ALT_RELATED_FAILURE = "ALT_RELATED_FAILURE"
RPC_FAILURE = "RPC_FAILURE"
QUOTE_OR_BUILD_FAILURE = "QUOTE_OR_BUILD_FAILURE"
INVALID_RESPONSE = "INVALID_RESPONSE"
NO_LIQUIDITY = "NO_LIQUIDITY"
OTHER = "OTHER"

#: Outcomes that mean an executable candidate existed and simulated clean.
SUCCESS_CLASSES = (RAPTOR_SUCCESS, JUPITER_SUCCESS,
                   V1_RECONSTRUCTION_SUCCESS)

#: Failure classes where the run learned NOTHING about the token: a transport
#: failure is not evidence that a token is untradable. They are reported
#: separately rather than counted as unsupported, and the harness retries them.
INCONCLUSIVE_CLASSES = (RPC_FAILURE, OTHER)

#: Bounded retries for TRANSPORT failures, in the measurement harness only.
#:
#: This is NOT a production behaviour change. The executor deliberately does not
#: retry a provider on a network error, because a provider outage must not
#: trigger a second attempt against the other provider. A coverage measurement
#: has the opposite obligation: a timeout measures the harness's luck, not the
#: pipeline's capability. Every retry is counted and reported.
TRANSPORT_RETRY_ATTEMPTS = 4
TRANSPORT_RETRY_BACKOFF_S = 2.0

#: Venue classification is by PROGRAM ID present in the built transaction and by
#: the provider's own route metadata. NEVER by ticker, name or mint suffix: a
#: "pump"-suffixed mint can be on any venue, and a token with an innocent name
#: is very often a fresh Pump.fun launch.
VENUE_BY_PROGRAM = {
    "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA": "PumpSwap",
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P": "Pump.fun",
    "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C": "Raydium",
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium",
    "routeUGWgWzqBWFcrCfv8tritsqukccJPu3q5GPP3xS": "Jupiter Router",
    "dbcij3LWUppWqq96dh6gJWwBifmcGfLSB5D4DuUaqg": "Meteora",
    "Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB": "Meteora",
    "cpamdpZCGKUy5JxQXB4dcpGPiikHawvSWAd6mEn1sGG": "Meteora",
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "Orca",
    "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "Orca",
    "ComputeBudget111111111111111111111111111111": None,   # not a venue
    "11111111111111111111111111111111": None,              # not a venue
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": None,
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1w7Zm4Yr5tHZwrmZDe": None,
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL": None,
    "So11111111111111111111111111111111111111112": None,
    "Sysvar1nstructions1111111111111111111111111": None,
    "SysvarRent111111111111111111111111111111111": None,
    "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr": None,
}
VENUE_BY_LABEL = {
    "pumpswap": "PumpSwap", "pump swap": "PumpSwap", "pumpfun": "Pump.fun",
    "pump.fun": "Pump.fun", "pump": "Pump.fun",
    "raydium": "Raydium", "clmm": "Raydium", "raydiumclmm": "Raydium",
    "raydiumlaunchlab": "Raydium", "amm-v4": "Raydium",
    "meteora": "Meteora", "meteoraDlmmPool": "Meteora",
    "meteoraVault": "Meteora", "damm2": "Meteora", "damm": "Meteora",
    "orca": "Orca", "whirlpool": "Orca",
    "moonshot": "Moonshot/Moonit", "moonit": "Moonshot/Moonit",
    "heaven": "Heaven", "bonk": "Bonk", "bonkswap": "Bonk",
    "sugar": "Sugar", "raydiumlaunchlabclmm": "Raydium",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def alert_mints(limit: int) -> list:
    """The bot's own recent alert mints, most recent last, deduplicated.

    Read from the candidate log rather than any hard-coded list, so the sample
    IS the alert universe rather than a convenient subset of it.
    """
    path = Path(__file__).resolve().parents[1] / "var" / "candidates.jsonl"
    seen: list = []
    known = set()
    if not path.exists():
        return seen
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        mint = row.get("mint") or row.get("token") or row.get("output_mint")
        if mint and mint not in known:
            known.add(mint)
            seen.append(mint)
    # Most recent alerts are the most representative of what the bot faces now.
    return seen[-limit:] if limit else seen


def _venue(transaction_b64: str, labels) -> tuple:
    """Venue from on-chain program ids first, provider labels second.

    Returns ``(venue, evidence)``. The program ids are read out of the built
    transaction, which is the strongest evidence available without a program
    account scan; the provider's own route labels are the fallback. A mint whose
    name ends in "pump" contributes NOTHING here on purpose.
    """
    programs: list = []
    try:
        message = VersionedTransaction.from_bytes(
            base64.b64decode(transaction_b64)).message
        programs = sorted({str(ix.program_id) for ix in message.instructions})
    except Exception:  # noqa: BLE001
        pass
    for program in programs:
        if program in VENUE_BY_PROGRAM and VENUE_BY_PROGRAM[program]:
            return VENUE_BY_PROGRAM[program], {
                "basis": "program_id", "program_id": program,
                "programs_seen": programs}
    for label in labels or []:
        mapped = VENUE_BY_LABEL.get(str(label).strip().lower())
        if mapped:
            return mapped, {"basis": "provider_label", "label": label,
                            "programs_seen": programs}
    return "other", {"basis": "unmatched", "labels": list(labels or []),
                     "programs_seen": programs}


def _status_of(exc: Exception) -> str:
    """Map a provider exception onto the coverage vocabulary."""
    status = getattr(exc, "status", None)
    if status:
        return status
    if isinstance(exc, MicroOrderUnsupported):
        return "MICRO_REAL_EXECUTION_UNSUPPORTED"
    if isinstance(exc, QuoteRejected):
        return "QUOTE_REJECTED"
    if isinstance(exc, QuoteUnavailable):
        return "QUOTE_UNAVAILABLE"
    if isinstance(exc, RaptorError):
        return getattr(exc, "status", "RAPTOR_ERROR")
    return type(exc).__name__


#: Substrings that mean "the call never reached a decision", as opposed to
#: "the venue reached a decision and it was no". Only these are retried.
_TRANSPORT_MARKERS = (
    "network error", "networkerror", "transport", "timed out", "timeout",
    "timeouterror", "httpx", "connect", "read timeout", "remoteprotocol",
    "rpc error", "rpc timed out", "rpc transport", "rate limit", "too many",
    "502", "503", "504", "429", "temporary failure", "name or service",
)


def is_transport_failure(status: str, detail: str) -> bool:
    """True when nothing was learned about the token.

    Deliberately narrow. A 429 is included because a rate-limited venue has not
    expressed an opinion about the token; an HTTP 500 carrying a route refusal is
    NOT, because that IS an opinion.
    """
    text = f"{status} {detail}".lower()
    if "unsupported_route" in text or "quote_rejected" in text:
        return False
    if "not tradable" in text or "no route" in text:
        return False
    return any(m in text for m in _TRANSPORT_MARKERS)


async def _with_transport_retry(coro_factory, *, attempts: int
                                = TRANSPORT_RETRY_ATTEMPTS):
    """Call ``coro_factory()``, retrying ONLY transport failures.

    Returns ``(result, exception, retries)``. The retry count is reported so the
    report can state how much of the run was affected by provider throttling
    rather than pretending the pipeline was measured cleanly.
    """
    retries = 0
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await coro_factory(), None, retries
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not is_transport_failure(_status_of(exc), str(exc)):
                return None, exc, retries
            retries += 1
            if attempt + 1 < attempts:
                await asyncio.sleep(TRANSPORT_RETRY_BACKOFF_S * (attempt + 1))
    return None, last, retries


def _is_route_refusal(status: str, detail: str) -> bool:
    """Did the provider say "this token cannot be traded", or something else?

    The distinction matters: "no route" means the token is outside the venue's
    coverage, while a timeout means nothing was learned and must not be counted
    as evidence the token is untradable.
    """
    text = f"{status} {detail}".lower()
    if "network" in text or "timeout" in text or "transport" in text:
        return False
    return any(m in text for m in (
        "unsupported_route", "quote_rejected", "quote_failed", "no route",
        "no multi-hop", "unable to fetch pools", "invalid mint",
        "invalid output mint", "invalid input mint", "not tradable", "no pools",
        "quote_unavailable", "invalid response", "no_pools",
    ))


# ---------------------------------------------------------------------------
# per-provider measurement
# ---------------------------------------------------------------------------
async def _measure_provider(provider, rpc, mint: str, *, jupiter: bool
                            ) -> dict:
    """quote -> build -> version-aware size gate -> v1 rescue -> simulate."""
    out = {
        "provider": provider.provider_name(),
        "quote": "NOT_ATTEMPTED", "build": "NOT_ATTEMPTED",
        "transaction_version": None, "raw_bytes": None, "max_bytes": None,
        "over_by": None, "size_ok": False, "simulation": None,
        "simulation_ok": False, "simulation_detail": "", "units": None,
        "route": [], "v1_rescue": None, "alt_load_rescue": None,
        "selected_tx": None, "venue": None,
        "venue_evidence": {}, "status": "", "detail": "",
        "quote_transport_failure": False, "build_transport_failure": False,
        "sim_transport_failure": False,
    }

    # -- quote
    t0 = time.perf_counter()
    try:
        quote = await provider.quote(WSOL_MINT, mint, POSITION_LAMPORTS,
                                     SLIPPAGE_BPS)
    except Exception as exc:  # noqa: BLE001
        out["quote"] = _status_of(exc)
        out["status"] = out["quote"]
        out["detail"] = str(exc)[:200]
        out["route_refusal"] = _is_route_refusal(out["status"], out["detail"])
        out["quote_transport_failure"] = bool(
            not out["route_refusal"]
            and is_transport_failure(out["quote"], out["detail"]))
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    out["quote"] = "OK"
    out["quote_out_amount"] = getattr(quote, "expected_out_amount", None)
    out["route"] = list(getattr(quote, "route_labels", []) or [])
    t_quote = time.perf_counter()

    # -- build (UNSIGNED; the venue never signs and never broadcasts)
    try:
        built = await provider.build_transaction(
            quote, _PAYER, wrap_and_unwrap_sol=True,
            priority_fee=PRIORITY_FEE_LAMPORTS)
    except Exception as exc:  # noqa: BLE001
        out["build"] = _status_of(exc)
        out["status"] = out["build"]
        out["detail"] = str(exc)[:200]
        out["route_refusal"] = _is_route_refusal(out["status"], out["detail"])
        out["build_transport_failure"] = bool(
            not out["route_refusal"]
            and is_transport_failure(out["build"], out["detail"]))
        out["latency_ms"] = {"quote": round((t_quote - t0) * 1000, 1)}
        return out
    out["build"] = "OK"
    t_build = time.perf_counter()

    tx_b64 = built.transaction_b64
    size = transaction_size_report(tx_b64)
    out["transaction_version"] = size.get("transaction_version")
    out["raw_bytes"] = size.get("raw_bytes")
    out["max_bytes"] = size.get("max_bytes")
    out["over_by"] = size.get("over_by")
    out["size_ok"] = bool(size.get("ok"))
    out["venue"], out["venue_evidence"] = _venue(tx_b64, out["route"])

    # -- v1 rescue: only for an oversized legacy/v0 transaction
    if not out["size_ok"] and out["transaction_version"] in ("legacy", "v0"):
        out["v1_rescue"] = await _rescue(rpc, tx_b64,
                                          getattr(built,
                                                  "prioritization_fee_lamports",
                                                  None))
        if out["v1_rescue"].get("ok"):
            tx_b64 = out["v1_rescue"]["transaction_b64"]
            size = transaction_size_report(tx_b64)
            out["transaction_version"] = size.get("transaction_version")
            out["raw_bytes"] = size.get("raw_bytes")
            out["max_bytes"] = size.get("max_bytes")
            out["over_by"] = size.get("over_by")
            out["size_ok"] = bool(size.get("ok"))
            out["venue"], out["venue_evidence"] = _venue(tx_b64, out["route"])
    out["size_detail"] = size.get("detail", "")
    # -- simulate (UNSIGNED, sigVerify off; nothing is broadcast)
    if out["size_ok"] and rpc is not None:
        out["simulation"] = await _simulate(rpc, tx_b64, out)
        # ALT-LOAD RESCUE. A v0 the node refuses to load is rejected whatever its
        # size, and v1 has no tables to load. This does not skip simulation: the
        # v0 WAS simulated and failed; the rebuild is simulated again from
        # scratch through the same gate.
        if (not out["simulation_ok"]
                and is_alt_load_refusal(out["simulation_detail"])
                and out["transaction_version"] in ("v0", "legacy")):
            out["alt_load_rescue"] = await _rescue(
                rpc, tx_b64,
                getattr(built, "prioritization_fee_lamports", None))
            if out["alt_load_rescue"].get("ok"):
                tx_b64 = out["alt_load_rescue"]["transaction_b64"]
                size = transaction_size_report(tx_b64)
                out["transaction_version"] = size.get("transaction_version")
                out["raw_bytes"] = size.get("raw_bytes")
                out["max_bytes"] = size.get("max_bytes")
                out["over_by"] = size.get("over_by")
                out["size_ok"] = bool(size.get("ok"))
                out["simulation"] = await _simulate(rpc, tx_b64, out)
                out["simulation_after_rescue"] = out["simulation"]
                out["venue"], out["venue_evidence"] = _venue(tx_b64, out["route"])
        out["selected_tx"] = tx_b64 if out["size_ok"] else None

    out["latency_ms"] = {
        "quote": round((t_quote - t0) * 1000, 1),
        "build": round((time.perf_counter() - t_build) * 1000, 1),
    }
    out["route_refusal"] = False
    return out


async def _simulate(rpc, tx_b64: str, out: dict) -> str:
    try:
        outcome = await asyncio.wait_for(
            rpc.simulate_transaction_detailed(tx_b64), timeout=45)
    except asyncio.TimeoutError:
        out["simulation_ok"] = False
        out["simulation_detail"] = "the node did not answer within 45s"
        out["sim_transport_failure"] = True
        return "SIMULATION_TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        out["simulation_ok"] = False
        out["simulation_detail"] = str(exc)[:160]
        out["sim_transport_failure"] = is_transport_failure(
            "rpc", str(exc))
        return f"SIMULATION_ERROR:{type(exc).__name__}"
    out["simulation_ok"] = bool(outcome.ok)
    out["simulation_detail"] = (outcome.detail or "")[:200]
    out["units"] = outcome.units_consumed
    return outcome.status


async def _rescue(rpc, tx_b64: str, priority_fee) -> dict:
    """Re-encode an oversized v0/legacy transaction as v1, with the proof."""
    report = {"attempted": True, "ok": False, "status": "", "detail": "",
              "verification": {}, "resource_config": {}, "transaction_b64": None,
              "source_version": None, "source_raw_bytes": None,
              "v1_raw_bytes": None, "max_bytes": MAX_BYTES_V1,
              "source_limit_bytes": MAX_BYTES_V0}
    try:
        raw = base64.b64decode(tx_b64)
        message = VersionedTransaction.from_bytes(raw).message
        tables = [str(l.account_key) for l in message.address_table_lookups]
        report["source_raw_bytes"] = len(raw)
        report["source_version"] = "v0" if tables else (
            "v0" if hasattr(message, "address_table_lookups") else "legacy")
    except Exception as exc:  # noqa: BLE001
        report["status"] = "DECODE_FAILED"
        report["detail"] = type(exc).__name__
        return report
    if rpc is None:
        report["status"] = "NO_RPC_FOR_ALT_RESOLUTION"
        return report
    try:
        resolved, unresolved = await asyncio.wait_for(
            rpc.resolve_lookup_tables(tables), timeout=45)
    except Exception as exc:  # noqa: BLE001
        report["status"] = "ALT_READ_FAILED"
        report["detail"] = f"{type(exc).__name__}: {exc}"[:160]
        return report
    if unresolved:
        report["status"] = "ALT_UNRESOLVED"
        report["detail"] = json.dumps(unresolved)[:200]
        return report
    rec = reconstruct_as_v1(tx_b64, resolved_tables=resolved,
                            priority_fee_lamports=priority_fee)
    report.update(ok=rec.ok, status=rec.status, detail=rec.detail[:300],
                  verification=rec.verification,
                  resource_config=rec.resource_config,
                  transaction_b64=rec.transaction_b64,
                  v1_raw_bytes=rec.raw_bytes)
    return report


# ---------------------------------------------------------------------------
# per-mint outcome
# ---------------------------------------------------------------------------
def _provider_verdict(m: Optional[dict]) -> dict:
    """Reduce one provider's measurement to the facts the classifier needs."""
    if not m:
        return {"reached_a_decision": False, "routable": False,
                "size_valid": False, "simulated_ok": False,
                "rescued": False, "rescue_attempted": False,
                "rescue_reason": None, "rescue_status": None,
                "failure": None, "transport": False}
    rescue = m.get("v1_rescue") or {}
    alt_rescue = m.get("alt_load_rescue") or {}
    used_rescue = rescue if rescue.get("attempted") else (
        alt_rescue if alt_rescue.get("attempted") else {})
    routed = m.get("quote") == "OK" and m.get("build") == "OK"
    transport = (not routed) and bool(
        m.get("quote_transport_failure") or m.get("build_transport_failure")
        or m.get("sim_transport_failure"))
    if m.get("quote") == "OK" and m.get("build") == "NOT_ATTEMPTED":
        failure = "BUILD_NOT_ATTEMPTED"
    elif m.get("quote") != "OK":
        failure = m.get("quote")
    elif m.get("build") != "OK":
        failure = m.get("build")
    elif not m.get("size_ok"):
        failure = (used_rescue.get("status") or "TX_TOO_LARGE")
    elif not m.get("simulation_ok"):
        failure = m.get("simulation")
    else:
        failure = None
    return {
        "reached_a_decision": bool(routed or not transport),
        "routable": routed,
        "size_valid": bool(m.get("size_ok")),
        "simulated_ok": bool(m.get("simulation_ok")),
        "rescued": bool(used_rescue.get("ok")),
        "rescue_attempted": bool(used_rescue.get("attempted")),
        "rescue_reason": ("ALT_LOAD_REFUSED"
                          if alt_rescue.get("ok") else
                          ("TX_TOO_LARGE" if rescue.get("attempted")
                           else None)),
        "rescue_status": used_rescue.get("status"),
        "failure": failure,
        "transport": transport,
    }


def _classify(rap: dict, jup: Optional[dict]) -> str:
    """One classification per mint, chosen by the DEEPEST reason reached.

    Ordering matters and is deliberate: a success is only claimed when a
    size-valid candidate actually SIMULATED clean, and a failure is attributed to
    the last thing that was learned rather than to the first branch that matched.
    A provider that routed but whose transaction would not fit is a SIZE
    failure, not "unsupported" -- the token is routable.
    """
    r = _provider_verdict(rap)
    j = _provider_verdict(jup) if jup else None

    # 1. An executable candidate that simulated clean. V1 wins the label
    #    because the bytes on the wire are ones no provider built.
    for verdict, direct, rescued in ((r, RAPTOR_SUCCESS,
                                     V1_RECONSTRUCTION_SUCCESS),
                                    (j, JUPITER_SUCCESS,
                                     V1_RECONSTRUCTION_SUCCESS)):
        if verdict and verdict["size_valid"] and verdict["simulated_ok"]:
            return rescued if verdict["rescued"] else direct

    # 2. A candidate existed and passed the size gate but would not simulate.
    for verdict in (r, j):
        if verdict and verdict["size_valid"] and not verdict["simulated_ok"]:
            detail = (rap if verdict is r else (jup or {})).get(
                "simulation_detail") or ""
            if is_alt_load_refusal(detail):
                return ALT_RELATED_FAILURE
            return SIMULATION_FAILED

    # 3. Both providers routed but neither transaction could be made sendable.
    routed = [v for v in (r, j) if v and v["routable"]]
    if routed:
        statuses = {v["rescue_status"] for v in routed if v["rescue_status"]}
        if any(str(s).startswith("ALT_") for s in statuses):
            return ALT_RELATED_FAILURE
        if any(v["rescue_attempted"] for v in routed):
            # A rebuild was tried and did not produce a fitting v1: the size
            # envelope is what stopped it.
            return TX_TOO_LARGE
        return TX_TOO_LARGE

    # 4. A provider reached a "this token cannot be traded" decision.
    refusals = [v for v in (r, j) if v and v["reached_a_decision"]
                and str(v["failure"] or "").find("UNSUPPORTED") >= 0
                or str(v["failure"] or "") in ("QUOTE_REJECTED",
                                               "RAPTOR_QUOTE_FAILED")]
    if refusals:
        return BOTH_UNSUPPORTED if len(refusals) >= 2 else (
            BOTH_UNSUPPORTED if (r and j and not r["transport"]
                                 and not j["transport"]) else
            BOTH_UNSUPPORTED)

    # 5. A provider answered with something unusable, or not at all.
    for verdict in (r, j):
        if verdict and verdict["transport"]:
            return RPC_FAILURE
    for verdict, m in ((r, rap), (j, jup)):
        if verdict and m:
            status = f"{m.get('quote')} {m.get('build')}"
            if "INVALID" in status:
                return INVALID_RESPONSE
    for verdict in (r, j):
        if verdict and str(verdict["failure"] or "").startswith("QUOTE"):
            return QUOTE_OR_BUILD_FAILURE
    if any((m or {}).get("sim_transport_failure") for m in (rap, jup)):
        return RPC_FAILURE
    return OTHER


# ---------------------------------------------------------------------------
# per-mint outcome
# ---------------------------------------------------------------------------
async def probe_mint(raptor, jupiter, rpc, mint: str) -> dict:
    """The executor's decision for one mint, measured rather than assumed."""
    row = {"mint": mint, "notional_lamports": POSITION_LAMPORTS,
           "slippage_bps": SLIPPAGE_BPS, "at": time.time(),
           "at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    rap = await _measure_provider(raptor, rpc, mint, jupiter=False)
    row["raptor"] = rap

    # The fallback fires ONLY for a route refusal or an unrescuable oversized
    # transaction, exactly as in the executor. A provider outage must never
    # cause a second attempt -- so a transport failure is re-asked rather than
    # answered by the other provider, and the retries are counted.
    rap_servable = (rap["quote"] == "OK" and rap["build"] == "OK"
                    and rap["size_ok"])
    rap_verdict = _provider_verdict(rap)
    needs_fallback = (
        rap.get("route_refusal")
        or (rap["quote"] == "OK" and rap["build"] == "OK" and not rap["size_ok"]
            and not (rap.get("v1_rescue") or {}).get("ok")
            and not (rap.get("alt_load_rescue") or {}).get("ok"))
    )
    if rap_verdict["transport"] and not needs_fallback:
        # Nothing was learned: re-ask the SAME provider rather than asking a
        # different one, which would measure the wrong thing.
        again = await _measure_provider(raptor, rpc, mint, jupiter=False)
        row["raptor_transport_retry"] = True
        rap = again
        row["raptor"] = rap
        rap_servable = (rap["quote"] == "OK" and rap["build"] == "OK"
                        and rap["size_ok"])
        needs_fallback = (
            rap.get("route_refusal")
            or (rap["quote"] == "OK" and rap["build"] == "OK"
                and not rap["size_ok"]
                and not (rap.get("v1_rescue") or {}).get("ok")
                and not (rap.get("alt_load_rescue") or {}).get("ok"))
        )
    row["fallback_attempted"] = bool(needs_fallback)

    jup = None
    if needs_fallback:
        jup = await _measure_provider(jupiter, rpc, mint, jupiter=True)
        if _provider_verdict(jup)["transport"]:
            jup = await _measure_provider(jupiter, rpc, mint, jupiter=True)
            row["jupiter_transport_retry"] = True
    row["jupiter"] = jup

    row["classification"] = _classify(rap, jup)

    jup_verdict = _provider_verdict(jup) if jup else None
    chosen = None
    if rap_servable and (rap["simulation_ok"] or not rap["simulation"]):
        chosen = ("raptor", rap)
    elif jup_verdict and jup_verdict["size_valid"]:
        chosen = ("jupiter", jup)

    row["selected_provider"] = chosen[0] if chosen else None
    if chosen:
        row["venue"] = chosen[1]["venue"]
        row["venue_evidence"] = chosen[1]["venue_evidence"]
        row["transaction_version"] = chosen[1]["transaction_version"]
        row["raw_bytes"] = chosen[1]["raw_bytes"]
        row["max_bytes"] = chosen[1]["max_bytes"]
        row["over_by"] = chosen[1]["over_by"]
        row["route"] = chosen[1]["route"]
        row["simulation"] = chosen[1]["simulation"]
    else:
        row["venue"] = (rap["venue"] if rap["venue"] != "other"
                        else ((jup or {}).get("venue") or "other"))

    # The ladder the coverage percentage is built from, plus the reason the v1
    # rescue was entered, per provider.
    for label, m in (("raptor", rap), ("jupiter", jup)):
        if m is None:
            continue
        verdict = _provider_verdict(m)
        row[f"{label}_route_status"] = m.get("quote")
        row[f"{label}_transaction_version"] = m.get("transaction_version")
        row[f"{label}_raw_bytes"] = m.get("raw_bytes")
        row[f"{label}_max_bytes"] = m.get("max_bytes")
        row[f"{label}_failure_reason"] = verdict["failure"]
        row[f"{label}_v1_rescue_attempted"] = verdict["rescue_attempted"]
        row[f"{label}_v1_rescue_reason"] = verdict["rescue_reason"]
        row[f"{label}_v1_rescue_status"] = verdict["rescue_status"]
        row[f"{label}_v1_raw_bytes"] = (
            (m.get("alt_load_rescue") or m.get("v1_rescue") or {})
            .get("v1_raw_bytes"))
        row[f"{label}_simulation"] = m.get("simulation")
        row[f"{label}_route"] = m.get("route")

    row["quoteable"] = bool(rap["quote"] == "OK" or (jup and jup["quote"] == "OK"))
    row["buildable"] = bool(rap["build"] == "OK" or (jup and jup["build"] == "OK"))
    row["size_valid"] = bool(rap_servable or (jup_verdict
                                              and jup_verdict["size_valid"]))
    row["size_valid_without_v1"] = bool(
        (rap["build"] == "OK" and rap["size_ok"]
         and not _provider_verdict(rap)["rescued"])
        or (jup and jup["build"] == "OK" and jup["size_ok"]
            and not _provider_verdict(jup)["rescued"]))
    row["simulation_valid"] = bool(chosen and chosen[1]["simulation_ok"])
    row["v1_rescued"] = bool(_provider_verdict(rap)["rescued"]
                             or (jup_verdict or {}).get("rescued"))
    return row


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def summarise(rows: list) -> dict:
    total = len(rows)
    def count(pred):
        return sum(1 for r in rows if pred(r))

    classes: dict = {}
    for r in rows:
        classes[r["classification"]] = classes.get(r["classification"], 0) + 1
    venues: dict = {}
    for r in rows:
        venues[r.get("venue") or "unknown"] = venues.get(r.get("venue") or "unknown", 0) + 1

    sim_attempted = [r for r in rows
                     if (r.get("selected_provider")
                         and ((r.get(r["selected_provider"]) or {}).get("simulation")))]
    sim_ok = count(lambda r: r.get("simulation_valid"))

    def pct(n):
        return round(100.0 * n / total, 1) if total else 0.0

    before = count(lambda r: r.get("size_valid_without_v1"))
    after = count(lambda r: r.get("size_valid"))
    successes = count(lambda r: r["classification"] in SUCCESS_CLASSES)
    rescued = count(lambda r: r.get("v1_rescued"))
    rescue_attempted = count(lambda r: r.get("raptor_v1_rescue_attempted")
                             or r.get("jupiter_v1_rescue_attempted"))
    inconclusive = count(lambda r: r["classification"] in INCONCLUSIVE_CLASSES
                         or r["classification"] == RPC_FAILURE)
    transport_retries = count(lambda r: r.get("raptor_transport_retry")
                              or r.get("jupiter_transport_retry"))
    return {
        "alerts_sampled": total,
        "classification_counts": dict(sorted(classes.items())),
        "venue_counts": dict(sorted(venues.items(), key=lambda kv: -kv[1])),
        "funnel": {
            "quoteable": count(lambda r: r["quoteable"]),
            "quoteable_pct": pct(count(lambda r: r["quoteable"])),
            "buildable": count(lambda r: r["buildable"]),
            "buildable_pct": pct(count(lambda r: r["buildable"])),
            "size_valid_before_v1": before,
            "execution_candidate_coverage_before_v1_pct": pct(before),
            "size_valid_after_v1": after,
            "execution_candidate_coverage_after_v1_pct": pct(after),
            "coverage_improvement_pct_points": round(pct(after) - pct(before), 1),
            "simulation_valid": sim_ok,
            "simulation_success_rate_pct": pct(sim_ok),
            "simulation_attempted": len(sim_attempted),
        },
        "headline": {
            "final_successful": successes,
            "final_successful_pct": pct(successes),
            "final_unsupported": total - successes,
            "final_unsupported_pct": pct(total - successes),
            "v1_rescue_attempted": rescue_attempted,
            "v1_rescue_succeeded": rescued,
            "v1_rescue_success_rate_pct": (round(100.0 * rescued / rescue_attempted, 1)
                                           if rescue_attempted else 0.0),
            "inconclusive_samples": inconclusive,
            "transport_retry_samples": transport_retries,
            "note": (
                "Coverage is reported for the SAMPLED alert universe only, at one "
                "timestamp, at the 1 071 843 lamport notional, on the configured "
                "public RPC. It is not a statement about every Solana token. "
                "'final_successful' means a size-valid transaction was produced "
                "AND it simulated clean on mainnet; nothing was broadcast."),
        },
        "coverage": {
            "RAPTOR_COVERAGE": count(
                lambda r: r["selected_provider"] == "raptor"),
            "RAPTOR_COVERAGE_PCT": pct(count(
                lambda r: r["selected_provider"] == "raptor")),
            "JUPITER_COVERAGE": count(
                lambda r: r["selected_provider"] == "jupiter"),
            "JUPITER_COVERAGE_PCT": pct(count(
                lambda r: r["selected_provider"] == "jupiter")),
            "V1_RESCUE_COVERAGE": count(
                lambda r: r["classification"] == V1_RECONSTRUCTION_SUCCESS),
            "V1_RESCUE_COVERAGE_PCT": pct(count(
                lambda r: r["classification"] == V1_RECONSTRUCTION_SUCCESS)),
            "V1_RESCUE_ATTEMPTED": rescue_attempted,
            "UNSUPPORTED_COVERAGE": count(
                lambda r: r["classification"] == BOTH_UNSUPPORTED),
            "UNSUPPORTED_COVERAGE_PCT": pct(count(
                lambda r: r["classification"] == BOTH_UNSUPPORTED)),
            "SIMULATION_SUCCESS_RATE_PCT": pct(sim_ok),
        },
        "size_distribution": _size_stats(rows),
    }



def _size_stats(rows: list) -> dict:
    out: dict = {}
    for version in ("legacy", "v0", "v1"):
        sizes = [r["raw_bytes"] for r in rows
                 if r.get("transaction_version") == version
                 and r.get("raw_bytes")]
        if not sizes:
            continue
        sizes.sort()
        out[version] = {
            "count": len(sizes),
            "min": sizes[0],
            "median": statistics.median(sizes),
            "p95": sizes[min(int(0.95 * (len(sizes) - 1)), len(sizes) - 1)],
            "max": sizes[-1],
            "limit": MAX_BYTES_V1 if version == "v1" else MAX_BYTES_V0,
        }
    # The rescue is the point of the exercise, so its before/after is reported
    # in full rather than summarised: each row is (v0 bytes, v0 limit, v1 bytes,
    # v1 limit, whether the rebuild then simulated clean).
    rescues = []
    for r in rows:
        for key in ("raptor", "jupiter"):
            provider_row = r.get(key) or {}
            rescue = (provider_row.get("v1_rescue")
                      or provider_row.get("alt_load_rescue") or {})
            if not rescue.get("attempted"):
                continue
            rescues.append({
                "mint": r["mint"], "provider": key,
                "trigger": ("ALT_LOAD_REFUSED"
                            if (provider_row.get("alt_load_rescue") or {}).get("ok")
                            else "TX_TOO_LARGE"),
                "source_version": rescue.get("source_version"),
                "source_raw_bytes": rescue.get("source_raw_bytes"),
                "source_max_bytes": rescue.get("source_limit_bytes"),
                "v1_raw_bytes": rescue.get("v1_raw_bytes"),
                "v1_max_bytes": rescue.get("max_bytes"),
                "ok": bool(rescue.get("ok")),
                "status": rescue.get("status"),
                "simulation": (r.get(key) or {}).get("simulation"),
                "selected": r.get("selected_provider") == key,
            })
    if rescues:
        out["v1_rescues"] = {
            "attempted": len(rescues),
            "succeeded": sum(1 for x in rescues if x["ok"]),
            "rescued_and_simulated_ok": sum(
                1 for x in rescues if x["ok"] and x["simulation"] == "SIMULATION_OK"),
            "rows": rescues,
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
_PAYER = "11111111111111111111111111111111"


async def main(limit: int, out_path: Path, pause: float) -> None:
    global _PAYER
    payer_source = "throwaway_public_key"
    try:
        from autopaper.config import build_config
        from autopaper.execution.solana import SolanaWallet
        path = build_config().real.wallet_keypair_path
        if path:
            _PAYER = SolanaWallet(path).pubkey
            payer_source = "operator_wallet_PUBLIC_ADDRESS_ONLY"
    except Exception:  # noqa: BLE001
        pass

    mints = alert_mints(limit)
    raptor = RaptorProvider()
    jupiter = JupiterProvider()
    rpc = None
    rpc_url = ""
    try:
        from autopaper.config import build_config
        rpc_url = build_config().real.rpc_url or ""
        if rpc_url:
            rpc = SolanaRpc(rpc_url, timeout=40.0)
    except Exception:  # noqa: BLE001
        rpc = None

    print(f"probe: {len(mints)} alert mints, notional {POSITION_LAMPORTS} "
          f"lamports, payer source {payer_source}, simulation-only",
          flush=True)

    rows: list = []
    started = time.time()
    for i, mint in enumerate(mints, 1):
        try:
            row = await asyncio.wait_for(
                probe_mint(raptor, jupiter, rpc, mint), timeout=180)
        except asyncio.TimeoutError:
            row = {"mint": mint, "classification": OTHER,
                   "detail": "probe timeout", "raptor": {}, "jupiter": None,
                   "quoteable": False, "buildable": False, "size_valid": False,
                   "size_valid_without_v1": False, "simulation_valid": False,
                   "venue": "unknown"}
        except Exception as exc:  # noqa: BLE001
            row = {"mint": mint, "classification": OTHER,
                   "detail": f"probe error {type(exc).__name__}",
                   "raptor": {}, "jupiter": None, "quoteable": False,
                   "buildable": False, "size_valid": False,
                   "size_valid_without_v1": False, "simulation_valid": False,
                   "venue": "unknown"}
        rows.append(row)
        print(f"[{i}/{len(mints)}] {row['classification']:28s} "
              f"v={row.get('transaction_version')} "
              f"b={row.get('raw_bytes')} venue={row.get('venue')} "
              f"({round(time.time() - started)}s)", flush=True)
        if pause:
            await asyncio.sleep(pause)
        if out_path and i % 10 == 0:
            out_path.write_text(json.dumps(rows, indent=1, default=str))

    summary = summarise(rows)
    payload = {
        "probe": "tx_v1_coverage_probe",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": {
            "notional_lamports": POSITION_LAMPORTS,
            "slippage_bps": SLIPPAGE_BPS,
            "priority_fee_lamports": PRIORITY_FEE_LAMPORTS,
            "v0_limit_bytes": MAX_BYTES_V0,
            "v1_limit_bytes": MAX_BYTES_V1,
            "rpc_url": rpc_url or "(none)",
            "payer_source": payer_source,
            "sample": "most recent unique mints from var/candidates.jsonl",
        },
        "guarantees": {
            "BROADCAST_PERFORMED": False,
            "REAL_TRADING_ENABLED": False,
            "private_key_never_shared": True,
            "no_real_BUY": True,
            "no_real_SELL": True,
            "no_WSOL_setup": True,
            "no_ATA_setup": True,
            "notional_never_increased": True,
            "L25_changed": False,
            "simulation_only": True,
        },
        "summary": summary,
        "rows": rows,
    }
    if out_path:
        out_path.write_text(json.dumps(payload, indent=1, default=str))
        print(f"wrote {out_path}", flush=True)
    print(json.dumps(summary, indent=1), flush=True)
    await raptor.aclose()
    await jupiter.aclose()
    if rpc is not None:
        await rpc.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds between mints, to respect provider quotas")
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    asyncio.run(main(args.limit, out, args.pause))
