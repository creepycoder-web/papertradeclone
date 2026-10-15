"""Live READ-ONLY probe: validate Raptor-primary with Jupiter fallback.

For each real alert token it measures the exact fallback decision the executor
would make: try Raptor; if Raptor has no route (or its transaction is
oversized), try Jupiter; then record the selected provider. No signing, no
broadcast, no key material sent anywhere, Raptor's /send-transaction is never
called, and no wSOL/ATA is created on chain.

    .venv/bin/python scripts/provider_fallback_probe.py --out PATH
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
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
from autopaper.execution.solana import MAX_SOLANA_TX_BYTES, SolanaRpc, WSOL_MINT  # noqa: E402

POSITION_LAMPORTS = 1_071_843
SLIPPAGE_BPS = 500
PRIORITY_FEE_LAMPORTS = 50_000

TOKENS: list[tuple[str, str, str]] = [
    ("A_AI_task", "H2LWQMQdEGmARqEQh8v4jiwBhW3hh1iqcDwU815xJ881t", "$AI task CA"),
    ("A_AI_repo", "H2LWQMQdEGamrEQh8v4jiwBhW3hh1iqcDwU815xJ881t", "$AI incident mint"),
    ("B_nonpump", "DcJsfF4F2Gj2Hw7SwjbLyn8PfTCXSjCZrPwMWdMQLL7", "non-Pump.fun"),
    ("C_pump1", "AxprAjtnzxPFoj91qifAYcD2V2KST3J2eSAn2xBPpump", "Pump.fun"),
    ("C_pump2", "Tp3HMESa9hRC9T3XJQZrhFricvH87ifBwMVZk7Qpump", "Pump.fun"),
    ("D_migrated", "9p3NuCz29u7KUsjfrZcBPNGB2pryDpACggjSjYWbkpds", "migrated/PumpSwap"),
    ("E_raydium", "2cuM75BppZWghS74hZXapyQonz4KKd36PeVkCUYqSTNK", "Meteora/Raydium"),
    ("E_raydium2", "JAccgVwonRw28jZ3fHdgwtyAfKJRPd5qMqaVP2xipump", "Raydium/Meteora"),
    ("F_fallback_moon", "C5yy7SB3tY81ZMpr4tgjVDcp6teSLRKrPvGktggNmoon", "Raptor-unsupported -> Jupiter"),
    ("F_fallback_other", "D8qiGA5eSA2E3JNyM89J5Qu9YJzWFYMZBFysXWgtJRYd", "Raptor-unsupported -> Jupiter"),
    ("F_fallback_pump", "6BkYiUKGVjrCy5AuU6zWoeL1iyCQhNKKxiL4WeXPpump", "Raptor-unsupported, Jupiter oversized"),
]


async def _payer() -> tuple[str, str]:
    try:
        from autopaper.config import build_config
        path = build_config().real.wallet_keypair_path
        if path:
            from autopaper.execution.solana import SolanaWallet
            return SolanaWallet(path).pubkey, "operator_wallet_public_only"
    except Exception:  # noqa: BLE001
        pass
    return str(Keypair().pubkey()), "throwaway_key"


def _measure(tx_b64: str) -> dict:
    rep = transaction_size_report(tx_b64)
    try:
        vtx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
        rep["instruction_count"] = len(vtx.message.instructions)
        rep["account_count"] = len(vtx.message.account_keys)
        rep["unsigned"] = all(bytes(s) == bytes(64) for s in vtx.signatures)
    except Exception as exc:  # noqa: BLE001
        rep["decode_error"] = type(exc).__name__
    return rep


async def _rap_probe(rap, rpc, mint) -> dict:
    t0 = time.perf_counter()
    try:
        q = await rap.quote(WSOL_MINT, mint, POSITION_LAMPORTS, SLIPPAGE_BPS)
    except RaptorError as exc:
        return {"status": exc.status, "detail": str(exc)[:180], "route": None,
                "tx_bytes": None, "fits": False, "simulation": None}
    except Exception as exc:  # noqa: BLE001
        return {"status": "ERROR", "detail": type(exc).__name__, "route": None,
                "tx_bytes": None, "fits": False, "simulation": None}
    t_quote = time.perf_counter()
    try:
        built = await rap.build_transaction(
            q, _PAYER, wrap_and_unwrap_sol=True,
            priority_fee=PRIORITY_FEE_LAMPORTS)
    except RaptorError as exc:
        return {"status": exc.status, "detail": str(exc)[:180],
                "route": q.route_labels, "tx_bytes": None, "fits": False,
                "simulation": None}
    m = _measure(built.transaction_b64)
    out = {"status": "OK", "detail": "", "route": q.route_labels,
           "selected_dex": q.selected_dex, "route_hops": len(q.route_steps),
           "tx_bytes": m.get("raw_bytes"), "fits": bool(m.get("ok")),
           "over_by": m.get("over_by"), "unsigned": m.get("unsigned"),
           "simulation": None,
           "latency_ms": {"quote": round((t_quote - t0) * 1000, 2),
                          "build": round((time.perf_counter() - t_quote) * 1000, 2)}}
    if rpc is not None and m.get("ok"):
        ts = time.perf_counter()
        sim = await rpc.simulate_transaction_detailed(built.transaction_b64)
        out["simulation"] = sim.status
        out["latency_ms"]["simulate"] = round(
            (time.perf_counter() - ts) * 1000, 2)
    return out


def _jup_status(exc: Exception) -> str:
    if isinstance(exc, QuoteRejected):
        return "JUPITER_UNSUPPORTED_ROUTE"
    if isinstance(exc, MicroOrderUnsupported):
        text = str(exc).lower()
        if any(m in text for m in ("not tradable", "invalid", "no route",
                                   "unknown token", "not found")):
            return "JUPITER_UNSUPPORTED_ROUTE"
        return "MICRO_REAL_EXECUTION_UNSUPPORTED"
    if isinstance(exc, QuoteUnavailable):
        return "JUPITER_QUOTE_FAILED"
    return type(exc).__name__


async def _jup_probe(jup, rpc, mint) -> dict:
    t0 = time.perf_counter()
    try:
        q = await jup.quote(WSOL_MINT, mint, POSITION_LAMPORTS, SLIPPAGE_BPS)
    except Exception as exc:  # noqa: BLE001
        return {"status": _jup_status(exc), "detail": str(exc)[:180],
                "route": None, "tx_bytes": None, "fits": False,
                "simulation": None}
    t_quote = time.perf_counter()
    try:
        # wrap_and_unwrap_sol=True matches the executor's fallback build: the
        # transaction must be self-contained rather than relying on a
        # pre-provisioned wSOL account.
        built = await jup.build_transaction(
            q, _PAYER, wrap_and_unwrap_sol=True,
            priority_fee=PRIORITY_FEE_LAMPORTS)
    except Exception as exc:  # noqa: BLE001
        return {"status": "JUPITER_BUILD_FAILED", "detail": type(exc).__name__,
                "route": q.route_labels, "tx_bytes": None, "fits": False,
                "simulation": None}
    m = _measure(built.transaction_b64)
    out = {"status": "OK", "detail": "", "route": q.route_labels,
           "tx_bytes": m.get("raw_bytes"), "fits": bool(m.get("ok")),
           "over_by": m.get("over_by"), "unsigned": m.get("unsigned"),
           "simulation": None,
           "latency_ms": {"quote": round((t_quote - t0) * 1000, 2),
                          "build": round((time.perf_counter() - t_quote) * 1000, 2)}}
    if rpc is not None and m.get("ok"):
        ts = time.perf_counter()
        sim = await rpc.simulate_transaction_detailed(built.transaction_b64)
        out["simulation"] = sim.status
        out["simulation_detail"] = (sim.detail or "")[:160]
        out["latency_ms"]["simulate"] = round(
            (time.perf_counter() - ts) * 1000, 2)
    return out


_PAYER = "11111111111111111111111111111111"


async def probe(out_path: Path | None) -> dict:
    global _PAYER
    payer, payer_source = await _payer()
    _PAYER = payer
    rap = RaptorProvider()
    jup = JupiterProvider()
    rpc = None
    try:
        from autopaper.config import build_config
        rpc_url = build_config().real.rpc_url
        if rpc_url:
            rpc = SolanaRpc(rpc_url)
    except Exception:  # noqa: BLE001
        rpc = None

    rows = []
    stats = {"raptor_ok": 0, "raptor_unsupported": 0, "raptor_other_fail": 0,
             "raptor_fit": 0, "raptor_total": 0,
             "jupiter_fallback_attempts": 0, "jupiter_fallback_ok": 0,
             "jupiter_total": 0, "jupiter_fit": 0,
             "both_unsupported": 0, "selected_raptor": 0,
             "selected_jupiter": 0, "blocked": 0}

    for key, mint, note in TOKENS:
        rap_res = await _rap_probe(rap, rpc, mint)
        stats["raptor_total"] += 1
        if rap_res["status"] == "OK":
            stats["raptor_ok"] += 1
            if rap_res["fits"]:
                stats["raptor_fit"] += 1
        elif rap_res["status"] == "RAPTOR_UNSUPPORTED_ROUTE":
            stats["raptor_unsupported"] += 1
        else:
            stats["raptor_other_fail"] += 1

        jup_res = None
        selected = None
        reason = ""
        # Fallback only when Raptor has no route or is oversized.
        needs_fallback = (
            rap_res["status"] in ("RAPTOR_UNSUPPORTED_ROUTE",)
            or (rap_res["status"] == "OK" and not rap_res["fits"])
        )
        if rap_res["status"] == "OK" and rap_res["fits"]:
            selected, reason = "raptor", "raptor routed and fit"
        elif needs_fallback:
            stats["jupiter_fallback_attempts"] += 1
            jup_res = await _jup_probe(jup, rpc, mint)
            stats["jupiter_total"] += 1
            if jup_res["status"] == "OK":
                if jup_res["fits"]:
                    stats["jupiter_fit"] += 1
                    stats["jupiter_fallback_ok"] += 1
                    selected, reason = "jupiter", (
                        f"raptor {rap_res['status']}; jupiter routed and fit")
                else:
                    selected, reason = "blocked", (
                        f"raptor {rap_res['status']}; jupiter oversized")
            else:
                if ("UNSUPPORTED" in jup_res["status"].upper()
                        or jup_res["status"] == "QuoteRejected"):
                    stats["both_unsupported"] += 1
                selected, reason = "blocked", (
                    f"raptor {rap_res['status']}; jupiter {jup_res['status']}")
        else:
            selected, reason = "blocked", f"raptor {rap_res['status']}"

        if selected == "raptor":
            stats["selected_raptor"] += 1
        elif selected == "jupiter":
            stats["selected_jupiter"] += 1
        else:
            stats["blocked"] += 1

        rows.append({
            "token": key, "mint": mint, "note": note,
            "raptor_status": rap_res["status"],
            "raptor_route": rap_res.get("route"),
            "raptor_tx_bytes": rap_res.get("tx_bytes"),
            "raptor_simulation": rap_res.get("simulation"),
            "raptor_latency_ms": rap_res.get("latency_ms"),
            "jupiter_status": (jup_res or {}).get("status"),
            "jupiter_route": (jup_res or {}).get("route"),
            "jupiter_tx_bytes": (jup_res or {}).get("tx_bytes"),
            "jupiter_simulation": (jup_res or {}).get("simulation"),
            "jupiter_latency_ms": (jup_res or {}).get("latency_ms"),
            "selected_provider": selected,
            "final_reason": reason,
        })
        print(f"[{key}] raptor={rap_res['status']} "
              f"({rap_res.get('tx_bytes')}) -> selected={selected} :: {reason}")

    n = len(rows)
    metrics = {
        "tokens": n,
        "raptor_routable_rate_pct": round(100.0 * stats["raptor_ok"] / n, 1),
        "jupiter_fallback_rate_pct": round(
            100.0 * stats["selected_jupiter"] / n, 1),
        "both_unsupported_rate_pct": round(
            100.0 * stats["both_unsupported"] / n, 1),
        "raptor_fit_rate_pct": (round(100.0 * stats["raptor_fit"]
                                      / stats["raptor_ok"], 1)
                                if stats["raptor_ok"] else 0.0),
        "jupiter_fit_rate_pct": (round(100.0 * stats["jupiter_fit"]
                                       / stats["jupiter_total"], 1)
                                 if stats["jupiter_total"] else 0.0),
        "overall_candidate_fit_rate_pct": round(
            100.0 * (stats["selected_raptor"] + stats["selected_jupiter"]) / n, 1),
        "counts": stats,
    }
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raptor_endpoint": rap.base_url,
        "jupiter_endpoint": jup.base_url,
        "payer_source": payer_source,
        "position_lamports": POSITION_LAMPORTS,
        "slippage_bps": SLIPPAGE_BPS,
        "priority_fee_lamports": PRIORITY_FEE_LAMPORTS,
        "tx_limit_bytes": MAX_SOLANA_TX_BYTES,
        "simulation_run": rpc is not None,
        "broadcast_performed": False,
        "private_key_shared": False,
        "metrics": metrics,
        "rows": rows,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {out_path}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = asyncio.run(probe(Path(args.out) if args.out else None))
    print("\n== metrics ==")
    print(json.dumps(rep["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
