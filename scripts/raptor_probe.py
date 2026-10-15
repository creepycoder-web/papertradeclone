"""Live READ-ONLY probe: compare Raptor vs Jupiter transaction footprints.

WHAT THIS DOES (and does NOT do)
--------------------------------
For a set of REAL alert mints taken from this bot's own candidate log, at the
exact production micro notional, this script:

  * asks Raptor for a quote and an UNSIGNED swap transaction,
  * asks Jupiter for the same, for comparison,
  * deserializes and reserializes each transaction and MEASURES the raw bytes,
  * optionally simulates the Raptor transaction on the configured Solana RPC.

It does NOT sign, does NOT broadcast, does NOT send a private key anywhere, does
NOT create a wSOL account and does NOT create an ATA. Raptor's ``/send-transaction``
endpoint is never called. It is a measurement harness, nothing more.

    .venv/bin/python scripts/raptor_probe.py [--out PATH] [--no-simulate]

The payer public key is the configured operator wallet (read for its PUBLIC
address only) when available, otherwise a throwaway key, so the built
transaction's ATA/key shape matches production as closely as possible.
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

from autopaper.execution.jupiter import JupiterProvider  # noqa: E402
from autopaper.execution.raptor import (  # noqa: E402
    RaptorError,
    RaptorProvider,
    transaction_size_report,
)
from autopaper.execution.solana import (  # noqa: E402
    MAX_SOLANA_TX_BYTES,
    SolanaRpc,
    WSOL_MINT,
)

#: The exact production micro notional from the incident
#: (var/real_execution_intents.jsonl requested_amount).
POSITION_LAMPORTS = 1_071_843
SLIPPAGE_BPS = 500
PRIORITY_FEE_LAMPORTS = 50_000

#: Real alert mints drawn from var/candidates.jsonl + the incident record.
#: Venue is NOT inferred from the mint; Raptor decides the route.
TOKENS: list[tuple[str, str, str]] = [
    ("A_AI_repo", "H2LWQMQdEGamrEQh8v4jiwBhW3hh1iqcDwU815xJ881t", "incident $AI"),
    ("A_AI_task", "H2LWQMQdEGmARqEQh8v4jiwBhW3hh1iqcDwU815xJ881t", "task-pasted $AI CA"),
    ("B_nonpump", "DcJsfF4F2Gj2Hw7SwjbLyn8PfTCXSjCZrPwMWdMQLL7", "non-Pump.fun alert"),
    ("C_pump1", "AxprAjtnzxPFoj91qifAYcD2V2KST3J2eSAn2xBPpump", "Pump.fun"),
    ("C_pump2", "Tp3HMESa9hRC9T3XJQZrhFricvH87ifBwMVZk7Qpump", "Pump.fun"),
    ("D_migrated", "9p3NuCz29u7KUsjfrZcBPNGB2pryDpACggjSjYWbkpds", "migrated/PumpSwap"),
    ("E_raydium", "2cuM75BppZWghS74hZXapyQonz4KKd36PeVkCUYqSTNK", "Whirlpool/Raydium"),
    ("E_raydium2", "JAccgVwonRw28jZ3fHdgwtyAfKJRPd5qMqaVP2xipump", "Raydium/Meteora"),
]


def _stats(sizes: list[int]) -> dict:
    if not sizes:
        return {"sample_count": 0}
    s = sorted(sizes)
    fit = [x for x in s if x <= MAX_SOLANA_TX_BYTES]
    return {
        "sample_count": len(s),
        "min": s[0],
        "median": statistics.median(s),
        "p95": s[min(int(0.95 * (len(s) - 1)), len(s) - 1)],
        "max": s[-1],
        "fit_count": len(fit),
        "fit_rate_pct": round(100.0 * len(fit) / len(s), 1),
        "tx_limit": MAX_SOLANA_TX_BYTES,
    }


def _measure(tx_b64: str) -> dict:
    rep = transaction_size_report(tx_b64)
    # ``fits`` is the readable alias of the report's ``ok``.
    rep["fits"] = rep.get("ok")
    counts = {"instruction_count": None, "account_count": None,
              "signature_slots": None, "unsigned": None}
    try:
        vtx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
        counts["instruction_count"] = len(vtx.message.instructions)
        counts["account_count"] = len(vtx.message.account_keys)
        counts["signature_slots"] = len(vtx.signatures)
        counts["unsigned"] = all(
            bytes(s) == bytes(64) for s in vtx.signatures)
    except Exception as exc:  # noqa: BLE001
        counts["decode_error"] = type(exc).__name__
    return {**rep, **counts}


async def _payer() -> tuple[str, str]:
    """Return (pubkey, source). PUBLIC key only; no signing anywhere here."""
    throwaway = str(Keypair().pubkey())
    try:
        from autopaper.config import build_config
        cfg = build_config()
        path = cfg.real.wallet_keypair_path
        if path:
            from autopaper.execution.solana import SolanaWallet
            return SolanaWallet(path).pubkey, "operator_wallet_public_only"
    except Exception:  # noqa: BLE001
        pass
    return throwaway, "throwaway_key"


async def probe(out_path: Path | None, simulate: bool) -> dict:
    payer, payer_source = await _payer()
    rap = RaptorProvider()
    jup = JupiterProvider()
    rpc = None
    rpc_url = None
    if simulate:
        try:
            from autopaper.config import build_config
            rpc_url = build_config().real.rpc_url
        except Exception:  # noqa: BLE001
            rpc_url = None
        if rpc_url:
            rpc = SolanaRpc(rpc_url)

    rows: list[dict] = []
    rap_sizes: list[int] = []
    jup_sizes: list[int] = []

    for key, mint, note in TOKENS:
        row: dict = {"token": key, "mint": mint, "note": note,
                     "amount_lamports": POSITION_LAMPORTS,
                     "slippage_bps": SLIPPAGE_BPS, "raptor": {}, "jupiter": {}}

        # ---- Raptor ----
        try:
            t0 = time.perf_counter()
            q = await rap.quote(WSOL_MINT, mint, POSITION_LAMPORTS, SLIPPAGE_BPS)
            t1 = time.perf_counter()
            built = await rap.build(
                q, payer, wrap_and_unwrap_sol=True,
                priority_fee=PRIORITY_FEE_LAMPORTS)
            t2 = time.perf_counter()
            m = _measure(built.transaction_b64)
            t3 = time.perf_counter()
            latency = {
                "quote_ms": round((t1 - t0) * 1000, 3),
                "build_ms": round((t2 - t1) * 1000, 3),
                "measure_ms": round((t3 - t2) * 1000, 3),
                "alert_to_tx_ready_ms": round((t2 - t0) * 1000, 3),
            }
            row["raptor"] = {
                "status": "OK",
                "latency_ms": latency,
                "route": q.route_labels,
                "selected_dex": q.selected_dex,
                "route_hops": len(q.route_steps),
                "expected_out": q.expected_out_amount,
                "min_out": q.min_out_amount,
                "price_impact_pct": round(q.price_impact_pct, 6),
                "prioritization_fee_lamports":
                    built.prioritization_fee_lamports,
                **{k: m[k] for k in (
                    "raw_bytes", "encoded_bytes", "fits", "status",
                    "over_by", "instruction_count", "account_count",
                    "signature_slots", "unsigned")},
            }
            if m.get("raw_bytes"):
                rap_sizes.append(m["raw_bytes"])
            if rpc is not None:
                ts0 = time.perf_counter()
                out = await rpc.simulate_transaction_detailed(
                    built.transaction_b64)
                row["raptor"]["latency_ms"]["simulate_ms"] = round(
                    (time.perf_counter() - ts0) * 1000, 3)
                row["raptor"]["simulation"] = {
                    "ok": out.ok, "status": out.status,
                    "detail": (out.detail or "")[:200], "stage": out.stage,
                }
        except RaptorError as exc:
            row["raptor"] = {"status": exc.status,
                             "detail": str(exc)[:200],
                             "http_status": exc.http_status}
        except Exception as exc:  # noqa: BLE001
            row["raptor"] = {"status": "ERROR",
                             "detail": f"{type(exc).__name__}: {exc}"[:200]}

        # ---- Jupiter (reference) ----
        try:
            jt0 = time.perf_counter()
            jq = await jup.quote(WSOL_MINT, mint, POSITION_LAMPORTS,
                                 SLIPPAGE_BPS)
            jresults = {"quote_ms": round(
                (time.perf_counter() - jt0) * 1000, 3)}
            for wrap in (False, True):
                try:
                    jb0 = time.perf_counter()
                    jb = await jup.build(
                        jq, payer, wrap_and_unwrap_sol=wrap,
                        priority_fee=PRIORITY_FEE_LAMPORTS)
                    jm = _measure(jb.transaction_b64)
                    jm["build_ms"] = round(
                        (time.perf_counter() - jb0) * 1000, 3)
                    jresults[f"wrap_{wrap}"] = {
                        k: jm[k] for k in (
                            "raw_bytes", "encoded_bytes", "fits", "status",
                            "instruction_count", "account_count",
                            "build_ms")}
                    if jm.get("raw_bytes"):
                        jup_sizes.append(jm["raw_bytes"])
                except Exception as exc:  # noqa: BLE001
                    jresults[f"wrap_{wrap}"] = {
                        "status": "BUILD_FAILED",
                        "detail": type(exc).__name__}
            row["jupiter"] = {
                "status": "OK",
                "route": jq.route_labels,
                "expected_out": jq.expected_out_amount,
                **jresults,
            }
        except Exception as exc:  # noqa: BLE001
            row["jupiter"] = {"status": "QUOTE_FAILED",
                              "detail": f"{type(exc).__name__}: {exc}"[:160]}

        rows.append(row)
        print(f"[{key}] {mint}\n  raptor={row['raptor'].get('status')} "
              f"bytes={row['raptor'].get('raw_bytes')} "
              f"route={row['raptor'].get('route')}\n  "
              f"jupiter={row['jupiter'].get('status')} "
              f"bytes={[v.get('raw_bytes') for v in row['jupiter'].values() if isinstance(v, dict)]}")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": rap.base_url,
        "payer_source": payer_source,
        "position_lamports": POSITION_LAMPORTS,
        "slippage_bps": SLIPPAGE_BPS,
        "priority_fee_lamports": PRIORITY_FEE_LAMPORTS,
        "tx_limit_bytes": MAX_SOLANA_TX_BYTES,
        "simulation_run": rpc is not None,
        "broadcast_performed": False,
        "private_key_shared": False,
        "raptor_distribution": _stats(rap_sizes),
        "jupiter_distribution": _stats(jup_sizes),
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
    ap.add_argument("--no-simulate", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    rep = asyncio.run(probe(out, simulate=not args.no_simulate))
    print("\n== distributions ==")
    print("raptor :", rep["raptor_distribution"])
    print("jupiter:", rep["jupiter_distribution"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
