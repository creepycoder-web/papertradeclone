"""Live READ-ONLY probe: verify Raptor's native-SOL input behavior.

WHAT THIS PROVES
----------------
That a Raptor SOL-input BUY is built from NATIVE SOL, with NO pre-provisioned
wrapped-SOL account, and that the wrap/unwrap is handled by Raptor inside the
transaction it returns. It inspects the ACTUAL transaction rather than trusting
configuration:

  * whether the operator wallet even HAS a wSOL account (read-only),
  * instruction count, account count,
  * whether the wSOL mint / token programs appear (i.e. the wrap is in-tx),
  * whether an ATA/setup program is present,
  * raw bytes and encoded bytes,
  * the RPC simulation result.

WHAT THIS DOES NOT DO
---------------------
No signing, no broadcast, no private key anywhere, no wSOL account created or
funded, no ATA created, and Raptor's ``/send-transaction`` endpoint is never
called. The payer public key is read for its PUBLIC address ONLY.

    .venv/bin/python scripts/raptor_wsol_probe.py [--out PATH] [--no-simulate]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solders.keypair import Keypair  # noqa: E402
from solders.pubkey import Pubkey  # noqa: E402
from solders.transaction import VersionedTransaction  # noqa: E402

from autopaper.execution.raptor import (  # noqa: E402
    RaptorError,
    RaptorProvider,
    transaction_size_report,
)
from autopaper.execution.solana import (  # noqa: E402
    MAX_SOLANA_TX_BYTES,
    SolanaRpc,
    SolanaWallet,
    WSOL_MINT,
)

#: Exact production micro notional (never enlarged).
POSITION_LAMPORTS = 1_071_843
SLIPPAGE_BPS = 500
PRIORITY_FEE_LAMPORTS = 50_000

#: Known-routable mints from prior read-only runs, plus real alert mints.
TOKENS: list[tuple[str, str]] = [
    ("pump_routable", "C5yy7SB3tY81ZMpr4tgjVDcp6teSLRKrPvGktggNmoon"),
    ("raydium_routable", "D8qiGA5eSA2E3JNyM89J5Qu9YJzWFYMZBFysXWgtJRYd"),
    ("pump1", "AxprAjtnzxPFoj91qifAYcD2V2KST3J2eSAn2xBPpump"),
    ("migrated", "9p3NuCz29u7KUsjfrZcBPNGB2pryDpACggjSjYWbkpds"),
    ("raydium2", "JAccgVwonRw28jZ3fHdgwtyAfKJRPd5qMqaVP2xipump"),
]

#: Well-known program ids used only to CLASSIFY instructions in the report.
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ATA_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"


async def _resolve_loaded_keys(msg, rpc) -> tuple[list[str], int, bool]:
    """Return (static+loaded keys, alt_count, wsol_present_in_alt).

    V0 transactions load most keys through address lookup tables, so the static
    ``account_keys`` list alone cannot answer "is wSOL referenced?". The lookup
    tables are fetched (READ-ONLY) and their address arrays extracted. The wSOL
    mint is also scanned for directly in the raw table data, so the answer is
    robust even if the address-array header ever changes.
    """
    static = [str(k) for k in msg.account_keys]
    lookups = list(getattr(msg, "address_table_lookups", []) or [])
    wsol_bytes = bytes(Pubkey.from_string(WSOL_MINT))
    wsol_in_alt = False
    if lookups and rpc is not None:
        addrs = [str(lp.account_key) for lp in lookups]
        try:
            res = await rpc._rpc(
                "getMultipleAccounts", [addrs, {"encoding": "base64"}])
            values = ((res or {}).get("value") or [])
        except Exception:  # noqa: BLE001
            values = []
        for _lp, val in zip(lookups, values):
            if not val:
                continue
            data = val.get("data")
            data_b64 = data[0] if isinstance(data, list) else data
            try:
                raw = base64.b64decode(data_b64)
            except Exception:  # noqa: BLE001
                continue
            if wsol_bytes in raw:
                wsol_in_alt = True
    return static, len(lookups), wsol_in_alt


async def _classify(tx_b64: str, rpc) -> dict:
    """Structural facts about the built (unsigned) transaction."""
    rep = transaction_size_report(tx_b64)
    facts: dict = {
        "raw_bytes": rep.get("raw_bytes"),
        "encoded_bytes": rep.get("b64_chars"),
        "limit_bytes": rep.get("limit_bytes"),
        "over_by": rep.get("over_by"),
        "fits": rep.get("ok"),
        "size_status": rep.get("status"),
        "instruction_count": None,
        "account_count": None,
        "alt_lookup_count": None,
        "signature_slots": None,
        "unsigned": None,
        "wsol_mint_referenced": None,
        "token_program_instructions": None,
        "token_2022_instructions": None,
        "ata_program_instructions": None,
    }
    try:
        vtx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
    except Exception as exc:  # noqa: BLE001
        facts["decode_error"] = type(exc).__name__
        return facts
    msg = vtx.message
    facts["instruction_count"] = len(msg.instructions)
    facts["account_count"] = len(msg.account_keys)
    facts["signature_slots"] = len(vtx.signatures)
    facts["unsigned"] = all(bytes(s) == bytes(64) for s in vtx.signatures)
    full, alt_count, wsol_in_alt = await _resolve_loaded_keys(msg, rpc)
    facts["alt_lookup_count"] = alt_count
    facts["wsol_mint_in_alt_tables"] = wsol_in_alt
    facts["wsol_mint_referenced"] = WSOL_MINT in full or wsol_in_alt
    prog = [full[ix.program_id_index]
            if ix.program_id_index < len(full) else ""
            for ix in msg.instructions]
    facts["token_program_instructions"] = sum(1 for p in prog if p == TOKEN_PROGRAM)
    facts["token_2022_instructions"] = sum(1 for p in prog if p == TOKEN_2022_PROGRAM)
    facts["ata_program_instructions"] = sum(1 for p in prog if p == ATA_PROGRAM)
    return facts


async def _payer() -> tuple[str, str, bool | None]:
    """Return (pubkey, source, wsol_ata_present|None). PUBLIC key only."""
    from autopaper.config import build_config
    cfg = build_config()
    path = cfg.real.wallet_keypair_path
    if not path:
        return str(Keypair().pubkey()), "throwaway_key", None
    wallet = SolanaWallet(path)
    present: bool | None = None
    if cfg.real.rpc_url:
        rpc = SolanaRpc(cfg.real.rpc_url)
        try:
            ready = await rpc.wsol_readiness(wallet.pubkey, 1)
            present = bool(ready.ata_present)
        except Exception:  # noqa: BLE001
            present = None
        finally:
            await rpc.aclose()
    return wallet.pubkey, "operator_wallet_public_only", present


async def probe(out_path: Path | None, simulate: bool) -> dict:
    payer, payer_source, wsol_ata_present = await _payer()
    rap = RaptorProvider()

    rpc = None
    if simulate:
        from autopaper.config import build_config
        rpc_url = build_config().real.rpc_url
        if rpc_url:
            rpc = SolanaRpc(rpc_url)

    rows: list[dict] = []
    built_ok = sim_ok = 0
    for key, mint in TOKENS:
        row: dict = {"token": key, "mint": mint,
                     "amount_lamports": POSITION_LAMPORTS}
        try:
            q = await rap.quote(WSOL_MINT, mint, POSITION_LAMPORTS, SLIPPAGE_BPS)
            # NO wrap flag is passed: the provider applies its own default
            # (wrapUnwrapSol=true), exactly as the executor's generic path does.
            built = await rap.build(q, payer, priority_fee=PRIORITY_FEE_LAMPORTS)
            facts = await _classify(built.transaction_b64, rpc)
            row["raptor"] = {
                "status": "OK",
                "route": q.route_labels,
                "route_hops": len(q.route_steps),
                "expected_out": q.expected_out_amount,
                "min_out": q.min_out_amount,
                "prioritization_fee_lamports": built.prioritization_fee_lamports,
                **facts,
            }
            built_ok += 1
            if rpc is not None:
                out = await rpc.simulate_transaction_detailed(built.transaction_b64)
                row["raptor"]["simulation"] = {
                    "ok": out.ok, "status": out.status,
                    "detail": (out.detail or "")[:200], "stage": out.stage,
                }
                if out.ok:
                    sim_ok += 1
        except RaptorError as exc:
            row["raptor"] = {"status": exc.status, "detail": str(exc)[:200],
                             "http_status": exc.http_status}
        except Exception as exc:  # noqa: BLE001
            row["raptor"] = {"status": "ERROR",
                             "detail": f"{type(exc).__name__}: {exc}"[:200]}
        rows.append(row)
        r = row["raptor"]
        print(f"[{key}] {r.get('status')} bytes={r.get('raw_bytes')} "
              f"ix={r.get('instruction_count')} acc={r.get('account_count')} "
              f"wsol_ref={r.get('wsol_mint_referenced')} "
              f"sim={r.get('simulation', {}).get('ok')}")

    if rpc is not None:
        await rpc.aclose()
    await rap.aclose()

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": rap.base_url,
        "payer_source": payer_source,
        "wsol_ata_present_before": wsol_ata_present,
        "position_lamports": POSITION_LAMPORTS,
        "slippage_bps": SLIPPAGE_BPS,
        "priority_fee_lamports": PRIORITY_FEE_LAMPORTS,
        "tx_limit_bytes": MAX_SOLANA_TX_BYTES,
        "simulation_run": rpc is not None,
        "raptor_built_count": built_ok,
        "raptor_simulated_ok_count": sim_ok,
        "broadcast_performed": False,
        "private_key_shared": False,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-simulate", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    rep = asyncio.run(probe(out, simulate=not args.no_simulate))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
