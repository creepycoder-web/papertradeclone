"""
REAL-path latency probe. NO signing, NO broadcast, NO key material.

Drives the ACTUAL production :class:`~autopaper.execution.real.RealExecutor`
through N deterministic dry-run ENTRY and EXIT orders with in-process doubles for
the wallet, the Solana RPC and the venue. Every stage the executor actually
performs is measured individually via its ``stage`` callback, so the report does
not invent a signing or broadcast latency: those stages are reported only because
the probe's fake signer/broadcast were actually called and timed.

Writes ``reports/forensics/real_latency_comparison.json`` and prints a summary.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autopaper.execution.real import LiveSolanaBackend, RealExecutor  # noqa: E402
from autopaper.execution.solana import WSOL_MINT  # noqa: E402
from autopaper.latency import percentiles  # noqa: E402

MINT = "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj"
OWNER = "SoWaLanaPubKey1111111111111111111111111111111"
N = 20


class _Cfg:
    wallet_keypair_path = "/tmp/fake-id.json"
    rpc_url = "https://rpc.invalid"
    slippage_bps = 500
    simulate_before_send = True
    priority_fee_lamports = 50_000
    confirmation_timeout_s = 5.0
    quote_max_age_ms = 8_000.0
    max_price_impact_pct = 5.0
    max_open_positions = 1
    reentry_enabled = False
    trading_enabled = False


class _Quote:
    def __init__(self, in_amount, out_amount):
        self.in_amount = in_amount
        self.expected_out_amount = out_amount
        self.min_out_amount = int(out_amount * 0.99)
        self.price_impact_pct = 0.4
        self.route_labels = ["RAYDIUM"]

    @property
    def price_per_token(self):
        return self.in_amount / max(1, self.expected_out_amount)


class _Built:
    transaction_b64 = "AQAB"
    last_valid_block_height = 123_456


class _Provider:
    def __init__(self, delays=None):
        self.d = delays or {}

    async def quote(self, i, o, amount, slip):
        await asyncio.sleep(self.d.get("venue_quote", 0.0) / 1000.0)
        return _Quote(amount, 12_345_678)

    async def build(self, quote, pubkey, priority_fee="auto", **kw):
        await asyncio.sleep(self.d.get("build", 0.0) / 1000.0)
        return _Built()

    async def health(self):
        return {"reachable": True, "detail": "mock"}

    async def aclose(self):
        return None


class _Rpc:
    def __init__(self, tx, delays=None):
        self.tx = tx
        self.sent = []
        self.d = delays or {}

    async def get_balance_lamports(self, addr=None):
        await asyncio.sleep(self.d.get("get_balance", 0.0) / 1000.0)
        return 3_000_000_000

    async def get_health(self):
        return SimpleNamespace(reachable=True, detail="ok")

    async def get_slot(self):
        return 120_000

    async def simulate_transaction(self, b64):
        await asyncio.sleep(self.d.get("simulate", 0.0) / 1000.0)
        return (True, "ok")

    async def send_transaction(self, b64):
        await asyncio.sleep(self.d.get("send", 0.0) / 1000.0)
        self.sent.append(b64)
        return "Sig111"

    async def wait_for_confirmation(self, sig, **kw):
        await asyncio.sleep(self.d.get("confirm", 0.0) / 1000.0)
        return {"status": "confirmed", "error": None, "signature": sig, "slot": 9}

    async def get_transaction(self, sig, commitment="confirmed"):
        await asyncio.sleep(self.d.get("get_transaction", 0.0) / 1000.0)
        return self.tx

    async def aclose(self):
        return None


class _Wallet:
    pubkey = OWNER
    masked = "SoWa…1111"

    def sign_wire(self, raw):
        return raw

    def sign_new_transaction(self, ix):
        return b"probe"

    @property
    def pubkey_obj(self):
        return None


def _entry_tx():
    return {"slot": 9, "transaction": {"message": {"accountKeys": [OWNER]}, "meta": {
        "err": None, "fee": 5000, "preBalances": [3_000_000_000],
        "postBalances": [2_625_000_000], "preTokenBalances": [],
        "postTokenBalances": [{"account": {"data": {"parsed": {"info": {
            "mint": MINT, "tokenAmount": {"amount": "12345678"}}}}}}]}}}


def _exit_tx():
    return {"slot": 11, "transaction": {"message": {"accountKeys": [OWNER]}, "meta": {
        "err": None, "fee": 5000, "preBalances": [3_000_000_000],
        "postBalances": [3_010_000_000], "preTokenBalances": [
            {"account": {"data": {"parsed": {"info": {
                "mint": MINT, "tokenAmount": {"amount": "12345678"}}}}}}],
        "postTokenBalances": []}}}


class _Timer:
    def __init__(self):
        self.marks: list[tuple[str, float]] = []

    def __call__(self, name: str) -> None:
        self.marks.append((name, time.perf_counter()))

    def durations_ms(self) -> dict[str, float]:
        starts = {n: t for n, t in self.marks if n.endswith("_start")}
        ends = {n: t for n, t in self.marks if n.endswith("_end")}
        out: dict[str, float] = {}
        for s, t0 in starts.items():
            e = s[:-6] + "_end"
            if e in ends:
                out[s[:-6]] = (ends[e] - t0) * 1000.0
        return out


async def _one(side: str, delays: dict | None = None) -> dict:
    delays = delays or {}
    tx = _entry_tx() if side == "ENTRY" else _exit_tx()
    backend = LiveSolanaBackend(_Cfg(), wallet=_Wallet(),
                                rpc=_Rpc(tx, delays),
                                provider=_Provider(delays))
    ex = RealExecutor(_Cfg(), backend=backend)
    timer = _Timer()
    t0 = time.perf_counter()
    if side == "ENTRY":
        fill = await ex.open_position(
            mint=MINT, position_lamports=375_000_000, slippage_bps=500,
            prechecks=[], stage=timer,
        )
    else:
        fill = await ex.close_position(
            mint=MINT, token_amount=12_345_678, slippage_bps=500,
            prechecks=[], stage=timer,
        )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {"side": side, "filled": fill.filled, "wall_ms": wall_ms,
            "stages": timer.durations_ms()}


def _paper_baseline() -> dict:
    path = ROOT / "var" / "stage_timings.jsonl"
    samples: list[float] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if str(row.get("mode") or "").upper() != "PAPER":
                continue
            v = row.get("trading_ms")
            if isinstance(v, (int, float)) and v == v:
                samples.append(float(v))
    return {"n": len(samples), **percentiles(samples)}


#: Declared, MODELLED per-operation network delays. These are NOT measurements
#: of a live endpoint; they are a fixed profile applied to the in-process doubles
#: so the ORCHESTRATION can be measured against a realistic distribution. The
#: report labels them as modelled.
PROFILES = {
    "orchestration_no_network": {},
    "modelled_network": {
        "venue_quote": 150.0, "build": 328.0, "simulate": 208.0,
        "send": 275.0, "confirm": 1250.0, "get_transaction": 205.0,
        "get_balance": 80.0,
    },
}


def _run_profile(name: str, delays: dict) -> dict:
    samples = [asyncio.run(_one("ENTRY", delays)) for _ in range(N)]
    samples += [asyncio.run(_one("EXIT", delays)) for _ in range(N)]
    stage_samples: dict[str, list[float]] = {}
    for s in samples:
        for stage, ms in s["stages"].items():
            stage_samples.setdefault(stage, []).append(ms)
    return {
        "profile": name,
        "network_delays_ms": delays,
        "real_executor_wall_ms": percentiles([s["wall_ms"] for s in samples]),
        "real_stages_ms": {
            k: percentiles(v) for k, v in sorted(stage_samples.items())
        },
        "samples": samples,
    }


def main() -> dict:
    report = {
        "provenance": {
            "method": "in-process dry-run of the ACTUAL RealExecutor",
            "n_per_side": N,
            "signed": False, "broadcast": False,
            "wallet": "in-process fake signer (returns bytes)",
            "rpc": "in-process counting double (sends nothing)",
            "venue": "in-process double",
            "note": ("Two profiles: orchestration_no_network measures pure code "
                     "overhead; modelled_network applies DECLARED per-operation "
                     "delays so wall time can be read against a realistic "
                     "distribution. Signing/broadcast appear only because the "
                     "fake signer/broadcast were actually CALLED and timed. No "
                     "real transaction was signed or sent."),
        },
        "paper_baseline_trading_ms": _paper_baseline(),
        "profiles": {name: _run_profile(name, d) for name, d in PROFILES.items()},
    }
    out = ROOT / "reports" / "forensics" / "real_latency_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)
    print("PAPER trading_ms:", report["paper_baseline_trading_ms"])
    for name, prof in report["profiles"].items():
        w = prof["real_executor_wall_ms"]
        print(f"[{name}] REAL executor wall_ms: p50={w['p50']:.2f} "
              f"p95={w['p95']:.2f} p99={w['p99']:.2f} max={w['max']:.2f}")
        for k, v in prof["real_stages_ms"].items():
            print(f"    {k}: p50={v['p50']:.2f} p95={v['p95']:.2f} max={v['max']:.2f}")
    return report


if __name__ == "__main__":
    main()
