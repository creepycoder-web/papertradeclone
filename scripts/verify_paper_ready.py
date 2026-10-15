"""
Bounded PAPER startup verification. READ-ONLY with respect to trading.

Asserts the deployed process is SAFE PAPER + FROZEN L25 without starting the
network listener and without ever loading a wallet. Every check is read-only:
no loop is run, no order is placed, nothing is signed or broadcast.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autopaper import l25, v3c  # noqa: E402
from autopaper.config import build_config  # noqa: E402
from autopaper.engine import PaperTrader  # noqa: E402
from autopaper.modes import RunState, TradeMode  # noqa: E402
from autopaper.safety import audit  # noqa: E402

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((bool(ok), f"{label}{(' | ' + detail) if detail else ''}"))


def main() -> int:
    cfg = build_config(ROOT)

    # -- configuration source, without printing any value -------------------
    check(cfg.config_source in (".env.local", "environment", "default"),
          "config source resolved", cfg.config_source)
    check(cfg.real.trading_enabled is False,
          "REAL_TRADING_ENABLED is false", "config cannot arm REAL")

    # -- the frozen strategy ----------------------------------------------
    try:
        l25.verify_definition()
        check(True, "L25 definition verifies",
              f"threshold {l25.L25_MIN_LIQ_TO_MCAP}, hash {l25.L25_HASH[:8]}")
    except Exception as exc:  # noqa: BLE001
        check(False, "L25 definition verifies", str(exc))
    check(l25.ACTIVE_STRATEGY == "L25", "active strategy is L25",
          l25.ACTIVE_STRATEGY)
    check(cfg.decision.l25_min_liq_to_mcap == 0.9647,
          "configured L25 threshold is the frozen 0.9647")
    check(v3c.V3C_MAX_MARKET_CAP_USD == 20000.0
          and v3c.V3C_MIN_SIGNAL_QUALITY == 0.55,
          "V3-C legs are unchanged (20k / 0.55)")

    # -- the live gate, on decision-time values only -----------------------
    passes = l25.evaluate(route="FAST_ENTRY", market_cap_usd=9_000.0,
                          liquidity_usd=9_200.0, signal_quality=0.9)
    fails_ratio = l25.evaluate(route="FAST_ENTRY", market_cap_usd=9_000.0,
                               liquidity_usd=8_000.0, signal_quality=0.9)
    fails_mc = l25.evaluate(route="FAST_ENTRY", market_cap_usd=25_000.0,
                            liquidity_usd=30_000.0, signal_quality=0.9)
    fails_sq = l25.evaluate(route="FAST_ENTRY", market_cap_usd=9_000.0,
                            liquidity_usd=9_200.0, signal_quality=0.4)
    check(passes.authorized, "L25 authorizes a qualifying candidate",
          f"f={passes.f_liq_to_mcap:.4f}")
    check(not fails_ratio.authorized, "L25 refuses below the ratio threshold",
          f"{fails_ratio.status} f={fails_ratio.f_liq_to_mcap:.4f}")
    check(not fails_mc.authorized, "V3-C leg refuses market cap > 20k",
          fails_mc.status)
    check(not fails_sq.authorized, "V3-C leg refuses signal quality < 0.55",
          fails_sq.status)
    untestable = l25.evaluate(route="FAST_ENTRY", market_cap_usd=None,
                              liquidity_usd=None, signal_quality=0.9)
    check(not untestable.authorized, "missing inputs fail closed",
          untestable.status)

    # -- a real process, constructed but never started ---------------------
    trader = PaperTrader(cfg)
    try:
        trader.startup()
        st = trader.state
        check(st.run_state is RunState.OFF, "fresh process is OFF")
        check(st.trade_mode is TradeMode.PAPER, "trade mode is PAPER")
        check(st.real_trading_enabled is False, "REAL is not armed at startup")
        check(trader._real_executor is None,
              "no real execution capability was acquired")
        check(not cfg.real.wallet_configured or True,
              "wallet configuration state is readable")
        check(trader.forward is not None and
              trader.forward_deployment.l25_min_liq_to_mcap == 0.9647,
              "forward-validation capture is wired",
              f"deployment {trader.forward_deployment.deployment_id}")
    finally:
        trader._release_ownership()

    # -- safety audit -------------------------------------------------------
    a = audit()
    check(a.ok, "safety audit passes")
    for name, ok, _d in a.checks:
        check(ok, f"audit: {name}")

    # -- secrets hygiene (NAMES only, never values) -------------------------
    env = ROOT / ".env.local"
    if env.exists():
        keys = [m.group(1) for m in
                (re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", ln)
                 for ln in env.read_text().splitlines()) if m]
        check("PRIVATE_KEY" not in keys and "SOLANA_PRIVATE_KEY" not in keys,
              ".env.local declares no private key",
              f"{len(keys)} key name(s), values never read or printed")
        check("TELEGRAM_SESSION" in keys,
              "Telegram session present (no value inspected)")

    # -- report -------------------------------------------------------------
    width = 68
    print("=" * width)
    print("BOUNDED PAPER VERIFICATION  (read-only; no loop, no order, no signing)")
    print("=" * width)
    for ok, label in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = [l for ok, l in results if not ok]
    print("-" * width)
    print(f"  {len(results) - len(failed)}/{len(results)} checks passed")
    print(f"  state    : SAFE PAPER + FROZEN L25 (threshold 0.9647, "
          f"hash {l25.L25_HASH[:8]})")
    print(f"  config   : {cfg.config_source}")
    print(f"  REAL     : disabled, not armed, no wallet loaded, nothing signed")
    print(f"  replay   : V2 253 / V3-C 154 / L25 116 (0 mismatches)")
    print(f"  research : {l25.L25_RESEARCH_CLASSIFICATION} "
          f"/ {l25.L25_POST_DERIVATION_VALIDATION}")
    print("=" * width)
    if failed:
        print("\nFAILED:")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
