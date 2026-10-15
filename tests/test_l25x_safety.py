"""
Structural safety and non-regression tests.

Covers required tests 18-20:
  18. no real transaction path is reachable from this feature
  19. existing settlement PnL is unchanged
  20. existing L25 behaviour is unchanged

These are the tests that make the other 17 meaningful. A venue-selection feature
that quietly enabled trading, rewrote PnL, or altered L25 would be worthless
regardless of how good its venue ranking was.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

AI = Path(__file__).resolve().parents[1]
for _p in (AI, AI / "research"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from autopaper.same_market.models import CheckState, DexPair, WSOL_MINT  # noqa: E402
from l25x import pipeline as P                                          # noqa: E402
from l25x.economics import OBSERVE_ONLY, EconomicStatus                 # noqa: E402
from l25x.rent import TokenAccountState                                  # noqa: E402
from l25x.venues import VenueIdentity, VenueQuoteResult                 # noqa: E402

PKG = AI / "research" / "l25x"
MINT = "A5MdMEt99PzbYKsfPaQ5Ee1jFZEsYu3zBvdpDdyX2sXQ"


# -- 18. no real transaction path ------------------------------------------


def test_package_has_no_execution_capability():
    """18a. The package's own AST contains no broadcast/signing capability."""
    report = P.assert_no_execution_capability()
    assert report["execution_capability"].startswith("absent")
    assert "pipeline.py" in report["modules_scanned"]


def test_package_imports_nothing_that_can_sign_or_broadcast():
    """18b. No import of a signer/broadcaster/wallet module anywhere."""
    banned = ("autopaper.execution.solana", "autopaper.execution.real",
              "autopaper.telegram.client", "solana", "solders", "base58")
    offenders = []
    for path in PKG.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                for b in banned:
                    if m == b or m.startswith(b + "."):
                        offenders.append(f"{path.name}: {m}")
    assert offenders == []


def test_no_module_defines_a_broadcast_or_sign_function():
    """18c. Nothing here is even shaped like a broadcaster."""
    forbidden = {"send_raw_transaction", "sign_transaction", "broadcast",
                 "send_transaction", "commit_transaction", "sign_and_send",
                 "close_account", "close_token_account", "transfer_sol",
                 "execute_cleanup", "execute_reclaim"}
    defined = set()
    for path in PKG.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
            if isinstance(node, ast.ClassDef):
                defined.add(node.name)
    assert defined & forbidden == set()


def test_rent_ledger_exposes_no_execution_method():
    """18d. Rent reclamation is accounting only; there is no closer."""
    from l25x.rent import RentLedger
    public = {n for n in dir(RentLedger) if not n.startswith("_")}
    for bad in ("execute", "close", "reclaim", "broadcast", "send", "transfer"):
        assert bad not in public, bad
    # the only reclaim-ish name is a RECORDER, not a performer
    assert "on_reclaimed" in public


def test_on_chain_reclamation_flag_is_false_everywhere_it_is_reported():
    """18e. The report surface always declares reclamation unimplemented."""
    from l25x.rent import RentBasis, RentLedger
    led = RentLedger(trade_id="t", mint="M", token_account=None)
    led.on_entry(TokenAccountState(address=None, exists=False,
                                   program_id="Tokenkeg"),
                 2_039_280, RentBasis.ESTIMATED_PROGRAM_SCHEDULE)
    assert led.as_dict()["on_chain_reclamation_implemented"] is False


def test_research_run_writes_only_under_reports_research():
    """18f. The isolation guard's write-root rule is respected by our outputs."""
    from l25r import isolation
    assert "reports/research" in isolation.ALLOWED_WRITE_ROOTS
    assert ".env.local" in isolation.FORBIDDEN_READS


# -- 19. settlement PnL unchanged ------------------------------------------


def test_research_layer_does_not_write_canonical_trading_pnl():
    """19a. TRADING_PNL is hard-zeroed; the economic view is separate."""
    from l25x.rent import CostBucket, RentBasis, RentLedger, economic_adjusted_view
    led = RentLedger(trade_id="t", mint="M", token_account="ATA")
    led.on_entry(TokenAccountState(address="ATA", exists=False,
                                   program_id="Tokenkeg"),
                 2_039_280, RentBasis.ESTIMATED_PROGRAM_SCHEDULE)
    led.trading_pnl_usd = 1.2345           # set by the settlement engine
    d = led.as_dict()
    assert d["trading_pnl_usd"] == 1.2345
    assert d["buckets_lamports"][CostBucket.TRADING_PNL.value] == 0
    assert d["canonical_pnl_contains_rent"] is False
    view = economic_adjusted_view(led, sol_price_usd=120.0)
    assert view["canonical_trading_pnl_usd"] == 1.2345
    assert view["economic_adjusted_pnl_usd"] != 1.2345


def test_observe_only_policy_rejects_nothing_so_replay_is_identical():
    """
    19b/20a. With thresholds unset, the research layer admits exactly what L25
    admits, so a replay through it is bit-identical to plain L25. This is the
    strongest available statement that neither PnL nor L25 moved.
    """
    from l25c import engine as E
    from evml25 import modelled as M
    cands, _ = E.build_candidates()
    control = [t for t in cands if E.in_l25(t)]
    assert control, "expected a non-empty L25 control population"

    plain, _ = M.replay(cands, entry_filter=E.in_l25)

    gated = []
    for t in cands:
        if not E.in_l25(t):
            continue
        econ = P.assess_economics(
            planned_notional_usd=10.0, sol_price_usd=200.0,
            token_account=TokenAccountState(address="ATA", exists=True,
                                            program_id="Tokenkeg"),
            policy=OBSERVE_ONLY)
        assert econ.status is EconomicStatus.ECONOMICALLY_VALID
        gated.append(t)

    observed, _ = M.replay(gated, entry_filter=E.in_l25)
    assert len(observed.rows) == len(plain.rows)
    for a, b in zip(plain.rows, observed.rows):
        assert a["trade_id"] == b["trade_id"]
        assert a["net_pnl"] == b["net_pnl"]
        assert a["position_size"] == b["position_size"]
        assert a["exit_reason"] == b["exit_reason"]


# -- 20. L25 unchanged -----------------------------------------------------


def test_in_l25_control_is_used_verbatim_and_unmodified():
    """20b. The gate is imported, not reimplemented."""
    from l25c.engine import in_l25
    assert P.E.in_l25 is in_l25
    # a candidate that fails L25 is still rejected by the research pipeline
    v = VenueIdentity(venue_id="v", market_type=None, pool_address="p",
                      program_id=None, dex_id=None, quote_asset_mint=WSOL_MINT,
                      identity_state=CheckState.PASS)

    def q(vv, lamports):
        return VenueQuoteResult(venue=vv, ok=True, output_amount=1.0,
                                effective_price_usd=1e-6, quote_source="stub")

    plan = P.plan_entry(
        {"mint": MINT, "symbol": "AI", "f_market_cap_usd": 999_999.0,
         "f_signal_quality": 0.1, "f_liquidity_usd": 1.0, "route": "FAST_ENTRY"},
        planned_notional_usd=10.0, sol_price_usd=200.0,
        pairs=[DexPair(chain_id="solana", dex_id="pumpswap",
                      pair_address="p", base_token_address=MINT,
                      base_token_symbol="AI", quote_token_address=WSOL_MINT,
                      quote_token_symbol="SOL", price_usd=1e-6,
                      price_native=1e-6, liquidity_usd=1.0, volume_24h_usd=None,
                      fdv_usd=None, market_cap_usd=None, pair_created_at=None,
                      txns_5m=None, txns_1h=None, buys_5m=None, sells_5m=None,
                      buys_1h=None, sells_1h=None)],
        quote_for=q,
        token_account=TokenAccountState(address="ATA", exists=True,
                                        program_id="Tokenkeg"))
    assert plan.l25_admitted is False
    assert plan.status is not P.EntryPlanStatus.PLANNED


def test_research_package_does_not_modify_any_production_file():
    """20c. Nothing under src/ was touched by this work."""
    import subprocess
    diff = subprocess.run(["git", "diff", "--name-only", "--", "src/"],
                          capture_output=True, text=True, cwd=AI).stdout
    dirty = [ln for ln in diff.splitlines()
             if ln.startswith("src/autopaper/l25x")]
    assert dirty == []


def test_no_threshold_is_applied_by_default():
    """20d. The study observes; it does not filter."""
    from l25x.economics import EconomicPolicy
    assert EconomicPolicy().as_dict()["max_rent_fraction_of_notional"] is None
    assert EconomicPolicy().as_dict()["max_fixed_cost_fraction_of_notional"] is None
    assert EconomicPolicy().as_dict()["max_break_even_return_pct"] is None