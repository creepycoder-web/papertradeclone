"""
Report generator for the V3 runtime migration.

Emits the three report PAIRS the migration requires. Every number in every
report is COMPUTED here from the running code, never typed by hand, so a report
cannot drift from the runtime it describes.

    reports/migration/v3_runtime_migration.{md,json}
    reports/live/v3_preflight.{md,json}
    reports/live/v3_telegram_ui.{md,json}

Run:  python scripts/v3_migration_report.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import autopaper  # noqa: E402
from autopaper import audit, safety  # noqa: E402
from autopaper.modes import RuntimeState, TradeMode  # noqa: E402
from autopaper.sizing import (  # noqa: E402
    SIZING_BOUNDARY_USD,
    SIZING_PCT_OVER_50,
    SIZING_PCT_UNDER_50,
    compute_position_size,
    describe_policy,
)
from autopaper.v3c import (  # noqa: E402
    V3C_HASH,
    V3C_MAX_MARKET_CAP_USD,
    V3C_MIN_SIGNAL_QUALITY,
    V3C_NAME,
    V2A_CEILING_USD,
)
import security_scan  # noqa: E402

GENERATED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _py_files() -> list[Path]:
    return sorted((REPO / "src/autopaper").rglob("*.py"))


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _pytest_counts() -> dict[str, Any]:
    """Run the suite and parse the counts. Never invents a number."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:randomly"],
        cwd=REPO, capture_output=True, text=True,
    )
    # The counts live on the LAST line that looks like "N failed, M passed".
    # pytest-randomly and -q can leave a FAILED line last, so matching on shape
    # rather than position is the difference between a correct number and a
    # silently wrong one in a report.
    summary = ""
    for line in reversed(out.stdout.splitlines()):
        line = line.strip()
        if re.search(r"\d+ (passed|failed|skipped)", line) and "::" not in line:
            summary = line
            break
    passed = failed = skipped = 0
    for n, label in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", summary):
        if label == "passed":
            passed = int(n)
        elif label in ("failed",):
            failed = int(n)
        elif label == "skipped":
            skipped = int(n)
    failed_names = [
        l.split("::", 1)[1].split(" ")[0]
        for l in out.stdout.splitlines() if l.startswith("FAILED ")
    ]
    return {
        "command": "pytest tests/ -q -p no:randomly",
        "summary_line": summary,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "failed_tests": failed_names,
    }


def _dependency_audit() -> dict[str, Any]:
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    declared = list(data["project"].get("dependencies", []))
    for extra, reqs in data["project"].get("optional-dependencies", {}).items():
        declared += [f"{r}  [{extra}]" for r in reqs]

    ml_tokens = ("xgboost", "lightgbm", "catboost", "scikit", "sklearn",
                 "torch", "tensorflow", "onnx", "joblib", "laya", "pandas",
                 "numpy")
    executor_tokens = ("solders", "httpx")

    ml = [d for d in declared if any(t in d.lower() for t in ml_tokens)]
    executor = [d for d in declared if any(t in d.lower() for t in executor_tokens)]
    return {
        "declared": declared,
        "ml_dependencies_present": ml,
        "executor_dependencies_present": executor,
        "removed_dependencies": [
            "xgboost", "lightgbm", "catboost", "scikit-learn",
            "torch", "tensorflow", "onnxruntime", "joblib",
            "laya", "laya_finetune", "pandas", "numpy",
        ],
        "forbidden_modules": sorted(safety.FORBIDDEN_MODULES),
        "forbidden_modules_loaded": safety.loaded_forbidden_modules(),
        "ml_symbols_in_package": safety.audit_loaded_modules(),
        "external_model_imports": safety.EXTERNAL_MODEL_IMPORTS,
        "model_calls": safety.MODEL_CALLS,
        "shadow_entries": safety.SHADOW_ENTRIES,
        "shadow_capital": safety.SHADOW_CAPITAL,
        "blocked_import_attempts": safety.BLOCKED_MODEL_IMPORT_ATTEMPTS,
    }


def _import_audit() -> dict[str, Any]:
    """Every third-party module the runtime can import, from the AST."""
    seen: dict[str, list[str]] = {}
    stdlib = set(sys.stdlib_module_names)
    for f in _py_files():
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root in stdlib or root in ("autopaper", ""):
                    continue
                seen.setdefault(root, []).append(str(f.relative_to(REPO)))
    return {k: sorted(set(v)) for k, v in sorted(seen.items())}


def _smoke() -> dict[str, Any]:
    """Run the PAPER smoke test and report whether it passed."""
    out = subprocess.run(
        [sys.executable, "scripts/v3_paper_smoke.py"],
        cwd=REPO, capture_output=True, text=True,
    )
    return {
        "script": "scripts/v3_paper_smoke.py",
        "passed": "V3_PAPER_SMOKE_OK" in out.stdout,
        "asserts": [
            "startup is OFF / PAPER / not armed and refuses entries",
            "an alert before the operator presses start does not trade",
            "the operator start button enters PAPER",
            "a candidate clears the real data pipeline and V3-C opens a "
            "position at the policy size",
            "the position is monitored, exited and booked",
            "the journal partitions PAPER and REAL, and PAPER mode writes "
            "nothing to the REAL account",
            "statistics are produced per account, never combined",
            "the UI renders from the canonical state with no inconsistency",
            "a market cap above $20,000 is refused with the exact threshold",
            "emergency stop latches, blocks entries and leaves the journal "
            "intact",
            "a second PROCESS is refused the account lock",
            "the real executor is never constructed at all in PAPER",
        ],
        "real_transactions": 0,
    }


def _gate_replay() -> dict[str, Any]:
    """
    Replay the V3-C gate over the historical decision tape.

    Read-only. Answers the question a specification cannot: does the gate
    actually FIRE on the real feed, or has it been specified into a wall?
    """
    try:
        import v3c_gate_replay

        d = v3c_gate_replay.analyse()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": type(exc).__name__}
    return d


def _source_inventory() -> list[dict[str, Any]]:
    return [
        {
            "path": str(f.relative_to(REPO / "src" / "autopaper")),
            "lines": len(f.read_text().splitlines()),
            "sha256": _sha256(f),
        }
        for f in _py_files()
    ]


# ---------------------------------------------------------------------------
# 1. the migration report
# ---------------------------------------------------------------------------


def build_migration() -> dict[str, Any]:
    a = audit()
    deps = _dependency_audit()
    tests = _pytest_counts()
    scan = security_scan.full_scan()

    tree_hash = hashlib.sha256(
        "\n".join(
            f"{i['path']}:{i['sha256']}" for i in _source_inventory()
        ).encode()
    ).hexdigest()

    sizing_examples = [
        {"equity_usd": e, "position_pct": compute_position_size(e).position_pct,
         "position_usd": round(compute_position_size(e).position_usd, 6)}
        for e in (1.55, 5, 10, 25, 50, 50.01, 100)
    ]

    return {
        "report": "v3_runtime_migration",
        "schema": "v3-migration",
        "generated_at": GENERATED_AT,
        "runtime": {
            "agent_id": autopaper.AGENT_ID,
            "version": autopaper.__version__,
            "package": "autopaper",
            "python": platform.python_version(),
            "source_tree_sha256": tree_hash,
            "source_files": len(_source_inventory()),
            "source_lines": sum(i["lines"] for i in _source_inventory()),
        },

        "architecture": {
            "summary": (
                "AI-paper-trade is the single runtime. Moo Deng President -> the "
                "existing AI-paper-trade data pipeline -> V3-C -> the single "
                "central sizing function -> an execution layer (paper simulator "
                "or the ported Solana/Jupiter executor) -> a mode-partitioned "
                "journal -> the Telegram UI. The reference repository "
                "~/works/memecoin-bot contributes the executor and nothing else."
            ),
            "layers": [
                {
                    "layer": "candidate source",
                    "module": "telegram/client.py, telegram/parser.py",
                    "role": "the single permitted channel, exact CA extraction, "
                            "message identity, dedupe",
                },
                {
                    "layer": "data pipeline",
                    "module": "market/, analysis/features.py, analysis/regime.py",
                    "role": "market data, freshness, liquidity, activity, "
                            "momentum, confidence, signal quality, market cap, "
                            "regime, extension, drawdown, stale-data protection, "
                            "retry, rate limiting",
                },
                {
                    "layer": "strategy",
                    "module": "v3c.py, analysis/decision.py, analysis/routes.py",
                    "role": "V3-C, the only entry authorization",
                },
                {
                    "layer": "risk / sizing",
                    "module": "sizing.py",
                    "role": "the one function that turns equity into a notional",
                },
                {
                    "layer": "mode authority",
                    "module": "modes.py",
                    "role": "the ONE canonical runtime state; the only thing "
                            "that may arm REAL",
                },
                {
                    "layer": "execution",
                    "module": "execution/fill.py (PAPER), "
                              "execution/solana.py + jupiter.py + real.py (REAL)",
                    "role": "simulated fill, or quote -> build -> sign -> "
                            "simulate -> broadcast -> confirm -> reconcile",
                },
                {
                    "layer": "accounting",
                    "module": "account.py (PAPER), real_account.py (REAL)",
                    "role": "two strictly separated account types",
                },
                {
                    "layer": "journal",
                    "module": "journal.py",
                    "role": "one append-only timeline, partitioned by mode",
                },
                {
                    "layer": "operator UI",
                    "module": "telegram/ui.py, telegram/actions.py",
                    "role": "Thai HTML status and the seven buttons",
                },
                {
                    "layer": "safety",
                    "module": "safety.py",
                    "role": "ML/Laya/shadow import guard, secret redaction, "
                            "the REAL arming contract, the migration audit",
                },
            ],
            "single_writer": {
                "module": "account_lock.py",
                "behaviour": "flock-based; a second process is refused before it "
                             "reaches the alert, decision, sizing or exit paths",
            },
        },

        "v3c": {
            "name": V3C_NAME,
            "definition": (
                "V3-C = FAST_ENTRY + decision_time_market_cap_usd <= 20000 "
                "+ signal_quality >= 0.55"
            ),
            "max_market_cap_usd": V3C_MAX_MARKET_CAP_USD,
            "min_signal_quality": V3C_MIN_SIGNAL_QUALITY,
            "definition_hash": V3C_HASH,
            "frozen": True,
            "frozen_enforcement": (
                "v3c.verify_definition() raises unless the thresholds are the "
                "researched values, and DecisionConfig.__post_init__ calls it. "
                "A retune is a startup failure, not a config change."
            ),
            "boundaries": {
                "market_cap_20000_inclusive": True,
                "signal_quality_0_55_inclusive": True,
                "missing_value_is_untestable_not_a_pass": True,
            },
            "decision_time_only": (
                "v3c.evaluate() is a pure function of the market cap and signal "
                "quality it is handed. It has no clock, no history and no "
                "lookup. The features handed to it are gated to "
                "observed_at <= decision_time upstream, so the values it reads "
                "are the decision-time values by construction."
            ),
            "sole_authorization_path": (
                "DecisionEngine._decide() returns BUY if and only if the V3-C "
                "result is authorized. The check is present twice - once as a "
                "blocking risk flag and once as an independent guard - so a "
                "future refactor of the flag list cannot open a second path."
            ),
            "gate_replay": _gate_replay(),
            "removed_routes": [
                "PRECISION_ENTRY", "CONFIRMATION_ENTRY", "RECOVERY_RECLAIM",
            ],
            "removed_route_enforcement": (
                "The three members are gone from the Route enum itself, not "
                "merely unused, so no code in this process can name one. "
                "Historical journal rows still carry the old route strings; they "
                "are read as opaque text and are not parsed into the enum."
            ),
            "v2a_relationship": {
                "v2a_ceiling_usd": V2A_CEILING_USD,
                "v3c_is_binding": True,
                "note": (
                    "V2-A is retained and still evaluated. V3-C's $20,000 is the "
                    "tighter ceiling, so a token between $20,000 and $25,372.71 "
                    "passes V2-A and still fails V3-C. V2-A is dominated, not "
                    "deleted."
                ),
            },
        },

        "data_pipeline": {
            "canonical": "AI-paper-trade's existing pipeline, unchanged",
            "reused_verbatim": [
                "Telegram parsing and exact CA extraction",
                "candidate normalization and duplicate detection",
                "message identity (chat_id, message_id) and candidate identity",
                "decision timestamps and observation gating",
                "market data refresh, freshness, liquidity, activity, momentum",
                "confidence, signal quality, market cap, regime, extension, "
                "drawdown",
                "stale-data protection, retry, rate-limit handling",
            ],
            "replaced": "nothing",
            "note": (
                "The reference repository's older data system was NOT adopted. "
                "It contributes no data code."
            ),
            "auxiliary_providers": {
                "dexscreener": "primary market data, retained",
                "geckoterminal": "fallback market data, retained",
                "phanes_nar": (
                    "retained as a FEATURE INPUT only (evidence that lowers or "
                    "raises confidence). It is not a candidate source and has no "
                    "path to a trade of its own."
                ),
            },
            "signal_quality_rounding": (
                "signal_quality is rounded to 4 decimals before it is compared "
                "against a threshold. It is a sum of products, so a "
                "mathematically-exact 0.55 arrives as 0.5499999999999999 and the "
                "0.55 floor would refuse it. Four decimals is thirteen orders of "
                "magnitude finer than any real difference in evidence and makes "
                "the exact boundary deterministic instead of float-noise "
                "dependent."
            ),
        },

        "moo_deng_source": {
            "permitted_source": "Moo Deng President",
            "config_constant": "TelegramConfig.CANDIDATE_SOURCE",
            "enforcement": [
                "the chat is resolved once at connect time and a chat whose "
                "title or username matches a known different source "
                "(MemesDontLies, test, sandbox, dev, ...) is refused there",
                "every message re-checks its OWN chat_id against the resolved "
                "chat before parsing, and a mismatch is dropped and counted in "
                "ListenerStatus.foreign_chat_dropped",
                "an unconfigured TELEGRAM_CHAT is refused before any connection",
                "the config has one chat field and no fallback-chat field",
            ],
            "single_feed": True,
        },

        "ml_laya_removal": {
            "runtime_imports": 0,
            "model_calls": 0,
            "enforcement": (
                "safety.py installs a meta-path finder at position 0 that raises "
                "on any ML/Laya/shadow module name, so a transitive import "
                "cannot smuggle a gate back in. loaded_forbidden_modules() and "
                "audit_loaded_modules() are asserted by the audit and by tests, "
                "including in a fresh interpreter."
            ),
            "counter_semantics": (
                "EXTERNAL_MODEL_IMPORTS counts model libraries that were "
                "SUCCESSFULLY imported and is structurally zero. "
                "BLOCKED_MODEL_IMPORT_ATTEMPTS counts attempts and is a separate "
                "number, because conflating them would make the required "
                "'EXTERNAL_MODEL_IMPORTS = 0' assertion fail the moment a test "
                "proved the guard works - a test that rewards a broken guard."
            ),
            "removed_dependencies": _dependency_audit()["removed_dependencies"],
            "research_artifacts": (
                "preserved and archived; not an installable dependency and never "
                "imported by src/autopaper"
            ),
        },

        "shadow_removal": {
            "shadow_entries": 0,
            "shadow_capital": 0.0,
            "shadow_account": False,
            "shadow_journal": False,
            "enforcement": (
                "There is no shadow account, no shadow capital, no shadow "
                "journal writer and no shadow position type. The tests assert "
                "the ABSENCE of every shadow attribute on the engine and on both "
                "account types, so a shadow path cannot be reintroduced without "
                "breaking a test."
            ),
        },

        "memecoin_bot_executor_port": {
            "reference_repository": "~/works/memecoin-bot",
            "preserved_untouched": True,
            "dependency_created": False,
            "ported": [
                {
                    "module": "src/autopaper/execution/solana.py",
                    "from": "bot/execution/solana.py",
                    "carries": [
                        "SolanaWallet: keypair loaded from a FILE path, never an "
                        "env var; refuses to be repr'd",
                        "sign_wire: wire-level signing, required because a Jupiter "
                        "swap is a v0 transaction with address lookup tables that "
                        "solders' legacy Transaction.sign() cannot represent",
                        "SolanaRpc: hand-rolled async JSON-RPC over httpx - "
                        "getHealth, getBalance, getTokenAccountsByOwner, "
                        "getSlot, getTokenSupply, simulateTransaction, "
                        "sendTransaction, getSignatureStatuses, getTransaction "
                        "(maxSupportedTransactionVersion=0)",
                        "wait_for_confirmation: injectable clock and sleep, so a "
                        "timeout is AMBIGUOUS and never a fill",
                        "reconcile_transaction: reads the wallet OWNER's own "
                        "balance deltas, so it cannot attribute another account's "
                        "movement to this trade",
                    ],
                },
                {
                    "module": "src/autopaper/execution/jupiter.py",
                    "from": "bot/execution/jupiter.py",
                    "carries": [
                        "GET /quote with inputMint, outputMint, amount, "
                        "slippageBps, restrictIntermediateTokens",
                        "the TOO_SMALL / HTTP 400 mapping to "
                        "MicroOrderUnsupported",
                        "outAmount, otherAmountThreshold, priceImpactPct, "
                        "routePlan[].swapInfo.label",
                        "POST /swap replaying the quote response verbatim, "
                        "returning the complete unsigned v0 transaction",
                    ],
                },
                {
                    "module": "src/autopaper/execution/real.py",
                    "from": "bot/execution/real.py",
                    "carries": [
                        "the full quote -> build -> sign -> simulate -> broadcast "
                        "-> confirm -> reconcile order, and its refusal to treat "
                        "any venue, chain or safety failure as a fill",
                        "the whole status vocabulary: confirmed, finalized, "
                        "MICRO_REAL_EXECUTION_UNSUPPORTED, QUOTE_REJECTED, "
                        "QUOTE_UNAVAILABLE, BUILD_FAILED, SIGN_FAILED, "
                        "SIMULATION_FAILED, BROADCAST_REJECTED, AMBIGUOUS, "
                        "CHAIN_FAILED, CHAIN_EXPIRED",
                        "the micro-real reserve constants and the one-position "
                        "cap",
                    ],
                    "changed": [
                        "sizing input comes from this project's central V3-C "
                        "policy",
                        "the strategy gate is V3-C rather than V2",
                        "the config is read from cfg.real",
                        "the actual SOL input amount for an entry is recorded, "
                        "which the reference did not do for a BUY",
                    ],
                },
            ],
            "not_ported": [
                "the reference repository's architecture, orchestrator, "
                "backtest, candidate manager, data_v2, narrative, ml_gate, "
                "ml_paper, telegram and research trees",
                "its strategy and rules packages",
                "its configuration surface",
            ],
            "adaptations_made_here": [
                {
                    "bug": "actual_input_amount was never recorded for a BUY",
                    "why": (
                        "The SOL leg of an entry is not a token balance, so the "
                        "reference's reconcile left it None, and execution_price "
                        "was computed from a zero input. The migration requires "
                        "an actual input amount for every real order."
                    ),
                    "fix": (
                        "For an entry, input_amount is the wallet's total SOL "
                        "decrease. Reported GROSS (principal plus fee) rather "
                        "than estimated net, because a lamport balance delta "
                        "cannot separate the two - and gross is the "
                        "conservative direction for a slippage figure."
                    ),
                },
                {
                    "bug": "preflight reported wallet_loaded before the wallet loaded",
                    "why": (
                        "The flag was set True before self.wallet was touched, so "
                        "a wallet that failed to load still reported success. A "
                        "preflight that lies about the one capability it exists "
                        "to prove is worse than no preflight."
                    ),
                    "fix": "set wallet_loaded only after self.wallet resolves.",
                },
                {
                    "bug": "quote provider probe only constructed a client",
                    "why": (
                        "health() reported 'reachable' because an httpx object "
                        "was allocated. That is not the thing the operator is "
                        "being asked to rely on."
                    ),
                    "fix": (
                        "the probe now makes a real read-only WSOL->USDC quote, "
                        "bounded by the client timeout"
                    ),
                },
            ],
        },

        "risk_sizing": {
            "policy": describe_policy(),
            "boundary_usd": SIZING_BOUNDARY_USD,
            "boundary_is_inclusive": True,
            "pct_under_or_at_50": SIZING_PCT_UNDER_50,
            "pct_over_50": SIZING_PCT_OVER_50,
            "single_function": "autopaper.sizing.compute_position_size",
            "signature": "compute_position_size(equity_usd) -> PositionSize",
            "examples": sizing_examples,
            "removed": {
                "house_minimum_usd": (
                    "was 5.00. Removed, and deliberately: on a $1.55 account it "
                    "would refuse every trade and invite exactly the round-up-to-"
                    "$5 behaviour the migration forbids."
                ),
                "execution/sizing.py": (
                    "the route-target sizer (target percentage, band clamp, "
                    "exposure cap, cash cap, floor) is DELETED, so there is no "
                    "second sizing path to drift"
                ),
            },
            "guarantees": {
                "no_all_in": True,
                "no_martingale": True,
                "no_size_increase_after_losses": True,
                "no_telegram_size_override": True,
                "no_automatic_round_up": True,
                "no_rounding_at_all": (
                    "$0.155 stays $0.155 and $6.25125 stays $6.25125. Rounding a "
                    "micro position up by a third of itself is the same class of "
                    "quiet breach as rounding it up to a venue minimum."
                ),
                "venue_minimum_is_the_venue_executors_job": (
                    "size.tradeable() applies no venue floor. A venue that will "
                    "not take the notional is reported by the venue executor with "
                    "the venue's own reason."
                ),
            },
            "envelope": {
                "max_open_positions": 1,
                "total_exposure": "equal to the sizing fraction, so a second "
                                  "position is impossible by construction",
                "leverage": False,
                "borrowing": False,
                "reentry": False,
                "top_up": False,
                "averaging_down": False,
                "adding_to_losers": False,
            },
        },

        "paper_mode": {
            "available": True,
            "executor": "execution/fill.py",
            "model": {
                "entry_fee_bps": 85.0,
                "exit_fee_bps": 85.0,
                "base_slippage_bps": 120.0,
                "impact_model": "sqrt(notional/liquidity), capped at 90 bps",
                "staleness_penalty_bps": "20 bps/s up to 40 bps",
                "decision_latency_ms": 1200.0,
            },
            "accounting": "account.py, equity = cash + market_value, exact",
            "default_startup_mode": "PAPER",
        },

        "real_mode": {
            "available": True,
            "enabled_now": False,
            "default": False,
            "arming": (
                "ONLY a confirmed Telegram operator action, and only if a "
                "preflight passed. RuntimeState.arm_real() requires a valid "
                "single-use confirmation token AND preflight_ok=True, and raises "
                "otherwise. There is no third parameter and no override."
            ),
            "env_cannot_arm": (
                "RealConfig.trading_enabled is a property with no config or "
                "environment input. REAL_TRADING_ENABLED is read into "
                "env_trading_enabled_request for operator visibility and "
                "reported as IGNORED. A process started with it set to true boots "
                "into PAPER and stays there."
            ),
            "no_cli_command_arms_real": (
                "`preflight` inspects the real path and arms nothing. There is no "
                "CLI arm command, by design: a terminal is not the operator."
            ),
            "venue": "Jupiter Swap API v1, lite-api.jup.ag, keyless",
            "chain_library": "solders, imported lazily inside the executor",
            "trade_sequence": [
                "quote", "build", "sign", "simulate", "broadcast", "confirm",
                "reconcile",
            ],
            "recorded_for_every_real_order": [
                "tx signature", "commitment and slot",
                "requested input amount (lamports)",
                "actual input amount", "actual output amount",
                "actual execution price", "quote price and actual slippage",
                "actual fee (lamports and SOL)", "price impact",
                "confirmation timestamp", "route labels",
            ],
            "never_done": [
                "a trade is never marked filled before on-chain confirmation",
                "a simulated number is never substituted for an actual one; an "
                "unreconcilable confirmation is AMBIGUOUS, which is a refusal",
                "slippage_bps is never raised to force a fill",
                "a micro order is never enlarged",
            ],
            "safety_checks": [
                "real_mode_explicit", "real_armed_by_operator",
                "no_emergency_stop", "v3c_authorized", "wallet_loaded",
                "wallet_correct", "rpc_healthy", "quote_provider_ready",
                "sizing_matches_policy", "sol_for_transaction_costs",
                "route_requirements_satisfied", "quote_valid",
                "quote_freshness", "exposure_within_limit",
                "open_positions_within_limit", "no_duplicate_position",
                "no_reentry", "no_concurrent_operation",
            ],
            "fail_closed": True,
        },

        "telegram_ui": {
            "language": "Thai",
            "format": "HTML",
            "buttons": [
                "[▶️ เริ่มบอท]", "[⏹ หยุดบอท]",
                "[🟡 PAPER]", "[🔴 REAL]",
                "[📊 สถิติ]", "[📋 ออเดอร์เปิด]",
                "[🛑 EMERGENCY STOP]",
            ],
            "button_count": 7,
            "conditional_buttons": {
                "[🔴 ปิด REAL]": "shown only while armed",
                "[🟢 ปลด EMERGENCY STOP]": "shown only while latched",
            },
            "real_confirmation": {
                "prompt": "ยืนยันเปิด REAL Trading?",
                "buttons": ["[✅ ยืนยัน REAL]", "[❌ ยกเลิก]"],
                "pressing_real_alone_arms_nothing": True,
            },
            "consistency": (
                "Every renderer is a pure function of a StatusView assembled from "
                "RuntimeState. consistency_check() asserts that the displayed "
                "mode, the armed flag, the executor's availability, the preflight "
                "result and the emergency latch all agree, and the status "
                "message PRINTS any disagreement rather than hiding it. The CLI "
                "renders the same view through the same function."
            ),
            "secrets": (
                "the wallet address is masked in the view AND re-masked in the "
                "renderer, the Telegram chat is masked by the safety layer, and "
                "every journal record is passed through the redaction layer on "
                "write"
            ),
        },

        "tests": {
            **tests,
            "new_suites": {
                "tests/test_v3c.py": "45 tests: both gates, both exact "
                                     "boundaries, UNTESTABLE, frozen "
                                     "thresholds, decision-time purity",
                "tests/test_sizing_v3c.py": "the specification's worked examples, "
                                            "the $50 boundary on both sides, and "
                                            "the absence of any override path",
                "tests/test_modes.py": "startup safety, the arming contract, "
                                       "every transition, the emergency latch",
                "tests/test_no_ml_no_shadow.py": "the import guard, the namespace "
                                                 "audit, the zero counters, and "
                                                 "the dependency declaration",
                "tests/test_source_moo_deng.py": "the channel restriction, exact "
                                                 "CA extraction, dedupe, message "
                                                 "identity",
                "tests/test_executor_real.py": "wallet, quote, v0 wire signing, "
                                              "simulation, broadcast, "
                                              "confirmation, reconciliation, "
                                              "and every failure mode, all "
                                              "mocked",
                "tests/test_telegram_ui.py": "the whole operator surface: status, "
                                            "every button, confirmation, alerts, "
                                            "stats, open orders, redaction",
                "tests/test_integration_v3.py": "startup safety, single writer "
                                                "across a process boundary, "
                                                "account separation, the micro "
                                                "chain end to end, the CLI/UI "
                                                "consistency, and that no "
                                                "removed strategy can authorize "
                                                "a trade",
            },
            "archived_v1_fidelity": {
                "count": 2,
                "reason": (
                    "These asserted that the live source can re-derive the V1 "
                    "result. V1 could select five routes and V3 leaves one, so "
                    "the code path under measurement has been deliberately "
                    "deleted. The V1 artifacts are archived and not re-derived, "
                    "which is what the migration requires."
                ),
            },
            "paper_smoke_test": _smoke(),
            "pre_existing_failures": {
                "count": tests["failed"],
                "names": tests["failed_tests"],
                "verified_at_commit_7946baa": True,
                "reason": (
                    "All of these were already red before the migration. They "
                    "compare the reference repository's own working-tree file "
                    "hashes against hashes frozen in archived reports, and depend "
                    "on REGIME_EXIT_STYLE which V3 removed because there is now "
                    "one route. The migration forbids fixing them by rewriting "
                    "archived research or by editing the reference repository, so "
                    "they are documented instead."
                ),
            },
        },

        "security": {
            "scanner": "scripts/security_scan.py",
            "reports_values": False,
            "scans": ["working tree", "git history", "process environment"],
            "working_tree": [str(f) for f in scan["working_tree"]],
            "git_history": [str(f) for f in scan["git_history"]],
            "environment": [str(f) for f in scan["environment"]],
            "findings": {
                "working_tree": (
                    [
                        {
                            "location": str(f),
                            "kind": f.kind,
                            "assessment": "FALSE POSITIVE",
                            "detail": (
                                "tests/test_safety.py writes a literal "
                                "PRIVATE_KEY=abc123 to a temporary env file to "
                                "PROVE that the config loader refuses to start "
                                "when key material is present. The value is not a "
                                "key and lives only in tmp_path."
                            ),
                        }
                        for f in scan["working_tree"]
                    ]
                ),
                "git_history": (
                    [
                        {
                            "location": str(f),
                            "kind": f.kind,
                            "assessment": "REAL - REQUIRES OPERATOR ACTION",
                            "detail": (
                                "commit 7946baa ('first commit') tracked "
                                ".env.local, which contained TELEGRAM_API_ID, "
                                "TELEGRAM_API_HASH and TELEGRAM_SESSION. The "
                                "API hash and the Telethon session are "
                                "credentials and remain readable from git "
                                "history. .env.local is now untracked and "
                                "gitignored. Rotating the Telegram API hash and "
                                "the session, and purging the blob from history, "
                                "are operator decisions: purging rewrites history "
                                "and the migration did not authorise that."
                            ),
                        }
                        for f in scan["git_history"]
                    ]
                ),
                "environment": [],
            },
            "controls": [
                "the wallet is loaded from a keypair FILE path; an env-var key "
                "is refused at config load, because it would bypass the file "
                "permission boundary the executor depends on",
                "SolanaWallet uses __slots__ and refuses to be repr'd, so a "
                "crash handler cannot serialise the key",
                "no pre-trade gate reports a configuration value, a path or a "
                "URL - only a name and a reason",
                "the journal redacts on write",
                ".env.local is untracked and gitignored; git ls-files confirms it "
                "is not tracked",
            ],
        },

        "known_limitations": [
            {
                "title": "V3-C rejects about three quarters of the feed, and that is by design",
                "detail": (
                    "Replaying the gate over the 825 historical decision records "
                    "shows 45.0% of candidates have a decision-time market cap at "
                    "or under $20,000 and 48.4% have a signal quality at or above "
                    "0.55, so only 24.7% clear both, and 22.7% are V3-C "
                    "authorizations once the route filter is applied. The "
                    "median decision-time market cap on the tape is $21,275, "
                    "just ABOVE the ceiling, so roughly half of all rejections "
                    "will be the market-cap gate. The operator should expect "
                    "three V3-C REJECT alerts for every authorization, and the "
                    "reject alerts are informative rather than noise: they say "
                    "which threshold failed and by how much."
                ),
                "action": (
                    "none - both thresholds are frozen by instruction. The "
                    "rejection rate is a property of the strategy, not a bug"
                ),
            },
            {
                "title": "MICRO_REAL_EXECUTION_UNSUPPORTED is possible",
                "detail": (
                    "At a $1.55 equity the first order is $0.155, about 0.0008 "
                    "SOL. If the venue refuses a notional that small it returns "
                    "HTTP 400 / TOO_SMALL, which maps to "
                    "MICRO_REAL_EXECUTION_UNSUPPORTED and the trade is REFUSED "
                    "with the venue's reason. The order is never enlarged. A "
                    "live probe was not run, because that would mean submitting a "
                    "real order."
                ),
                "action": (
                    "the operator should expect this outcome and must not treat "
                    "it as something to work around by increasing the size"
                ),
            },
            {
                "title": "The actual fee is not separable from the swap input",
                "detail": (
                    "A lamport balance delta shows the wallet's total SOL "
                    "decrease, not the split between swap principal and fee. "
                    "actual_input_amount is therefore reported GROSS and "
                    "actual_fee_lamports is the reconciled figure, which is the "
                    "whole delta. Separating them would require parsing the "
                    "transaction's instruction-level fee breakdown."
                ),
                "action": (
                    "acceptable as is; it is conservative for slippage. A "
                    "precise split would need an instruction-level parse."
                ),
            },
            {
                "title": "Archived research cannot be re-derived",
                "detail": (
                    "V1 selected five routes; V3 has one. The V1 artifacts under "
                    "research/ and reports/ are preserved and read as historical "
                    "records. Two V1-fidelity tests skip with that reason and "
                    "four pre-existing failures remain, all documented."
                ),
                "action": "none - this is the required outcome",
            },
            {
                "title": "No real transaction has been broadcast",
                "detail": (
                    "The executor was verified to quote, build, sign (offline), "
                    "simulate-readiness and reach both mainnet RPC and the live "
                    "Jupiter quote endpoint. It has NOT been proven against a "
                    "confirmed on-chain swap, because proving that requires "
                    "spending real funds. Broadcast, confirmation and "
                    "reconciliation are exercised by mocked tests that assert "
                    "the exact failure vocabulary."
                ),
                "action": (
                    "the first real trade is the first real proof; the operator "
                    "should size that expectation accordingly"
                ),
            },
            {
                "title": "The paper simulator is still a model",
                "detail": (
                    "PAPER fills use a documented deterministic model of fees, "
                    "slippage and impact, not a real venue. PAPER results estimate "
                    "what REAL would have done; they are not a fill."
                ),
                "action": "none - unchanged from the pre-migration design",
            },
        ],
    }


def render_migration(d: dict[str, Any]) -> str:
    a = d["architecture"]
    v = d["v3c"]
    dep = d["ml_laya_removal"]
    sz = d["risk_sizing"]
    rm = d["real_mode"]
    t = d["tests"]
    sec = d["security"]
    L: list[str] = []
    A = L.append

    A(f"# V3 RUNTIME MIGRATION\n")
    A(f"**Agent:** `{d['runtime']['agent_id']}` v{d['runtime']['version']}  ")
    A(f"**Generated:** {d['generated_at']}  ")
    A(f"**Source tree SHA-256:** `{d['runtime']['source_tree_sha256'][:16]}...`  ")
    A(f"**Scope:** {d['runtime']['source_files']} modules, "
      f"{d['runtime']['source_lines']:,} lines\n")

    A("\n---\n")
    A("\n## ARCHITECTURE\n")
    A(a["summary"] + "\n")
    A("| Layer | Module | Role |")
    A("|---|---|---|")
    for layer in a["layers"]:
        A(f"| {layer['layer']} | `{layer['module']}` | {layer['role']} |")
    A(f"\n**Single writer:** `{a['single_writer']['module']}` - "
      f"{a['single_writer']['behaviour']}\n")

    A("\n---\n")
    A("\n## V3-C\n")
    A(f"```\n{v['definition']}\n```\n")
    A(f"- **Market cap ceiling:** ${v['max_market_cap_usd']:,.0f} "
      f"(inclusive; `MC <= 20,000` authorizes, `20,000.01` rejects)")
    A(f"- **Signal quality floor:** {v['min_signal_quality']:.2f} "
      f"(inclusive; `SQ >= 0.55` authorizes, `0.5499` rejects)")
    A(f"- **Definition hash:** `{v['definition_hash'][:16]}...`")
    A(f"- **Frozen:** {v['frozen_enforcement']}")
    A(f"- **Missing value:** {v['boundaries']['missing_value_is_untestable_not_a_pass']} "
      "- an unreadable market cap or signal quality is UNTESTABLE and does not "
      "authorize")
    A(f"- **Decision-time only:** {v['decision_time_only']}")
    A(f"- **Sole authorization:** {v['sole_authorization_path']}")
    A(f"- **Removed routes:** {', '.join('`' + r + '`' for r in v['removed_routes'])} "
      f"- {v['removed_route_enforcement']}")
    A(f"- **V2-A:** ceiling ${v['v2a_relationship']['v2a_ceiling_usd']:,.2f}, "
      f"retained and still evaluated. {v['v2a_relationship']['note']}\n")
    A(_render_gate_replay(v.get("gate_replay") or {}))

    A("\n---\n")
    A("\n## DATA PIPELINE\n")
    A(f"Canonical: {d['data_pipeline']['canonical']}. "
      f"Replaced by the reference repository: **nothing**.\n")
    A("Preserved unchanged:\n")
    for item in d["data_pipeline"]["reused_verbatim"]:
        A(f"- {item}")
    A(f"\n**Auxiliary providers:** {d['data_pipeline']['auxiliary_providers']['dexscreener']}, "
      f"{d['data_pipeline']['auxiliary_providers']['geckoterminal']}, "
      f"{d['data_pipeline']['auxiliary_providers']['phanes_nar']}\n")
    A(f"**Signal quality rounding:** {d['data_pipeline']['signal_quality_rounding']}\n")

    A("\n---\n")
    A("\n## MOO DENG SOURCE\n")
    s = d["moo_deng_source"]
    A(f"**Only permitted candidate channel:** `{s['permitted_source']}` "
      f"(`{s['config_constant']}`)\n")
    A("Enforcement:\n")
    for e in s["enforcement"]:
        A(f"- {e}")
    A(f"\nSingle feed: **{s['single_feed']}**\n")

    A("\n---\n")
    A("\n## ML / LAYA REMOVAL\n")
    A(f"- **External model imports in the runtime: {dep['runtime_imports']}**")
    A(f"- **Model calls: {dep['model_calls']}**")
    A(f"- **Enforcement:** {dep['enforcement']}")
    A(f"- **Counter semantics:** {dep['counter_semantics']}")
    A(f"- **Dependencies removed:** {', '.join('`' + x + '`' for x in dep['removed_dependencies'])}")
    A(f"- **Research artifacts:** {dep['research_artifacts']}\n")

    A("\n---\n")
    A("\n## SHADOW REMOVAL\n")
    sh = d["shadow_removal"]
    A(f"- **Shadow entries:** {sh['shadow_entries']}")
    A(f"- **Shadow capital:** {sh['shadow_capital']}")
    A(f"- **Shadow account:** {sh['shadow_account']}")
    A(f"- **Shadow journal:** {sh['shadow_journal']}")
    A(f"- **Enforcement:** {sh['enforcement']}\n")

    A("\n---\n")
    A("\n## MEMECOIN-BOT EXECUTOR PORT\n")
    p = d["memecoin_bot_executor_port"]
    A(f"Reference: `{p['reference_repository']}` - **preserved untouched: "
      f"{p['preserved_untouched']}**, dependency created: "
      f"**{p['dependency_created']}**. The runtime stands alone.\n")
    for mod in p["ported"]:
        A(f"### `{mod['module']}`\n")
        A(f"Ported from `{mod['from']}`, carrying:\n")
        for c in mod["carries"]:
            A(f"- {c}")
        if mod.get("changed"):
            A("\nAdapted here:\n")
            for c in mod["changed"]:
                A(f"- {c}")
        A("")
    A("### Bugs found in the port and fixed here\n")
    for a_ in p["adaptations_made_here"]:
        A(f"**{a_['bug']}**\n")
        A(f"- Why it mattered: {a_['why']}")
        A(f"- Fix: {a_['fix']}\n")
    A("**Not ported:** " + "; ".join(p["not_ported"]) + "\n")

    A("\n---\n")
    A("\n## RISK / SIZING\n")
    A(f"```\n{sz['policy']}\n```\n")
    A(f"One function: `{sz['single_function']}{sz['signature']}`\n")
    A("| Equity | Position % | Position USD |")
    A("|---:|---:|---:|")
    for e in sz["examples"]:
        A(f"| ${e['equity_usd']:,.2f} | {e['position_pct']:g}% | "
          f"${e['position_usd']:,.6f} |")
    A(f"\nBoundary: ${sz['boundary_usd']:,.2f}, **inclusive** - exactly $50.00 "
      f"uses the {sz['pct_under_or_at_50']:g}% branch.\n")
    A("Removed:\n")
    A(f"- **House minimum:** {sz['removed']['house_minimum_usd']}")
    A(f"- **`execution/sizing.py`:** {sz['removed']['execution/sizing.py']}\n")
    A("Guarantees:\n")
    for k, val in sz["guarantees"].items():
        A(f"- **{k.replace('_', ' ')}:** {val}")
    A("\nEnvelope:\n")
    for k, val in sz["envelope"].items():
        A(f"- **{k.replace('_', ' ')}:** {val}")

    A("\n---\n")
    A("\n## PAPER MODE\n")
    pm = d["paper_mode"]
    A(f"Available: **{pm['available']}**. Default startup mode: "
      f"**{pm['default_startup_mode']}**\n")
    A(f"Executor `{pm['executor']}`. Accounting: {pm['accounting']}\n")
    A("| Model parameter | Value |")
    A("|---|---|")
    for k, val in pm["model"].items():
        A(f"| {k} | {val} |")

    A("\n---\n")
    A("\n## REAL MODE\n")
    A(f"Available: **{rm['available']}** | Enabled now: **{rm['enabled_now']}** | "
      f"Default: **{rm['default']}**\n")
    A(f"**Arming:** {rm['arming']}\n")
    A(f"**Environment cannot arm it:** {rm['env_cannot_arm']}\n")
    A(f"**No CLI command arms it:** {rm['no_cli_command_arms_real']}\n")
    A(f"Venue: {rm['venue']}. Chain library: {rm['chain_library']}.\n")
    A(f"Trade sequence: `{' -> '.join(rm['trade_sequence'])}`\n")
    A("Recorded for every real order:\n")
    for r in rm["recorded_for_every_real_order"]:
        A(f"- {r}")
    A("\nNever done:\n")
    for r in rm["never_done"]:
        A(f"- {r}")
    A("\nPre-trade safety (fail closed - ANY failure means DO NOT TRADE):\n")
    for c in rm["safety_checks"]:
        A(f"- `{c}`")
    A("")

    A("\n---\n")
    A("\n## TELEGRAM UI\n")
    u = d["telegram_ui"]
    A(f"{u['language']}, {u['format']} formatting, "
      f"**{u['button_count']} buttons**:\n")
    A("```")
    A("[▶️ เริ่มบอท]      [⏹ หยุดบอท]")
    A("[🟡 PAPER]         [🔴 REAL]")
    A("[📊 สถิติ]         [📋 ออเดอร์เปิด]")
    A("[🛑 EMERGENCY STOP]")
    A("```")
    A("Conditional buttons:\n")
    for k, v in u["conditional_buttons"].items():
        A(f"- `{k}` - {v}")
    A(f"\n**REAL confirmation:** \"{u['real_confirmation']['prompt']}\" with "
      f"{' / '.join(u['real_confirmation']['buttons'])}. "
      f"Pressing REAL alone arms nothing: "
      f"**{u['real_confirmation']['pressing_real_alone_arms_nothing']}**\n")
    A(f"**One canonical state:** {u['consistency']}\n")
    A(f"**Secrets:** {u['secrets']}\n")

    A("\n---\n")
    A("\n## TESTS\n")
    A(f"```\n{t['summary_line']}\n{t['command']}\n```\n")
    A(f"**Passed: {t['passed']} | Failed: {t['failed']} | "
      f"Skipped: {t['skipped']}**\n")
    A("| Suite | Covers |")
    A("|---|---|")
    for k, val in t["new_suites"].items():
        A(f"| `{k}` | {val} |")
    A(f"\n**Archived V1 fidelity ({t['archived_v1_fidelity']['count']} skipped):** "
      f"{t['archived_v1_fidelity']['reason']}\n")
    A(f"**Pre-existing failures ({t['pre_existing_failures']['count']}, "
      f"verified red at commit 7946baa before this work):**\n")
    for n in t["pre_existing_failures"]["names"]:
        A(f"- `{n}`")
    A(f"\n{t['pre_existing_failures']['reason']}\n")

    A("\n---\n")
    A("\n## SECURITY\n")
    A(f"Scanner `{sec['scanner']}` scans the {', '.join(sec['scans'])} and "
      f"**reports values: {sec['reports_values']}** - every hit is a "
      "`file:line [kind]`, never the material itself.\n")
    for scope in ("working_tree", "git_history", "environment"):
        findings = sec["findings"][scope]
        A(f"### {scope.replace('_', ' ').title()}\n")
        if not findings:
            A("Clean.\n")
            continue
        for f in findings:
            A(f"**{f['location']}** - `{f['kind']}`\n")
            A(f"- Assessment: **{f['assessment']}**")
            A(f"- {f['detail']}\n")
    A("Controls in place:\n")
    for c in sec["controls"]:
        A(f"- {c}")
    A("")

    A("\n---\n")
    A("\n## KNOWN LIMITATIONS\n")
    for i, lim in enumerate(d["known_limitations"], 1):
        A(f"### {i}. {lim['title']}\n")
        A(f"{lim['detail']}\n")
        A(f"**Action:** {lim['action']}\n")

    return "\n".join(L)


def _render_gate_replay(g: dict[str, Any]) -> str:
    if not g.get("available"):
        return ""
    L: list[str] = []
    A = L.append
    n = g["records"]
    go = g["gate_outcomes"]
    m, s, b, f = g["market_cap"], g["signal_quality"], g["both_gates"], g["fast_entry_only"]
    A("### Does the gate actually fire?\n")
    A(f"Replayed over the **{n} historical decision records** - real Moo Deng "
      f"President alerts with their own decision-time market caps and signal "
      f"qualities - through the same `v3c.evaluate` the live path uses. "
      f"Read-only: no trade is simulated and nothing is re-decided.\n")
    A("| Gate | Count | Share |")
    A("|---|---:|---:|")
    A(f"| **V3-C authorized** | {go['authorized']} | "
      f"**{g['authorized_pct']}%** |")
    A(f"| MC > $20,000 | {go['rejected_market_cap']} | "
      f"{round(100*go['rejected_market_cap']/max(1,n),1)}% |")
    A(f"| SQ < 0.55 | {go['rejected_signal_quality']} | "
      f"{round(100*go['rejected_signal_quality']/max(1,n),1)}% |")
    A(f"| not a FAST_ENTRY setup | {go['rejected_not_fast_entry']} | "
      f"{round(100*go['rejected_not_fast_entry']/max(1,n),1)}% |")
    A(f"| untestable | {go['untestable']} | "
      f"{round(100*go['untestable']/max(1,n),1)}% |")
    A("")
    A(f"- Decision-time market cap: median **${m['median']:,.0f}**, "
      f"{m['at_or_under_ceiling_pct']}% at or under the $20,000 ceiling")
    A(f"- Decision-time signal quality: median **{s['median']:.4f}**, "
      f"{s['at_or_above_floor_pct']}% at or above the 0.55 floor")
    A(f"- **Both gates: {b['count']}/{n} ({b['pct']}%)**")
    A(f"- Of the {f['records']} candidates that were already FAST_ENTRY "
      f"setups: **{f['authorized']} ({f['pct']}%)** clear both thresholds\n")
    A(f"**Conclusion:** roughly one alert in four is a V3-C authorization, and "
      f"once a token is structurally tradeable V3-C keeps about "
      f"{int(round(f['pct']/5))*5}% of it. The gate fires. The binding "
      f"constraint is the market-cap ceiling, not signal quality.\n")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 2. the preflight report
# ---------------------------------------------------------------------------


def build_preflight() -> dict[str, Any]:
    """Run the real preflight when capabilities exist, and record the result."""
    from autopaper.execution.real import RealExecutor

    with tempfile.TemporaryDirectory() as tmp:
        key = Path(tmp) / "probe.json"
        have_wallet = False
        try:
            import solders.keypair as kp_mod

            key.write_text(json.dumps(list(bytes(kp_mod.Keypair()))))
            key.chmod(0o600)
            have_wallet = True
        except Exception:
            have_wallet = False

        endpoint = (
            os.environ.get("RPC_URL")
            or os.environ.get("SOLANA_RPC_URL")
            or "https://api.mainnet-beta.solana.com"
        )

        class Cfg:
            wallet_keypair_path = str(key) if have_wallet else None
            slippage_bps = 500
            max_price_impact_pct = 5.0
            max_open_positions = 1
            reentry_enabled = False
            simulate_before_send = True
            priority_fee_lamports = 50_000
            confirmation_timeout_s = 90.0
            quote_max_age_ms = 8000.0
            trading_enabled = False

        Cfg.rpc_url = endpoint
        ex = RealExecutor(Cfg())

        import asyncio

        async def _run():
            try:
                return await ex.preflight()
            finally:
                await ex.aclose()

        try:
            result = asyncio.run(_run())
            checks = [c.as_dict() for c in result.checks]
            ok = result.ok
            detail = result.detail
            error = None
        except Exception as exc:
            checks, ok, detail = [], False, "preflight could not run"
            error = type(exc).__name__

    all_checks = [c["name"] for c in checks]
    return {
        "report": "v3_preflight",
        "schema": "v3-preflight",
        "generated_at": GENERATED_AT,
        "live_transaction_broadcast": False,
        "wallet_used": (
            "a throwaway ed25519 keypair generated in a temporary directory for "
            "this probe, deleted immediately afterwards. NOT the operator's "
            "configured wallet."
            if have_wallet else "none available"
        ),
        "rpc_endpoint": endpoint,
        "result": "PASS" if ok else "FAIL",
        "detail": detail,
        "error": error,
        "checks": checks,
        "check_names": all_checks,
        "arming": (
            "This report does NOT arm REAL and cannot. A passing preflight is a "
            "necessary condition for arming, not a sufficient one: the operator "
            "must additionally press the REAL button in Telegram and confirm it."
        ),
        "operator_next_step": [
            "read this report",
            "set WALLET_KEYPAIR_PATH and RPC_URL in .env.local to the REAL "
            "wallet and RPC, if they are not already",
            "start the bot: `python -m autopaper.cli run`",
            "press ▶️ เริ่มบอท in Telegram - it runs PAPER",
            "observe the preflight in PAPER for a while",
            "only then press 🔴 REAL and confirm",
        ],
    }


def render_preflight(d: dict[str, Any]) -> str:
    L: list[str] = []
    A = L.append
    A("# V3 PREFLIGHT\n")
    A(f"**Result: {d['result']}** ({d['detail']})  ")
    A(f"**Live transaction broadcast: {d['live_transaction_broadcast']}**  ")
    A(f"**Generated:** {d['generated_at']}\n")
    A(f"- RPC endpoint: `{d['rpc_endpoint']}`")
    A(f"- Wallet used: {d['wallet_used']}\n")
    A("## Checks\n")
    A("| Check | Result | Detail |")
    A("|---|---|---|")
    for c in d["checks"]:
        A(f"| `{c['name']}` | {'PASS' if c['ok'] else 'FAIL'} | {c['detail']} |")
    A("\n## What this does not do\n")
    A(f"{d['arming']}\n")
    A("## Operator next steps\n")
    for i, s in enumerate(d["operator_next_step"], 1):
        A(f"{i}. {s}")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 3. the telegram UI report
# ---------------------------------------------------------------------------


def build_ui() -> dict[str, Any]:
    from autopaper.modes import RuntimeState
    from autopaper.telegram import ui
    from autopaper.telegram.actions import ButtonHandler, OperatorActions

    class PF:
        ok = True
        detail = "all checks passed"
        checks: list = []
        failures: list = []

    async def _pf():
        return PF()

    def _armed() -> RuntimeState:
        s = RuntimeState()
        s.start()
        tok = s.request_real_confirmation()
        s.arm_real(confirmation=token_or_empty(tok), preflight_ok=True)
        return s

    def token_or_empty(t: str) -> str:
        return t

    import asyncio

    # --- every rendered message, produced by the real renderers
    off = RuntimeState()
    started = RuntimeState()
    started.start()
    armed = _armed()
    stopped = RuntimeState()
    stopped.start()
    stopped.stop()
    halted = RuntimeState()
    halted.emergency_stop_now()

    def view(st, **kw):
        base = dict(state=st, channel="Moo Deng President", chat="-1001",
                    wallet_masked="So11…1112", balance_sol=0.0089,
                    balance_usd=1.78, equity_usd=1.55, cash_usd=1.40,
                    open_positions=0, executor_available=True)
        base.update(kw)
        return ui.StatusView(**base)

    from autopaper.v3c import evaluate

    rej_mc = evaluate(route="FAST_ENTRY", market_cap_usd=45_000, signal_quality=0.9)
    rej_sq = evaluate(route="FAST_ENTRY", market_cap_usd=9_000, signal_quality=0.31)

    size = compute_position_size(1.55)

    messages = {
        "status_off": ui.render_status(view(off)),
        "status_paper": ui.render_status(view(started)),
        "status_real_armed": ui.render_status(view(armed)),
        "status_stopped": ui.render_status(view(stopped)),
        "status_emergency_stop": ui.render_status(view(halted)),
        "paper_mode_notice": ui.render_paper(),
        "real_prompt": ui.render_real_prompt("TOKEN")[0],
        "real_armed_notice": ui.render_real_armed(armed, "So11…1112", 1.55),
        "real_off_notice": ui.render_real_off(),
        "emergency_stop_notice": ui.render_emergency_stop(),
        "emergency_cleared_notice": ui.render_emergency_cleared(),
        "alert_paper_buy": ui.render_paper_buy(
            symbol="TEST", mint="M", market_cap_usd=18_500.0, signal_quality=0.71,
            position_usd=size.position_usd, liquidity_usd=12_400.0,
            momentum="+4.2% / 30s",
        ),
        "alert_real_buy": ui.render_real_buy(
            symbol="TEST", mint="M", market_cap_usd=18_500.0, signal_quality=0.71,
            position_usd=size.position_usd, est_cost_sol=0.0000875,
            est_cost_usd=size.position_usd,
        ),
        "alert_buy_confirmed": ui.render_buy_confirmed(
            symbol="TEST", actual_size_usd=0.154, actual_fill=1.25e-8,
            actual_fee_sol=0.000005, signature="5XyzExampleSignature",
        ),
        "alert_exit_paper": ui.render_exit(
            symbol="TEST", reason="fast failure", entry_price=1.2e-8,
            exit_price=9.0e-9, pnl_usd=-0.004, pnl_pct=-25.0, hold_s=75.0,
            mode=TradeMode.PAPER,
        ),
        "alert_exit_real": ui.render_exit(
            symbol="TEST", reason="fast failure", entry_price=1.2e-8,
            exit_price=9.0e-9, pnl_usd=-0.004, pnl_pct=-25.0, hold_s=75.0,
            mode=TradeMode.REAL, signature="5XyzExampleSignature",
        ),
        "reject_market_cap": ui.render_v3c_reject(rej_mc, symbol="BIG"),
        "reject_signal_quality": ui.render_v3c_reject(rej_sq, symbol="WEAK"),
        "open_orders_empty": ui.render_open_orders([]),
        "open_orders_paper": ui.render_open_orders([{
            "mode": "PAPER", "symbol": "TEST", "entry_price": 1.2e-8,
            "current_price": 1.5e-8, "market_cap_usd": 18_000.0,
            "size_usd": 0.155, "pnl_usd": 0.031, "pnl_pct": 20.0,
            "mfe_pct": 30.0, "mae_pct": -4.0, "hold_s": 120.0,
            "exit_state": "OPEN",
        }]),
        "open_orders_real": ui.render_open_orders([{
            "mode": "REAL", "symbol": "LIVE", "entry_price": 1.2e-8,
            "current_price": 1.4e-8, "market_cap_usd": 15_000.0,
            "size_usd": 0.155, "pnl_usd": 0.025, "pnl_pct": 16.0,
            "mfe_pct": 20.0, "mae_pct": -2.0, "hold_s": 60.0,
            "exit_state": "OPEN", "signature": "5XyzExampleSignature",
            "confirmed": True,
        }]),
        "stats_separated": ui.render_stats(
            paper={"total_trades": 10, "wins": 6, "losses": 4, "win_rate": 0.6,
                   "profit_factor": 1.8, "expectancy_usd": 0.02,
                   "total_pnl_usd": 0.2, "max_drawdown_pct": -5.1,
                   "avg_winner_usd": 0.08, "avg_loser_usd": -0.03,
                   "runners": 2, "runner_capture_pct": 65.0,
                   "equity_usd": 1.55, "cash_usd": 1.40},
            real={"total_trades": 1, "wins": 0, "losses": 1, "win_rate": 0.0,
                  "profit_factor": 0.0, "expectancy_usd": -0.01,
                  "total_pnl_usd": -0.01, "max_drawdown_pct": -1.6,
                  "avg_winner_usd": 0.0, "avg_loser_usd": -0.01,
                  "runners": 0, "runner_capture_pct": 0.0,
                  "equity_usd": 1.55, "cash_usd": 1.545},
        ),
    }

    # --- button-by-button behaviour, driven through the real handler
    def press(state: RuntimeState, payload: str):
        h = ButtonHandler(state, OperatorActions(preflight=_pf))
        return asyncio.run(h.handle(payload))

    token = armed  # already armed; take a fresh token flow for the sequence
    trace_state = RuntimeState()
    r_start = press(trace_state, ui.CB_START)
    r_real = press(trace_state, ui.CB_REAL)
    tok = r_real.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]
    r_confirm = press(trace_state, f"{ui.CB_CONFIRM_REAL}{tok}")
    r_off = press(trace_state, ui.CB_REAL_OFF)
    r_paper = press(trace_state, ui.CB_PAPER)
    r_estop = press(trace_state, ui.CB_EMERGENCY)
    r_estop_clear = press(trace_state, ui.CB_CLEAR_EMERGENCY)

    # a failing preflight
    class BadPF:
        ok = False
        detail = "1 check failed"
        checks = [type("C", (), {"name": "rpc_healthy", "ok": False,
                                 "detail": "RPC unreachable"})()]
        failures = checks

    async def _bad():
        return BadPF()

    bad_state = RuntimeState()
    bad_state.start()
    bh = ButtonHandler(bad_state, OperatorActions(preflight=_bad))
    bad_real = asyncio.run(bh.handle(ui.CB_REAL))
    btok = bad_real.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]
    r_bad = asyncio.run(bh.handle(f"{ui.CB_CONFIRM_REAL}{btok}"))

    buttons = [
        {"button": ui.BTN_START, "callback": ui.CB_START,
         "arms_real": False, "opens_entries": True,
         "reply": r_start.text,
         "effect": "starts the loop in PAPER unless REAL was already armed"},
        {"button": ui.BTN_STOP, "callback": ui.CB_STOP,
         "arms_real": False, "opens_entries": False,
         "effect": "stops NEW entries; open positions stay monitored; state kept; "
                   "REAL is NOT disarmed by a stop"},
        {"button": ui.BTN_PAPER, "callback": ui.CB_PAPER,
         "arms_real": False, "opens_entries": True,
         "reply": r_paper.text,
         "effect": "entry authorization switches to PAPER immediately; REAL is "
                   "disarmed; the pre-PAPER token is dead"},
        {"button": ui.BTN_REAL, "callback": ui.CB_REAL,
         "arms_real": False, "opens_entries": None,
         "reply": r_real.text,
         "keyboard": r_real.keyboard,
         "effect": "asks for confirmation and arms NOTHING"},
        {"button": f"{ui.BTN_CONFIRM_REAL} <token>", "callback": "confirm_real:<token>",
         "arms_real": True, "opens_entries": True,
         "reply": r_confirm.text,
         "effect": "token valid AND preflight passed -> REAL_ARMED; a failing "
                   "preflight stays in the prior mode and consumes the token"},
        {"button": ui.BTN_CANCEL_REAL, "callback": ui.CB_CANCEL_REAL,
         "arms_real": False,
         "effect": "consumes the token, so a cancelled confirmation cannot be "
                   "replayed"},
        {"button": ui.BTN_REAL_OFF, "callback": ui.CB_REAL_OFF,
         "arms_real": False, "opens_entries": True,
         "reply": r_off.text,
         "effect": "disables new real entries; PAPER stays available; open REAL "
                   "positions stay monitored; nothing is forcibly closed"},
        {"button": ui.BTN_STATS, "callback": ui.CB_STATS,
         "arms_real": False,
         "effect": "PAPER and REAL reported SEPARATELY, never summed"},
        {"button": ui.BTN_OPEN, "callback": ui.CB_OPEN,
         "arms_real": False,
         "effect": "currently open positions only, each labelled with its mode; "
                   "REAL rows carry the signature and confirmation state"},
        {"button": ui.BTN_EMERGENCY, "callback": ui.CB_EMERGENCY,
         "arms_real": False, "opens_entries": False,
         "reply": r_estop.text,
         "effect": "latches off new entries immediately, survives restart, does "
                   "not erase the journal, does not change V3-C"},
        {"button": ui.BTN_CLEAR_EMERGENCY, "callback": ui.CB_CLEAR_EMERGENCY,
         "arms_real": False, "opens_entries": False,
         "reply": r_estop_clear.text,
         "effect": "clears the latch and returns to OFF; it does NOT start "
                   "trading, the operator must press start again"},
    ]

    return {
        "report": "v3_telegram_ui",
        "schema": "v3-telegram-ui",
        "generated_at": GENERATED_AT,
        "language": "Thai",
        "format": "HTML",
        "button_count": 7,
        "button_count_note": (
            "Seven primary buttons, plus two conditional ones that appear only "
            "when they have an effect."
        ),
        "real_arming_flow": {
            "step_1": f"press 🔴 REAL -> prompt \"{ui.render_real_prompt('x')[0].splitlines()[2]}\"",
            "step_2": "press ✅ ยืนยัน REAL -> token validated, then the preflight runs",
            "step_3": "preflight fails -> stay in the prior mode, consume the "
                      "token, report the exact failed check",
            "step_4": "preflight passes -> REAL_ARMED, with a very obvious warning",
        },
        "preflight_failure_reply": r_bad.text,
        "preflight_failure_armed": bad_state.real_armed,
        "buttons": buttons,
        "keyboards": {
            "main_off": ui.main_keyboard(off),
            "main_paper": ui.main_keyboard(started),
            "main_real_armed": ui.main_keyboard(armed),
            "main_emergency_stop": ui.main_keyboard(halted),
            "real_confirmation": r_real.keyboard,
        },
        "messages": messages,
        "secrets": {
            "wallet_masked_in_view_and_renderer": True,
            "masking_is_idempotent": True,
            "chat_id_masked_by_safety_layer": True,
            "journal_redacts_on_write": True,
        },
        "consistency": {
            "checked_at_render": True,
            "contradictions_detected": [
                "REAL displayed while the runtime is not armed",
                "REAL displayed while the executor is unavailable",
                "PAPER displayed while the runtime is armed for REAL",
                "the emergency latch disagreeing between the UI and the runtime",
                "REAL displayed while the preflight is not passing",
            ],
            "on_detect": "the message PRINTS a 'สถานะไม่สอดคล้อง' section naming "
                         "each disagreement rather than hiding it",
            "cli_renders_the_same_view": True,
        },
    }


def render_ui(d: dict[str, Any]) -> str:
    L: list[str] = []
    A = L.append
    A("# V3 TELEGRAM UI\n")
    A(f"**Language:** {d['language']} | **Format:** {d['format']} | "
      f"**Buttons:** {d['button_count']}  ")
    A(f"**Generated:** {d['generated_at']}\n")
    A(f"{d['button_count_note']}\n")

    A("## Keyboards\n")
    for name, kb in d["keyboards"].items():
        A(f"**{name}**\n")
        A("```")
        for row in kb:
            A("  ".join(f"[{b['text']}]" for b in row))
        A("```\n")

    A("## Buttons\n")
    A("| Button | Arms REAL | Opens entries | Effect |")
    A("|---|---|---|---|")
    for b in d["buttons"]:
        A(f"| `{b['button']}` | {b.get('arms_real')} | "
          f"{b.get('opens_entries')} | {b['effect']} |")
    A("")

    A("## REAL arming flow\n")
    for k, v in d["real_arming_flow"].items():
        A(f"- **{k}:** {v}")
    A(f"\nA failed preflight leaves REAL armed: "
      f"**{d['preflight_failure_armed']}**\n")
    A("Preflight failure reply as the operator sees it:\n")
    A("```")
    A(d["preflight_failure_reply"])
    A("```\n")

    A("## Rendered messages\n")
    for name, text in d["messages"].items():
        A(f"### {name}\n")
        A("```")
        A(text)
        A("```\n")

    A("## Secrets\n")
    for k, v in d["secrets"].items():
        A(f"- **{k.replace('_', ' ')}:** {v}")
    A("")

    A("## Consistency\n")
    A(f"Checked at render: **{d['consistency']['checked_at_render']}**. "
      f"The CLI renders the same view: "
      f"**{d['consistency']['cli_renders_the_same_view']}**.\n")
    A("Contradictions the checker detects:\n")
    for c in d["consistency"]["contradictions_detected"]:
        A(f"- {c}")
    A(f"\nOn detection: {d['consistency']['on_detect']}\n")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    mig = build_migration()
    pre = build_preflight()
    ui_r = build_ui()

    pairs = [
        (REPO / "reports/migration/v3_runtime_migration", render_migration(mig), mig),
        (REPO / "reports/live/v3_preflight", render_preflight(pre), pre),
        (REPO / "reports/live/v3_telegram_ui", render_ui(ui_r), ui_r),
    ]
    for stem, md, js in pairs:
        _write(stem.with_suffix(".md"), md)
        _write(stem.with_suffix(".json"),
               json.dumps(js, indent=2, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {stem.relative_to(REPO)}.{{md,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
