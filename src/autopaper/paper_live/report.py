"""
Report writers for the execution-aware PAPER forward-capture dataset.

WHAT IS WRITTEN
===============

* ``execution_aware_paper.json`` - the full machine-readable dataset.
* ``execution_aware_paper.md``   - the human summary.
* ``execution_aware_paper_trades.csv``   - one row per simulated trade.
* ``execution_aware_paper_signals.csv`` - one row per SIGNAL PER ARM.

The signals file is the one that matters most and is deliberately not a subset
of the trades file. A dataset of trades alone cannot answer "is execution
quality itself filtering out candidates?", because the candidates that never
became trades are exactly the ones that would answer it.

WHAT IS NEVER WRITTEN
=====================

No credential, no Telegram session, no key, no wallet state. The raw signal text
is stored only as a SHA-256 plus its length; the token fields the decision used
are written in full. That keeps the dataset joinable to a message by id without
it becoming a copy of every message the bot ever saw.

DETERMINISM
===========

Row order is the order the records were produced, and every float is rendered
through a single quantiser, so two runs over the same records produce
byte-identical files. A dataset that reorders itself between runs is a dataset
whose diffs cannot be read.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from autopaper.same_market.numeric import quantise, to_float

from .arms import ArmSpec
from .engine import BASELINE_HASH, SCHEMA_VERSION
from .state import atomic_write_text
from .schema import (
    ACTUAL_FILL_AVAILABLE,
    SIMULATED_FILL_KIND,
    SignalRecord,
    TradeRecord,
)

REPORT_DIR = Path("reports") / "live"

#: The standing statement every artefact carries.
NO_EXECUTION_NOTE = "NO TRANSACTION WAS BUILT, SIGNED, OR BROADCAST."


def _q(value: Any, places: int = 10) -> str:
    """One quantiser for every float, so the output is stable."""
    if value is None:
        return ""
    try:
        return str(quantise(value, places))
    except Exception:  # noqa: BLE001 - a display failure must not lose the row
        return str(value)


class DatasetWriter:
    """Writes the four artefacts under ``root/reports/live``."""

    def __init__(self, root: Path | None = None, *, arms: Sequence[ArmSpec] = ()) -> None:
        self.base = (root or Path.cwd()) / REPORT_DIR
        self.arms = list(arms)

    def _ensure(self) -> Path:
        self.base.mkdir(parents=True, exist_ok=True)
        return self.base

    # -- json ---------------------------------------------------------------

    def write_json(
        self,
        *,
        signals: Sequence[SignalRecord],
        trades: Sequence[TradeRecord],
        status: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        payload = {
            "schema": SCHEMA_VERSION,
            "generated_at": time.time(),
            "mode": "PAPER_ONLY",
            "price_layers": {
                "OBSERVED_MARK": "DexScreener observation. NEVER a fill.",
                "EXECUTABLE_QUOTE": (
                    "a read-only price from on-chain state for this exact order "
                    "size. Not a fill."
                ),
                "SIMULATED_FILL": SIMULATED_FILL_KIND,
                "ACTUAL_FILL": None,
                "ACTUAL_FILL_AVAILABLE": ACTUAL_FILL_AVAILABLE,
                "statement": (
                    "OBSERVED_MARK != EXECUTABLE_QUOTE != SIMULATED_FILL != "
                    "ACTUAL_FILL"
                ),
            },
            "realized_pnl_basis": (
                "simulated settlement between two SIMULATED_FILLs; never "
                "exit_mark / entry_mark - 1"
            ),
            "baseline_hash": BASELINE_HASH,
            "execution_note": NO_EXECUTION_NOTE,
            "arms": [spec.as_dict() for spec in self.arms],
            "status": dict(status),
            "signals": [r.as_dict() for r in signals],
            "trades": [r.as_dict() for r in trades],
            "mark_to_quote_gaps": self._gap_distribution(signals),
            "mark_to_executable_premium": self._premium_distribution(signals),
            "mark_to_executable_gap": self._mark_to_executable_distribution(signals),
            "failures": self._failure_distribution(signals, trades),
            **dict(extra or {}),
        }
        path = self._ensure() / "execution_aware_paper.json"
        # ATOMIC. A kill during a write must not leave a truncated dataset that
        # the next run reads as if it were whole.
        atomic_write_text(path, json.dumps(payload, indent=2, default=str))
        return path

    # -- the two CSVs -------------------------------------------------------

    SIGNAL_COLUMNS: tuple[str, ...] = (
        "signal_id", "message_id", "telegram_timestamp", "receive_timestamp",
        "channel", "token_mint", "symbol", "strategy", "strategy_version",
        "baseline_hash", "l25_decision", "l25_status", "l25_authorized",
        "arm_filter_reason", "observed_entry_mark", "buy_quote_price",
        "buy_quote_gap_pct", "simulated_entry_price", "entry_quote_timestamp",
        "entry_quote_age_ms", "entry_quote_provider", "entry_quote_venue",
        "entry_quote_pool", "entry_quote_route", "execution_identity",
        "position_size_usd", "position_size_pct", "bankroll_before", "traded",
        "failure_reason", "failure_detail",
        # -- ECONOMIC-VERDICT DIAGNOSTIC (additive) ----------------------
        # A MICRO_ACCOUNT_ECONOMICALLY_INVALID block records the FULL verdict
        # on the row, not only the reason sentence, so the notional / rent /
        # break-even that tripped the guard are visible without reading JSON.
        # Blank means the economic guard did not run on this signal. The eight
        # columns below are emitted in EXACTLY the order ``_signal_row`` writes
        # them; a blank cell means the guard did not run, never a zero verdict.
        "economic_planned_notional_usd",
        "economic_rent_usd",
        "economic_fixed_cost_usd",
        "economic_rent_over_notional",
        "economic_fixed_cost_over_notional",
        "economic_break_even_return_pct",
        "economic_classification",
        "economic_sol_usd_reference",
        "decision_latency_ms",
        "buy_quote_latency_ms", "signal_to_entry_ms",
        # -- ENTRY EXECUTION-GAP OBSERVABILITY (additive columns) -----------
        # Emitted only when the underlying values exist; blank cells mean the
        # value was unavailable, never fabricated. These columns make the entry
        # price-quality question answerable per row. They do NOT change any
        # existing historical column.
        "entry_mark_timestamp",
        "entry_mark_age_ms",
        "entry_native_reference_price_usd",
        "entry_native_reference_method",
        "entry_native_reference_available",
        "entry_executable_buy_price_usd",
        "entry_executable_quote_timestamp",
        "entry_executable_quote_age_ms",
        "entry_mark_to_executable_gap_pct",
        "entry_native_to_executable_gap_pct",
        "entry_venue_id",
        "entry_pool_address",
    )

    TRADE_COLUMNS: tuple[str, ...] = (
        "signal_id", "arm", "mint", "symbol", "strategy_version", "baseline_hash",
        "bankroll_before", "position_size", "position_size_pct", "usd_notional",
        "sol_notional", "sol_usd_reference", "observed_entry_mark",
        "buy_quote_price", "simulated_entry_price", "entry_quote_provider",
        "entry_quote_venue", "entry_quote_pool", "entry_quote_timestamp",
        "entry_quote_age_ms", "exit_trigger_timestamp", "observed_exit_mark",
        "sell_quote_price", "simulated_exit_price", "exit_quote_timestamp",
        "exit_quote_age_ms", "exit_quote_provider", "exit_quote_venue",
        "exit_quote_pool", "exit_quote_route",
        # MFE/MAE are LABELLED by source. MARK-based excursions are named with
        # the MARK_ prefix so they are never read as executable returns.
        "observed_mark_mfe", "observed_mark_mae",
        # Executable excursions are emitted ONLY when executable evidence exists.
        "executable_mfe_pct", "executable_mae_pct",
        "executable_mfe_available", "executable_mae_available",
        "hold_seconds", "gross_pnl_quote", "fees_quote",
        "costs_quote", "net_pnl_quote", "realized_pnl_usd", "realized_pnl_sol",
        "return_pct", "bankroll_after", "drawdown_pct", "exit_reason",
        "telegram_latency_ms", "decision_latency_ms", "buy_quote_latency_ms",
        "signal_to_entry_ms", "sell_quote_latency_ms", "total_lifecycle_ms",
        "failure_reason",
    )

    def write_signals_csv(self, signals: Sequence[SignalRecord]) -> Path:
        path = self._ensure() / "execution_aware_paper_signals.csv"
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(self.SIGNAL_COLUMNS)
        for record in signals:
            writer.writerow(self._signal_row(record))
        atomic_write_text(path, buf.getvalue())
        return path

    def write_trades_csv(self, trades: Sequence[TradeRecord]) -> Path:
        path = self._ensure() / "execution_aware_paper_trades.csv"
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(self.TRADE_COLUMNS)
        for trade in trades:
            writer.writerow(self._trade_row(trade))
        atomic_write_text(path, buf.getvalue())
        return path

    # -- row builders -------------------------------------------------------

    @staticmethod
    def _signal_row(record: SignalRecord) -> list[str]:
        q = record.buy_quote
        lat = record.latency.as_dict()
        econ = record.economics
        return [
            record.signal.signal_id,
            "" if record.signal.message_id is None else str(record.signal.message_id),
            _q(record.signal.telegram_timestamp, 6),
            _q(record.signal.receive_timestamp, 6),
            record.signal.channel,
            record.signal.token_mint or "",
            record.signal.symbol or "",
            record.arm, record.arm_version, record.baseline_hash,
            record.l25_decision, record.l25_status,
            "true" if record.l25_authorized else "false",
            record.arm_filter_reason,
            _q(record.observed_mark.value_usd),
            _q(q.effective_price_usd),
            _q(record.mark_to_quote_gap_pct, 6),
            # USD form. This column sits between two USD/token columns
            # (observed_entry_mark, buy_quote_price); emitting the quote-asset
            # (SOL) form here put a SOL price next to USD prices in one row and
            # is the exact token/SOL-vs-token/USD mix this project forbids.
            _q(q.effective_price_usd),
            _q(q.quote_timestamp, 6),
            _q(q.quote_age_ms, 3),
            q.provider or "", q.venue or "", q.pool_or_curve or "",
            "|".join(q.route),
            record.execution_identity,
            _q(record.position_size_usd, 6),
            _q(record.position_size_pct, 4),
            _q(record.bankroll_before, 6),
            "true" if record.traded else "false",
            record.failure.value,
            record.failure_detail[:300],
            # Economics (blank when the guard did not run).
            _q(None if econ is None else econ.get("planned_notional_usd"), 6),
            _q(None if econ is None else econ.get("estimated_new_token_account_rent_usd"), 6),
            _q(None if econ is None else econ.get("fixed_cost_usd"), 6),
            _q(None if econ is None else econ.get("rent_over_notional"), 4),
            _q(None if econ is None else econ.get("fixed_cost_over_notional"), 4),
            _q(None if econ is None else econ.get("break_even_return_pct"), 4),
            (econ or {}).get("classification", ""),
            _q(None if econ is None else econ.get("sol_usd_reference"), 4),
            _q(lat.get("decision_latency_ms"), 3),
            _q(lat.get("buy_quote_latency_ms"), 3),
            _q(lat.get("signal_to_simulated_entry_ms"), 3),
            # -- ENTRY EXECUTION-GAP OBSERVABILITY (additive) ----------------
            _q(record.entry_observed_mark_timestamp, 6),
            _q(record.entry_mark_age_ms, 3),
            _q(record.entry_native_reference_price_usd),
            record.entry_native_reference_method or "",
            "true" if record.entry_native_reference_available else "false",
            _q(record.entry_executable_buy_price_usd),
            _q(record.entry_executable_quote_timestamp, 6),
            _q(record.entry_executable_quote_age_ms, 3),
            _q(record.entry_mark_to_executable_gap_pct, 6),
            _q(record.entry_native_to_executable_gap_pct, 6),
            record.entry_venue_id or "",
            record.entry_pool_address or "",
        ]

    @staticmethod
    def _trade_row(trade: TradeRecord) -> list[str]:
        p = trade.position
        entry = p.entry_fill
        exit_fill = trade.exit_fill
        sell = trade.sell_quote
        pnl = trade.pnl
        lat = trade.latency.as_dict()
        lifecycle = (
            None if lat.get("signal_to_simulated_entry_ms") is None
            or lat.get("exit_trigger_to_fill_ms") is None
            else lat["signal_to_simulated_entry_ms"] + lat["exit_trigger_to_fill_ms"]
        )
        return [
            p.signal_id, p.arm, p.mint, p.symbol or "", SCHEMA_VERSION, BASELINE_HASH,
            _q(pnl.bankroll_before if pnl else p.bankroll_before, 8),
            _q(p.position_size_usd, 6),
            _q(p.position_size_pct, 4),
            _q(p.usd_notional, 6),
            _q(p.sol_notional, 12),
            _q(p.sol_usd_reference, 4),
            _q(p.entry_observed_mark),
            _q(entry.effective_price_usd),
            # USD form: see the note in _signal_row. Every price column in this
            # row is USD/whole-token, so the columns are directly comparable.
            _q(entry.effective_price_usd),
            entry.provider, entry.venue or "", entry.pool_or_curve or "",
            _q(entry.quote_timestamp, 6),
            _q(entry.quote_age_ms, 3),
            # WALL-CLOCK. ``exit_trigger_at`` is monotonic -- its value is
            # seconds since boot (~3e5), while every other timestamp in this
            # row is a UNIX stamp (~1.79e9). Emitting the monotonic value in a
            # timestamp column made ``exit_trigger_timestamp -
            # entry_quote_timestamp`` a meaningless ~-1.79e9. Durations keep
            # using the monotonic stamp; only the reported timestamp is wall.
            _q(trade.latency.exit_trigger_wall
               if trade.latency.exit_trigger_wall is not None
               else trade.latency.exit_trigger_at, 6),
            _q(trade.exit_observed_mark),
            _q(sell.effective_price_usd),
            # USD form: see the note in _signal_row.
            _q(exit_fill.effective_price_usd if exit_fill else None),
            _q(sell.quote_timestamp, 6),
            _q(sell.quote_age_ms, 3),
            sell.provider, sell.venue or "", sell.pool_or_curve or "",
            "|".join(sell.route),
            _q(p.observed_mark_mfe, 6),
            _q(p.observed_mark_mae, 6),
            _q(trade.executable_mfe_pct if trade.executable_mfe_available else None, 6),
            _q(trade.executable_mae_pct if trade.executable_mae_available else None, 6),
            "true" if trade.executable_mfe_available else "false",
            "true" if trade.executable_mae_available else "false",
            _q(pnl.hold_seconds if pnl else None, 3),
            _q(pnl.gross_pnl_quote if pnl else None, 12),
            _q(pnl.trading_fees_quote if pnl else None, 12),
            _q((pnl.network_costs_quote + pnl.other_costs_quote) if pnl else None, 12),
            _q(pnl.net_pnl_quote if pnl else None, 12),
            _q(pnl.realized_pnl_usd if pnl else None, 8),
            _q(pnl.realized_pnl_sol if pnl else None, 12),
            _q(pnl.return_pct if pnl else None, 6),
            _q(pnl.bankroll_after if pnl else None, 8),
            _q(pnl.drawdown_pct if pnl else None, 6),
            trade.exit_trigger,
            _q(lat.get("telegram_to_receive_ms_cross_system"), 3),
            _q(lat.get("decision_latency_ms"), 3),
            _q(lat.get("buy_quote_latency_ms"), 3),
            _q(lat.get("signal_to_simulated_entry_ms"), 3),
            _q(lat.get("sell_quote_latency_ms"), 3),
            _q(lifecycle, 3),
            trade.failure.value,
        ]

    # -- distributions ------------------------------------------------------

    @staticmethod
    def _gap_distribution(
        signals: Sequence[SignalRecord],
    ) -> dict[str, Any]:
        """
        The EXECUTION-NATIVE gap distribution: executable vs this market's
        own on-chain price.

        This is the measure of what entering right now costs in this pool --
        fee plus size impact -- and it is deliberately NOT a mark comparison.
        The observed mark lags the curve by far more than execution costs,
        so the two numbers must never be printed under one heading. The mark
        comparison is :meth:`_premium_distribution`.
        """
        gaps = [
            r.mark_to_quote_gap_pct for r in signals
            if r.mark_to_quote_gap_pct is not None
        ]
        if not gaps:
            return {
                "count": 0,
                "note": (
                    "no signal produced both an observed mark and an executable "
                    "quote, so no distribution exists yet; nothing is imputed"
                ),
            }
        ordered = sorted(gaps)
        mid = len(ordered) // 2
        median = (
            ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0
        )
        return {
            "count": len(gaps),
            "min_pct": ordered[0],
            "median_pct": median,
            "mean_pct": sum(gaps) / len(gaps),
            "p90_pct": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
            "max_pct": ordered[-1],
            "basis": (
                "(executable_average_price / execution_native_reference_price "
                "- 1) * 100, where the reference is THIS MARKET's own "
                "on-chain marginal price from the same slot. Positive means "
                "entering now costs more than the book's current price, "
                "i.e. the trading fee plus the size impact of this order. "
                "It is NOT a comparison against the observed mark."
            ),
            "threshold_configured": False,
            "note": (
                "no execution-cost threshold is configured for the first "
                "forward-capture run; the distribution is the deliverable"
            ),
        }

    @staticmethod
    def _premium_distribution(
        signals: Sequence[SignalRecord],
    ) -> dict[str, Any]:
        """
        The MARK PREMIUM distribution: executable vs the external mark.

        This is the number the old ``Gap:`` line was believed to be showing.
        It measures the aggregator's snapshot of the market against what this
        market would actually charge, so it mixes execution cost with mark
        lag -- which is why it was replaced as a DECISION input, and why it is
        still recorded as a diagnostic on the same record.
        """
        gaps = [
            r.legacy_observed_mark_gap_pct for r in signals
            if r.legacy_observed_mark_gap_pct is not None
        ]
        if not gaps:
            return {
                "count": 0,
                "note": (
                    "no signal carried a mark-based premium; nothing is imputed"
                ),
            }
        ordered = sorted(gaps)
        mid = len(ordered) // 2
        median = (
            ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0
        )
        return {
            "count": len(gaps),
            "min_pct": ordered[0],
            "median_pct": median,
            "mean_pct": sum(gaps) / len(gaps),
            "p90_pct": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
            "max_pct": ordered[-1],
            "basis": (
                "(executable_average_price / observed_mark - 1) * 100. "
                "Positive means the executable quote is WORSE than the mark, "
                "so a mark-based backtest would have UNDERSTATED the real "
                "cost of entry. This mixes execution cost with aggregator "
                "lag and is a diagnostic, never a decision input."
            ),
            "threshold_configured": False,
            "note": (
                "research-only distribution: the production entry guard "
                "compares against the execution-native reference, not this"
            ),
        }

    @staticmethod
    def _mark_to_executable_distribution(
        signals: Sequence[SignalRecord],
    ) -> dict[str, Any]:
        """
        The RESEARCH-ONLY mark-to-executable gap: executable buy USD price vs the
        external DexScreener observed mark. This number mixes execution cost with
        aggregator lag and is a DIAGNOSTIC, never a decision input. It is
        retained on every record so the old and new references stay comparable,
        but it is labelled as such everywhere it appears.
        """
        gaps = [
            r.entry_mark_to_executable_gap_pct for r in signals
            if r.entry_mark_to_executable_gap_pct is not None
        ]
        if not gaps:
            return {
                "count": 0,
                "note": (
                    "no signal recorded an entry mark-to-executable gap; "
                    "nothing is imputed"
                ),
            }
        ordered = sorted(gaps)
        mid = len(ordered) // 2
        median = (
            ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0
        )
        return {
            "count": len(gaps),
            "min_pct": ordered[0],
            "median_pct": median,
            "mean_pct": sum(gaps) / len(gaps),
            "p90_pct": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
            "max_pct": ordered[-1],
            "basis": (
                "(executable_buy_price_usd / observed_mark_usd - 1) * 100. "
                "This mixes execution cost with aggregator lag and is a "
                "DIAGNOSTIC ONLY, never a decision input. The production guard "
                "uses entry_native_to_executable_gap_pct instead."
            ),
        }

    @staticmethod
    def _failure_distribution(
        signals: Sequence[SignalRecord], trades: Sequence[TradeRecord],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in list(signals) + list(trades):
            key = record.failure.value
            if key == "NONE":
                continue
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    # -- markdown -----------------------------------------------------------

    def write_markdown(
        self,
        *,
        signals: Sequence[SignalRecord],
        trades: Sequence[TradeRecord],
        status: Mapping[str, Any],
    ) -> Path:
        gaps = self._gap_distribution(signals)
        lines: list[str] = []
        add = lines.append
        add("# Execution-aware PAPER forward capture")
        add("")
        add("**PAPER ONLY - NO TRANSACTION EXECUTION**")
        add("")
        add(f"`{NO_EXECUTION_NOTE}`")
        add("")
        add("```")
        add("OBSERVED_MARK  !=  EXECUTABLE_QUOTE  !=  SIMULATED_FILL  !=  ACTUAL_FILL")
        add("```")
        add("")
        add(f"- schema: `{SCHEMA_VERSION}`")
        add(f"- L25 definition hash: `{BASELINE_HASH[:16]}…`")
        add(f"- actual fill available: **{ACTUAL_FILL_AVAILABLE}**")
        add(f"- simulated fills: **{SIMULATED_FILL_KIND}**")
        add("")

        add("## Arms")
        add("")
        add("| arm | production profile | max hold | extra entry rule |")
        add("|---|---|---|---|")
        for spec in self.arms:
            add(
                f"| `{spec.name.value}` | `{spec.production_profile.value}` | "
                f"{spec.max_hold_s if spec.max_hold_s else 'none'} | "
                f"{spec.extra_entry_rule or 'none (pure production L25)'} |"
            )
        add("")

        add("## Bankrolls (moved only by fill-based settlement)")
        add("")
        add("| arm | starting | now | trades |")
        add("|---|---|---|---|")
        for name, acc in (status.get("accounts") or {}).items():
            add(
                f"| `{name}` | ${float(acc['starting_bankroll_usd']):.4f} | "
                f"${float(acc['bankroll_usd']):.4f} | {acc['realized_trades']} |"
            )
        add("")

        add("## Executable quote vs curve spot (execution-native gap)")
        add("")
        if gaps.get("count"):
            add(f"- observations: **{gaps['count']}**")
            add(f"- median gap: **{gaps['median_pct']:+.4f}%**")
            add(f"- mean gap: **{gaps['mean_pct']:+.4f}%**")
            add(f"- range: {gaps['min_pct']:+.4f}% .. {gaps['max_pct']:+.4f}%")
            add("")
            add(gaps["basis"])
            add("")
            add(gaps["note"])
        else:
            add(gaps.get("note", "no distribution yet"))
        add("")

        premium = self._premium_distribution(signals)
        add("## Observed mark vs executable quote (mark premium)")
        add("")
        if premium.get("count"):
            add(f"- observations: **{premium['count']}**")
            add(f"- median premium: **{premium['median_pct']:+.4f}%**")
            add(f"- mean premium: **{premium['mean_pct']:+.4f}%**")
            add(
                f"- range: {premium['min_pct']:+.4f}% .. "
                f"{premium['max_pct']:+.4f}%"
            )
            add("")
            add(premium["basis"])
            add("")
            add(premium["note"])
        else:
            add(premium.get("note", "no distribution yet"))
        add("")

        add("## Entry execution-gap observability (per-signal)")
        add("")
        add("Every PAPER entry records, additively and without touching history:")
        add("")
        add("- observed mark (DexScreener) and its timestamp")
        add("- observed mark age at decision (ms, or blank when the source has no per-observation timestamp)")
        add("- execution-native reference price (USD) and calculation method")
        add("- execution-native reference availability")
        add("- executable BUY price (USD) and its quote timestamp")
        add("- executable quote age at decision (ms)")
        add("- mark-to-executable gap (USD): DIAGNOSTIC ONLY, mixes aggregator lag with execution cost")
        add("- native-to-executable gap: the production number (fee + size impact)")
        add("- entry venue id and pool address")
        add("")
        mark_gap = self._mark_to_executable_distribution(signals)
        if mark_gap.get("count"):
            add(f"- mark-to-executable gap observations: **{mark_gap['count']}**")
            add(f"- median: **{mark_gap['median_pct']:+.4f}%**")
            add(f"- mean: **{mark_gap['mean_pct']:+.4f}%**")
            add(f"- range: {mark_gap['min_pct']:+.4f}% .. {mark_gap['max_pct']:+.4f}%")
            add("")
            add(mark_gap["basis"])
        else:
            add(mark_gap.get("note", "no distribution yet"))
        add("")

        add("## Coverage")
        add("")
        add(f"- signals recorded (per arm): **{len(signals)}**")
        add(f"- simulated trades closed: **{len(trades)}**")
        add("")
        failures = self._failure_distribution(signals, trades)
        if failures:
            add("| failure reason | count |")
            add("|---|---|")
            for key, count in failures.items():
                add(f"| `{key}` | {count} |")
        else:
            add("no failures recorded")
        add("")

        add("## Accounting basis")
        add("")
        add("- entry fill: a fresh executable **BUY** quote, in SOL")
        add("- exit fill: a fresh executable **SELL** quote, for the exact tokens held")
        add("- realized: `exit_proceeds - entry_cost - fees - costs`, in the quote asset")
        add("- **never** `exit_mark / entry_mark - 1`")
        add("- MFE/MAE are labelled `OBSERVED_MARK_MFE` / `OBSERVED_MARK_MAE` and are")
        add("  mark observations, not executable returns")
        add("- no slippage is added on top of a quote; the quote already prices this size")
        add("")

        add("## Safety")
        add("")
        audit = status.get("safety") or {}
        add(f"- capability to broadcast: **{'absent (structural)' if audit.get('ok') else 'PRESENT'}**")
        add(f"- forbidden attrs found: **{len(audit.get('forbidden_attrs_found') or [])}**")
        add(f"- forbidden modules found: **{len(audit.get('forbidden_modules_found') or [])}**")
        add("")
        add(f"`{NO_EXECUTION_NOTE}`")
        add("")

        path = self._ensure() / "execution_aware_paper.md"
        atomic_write_text(path, "\n".join(lines) + "\n")
        return path


__all__ = ["DatasetWriter", "NO_EXECUTION_NOTE", "REPORT_DIR"]
