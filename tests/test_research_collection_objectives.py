"""
Focused tests for the research-collection objectives.

  A  independent read-only ingestion, process health, graceful signal handling
  B  SELL quote parameters, route/identity separation, no signing or broadcast,
     quote failure classification
  C  complete checkpoint denominators and horizon keys, no mark-to-executable
     substitution

Run:
    .venv/bin/python -m pytest tests/test_research_collection_objectives.py -q

These tests are hermetic: no network, no Telegram session, no subprocess, no
production state written or read for mutation.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research/strategy_ab_comparison/scripts"))

import research_ingest as RI  # noqa: E402

WSOL = "So11111111111111111111111111111111111111112"
MINT = "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump"
POOL = "2wT8Yq49kHgDzXuPxZSaeLaH1qbmGXtEyPy64bL7aD3c"


# ===========================================================================
# OBJECTIVE A -- independent read-only ingestion
# ===========================================================================
ALERT_CARD = (
    "\U0001F525 NEW TOKEN\n"
    f"CA: {MINT}\n"
    "Name: Research Ingest Test\n"
    "Symbol: RIT\n"
    "Market Cap: $45,000\n"
    "Liquidity: $22,000\n"
    "Volume(5m): $12,300\n"
    "Holders: 88\n"
)


class TestIndependentIngestion:
    def test_alert_store_dedups_on_mint_and_message_id(self, tmp_path):
        p = tmp_path / "alerts.jsonl"
        store = RI.AlertStore(p)
        rec = {"mint": MINT, "message_id": "163836"}
        assert store.append_once(rec) is True
        assert store.append_once(dict(rec)) is False, "a replayed alert must not append"
        assert store.append_once({**rec, "message_id": "999"}) is True
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        assert len(rows) == 2

    def test_alert_store_survives_restart_without_duplicating(self, tmp_path):
        p = tmp_path / "alerts.jsonl"
        first = RI.AlertStore(p)
        first.append_once({"mint": MINT, "message_id": "1"})
        second = RI.AlertStore(p)
        assert second.load() == 1
        assert second.append_once({"mint": MINT, "message_id": "1"}) is False

    def test_a_record_without_a_mint_is_never_written(self, tmp_path):
        store = RI.AlertStore(tmp_path / "a.jsonl")
        assert store.append_once({"mint": None, "message_id": "5"}) is False
        assert not (tmp_path / "a.jsonl").exists()

    def test_ingestion_uses_the_repository_parser(self):
        """The mint must come from the EXISTING parser, not a second one."""
        from autopaper.telegram.parser import parse_alert
        parsed = parse_alert(ALERT_CARD)
        assert parsed.mint == MINT, "fixture must be parseable by the real parser"

    def test_on_message_writes_a_chronological_deduped_record(self, tmp_path):
        out = tmp_path / "alerts.jsonl"
        ing = RI.IndependentIngest(out_path=out, health_path=tmp_path / "health.json")

        asyncio.run(ing.on_message(ALERT_CARD, 163836, 1791630000.0, 1791630001.25))
        asyncio.run(ing.on_message(ALERT_CARD, 163836, 1791630000.0, 1791630001.25))

        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert len(rows) == 1, "the same alert must not be written twice"
        r = rows[0]
        assert r["mint"] == MINT
        # BOTH timestamps are kept: Telegram's causal stamp and local ingest.
        assert r["source_timestamp"] == 1791630000.0
        assert r["received_timestamp"] == 1791630001.25
        assert r["transit_latency_ms"] == pytest.approx(1250.0)
        assert r["kind"] == "RESEARCH_ALERT"
        assert ing.health.parsed == 1
        assert ing.health.duplicates == 1

    def test_a_message_with_no_parseable_mint_is_counted_not_written(self, tmp_path):
        out = tmp_path / "alerts.jsonl"
        ing = RI.IndependentIngest(out_path=out, health_path=tmp_path / "h.json")
        asyncio.run(ing.on_message("good morning, nothing to parse here", 1, None, 1.0))
        assert ing.health.no_mint == 1
        assert not out.exists() or out.read_text().strip() == ""

    def test_a_parser_error_does_not_stop_ingest(self, tmp_path, monkeypatch):
        ing = RI.IndependentIngest(out_path=tmp_path / "a.jsonl",
                                   health_path=tmp_path / "h.json")

        def boom(_text):
            raise RuntimeError("parser exploded")

        monkeypatch.setattr(RI, "parse_alert", boom)
        asyncio.run(ing.on_message(ALERT_CARD, 1, None, 1.0))
        assert any("parser exploded" in e for e in ing.health.errors)

    def test_ingest_defaults_write_only_under_the_research_directory(self):
        """The defaults must be research-owned, never engine-owned."""
        assert RI.DEFAULT_OUT == REPO / "var/research/alerts.jsonl"
        assert RI.DEFAULT_HEALTH == REPO / "var/research/ingest_health.json"
        for p in (RI.DEFAULT_OUT, RI.DEFAULT_HEALTH):
            assert p.is_relative_to(REPO / "var/research")

    def test_ingest_never_opens_an_engine_owned_file_for_writing(self):
        """
        Structural: the only write targets in the module are its two store
        methods, and neither names an engine-owned path. Prose may mention
        those paths to explain what is NOT written; code may not open them.
        """
        import ast
        tree = ast.parse(Path(RI.__file__).read_text())
        # Docstrings may NAME an engine-owned path to explain what is not
        # written; only non-docstring literals are executable intent.
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and \
                        isinstance(body[0].value, ast.Constant) and \
                        isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
        literals = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and id(n) not in docstrings]
        engine_paths = ("candidates.jsonl", "journal_decisions", "ledger.jsonl",
                        "paper_state", "paper_account", ".env.local")
        # a path literal may only appear if it is one of the two research defaults
        for lit in literals:
            for bad in engine_paths:
                assert bad not in lit or lit.startswith("var/research"), (
                    f"a code literal references an engine-owned path: {lit!r}")


class TestProcessHealth:
    def test_snapshot_reports_awaiting_first_event(self, tmp_path):
        h = RI.IngestHealth(tmp_path / "h.json")
        s = h.snapshot(900.0)
        assert s["feed_status"] == "AWAITING_FIRST_EVENT"
        assert s["feed_stalled"] is False

    def test_a_live_feed_is_not_reported_stalled(self, tmp_path):
        h = RI.IngestHealth(tmp_path / "h.json")
        h.note(event=True)
        s = h.snapshot(900.0)
        assert s["feed_status"] == "LIVE"
        assert s["feed_stalled"] is False

    def test_a_silent_feed_IS_reported_stalled(self, tmp_path):
        """The failure that was invisible on 2026-10-10: alive but silent."""
        h = RI.IngestHealth(tmp_path / "h.json")
        h.note(event=True)
        h.last_event_at = h.last_event_at - 3600.0
        s = h.snapshot(900.0)
        assert s["feed_stalled"] is True
        assert s["feed_status"] == "STALLED"
        assert s["silence_s"] > 900.0

    def test_health_declares_itself_read_only(self, tmp_path):
        s = RI.IngestHealth(tmp_path / "h.json").snapshot(900.0)
        assert s["read_only"] is True
        assert s["writes_to_journal"] is False
        assert s["writes_to_candidates"] is False
        assert s["sends_messages"] is False

    def test_health_file_is_written_atomically(self, tmp_path):
        p = tmp_path / "h.json"
        h = RI.IngestHealth(p)
        h.note(event=True)
        h.write(900.0)
        assert p.exists()
        assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"
        assert json.loads(p.read_text())["feed_status"] == "LIVE"


class TestGracefulSignalHandling:
    def test_signal_handlers_are_installed_for_term_and_int(self, tmp_path):
        ing = RI.IndependentIngest(out_path=tmp_path / "a.jsonl",
                                   health_path=tmp_path / "h.json")

        async def go():
            loop = asyncio.get_running_loop()
            names = ing.install_signal_handlers(loop)
            # raise the real signal: the handler must convert it into a clean stop
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(0.05)
            for s in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(s)
            return names

        names = asyncio.run(go())
        assert "SIGTERM" in names and "SIGINT" in names
        assert ing._stop.is_set(), "SIGTERM must set the stop flag, not kill the loop"
        assert any("SIGTERM" in e for e in ing.health.errors)

    def test_request_stop_is_idempotent(self, tmp_path):
        ing = RI.IndependentIngest(out_path=tmp_path / "a.jsonl",
                                   health_path=tmp_path / "h.json")
        ing.request_stop("SIGTERM")
        ing.request_stop("SIGINT")
        assert ing._stop.is_set()
        assert len([e for e in ing.health.errors if e.startswith("signal")]) == 2


# ===========================================================================
# OBJECTIVE B -- SELL quote: parameters, identity separation, safety
# ===========================================================================
SELL_BODY = {
    "inputMint": MINT, "outputMint": WSOL,
    "inAmount": "64523000", "outAmount": "1128000",
    "otherAmountThreshold": "1100000", "priceImpactPct": "0.51",
    "routePlan": [
        {"swapInfo": {"label": "Obsidian", "ammKey": POOL, "inputMint": MINT,
                      "outputMint": WSOL, "inAmount": "30000000",
                      "outAmount": "520000", "feeAmount": "15000", "percent": 46.5}},
        {"swapInfo": {"label": "Flux", "ammKey": "p2", "inputMint": WSOL,
                      "outputMint": WSOL, "inAmount": "520000",
                      "outAmount": "1128000", "feeAmount": "0", "percent": 53.5}},
    ],
}


def _sell_client(body=None, *, error=None):
    from autopaper.same_market.jupiter import JupiterQuoteClient

    def _get(params):
        if error is not None:
            raise error
        return dict(body if body is not None else SELL_BODY)

    return JupiterQuoteClient(http_get=_get, cache_ttl_s=0.0, slippage_bps=1000)


class TestSellQuoteParameters:
    def test_the_exact_raw_token_quantity_is_requested(self):
        from autopaper.same_market.jupiter import JupiterQuoteClient
        seen = {}

        def _get(params):
            seen.update(params)
            return dict(SELL_BODY)

        c = JupiterQuoteClient(http_get=_get, cache_ttl_s=0.0)
        asyncio.run(c.quote_sell(token_mint=MINT, quote_asset_mint=WSOL,
                                 token_amount=64_523_000, token_decimals=6))
        assert seen["inputMint"] == MINT, "a SELL must give up the TOKEN"
        assert seen["outputMint"] == WSOL, "a SELL must receive the QUOTE asset"
        assert seen["amount"] == "64523000", "the exact raw quantity is required"

    def test_output_route_impact_and_latency_are_all_recorded(self):
        c = _sell_client()
        r = asyncio.run(c.quote_sell(token_mint=MINT, quote_asset_mint=WSOL,
                                     token_amount=64_523_000, token_decimals=6))
        d = r.as_dict()
        assert d["out_amount"] == 1128000
        assert d["price_impact_pct"] == pytest.approx(0.51, abs=1e-9)
        assert len(d["route_plan"]) == 2
        assert d["route_plan"][0]["feeAmount"] == 15000, "the hop fee must survive"
        assert d["request_latency_ms"] is not None and d["request_latency_ms"] >= 0

    @pytest.mark.parametrize("amount,decimals", [(0, 6), (-5, 6), (1000, 0)])
    def test_a_non_positive_or_undecodable_amount_is_refused(self, amount, decimals):
        from autopaper.same_market.jupiter import QuoteUnavailable
        c = _sell_client()
        with pytest.raises(QuoteUnavailable):
            asyncio.run(c.quote_sell(token_mint=MINT, quote_asset_mint=WSOL,
                                     token_amount=amount, token_decimals=decimals))


class TestNoSigningOrBroadcast:
    def test_the_quote_client_has_no_transaction_capability(self):
        from autopaper.same_market.jupiter import JupiterQuoteClient
        forbidden = ("build", "build_transaction", "send", "sign", "submit",
                     "broadcast", "execute", "swap_and_send", "send_raw")
        for name in forbidden:
            assert not hasattr(JupiterQuoteClient, name), name
        assert JupiterQuoteClient.read_methods == ("quote",)

    def test_the_sell_path_only_performs_a_read(self):
        """The sell method must not call anything but the shared read helper."""
        import ast
        import inspect
        import textwrap
        from autopaper.same_market.jupiter import JupiterQuoteClient
        src = textwrap.dedent(inspect.getsource(JupiterQuoteClient.quote_sell))
        calls = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute):
                    calls.add(f.attr)
                elif isinstance(f, ast.Name):
                    calls.add(f.id)
        # Raising and comparing are fine; anything that could PRODUCE or
        # TRANSMIT a transaction is not.
        allowed = {"quote_sell", "assert_sell_direction", "int", "str", "dict",
                   "time", "abs", "float", "max", "self", "run", "acquire",
                   "parse", "monotonic", "_fetch", "_cache", "get",
                   "QuoteUnavailable", "QuoteDirectionError"}
        unexpected = calls - allowed
        assert not unexpected, f"quote_sell calls something unexpected: {unexpected}"
        assert "_fetch" in calls, "it must go through the shared read helper"

    def test_the_production_signer_module_is_not_imported_by_the_sell_path(self):
        import inspect
        from autopaper.same_market.jupiter import JupiterQuoteClient
        src = inspect.getsource(JupiterQuoteClient.quote_sell)
        for forbidden in ("txcodec", "real", "keypair", "signer"):
            assert forbidden not in src.lower(), f"quote_sell references {forbidden}"


class TestRouteIdentitySeparation:
    def _point(self, body=None, entry_market=POOL):
        from autopaper.paper_live.quotes import QuoteService
        c = _sell_client(body)
        res = asyncio.run(c.quote_sell(token_mint=MINT, quote_asset_mint=WSOL,
                                       token_amount=64_523_000, token_decimals=6))
        svc = QuoteService.__new__(QuoteService)
        svc.sol_reference = None
        svc.max_quote_age_ms = 30_000
        return QuoteService._point_from_liquidation_route(
            svc, res, mint=MINT, decimals=6, entry_market=entry_market,
            latency_ms=1.0)

    def test_a_multi_hop_route_is_never_identity_PASS(self):
        p = self._point()
        assert p.available is True
        assert p.identity == "UNKNOWN", "a multi-hop route is not a market identity"
        assert p.identity != "PASS"

    def test_a_single_hop_route_is_also_not_identity_PASS(self):
        """One hop is not proof either: the entry's market is verified on chain,
        not inferred from the router's choice."""
        single = dict(SELL_BODY)
        single["routePlan"] = SELL_BODY["routePlan"][:1]
        p = self._point(single)
        assert p.identity == "UNKNOWN"
        assert p.route == ("Obsidian",)

    def test_the_entrys_verified_market_is_preserved_separately(self):
        p = self._point(entry_market=POOL)
        assert p.observed_pair_address == POOL
        assert POOL in (p.identity_evidence or "")
        assert "OFF-MARKET" in (p.identity_evidence or "")

    def test_the_route_is_recorded_as_provenance_not_as_pool_identity(self):
        p = self._point()
        assert p.route == ("Obsidian", "Flux")
        assert p.pool_or_curve is None, (
            "a route must not be presented as the position's pool")

    def test_proceeds_are_quote_per_token_not_the_reciprocal(self):
        p = self._point()
        # out/in = 1.128 SOL for 64.523 tokens, NOT the buy-direction reciprocal
        assert float(p.effective_price) == pytest.approx(
            1_128_000 / 1e9 / (64_523_000 / 1e6))

    def test_a_venue_echoing_a_different_pair_is_refused(self):
        from autopaper.paper_live.quotes import QuoteService
        svc = QuoteService.__new__(QuoteService)
        svc.sol_reference = None
        svc.max_quote_age_ms = 30_000
        bogus = type("R", (), {
            "input_mint": WSOL, "output_mint": MINT,
            "in_amount": 1, "out_amount": 1, "route_plan": (),
            "price_impact_pct": None, "timestamp": 0.0})()
        p = QuoteService._point_from_liquidation_route(
            svc, bogus, mint=MINT, decimals=6, entry_market=POOL, latency_ms=1.0)
        assert p.available is False
        assert p.output_amount_raw is None, "a refused leg must have no amounts"
        assert "different pair" in (p.missing_reason or "")


class TestQuoteFailureClassification:
    @pytest.mark.parametrize("exc_name,expected", [
        ("QuoteUnavailable", "NO_ROUTE"),
        ("ProviderRateLimited", "RATE_LIMITED"),
    ])
    def test_a_venue_refusal_is_named_not_swallowed(self, exc_name, expected):
        import autopaper.same_market.jupiter as J
        exc = getattr(J, exc_name)("the venue declined")
        c = _sell_client(error=exc)
        with pytest.raises(getattr(J, exc_name)):
            asyncio.run(c.quote_sell(token_mint=MINT, quote_asset_mint=WSOL,
                                     token_amount=1_000_000, token_decimals=6))

    def test_a_refused_liquidation_records_a_status_not_a_price(self):
        from autopaper.paper_live.quotes import QuoteService
        svc = QuoteService.__new__(QuoteService)
        svc.sol_reference = None
        svc.max_quote_age_ms = 30_000
        p = QuoteService._point_from_liquidation_route(
            svc, None, mint=MINT, decimals=6, entry_market=POOL, latency_ms=None)
        assert p.available is False
        assert p.output_amount_raw is None
        assert p.effective_price is None, "a refusal must not invent a price"
        assert p.missing_reason

    def test_a_zero_input_route_yields_no_proceeds(self):
        from autopaper.paper_live.quotes import QuoteService
        svc = QuoteService.__new__(QuoteService)
        svc.sol_reference = None
        svc.max_quote_age_ms = 30_000
        zero = type("R", (), {
            "input_mint": MINT, "output_mint": WSOL,
            "in_amount": 0, "out_amount": 5, "route_plan": (),
            "price_impact_pct": None, "timestamp": 0.0})()
        p = QuoteService._point_from_liquidation_route(
            svc, zero, mint=MINT, decimals=6, entry_market=POOL, latency_ms=None)
        assert p.available is False
        assert "zero input" in (p.missing_reason or "")


# ===========================================================================
# OBJECTIVE C -- denominators, horizons, no mark substitution
# ===========================================================================
class TestCheckpointDenominators:
    def test_collectors_coverage_report_counts_missed_checkpoints(self):
        cov = json.loads((REPO / "research/new_strategy_study/collector_data"
                          / "forward_coverage.json").read_text())
        sched = cov["denominator"]["scheduled_observations"]
        written = cov["denominator"]["observations_written"]
        assert written == sched, "every scheduled checkpoint must be written"
        # a missed schedule still occupies its slot in the exit-leg table
        total = sum(h["scheduled"] for h in cov["exit_leg"]["by_horizon"].values())
        assert total == cov["exit_leg"]["checkpoints_scheduled_total"]
        for h, v in cov["exit_leg"]["by_horizon"].items():
            assert v["scheduled"] >= 1, f"horizon {h} is missing from the denominator"

    def test_all_six_required_horizons_are_declared(self):
        cov = json.loads((REPO / "research/new_strategy_study/collector_data"
                          / "forward_coverage.json").read_text())
        assert [float(x) for x in cov["checkpoints_scheduled_s"]] == \
            [10.0, 30.0, 60.0, 120.0, 300.0, 600.0]

    def test_a_failed_entry_still_writes_its_anchor_row(self, tmp_path):
        """Re-run the collector's own invariant on live data."""
        rows = [json.loads(l) for l in (REPO / "research/new_strategy_study"
                                        / "collector_data/observations.jsonl")
                .read_text().splitlines() if l.strip()]
        shadow = [r for r in rows if r.get("observation_kind") == "forward_shadow"]
        for r in shadow:
            assert r.get("status") is not None, "every row carries a status"
        # every checkpoint row belongs to a mint that also has an anchor row
        mints_with_entry = {r["mint"] for r in shadow if r.get("horizon_s") == 0.0}
        for r in shadow:
            assert r["mint"] in mints_with_entry, \
                "a checkpoint exists with no anchor row"


class TestNoMarkToExecutableSubstitution:
    def test_shadow_rows_carry_no_mark_in_the_return(self):
        rows = [json.loads(l) for l in (REPO / "research/new_strategy_study"
                                        / "collector_data/observations.jsonl")
                .read_text().splitlines() if l.strip()]
        checked = 0
        for r in rows:
            if r.get("observation_kind") != "forward_shadow":
                continue
            if r.get("shadow_return_pct") is None:
                continue
            checked += 1
            assert r.get("executable_sell") is not None, \
                "a shadow return must be backed by an executable SELL leg"
            assert r.get("executable_buy") is not None, \
                "a shadow return must be backed by an executable BUY leg"
        assert checked >= 0

    def test_a_mark_only_observation_never_produces_a_shadow_return(self):
        rows = [json.loads(l) for l in (REPO / "research/new_strategy_study"
                                        / "collector_data/observations.jsonl")
                .read_text().splitlines() if l.strip()]
        for r in rows:
            if r.get("observation_kind") == "forward_shadow":
                continue
            assert r.get("shadow_return_pct") is None, \
                "a delayed/mark observation must not carry a shadow return"

    def test_the_liquidation_point_never_carries_a_native_reference(self):
        """An aggregator route has no chain-decoded account to point at."""
        from autopaper.paper_live.quotes import QuoteService
        svc = QuoteService.__new__(QuoteService)
        svc.sol_reference = None
        svc.max_quote_age_ms = 30_000
        c = _sell_client()
        res = asyncio.run(c.quote_sell(token_mint=MINT, quote_asset_mint=WSOL,
                                       token_amount=64_523_000, token_decimals=6))
        p = QuoteService._point_from_liquidation_route(
            svc, res, mint=MINT, decimals=6, entry_market=POOL, latency_ms=1.0)
        assert p.execution_native_reference is None