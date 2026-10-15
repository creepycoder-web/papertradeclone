"""
End-to-end pipeline and CLI wiring, driven entirely by fixtures.

No network anywhere in this file: the DexScreener client is given an
``http_get`` stub, Jupiter an ``http_get`` stub, and the Solana RPC a fake
account. That is deliberate - the harness must be provable offline, and a test
that reaches mainnet would make the suite's own results unverifiable.

The cases here cover the whole pipeline on the paths that matter: a proven
same-market pass, a proven pool mismatch, an unprovable identity, a missing
quote, and the ambiguity stop.
"""


from __future__ import annotations

import asyncio
import json

import pytest

from autopaper.same_market.dexscreener import DexScreenerProbe
from autopaper.same_market.jupiter import JupiterQuoteClient
from autopaper.same_market.models import WSOL_MINT
from autopaper.same_market.pipeline import (
    SameMarketPipeline,
    SameMarketTestError,
    TestConfig,
)

pytestmark = pytest.mark.same_market

WSOL = WSOL_MINT
MINT = "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump"
MARK_POOL = "2wT8Yq49kHgDzXuPxZSaeLaH1qbmGXtEyPy64bL7aD3c"
OTHER_POOL = "9rPYCUjkgH3rMCbB9fB7f4nUqNQ6EqFNzVvCGrXvpS"
MARK = 0.00001783
CHANNEL = "MooDengPresidentCallers"

SIGNAL = (
    f"⚡️[SOL]  - earn by post ｜ $earn\n\n"
    f"🎯 CA: {MINT}\n\n"
    f"┏ 🧢 Market Cap:   $22.6K ｜🔝$24.9K\n"
    f"┣ ⌛️ Pool Age:  10m [+45%]\n"
    f"┣ 💧 Liquid:  $19K [76.7 SOL]\n"
    f"┗ 🚀 Volume: $59.5k\n\n"
    f"┏ 🦅 DEX:  Paid❌ Ads❌ 0⚡\n"
)


def _dex_pair(address=MARK_POOL, *, dex="pumpswap", liq=19_000.0, price=MARK):
    return {
        "chainId": "solana", "dexId": dex, "pairAddress": address,
        "baseToken": {"address": MINT, "symbol": "earn"},
        "quoteToken": {"address": WSOL, "symbol": "SOL"},
        "priceUsd": price, "priceNative": 0.000000089,
        "liquidity": {"usd": liq}, "volume": {"h24": 59_500.0},
        "fdv": 22_600.0, "pairCreatedAt": 1_788_000_000_000,
    }


#: Tokens out of a 0.00115 SOL BUY at ~$200/SOL.
#:
#: 0.00115 SOL / 12,921.348 tokens = 8.9003e-8 SOL/token, which at $200/SOL is
#: $1.78007e-5 - deliberately close to the observed mark of 1.783e-5 so the
#: identity-proven case exercises a SMALL gap. An output near 64,000 would
#: price the token roughly 100x lower, which is not the scenario under test.
TOKENS_OUT = "12921348000"


def _jup_body(amm_key=MARK_POOL, label="PumpSwap", out=TOKENS_OUT):
    return {
        "inputMint": WSOL, "outputMint": MINT,
        "inAmount": "1150000", "outAmount": out,
        "otherAmountThreshold": "12800000000",
        "priceImpactPct": "0.42",
        "routePlan": [{"swapInfo": {
            "label": label, "ammKey": amm_key,
            "inputMint": WSOL, "outputMint": MINT,
            "inAmount": "1150000", "outAmount": out,
            "feeAmount": "0", "percent": 100,
        }}],
    }


def _pipeline(*, dexs=None, jup=None, order_usd=0.23, sol_usd=200.0,
              threshold=5.0, staleness=True):
    config = TestConfig(
        order_usd=order_usd,
        sol_reference_usd=sol_usd,
        max_mark_gap_pct=threshold,
    )
    if staleness:
        from autopaper.same_market.models import StalenessPolicy
        config.staleness = StalenessPolicy(
            max_mark_age_ms=15_000.0, max_state_age_ms=15_000.0,
            max_quote_age_ms=15_000.0,
        )
    pairs = dexs if dexs is not None else [_dex_pair()]
    return SameMarketPipeline(
        config=config,
        dexscreener=DexScreenerProbe(
            http_get=lambda url, t: {"pairs": pairs}, cache_ttl_s=0.0),
        rpc=None,
        jupiter=JupiterQuoteClient(
            http_get=lambda p: (jup or _jup_body()), cache_ttl_s=0.0),
    )


def _run(pipeline, text=SIGNAL, **kw):
    return asyncio.run(pipeline.run(
        raw_text=text, channel=CHANNEL, **kw))


# ---------------------------------------------------------------------------
# the proven-same-market path
# ---------------------------------------------------------------------------

def test_a_proven_same_market_with_a_small_gap_passes():
    report = _run(_pipeline())
    assert report.identity.pool_match.value == "PASS"
    assert report.identity.market_match.value == "PASS"
    assert report.verdict.decision.value == "PASS"
    # 0.00115 SOL / 12,921.348 tokens = 8.9003e-08 SOL/token, which at the
    # logged $200/SOL is $1.78007e-05 against a $1.783e-05 mark.
    assert report.quote.effective_price_usd == pytest.approx(
        0.00115 / 12921.348 * 200.0, rel=1e-9
    )
    assert report.verdict.abs_gap_pct == pytest.approx(0.164, abs=0.01)
    assert report.verdict.gap_pct < 0, "the quote is marginally better"
    # The SOL figure is retained alongside, never replaced.
    assert report.quote.effective_price == pytest.approx(8.9003e-08, rel=1e-4)
    assert report.quote.usd_reference_price == 200.0
    assert report.failures == ()


def test_the_report_states_the_order_is_hypothetical_and_unexecuted():
    report = _run(_pipeline())
    payload = report.as_dict()
    assert payload["test_order"]["hypothetical"] is True
    assert payload["test_order"]["executed"] is False
    assert payload["actual_fill"] is None
    assert payload["actual_fill_available"] is False
    assert "NO TRANSACTION" in payload["execution_note"]


def test_the_selection_reason_is_always_in_the_report():
    report = _run(_pipeline())
    payload = report.as_dict()
    reason = payload["dexscreener"]["dexscreener_pair_selection_reason"]
    assert reason.strip()
    assert payload["dexscreener"]["selection"]["rule"].strip()


def test_the_full_latency_ladder_is_present():
    report = _run(_pipeline(), telegram_timestamp=1_700_000_000.0)
    payload = report.as_dict()["latency"]
    for key in ("telegram_to_receive_ms", "receive_to_mark_ms",
                "mark_to_state_ms", "state_to_quote_ms",
                "quote_to_decision_ms", "total_signal_to_decision_ms"):
        assert key in payload
    assert payload["telegram_to_receive_approximate"] is True
    assert payload["total_signal_to_decision_ms"] is not None


def test_the_quote_direction_is_buy_throughout_the_pipeline():
    report = _run(_pipeline())
    assert report.quote.input_mint == WSOL
    assert report.quote.output_mint == MINT
    assert report.quote.input_amount_raw == "1150000"   # 0.00115 SOL @ $200


def test_the_three_values_are_serialised_separately():
    payload = _run(_pipeline()).as_dict()
    assert "observed_mark" in payload
    assert "executable_quote" in payload
    assert payload["actual_fill"] is None
    assert payload["observed_mark"]["value_usd"] == pytest.approx(MARK)
    assert payload["executable_quote"]["effective_price"] is not None


# ---------------------------------------------------------------------------
# identity failures
# ---------------------------------------------------------------------------

def test_a_route_on_a_different_pool_is_a_proven_mismatch():
    report = _run(_pipeline(jup=_jup_body(amm_key=OTHER_POOL)))
    assert report.identity.pool_match.value == "FAIL"
    assert report.verdict.decision.value == "REJECT"
    assert "POOL_MISMATCH" in [f.value for f in report.failures]


def test_a_route_that_discloses_no_pool_cannot_pass():
    body = _jup_body()
    body["routePlan"][0]["swapInfo"].pop("ammKey")
    report = _run(_pipeline(jup=body))
    assert report.identity.pool_match.value == "UNKNOWN"
    assert report.verdict.decision.value == "REJECT"
    assert "MARKET_IDENTITY_UNKNOWN" in [f.value for f in report.failures]


def test_an_ambiguous_market_is_rejected_even_at_zero_gap():
    """Two pools, identical prices. The prices agreeing proves nothing."""
    report = _run(_pipeline(dexs=[
        _dex_pair(MARK_POOL, dex="pumpswap", liq=50_000.0),
        _dex_pair(OTHER_POOL, dex="raydium", liq=30_000.0),
    ]))
    assert report.selection.ambiguous is True
    assert report.verdict.decision.value == "REJECT"
    assert "PAIR_AMBIGUOUS" in [f.value for f in report.failures]


def test_a_far_quote_is_rejected_against_the_threshold():
    # 1/6th of the tokens out is a ~5x worse price.
    report = _run(_pipeline(jup=_jup_body(out="2153558000")))
    assert report.identity.pool_match.value == "PASS"
    assert report.verdict.decision.value == "REJECT"
    assert report.verdict.abs_gap_pct > 5.0


def test_no_threshold_means_informative_not_pass():
    report = _run(_pipeline(threshold=None))
    assert report.verdict.decision.value == "INFORMATIVE"
    assert report.verdict.threshold_pct is None
    assert report.verdict.gap_pct is not None, "the gap is still measured"


# ---------------------------------------------------------------------------
# early exits
# ---------------------------------------------------------------------------

def test_an_ambiguous_mint_stops_before_any_lookup():
    calls: list[str] = []
    pipeline = SameMarketPipeline(
        config=TestConfig(sol_reference_usd=200.0, max_mark_gap_pct=5.0),
        dexscreener=DexScreenerProbe(
            http_get=lambda url, t: calls.append(url) or {"pairs": []}),
        rpc=None, jupiter=None,
    )
    report = _run(pipeline, text=f"a {MINT} b {MARK_POOL}")
    assert report.token.ambiguous is True
    assert "TOKEN_EXTRACTION_AMBIGUOUS" in [f.value for f in report.failures]
    assert calls == [], "no lookup should be attempted for an ambiguous signal"
    assert report.verdict.decision.value == "REJECT"


def test_a_message_with_no_mint_stops_before_any_lookup():
    report = _run(_pipeline(), text="🔥 narrative post, no contract address")
    assert report.token.mint is None
    assert "TOKEN_NOT_FOUND" in [f.value for f in report.failures]
    assert report.verdict.decision.value == "REJECT"


def test_a_missing_pair_is_reported_as_pair_not_found():
    pipeline = SameMarketPipeline(
        config=TestConfig(sol_reference_usd=200.0, max_mark_gap_pct=5.0),
        dexscreener=DexScreenerProbe(
            http_get=lambda url, t: {"pairs": []}, cache_ttl_s=0.0),
        rpc=None, jupiter=None,
    )
    report = _run(pipeline)
    assert "PAIR_NOT_FOUND" in [f.value for f in report.failures]
    assert report.observed_mark.value_usd is None
    assert report.verdict.decision.value == "REJECT"


def test_an_unavailable_quote_is_named_not_zeroed():
    from autopaper.same_market.jupiter import QuoteUnavailable

    def _fail(_params):
        raise QuoteUnavailable("TOO_SMALL")

    pipeline = SameMarketPipeline(
        config=TestConfig(sol_reference_usd=200.0, max_mark_gap_pct=5.0),
        dexscreener=DexScreenerProbe(
            http_get=lambda url, t: {"pairs": [_dex_pair()]}, cache_ttl_s=0.0),
        rpc=None, jupiter=JupiterQuoteClient(http_get=_fail),
    )
    report = _run(pipeline)
    assert "QUOTE_UNAVAILABLE" in [f.value for f in report.failures]
    assert report.quote.effective_price is None
    assert report.quote.output_amount_raw == ""
    assert report.verdict.decision.value == "REJECT"


def test_no_sol_reference_means_no_quote_rather_than_a_guessed_size():
    pipeline = SameMarketPipeline(
        config=TestConfig(order_usd=0.23, sol_reference_usd=None,
                          max_mark_gap_pct=5.0),
        dexscreener=DexScreenerProbe(
            http_get=lambda url, t: {"pairs": [_dex_pair()]}, cache_ttl_s=0.0),
        rpc=None, jupiter=None,
    )
    with pytest.raises(SameMarketTestError):
        pipeline.config.lamports_for_usd()

    report = _run(pipeline)
    assert report.quote.failure is not None
    assert "QUOTE_UNAVAILABLE" in [f.value for f in report.failures]


def test_a_stale_mark_window_rejects():
    from autopaper.same_market.models import StalenessPolicy

    config = TestConfig(sol_reference_usd=200.0, max_mark_gap_pct=5.0)
    config.staleness = StalenessPolicy(max_mark_age_ms=0.000001)
    pipeline = SameMarketPipeline(
        config=config,
        dexscreener=DexScreenerProbe(
            http_get=lambda url, t: {"pairs": [_dex_pair()]}, cache_ttl_s=0.0),
        rpc=None,
        jupiter=JupiterQuoteClient(http_get=lambda p: _jup_body()),
    )
    report = _run(pipeline)
    assert report.verdict.decision.value == "REJECT"
    assert "MARK_STALE" in [f.value for f in report.failures]


# ---------------------------------------------------------------------------
# order sizing
# ---------------------------------------------------------------------------

def test_the_configured_usd_size_becomes_the_expected_lamports():
    assert _pipeline(order_usd=0.23, sol_usd=200.0).config.lamports_for_usd() \
        == 1_150_000
    assert _pipeline(order_usd=1.00, sol_usd=200.0).config.lamports_for_usd() \
        == 5_000_000


def test_a_usd_size_that_rounds_to_zero_lamports_is_refused():
    config = TestConfig(order_usd=1e-12, sol_reference_usd=200.0)
    with pytest.raises(SameMarketTestError):
        config.lamports_for_usd()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_the_cli_exposes_the_command_with_the_documented_flags():
    from autopaper.cli import build_parser

    args = build_parser().parse_args([
        "same-market-test",
        "--channel", CHANNEL,
        "--order-usd", "0.23",
        "--max-mark-gap-pct", "3",
    ])
    assert args.func.__name__ == "cmd_same_market_test"
    assert args.channel == CHANNEL
    assert args.order_usd == 0.23
    assert args.max_mark_gap_pct == 3.0


def test_the_threshold_defaults_to_unconfigured_not_to_a_number():
    """
    The default must be None so an unconfigured run reports INFORMATIVE. A
    default threshold would manufacture verdicts the operator never chose.
    """
    from autopaper.cli import build_parser

    args = build_parser().parse_args(["same-market-test"])
    assert args.max_mark_gap_pct is None
    assert args.max_mark_age_ms is None
    assert args.max_state_age_ms is None
    assert args.max_quote_age_ms is None


def test_the_default_channel_and_order_size_match_the_brief():
    from autopaper.cli import build_parser

    args = build_parser().parse_args(["same-market-test"])
    assert args.channel == "MooDengPresidentCallers"
    assert args.order_usd == 0.23


def test_the_cli_does_not_arm_real_and_has_no_execute_flag():
    from autopaper.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["same-market-test"])
    for forbidden in ("execute", "confirm", "live", "real", "dry_run"):
        assert not hasattr(args, forbidden), (
            f"same-market-test exposes a {forbidden} flag; it must have no way "
            "to be turned into a trading action"
        )


def test_the_offline_message_path_writes_json_and_markdown(tmp_path, capsys):
    from autopaper.cli import build_parser
    from autopaper.same_market.cli import run_same_market_test

    args = build_parser().parse_args([
        "--root", str(tmp_path),          # a global option, so it leads
        "same-market-test", "--message", SIGNAL,
        "--sol-usd", "200", "--no-rpc", "--no-jupiter", "--json",
    ])
    code = run_same_market_test(None, args)
    assert code == 0

    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):out.rindex("}") + 1])
    assert payload["actual_fill_available"] is False

    assert (tmp_path / "reports" / "tests" / "same_market_quote").exists()


def test_a_report_file_contains_no_credentials(tmp_path, monkeypatch):
    """
    The JSON report is built from report dataclasses, so it has no path to a
    session, a key or a token. Asserted rather than assumed.
    """
    from autopaper.same_market.cli import write_report_files

    report = _run(_pipeline())
    written, latest = write_report_files([report], root=tmp_path)
    body = written[0].read_text(encoding="utf-8")
    latest_body = latest.read_text(encoding="utf-8")

    for secret_marker in ("TELEGRAM_SESSION", "api_hash", "bot_token",
                          "private_key", "seed", "StringSession", "secret"):
        assert secret_marker.lower() not in body.lower(), secret_marker
        assert secret_marker.lower() not in latest_body.lower(), secret_marker
    assert "1B7" not in body      # a StringSession always begins like this


def test_the_markdown_report_always_shows_the_three_values(tmp_path):
    from autopaper.same_market.cli import write_report_files

    written, latest = write_report_files([_run(_pipeline())], root=tmp_path)
    text = latest.read_text(encoding="utf-8")
    assert "OBSERVED_MARK" in text
    assert "EXECUTABLE_QUOTE" in text
    assert "ACTUAL_FILL" in text
    assert "GOAT REGRESSION" in text
    assert "NO TRANSACTION WAS BROADCAST" in text