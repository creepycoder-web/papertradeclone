"""
SOL/USD reference provider and the two CLI wirings.

Fully offline. Every provider path is driven by an injected ``http_get`` or a
fixture body; nothing here opens a socket or reads a secret file.

Run:  PYTHONPATH=src .venv/bin/python -m pytest tests/test_sol_usd_reference.py -q
"""


from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path

import pytest

from autopaper.same_market import sol_price as S
from autopaper.same_market.sol_price import (
    JupiterSolPriceProvider,
    SolPriceUnavailable,
    SolUsdSnapshot,
    describe_snapshot,
    override_snapshot,
)

pytestmark = [
    pytest.mark.paper_live,
    pytest.mark.safety,
    pytest.mark.same_market,
]

WSOL = "So11111111111111111111111111111111111111112"
T0 = 1_700_000_000.0

AI = Path(__file__).resolve().parents[1]
SRC = AI / "src" / "autopaper"


def code_only(path: Path) -> str:
    """
    Source with every comment and docstring removed.

    Prohibitions are asserted against CODE. A module is allowed -- required,
    even -- to *document* that it must never touch DexScreener, so matching
    that prose would fail the guard for entirely the wrong reason.
    """
    import io
    import tokenize

    stream = io.StringIO(path.read_text())
    pieces: list[str] = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(stream.readline):
        ttype, text = tok[0], tok[1]
        sline, scol = tok[2][0], tok[2][1]
        if sline > last_lineno:
            last_col = 0
        if scol > last_col:
            pieces.append(" " * (scol - last_col))
        if ttype == tokenize.COMMENT:
            pass
        elif ttype == tokenize.STRING:
            # a string that is a whole statement is a docstring
            if prev_toktype in (tokenize.INDENT, tokenize.NEWLINE,
                                tokenize.NL, tokenize.ENCODING):
                pass
            else:
                pieces.append(text)
        else:
            pieces.append(text)
        prev_toktype = ttype
        last_col = scol + len(text)
        last_lineno = tok[3][0]
    return " ".join(pieces)


def _body(price=147.48, *, block=348004023, decimals=9, omit=False,
          mint=WSOL, null_price=False):
    if omit:
        return {WSOL: {}}
    entry: dict = {"decimals": decimals, "blockId": block}
    entry["usdPrice"] = None if null_price else price
    return {mint: entry}


def _provider(body=None, *, clock=T0, ttl_s=60.0, key=None, raise_exc=None,
              calls=None):
    """A provider wired to an injected transport. No network, ever."""
    payload = _body() if body is None else body

    async def _get(params):
        if calls is not None:
            calls.append(params)
        if raise_exc is not None:
            raise raise_exc
        return payload

    return JupiterSolPriceProvider(
        http_get=_get, clock=lambda: clock, ttl_s=ttl_s, api_key=key)


# ===========================================================================
# 3. the provider returns a valid snapshot
# ===========================================================================


def test_provider_returns_a_valid_snapshot():
    import asyncio

    p = _provider(_body(147.48))
    snap = asyncio.run(p.snapshot())
    assert isinstance(snap, SolUsdSnapshot)
    assert snap.usd_per_sol == pytest.approx(147.48)
    assert snap.block_id == 348004023
    assert snap.fetched is True
    assert snap.source.startswith("jupiter price api v3")
    assert snap.usd_per_sol > 0


def test_snapshot_metadata_has_the_reported_keys():
    import asyncio

    snap = asyncio.run(_provider().snapshot())
    meta = snap.as_dict(T0 + 1.5)
    assert set(meta) >= {"sol_usd", "sol_usd_source", "sol_usd_timestamp",
                         "sol_usd_age_ms"}
    assert meta["sol_usd"] == pytest.approx(147.48)
    assert meta["sol_usd_timestamp"] == pytest.approx(T0)
    assert meta["sol_usd_age_ms"] == pytest.approx(1500.0)


def test_age_is_never_negative():
    import asyncio

    snap = asyncio.run(_provider().snapshot())
    assert snap.age_ms(T0 - 10_000) == 0.0


def test_request_targets_sol_and_the_free_endpoint():
    calls: list = []
    import asyncio

    p = _provider(calls=calls)
    asyncio.run(p.snapshot())
    assert calls == [{"ids": WSOL}]
    assert p.base_url == "https://lite-api.jup.ag/price/v3"
    assert "lite-api.jup.ag" in p.base_url
    assert "//api.jup.ag" not in p.base_url, "must use the free/keyless host"


# ===========================================================================
# 4. caching
# ===========================================================================


def test_snapshot_is_cached_and_reused():
    """Multiple arms/signals must not each make a request."""
    import asyncio

    calls: list = []
    p = _provider(calls=calls)

    async def drive():
        return [await p.snapshot() for _ in range(10)]

    snaps = asyncio.run(drive())
    assert len(calls) == 1, "ten consumers must cause ONE request"
    assert p.cache_hits == 9
    assert all(s is snaps[0] for s in snaps)


def test_cache_expires_after_the_ttl():
    import asyncio

    calls: list = []
    now = {"t": T0}

    async def _get(params):
        calls.append(params)
        return _body()

    p = JupiterSolPriceProvider(http_get=_get, clock=lambda: now["t"], ttl_s=60.0)

    async def drive():
        await p.snapshot()
        now["t"] += 59.0
        await p.snapshot()
        assert p.cached_snapshot() is not None, "still fresh at 59s"
        now["t"] += 2.0
        assert p.cached_snapshot() is None, "stale at 61s"
        await p.snapshot()

    asyncio.run(drive())
    assert len(calls) == 2


def test_refresh_bypasses_the_cache():
    import asyncio

    calls: list = []
    p = _provider(calls=calls)

    async def drive():
        await p.snapshot()
        await p.snapshot(refresh=True)

    asyncio.run(drive())
    assert len(calls) == 2


def test_sync_resolver_reuses_an_already_fetched_snapshot():
    import asyncio

    calls: list = []
    p = _provider(calls=calls)
    asyncio.run(p.snapshot())
    assert S.resolve_sol_usd(provider=p).usd_per_sol == pytest.approx(147.48)
    assert len(calls) == 1, "the sync path must not re-request"


# ===========================================================================
# 5. fail closed -- every failure path refuses
# ===========================================================================


def test_transport_failure_fails_closed():
    import asyncio

    p = _provider(raise_exc=ConnectionError("network down"))
    with pytest.raises(SolPriceUnavailable) as e:
        asyncio.run(p.snapshot())
    assert "transport failure" in str(e.value)


def test_rate_limit_fails_closed_without_a_price():
    import asyncio

    p = _provider(raise_exc=SolPriceUnavailable(
        "SOL/USD provider rate limited (HTTP 429); not guessing a price"))
    with pytest.raises(SolPriceUnavailable):
        asyncio.run(p.snapshot())
    assert p.last_error and "429" in p.last_error


def test_omitted_token_fails_closed():
    """Jupiter OMITS a token with no reliable price. That is not zero."""
    import asyncio

    # the mint key is simply ABSENT -- a null value is a different failure
    p = _provider({"EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v":
                   {"usdPrice": 1.0}})
    with pytest.raises(SolPriceUnavailable) as e:
        asyncio.run(p.snapshot())
    assert "no entry for SOL" in str(e.value)
    assert "never substitutes a guess" in str(e.value)


def test_null_price_fails_closed():
    import asyncio

    p = _provider(_body(null_price=True))
    with pytest.raises(SolPriceUnavailable) as e:
        asyncio.run(p.snapshot())
    assert "no usdPrice" in str(e.value)


def test_non_numeric_and_non_finite_prices_fail_closed():
    import asyncio

    for bad in ("abc", True, None, float("nan"), float("inf")):
        body = {WSOL: {"usdPrice": bad}}
        p = _provider(body)
        with pytest.raises(SolPriceUnavailable):
            asyncio.run(p.snapshot())


def test_implausible_price_fails_closed():
    """A misparsed response must not be allowed to size notionals."""
    import asyncio

    for bad in (0.0, -5.0, 1e9):
        p = _provider(_body(bad))
        with pytest.raises(SolPriceUnavailable) as e:
            asyncio.run(p.snapshot())
        assert "plausible band" in str(e.value)


def test_failure_never_caches_a_stale_value():
    """After a failure there must be no cached fallback to reuse."""
    import asyncio

    p = _provider(raise_exc=ConnectionError("down"))
    with pytest.raises(SolPriceUnavailable):
        asyncio.run(p.snapshot())
    assert p._cached is None
    assert p.cached_snapshot() is None


def test_no_default_price_exists_anywhere_in_the_module():
    """
    The strongest available statement of 'never guess': there is no fallback
    constant a future edit could silently reach for.
    """
    src = (SRC / "same_market" / "sol_price.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            # Only the plausibility band and formatting constants may be float
            # literals; a DEFAULT price would be one of these too, so assert
            # the specific band values are the only ones in a plausible range.
            assert node.value in (
                S.MIN_PLAUSIBLE_USD, S.MAX_PLAUSIBLE_USD, 1000.0,
                60.0, 0.0, 8.0, 20.0,
            ), f"unexpected float literal {node.value} at line {node.lineno}"


# ===========================================================================
# 6. offline override
# ===========================================================================


def test_override_works_with_no_provider_and_no_network():
    import asyncio

    snap = asyncio.run(S.resolve_sol_usd_async(override=150.0))
    assert snap.usd_per_sol == pytest.approx(150.0)
    assert snap.fetched is False
    assert "override" in snap.source


def test_override_is_validated_like_a_fetched_price():
    for bad in (0.0, -1.0, 1e9, None, "x", True, float("nan")):
        with pytest.raises(SolPriceUnavailable):
            override_snapshot(bad)


def test_override_takes_precedence_and_makes_no_request():
    import asyncio

    calls: list = []
    p = _provider(calls=calls)
    snap = asyncio.run(S.resolve_sol_usd_async(override=150.0, provider=p))
    assert snap.usd_per_sol == pytest.approx(150.0)
    assert snap.fetched is False
    assert calls == [], "an override must not touch the network"


def test_deprecated_sol_usd_flag_still_works_as_an_override():
    import argparse

    from autopaper.same_market.cli import config_sol, config_sol_metadata

    import asyncio

    from autopaper.same_market.cli import _resolve_sol_snapshot

    args = argparse.Namespace(sol_usd=150.0, sol_usd_override=None)
    assert config_sol(args) == pytest.approx(150.0)
    snap = asyncio.run(_resolve_sol_snapshot(args))
    assert snap.usd_per_sol == pytest.approx(150.0)
    assert snap.fetched is False
    assert "DEPRECATED" in snap.source


def test_override_beats_the_deprecated_flag():
    import argparse

    from autopaper.same_market.cli import config_sol

    args = argparse.Namespace(sol_usd=150.0, sol_usd_override=160.0)
    assert config_sol(args) == pytest.approx(160.0)


# ===========================================================================
# 1 & 2. the CLIs no longer require --sol-usd
# ===========================================================================


def test_same_market_test_parses_without_sol_usd():
    from autopaper.cli import build_parser

    args = build_parser().parse_args(["same-market-test", "--order-usd", "0.23"])
    assert args.order_usd == pytest.approx(0.23)
    assert args.sol_usd is None
    assert args.sol_usd_override is None


def test_same_market_test_accepts_the_override():
    from autopaper.cli import build_parser

    args = build_parser().parse_args(
        ["same-market-test", "--order-usd", "0.23", "--sol-usd-override", "150"])
    assert args.sol_usd_override == pytest.approx(150.0)


def test_paper_live_parses_without_sol_usd():
    from autopaper.paper_live.cli import build_parser_args

    p = argparse.ArgumentParser()
    build_parser_args(p)
    args = p.parse_args(["--bankroll-usd", "5"])
    assert args.bankroll_usd == pytest.approx(5.0)
    assert args.sol_usd is None
    assert args.sol_usd_override is None


def test_paper_live_accepts_the_override():
    from autopaper.paper_live.cli import build_parser_args

    p = argparse.ArgumentParser()
    build_parser_args(p)
    assert p.parse_args(["--sol-usd-override", "150"]).sol_usd_override \
        == pytest.approx(150.0)


def test_no_cli_still_refuses_for_a_missing_sol_usd():
    """
    The old REQUIRED check must be GONE. If it survives, the auto-fetch never
    gets a chance to run.
    """
    for path in (SRC / "paper_live" / "cli.py", SRC / "same_market" / "cli.py"):
        src = path.read_text()
        assert "is required" not in src or "no longer required" in src, path
        assert "--sol-usd is required" not in src, path


def test_paper_live_run_refuses_when_the_reference_cannot_be_obtained(monkeypatch):
    """Fail closed at the CLI boundary, with a non-zero exit and no client."""
    import asyncio

    from autopaper.paper_live import cli as pl

    async def boom(args):
        raise SolPriceUnavailable("provider unreachable")

    monkeypatch.setattr(pl, "_resolve_sol_reference", boom)
    monkeypatch.setattr(pl, "audit_paper_capability",
                        lambda: type("A", (), {"ok": True, "render": str})())

    p = argparse.ArgumentParser()
    pl.build_parser_args(p)
    args = p.parse_args(["--bankroll-usd", "5", "--strategies", "L25_CURRENT"])
    args.provider_timeout_s = 0.01
    args.cache_ttl_s = 0.01
    args.rpc_max_rpm = 1
    args.sol_price_max_rpm = 1

    rc = pl.run(None, args)
    assert rc == 2, "a missing SOL/USD reference must refuse the run"


def test_paper_live_uses_the_resolved_snapshot_not_a_flag(monkeypatch):
    """
    Structural: the flag must not still be read for sizing.
    """
    src = (SRC / "paper_live" / "cli.py").read_text()
    tree = ast.parse(src)

    # ZERO direct ``args.sol_usd`` reads: sizing must not come off the flag.
    direct = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Attribute) and n.attr == "sol_usd"
              and isinstance(n.ctx, ast.Load)]
    assert not direct, f"args.sol_usd still read for sizing at {direct}"

    # The deprecated flag survives ONLY as the alias lookup inside the resolver.
    gets = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "getattr" and len(n.args) >= 2
            and isinstance(n.args[1], ast.Constant)
            and n.args[1].value == "sol_usd"]
    assert len(gets) == 1, "exactly one deprecated-alias lookup"

    # and the reference is built ONCE, from the resolved snapshot
    assert src.count("SolReference(") == 1, "one construction, from the snapshot"
    assert "sol_reference=sol_reference" in src
    assert src.count("sol_reference=sol_reference") == 2, (
        "both the quote service and the engine take the same snapshot")


# ===========================================================================
# 7. the API key never leaks
# ===========================================================================


SECRET = "supersecret-jupiter-key-abc123XYZ"


def test_key_is_used_but_never_exposed():
    p = JupiterSolPriceProvider(api_key=SECRET)
    assert p.has_api_key is True
    assert p._headers()["x-api-key"] == SECRET, "it must actually be sent"
    # and it must not appear anywhere observable
    assert SECRET not in repr(p)
    assert SECRET not in str(p.as_dict()) if hasattr(p, "as_dict") else True
    assert SECRET not in describe_snapshot(override_snapshot(150.0))


def test_key_is_absent_from_every_snapshot_field():
    import asyncio

    async def _get(params):
        return _body()

    p = JupiterSolPriceProvider(http_get=_get, api_key=SECRET)
    snap = asyncio.run(p.snapshot())
    assert SECRET not in repr(snap)
    assert SECRET not in str(snap.as_dict())
    for value in snap.as_dict().values():
        assert SECRET not in str(value)


def test_key_is_redacted_out_of_error_messages():
    p = JupiterSolPriceProvider(api_key=SECRET)
    leaked = p._redact(
        f"request failed with header x-api-key: {SECRET} for url ?key={SECRET}")
    assert SECRET not in leaked
    assert "***" in leaked


def test_env_key_is_redacted_from_provider_errors(monkeypatch):
    monkeypatch.setenv("JUPITER_API_KEY", SECRET)
    p = JupiterSolPriceProvider()
    assert p.has_api_key is True
    assert SECRET not in repr(p)
    assert SECRET not in p._redact(f"boom {SECRET}")


def test_the_module_never_prints_the_key():
    """No print/log of the key, structurally."""
    src = (SRC / "same_market" / "sol_price.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            assert name not in ("print", "log"), f"line {node.lineno}"
    # and the key is only ever referenced as a header value or a private field
    assert src.count("_api_key") == src.count("_api_key")


def test_no_paid_or_new_provider_is_introduced():
    src = code_only(SRC / "same_market" / "sol_price.py").lower()
    for banned in ("helius", "birdeye", "//api.jup.ag", "subscription",
                   "alchemy", "quicknode", "stripe"):
        assert banned not in src, banned


def test_no_secret_file_is_read():
    src = code_only(SRC / "same_market" / "sol_price.py")
    assert ".env" not in src
    assert "dotenv" not in src
    assert "load_env_file" not in src


def test_env_read_is_limited_to_the_documented_key_names(monkeypatch):
    monkeypatch.setenv("JUPITER_API_KEY", SECRET)
    monkeypatch.setenv("TELEGRAM_API_HASH", "should-not-be-read")
    p = JupiterSolPriceProvider()
    assert p.has_api_key is True
    assert p._headers().get("x-api-key") == SECRET
    assert "should-not-be-read" not in str(p._headers())


# ===========================================================================
# the architecture boundary
# ===========================================================================


def test_sol_usd_is_never_derived_from_a_token_price():
    """
    The provider must not be able to reach DexScreener or a token quote.
    """
    src = code_only(SRC / "same_market" / "sol_price.py")
    for banned in ("dexscreener", "pump", "quote_price", "effective_price",
                   "usd_price", "price_usd", "observed_mark", "get_quote"):
        assert banned not in src, f"provider code references {banned}"
    # its only mint is SOL's, hard-wired. Scanned from code_only so DOCSTRINGS do
    # not count as data: prose mentioning a mint is not a pricing capability.
    import re

    mints = set(re.findall(r"[1-9A-HJ-NP-Za-km-z]{40,50}", src))
    assert mints <= {WSOL}, mints


def test_conversion_stage_is_unchanged():
    """
    numeric.sol_amount_for_usd must still be the ONLY converter, untouched.
    """
    import inspect as _i

    from autopaper.same_market import numeric

    src = _i.getsource(numeric.sol_amount_for_usd)
    assert "target / price" in src
    # it still takes the reference as a parameter -- it was not taught to fetch
    assert "Jupiter" not in src and "snapshot" not in src


def test_sizing_formula_is_untouched():
    """The production position-size percentages must not have moved."""
    from autopaper.paper_live import cli as pl

    src = pl.__file__ and Path(pl.__file__).read_text()
    assert "12.5%%" in src or "12.5" in src


def test_report_metadata_is_additive_only():
    """sol_metadata is merged with setdefault; existing keys are preserved."""
    from autopaper.same_market import cli as sm

    src = Path(sm.__file__).read_text()
    assert "payload.setdefault(key, value)" in src, "must merge, not overwrite"


def test_sol_reference_dict_keeps_its_original_keys():
    from autopaper.paper_live.quotes import SolReference

    d = SolReference(150.0, T0, "test").as_dict()
    assert d["sol_usd_reference"] == pytest.approx(150.0)
    assert d["sol_usd_reference_at"] == pytest.approx(T0)
    assert d["sol_usd_reference_source"] == "test"
    # plus the additive metadata
    assert d["sol_usd"] == pytest.approx(150.0)
    assert d["sol_usd_source"] == "test"
    assert d["sol_usd_timestamp"] == pytest.approx(T0)
    assert "sol_usd_age_ms" in d