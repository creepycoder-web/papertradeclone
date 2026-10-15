"""
The pipeline: one Telegram signal in, one verdict out.

    Telegram signal
      -> extract token mint          (AMBIGUOUS stops here)
      -> DexScreener: enumerate candidate pairs
      -> select ONE pair, record why
      -> verify pair identity
      -> read live pool / curve state
      -> calculate a BUY quote for the EXACT test size
      -> compare executable price against the observed mark
      -> PASS / REJECT / INFORMATIVE
      -> NO TRANSACTION IS EVER SUBMITTED

EVERY STAGE CAN STOP THE PIPELINE
=================================

Each stage returns a named failure and the pipeline stops. The design rule is
that a missing observation is ``None`` plus a reason, never a zero: a zero
price would flow into the gap calculation and produce a confident, meaningless
percentage. So ``REJECT`` always arrives with the reason attached.

FRESHNESS IS CHECKED BEFORE COMPARISON
=====================================

A stale mark compared against a fresh quote looks exactly like a healthy
comparison. The staleness policy is applied first and any stale input yields
REJECT before the gap is even computed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Sequence

from .dexscreener import (
    DexScreenerProbe,
    PairSelection,
    observed_mark_from,
    select_pair_from,
)
from .identity import compare_mark_to_executable, verify_identity
from .jupiter import JupiterQuoteClient, JupiterQuoteResult, to_executable_quote
from .models import (
    ACTUAL_FILL_AVAILABLE,
    SOL_DECIMALS,
    WSOL_MINT,
    CheckState,
    Decision,
    DexPair,
    ExecutableQuote,
    FailureMode,
    Freshness,
    IdentityChecks,
    MarketKind,
    ObservedMark,
    PoolState,
    SameMarketVerdict,
    StalenessPolicy,
    TokenCandidate,
)
from .numeric import (
    DecimalError,
    sol_amount_for_usd,
    to_decimal,
    to_float,
)
from .pump import CurveUnavailable, quote_bonding_curve_buy, quote_pumpswap_buy
from .signals import extract_token
from .trace import LatencyBreakdown, StageTrace, build_breakdown


#: Input assets whose price is already in USD, so an executable quote taken
#: against one needs no SOL conversion before being compared to the mark.
_USD_DENOMINATED: frozenset[str] = frozenset({
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",   # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",   # USDT
})


class SameMarketTestError(RuntimeError):
    """The harness could not run at all (configuration or transport setup)."""


@dataclass
class TestConfig:
    """
    Everything the operator configures, in one place.

    ``max_mark_gap_pct`` defaults to ``None``, meaning "no threshold". That
    default is the point: the harness will report a gap without claiming a
    verdict, rather than inventing a tolerance that would make results look
    good.
    """

    order_usd: float = 0.23
    sol_reference_usd: float | None = None
    quote_asset_mint: str = WSOL_MINT
    max_mark_gap_pct: float | None = None
    staleness: StalenessPolicy = field(default_factory=StalenessPolicy)
    slippage_bps: int = 1000
    prefer_pump: bool = True
    #: Candidate bonding-curve addresses to probe when the caller knows them.
    curve_candidates: Sequence[str] = ()

    #: Tolerance for CLOSE / DIVERGENT classification of a pairwise gap.
    #: ``None`` means "no tolerance", which is the default and the point: the
    #: harness reports the raw percentage and claims no verdict rather than
    #: inventing a band that would make results look tidy.
    divergence_pct: float | None = None

    #: Run the read-only direct Pump / PumpSwap quote stage. Requires an RPC.
    direct_quotes: bool = True

    #: Tells pytest this is a configuration dataclass, not a test class. It
    #: collects anything named ``Test*`` and would otherwise warn on the name.
    __test__ = False

    def lamports_for_usd(self) -> int:
        """
        Convert the USD test size into lamports via an explicitly logged SOL
        reference price.

        Requires ``sol_reference_usd``. The harness deliberately does not fetch
        a SOL price itself: an unlogged reference would make the order size
        unreproducible, and an unreproducible order size makes the entire
        comparison unverifiable.
        """
        if self.sol_reference_usd is None:
            raise SameMarketTestError(
                "a SOL reference price is required to size the test order in "
                "lamports. Pass --sol-usd <price> so the notional is logged and "
                "reproducible; the harness will not fetch one silently"
            )
        sol = sol_amount_for_usd(self.order_usd, self.sol_reference_usd)
        lamports = int(sol.scaleb(SOL_DECIMALS))
        if lamports <= 0:
            raise SameMarketTestError(
                f"a ${self.order_usd:g} order at ${self.sol_reference_usd:g}/SOL "
                "rounds to zero lamports, which no venue will price"
            )
        return lamports


@dataclass
class SignalReport:
    """Everything observed about one signal. The unit of the JSON report."""

    channel: str
    message_id: int | None
    telegram_timestamp: float | None
    local_receive_timestamp: float | None
    receive_latency_ms: float | None
    raw_text: str
    token: TokenCandidate
    pairs: tuple[DexPair, ...]
    selection: PairSelection
    observed_mark: ObservedMark
    identity: IdentityChecks
    market_kind: MarketKind
    pool_state: PoolState | None
    quote: ExecutableQuote
    jupiter: JupiterQuoteResult | None
    verdict: SameMarketVerdict
    failures: tuple[FailureMode, ...]
    latency: LatencyBreakdown
    mark_freshness: str
    state_freshness: str
    quote_freshness: str
    test_order_usd: float
    test_order_lamports: int | None
    sol_reference_usd: float | None
    started_at: float
    finished_at: float
    execution_note: str = "NOT EXECUTED - NO TRANSACTION BUILT, SIGNED OR BROADCAST"
    #: The four-way measurement. ``None`` when no RPC was configured, which is
    #: how the existing single-quote tests keep their original behaviour.
    direct: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "same_market_quote/v1",
            "actual_fill_available": ACTUAL_FILL_AVAILABLE,
            "signal": {
                "channel": self.channel,
                "message_id": self.message_id,
                "telegram_timestamp": self.telegram_timestamp,
                "local_receive_timestamp": self.local_receive_timestamp,
                "receive_latency_ms": self.receive_latency_ms,
            },
            "test_order": {
                "usd": self.test_order_usd,
                "lamports": self.test_order_lamports,
                "sol_reference_usd": self.sol_reference_usd,
                "hypothetical": True,
                "executed": False,
            },
            "token": self.token.as_dict(),
            "dexscreener": {
                "candidates": [p.as_dict() for p in self.pairs],
                "selection": self.selection.as_dict(),
                "dexscreener_pair_selection_reason": self.selection.reason,
            },
            "observed_mark": {
                **self.observed_mark.as_dict(),
                "freshness": self.mark_freshness,
            },
            "identity_checks": self.identity.as_dict(),
            "execution_market": {
                "kind": self.market_kind.value,
                "pool_state": self.pool_state.as_dict() if self.pool_state else None,
                "state_freshness": self.state_freshness,
                "jupiter_route": (
                    [h.as_dict() for h in self.jupiter.route_plan] if self.jupiter
                    else []
                ),
            },
            "executable_quote": {
                **self.quote.as_dict(),
                "freshness": self.quote_freshness,
            },
            "comparison": self.verdict.as_dict(),
            "failures": [f.value for f in self.failures],
            "latency": self.latency.as_dict(),
            # The four-way measurement, when it ran. These keys are ADDITIVE:
            # everything above is unchanged, so existing consumers and tests
            # keep working while the new schema is available.
            "direct_measurement": self.direct.as_dict() if self.direct else None,
            "price_layers": {
                "OBSERVED_MARK": (self.observed_mark.value_usd
                                  if self.observed_mark.value_usd else None),
                "DIRECT_EXECUTABLE_QUOTE": (
                    {
                        "pump_direct": (
                            self.direct.pump_direct.as_dict()
                            if self.direct and self.direct.pump_direct else None
                        ),
                        "pumpswap_direct": (
                            self.direct.pumpswap_direct.as_dict()
                            if self.direct and self.direct.pumpswap_direct else None
                        ),
                    } if self.direct else None
                ),
                "JUPITER_QUOTE": (
                    self.quote.as_dict() if self.quote.failure is None else None
                ),
                "ACTUAL_FILL": None,
                "ACTUAL_FILL_AVAILABLE": ACTUAL_FILL_AVAILABLE,
                "note": (
                    "OBSERVED_MARK is an aggregator observation, never an "
                    "execution price. DIRECT_EXECUTABLE_QUOTE is priced from "
                    "on-chain state for this exact order size. JUPITER_QUOTE is "
                    "a route. ACTUAL_FILL does not exist in this harness: no "
                    "transaction is built, signed or broadcast."
                ),
            },
            "actual_fill": None,
            "execution_note": self.execution_note,
            "timing": {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            },
        }


class SameMarketPipeline:
    """
    Orchestrates the measurement for one signal.

    Providers are injected so the whole pipeline can be driven by fixtures in
    tests with no network at all.
    """

    def __init__(
        self,
        *,
        config: TestConfig,
        dexscreener: DexScreenerProbe,
        rpc: Any = None,
        jupiter: JupiterQuoteClient | None = None,
        token_decimals_for: Callable[[str], int] | None = None,
    ) -> None:
        self.config = config
        self.dexscreener = dexscreener
        self.rpc = rpc
        self.jupiter = jupiter
        self._token_decimals = token_decimals_for or (lambda _mint: 6)

    # -- pipeline ----------------------------------------------------------

    async def run(
        self,
        *,
        raw_text: str,
        channel: str,
        message_id: int | None = None,
        telegram_timestamp: float | None = None,
    ) -> SignalReport:
        """
        Measure one signal end to end.

        Never raises for an expected failure: every failure becomes a named
        :class:`FailureMode` on the returned report. It raises only if the
        harness itself is misconfigured, which is a different class of problem
        and should be loud.
        """
        trace = StageTrace()
        started = time.time()
        trace.mark("receive")

        failures: list[FailureMode] = []
        token = extract_token(raw_text or "")

        trace.mark("parse_start")
        # Extraction is synchronous and pure, so the parse window is closed
        # immediately: the marks still exist so the stage ladder is complete
        # and a future async extractor has somewhere to record.
        trace.mark("parse_end")

        if token.ambiguous:
            failures.append(FailureMode.TOKEN_EXTRACTION_AMBIGUOUS)
        elif not token.mint:
            failures.append(FailureMode.TOKEN_NOT_FOUND)

        selection = PairSelection(
            pair=None,
            rule="not_attempted",
            reason="token extraction did not yield a single mint, so no "
                   "DexScreener lookup was attempted",
            candidates=(),
            considered=0,
            ambiguous=False,
        )
        mark = ObservedMark(
            value_usd=None,
            pair_address=None,
            dex_id=None,
            source="dexscreener",
            timestamp=time.time(),
            latency_ms=None,
            failure=FailureMode.TOKEN_NOT_FOUND,
        )
        identity = IdentityChecks(
            token_mint_match=CheckState.UNKNOWN,
            quote_mint_match=CheckState.UNKNOWN,
            chain_match=CheckState.UNKNOWN,
            venue_match=CheckState.UNKNOWN,
            pool_match=CheckState.UNKNOWN,
            execution_can_identify_market=None,
            notes=("no executable quote was obtained",),
        )
        quote = _empty_quote(FailureMode.TOKEN_NOT_FOUND)
        jupiter_result: JupiterQuoteResult | None = None
        pool_state: PoolState | None = None
        market_kind = MarketKind.UNKNOWN

        mint = token.mint

        # -- DexScreener ----------------------------------------------------
        if mint and not failures:
            trace.mark("dex_request_start")
            try:
                selection = self.dexscreener.select_pair(
                    mint, venue_hint=token.venue_hint
                )
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                selection = PairSelection(
                    pair=None,
                    rule="provider_error",
                    reason=f"dexscreener lookup failed: {exc}",
                    candidates=(),
                    considered=0,
                    ambiguous=False,
                    failure=FailureMode.PAIR_NOT_FOUND,
                )
            trace.mark("dex_response")

            if selection.failure is not None:
                failures.append(selection.failure)
            mark = observed_mark_from(
                selection,
                timestamp=trace.wall_at("dex_response") or time.time(),
                latency_ms=trace.duration_ms("dex_request_start", "dex_response"),
            )
            if mark.value_usd is None:
                failures.append(FailureMode.PAIR_NOT_FOUND)

            pair = selection.pair
            if pair is not None:
                market_kind = await self._classify_market(pair, trace)
                quote, pool_state, jupiter_result, quote_failures = (
                    await self._quote_exact_size(
                        mint=mint,
                        pair=pair,
                        market_kind=market_kind,
                        trace=trace,
                    )
                )
                failures.extend(quote_failures)
                identity = verify_identity(
                    expected_mint=mint,
                    pair=pair,
                    quote=quote,
                    expected_quote_mint=self.config.quote_asset_mint,
                    jupiter_result=jupiter_result,
                )
                if identity.pool_match is CheckState.FAIL:
                    failures.append(FailureMode.POOL_MISMATCH)
                elif identity.pool_match is CheckState.UNKNOWN:
                    failures.append(FailureMode.MARKET_IDENTITY_UNKNOWN)

        # -- freshness ------------------------------------------------------
        now = time.time()
        mark_stamp = self.config.staleness.stamp(
            "observed_mark",
            mark.timestamp if mark.value_usd is not None else None,
            now,
            self.config.staleness.max_mark_age_ms,
        )
        quote_stamp = self.config.staleness.stamp(
            "executable_quote",
            quote.quote_timestamp if quote.failure is None else None,
            now,
            self.config.staleness.max_quote_age_ms,
        )
        state_ts = pool_state.timestamp if pool_state else None
        state_stamp = self.config.staleness.stamp(
            "pool_state",
            state_ts,
            now,
            self.config.staleness.max_state_age_ms,
        )

        def fresh(stamp: Any) -> bool:
            # An UNKNOWN freshness window is not "fresh". It is unknown, and
            # unknown must not pass as verified.
            return stamp.freshness is Freshness.FRESH

        # Surface staleness in the report's own FAILURE MODES list, not only in
        # the verdict's prose. A stale mark is a named outcome of the
        # measurement and belongs beside every other named outcome.
        if mark_stamp.freshness is Freshness.STALE:
            failures.append(FailureMode.MARK_STALE)
        if quote_stamp.freshness is Freshness.STALE:
            failures.append(FailureMode.QUOTE_STALE)
        if state_stamp.freshness is Freshness.STALE:
            failures.append(FailureMode.STATE_STALE)

        # Convert to USD BEFORE the verdict. DexScreener's mark is USD per
        # token; a quote taken with a SOL-denominated input is SOL per token.
        # Comparing the two directly yields a gap of a few hundred percent that
        # reads like an extreme market rather than a unit error, and it is
        # silent because both numbers are finite and positive.
        quote = self._express_quote_in_usd(quote)

        verdict = compare_mark_to_executable(
            observed_mark=mark.value_usd,
            executable_price=quote.effective_price_usd,
            threshold_pct=self.config.max_mark_gap_pct,
            identity=identity,
            failures=failures,
            mark_freshness=mark_stamp.freshness,
            quote_freshness=quote_stamp.freshness,
        )

        trace.mark("decision")
        finished = time.time()
        latency = build_breakdown(trace, telegram_message_ts=telegram_timestamp)

        try:
            lamports = self.config.lamports_for_usd()
        except SameMarketTestError:
            lamports = None

        # The four-way stage runs LAST and is additive. It needs an RPC, and
        # when there is none it simply does not run - which is what keeps every
        # existing single-quote test behaving exactly as before.
        direct = None
        if self.rpc is not None and self.config.direct_quotes and mint:
            from .direct import run_direct_measurement

            try:
                direct = await run_direct_measurement(
                    mint=mint,
                    telegram_message_at=telegram_timestamp,
                    pair=selection.pair,
                    mark=mark,
                    jupiter_result=jupiter_result,
                    order_usd=self.config.order_usd,
                    order_lamports=lamports,
                    sol_reference_usd=self.config.sol_reference_usd,
                    rpc=self.rpc,
                    jupiter=self.jupiter,
                    staleness=self.config.staleness,
                    divergence_pct=self.config.divergence_pct,
                )
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                failures.append(FailureMode.STATE_UNAVAILABLE)
                trace.mark("direct_failed")
                direct = _direct_failed(str(exc))

        return SignalReport(
            channel=channel,
            message_id=message_id,
            telegram_timestamp=telegram_timestamp,
            local_receive_timestamp=trace.wall_at("receive"),
            receive_latency_ms=latency.telegram_to_receive_ms,
            raw_text=raw_text,
            token=token,
            pairs=tuple(selection.candidates),
            selection=selection,
            observed_mark=mark,
            identity=identity,
            market_kind=market_kind,
            pool_state=pool_state,
            quote=quote,
            jupiter=jupiter_result,
            verdict=verdict,
            failures=tuple(dict.fromkeys(failures)),
            latency=latency,
            mark_freshness=mark_stamp.freshness.value,
            state_freshness=state_stamp.freshness.value,
            quote_freshness=quote_stamp.freshness.value,
            test_order_usd=self.config.order_usd,
            test_order_lamports=lamports,
            sol_reference_usd=self.config.sol_reference_usd,
            started_at=started,
            finished_at=finished,
            direct=direct,
        )

    # -- unit normalisation -------------------------------------------------

    def _express_quote_in_usd(self, quote: ExecutableQuote) -> ExecutableQuote:
        """
        Give the quote a USD price comparable with the mark.

        A BUY quoted with a SOL-denominated input produces SOL per token.
        DexScreener's ``priceUsd`` is USD per token. The comparison has to be
        made in one unit, and USD is the unit the mark is in.

        The conversion uses the SAME logged ``--sol-usd`` reference that sized
        the order, so one number governs the whole measurement and it appears
        in the report next to the result. If no reference is available the USD
        price stays ``None`` and the verdict rejects, rather than falling back
        to the SOL figure and reporting a nonsense gap.

        A quote already denominated in a stable (USDC/USDT) is already USD and
        is passed through untouched.
        """
        if quote.failure is not None or quote.effective_price is None:
            return quote
        if quote.input_mint in _USD_DENOMINATED:
            # Already USD per token.
            return replace(
                quote,
                effective_price_usd=quote.effective_price,
                usd_reference_price=None,
            )

        reference = self.config.sol_reference_usd
        if not reference or reference <= 0:
            return quote

        usd = float(quote.effective_price) * float(reference)
        detail = quote.detail or ""
        detail = (
            f"{detail} | effective price converted from SOL to USD at the "
            f"logged reference ${reference:g}/SOL"
        ).strip(" |")
        return replace(
            quote,
            effective_price_usd=usd,
            usd_reference_price=float(reference),
            detail=detail,
        )

    # -- stages ------------------------------------------------------------

    async def _classify_market(self, pair: DexPair, trace: StageTrace) -> MarketKind:
        """
        Decide what KIND of market this is, from chain state where possible.

        Falls back to the venue label when the chain cannot be read, and says
        so via :attr:`MarketKind.UNKNOWN` rather than asserting a venue it did
        not confirm.
        """
        from .dexscreener import market_kind_for

        if not self.config.prefer_pump or self.rpc is None:
            return market_kind_for(pair.dex_id)

        label_kind = market_kind_for(pair.dex_id)
        if label_kind is not MarketKind.UNKNOWN:
            return label_kind

        trace.mark("rpc_request_start")
        try:
            account = await self.rpc.get_account_info(pair.pair_address)
        except Exception:  # noqa: BLE001 - classification is best-effort
            account = None
        trace.mark("rpc_response")
        trace.mark("pool_state")

        from .pump import PUMP_FUN_PROGRAM, PUMP_SWAP_PROGRAM

        if account is not None and not account.is_empty:
            if account.owner == PUMP_FUN_PROGRAM:
                return MarketKind.PUMP_BONDING_CURVE
            if account.owner == PUMP_SWAP_PROGRAM:
                return MarketKind.PUMPSWAP
        return label_kind

    async def _quote_exact_size(
        self,
        *,
        mint: str,
        pair: DexPair,
        market_kind: MarketKind,
        trace: StageTrace,
    ) -> tuple[ExecutableQuote, PoolState | None, JupiterQuoteResult | None,
               list[FailureMode]]:
        """
        Produce the executable quote for the EXACT configured test size.

        Tries the market-specific path first (bonding curve, PumpSwap) and
        falls back to Jupiter. Every path is read-only and every path records
        which market it priced against, because "same market" is the question
        and an unnamed venue cannot answer it.
        """
        failures: list[FailureMode] = []
        token_decimals = self._token_decimals(mint)
        # Why a market-specific path declined to quote, if it did. Carried into
        # the Jupiter fallback so the report can say WHY the venue route was
        # used instead of the pool we had already identified.
        curve_detail: str | None = None
        pool_detail: str | None = None
        pool_state: PoolState | None = None

        try:
            lamports = self.config.lamports_for_usd()
        except SameMarketTestError as exc:
            return (_empty_quote(FailureMode.QUOTE_UNAVAILABLE, detail=str(exc)),
                    None, None, [FailureMode.QUOTE_UNAVAILABLE])

        # -- bonding curve ---------------------------------------------------
        if market_kind is MarketKind.PUMP_BONDING_CURVE and self.rpc is not None:
            trace.mark("rpc_request_start")
            state: Any = None
            try:
                from .pump import find_bonding_curve

                state, curve_failure = await find_bonding_curve(
                    self.rpc, mint, candidates=tuple(self.config.curve_candidates)
                )
            except Exception as exc:  # noqa: BLE001 - named, never swallowed
                state, curve_failure = None, _classify_provider_error(exc)
            trace.mark("rpc_response")

            if state is not None:
                pool_state = PoolState(
                    kind=MarketKind.PUMP_BONDING_CURVE,
                    address=state.curve_address,
                    program_id=state.program_id,
                    identity_verified=state.identity_verified,
                    timestamp=state.timestamp,
                    fetch_latency_ms=None,
                    detail=state.as_dict(),
                )
                trace.mark("pool_state")
                trace.mark("quote_start")
                try:
                    quote = quote_bonding_curve_buy(
                        state,
                        sol_amount_lamports=lamports,
                        token_decimals=token_decimals,
                        quote_latency_ms=None,
                    )
                    trace.mark("quote_end")
                    return quote, pool_state, None, failures
                except CurveUnavailable as exc:
                    trace.mark("quote_end")
                    failures.append(FailureMode.CURVE_UNAVAILABLE)
                    # Do NOT return. Fall through to Jupiter: this curve path
                    # could not produce a quote, and the venue may still price
                    # the same token. Returning here would make the executable
                    # quote unreachable for every Pump.fun token whose curve
                    # cannot be decoded - which is the majority of this
                    # channel's traffic.
                    curve_detail = str(exc)

        # -- pumpswap ---------------------------------------------------------
        if market_kind is MarketKind.PUMPSWAP and self.rpc is not None:
            trace.mark("rpc_request_start")
            try:
                from .pump import decode_pumpswap_pool

                account = await self.rpc.get_account_info(pair.pair_address)
            except Exception as exc:  # noqa: BLE001 - named, never swallowed
                account = None
                failures.append(_classify_provider_error(exc))
            trace.mark("rpc_response")

            if account is None or account.is_empty:
                # Do not add STATE_UNAVAILABLE on top of a rate limit: the node
                # being busy is a different diagnosis from the pool being gone.
                if FailureMode.RPC_RATE_LIMIT not in failures:
                    failures.append(FailureMode.STATE_UNAVAILABLE)
            else:
                try:
                    pool_state = decode_pumpswap_pool(
                        account.raw(),
                        pool_address=pair.pair_address,
                        program_id=account.owner,
                        expected_base_mint=mint,
                        expected_quote_mint=self.config.quote_asset_mint,
                    )
                except CurveUnavailable as exc:
                    failures.append(FailureMode.POOL_MISMATCH)
                    trace.mark("pool_state")
                    trace.mark("quote_start")
                    trace.mark("quote_end")
                    # Fall through: a pool that does not match is not the
                    # execution market, but Jupiter can still price the token
                    # and reveal WHICH market it actually uses.
                    pool_detail = str(exc)
                else:
                    trace.mark("pool_state")
                    trace.mark("quote_start")
                    try:
                        quote = quote_pumpswap_buy(
                            base_mint=mint,
                            quote_mint=self.config.quote_asset_mint,
                            pool_address=pair.pair_address,
                            # Effective reserves, not raw vault balances. When
                            # the protocol's own accounting cannot be read, this
                            # raises rather than substituting a vault balance.
                            base_reserve=int(
                                (pool_state.detail or {}).get("base_reserve", 0) or 0
                            ),
                            quote_reserve=int(
                                (pool_state.detail or {}).get("quote_reserve", 0) or 0
                            ),
                            sol_amount_lamports=lamports,
                            base_decimals=token_decimals,
                            state_timestamp=pool_state.timestamp,
                        )
                    except CurveUnavailable as exc:
                        trace.mark("quote_end")
                        failures.append(FailureMode.QUOTE_UNAVAILABLE)
                        # Fall through to Jupiter rather than giving up: the
                        # venue route is the executable quote of last resort
                        # and the whole experiment needs one.
                        pool_detail = str(exc)
                    else:
                        trace.mark("quote_end")
                        return quote, pool_state, None, failures

        # -- jupiter ----------------------------------------------------------
        if self.jupiter is None:
            failures.append(FailureMode.QUOTE_UNAVAILABLE)
            return (_empty_quote(FailureMode.QUOTE_UNAVAILABLE,
                                 detail="no quote provider configured"),
                    None, None, failures)

        trace.mark("jupiter_request_start")
        try:
            result = await self.jupiter.quote_buy(
                token_mint=mint,
                quote_asset_mint=self.config.quote_asset_mint,
                in_amount=lamports,
                output_decimals=token_decimals,
                slippage_bps=self.config.slippage_bps,
            )
        except Exception as exc:  # noqa: BLE001 - named, not swallowed
            trace.mark("jupiter_response")
            trace.mark("quote_start")
            trace.mark("quote_end")
            failure = _classify_provider_error(exc)
            failures.append(failure)
            detail = str(exc)
            if curve_detail:
                detail = f"bonding curve: {curve_detail}; jupiter: {detail}"
            elif pool_detail:
                detail = f"pumpswap: {pool_detail}; jupiter: {detail}"
            return (_empty_quote(failure, detail=detail), pool_state, None, failures)
        trace.mark("jupiter_response")
        trace.mark("quote_start")
        quote = to_executable_quote(
            result,
            token_mint=mint,
            quote_asset_mint=self.config.quote_asset_mint,
            output_decimals=token_decimals,
        )
        trace.mark("quote_end")
        if quote.failure is not None:
            failures.append(quote.failure)

        # Keep the chain state that WAS read, even though the quote came from
        # the venue. It is the only evidence about the pool the mark describes,
        # and discarding it would leave the report claiming no state was seen.
        if curve_detail or pool_detail:
            quote = replace(quote, detail=(
                f"{quote.detail or ''} | the market-specific path could not be "
                f"quoted ({curve_detail or pool_detail}); this executable quote "
                f"came from Jupiter instead"
            ).strip(" |"))

        return quote, pool_state, result, failures


def _classify_provider_error(exc: Exception) -> FailureMode:
    """
    Name a provider failure.

    ``RPC_RATE_LIMIT`` is kept distinct from ``PROVIDER_RATE_LIMIT`` because
    they have different causes and different remedies: the first means the
    shared public node is overloaded (back off, lower ``--rpc-max-rpm``), the
    second means a data provider is throttling us. Collapsing them into one
    label would hide which knob to turn.
    """
    from .jupiter import ProviderRateLimited as JupiterRateLimit
    from .jupiter import QuoteDirectionError
    from .jupiter import QuoteUnavailable as JupiterUnavailable
    from .rpc import RpcRateLimited

    if isinstance(exc, QuoteDirectionError):
        return FailureMode.QUOTE_DIRECTION_ERROR
    if isinstance(exc, RpcRateLimited):
        return FailureMode.RPC_RATE_LIMIT
    if isinstance(exc, JupiterRateLimit):
        return FailureMode.PROVIDER_RATE_LIMIT
    if isinstance(exc, JupiterUnavailable):
        return FailureMode.QUOTE_UNAVAILABLE
    return FailureMode.QUOTE_UNAVAILABLE


def _direct_failed(detail: str) -> Any:
    """
    A four-way stage that could not run at all.

    Present-but-empty rather than absent, so "the direct stage broke" is
    distinguishable from "the direct stage was not configured".
    """
    return _DirectUnavailable(detail)


class _DirectUnavailable:
    """Placeholder carrying only the reason the stage did not produce output."""

    def __init__(self, detail: str) -> None:
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": False,
            "missing_reason": self.detail,
            "note": (
                "the direct Pump/PumpSwap stage did not complete; no direct "
                "quote is reported and nothing was substituted"
            ),
        }


def _empty_quote(
    failure: FailureMode,
    *,
    detail: str | None = None,
) -> ExecutableQuote:
    """A quote that was never obtained. No amounts, no price, no venue."""
    from .pump import failed_quote

    return failed_quote(failure, detail=detail)


__all__ = [
    "SameMarketPipeline",
    "SameMarketTestError",
    "SignalReport",
    "TestConfig",
]