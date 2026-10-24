"""
Configuration loader for the V3 runtime.

Reads ``.env.local`` as provided. Nothing is invented: a missing Telegram
session, wallet or RPC is reported as absent and degrades the capability rather
than being faked.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------

It cannot arm REAL trading. ``RealConfig.trading_enabled`` is a CAPABILITY
report, not an authorization: it is true only when the operator has armed the
runtime, and arming happens exclusively through a confirmed Telegram action
plus a passing preflight (see :mod:`autopaper.modes`). A config file or an
environment variable that says ``REAL_TRADING_ENABLED=true`` starts a PAPER
process exactly as a file saying ``false`` does.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, ClassVar

from .safety import RealExecutionBlocked

ENV_FILENAME = ".env.local"


# ---------------------------------------------------------------------------
# execution model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionModel:
    """
    Documented paper execution model.

    Defaults reflect a small, fast memecoin account on a public venue:

    * ``entry_fee_bps`` 85  -> 0.85% (typical DEX aggregator taker fee + priority)
    * ``exit_fee_bps``  85  -> 0.85%
    * ``entry_slippage_bps`` 120 / ``exit_slippage_bps`` 120
      Base 60 bps for memecoin books this thin, plus up to 90 bps of size impact
      as a deterministic function of notional vs. liquidity, plus up to 40 bps of
      staleness penalty. See ``execution/fill.py`` for the exact formula.
    * ``decision_latency_ms`` 1200  Bot reaction + analysis latency before the
      order is considered placed. During this window price moves against us,
      which is simulated for real.
    * ``market_impact_model`` "sqrt_notional_over_liquidity"
    """

    entry_fee_bps: float = 85.0
    exit_fee_bps: float = 85.0
    entry_slippage_bps: float = 120.0
    exit_slippage_bps: float = 120.0
    decision_latency_ms: float = 1200.0
    market_impact_model: str = "sqrt_notional_over_liquidity"
    #: HARD max-hold cap for the L25_MAX_HOLD PAPER profile, in seconds.
    #: PAPER research only: it exists so the cap can be swept without editing
    #: code. It has NO effect on L25_CURRENT, which has no cap at all, or on
    #: L25_60S, whose name states its fixed 60s.
    max_hold_seconds: float = 60.0
    # impact coefficients
    impact_base_bps: float = 60.0
    impact_max_bps: float = 90.0
    staleness_penalty_bps_per_sec: float = 20.0
    staleness_penalty_max_bps: float = 40.0
    # fill validity
    max_quote_age_ms: float = 8000.0
    fill_quote_timeout_ms: float = 5000.0
    # liquidity depth used when a pool has no published depth
    # ---------------------------------------------------------------------------
    # DEPRECATED. This value is NO LONGER substituted for a missing liquidity in
    # any fill path (see autopaper.execution.fill.impact_bps). It is retained
    # only so historical fill records and forensic reports that quote the old
    # assumption remain reproducible. Setting it has no effect on a new fill.
    assume_sol_usd: float = 60_000.0
    #: Impact charged when the venue's liquidity could NOT be read at fill time.
    #: ``None`` means "use ``impact_max_bps``", i.e. charge the worst case the
    #: model is willing to represent rather than inventing a pool size.
    impact_unknown_bps: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# risk limits (simulation constraints, not production advice)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskLimits:
    """
    Risk limits for the V3-C runtime.

    ``min_position_notional_usd`` is now ``0.0``. The previous value was $5,
    which is unreachable on a ~$1.55 account: 10% of $1.55 is $0.155, so a $5
    floor would have refused every trade on the micro account and invited
    exactly the "round up to $5" behaviour this migration forbids. The floor
    that matters for micro is the VENUE's, and the venue executor reports it
    with the venue's own reason.

    Position percentage itself is NOT configured here. It is the frozen V3-C
    policy in :mod:`autopaper.sizing`: 10% at or below $50 equity, 12.5% above.
    """

    starting_bankroll_usd: float = 100.0
    #: Hard ceiling on a single position, as a percentage of equity. The V3-C
    #: policy tops out at 12.5%, so this is a backstop, not the policy.
    max_position_pct_equity: float = 12.5
    #: Hard ceiling on total open exposure. Equal to the sizing fraction, so a
    #: second position is impossible by construction.
    max_total_exposure_pct_equity: float = 12.5
    #: The V3-C sizing band. Retained for display; the policy lives in sizing.py.
    default_position_pct_low: float = 10.0
    default_position_pct_high: float = 12.5
    #: No house minimum. See the class docstring.
    min_position_notional_usd: float = 0.0
    #: Exactly one concurrent position.
    max_simultaneous_positions: int = 1
    #: Never re-enter. There is no override in the config.
    reentry_enabled: bool = False
    reentry_cooldown_s: float = 900.0
    #: simulation safety boundary; NOT the primary strategy exit
    disaster_cap_pct: float = -45.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# market data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketConfig:
    primary: str = "dexscreener"
    fallback: str = "geckoterminal"
    poll_interval_s: float = 5.0
    open_position_poll_interval_s: float = 3.0
    max_requests_per_minute: int = 110
    # When an alert names a token we have never seen, take one print, wait this
    # long, then take another. Nobody decides on a single tick, and without this
    # the agent would hold exactly one observation at analysis time and refuse
    # every opportunity as UNCERTAIN.
    alert_settle_s: float = 2.0
    #: WALL-CLOCK budget for ONE provider call, retries and backoff included.
    #: Read this as a deadline, not as an httpx knob: httpx applies a single
    #: number to each request phase independently, and charges the connect
    #: phase once per resolved address, so ``timeout_s=8`` became a 16s request
    #: on a host listing two unroutable IPv6 addresses. See
    #: ``market.ratelimit.run_with_deadline``.
    http_timeout_s: float = 8.0
    #: Connect-phase budget, charged ONCE PER CANDIDATE ADDRESS by
    #: ``socket.create_connection``, so it is deliberately short: the cost of an
    #: unreachable address is this number, not ``http_timeout_s``.
    #:
    #: Measured on this host, 2026-09-30: ``api.dexscreener.com`` and
    #: ``api.geckoterminal.com`` each resolve to two unroutable IPv6 addresses
    #: ahead of two reachable IPv4 addresses. A reachable handshake costs
    #: 13-45ms, so 1.0s is 20-70x the observed need, while the dead-address bill
    #: drops from ``2 x http_timeout_s`` (16.0s) to 2.0s. The remaining bound is
    #: ``http_timeout_s``, enforced as a WALL-CLOCK deadline regardless.
    connect_timeout_s: float = 1.0
    max_retries: int = 2
    cache_ttl_s: float = 2.0
    # an observation older than this cannot be used to mark or fill
    stale_after_ms: float = 15_000.0
    # an observation older than this is unusable for any decision
    unusable_after_ms: float = 45_000.0
    min_pair_created_age_s: float = 60.0

    # -- NATIVE MARKET MONITORING (event-driven) --------------------------
    #
    # These govern the FAST MARKET MONITOR, which answers "what is the venue
    # doing right now" from PUSHED Solana account data with a local price
    # calculation and zero RPC. They deliberately do NOT govern the
    # ``poll_interval_s`` above, which remains the RECOVERY fallback cadence:
    # a shorter poll interval would spend more of the very budget the push
    # channel exists to protect.
    #
    # ``native_monitor_enabled=False`` returns the runtime to the previous
    # poll-and-quote behaviour, so this is reversible in one flag.
    native_monitor_enabled: bool = True
    #: ``wss://`` push endpoint. Empty disables native monitoring and falls back
    #: to the polling path, which is the correct behaviour on an operator who
    #: has no websocket endpoint rather than a silent degradation.
    native_monitor_ws_url: str = ""
    #: A silent stream is not stale immediately: a quiet token produces NO pushes
    #: at all, so this must be generous enough that an untraded market is never
    #: mistaken for a broken stream.
    native_monitor_stale_after_ms: float = 8_000.0
    native_monitor_open_timeout_s: float = 10.0
    #: QUOTE PROXIMITY. Within this many percentage points of an EXISTING exit
    #: trigger, an executable SELL quote may be requested for validation; beyond
    #: it the position is monitored natively and no quote is spent. This gates
    #: QUOTE REQUESTS ONLY and cannot create or suppress an exit -- see
    #: ``autopaper.paper_live.quote_proximity``.
    quote_proximity_enabled: bool = True
    quote_proximity_band_pct: float = 3.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# decision / exit policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionConfig:
    # NOTE: this is the *pipeline* floor, below V3-C's 0.55. It exists to
    # stop obviously bad data reaching the decision at all. V3-C's 0.55 is
    # the binding entry gate and is applied separately in v3c.evaluate().
    min_signal_quality_to_trade: float = 0.45
    max_extension_to_chase: float = 0.60          # 60% above first valid price
    perfect_extension_to_chase: float = 0.30      # allow a little more for runners
    prefer_perfect_extension: bool = True
    # Recalibrated against the configured channel's actual distribution. The
    # feed publishes 13K-40K market caps with 3.5K-25K pools; the original
    # 25K/8K floors rejected essentially every candidate in it, which is a
    # configuration that measures my assumptions rather than the market.
    min_liquidity_usd: float = 3_000.0
    max_market_cap_usd: float = 120_000_000.0
    min_market_cap_usd: float = 8_000.0
    min_data_freshness: float = 0.5
    min_buy_sell_ratio: float = 1.05
    max_data_age_s: float = 30.0
    min_confidence_to_buy: float = 0.50
    # A lower bar for an entry the extension ladder explicitly permitted. Without
    # this, the second and third rungs of the ladder would be unreachable: the
    # very thing that earns an extended entry permission (being past the
    # chaseable band) is also what pushes confidence down, so the normal bar
    # would cancel the ladder out.
    min_confidence_to_buy_extended: float = 0.45

    # --- V3-C: the frozen decision-time gates ----------------------------
    # These exist so an operator can SEE the thresholds, never to change them.
    # __post_init__ refuses any other value, so neither field can become the
    # knob that quietly invalidates the V3-C result.
    v3c_max_market_cap_usd: float = 20_000.0
    v3c_min_signal_quality: float = 0.55

    # --- L25: the ACTIVE strategy ---------------------------------------
    # L25 = V3-C AND f_liq_to_mcap >= this. Frozen from the Round-2 research
    # (TRAIN-only quantile of f_liq_to_mcap over V3-C survivors). Reported here
    # for operator visibility; l25.verify_definition() refuses any other value.
    l25_min_liq_to_mcap: float = 0.9647

    # --- SELECTABLE STRATEGY PROFILE -------------------------------------
    # Which exit-policy variant the runtime trades under. The ENTRY strategy is
    # L25 in every profile and is not selectable; see strategy_profile.py.
    #
    # L25_CURRENT is the default and is byte-equivalent to the behaviour that
    # shipped before profiles existed. L25_60S is the same L25 entry and the
    # same existing exit rules PLUS a 60s maximum hold.
    #
    # A value that is not one of the known profiles is REFUSED in __post_init__
    # rather than defaulted, so a typo cannot start the bot trading a strategy
    # nobody selected.
    strategy_profile: str = "L25_CURRENT"

    # HARD max-hold cap, in seconds, for the L25_MAX_HOLD PAPER profile.
    #
    # It exists so the cap is a CONFIGURABLE research parameter rather than a
    # constant frozen into a profile name. At its 60s default the profile
    # behaves exactly like L25_60S.
    #
    # It has NO effect on L25_CURRENT (no cap at all) or on L25_60S (fixed at
    # 60s by its own name). Validated in __post_init__: a cap of 0 would exit
    # every position instantly and a negative cap is not a hold at all, so
    # both are refused rather than quietly clamped.
    max_hold_seconds: float = 60.0

    # --- V2-A: the retained, wider decision-time market-cap ceiling ------
    # V2-A is still evaluated and still reported. V3-C's $20,000 is the
    # binding ceiling, so a token between $20,000 and $25,372.71 passes V2-A
    # and still fails V3-C. V2-A is dominated, not deleted.
    v2a_ceiling_usd: float = 25_372.71

    def __post_init__(self) -> None:
        from .strategy_profile import (
            DEFAULT_PROFILE,
            StrategyProfileError,
            parse_profile,
        )
        from .v2a import V2A_CEILING_USD, verify_ceiling
        from .v3c import (
            V3C_MAX_MARKET_CAP_USD,
            V3C_MIN_SIGNAL_QUALITY,
            verify_definition,
        )

        # The selectable profile is validated FIRST, and fails closed: an
        # unrecognised STRATEGY_PROFILE must stop the boot rather than fall back
        # to a default the operator did not ask for.
        try:
            parse_profile(self.strategy_profile)
        except StrategyProfileError as exc:
            raise ValueError(str(exc)) from exc

        # The max-hold cap must be a real positive duration. A cap of 0 would
        # exit every position the instant it opened, and a negative cap is not
        # a hold at all; both are refused rather than clamped, because a
        # silently-wrong cap is indistinguishable from a working one.
        if float(self.max_hold_seconds) <= 0:
            raise ValueError(
                f"max_hold_seconds must be positive, got "
                f"{self.max_hold_seconds!r}")

        verify_definition(
            self.v3c_max_market_cap_usd, self.v3c_min_signal_quality
        )
        from .l25 import L25_MIN_LIQ_TO_MCAP
        from .l25 import verify_definition as verify_l25_definition

        verify_l25_definition(self.l25_min_liq_to_mcap)
        if float(self.v3c_max_market_cap_usd) != float(V3C_MAX_MARKET_CAP_USD):
            raise ValueError(
                f"V3C_MAX_MARKET_CAP_USD is frozen at "
                f"${V3C_MAX_MARKET_CAP_USD:,.0f} USD"
            )
        if float(self.v3c_min_signal_quality) != float(V3C_MIN_SIGNAL_QUALITY):
            raise ValueError(
                f"V3C_MIN_SIGNAL_QUALITY is frozen at {V3C_MIN_SIGNAL_QUALITY}"
            )
        if float(self.l25_min_liq_to_mcap) != float(L25_MIN_LIQ_TO_MCAP):
            raise ValueError(
                f"L25_MIN_LIQ_TO_MCAP is frozen at {L25_MIN_LIQ_TO_MCAP:.4f}"
            )
        verify_ceiling(self.v2a_ceiling_usd)
        if float(self.v2a_ceiling_usd) != float(V2A_CEILING_USD):
            raise ValueError(
                f"V2A_CEILING_USD is frozen at {V2A_CEILING_USD:,.2f} USD")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExitConfig:
    """Deterministic exit policy parameters, per exit style."""

    # FAST_FAILURE: fragile / explosive entries
    fast_failure_stop_pct: float = -12.0
    fast_failure_window_s: float = 75.0
    # MOMENTUM_FAILURE: trailing give-back from peak
    momentum_trail_pct: float = -22.0
    momentum_min_hold_s: float = 25.0
    # RUNNER
    runner_arm_pct: float = 25.0
    runner_trail_pct: float = -18.0
    runner_hard_floor_pct: float = -8.0
    runner_min_hold_s: float = 40.0
    # FIXED_TARGET
    fixed_target_pct: float = 45.0
    # TIME_DECAY
    time_decay_no_progress_s: float = 240.0
    time_decay_no_progress_band_pct: float = 6.0
    time_decay_max_hold_s: float = 900.0
    # momentum collapse detector (any style)
    momentum_collapse_lookback_s: float = 30.0
    momentum_collapse_drop_pct: float = -18.0
    # structure break detector
    structure_break_lookback_s: float = 45.0
    structure_break_pct: float = -14.0
    # a hard boundary for the simulator only
    disaster_cap_pct: float = -45.0
    # if data goes bad for this long, exit to a known state rather than guess
    data_invalid_exit_s: float = 120.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# real execution (Solana / Jupiter)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealConfig:
    """
    Capability configuration for the REAL execution path.

    THE ONE THING TO UNDERSTAND ABOUT THIS CLASS
    --------------------------------------------
    ``trading_enabled`` here is NOT authorization. It reports whether the
    operator has armed the runtime, and the operator arms it through a confirmed
    Telegram action. The value in ``.env.local`` is read into
    ``env_trading_enabled_request`` and is deliberately NEVER copied into
    ``trading_enabled``.

    A process started with ``REAL_TRADING_ENABLED=true`` in its environment
    boots into PAPER and stays there until a human presses two buttons and a
    preflight passes. That is the whole design.
    """

    #: Wallet keypair file path. A PATH, never the key material itself. The
    #: private key is never accepted as a config value, never read from the
    #: environment and never logged.
    wallet_keypair_path: str | None = None
    #: Solana JSON-RPC endpoint. Read-only capabilities; the key is separate.
    rpc_url: str | None = None

    #: The value found in the environment, reported for operator visibility.
    #: It authorizes NOTHING.
    env_trading_enabled_request: bool = False
    #: Set by the operator's "I understand" answer, which is required before
    #: the REAL button will even offer a confirmation. Also authorizes nothing
    #: on its own.
    risk_acknowledged: bool = False

    #: Slippage tolerance sent to the venue, in basis points. Never raised to
    #: force a fill.
    slippage_bps: int = 500
    #: Refuse a quote whose price impact exceeds this percentage.
    max_price_impact_pct: float = 5.0
    #: Simulate before broadcast.
    simulate_before_send: bool = True
    #: Priority fee in lamports.
    priority_fee_lamports: int = 50_000
    #: Use the Solana Tracker Raptor aggregator as the swap build layer.
    #: Off by default: production keeps the existing path until Raptor has
    #: passed validation. A flag only, never an authorization for REAL.
    raptor_enabled: bool = False
    #: Hosted Raptor base URL. Public/no-key; no secret lives here.
    raptor_base_url: str = "https://raptor.solanatracker.io"
    #: When Raptor cannot construct a route (or its transaction is oversized),
    #: fall back to Jupiter BEFORE simulation/signing/broadcast. Only the
    #: RAPTOR_UNSUPPORTED_ROUTE / oversized cases fall back; network and build
    #: failures do not, so a provider outage can never cause a double attempt.
    raptor_fallback_enabled: bool = True
    #: When a provider returns a legacy/v0 transaction that exceeds its own
    #: 1232-byte ceiling, try to re-encode it as a v1 transaction (ceiling
    #: 4096) before declaring the token un-executable. A RE-ENCODING only: the
    #: instructions, accounts, fee payer, blockhash, amount and slippage are
    #: preserved and the result is proved equivalent before it is used. Fails
    #: closed, so turning it off only costs coverage, never safety.
    v1_rescue_enabled: bool = True
    #: How long to wait for on-chain confirmation.
    confirmation_timeout_s: float = 90.0
    #: A quote older than this cannot be used.
    quote_max_age_ms: float = 8000.0
    #: The one position cap the REAL envelope allows. Exactly one: the sizing
    #: fraction is also the total-exposure ceiling, so a second position is
    #: impossible by construction.
    max_open_positions: int = 1
    #: Never re-enter. Not configurable; stated here so the preflight can read
    #: it and report it rather than assuming it.
    reentry_enabled: bool = False

    # -- executable quote / mark divergence guard --------------------------
    #
    # ADDED 2026-10-02 AFTER THE $GOAT INCIDENT. Before these, nothing compared
    # the price the venue would charge against the mark the bot was displaying.
    # $GOAT was bought 42.6% above the mark it had just recorded and reported a
    # +71.56% result built from the mark moving toward the truth.
    #
    # ``entry_quote_divergence_max_pct`` bounds ``(quote / mark - 1) * 100`` on a
    # BUY: how much more the venue would charge than the observed market says the
    # token costs. A 10% band is a POLICY choice about tolerated disagreement
    # between two price sources. It is NOT a fitted parameter -- no backtest in
    # this repository chose it -- and it must never be tuned to make a
    # historical record look better.
    entry_quote_guard_enabled: bool = True
    entry_quote_divergence_max_pct: float = 10.0
    #: The reverse direction: an executable quote far CHEAPER than the mark is an
    #: integrity alarm (different venue, or a mark fresh in transit and wrong in
    #: substance), not an opportunity. Deliberately looser than the band above.
    entry_quote_alarm_max_pct: float = 50.0
    #: Refuse when the observed mark describes a pool the quote did not come
    #: from. Recorded either way; see the guard's docstring for why a cross-venue
    #: match is surfaced rather than fatal on its own.
    entry_quote_reject_cross_venue: bool = True

    # -- quote provider selection -----------------------------------------
    #
    # EXPLICIT, ORDERED, AND DEFAULTING TO A FREE TIER. The chain is a list of
    # provider keys resolved by the factory; an unknown key is refused at
    #: startup rather than silently skipped.
    quote_provider_chain: str = "pumpswap,pump_bonding_curve,jupiter"
    #: Attest a routed pool's owning program on chain before accepting a quote.
    #: This is what makes a venue-scoped quote a statement about a POOL and not
    #: about a label. It costs one cached RPC read per decision.
    pool_attestation_required: bool = True
    #: Show the observed-mark return in normal operator output. Off by default:
    #: a mark return printed next to a realised figure is the exact confusion
    #: $GOAT caused.
    show_mark_return_in_operator_output: bool = False

    # -- free public RPC ---------------------------------------------------
    #
    # The default on-chain read source. Free, keyless, and with NO SERVICE LEVEL
    # AGREEMENT of any kind, which is why everything reading through it is
    # cached, deduplicated, rate-limited and fail-closed.
    public_rpc_url: str = "https://api.mainnet-beta.solana.com"
    # Optional comma-separated additional read-only endpoints. The default is
    # empty on purpose: this runtime must not require a paid provider, and it
    # must not ship traffic to a third party the operator did not choose. When
    # an operator supplies more than one, a throttled node is skipped in favour
    # of another instead of deciding availability on its own.
    public_rpc_urls: str = ""
    public_rpc_cache_ttl_s: float = 2.0
    public_rpc_rate_per_s: float = 8.0
    # How long a request may WAIT for local capacity before it is refused as a
    # named rate limit. Waiting is what keeps a burst from being reported as a
    # missing market; the ceiling is what keeps the decision bounded.
    public_rpc_limiter_wait_s: float = 2.0
    public_rpc_max_concurrency: int = 4
    public_rpc_max_retries: int = 3
    public_rpc_timeout_s: float = 8.0
    # THE TELEMETRY LANE'S SHARE OF THE RPC BUDGET.
    #
    # This is a RESERVATION, not a rate increase. The total budget is still
    # ``public_rpc_rate_per_s * 60``; this only caps how much of it the
    # background hold-path sampler may take, so the remainder is structurally
    # unreachable by telemetry and always available to an open position trying
    # to sell.
    #
    # It exists because a sampler that shares one FIFO budget with the exit path
    # can consume all of it: at the old 1 Hz default a single position requested
    # roughly the entire budget, and every SELL then failed for want of capacity
    # while the position it belonged to stayed open. Set to 0 to disable
    # telemetry's RPC use entirely.
    public_rpc_telemetry_max_per_minute: int = 60

    # -- PAPER telemetry ---------------------------------------------------
    #
    # PAPER keeps three distinct price layers (OBSERVED_MARK, EXECUTABLE_QUOTE,
    # SIMULATED_FILL) rather than collapsing them into one number. This flag
    #: governs whether the simulated executable quote is fetched at all; when
    #: it is off, the layer is recorded as MISSING WITH A REASON and never as
    #: zero.
    paper_executable_quote_enabled: bool = True
    #: PAPER fills are executed against a real executable quote through the
    #: verified ``paper_live`` engine instead of the observed mark.
    #:
    #: TRUE BY DEFAULT. This is the runtime's execution model: ``autopaper run``
    #: in PAPER mode quotes every entry and every exit on a same-market
    #: executable venue, fills from that quote, and settles between the two
    #: simulated fills. An operator does not opt into it.
    #:
    #: A missing, stale or unidentifiable quote produces NO fill. It is never
    #: replaced with the observed mark, and an exit whose pinned venue cannot be
    #: quoted leaves the position OPEN.
    #:
    #: It affects PAPER only: this cannot arm REAL and reads no credential.
    paper_execution_aware_enabled: bool = True
    #: ROLLBACK / DEBUG ONLY. Set true to restore the legacy mark-based PAPER
    #: fill path, which exists so a regression in the executable-quote runtime
    #: is one environment variable to undo rather than a revert. Leave it false
    #: in normal operation; nothing else reads it.
    paper_legacy_mark_fill_rollback: bool = False

    # -- PAPER hold-path telemetry ----------------------------------------
    #
    # The instrument records, for every open PAPER position, that position's OWN
    # position-sized executable SELL reference sampled on its own schedule, on the
    # same venue the position would actually be sold through. It exists because the
    # runtime persists an observation stream only while a position is open and
    # nothing about that stream says which price the exit would really get.
    #
    # ON BY DEFAULT. It was previously unreachable: the sink was a module global that
    # nothing ever set, so ``_hold_path_config`` returned ``disabled()`` on every
    # position and no hold path was ever recorded. A measurement that cannot be
    # switched on cannot be trusted, so it now resolves from configuration.
    #
    # It is RESEARCH TELEMETRY. It cannot change an entry, a size, an exit or an
    # order: the sampler runs on its own asyncio task, quotes through the same
    # read-only quote service the executor sells through, and every failure is
    # recorded rather than raised.
    paper_hold_path_telemetry_enabled: bool = True
    #: Where observations are appended, one JSON object per line. Relative paths
    #: resolve against the runtime's ``var_dir``.
    paper_hold_path_sink: str = "paper_hold_path.jsonl"
    #: Target seconds between samples. Below the 3 s monitor cadence on purpose:
    #: the point is to see the execution venue's own path, not to imitate a tick
    #: source that does not exist.
    paper_hold_path_interval_s: float = 1.0
    #: Hard cap on sampler lifetime for one position.
    paper_hold_path_max_duration_s: float = 180.0

    # -- derived capability views ----------------------------------------

    @property
    def trading_enabled(self) -> bool:
        """
        Whether REAL is actually armed.

        This property intentionally has no config-file or environment input. It
        is overridden at runtime by the mode state machine, which is the only
        thing that may set it. Reading it on a fresh Config is always ``False``.
        """
        return False

    @property
    def wallet_configured(self) -> bool:
        return bool(self.wallet_keypair_path)

    @property
    def rpc_configured(self) -> bool:
        return bool(self.rpc_url)

    def missing_capabilities(self) -> list[str]:
        """Capability names that are absent. Never a value, never a path."""
        missing: list[str] = []
        if not self.wallet_configured:
            missing.append("WALLET_KEYPAIR_PATH")
        if not self.rpc_configured:
            missing.append("RPC_URL")
        return missing

    def as_dict(self) -> dict[str, Any]:
        # Paths and URLs are operator-facing configuration, not secrets, but
        # the keypair PATH is reduced to a bare filename so a report cannot
        # leak a home directory layout.
        return {
            "trading_enabled": self.trading_enabled,
            "env_trading_enabled_request": self.env_trading_enabled_request,
            "risk_acknowledged": self.risk_acknowledged,
            "wallet_configured": self.wallet_configured,
            "rpc_configured": self.rpc_configured,
            "missing_capabilities": self.missing_capabilities(),
            "slippage_bps": self.slippage_bps,
            "max_price_impact_pct": self.max_price_impact_pct,
            "simulate_before_send": self.simulate_before_send,
            "priority_fee_lamports": self.priority_fee_lamports,
            "raptor_enabled": self.raptor_enabled,
            "raptor_base_url": self.raptor_base_url,
            "raptor_fallback_enabled": self.raptor_fallback_enabled,
            "v1_rescue_enabled": self.v1_rescue_enabled,
            "confirmation_timeout_s": self.confirmation_timeout_s,
            "quote_max_age_ms": self.quote_max_age_ms,
            "max_open_positions": self.max_open_positions,
            "reentry_enabled": self.reentry_enabled,
            "entry_quote_guard_enabled": self.entry_quote_guard_enabled,
            "entry_quote_divergence_max_pct": self.entry_quote_divergence_max_pct,
            "entry_quote_alarm_max_pct": self.entry_quote_alarm_max_pct,
            "entry_quote_reject_cross_venue": self.entry_quote_reject_cross_venue,
            "quote_provider_chain": self.quote_provider_chain,
            "pool_attestation_required": self.pool_attestation_required,
            "show_mark_return_in_operator_output":
                self.show_mark_return_in_operator_output,
            "public_rpc_url": self.public_rpc_url,
            "public_rpc_cache_ttl_s": self.public_rpc_cache_ttl_s,
            "public_rpc_rate_per_s": self.public_rpc_rate_per_s,
            "public_rpc_max_concurrency": self.public_rpc_max_concurrency,
            "public_rpc_max_retries": self.public_rpc_max_retries,
            "public_rpc_timeout_s": self.public_rpc_timeout_s,
            "paper_executable_quote_enabled":
                self.paper_executable_quote_enabled,
            "paper_execution_aware_enabled":
                self.paper_execution_aware_enabled,
            "paper_legacy_mark_fill_rollback":
                self.paper_legacy_mark_fill_rollback,
        }


# ---------------------------------------------------------------------------
# telegram
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelegramConfig:
    api_id: int | None = None
    api_hash: str | None = None
    session: str | None = None
    chat: str | None = None
    # Operational signal-source fallback.  This is intentionally a code
    # default so enabling the fallback does not require editing .env.local.
    fallback_chat: str | None = "@painaped"
    fallback_threshold_s: float = 5 * 60 * 60
    allow_backlog: bool = False
    reconnect_backoff_s: float = 5.0
    reconnect_backoff_max_s: float = 120.0
    dedupe_window: int = 4096
    # MTProto read cadence for "new messages only". One request per interval;
    # get_messages costs nothing when the chat is quiet.
    poll_interval_s: float = 1.5
    poll_error_backoff_s: float = 2.0

    # -- Bot API (operator control + notification) ------------------------
    #:
    #: Bot token from ``TELEGRAM_BOT_TOKEN``. ``repr=False`` because the token
    #: would otherwise appear in any repr, log line, traceback or report that
    #: happens to include the config. The Bot API address is
    #: ``https://api.telegram.org/bot<token>/<method>``, so the token is a
    #: bearer credential for the bot itself.
    bot_token: Optional[str] = field(default=None, repr=False)
    #: The ONE chat allowed to control the runtime, from ``TELEGRAM_BOT_CHAT``.
    #: This is the authorization boundary: an update from any other chat id is
    #: refused before it is parsed, let alone dispatched.
    bot_chat: Optional[str] = None

    # -- Bot API transport behaviour -------------------------------------
    #: Long-poll timeout handed to Telegram. Bounded, never infinite.
    bot_poll_timeout_s: float = 20.0
    #: Per-request timeout. Must exceed ``bot_poll_timeout_s`` or a healthy
    #: long-poll would be aborted as a client timeout.
    bot_request_timeout_s: float = 30.0
    #: Bounded retries for a transient failure. Never unbounded: a permanent
    #: error must not become a hot loop against the API.
    bot_max_retries: int = 3
    bot_retry_backoff_s: float = 0.5
    bot_retry_backoff_max_s: float = 8.0

    #: The primary permitted candidate channel.
    CANDIDATE_SOURCE: ClassVar[str] = "Moo Deng President"
    FALLBACK_SOURCE: ClassVar[str] = "PainApe"

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_id and self.api_hash)

    @property
    def has_session(self) -> bool:
        return bool(self.session)

    @property
    def has_bot(self) -> bool:
        """Is a Bot API token configured? A boolean, never the token."""
        return bool(self.bot_token)

    @property
    def has_bot_chat(self) -> bool:
        """Is an operator chat allow-listed? A boolean, never the value."""
        return bool(self.bot_chat)

    @property
    def bot_control_ready(self) -> bool:
        """
        Can the Bot API control path run at all?

        Both halves are required. With a token but no chat there is nothing to
        allow-list, and allowing every chat instead would be the worst possible
        default, so this reports False rather than falling open.
        """
        return bool(self.bot_token and self.bot_chat)

    @property
    def source_name(self) -> str:
        """Human name of the configured channel, for the status message."""
        return self.CANDIDATE_SOURCE

    def as_dict(self) -> dict[str, Any]:
        """
        Safe for reports and logs.

        ``bot_token`` is dropped entirely rather than masked. A mask still
        discloses length and prefix, which is more than a report needs, and
        ``asdict`` on a field with ``repr=False`` still copies the real value -
        so it is removed here instead of being trusted to be redacted later.
        """
        d = asdict(self)
        d["fallback_source"] = self.FALLBACK_SOURCE
        d["api_hash"] = "***" if self.api_hash else None
        d["session"] = "***" if self.session else None
        d.pop("bot_token", None)
        d["bot_token_configured"] = bool(self.bot_token)
        d["bot_chat_configured"] = bool(self.bot_chat)
        d["candidate_source"] = self.CANDIDATE_SOURCE
        return d


# ---------------------------------------------------------------------------
# NAR / narrative source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NarConfig:
    """
    The NAR (narrative) source.

    ``.env.local`` as supplied configures exactly one Telegram chat, which
    carries the alerts. It contains NO separate NAR endpoint, no NAR API key and
    no NAR bot token. Those fields are therefore ``None``.

    The agent will use the configured Telegram chat as the alert source and will
    additionally attempt an optional NAR narrative lookup if the user later adds
    credentials under the ``NAR_*`` names. Until then every NAR-derived field
    (narrative label, social activity, buy/sell counts, holder counts, wallet
    flow, security flags) is reported as ``None`` and the decision engine
    downgrades its confidence. It never fabricates these values.
    """

    source: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    enabled: bool = True
    # Telegram-bot narrative source. It rides the SAME connection as the alert
    # listener, so it costs no extra login and no extra socket.
    bot_username: str = "Phanes_bot"
    bot_command: str = "/nar"
    #: Narrative FALLBACK bot. The primary bot is used until it fails
    #: ``fallback_after_failures`` consecutive OPERATIONAL times; the fallback is
    #: then used for subsequent queries, and the primary is restored the moment
    #: it answers successfully again. Reuses the identical ``/nar <CA>`` protocol.
    fallback_bot_username: str | None = "tokenscan"
    #: Consecutive primary failures before the fallback bot is used. Counted only
    #: for real operational failures (timeout / no reply / transport / malformed),
    #: never for a valid but unfavourable narrative answer.
    fallback_after_failures: int = 3
    #: Per-request narrative budget. 12s was tuned when the decision path was
    #: not latency-gated; the observed median is ~1.7s, so a 2.5s budget keeps
    #: the fast path fast while still covering the normal case.
    timeout_s: float = 2.5
    #: Grace the engine adds ON TOP of timeout_s when awaiting the narrative.
    #: Explicit and small: it used to be a hardcoded +2.0, which meant a 2.5s
    #: budget still cost 4.5s. This is a cancellation-shutdown allowance, not
    #: extra collection time.
    timeout_grace_s: float = 0.5
    min_query_gap_s: float = 4.0

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    @property
    def is_bot(self) -> bool:
        return (self.source or "").lower() in (
            "phanes_bot", "phanes", "telegram_bot"
        )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["api_key"] = "***" if self.api_key else None
        return d


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualConfig:
    """
    Counterfactual tracking of the decisions the agent did NOT take.

    * ``enabled`` True. It is the only way to answer "which gate blocks
      winners", and it changes no trading decision: see
      :mod:`autopaper.counterfactual` for why a counterfactual has no path to a live decision
      or to ``ResearchMemory``.
    * ``window_s`` 1800 -> how long a refused candidate is followed. Thirty
      minutes is long enough to cover the whole life of a memecoin entry under
      every exit style the runtime declares (the longest cap is 900s) and short
      enough that the tracked set turns over.
    * ``max_tracked`` 64 -> concurrent counterfactuals. A busy channel degrades by
      tracking fewer candidates rather than by growing without bound, and every
      refused candidate increments a counter so lost coverage is a number.
    """

    enabled: bool = True
    window_s: float = 1800.0
    max_tracked: int = 64

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationsConfig:
    """
    Append-only capture of every market observation, for ``replay``.

    * ``enabled`` True. Capture is the input ``replay`` reads, and an
      unattended run that forgot to turn it on produces weeks of trading with
      nothing to analyse afterwards. It is telemetry only: see
      :mod:`autopaper.observations` for why it can never refuse an observation
      the agent needs in order to decide.
    * ``max_mb`` 2000 -> the total disk the stream may occupy. On reaching it
      the OLDEST ROTATED file is deleted first; the active file is never
      deleted, and no file outside the stream's own directory is ever touched.
    * ``rotate_mb`` 64 -> the size at which the active file is rotated. Chosen
      so a single file stays readable with ordinary tools while the rotation
      count over weeks stays in the hundreds, not the millions. Not derived
      from ``max_mb``: the two answer different questions.
    """

    enabled: bool = True
    max_mb: float = 2000.0
    rotate_mb: float = 64.0
    #: Seconds between flushes of the write buffer. 5s because the stream is
    #: telemetry: an fsync per observation is latency the trading loop waits
    #: behind, and losing the last few seconds to a kill -9 costs a replay
    #: nothing. An orderly stop flushes, so it loses nothing at all.
    flush_s: float = 5.0
    #: Free disk required ON TOP of ``max_mb`` before `health --startup`
    #: passes. The stream is allowed to grow to its cap, so a machine with
    #: exactly ``max_mb`` free is already full.
    disk_margin_mb: float = 500.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HealthConfig:
    """
    Watchdog thresholds. The exit code is the interface; see
    :mod:`autopaper.health`.

    * ``interval_s`` 30 -> how often the heartbeat is written.
    * ``degraded_s`` 90 -> three missed beats is a warning, not a restart.
    * ``stale_s`` 300 -> ten missed beats means the loop is gone; a watchdog
      restarts on this and on nothing else.
    * ``error_rate_degraded`` 0.25 -> provider errors at or above this are
      DEGRADED, never STALE: restarting on a provider wobble would turn it into
      a restart loop, and the runtime already marks data stale rather than
      guessing.
    """

    interval_s: float = 30.0
    degraded_s: float = 90.0
    stale_s: float = 300.0
    error_rate_degraded: float = 0.25

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Config:
    agent_id: str
    root: Path
    var_dir: Path
    log_dir: Path
    state_file: Path
    ledger_file: Path
    decisions_file: Path
    candidates_file: Path
    updates_file: Path
    control_file: Path
    telegram: TelegramConfig
    nar: NarConfig
    market: MarketConfig
    execution: ExecutionModel
    risk: RiskLimits
    decision: DecisionConfig
    exits: ExitConfig
    real: RealConfig = RealConfig()
    observations: ObservationsConfig = ObservationsConfig()
    counterfactual: CounterfactualConfig = CounterfactualConfig()
    health: HealthConfig = HealthConfig()
    #: CLI-only mode. The trading core and the upstream CHANNEL listener keep
    #: working; every OUTBOUND Telegram bot call is disabled, so nothing the bot
    #: does can sit on the critical path. Telegram is two things: the channel
    #: that PRODUCES alerts (upstream, stays) and the bot that presents results
    #: (downstream, disabled here).
    cli_only: bool = False
    #: Push every strategy rejection to Telegram. Off by default: rejections are
    #: the expected outcome for most candidates. They stay in the log, journal,
    #: dashboard LAST EVENT row and on-demand /status.
    notify_rejects: bool = False
    dashboard_interval_s: float = 2.0
    real_trading_enabled: bool = False
    env: dict[str, str] = field(default_factory=dict, repr=False)
    #: Which source supplied each resolved setting. Names only, never values.
    #: Populated by the loader; empty when the Config was built by hand.
    env_origin: dict[str, str] = field(default_factory=dict, repr=False)
    #: Configuration diagnostics collected while loading: what is misconfigured,
    #: what is absent, and the key to set. Names and fixes only -- no values, so
    #: an endpoint carrying a token can never be leaked through a log line or a
    #: ``config-source`` dump.
    config_warnings: tuple[str, ...] = field(default=(), repr=False)

    @property
    def config_source(self) -> str:
        """Reported as a NAME, never contents. Returns ``.env.local`` when the
        operator's local file supplied at least one value, ``environment`` when
        only the process environment did, and ``defaults`` when neither did.
        """
        sources = set(self.env_origin.values()) - {"default"}
        if ENV_FILENAME in sources:
            return ENV_FILENAME
        if "environment" in sources:
            return "environment"
        return "defaults"

    @property
    def is_live_allowed(self) -> bool:
        return bool(
            self.telegram.has_credentials
            and self.telegram.has_session
            and self.telegram.chat
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "root": str(self.root),
            "var_dir": str(self.var_dir),
            "telegram": self.telegram.as_dict(),
            "nar": {
                **self.nar.as_dict(),
                "configured": self.nar.available,
            },
            "market": self.market.as_dict(),
            "execution_model": self.execution.as_dict(),
            "risk_limits": self.risk.as_dict(),
            "decision": self.decision.as_dict(),
            "exits": self.exits.as_dict(),
            "real": self.real.as_dict(),
            "observations": self.observations.as_dict(),
            "counterfactual": self.counterfactual.as_dict(),
            "health": self.health.as_dict(),
            "dashboard_interval_s": self.dashboard_interval_s,
        }


def _parse_env_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return _parse_env_text(path.read_text(encoding="utf-8"))


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_int(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


#: Configuration keys the native market monitor reads. Listed so the diagnostic
#: below and ``.env.example`` cannot drift apart.
NATIVE_MONITOR_ENV_KEYS = (
    "NATIVE_MONITOR_ENABLED",
    "NATIVE_MONITOR_WS_URL",
    "NATIVE_MONITOR_STALE_AFTER_MS",
    "NATIVE_MONITOR_OPEN_TIMEOUT_S",
)


def resolve_native_monitor_ws_url(value: str | None) -> tuple[str, str]:
    """
    Validate the native monitor's websocket endpoint, without inventing one.

    Returns ``(ws_url, problem)``.

    * An EMPTY value is the documented "no push endpoint" state, not an error:
      the runtime stays on the polling recovery path exactly as before.
    * A NON-empty value that is not a ``wss://`` (or local-test ``ws://``)
      endpoint is a MISCONFIGURATION and fails closed to ``""`` with a
      diagnostic naming the key. It is deliberately NOT repaired by swapping
      ``https://`` for ``wss://``: the Solana HTTP and PubSub endpoints are
      separate services, an HTTP URL proves nothing about a push endpoint, and
      an operator who set ``RPC_URL`` must not silently acquire a monitor pointed
      at a host that may not serve websockets.

    Only the SCHEME and HOST are checked here. No connection is opened, so this
    cannot be used to discover whether the endpoint actually works -- that is
    the diagnostic probe's job.
    """
    from .logging_setup import log
    from .same_market.ws import WebSocketUnavailable, parse_ws_url

    raw = (value or "").strip()
    if not raw:
        return "", ""
    # Never log the whole value: a hosted provider's endpoint routinely carries
    # an API key in the path or query, and this is a log line.
    try:
        target = parse_ws_url(raw)
    except WebSocketUnavailable as exc:
        return "", (
            f"NATIVE_MONITOR_WS_URL is set but is not a websocket endpoint "
            f"({exc}); the native market monitor is DISABLED and monitoring "
            f"falls back to RPC polling. Set it to a wss:// endpoint or leave "
            f"it empty.")
    if target.ssl_context is None:
        log("SYSTEM",
            "NATIVE_MONITOR_WS_URL uses an UNENCRYPTED ws:// endpoint "
            "(%s); acceptable for a local test only", target.host,
            level=logging.WARNING)
    return raw, ""


def build_config(root: Path | str | None = None) -> Config:
    root_path = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    env_path = root_path / ENV_FILENAME
    env = load_env_file(env_path)

    # process env wins only for explicit overrides used by tests; the file is the
    # source of truth for credentials, exactly as the user supplied it.
    # PRECEDENCE (Phase 6 stabilization)
    #
    #   .env.local  >  process environment  >  hard-coded default
    #
    # .env.local is the OPERATOR'S LOCAL SOURCE OF TRUTH. The process
    # environment is a FALLBACK for keys the operator has not set locally, so
    # CI and tests can still supply a value without editing a secret file.
    #
    # The previous order (environment first) contradicted the loader's own
    # docstring, which said the file "is the source of truth for credentials,
    # exactly as the user supplied it". An operator who set a value locally
    # could have it silently shadowed by a stale variable inherited from a shell
    # profile or a previous export. That is the opposite of a source of truth.
    #
    # This is a PRECEDENCE change only. It cannot enable REAL trading:
    # RealConfig.trading_enabled is a property with no config or environment
    # input at all, and REAL is armed only by a confirmed Telegram operator
    # action plus a passing preflight.
    _origin: dict[str, str] = {}

    def get(key: str, default: str | None = None) -> str | None:
        if key in env:
            _origin[key] = ENV_FILENAME
            return env[key]
        val = os.environ.get(key)
        if val is not None:
            _origin[key] = "environment"
            return val
        _origin[key] = "default"
        return default

    var_dir = root_path / "var"
    log_dir = root_path / "logs"
    var_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    #: Human-readable configuration problems collected while loading. Each entry
    #: names the key and the fix; a VALUE is never included, so a hosted
    #: provider endpoint carrying a token cannot leak through a log line.
    problems: list[str] = []
    _native_ws_url, _native_ws_problem = resolve_native_monitor_ws_url(
        get("NATIVE_MONITOR_WS_URL"))
    if _native_ws_problem:
        problems.append(_native_ws_problem)
    elif not _native_ws_url and _as_bool(
            get("NATIVE_MONITOR_ENABLED"), MarketConfig.native_monitor_enabled):
        # Not an error -- the operator may have no push endpoint at all -- but it
        # is the single setting that decides whether monitoring is event-driven
        # or polling, so it must say which state this run is in instead of
        # leaving a silent gap. The polling path remains correct, just expensive.
        problems.append(
            "NATIVE_MONITOR_WS_URL is not set, so the native market monitor is "
            "DISABLED and every monitoring tick uses the RPC polling recovery "
            "path. Set it to a wss:// endpoint to use the push channel.")

    session_raw = (get("TELEGRAM_SESSION") or "").strip()
    api_hash = (get("TELEGRAM_API_HASH") or "").strip() or None
    api_id = _as_int(get("TELEGRAM_API_ID"))

    telegram = TelegramConfig(
        api_id=api_id,
        api_hash=api_hash,
        session=session_raw or None,
        chat=(get("TELEGRAM_CHAT") or "").strip() or None,
        allow_backlog=_as_bool(get("TELEGRAM_ALLOW_BACKLOG"), False),
        # Bot API credentials. Read through the same precedence helper as every
        # other setting, so they are never hardcoded and never printed.
        bot_token=(get("TELEGRAM_BOT_TOKEN") or "").strip() or None,
        bot_chat=(get("TELEGRAM_BOT_CHAT") or "").strip() or None,
    )

    nar_source = (get("NAR_SOURCE") or "").strip() or None
    nar = NarConfig(
        source=nar_source or "phanes_bot",
        api_key=(get("NAR_API_KEY") or "").strip() or None,
        base_url=(get("NAR_BASE_URL") or "").strip() or None,
        enabled=_as_bool(get("NAR_ENABLED"), True),
        bot_username=(get("NAR_BOT") or "Phanes_bot").strip(),
        bot_command=(get("NAR_COMMAND") or "/nar").strip(),
        fallback_bot_username=(
            (get("NAR_FALLBACK_BOT") or "tokenscan").strip() or None
        ),
        fallback_after_failures=(
            _as_int(get("NAR_FALLBACK_AFTER_FAILURES"))
            or NarConfig.fallback_after_failures
        ),
        # The FALLBACK must equal the dataclass default (2.5), not the old 12.0.
        # 12.0 as a fallback meant that a deployment which did not set
        # NAR_TIMEOUT_S in .env.local silently ran with a 12s narrative budget
        # while every report and the dataclass said 2.5. The two places now
        # agree, and .env.local remains the source of truth when it sets one.
        timeout_s=_as_float(get("NAR_TIMEOUT_S"), NarConfig.timeout_s),
        min_query_gap_s=_as_float(get("NAR_MIN_QUERY_GAP_S"), 4.0),
    )

    market = MarketConfig(
        alert_settle_s=_as_float(get("MARKET_ALERT_SETTLE_S"), 2.0),
        poll_interval_s=_as_float(get("MARKET_POLL_INTERVAL_S"), 5.0),
        open_position_poll_interval_s=_as_float(
            get("MARKET_OPEN_POLL_INTERVAL_S"), 3.0
        ),
        stale_after_ms=_as_float(get("MARKET_STALE_AFTER_MS"), 15_000.0),
        unusable_after_ms=_as_float(get("MARKET_UNUSABLE_AFTER_MS"), 45_000.0),
        max_requests_per_minute=_as_int(
            get("MARKET_MAX_RPM")
        )
        or MarketConfig.max_requests_per_minute,
        # THE TIMEOUTS ARE NOW READABLE.
        #
        # `http_timeout_s`, `max_retries` and `cache_ttl_s` were silently NOT
        # read by the loader, so an operator who put MARKET_HTTP_TIMEOUT_S in
        # .env.local got the dataclass default and no warning. A setting that
        # looks configurable but is not is worse than one that is absent: it is
        # exactly how a latency fix gets "applied" in a config file and never
        # takes effect. Same resolution order as everything else, and the same
        # dataclass default as the fallback, so nothing changes for an operator
        # who sets nothing.
        http_timeout_s=_as_float(
            get("MARKET_HTTP_TIMEOUT_S"), MarketConfig.http_timeout_s
        ),
        connect_timeout_s=_as_float(
            get("MARKET_CONNECT_TIMEOUT_S"), MarketConfig.connect_timeout_s
        ),
        max_retries=(
            _as_int(get("MARKET_MAX_RETRIES")) or MarketConfig.max_retries
        ),
        cache_ttl_s=_as_float(
            get("MARKET_CACHE_TTL_S"), MarketConfig.cache_ttl_s
        ),
        # -- NATIVE (EVENT-DRIVEN) MARKET MONITORING ------------------------
        #
        # THESE WERE DECLARED BUT NEVER READ. ``MarketConfig`` carried the four
        # ``native_monitor_*`` fields since the push channel shipped, and
        # ``build_config`` did not pass ANY of them, so an operator who set
        # ``NATIVE_MONITOR_WS_URL`` in .env.local received the dataclass default
        # ``""`` silently. The effect was structural: ``PaperLiveEngine`` requires
        # both the enable flag AND a non-empty URL, so the push channel was
        # unavailable on every run and every monitoring tick fell back to the
        # RPC polling recovery path -- several calls per position per tick, which
        # is what exhausted the shared RPC budget and surfaced as
        # "market data unusable for 123s".
        #
        # Same resolution order as every other setting, and the URL is VALIDATED
        # rather than assumed convertible from HTTP (see
        # :func:`resolve_native_monitor_ws_url`).
        native_monitor_enabled=_as_bool(
            get("NATIVE_MONITOR_ENABLED"), MarketConfig.native_monitor_enabled
        ),
        native_monitor_ws_url=_native_ws_url,
        native_monitor_stale_after_ms=_as_float(
            get("NATIVE_MONITOR_STALE_AFTER_MS"),
            MarketConfig.native_monitor_stale_after_ms,
        ),
        native_monitor_open_timeout_s=_as_float(
            get("NATIVE_MONITOR_OPEN_TIMEOUT_S"),
            MarketConfig.native_monitor_open_timeout_s,
        ),
    )

    execution = ExecutionModel(
        entry_fee_bps=_as_float(get("PAPER_ENTRY_FEE_BPS"), ExecutionModel.entry_fee_bps),
        exit_fee_bps=_as_float(get("PAPER_EXIT_FEE_BPS"), ExecutionModel.exit_fee_bps),
        entry_slippage_bps=_as_float(
            get("PAPER_ENTRY_SLIPPAGE_BPS"), ExecutionModel.entry_slippage_bps
        ),
        exit_slippage_bps=_as_float(
            get("PAPER_EXIT_SLIPPAGE_BPS"), ExecutionModel.exit_slippage_bps
        ),
        decision_latency_ms=_as_float(
            get("PAPER_DECISION_LATENCY_MS"), ExecutionModel.decision_latency_ms
        ),
    )

    # re-entry has no override. The field exists in RiskLimits for the
    # accounting layer's cooldown bookkeeping, but the value is not read from
    # the environment: no configuration can turn it on.
    risk = RiskLimits(
        starting_bankroll_usd=_as_float(
            get("PAPER_START_BANKROLL_USD"), RiskLimits.starting_bankroll_usd
        ),
        reentry_enabled=False,
        max_simultaneous_positions=RiskLimits.max_simultaneous_positions,
    )

    real = RealConfig(
        wallet_keypair_path=(get("WALLET_KEYPAIR_PATH") or "").strip() or None,
        rpc_url=(get("RPC_URL") or get("SOLANA_RPC_URL") or "").strip() or None,
        # Recorded, never honoured as authorization.
        env_trading_enabled_request=_as_bool(
            get("REAL_TRADING_ENABLED"), False
        ),
        risk_acknowledged=_as_bool(
            get("I_UNDERSTAND_REAL_TRADING_RISKS"), False
        ),
        slippage_bps=_as_int(get("REAL_SLIPPAGE_BPS")) or RealConfig.slippage_bps,
        max_price_impact_pct=_as_float(
            get("REAL_MAX_PRICE_IMPACT_PCT"), RealConfig.max_price_impact_pct
        ),
        simulate_before_send=_as_bool(
            get("REAL_SIMULATE_BEFORE_SEND"), RealConfig.simulate_before_send
        ),
        priority_fee_lamports=_as_int(get("REAL_PRIORITY_FEE_LAMPORTS"))
        or RealConfig.priority_fee_lamports,
        raptor_enabled=_as_bool(get("RAPTOR_ENABLED"), RealConfig.raptor_enabled),
        raptor_base_url=(get("RAPTOR_BASE_URL") or "").strip()
        or RealConfig.raptor_base_url,
        raptor_fallback_enabled=_as_bool(
            get("RAPTOR_FALLBACK_ENABLED"), RealConfig.raptor_fallback_enabled
        ),
        v1_rescue_enabled=_as_bool(
            get("V1_RESCUE_ENABLED"), RealConfig.v1_rescue_enabled
        ),
        confirmation_timeout_s=_as_float(
            get("REAL_CONFIRMATION_TIMEOUT_S"), RealConfig.confirmation_timeout_s
        ),
        quote_max_age_ms=_as_float(
            get("REAL_QUOTE_MAX_AGE_MS"), RealConfig.quote_max_age_ms
        ),
        # NOT read from the environment. The micro envelope allows exactly one
        # concurrent position and never re-entry, and there is no override.
        max_open_positions=RealConfig.max_open_positions,
        reentry_enabled=RealConfig.reentry_enabled,
        # -- executable quote / mark divergence guard ----------------------
        entry_quote_guard_enabled=_as_bool(
            get("REAL_ENTRY_QUOTE_GUARD_ENABLED"),
            RealConfig.entry_quote_guard_enabled,
        ),
        entry_quote_divergence_max_pct=_as_float(
            get("REAL_ENTRY_QUOTE_DIVERGENCE_MAX_PCT"),
            RealConfig.entry_quote_divergence_max_pct,
        ),
        entry_quote_alarm_max_pct=_as_float(
            get("REAL_ENTRY_QUOTE_ALARM_MAX_PCT"),
            RealConfig.entry_quote_alarm_max_pct,
        ),
        entry_quote_reject_cross_venue=_as_bool(
            get("REAL_ENTRY_QUOTE_REJECT_CROSS_VENUE"),
            RealConfig.entry_quote_reject_cross_venue,
        ),
        quote_provider_chain=(get("REAL_QUOTE_PROVIDER_CHAIN") or "").strip()
        or RealConfig.quote_provider_chain,
        pool_attestation_required=_as_bool(
            get("REAL_POOL_ATTESTATION_REQUIRED"),
            RealConfig.pool_attestation_required,
        ),
        show_mark_return_in_operator_output=_as_bool(
            get("SHOW_MARK_RETURN_IN_OPERATOR_OUTPUT"),
            RealConfig.show_mark_return_in_operator_output,
        ),
        # -- free public RPC ------------------------------------------------
        # Defaults to the free public endpoint. A private endpoint may be
        # supplied, but none is required: nothing in this runtime depends on a
        # paid RPC or a paid data tier.
        public_rpc_url=(get("PUBLIC_RPC_URL") or "").strip()
        or RealConfig.public_rpc_url,
        public_rpc_urls=(get("PUBLIC_RPC_URLS") or "").strip(),
        public_rpc_limiter_wait_s=_as_float(
            get("PUBLIC_RPC_LIMITER_WAIT_S"),
            RealConfig.public_rpc_limiter_wait_s),
        public_rpc_cache_ttl_s=_as_float(
            get("PUBLIC_RPC_CACHE_TTL_S"), RealConfig.public_rpc_cache_ttl_s),
        public_rpc_rate_per_s=_as_float(
            get("PUBLIC_RPC_RATE_PER_S"), RealConfig.public_rpc_rate_per_s),
        public_rpc_max_concurrency=_as_int(
            get("PUBLIC_RPC_MAX_CONCURRENCY"))
        or RealConfig.public_rpc_max_concurrency,
        public_rpc_telemetry_max_per_minute=_as_int(
            get("PUBLIC_RPC_TELEMETRY_MAX_PER_MINUTE"))
        if get("PUBLIC_RPC_TELEMETRY_MAX_PER_MINUTE") not in (None, "")
        else RealConfig.public_rpc_telemetry_max_per_minute,
        public_rpc_max_retries=_as_int(get("PUBLIC_RPC_MAX_RETRIES"))
        if _as_int(get("PUBLIC_RPC_MAX_RETRIES")) is not None
        else RealConfig.public_rpc_max_retries,
        public_rpc_timeout_s=_as_float(
            get("PUBLIC_RPC_TIMEOUT_S"), RealConfig.public_rpc_timeout_s),
        # -- PAPER telemetry -------------------------------------------------
        paper_executable_quote_enabled=_as_bool(
            get("PAPER_EXECUTABLE_QUOTE_ENABLED"),
            RealConfig.paper_executable_quote_enabled,
        ),
        paper_execution_aware_enabled=_as_bool(
            get("PAPER_EXECUTION_AWARE_ENABLED"),
            RealConfig.paper_execution_aware_enabled,
        ),
        paper_legacy_mark_fill_rollback=_as_bool(
            get("PAPER_LEGACY_MARK_FILL_ROLLBACK"),
            RealConfig.paper_legacy_mark_fill_rollback,
        ),
        paper_hold_path_telemetry_enabled=_as_bool(
            get("PAPER_HOLD_PATH_TELEMETRY"),
            RealConfig.paper_hold_path_telemetry_enabled,
        ),
        paper_hold_path_sink=get("PAPER_HOLD_PATH_SINK")
        or RealConfig.paper_hold_path_sink,
        paper_hold_path_interval_s=_as_float(
            get("PAPER_HOLD_PATH_INTERVAL_S"),
            RealConfig.paper_hold_path_interval_s),
        paper_hold_path_max_duration_s=_as_float(
            get("PAPER_HOLD_PATH_MAX_DURATION_S"),
            RealConfig.paper_hold_path_max_duration_s),
    )

    cfg = Config(
        agent_id="AUTONOMOUS_PAPER_AGENT_V1",
        root=root_path,
        var_dir=var_dir,
        log_dir=log_dir,
        state_file=var_dir / "paper_account.json",
        ledger_file=var_dir / "ledger.jsonl",
        decisions_file=var_dir / "decisions.jsonl",
        candidates_file=var_dir / "candidates.jsonl",
        updates_file=var_dir / "position_updates.jsonl",
        control_file=var_dir / "control.json",
        telegram=telegram,
        nar=nar,
        market=market,
        execution=execution,
        risk=risk,
        decision=DecisionConfig(
            # Same read idiom as WALLET_KEYPAIR_PATH: strip, and treat an absent
            # or blank value as "not configured" so the default applies. A
            # NON-blank value that is not a known profile is refused by
            # DecisionConfig.__post_init__ rather than defaulted.
            strategy_profile=(get("STRATEGY_PROFILE") or "").strip()
            or "L25_CURRENT",
            max_hold_seconds=_as_float(
                get("MAX_HOLD_SECONDS"), DecisionConfig.max_hold_seconds),
        ),
        exits=ExitConfig(disaster_cap_pct=risk.disaster_cap_pct),
        real=real,
        observations=ObservationsConfig(
            enabled=_as_bool(get("OBS_ENABLED"),
                             ObservationsConfig.enabled),
            max_mb=_as_float(get("OBS_MAX_MB"),
                             ObservationsConfig.max_mb),
            rotate_mb=_as_float(get("OBS_ROTATE_MB"),
                                ObservationsConfig.rotate_mb),
            flush_s=_as_float(get("OBS_FLUSH_S"), ObservationsConfig.flush_s),
            disk_margin_mb=_as_float(get("OBS_DISK_MARGIN_MB"),
                                     ObservationsConfig.disk_margin_mb),
        ),
        health=HealthConfig(
            interval_s=_as_float(get("HEARTBEAT_INTERVAL_S"),
                                 HealthConfig.interval_s),
            degraded_s=_as_float(get("HEARTBEAT_DEGRADED_S"),
                                 HealthConfig.degraded_s),
            stale_s=_as_float(get("HEARTBEAT_STALE_S"), HealthConfig.stale_s),
            error_rate_degraded=_as_float(
                get("HEALTH_ERROR_RATE_DEGRADED"),
                HealthConfig.error_rate_degraded),
        ),
        counterfactual=CounterfactualConfig(
            enabled=_as_bool(get("COUNTERFACTUAL_ENABLED"), CounterfactualConfig.enabled),
            window_s=_as_float(get("COUNTERFACTUAL_WINDOW_S"),
                               CounterfactualConfig.window_s),
            max_tracked=_as_int(get("COUNTERFACTUAL_MAX_TRACKED"))
            or CounterfactualConfig.max_tracked,
        ),
        env=env,
    )

    if api_hash and cfg.telegram.chat and not cfg.telegram.session:
        # not an error: the operator may complete login later
        pass

    # Attach the per-key origin map. The frozen dataclass is replaced via
    # object.__setattr__ because `env_origin` exists precisely so this
    # diagnostic can be added without touching the resolution order.
    object.__setattr__(cfg, "env_origin", dict(_origin))

    # Report every configuration diagnostic collected above. Each one is a named
    # key plus the fix; none of them carries a value, so nothing here can leak a
    # credential. They are non-fatal on purpose: a misconfigured monitor must
    # leave the runtime on its existing conservative polling path rather than
    # refuse to boot a PAPER run.
    if problems:
        object.__setattr__(cfg, "config_warnings", tuple(problems))
        for _problem in problems:
            from .logging_setup import log

            log("SYSTEM", "CONFIG: %s", _problem, level=logging.WARNING)

    # A private key in the environment file is a credential-storing mistake,
    # not a capability. It is reported and refused, because the wallet is
    # loaded from a keypair FILE and an env-var key would bypass the file
    # permission boundary the executor depends on.
    for _bad in ("PRIVATE_KEY", "SOLANA_PRIVATE_KEY", "SEED_PHRASE", "MNEMONIC"):
        if _bad in env or _bad in os.environ:
            raise RealExecutionBlocked(
                f"Refusing to load config: {_bad} is present in {ENV_FILENAME}. "
                "The wallet is loaded from a keypair FILE "
                "(WALLET_KEYPAIR_PATH); key material must not live in an "
                "environment file."
            )
    return cfg


def describe(cfg: Config) -> str:
    # Imported here, matching the local-import convention used elsewhere in this
    # module, so the strategy IDENTITY is read from the one authoritative
    # constant rather than a name typed into a report string.
    from .l25 import ACTIVE_STRATEGY
    from .logging_setup import log
    from .strategy_profile import describe as describe_profile

    # Whether market monitoring is event-driven or on the polling recovery path
    # is the one line an operator most often needs to see after a
    # "market data unusable for 123s" run, so it is printed rather than inferred.
    ws_cfg = cfg.market
    native = ("wss:// (configured)" if ws_cfg.native_monitor_ws_url
              else "DISABLED - no NATIVE_MONITOR_WS_URL, polling recovery")
    lines = (
        f"agent={cfg.agent_id}\n"
        f"root={cfg.root}\n"
        f"telegram.credentials={'yes' if cfg.telegram.has_credentials else 'NO'}\n"
        f"telegram.session={'yes' if cfg.telegram.has_session else 'MISSING'}\n"
        f"telegram.chat={cfg.telegram.chat}\n"
        f"telegram.allow_backlog={cfg.telegram.allow_backlog}\n"
        f"nar.source={cfg.nar.source or 'NOT CONFIGURED'}\n"
        f"nar.available={'yes' if cfg.nar.available else 'no (fields -> null)'}\n"
        f"market.primary={cfg.market.primary} fallback={cfg.market.fallback}\n"
        f"native_monitor={native}\n"
        f"bankroll=${cfg.risk.starting_bankroll_usd:.2f}\n"
        # The strategy IDENTITY, so it must track the active strategy rather
        # than name one leg of it. The legs are listed separately below.
        f"strategy={ACTIVE_STRATEGY}\n"
        f"  V3-C leg            : MC<=$20,000 SQ>=0.55\n"
        f"{describe_profile(cfg.decision.strategy_profile)}\n"
        f"candidate_source={cfg.telegram.CANDIDATE_SOURCE} (only)\n"
        f"real.trading_enabled={cfg.real.trading_enabled}\n"
        f"real.env_request_ignored={cfg.real.env_trading_enabled_request}\n"
        f"real.missing_capabilities={cfg.real.missing_capabilities() or 'none'}\n"
    )
    for _warning in cfg.config_warnings:
        lines += f"config_warning={_warning}\n"
        log("SYSTEM", "CONFIG: %s", _warning, level=logging.WARNING)
    return lines
