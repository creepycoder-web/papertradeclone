"""
Event-driven market monitoring: the split, and the guarantees around it.

THE ARCHITECTURE UNDER TEST
---------------------------
    MARKET STATE MONITORING -> event-driven native WebSocket account updates
    EXECUTION VALIDATION   -> an executable SELL quote, only when one is needed

Before this, one 5-second timer asked for a FULL executable SELL quote for
every open position on every tick and discarded it on every HOLD. These tests
assert the split holds, and -- just as importantly -- assert the things that
could have been broken by making it: the exit thresholds, the identity gate,
the quote freshness gate, and the fail-closed behaviour when the push channel
is not available.

The most important negative assertions in this file are the ones that say a
stale push is NEVER sold as a fresh price, because that failure mode is silent:
the number looks reasonable and the position quietly trades on stale
information.
"""

from __future__ import annotations

import asyncio
import base64
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from autopaper.config import ExitConfig
from autopaper.paper_live.native_monitor import (
    MonitorConfig,
    NativeMarketMonitor,
    NativeMonitorRegistry,
    StaticVenueIdentity,
    SOURCE_PUSH,
    STATE_HEALTHY,
    STATE_STALE,
    STATE_STOPPED,
    _base64_account,
    _notification_slot,
)
from autopaper.paper_live.quote_proximity import (
    DEFAULT_PROXIMITY,
    MONITOR_ONLY,
    QUOTE_ALLOWED,
    QUOTE_REQUIRED,
    ProximityPolicy,
    evaluate,
    existing_price_triggers,
)
from autopaper.same_market import pumpdirect as pd
from autopaper.same_market.pumpdirect import (
    MarketKind,
    SOL_DECIMALS,
    _spot_price,
    decode_bonding_curve,
    decode_pool,
    derive_bonding_curve,
    derive_canonical_pool,
)
from autopaper.same_market.ws import (
    WebSocketConnection,
    WebSocketUnavailable,
    encode_frame,
    parse_ws_url,
)
from autopaper.same_market.pda import b58encode

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from tests import pumpdirect_accounts as _acct
except ImportError:
    import pumpdirect_accounts as _acct


MINT = "2CNVaZ1sr2qpFvvbZbmmyChtifMBScMK7ktQGV8hpump"
CURVE_ADDRESS, _CURVE_BUMP = derive_bonding_curve(MINT)
POOL_ADDRESS, _POOL_BUMP = derive_canonical_pool(MINT)
VAULT_BASE = "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin"
VAULT_QUOTE = "2wT8Yq49kHgDzXuPxZSaeLaH1qbmGXtEyPy64bL7aD3c"


def _identity(market_type: str = MarketKind.PUMP_BONDING_CURVE.value,
              address: str = CURVE_ADDRESS, **kw) -> StaticVenueIdentity:
    base = dict(
        mint=MINT, market_type=market_type, market_address=address,
        base_mint=MINT, quote_mint="So11111111111111111111111111111111111111112",
        token_decimals=6,
    )
    base.update(kw)
    return StaticVenueIdentity(**base)


def _curve_notification(raw: bytes, *, slot: int = 1000,
                        sub_id: int = 7) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "accountNotification",
        "params": {
            "subscription": sub_id,
            "result": {
                "context": {"slot": slot},
                "value": {
                    "lamports": 1,
                    "data": [base64.b64encode(raw).decode(), "base64"],
                    "owner": pd.PUMP_FUN_PROGRAM,
                },
            },
        },
    }


def _bind(monitor: NativeMarketMonitor, sub_id: int = 7) -> None:
    """
    Pretend ``accountSubscribe`` already succeeded, offline.

    Routing now lives on the shared stream, so the fake binding stands up a
    stream with one route per account rather than pretending the monitor owns a
    socket.
    """
    from autopaper.paper_live.native_monitor import SharedSubscriptionStream

    stream = SharedSubscriptionStream("wss://unused", config=monitor.config)
    monitor._stream = stream
    stream._monitors.append(monitor)
    for address, role in monitor.identity.subscription_addresses():
        monitor._subscriptions[sub_id] = (address, role)
        monitor.stats.subscription_ids[role] = sub_id
        stream._routes[sub_id] = (monitor, address, role)
    monitor.state = STATE_HEALTHY


def _push(monitor: NativeMarketMonitor, raw: bytes, *, slot: int = 1000,
          sub_id: int = 7) -> None:
    """One account push, routed exactly as the shared stream would route it."""
    monitor._stream._dispatch(_curve_notification(raw, slot=slot,
                                                  sub_id=sub_id))


# ===========================================================================
# 1. the local price calculation
# ===========================================================================


def test_a_pushed_curve_price_matches_the_polled_quote_exactly():
    """
    The decisive equivalence.

    The pushed price is computed by the SAME ``_spot_price`` the executable
    quote path uses for its ``execution_native_reference``. If these two ever
    differ, a position would be evaluated on a different number depending on
    whether the channel was healthy, and "the monitoring price" would become a
    second definition of the truth. So they are compared on the same bytes.
    """
    raw = _acct.bonding_curve_bytes(
        virtual_token_reserves=320_098_471_659_904,
        virtual_quote_reserves=100_562_804_716,
    )
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, raw, slot=555)

    point = monitor.latest()
    assert point is not None and point.usable

    curve = decode_bonding_curve(raw, curve_address=CURVE_ADDRESS, mint=MINT)
    expected = _spot_price(
        numerator=curve.virtual_quote_reserves,
        denominator=curve.virtual_token_reserves,
        input_decimals=SOL_DECIMALS,
        output_decimals=6,
    )
    assert point.price_sol_per_token == expected
    assert point.slot == 555
    assert point.source == SOURCE_PUSH


def test_the_pushed_price_moves_when_the_reserves_move():
    """It is a live price, not a cached one."""
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, _acct.bonding_curve_bytes(
        virtual_token_reserves=320_098_471_659_904,
        virtual_quote_reserves=100_562_804_716), slot=1)
    first = monitor.latest().price_sol_per_token
    _push(monitor, _acct.bonding_curve_bytes(
        virtual_token_reserves=320_098_471_659_904,
        virtual_quote_reserves=200_562_804_716), slot=2)
    second = monitor.latest().price_sol_per_token
    assert second > first, "the pushed price did not respond to the reserves"
    assert monitor.stats.updates == 2


def test_many_updates_arrive_between_normal_exit_checks():
    """
    The whole point of the change, stated as a test.

    A 5s poller cannot produce more than one observation per five seconds, so a
    test that drives 20 pushes in a tight loop and asserts 20 distinct
    observations is asserting the opposite of polling. No sleeping is involved:
    this is the property that matters, and sleeping would only make it flaky.
    """
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    seen = []
    for index in range(20):
        reserves = 100_562_804_716 + index
        _push(monitor, _acct.bonding_curve_bytes(
            virtual_token_reserves=320_098_471_659_904,
            virtual_quote_reserves=reserves), slot=1000 + index)
        seen.append(monitor.latest().price_sol_per_token)
    assert len(set(seen)) == 20, "the monitor did not resolve 20 distinct prices"
    assert monitor.stats.quotes_from_push == 20


def test_a_pump_swap_pool_price_uses_the_vault_balances_plus_the_virtual_part():
    """
    A pool's reserves are NOT in the pool account, so the monitor subscribes to
    the two vaults the pool names as well. The price uses
    ``vault_quote_balance + virtual_quote_reserves`` over ``vault_base_balance``
    -- the same composition the executable pool quote uses.
    """
    pool = _acct.pool_bytes(
        creator=VAULT_BASE,
        base_mint=MINT,
        pool_base_token_account=VAULT_BASE,
        pool_quote_token_account=VAULT_QUOTE,
        virtual_quote_reserves=1_000_000_000,
    )
    monitor = NativeMarketMonitor(
        _identity(market_type=MarketKind.PUMPSWAP.value, address=POOL_ADDRESS,
                  pool_base_vault=VAULT_BASE, pool_quote_vault=VAULT_QUOTE,
                  pool_base_decimals=6, pool_base_supply=1_000_000_000_000_000),
        config=MonitorConfig(ws_url="wss://unused"))
    # A pool needs THREE subscriptions, so they get distinct ids. Binding all
    # three to one id would model a stream that cannot exist.
    from autopaper.paper_live.native_monitor import SharedSubscriptionStream

    stream = SharedSubscriptionStream("wss://unused", config=monitor.config)
    monitor._stream = stream
    stream._monitors.append(monitor)
    monitor._subscriptions = {
        7: (POOL_ADDRESS, "POOL"),
        8: (VAULT_BASE, "BASE_VAULT"),
        9: (VAULT_QUOTE, "QUOTE_VAULT"),
    }
    for sid, (address, role) in monitor._subscriptions.items():
        stream._routes[sid] = (monitor, address, role)
    monitor.state = STATE_HEALTHY

    def _push(role: str, raw: bytes, slot: int) -> None:
        monitor._store_bytes(role, raw)
        monitor._recompute(slot, POOL_ADDRESS)
        monitor._record_arrival(slot, POOL_ADDRESS)

    # The pool account alone is not enough: it carries no balances.
    _push("POOL", pool, 10)
    assert monitor.latest() is None, (
        "a pool price appeared with no vault balances at all")
    _push("BASE_VAULT", _acct.token_account_bytes(
        amount=1_000_000_000, mint=MINT), 11)
    assert monitor.latest() is None, (
        "a pool price appeared with only one vault read")
    _push("QUOTE_VAULT", _acct.token_account_bytes(
        amount=50_000_000_000, mint="So11111111111111111111111111111111111111112"), 12)

    point = monitor.latest()
    assert point is not None and point.usable
    fields = decode_pool(pool, pool_address=POOL_ADDRESS)
    expected = _spot_price(
        numerator=50_000_000_000 + fields["virtual_quote_reserves"],
        denominator=1_000_000_000,
        input_decimals=SOL_DECIMALS,
        output_decimals=6,
    )
    assert point.price_sol_per_token == expected
    assert point.market_address == POOL_ADDRESS


# ===========================================================================
# 2. staleness, recovery and "never sell silence as freshness"
# ===========================================================================


def test_a_silent_stream_is_never_reported_as_a_fresh_price():
    """
    THE most important negative assertion in this file.

    A quiet token produces NO pushes at all. If a monitor reported its last
    price as current after the stream went quiet, a position would keep trading
    on a price nobody is updating, and every log line would say the market was
    live. So exceeding the staleness bound makes the point unusable, with a
    reason, and the caller falls back.
    """
    monitor = NativeMarketMonitor(
        _identity(),
        config=MonitorConfig(ws_url="wss://unused", stale_after_ms=50.0))
    _bind(monitor)
    _push(monitor, _acct.bonding_curve_bytes(), slot=1)
    assert monitor.latest().usable is True

    # Age the stream past the bound by rewinding the monotonic stamp.
    monitor._last_update_monotonic -= 1.0     # 1000ms, bound is 50ms
    point = monitor.latest()
    assert point is not None, "a price object is still returned"
    assert point.usable is False
    assert "old" in point.reason or "stale" in point.reason.lower()


def test_a_price_is_unusable_unless_the_stream_is_healthy():
    for state in (STATE_STALE, STATE_STOPPED, "WARMING", "RECONNECTING", "IDLE"):
        monitor = NativeMarketMonitor(
            _identity(), config=MonitorConfig(ws_url="wss://unused"))
        _bind(monitor)
        _push(monitor, _acct.bonding_curve_bytes(), slot=1)
        monitor.state = state
        point = monitor.latest()
        assert point is not None and point.usable is False, state
        assert state in point.reason


def test_no_price_at_all_before_the_first_push():
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    assert monitor.latest() is None


def test_a_decode_failure_is_counted_not_crashed():
    """Garbage bytes must not take the monitor down or invent a price."""
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, b"\x00" * 200, slot=1)
    assert monitor.stats.decode_failures >= 1
    assert monitor.latest() is None, "garbage produced a price"


def test_a_completed_curve_offers_no_curve_price():
    """
    A graduated curve is not a market. The monitor reports no price for it and
    records the migration, rather than quoting a curve that no longer exists.
    """
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, _acct.bonding_curve_bytes(complete=True), slot=3)

    assert monitor.latest() is None, (
        "a graduated curve still produced a price")
    migration = monitor.migration()
    assert migration is not None
    assert migration.from_market_address == CURVE_ADDRESS
    assert migration.to_market_address == POOL_ADDRESS
    assert migration.to_venue_id == f"{MarketKind.PUMPSWAP.value}:{POOL_ADDRESS}"
    assert monitor._curve_bytes == b"", (
        "the graduated curve's price was retained after the switch")


def test_a_migration_records_the_switch_so_pre_and_post_prices_never_mix():
    """
    The curve's price is DISCARDED on the switch. A position evaluated
    immediately after a graduation must get "no price from this market", not
    the last curve number.
    """
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, _acct.bonding_curve_bytes(), slot=1)
    assert monitor.latest() is not None
    _push(monitor, _acct.bonding_curve_bytes(
        virtual_quote_reserves=900_000_000_000, complete=True), slot=2)
    assert monitor.latest() is None


def test_a_second_graduation_push_does_not_re_record_the_migration():
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, _acct.bonding_curve_bytes(complete=True), slot=1)
    first = monitor.migration()
    _push(monitor, _acct.bonding_curve_bytes(
        virtual_quote_reserves=900_000_000_000, complete=True), slot=2)
    assert monitor.migration() is first


# ===========================================================================
# 3. identity is not relaxed
# ===========================================================================


def test_a_curve_push_for_an_account_that_is_not_the_pinned_market_is_refused():
    """
    The monitor recomputes the pin from the mint and refuses a payload for any
    other account. A wrong-but-well-formed price must not reach the exit policy.
    """
    other = StaticVenueIdentity(
        mint=MINT, market_type=MarketKind.PUMP_BONDING_CURVE.value,
        market_address="9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin",
        token_decimals=6)
    monitor = NativeMarketMonitor(
        other, config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, _acct.bonding_curve_bytes(), slot=1)
    assert monitor.latest() is None, (
        "a push for a market the position is not pinned to produced a price")


def test_the_monitor_identity_reports_the_market_it_is_pinned_to():
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    assert monitor.venue_id == f"{MarketKind.PUMP_BONDING_CURVE.value}:{CURVE_ADDRESS}"
    assert monitor.market_address == CURVE_ADDRESS


def test_the_subscription_set_is_exactly_the_markets_own_accounts():
    curve = _identity().subscription_addresses()
    assert curve == ((CURVE_ADDRESS, "CURVE"),), (
        "a bonding curve needs exactly one account: the curve carries both "
        "virtual reserves, so subscribing to anything else is waste")

    pool = _identity(
        market_type=MarketKind.PUMPSWAP.value, address=POOL_ADDRESS,
        pool_base_vault=VAULT_BASE, pool_quote_vault=VAULT_QUOTE,
    ).subscription_addresses()
    assert set(pool) == {
        (POOL_ADDRESS, "POOL"),
        (VAULT_BASE, "BASE_VAULT"),
        (VAULT_QUOTE, "QUOTE_VAULT"),
    }


def test_static_identity_is_not_rediscovered_per_update():
    """
    The pin, the addresses and the decimals are carried, not re-derived. A
    subscription set that changed between updates would mean the monitor was
    silently re-resolving the market on every tick, which is the cost this
    architecture removes.
    """
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    before = monitor.identity
    for index in range(5):
        _push(monitor, _acct.bonding_curve_bytes(
            virtual_quote_reserves=100_000_000_000 + index), slot=index)
    assert monitor.identity == before
    assert set(monitor._subscriptions) == {7}, (
        "the subscription set changed while updates were arriving")


# ===========================================================================
# 4. gaps, reconnects and counters
# ===========================================================================


def test_a_backwards_slot_is_counted_as_a_stream_gap():
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, _acct.bonding_curve_bytes(), slot=100)
    _push(monitor, _acct.bonding_curve_bytes(), slot=200)
    _push(monitor, _acct.bonding_curve_bytes(), slot=150)
    assert monitor.stats.stream_gaps == 1
    assert monitor._high_water_slot == 200, (
        "the high-water slot was lowered by an out-of-order update")


def test_a_forward_slot_is_not_a_gap():
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    for slot in (10, 11, 12):
        _push(monitor, _acct.bonding_curve_bytes(), slot=slot)
    assert monitor.stats.stream_gaps == 0


def test_a_notification_for_an_unknown_subscription_is_ignored():
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, _acct.bonding_curve_bytes(), slot=1, sub_id=999)
    assert monitor.stats.updates == 0
    assert monitor.stats.stream_gaps == 0, (
        "a post-migration straggler was counted as a gap")


def test_update_interval_statistics_summarise_the_trailing_window():
    """
    Median, p95 and max are what turn "it is event-driven" into a number.

    The intervals are computed by the monitor from MONOTONIC deltas, so an NTP
    step cannot invent or hide a gap; the window is bounded so a long run cannot
    turn a rate into a leak.
    """
    from autopaper.paper_live.native_monitor import MonitorStats

    stats = MonitorStats()
    for value in (100.0,) * 40 + (5_000.0,):
        stats.note_interval(value)
    assert stats.interval_samples == 41
    assert stats.update_interval_median_ms == 100.0
    assert stats.update_interval_p95_ms == 100.0
    assert stats.update_interval_max_ms == 5_000.0, (
        "a single stall must not disappear from the max")

    # And a genuinely tail-heavy window surfaces in p95, not just in max. A
    # 10% tail must move p95, or p95 is really reporting the median.
    stats = MonitorStats()
    for value in (10.0,) * 90 + (2_000.0,) * 10:
        stats.note_interval(value)
    assert stats.update_interval_p95_ms > 100.0
    assert stats.update_interval_median_ms == 10.0


def test_the_arrival_path_records_an_interval_from_a_monotonic_delta():
    """The monitor stamps arrivals with a MONOTONIC clock, not wall time."""
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)

    base = [1000.0]
    monitor._clock = SimpleNamespace(monotonic=lambda: base[0])

    _push(monitor, _acct.bonding_curve_bytes(), slot=1)
    base[0] += 0.25          # 250ms later
    _push(monitor, _acct.bonding_curve_bytes(), slot=2)
    base[0] += 0.25
    _push(monitor, _acct.bonding_curve_bytes(), slot=3)

    assert monitor.stats.interval_samples == 2
    assert abs(monitor.stats.update_interval_median_ms - 250.0) < 1.0
    assert monitor.age_ms() == 0.0, (
        "the age must be measured from the injected monotonic clock, not from "
        "a wall clock that could step under it")


def test_the_interval_window_is_bounded():
    """A rate that grows with the length of the run is a leak, not a metric."""
    from autopaper.paper_live.native_monitor import _INTERVAL_WINDOW
    stats = SimpleNamespace()
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    for index in range(_INTERVAL_WINDOW + 500):
        monitor.stats.note_interval(float(index % 97))
    assert len(monitor.stats.intervals) == _INTERVAL_WINDOW


def test_going_disconnected_marks_the_stream_stale_rather_than_healthy():
    """
    Losing the shared socket must not leave the last price looking current.

    This is the shared-connection cost taken seriously: one socket carries every
    market, so a drop is everyone's problem, and every affected market has to
    stop being trusted at the same moment.
    """
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, _acct.bonding_curve_bytes(), slot=1)
    assert monitor.latest().usable is True

    monitor.stream_connected(False)
    assert monitor.state == STATE_STALE
    # The last price is STILL THERE -- dropping it would lose information -- but
    # it is no longer usable, because nobody is updating it any more.
    assert monitor.latest() is not None
    assert monitor.latest().usable is False


def test_reconnecting_restores_health_and_rebinds_the_subscription_ids():
    """
    After a reconnect the OLD subscription ids are gone.

    A monitor that kept using them would silently receive nothing while
    reporting HEALTHY, which is the worst possible combination: it looks fine
    and it is blind.
    """
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    _push(monitor, _acct.bonding_curve_bytes(), slot=1)
    monitor.stream_connected(False)

    monitor.stream_connected(True)
    monitor.stream_resubscribed({99: (CURVE_ADDRESS, "CURVE")})
    assert set(monitor._subscriptions) == {99}
    assert 7 not in monitor._subscriptions, (
        "a dead subscription id survived a reconnect")
    assert monitor.state == STATE_HEALTHY


def test_a_stream_error_is_recorded_and_marks_stale():
    monitor = NativeMarketMonitor(
        _identity(), config=MonitorConfig(ws_url="wss://unused"))
    _bind(monitor)
    monitor.stream_error("HTTP/1.1 429 Too Many Requests")
    assert monitor.stats.last_error.startswith("HTTP/1.1 429")
    assert monitor.state == STATE_STALE


def test_the_registry_uses_one_connection_for_every_market():
    """
    THE STRUCTURAL FIX.

    One connection per market was measured to earn ``429 Too Many Requests`` on
    the WebSocket HANDSHAKE and left every monitor permanently reconnecting with
    zero updates. Solana serves many subscriptions per connection, so the scarce
    resource is connections, not subscriptions: N markets must cost ONE socket.
    """
    from autopaper.paper_live.native_monitor import SharedSubscriptionStream

    async def _run():
        reg = NativeMonitorRegistry()
        cfg = MonitorConfig(ws_url="wss://example.invalid")
        monitors = [
            reg.acquire(StaticVenueIdentity(
                mint=MINT, market_type=MarketKind.PUMP_BONDING_CURVE.value,
                market_address=f"addr{i}", token_decimals=6), cfg)
            for i in range(25)
        ]
        assert len(reg.monitors()) == 25, "markets collapsed into one"
        assert len(reg.streams()) == 1, (
            f"25 markets opened {len(reg.streams())} connections; the endpoint "
            f"rate limits handshakes, so this must be 1")
        stream = reg.streams()[0]
        assert stream in [m._stream for m in monitors]
        await reg.aclose()

    asyncio.run(_run())


def test_one_frame_reaches_exactly_one_market():
    """
    Multiplexing must not blur markets.

    Two markets on one socket, two distinct subscription ids: a push for one is
    handed to that market and changes nothing about the other. Blending them
    would put one token's price on another token's position.
    """
    async def _run():
        from autopaper.paper_live.native_monitor import SharedSubscriptionStream

        cfg = MonitorConfig(ws_url="wss://example.invalid")
        stream = SharedSubscriptionStream("wss://example.invalid", config=cfg)
        first = NativeMarketMonitor(
            StaticVenueIdentity(
                mint=MINT, market_type=MarketKind.PUMP_BONDING_CURVE.value,
                market_address=CURVE_ADDRESS, token_decimals=6),
            config=cfg, stream=stream)
        other_curve, _ = pd.derive_bonding_curve(MINT)
        second = NativeMarketMonitor(
            StaticVenueIdentity(
                mint="9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin",
                market_type=MarketKind.PUMP_BONDING_CURVE.value,
                market_address=other_curve, token_decimals=6),
            config=cfg, stream=stream)
        stream._routes[1] = (first, CURVE_ADDRESS, "CURVE")
        stream._routes[2] = (second, other_curve, "CURVE")
        stream._monitors.extend([first, second])
        first.stream_resubscribed({1: (CURVE_ADDRESS, "CURVE")})
        second.stream_resubscribed({2: (other_curve, "CURVE")})

        raw = _acct.bonding_curve_bytes(virtual_quote_reserves=555_000_000)
        stream._dispatch(_curve_notification(raw, slot=77, sub_id=1))
        assert first.stats.updates == 1 and first.stats.quotes_from_push == 1
        assert second.stats.updates == 0, (
            "a push for one market was delivered to another")
        assert second.latest() is None
        await stream.aclose()

    asyncio.run(_run())


def test_an_unknown_subscription_id_is_dropped_by_the_stream():
    async def _run():
        from autopaper.paper_live.native_monitor import SharedSubscriptionStream

        cfg = MonitorConfig(ws_url="wss://example.invalid")
        stream = SharedSubscriptionStream("wss://example.invalid", config=cfg)
        monitor = NativeMarketMonitor(_identity(), config=cfg, stream=stream)
        stream._monitors.append(monitor)
        stream._dispatch(_curve_notification(
            _acct.bonding_curve_bytes(), slot=1, sub_id=999))
        assert monitor.stats.updates == 0
        assert monitor.stats.stream_gaps == 0, (
            "a post-reconnect straggler was counted as a gap")
        await stream.aclose()

    asyncio.run(_run())


def test_detaching_one_market_leaves_the_others_subscribed():
    async def _run():
        from autopaper.paper_live.native_monitor import SharedSubscriptionStream

        cfg = MonitorConfig(ws_url="wss://example.invalid")
        stream = SharedSubscriptionStream("wss://example.invalid", config=cfg)
        first = NativeMarketMonitor(_identity(), config=cfg, stream=stream)
        other_curve, _ = pd.derive_bonding_curve(MINT)
        second = NativeMarketMonitor(
            StaticVenueIdentity(
                mint="9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin",
                market_type=MarketKind.PUMP_BONDING_CURVE.value,
                market_address=other_curve, token_decimals=6),
            config=cfg, stream=stream)
        stream._routes = {
            1: (first, CURVE_ADDRESS, "CURVE"),
            2: (second, other_curve, "CURVE"),
        }
        stream._monitors.extend([first, second])
        first.stream_resubscribed({1: (CURVE_ADDRESS, "CURVE")})
        second.stream_resubscribed({2: (other_curve, "CURVE")})
        # No live socket, so detach is a local route removal.
        stream._connection = None
        await stream.detach(first)
        assert set(stream._routes) == {2}
        assert set(second._subscriptions) == {2}
        await stream.aclose()

    asyncio.run(_run())


def test_the_registry_shares_one_monitor_per_market():
    """
    The venue account belongs to the MARKET, not to a position. Two positions
    on one mint must share one subscription, or the push rate doubles for the
    same bytes.
    """
    async def _run():
        reg = NativeMonitorRegistry()
        cfg = MonitorConfig(ws_url="")
        first = reg.acquire(_identity(), cfg)
        second = reg.acquire(_identity(), cfg)
        assert first is second, (
            "two positions on one mint each opened their own subscription")
        assert len(reg.monitors()) == 1
        await reg.aclose()

    asyncio.run(_run())


def test_the_registry_releases_a_market_only_when_no_position_holds_it():
    async def _run():
        reg = NativeMonitorRegistry()
        cfg = MonitorConfig(ws_url="")
        monitor = reg.acquire(_identity(), cfg)
        reg.acquire(_identity(), cfg)          # a second holder
        reg.release(monitor)
        assert len(reg.monitors()) == 1, (
            "the market was unsubscribed while a position still held it")
        reg.release(monitor)
        assert len(reg.monitors()) == 0, (
            "the market was never unsubscribed once nothing held it")
        await reg.aclose()

    asyncio.run(_run())


# ===========================================================================
# 5. notification envelope parsing
# ===========================================================================


def test_the_account_bytes_are_read_from_the_value_envelope():
    raw = _acct.bonding_curve_bytes()
    payload = {"context": {"slot": 5}, "value": {
        "data": [base64.b64encode(raw).decode(), "base64"]}}
    assert _base64_account(payload) == raw


def test_a_nested_account_envelope_is_also_accepted():
    raw = _acct.bonding_curve_bytes()
    payload = {"context": {"slot": 5}, "value": {"account": {
        "data": [base64.b64encode(raw).decode(), "base64"]}}}
    assert _base64_account(payload) == raw


def test_an_empty_envelope_yields_no_bytes_rather_than_a_wrong_price():
    """
    The failure this guards: reading the wrong key gives ZERO bytes, which
    decodes as a curve with zero reserves rather than as a parse failure.
    """
    assert _base64_account({"value": {}}) == b""
    assert _base64_account({}) == b""
    assert _base64_account({"value": {"data": []}}) == b""


def test_the_slot_is_read_from_the_context():
    assert _notification_slot({"context": {"slot": 42}}) == 42
    assert _notification_slot({}) is None


# ===========================================================================
# 6. the quote proximity band
# ===========================================================================


def _pos(profile: str = "L25_CURRENT") -> SimpleNamespace:
    return SimpleNamespace(strategy_profile=profile)


def test_the_band_watches_only_thresholds_the_profile_can_reach():
    current = {n for n, _ in existing_price_triggers(_pos(), ExitConfig())}
    assert "fast_failure_stop" in current and "disaster_cap" in current
    assert "tp100" not in current, "L25_CURRENT cannot reach TP100"

    tp100 = {n for n, _ in existing_price_triggers(
        _pos("L25_TP100"), ExitConfig())}
    assert "tp100" in tp100

    v2 = {n for n, _ in existing_price_triggers(_pos("V2_FLOOR"), ExitConfig())}
    assert "v2_take_profit" in v2 and "v2_stop_loss" in v2


def test_the_band_reads_its_thresholds_from_the_live_exit_config():
    """A configured stop must move the band, or the band describes a fiction."""
    moved = ExitConfig(fast_failure_stop_pct=-30.0)
    verdict = evaluate(_pos(), -26.0, cfg=moved)
    assert verdict.band == MONITOR_ONLY, (
        "-26% is 4 points from -30%, beyond a 3 point band")
    assert evaluate(_pos(), -28.0, cfg=moved).band in (
        QUOTE_ALLOWED, QUOTE_REQUIRED)


def test_far_from_a_trigger_no_quote_is_permitted():
    verdict = evaluate(_pos(), 0.0, cfg=ExitConfig())
    assert verdict.band == MONITOR_ONLY
    assert verdict.permits_quote is False
    assert verdict.requires_quote is False


def test_at_a_trigger_a_quote_is_required():
    verdict = evaluate(_pos(), -12.0, cfg=ExitConfig())
    assert verdict.band == QUOTE_REQUIRED
    assert verdict.permits_quote is True
    assert verdict.nearest_trigger == "fast_failure_stop"


def test_near_a_trigger_a_quote_is_allowed():
    verdict = evaluate(_pos(), -8.0, cfg=ExitConfig())
    assert verdict.band == MONITOR_ONLY, "4 points away is beyond a 3 point band"
    verdict = evaluate(_pos(), -9.5, cfg=ExitConfig())
    assert verdict.band == QUOTE_ALLOWED
    assert verdict.permits_quote is True
    assert verdict.requires_quote is False


def test_a_take_profit_is_measured_upward_not_as_an_absolute_distance():
    """
    A stop fires DOWN and a take-profit fires UP. Measuring both as |distance|
    would put a position at +50% "near" a -45% disaster cap, which is nonsense,
    and would spend a quote on almost every profitable position.
    """
    verdict = evaluate(_pos("L25_TP100"), 50.0, cfg=ExitConfig())
    assert verdict.band == MONITOR_ONLY, (
        "a position at +50% is not near a -45% cap or a +100% take-profit")
    verdict = evaluate(_pos("L25_TP100"), 99.0, cfg=ExitConfig())
    assert verdict.band == QUOTE_ALLOWED
    assert verdict.nearest_trigger == "tp100"


def test_an_unprovable_reference_requests_a_quote_rather_than_assuming_safety():
    """
    A position whose price cannot be trusted is exactly the case where the
    existing fail-closed data-integrity path may act. Failing closed here must
    not mean "spend nothing and hope".
    """
    verdict = evaluate(_pos(), None, cfg=ExitConfig())
    assert verdict.band == QUOTE_REQUIRED


def test_a_reached_max_hold_always_requires_a_quote():
    profile = SimpleNamespace(strategy_profile="L25_MAX_HOLD")
    assert evaluate(profile, 0.0, cfg=ExitConfig(), held_s=10.0).band \
        == MONITOR_ONLY
    assert evaluate(profile, 0.0, cfg=ExitConfig(), held_s=60.0).band \
        == QUOTE_REQUIRED


def test_disabling_the_band_restores_conditional_quoting():
    verdict = evaluate(_pos(), 0.0, cfg=ExitConfig(),
                       policy=ProximityPolicy(enabled=False))
    assert verdict.band == QUOTE_ALLOWED
    assert verdict.permits_quote is True


def test_an_absurd_band_is_clamped_rather_than_trusted():
    """
    ``band_pct=1000`` would silently restore unconditional quoting while
    looking like a tuning choice, so it is clamped to the declared maximum.
    """
    assert ProximityPolicy(band_pct=1000.0).effective_band_pct == \
        ProximityPolicy().max_band_pct
    assert ProximityPolicy(band_pct=-5.0).effective_band_pct == 0.0


def test_the_band_declares_no_threshold_of_its_own():
    """
    Every watched number is an EXISTING exit threshold. If this module ever
    grows a constant that is not read from ``ExitConfig``, the band would be a
    strategy change wearing a resource knob's clothes.
    """
    from autopaper.strategy_profile import (
        V2F_STOP_LOSS_PCT, V2F_TAKE_PROFIT_PCT,
    )

    from autopaper.paper_live.quote_proximity import TP100_PCT

    cfg = ExitConfig()
    watched = {t for _, t in existing_price_triggers(
        _pos("V2_FLOOR"), cfg)}
    declared = {cfg.fast_failure_stop_pct, cfg.disaster_cap_pct, TP100_PCT,
                V2F_TAKE_PROFIT_PCT, V2F_STOP_LOSS_PCT}
    assert watched <= declared, watched - declared


# ===========================================================================
# 7. the websocket transport
# ===========================================================================


def test_a_client_frame_is_masked_as_the_rfc_requires():
    frame = encode_frame(b'{"jsonrpc":"2.0"}')
    assert frame[0] & 0x80, "FIN bit must be set"
    assert frame[1] & 0x80, "client-to-server frames MUST be masked"
    mask = frame[2:6]
    payload = frame[6:]
    decoded = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    assert decoded == b'{"jsonrpc":"2.0"}'


@pytest.mark.parametrize("size", [0, 5, 125, 126, 127, 300, 70_000])
def test_frame_length_is_encoded_for_every_size_class(size):
    frame = encode_frame(b"x" * size)
    assert frame[0] == 0x81, "opcode must be text and FIN set"
    masked = frame[1] & 0x80
    length = frame[1] & 0x7F
    if size < 126:
        assert masked and length == size
    elif size < 65_536:
        assert length == 126 and int.from_bytes(frame[2:4], "big") == size
    else:
        assert length == 127 and int.from_bytes(frame[2:10], "big") == size


def test_a_control_frame_may_not_carry_a_large_payload():
    with pytest.raises(ValueError):
        encode_frame(b"x" * 200, opcode=0x9)


def test_only_a_secure_or_local_url_is_accepted():
    parsed = parse_ws_url("wss://api.mainnet.solana.com")
    assert parsed.host == "api.mainnet.solana.com" and parsed.port == 443
    assert parsed.request_target == "/"
    assert parse_ws_url("wss://host/path?x=1").request_target == "/path?x=1"
    for bad in ("http://example.com", "https://example.com", "not a url"):
        with pytest.raises(WebSocketUnavailable):
            parse_ws_url(bad)


def test_the_transport_does_not_import_a_websocket_library():
    """
    The dependency budget is part of the design. ``pyproject`` declares three
    runtime dependencies and none of them is a websocket client, so this module
    must not quietly become the thing that requires one.
    """
    import ast

    source = Path("src/autopaper/same_market/ws.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {"websockets", "websocket", "solana", "solders",
                            "anchorpy", "pylana"}), imported
    pyproject = Path("pyproject.toml").read_text()
    assert "websockets" not in pyproject
    assert "solana-py" not in pyproject


def test_the_transport_never_signs_or_broadcasts():
    """
    A websocket that carries observations must not be able to authorise an
    action. ``sendTransaction`` and ``sendRawTransaction`` are simply not
    anywhere in the monitoring path.
    """
    import ast

    for path in ("src/autopaper/same_market/ws.py",
                 "src/autopaper/paper_live/native_monitor.py",
                 "src/autopaper/paper_live/quote_proximity.py"):
        tree = ast.parse(Path(path).read_text())
        # Any JSON-RPC METHOD NAME is a string literal in the source, so walk
        # the constants rather than the prose: a docstring that SAYS it signs
        # nothing must not fail this, and a real call must.
        methods = {n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        for banned in ("sendTransaction", "sendRawTransaction",
                       "signTransaction", "simulateTransaction",
                       "requestAirdrop", "private_key", "keypair"):
            assert banned not in methods, f"{path} names {banned}"
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "solders" not in imported, f"{path} imports solders"


# ===========================================================================
# 8. the engine seam
# ===========================================================================


def test_the_validation_quote_must_be_fresh_and_same_market():
    """
    The EXISTING freshness and identity gate, applied to the new validation
    quote. Loosening it would let a quote for a different market, or a stale
    one, stand in for executable validation.
    """
    from autopaper.paper_live.engine import PaperLiveEngine

    position = SimpleNamespace(
        entry_pool_address=CURVE_ADDRESS, entry_venue_id="", exit_venue_id="")

    def _quote(**kw):
        base = dict(available=True, identity="PASS", stale=False,
                    quote_age_ms=10.0, max_quote_age_ms=30_000.0,
                    pool_or_curve=CURVE_ADDRESS, missing_reason=None)
        base.update(kw)
        return SimpleNamespace(**base)

    ok, _ = PaperLiveEngine._quote_is_fresh_and_same_market(_quote(), position)
    assert ok is True

    for broken, expect in (
        (_quote(available=False), "unavailable"),
        (_quote(identity="FAIL"), "identity"),
        (_quote(identity="UNKNOWN"), "identity"),
        (_quote(stale=True), "freshness"),
        (_quote(quote_age_ms=99_999.0), "freshness"),
        (_quote(pool_or_curve="9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin"),
         "pinned market"),
    ):
        ok, why = PaperLiveEngine._quote_is_fresh_and_same_market(
            broken, position)
        assert ok is False and expect in why, (expect, why)


def test_the_entry_fill_price_is_read_for_the_return_basis():
    from autopaper.paper_live.engine import PaperLiveEngine
    position = SimpleNamespace(entry_fill=SimpleNamespace(
        effective_price_usd=2.0, price_usd=1.0))
    assert PaperLiveEngine._entry_fill_usd(position) == 2.0
    position = SimpleNamespace(entry_fill=SimpleNamespace(
        effective_price_usd=None, price_usd=1.5))
    assert PaperLiveEngine._entry_fill_usd(position) == 1.5
    assert PaperLiveEngine._entry_fill_usd(
        SimpleNamespace(entry_fill=None)) == 0.0


def test_monitoring_is_disabled_unless_a_websocket_endpoint_is_configured():
    """
    An operator with no push endpoint must get the previous behaviour, not a
    monitor that quietly never connects and silently reports no price.
    """
    from tests.paper_bridge_fakes import bridge

    engine = bridge()
    assert engine.native_monitor_enabled is False
    assert engine._ensure_native_monitor(
        SimpleNamespace(mint="m", token_decimals=6, entry_pool_address=None,
                        entry_venue_id="", cached_identity=None),
        SimpleNamespace(market_type="PUMP_BONDING_CURVE", pool_or_curve=""),
    ) is None


def test_the_engine_exposes_the_quote_rate_per_position():
    """The counter that proves the quote side collapsed while monitoring did not."""
    from tests.paper_bridge_fakes import bridge

    engine = bridge()
    position = SimpleNamespace(signal_id="sig", arm="L25_CURRENT",
                               quote_requests_last_minute=0,
                               quote_request_reason="")
    for _ in range(3):
        engine._note_quote_request(position, "QUOTE_REQUIRED")
    assert position.quote_requests_last_minute == 3
    assert position.quote_request_reason == "QUOTE_REQUIRED"
    assert ("sig", "L25_CURRENT") in engine._quote_request_times


def test_the_quote_rate_window_expires():
    """A per-minute rate must actually be a rate."""
    from tests.paper_bridge_fakes import bridge
    import time as _time

    engine = bridge()
    position = SimpleNamespace(signal_id="sig", arm="L25_CURRENT",
                               quote_requests_last_minute=0,
                               quote_request_reason="")
    engine._note_quote_request(position, "x")
    stamps = engine._quote_request_times[("sig", "L25_CURRENT")]
    stamps[0] = _time.monotonic() - 120.0
    engine._note_quote_request(position, "y")
    assert position.quote_requests_last_minute == 1, (
        "a quote from two minutes ago is still counted in the last minute")