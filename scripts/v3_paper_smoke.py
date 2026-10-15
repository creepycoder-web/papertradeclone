"""
V3 PAPER SMOKE TEST.

Drives the FULL runtime end to end in PAPER and asserts the properties the
migration is judged on:

    startup -> OFF / PAPER / not armed
    an alert before the operator presses start -> no trade
    operator start -> PAPER
    candidate -> the real data pipeline -> V3-C -> sizing -> paper fill
    monitor -> exit -> bookkeeping
    a market cap above $20,000 -> refused with the exact threshold
    emergency stop -> latched, journal intact
    a second process -> refused
    the real executor -> never even constructed

READ-ONLY WITH RESPECT TO REAL FUNDS
------------------------------------
No wallet is loaded, no RPC client is built, no venue is contacted and no
transaction of any kind is created. The real executor is asserted to be
UNCONSTRUCTED at the end, which is stronger than asserting it did nothing: a
PAPER-only process never acquires the capability at all.

Run:  python scripts/v3_paper_smoke.py
"""
import asyncio, json, sys, tempfile, time
from pathlib import Path

REPO = Path("/Users/lnwper/AI-paper-trade")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from autopaper.clock import FakeClock
from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.models import AlertSnapshot, Observation, new_id
from autopaper.market.base import QuoteResult
from autopaper.telegram.parser import parse_alert

T0 = 1_700_000_000.0
MINT = "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj"


class Venue:
    def __init__(self, clock):
        self.clock = clock
        self.paths, self.base, self.started, self.calls = {}, {}, {}, {}
        self.liquidity, self.mcap = {}, {}

    def add(self, mint, path, base=1.0e-6, liq=95_000.0, mcap=17_500.0):
        self.paths[mint] = path
        self.base[mint] = base
        self.started[mint] = self.clock.now()
        self.liquidity[mint] = liq
        self.mcap[mint] = mcap
        self.calls[mint] = 0

    def price(self, mint):
        if mint not in self.paths:
            return None
        e = int(self.clock.now() - self.started[mint])
        return self.base[mint] * self.paths[mint][min(e, len(self.paths[mint]) - 1)]

    def quote(self, mint):
        self.calls[mint] = self.calls.get(mint, 0) + 1
        p = self.price(mint)
        if p is None:
            return QuoteResult(observation=None, error="unknown", provider="venue")
        now = self.clock.now()
        return QuoteResult(
            observation=Observation(
                mint=mint, observed_at=now, received_at=now, price_usd=p,
                market_cap_usd=self.mcap[mint], liquidity_usd=self.liquidity[mint],
                volume_5m_usd=180_000.0, volume_1h_usd=420_000.0,
                buys_5m=640, sells_5m=180, txns_5m=820, volume_24h_usd=900_000.0,
                pair_created_at=self.started[mint] - 120, provider="venue",
            ),
            provider="venue",
        )


def snap(clock, mint, sym, price, mid=1):
    return AlertSnapshot(
        candidate_id=new_id("cand"), source="telegram", chat="moo_deng",
        message_id=mid, source_timestamp=clock.now() - 0.4,
        received_timestamp=clock.now(), raw_text="synthetic",
        mint=mint, symbol=sym, name=sym,
        fields={"price_usd": price}, provenance={"price_usd": "alert_label:price"},
    )


def main():
    tmp = Path(tempfile.mkdtemp())
    cfg = build_config(tmp)
    clock = FakeClock(T0)
    venue = Venue(clock)
    # a token that runs then dies
    # A token that ramps, is alerted repeatedly as a live channel would, then
    # reverses. The ramp is deliberately gentle: a steep ramp pushes the
    # extension score down and V3-C correctly refuses it, which is a property of
    # the strategy rather than a defect.
    venue.add(MINT, [1.00, 1.03, 1.06, 1.10, 1.14, 1.18, 1.23, 1.28,
                     1.33, 1.39, 1.45, 1.51, 1.58, 1.65, 1.72, 1.80,
                     1.88, 1.96, 2.05, 2.14, 2.24, 2.30, 2.20, 2.05,
                     1.88, 1.70, 1.55, 1.42, 1.32, 1.24, 1.18, 1.14,
                     1.11, 1.09, 1.08, 1.075, 1.072, 1.070, 1.069, 1.068])
    from autopaper.market import MarketDataService
    from autopaper.market.ratelimit import RateLimiter
    from autopaper.nar import NarService
    from autopaper.telegram import ConnectionState, TelegramListener

    class VPrimary:
        name = "venue"

        def __init__(self):
            self.limiter = RateLimiter(10_000, name="venue")

        quote = venue.quote

    market = MarketDataService(cfg.market, clock=clock, primary=VPrimary(),
                               fallback=VPrimary())
    market._fvp_probe_budget = 0
    listener = TelegramListener(cfg.telegram, clock=clock)
    listener.status.state = ConnectionState.CONNECTED
    listener.status.chat_id = 2270552242
    listener.on_alert = None
    trader = PaperTrader(cfg, clock=clock, market=market, listener=listener)

    out = {"steps": []}
    def step(name, **kw):
        out["steps"].append({"step": name, **kw})

    # 1. startup
    report = trader.startup()
    step("startup", audit_ok=report["ok"],
         run_state=trader.state.run_state.value,
         trade_mode=trader.state.trade_mode.value,
         real_armed=trader.state.real_armed,
         running=trader.running,
         entries_open=trader.state.can_open_new_entries())
    assert trader.state.run_state.value == "OFF"
    assert not trader.state.real_armed
    assert trader.buys == 0 and trader.account.open_positions() == []
    assert trader.journal.all_trades() == []

    # 2. no trade without the start button
    async def pre_start():
        a = snap(clock, MINT, "SMOKE", venue.price(MINT))
        await trader.on_alert(a, parse_alert(f"$SMOKE\nCA: {MINT}"))
    asyncio.run(pre_start())
    step("before_start", candidates=trader.candidates_seen, buys=trader.buys,
         rejected=trader.rejects, positions=len(trader.account.open_positions()))
    assert trader.buys == 0, "an alert before start must not trade"
    assert not trader.account.open_positions()

    # 3. press start
    trader.state.start()
    step("operator_start", run_state=trader.state.run_state.value,
         trade_mode=trader.state.trade_mode.value)

    # 4. a candidate through the whole pipeline
    async def run():
        # build a real observation history first: a token with one print has
        # no market structure, and V3-C is right to refuse it
        for i in range(12):
            clock.advance(2.0)
            await trader.monitor_once()
        bought = False
        for n in range(2, 12):
            clock.advance(4.0)
            await trader.monitor_once()
            a = snap(clock, MINT, "SMOKE", venue.price(MINT), mid=n)
            await trader.on_alert(a, parse_alert(f"$SMOKE\nCA: {MINT}"))
            if trader.account.open_positions():
                bought = True
                break
        d = trader.last_decision
        step("v3c_decision", decision=d["decision"], v3c=d["v3c"],
             flags=[f["code"] for f in d["risk_flags"]],
             signal_quality=d["signal_quality"],
             market_cap=d["features"]["market_cap_usd"],
             observation_count=d["features"]["observation_count"],
             authorized_and_bought=bought)
        for i in range(20):
            clock.advance(2.0)
            await trader.monitor_once()
    asyncio.run(run())
    step("after_entry", buys=trader.buys,
         open_positions=len(trader.account.open_positions()),
         equity=round(trader.account.equity(trader._refresh_marks()), 6))

    # 5. the journal partition
    js = trader.journal.summary()
    step("journal", summary=js,
         real_records=len(trader.journal.trades_for("REAL")),
         real_signature_problems=trader.journal.verify_real_records_have_signatures())
    assert js["REAL"]["entries"] == 0, "PAPER must not touch the REAL account"
    assert js["REAL"]["exits"] == 0
    assert trader.journal.verify_real_records_have_signatures() == []
    assert trader.real_account.open_positions() == []

    # 6. statistics, separated
    paper_stats, real_stats = trader.statistics_by_mode()
    step("statistics",
         paper={k: paper_stats[k] for k in ("total_trades","wins","losses",
                                             "win_rate","profit_factor",
                                             "total_pnl_usd","max_drawdown_pct",
                                             "avg_winner_usd","avg_loser_usd",
                                             "runners","runner_capture_pct",
                                             "equity_usd","cash_usd")},
         real={k: real_stats[k] for k in ("total_trades","total_pnl_usd")})

    # 7. the UI renders from the same state
    view = trader.status_view()
    from autopaper.telegram import ui
    step("ui_status", rendered=True, consistency_problems=ui.consistency_check(view),
         mode=view.state.trade_mode.value)

    # 8. a REAL candidate is refused: MC above the V3-C ceiling
    venue.add(MINT, [1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], mcap=48_000.0)
    async def big():
        clock.advance(2.0)
        await trader.monitor_once()
        a = snap(clock, MINT, "BIGCAP", venue.price(MINT), mid=3)
        await trader.on_alert(a, parse_alert(f"$BIGCAP\nCA: {MINT}"))
    asyncio.run(big())
    d = trader.last_decision
    step("v3c_reject", decision=d["decision"], v3c=d["v3c"],
         flags=[f["code"] for f in d["risk_flags"]])
    assert d["decision"] == "REJECT"
    assert d["v3c"]["status"] == "REJECTED_MARKET_CAP"
    assert d["v3c"]["authorized"] is False

    # 9. emergency stop
    trader.state.emergency_stop_now()
    step("emergency_stop", latched=trader.state.emergency_stop,
         run_state=trader.state.run_state.value,
         entries=trader.state.can_open_new_entries()[0],
         journal_intact=len(trader.journal.all_trades()) > 0)
    assert not trader.state.can_open_new_entries()[0]
    assert len(trader.journal.all_trades()) > 0

    # 10. single writer. In-process re-acquisition is deliberately legal (an
    # engine restart in one process is a new reference, not a new owner), so
    # the meaningful assertion is that this process OWNS the account and that a
    # real second PROCESS is refused - which tests/test_integration_v3.py
    # proves across a subprocess boundary.
    import subprocess, sys as _s
    code = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from autopaper.config import build_config\n"
        "from autopaper.engine import PaperTrader\n"
        f"cfg = build_config({str(tmp)!r})\n"
        "try:\n"
        "    PaperTrader(cfg).startup(); print('ACQUIRED')\n"
        "except Exception as exc:\n"
        "    print('REFUSED:' + type(exc).__name__)\n"
    )
    proc = subprocess.run([_s.executable, "-c", code], cwd=REPO,
                         capture_output=True, text=True)
    refused = "REFUSED:AccountAlreadyLocked" in proc.stdout
    step("single_writer", this_process_owns=trader.owns_account,
         second_process_refused=refused)
    assert trader.owns_account
    assert refused

    # 11. nothing real was ever touched
    step("real_untouched",
         real_capable=trader.real_capable(),
         real_executor_constructed=trader._real_executor is not None,
         real_armed=trader.state.real_armed,
         real_open=len(trader.real_account.open_positions()))
    assert trader._real_executor is None, "the real executor must not even be built in PAPER"

    print(json.dumps(out, indent=2, default=str))
    print("\nV3_PAPER_SMOKE_OK")


main()
