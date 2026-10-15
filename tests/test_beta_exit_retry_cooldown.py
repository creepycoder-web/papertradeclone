"""
V2 FLOOR (BETA) EXIT-RETRY COOLDOWN.

THE DEFECT
----------
The beta exit had NO retry policy. An unresolved arm re-quoted on EVERY monitor
tick -- one full three-leg sweep (aggregator + bonding curve + PumpSwap) per arm,
every ~5 seconds, for as long as the position stayed outstanding. The stranded
live experiment logged **10,445 SELL-quote attempts in ten hours**, and the
throttling those attempts caused is itself one of the two refusal reasons in its
log: a quote that fails because the shared RPC/aggregator budget is spent is a
CAPACITY failure, and a retry loop that spends the budget cannot recover from it.

WHAT IS PINNED HERE
-------------------
1. The PRODUCTION default defers a retry: a second tick inside the cooldown
   issues NO quote request and does NOT advance ``exit_attempts`` (so the attempt
   counter keeps meaning "real attempts", which is what the incident-recovery
   classification is derived from).
2. The cooldown is a BOUNDED exponential curve, and it is bounded.
3. The retry DOES happen once the cooldown expires, and a recovered quote settles
   the arm -- the arm is never abandoned.
4. A settlement CLEARS the cooldown, so the next position on that arm is not
   penalised for a previous one.
5. The cooldown suppresses quote requests without suppressing the EXIT BLOCKED
   notification, and the CLOSED notification is a DIFFERENT message.

NOTHING here settles from a mark, invents a price or weakens identity: this file
is about WHEN the engine is allowed to ask, not about what it accepts.

PAPER only. No network, no signer, no transaction.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from autopaper.paper_live.engine import (  # noqa: E402
    BETA_EXIT_RETRY_BASE_S,
    BETA_EXIT_RETRY_CAP_S,
)

from test_paper_live import FakeQuoteService, run, sell_point  # noqa: E402
from test_v2_floor_beta import beta_engine, eligible_signal  # noqa: E402

pytestmark = [pytest.mark.paper_live, pytest.mark.strategy]


class _RefusingQuotes(FakeQuoteService):
    """Every SELL answers an UNAVAILABLE quote with a named reason."""

    def __init__(self, reason: str = "no executable SELL route right now"):
        super().__init__(sell=sell_point(available=False, missing_reason=reason))
        self.available = False

    async def quote_sell(self, **kw):
        self.sell_calls.append(kw)
        if not self.available:
            return sell_point(available=False,
                              missing_reason="no executable SELL route right now")
        return self._restamp(sell_point(), kw["mint"])


def _armed(tmp_path, quotes=None, **kwargs):
    """A beta engine carrying ONE exit-due arm on arm A."""
    eng = beta_engine(tmp_path, quotes=quotes or _RefusingQuotes(), **kwargs)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    return eng, exp, pid


def _arm(exp, pid, arm="A"):
    return exp.positions[pid][arm]


def _sell_calls(quotes) -> int:
    return len(quotes.sell_calls)


# ---------------------------------------------------------------------------
# 1. the default cooldown defers the retry
# ---------------------------------------------------------------------------


def test_the_default_cooldown_defers_the_second_attempt(tmp_path):
    """One refusal, then a second tick inside the cooldown: NO new quote."""
    quotes = _RefusingQuotes()
    eng, exp, pid = _armed(tmp_path, quotes)
    run(eng._monitor_beta_experiment())
    pos = _arm(exp, pid)
    assert pos.status == "unresolved"
    assert pos.exit_attempts == 1
    first_calls = _sell_calls(quotes)
    assert first_calls >= 1

    # THE SAME TICK AGAIN, immediately. The production cooldown is seconds long,
    # so this tick must NOT spend another three-leg quote sweep.
    run(eng._monitor_beta_experiment())
    assert _sell_calls(quotes) == first_calls, (
        "a retry inside the cooldown issued a fresh quote sweep: the exit loop "
        "is hammering the provider again")
    assert pos.exit_attempts == 1, (
        "a suppressed retry must not advance the attempt counter, or the "
        "incident-recovery classification (exit_attempts > 0) is no longer "
        "derived from real attempts")
    assert pos.status == "unresolved"
    assert eng._monitor_diag["BETA_EXIT_RETRY_DEFERRED"] >= 1


def test_the_default_is_the_production_policy(tmp_path):
    """The engine's out-of-the-box bounds ARE the declared production policy."""
    eng, _exp, _pid = _armed(tmp_path)
    assert eng.beta_exit_retry_base_s == BETA_EXIT_RETRY_BASE_S
    assert eng.beta_exit_retry_cap_s == BETA_EXIT_RETRY_CAP_S
    assert BETA_EXIT_RETRY_BASE_S > 0.0, (
        "a zero default would reinstate the per-tick retry storm")


# ---------------------------------------------------------------------------
# 2. the curve is exponential and BOUNDED
# ---------------------------------------------------------------------------


def test_the_cooldown_grows_and_is_bounded(tmp_path):
    eng, _exp, _pid = _armed(tmp_path)
    delays = [eng._beta_exit_backoff_s(n) for n in range(1, 12)]
    assert delays[0] == pytest.approx(BETA_EXIT_RETRY_BASE_S)
    assert delays[1] == pytest.approx(BETA_EXIT_RETRY_BASE_S * 2)
    assert delays[2] == pytest.approx(BETA_EXIT_RETRY_BASE_S * 4)
    assert delays == sorted(delays), "the cooldown must not shrink"
    assert max(delays) == pytest.approx(BETA_EXIT_RETRY_CAP_S), (
        "the cooldown must be BOUNDED; an unbounded curve is a stuck position")


def test_the_scheduled_retry_time_follows_the_curve(tmp_path):
    eng, exp, pid = _armed(tmp_path)
    delays = []
    for _ in range(4):
        before = time.time()
        run(eng._monitor_beta_experiment())
        state = eng._beta_exit_retry[(pid, "A")]
        delays.append(round(state["next_attempt_at"] - before, 3))
        # move the clock forward past the cooldown so the next attempt happens
        eng._beta_exit_retry[(pid, "A")]["next_attempt_at"] = 0.0
    assert delays[0] < delays[1] < delays[2], (
        "consecutive refusals must grow the cooldown: " + str(delays))
    assert _arm(exp, pid).exit_attempts == 4


# ---------------------------------------------------------------------------
# 3. the retry still happens -- and a recovered quote settles the arm
# ---------------------------------------------------------------------------


def test_the_retry_happens_once_the_cooldown_expires(tmp_path):
    quotes = _RefusingQuotes()
    eng, exp, pid = _armed(tmp_path, quotes)
    run(eng._monitor_beta_experiment())
    calls = _sell_calls(quotes)
    # Expire the cooldown the way wall-clock time would.
    eng._beta_exit_retry[(pid, "A")]["next_attempt_at"] = 0.0
    run(eng._monitor_beta_experiment())
    assert _sell_calls(quotes) > calls, (
        "the arm was abandoned instead of retried")
    assert _arm(exp, pid).exit_attempts == 2


def test_a_recovered_quote_settles_the_arm_after_the_cooldown(tmp_path):
    quotes = _RefusingQuotes()
    eng, exp, pid = _armed(tmp_path, quotes)
    run(eng._monitor_beta_experiment())
    pos = _arm(exp, pid)
    assert pos.status == "unresolved" and pos.exit_attempts == 1

    # The provider recovers and the cooldown expires.
    quotes.available = True
    eng._beta_exit_retry[(pid, "A")]["next_attempt_at"] = 0.0
    run(eng._monitor_beta_experiment())
    assert pos.status == "closed", (
        "a recovered quote did not settle the outstanding arm")
    assert pos.exit_basis == "EXECUTABLE_QUOTE_SIMULATED_FILL"
    assert pos.net_pnl_usd is not None


# ---------------------------------------------------------------------------
# 4. a settlement clears the cooldown
# ---------------------------------------------------------------------------


def test_a_settlement_clears_the_cooldown_state(tmp_path):
    quotes = _RefusingQuotes()
    eng, exp, pid = _armed(tmp_path, quotes)
    run(eng._monitor_beta_experiment())
    assert (pid, "A") in eng._beta_exit_retry
    quotes.available = True
    eng._beta_exit_retry[(pid, "A")]["next_attempt_at"] = 0.0
    run(eng._monitor_beta_experiment())
    assert (pid, "A") not in eng._beta_exit_retry, (
        "a settled arm left a stale penalty behind for its next position")


# ---------------------------------------------------------------------------
# 5. the number of requests over many ticks is BOUNDED (the anti-hammering
#    property this whole change exists for)
# ---------------------------------------------------------------------------


def test_many_ticks_inside_the_cooldown_issue_no_further_requests(tmp_path):
    quotes = _RefusingQuotes()
    eng, exp, pid = _armed(tmp_path, quotes)
    run(eng._monitor_beta_experiment())
    calls = _sell_calls(quotes)
    for _ in range(50):
        run(eng._monitor_beta_experiment())
    assert _sell_calls(quotes) == calls, (
        "50 ticks inside one cooldown produced fresh provider requests; this is "
        "the retry storm the cooldown exists to stop")
    pos = _arm(exp, pid)
    assert pos.exit_attempts == 1
    assert pos.status == "unresolved", "the arm must stay outstanding"
    # ONE deferral per arm per tick: the parent signal carries all three arms,
    # so 50 ticks over A/B/C is 150 suppressed retries and still ONE sweep each.
    assert eng._monitor_diag["BETA_EXIT_RETRY_DEFERRED"] == 150


# ---------------------------------------------------------------------------
# 6. the cooldown does not silence the operator
# ---------------------------------------------------------------------------


def test_the_first_refusal_is_announced_even_though_the_retry_is_deferred(
        tmp_path):
    quotes = _RefusingQuotes()
    eng, exp, pid = _armed(tmp_path, quotes)
    run(eng._monitor_beta_experiment())
    msgs = eng.render_beta_exit_blocked()
    assert len(msgs) == 1
    text = msgs[0]["text"]
    assert "EXIT BLOCKED" in text and "UNRESOLVED" in text
    assert "NO SALE WAS BOOKED" in text
    assert "PAPER CLOSED" not in text
    assert "net PAPER P&L" not in text

    # ... and the deferral does not re-announce it on every tick either, once the
    # notice has actually been DELIVERED (the marker is what records that).
    for _pid, arm, attempt in msgs[0]["blocked_arms"]:
        assert exp.mark_exit_block_notified(_pid, arm, attempt)
    for _ in range(5):
        run(eng._monitor_beta_experiment())
    assert eng.render_beta_exit_blocked() == [], (
        "the deferred retries re-announced the same refusal")
    assert _arm(exp, pid).exit_attempts == 1, (
        "a deferred tick is not an attempt")


def test_a_settlement_is_a_different_message_from_a_refusal(tmp_path):
    quotes = _RefusingQuotes()
    eng, exp, pid = _armed(tmp_path, quotes)
    run(eng._monitor_beta_experiment())
    blocked = eng.render_beta_exit_blocked()
    assert blocked and "NO SALE WAS BOOKED" in blocked[0]["text"]
    quotes.available = True
    eng._beta_exit_retry[(pid, "A")]["next_attempt_at"] = 0.0
    settled = run(eng._monitor_beta_experiment())
    closed = [p for p in settled if p.status == "closed"]
    assert closed, "the recovered quote did not settle"
    receipts = eng.render_beta_exits(closed)
    assert len(receipts) == 1
    assert "PAPER CLOSED" in receipts[0]["text"]
    assert "NO SALE WAS BOOKED" not in receipts[0]["text"]
    assert blocked[0]["kind"] != receipts[0]["kind"], (
        "a refusal and a sale must not share a message kind")
    assert blocked[0]["dedupe_key"] != receipts[0]["dedupe_key"]


def test_the_refusal_reason_reaches_the_operator_verbatim(tmp_path):
    """The provider's own words reach the position AND the operator.

    The position carries the CLASSIFIED refusal (it stays unresolved because no
    valid executable SELL quote was obtained for the acquired quantity); the
    message carries the CLASSIFIED refusal plus the provider's own detail, so the
    operator never has to guess between a throttled venue and a dead route.
    """
    quotes = _RefusingQuotes()
    eng, exp, pid = _armed(tmp_path, quotes)
    run(eng._monitor_beta_experiment())
    pos = _arm(exp, pid)
    assert pos.exit_blocked_reason
    assert pos.net_pnl_usd is None and pos.exit_fill_price is None
    assert "no valid executable SELL quote" in pos.exit_blocked_reason
    text = eng.render_beta_exit_blocked()[0]["text"]
    assert "no executable SELL quote" in text
    assert "no executable SELL route right now" in text, (
        "the provider's own refusal did not reach the operator")


def test_a_missing_quote_records_the_classified_failure_reason(tmp_path):
    quotes = _RefusingQuotes()
    eng, exp, pid = _armed(tmp_path, quotes)
    settled = run(eng._monitor_beta_experiment())
    failed = [p for p in settled if p.status == "unresolved"]
    assert failed, "the arm was not reported as unresolved"
    # No fabricated settlement of ANY kind.
    assert exp.completed == []
    assert _arm(exp, pid).net_pnl_usd is None
