"""
V2 FLOOR (BETA) -- three parallel PAPER-only exit arms on the SAME V2 FLOOR entry.

Selecting ``V2 FLOOR (BETA)`` starts an experiment session. Every signal that
passes the EXISTING V2 FLOOR entry gate (:func:`autopaper.v2floor.evaluate`,
unchanged) is routed to THREE independent virtual PAPER arms. The arms share
ONE entry decision and ONE entry instant; they differ in their exit rule -- and,
because each is ALL-IN on its own bankroll, in the size it commits:

    Arm A -- Baseline : TP +30%  SL -30%  max hold  90s   (the researched bracket)
    Arm B -- TP Test   : TP +50%  SL -30%  max hold  90s
    Arm C -- Hold Test : TP +50%  SL -30%  max hold 120s

    A vs B  isolates the take-profit level (+30 vs +50, SL and hold fixed).
    B vs C  isolates the maximum hold   (90s vs 120s, TP and SL fixed).

Each arm owns an ISOLATED $10.00 virtual ledger (three independent accounts,
$30.00 total experiment capital -- never a single shared $10). An exit in one
arm can never close, modify or realize P&L on another arm's position.

SIZING IS ALL-IN. Every new entry commits 100% of THAT arm's uncommitted cash
(:meth:`BetaExperiment.all_in_notional_usd`), so each ledger compounds its own
realized proceeds and an arm that is holding has nothing left to trade with --
which is the same fact as the no-overlapping-positions rule, enforced per arm by
:meth:`BetaExperiment.arm_entry_plan` while the other arms carry on
independently. The 10%-of-equity rule this replaced left 90% of every ledger
idle and could not compound; trades booked under it carry no ``sizing_rule`` and
are reported as LEGACY, never as ALL-IN.

NOT a new entry strategy. The V2 FLOOR gate is used verbatim; this module never
re-derives it, never changes its thresholds, and never touches L25. NOT live:
it builds, signs and broadcasts NOTHING. It routes to the PAPER simulator only
(:meth:`BetaExperiment.audit` runs ``assert_paper_only`` and fails CLOSED if the
paper-only capability audit does not pass).

REUSES, does not reimplement: :class:`VirtualAccount` (the isolated per-arm
ledger), :class:`FillModel` (the PAPER fill model), and
:class:`PersistedState`/:class:`PaperStateStore` (the durable, restart-safe,
append-and-reconcile store).

EXIT FILL SEMANTICS. A take-profit target is a LIMIT, not a fill: a TP books at
the LIMIT price (target), never at a later gapped-up print. A stop books at the
observed trough when one is available (stop-through). A time exit books at the
last observed return. When the path cannot resolve whether TP or SL came first,
the trade is booked CONSERVATIVELY as the stop and FLAGGED
``path_ambiguous``. Every realized return is NET of the modeled exit friction
(the same paper basis the rest of the harness uses).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from autopaper.paper_live.account import VirtualAccount
from autopaper.paper_live.execution_costs import (
    PAPER_PRIORITY_FEE_LAMPORTS,
    OutsideSwapCosts,
    deployable_sol_lamports,
    failed_execution_accounting,
    intended_transaction_config,
    slippage_boundary,
    outside_swap_costs as _outside_swap_costs,
)
from autopaper.paper_live.fills import FillModel
from autopaper.paper_live.guards import assert_paper_only
from autopaper.paper_live.state import PersistedState, PaperStateStore
from autopaper.paper_live.schema import SignalSnapshot
from autopaper.v2floor import V2F_HASH as BETA_BASELINE_HASH, evaluate as v2floor_evaluate

ARM_A = "A"
ARM_B = "B"
ARM_C = "C"
BETA_ARMS = (ARM_A, ARM_B, ARM_C)

#: Fill-provenance labels. Every recorded entry and exit carries exactly one,
#: so a consumer can never mistake a modelled number for an executable quote.
#: ``EXECUTABLE_QUOTE`` means both legs came from the production quote stack.
#: ``MARK_MODEL`` is retained only to label state restored from a session
#: written before quotes were wired in -- it is never produced by new entries.
EXECUTABLE_QUOTE_BASIS = "EXECUTABLE_QUOTE_SIMULATED_FILL"
MARK_MODEL_BASIS = "MARK_MODEL_LEGACY"

#: Fill-provenance label for a HISTORICAL / BACKTEST settlement. It is the
#: modelled limit-or-trough booking (a TP books at the target, a stop at the
#: observed trough, a time exit at the last observed return, net of the
#: ``PAPER_EXIT_MULT`` friction) and it is NOT an executable quote. It is used
#: only when a caller explicitly settles WITHOUT a quote -- i.e. the research /
#: backtest path -- and never by the production engine, which passes
#: ``require_quote=True`` so a missing quote stays UNRESOLVED.
MODELED_BACKTEST_BASIS = "MODELED_LIMIT_OR_TROUGH"

#: Fill-provenance label for a QUOTE-SETTLED settlement. Both legs came from
#: validated executable quotes and the exit books the accepted SELL quote itself.
#: It is written by the quote path ONLY. The previous default left every settled
#: quote trade carrying ``MODELED_LIMIT_OR_TROUGH`` -- a label describing the
#: opposite of how it was booked -- which is fixed here and on the position.
QUOTE_FILL_MODEL = "EXECUTABLE_SELL_QUOTE_SIMULATED_FILL"

# ---------------------------------------------------------------------------
# THE RETURN BASIS -- what "up" and "down" MEAN for a TP/SL threshold
# ---------------------------------------------------------------------------
#
# WHAT WAS WRONG
# ==============
# ``_exit_decision`` fired TP and SL off ``peak_return_pct`` / ``trough_return_pct``,
# which :meth:`BetaExperiment.observe` computed from the AGGREGATOR MARK::
#
#     ret = (mark_price - entry_fill_price) / entry_fill_price * 100
#
# but the P&L was booked from a completely different number -- the accepted
# executable SELL quote, via :class:`BetaExitFill`. Two different books, one rule.
# Measured over the live clean cohort the two disagree by a MEDIAN 29.2% and by
# up to 62.5% on the same token in the same instant, which is larger than either
# the +30..+50% target or the -30% stop. The consequences were not subtle:
#
#     * 18 of 20 trades labelled ``SL`` realised BETTER than the -30% stop
#       (one labelled ``SL`` at +1.00%);
#     * all 3 trades labelled ``TP`` realised WORSE than +30% (one at -22.78%);
#     * ``A`` and ``B`` on one signal booked +$2.114 and -$0.025 off the SAME
#       token, because only the mark-relative rule let A hold.
#
# A TP/SL label therefore described a price the position could not have traded
# at, while the P&L came from the price it actually could. Every hit-rate
# derived from those labels was meaningless.
#
# THE FIX: TWO EXPLICIT BASES, NEVER MIXED
# =======================================
#: The PRODUCTION basis, and the only one that may satisfy TP/SL on a quote
#: path. One number, computed exactly as the settlement computes its own::
#:
#:     net_return_pct = (SELL gross proceeds - SELL fee) / total BUY cost - 1
#:
#: The denominator is the entry's TOTAL cost in the quote asset (swap input plus
#: the quote's own fee plus modelled network cost), so the trigger and the booked
#: P&L are the same arithmetic on the same two receipts. It requires a validated
#: executable SELL quote for the EXACT acquired quantity on a verified market.
RETURN_BASIS_QUOTE = "EXECUTABLE_SELL_QUOTE_NET_VS_ENTRY_COST"
#: The HISTORICAL / BACKTEST basis, unchanged and still used by
#: :meth:`_settle_modeled`. The modelled path has no quotes by construction --
#: that is what ``PAPER_EXIT_MULT`` exists for -- so it samples marks, and its
#: records stay tagged ``MODELED_BACKTEST_BASIS`` and stay out of the forward
#: cohort. Its historical numbers are NOT recomputed and NOT corrected: they
#: were produced by a different model and are preserved exactly as they are.
RETURN_BASIS_MARK = "MARK_OBSERVATION_MODELLED"
#: What the MARK excursion is, on every record including quote-settled ones. A
#: mark return is a DIFFERENT book from the executable one, so it is retained
#: strictly as a diagnostic and can never satisfy a threshold.
RETURN_BASIS_MARK_DIAGNOSTIC = "MARK_DIAGNOSTIC_ONLY"

#: Canonical arm order for every operator-facing rendering, so an entry or exit
#: notification lists A, B, C in the same sequence every time.
BETA_ARMS_ORDER: tuple[str, ...] = BETA_ARMS

#: The operator-facing profile name. The experiment is a PAPER simulation and
#: is labelled as such in every message it produces.
BETA_PROFILE_LABEL = "V2 FLOOR (BETA)"

BETA_STARTING_BANKROLL_USD = 10.0
BETA_LABEL = "PAPER ONLY -- V2 FLOOR (BETA) -- 3 Parallel Exit Arms"
BETA_SELECTOR = "V2_FLOOR_BETA"
BETA_ENTRY_PROFILE = "V2_FLOOR"

#: THE BETA SIZING RULE, and it is ALL-IN.
#:
#: The 10%-of-equity rule this replaced left 90% of every arm's capital idle, so
#: a +30% arm win realized $0.30 against a $10 bankroll and the experiment could
#: not compound. Each arm now commits 100% of its OWN available (uncommitted)
#: cash on every new entry. The percentage is written onto every position and
#: trade record as ``sizing_pct``, so a result can always be traced back to the
#: rule that produced it and a historical 10%-sized trade is never read as an
#: all-in one.
BETA_SIZING_PCT = 100.0
BETA_SIZING_RULE = "ALL_IN_100_PCT_OF_AVAILABLE_CASH"
#: How a position's size was decided. ``ALL_IN_100_PCT`` is the production beta
#: rule; ``EXPLICIT_SIZE`` is the fixture/probe entry point where the caller
#: declares the size; an absent value means the record predates this change and
#: was produced by the removed 10% rule.
SIZING_ALL_IN = "ALL_IN_100_PCT"
SIZING_EXPLICIT = "EXPLICIT_SIZE"
LEGACY_BETA_SIZING_PCT = 10.0
#: An arm whose uncommitted cash is at or below this cannot fund a BUY and is
#: refused rather than opened at a size it cannot pay for.
BETA_MIN_ENTRY_USD = 0.0

#: How an EXIT BLOCKED / UNRESOLVED notification escalates. The FIRST refused
#: exit for an arm is always reported, then every Nth retry -- the same cadence
#: the REAL exit path uses (``_SELL_BLOCK_REPEAT_EVERY``). The attempt number
#: shown is always the TRUE count, so the cadence decides only how often the
#: operator is interrupted, never what the number means. It is one half of the
#: duplicate guard; the other half is ``BetaPosition.exit_notified_attempt``,
#: which is PERSISTED and stops an already-reported attempt from being
#: re-announced by a repeated tick, a retry loop or a process restart.
BETA_EXIT_BLOCK_NOTIFY_EVERY = 10

PAPER_EXIT_MULT = 0.9852


def net_from_gross(gross_frac: float) -> float:
    """Gross mark return (fraction) -> net-of-exit-friction paper return."""
    return (1.0 + gross_frac) * PAPER_EXIT_MULT - 1.0


def record_pnl_basis(rec: "dict[str, Any]") -> str:
    """The accounting basis of one persisted trade record.

    A record written by the executable-quote path says so; one written before
    quote wiring existed (or by the explicit backtest path) is MODELLED. The
    distinction is reported on every surface so a historical modelled number is
    never presented as quote-based performance.
    """
    basis = rec.get("pnl_basis")
    if basis:
        return str(basis)
    if str(rec.get("exit_basis") or "") == EXECUTABLE_QUOTE_BASIS:
        return "V2_FLOOR_BETA_QUOTE_SIMULATED_FILL"
    return "V2_FLOOR_BETA_MODELED_BACKTEST"


#: The two provenance buckets the reporting separates by.
BASIS_QUOTE = "quote"
BASIS_MODELED = "modeled"


def basis_bucket(rec: "dict[str, Any]") -> str:
    """``"quote"`` or ``"modeled"`` for one trade record."""
    return (BASIS_QUOTE if record_pnl_basis(rec).endswith("QUOTE_SIMULATED_FILL")
            else BASIS_MODELED)


#: The three sizing buckets the reporting separates by.
SIZING_BUCKET_ALL_IN = "all_in"
SIZING_BUCKET_EXPLICIT = "explicit"
SIZING_BUCKET_LEGACY = "legacy"


def sizing_bucket(rec: "dict[str, Any]") -> str:
    """
    Which sizing rule produced a recorded trade.

    ``legacy`` is the honest label for a record with no ``sizing_rule``: those
    are the trades booked before this change, at 10% of equity. Reporting an
    all-in total over them would credit the ALL-IN experiment with results the
    old rule produced.
    """
    rule = str(rec.get("sizing_rule") or "")
    # Accept both the short constant form ("ALL_IN_100_PCT") and the full
    # descriptive form ("ALL_IN_100_PCT_OF_AVAILABLE_CASH") so records written
    # under either convention are recognized as all-in.
    if rule == SIZING_ALL_IN or rule == "ALL_IN_100_PCT_OF_AVAILABLE_CASH":
        return SIZING_BUCKET_ALL_IN
    if rule == SIZING_EXPLICIT:
        return SIZING_BUCKET_EXPLICIT
    pct = rec.get("sizing_pct")
    if pct is not None:
        try:
            return (SIZING_BUCKET_ALL_IN if float(pct) >= BETA_SIZING_PCT
                    else SIZING_BUCKET_LEGACY)
        except (TypeError, ValueError):
            return SIZING_BUCKET_LEGACY
    return SIZING_BUCKET_LEGACY


#: The cost model a record was booked under.
COST_MODEL_MODELED = "outside_swap_costs_modeled"
COST_MODEL_LEGACY = "legacy_swap_quote_only"


def cost_model_bucket(rec: "dict[str, Any]") -> str:
    """Whether a record was costed OUTSIDE the swap quote, or only within it.

    ``legacy`` is the honest label for a record with no outside-swap cost model:
    it was charged the swap quote's own fee and NOTHING for the Solana base fee,
    the priority fee or the token-account rent. Reporting a single total over
    such records would present them as fully costed, so the split is always
    carried alongside.
    """
    costs = rec.get("outside_swap_costs")
    if isinstance(costs, Mapping) and costs.get("model") == \
            "OUTSIDE_SWAP_COSTS_MODELED":
        return COST_MODEL_MODELED
    return COST_MODEL_LEGACY


# ===========================================================================
# THE FORWARD EXPERIMENT: an ISOLATED store for NEW ALL-IN quote-based results
# ===========================================================================
#
# WHY A SECOND FILE AND NOT A FILTER
# ---------------------------------
# The main store already carries 51 HISTORICAL MODELLED trades (17 parents x 3
# arms) booked under the removed 10%-of-equity rule and priced off the observed
# MARK. They are the experiment's provenance and must stay exactly as they are.
# They must also never enter a forward performance statistic, because a modelled
# mark-based result and a quote-based one are different accounting bases.
#
# Separating them by a REPORTING FILTER would be one edit away from being wrong:
# a later change to the filter, or a sum taken somewhere that forgets it, would
# silently mix the two. So the forward results get their OWN file, and every
# record in it is eligible BY CONSTRUCTION -- the mirror below is the only
# writer, and it refuses anything that is not both QUOTE-based and ALL-IN.
#
# NOTHING IS MOVED, REWRITTEN OR DELETED. The main store is untouched by this
# path, the historical trades stay where they are, and the forward file starts
# empty and accumulates only new results.
#
#: The forward experiment's own durable store, a sibling of the main one.
FORWARD_STATE_FILENAME = "v2_floor_beta_forward_state.json"

#: The forward sample target, in UNIQUE ELIGIBLE PARENT SIGNALS. A parent signal
#: contributes ONE sample however many of its three arms settle, because A/B/C
#: are three exit rules measured on one entry, not three independent trades.
FORWARD_TARGET_PARENT_SIGNALS = 50

FORWARD_LABEL = "PAPER ONLY -- V2 FLOOR (BETA) FORWARD -- ALL-IN quote-based"

#: The eligibility rule, stated once and written into the store so a reader of
#: the file alone can tell what belongs in it.
FORWARD_ELIGIBILITY = (
    "pnl_basis == V2_FLOOR_BETA_QUOTE_SIMULATED_FILL (basis bucket 'quote') "
    "AND sizing_rule == ALL_IN_100_PCT_OF_AVAILABLE_CASH (sizing bucket "
    "'all_in')")
FORWARD_EXCLUDES = (
    "every MODELLED / mark-priced record and every legacy 10%-of-equity "
    "record, including the 51 historical modelled trades held in the main "
    "v2_floor_beta_state.json store; those stay in place and are never copied, "
    "summed or re-based here")

#: The cohort label for a settlement that DEPENDS ON AN INCIDENT, not on the
#: exit rule. See ``_restore_from_state`` and ``_persist_forward``.
INCIDENT_RECOVERY_COHORT = "INCIDENT_RECOVERY"

#: The clean cohort, and the ONLY label a forward result may carry.
FORWARD_COHORT = "FORWARD"

INCIDENT_RECOVERY_EXCLUSION = (
    "a DELAYED settlement: the exit was REFUSED at least once before it "
    "executed (exit_attempts > 0), so it did not happen when the exit rule "
    "fired and its P&L carries the delay. The incident that produced the "
    "longest of them -- the pinned-SELL pair-unwrapping defect, which refused "
    "every pinned SELL quote -- is recorded per record in 'incident_recovery', "
    "with the attempt count, the refusal and, when the arm was still "
    "unresolved across a restart, the attempt count the previous process left "
    "behind. These settlements are PRESERVED in full (they are in the main "
    "store, in the ledger and in this file's incident_recovery section) and are "
    "EXCLUDED from every clean forward figure")


def _session_id_of(payload: "Mapping[str, Any] | None") -> str | None:
    """The session id a PERSISTED state file was written under, or ``None``.

    Read from the records rather than from memory, because the id that matters
    for an audit is the one an already-written file carries -- the in-memory
    ``session_id`` of the process doing the reading is a different thing, and
    after a restart it IS a different value.

    The sources are tried in order of authority:

    1. the LAST trade record's own ``session_id`` -- the run that last booked a
       result into this file;
    2. the last ``beta session <id>`` note, which every :meth:`_persist` writes
       and which therefore names the run that LAST WROTE this file;
    3. an incident-recovery record's ``session_id``.

    WHY THE INCIDENT RECORDS ARE LAST, AND NOT SECOND
    =================================================
    Their ``session_id`` deliberately names the session that PRODUCED the
    incident, which after a rollover is the PREVIOUS one -- that is the whole
    point of the field, and it is what makes a carried position traceable back
    to the experiment it belonged to. Reading it before the save note therefore
    answered the wrong question: on a freshly rolled session with no trades yet,
    it reported the PRIOR session id as the current one, so a SECOND rollover
    would name the session it was ending as the one before that. A rollover's
    ``prior_session_id`` is exactly what this function exists to answer
    correctly, so the save note has to outrank a historical provenance field.
    """
    if not isinstance(payload, Mapping):
        return None
    for rec in reversed(list(payload.get("trades") or ())):
        if isinstance(rec, Mapping) and rec.get("session_id"):
            return str(rec["session_id"])
    for note in reversed(list(payload.get("write_log") or ())):
        text = str((note or {}).get("note") or "")
        if text.startswith("beta session "):
            return text[len("beta session "):].strip() or None
    for rec in reversed(list(payload.get("incident_recovery") or ())):
        if isinstance(rec, Mapping) and rec.get("session_id"):
            return str(rec["session_id"])
    return None


def _next_session_id(prior: "str | None", *, prefix: str = "beta") -> str:
    """A session id for a NEW session: distinct from ``prior``, and ORDERED.

    ``session_id`` is a SECOND-RESOLUTION stamp (``beta_<epoch>``), so a session
    boundary that lands inside the same second as the one it replaces would give
    two different sessions the SAME label -- and every figure, every trade record
    and every incident would then be attributable to either of them. That is the
    same defect the forward store's id already guards against, with the same fix:
    advance the stamp to the next second rather than randomising it, so the ids
    stay ordered and comparable (a hash would destroy that ordering).
    """
    stamp = f"{prefix}_{int(time.time())}"
    if not prior or stamp != prior:
        return stamp
    try:
        return f"{prefix}_{int(str(prior).split('_', 1)[1]) + 1}"
    except (IndexError, ValueError):  # pragma: no cover - unreachable for a stamp
        return stamp


def _is_incident_recovery(rec: "Mapping[str, Any] | None") -> bool:
    """Was this settlement DELAYED by an incident rather than produced on time?

    ONE predicate, one definition, used by the mirror and by the counter, so a
    record cannot be excluded from one figure and counted in another.

    It is DERIVED, not taken on trust: the label is read when it is present, and
    otherwise the record's own ``exit_attempts`` decides. That is what makes the
    classification survive records written before this rule existed -- the three
    LARP arms the pinned-SELL defect stranded carry ``exit_attempts`` 1374/1374/
    1368 on their settled records, so they are identified as delayed without
    anybody having to remember what happened to them.

    ``exit_attempts > 0`` means the exit did NOT execute when its rule first
    fired. That is precisely the property the clean cohort must not contain: a
    forward result measures the exit rule, and a rule that is retried is being
    measured with an incident folded into its P&L.
    """
    if not isinstance(rec, Mapping):
        return False
    if str(rec.get("cohort") or "") == INCIDENT_RECOVERY_COHORT:
        return True
    if rec.get("incident_recovery"):
        return True
    return int(rec.get("exit_attempts") or 0) > 0


def forward_eligible(rec: "Mapping[str, Any] | None") -> bool:
    """Is a COMPLETED trade record part of the forward ALL-IN quote-based sample?

    Two independent conditions, and both are required:

    * the P&L came from EXECUTABLE QUOTES on both legs (``basis_bucket ==
      'quote'``), so a modelled mark-priced backtest result can never enter; and
    * the size was decided by the ALL-IN rule (``sizing_bucket == 'all_in'``),
      so a legacy 10%-of-equity record can never enter.

    A record with no ``sizing_rule`` and no ``sizing_pct`` is LEGACY by
    :func:`sizing_bucket` and is therefore refused -- which is exactly what keeps
    the 51 historical trades out.
    """
    if not isinstance(rec, Mapping):
        return False
    return (basis_bucket(rec) == BASIS_QUOTE
            and sizing_bucket(rec) == SIZING_BUCKET_ALL_IN)


def forward_eligible_open(pos: "Mapping[str, Any] | None") -> bool:
    """Is an OUTSTANDING arm part of the forward sample?

    An open position has no exit, so it has no ``pnl_basis`` to bucket. The same
    two conditions are therefore read from its ENTRY: the entry fill must have
    come from an executable quote and the size must have come from the ALL-IN
    rule. An outstanding arm is not a result -- it is counted as UNRESOLVED
    progress and contributes no realized P&L.
    """
    if not isinstance(pos, Mapping):
        return False
    return (str(pos.get("entry_basis") or "") == EXECUTABLE_QUOTE_BASIS
            and (str(pos.get("sizing_rule") or "") == SIZING_ALL_IN
                 or str(pos.get("sizing_rule") or "") == "ALL_IN_100_PCT_OF_AVAILABLE_CASH"))


@dataclass(frozen=True)
class BetaEntryFill:
    """
    The EXECUTABLE QUOTE an entry was booked from.

    Every field is required. There is no ``mark_price`` field and no default,
    because the defect being closed was that the entry was booked at the
    observed MARK: a mark is not a price a fill could execute at, and a
    missing quote could not be told apart from a good one.

    ``cost_quote`` is the TOTAL cost in the quote asset -- the quote's own
    input amount, which already carries its fee -- and it is the denominator of
    every later return. ``tokens_raw`` is the quantity the BUY actually
    acquired, and the exit must quote exactly this.
    """
    price_usd: float
    tokens_raw: int
    token_decimals: int
    cost_quote: float
    fee_quote: float
    mark_price: float | None
    quote_timestamp: float
    quote_age_ms: float | None
    venue: str | None
    pool_or_curve: str | None
    provider: str | None
    route: tuple[str, ...]
    identity: str
    basis: str = EXECUTABLE_QUOTE_BASIS
    #: The SAME total BUY cost as ``cost_quote``, expressed in USD. These are
    #: two UNITS of one quantity, not two measurements: ``cost_quote`` is the
    #: quote-asset (SOL) leg the return denominator must stay in, and
    #: ``cost_usd`` is what the arm actually committed out of its USD bankroll.
    #: ``None`` means "this caller's quote asset is USD-equivalent" (the
    #: synthetic fixture path), in which case the two are equal.
    cost_usd: float | None = None
    #: THE RECORDED QUOTE PROVENANCE, verbatim. Everything a reader needs to
    #: reproduce this entry: both amounts and their decimals, the slippage
    #: setting and the resulting MINIMUM-OUTPUT floor, the venue's price impact,
    #: the venue/provider/route identity, the quote's age and latency, the fee,
    #: the SOL/USD source and timestamp, the observed TRIGGER price, and the
    #: MEASURED signal->quote->fill latency. Empty on a fixture-built fill, and
    #: reported as absent rather than as zero.
    provenance: Mapping[str, Any] = field(default_factory=dict)
    #: The round trip's costs that live OUTSIDE the swap input -- the Solana base
    #: fee, the priority fee and the token-account rent -- with their provenance.
    #: ``None`` means this fill did NOT model them (the fixture/synthetic path),
    #: which is reported as LEGACY and NEVER as "free".
    outside_costs: "OutsideSwapCosts | None" = None

    @property
    def usd_cost(self) -> float:
        """The USD this entry committed -- the affordability denominator."""
        if self.cost_usd is None:
            return float(self.cost_quote)
        return float(self.cost_usd)

    def validate(self, *, allocation_usd: float | None = None) -> None:
        """Fail closed. An unreadable quote, or an unaffordable one, is not a fill."""
        if not (self.price_usd and self.price_usd > 0):
            raise ValueError(
                "an entry needs a positive price from an executable quote; "
                f"got {self.price_usd!r}. A MARK must never be substituted.")
        if self.tokens_raw is None or self.tokens_raw <= 0:
            raise ValueError(
                f"an entry needs the acquired token quantity; got "
                f"{self.tokens_raw!r}. An exit cannot quote an unknown amount.")
        if self.token_decimals is None or self.token_decimals < 0:
            raise ValueError(
                f"an entry needs the token decimals; got "
                f"{self.token_decimals!r}.")
        if not (self.cost_quote and float(self.cost_quote) > 0):
            raise ValueError(
                f"an entry needs a positive total BUY cost, which is the return "
                f"denominator; got {self.cost_quote!r}")
        # THE AFFORDABILITY GUARD. An all-in entry is funded entirely by the
        # arm's own available cash, so a BUY that costs more than that cash is
        # refused -- never opened at a size the ledger cannot pay for. The
        # venue's fee is already inside the quoted input, so this comparison is
        # the "after required fees" test: what is checked is the GROSS input
        # the venue says it will take.
        if allocation_usd is not None and (
                self.usd_cost > float(allocation_usd) + 1e-9):
            raise ValueError(
                f"the BUY costs ${self.usd_cost:.9f} but the arm has only "
                f"${float(allocation_usd):.9f} available; an all-in entry may "
                "never exceed the capital it is funded from")


@dataclass(frozen=True)
class BetaExitFill:
    """
    The EXECUTABLE SELL quote that settles one arm.

    The return is computed here from the two receipts, with the denominator
    being the BUY's total cost:

        net SELL proceeds = gross proceeds - this quote's own fee
        net P&L            = net SELL proceeds - total BUY cost
        net return         = net P&L / total BUY cost

    Both legs come from quotes, so a mark cannot enter the arithmetic, and
    there is no field a mark could be written to. ``tokens_raw`` must equal the
    quantity the BUY acquired -- see :meth:`validate`.
    """
    price_usd: float
    tokens_raw: int
    token_decimals: int
    #: Gross proceeds in the quote asset for the exact quantity sold.
    cost_quote: float
    #: The modelled fee on this exit, in the quote asset.
    fee_quote: float
    #: The arm's total BUY cost. The return DENOMINATOR.
    entry_cost_quote: float
    quote_timestamp: float
    identity: str
    basis: str = EXECUTABLE_QUOTE_BASIS
    #: The SELL leg's recorded quote provenance. See ``BetaEntryFill.provenance``.
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, pos: "BetaPosition") -> None:
        """Fail closed. An unusable quote leaves the arm unresolved."""
        if pos.entry_cost_quote is None or float(pos.entry_cost_quote) <= 0:
            raise ValueError(
                "the arm has no recorded BUY cost, so a return has no "
                f"denominator (entry basis: {pos.entry_basis})")
        if pos.entry_tokens_raw is None or pos.entry_tokens_raw <= 0:
            raise ValueError(
                "the arm has no recorded acquired quantity, so an exit cannot "
                "quote the right amount")
        if self.tokens_raw != pos.entry_tokens_raw:
            raise ValueError(
                f"the SELL quote must cover the EXACT acquired quantity "
                f"{pos.entry_tokens_raw}; got {self.tokens_raw}. Re-estimating "
                "the size would fabricate the outcome.")
        if not (self.price_usd and self.price_usd > 0):
            raise ValueError(
                f"a SELL quote needs a positive price; got {self.price_usd!r}")
        if not (self.entry_cost_quote and self.entry_cost_quote > 0):
            raise ValueError(
                "a SELL quote needs the BUY cost it is being measured against "
                f"({self.entry_cost_quote!r})")

    @property
    def gross_proceeds_quote(self) -> float:
        return float(self.cost_quote)

    @property
    def net_proceeds_quote(self) -> float:
        """Gross proceeds less this exit's own fee. The exit leg of the return."""
        return float(self.cost_quote) - float(self.fee_quote or 0.0)

    @property
    def net_pnl_usd(self) -> float:
        return self.net_proceeds_quote - float(self.entry_cost_quote)

    @property
    def gross_return_pct(self) -> float:
        """Return on gross proceeds, before the exit fee. Diagnostic only."""
        return (float(self.cost_quote / self.entry_cost_quote - 1.0) * 100.0)

    @property
    def net_return_pct(self) -> float:
        """net SELL proceeds / total BUY cost - 1, in percent."""
        return (self.net_proceeds_quote / float(self.entry_cost_quote)
                - 1.0) * 100.0


@dataclass(frozen=True)
class BetaArmConfig:
    """One arm's fixed exit rule. Frozen: experiment parameters do not change."""

    arm: str
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_hold_s": self.max_hold_s,
        }


BETA_ARM_CONFIGS: tuple[BetaArmConfig, ...] = (
    BetaArmConfig(arm=ARM_A, take_profit_pct=30.0, stop_loss_pct=-30.0, max_hold_s=90.0),
    BetaArmConfig(arm=ARM_B, take_profit_pct=50.0, stop_loss_pct=-30.0, max_hold_s=90.0),
    BetaArmConfig(arm=ARM_C, take_profit_pct=50.0, stop_loss_pct=-30.0, max_hold_s=120.0),
)


def arm_config(arm: str) -> BetaArmConfig:
    for cfg in BETA_ARM_CONFIGS:
        if cfg.arm == arm:
            return cfg
    raise ValueError(
        f"unknown beta arm {arm!r}; the beta experiment has exactly "
        f"{', '.join(BETA_ARMS)}")


@dataclass
class BetaPosition:
    """One arm's open virtual position for one parent signal."""

    parent_signal_id: str
    arm: str
    mint: str
    symbol: str | None
    #: USD price the entry was booked at. This is the EXECUTABLE QUOTE's
    #: effective price, never a mark -- see ``entry_basis``.
    entry_fill_price: float
    entry_mark_price: float | None
    position_size_usd: float
    entry_at: float

    # -- THE SIZING DECISION ---------------------------------------------
    # Recorded ON the position, so the number in a report can always be traced
    # back to the rule that produced it and a historical 10%-sized trade can
    # never be read as an all-in one.
    #:
    #: The arm's uncommitted cash at entry. Under the ALL-IN rule this is the
    #: whole of it -- "100% of available capital" is a record, not a claim.
    allocated_usd: float | None = None
    #: The USD this entry actually committed. ``<= allocated_usd``; the rest, if
    #: any, is lamport/lot granularity dust and is recorded rather than rounded
    #: away.
    entry_cost_usd: float | None = None
    #: Uncommitted cash left in the arm immediately after this entry.
    remaining_cash_usd: float | None = None
    #: 100.0 under the ALL-IN rule; ``None`` on a record written before it.
    sizing_pct: float | None = None
    sizing_rule: str | None = None

    # -- QUOTE-DERIVED ENTRY BASIS ---------------------------------------
    # Added so the exit can quote the EXACT quantity the BUY acquired and so
    # the return can be recomputed from the receipts alone. Before this, the
    # entry was the observed MARK (``mark.value_usd``) and the exit was a
    # percentage of a mark: neither leg was an executable quote, and a mark
    # failure could not be distinguished from a real price.
    #:
    # ``entry_basis`` names which model produced these numbers. It is never
    # absent, so no consumer has to guess.
    entry_basis: str = EXECUTABLE_QUOTE_BASIS
    #: RAW token units the accepted BUY quote acquired, and their decimals.
    #: The exit MUST quote exactly this quantity -- not a re-estimated one.
    entry_tokens_raw: int | None = None
    entry_token_decimals: int | None = None
    #: Total BUY cost in the quote asset (input), INCLUDING the quote's own
    #: fee and the modelled network cost. This is the return DENOMINATOR.
    entry_cost_quote: float | None = None
    entry_fee_quote: float | None = None
    entry_quote_timestamp: float | None = None
    entry_quote_age_ms: float | None = None
    entry_venue: str | None = None
    entry_pool_or_curve: str | None = None
    entry_provider: str | None = None
    entry_route: tuple[str, ...] = ()
    entry_identity: str = "UNKNOWN"
    #: The entry's recorded quote provenance (amounts, min-output floor, slippage,
    #: impact, venue identity, quote age/latency, SOL/USD source+timestamp and
    #: the measured latency marks). ``{}`` means NOT RECORDED -- never zero.
    entry_provenance: dict[str, Any] = field(default_factory=dict)

    # -- OUTSIDE-SWAP COSTS (paid BESIDE the swap input) ------------------
    #: The round trip's costs that are NOT inside the swap quote -- Solana base
    #: fee, priority fee and the token-account rent -- with a provenance label
    #: per component. ``{}`` on a record written before this model existed;
    #: those are reported as LEGACY (swap-quote costs only) and never as fully
    #: costed.
    outside_swap_costs: dict[str, Any] = field(default_factory=dict)
    #: The USD swap-input budget this arm actually deployed: 100% of its
    #: available DEPLOYABLE capital (available cash less the reserve).
    deployable_usd: float | None = None
    #: ``allocated_usd - deployable_usd``: the fees and recoverable rent the
    #: wallet must hold BESIDE the swap input, or it cannot execute at all.
    reserved_outside_swap_usd: float | None = None
    #: The part of the reserve that is genuinely SPENT (both legs' base +
    #: priority fees). Charged against the arm's bankroll at settlement.
    outside_swap_cost_usd: float | None = None
    #: The QUOTE-based P&L before that charge. ``net_pnl_usd`` is this MINUS
    #: ``outside_swap_cost_usd`` -- i.e. after ALL modelled costs.
    trading_pnl_usd: float | None = None

    # -- QUOTE-DERIVED EXIT BASIS ----------------------------------------
    exit_basis: str | None = None
    exit_cost_quote: float | None = None
    exit_fee_quote: float | None = None
    exit_tokens_raw: int | None = None
    exit_quote_timestamp: float | None = None
    exit_identity: str = "UNKNOWN"
    #: The exit's recorded quote provenance: the exit TRIGGER, the measured
    #: detection->quote latency, the SELL's min-output floor and slippage, the
    #: venue identity and the quote's age. ``{}`` means NOT RECORDED.
    exit_provenance: dict[str, Any] = field(default_factory=dict)
    #: The settled proceeds in the SAME USD the entry committed
    #: (``position_size_usd + net_pnl_usd``).
    exit_proceeds_usd: float | None = None
    #: Why no exit could be booked, when the arm is UNRESOLVED. Never a price.
    exit_blocked_reason: str | None = None
    #: How many exit-quote attempts have been made for this arm.
    exit_attempts: int = 0
    #: THE BLOCKED-EXIT NOTIFICATION BOOKKEEPING. The value of ``exit_attempts``
    #: whose EXIT BLOCKED / UNRESOLVED notification has already been EMITTED.
    #: It is PERSISTED, so a retry, a repeated monitor tick or a process restart
    #: cannot re-announce an attempt the operator has already been told about.
    #: Bookkeeping only: no price, no ledger effect, and it never settles the arm.
    exit_notified_attempt: int = 0
    peak_return_pct: float = 0.0
    trough_return_pct: float = 0.0
    last_return_pct: float = 0.0
    last_observed_at: float = 0.0
    exit_reason: str | None = None
    exit_fill_price: float | None = None
    exit_at: float | None = None
    gross_return_pct: float | None = None
    net_return_pct: float | None = None
    net_pnl_usd: float | None = None
    path_ambiguous: bool = False
    fill_model: str = MODELED_BACKTEST_BASIS
    status: str = "open"

    # -- THE EXECUTABLE-QUOTE RETURN SERIES (the production trigger source) --
    #
    # ``peak_return_pct`` / ``trough_return_pct`` / ``last_return_pct`` above are
    # the MODELLED mark series and drive ``_settle_modeled`` only. These are a
    # SEPARATE series, computed from validated executable SELL quotes, and they
    # are the ONLY thing that may satisfy TP or SL on a quote path.
    #
    # They are kept apart rather than merged because merging them would make a
    # mark excursion silently able to satisfy an executable threshold again --
    # which is the exact defect recorded on :data:`RETURN_BASIS_QUOTE`.
    #: The basis ``peak/trough/last_return_pct`` were computed on. Recorded so a
    #: reader can tell a modelled record from a quote one without inferring it
    #: from the fill label.
    return_basis: str = RETURN_BASIS_MARK
    #: Highest executable SELL quote net return observed, in percent, on
    #: :data:`RETURN_BASIS_QUOTE`. ``0.0`` with ``quote_observation_count == 0``
    #: means NOTHING WAS OBSERVED -- never "the position never went up".
    quote_peak_return_pct: float = 0.0
    quote_trough_return_pct: float = 0.0
    quote_last_return_pct: float = 0.0
    quote_peak_at: float | None = None
    quote_trough_at: float | None = None
    quote_peak_basis: str | None = None
    quote_trough_basis: str | None = None
    #: How many validated executable SELL quotes were observed while the arm was
    #: open. Zero means the TP/SL series has NO evidence behind it, and the
    #: excursion must then be reported as unmeasured rather than as 0.0.
    quote_observation_count: int = 0
    #: Why the most recent candidate quote was not admitted to the series, so a
    #: stale or identity-failed quote is visible rather than silently absent.
    quote_last_rejection_reason: str | None = None
    #: The MARK excursion, kept strictly for diagnosis and never for a trigger.
    mark_peak_return_pct: float | None = None
    mark_trough_return_pct: float | None = None
    mark_last_return_pct: float | None = None

    # -- WHY THIS EXIT FIRED (auditable, on the settled record) --------------
    #
    # ``exit_reason`` alone says WHAT happened. These say what the engine saw
    # when it decided, on which basis, and from which quote -- so a reader can
    # re-derive the decision instead of taking the label on trust. They are the
    # fields whose absence made the 18-of-20 mislabelled stops undetectable.
    #: The return observed at the moment the exit fired, on the basis that
    #: actually decided it.
    trigger_return_pct: float | None = None
    trigger_basis: str | None = None
    trigger_at: float | None = None
    trigger_price_usd: float | None = None
    trigger_quote_identity: str = ""
    trigger_quote_age_ms: float | None = None
    trigger_quote_timestamp: float | None = None
    #: Expected net proceeds in the QUOTE ASSET implied by the deciding quote:
    #: gross proceeds less this leg's fee. Compared against
    #: ``settlement_proceeds_quote`` to show whether the booking matched the
    #: quote it was triggered by.
    trigger_expected_net_proceeds_quote: float | None = None
    #: What the settlement ACTUALLY booked, in the quote asset, and the return
    #: it implies. Always from the accepted quote, never from a threshold.
    settlement_proceeds_quote: float | None = None
    settlement_return_pct: float | None = None
    settlement_net_proceeds_quote: float | None = None
    #: True when the accepted settlement quote IS the quote that triggered the
    #: exit. False only when a documented execution-delay model books a later
    #: quote; ``None`` when it was not recorded.
    settlement_quote_is_trigger_quote: bool | None = None

    def config(self) -> BetaArmConfig:
        return arm_config(self.arm)


class BetaExperiment:
    """The V2 FLOOR (BETA) experiment session: 3 isolated ledgers, 1 gate."""

    def __init__(self, *, starting_bankroll_usd: float = BETA_STARTING_BANKROLL_USD,
                 state_root: Any = None, runtime: bool = False) -> None:
        """
        ``state_root`` is REQUIRED and ``runtime`` defaults to ``False``.

        This session owns a durable file -- ``v2_floor_beta_state.json`` -- and
        it is the one a probe reaches most easily by accident, because the
        experiment is constructed internally by the engine rather than by the
        caller. It used to default to ``None``, which
        ``PaperStateStore`` resolved to ``Path("var")`` RELATIVE TO THE PROCESS
        WORKING DIRECTORY: a synthetic session named ``beta_<now>``, complete
        with synthetic open positions, was written straight into the operator's
        live experiment file.

        Now an omitted root raises, and a root pointing at the runtime state
        directory is refused unless the runtime says ``runtime=True``.
        """
        self.starting_bankroll_usd = float(starting_bankroll_usd)
        self.audit = assert_paper_only()
        self.fill_model = FillModel()
        self.state_store = PaperStateStore(root=state_root,
                                           filename="v2_floor_beta_state.json",
                                           runtime=runtime)
        self.state = self.state_store.load()
        # THE FORWARD EXPERIMENT'S OWN STORE. A second file in the SAME root,
        # written by the mirror in :meth:`_persist_forward`, holding only ALL-IN
        # quote-based results. It is created on the first forward write and never
        # touches the main store's 51 historical modelled trades.
        self.forward_state_store = PaperStateStore(
            root=state_root, filename=FORWARD_STATE_FILENAME, runtime=runtime)
        self.forward_state = self.forward_state_store.load()
        # STABLE ACROSS RESTARTS: derived from the forward store's own creation
        # stamp, which is read back from the file when one exists. A random or
        # clock-derived id would relabel the same sample on every process start.
        self.forward_session_id = (
            f"beta_fwd_{int(self.forward_state.created_at)}")
        self.session_id = f"beta_{int(time.time())}"
        self.session_started_at = time.time()
        self.positions: dict[str, dict[str, BetaPosition]] = {}
        self._seen_signals: set[str] = set()
        self.completed: list[dict[str, Any]] = []
        self._init_accounts()
        self._restore_from_state()

    def _init_accounts(self) -> None:
        self.accounts: dict[str, VirtualAccount] = {
            arm: VirtualAccount(
                arm=f"V2BETA_{arm}",
                arm_version="V2_FLOOR_BETA",
                baseline_hash=BETA_BASELINE_HASH,
                starting_bankroll_usd=self.starting_bankroll_usd,
            )
            for arm in BETA_ARMS
        }

    @property
    def total_starting_capital_usd(self) -> float:
        return self.starting_bankroll_usd * len(BETA_ARMS)

    def open_for_signal(
        self, *, parent_signal_id: str, mint: str, symbol: str | None,
        position_size_usd: float, entry_fill: BetaEntryFill,
        entry_at: float | None = None,
    ) -> dict[str, BetaPosition]:
        """
        EXPLICIT-SIZE entry: the caller declares the size and one BUY quote.

        This is the fixture / synthetic-probe primitive. It deliberately does
        NOT derive the size from the arm (the caller has already decided it) and
        does NOT apply the overlap rule, because a probe opens several parents
        against one arm on purpose, to exercise path safety. The PRODUCTION path
        is :meth:`open_arms`, which does both. The rule used is recorded as
        ``EXPLICIT_SIZE`` so a probe position is never reported as ALL-IN.

        ``entry_fill`` is the EXECUTABLE QUOTE the entry is booked from. It is
        required and validated: an entry priced at a mark, or one whose acquired
        quantity is unknown, is refused rather than quietly accepted. The three
        arms share this ONE fill by design -- one parent signal, one entry
        basis -- and they differ only in exit bracket and max hold.

        ``entry_mark_price`` is no longer accepted. It used to be the entry
        price, which is precisely what made a failed quote indistinguishable
        from a good price.
        """
        if parent_signal_id in self._seen_signals:
            return self.positions.get(parent_signal_id, {})
        size = float(position_size_usd)
        if size <= 0:
            raise ValueError(
                "position_size_usd must be positive; got "
                f"{position_size_usd!r}")
        # The declared size IS the allocation here, so an entry that costs more
        # than it is refused -- the affordability guard is not bypassed just
        # because the caller named the size.
        entry_fill.validate(allocation_usd=size)
        now = time.time() if entry_at is None else float(entry_at)
        # ONE entry basis shared by A/B/C, taken from the quote itself.
        return self._create_arms(
            parent_signal_id=parent_signal_id, mint=mint, symbol=symbol,
            arms=BETA_ARMS, fills={arm: entry_fill for arm in BETA_ARMS},
            allocations={arm: size for arm in BETA_ARMS},
            sizing_rule=SIZING_EXPLICIT, sizing_pct=None, entry_at=now,
        )

    def observe(self, parent_signal_id: str, arm: str, *, mark_price: float,
                at: float | None = None) -> None:
        """Record one MARK observation.

        THE MARK SERIES IS NOW DIAGNOSTIC ONLY.

        This feeds ``peak_return_pct`` / ``trough_return_pct``, which drive
        :meth:`_settle_modeled` and nothing else. It is retained because the
        historical backtest has no quotes by construction, and because a mark
        excursion is worth keeping next to the executable one as a label of
        which book the price came from.

        It CANNOT satisfy TP or SL on a quote path, and :meth:`observe_quote` is
        the only producer of the series that can. See :data:`RETURN_BASIS_QUOTE`
        for what that defect looked like in production.
        """
        pos = self.positions.get(parent_signal_id, {}).get(arm)
        if pos is None or pos.status not in ("open", "unresolved"):
            return None
        if mark_price <= 0:
            return None
        now = time.time() if at is None else float(at)
        ret = (mark_price - pos.entry_fill_price) / pos.entry_fill_price * 100.0
        pos.peak_return_pct = max(pos.peak_return_pct, ret)
        pos.trough_return_pct = min(pos.trough_return_pct, ret)
        pos.last_return_pct = ret
        pos.last_observed_at = now
        # THE DIAGNOSTIC MIRROR, so a reader never has to guess which series a
        # number came from: the mark excursion is recorded under its own name
        # and its own basis, separately from the executable one.
        pos.mark_peak_return_pct = pos.peak_return_pct
        pos.mark_trough_return_pct = pos.trough_return_pct
        pos.mark_last_return_pct = ret
        return None

    def observe_quote(self, parent_signal_id: str, arm: str, *,
                      exit_fill: "BetaExitFill", at: float | None = None
                      ) -> float | None:
        """
        Fold ONE validated executable SELL quote into the trigger series.

        THE ONLY PRODUCER OF THE SERIES THAT MAY SATISFY TP OR SL.

        The return is computed by the same arithmetic the settlement uses --
        :attr:`BetaExitFill.net_return_pct`, i.e. net SELL proceeds over the
        entry's total BUY cost -- so a threshold and the booked P&L are the same
        calculation on the same two receipts and cannot disagree about what
        "up" meant.

        The quote must already have passed :meth:`BetaExitFill.validate`: the
        EXACT acquired quantity, a positive price, and a recorded total BUY cost
        to be the denominator. Identity is checked here rather than assumed,
        because this series is what settles the arm: an UNVERIFIED quote is not
        admitted, and the reason is recorded so an absent excursion is
        diagnosable rather than silently short.

        Returns the observed return in percent, or ``None`` when the quote was
        refused -- in which case NOTHING was recorded, because a rejected quote
        is not an observation at zero.
        """
        pos = self.positions.get(parent_signal_id, {}).get(arm)
        if pos is None or pos.status not in ("open", "unresolved"):
            return None
        try:
            exit_fill.validate(pos)
        except ValueError as exc:
            pos.quote_last_rejection_reason = str(exc)[:200]
            return None
        identity = str(getattr(exit_fill, "identity", "") or "")
        if identity and identity != "PASS":
            pos.quote_last_rejection_reason = (
                f"same-market identity not proven ({identity})")
            return None
        if identity == "":
            # Absence of a verdict is not a failed verdict, but it is not a pass
            # either: it is recorded as unknown so the series never claims more
            # provenance than the quote carried.
            pos.quote_peak_basis = RETURN_BASIS_QUOTE
            pos.quote_trough_basis = RETURN_BASIS_QUOTE
        now = time.time() if at is None else float(at)
        ret = float(exit_fill.net_return_pct)
        if ret > pos.quote_peak_return_pct:
            pos.quote_peak_return_pct = ret
            pos.quote_peak_at = now
            pos.quote_peak_basis = RETURN_BASIS_QUOTE
        if ret < pos.quote_trough_return_pct:
            pos.quote_trough_return_pct = ret
            pos.quote_trough_at = now
            pos.quote_trough_basis = RETURN_BASIS_QUOTE
        pos.quote_last_return_pct = ret
        pos.quote_observation_count = int(pos.quote_observation_count or 0) + 1
        pos.quote_last_rejection_reason = None
        # ``last_observed_at`` marks the last time ANY evidence was seen. The
        # modelled settle path uses it to tell "never observed" from "observed
        # and stale"; a quote observation is strictly better evidence and
        # advances it, so the two paths cannot disagree about whether the arm
        # was ever priced.
        pos.last_observed_at = max(float(pos.last_observed_at or 0.0), now)
        pos.return_basis = RETURN_BASIS_QUOTE
        return ret

    def _exit_decision(self, pos: BetaPosition, *, now: float,
                       basis: str = RETURN_BASIS_MARK) -> tuple[str, bool]:
        """
        Exit reason + ambiguity flag; ambiguous TP/SL -> conservative SL.

        ``basis`` SELECTS WHICH RETURN SERIES MAY SATISFY TP OR SL, and it is
        the whole point of this method.

        ``RETURN_BASIS_QUOTE`` (production) reads only
        ``quote_peak_return_pct`` / ``quote_trough_return_pct``, which
        :meth:`observe_quote` fills from validated executable SELL quotes. A
        mark excursion cannot reach it, so a mark threshold alone can never
        produce a quote-settled TP or SL fill -- and with no quote observed yet,
        only a TIME exit can fire, because a deadline is a fact about the clock
        rather than a claim about a price.

        ``RETURN_BASIS_MARK`` (the historical backtest, which has no quotes by
        construction) reads the modelled series exactly as before. Its behaviour
        and its persisted numbers are unchanged.

        Returning a reason is NOT a fill: the caller still has to obtain and
        accept an executable quote before anything is booked.
        """
        cfg = pos.config()
        held = max(0.0, now - pos.entry_at)
        if basis == RETURN_BASIS_QUOTE:
            if int(pos.quote_observation_count or 0) <= 0:
                # NO EXECUTABLE EVIDENCE. A threshold cannot be satisfied by a
                # price nobody quoted.
                tp_touched = sl_touched = False
            else:
                tp_touched = pos.quote_peak_return_pct >= cfg.take_profit_pct
                sl_touched = pos.quote_trough_return_pct <= cfg.stop_loss_pct
        else:
            tp_touched = pos.peak_return_pct >= cfg.take_profit_pct
            sl_touched = pos.trough_return_pct <= cfg.stop_loss_pct
        if tp_touched and sl_touched:
            return "SL", True
        if sl_touched:
            return "SL", False
        if tp_touched:
            return "TP", False
        if held >= cfg.max_hold_s:
            return "TIME", False
        return "", False

    def _settle_modeled(self, pos: BetaPosition, *, reason: str,
                        ambiguous: bool, now: float) -> BetaPosition:
        """
        HISTORICAL / BACKTEST settlement: the modelled limit-or-trough booking.

        A TP books at the LIMIT target, a stop at the observed trough, a time
        exit at the last observed return, and every return is NET of the
        modelled ``PAPER_EXIT_MULT`` friction. This is the path the research
        harness and the historical backtests use; it is NOT production, it is
        NEVER reached by the engine (which requires a quote), and every record
        it produces is tagged ``MODELED_BACKTEST_BASIS``.

        The ambiguity rule is preserved: when the path cannot prove whether TP
        or SL came first the trade is booked CONSERVATIVELY as the stop and
        flagged ``path_ambiguous``.
        """
        cfg = pos.config()
        if reason == "TP":
            gross_pct = float(cfg.take_profit_pct)
        elif reason == "SL":
            gross_pct = float(pos.trough_return_pct)
        else:  # TIME -- book the last observed return
            gross_pct = float(pos.last_return_pct)
        gross = gross_pct / 100.0
        net = net_from_gross(gross)
        pos.exit_reason = reason
        pos.exit_fill_price = pos.entry_fill_price * (1.0 + gross)
        pos.exit_at = now
        pos.gross_return_pct = gross_pct
        pos.net_return_pct = net * 100.0
        pos.net_pnl_usd = pos.position_size_usd * net
        pos.path_ambiguous = bool(ambiguous)
        pos.status = "closed"
        pos.fill_model = MODELED_BACKTEST_BASIS
        # THE MODELLED TRIGGER, RECORDED AS SUCH. A modelled record says on its
        # face that its TP/SL came from sampled marks and that its settlement is
        # a limit-or-trough booking, so it can never be read as an executable
        # result. ``trigger_return_pct`` is the modelled return the rule acted
        # on, and the executable settlement fields stay ``None`` because no
        # quote was involved -- unknown, not zero.
        pos.return_basis = RETURN_BASIS_MARK
        pos.trigger_basis = RETURN_BASIS_MARK
        pos.trigger_at = now
        pos.trigger_return_pct = float(gross_pct)
        pos.settlement_return_pct = net * 100.0
        pos.settlement_quote_is_trigger_quote = False
        if not pos.entry_basis:
            pos.entry_basis = MODELED_BACKTEST_BASIS
        pos.exit_basis = MODELED_BACKTEST_BASIS
        account = self.accounts[pos.arm]
        account.bankroll_usd = round(account.bankroll_usd + pos.net_pnl_usd, 6)
        account.peak_bankroll_usd = max(account.peak_bankroll_usd,
                                        account.bankroll_usd)
        account.position = None
        account.closed_trades += 1
        self.completed.append(self._trade_record(pos))
        self._persist()
        return pos

    def settle_arm(self, parent_signal_id: str, arm: str,
                   now: float | None = None,
                   exit_fill: "BetaExitFill | None" = None,
                   *, require_quote: bool = False) -> BetaPosition | None:
        """
        Resolve + settle ONE arm; never touches the other arms' ledger.

        ``exit_fill`` is the EXECUTABLE SELL quote for this arm's exact
        remaining quantity. When it is absent the arm stays UNRESOLVED and the
        reason is recorded -- it is NEVER priced from a mark.

        A trigger is not a fill. ``_exit_decision`` reports THAT an exit is due
        (a target was touched, or the deadline passed); the P&L comes from the
        quote, and a missing quote means the trade has not happened yet.

        THE RETURN BASIS FOLLOWS ``require_quote``, and that is deliberate.
        Production (``require_quote=True``) evaluates TP/SL on
        :data:`RETURN_BASIS_QUOTE` only, so the deciding series and the booked
        P&L are the same arithmetic on the same two receipts. The historical
        backtest (``require_quote=False``) evaluates on
        :data:`RETURN_BASIS_MARK`, which is what it has.
        """
        pos = self.positions.get(parent_signal_id, {}).get(arm)
        # AN UNRESOLVED ARM IS RETRIED, not abandoned. A quote can fail
        # transiently (expiry, a 429, a provider failover), so the arm stays
        # outstanding and the next settle attempt quotes it again. It only
        # leaves ``unresolved`` when a real quote settles it.
        if pos is None or pos.status not in ("open", "unresolved"):
            return pos
        now = time.time() if now is None else float(now)
        cfg = pos.config()
        basis = RETURN_BASIS_QUOTE if require_quote else RETURN_BASIS_MARK
        reason, ambiguous = self._exit_decision(pos, now=now, basis=basis)
        if not reason:
            return pos
        if exit_fill is None:
            # No executable SELL quote was supplied.
            #
            # PRODUCTION (`require_quote=True`) and any arm that has NEVER been
            # observed stay UNRESOLVED: no quote, no settlement, and no bankroll
            # move. Only the BACKTEST path (`require_quote=False` AND an observed
            # path) may book the modelled limit-or-trough fill, and that fill is
            # tagged ``MODELED_BACKTEST_BASIS`` so it can never be mistaken for
            # an executable-quote settlement in reporting.
            if require_quote or pos.last_observed_at <= 0:
                pos.status = "unresolved"
                pos.exit_reason = "TIME_NO_QUOTE" if reason == "TIME" else \
                    f"{reason}_NO_QUOTE"
                pos.exit_blocked_reason = (
                    "no valid executable SELL quote for the acquired quantity; "
                    "the position is left unresolved rather than priced from a mark")
                pos.exit_attempts = int(pos.exit_attempts) + 1
                # THE TRIGGER THAT COULD NOT BE BOOKED IS STILL RECORDED, on the
                # basis it was evaluated. A blocked arm that later settles must
                # be able to show that it was a TIME expiry and not a stop, and
                # must never show a threshold it could not evidence.
                pos.trigger_basis = basis
                pos.trigger_at = now
                pos.settlement_quote_is_trigger_quote = None
                self._persist()
                return pos
            return self._settle_modeled(
                pos, reason=reason, ambiguous=ambiguous, now=now)
        exit_fill.validate(pos)
        gross = exit_fill.gross_return_pct
        net = exit_fill.net_return_pct
        # THE RETURN IS UNIT-FREE; THE P&L IS NOT.
        #
        # ``net`` is (net SELL proceeds / total BUY cost - 1), a ratio, so it
        # needs no FX rate and cannot drift with one. The P&L is that ratio
        # applied to the USD the ENTRY committed, which is what the arm's
        # bankroll is denominated in: an all-in $10.00 arm that returns +30%
        # realizes $3.00, not a SOL figure that has lost a factor of the SOL/USD
        # rate on the way into a dollar ledger. ``entry_cost_quote`` stays in the
        # quote asset because it is the return DENOMINATOR; ``position_size_usd``
        # is its USD twin, recorded at entry.
        base_usd = float(pos.position_size_usd)
        # THE TRADING P&L IS NOT THE NET P&L. The quote ratio gives what the
        # round trip EARNED on the deployed notional; the costs paid OUTSIDE the
        # swap (both legs' base + priority fees) come off afterwards, so the
        # number that moves the bankroll is the fully-costed one. A record with
        # no cost model (``outside_swap_costs == {}``) charges nothing and the
        # two are identical -- that is the fixture/backtest path, reported as
        # LEGACY rather than as cost-free.
        trading = round(base_usd * (net / 100.0), 9)
        charged = float(pos.outside_swap_cost_usd or 0.0)
        pos.exit_reason = reason
        pos.exit_fill_price = exit_fill.price_usd
        pos.exit_at = now
        pos.gross_return_pct = gross
        pos.net_return_pct = net
        pos.trading_pnl_usd = trading
        pos.net_pnl_usd = round(trading - charged, 9)
        pos.exit_proceeds_usd = round(base_usd + pos.net_pnl_usd, 9)
        pos.exit_provenance = dict(exit_fill.provenance or {})
        pos.path_ambiguous = ambiguous
        pos.status = "closed"
        pos.exit_basis = exit_fill.basis
        pos.exit_cost_quote = exit_fill.cost_quote
        pos.exit_fee_quote = exit_fill.fee_quote
        pos.exit_tokens_raw = exit_fill.tokens_raw
        pos.exit_quote_timestamp = exit_fill.quote_timestamp
        pos.exit_identity = exit_fill.identity

        # THE ACCEPTED QUOTE IS THE FILL. Recorded explicitly, because the whole
        # defect this file documents was a threshold being booked instead of the
        # quote: every figure below is read off ``exit_fill``, and
        # ``settlement_quote_is_trigger_quote`` states whether that quote is
        # also the one whose return satisfied the rule. There is NO documented
        # execution-delay model in this path, so the two are always the same
        # object and the flag says so rather than leaving it unknown.
        pos.return_basis = RETURN_BASIS_QUOTE
        pos.fill_model = QUOTE_FILL_MODEL
        pos.settlement_proceeds_quote = float(exit_fill.cost_quote)
        pos.settlement_net_proceeds_quote = float(exit_fill.net_proceeds_quote)
        pos.settlement_return_pct = float(net)
        pos.settlement_quote_is_trigger_quote = True
        pos.trigger_basis = RETURN_BASIS_QUOTE
        pos.trigger_at = now
        # The return the RULE saw. On a TP/SL that is the peak/trough that
        # satisfied the threshold; on a TIME it is this same quote's own return.
        # Recording it is what makes ``TP at -22.78%`` detectable: a reader can
        # see the decision return and the booked return side by side.
        if reason == "TP":
            pos.trigger_return_pct = float(pos.quote_peak_return_pct)
            pos.trigger_at = pos.quote_peak_at or now
        elif reason == "SL":
            pos.trigger_return_pct = float(pos.quote_trough_return_pct)
            pos.trigger_at = pos.quote_trough_at or now
        else:  # TIME -- the deciding quote is the settlement quote itself
            pos.trigger_return_pct = float(net)
        pos.trigger_price_usd = float(exit_fill.price_usd)
        pos.trigger_quote_identity = str(
            getattr(exit_fill, "identity", "") or "")
        pos.trigger_quote_timestamp = float(exit_fill.quote_timestamp)
        pos.trigger_quote_age_ms = (
            (pos.exit_provenance or {}).get("quote_age_ms")
            if isinstance(pos.exit_provenance, dict) else None)
        pos.trigger_expected_net_proceeds_quote = float(
            exit_fill.net_proceeds_quote)
        # The excursion the exit was decided on, with the basis named. A record
        # written before this change has none, and says so by leaving them None.
        pos.quote_peak_basis = pos.quote_peak_basis or RETURN_BASIS_QUOTE
        pos.quote_trough_basis = pos.quote_trough_basis or RETURN_BASIS_QUOTE
        account = self.accounts[arm]
        # NEVER NEGATIVE. A linear (all-in) size means a total loss zeroes the
        # arm; the clamp is the last line of defence against an arithmetic
        # regression turning that into a negative ledger, and it is reported
        # rather than silent if it ever binds.
        settled = round(account.bankroll_usd + pos.net_pnl_usd, 9)
        if settled < 0.0:
            import logging
            logging.getLogger(__name__).error(
                "beta arm %s settlement would take the bankroll to %.9f; "
                "clamping to 0.00 (entry cost %.9f, pnl %.9f)",
                arm, settled, base_usd, pos.net_pnl_usd)
        account.bankroll_usd = max(0.0, settled)
        account.peak_bankroll_usd = max(account.peak_bankroll_usd,
                                        account.bankroll_usd)
        account.position = None
        account.closed_trades += 1
        self.completed.append(self._trade_record(pos))
        self._persist()
        return pos

    def monitor_once(self, now: float | None = None) -> list[BetaPosition]:
        """Attempt to settle every open position across all arms."""
        now = time.time() if now is None else float(now)
        settled: list[BetaPosition] = []
        for pid in list(self.positions.keys()):
            for arm in BETA_ARMS:
                # A parent signal may already have settled some of its arms
                # (A booked, B and C still open), so an arm can be legitimately
                # absent. Skipping is correct; indexing it is a KeyError.
                pos = self.positions[pid].get(arm)
                if pos is None or pos.status not in ("open", "unresolved"):
                    continue
                result = self.settle_arm(pid, arm, now=now)
                if result is None or result.status not in ("closed", "unresolved"):
                    continue
                settled.append(result)
        return settled

    def _trade_record(self, pos: BetaPosition) -> dict[str, Any]:
        cfg = pos.config()
        # WHICH COHORT THIS SETTLEMENT BELONGS TO, recorded on the record itself
        # so every reader -- the forward mirror, the stats report, a later
        # analysis of the JSON -- reaches the same answer without re-deriving it.
        #
        # A settlement is INCIDENT RECOVERY when the exit was REFUSED before it
        # executed (``exit_attempts > 0``): the arm was stranded, its bracket
        # kept firing and the P&L that finally booked includes however long the
        # delay lasted. Everything settled on the rule's first firing is
        # FORWARD. The restart evidence, when there is any, travels with it.
        attempts = int(pos.exit_attempts or 0)
        delayed = attempts > 0
        evidence = getattr(pos, "delayed_across_restart", None) or {}
        return {
            "session_id": self.session_id,
            "parent_signal_id": pos.parent_signal_id,
            "cohort": (INCIDENT_RECOVERY_COHORT if delayed else FORWARD_COHORT),
            "incident_recovery": (
                {"exit_attempts_at_settlement": attempts,
                 "exit_reason_attempted": pos.exit_reason,
                 "refusals": (pos.exit_blocked_reason or "")[:300],
                 **dict(evidence)} if delayed else None),
            "arm": pos.arm,
            "mint": pos.mint,
            "symbol": pos.symbol,
            "entry_profile": BETA_ENTRY_PROFILE,
            "take_profit_pct": cfg.take_profit_pct,
            "stop_loss_pct": cfg.stop_loss_pct,
            "max_hold_s": cfg.max_hold_s,
            "entry_fill_price": pos.entry_fill_price,
            "entry_mark_price": pos.entry_mark_price,
            "position_size_usd": pos.position_size_usd,
            # -- THE SIZING DECISION ---------------------------------------
            # Which rule produced the size, what it allocated, what it actually
            # committed and what was left. Recorded on the TRADE, not just on the
            # position, because a closed trade is the artefact a later reader
            # has: without these a historical 10%-sized trade and an all-in one
            # are the same row.
            "sizing_rule": pos.sizing_rule,
            "sizing_pct": pos.sizing_pct,
            "allocated_usd": pos.allocated_usd,
            "entry_cost_usd": (pos.entry_cost_usd
                               if pos.entry_cost_usd is not None
                               else pos.position_size_usd),
            "remaining_cash_usd": pos.remaining_cash_usd,
            "exit_proceeds_usd": pos.exit_proceeds_usd,
            "entry_at": pos.entry_at,
            "exit_reason": pos.exit_reason,
            "exit_fill_price": pos.exit_fill_price,
            "exit_at": pos.exit_at,
            "gross_return_pct": pos.gross_return_pct,
            "net_return_pct": pos.net_return_pct,
            "net_pnl_usd": pos.net_pnl_usd,
            "path_ambiguous": pos.path_ambiguous,
            "fill_model": pos.fill_model,
            "status": pos.status,

            # -- WHY THIS EXIT FIRED ----------------------------------------
            # ``exit_reason`` says WHAT happened. These say what the engine
            # saw when it decided, on WHICH basis, from WHICH quote, and what
            # was actually booked. Without them a mislabelled stop is invisible
            # -- which is exactly how 18 of 20 live ``SL`` labels came to
            # describe a return better than the -30% they claim to have hit.
            "return_basis": pos.return_basis,
            "trigger_return_pct": pos.trigger_return_pct,
            "trigger_basis": pos.trigger_basis,
            "trigger_at": pos.trigger_at,
            "trigger_price_usd": pos.trigger_price_usd,
            "trigger_quote_identity": pos.trigger_quote_identity,
            "trigger_quote_age_ms": pos.trigger_quote_age_ms,
            "trigger_quote_timestamp": pos.trigger_quote_timestamp,
            "trigger_expected_net_proceeds_quote":
                pos.trigger_expected_net_proceeds_quote,
            "settlement_proceeds_quote": pos.settlement_proceeds_quote,
            "settlement_net_proceeds_quote":
                pos.settlement_net_proceeds_quote,
            "settlement_return_pct": pos.settlement_return_pct,
            "settlement_quote_is_trigger_quote":
                pos.settlement_quote_is_trigger_quote,

            # -- THE TWO RETURN SERIES, NEVER MIXED ------------------------
            # ``quote_*`` is the executable one that may satisfy a threshold.
            # ``peak/trough/last_return_pct`` is the modelled MARK series, and
            # ``mark_*`` is its diagnostic mirror. Both are persisted with their
            # basis named so no reader has to infer which book a number came
            # from, and so a historical mark excursion is never re-presented as
            # an executable Quote-MFE/MAE.
            "quote_peak_return_pct": pos.quote_peak_return_pct,
            "quote_trough_return_pct": pos.quote_trough_return_pct,
            "quote_last_return_pct": pos.quote_last_return_pct,
            "quote_peak_at": pos.quote_peak_at,
            "quote_trough_at": pos.quote_trough_at,
            "quote_peak_basis": pos.quote_peak_basis,
            "quote_trough_basis": pos.quote_trough_basis,
            "quote_observation_count": int(pos.quote_observation_count or 0),
            "quote_last_rejection_reason": pos.quote_last_rejection_reason,
            "mark_peak_return_pct": pos.mark_peak_return_pct,
            "mark_trough_return_pct": pos.mark_trough_return_pct,
            "mark_last_return_pct": pos.mark_last_return_pct,
            "peak_return_pct": pos.peak_return_pct,
            "trough_return_pct": pos.trough_return_pct,
            "last_return_pct": pos.last_return_pct,

            # -- QUOTE DERIVATION -------------------------------------------
            # Both legs' provenance and their raw quantities, so the outcome can
            # be recomputed from the receipts and a mark-priced trade can never
            # be mistaken for a quote-priced one.
            "entry_basis": pos.entry_basis,
            "entry_tokens_raw": pos.entry_tokens_raw,
            "entry_token_decimals": pos.entry_token_decimals,
            "entry_cost_quote": pos.entry_cost_quote,
            "entry_fee_quote": pos.entry_fee_quote,
            "entry_quote_timestamp": pos.entry_quote_timestamp,
            "entry_quote_age_ms": pos.entry_quote_age_ms,
            "entry_venue": pos.entry_venue,
            "entry_pool_or_curve": pos.entry_pool_or_curve,
            "entry_provider": pos.entry_provider,
            "entry_route": list(pos.entry_route or ()),
            "entry_identity": pos.entry_identity,

            "exit_basis": pos.exit_basis,
            "exit_cost_quote": pos.exit_cost_quote,
            "exit_fee_quote": pos.exit_fee_quote,
            "exit_tokens_raw": pos.exit_tokens_raw,
            "exit_quote_timestamp": pos.exit_quote_timestamp,
            "exit_identity": pos.exit_identity,
            "exit_blocked_reason": pos.exit_blocked_reason,
            "exit_attempts": pos.exit_attempts,

            # -- COSTS PAID OUTSIDE THE SWAP ---------------------------------
            # The round trip's non-quote costs, so a reader can see whether a
            # record was FULLY COSTED (fees + rent reserved) or charged only the
            # swap quote's own fee. ``{}`` means NOT MODELED -- LEGACY, never
            # "free". ``trading_pnl_usd`` is the quote-only P&L and
            # ``net_pnl_usd`` is it MINUS the charge, so the size of the charge
            # is always visible as the difference rather than hidden in the net.
            "outside_swap_costs": dict(pos.outside_swap_costs or {}),
            "outside_swap_cost_usd": pos.outside_swap_cost_usd,
            "outside_swap_charged_usd": (
                float(pos.outside_swap_cost_usd)
                if pos.outside_swap_cost_usd is not None else 0.0),
            "deployable_usd": pos.deployable_usd,
            "reserved_outside_swap_usd": pos.reserved_outside_swap_usd,
            "trading_pnl_usd": (pos.trading_pnl_usd
                                if pos.trading_pnl_usd is not None
                                else pos.net_pnl_usd),
            # Both legs' provenance, verbatim: amounts, min-output floor,
            # slippage, impact, venue identity, quote age and the measured
            # latency. Enough to recompute the trade from its receipts.
            "entry_provenance": dict(pos.entry_provenance or {}),
            "exit_provenance": dict(pos.exit_provenance or {}),

            # What this record is, for the reporting views that consume it. A
            # settlement that came from two EXECUTABLE QUOTES and one that came
            # from the MODELLED backtest path are DIFFERENT accounting bases and
            # must be separately identifiable in every consumer.
            "pnl_basis": (
                "V2_FLOOR_BETA_QUOTE_SIMULATED_FILL"
                if pos.exit_basis == EXECUTABLE_QUOTE_BASIS
                else "V2_FLOOR_BETA_MODELED_BACKTEST"),
            "strategy": BETA_ENTRY_PROFILE,
            "record_type": "beta_exit",
        }

    def _persist(self) -> None:
        open_positions = [
            p for pid in self.positions.values() for p in pid.values()
            # UNRESOLVED OUTSTANDING ARMS ARE PERSISTED TOO. An arm that could
            # not be quoted is NOT a settled trade and must not be booked as
            # one -- but it is still an outstanding position, and dropping it
            # from the durable state would make a restart lose it silently,
            # along with the reason it is stuck. It is written here with its
            # ``status``, restored as unresolved, reported as unresolved and
            # RETRIED; it never enters ``completed`` until it actually settles.
            if p.status in ("open", "unresolved")
        ]
        records = [self._position_dict(p) for p in open_positions]
        self.state.merge(trades=self.completed, open_positions=records)
        # AN OUTSTANDING ARM'S RECORD IS UPSERTED, NOT ADD-ONLY.
        #
        # ``merge`` keeps an existing key untouched, which is right for the
        # generic engine's append-and-reconcile dataset. Here it is wrong: an
        # arm's STATUS changes while it is outstanding (open -> unresolved when
        # an exit quote is refused, and ``exit_attempts`` grows on every retry),
        # so an add-only write would leave the durable record claiming "open"
        # with zero attempts and a restart would resurrect the stale one. This
        # file is the experiment's own, so the current record wins.
        for rec in records:
            self.state.open_positions[
                f"{rec.get('arm', '')}::{rec.get('signal_id', '')}"] = rec
        # A position that has settled must LEAVE the rehydratable open set.
        # ``merge`` only ever ADDS, so without this a settled position would
        # still be listed as open after the trade was booked -- and a restart
        # would revive it, re-settle it and count the same P&L twice.
        wanted = {self._position_key(p) for p in open_positions}
        stale = [k for k in self.state.open_positions if k not in wanted]
        if stale:
            self.state.close_positions(stale)
        self.state_store.save(self.state, note=f"beta session {self.session_id}")
        # THE SAME WRITE ALSO MIRRORS THE FORWARD-ELIGIBLE PART into the forward
        # experiment's own store. Done HERE, at the single durable-write point,
        # so the forward file can never lag the ledger it mirrors.
        self._persist_forward()

    def _persist_forward(self) -> None:
        """Mirror the FORWARD-eligible results into the forward store.

        THE ONLY WRITER of ``v2_floor_beta_forward_state.json``. It COPIES: it
        reads the session's own ledger and writes the subset that is both
        quote-based and ALL-IN. It never mutates the main store, never removes
        anything from it and never re-labels a historical record.

        An OUTSTANDING forward arm is mirrored too, under ``open_positions``,
        because an unresolved exit is progress the operator must be able to see
        -- but it is written with its own status and contributes no realized
        P&L anywhere. A settled arm leaves the forward open set through the same
        staleness sweep the main store uses, so a restart cannot revive it.

THE INCIDENT-RECOVERY COHORT IS A LABEL, NOT AN EXCLUSION
        ==========================================================
        A settlement that spanned a restart while the arm was UNRESOLVED was
        delayed by an incident (today: the pinned-SELL pair-unwrapping defect,
        which refused every pinned quote), not produced by the exit rule on
        schedule.

        IT IS NOW RETAINED IN ``trades`` ALONGSIDE THE FLAG. It used to be moved
        OUT of ``trades`` into ``incident_recovery``, and that was wrong for a
        reason the live data made plain: the incident classification keys off
        ``exit_attempts > 0``, and a refusal is an event that happens to ONE ARM
        of a parent signal and not to its siblings. UrMom's arms B and C were
        each refused three times and settled; arm A quoted first time and never
        was. The exclusion therefore produced ``A:14, B:13, C:13`` in the mirror
        -- an ARM-ASYMMETRIC matched set whose two missing rows happened to be
        the single largest losses in the cohort (-$2.515 and -$2.402).

        A matched A/B/C comparison read off a set that no longer contains the
        same trades is worse than no comparison, and the bias runs one way:
        delayed arms are by construction the ones that lost. The record is kept
        with its full ``cohort`` label and its ``incident_recovery`` evidence, so
        a reader who wants the strict cohort excludes it explicitly; a reader who
        forgets still gets a complete, matched, honestly-labelled set.

        THE INVARIANT
        -------------
        :meth:`_assert_forward_mirror_complete` then proves, after every write,
        that the mirror holds EXACTLY the authoritative eligible set for this
        session. A silent drop is the failure mode this whole section exists to
        end, so it is checked rather than assumed.
        """
        def _incident(rec: Any) -> bool:
            return _is_incident_recovery(rec)

        # RETAIN EVERY ELIGIBLE ARM. The incident flag travels on the record;
        # it does not decide membership.
        eligible = [rec for rec in self.completed if forward_eligible(rec)]
        incident = [rec for rec in eligible if _incident(rec)]
        open_eligible = [
            self._position_dict(p)
            for pid in self.positions.values() for p in pid.values()
            if p.status in ("open", "unresolved")
            and forward_eligible_open({
                "entry_basis": p.entry_basis, "sizing_rule": p.sizing_rule})
        ]
        self.forward_state.merge(trades=eligible, open_positions=open_eligible)
        # Upsert: an outstanding arm's record CHANGES while it is unresolved
        # (status and ``exit_attempts``), and ``merge`` only ever adds.
        for rec in eligible:
            self.forward_state.trades[self.forward_state.trade_key(rec)] = rec
        # The incident-recovery settlements are ALSO indexed under their own
        # section, purely so the delayed set is enumerable without a scan. The
        # record itself stays in ``trades`` -- see the section docstring.
        for rec in incident:
            self.forward_state.incident_recovery[
                self.forward_state.trade_key(rec)] = rec
        # A record that this build now classifies as incident recovery must no
        # longer be counted as a clean forward result. It is RETAINED and
        # flagged; only the counting is separated. Nothing is deleted from the
        # main store, and nothing is deleted here either.
        for rec in self.forward_state.trades.values():
            if _incident(rec):
                rec["cohort"] = INCIDENT_RECOVERY_COHORT
        for rec in open_eligible:
            self.forward_state.open_positions[
                f"{rec.get('arm', '')}::{rec.get('signal_id', '')}"] = rec
        wanted = {f"{p.get('arm', '')}::{p.get('signal_id', '')}"
                  for p in open_eligible}
        stale = [k for k in self.forward_state.open_positions if k not in wanted]
        if stale:
            self.forward_state.close_positions(stale)
        self.forward_state_store.save(
            self.forward_state, note=f"forward session {self.forward_session_id}")
        self._assert_forward_mirror_complete()

    def _assert_forward_mirror_complete(self) -> dict[str, Any]:
        """
        PROVE the forward mirror equals the authoritative eligible set.

        Checked after every forward write, because the failure this exists to
        catch is silent: an add-only ``merge`` plus a filter that can transiently
        exclude one arm produced a mirror that looked healthy and was missing two
        of the cohort's largest losses for a whole session.

        A discrepancy is REPORTED, never repaired by deletion and never
        repaired by inventing a record. The forward store is a mirror; the main
        store is the truth, and a mirror that disagrees is a bug to surface.
        """
        expected = {
            self.forward_state.trade_key(rec): rec
            for rec in self.completed if forward_eligible(rec)
        }
        mirrored = dict(self.forward_state.trades)
        missing = sorted(set(expected) - set(mirrored))
        unexpected = sorted(set(mirrored) - set(expected))
        mismatched = sorted(
            key for key in set(expected) & set(mirrored)
            if mirrored[key] is not expected[key]
            and mirrored[key] != expected[key]
        )
        report = {
            "expected": len(expected),
            "mirrored": len(mirrored),
            "missing": missing,
            "unexpected": unexpected,
            "mismatched": mismatched,
            "ok": not (missing or unexpected or mismatched),
        }
        if not report["ok"]:
            import logging
            logging.getLogger(__name__).error(
                "forward mirror INCOMPLETE for session %s: %d expected, %d "
                "mirrored; missing=%s unexpected=%s mismatched=%s. The main "
                "store is authoritative and nothing was deleted to make this "
                "look clean.",
                self.session_id, len(expected), len(mirrored),
                missing[:8], unexpected[:8], mismatched[:8])
        self.last_forward_mirror_report = report
        return report

    def forward_progress(self) -> dict[str, Any]:
        """The forward experiment's PROGRESS: unique parents, A/B/C grouped.

        WHAT THIS IS AND WHAT IT IS NOT
        -------------------------------
        It is a COUNTER of the forward sample: how many UNIQUE ELIGIBLE PARENT
        SIGNALS have produced a quote-based ALL-IN result, with each parent's
        three arms reported under the SAME parent id so A/B/C are three exit
        rules on ONE entry. It is NOT a performance result: the sample is tiny
        by construction at the start and no profitability claim is made at any
        sample size. The count is read from the DURABLE forward store, so it
        survives a restart and cannot be inflated in memory.

        An arm that could not be quoted stays in its parent's group as
        UNRESOLVED, with the reason, and is never counted as a settled trade.
        """
        # THE CLEAN COHORT, AND ONLY IT. A settlement labelled incident recovery
        # is reported separately and counted in NONE of the figures below --
        # not in completed_trades, not in by_arm, not in by_cost_model, not in
        # a parent's realized total. It stays visible in
        # ``incident_recovery_trades`` and in its own grouped list.
        trades = [rec for rec in self.forward_state.trades.values()
                  if not _is_incident_recovery(rec)]
        incident_trades = list(self.forward_state.incident_recovery.values())
        open_arms = list(self.forward_state.open_positions.values())

        def _pid(rec: Mapping[str, Any]) -> str:
            return str(rec.get("parent_signal_id") or rec.get("signal_id") or "")

        parents: dict[str, dict[str, Any]] = {}
        for rec in trades:
            pid = _pid(rec)
            if not pid:
                continue
            group = parents.setdefault(pid, {"arms": {}, "unresolved": {}})
            group["arms"][str(rec.get("arm") or "?")] = {
                "status": "closed",
                "exit_reason": rec.get("exit_reason"),
                "net_pnl_usd": rec.get("net_pnl_usd"),
                "net_return_pct": rec.get("net_return_pct"),
                "cost_model": cost_model_bucket(rec),
            }
        for rec in open_arms:
            pid = _pid(rec)
            if not pid:
                continue
            group = parents.setdefault(pid, {"arms": {}, "unresolved": {}})
            group["unresolved"][str(rec.get("arm") or "?")] = {
                "status": str(rec.get("status") or "open"),
                "exit_attempts": int(rec.get("exit_attempts") or 0),
                "reason": rec.get("exit_blocked_reason") or "",
            }
        grouped = []
        for pid in sorted(parents):
            g = parents[pid]
            settled_arms = sorted(g["arms"])
            grouped.append({
                "parent_signal_id": pid,
                "arms": {a: g["arms"].get(a) for a in BETA_ARMS},
                "arms_settled": settled_arms,
                "unresolved": {a: g["unresolved"].get(a) for a in BETA_ARMS},
                "all_three_arms_settled": settled_arms == list(BETA_ARMS),
                "realized_pnl_usd": round(sum(
                    float(v["net_pnl_usd"] or 0.0) for v in g["arms"].values()), 6),
            })
        by_arm: dict[str, dict[str, Any]] = {
            arm: {"completed_trades": 0, "realized_pnl_usd": 0.0}
            for arm in BETA_ARMS}
        by_cost_model: dict[str, dict[str, Any]] = {
            COST_MODEL_MODELED: {"completed_trades": 0, "realized_pnl_usd": 0.0},
            COST_MODEL_LEGACY: {"completed_trades": 0, "realized_pnl_usd": 0.0}}
        for rec in trades:
            arm = str(rec.get("arm") or "?")
            if arm in by_arm:
                by_arm[arm]["completed_trades"] += 1
                by_arm[arm]["realized_pnl_usd"] = round(
                    by_arm[arm]["realized_pnl_usd"]
                    + float(rec.get("net_pnl_usd") or 0.0), 6)
            b = by_cost_model[cost_model_bucket(rec)]
            b["completed_trades"] += 1
            b["realized_pnl_usd"] = round(
                b["realized_pnl_usd"] + float(rec.get("net_pnl_usd") or 0.0), 6)
        n_parents = len(grouped)
        incident_parents = {_pid(rec) for rec in incident_trades if _pid(rec)}
        return {
            "store": str(self.forward_state_store.path),
            "session_id": self.forward_session_id,
            "label": FORWARD_LABEL,
            "eligibility": FORWARD_ELIGIBILITY,
            "excludes": FORWARD_EXCLUDES + "; and " + INCIDENT_RECOVERY_EXCLUSION,
            "target_unique_parent_signals": FORWARD_TARGET_PARENT_SIGNALS,
            "unique_parent_signals": n_parents,
            "parents_with_all_three_arms_settled": sum(
                1 for g in grouped if g["all_three_arms_settled"]),
            "completed_trades": len(trades),
            "unresolved_arms": len(open_arms),
            "remaining_to_target": max(
                0, FORWARD_TARGET_PARENT_SIGNALS - n_parents),
            "progress_pct": round(
                n_parents / FORWARD_TARGET_PARENT_SIGNALS * 100.0, 2),
            "target_reached": n_parents >= FORWARD_TARGET_PARENT_SIGNALS,
            "by_arm": by_arm,
            "by_cost_model": by_cost_model,
            "grouped_by_parent": grouped,
            # INCIDENT RECOVERY: PRESERVED, COUNTED HERE, EXCLUDED FROM ABOVE.
            "incident_recovery_cohort": INCIDENT_RECOVERY_COHORT,
            "incident_recovery_trades": len(incident_trades),
            "incident_recovery_parents": len(incident_parents),
            "incident_recovery_realized_pnl_usd": round(sum(
                float(rec.get("net_pnl_usd") or 0.0)
                for rec in incident_trades), 6),
            "incident_recovery_records": [
                {"parent_signal_id": _pid(rec), "arm": rec.get("arm"),
                 "symbol": rec.get("symbol"),
                 "net_pnl_usd": rec.get("net_pnl_usd"),
                 "net_return_pct": rec.get("net_return_pct"),
                 "exit_reason": rec.get("exit_reason"),
                 "cohort": rec.get("cohort"),
                 "evidence": rec.get("incident_recovery")}
                for rec in incident_trades],
            "performance_claim": False,
            "note": (
                "PROGRESS COUNTER, NOT A PERFORMANCE RESULT. A parent signal "
                "counts ONCE however many of its three arms settle, because "
                "A/B/C are three exit rules on ONE entry. The 51 historical "
                "MODELLED trades in the main store are EXCLUDED by "
                "construction (they are neither quote-based nor ALL-IN) and are "
                "reported separately. Reaching the target is not evidence of "
                "edge and no profitability claim is made here at any sample "
                "size; an arm that could not be quoted remains UNRESOLVED and "
                "is never assumed to have exited. A settlement that spanned a "
                "restart while UNRESOLVED is reported as INCIDENT RECOVERY and "
                "is EXCLUDED from every figure above."),
        }

    @staticmethod
    def _position_key(pos: BetaPosition) -> str:
        """The durable key of one arm's open position: ``arm::signal_id``.

        Must match the key :class:`PersistedState` derives for a position dict,
        or the staleness sweep in :meth:`_persist` cannot find what to close.
        """
        return f"{pos.arm}::{pos.parent_signal_id}"

    @staticmethod
    def _position_dict(pos: BetaPosition) -> dict[str, Any]:
        return {
            "arm": pos.arm,
            "signal_id": pos.parent_signal_id,
            "mint": pos.mint,
            "symbol": pos.symbol,
            "entry_fill_price": pos.entry_fill_price,
            "position_size_usd": pos.position_size_usd,
            # The sizing decision and the cash it left, so a restart restores
            # the arm's committed capital (and therefore its AVAILABLE cash)
            # instead of believing the whole bankroll is still free.
            "allocated_usd": pos.allocated_usd,
            "entry_cost_usd": (pos.entry_cost_usd
                               if pos.entry_cost_usd is not None
                               else pos.position_size_usd),
            "remaining_cash_usd": pos.remaining_cash_usd,
            "sizing_pct": pos.sizing_pct,
            "sizing_rule": pos.sizing_rule,
            "entry_at": pos.entry_at,
            "peak_return_pct": pos.peak_return_pct,
            "trough_return_pct": pos.trough_return_pct,
            "last_return_pct": pos.last_return_pct,
            "status": pos.status,
            # THE EXECUTABLE TRIGGER SERIES, on the open record too. An
            # unresolved arm that is restarted must not lose the evidence it
            # already gathered, and a later reader must be able to see that the
            # TP/SL series was EMPTY rather than absent.
            "return_basis": pos.return_basis,
            "quote_peak_return_pct": pos.quote_peak_return_pct,
            "quote_trough_return_pct": pos.quote_trough_return_pct,
            "quote_last_return_pct": pos.quote_last_return_pct,
            "quote_peak_at": pos.quote_peak_at,
            "quote_trough_at": pos.quote_trough_at,
            "quote_observation_count": int(pos.quote_observation_count or 0),
            "quote_last_rejection_reason": pos.quote_last_rejection_reason,
            "mark_peak_return_pct": pos.mark_peak_return_pct,
            "mark_trough_return_pct": pos.mark_trough_return_pct,
            "trigger_return_pct": pos.trigger_return_pct,
            "trigger_basis": pos.trigger_basis,
            "trigger_at": pos.trigger_at,
            # the durable entry basis: an open position must reopen with the
            # quantity it acquired, or its exit could never be quoted exactly.
            "entry_basis": pos.entry_basis,
            "entry_tokens_raw": pos.entry_tokens_raw,
            "entry_token_decimals": pos.entry_token_decimals,
            "entry_cost_quote": pos.entry_cost_quote,
            "entry_fee_quote": pos.entry_fee_quote,
            "entry_quote_timestamp": pos.entry_quote_timestamp,
            "entry_quote_age_ms": pos.entry_quote_age_ms,
            "entry_venue": pos.entry_venue,
            "entry_pool_or_curve": pos.entry_pool_or_curve,
            "entry_provider": pos.entry_provider,
            "entry_route": list(pos.entry_route or ()),
            "entry_identity": pos.entry_identity,
            "exit_blocked_reason": pos.exit_blocked_reason,
            "exit_attempts": pos.exit_attempts,
            "exit_notified_attempt": pos.exit_notified_attempt,
            # The cost model and the entry provenance survive a restart too: a
            # restored open arm must still know what it reserved outside the
            # swap and what the market looked like at entry, or its settlement
            # would charge the wrong amount and its record would lose its source.
            "outside_swap_costs": dict(pos.outside_swap_costs or {}),
            "outside_swap_cost_usd": pos.outside_swap_cost_usd,
            "deployable_usd": pos.deployable_usd,
            "reserved_outside_swap_usd": pos.reserved_outside_swap_usd,
            "entry_provenance": dict(pos.entry_provenance or {}),
        }

    def exit_block_notifiable(self, pos: "BetaPosition") -> bool:
        """Whether an EXIT BLOCKED / UNRESOLVED notification is DUE for an arm.

        THE RULE, and it is the whole duplicate-prevention mechanism:

        * nothing is reported until an exit has actually been ATTEMPTED
          (``exit_attempts >= 1``) -- an open arm has no refusal to announce;
        * the FIRST refused attempt is always reported;
        * after that, only every ``BETA_EXIT_BLOCK_NOTIFY_EVERY``th attempt;
        * an attempt whose notification was already EMITTED -- recorded durably
          in ``exit_notified_attempt`` -- is never reported again, so a repeated
          monitor tick, a retry loop or a process restart cannot re-announce it.

        No price, no ledger effect: this is a bookkeeping predicate.
        """
        if pos.status != "unresolved":
            return False
        attempts = int(pos.exit_attempts or 0)
        if attempts < 1:
            return False
        if attempts == int(pos.exit_notified_attempt or 0):
            return False
        return attempts == 1 or attempts % BETA_EXIT_BLOCK_NOTIFY_EVERY == 0

    def mark_exit_block_notified(self, parent_signal_id: str, arm: str,
                                 attempt: int) -> bool:
        """Record that an EXIT BLOCKED notification was EMITTED for ``attempt``.

        Called ONLY after the notifier returned without raising: a send that
        FAILED must stay un-announced so the next tick retries it, while a send
        that succeeded must never be repeated. The marker is PERSISTED, which is
        what makes "no duplicate across a restart" a fact rather than an
        intention. It moves no ledger, books no price and does NOT settle the
        arm -- the position is still outstanding until a real SELL quote (or the
        flatten path) resolves it.
        """
        pos = self.positions.get(parent_signal_id, {}).get(arm)
        if pos is None or pos.status != "unresolved":
            return False
        pos.exit_notified_attempt = max(int(pos.exit_notified_attempt or 0),
                                        int(attempt))
        self._persist()
        return True

    def _restore_from_state(self) -> None:
        realized: dict[str, float] = {arm: 0.0 for arm in BETA_ARMS}
        for rec in self.state.trades.values():
            arm = rec.get("arm")
            if arm in realized and rec.get("net_pnl_usd") is not None:
                realized[arm] += float(rec["net_pnl_usd"])
                self.completed.append(rec)
        for arm, total in realized.items():
            self.accounts[arm].restore_settled_bankroll(total)
        for rec in self.state.open_positions.values():
            arm = rec.get("arm")
            sid = rec.get("signal_id")
            if arm not in BETA_ARMS or not sid:
                continue
            # ``unresolved`` survives a restart as unresolved: no quote had
            # arrived, no fill was booked, and the arm is still outstanding.
            # Restoring it as merely "open" would erase the fact that an exit
            # was already attempted and refused.
            restored_status = str(rec.get("status") or "open")
            if restored_status not in ("open", "unresolved"):
                restored_status = "open"
            pos = BetaPosition(
                parent_signal_id=sid, arm=arm,
                mint=rec.get("mint", ""), symbol=rec.get("symbol"),
                entry_fill_price=float(rec.get("entry_fill_price") or 0.0),
                entry_mark_price=None,
                position_size_usd=float(rec.get("position_size_usd") or 0.0),
                entry_at=float(rec.get("entry_at") or time.time()),
                peak_return_pct=float(rec.get("peak_return_pct") or 0.0),
                trough_return_pct=float(rec.get("trough_return_pct") or 0.0),
                last_return_pct=float(rec.get("last_return_pct") or 0.0),
                # The acquired quantity and its cost MUST survive a restart, or
                # the restored arm could never be quoted on exit and would be
                # forced into the mark path this design refuses.
                entry_basis=rec.get("entry_basis") or MARK_MODEL_BASIS,
                entry_tokens_raw=rec.get("entry_tokens_raw"),
                entry_token_decimals=rec.get("entry_token_decimals"),
                entry_cost_quote=rec.get("entry_cost_quote"),
                entry_fee_quote=rec.get("entry_fee_quote"),
                entry_quote_timestamp=rec.get("entry_quote_timestamp"),
                entry_quote_age_ms=rec.get("entry_quote_age_ms"),
                entry_venue=rec.get("entry_venue"),
                entry_pool_or_curve=rec.get("entry_pool_or_curve"),
                entry_provider=rec.get("entry_provider"),
                entry_route=tuple(rec.get("entry_route") or ()),
                entry_identity=rec.get("entry_identity") or "UNKNOWN",
                exit_blocked_reason=rec.get("exit_blocked_reason"),
                exit_attempts=int(rec.get("exit_attempts") or 0),
                # The announced-attempt marker survives a restart too, so a
                # recovered unresolved arm does NOT re-announce an exit block
                # the operator has already seen.
                exit_notified_attempt=int(
                    rec.get("exit_notified_attempt") or 0),
                # The sizing decision survives a restart too: without it a
                # restored arm would look like it never committed anything and
                # the all-in rule (and the report) would be wrong after every
                # process start.
                allocated_usd=rec.get("allocated_usd"),
                entry_cost_usd=rec.get("entry_cost_usd"),
                remaining_cash_usd=rec.get("remaining_cash_usd"),
                sizing_pct=rec.get("sizing_pct"),
                sizing_rule=rec.get("sizing_rule"),
                # The cost model survives a restart: an arm that reserved fees
                # + rent outside the swap must still be charged them at
                # settlement after the process restarted.
                outside_swap_costs=dict(rec.get("outside_swap_costs") or {}),
                outside_swap_cost_usd=rec.get("outside_swap_cost_usd"),
                deployable_usd=rec.get("deployable_usd"),
                reserved_outside_swap_usd=rec.get("reserved_outside_swap_usd"),
                entry_provenance=dict(rec.get("entry_provenance") or {}),
                status=restored_status,
            )
            if restored_status == "unresolved":
                # THIS ARM WAS ALREADY STRANDED WHEN THIS PROCESS LOADED IT.
                #
                # Its exit was due, its quotes were being refused and it was
                # being retried. Whatever eventually settles it therefore did
                # NOT settle on the exit rule's schedule, so counting it as a
                # clean forward result would report an infrastructure delay as
                # if the rule had produced it. The cohort label itself is
                # derived at settlement from ``exit_attempts`` (see
                # ``_trade_record``); what is captured HERE is the EVIDENCE that
                # the delay spanned a process: the attempt count and the refusal
                # the previous process recorded.
                #
                # Nothing is rewritten and nothing is dropped -- the main store
                # keeps the trade in full and the forward store keeps the
                # settlement under ``incident_recovery``.
                pos.delayed_across_restart = {
                    "restored_unresolved_at": time.time(),
                    "exit_attempts_at_restore": int(rec.get("exit_attempts") or 0),
                    "exit_reason_at_restore": rec.get("exit_reason"),
                    "refusal_at_restore": (rec.get("exit_blocked_reason") or "")[:300],
                }
            self.positions.setdefault(sid, {})[arm] = pos
            self._seen_signals.add(sid)

    def reset(self, archive: bool = True) -> dict[str, Any]:
        """
        Fresh session: settle outstanding arms, archive, reset 3 ledgers to $10.

        ORDER MATTERS. Open positions are FLATTENED first -- through the same
        no-fabricated-price settle the kill switch uses -- so every arm with an
        observation is closed at its last observed return and BOOKED INTO the
        session being archived, and an arm with no observation is refused and
        left recorded as open in the archive rather than dropped. Only then is
        the previous state archived and a fresh $10-per-arm session started.
        A reset therefore can never silently delete a trade, lose an unresolved
        position, or touch anything outside this experiment's own state file
        (L25 and every other profile live elsewhere and are not reachable from
        here).
        """
        flattened = self.flatten_all()
        summary = self.stats()
        prior_session_id = _session_id_of(self.state.as_dict()) or self.session_id
        archived_to = self.state_store.archive_current() if archive else None
        self.state = PersistedState()
        # A boundary, so the id must be DISTINCT from the session it ends -- see
        # ``_next_session_id``: two boundaries inside one second would otherwise
        # label two sessions identically.
        self.session_id = _next_session_id(prior_session_id)
        self.session_started_at = time.time()
        self.positions = {}
        self._seen_signals = set()
        self.completed = []
        self._init_accounts()
        self._persist()
        return {
            "archived_to": archived_to,
            "prior_session_summary": summary,
            "flattened_before_reset": flattened,
        }

    def rollover_session(self, *, reason: str, archive: bool = True) -> dict[str, Any]:
        """
        Open a NEW isolated $10-per-arm session and PRESERVE this one intact.

        WHY THIS IS NOT :meth:`reset`
        =============================
        ``reset`` FLATTENS first: it books every OBSERVED arm closed at its LAST
        OBSERVED return. That is right for a kill switch or a profile switch,
        where the outstanding arms belong to the session being ended. It is
        WRONG for a rollover whose outstanding arms have NO executable exit at
        all: it would stamp an exit price and a P&L onto a position whose market
        never paid one, which is the single thing this harness exists to
        prevent. Every argument for a rollover -- "the previous session must stay
        auditable", "the unresolved position must stay unresolved" -- is an
        argument AGAINST flattening.

        So this method prices NOTHING, settles NOTHING and deletes NOTHING. It:

        1. ARCHIVES the whole session file verbatim -- every historical modelled
           trade, every incident-recovery record and every unresolved position
           with its exact acquired quantity, its venue pin and its full refusal
           history -- under a timestamped sibling name;
        2. archives the forward store the same way, so the clean forward cohort
           also begins empty instead of inheriting the previous sample;
        3. starts the three ledgers at ``starting_bankroll_usd`` with no
           positions, no completed trades and NO inherited realized P&L;
        4. writes the outstanding arms into the NEW state's ``incident_recovery``
           section as explicitly NON-SETTLED incident records: no exit price, no
           proceeds, ``net_pnl_usd`` None, and a flag saying so. They are not
           trades, they are not counted by any figure, and -- because the new
           session's ``open_positions`` is EMPTY -- they can never settle into
           this session's performance.

        WHAT IS DELIBERATELY PRESERVED IN CODE (not just in the archive): the
        per-arm ledgers stay independent, ALL-IN sizing still commits 100% of
        each arm's own deployable cash, the affordability checks are untouched,
        an outstanding position still BLOCKS its own arm's next entry
        (``arm_entry_plan``), and the entry and exit rules are unchanged. Only
        the SESSION's starting capital and its inherited P&L are new.

        Refuses to run while the session still holds a SETTLED result it has not
        persisted, so a rollover can never race a write.
        """
        prior_session = self.session_id
        prior_forward_session = self.forward_session_id
        prior_summary = self.stats()
        prior_payload = self.state.as_dict()
        prior_session_id = _session_id_of(prior_payload) or prior_session
        outstanding = self._outstanding_incidents()

        archived_to = self.state_store.archive_current() if archive else None
        archived_forward_to = (
            self.forward_state_store.archive_current() if archive else None)

        # THE NEW SESSION. A brand-new PersistedState: no trades, no signals, no
        # open positions, so ``_restore_from_state`` has nothing to revive and
        # each ledger starts at its own starting bankroll.
        self.state = PersistedState()
        self.session_id = _next_session_id(prior_session_id)
        self.session_started_at = time.time()
        self.positions = {}
        self._seen_signals = set()
        self.completed = []
        self._init_accounts()

        self.forward_state = PersistedState()
        # STABLE ACROSS RESTARTS, and now also stable across a ROLLOVER: it is
        # derived from the NEW forward store's own creation stamp, so the fresh
        # sample gets a fresh, durable id rather than the previous one's.
        self.forward_session_id = f"beta_fwd_{int(self.forward_state.created_at)}"
        if self.forward_session_id == prior_forward_session:
            # THE STAMP IS SECOND-RESOLUTION, so two sessions created inside one
            # second would share a label and be indistinguishable in a report --
            # which would silently pool two samples under one id. A rollover is
            # a session BOUNDARY, so the stamp is advanced to the next second
            # rather than the id being re-randomised: the ids stay ordered and
            # comparable, which a hash would destroy.
            self.forward_state.created_at = float(
                int(self.forward_state.created_at) + 1)
            self.forward_session_id = (
                f"beta_fwd_{int(self.forward_state.created_at)}")

        # INCIDENTS CARRIED BY AN EARLIER BOUNDARY ARE PART OF THIS SESSION'S
        # AUDIT TRAIL, so they are copied forward rather than dropped.
        #
        # ``self.state`` is a brand-new ``PersistedState``, so anything not
        # written here is GONE from the live file -- still in the archive, but no
        # longer in the file the operator, the report and the incident counter
        # read. A SECOND rollover used to therefore end with an EMPTY
        # ``incident_recovery``: the unresolved positions the file had just
        # carried over were erased by the very next boundary, which is precisely
        # what a carry-over must not do.
        #
        # The records are copied VERBATIM: no field is rewritten, because
        # ``session_id`` and ``carried_into_session`` are provenance and name the
        # boundary at which the incident entered the experiment. The only thing
        # added is the list of boundaries it has since survived, so a reader of
        # the current file can tell a still-open incident from a stale one.
        prior_incidents = [r for r in (prior_payload.get("incident_recovery") or ())
                           if isinstance(r, Mapping)]
        for rec in prior_incidents:
            key = f"{rec.get('parent_signal_id', '')}::{rec.get('arm', '')}::"
            kept = dict(rec)
            kept["carried_forward_through"] = list(
                kept.get("carried_forward_through") or []) + [self.session_id]
            self.state.incident_recovery.setdefault(key, kept)

        # THE INCIDENT RECORD. Written for EVERY arm that was still outstanding,
        # including the unresolved positions carried from an earlier incident.
        # It is a RECORD OF NOT SETTLING: no price, no proceeds, no P&L, and an
        # explicit flag so no reader has to infer it from the missing fields.
        for row in outstanding:
            key = f"{row['parent_signal_id']}::{row['arm']}::"
            self.state.incident_recovery[key] = {
                "record_type": "beta_unresolved_incident",
                "parent_signal_id": row["parent_signal_id"],
                "arm": row["arm"],
                "mint": row["mint"],
                "symbol": row["symbol"],
                "session_id": prior_session_id,
                "carried_into_session": self.session_id,
                "status": "UNRESOLVED_INCIDENT",
                "settled": False,
                "counted_as_trade": False,
                "exit_price": None,
                "exit_proceeds_usd": None,
                "net_pnl_usd": None,
                "entry_at": row["entry_at"],
                "entry_cost_usd": row["entry_cost_usd"],
                "entry_tokens_raw": row["entry_tokens_raw"],
                "entry_token_decimals": row["entry_token_decimals"],
                "entry_venue": row["entry_venue"],
                "entry_pool_or_curve": row["entry_pool_or_curve"],
                "exit_attempts": row["exit_attempts"],
                "exit_blocked_reason": row["exit_blocked_reason"],
                "note": (
                    "carried over UNRESOLVED with no fabricated exit price and "
                    "no P&L; it is preserved in the archived session and is "
                    "EXCLUDED from this session's performance"),
                "archived_session": archived_to,
            }

        self.state.write_log.append({
            "at": time.time(), "event": "session_rollover",
            "reason": str(reason),
            "prior_session_id": prior_session_id,
            "new_session_id": self.session_id,
            "archived_to": archived_to,
            "archived_forward_to": archived_forward_to,
            "prior_forward_session_id": prior_forward_session,
            "new_forward_session_id": self.forward_session_id,
            "starting_bankroll_usd_per_arm": self.starting_bankroll_usd,
            "outstanding_carried_as_incidents": len(outstanding),
            "incidents_carried_forward": len(prior_incidents),
            "flattened": False,
            "prior_counts": prior_payload.get("counts"),
        })
        self.forward_state.write_log.append({
            "at": time.time(), "event": "session_rollover",
            "reason": str(reason),
            "prior_forward_session_id": prior_forward_session,
            "new_forward_session_id": self.forward_session_id,
            "archived_forward_to": archived_forward_to,
            "note": "the clean forward cohort starts empty; the previous "
                    "sample is preserved in the archived file",
        })
        self._persist()
        return {
            "reason": str(reason),
            "prior_session_id": prior_session_id,
            "new_session_id": self.session_id,
            "prior_forward_session_id": prior_forward_session,
            "new_forward_session_id": self.forward_session_id,
            "archived_to": archived_to,
            "archived_forward_to": archived_forward_to,
            "prior_summary": prior_summary,
            "outstanding_carried": outstanding,
            "flattened": False,
            "bankroll_per_arm_after": {
                arm: self.accounts[arm].bankroll_usd for arm in BETA_ARMS},
            "total_starting_capital_usd": self.total_starting_capital_usd,
        }

    def _outstanding_incidents(self) -> list[dict[str, Any]]:
        """Every arm still open or unresolved, described WITHOUT pricing it.

        Deliberately excludes any return, mark or price: the caller of this is
        the rollover, and the whole point of that path is that no number is
        invented for an arm that never got a fill.
        """
        rows: list[dict[str, Any]] = []
        for pid in sorted(self.positions):
            for arm in BETA_ARMS:
                pos = self.positions.get(pid, {}).get(arm)
                if pos is None or pos.status not in ("open", "unresolved"):
                    continue
                rows.append({
                    "parent_signal_id": pid, "arm": arm,
                    "mint": pos.mint, "symbol": pos.symbol,
                    "status": pos.status,
                    "entry_at": pos.entry_at,
                    "entry_cost_usd": (pos.entry_cost_usd
                                       if pos.entry_cost_usd is not None
                                       else pos.position_size_usd),
                    "entry_tokens_raw": pos.entry_tokens_raw,
                    "entry_token_decimals": pos.entry_token_decimals,
                    "entry_venue": pos.entry_venue,
                    "entry_pool_or_curve": pos.entry_pool_or_curve,
                    "exit_attempts": int(pos.exit_attempts or 0),
                    "exit_blocked_reason": pos.exit_blocked_reason,
                })
        return rows

    def flatten_all(self, now: float | None = None) -> dict[str, Any]:
        """
        EMERGENCY/PROFILE-SWITCH flatten of every open arm.

        This is the beta counterpart to the harness's mandatory flatten
        protection: an arm is never left open and unaccounted because the
        operator hit kill, or because they selected a different strategy while a
        beta position was running.

        IT DOES NOT INVENT A PRICE
        ---------------------------
        The exit basis is the LAST OBSERVED return -- the same basis a time exit
        uses -- and an arm with no observation at all is REFUSED, reported as
        still-open and left for the next tick rather than closed at a fabricated
        number. A made-up fill is exactly what this harness exists to prevent.

        Every flattened trade is stamped ``FLATTEN`` as its exit reason, so the
        forced exit is visible in the ledger and in execution-quality reporting
        instead of being indistinguishable from an ordinary time exit.
        """
        now = time.time() if now is None else float(now)
        closed: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        for pid in list(self.positions.keys()):
            for arm in BETA_ARMS:
                pos = self.positions.get(pid, {}).get(arm)
                if pos is None or pos.status not in ("open", "unresolved"):
                    continue
                if pos.last_observed_at <= 0:
                    # NO PRICE, NO TRADE. The position stays open and is
                    # reported so it cannot be silently dropped.
                    unresolved.append({
                        "parent_signal_id": pid, "arm": arm,
                        "mint": pos.mint, "reason": "NO_OBSERVATION",
                    })
                    continue
                gross = pos.last_return_pct / 100.0
                net = net_from_gross(gross)
                pos.exit_reason = "FLATTEN"
                pos.exit_fill_price = pos.entry_fill_price * (1.0 + gross)
                pos.exit_at = now
                pos.gross_return_pct = gross * 100.0
                pos.net_return_pct = net * 100.0
                pos.net_pnl_usd = pos.position_size_usd * net
                pos.status = "closed"
                account = self.accounts[arm]
                account.bankroll_usd = round(
                    account.bankroll_usd + pos.net_pnl_usd, 6)
                account.peak_bankroll_usd = max(
                    account.peak_bankroll_usd, account.bankroll_usd)
                account.position = None
                account.closed_trades += 1
                self.completed.append(self._trade_record(pos))
                closed.append({
                    "parent_signal_id": pid, "arm": arm,
                    "mint": pos.mint,
                    "exit_fill_price": pos.exit_fill_price,
                    "net_return_pct": pos.net_return_pct,
                    "net_pnl_usd": pos.net_pnl_usd,
                    "exit_reason": "FLATTEN",
                })
        self._persist()
        # Mirrors are dropped by the caller; the experiment's own book no longer
        # holds any of the arms settled here, because ``settle``/this method
        # already moved them to ``closed``.
        return {"closed": closed, "unresolved": unresolved}

    def arm_stats(self, arm: str) -> dict[str, Any]:
        cfg = arm_config(arm)
        acct = self.accounts[arm]
        closed = [t for t in self.completed
                  if t.get("arm") == arm and t.get("status") == "closed"]
        open_n = sum(
            1 for pid in self.positions.values()
            if arm in pid and pid[arm].status == "open"
        )
        unresolved_n = sum(
            1 for pid in self.positions.values()
            if arm in pid and pid[arm].status == "unresolved"
        )
        n = len(closed)
        nets = [float(t["net_return_pct"]) for t in closed]
        usd = [float(t["net_pnl_usd"]) for t in closed]
        wins = [x for x in nets if x > 0]
        losses = [x for x in nets if x <= 0]
        gross_win = sum(wins)
        gross_loss = -sum(losses)
        pf = gross_win / gross_loss if gross_loss > 0 else None
        # SPLIT THE REALIZED NUMBER BY ACCOUNTING BASIS. An arm's history can
        # span the model change (pre-quote MODELLED trades and new quote-based
        # ones); a single summed $ figure would present historical modelled P&L
        # as if it were quote-based, so the split is always carried alongside.
        realized_by_basis: dict[str, dict[str, Any]] = {
            BASIS_QUOTE: {"completed_trades": 0, "realized_pnl_usd": 0.0},
            BASIS_MODELED: {"completed_trades": 0, "realized_pnl_usd": 0.0},
        }
        for t in closed:
            bucket = realized_by_basis[basis_bucket(t)]
            bucket["completed_trades"] += 1
            bucket["realized_pnl_usd"] = round(
                bucket["realized_pnl_usd"] + float(t.get("net_pnl_usd") or 0.0), 4)
        # THE SIZING SPLIT, for the same reason as the basis split: this arm's
        # history spans the sizing change, and the 10%-of-equity trades that
        # came before must not be presented as ALL-IN results.
        sizing_split: dict[str, dict[str, Any]] = {
            SIZING_BUCKET_ALL_IN: {"completed_trades": 0, "realized_pnl_usd": 0.0},
            SIZING_BUCKET_EXPLICIT: {"completed_trades": 0, "realized_pnl_usd": 0.0},
            SIZING_BUCKET_LEGACY: {"completed_trades": 0, "realized_pnl_usd": 0.0},
        }
        for t in closed:
            b = sizing_split[sizing_bucket(t)]
            b["completed_trades"] += 1
            b["realized_pnl_usd"] = round(
                b["realized_pnl_usd"] + float(t.get("net_pnl_usd") or 0.0), 4)
        # WHAT THIS ARM CAN TRADE WITH RIGHT NOW: the settled bankroll less what
        # its outstanding position committed. It is the all-in budget for the
        # next entry, so it belongs on the report next to the sizing rule.
        held = self.outstanding_arms().get(arm)
        committed = (float(held.entry_cost_usd
                           if held is not None
                           and held.entry_cost_usd is not None
                           else (held.position_size_usd if held is not None else 0.0))
                     if held is not None else 0.0)
        available_cash = max(0.0, float(acct.bankroll_usd) - committed)
        tp = sum(1 for t in closed if t.get("exit_reason") == "TP")
        sl = sum(1 for t in closed if t.get("exit_reason") == "SL")
        tm = sum(1 for t in closed if t.get("exit_reason") == "TIME")
        amb = sum(1 for t in closed if t.get("path_ambiguous"))
        equity = acct.starting_bankroll_usd
        peak = equity
        max_dd = 0.0
        for p in usd:
            equity += p
            peak = max(peak, equity)
            if peak > 0:
                max_dd = min(max_dd, (equity - peak) / peak * 100.0)
        streak = 0
        longest = 0
        for x in nets:
            if x <= 0:
                streak += 1
                longest = max(longest, streak)
            else:
                streak = 0
        # UNREALIZED, from the LAST OBSERVED return of this arm's open
        # positions -- priced on the same net-of-exit-friction basis as a
        # settled trade. An open arm that has never been observed contributes
        # nothing and is counted in ``unrealized_unpriced`` rather than being
        # shown as a confident $0: no observation, no price.
        unrealized = 0.0
        unrealized_priced = 0
        unrealized_unpriced = 0
        for pid in self.positions.values():
            pos = pid.get(arm)
            if pos is None or pos.status != "open":
                continue
            if pos.last_observed_at <= 0:
                unrealized_unpriced += 1
                continue
            unrealized += pos.position_size_usd * net_from_gross(
                pos.last_return_pct / 100.0)
            unrealized_priced += 1
        # -- COST COVERAGE AND QUOTE TELEMETRY, PER ARM --------------------
        #
        # The fee breakdown the audit asks for, taken from the RECORDS rather
        # than recomputed, so it reflects what was actually charged. A record
        # with ``{}`` costs is counted as LEGACY: it was charged only the swap
        # quote's own fee, and that fact is reported instead of being averaged
        # away into a total that looks fully costed.
        cost_models: dict[str, int] = {COST_MODEL_MODELED: 0, COST_MODEL_LEGACY: 0}
        charged_usd = 0.0
        reserve_usd = 0.0
        fees_lamports = 0
        rent_lamports = 0
        swap_fee_quote = 0.0
        quote_latencies: list[float] = []
        quote_ages: list[float] = []
        for t in closed:
            cost_models[cost_model_bucket(t)] += 1
            charged_usd += float(t.get("outside_swap_charged_usd") or 0.0)
            costs = t.get("outside_swap_costs")
            if isinstance(costs, Mapping) and costs:
                fees_lamports += int(costs.get("non_recoverable_lamports") or 0)
                rent_lamports += int(costs.get("recoverable_lamports") or 0)
                reserve_usd += float(costs.get("total_reserve_usd") or 0.0)
            swap_fee_quote += (float(t.get("entry_fee_quote") or 0.0)
                               + float(t.get("exit_fee_quote") or 0.0))
            for key in ("entry_provenance", "exit_provenance"):
                prov = t.get(key)
                if not isinstance(prov, Mapping):
                    continue
                lat = prov.get("quote_latency_ms")
                if isinstance(lat, (int, float)):
                    quote_latencies.append(float(lat))
                age = prov.get("quote_age_ms")
                if isinstance(age, (int, float)):
                    quote_ages.append(float(age))
        # QUOTE FAILURES: recorded FAILED exit attempts, on settled records and
        # on arms still outstanding. ``exit_attempts`` counts only the attempts
        # that produced no valid quote, so this is a failure count and not a
        # request count.
        quote_failures = 0
        for t in closed:
            quote_failures += int(t.get("exit_attempts") or 0)
        for pid in self.positions.values():
            pos = pid.get(arm)
            if pos is None:
                continue
            quote_failures += int(pos.exit_attempts or 0)

        def _r(v, nd=3):
            if v is None:
                return None
            return round(float(v), nd)

        def _mean(xs: list[float], nd: int = 3) -> float | None:
            return round(sum(xs) / len(xs), nd) if xs else None

        return {
            "arm": arm,
            "take_profit_pct": cfg.take_profit_pct,
            "stop_loss_pct": cfg.stop_loss_pct,
            "max_hold_s": cfg.max_hold_s,
            "starting_bankroll_usd": round(acct.starting_bankroll_usd, 2),
            "bankroll_usd": round(acct.bankroll_usd, 4),
            "equity_usd": round(acct.bankroll_usd, 4),
            "realized_pnl_usd": round(sum(usd), 4),
            "realized_basis_note": (
                "all-basis total; do NOT read as quote-based performance -- "
                "see realized_by_basis"),
            "realized_by_basis": realized_by_basis,
            "realized_by_sizing": sizing_split,
            # -- ALL-IN SIZING, ON THE ARM ----------------------------------
            "sizing_rule": BETA_SIZING_RULE,
            "sizing_pct": BETA_SIZING_PCT,
            "available_bankroll_usd": round(available_cash, 4),
            "capital_committed_usd": round(committed, 4),
            "entry_cost_usd": (round(float(held.entry_cost_usd), 4)
                               if held is not None
                               and held.entry_cost_usd is not None else None),
            "allocated_usd": (round(float(held.allocated_usd), 4)
                              if held is not None
                              and held.allocated_usd is not None else None),
            "remaining_cash_usd": (round(float(held.remaining_cash_usd), 4)
                                   if held is not None
                                   and held.remaining_cash_usd is not None
                                   else None),
            "position_sizing_pct": (float(held.sizing_pct)
                                    if held is not None
                                    and held.sizing_pct is not None else None),
            "realized_pnl_pct": _r(sum(usd) / acct.starting_bankroll_usd * 100.0
                                  if acct.starting_bankroll_usd else None, 3),
            "unrealized_pnl_usd": round(unrealized, 4),
            "unrealized_priced_positions": unrealized_priced,
            "unrealized_unpriced_positions": unrealized_unpriced,
            "entries": n + open_n,
            "completed_trades": n,
            "open_positions": open_n,
            "unresolved": unresolved_n,
            "win_rate_pct": _r(len(wins) / n * 100.0 if n else None, 2),
            "expectancy_pct": _r(sum(nets) / n if n else None, 4),
            "profit_factor": _r(pf, 4),
            "max_drawdown_pct": _r(max_dd, 3) if n else None,
            "avg_win_pct": _r(sum(wins) / len(wins) if wins else None, 3),
            "avg_loss_pct": _r(sum(losses) / len(losses) if losses else None, 3),
            "longest_losing_streak": longest if n else None,
            "exits": {
                "TP": tp,
                "SL": sl,
                "TIME": tm,
                "path_ambiguous": amb,
            },
            # -- FEES, RESERVES AND LATENCY, HONESTLY SPLIT -----------------
            # Costs paid OUTSIDE the swap: charged to P&L (the two legs' base
            # and priority fees) vs merely RESERVED (the token-account rent,
            # a recoverable deposit that is never charged as a loss).
            "outside_swap_costs_usd": round(charged_usd, 6),
            "outside_swap_reserve_usd": round(reserve_usd, 6),
            "outside_swap_fees_lamports": fees_lamports,
            "outside_swap_rent_reserved_lamports": rent_lamports,
            "outside_swap_rent_recoverable": True,
            "outside_swap_cost_model": dict(cost_models),
            "outside_swap_cost_model_note": (
                "legacy = charged the swap quote's own fee only; the Solana base "
                "fee, the priority fee and the token-account rent were NOT "
                "modelled. Legacy and modelled results must never be summed "
                "into one fully-costed number."),
            # The venue's own fee, which lives INSIDE the quoted input. Reported
            # separately so it is visible next to the outside-swap costs.
            "swap_fee_quote_total": round(swap_fee_quote, 9),
            "quote_telemetry": {
                "quote_failures": quote_failures,
                "unresolved_open": unresolved_n,
                "mean_quote_latency_ms": _mean(quote_latencies),
                "mean_quote_age_ms": _mean(quote_ages),
                "priced_quotes": len(quote_latencies),
            },
            "basis": "bankroll moves only on realized settlement between two "
                     "SIMULATED fills; never marked to the observed market",
        }

    def _v2_gate_result(self, signal: SignalSnapshot) -> tuple[bool, str]:
        """Run the production V2 FLOOR gate once; return (eligible, reason)."""
        vol = getattr(signal, "v2_volume_5m_usd", None)
        chg = getattr(signal, "v2_change_5m_pct", None)
        mcap = getattr(signal, "v2_market_cap_usd", None)
        if vol is None or chg is None or mcap is None:
            return False, "V2_FLOOR gate requires v2_volume_5m_usd, v2_change_5m_pct, v2_market_cap_usd"
        result = v2floor_evaluate(
            volume_5m_usd=vol,
            change_5m_pct=chg,
            market_cap_usd=mcap,
        )
        if result.authorized:
            return True, "V2_FLOOR gate passed"
        return False, f"V2_FLOOR refused: {result.detail}"

    def _stable_parent_id(self, signal: SignalSnapshot) -> str:
        """Deterministic parent signal ID from channel+mint+timestamp."""
        base = f"{signal.channel}:{signal.token_mint}:{signal.telegram_timestamp}"
        return hashlib.sha256(base.encode()).hexdigest()[:24]

    # -- ALL-IN SIZING + THE OVERLAP RULE --------------------------------

    def outstanding_arms(self) -> dict[str, "BetaPosition"]:
        """arm -> its ONE outstanding position (open or unresolved).

        "Outstanding" is the operative word: an arm that could not be quoted on
        exit is still holding, so it is outstanding until it settles.
        """
        out: dict[str, BetaPosition] = {}
        for by_arm in self.positions.values():
            for arm, pos in by_arm.items():
                if pos.status in ("open", "unresolved"):
                    out[arm] = pos
        return out

    def committed_capital_usd(self, arm: str) -> float:
        """USD this arm has committed to its outstanding position (0 if none)."""
        pos = self.outstanding_arms().get(arm)
        if pos is None:
            return 0.0
        return float(pos.entry_cost_usd
                     if pos.entry_cost_usd is not None
                     else pos.position_size_usd or 0.0)

    def available_cash_usd(self, arm: str) -> float:
        """
        This arm's UNCOMMITTED cash -- the budget its next all-in entry spends.

        ``bankroll_usd`` is the SETTLED total and deliberately does not move
        while a position is open (it only ever changes on realized settlement).
        So available cash is the settled bankroll MINUS what the outstanding
        position committed. Under an all-in rule those are equal, and the arm has
        nothing left to trade with -- which is the same fact as the overlap rule.
        """
        return max(0.0, float(self.accounts[arm].bankroll_usd)
                   - self.committed_capital_usd(arm))

    def all_in_notional_usd(self, arm: str) -> float:
        """
        The ALL-IN notional for this arm: 100% of its uncommitted cash.

        The ONE definition of the sizing rule. The 10%-of-equity version this
        replaced is what the experiment must never silently regress to, and a
        single call site is what makes that checkable.
        """
        return self.available_cash_usd(arm)

    def arm_entry_plan(self, arms: tuple[str, ...] | None = None
                       ) -> dict[str, dict[str, Any]]:
        """
        Per-arm readiness AND sizing for one new entry, taken BEFORE any quote.

        THE OVERLAP RULE LIVES HERE. Every arm is all-in, so an arm that still
        holds an open or unresolved position has no free capital and is BLOCKED
        (with its reason) until that position is resolved. The other arms are
        untouched: they are independent ledgers, and an eligible arm must never
        wait on an arm that is stuck.
        """
        keys = tuple(arms) if arms is not None else BETA_ARMS
        outstanding = self.outstanding_arms()
        plan: dict[str, dict[str, Any]] = {}
        for arm in keys:
            bankroll = float(self.accounts[arm].bankroll_usd)
            held = outstanding.get(arm)
            committed = (float(held.entry_cost_usd
                               if held.entry_cost_usd is not None
                               else held.position_size_usd or 0.0)
                         if held is not None else 0.0)
            available = max(0.0, bankroll - committed)
            blocked = False
            reason = ""
            if held is not None:
                blocked = True
                reason = (
                    f"OVERLAP: arm {arm} still holds a {held.status} position "
                    f"({held.parent_signal_id}); an all-in arm takes no second "
                    "entry until the first is resolved")
            elif available <= BETA_MIN_ENTRY_USD:
                blocked = True
                reason = (
                    f"INSUFFICIENT_CAPITAL: arm {arm} has ${available:.9f} "
                    "uncommitted, so there is nothing to go all-in with")
            plan[arm] = {
                "arm": arm,
                "bankroll_usd": round(bankroll, 9),
                "committed_capital_usd": round(committed, 9),
                "available_cash_usd": round(available, 9),
                "notional_usd": (round(available, 9) if held is None else 0.0),
                "sizing_pct": BETA_SIZING_PCT if held is None else 0.0,
                "sizing_rule": SIZING_ALL_IN,
                "blocked": blocked,
                "reason": reason,
            }
        return plan

    def shared_liquidity_scenario(self) -> dict[str, Any]:
        """
        The A/B/C AGGREGATE demand of one signal, reported SEPARATELY.

        THE BLIND SPOT THIS NAMES. The three arms are independent ledgers and
        are NEVER priced as one combined order: each is quoted at its OWN
        all-in notional, and two arms that happen to hold the same budget SHARE
        one quote snapshot for that same size (the engine's single-flight
        coalescing). What no per-arm quote can therefore show is the impact of
        the three arms acting on the SAME mint at the same instant: the pool
        absorbs the SUM of their notionals, while each arm's quote prices only
        its own share -- so per-arm quotes UNDERSTATE the combined market impact.

        This is an AGGREGATE STRESS VIEW, not a result. It moves no ledger,
        prices nothing, and is deliberately kept out of every per-arm number.
        """
        plan = self.arm_entry_plan()
        free = [a for a in BETA_ARMS if not plan[a]["blocked"]]
        blocked = [a for a in BETA_ARMS if plan[a]["blocked"]]
        total = sum(float(plan[a]["available_cash_usd"]) for a in BETA_ARMS)
        at_once = sum(float(plan[a]["available_cash_usd"]) for a in free)
        per_arm = (at_once / len(free)) if free else 0.0
        return {
            "arms_free": free,
            "arms_blocked": blocked,
            "aggregate_available_usd": round(total, 9),
            "simultaneous_available_usd": round(at_once, 9),
            "per_arm_available_usd": round(per_arm, 9),
            "aggregate_multiple_of_one_arm": (
                round(at_once / per_arm, 6) if per_arm > 0 else None),
            "is_a_result": False,
            "note": (
                "AGGREGATE STRESS VIEW, NOT A RESULT. The arms are independent "
                "ledgers and are never priced as one order: each is quoted at "
                "its OWN all-in notional, and two arms holding the same budget "
                "share one quote snapshot for that same size. Acting on the "
                "same mint at the same instant, the pool absorbs their SUM while "
                "each arm's quote prices only its own share, so per-arm quotes "
                "UNDERSTATE the combined market impact. The deployable "
                "aggregate is these amounts less each arm's own outside-swap "
                "reserve (see the arm reports). This block moves no ledger."),
        }

    def _create_arms(self, *, parent_signal_id: str, mint: str,
                     symbol: str | None, arms: tuple[str, ...],
                     fills: dict[str, "BetaEntryFill"],
                     allocations: dict[str, float],
                     sizing_rule: str, sizing_pct: float | None,
                     entry_at: float) -> dict[str, BetaPosition]:
        """Materialise the arms that are actually opening, then persist once.

        ``allocated_usd`` is the sizing DECISION (100% of the arm's cash on the
        all-in path), ``entry_cost_usd`` is what the quote actually took, and
        ``remaining_cash_usd`` records the difference rather than rounding it
        away. ``position_size_usd`` -- the USD base every later P&L is
        measured against -- is the COMMITTED cost, not the intention.
        """
        created: dict[str, BetaPosition] = {}
        for arm in arms:
            fill = fills[arm]
            allocated = float(allocations[arm])
            cost = float(fill.usd_cost)
            # THE RESERVE. An ALL-IN arm commits 100% of its available cash to
            # the SWAP INPUT, but the wallet must also hold the fees and the
            # token-account rent BESIDE it or the BUY cannot execute at all. The
            # difference is recorded, not spent: it is capital held back, of
            # which only the non-recoverable part (both legs' fees) is later
            # charged to P&L and the rent is a recoverable deposit.
            outside = fill.outside_costs
            charged = float(outside.charged_to_pnl_usd) if outside else 0.0
            reserve = max(0.0, allocated - cost)
            created[arm] = BetaPosition(
                parent_signal_id=parent_signal_id, arm=arm, mint=mint,
                symbol=symbol,
                entry_fill_price=float(fill.price_usd),
                entry_mark_price=fill.mark_price,
                position_size_usd=cost,
                entry_at=entry_at,
                allocated_usd=allocated,
                entry_cost_usd=cost,
                remaining_cash_usd=reserve,
                deployable_usd=cost,
                reserved_outside_swap_usd=reserve,
                outside_swap_cost_usd=(charged if outside else None),
                outside_swap_costs=(outside.as_dict() if outside else {}),
                entry_provenance=dict(fill.provenance or {}),
                sizing_pct=sizing_pct,
                sizing_rule=sizing_rule,
                entry_basis=fill.basis,
                entry_tokens_raw=fill.tokens_raw,
                entry_token_decimals=fill.token_decimals,
                entry_cost_quote=fill.cost_quote,
                entry_fee_quote=fill.fee_quote,
                entry_quote_timestamp=fill.quote_timestamp,
                entry_quote_age_ms=fill.quote_age_ms,
                entry_venue=fill.venue,
                entry_pool_or_curve=fill.pool_or_curve,
                entry_provider=fill.provider,
                entry_route=fill.route,
                entry_identity=fill.identity,
            )
        self.positions[parent_signal_id] = created
        self._seen_signals.add(parent_signal_id)
        self._persist()
        return created

    def open_arms(
        self, *, parent_signal_id: str, mint: str, symbol: str | None,
        entry_fills: dict[str, "BetaEntryFill"],
        allocations: dict[str, float] | None = None,
        entry_at: float | None = None,
        sizing_pct: float = BETA_SIZING_PCT,
    ) -> tuple[dict[str, BetaPosition], dict[str, str]]:
        """
        THE PRODUCTION ENTRY: open every arm that is ready AND has a fill.

        Returns ``(opened, blocked)``. An arm is blocked when it already holds
        an open/unresolved position (OVERLAP), when it has no spare capital, or
        when the BUY quote offered for it costs more than that arm's available
        cash. A blocked arm does not open and does not affect any other arm.

        Idempotent: re-delivering the same parent signal returns the existing
        book instead of opening a second time.
        """
        if parent_signal_id in self._seen_signals:
            return dict(self.positions.get(parent_signal_id, {})), {}
        now = time.time() if entry_at is None else float(entry_at)
        plan = self.arm_entry_plan()
        fills: dict[str, BetaEntryFill] = {}
        alloc: dict[str, float] = {}
        blocked: dict[str, str] = {}
        for arm in BETA_ARMS:
            row = plan[arm]
            if row["blocked"]:
                blocked[arm] = row["reason"]
                continue
            fill = entry_fills.get(arm)
            if fill is None:
                blocked[arm] = "no executable BUY quote for this arm"
                continue
            allocation = float(
                (allocations or {}).get(arm, row["available_cash_usd"]))
            try:
                fill.validate(allocation_usd=allocation)
            except ValueError as exc:
                blocked[arm] = f"AFFORDABILITY: {exc}"
                continue
            fills[arm] = fill
            alloc[arm] = allocation
        if not fills:
            return {}, blocked
        created = self._create_arms(
            parent_signal_id=parent_signal_id, mint=mint, symbol=symbol,
            arms=tuple(fills), fills=fills, allocations=alloc,
            sizing_rule=SIZING_ALL_IN, sizing_pct=float(sizing_pct),
            entry_at=now,
        )
        return created, blocked

    def process_signal(
        self,
        signal: SignalSnapshot,
        *,
        entry_fill: "BetaEntryFill",
    ) -> tuple[bool, str, dict[str, BetaPosition] | None]:
        """
        Main gateway: run the V2 FLOOR gate once, open every READY arm.

        Returns (eligible, reason, positions_or_None). On rejection, positions is
        None. Idempotent on repeated delivery of the same parent signal.

        The caller supplies ONE executable BUY quote, which is what the
        experiment-level API can express; the QUOTE-SIZED production path is
        :meth:`open_arms`, where the engine acquires a quote per arm at that
        arm's own all-in notional. Sizing here is still per-arm and all-in: each
        ready arm's allocation is 100% of ITS OWN uncommitted cash, and an arm
        that already holds a position is blocked rather than topped up.

        ``entry_fill`` is the EXECUTABLE BUY quote, required. The old signature
        took ``entry_fill_price`` and ``entry_mark_price``, which the caller
        filled from ``mark.value_usd`` -- so every beta entry was priced at a
        MARK and no quote was ever acquired or validated for it.
        """
        eligible, reason = self._v2_gate_result(signal)
        if not eligible:
            return False, reason, None
        parent_id = self._stable_parent_id(signal)
        if parent_id in self._seen_signals:
            return True, "duplicate signal (idempotent)", self.positions.get(parent_id, {})
        plan = self.arm_entry_plan()
        ready = tuple(a for a in BETA_ARMS if not plan[a]["blocked"])
        if not ready:
            details = "; ".join(f"{a}: {plan[a]['reason']}" for a in BETA_ARMS)
            return False, f"no beta arm could take an entry ({details})", None
        positions, blocked = self.open_arms(
            parent_signal_id=parent_id, mint=signal.token_mint or "",
            symbol=signal.symbol,
            entry_fills={a: entry_fill for a in ready},
            allocations={a: plan[a]["notional_usd"] for a in ready},
        )
        if not positions:
            details = "; ".join(f"{a}: {blocked.get(a, 'not opened')}"
                                for a in ready)
            return False, f"no beta arm could be opened ({details})", None
        return True, reason, positions

    def provenance_summary(self) -> dict[str, Any]:
        """
        Completed results SPLIT by accounting basis and by session.

        The experiment's history crosses a model change: earlier trades were
        booked on the MODELLED backtest path (no executable quote), later ones on
        the executable-quote path. They are different accounting bases, so the
        totals are reported SEPARATELY and a caller can never present the
        historical modelled P&L as quote-based performance.
        """
        by_basis: dict[str, dict[str, Any]] = {
            BASIS_QUOTE: {"completed_trades": 0, "wins": 0, "losses": 0,
                          "realized_pnl_usd": 0.0},
            BASIS_MODELED: {"completed_trades": 0, "wins": 0, "losses": 0,
                            "realized_pnl_usd": 0.0},
        }
        by_session: dict[str, dict[str, Any]] = {}
        for rec in self.completed:
            if not isinstance(rec, dict) or rec.get("status") != "closed":
                continue
            pnl = rec.get("net_pnl_usd")
            if pnl is None:
                continue
            bucket = basis_bucket(rec)
            sid = str(rec.get("session_id") or "?")
            for target, key in ((by_basis, bucket), (by_session, sid)):
                entry = target.setdefault(
                    key, {"completed_trades": 0, "wins": 0, "losses": 0,
                          "realized_pnl_usd": 0.0})
                entry["completed_trades"] += 1
                entry["realized_pnl_usd"] = round(
                    entry["realized_pnl_usd"] + float(pnl), 6)
                if float(rec.get("net_return_pct") or 0.0) > 0.0:
                    entry["wins"] += 1
                else:
                    entry["losses"] += 1
        # THE SIZING SPLIT. Same reasoning as the basis split: the experiment's
        # history crosses the sizing change (10%-of-equity -> ALL-IN), and a
        # single total would credit the ALL-IN experiment with what the old rule
        # produced. A record with no ``sizing_rule`` is the HISTORICAL 10% one.
        by_sizing: dict[str, dict[str, Any]] = {
            SIZING_BUCKET_ALL_IN: {"completed_trades": 0, "wins": 0,
                                   "losses": 0, "realized_pnl_usd": 0.0},
            SIZING_BUCKET_EXPLICIT: {"completed_trades": 0, "wins": 0,
                                     "losses": 0, "realized_pnl_usd": 0.0},
            SIZING_BUCKET_LEGACY: {"completed_trades": 0, "wins": 0,
                                   "losses": 0, "realized_pnl_usd": 0.0},
        }
        for rec in self.completed:
            if not isinstance(rec, dict) or rec.get("status") != "closed":
                continue
            pnl = rec.get("net_pnl_usd")
            if pnl is None:
                continue
            bucket = by_sizing[sizing_bucket(rec)]
            bucket["completed_trades"] += 1
            bucket["realized_pnl_usd"] = round(
                bucket["realized_pnl_usd"] + float(pnl), 6)
            if float(rec.get("net_return_pct") or 0.0) > 0.0:
                bucket["wins"] += 1
            else:
                bucket["losses"] += 1
        # THE COST-MODEL SPLIT. Same reasoning a third time: the history crosses
        # the cost-model change, and a record charged ONLY the swap quote's own
        # fee must not be presented as one that also paid the outside-swap fees
        # and reserved the rent.
        by_cost_model: dict[str, dict[str, Any]] = {
            COST_MODEL_MODELED: {"completed_trades": 0, "wins": 0, "losses": 0,
                                 "realized_pnl_usd": 0.0,
                                 "outside_swap_charged_usd": 0.0},
            COST_MODEL_LEGACY: {"completed_trades": 0, "wins": 0, "losses": 0,
                                "realized_pnl_usd": 0.0,
                                "outside_swap_charged_usd": 0.0},
        }
        for rec in self.completed:
            if not isinstance(rec, dict) or rec.get("status") != "closed":
                continue
            if rec.get("net_pnl_usd") is None:
                continue
            bucket = by_cost_model[cost_model_bucket(rec)]
            bucket["completed_trades"] += 1
            bucket["realized_pnl_usd"] = round(
                bucket["realized_pnl_usd"] + float(rec.get("net_pnl_usd") or 0.0),
                6)
            bucket["outside_swap_charged_usd"] = round(
                bucket["outside_swap_charged_usd"]
                + float(rec.get("outside_swap_charged_usd") or 0.0), 6)
            if float(rec.get("net_return_pct") or 0.0) > 0.0:
                bucket["wins"] += 1
            else:
                bucket["losses"] += 1
        for entry in list(by_basis.values()) + list(by_session.values()) \
                + list(by_sizing.values()) + list(by_cost_model.values()):
            n = entry["completed_trades"]
            entry["win_rate_pct"] = round(entry["wins"] / n * 100.0, 2) if n else None
        mixed = (by_basis[BASIS_QUOTE]["completed_trades"] > 0
                 and by_basis[BASIS_MODELED]["completed_trades"] > 0)
        return {
            "by_basis": by_basis,
            "by_session": by_session,
            "by_sizing": by_sizing,
            "by_cost_model": by_cost_model,
            "sizing_rule": BETA_SIZING_RULE,
            "sizing_pct": BETA_SIZING_PCT,
            "mixed": mixed,
            "note": (
                "quote-based and modelled-backtest results are DIFFERENT "
                "accounting bases and are never summed into one number; the "
                "modelled backtest is a historical simulation, NOT executable-"
                "quote performance. ALL-IN and legacy 10%-of-equity results "
                "are likewise different rules and are reported separately, as "
                "are results charged the swap quote's fee only (legacy) and "
                "results that also paid the outside-swap fees and reserved "
                "the token-account rent (modelled)."),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_started_at": self.session_started_at,
            "label": BETA_LABEL,
            "entry_profile": BETA_ENTRY_PROFILE,
            "paper_only": True,
            "starting_bankroll_per_arm_usd": round(self.starting_bankroll_usd, 2),
            "total_starting_capital_usd": round(self.total_starting_capital_usd, 2),
            "arms": {arm: self.arm_stats(arm) for arm in BETA_ARMS},
            "provenance": self.provenance_summary(),
            # The three arms are separate books; this is the one place their
            # COMBINED demand is stated, and it is labelled a stress view.
            "shared_liquidity_scenario": self.shared_liquidity_scenario(),
            # WHAT THE COST MODEL COVERS, stated ONCE at the top so a reader
            # never has to open a trade record to find out. ``Modeled`` is the
            # only basis that is fully costed.
            "cost_coverage": self.cost_coverage(),
            # THE FORWARD EXPERIMENT: its own store, its own progress counter.
            # Reported inside the SAME stats block so an operator cannot read an
            # arm figure without also seeing how much forward evidence exists.
            "forward_experiment": self.forward_progress(),
        }

    def cost_coverage(self) -> dict[str, Any]:
        """What the cost model covers, what it derives, and what it does NOT.

        THE FOUR PROVENANCE KINDS ARE KEPT APART, by name:

        * ``DERIVED``    -- an arithmetic identity over the INTENDED transaction
          configuration (the base fee is ``signatures x 5000``).
        * ``CONFIGURED`` -- a value this project sets. The priority fee is the
          CEILING the intended transaction requests; the transaction's actual
          observed fee is ``ceil(cu_used x price_microlamports / 1e6)`` and can
          be LOWER, so it is never presented as measured.
        * ``MEASURED``   -- an observation from a live provider read (the round
          trip probe's price impact, latencies, SOL/USD reference).
        * ``ASSUMED``    -- a worst case chosen to FAIL CLOSED (a NEW token
          account is required for the rent).

        FAILED ATTEMPTS ARE SPLIT TOO, because a refused QUOTE REQUEST and a
        failed BROADCAST do not cost the same: the first builds and sends
        nothing and costs ZERO lamports, the second pays its fees. PAPER never
        broadcasts, so its failure cost is zero by construction -- stated rather
        than left to be inferred.
        """
        cfg = intended_transaction_config()
        # Every failed EXIT attempt seen on this book, whether the arm has since
        # settled (the count is on its record) or is still outstanding (it is on
        # the position). A BetaPosition is an object, a trade record is a dict.
        quote_failures = 0
        for rec in self.completed:
            quote_failures += int(rec.get("exit_attempts") or 0)
        for pid in self.positions.values():
            for pos in pid.values():
                quote_failures += int(pos.exit_attempts or 0)
        return {
            "model": "OUTSIDE_SWAP_COSTS_MODELED",
            "charged_to_pnl": (
                "Solana base fee + priority fee, both legs (non-recoverable)"),
            "reserved_not_charged": (
                "token-account rent (recoverable deposit; reserved so an "
                "ALL-IN arm never over-commits, never charged as a loss)"),
            "inside_the_swap_quote": (
                "the venue/aggregator/route fee, already in the quoted input"),
            "intended_transaction": cfg,
            "provenance": {
                "base_fee": (
                    f"DERIVED from the intended transaction: "
                    f"{cfg['fee_payer_signatures_per_leg']} signature(s) per leg "
                    "x 5000 lamports/signature x 2 legs"),
                "priority_fee": (
                    "CONFIGURED CEILING: the total the intended transaction "
                    "requests (mirrors RealConfig.priority_fee_lamports (50000) "
                    "x 2 legs). NOT measured and NOT observed: a "
                    "dynamic-compute-unit-limit transaction pays "
                    "ceil(cu_used x price_microlamports / 1e6) <= this, and "
                    "deriving it exactly would require simulating and signing "
                    "the intended transaction, which PAPER never does"),
                "rent": (
                    "ASSUMED: a NEW token account is required "
                    "(conservative; an existing one only makes it cheaper)"),
            },
            "measured": [
                "both legs' quote amounts, routes, venue identity, quote age "
                "and latencies (recorded per fill)",
                "the SOL/USD reference used to size and value the reserve "
                "(source and timestamp recorded)",
            ],
            "failed_execution": failed_execution_accounting(
                quote_failures=quote_failures,
                broadcast_attempts=0,
                signatures_per_leg=cfg["fee_payer_signatures_per_leg"]),
            "slippage": slippage_boundary(),
            "not_modelled": (
                "failed-attempt on-chain cost (PAPER submits nothing, so its "
                "failed requests are provider-side quote refusals at ZERO "
                "network cost), and the loss a real minimum-output boundary "
                "would cause (PAPER sets no boundary because it builds no "
                "transaction)"),
        }


__all__ = [
    "BETA_LABEL", "BETA_SELECTOR", "BETA_ARMS", "BETA_ARM_CONFIGS",
    "BETA_STARTING_BANKROLL_USD", "BETA_ENTRY_PROFILE",
    "FORWARD_ELIGIBILITY", "FORWARD_EXCLUDES", "FORWARD_LABEL",
    "FORWARD_STATE_FILENAME", "FORWARD_TARGET_PARENT_SIGNALS",
    "BetaArmConfig", "BetaPosition", "BetaExperiment",
    "arm_config", "forward_eligible", "forward_eligible_open",
    "net_from_gross", "PAPER_EXIT_MULT",
]