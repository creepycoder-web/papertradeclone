"""
Strategy profiles: the entry strategy, plus the exit policy variant.

A "profile" is a SELECTABLE runtime configuration. It is deliberately narrow:
it names the entry strategy and, where a variant exists, the maximum-hold
override for the exit policy. Nothing else about the strategy is reachable
through it.

WHAT A PROFILE IS NOT
---------------------
It is not a way to change a threshold, a size, a fee, a cooldown or a re-entry
rule. Every one of those stays frozen in its own module and is not reachable
from here. ``L25_60S`` differs from ``L25_CURRENT`` by exactly one thing: a
maximum hold on the exit policy. The entry decision is the SAME L25
implementation, called the SAME way, in both profiles.

WHY THE PROFILE LIVES ON THE POSITION AND NOT ONLY ON THE STATE
----------------------------------------------------------------
The operator can change the selected profile at any time, including while a
position is open. A position therefore records the profile it was OPENED under
and the exit policy reads THAT, never the current global selection. Otherwise a
profile switch would silently rewrite the exit policy of a live position, which
is a different strategy than the one that was opened.

THE 60s VARIANT'S PROVENANCE, STATED HONESTLY
----------------------------------------------
``L25_60S`` comes from a retrospective counterfactual replay of the frozen L25
entry set (``reports/research/l25_60s_backtest.{md,json}``). On that replay it
measured: win rate 56.90% -> 78.45%, max drawdown -12.71% -> -5.86%, profit
factor 6.53 -> 9.51, and ending bankroll $2,027.04 -> $1,977.04 on a $10
starting account. That is OBSERVED REPLAY OUTPUT, NOT A GUARANTEE, and it is
not a claim of superiority:

  * it was retrospective, on the same historical sample L25 was derived from;
  * chronological consistency was MIXED -- the 60s cap was negative on the
    TRAIN block and on the post-derivation TEST block, and positive only on
    VALIDATION;
  * the tested fill was the mark at the trigger, an optimistic bound.

Production behaviour therefore has to be validated prospectively. That is why
``L25_CURRENT`` remains the default and why selecting a profile never changes
anything about arming or starting the runtime.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from .logging_setup import log


class StrategyProfileError(ValueError):
    """
    An unrecognised strategy profile.

    Raised rather than defaulted, so a typo in ``STRATEGY_PROFILE`` or a bad
    operator selection FAILS CLOSED: the caller cannot start on a profile that
    was never defined.
    """


class StrategyProfile(str, Enum):
    """The complete, closed set of selectable profiles."""

    #: Exactly today's behaviour: L25 entry, existing exit rules, no cap.
    L25_CURRENT = "L25_CURRENT"

    #: A SECOND, INDEPENDENT entry strategy (autopaper.v2floor), not a variant
    #: of L25. Selected here because this module is the single closed
    #: vocabulary the config parser, the operator selector, the position stamp,
    #: the exit max-hold lookup, the Telegram status line and the safety audit
    #: all read -- so registering the profile anywhere else would leave one of
    #: those surfaces unable to name it.
    #:
    #: It is deliberately NOT layered on top of L25. L25 is V3-C AND a
    #: liquidity/MC ratio, and V3-C carries a $20,000 market-cap ceiling and a
    #: 0.55 signal-quality floor. V2_FLOOR has no market-cap cap and no
    #: liquidity filter, because the research tested and REJECTED both. Stacking
    #: it on L25 would silently reinstate those two filters and trade a
    #: different strategy from the one that was measured.
    #:
    #: The 90s cap here is the research ``max_hold_s``. It is appended LAST in
    #: the exit precedence by autopaper/exits.py, exactly as L25_60S's is: it is
    #: a cap, not a replacement for the existing exit triggers.
    V2_FLOOR = "V2_FLOOR"

    #: Same L25 entry, same existing exit rules, PLUS a 60s maximum hold.
    #: Fixed at 60s; kept for continuity with the original replay.
    L25_60S = "L25_60S"

    #: Same L25 entry, same existing exit rules, PLUS a CONFIGURABLE hard
    #: max-hold cap (default 60s).
    #:
    #: At its default this behaves exactly like ``L25_60S``. It exists as a
    #: separate profile so the cap is an explicit, TUNABLE strategy parameter
    #: for PAPER research rather than a constant baked into a profile name --
    #: ``L25_60S`` cannot be re-tuned without lying about what it is.
    L25_MAX_HOLD = "L25_MAX_HOLD"

    #: Same L25 entry, same existing exit rules, PLUS a take-profit at +100%.
    #:
    #: This is a PAPER research profile based on the exit research finding that
    #: TP100 captures the single spike of strong memecoin pumps before collapse.
    #: The existing downside triggers (fast-failure stop -12%, momentum collapse
    #: -18%/30s, structure break -14%/45s, no-follow-through 75s, disaster cap
    #: -45%, data-invalid 120s) are ALL PRESERVED. TP100 is an ADDITIONAL
    #: profit-taking exit, not a replacement.
    L25_TP100 = "L25_TP100"

    #: V2 FLOOR (BETA) experiment: three parallel PAPER-only exit arms off ONE
    #: V2 FLOOR entry. Arms share entry decision/fill; differ ONLY in exit rule:
    #:   Arm A: TP +30%  SL -30%  max hold  90s
    #:   Arm B: TP +50%  SL -30%  max hold  90s
    #:   Arm C: TP +50%  SL -30%  max hold 120s
    #: Each arm owns an ISOLATED $10 virtual ledger ($30 total). PAPER ONLY.
    V2_FLOOR_BETA = "V2_FLOOR_BETA"

    def __str__(self) -> str:      # pragma: no cover - trivial
        return self.value


#: The entry strategy each profile runs. The L25 family shares ``L25``;
#: ``V2_FLOOR`` runs its own gate in ``autopaper.v2floor``.
#:
#: This WAS a single ``ENTRY_STRATEGY: str`` constant asserting that one entry
#: strategy exists and a profile cannot change it. That assertion was true when
#: written and is now false, so it became a per-profile mapping rather than
#: being quietly dropped: the invariant it protected -- that selecting a profile
#: cannot silently swap the entry gate -- is enforced by
#: :data:`DEFAULT_ENTRY_STRATEGY` being unchanged and by every L25 profile
#: still mapping to L25.
DEFAULT_ENTRY_STRATEGY: str = "L25"

#: The two entry-gate tokens the decision layer can be asked to apply. Defined
#: HERE, next to the profile mapping that names them, so the gate vocabulary has
#: exactly one home and both the decision layer and the arm registry import it
#: rather than restating the strings.
GATE_L25: str = "L25"
GATE_V2_FLOOR: str = "V2_FLOOR"

#: The PumpView 'Amarih' paired-exit experiment gate. Amarih is NOT an L25 or
#: V2_FLOOR candidate: its signals are PumpView's own already-decided community
#: strategy events (see amarih_pumpview_integration.md), not candidates our L25
#: or V2_FLOOR gates evaluate. It gets its OWN gate token so selecting the
#: Amarih experiment can never be silently funneled through -- or vetoed by --
#: the L25 gate, exactly the failure the effective-entry-gate contract exists to
#: prevent for the beta experiment.
GATE_AMARIH: str = "AMARIH"

#: Every gate token the system recognises. A value outside this set is not a
#: gate and must never be treated as one.
KNOWN_ENTRY_GATES: tuple[str, ...] = (GATE_L25, GATE_V2_FLOOR, GATE_AMARIH)

#: Backwards-compatible alias for the L25 family's entry strategy.
ENTRY_STRATEGY: str = DEFAULT_ENTRY_STRATEGY

#: Which entry gate each profile authorizes through.
PROFILE_ENTRY_STRATEGY: dict[StrategyProfile, str] = {
    StrategyProfile.L25_CURRENT: "L25",
    StrategyProfile.L25_60S: "L25",
    StrategyProfile.L25_MAX_HOLD: "L25",
    StrategyProfile.L25_TP100: "L25",
    StrategyProfile.V2_FLOOR: "V2_FLOOR",
    StrategyProfile.V2_FLOOR_BETA: "V2_FLOOR",
}


def entry_strategy(profile: object) -> str:
    """
    The entry strategy a profile authorizes through.

    Resolved through :func:`parse_profile`, so an unrecognised value REFUSES
    rather than silently reporting L25. A caller that fell back to the default
    here would authorize a real entry under the wrong gate, which is the one
    failure this module exists to make impossible.
    """
    return PROFILE_ENTRY_STRATEGY[parse_profile(profile)]


def uses_l25_entry(profile: object) -> bool:
    """Whether this profile authorizes through the L25 gate."""
    return entry_strategy(profile) == DEFAULT_ENTRY_STRATEGY

#: The default, and the profile a missing or unreadable setting resolves to.
#: The default is today's behaviour, so an operator who never touches the
#: selector is running exactly what they were running before this existed.
DEFAULT_PROFILE: StrategyProfile = StrategyProfile.L25_CURRENT

#: Maximum hold in seconds, per profile. ``None`` means "no cap", which is
#: today's behaviour: the exit is decided only by the existing rules.
PROFILE_MAX_HOLD_S: dict[StrategyProfile, Optional[float]] = {
    StrategyProfile.L25_CURRENT: None,
    StrategyProfile.L25_60S: 60.0,
    #: DEFAULT cap. Override at runtime with :func:`set_max_hold_s` for PAPER
    #: research; the default is what an operator who never touches it gets.
    StrategyProfile.L25_MAX_HOLD: 60.0,
    #: L25_TP100 has no max hold cap; it relies on the existing exit rules plus
    #: the TP100 take-profit. The cap is None like L25_CURRENT.
    StrategyProfile.L25_TP100: None,
    #: The V2_FLOOR research rule's max hold. NOT tunable through
    #: :func:`set_max_hold_s`, which is scoped to L25_MAX_HOLD on purpose: the
    #: override exists so L25's cap can be retuned without renaming a profile,
    #: and 90s here is part of what the name V2_FLOOR says it is.
    StrategyProfile.V2_FLOOR: 90.0,
    #: V2_FLOOR_BETA has no single max hold; each arm has its own (90/90/120s).
    #: The experiment engine enforces per-arm caps internally.
    StrategyProfile.V2_FLOOR_BETA: None,
}

#: Runtime override for the TUNABLE profile only, set from configuration.
#: ``None`` means "use the default above". Scoped to L25_MAX_HOLD on purpose:
#: ``L25_CURRENT`` must never acquire a cap, and ``L25_60S`` is fixed because
#: its name states its cap.
_MAX_HOLD_OVERRIDE_S: Optional[float] = None


def set_max_hold_s(seconds: Optional[float]) -> None:
    """
    Configure the ``L25_MAX_HOLD`` cap. PAPER research only.

    Fails closed on nonsense rather than silently clamping: a cap of 0 would
    exit every position instantly, and a negative cap is not a hold at all.
    """
    global _MAX_HOLD_OVERRIDE_S
    if seconds is None:
        _MAX_HOLD_OVERRIDE_S = None
        return
    value = float(seconds)
    if not (value > 0):
        raise StrategyProfileError(
            f"max_hold_seconds must be a positive number of seconds, got "
            f"{seconds!r}")
    _MAX_HOLD_OVERRIDE_S = value


def configured_max_hold_s_for(profile: object) -> Optional[float]:
    """
    The cap a SPECIFIC profile runs right now.

    Resolves through :func:`parse_profile` first, so an unrecognised value
    refuses instead of silently reporting the L25_MAX_HOLD override. Used by
    surfaces that need one profile's cap without hardcoding which profile it is.
    """
    p = parse_profile(profile)
    if p is StrategyProfile.L25_MAX_HOLD:
        return configured_max_hold_s()
    return PROFILE_MAX_HOLD_S[p]


def configured_max_hold_s() -> Optional[float]:
    """The cap ``L25_MAX_HOLD`` will actually run right now."""
    if _MAX_HOLD_OVERRIDE_S is not None:
        return _MAX_HOLD_OVERRIDE_S
    return PROFILE_MAX_HOLD_S[StrategyProfile.L25_MAX_HOLD]

#: The exit trigger recorded when the cap fires. Surfaced verbatim in trade
#: records and operator notifications so a max-hold exit is never mistaken for
#: a momentum or structure exit.
#:
#: This is the trigger for the FIXED 60s cap. ``L25_MAX_HOLD`` is tunable, so
#: its trigger is generated from the cap actually in force by
#: :func:`max_hold_trigger` -- otherwise a 90s exit would be reported as a 60s
#: one, which is exactly the kind of mislabel that makes a result untrustworthy.
MAX_HOLD_TRIGGER: str = "max_hold_60s"


def max_hold_trigger(cap_s: float) -> str:
    """The trigger token for a cap of ``cap_s`` seconds."""
    return f"max_hold{float(cap_s):.0f}s".replace("max_hold", "max_hold_")

#: Exit style token for the bracket, matching the existing ``ExitStyle`` values.
#: Declared before ``PROFILE_EXIT_STRATEGY`` because that mapping names it.
V2_BRACKET_STYLE: str = "V2_BRACKET"

#: The exit strategy each profile runs.
#:
#: The L25 family runs the SHARED ``FAST_FAILURE`` stack. ``V2_FLOOR`` runs its
#: own researched bracket instead -- see :data:`PROFILE_EXIT_BRACKET`. This is
#: not a second exit ENGINE: both branches are reached from the same
#: ``ExitPolicy.assess`` call and produce the same ``ExitDecision``, which the
#: caller then satisfies with the one common quote/fill pipeline. Only the
#: DECISION differs.
#:
#: L25_TP100 uses the same FAST_FAILURE stack but with an additional TP100
#: take-profit check. The TP100 logic is implemented in the FAST_FAILURE block
#: of ExitPolicy.assess, gated by the position's strategy_profile.
#: Exit style for the beta experiment. Three parallel arms, each with its own
#: bracket; the experiment engine runs them independently. Not a shared stack.
V2_FLOOR_BETA_STYLE: str = "V2_FLOOR_BETA"

PROFILE_EXIT_STRATEGY: dict[StrategyProfile, str] = {
    StrategyProfile.L25_CURRENT: "FAST_FAILURE",
    StrategyProfile.L25_60S: "FAST_FAILURE",
    StrategyProfile.L25_MAX_HOLD: "FAST_FAILURE",
    StrategyProfile.L25_TP100: "FAST_FAILURE",
    StrategyProfile.V2_FLOOR: V2_BRACKET_STYLE,
    StrategyProfile.V2_FLOOR_BETA: V2_FLOOR_BETA_STYLE,
}

# ---------------------------------------------------------------------------
# the V2_FLOOR exit bracket
# ---------------------------------------------------------------------------
#
# The researched V2 FLOOR exit rule, exactly:
#
#     TP = +30%   SL = -30%   MAX HOLD = 90s
#
#     if return <= -30%:            SELL
#     elif return >= +30%:          SELL
#     elif hold_time >= 90s:        SELL
#     else:                         HOLD
#
# WHY THIS IS A SEPARATE BRANCH AND NOT THE SHARED STACK
#
# The shared ``FAST_FAILURE`` stack cannot express this rule. Its triggers are
# a -12% stop, a -45% disaster cap, a 75s no-follow-through exit, and momentum
# and structure exits -- none of which appear in the V2 research, and the first
# two of which fire long before the researched -30% stop. Running the researched
# ENTRY rule with the shared EXIT stack is not a conservative approximation of
# the strategy; it is a DIFFERENT strategy that happens to share a name. So the
# bracket is evaluated on its own, and every shared trigger is skipped for this
# profile.
#
# WHAT SURVIVES
#
# Only gates that are not strategy decisions:
#
#   * the data-integrity gate (a TP/SL is a PRICE action and needs a real mark,
#     so it is evaluated AFTER that gate, not before it -- unlike
#     ``L25_MAX_HOLD``'s time-only hard hold, which correctly sits before it);
#   * ``data_invalid``, which is bounded and exits to a known state;
#   * ``STOP_COMMAND`` / ``FLATTEN``, which are operator actions outside the
#     strategy entirely.
#
# ``disaster_cap`` (-45%) also survives, and is in fact UNREACHABLE for this
# profile: the -30% stop fires first at every price.

#: The researched V2_FLOOR take-profit. Positive, percentage of fill price.
V2F_TAKE_PROFIT_PCT: float = 30.0

#: The researched V2_FLOOR stop. Negative; the sell trigger is ``<= this``.
V2F_STOP_LOSS_PCT: float = -30.0

#: The bracket's own exit triggers. Distinct from every shared trigger name, so a
#: V2 exit can never be read as an L25 exit in the journal.
V2F_TRIGGER_TAKE_PROFIT: str = "v2_take_profit"
V2F_TRIGGER_STOP_LOSS: str = "v2_stop_loss"
V2F_TRIGGER_MAX_HOLD: str = "v2_max_hold_90s"

#: Precedence of the researched exits, highest first. Asserted in tests, because
#: a precedence order that is merely "whatever the code happens to do" is not a
#: specification.
V2F_EXIT_PRECEDENCE: tuple[str, ...] = (
    V2F_TRIGGER_STOP_LOSS,
    V2F_TRIGGER_TAKE_PROFIT,
    V2F_TRIGGER_MAX_HOLD,
)

#: ``(take_profit_pct, stop_loss_pct)`` per profile, or ``None`` for every
#: profile that runs the shared stack.
#:
#: ``None`` is the DEFAULT and is what all L25 profiles carry, so adding
#: this branch could not have changed any existing profile's behaviour even if
#: the lookup were wrong.
#:
#: L25_TP100 uses None here because its TP100 is implemented as an ADDITIONAL
#: check within the shared FAST_FAILURE stack (see exits.py), not as a
#: replacement bracket like V2_FLOOR.
#: V2_FLOOR_BETA has no single bracket; the experiment engine runs three arms
#: with different brackets internally (A: TP+30/SL-30, B: TP+50/SL-30, C: TP+50/SL-30).
PROFILE_EXIT_BRACKET: dict[StrategyProfile, Optional[tuple[float, float]]] = {
    StrategyProfile.L25_CURRENT: None,
    StrategyProfile.L25_60S: None,
    StrategyProfile.L25_MAX_HOLD: None,
    StrategyProfile.L25_TP100: None,
    StrategyProfile.V2_FLOOR: (V2F_TAKE_PROFIT_PCT, V2F_STOP_LOSS_PCT),
    StrategyProfile.V2_FLOOR_BETA: None,
}


def exit_bracket_for(profile: Any) -> Optional[tuple[float, float]]:
    """
    ``(take_profit_pct, stop_loss_pct)`` for a profile, or ``None`` for the
    shared stack.

    FAILS CLOSED on an unrecognised profile: returns ``None`` and logs, rather
    than raising into the exit loop or guessing a bracket. ``None`` means "run
    the shared stack", which is the pre-existing behaviour, so the failure mode
    is the old behaviour and not a silent invented stop.
    """
    try:
        return PROFILE_EXIT_BRACKET[parse_profile(profile)]
    except StrategyProfileError:
        log("ERROR",
            "unrecognised strategy_profile=%r; running the shared exit stack "
            "rather than guessing a bracket", profile)
        return None


#: Exit triggers a profile declares as its OWN strategy exits, beyond the shared
#: stack. Empty for every L25 profile except L25_TP100. ``V2_FLOOR`` is the one
#: profile with a researched bracket that replaces the shared stack.
#:
#: L25_TP100 declares "tp100" as its own trigger, but the actual logic is
#: implemented within the shared FAST_FAILURE stack. The trigger is recorded
#: when TP100 fires so it can be distinguished in the journal.
#: V2_FLOOR_BETA declares its own triggers for the three arms. These are the
#: experiment-internal exit reason tokens, distinct from any shared stack triggers.
V2B_EXIT_PRECEDENCE: tuple[str, ...] = (
    "v2b_A_take_profit",
    "v2b_A_stop_loss",
    "v2b_A_max_hold_90s",
    "v2b_B_take_profit",
    "v2b_B_stop_loss",
    "v2b_B_max_hold_90s",
    "v2b_C_take_profit",
    "v2b_C_stop_loss",
    "v2b_C_max_hold_120s",
)

PROFILE_OWN_EXIT_TRIGGERS: dict[StrategyProfile, tuple[str, ...]] = {
    StrategyProfile.L25_CURRENT: (),
    StrategyProfile.L25_60S: (),
    StrategyProfile.L25_MAX_HOLD: (),
    StrategyProfile.L25_TP100: ("tp100",),
    StrategyProfile.V2_FLOOR: V2F_EXIT_PRECEDENCE,
    StrategyProfile.V2_FLOOR_BETA: V2B_EXIT_PRECEDENCE,
}

#: Operator-facing names. Deliberately not the raw enum value, because
#: "L25_60S" on a status line is a configuration token, not a description.

def position_pct_override(profile: Any) -> Optional[float]:
    """
    The profile's own percentage-of-equity, or ``None`` for the shared policy.

    FAILS CLOSED the same way :func:`exit_bracket_for` does: an unrecognised
    profile falls back to the shared policy rather than inventing a size.
    """
    try:
        return PROFILE_POSITION_PCT[parse_profile(profile)]
    except StrategyProfileError:
        log("ERROR",
            "unrecognised strategy_profile=%r; using the shared sizing policy "
            "rather than guessing a percentage", profile)
        return None


#: Operator-facing labels. ``PROFILE_LABELS`` / ``PROFILE_NOTES`` / this sit
#: together so the operator sees entry, exit, hold and size as one profile.
PROFILE_LABELS: dict[StrategyProfile, str] = {
    StrategyProfile.L25_CURRENT: "L25 CURRENT",
    StrategyProfile.L25_60S: "L25 + 60s",
    StrategyProfile.L25_MAX_HOLD: "L25 + MAX HOLD",
    StrategyProfile.L25_TP100: "L25 + TP100",
    StrategyProfile.V2_FLOOR: "V2 FLOOR (vol/mcap>=1, 5m +10%..+300%, 90s)",
    StrategyProfile.V2_FLOOR_BETA: "V2 FLOOR (BETA) -- 3 Parallel Exit Arms",
}

#: One-line provenance, shown in the operator menu so nobody has to guess why a
#: profile exists or what it was measured on.
PROFILE_NOTES: dict[StrategyProfile, str] = {
    StrategyProfile.L25_CURRENT: "default; L25 entry + current exit rules",
    StrategyProfile.L25_60S: (
        "same L25 entry + current exit rules + 60s max hold. From a "
        "retrospective replay: win rate 56.9%->78.5%, max DD -12.7%->-5.9%, "
        "PF 6.53->9.51, but ending bankroll $2,027->$1,977 on $10. Mixed "
        "chronological consistency (negative on TRAIN and TEST). NOT proven "
        "superior; validate prospectively."
    ),
    StrategyProfile.L25_MAX_HOLD: (
        "same L25 entry + current exit rules + a CONFIGURABLE hard max-hold "
        f"cap (default {PROFILE_MAX_HOLD_S[StrategyProfile.L25_MAX_HOLD]:.0f}s, "
        "settable via max_hold_seconds for PAPER research). The cap is an "
        "extra exit TRIGGER only: it requests an executable SELL and never "
        "forces a fill. At its default it matches L25_60S; it exists so the cap "
        "can be tuned without renaming the profile. The drawdown benefit is "
        "from a RETROSPECTIVE replay with MIXED chronological consistency, and "
        "is NOT PROVEN SUPERIOR: validate prospectively on PAPER."
    ),
    StrategyProfile.L25_TP100: (
        "same L25 entry + current exit rules + TAKE PROFIT at +100% (executable/"
        "native return). PAPER research profile from exit research: TP100 was "
        "the best fixed level on the common sample (+183 net, PF 9.51, "
        "expectancy 17.54, 49% MFE captured) but FAILS chronological validation "
        "(loses on TRAIN, wins on TEST). This is a RETROSPECTIVE finding with "
        "MIXED chronological consistency. ALL existing downside triggers "
        "PRESERVED: fast-failure stop -12%, momentum collapse -18%/30s, "
        "structure break -14%/45s, no-follow-through 75s, disaster cap -45%, "
        "data-invalid 120s. TP100 is ADDITIONAL profit-taking, not a replacement. "
        "NOT PROVEN SUPERIOR; unvalidated forward test only."
    ),
    StrategyProfile.V2_FLOOR: (
        "SECOND entry strategy, NOT an L25 variant: "
        "autopaper/v2floor.py, vol5/mcap >= 1.0 AND 5m change in "
        "[+10%, +300%]. Its OWN researched exit bracket, TP +30% / SL -30% / "
        "max hold 90s, which SUPPRESSES the shared L25 exit triggers for this "
        "profile (no -12% stop, no 75s no-follow-through, no momentum or "
        "structure exit). Sized at a FLAT 10% of current equity. From a "
        "RETROSPECTIVE replay over 193 trades on a frozen tape: win rate "
        "66.9%->70.5%, expectancy +23.84%->+28.37%, PF 3.81->4.98, max DD "
        "18.05%->11.91%, longest loss streak 5->4. THREE REMAINING "
        "DISCLOSURES: (1) NO market-cap cap and NO liquidity filter -- both "
        "were tested and rejected, so this is NOT L25 plus a momentum floor; "
        "(2) the shared structural route gates (data freshness, liquidity "
        "sanity, market-cap sanity band, confidence) still apply, so the live "
        "population is NARROWER than the research population; (3) the backtest "
        "break-even execution degradation is 12.4% against a measured buy-side "
        "quote slip of 11.5% median on only 7 non-stale records, so "
        "profitability is UNRESOLVED. NOT PROVEN SUPERIOR and not "
        "prospectively validated. PAPER ONLY."
    ),
    StrategyProfile.V2_FLOOR_BETA: (
        "RETROSPECTIVE EXPERIMENT WITH MIXED CHRONOLOGICAL CONSISTENCY: three parallel PAPER-only exit arms off ONE V2 FLOOR entry. "
        "Uses the SAME V2 FLOOR gate (vol5/mcap>=1, 5m change +10%..+300%). "
        "One entry decision, one fill, three isolated $10 virtual ledgers ($30 total). "
        "Arms: A=TP+30/SL-30/90s, B=TP+50/SL-30/90s, C=TP+50/SL-30/120s. "
        "A vs B isolates TP level; B vs C isolates max hold. "
        "Sized at FLAT 10% of equity. PAPER ONLY -- NOT live, broadcasts NOTHING. "
        "No prospective validation; this is a research tool to compare exit variants "
        "on identical entries. Gate reuse: v2floor.evaluate (unchanged). "
        "Ledger reuse: VirtualAccount, FillModel, PersistedState/PaperStateStore. "
        "Exit semantics: TP at LIMIT price, SL at observed trough (stop-through), "
        "TIME at last observed return. Ambiguous TP/SL -> conservative SL + flag. "
        "NOT PROVEN SUPERIOR; unvalidated forward test only."
    ),
}

# --- per-profile position sizing ---------------------------------------
#:
#: The researched V2 FLOOR size is a FLAT 10% of current equity at every balance,
#: where the shared policy steps to 12.5% above $50. The override is declared
#: HERE, in the profile registry, and ``autopaper.sizing`` reads it -- so the
#: researched number lives with the other profile facts and the shared risk code
#: keeps exactly one branch instead of growing a per-strategy ladder.
#:
#: ``None`` means "use the shared sizing policy", which is every L25 profile.
PROFILE_POSITION_PCT: dict[StrategyProfile, Optional[float]] = {
    StrategyProfile.L25_CURRENT: None,
    StrategyProfile.L25_60S: None,
    StrategyProfile.L25_MAX_HOLD: None,
    StrategyProfile.L25_TP100: None,
    StrategyProfile.V2_FLOOR: 10.0,
    # V2 FLOOR (BETA) is ALL-IN: each of its three arms commits 100% of its own
    # uncommitted cash on every entry. The 10% declared here until 2026-10-13
    # left 90% of every arm idle and is gone. The number is declared so that the
    # registry cannot disagree with the experiment's own rule, and so a generic
    # lookup can never quietly size a beta arm at 10% again. ``sizing.py`` is not
    # what sizes beta arms -- ``BetaExperiment.all_in_notional_usd`` is -- and the
    # two now say the same thing.
    StrategyProfile.V2_FLOOR_BETA: 100.0,
}


#: Every accepted raw value. Used by the config parser and asserted in tests so
#: the set of profiles cannot grow without the vocabulary changing too.
PROFILE_VALUES: tuple[str, ...] = tuple(p.value for p in StrategyProfile)


def parse_profile(value: object) -> StrategyProfile:
    """
    Resolve a raw configured value to a profile, or refuse.

    Fails closed. A value that is not exactly one of the known profiles raises
    :class:`StrategyProfileError`; it is never coerced to the default, because
    silently running a different strategy than the operator asked for is worse
    than refusing to start.

    ``None`` and the empty string resolve to the DEFAULT, so an absent setting
    is a legitimate configuration and not a typo. Anything else non-matching is
    an error.
    """
    if value is None:
        return DEFAULT_PROFILE
    if isinstance(value, StrategyProfile):
        return value
    text = str(value).strip()
    if text == "":
        return DEFAULT_PROFILE
    upper = text.upper()
    for profile in StrategyProfile:
        if upper == profile.value:
            return profile
    raise StrategyProfileError(
        f"unknown strategy profile {text!r}; allowed values are "
        f"{', '.join(PROFILE_VALUES)} (no default is assumed on a bad value)"
    )


def max_hold_s(profile: object) -> Optional[float]:
    """The maximum hold for a profile, or ``None`` for no cap."""
    p = parse_profile(profile)
    if p is StrategyProfile.L25_MAX_HOLD:
        return configured_max_hold_s()
    return PROFILE_MAX_HOLD_S[p]


def label(profile: object) -> str:
    """Operator-facing name."""
    return PROFILE_LABELS[parse_profile(profile)]


def exit_strategy(profile: object) -> str:
    """The exit style the profile runs."""
    return PROFILE_EXIT_STRATEGY[parse_profile(profile)]


def note(profile: object) -> str:
    """Provenance line for the operator menu."""
    return PROFILE_NOTES[parse_profile(profile)]


def status_line(profile: object) -> str:
    """The exact line the operator sees on status surfaces."""
    return f"STRATEGY: {label(profile)}"


def describe(profile: object) -> str:
    """The full profile definition, for status/diagnostics surfaces."""
    p = parse_profile(profile)
    cap = max_hold_s(p)
    cap_text = "none (no cap; existing exit rules only)" if cap is None \
        else f"{cap:g}s"
    entry_text = PROFILE_ENTRY_STRATEGY[p]
    if entry_text == DEFAULT_ENTRY_STRATEGY:
        entry_text = f"{DEFAULT_ENTRY_STRATEGY} (shared by the L25 family)"
    else:
        entry_text = f"{entry_text} (own gate; NOT L25)"
    bracket = PROFILE_EXIT_BRACKET[p]
    if bracket is None:
        exit_text = (f"{PROFILE_EXIT_STRATEGY[p]} (shared stack, shared with "
                     f"the L25 family)")
    else:
        tp, sl = bracket
        exit_text = (f"{PROFILE_EXIT_STRATEGY[p]} (own researched bracket: "
                     f"TP {tp:+.0f}%, SL {sl:+.0f}%; the shared L25 exit "
                     f"triggers are suppressed)")
    own = PROFILE_OWN_EXIT_TRIGGERS[p]
    own_text = ", ".join(own) if own else \
        "none (runs the shared exit stack)"
    pct = PROFILE_POSITION_PCT[p]
    size_text = "shared sizing policy" if pct is None else \
        f"{pct:g}% of current equity (flat, own researched sizing)"
    return (
        f"{PROFILE_LABELS[p]}\n"
        f"  entry strategy : {entry_text}\n"
        f"  exit strategy  : {exit_text}\n"
        f"  max hold       : {cap_text}\n"
        f"  own triggers   : {own_text}\n"
        f"  position size  : {size_text}\n"
        f"  note           : {PROFILE_NOTES[p]}"
    )


def as_dict(profile: object) -> dict:
    """Machine-readable profile definition."""
    p = parse_profile(profile)
    cap = PROFILE_MAX_HOLD_S[p]
    bracket = PROFILE_EXIT_BRACKET[p]
    return {
        "profile": p.value,
        "label": PROFILE_LABELS[p],
        "entry_strategy": PROFILE_ENTRY_STRATEGY[p],
        "exit_strategy": PROFILE_EXIT_STRATEGY[p],
        "exit_bracket": (
            None if bracket is None
            else {"take_profit_pct": bracket[0], "stop_loss_pct": bracket[1]}
        ),
        "max_hold_s": cap,
        "max_hold_trigger": max_hold_trigger(cap) if cap is not None else None,
        "own_exit_triggers": list(PROFILE_OWN_EXIT_TRIGGERS[p]),
        "position_pct_override": PROFILE_POSITION_PCT[p],
        "note": PROFILE_NOTES[p],
        "is_default": p is DEFAULT_PROFILE,
    }
