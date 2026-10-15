"""
The research arms, and where each one's rule already exists.

WHY THESE THREE
===============

The arms are not invented for this harness. Each maps onto something that is
already written down in this repository, and the provenance is carried in
:attr:`ResearchArm.provenance` so a reader can check the claim:

``L25_CURRENT``
    The production default. :data:`autopaper.strategy_profile.DEFAULT_PROFILE`.
    Entry gate = L25, exit precedence = FAST_FAILURE, no max hold.

``L25_TP100``
    The production default L25 entry with an ADDITIONAL take-profit at +100%
    executable/native return. All existing downside triggers are preserved
    (fast-failure stop -12%, momentum collapse -18%/30s, structure break
    -14%/45s, no-follow-through 75s, disaster cap -45%, data-invalid 120s).
    Implemented as an additional check in the shared FAST_FAILURE stack
    (exits.py). From exit research: best fixed level on common sample but
    FAILS chronological validation. UNVALIDATED forward test only.

``H2_CAP60``
    Preregistered hypothesis ``H2_turnover_ceiling`` from
    ``research/l25r/hypotheses.py`` with the 60-second cap appended - the
    ``H2_CAP60`` cell of the arm family built in ``research/l25c/registry.py``.
    The rule is ``f_volume_to_liquidity <= 2.0``, and the preregistration is
    explicit that 2.0 was "a round number fixed by mechanism, NOT fitted", with
    coverage checked before any outcome was computed.

    Its exit behaviour uses the production 60s max hold profile
    (``L25_60S``), so it differs from L25_TP100 ONLY by the entry filter.
    That is the point of the comparison, and it is why no new exit code exists.

WHAT IS NOT HERE
================

No arm changes the entry gate's thresholds, the exit precedence, the cooldown,
the re-entry rule or the sizing formula. The only behavioural differences
between arms are the exit policy variant (via strategy_profile) and the
extra entry predicate for H2_CAP60.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from autopaper.strategy_profile import (
    GATE_AMARIH as _GATE_AMARIH,
    GATE_L25 as _GATE_L25,
    GATE_V2_FLOOR as _GATE_V2_FLOOR,
    StrategyProfile,
)


class ResearchArm(str, Enum):
    """The forward-capture arms. One virtual bankroll each."""

    L25_CURRENT = "L25_CURRENT"

    #: RETIRED AS AN ARM. L25 + take profit used to be a whole arm whose only
    #: difference from L25_CURRENT was a hard-coded +100% exit. Take profit is a
    #: PARAMETER, not a strategy, so it now lives in ``paper_live.take_profit``
    #: as a selectable setting (which still includes +100%), persisted and
    #: snapshotted onto each position.
    #:
    #: The token is kept so historical ledgers and already-open positions still
    #: name what they were opened under, and so ``take_profit.for_position`` can
    #: map such a position to +100% rather than silently disarming it. It is NOT
    #: offered for new selection.
    L25_TP100 = "L25_TP100"

    #: L25 entry + the existing exit rules + the production 60s MAX HOLD.
    #:
    #: RESTORED. ``StrategyProfile.L25_60S`` has always existed and is still the
    #: production profile for the 60s cap, and the selector button for it was
    #: still rendered -- but the selection path resolves through
    #: :func:`arm_by_name`, which did not know this name, so clicking the button
    #: was REFUSED with "unknown research arm". The profile was unreachable in
    #: practice while still being visible on screen.
    #:
    #: This maps onto ``StrategyProfile.L25_60S`` unchanged. The 60s cap is a cap
    #: appended LAST in the exit precedence by exits.py; every downside trigger
    #: is preserved. Nothing about the rule is redefined here -- only its
    #: reachability is restored.
    L25_60S = "L25_60S"

    #: L25 entry + the existing exit rules + a CONFIGURABLE hard max hold.
    #: At its 60s default this behaves exactly like ``L25_60S``. It exists so the
    #: cap is an explicit, tunable strategy parameter for PAPER research rather
    #: than a constant baked into a profile name.
    L25_MAX_HOLD = "L25_MAX_HOLD"

    H2_CAP60 = "H2_CAP60"

    #: V2 FLOOR (BETA) parallel exit experiment -- three arms, ONE entry gate.
    #:
    #: Each of these authorizes through the EXISTING V2_FLOOR entry gate
    #: (``autopaper.v2floor.evaluate``, unchanged) and stamps
    #: ``StrategyProfile.V2_FLOOR`` onto its positions, exactly like the
    #: production V2_FLOOR arm would. They differ ONLY in their exit parameters,
    #: carried on :attr:`ArmSpec.exit_bracket` (TP/SL) and
    #: :attr:`ArmSpec.max_hold_s` (cap), which are snapshotted onto each
    #: position at open and read off the position afterwards. No profile, no
    #: gate and no threshold is changed: Arm A runs the exact researched
    #: bracket, and Arms B and C vary ONE exit parameter each.
    V2_BETA_A = "V2_BETA_A"
    V2_BETA_B = "V2_BETA_B"
    V2_BETA_C = "V2_BETA_C"

    #: PumpView 'Amarih' paired-exit PAPER experiment -- three arms, ONE
    #: canonical entry observation, differing only in the scheduled holding
    #: period at which a fresh executable SELL quote settles each arm's own
    #: isolated virtual ledger. Opt-in research; see amarih_experiment.py.
    AMARIH_T60 = "AMARIH_T60"
    AMARIH_T120 = "AMARIH_T120"
    AMARIH_T300 = "AMARIH_T300"


def _ratio_of(features: Any) -> float | None:
    """
    Read ``f_volume_to_liquidity`` off whatever the caller passed.

    THE PRODUCTION REPORT CARRIES A ``dict``, NOT A ``FeatureSet``:
    :meth:`DecisionReport` is built with ``features=features.as_dict()``. An
    earlier version of this filter used ``getattr`` only, so against a real
    production report it read ``None`` and H2_CAP60 refused EVERY signal,
    including ones whose true ratio was far below the ceiling. That is the
    worst possible failure for a research arm: not a loud crash, but a dataset
    that is empty for a reason that has nothing to do with the hypothesis.

    Both shapes are therefore accepted, and an absent value is still refused
    rather than passed -- a filter that admits what it cannot see is not a
    filter.
    """
    if features is None:
        return None
    if isinstance(features, Mapping):
        value = features.get("volume_to_liquidity")
    else:
        value = getattr(features, "volume_to_liquidity", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ArmSpec:
    """
    One arm's rule, with its provenance.

    ``production_profile`` is the :class:`StrategyProfile` stamped onto the
    position. The exit policy reads the max hold off the POSITION, never off a
    global, which is what makes one shared exit implementation produce two
    different hold limits for two arms without a second exit code path.

    ``exit_bracket`` is an OPTIONAL per-arm exit parameter. When set, it is the
    ``(take_profit_pct, stop_loss_pct)`` this arm's V2_FLOOR exit rule uses for
    ITS OWN positions, stamped at open and read off the position afterwards
    (exactly like ``production_profile``). When ``None`` the arm uses whatever
    its production profile declares, which is the pre-existing behaviour:
    adding this field changes NO existing arm. A take-profit is a PARAMETER of
    this experiment, not a new strategy, so the beta arms carry their exit
    numbers here instead of multiplying the closed profile vocabulary.
    ``max_hold_s`` is the arm's maximum hold, stamped the same way.
    """
    name: ResearchArm
    label: str
    production_profile: StrategyProfile
    max_hold_s: float | None
    #: Extra entry predicate applied ON TOP of the production L25 decision. Empty
    #: for the two arms that are pure L25.
    extra_entry_rule: str | None = None
    provenance: str = ""
    #: Optional per-arm V2 exit bracket ``(take_profit_pct, stop_loss_pct)``.
    #: ``None`` (every existing arm) means "use the production profile's".
    exit_bracket: tuple[float, float] | None = None

    def entry_allows(self, features: Any) -> tuple[bool, str]:
        """
        Does this arm's EXTRA rule (if any) admit the signal?

        Only ever called after the production L25 decision has been taken, so a
        rejection here means "L25 accepted and this arm's refinement did not" -
        which is recorded distinctly so the two causes never blur.
        """
        if self.name is ResearchArm.H2_CAP60:
            ratio = _ratio_of(features)
            if ratio is None:
                return False, (
                    "H2 requires f_volume_to_liquidity at decision time and it "
                    "is unavailable, so the filter refuses rather than passing "
                    "an absent value"
                )
            if float(ratio) <= 2.0:
                return True, f"f_volume_to_liquidity {float(ratio):.4f} <= 2.0"
            return False, (
                f"f_volume_to_liquidity {float(ratio):.4f} > 2.0 (the "
                "preregistered H2 turnover ceiling)"
            )
        return True, "no extra rule; this arm is pure production L25"

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.name.value,
            "label": self.label,
            "production_profile": self.production_profile.value,
            "max_hold_s": self.max_hold_s,
            "extra_entry_rule": self.extra_entry_rule,
            "provenance": self.provenance,
            "exit_bracket": (
                None if self.exit_bracket is None
                else {"take_profit_pct": self.exit_bracket[0],
                      "stop_loss_pct": self.exit_bracket[1]}
            ),
        }


_PROV_L25 = (
    "autopaper/strategy_profile.py DEFAULT_PROFILE; entry gate "
    "autopaper/l25.py evaluate() (hash-locked and re-verified on every call)"
)
_PROV_TP100 = (
    "autopaper/strategy_profile.py StrategyProfile.L25_TP100; TP100 take-profit "
    "at +100% executable/native return, implemented as ADDITIONAL check in "
    "shared FAST_FAILURE stack (exits.py). All existing L25 downside triggers "
    "preserved. From exit research: best fixed level on common sample but "
    "FAILS chronological validation. UNVALIDATED forward test only."
)
_PROV_H2 = (
    "research/l25r/hypotheses.py H2_turnover_ceiling (preregistered, rule "
    "'f_volume_to_liquidity <= 2.0', threshold explicitly NOT fitted) combined "
    "with the 60s cap as research/l25c/registry.py H2_CAP60"
)

_PROV_60S = (
    "Production profile autopaper.strategy_profile.StrategyProfile.L25_60S: same L25 "
    "entry gate, same existing exit rules, PLUS a 60s maximum hold appended LAST in "
    "the exit precedence (autopaper/exits.py). All downside triggers are preserved "
    "(fast-failure stop -12%, momentum collapse -18%/30s, structure break -14%/45s, "
    "no-follow-through, disaster cap -45%, data-invalid). The rule is not redefined "
    "here; this arm only makes an existing, already-shipped profile selectable again."
)

_PROV_MAX_HOLD = (
    "Production profile autopaper.strategy_profile.StrategyProfile.L25_MAX_HOLD: "
    "same L25 entry, same exit rules, plus a configurable hard max-hold cap (default "
    "60s, i.e. identical to L25_60S at its default). Exists so the cap is a tunable "
    "PAPER research parameter rather than a constant in a profile name."
)

_PROV_BETA = (
    "PAPER-ONLY parallel exit experiment on the EXISTING V2_FLOOR gate "
    "(autopaper/v2floor.py, unchanged; entry profile StrategyProfile.V2_FLOOR, "
    "sized at the profile's flat 10%%). The arm differs from production ONLY by "
    "its own exit parameters, snapshotted onto its positions at open. Arm A "
    "runs the exact researched bracket (TP +30 / SL -30 / 90s); Arm B raises "
    "only the take-profit (+50); Arm C raises only the take-profit (+50) and "
    "only the max hold (120s)."
)

_PROV_AMARIH = (
    "PAPER-ONLY paired-exit experiment on the PumpView 'Amarih' COMMUNITY "
    "strategy (pumpview.fun, owner tg:1458395479). Signals are read from the "
    "public /api/signal-history endpoint via autopaper.paper_live.pumpview; the "
    "three arms share ONE canonical executable-BUY-quote entry observation and "
    "differ ONLY in the scheduled holding period at which a fresh executable "
    "SELL quote settles each arm's OWN isolated virtual ledger (T30/T60/T120). "
    "The Amarih entry thresholds are PumpView's PRIVATE config and are NOT "
    "reimplemented here; this harness observes PumpView's already-decided "
    "signals. Experimental holding periods, NOT validated optima. See "
    "reports/research/amarih_pumpview_integration.md."
)

ARMS: Mapping[ResearchArm, ArmSpec] = {
    ResearchArm.L25_CURRENT: ArmSpec(
        name=ResearchArm.L25_CURRENT,
        label="L25 (production default, no max hold)",
        production_profile=StrategyProfile.L25_CURRENT,
        max_hold_s=None,
        provenance=_PROV_L25,
    ),
    ResearchArm.L25_TP100: ArmSpec(
        name=ResearchArm.L25_TP100,
        label="L25 + TP100 (take profit at +100%)",
        production_profile=StrategyProfile.L25_TP100,
        max_hold_s=None,
        provenance=_PROV_TP100,
    ),
    ResearchArm.L25_60S: ArmSpec(
        name=ResearchArm.L25_60S,
        label="L25 + 60s max hold",
        production_profile=StrategyProfile.L25_60S,
        max_hold_s=60.0,
        extra_entry_rule=None,
        provenance=_PROV_60S,
    ),
    ResearchArm.L25_MAX_HOLD: ArmSpec(
        name=ResearchArm.L25_MAX_HOLD,
        label="L25 + configurable max hold (default 60s)",
        production_profile=StrategyProfile.L25_MAX_HOLD,
        max_hold_s=60.0,
        extra_entry_rule=None,
        provenance=_PROV_MAX_HOLD,
    ),
    ResearchArm.H2_CAP60: ArmSpec(
        name=ResearchArm.H2_CAP60,
        label="L25 + H2 turnover ceiling (<= 2.0) + 60s max hold",
        production_profile=StrategyProfile.L25_60S,
        max_hold_s=60.0,
        extra_entry_rule="f_volume_to_liquidity <= 2.0",
        provenance=_PROV_H2,
    ),
    ResearchArm.V2_BETA_A: ArmSpec(
        name=ResearchArm.V2_BETA_A,
        label="V2 FLOOR (BETA) Arm A -- baseline TP +30 / SL -30 / 90s",
        production_profile=StrategyProfile.V2_FLOOR,
        max_hold_s=90.0,
        extra_entry_rule=None,
        provenance=_PROV_BETA,
        exit_bracket=(30.0, -30.0),
    ),
    ResearchArm.V2_BETA_B: ArmSpec(
        name=ResearchArm.V2_BETA_B,
        label="V2 FLOOR (BETA) Arm B -- TP test TP +50 / SL -30 / 90s",
        production_profile=StrategyProfile.V2_FLOOR,
        max_hold_s=90.0,
        extra_entry_rule=None,
        provenance=_PROV_BETA,
        exit_bracket=(50.0, -30.0),
    ),
    ResearchArm.V2_BETA_C: ArmSpec(
        name=ResearchArm.V2_BETA_C,
        label="V2 FLOOR (BETA) Arm C -- hold test TP +50 / SL -30 / 120s",
        production_profile=StrategyProfile.V2_FLOOR,
        max_hold_s=120.0,
        extra_entry_rule=None,
        provenance=_PROV_BETA,
        exit_bracket=(50.0, -30.0),
    ),
    ResearchArm.AMARIH_T60: ArmSpec(
        name=ResearchArm.AMARIH_T60,
        label="Amarih Research -- exit observed at +60s",
        production_profile=StrategyProfile.L25_CURRENT,
        max_hold_s=60.0,
        extra_entry_rule=None,
        provenance=_PROV_AMARIH,
    ),
    ResearchArm.AMARIH_T120: ArmSpec(
        name=ResearchArm.AMARIH_T120,
        label="Amarih Research -- exit observed at +120s",
        production_profile=StrategyProfile.L25_CURRENT,
        max_hold_s=120.0,
        extra_entry_rule=None,
        provenance=_PROV_AMARIH,
    ),
    ResearchArm.AMARIH_T300: ArmSpec(
        name=ResearchArm.AMARIH_T300,
        label="Amarih Research -- exit observed at +300s",
        production_profile=StrategyProfile.L25_CURRENT,
        max_hold_s=300.0,
        extra_entry_rule=None,
        provenance=_PROV_AMARIH,
    ),
}


#: Arms that are NOT offered for new selection.
#:
#: ``L25_TP100`` is retired as an ARM because a take-profit is a parameter, not
#: a strategy. It remains resolvable (so a persisted selection or a historical
#: position still names it) and still maps to +100%, but it is not SELECTABLE.
#: Keeping both properties is what makes removing the duplicate option safe: the
#: UI cannot offer it, while nothing already recorded loses its meaning.
NON_SELECTABLE_ARMS: frozenset[ResearchArm] = frozenset({ResearchArm.L25_TP100})


#: The V2 FLOOR (BETA) group selector: selecting it starts the three beta arms
#: together. It is NOT an arm itself -- it is the one option that expands to
#: three, so the operator chooses ONE EXPERIMENT and the harness runs three
#: independent ledgers. Individual beta arms remain resolvable by name (so a
#: persisted record or a diagnostics run can address one), but they are NOT
#: offered as standalone selections: starting Arm B alone would defeat the
#: comparison the experiment exists to produce.
BETA_GROUP = "V2_FLOOR_BETA"

BETA_ARMS: tuple[ResearchArm, ...] = (
    ResearchArm.V2_BETA_A,
    ResearchArm.V2_BETA_B,
    ResearchArm.V2_BETA_C,
)

#: Mapping from ResearchArm enum to beta experiment arm key ('A', 'B', 'C')
BETA_ARM_KEYS: dict[ResearchArm, str] = {
    ResearchArm.V2_BETA_A: "A",
    ResearchArm.V2_BETA_B: "B",
    ResearchArm.V2_BETA_C: "C",
}


#: The PumpView 'Amarih' paired-exit experiment group token. Selecting it starts
#: the three Amarih arms together (ONE experiment, THREE paired exit horizons).
#: Like the beta group it is NOT an arm itself; the three arm identities stay
#: private to the experiment and are resolvable by name for records but are not
#: offered as standalone selections (a single horizon is not the paired
#: comparison the experiment exists to produce).
AMARIH_GROUP = "AMARIH"

#: The three Amarih arms, in canonical horizon order T60/T120/T300.
AMARIH_ARMS: tuple[ResearchArm, ...] = (
    ResearchArm.AMARIH_T60,
    ResearchArm.AMARIH_T120,
    ResearchArm.AMARIH_T300,
)


def is_amarih_group(name: str) -> bool:
    """Whether a selection token is the Amarih paired-exit experiment group."""
    return str(name or "").strip().upper() == AMARIH_GROUP


def amarih_arm_specs() -> tuple[ArmSpec, ...]:
    """The three Amarih arms, in canonical order."""
    return tuple(ARMS[a] for a in AMARIH_ARMS)


def is_beta_group(name: str) -> bool:
    """Whether a selection token is the beta experiment group."""
    return str(name or "").strip().upper() == BETA_GROUP


def _is_beta_arm_token(name: str) -> bool:
    """Whether a token NAMES one of the beta arms (rather than the group).

    Used so a beta arm token still resolves to the WHOLE experiment: the arms
    are only ever meaningful together, and a selection that has been persisted
    as three arm names must round-trip back to three arms rather than being
    refused as "not selectable on its own".
    """
    return str(name or "").strip().upper() in {a.value for a in BETA_ARMS}


def _is_amarih_arm_token(name: str) -> bool:
    """Whether a token NAMES one of the Amarih arms (rather than the group).

    Used so an Amarih arm token resolves to the WHOLE paired experiment: the
    three horizons are only meaningful together, and a selection persisted as
    three arm names must round-trip back to the three-arm group rather than
    being refused or reduced to a single horizon.
    """
    return str(name or "").strip().upper() in {a.value for a in AMARIH_ARMS}


def beta_arm_specs() -> tuple[ArmSpec, ...]:
    """The three beta exit arms, in canonical order A, B, C."""
    return tuple(ARMS[a] for a in BETA_ARMS)


#: The two entry-gate tokens, RE-EXPORTED from ``strategy_profile`` rather than
#: restated here, so the gate vocabulary has exactly one home and a new gate
#: cannot be added to one place and forgotten in the other.
GATE_L25 = _GATE_L25
GATE_V2_FLOOR = _GATE_V2_FLOOR


def entry_gate_for_arm(spec: ArmSpec) -> str:
    """
    The entry gate ONE arm is governed by.

    This is the per-strategy half of the authorization contract: an arm is
    authorized by its OWN gate, never by whatever gate happens to be the
    runtime's global profile. That is what stops one strategy's approval from
    becoming another strategy's entry.

    The beta arms answer to ``V2_FLOOR`` because the experiment's gate IS
    ``autopaper.v2floor.evaluate``. Every other arm answers to the entry gate of
    the profile it was registered against -- so ``L25_CURRENT`` and the L25
    family resolve to ``L25``, and the single-book ``V2_FLOOR`` arm resolves to
    ``V2_FLOOR`` through its own ``production_profile`` rather than through a
    special case here.
    """
    from autopaper.strategy_profile import PROFILE_ENTRY_STRATEGY

    if spec.name in BETA_ARMS:
        return GATE_V2_FLOOR
    if spec.name in AMARIH_ARMS:
        return _GATE_AMARIH
    return PROFILE_ENTRY_STRATEGY[spec.production_profile]


def effective_entry_gates(selected_arms: Sequence[str], *,
                          profile: str | None = None) -> tuple[str, ...]:
    """
    THE EFFECTIVE-ENTRY-GATE CONTRACT.

    Given the arms the operator actually selected, return the set of entry
    gates that are OPERATIVE, in a stable order.

    WHY THIS EXISTS
    ---------------
    The decision layer used to take its gate from ``strategy_profile`` alone,
    and nothing connected that profile to the arms that were selected. The two
    axes are written by different code -- ``RuntimeState.select_strategies``
    writes ``selected_arms``, and the Telegram profile callback never writes
    ``strategy_profile`` in PAPER -- so selecting the V2 FLOOR (BETA) experiment
    left the profile on its shipped default of ``L25_CURRENT``. The decision
    layer then applied the L25 gate, and a candidate that satisfied the V2 FLOOR
    rule but failed L25 was REJECTED at ``engine.on_alert`` and never reached
    ``BetaExperiment.process_signal`` at all. The experiment could only ever
    trade the subset of candidates L25 already liked, which is not the
    experiment that was selected.

    THE RULE
    --------
    Each selected arm contributes ITS OWN gate (see :func:`entry_gate_for_arm`)
    and the union is operative. So:

    * the beta group selected -> ``V2_FLOOR`` is operative, and an L25-only
      refusal can no longer exclude a V2 FLOOR-eligible candidate;
    * ``L25_CURRENT`` selected alongside it -> ``L25`` is operative TOO, and the
      decision authorizes on EITHER. Each arm is then filtered by its own gate
      downstream, so L25 still refuses what L25 refuses and beta still refuses
      what V2 FLOOR refuses;
    * beta is NEVER contributed by an empty or unknown selection -- an empty
      selection resolves to the declared default arm, so this cannot activate
      the experiment by default or from a legacy config.

    WHAT ``profile`` DOES, AND WHY IT IS NOT SIMPLY ADDED
    -----------------------------------------------------
    ``profile`` contributes its gate ONLY when it is a NON-DEFAULT profile.
    That asymmetry is the whole point:

    * An operator who explicitly selects ``V2_FLOOR`` / ``V2_FLOOR_BETA`` in
      config has stated an entry-gate intent, and it is honoured. There is no
      ``V2_FLOOR`` research ARM, so without this an explicit V2 FLOOR profile
      would be silently downgraded to L25 by its own arms.
    * The shipped DEFAULT (``L25_CURRENT``) is not an instruction -- it is what
      an operator who never chose anything is left holding. Letting it
      contribute would put L25 back in front of the beta experiment and
      reintroduce exactly the veto this contract exists to remove.

    So: explicit choice is authoritative, an unset default is not. An empty or
    wholly unresolvable selection falls back to the default arm's gate, never
    to beta: "nothing was chosen" is not "run the experiment".
    """
    gates: list[str] = []
    try:
        specs = parse_arms(list(selected_arms or ()))
    except Exception:  # noqa: BLE001 - an unresolvable selection refuses
        specs = ()
    if not specs:
        return (entry_gate_for_arm(ARMS[DEFAULT_ARM]),)
    for spec in specs:
        gate = entry_gate_for_arm(spec)
        if gate not in gates:
            gates.append(gate)
    if profile:
        from autopaper.strategy_profile import (
            DEFAULT_PROFILE,
            PROFILE_ENTRY_STRATEGY,
            parse_profile,
        )

        try:
            chosen = parse_profile(profile)
        except Exception:  # noqa: BLE001 - an unreadable profile contributes
            chosen = None
        if chosen is not None and chosen is not DEFAULT_PROFILE:
            gate = PROFILE_ENTRY_STRATEGY[chosen]
            if gate not in gates:
                gates.append(gate)
    # Stable order: L25 first, then V2_FLOOR. Deterministic for the journal,
    # the UI status line and the safety audit.
    return tuple(sorted(gates, key=lambda g: (g != GATE_L25, g)))


def selectable_arms() -> tuple[ResearchArm, ...]:
    """Arms offered for new selection, in canonical order.

    The beta arms are selectable ONLY as the group (``V2_FLOOR_BETA``), which is
    included here so the picker offers exactly one beta option. Listing the
    three beta arms separately would let an operator start a partial experiment;
    each beta arm stays resolvable by :func:`arm_by_name` for records and
    diagnostics.
    """
    return tuple(a for a in ResearchArm
                 if a not in NON_SELECTABLE_ARMS
                 and a not in BETA_ARMS
                 and a not in AMARIH_ARMS)


def selectable_arm_names() -> tuple[str, ...]:
    """The per-arm selector vocabulary (excludes the beta GROUP token).

    :func:`arm_names` -- what the picker actually shows -- is this list with
    ``V2_FLOOR_BETA`` appended, so the operator sees ONE beta experiment option,
    not three partial ones.
    """
    return tuple(a.value for a in selectable_arms())


def arm_names() -> tuple[str, ...]:
    """
    Every SELECTABLE arm name, in canonical order.

    THE selector's list. The UI and the config layer read this instead of
    keeping their own copy, so a strategy added to the registry cannot be
    silently missing from the picker -- which is exactly how H2_CAP60 went
    missing in the first place.

    This is the SELECTABLE set, not every registered enum value: an arm retired
    as a choice must disappear from the picker without its token ceasing to
    resolve for existing records.

    BETA IS DELIBERATELY ABSENT
    ---------------------------
    ``V2_FLOOR_BETA`` is not an ARM -- it is the GROUP token that expands to
    three. It is offered by the SELECTOR (the Telegram menu appends it as its
    own row) and accepted by :func:`parse_arms`, but it is not part of the
    arm-name vocabulary: one entry here means one arm, and
    ``len(arm_names()) == len(arm_specs())`` is an invariant other tests assert.

    Because it is absent here, an EMPTY selection -- a fresh install, a missing
    config, a legacy state file -- resolves to the arms this list names and
    NEVER starts the three-arm exit experiment. Beta has to be asked for by
    name, in exactly one place: :func:`is_beta_group`.

    ``AMARIH`` -- the PumpView paired-exit experiment GROUP -- follows the SAME
    pattern for the SAME reason: it is not an arm (so it is NOT appended here
    and ``len(arm_names()) == len(arm_specs())`` stays true), it expands to
    three (T30/T60/T120) via :func:`parse_arms`, and it can never be started by
    an empty or legacy selection. It is offered by the SELECTOR as its own row
    and recognised by :func:`is_selectable_arm` / :func:`is_amarih_group`.
    Unlike beta, Amarih is ENABLED (opt-in research).
    """
    return tuple(a.value for a in selectable_arms())


def is_selectable_arm(name: str) -> bool:
    text = str(name or "").strip().upper()
    return text in set(selectable_arm_names()) | {BETA_GROUP, AMARIH_GROUP}


#: THE arm a runtime runs when NO selection has been restored.
#:
#: Declared here, next to the registry it names, rather than at the call site, so
#: "the default" is one value with one definition. It is
#: :class:`ResearchArm` ``L25_CURRENT`` because that is the production default
#: strategy -- the same one ``strategy_profile.DEFAULT_PROFILE`` names for the
#: profile vocabulary, and the two are asserted to agree in tests so the arm
#: layer and the profile layer cannot drift into disagreeing about what the
#: default is.
#:
#: It is emphatically NOT ``parse_arms([])``, which resolves to EVERY arm. That
#: distinction is the whole ``L25_TP100`` selection bug: an unselected default
#: that happens to include a research profile is not a default, it is a strategy
#: the operator never asked for.
DEFAULT_ARM: ResearchArm = ResearchArm.L25_CURRENT


def arm_specs() -> tuple[ArmSpec, ...]:
    """Every SELECTABLE :class:`ArmSpec`, in canonical order."""
    return tuple(ARMS[a] for a in selectable_arms())


def arm_by_name(name: str) -> ArmSpec:
    """
    Resolve an arm by name, failing closed.

    An unrecognised name RAISES rather than defaulting. Silently falling back to
    ``L25_CURRENT`` would mean an operator who typo'd an arm name spent a
    research run collecting data for a different strategy than the one they
    asked for, and nothing in the output would say so.
    """
    text = str(name or "").strip().upper()
    for arm in ResearchArm:
        if arm.value == text:
            return ARMS[arm]
    raise ValueError(
        f"unknown research arm {name!r}; supported arms are "
        + ", ".join(a.value for a in ResearchArm)
    )


def arm_by_name_or_none(name: str) -> ArmSpec | None:
    """
    Resolve a RETIRED arm to its spec, or ``None`` for a genuinely unknown name.

    Only used to keep a persisted selection meaningful after a retira: it is not
    offered by any UI path and cannot be used to START something new.
    """
    text = str(name or "").strip().upper()
    for arm in NON_SELECTABLE_ARMS:
        if arm.value == text:
            return ARMS[arm]
    return None


def parse_arms(names: Sequence[str]) -> tuple[ArmSpec, ...]:
    """
    Resolve a comma-separated arm list.

    An EMPTY list means EVERY SELECTABLE arm, and that is a DELIBERATE,
    EXPLICIT meaning rather than a default: ``parse_arms([])`` is how the
    standalone multi-arm research harness asks for all of them.

    "Every" is the SELECTABLE set, and it must be exactly what
    :func:`arm_names` returns -- the same vocabulary the picker and the config
    layer read. It used to be every REGISTERED arm, which silently included a
    retired one and made the two disagree about how many arms exist.

    It is NOT what an unselected runtime resolves to. That is
    :data:`DEFAULT_ARM`, resolved by ``RuntimeState.arms``, because "run
    everything" and "the operator chose nothing" are different situations and
    conflating them is what produced the ``L25_TP100`` selection defect.
    A RETIRED arm is refused rather than resolved: an arm that can no longer be
    selected must not be startable through the back door of a persisted list or
    a config file.

    ``V2_FLOOR_BETA`` (the group) is NO LONGER ACTIVE. The V2 FLOOR (BETA)
    experiment has been disabled from the active bot. Any selection that names
    the group token or a beta arm token (V2_BETA_A, V2_BETA_B, V2_BETA_C) is
    REFUSED rather than expanded, so the experiment cannot be started through
    the selector, a persisted list, or a config file.

    A token that names a BETA ARM (``V2_BETA_A``) previously resolved to the
    SAME group, not to that arm alone. Running one arm of a three-arm comparison
    is a different experiment and must never be mistaken for it, so a partial
    group is not a reachable outcome.

    The beta arm specs, enum values, and historical records are preserved: they
    remain resolvable by :func:`arm_by_name` for diagnostics and existing
    position records, but they are NOT offered for new selection and the group
    token is refused.
    """
    wanted = [n.strip() for n in names if str(n).strip()]
    if not wanted:
        # Empty means every SELECTABLE arm -- and the beta arms are NOT
        # selectable on their own, so an unselected configuration cannot
        # accidentally start the V2 FLOOR (BETA) experiment. The Amarih arms
        # are likewise group-only, so an empty selection never starts Amarih
        # either: it resolves to the L25-family production arms.
        return tuple(ARMS[a] for a in selectable_arms())
    out: list[ArmSpec] = []
    for n in wanted:
        if is_beta_group(n) or _is_beta_arm_token(n):
            # V2 FLOOR (BETA) is disabled. Refuse rather than expand.
            raise ValueError(
                f"V2 FLOOR (BETA) is disabled from the active bot. "
                f"Selection of {n!r} is no longer accepted. "
                "The experiment's code and historical records are preserved, "
                "but no new beta positions can be opened.")
        if is_amarih_group(n) or _is_amarih_arm_token(n):
            # Amarih is a GROUP: selecting the group token OR any single arm
            # token starts the WHOLE paired experiment (T30/T60/T120), never a
            # partial one -- running one horizon is a different experiment and
            # must never be mistaken for the paired comparison.
            out.extend(amarih_arm_specs())
            continue
        spec = arm_by_name(n)
        if spec.name in NON_SELECTABLE_ARMS:
            raise ValueError(
                f"arm {spec.name.value!r} is retired and is no longer selectable; "
                "take profit is a SETTING (paper_live.take_profit), not an arm")
        out.append(spec)
    # de-duplicate while preserving order, so "V2_FLOOR_BETA,V2_FLOOR_BETA" or a
    # persisted list that already contains a beta arm cannot double-book it.
    seen: set[ResearchArm] = set()
    deduped: list[ArmSpec] = []
    for spec in out:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        deduped.append(spec)
    return tuple(deduped)


def describe_arm(arm: ResearchArm | str) -> dict[str, Any]:
    spec = arm if isinstance(arm, ArmSpec) else arm_by_name(str(arm))
    return spec.as_dict()


__all__ = [
    "arm_names",
    "arm_specs",
    "selectable_arms",
    "selectable_arm_names",
    "is_selectable_arm",
    "is_beta_group",
    "beta_arm_specs",
    "is_amarih_group",
    "amarih_arm_specs",
    "AMARIH_GROUP",
    "AMARIH_ARMS",
    "entry_gate_for_arm",
    "effective_entry_gates",
    "GATE_L25",
    "GATE_V2_FLOOR",
    "BETA_ARMS",
    "BETA_GROUP",
    "ARMS",
    "ArmSpec",
    "ResearchArm",
    "arm_by_name",
    "describe_arm",
    "parse_arms",
]
