"""
The AI decision engine.

Produces a ``DecisionReport`` for every candidate: BUY / WAIT / REJECT, with a
machine-readable rationale, an explicit pre-declared invalidation set, and an
exit plan written before any position exists.

Hard rules enforced here:

* Only information available at ``decision_time`` is consulted.
* No future outcomes are read. The engine receives a ``FeatureSet`` that was
  already gated by ``ObservationSet.as_of`` and a ``ResearchMemory`` snapshot
  that was already gated by ``exit_timestamp <= decision_time``.
* ``confidence`` is a decision-strength score, explicitly not a probability of
  profit.
* Weak evidence resolves to WAIT or REJECT, never to a forced BUY.

THE SELECTED PROFILE'S GATE IS THE ONLY ENTRY AUTHORIZATION
-----------------------------------------------------------
``BUY`` is returned if and only if the entry gate of the SELECTED profile
authorizes the candidate: L25 (which delegates its first leg to V3-C) for the
L25 family, or ``autopaper.v2floor.evaluate`` for ``V2_FLOOR``. Those gates are
BLOCKING, so a candidate that fails one comes back as ``REJECT`` with the exact
threshold and value in its rationale, and a blocking risk flag is journaled with
the decision. There is no other route to ``BUY`` in this class.

The gate is chosen by ``DecisionConfig.strategy_profile``, which is validated
fail-closed at config load. The profile is read HERE and never inside an entry
module, so ``l25.py`` / ``v3c.py`` / ``v2floor.py`` remain unable to know a
profile exists -- the invariant the existing entry-immutability tests assert.

Everything downstream of the gate is shared and unchanged: the pipeline route
selection and its structural refusals, the confidence bar, the data-state
handling, and the entire execution path. V2_FLOOR changes WHICH gate
authorizes; it does not weaken any structural or safety check.
"""

from __future__ import annotations

from typing import Mapping

from ..clock import iso
from ..config import DecisionConfig, ExitConfig
from ..logging_setup import log
from ..memory import ResearchMemory
from ..models import (
    AlertSnapshot,
    DataState,
    Decision,
    DecisionReport,
    NarSnapshot,
    Regime,
    RiskFlag,
    Route,
)
from ..l25 import (
    ACTIVE_STRATEGY,
    REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD,
    UNTESTABLE as L25_UNTESTABLE,
    L25Result,
    evaluate as evaluate_l25,
    f_liq_to_mcap,
)
from ..strategy_profile import (
    GATE_L25 as _GATE_L25,
    GATE_V2_FLOOR as _GATE_V2_FLOOR,
    KNOWN_ENTRY_GATES as _KNOWN_ENTRY_GATES,

    PROFILE_ENTRY_STRATEGY,
    StrategyProfile,
    parse_profile,
)
from ..v2a import V2A_UNTESTABLE, apply_v2a
from ..v2floor import V2FloorResult
from ..v2floor import evaluate as evaluate_v2floor
from ..v2floor import flag_code as v2floor_flag_code
from ..v3c import (
    REJECTED_MARKET_CAP,
    REJECTED_SIGNAL_QUALITY,
    UNTESTABLE as UNTESTABLE_STATUS,
    V3C_MAX_MARKET_CAP_USD,
    V3C_MIN_SIGNAL_QUALITY,
    V3C_NAME,
    V3CResult,
    evaluate as evaluate_v3c,
    v2a_note,
)
from .features import FeatureSet
from .regime import classify
from .routes import (
    build_exit_plan,
    exit_style_for,
    select_route,
)


class DecisionEngine:
    def __init__(
        self,
        decision_cfg: DecisionConfig,
        exit_cfg: ExitConfig,
        *,
        memory: ResearchMemory | None = None,
    ) -> None:
        self.cfg = decision_cfg
        self.exit_cfg = exit_cfg
        self.memory = memory
        # Resolved ONCE here and re-asserted in ``decide``. ``DecisionConfig``
        # validates the profile fail-closed at construction, so a bad value
        # cannot reach this point; parsing again costs nothing and means the
        # selected gate cannot be anything other than a declared profile even
        # if a caller mutated the dataclass after construction.
        self._selected_profile: StrategyProfile = parse_profile(
            decision_cfg.strategy_profile
        )
        self._selected_entry: str = PROFILE_ENTRY_STRATEGY[self._selected_profile]
        #: The entry gates OPERATIVE for the current selection. Defaults to the
        #: profile's own gate, which is exactly the pre-existing behaviour, so a
        #: caller that knows nothing about arms (the research harness, the
        #: unit tests) is unaffected. :meth:`set_entry_gates` is how the
        #: runtime narrows or widens it to match the arms actually selected --
        #: see ``paper_live.arms.effective_entry_gates``.
        self._entry_gates: tuple[str, ...] = (self._selected_entry,)

    # -- the effective-entry-gate contract ---------------------------------

    @property
    def entry_gates(self) -> tuple[str, ...]:
        """The entry gates currently operative, in stable order."""
        return self._entry_gates

    def set_entry_gates(self, gates: object) -> tuple[str, ...]:
        """
        Set the operative entry gates from the runtime's arm selection.

        Fails CLOSED: an empty, unreadable or entirely unrecognised value falls
        back to the profile's own gate rather than to "no gate", because a gate
        set that authorizes everything is the one value that must never be
        reachable by accident. The result is always a non-empty tuple drawn
        from the DECLARED gate tokens -- :data:`KNOWN_ENTRY_GATES`, not a
        hand-copied pair.

        WHY THE SHARED CONSTANT MATTERS
        ------------------------------
        This filter used to hard-code ``(_GATE_L25, _GATE_V2_FLOOR)`` while
        ``strategy_profile`` already declared ``GATE_AMARIH`` in
        ``KNOWN_ENTRY_GATES``. So ``effective_entry_gates`` correctly returned
        ``("AMARIH",)`` for a selected Amarih experiment and this method threw
        the token away, fell through to the profile's own gate and left ``L25``
        operative -- silently, and with no log line, because the caller only
        logs when the APPLIED value changes and ``L25 -> L25`` is not a change.
        The persisted selection then said "AMARIH" while the runtime traded L25.

        Two lists can drift, and this one did. Filtering against the single
        declared vocabulary makes a newly declared gate operative the moment it
        is declared, and keeps the "unrecognised values fail closed" property
        exactly as it was.
        """
        fallback = (self._selected_entry,)
        try:
            wanted = [str(g).strip().upper() for g in (gates or ())]
        except TypeError:
            wanted = []
        clean: list[str] = []
        for gate in wanted:
            if gate in _KNOWN_ENTRY_GATES and gate not in clean:
                clean.append(gate)
        self._entry_gates = tuple(clean) if clean else fallback
        return self._entry_gates

    @property
    def _operative_gates(self) -> tuple[str, ...]:
        return self._entry_gates or (self._selected_entry,)

    # -- public ------------------------------------------------------------

    def decide(
        self,
        *,
        features: FeatureSet,
        alert: AlertSnapshot | None,
        nar: NarSnapshot | None = None,
        analysis_start: float,
        analysis_complete: float,
        blocked_reason: str | None = None,
        account_blocked_reason: str | None = None,
    ) -> DecisionReport:
        regime, regime_reason = classify(features)
        choice = select_route(features, self.cfg)

        flags: list[RiskFlag] = self._risk_flags(features, nar, regime)
        flags.extend(self._alert_risk_flags(alert))
        if blocked_reason:
            flags.append(RiskFlag("market_data_blocked", blocked_reason, "block"))
        if account_blocked_reason:
            flags.append(RiskFlag("account_limit", account_blocked_reason, "block"))

        if (
            nar is not None
            and nar.error
            and str(nar.error).startswith("NARRATIVE_PENDING:")
        ):
            flags.append(
                RiskFlag(
                    "nar_pending",
                    "narrative task was still running at the L25 decision; "
                    "no narrative evidence was used",
                    "info",
                )
            )
        elif not nar or not nar.available:
            detail = (
                "narrative bot has nothing for this token"
                if (nar is not None and nar.error and "PLACEHOLDER" in nar.error)
                else "narrative unavailable: security, holder and wallet-flow "
                     "evidence is absent (reported as null, not assumed)"
            )
            flags.append(RiskFlag("nar_unavailable", detail, "warn"))
        elif nar.narrative:
            # The bot supplies prose only. Say so, so nobody reads the presence
            # of a narrative as a metrics read-out.
            flags.append(
                RiskFlag(
                    "nar_narrative_only",
                    f"narrative supplied ({len(nar.narrative)} chars, untrusted "
                    "prose); no holder, security or wallet-flow data from this "
                    "source",
                    "info",
                )
            )

        # --- the SELECTED PROFILE's entry gate -------------------------
        #
        # Two independent, independently selectable gates:
        #
        #   L25        = V3-C AND f_liq_to_mcap >= 0.9647
        #   V2_FLOOR   = vol5/mcap >= 1.0 AND 5m change in [+10%, +300%]
        #
        # V2_FLOOR is NOT a superset or subset of L25. It carries no
        # market-cap ceiling and no liquidity floor, because the research
        # tested and rejected both. Which gate runs is decided by the profile
        # validated at config load, and BOTH are always evaluated and reported
        # so the journal records the operative one and a refusal can always be
        # attributed to a specific threshold.
        #
        # L25's own body (its V3-C delegation and its thresholds) is UNCHANGED,
        # and is still evaluated under a V2_FLOOR profile so its refusal reason
        # is available for diagnosis. It simply does not authorize that profile.
        #
        #     L25  = V3-C AND f_liq_to_mcap >= 0.9647
        #     V3-C = route == FAST_ENTRY
        #            AND decision-time market cap <= $20,000
        #            AND decision-time signal quality >= 0.55
        #
        # L25 delegates its V3-C leg to autopaper.v3c, so the V3-C thresholds
        # have exactly one implementation and are not restated here.
        #
        # Applied AFTER route classification so the pipeline can still refuse a
        # structurally untradeable setup, and read from ``features`` so every
        # value is the one observed at the decision time - never the alert-time
        # figure, never a later print.
        #
        # The gates are BLOCKING, so _decide() returns REJECT and the exact
        # threshold and value are journaled with every decision. An unreadable
        # value is UNTESTABLE, never an automatic pass - including a missing or
        # non-positive liquidity, which refuses rather than substituting a value.
        #
        # BOTH gates are ALWAYS evaluated. Only an OPERATIVE one authorizes.
        # Evaluating a non-operative one too is deliberate: it means a
        # V2_FLOOR journal row can still answer "would L25 have taken this?",
        # which is what a control A/B needs, and a refusal is never silent
        # about which gate was operative.
        self._selected_profile = parse_profile(self.cfg.strategy_profile)
        self._selected_entry = PROFILE_ENTRY_STRATEGY[self._selected_profile]
        operative = self._operative_gates

        l25 = evaluate_l25(
            route=choice.route.value,
            market_cap_usd=features.market_cap_usd,
            signal_quality=features.signal_quality,
            liquidity_usd=features.liquidity_usd,
            min_liq_to_mcap=self.cfg.l25_min_liq_to_mcap,
        )
        # The delegated V3-C sub-result, so a refusal can always say WHICH gate
        # was the operative one.
        v3c = l25.v3c if l25.v3c is not None else evaluate_v3c(
            route=choice.route.value,
            market_cap_usd=features.market_cap_usd,
            signal_quality=features.signal_quality,
            max_market_cap_usd=self.cfg.v3c_max_market_cap_usd,
            min_signal_quality=self.cfg.v3c_min_signal_quality,
        )
        v2floor = evaluate_v2floor(
            volume_5m_usd=features.volume_5m_usd,
            market_cap_usd=features.market_cap_usd,
            change_5m_pct=features.change_5m,
        )
        # Only the SELECTED gate may raise a BLOCKING flag. The other gate's
        # refusal is still attached at "info" severity so it is journaled and
        # never mistaken for the reason this decision was refused.
        #
        # OPERATIVE, NOT "SELECTED". A gate blocks when it is operative for the
        # CURRENT SELECTION -- which, once the beta experiment is running, is
        # more than one gate at a time. A gate that is not operative still has
        # its refusal recorded at "info" so the journal keeps the counterfactual
        # ("would L25 have taken this?"), exactly as before, but it can no
        # longer be what refuses a candidate another selected strategy wanted.
        if _GATE_V2_FLOOR in operative:
            if v2floor.rejects:
                flags.append(
                    RiskFlag(v2floor_flag_code(v2floor), v2floor.detail,
                             "block")
                )
        if l25.rejects:
            flags.append(
                RiskFlag(_strategy_flag_code(l25), l25.detail, "block")
                if _GATE_L25 in operative
                else RiskFlag(_strategy_flag_code(l25),
                              f"[not the selected gate for this selection; "
                              f"operative gate(s) {','.join(operative)}] "
                              f"{l25.detail}", "info")
            )
        if v2floor.rejects and _GATE_V2_FLOOR not in operative:
            flags.append(
                RiskFlag(v2floor_flag_code(v2floor),
                         f"[not the selected gate for this selection; "
                         f"operative gate(s) {','.join(operative)}] "
                         f"{v2floor.detail}", "info")
            )

        # --- V2-A: the retained, wider ceiling ---------------------------
        # Still evaluated and still reported. V3-C's $20,000 is the binding
        # limit, so a token between $20,000 and $25,372.71 passes V2-A and
        # still fails V3-C. V2-A is dominated, not deleted.
        #
        # SCOPED TO THE L25 FAMILY. V2-A is an L25 sub-result: it exists to
        # report that a candidate cleared the wider ceiling and still failed
        # V3-C's tighter one. Under V2_FLOOR there is no $20,000 V3-C ceiling,
        # so a V2-A block would be a market-cap filter the V2_FLOOR rule
        # explicitly does not have -- precisely the "MC cap" the research
        # rejected. It is therefore attached at "info" under V2_FLOOR: still
        # journaled, never authoritative, never blocking.
        v2a = apply_v2a(choice.route.value, features.market_cap_usd,
                        self.cfg.v2a_ceiling_usd)
        # V2-A is an L25 sub-result, so it binds whenever L25 is OPERATIVE --
        # which is now a question about the selection rather than about the
        # profile. With beta alone selected, L25 is not operative and V2-A is
        # correctly journaled but not authoritative.
        if _GATE_L25 in operative:
            if v2a.rejects and v3c.status != REJECTED_MARKET_CAP:
                flags.append(
                    RiskFlag("v2a_market_cap_ceiling", v2a.detail, "block")
                )
            elif v2a.status == V2A_UNTESTABLE:
                flags.append(
                    RiskFlag("v2a_untestable", v2a.detail, "warn")
                )
        else:
            if v2a.rejects:
                flags.append(
                    RiskFlag(
                        "v2a_market_cap_ceiling",
                        f"[L25 sub-result, not authoritative under "
                        f"{self._selected_profile.value}] {v2a.detail}",
                        "info",
                    )
                )
            elif v2a.status == V2A_UNTESTABLE:
                flags.append(
                    RiskFlag(
                        "v2a_untestable",
                        f"[L25 sub-result, not authoritative under "
                        f"{self._selected_profile.value}] {v2a.detail}",
                        "info",
                    )
                )

        blocking = [f for f in flags if f.severity == "block"]

        # SCOPE THE BLOCKING FLAGS TO WHO OWNS THEM, once, here.
        #
        # A strategy-gate refusal is that strategy's verdict on this candidate;
        # a structural refusal is a property of the candidate. With one gate
        # selected the two were interchangeable, which is why this split was
        # never needed. With the beta experiment selected ALONGSIDE L25 they are
        # not interchangeable -- L25 declining a $50,000 market cap is the L25
        # arm declining, not the beta experiment being refused, and treating it
        # as a blanket block is precisely the veto this contract removes.
        gate_blocking, structural_blocking = _split_blocking(blocking)
        _gate_results = {_GATE_L25: l25, _GATE_V2_FLOOR: v2floor}
        operative_results = [_gate_results[g] for g in operative
                             if g in _gate_results]
        any_operative_authorized = any(r.authorized for r in operative_results)
        # What collapses confidence: a structural refusal always, and a gate
        # refusal only when NO operative gate authorized. With a single gate
        # those two conditions are exactly the old ``if blocking: return 0.05``,
        # so single-gate confidence is unchanged.
        confidence_blocking = (structural_blocking
                               or (gate_blocking
                                   if not any_operative_authorized else []))

        # -- memory-aware route adjustment (time-correct) ----------------
        multiplier, sample = 1.0, 0
        if self.memory is not None and choice.route is not Route.NO_TRADE:
            multiplier, sample = self.memory.route_multiplier(
                choice.route, features.decision_time, regime.value
            )
            if sample >= self.memory.min_sample and multiplier < 0.85:
                flags.append(
                    RiskFlag(
                        "route_demoted",
                        f"{choice.route.value} in {regime.value} has paper history of "
                        f"x{multiplier:.2f} over {sample} closed trades",
                        "warn",
                    )
                )

        # ResearchMemory's multiplier is OBSERVABILITY ONLY. It used to scale
        # confidence via `base * (0.75 + 0.25 * multiplier)`, so adaptive
        # behaviour that is not part of the researched L25 could lower the
        # score for a candidate. The researched population had no such
        # adjustment, so 1.0 is passed to keep live equal to research. The
        # multiplier is still computed, still reported, and still journaled
        # below - it simply cannot alter the decision.
        confidence = self._confidence(
            features, choice.route, regime, confidence_blocking, 1.0
        )

        extended_permission = any(
            m.startswith("extended_") for m in choice.modifiers
        )
        decision = self._decide(
            features=features,
            regime=regime,
            route=choice.route,
            blocking=blocking,
            confidence=confidence,
            extended_permission=extended_permission,
            v3c=v3c,
            l25=l25,
            v2floor=v2floor,
            gate_blocking=gate_blocking,
            structural_blocking=structural_blocking,
            any_operative_authorized=any_operative_authorized,
        )

        thesis = self._thesis(
            features, regime, regime_reason, choice.rationale, decision, nar,
            v3c, l25, v2floor,
        )
        invalidation = self._invalidation(features, choice, regime)
        style = exit_style_for(choice.route, regime, choice.style_override)
        # target_position_pct is deliberately not a route output. Sizing is the
        # frozen V3-C policy in autopaper.sizing and is applied at execution
        # time from live equity, so no plan can carry a stale percentage.
        plan = build_exit_plan(
            choice.route, regime, style, features, self.exit_cfg,
        )
        plan["route_rationale"] = choice.rationale
        plan["route_modifiers"] = list(choice.modifiers)

        if decision is Decision.BUY:
            entry_conditions = self._entry_conditions(features, choice, plan)
        else:
            entry_conditions = self._watch_conditions(features, decision)

        report = DecisionReport(
            candidate_id=features.candidate_id,
            mint=features.mint,
            symbol=(alert.symbol if alert else None),
            decision=decision,
            regime=regime,
            route=choice.route,
            signal_quality=features.signal_quality,
            confidence=confidence,
            thesis=thesis,
            risk_flags=tuple(flags),
            entry_conditions=tuple(entry_conditions),
            invalidation_conditions=tuple(invalidation),
            exit_plan=plan,
            preferred_exit_style=style,
            decided_at=analysis_complete,
            snapshot=alert,
            features=features.as_dict(),
            data_state=features.data_state,
            latency_ms=max(0.0, (analysis_complete - analysis_start) * 1000.0),
            nar_used=bool(nar and nar.available),
            route_multiplier=multiplier,
            route_sample_size=sample,
            v3c=v3c,
            l25=l25,
            v2floor=v2floor,
            strategy_profile=self._selected_profile.value,
        )

        # The logged strategy name is the SELECTED gate, not a constant. Under
        # the L25 family this is byte-identical to the previous log line
        # (ACTIVE_STRATEGY == "L25", l25.status); under V2_FLOOR it names the
        # gate that actually authorized or refused.
        log(
            "ANALYSIS",
            "%s %s %s/%s reg=%s route=%s sq=%.2f conf=%.2f ext=%s data=%s "
            "(%dms) %s",
            report.decision.value,
            (report.symbol or report.mint or "?")[:12],
            self._selected_entry,
            (v2floor if self._selected_entry == "V2_FLOOR" else l25).status,
            report.regime.value,
            report.route.value,
            report.signal_quality,
            report.confidence,
            _fmt_pct(features.extension_from_first_valid_price),
            features.data_state.value,
            report.latency_ms,
            "; ".join(f.code for f in report.risk_flags) or "no flags",
        )
        return report

    # -- decision ----------------------------------------------------------

    def _decide(
        self,
        *,
        features: FeatureSet,
        regime,
        route: Route,
        blocking: list[RiskFlag],
        confidence: float,
        extended_permission: bool = False,
        v3c: V3CResult,
        l25: L25Result,
        v2floor: V2FloorResult,
        gate_blocking: list[RiskFlag] | None = None,
        structural_blocking: list[RiskFlag] | None = None,
        any_operative_authorized: bool | None = None,
    ) -> Decision:
        # Re-read the operative set here rather than threading it through the
        # signature: it is engine state, not a per-candidate argument, and a
        # caller that has just changed the arm selection must see the change on
        # the very next candidate.
        operative = self._operative_gates
        if features.data_state is DataState.STALE:
            # Out-of-date data is a TIMING problem, not a verdict on the token.
            # If nothing blocks it structurally, the answer is WAIT: re-examine
            # on a live print. Only a structural blocker (thin liquidity, a
            # reversing token, an exhausted entry) makes it a refusal, because
            # those would still be true on a fresher print.
            _gate_flags, structural = _split_blocking(blocking)
            if not structural:
                return Decision.WAIT
        # A STRATEGY-GATE refusal and a STRUCTURAL refusal are not the same
        # kind of thing, and with more than one gate operative they must not be
        # treated as one.
        #
        # A gate flag (``l25_``/``v3c_``/``v2floor_``) is the verdict of ONE
        # strategy on this candidate. When several strategies are selected, one
        # of them saying no is exactly what per-strategy authorization means:
        # that arm declines, the others still get their own answer. Letting it
        # REJECT the whole candidate is how the L25 gate suppressed the beta
        # experiment.
        #
        # Everything else -- absent market data, a refused route, an account
        # limit -- is structural: it is a property of the candidate, not of any
        # strategy, and it refuses for everyone.
        if gate_blocking is None or structural_blocking is None:
            gate_blocking, structural_blocking = _split_blocking(blocking)
        gate_flags, structural_flags = gate_blocking, structural_blocking
        if structural_flags:
            return Decision.REJECT
        if any_operative_authorized is None:
            # Only reached by a direct ``_decide`` call that did not come from
            # ``decide()``; recomputed here so the fallback cannot reference a
            # name that lives in the caller.
            _results = {_GATE_L25: l25, _GATE_V2_FLOOR: v2floor}
            any_operative_authorized = any(
                _results[g].authorized for g in operative if g in _results)
        any_authorized = any_operative_authorized
        if gate_flags and not any_authorized:
            # Every operative gate that flagged refused, and none authorized.
            # With a SINGLE operative gate this is exactly the old behaviour:
            # its own refusal refuses the candidate.
            return Decision.REJECT
        if features.data_state is not DataState.LIVE:
            # Out-of-date data is a TIMING problem, not a verdict on the token.
            # A genuinely good setup judged on a 20s-old print is WAIT - look
            # again in a few seconds - not a permanent REJECT. (Blocking flags
            # above have already handled the cases that are reject-worthy on
            # their own merits, e.g. genuinely thin liquidity.)
            return Decision.WAIT
        if route is Route.NO_TRADE:
            return Decision.REJECT
        bar = (
            self.cfg.min_confidence_to_buy_extended
            if extended_permission
            else self.cfg.min_confidence_to_buy
        )
        if confidence < bar:
            return Decision.REJECT
        return Decision.BUY

    def _confidence(
        self, f: FeatureSet, route: Route, regime: Regime, blocking: list[RiskFlag],
        multiplier: float,
    ) -> float:
        """
        Decision strength in [0, 1].

        Deliberately NOT a probability of profit. It measures how much evidence
        supports the action relative to the alternatives right now.
        """
        if blocking:
            return 0.05
        if route is Route.NO_TRADE:
            return min(0.5, f.signal_quality)

        base = 0.35 * f.signal_quality + 0.20 * f.freshness + 0.15 * f.liquidity_score
        base += 0.15 * f.activity_score + 0.15 * f.momentum_score

        if regime in (Regime.VERY_EARLY, Regime.EARLY, Regime.ACCELERATING):
            base += 0.08
        if regime is Regime.UNCERTAIN:
            base -= 0.12
        if f.extension_from_first_valid_price is not None and \
                f.extension_from_first_valid_price > 0.6:
            base -= 0.10
        if f.drawdown_from_high_pct is not None and f.drawdown_from_high_pct < -30:
            base -= 0.15
        if f.observation_count < 3:
            base -= 0.08
        # REMOVED: Narrative availability penalty - narrative is optional telemetry
        # if not f.nar_available:
        #     base -= 0.04

        conf = base * (0.75 + 0.25 * multiplier)
        return max(0.0, min(1.0, conf))

    # -- narrative parts ---------------------------------------------------

    def _alert_risk_flags(self, alert: AlertSnapshot | None) -> list[RiskFlag]:
        """
        Risk flags derived from the ALERT's own reported figures.

        The configured bot publishes holder count, top-10 concentration, dev
        supply, bundle count, bundle/sold percentages and DEX-paid status. These
        are real, source-reported numbers, so they belong in the risk read - and
        they are the strongest supply-side evidence available, since no market
        data provider reports them.

        Each is read from the alert's own fields, never inferred. Anything absent
        is absent.
        """
        flags: list[RiskFlag] = []
        if alert is None:
            return flags

        def num(key: str) -> float | None:
            return alert.num(key)

        holders = num("holder_count")
        if holders is not None and holders < 50:
            flags.append(RiskFlag(
                "thin_holder_base",
                f"only {holders:.0f} holders",
                "block" if holders < 20 else "warn",
            ))

        top10 = num("top10_pct")
        if top10 is not None:
            if top10 >= 60:
                flags.append(RiskFlag("concentrated_holders",
                                      f"top 10 hold {top10:.1f}%", "block"))
            elif top10 >= 40:
                flags.append(RiskFlag("concentrated_holders",
                                      f"top 10 hold {top10:.1f}%", "warn"))

        dev = num("dev_holdings_pct")
        if dev is not None:
            if dev >= 5:
                flags.append(RiskFlag("dev_supply",
                                      f"dev holds {dev:.1f}% of supply", "block"))
            elif dev >= 1:
                flags.append(RiskFlag("dev_supply",
                                      f"dev holds {dev:.1f}% of supply", "warn"))

        bundles = num("bundle_count")
        bundled = num("bundled_pct")
        if bundles is not None and bundles >= 8:
            flags.append(RiskFlag("bundling",
                                  f"{bundles:.0f} bundle wallets", "warn"))
        if bundled is not None and bundled >= 50:
            flags.append(RiskFlag("bundled_supply",
                                  f"{bundled:.0f}% of supply bundled", "warn"))

        sold = num("sold_pct")
        if sold is not None and sold >= 50:
            flags.append(RiskFlag("dev_sold",
                                  f"{sold:.0f}% of dev supply already sold", "warn"))

        if alert.fields.get("dex_paid") is True:
            flags.append(RiskFlag("dex_paid",
                                  "DEX promotion paid - paid promotion is a "
                                  "crowding signal, not a quality one", "warn"))

        ath = num("ath_usd")
        price = num("price_usd")
        if ath and price and ath > 0:
            from_ath = (price - ath) / ath * 100.0
            if from_ath <= -60:
                flags.append(RiskFlag("deep_from_ath",
                                      f"{from_ath:.0f}% below the reported high",
                                      "warn"))
        return flags

    def _risk_flags(self, f: FeatureSet, nar: NarSnapshot | None, regime) -> list[RiskFlag]:
        flags: list[RiskFlag] = []

        if f.data_state is DataState.MISSING:
            flags.append(RiskFlag("no_market_data", "no observation at decision time", "block"))
        elif f.data_state is DataState.UNUSABLE:
            flags.append(
                RiskFlag("stale_market_data",
                         f"newest observation is {f.data_age_ms / 1000.0:.0f}s old", "block")
            )
        elif f.data_state is DataState.STALE:
            flags.append(
                RiskFlag("aging_market_data",
                         f"observation is {f.data_age_ms / 1000.0:.0f}s old", "warn")
            )

        if f.liquidity_usd is None:
            flags.append(RiskFlag("liquidity_unknown", "liquidity not reported by any source", "warn"))
        elif f.liquidity_usd < self.cfg.min_liquidity_usd:
            flags.append(
                RiskFlag("thin_liquidity",
                         f"${f.liquidity_usd:,.0f} < ${self.cfg.min_liquidity_usd:,.0f}", "block")
            )

        if f.buy_sell_ratio is not None:
            if f.buy_sell_ratio < 0.85:
                flags.append(
                    RiskFlag("sell_pressure", f"buy/sell {f.buy_sell_ratio:.2f} over 5m", "warn")
                )
            elif f.buy_sell_ratio > 6.0:
                flags.append(
                    RiskFlag("one_sided_buys",
                             f"buy/sell {f.buy_sell_ratio:.1f} is likely coordinated", "warn")
                )

        if f.buys_5m == 0 and f.sells_5m == 0 and f.observation_count > 0:
            flags.append(RiskFlag("no_transactions", "no trades in the last 5m window", "warn"))

        ext = f.extension_from_first_valid_price
        if ext is not None:
            # Two independent `if`s, deliberately not `if/elif`: an `elif` here
            # means the >1.0 case can never be reached, because >1.0 always also
            # satisfies >0.6. The severe case would silently degrade to a
            # warning and a badly chased entry would look merely notable.
            if ext > 0.6:
                flags.append(
                    RiskFlag("extended_entry",
                             f"{ext * 100:.0f}% above first valid price", "warn")
                )
            if ext > 1.0:
                flags.append(
                    RiskFlag("exhausted_entry",
                             f"{ext * 100:.0f}% above first valid price", "block")
                )

        if f.drawdown_from_high_pct is not None and f.drawdown_from_high_pct < -30:
            flags.append(
                RiskFlag("off_the_high",
                         f"{f.drawdown_from_high_pct:.0f}% from the observed high", "warn")
            )
        # OBSERVABILITY ONLY. Both were 'block' severity, which put them in the
        # `blocking` list and so REJECTed the candidate outright. Neither is part
        # of the researched L25 definition, so live was refusing candidates the
        # research never would have. They are retained as 'warn' so the operator
        # still sees the condition, and they can no longer change the decision,
        # the confidence, or entry eligibility.
        if f.change_5m is not None and f.change_5m < -10:
            flags.append(
                RiskFlag("reversing_5m", f"5m change {f.change_5m:.0f}%", "warn")
            )
        if f.momentum_30s_pct is not None and f.momentum_30s_pct < -10:
            flags.append(
                RiskFlag("reversing_30s",
                         f"{f.momentum_30s_pct:.0f}% in the last 30s", "warn")
            )
        if f.realized_vol_60s is not None and f.realized_vol_60s > 35:
            flags.append(
                RiskFlag("extreme_volatility",
                         f"{f.realized_vol_60s:.0f}% stdev over 60s", "warn")
            )
        if f.observation_count <= 1:
            flags.append(
                RiskFlag("thin_history", "one observation only; no market structure yet", "warn")
            )
        if f.market_cap_usd is None:
            flags.append(RiskFlag("mcap_unknown", "market cap not reported", "info"))
        if not f.extension_anchored:
            flags.append(
                RiskFlag(
                    "extension_unanchored",
                    "no earlier price to measure extension from; the chaseable "
                    "band cannot be applied, so the entry is sized down instead",
                    "warn",
                )
            )
        if f.liq_to_mcap is not None and f.liq_to_mcap < 0.02:
            flags.append(
                RiskFlag("liq_mcap_ratio", f"liquidity is {f.liq_to_mcap * 100:.1f}% of mcap", "warn")
            )

        # NAR-sourced flags are purely INFORMATIONAL - they never block
        # The narrative is optional telemetry; it never blocks a trade
        if nar and nar.available:
            if nar.bundler_count and nar.bundler_count >= 3:
                flags.append(
                    RiskFlag("bundlers", f"{nar.bundler_count} bundler wallets reported", "info")
                )
            if nar.sniper_count and nar.sniper_count >= 5:
                flags.append(
                    RiskFlag("snipers", f"{nar.sniper_count} sniper buys reported", "info")
                )
            if nar.dev_holdings_pct is not None and nar.dev_holdings_pct >= 5:
                flags.append(
                    RiskFlag("dev_supply",
                             f"dev holds {nar.dev_holdings_pct:.1f}% of supply", "block")
                )
            if nar.top10_holders_pct is not None and nar.top10_holders_pct >= 60:
                flags.append(
                    RiskFlag("concentrated_holders",
                             f"top10 hold {nar.top10_holders_pct:.0f}%", "block")
                )
            for f_ in nar.security_flags or ():
                flags.append(RiskFlag(f"nar_{f_}", f"NAR security flag: {f_}", "warn"))
        return flags

    def _thesis(self, f: FeatureSet, regime, regime_reason: str,
                route_reason: str, decision: Decision, nar: NarSnapshot | None = None,
                v3c: V3CResult = None, l25: L25Result = None,
                v2floor: V2FloorResult = None) -> str:
        parts: list[str] = []
        sym = "token"
        if f.price is not None:
            parts.append(f"price ${f.price:.3e}")
        parts.append(f"mcap ${f.market_cap_usd:,.0f}" if f.market_cap_usd else "mcap unknown")
        parts.append(f"liq ${f.liquidity_usd:,.0f}" if f.liquidity_usd else "liq unknown")
        parts.append(
            f"age {f.token_age_s / 60:.1f}m" if f.token_age_s else "age unknown"
        )
        parts.append(
            f"ext from first valid price {_fmt_pct(f.extension_from_first_valid_price)}"
        )
        if f.change_5m is not None:
            parts.append(f"5m {f.change_5m:+.0f}%")
        if f.buy_sell_ratio is not None:
            parts.append(f"buy/sell {f.buy_sell_ratio:.2f}")
        parts.append(f"data {f.data_state.value}")
        if nar is not None and nar.narrative:
            parts.append(f"narrative {len(nar.narrative)}c")
        else:
            parts.append("NAR evidence absent")
        parts.append(
            f"{self._selected_entry} "
            f"{(v2floor if self._selected_entry == 'V2_FLOOR' else l25).status.lower()}"
            + (
                f" (v3c {v3c.status.lower()})"
                if self._selected_entry != "V2_FLOOR" and not l25.authorized
                else ""
            )
        )
        head = f"{regime.value} ({regime_reason})"
        verdict = {
            Decision.BUY: "BUY",
            Decision.WAIT: "WAIT for a live print",
            Decision.REJECT: "NO TRADE",
        }[decision]
        return f"{verdict}: {head}; {route_reason}. " + ", ".join(parts)

    def _invalidation(self, f: FeatureSet, choice, regime) -> list[str]:
        style = exit_style_for(choice.route, regime, choice.style_override)
        plan = build_exit_plan(choice.route, regime, style, f, self.exit_cfg)
        return [str(x) for x in plan["invalidation_conditions"]]  # type: ignore[index]

    def _entry_conditions(self, f: FeatureSet, choice, plan) -> list[str]:
        # The pre-declared entry conditions name the SELECTED gate's own
        # thresholds. Under the L25 family this is byte-identical to before.
        # Under V2_FLOOR it states V2_FLOOR's conditions and explicitly records
        # that NO market-cap cap and NO liquidity floor apply -- so a stored
        # record can never imply a filter that was not enforced.
        if self._selected_entry == "V2_FLOOR":
            from ..v2floor import (
                V2F_MAX_CHANGE_5M_PCT,
                V2F_MIN_CHANGE_5M_PCT,
                V2F_MIN_VOL5_PER_MC,
                rule_summary,
            )
            out = [
                f"V2_FLOOR authorized: {rule_summary()}",
                f"decision-time 5m USD volume / market cap >= {V2F_MIN_VOL5_PER_MC:g}",
                f"decision-time 5m price change >= {V2F_MIN_CHANGE_5M_PCT:+g}%",
                f"decision-time 5m price change <= {V2F_MAX_CHANGE_5M_PCT:+g}%",
                "NO market-cap ceiling (V3-C's is L25-only; V2-FLOOR has none)",
                "NO liquidity/MC floor (V2-FLOOR has none)",
                f"pipeline route == {choice.route.value} (shared structural gate)",
                f"exit style {choice.exit_style.value}",
                f"data must remain {self.cfg.min_data_freshness:.2f}+ fresh at fill time",
                "fill must obtain a valid quote within the quote timeout; otherwise no trade",
            ]
        else:
            out = [
                f"{ACTIVE_STRATEGY} authorized: {V3C_NAME} + liquidity/MC",
                f"route == FAST_ENTRY",
                f"decision-time market cap <= ${V3C_MAX_MARKET_CAP_USD:,.0f}",
                f"decision-time signal quality >= {V3C_MIN_SIGNAL_QUALITY:.2f}",
                f"f_liq_to_mcap >= {self.cfg.l25_min_liq_to_mcap:.4f}",
                f"exit style {choice.exit_style.value}",
                f"data must remain {self.cfg.min_data_freshness:.2f}+ fresh at fill time",
                "fill must obtain a valid quote within the quote timeout; otherwise no trade",
            ]
        if f.liquidity_usd:
            out.append(f"liquidity >= ${self.cfg.min_liquidity_usd:,.0f} at fill")
        if f.buy_sell_ratio is not None:
            out.append(f"buy/sell {f.buy_sell_ratio:.2f} at analysis time")
        return out

    def _watch_conditions(self, f: FeatureSet, decision: Decision) -> list[str]:
        if decision is Decision.WAIT:
            return [
                "a LIVE price print (data state STALE -> LIVE)",
                f"signal quality >= {self.cfg.min_signal_quality_to_trade:.2f}",
            ]
        blockers = []
        if f.data_state is not DataState.LIVE:
            blockers.append("live market data")
        if f.liquidity_usd is None or f.liquidity_usd < self.cfg.min_liquidity_usd:
            blockers.append(f"liquidity >= ${self.cfg.min_liquidity_usd:,.0f}")
        if f.extension_from_first_valid_price is not None and \
                f.extension_from_first_valid_price > 0.6:
            blockers.append("extension back under control")
        if f.signal_quality < self.cfg.min_signal_quality_to_trade:
            blockers.append(
                f"signal quality >= {self.cfg.min_signal_quality_to_trade:.2f}"
            )
        return [f"would need: {', '.join(blockers)}"] if blockers else ["would need: new evidence"]


def _fmt_pct(value: float | None) -> str:
    return "unknown" if value is None else f"{value * 100:+.0f}%"


def _v3c_flag_code(v3c: V3CResult) -> str:
    """
    The risk-flag code for a V3-C refusal.

    Distinct codes per gate, so the journal says WHICH threshold failed rather
    than "V3-C said no", and so the Telegram reject alert is generated from the
    same fact the decision was made on.
    """
    if v3c.status == REJECTED_MARKET_CAP:
        return "v3c_market_cap_over_ceiling"
    if v3c.status == REJECTED_SIGNAL_QUALITY:
        return "v3c_signal_quality_below_floor"
    if v3c.status == UNTESTABLE_STATUS:
        return "v3c_untestable"
    return "v3c_not_authorized"


#: Risk-flag code prefixes that identify a STRATEGY-GATE verdict rather than a
#: structural refusal of the candidate. ``v2a_`` is included because V2-A is an
#: L25 sub-result (see the V2-A scoping in ``_risk_flags``).
GATE_FLAG_PREFIXES = ("v3c_", "v2a_", "l25_", "v2floor_")


def _split_blocking(blocking: list[RiskFlag]) -> tuple[list[RiskFlag],
                                                       list[RiskFlag]]:
    """
    Split blocking flags into (strategy-gate verdicts, structural refusals).

    The distinction is the whole of per-strategy authorization. A gate verdict
    belongs to one selected strategy and must not refuse a candidate another
    selected strategy authorized; a structural refusal belongs to the candidate
    and refuses it for everyone.

    ``v3c_not_authorized`` is the one ``v3c_`` code that is structural rather
    than gate-scoped: it means the route classifier refused the candidate
    outright ("not a FAST_ENTRY setup"), which is a statement about the
    candidate and not a verdict L25 reached on a tradeable setup.
    """
    gate_flags: list[RiskFlag] = []
    structural: list[RiskFlag] = []
    for flag in blocking:
        if flag.code == "v3c_not_authorized" or not flag.code.startswith(
                GATE_FLAG_PREFIXES):
            structural.append(flag)
        else:
            gate_flags.append(flag)
    return gate_flags, structural


def _strategy_flag_code(l25: L25Result) -> str:
    """
    The risk-flag code for the ACTIVE strategy's refusal.

    When L25's V3-C leg refused, the delegated V3-C code is used unchanged, so
    a market-cap or signal-quality refusal keeps its established meaning. When
    V3-C passed and only the liquidity ratio refused, an L25-prefixed code is
    used, so the journal distinguishes "V3-C said no" from "L25's liquidity
    gate said no".
    """
    if l25.v3c is not None and not l25.v3c.authorized:
        return _v3c_flag_code(l25.v3c)
    if l25.status == REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD:
        return "l25_liq_to_mcap_below_threshold"
    if l25.status == L25_UNTESTABLE:
        return "l25_untestable"
    return "l25_not_authorized"
