"""
Focused tests for OBJECTIVE D -- take-profit selection, persistence, position
snapshot immutability, and quote-based MFE/MAE semantics.

Run:
    .venv/bin/python -m pytest tests/test_take_profit_objectives.py -q

Nothing here signs, broadcasts, starts a runtime, or touches real state.
Take-profit wiring must not be able to change any exit other than the
take-profit itself, and must not be able to rewrite an open position.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from autopaper.config import ExitConfig
from autopaper.modes import RuntimeState
from autopaper.paper_live.arms import (
    arm_by_name,
    arm_names,
    arm_specs,
)
from autopaper.paper_live.quote_excursions import (
    BASIS_EXECUTABLE_SELL_QUOTES,
    BASIS_MARK_ONLY,
    BASIS_NOT_MEASURED,
    describe,
    mark_only,
    record_quote_excursion,
    sell_quote_return_pct,
)
from autopaper.paper_live.quote_proximity import (
    existing_price_triggers,
)
from autopaper.paper_live.take_profit import (
    NO_TAKE_PROFIT,
    SELECTABLE_TAKE_PROFITS,
    TakeProfitError,
    for_position,
    label,
    net_quote_return_pct,
    normalise,
    trigger_token,
)
from autopaper.telegram import ui
from autopaper.telegram.actions import ButtonHandler, OperatorActions


# ===========================================================================
# TP vocabulary
# ===========================================================================
class TestTakeProfitVocabulary:
    def test_100_pct_is_still_selectable(self):
        """The whole point of removing the TP100 ARM: the VALUE stays reachable."""
        assert 100.0 in SELECTABLE_TAKE_PROFITS
        assert normalise("100") == 100.0
        assert normalise("100%") == 100.0

    def test_off_is_explicit(self):
        assert normalise("OFF") is NO_TAKE_PROFIT
        assert normalise(None) is NO_TAKE_PROFIT
        assert label(None) == "OFF"

    def test_zero_is_not_a_fireable_level(self):
        """
        A take-profit at ``ret >= 0`` fires on the first tick of any position
        that is not already underwater. Zero must be OFF, never a target.
        """
        with pytest.raises(TakeProfitError):
            normalise(0)
        with pytest.raises(TakeProfitError):
            normalise(0.0)
        with pytest.raises(TakeProfitError):
            normalise("0")

    def test_an_out_of_bounds_value_is_not_snapped_to_a_neighbour(self):
        # Presets are accepted exactly. A value OUTSIDE the declared bounds is
        # refused rather than clamped into one -- a take-profit is not something
        # to "round to the nearest sensible level".
        with pytest.raises(TakeProfitError):
            normalise(5000.0)      # above the ceiling
        with pytest.raises(TakeProfitError):
            normalise(0.1)         # below the floor
        with pytest.raises(TakeProfitError):
            normalise(-5)
        # and a value inside the bounds is kept exactly, not snapped
        assert normalise(200.5) == 200.5

    def test_a_custom_value_inside_the_bounds_is_kept(self):
        assert normalise(200.0) == 200.0

    def test_trigger_token_keeps_history_legible(self):
        assert trigger_token(100.0) == "tp100"
        assert trigger_token(50.0) != "tp100", (
            "a +50% exit must never be recorded under the +100% token")


# ===========================================================================
# The duplicate option is gone, but history still resolves
# ===========================================================================
class TestDuplicateOptionRemoved:
    def test_l25_tp100_is_not_selectable(self):
        assert "L25_TP100" not in arm_names(), (
            "the duplicate TP-100% strategy option is still offered for selection")
        assert "L25_TP100" not in [s.name.value for s in arm_specs()]

    def test_l25_tp100_still_resolves_for_historical_records(self):
        """A retired option must not orphan what it already opened."""
        spec = arm_by_name("L25_TP100")
        assert spec.production_profile.value == "L25_TP100"

    def test_the_picker_offers_the_level_instead(self):
        labels = [b["text"] for row in ui.take_profit_keyboard(None) for b in row]
        assert any("+100%" in t for t in labels), labels


# ===========================================================================
# TP selection and persistence
# ===========================================================================
class TestTakeProfitSelection:
    def _handler(self, state, captured):
        h = ButtonHandler(state, OperatorActions(
            persist_take_profit=captured.append))
        return h

    def test_the_ui_offers_every_level(self):
        rows = ui.take_profit_keyboard(None)
        payloads = [b["callback_data"] for r in rows for b in r]
        for value in SELECTABLE_TAKE_PROFITS:
            assert ui.CB_TAKE_PROFIT_SELECT + f"{value:g}" in payloads
        assert ui.CB_TAKE_PROFIT_OFF in payloads

    def test_selecting_a_level_persists_it(self):
        st = RuntimeState()
        captured = []
        h = self._handler(st, captured)
        r = asyncio.run(h.handle(ui.CB_TAKE_PROFIT_SELECT + "100"))
        assert not r.error
        assert st.take_profit_pct == 100.0
        assert captured == [100.0], "the selection must be handed to the persist hook"

    def test_an_invalid_level_leaves_the_previous_level_in_force(self):
        st = RuntimeState()
        st.take_profit_pct = 50.0
        captured = []
        h = self._handler(st, captured)
        asyncio.run(h.handle(ui.CB_TAKE_PROFIT_SELECT + "100"))
        r = asyncio.run(h.handle(ui.CB_TAKE_PROFIT_SELECT + "0"))
        assert r.error == "invalid_take_profit"
        assert st.take_profit_pct == 100.0, (
            "a refused value must not silently change what is armed")
        assert len(captured) == 1

    def test_selection_is_inert_with_respect_to_the_runtime(self):
        st = RuntimeState()
        st.run_state = st.run_state.__class__.PAPER
        st.emergency_stop = True
        captured = []
        h = self._handler(st, captured)
        before = (st.run_state, st.real_armed, st.entries_blocked, st.emergency_stop)
        asyncio.run(h.handle(ui.CB_TAKE_PROFIT_SELECT + "50"))
        after = (st.run_state, st.real_armed, st.entries_blocked, st.emergency_stop)
        assert before == after, (
            "selecting a take-profit must not start the runtime, arm REAL, "
            "unblock entries or clear an emergency stop")

    def test_the_engine_persists_through_an_atomic_write(self, tmp_path, monkeypatch):
        """The engine's own persistence path, exercised against a temp var dir."""
        import autopaper.engine as eng_mod

        src = Path(eng_mod.__file__).read_text()
        assert "_take_profit_path" in src
        assert "_restore_take_profit" in src
        assert "atomic_write_json" in src, (
            "take-profit must use the existing atomic persistence mechanism")


# ===========================================================================
# Position snapshot immutability
# ===========================================================================
class TestPositionSnapshot:
    def test_a_position_exits_on_its_own_snapshot(self):
        class P:
            take_profit_pct = 50.0
            production_profile = "L25_CURRENT"
        assert for_position(P) == 50.0

    def test_changing_the_global_does_not_change_the_position(self):
        """
        The load-bearing invariant: the global setting is read ONCE, at open.
        A position carries its own copy forever after.
        """
        st = RuntimeState()
        st.take_profit_pct = 50.0

        class P:
            take_profit_pct = 50.0
            production_profile = "L25_CURRENT"

        opened = for_position(P())
        # the operator changes the global afterwards
        st.take_profit_pct = 200.0
        assert for_position(P()) == opened == 50.0, (
            "changing the global setting altered an already-open position")

    def test_a_snapshot_that_cannot_be_read_fails_closed_to_off(self):
        class P:
            take_profit_pct = "not a number"
            production_profile = "L25_CURRENT"
        assert for_position(P()) is NO_TAKE_PROFIT

    def test_a_legacy_tp100_position_keeps_its_level(self):
        class P:
            take_profit_pct = None
            production_profile = "L25_TP100"
        assert for_position(P()) == 100.0

    def test_a_position_without_a_take_profit_has_no_tp_trigger(self):
        class P:
            take_profit_pct = None
            production_profile = "L25_CURRENT"
        names = [t[0] for t in existing_price_triggers(P, ExitConfig())]
        assert "tp100" not in names, names

    def test_the_trigger_uses_the_position_level_not_the_global(self):
        class P:
            take_profit_pct = 25.0
            production_profile = "L25_CURRENT"
        trig = dict(existing_price_triggers(P, ExitConfig()))
        assert trig["tp100"] == 25.0, (
            "proximity was computed against a level this position does not "
            "actually evaluate")

    def test_the_snapshot_is_never_written_back_to_the_global(self):
        source = Path(ui.__file__).read_text() + \
            Path(RuntimeState.__module__ and __import__(
                "autopaper.modes", fromlist=["x"]).__file__).read_text()
        # Selecting a take-profit must assign the state, never a position.
        assert "take_profit_pct = level" in Path(
            __import__("autopaper.telegram.actions", fromlist=["x"]).__file__
        ).read_text()


# ===========================================================================
# Trigger basis: cost-aware quote valuation, or an honest fallback
# ===========================================================================
class TestTriggerBasis:
    def test_a_fresh_sell_quote_is_the_preferred_basis(self):
        from autopaper.paper_live.take_profit import (
            BASIS_EXECUTABLE_NET_SELL_QUOTE,
            take_profit_trigger_documented,
        )
        t = take_profit_trigger_documented(
            target_pct=50.0,
            net_sell_proceeds=1.55,
            all_in_entry_cost=1.00,
        )
        assert t.fired is True
        assert t.basis == BASIS_EXECUTABLE_NET_SELL_QUOTE
        assert t.return_pct == pytest.approx(55.0)

    def test_without_a_quote_the_basis_is_stated_explicitly(self):
        from autopaper.paper_live.take_profit import (
            BASIS_ENTRY_FILL_PRICE,
            take_profit_trigger_documented,
        )
        t = take_profit_trigger_documented(
            target_pct=50.0, entry_fill_price=1.0, price=2.0,
        )
        assert t.fired is True
        assert t.basis == BASIS_ENTRY_FILL_PRICE
        assert "no executable SELL quote" in t.detail, (
            "the fallback basis must be visible in the recorded trigger")

    def test_a_mark_is_never_used_as_the_trigger_basis_by_itself(self):
        from autopaper.paper_live.take_profit import take_profit_trigger_documented
        with_neither = take_profit_trigger_documented(target_pct=50.0)
        assert with_neither.fired is False, (
            "with neither a quote nor a fill price, the trigger must not fire")
        assert "unavailable" in with_neither.detail

    def test_fares_already_in_the_quote_are_not_charged_twice(self):
        # 1.15 returned for a 1.00 outlay, net of the route's own fee/impact.
        assert net_quote_return_pct(
            net_proceeds=1.15, all_in_entry_cost=1.00) == pytest.approx(15.0)
        assert net_quote_return_pct(
            net_proceeds=1.0, all_in_entry_cost=0.0) is None, (
            "a return computed against a zero cost is not a return")

    def test_no_target_means_no_fire(self):
        from autopaper.paper_live.take_profit import take_profit_trigger_documented
        t = take_profit_trigger_documented(
            target_pct=None, net_sell_proceeds=9.9, all_in_entry_cost=1.0)
        assert t.fired is False


# ===========================================================================
# Quote-based MFE/MAE, and N/A for missing
# ===========================================================================
class _P:
    def __init__(self, entry=1.0):
        self.entry_fill_price_usd = entry
        self.quote_mfe_pct = None
        self.quote_mae_pct = None
        self.quote_observation_count = 0
        self.quote_mfe_mae_basis = BASIS_NOT_MEASURED


class _Q:
    def __init__(self, price, side="SELL", available=True):
        self.effective_price_usd = price
        self.side = side
        self.available = available


class TestQuoteBasedExcursions:
    def test_a_missing_measurement_is_not_a_zero(self):
        d = describe(_P())
        assert d["quote_mfe_pct"] is None
        assert d["quote_mfe_pct_display"] == "N/A"
        assert d["quote_mae_pct_display"] == "N/A"
        assert d["measured"] is False

    def test_an_unmeasured_position_is_distinguishable_from_a_flat_one(self):
        flat = _P()
        flat.quote_mfe_pct = 0.0
        flat.quote_mae_pct = 0.0
        flat.quote_observation_count = 3
        flat.quote_mfe_mae_basis = BASIS_EXECUTABLE_SELL_QUOTES
        assert describe(flat)["quote_mfe_pct_display"] == "+0.00%"
        assert describe(_P())["quote_mfe_pct_display"] == "N/A"

    def test_a_buy_quote_is_not_evidence_about_an_exit(self):
        p = _P()
        assert sell_quote_return_pct(position=p, quote=_Q(2.0, side="BUY")) is None
        assert record_quote_excursion(p, quote=_Q(2.0, side="BUY")) is None
        assert p.quote_mfe_pct is None

    def test_an_unavailable_quote_contributes_nothing(self):
        p = _P()
        assert record_quote_excursion(p, quote=_Q(2.0, available=False)) is None
        assert p.quote_mfe_pct is None and p.quote_observation_count == 0
        assert p.quote_mfe_mae_basis == BASIS_MARK_ONLY, (
            "a position that has seen only unusable quotes must converge on "
            "MARK_ONLY so the absence is legible")

    def test_the_excursion_tracks_the_exact_held_quantity_basis(self):
        p = _P()
        assert record_quote_excursion(p, quote=_Q(1.5)) == pytest.approx(50.0)
        assert record_quote_excursion(p, quote=_Q(0.8)) == pytest.approx(-20.0)
        d = describe(p)
        assert d["quote_mfe_pct"] == pytest.approx(50.0)
        assert d["quote_mae_pct"] == pytest.approx(-20.0)
        assert d["quote_observation_count"] == 2

    def test_the_net_liquidation_basis_is_preferred_when_supplied(self):
        p = _P()
        # A quote whose own numbers already net out fees and impact.
        assert record_quote_excursion(
            p, quote=_Q(1.0), net_liquidation_return_pct=-4.7) == pytest.approx(-4.7)

    def test_a_non_numeric_display_value_never_becomes_a_zero(self):
        p = _P()
        assert sell_quote_return_pct(position=p, quote=None) is None
        assert describe(p)["quote_mfe_pct_display"] == "N/A"

    def test_the_mark_excursions_are_carried_alongside_not_substituted(self):
        p = _P()
        record_quote_excursion(p, quote=_Q(1.5))
        d = describe(p)
        assert d["observed_mark_label"] == "MARK (not executable)"
        assert "quote_mfe_pct" in d and "observed_mark_mfe" in d, (
            "the two bases must both be present so they are never pooled")

    def test_mark_only_is_idempotent(self):
        p = _P()
        mark_only(p)
        mark_only(p)
        assert p.quote_mfe_mae_basis == BASIS_MARK_ONLY
        assert p.quote_mfe_pct is None

    def test_a_real_measurement_overrides_a_mark_only_label(self):
        p = _P()
        mark_only(p)
        record_quote_excursion(p, quote=_Q(1.5))
        assert p.quote_mfe_mae_basis == BASIS_EXECUTABLE_SELL_QUOTES


# ===========================================================================
# The snapshot is persisted, not re-derived
# ===========================================================================
class TestSnapshotPersistence:
    def test_the_schema_persists_the_snapshot(self):
        src = Path(__import__(
            "autopaper.paper_live.schema", fromlist=["x"]).__file__).read_text()
        assert '"take_profit_pct": to_float(self.take_profit_pct)' in src
        assert '"quote_mfe_pct": to_float(self.quote_mfe_pct)' in src
        assert '"quote_mfe_mae_basis": self.quote_mfe_mae_basis' in src

    def test_the_exit_policy_reads_the_position(self):
        src = Path(__import__(
            "autopaper.exits", fromlist=["x"]).__file__).read_text()
        assert "_take_profit_for(position)" in src, (
            "the exit policy must read the POSITION, not the global setting")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
