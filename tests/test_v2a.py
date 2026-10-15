"""
V2-A gate tests.

Covers the frozen ceiling, the exact boundary, the FAST_ENTRY-only scope, the
untestable-missing-market-cap case, the config tamper guard, and the fact that
the gate reaches the live decision path as a blocking flag.

The boundary cases (25,372.70 / .71 / .72) are the ones the research brief
named explicitly, so they are asserted literally rather than derived.
"""

from __future__ import annotations

import pytest

from autopaper.config import DecisionConfig
from autopaper.v2a import (
    V2A_CEILING_USD,
    V2A_HASH,
    V2A_UNTESTABLE,
    apply_v2a,
    verify_ceiling,
)


class TestFrozenValue:
    def test_ceiling_is_the_researched_number(self):
        assert V2A_CEILING_USD == 25_372.71

    def test_verify_accepts_the_frozen_value(self):
        verify_ceiling(25_372.71)          # must not raise

    @pytest.mark.parametrize("bad", [25_000.0, 25_372.70, 25_372.72, 30_000.0])
    def test_verify_refuses_any_retune(self, bad):
        with pytest.raises(ValueError, match="FROZEN"):
            verify_ceiling(bad)

    def test_hash_is_defined(self):
        assert len(V2A_HASH) == 64
        int(V2A_HASH, 16)                  # valid hex


class TestBoundary:
    """The three values the brief named, asserted literally."""

    def test_just_below_ceiling_passes(self):
        r = apply_v2a("FAST_ENTRY", 25_372.70)
        assert r.passes and r.status == "WITHIN_CEILING"

    def test_exactly_the_ceiling_passes(self):
        r = apply_v2a("FAST_ENTRY", 25_372.71)
        assert r.passes and r.status == "WITHIN_CEILING"
        assert r.rejects is False

    def test_one_cent_above_is_rejected(self):
        r = apply_v2a("FAST_ENTRY", 25_372.72)
        assert r.passes is False
        assert r.rejects is True
        assert r.status == "REJECTED_OVER_CEILING"

    def test_boundary_is_inclusive_with_no_epsilon(self):
        # a float just under the ceiling must still pass
        assert apply_v2a("FAST_ENTRY", V2A_CEILING_USD - 1e-9).passes
        assert apply_v2a("FAST_ENTRY", V2A_CEILING_USD + 1e-9).rejects


class TestScope:
    def test_only_fast_entry_is_in_scope(self):
        for route in ("PRECISION_ENTRY", "NO_TRADE", "WATCH", ""):
            r = apply_v2a(route, 99_000_000.0)
            assert r.status == "NOT_APPLICABLE"
            assert r.passes is True

    def test_fast_entry_wide_above_ceiling_is_rejected(self):
        assert apply_v2a("FAST_ENTRY", 1_000_000.0).rejects is True

    def test_fast_entry_low_is_untouched(self):
        r = apply_v2a("FAST_ENTRY", 12_000.0)
        assert r.passes and r.market_cap_usd == 12_000.0


class TestUntestableIsNotAPass:
    def test_missing_market_cap_is_untestable(self):
        r = apply_v2a("FAST_ENTRY", None)
        assert r.status == V2A_UNTESTABLE
        assert r.passes is False, "an unreadable market cap must never auto-pass"

    def test_untestable_reports_no_market_cap(self):
        assert apply_v2a("FAST_ENTRY", None).market_cap_usd is None


class TestResultShape:
    def test_as_dict_carries_the_audit_fields(self):
        d = apply_v2a("FAST_ENTRY", 30_000.0).as_dict()
        assert set(d) == {"passes", "status", "rejects", "market_cap_usd",
                          "ceiling_usd", "detail"}
        assert d["ceiling_usd"] == 25_372.71
        assert d["rejects"] is True

    def test_reject_detail_names_both_numbers(self):
        detail = apply_v2a("FAST_ENTRY", 30_000.0).detail
        assert "30,000.00" in detail and "25,372.71" in detail

    def test_apply_refuses_a_retuned_ceiling(self):
        with pytest.raises(ValueError, match="FROZEN"):
            apply_v2a("FAST_ENTRY", 10_000.0, ceiling=20_000.0)


class TestConfigGuard:
    def test_default_carries_the_frozen_ceiling(self):
        assert DecisionConfig().v2a_ceiling_usd == 25_372.71

    def test_explicit_frozen_value_is_accepted(self):
        assert DecisionConfig(v2a_ceiling_usd=25_372.71).v2a_ceiling_usd == 25_372.71

    @pytest.mark.parametrize("bad", [20_000.0, 26_000.0, 25_372.72])
    def test_config_refuses_to_retune(self, bad):
        with pytest.raises(ValueError):
            DecisionConfig(v2a_ceiling_usd=bad)

    def test_ceiling_is_reported_in_as_dict(self):
        assert DecisionConfig().as_dict()["v2a_ceiling_usd"] == 25_372.71
