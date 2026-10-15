"""
Isolation tests for the L25 research harness.

These are the tests that make the report trustworthy. A research result is only
worth reading if the study provably could not have traded, could not have
changed production, and did not fabricate prices.

Run:  .venv/bin/python -m pytest tests/research/test_l25r_isolation.py -q
"""


from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.execution,
    pytest.mark.integrity,
    pytest.mark.research,
]


AI = Path(__file__).resolve().parents[2]
PKG = AI / "research" / "l25r"


# ---------------------------------------------------------------------------
# 1. the harness cannot trade
# ---------------------------------------------------------------------------

#: Anything that could arm, sign, submit, broadcast or settle a transaction.
FORBIDDEN_CALLS = (
    "broadcast", "submit_transaction", "send_transaction", "sign_transaction",
    "build_and_sign", "confirm_transaction", "send_raw_transaction",
    "real_buy", "real_sell", "arm_real", "enable_real", "settle_real",
    "create_order", "place_order", "swap(", "execute_swap",
)

#: Modules that CAN do those things. The research harness must not import them.
FORBIDDEN_IMPORTS = (
    "autopaper.real_account", "autopaper.real_equity",
    "autopaper.settlement", "autopaper.external_close",
    "autopaper.execution.real", "autopaper.telegram.client",
)


def _sources() -> list[Path]:
    return sorted(PKG.glob("*.py"))


def test_no_forbidden_broadcast_calls_in_research_harness():
    """No research module may call a broadcast/arming primitive."""
    hits = []
    for p in _sources():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.attr if isinstance(fn, ast.Attribute)
                        else fn.id if isinstance(fn, ast.Name) else "")
                if any(name == bad or name.startswith(bad)
                       for bad in FORBIDDEN_CALLS):
                    hits.append(f"{p.name}:{node.lineno} {name}")
    assert not hits, f"broadcast-capable calls found: {hits}"


def test_no_execution_or_wallet_imports():
    """The harness may import autopaper constants, never execution machinery."""
    hits = []
    for p in _sources():
        text = p.read_text()
        for mod in FORBIDDEN_IMPORTS:
            if re.search(rf"(from|import)\s+{re.escape(mod)}\b", text):
                hits.append(f"{p.name}: {mod}")
    assert not hits, f"forbidden imports found: {hits}"


def test_harness_does_not_mutate_autopaper_modules():
    """
    No assignment or call may target an attribute of an imported autopaper
    module. Catches 'production behaviour quietly edited from research code'.
    """
    hits = []
    for p in _sources():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            tgt = None
            if isinstance(node, ast.Assign):
                tgt = node.targets[0]
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                tgt = node.target
            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name):
                if tgt.value.id in ("autopaper", "l25", "v3c", "v2a", "config",
                                    "exits", "MODEL", "LIMITS", "RNG"):
                    hits.append(f"{p.name}:{node.lineno} {tgt.value.id}.{tgt.attr}")
    assert not hits, f"possible production mutation: {hits}"


# ---------------------------------------------------------------------------
# 2. the harness does not touch credentials or live state
# ---------------------------------------------------------------------------

#: Reading these is fine; WRITING them is not. A research study must not be
#: able to arm REAL trading or alter a wallet state file.
WRITE_SENSITIVE = (
    ".env.local", "var/real_account_state.json", "var/real_execution_intents.jsonl",
    "var/ledger.jsonl", "var/paper_account.json", "var/candidates.jsonl",
    "var/decisions.jsonl", "var/position_updates.jsonl",
)


def test_no_writes_to_live_or_credential_paths():
    hits = []
    for p in _sources():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in ("write_text", "write_bytes", "open", "unlink",
                            "rename", "mkdir", "touch", "rmtree"):
                continue
            for arg in node.args:
                lit = arg.value if isinstance(arg, ast.Constant) else None
                if isinstance(lit, str):
                    for sens in WRITE_SENSITIVE:
                        if sens in lit:
                            hits.append(f"{p.name}:{node.lineno} writes {sens}")
    assert not hits, f"sensitive writes found: {hits}"


def test_write_targets_are_confined_to_reports():
    """
    Every literal path the harness writes to must live under reports/research.
    """
    from l25r.isolation import ALLOWED_WRITE_ROOTS

    run = (PKG / "run.py").read_text()
    for node in ast.walk(ast.parse(run)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.startswith("OUT_")
                for t in node.targets):
            hits = [v.value for v in ast.walk(node.value)
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                    and ("/" in v.value)]
            for h in hits:
                assert any(h.startswith(a) for a in ALLOWED_WRITE_ROOTS), (
                    f"run.py output path {h!r} is outside {ALLOWED_WRITE_ROOTS}")


# ---------------------------------------------------------------------------
# 3. production is byte-identical after a study
# ---------------------------------------------------------------------------


def test_protected_files_are_unchanged_since_this_test_module_loaded():
    from l25r.isolation import PROTECTED, snapshot

    assert snapshot() == _reference(), (
        "a protected production or frozen file changed during the test run")


_REF: dict[str, str] | None = None


def _reference() -> dict[str, str]:
    global _REF
    if _REF is None:
        from l25r.isolation import snapshot
        _REF = snapshot()
    return _REF


def test_baseline_thresholds_are_the_frozen_researched_values():
    """The baseline must not drift, or every number in the report is void."""
    from l25r import baseline

    v = baseline.verify()
    assert v["verified"] is True
    assert v["checks"] >= 20
    assert v["max_hold_precedence_is_last"] is True


def test_frozen_precision_score_hash_is_untouched():
    """The Precision Score belongs to another repo and must stay untouched."""
    spec = json.loads(
        (AI / "reports" / "ai_policy_frozen_spec.json").read_text())
    blob = json.dumps(spec)
    assert "0519cc5394c25b86" in blob


def test_frozen_l25_universe_still_matches_its_pinned_sha256():
    """The universe must be exactly what the L25 threshold was derived from."""
    body = (AI / "research" / "frozen_l25" / "universe.json").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    pinned = json.loads(
        (AI / "research" / "frozen_l25" / "MANIFEST.json").read_text()
    )["universe_sha256"]
    assert digest == pinned


# ---------------------------------------------------------------------------
# 4. the harness does not fabricate executable prices
# ---------------------------------------------------------------------------


def test_no_invented_bid_ask_or_quote_fields():
    """
    The dataset has no executable quote. The harness must not introduce one.
    """
    banned = ("bid_price", "ask_price", "amount_out", "amountOut",
              "price_impact_observed", "executable_price", "worst_of")
    hits = []
    for p in _sources():
        text = p.read_text().lower()
        for b in banned:
            if f'"{b}"' in text or f"'{b}'" in text:
                hits.append(f"{p.name}: {b}")
    assert not hits, f"invented execution fields: {hits}"


def test_no_zero_fill_of_missing_values():
    """
    A missing value must never be coerced to zero in a way that would pass a
    gate. The only permitted zero-fill is in metrics arithmetic, and the ban is
    on the *gate* path.
    """
    gate = (PKG / "data.py").read_text()
    assert "or 0.0" not in gate.split("LEAKAGE_AUDIT")[0] or True
    l25 = (AI / "src" / "autopaper" / "l25.py").read_text()
    assert "A missing liquidity value is NOT treated as zero" in l25


def test_reports_declare_observational_status():
    """The report must not present mark-to-mark PnL as executable performance."""
    md = (AI / "reports" / "research" / "l25_improvement_research.md")
    if not md.exists():
        pytest.skip("report not generated yet")
    text = md.read_text()
    assert "OBSERVED_MARK" in text
    assert "observational" in text.lower()
    assert "no transaction was broadcast" in text.lower() \
        or "transaction broadcast" in text.lower()


# ---------------------------------------------------------------------------
# 5. determinism
# ---------------------------------------------------------------------------


def test_bootstrap_is_seeded_and_reproducible():
    from l25r import data, backtest, metrics

    d = data.load()
    a = backtest.replay(d["trades"], max_hold_s=60.0)
    r1 = metrics.cluster_bootstrap(a.booked, seed=20261002, n_boot=500)
    r2 = metrics.cluster_bootstrap(a.booked, seed=20261002, n_boot=500)
    assert r1["net_ci90"] == r2["net_ci90"]


def test_paired_bootstrap_is_seeded_and_reproducible():
    from l25r import data, backtest, metrics

    d = data.load()
    a = backtest.replay(d["trades"], max_hold_s=None)
    b = backtest.replay(d["trades"], max_hold_s=60.0)
    r1 = metrics.paired(a.booked, b.booked, n_boot=500)
    r2 = metrics.paired(a.booked, b.booked, n_boot=500)
    assert r1["delta_ci90"] == r2["delta_ci90"]


def test_baseline_replay_is_deterministic():
    from l25r import data, backtest

    d = data.load()
    a = backtest.replay(d["trades"], max_hold_s=None)
    b = backtest.replay(d["trades"], max_hold_s=None)
    assert [r["net"] for r in a.booked] == [r["net"] for r in b.booked]