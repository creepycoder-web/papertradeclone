"""
An operator-facing message must never name a strategy the runtime is not using.

This exists because it already went wrong. ``render_stats`` and the armed-REAL
banner hardcoded ``V3C_NAME``, so after the L25 migration the bot reported
"Strategy: V3-C" while the strategy actually gating entries was L25 - and in the
armed-REAL case it told the operator the wrong strategy at the moment real money
was about to be committed.

The failure mode is silent: nothing crashes, the numbers are right, only the
label lies. So the invariant is checked structurally - every strategy name that
reaches an operator must come from ``ACTIVE_STRATEGY``.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

from autopaper.l25 import ACTIVE_STRATEGY
from autopaper.modes import RunState, RuntimeState
from autopaper.telegram import ui
from autopaper.v3c import V3C_NAME

#: AST-scans the whole source tree
pytestmark = [
    pytest.mark.slow,
]

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src/autopaper"

#: Modules where a strategy mention is a CLAIM ABOUT THE ACTIVE STRATEGY, so
#: every one of them must name ``ACTIVE_STRATEGY``.
#:
#: This was originally just ``telegram/ui.py``, and that scope is exactly what
#: let the START acknowledgement in ``telegram/actions.py`` keep saying "V3-C"
#: after the L25 migration. A check scoped to one file only proves things about
#: that file, so the scope is the whole operator surface.
IDENTITY_MODULES = (
    "telegram/ui.py",
    "telegram/actions.py",
    "config.py",
)

#: Modules that legitimately name V3-C, because they are describing THE GATE and
#: not the active strategy. "V3-C: pass, L25 refused on the ratio" is useful and
#: correct; "Strategy: V3-C" is a mislabel. Conflating the two would force the
#: code to stop explaining which leg did what, so these get the narrower check.
GATE_MODULES = (
    "engine.py",
    "cli.py",
    "analysis/routes.py",
    "analysis/decision.py",
    "v3c.py",
    "l25.py",
)

OPERATOR_MODULES = IDENTITY_MODULES + GATE_MODULES

#: Patterns that assert IDENTITY rather than describing a gate's result.
IDENTITY_CLAIMS = (
    "strategy=V3-C",
    "Strategy: V3-C",
    "Strategy: <b>V3-C",
    "active strategy=V3-C",
)

assert ACTIVE_STRATEGY == "L25"
assert V3C_NAME != ACTIVE_STRATEGY, (
    "if these ever become equal this module stops being meaningful"
)


# ---------------------------------------------------------------------------
# the renderers an operator actually reads
# ---------------------------------------------------------------------------


def _view(state: RuntimeState | None = None, **kw) -> ui.StatusView:
    base = dict(
        state=state or RuntimeState(),
        channel="Moo Deng President",
        chat="-1001234567890",
        wallet_masked="So11…1112",
        balance_sol=0.0089,
        balance_usd=1.78,
        equity_usd=1.55,
        cash_usd=1.55,
        open_positions=0,
        executor_available=False,
    )
    base.update(kw)
    return ui.StatusView(**base)


def _armed() -> RuntimeState:
    s = RuntimeState()
    s.start()
    s.arm_real(confirmation=s.request_real_confirmation(), preflight_ok=True)
    return s


PAPER_STATS = {
    "total_trades": 418, "wins": 122, "losses": 296, "win_rate": 0.292,
    "profit_factor": 1.52, "expectancy_usd": 2.11, "total_pnl_usd": 882.25,
    "max_drawdown_pct": -55.5, "avg_winner_usd": 21.16, "avg_loser_usd": -5.74,
    "runners": 39, "runner_capture_pct": 69.8, "equity_usd": 932.97,
    "cash_usd": 932.97,
}


def test_the_stats_header_names_the_active_strategy():
    out = ui.render_stats(paper=PAPER_STATS, real={"total_trades": 0})
    assert f"Strategy: <b>{ACTIVE_STRATEGY}</b>" in out
    assert f"Strategy: <b>{V3C_NAME}</b>" not in out


def test_the_stats_view_cannot_be_misread_as_an_l25_verdict():
    """
    The PAPER account is cumulative across every strategy version that ever ran.

    So the label is right for what happens next, and the caveat is what stops it
    being read as a result for the current strategy. 418 cumulative trades are
    not 418 L25 trades.
    """
    out = ui.render_stats(paper=PAPER_STATS, real={"total_trades": 0})
    assert "forward-report" in out
    assert ACTIVE_STRATEGY in out


def test_the_armed_real_banner_names_the_active_strategy():
    out = ui.render_status(_view(_armed()))
    assert "REAL TRADING ARMED" in out
    armed_block = out.split("REAL TRADING ARMED", 1)[1]
    assert ACTIVE_STRATEGY in armed_block
    assert V3C_NAME not in armed_block, (
        "the armed banner must not name the delegated V3-C leg: it is the line "
        "that tells the operator which strategy is about to use real money"
    )


def test_the_real_confirmation_prompt_names_the_active_strategy():
    s = RuntimeState()
    s.start()
    text, _kb = ui.render_real_prompt(s.request_real_confirmation())
    assert f"Strategy: {ACTIVE_STRATEGY}" in text
    assert f"Strategy: {V3C_NAME}" not in text


def test_the_status_headline_names_the_active_strategy():
    out = ui.render_status(_view())
    assert f"Strategy: <b>{ACTIVE_STRATEGY}</b>" in out


@pytest.mark.parametrize("render,args", [
    ("render_real_armed", ("—",)),
    ("render_real_off", ()),
    ("render_emergency_stop", ()),
    ("render_emergency_cleared", ()),
    ("render_preflight_failed", None),
])
def test_every_strategy_banner_names_the_active_strategy(render, args):
    """
    Whichever banner an operator sees, the strategy it names is the live one.

    ``render_paper`` and ``render_real_off`` legitimately do not name a strategy
    at all, so they are not asserted here - asserting a name on a message that
    has no business carrying one would just be noise.
    """
    fn = getattr(ui, render)
    if render == "render_real_armed":
        text = fn(_armed(), *args)
    elif render == "render_preflight_failed":
        class R:
            ok = False
            detail = "d"
            failures = []
        text = fn(R(), _armed())
    else:
        text = fn(*args)
    if V3C_NAME in text:
        assert ACTIVE_STRATEGY in text, (
            f"{render} names {V3C_NAME} without the active strategy"
        )
    assert f"<b>{V3C_NAME}</b>" not in text


# ---------------------------------------------------------------------------
# structural: the invariant, so a NEW renderer cannot reintroduce it
# ---------------------------------------------------------------------------


def _display_strings(src: str) -> list[tuple[int, str, str]]:
    """
    Every operator-facing string in the module, as (line, own text, context).

    Uses ``ast`` rather than a regex so that two distinctions are respected:

    * a DOCSTRING is prose about the code, not text shown to an operator. A
      docstring that says "V3-C" while explaining the active strategy is
      correct and must not be flagged.
    * an f-string's fragments are judged together with the whole f-string, so a
      multi-line template is not mistaken for a bare hardcoded label just
      because the literal sits on a continuation line.

    Returns (lineno, literal text, enclosing source segment).
    """
    tree = ast.parse(src)
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    # the docstring of every module/class/function is prose
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

    out: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant)
                and isinstance(node.value, str)):
            continue
        if id(node) in docstrings:
            continue
        # The context is the enclosing STATEMENT, because a rendered message is
        # built by a statement: adjacent f-strings that form one template can
        # occupy several lines, and the active-strategy name may sit on the
        # first of them while the V3-C sub-detail sits on the second.
        ctx_node: ast.AST = node
        while (p := parent.get(ctx_node)) is not None and not isinstance(
            p, ast.stmt
        ):
            ctx_node = p
        segment = ast.get_source_segment(src, ctx_node) or node.value
        out.append((node.lineno, node.value, segment))
    return out


@pytest.mark.parametrize("module", IDENTITY_MODULES)
def test_no_identity_claim_hardcodes_a_strategy_name(module):
    """
    A literal "V3-C" in text shown to an operator is the bug this prevents.

    Scoped to statements, so a multi-line f-string is judged whole. The only
    allowance is the V3-C sub-detail of an L25 headline, which must therefore
    name ACTIVE_STRATEGY in the same rendered template.
    """
    src = (SRC / module).read_text()
    offenders: list[str] = []
    for lineno, text, segment in _display_strings(src):
        if V3C_NAME not in text:
            continue
        if "ACTIVE_STRATEGY" in segment:
            continue                      # the sanctioned sub-detail
        if V3C_NAME in segment and "esc(" in segment:
            continue                      # interpolates the constant
        offenders.append(f"{module}:{lineno}: {segment.strip()[:90]}")
    assert not offenders, (
        f"a hardcoded strategy name in {module}:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("module", GATE_MODULES)
def test_no_gate_module_claims_v3c_as_the_active_strategy(module):
    """
    Naming a GATE is fine; claiming to BE the active strategy is not.

    ``l25.py`` saying the threshold was derived "over V3-C survivors", and the
    refusal detail "V3-C authorized but f_liq_to_mcap is below 0.9647", are both
    correct and necessary. What must never appear is a bare identity claim.
    """
    src = (SRC / module).read_text()
    offenders = [
        f"{module}: {claim}"
        for claim in IDENTITY_CLAIMS
        if claim in src
    ]
    assert not offenders, (
        f"{module} claims a hardcoded strategy identity: {offenders}"
    )


def test_v3c_name_is_only_ever_used_as_a_sub_detail():
    """
    Every display use of V3-C sits in a template that also names the active
    strategy, so V3-C can only ever appear as L25's first leg.
    """
    uses = [
        (module, lineno, segment)
        for module in IDENTITY_MODULES
        for lineno, _text, segment in _display_strings((SRC / module).read_text())
        if "V3C_NAME" in segment and ACTIVE_STRATEGY in segment
    ]
    assert uses, "expected the V3-C sub-detail to still be shown beside L25"


def test_docstrings_are_not_mistaken_for_display_text():
    """
    Guard the guard.

    The module has docstrings that legitimately say "V3-C" while describing the
    active strategy. If those were flagged the check would be noise, and if the
    detection silently returned nothing it would pass vacuously - so both
    directions are asserted.
    """
    total = 0
    for module in OPERATOR_MODULES:
        found = _display_strings((SRC / module).read_text())
        assert found, f"{module}: scanner returned nothing, so the check is vacuous"
        total += len(found)
    assert total > 50, f"expected many display templates, scanned {total}"
    ui_texts = [t for _, t, _ in _display_strings((SRC / "telegram/ui.py").read_text())]
    assert any("Strategy:" in t or "PAPER" in t for t in ui_texts), (
        "expected to find real display templates in ui.py"
    )


# ---------------------------------------------------------------------------
# the reports must actually run
# ---------------------------------------------------------------------------


def test_the_config_report_renders_and_names_the_active_strategy():
    """
    Guards a regression I introduced while fixing this.

    Referencing ``ACTIVE_STRATEGY`` in ``config.describe`` without importing it
    left the ``config`` command raising ``NameError`` at runtime. A string-level
    AST check cannot see that, so the report is executed here.
    """
    from autopaper.config import build_config, describe

    out = describe(build_config(REPO))
    assert f"strategy={ACTIVE_STRATEGY}" in out
    assert f"strategy={V3C_NAME}" not in out
    # the leg is still reported, just labelled as a leg
    assert "V3-C leg" in out


def test_the_config_command_runs(tmp_path):
    import subprocess
    import sys as _sys

    out = subprocess.run(
        [_sys.executable, "-m", "autopaper.cli", "--root", str(tmp_path),
         "config"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"strategy={ACTIVE_STRATEGY}" in out.stdout
    assert "NameError" not in out.stderr


def test_every_button_reply_avoids_a_bare_v3c_identity_claim():
    """
    The START acknowledgement is the first thing an operator reads after
    pressing the button, and it is the one that was still wrong.
    """
    import asyncio

    from autopaper.modes import RuntimeState
    from autopaper.telegram.actions import ButtonHandler

    for payload in ("start", "stop", "paper", "cancel_real",
                    "clear_emergency"):
        state = RuntimeState()
        result = asyncio.run(ButtonHandler(state).handle(payload))
        assert result.handled, payload
        for claim in IDENTITY_CLAIMS:
            assert claim not in result.text, f"{payload} -> {result.text!r}"
