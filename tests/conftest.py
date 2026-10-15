"""Shared test fixtures. Everything is offline: no network, no Telegram, no RPC."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autopaper.clock import FakeClock
from autopaper.config import (
    Config,
    DecisionConfig,
    ExitConfig,
    ExecutionModel,
    MarketConfig,
    NarConfig,
    RiskLimits,
    TelegramConfig,
)
from autopaper.ledger import Ledger
from autopaper.models import AlertSnapshot, Observation

BASE_T = 1_700_000_000.0
MINT = "So11111111111111111111111111111111111111112"
MINT2 = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


@pytest.fixture(autouse=True)
def isolate_operator_env(monkeypatch, tmp_path, request):
    """
    Tests must NEVER read the operator's real ``.env.local``.

    Several tests call ``build_config(".")``, which resolves to the repository
    root and therefore loaded whatever the operator happens to have configured
    there. That made assertions like "REAL is not enabled by default" depend on
    the developer's local file: an operator with ``REAL_TRADING_ENABLED=true`` or
    ``NAR_TIMEOUT_S=5`` in ``.env.local`` turned documented-default tests red,
    and the only ways to "fix" them were to edit a secret file or weaken the
    assertion. Both are worse than isolating the test.

    This fixture makes any read of a ``.env.local`` OUTSIDE a per-test tmp_path
    return nothing, so ``build_config`` sees an empty environment and falls back
    to the documented dataclass defaults. ``.env.example`` is a tracked template
    and is NOT isolated, so the tests that assert its contents still read it.

    Nothing here modifies ``.env.local``; it only intercepts the read.
    """
    from autopaper import config as _cfgmod

    real_loader = _cfgmod.load_env_file
    allowed_root = str(Path(tmp_path).resolve())

    def _guarded(path):
        p = Path(path)
        if p.name == ".env.local" and not str(p.resolve()).startswith(
                allowed_root):
            # Not this test's own file -> the operator's. Return nothing.
            return {}
        return real_loader(path)

    monkeypatch.setattr(_cfgmod, "load_env_file", _guarded)

    # TESTS MUST NEVER WRITE INTO THE OPERATOR'S var/ DIRECTORY.
    #
    # Hold-path telemetry is now ON BY DEFAULT and resolves its sink under the
    # runtime's var_dir, which for ``build_config(".")`` is the repository's own
    # ``var/``. Without this, every engine test that opened a PAPER position
    # appended observations to the real file -- and did so with the test
    # fixture's constants (BUY_TOKENS_RAW = 5_176_254_910), so 1,262 fake
    # "observations" of the live strategy's hold path were sitting in the
    # production evidence base. A measurement is worth nothing if its input can
    # be written by a test, so the sink is redirected per test.
    # The SINK OVERRIDE IS PATCHED, NOT SET AS AN ENVIRONMENT VARIABLE.
    # ``build_config`` reports ``config_source == "environment"`` when any
    # variable is present, and ``test_config_source_reports_defaults_without_a_file``
    # asserts the clean-defaults path -- so redirecting via env would make an
    # unrelated invariant fail for an unrelated reason.
    from autopaper.paper_live import engine as _engmod

    _redirected = Path(tmp_path) / "paper_hold_path.jsonl"
    monkeypatch.setattr(_engmod, "_HOLD_PATH_SINK", _redirected)

    # ...and REPAIR it after every test, because a test that reassigns the module
    # global without restoring it would otherwise leave the NEXT test writing into
    # the operator's var/. Relying on each test's own teardown to be correct made
    # this depend on every future test remembering to clean up, which is not a
    # property a repository can rely on.
    def _repair():
        if _engmod._HOLD_PATH_SINK is not _redirected:
            _engmod._set_hold_path_sink(_redirected)
    request.addfinalizer(_repair)
    return _guarded


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(t=BASE_T)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    var = tmp_path / "var"
    logs = tmp_path / "logs"
    var.mkdir()
    logs.mkdir()
    risk = RiskLimits()
    return Config(
        agent_id="AUTONOMOUS_PAPER_AGENT_V1",
        root=tmp_path,
        var_dir=var,
        log_dir=logs,
        state_file=var / "paper_account.json",
        ledger_file=var / "ledger.jsonl",
        decisions_file=var / "decisions.jsonl",
        candidates_file=var / "candidates.jsonl",
        updates_file=var / "position_updates.jsonl",
        control_file=var / "control.json",
        telegram=TelegramConfig(api_id=1, api_hash="h", session=None, chat="testchat"),
        nar=NarConfig(),
        market=MarketConfig(),
        execution=ExecutionModel(),
        risk=risk,
        decision=DecisionConfig(),
        exits=ExitConfig(disaster_cap_pct=risk.disaster_cap_pct),
        env={},
    )


@pytest.fixture
def ledger(config: Config, clock: FakeClock) -> Ledger:
    return Ledger(
        config.ledger_file,
        config.decisions_file,
        config.candidates_file,
        config.updates_file,
        clock=clock,
    )


def obs(
    t: float,
    price: float = 1e-4,
    mint: str = MINT,
    **kw,
) -> Observation:
    base = dict(
        mint=mint,
        observed_at=t,
        received_at=t + 0.05,
        price_usd=price,
        market_cap_usd=1_200_000.0,
        liquidity_usd=60_000.0,
        volume_5m_usd=40_000.0,
        volume_1h_usd=200_000.0,
        buys_5m=180,
        sells_5m=110,
        txns_5m=290,
        pair_created_at=t - 600,
        provider="test",
    )
    base.update(kw)
    return Observation(**base)


def alert(t: float = BASE_T, mint: str = MINT, **fields) -> AlertSnapshot:
    return AlertSnapshot(
        candidate_id="cand_test0001",
        source="telegram",
        chat="testchat",
        message_id=1234,
        source_timestamp=t - 0.4,
        received_timestamp=t,
        raw_text="synthetic",
        mint=mint,
        symbol=fields.pop("symbol", "TEST"),
        name=fields.pop("name", "Test Token"),
        fields=fields,
        provenance={k: "test" for k in fields},
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


#: Marks a test whose premise was "the live source tree can re-derive the
#: archived V1 research result". That was already untrue after the V2 cut and is
#: definitively untrue after the V3 migration: V1 could select five entry routes
#: and V3 leaves exactly one (FAST_ENTRY), so the code path under measurement
#: has been deliberately deleted.
#:
#: The V1 artifacts under research/ and reports/ are ARCHIVED. The migration
#: instruction is explicit - "do not modify historical records", "do not rewrite
#: historical research", "preserve research provenance" - so they are read as
#: historical records and are NOT re-derived. These tests skip with a stated
#: reason rather than failing, so the intent stays visible instead of being
#: inferred from a red mark.
archived_v1 = pytest.mark.skipif(
    True,
    reason=(
        "ARCHIVED: asserts the live source can re-derive the V1 result. V1 "
        "selected five entry routes; V3 leaves one (FAST_ENTRY), so the code "
        "path under measurement has been deliberately removed. The V1 artifacts "
        "are preserved as historical records and are not re-derived."
    ),
)


# ---------------------------------------------------------------------------
# THE OFFLINE GUARANTEE
# ---------------------------------------------------------------------------
#
# Execution-aware PAPER is now the DEFAULT execution model, so a PAPER test
# reaches the executable-quote path without asking for it. That created a real
# hazard: the runtime builds an RPC client and a chain reader when it needs a
# quote, so an offline test could reach Solana simply because the default was
# flipped.
#
# The fix is injection at the test boundary, NOT disabling executable quotes.
# This fixture refuses to let a test construct a live quote client unless it
# declares that it wants one, by patching the ONE factory that builds the
# read-only stack. A test that genuinely needs the real stack overrides this
# with ``offline_network_ok``, and such a test is marked ``live``.
#
# The consequence is that "the default is executable quotes" and "the suite is
# offline" are both true at once, which is the requirement.
# ---------------------------------------------------------------------------

_OFFLINE_GUARD = "autopaper.paper_live.quotes.QuoteService"


class _RefuseNetworkQuoteService:
    """A quote service that cannot reach anything, and says why loudly."""

    def __init__(self, reason: str):
        self.reason = reason

    def _refuse(self, **kwargs):
        raise AssertionError(
            "an offline test tried to build a REAL quote service. Inject a fake "
            f"with PaperTrader(quote_service=...), or mark the test "
            f"`@pytest.mark.live` and request `offline_network_ok`. "
            f"({self.reason})"
        )

    quote_buy = quote_sell = quote_candidates = quote_buy_best = (
        quote_sell_pinned
    ) = _refuse


@pytest.fixture(autouse=True)
def offline_by_default(monkeypatch, request):
    """
    Block accidental network access from the executable-quote path.

    Only the QuoteService CONSTRUCTION is blocked. Everything a test does
    around it -- the strategy, sizing, the economic guard, fills, settlement,
    persistence -- runs exactly as production runs it. Nothing is weakened; the
    only thing prevented is a test reaching the internet by accident.
    """
    if request.node.get_closest_marker("live"):
        return None
    # The hazard is SPECIFIC: an offline test causing PaperTrader to build a
    # chain client because the default flipped. It is NOT "no test may name
    # SolanaReadOnlyRpc" -- the same_market tests legitimately construct fakes
    # and injected clients from these very classes, and blocking the names would
    # break correct tests.
    #
    # So the guard is placed on the construction SITE, and only refuses when the
    # runtime was not given an injected quote service.
    from autopaper.engine import PaperTrader

    original = PaperTrader._paper_bridge_instance

    async def _guarded(self):
        # An already-built bridge is an INJECTED bridge: the test supplied its
        # own collaborators, so there is nothing left to construct and nothing
        # to reach for. Async because bridge construction now RESOLVES SOL/USD
        # through the canonical provider, so it can await.
        if self.quote_service_override is not None or self._paper_bridge is not None:
            return await original(self)
        raise AssertionError(
            "an offline test reached the executable-quote runtime without "
            "injecting a quote service, so it would have built a real Solana "
            "RPC client. Inject one with "
            "PaperTrader(config, quote_service=<fake>), set the explicit "
            "rollback flag, or mark the test @pytest.mark.live."
        )

    monkeypatch.setattr(PaperTrader, "_paper_bridge_instance", _guarded)

    # The same hazard, one layer over: the canonical SOL/USD reference provider.
    # Production builds it lazily on the first price read, which is correct --
    # but it means a test that reaches a price without injecting one would build
    # a real HTTP client and call the free keyless Jupiter price API. Refuse the
    # lazy build instead, so the offline guarantee is enforced at the same kind
    # of boundary as the quote service above, and a test says what it means:
    #     PaperTrader(config, sol_usd_provider=<fake>)
    original_sol = PaperTrader._sol_usd_reference

    def _guarded_sol(self):
        if self.sol_usd_provider is None:
            raise AssertionError(
                "an offline test reached the SOL/USD reference without "
                "injecting a provider, so it would have built a real Jupiter "
                "price client. Inject one with "
                "PaperTrader(config, sol_usd_provider=<fake>), or mark the "
                "test @pytest.mark.live."
            )
        return original_sol(self)

    monkeypatch.setattr(PaperTrader, "_sol_usd_reference", _guarded_sol)
    return None
