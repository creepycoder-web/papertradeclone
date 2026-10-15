"""
Offline tests for the read-only forensics modules and the Telegram reader fix.

Everything here is stubbed. No network, no RPC, no Telegram, no filesystem
write outside pytest's own tmp_path.

Run:  PYTHONPATH=src .venv/bin/python -m pytest tests/research/test_forensics_isolation.py -q
"""


from __future__ import annotations

import ast
import asyncio
import sys
import base64
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.execution,
    pytest.mark.integrity,
    pytest.mark.paper_live,
    pytest.mark.research,
    pytest.mark.safety,
    pytest.mark.same_market,
]


AI = Path(__file__).resolve().parents[2]
SRC = AI / "src" / "autopaper"
PKG = SRC / "forensics"

#: Anything that could move money. A research reader must contain none of it.
FORBIDDEN_NAMES = (
    "broadcast", "send_transaction", "sign_transaction", "build_and_sign",
    "confirm_transaction", "submit_transaction", "real_buy", "real_sell",
    "arm_real", "settle_real", "place_order", "create_order", "execute_swap",
)
#: Modules that CAN do those things. A forensics module must not import them.
FORBIDDEN_IMPORTS = (
    "autopaper.real_account", "autopaper.real_equity", "autopaper.settlement",
    "autopaper.external_close", "autopaper.execution.real",
    "autopaper.execution.solana", "autopaper.execution.chainstate",
    "autopaper.execution.signing", "autopaper.telegram.client",
    "autopaper.paper_live",
)
#: Paths no forensics module may write to.
SENSITIVE_PATHS = (
    ".env.local", "var/", "paper_account.json", "real_account_state.json",
    "real_execution_intents.jsonl", "ledger.jsonl",
)


def _pyfiles() -> list[Path]:
    return sorted(PKG.glob("*.py"))


# ===========================================================================
# 1. ISOLATION
# ===========================================================================


def test_no_broadcast_capable_calls_in_forensics():
    hits = []
    for p in _pyfiles():
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.attr if isinstance(fn, ast.Attribute)
                        else fn.id if isinstance(fn, ast.Name) else "")
                if any(name == b or name.startswith(b) for b in FORBIDDEN_NAMES):
                    hits.append(f"{p.name}:{node.lineno} {name}")
    assert not hits, hits


def test_no_execution_or_wallet_imports_in_forensics():
    hits = []
    for p in _pyfiles():
        t = p.read_text()
        for m in FORBIDDEN_IMPORTS:
            if f"import {m}" in t or f"from {m}" in t:
                hits.append(f"{p.name}: {m}")
    assert not hits, hits


def test_forensics_does_not_import_paper_live():
    """Names may appear in PROSE (to document the prohibition) but never as an
    import edge -- an import edge is what would let this code affect the
    strategy."""
    for p in _pyfiles():
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert "paper_live" not in a.name, f"{p.name}: {a.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "paper_live" not in mod, f"{p.name}: {mod}"
                for a in node.names:
                    assert "paper_live" not in a.name, f"{p.name}: {a.name}"


def test_no_execution_or_wallet_modules_reachable_from_forensics():
    """Transitive check: the forensics package must not pull in anything that
    can move money, even indirectly."""
    import subprocess
    code = (
        "import sys, importlib;"
        "[importlib.import_module('autopaper.forensics.' + m) "
        " for m in ('telegram_guard','transfer_fee','market_timing',"
        "'cost_forensics','min_trade')];"
        "bad=[m for m in sys.modules if m.startswith(('autopaper.paper_live',"
        "'autopaper.execution','autopaper.settlement','autopaper.real'))];"
        "print(sorted(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONPATH": str(SRC.parent), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


def test_no_writes_to_sensitive_paths_in_forensics():
    hits = []
    for p in _pyfiles():
        for node in ast.walk(ast.parse(p.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in ("write_text", "write_bytes", "open", "unlink",
                            "mkdir", "touch", "rename"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    for s in SENSITIVE_PATHS:
                        if s in arg.value:
                            hits.append(f"{p.name}:{node.lineno} {s}")
    assert not hits, hits


def test_no_credential_access_in_forensics():
    for p in _pyfiles():
        t = p.read_text()
        for bad in ("load_env_file", "dotenv", "bot_token", "api_hash",
                    "api_id", "private_key", "secret_key", "keypair",
                    "mnemonic", "seed_phrase"):
            assert bad not in t, f"{p.name} references {bad!r}"


# ===========================================================================
# 2. TASK 1 -- Telegram reader
# ===========================================================================

from autopaper.forensics import telegram_guard as G  # noqa: E402


def _msg(mid: int = 7, text: str = "x" * 40):
    class M:
        id = mid
        chat_id = -100123
        symbol = "LEGS"
        mint = "BQyKKFQCYhBQJgDARvrzUKqN4yMM1VB2rhZte8JMpump"
        message = text
    return M()


def test_none_cannot_reach_the_arithmetic_site():
    """
    The regression the crash needed. `coerce_positive_float` is the only value
    that is allowed to reach a `X * <int>`, so None is structurally impossible.
    """
    assert G.coerce_positive_float(None, 5.0) == 5.0
    assert G.coerce_positive_float(None, 120.0) == 120.0
    for bad in (None, "", "abc", 0, -1, float("nan"), float("inf"),
                float("-inf"), True, False, object()):
        got = G.coerce_positive_float(bad, 5.0)
        assert isinstance(got, float) and got > 0, (bad, got)
        # and the arithmetic the crash site performed must not raise
        assert (got * 2) > 0
    assert G.coerce_positive_float(7.5, 5.0) == 7.5


def test_arithmetic_site_does_not_raise_for_any_config_value():
    """The exact expression from the crash site, over every hostile input."""
    for raw in (None, 0, -5, "x", float("nan"), float("inf"), True, 1e308):
        backoff = G.coerce_positive_float(raw, 5.0)
        cap = G.coerce_positive_float(raw, 120.0)
        result = min(cap, backoff * 2)          # must never raise
        assert result > 0


def test_context_captures_type_message_cause_and_traceback():
    """The exact evidence the historical log discarded."""
    try:
        try:
            raise ValueError("inner cause")
        except ValueError as inner:
            raise RuntimeError("outer failure") from inner
    except RuntimeError as exc:
        ctx = G.CallbackContext.capture(exc, stage="handle_message",
                                       message=_msg(42), text="hello")
    assert ctx.exception_type == "RuntimeError"
    assert ctx.exception_message == "outer failure"
    assert ctx.exception_repr == "RuntimeError('outer failure')"
    assert ctx.cause_type == "ValueError"
    assert ctx.cause_message == "inner cause"
    assert "ValueError" in ctx.traceback_text
    assert "RuntimeError" in ctx.traceback_text
    assert "Traceback (most recent call last)" in ctx.traceback_text
    assert ctx.stage == "handle_message"
    assert ctx.message_id == 42
    assert ctx.chat_id == -100123
    assert ctx.symbol == "LEGS"
    assert ctx.mint.startswith("BQyK")
    assert ctx.text_length == 5
    assert ctx.is_cancellation is False


def test_describe_failure_includes_the_traceback():
    try:
        raise TypeError("unsupported operand type(s) for *: 'NoneType' and 'int'")
    except TypeError as exc:
        ctx = G.CallbackContext.capture(exc, stage="handle_message",
                                       message=_msg(), text="t")
    block = G.describe_failure(ctx)
    assert "stream CONTINUES" in block
    assert "traceback" in block
    assert "TypeError" in block
    assert "NoneType" in block


def test_reader_survives_an_isolated_callback_failure():
    """
    The availability defect: one bad message must NOT terminate the stream.
    Three messages, the middle one explodes, all three must be attempted.
    """
    seen: list[int] = []

    async def handle(message):
        seen.append(message.id)
        if message.id == 2:
            raise ValueError("bad message")

    async def drive():
        failures = G.FailureLog()
        for mid in (1, 2, 3):
            ok = await G.safe_handle(handle, _msg(mid), text="t",
                                     failures=failures, stage="handle_message")
            assert (ok is False) if mid == 2 else (ok is True)
        return failures

    failures = asyncio.run(drive())
    assert seen == [1, 2, 3], "every message must be attempted"
    assert failures.total == 1
    assert failures.counts_by_type == {"ValueError": 1}
    assert failures.counts_by_stage == {"handle_message": 1}
    assert failures.contexts[0].message_id == 2


def test_repeated_failures_do_not_grow_without_limit():
    async def handle(message):
        raise ValueError("always")

    async def drive():
        failures = G.FailureLog(history=8)
        for mid in range(50):
            await G.safe_handle(handle, _msg(mid), failures=failures)
        return failures

    f = asyncio.run(drive())
    assert f.total == 50, "the COUNT must be complete"
    assert len(f.contexts) == 8, "but retained CONTEXT must be bounded"
    assert f.history == 8


def test_cancellation_is_not_swallowed():
    """Shutdown behaviour must be preserved exactly."""
    async def handle(message):
        raise asyncio.CancelledError()

    async def drive():
        with pytest.raises(asyncio.CancelledError):
            await G.safe_handle(handle, _msg(), failures=G.FailureLog())

    asyncio.run(drive())


def test_stop_event_semantics_unchanged():
    """A set stop event must still break the loop before handling."""
    async def drive():
        stop = asyncio.Event()
        stop.set()
        assert stop.is_set()
        handled = []

        async def handle(m):
            handled.append(m.id)

        for m in (1, 2):
            if stop.is_set():
                break
            await G.safe_handle(handle, _msg(m), failures=G.FailureLog())
        return handled

    assert asyncio.run(drive()) == [], "a set stop must skip all handling"


def test_a_broken_reporter_does_not_become_a_second_failure():
    async def handle(message):
        raise ValueError("boom")

    def bad_reporter(ctx):
        raise RuntimeError("reporter itself is broken")

    async def drive():
        failures = G.FailureLog()
        ok = await G.safe_handle(handle, _msg(), failures=failures,
                                 on_failure=bad_reporter)
        return ok, failures

    ok, failures = asyncio.run(drive())
    assert ok is False
    assert failures.total == 1


def test_real_listener_uses_the_isolation_boundary():
    """
    Structural proof the fix is wired in: the reader must route every message
    through the guard, and must still log a traceback on a stream-level error.
    """
    src = (SRC / "telegram" / "client.py").read_text()
    assert "_safe_handle_message" in src, "reader does not use the guard"
    assert "telegram stream error traceback" in src, "no traceback on the log"
    assert "coerce_positive_float" in src, "arithmetic site not hardened"
    # No BARE await of _handle may remain. Matching on the raw text would false
    # -positive on the explanatory comment, so walk the AST instead.
    bare = [n.lineno for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "_handle"]
    assert not bare, f"bare await self._handle(...) still present at {bare}"
    # and the pre-existing one-line log is retained for continuity
    assert 'log("ERROR", "telegram stream error: %s", exc)' in src


def test_listener_registers_the_failure_log_and_reports_stats():
    from autopaper.config import TelegramConfig
    from autopaper.telegram.client import TelegramListener

    lis = TelegramListener(config=TelegramConfig())
    stats = lis.callback_failure_stats()
    assert stats["isolated_failures"] == 0
    assert stats["stream_alive"] is False       # never connected in this test
    assert stats["retained_contexts"] == 0
    lis._on_callback_failure(
        G.CallbackContext.capture(ValueError("x"), stage="s", message=_msg()))
    assert lis.callback_failure_stats()["isolated_failures"] == 1
    assert "1 message(s) failed" in lis.status.detail


def test_reconnect_backoff_is_coerced_at_both_sites():
    src = (SRC / "telegram" / "client.py").read_text()
    # the raw unguarded doubling must be gone
    assert "reconnect_backoff_max_s, backoff * 2" not in src
    assert "backoff = self.config.reconnect_backoff_s" not in src


def test_every_recorded_arithmetic_hazard_is_documented():
    for h in G.UNGUARDED_MULTIPLICATION_SITES:
        assert h.file and h.line and h.expression and h.operand, h
        assert h.note, "every hazard must carry a note explaining its status"
    assert {h.file for h in G.UNGUARDED_MULTIPLICATION_SITES} >= {
        "src/autopaper/telegram/client.py", "src/autopaper/engine.py"}


def test_no_arithmetic_hazard_is_still_reachable_with_a_possible_none():
    """
    The invariant the crash violated.

    A hazard may still EXIST in the source, but if it is reachable from the
    reader then the current tree must not admit a None into it. That is the
    whole difference between "documented" and "fixed".
    """
    live = [h for h in G.UNGUARDED_MULTIPLICATION_SITES
            if h.reachable_from_reader and h.can_be_none_in_current_tree]
    assert not live, [f"{h.file}:{h.line} {h.expression}" for h in live]


# ===========================================================================
# 3. TASK 2 -- transfer fee / Token-2022
# ===========================================================================

from autopaper.forensics import transfer_fee as TF  # noqa: E402

_PUBKEY_A = b"\x11" * 32
_PUBKEY_B = b"\x22" * 32


def _fee_payload(older_bps, older_cap, older_epoch,
                 newer_bps, newer_cap, newer_epoch,
                 *, withheld=12_345, auth=_PUBKEY_A, wdr=_PUBKEY_B):
    """A byte-exact 108-byte TransferFeeConfig payload, real layout."""
    p = bytearray(TF.TRANSFER_FEE_CONFIG_SIZE)
    p[0:32] = auth
    p[32:64] = wdr
    p[64:72] = int(withheld).to_bytes(8, "little")
    for base, (bps, cap, ep) in ((TF.OLDER_FEE_OFFSET,
                                  (older_bps, older_cap, older_epoch)),
                                 (TF.NEWER_FEE_OFFSET,
                                  (newer_bps, newer_cap, newer_epoch))):
        p[base + 0:base + 8] = int(ep).to_bytes(8, "little")
        p[base + 8:base + 16] = int(cap).to_bytes(8, "little")
        p[base + 16:base + 18] = int(bps).to_bytes(2, "little")
    return bytes(p)


def _tlv(etype: int, payload: bytes) -> bytes:
    return (int(etype).to_bytes(2, "little")
            + len(payload).to_bytes(2, "little") + payload)


def _acct(payload: bytes, *, etype: int = 1,
          account_type: int = TF.ACCOUNT_TYPE_MINT,
          pad: bytes = b"\x00") -> bytes:
    """Full extended-mint account: 82-byte body, zero-padded to 165, the
    account-type byte at 165, then the TLV stream."""
    body = bytes(82) + pad * 83 + bytes([account_type])
    return body + _tlv(etype, payload)


def _mint_with_fee(bps=500, cap=1_000_000_000, epoch=700):
    """One live configuration (older is zero, newer takes effect at `epoch`)."""
    return _acct(_fee_payload(0, 0, 1, bps, cap, epoch))


def _mint_without_fee() -> bytes:
    """An extended mint carrying a DIFFERENT extension, not a transfer fee."""
    return _acct(b"\x00" * 6, etype=9)


def _legacy_reader(mint_bytes, owner):
    r = TF.StubAccountReader()
    r.set("M", mint_bytes, owner)
    return r


# ---- the three states -------------------------------------------------------


def test_legacy_spl_token_is_provably_fee_free():
    rep = TF.inspect_transfer_fee(
        "M", _legacy_reader(bytes(82), TF.TOKEN_PROGRAM_ID))
    assert rep.transfer_fee_status is TF.TransferFeeStatus.NONE
    assert rep.token_program is TF.TokenProgram.LEGACY_SPL
    assert rep.transfer_fee_bps is None
    assert "no on-chain transfer fee exists" in rep.reason


def test_token_2022_without_the_extension_is_NONE():
    """Extended mint, real account-type byte, but no TransferFeeConfig."""
    r = TF.StubAccountReader()
    r.set("M", _mint_without_fee(), TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.NONE
    assert "NO TransferFeeConfig extension" in rep.reason


def test_token_2022_with_zero_fee_in_both_configs_is_NONE():
    r = TF.StubAccountReader()
    r.set("M", _acct(_fee_payload(0, 0, 1, 0, 0, 700)),
          TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r, epoch=999)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.NONE
    assert rep.transfer_fee_bps == 0


def test_unreadable_account_is_UNKNOWN_not_NONE():
    rep = TF.inspect_transfer_fee("NOPE", TF.StubAccountReader())
    assert rep.transfer_fee_status is TF.TransferFeeStatus.UNKNOWN
    assert "not found" in rep.reason


# ---- the epoch-gated configurations -----------------------------------------


def test_transfer_fee_config_is_decoded_from_the_real_108_byte_layout():
    r = TF.StubAccountReader()
    r.set("M", _acct(_fee_payload(100, 10_000, 1, 500, 1_000_000_000, 700)),
          TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r, epoch=999)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.PRESENT
    assert rep.token_program is TF.TokenProgram.TOKEN_2022
    assert rep.older_transfer_fee_bps == 100
    assert rep.newer_transfer_fee_bps == 500
    assert rep.active_config == "newer"
    assert rep.transfer_fee_bps == 500
    assert rep.buy_fee_bps == 500 and rep.sell_fee_bps == 500
    assert rep.max_fee_lamports == 1_000_000_000
    assert rep.max_fee_sol == pytest.approx(1.0)


def test_active_config_switches_at_the_newer_epoch():
    """spl-token-2022's get_epoch_fee: newer wins once the epoch is reached."""
    r = TF.StubAccountReader()
    r.set("M", _acct(_fee_payload(100, 10_000, 1, 500, 1_000_000_000, 700)),
          TF.TOKEN_2022_PROGRAM_ID)
    assert TF.inspect_transfer_fee("M", r, epoch=0).active_config == "older"
    assert TF.inspect_transfer_fee("M", r, epoch=699).active_config == "older"
    assert TF.inspect_transfer_fee("M", r, epoch=699).transfer_fee_bps == 100
    assert TF.inspect_transfer_fee("M", r, epoch=700).active_config == "newer"
    assert TF.inspect_transfer_fee("M", r, epoch=700).transfer_fee_bps == 500


def test_without_a_cluster_epoch_no_single_rate_is_invented():
    """The canonical Token-2022 bug: reporting only `newer`."""
    r = TF.StubAccountReader()
    r.set("M", _acct(_fee_payload(100, 10_000, 1, 500, 1_000_000_000, 700)),
          TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.PRESENT
    assert rep.transfer_fee_bps is None, "must not default to the newer config"
    assert rep.max_fee_lamports is None
    assert rep.older_transfer_fee_bps == 100
    assert rep.newer_transfer_fee_bps == 500
    assert "epoch" in " ".join(rep.warnings)


def test_a_pending_fee_increase_is_surfaced():
    r = TF.StubAccountReader()
    r.set("M", _acct(_fee_payload(100, 10_000, 1, 500, 1_000_000_000, 700)),
          TF.TOKEN_2022_PROGRAM_ID)
    assert "already scheduled" in TF.inspect_transfer_fee("M", r, epoch=699).reason
    assert "no longer applied" in TF.inspect_transfer_fee("M", r, epoch=700).reason


def test_authorities_and_withheld_state_are_readable():
    r = TF.StubAccountReader()
    r.set("M", _acct(_fee_payload(100, 1, 1, 200, 2, 700, withheld=987_654_321)),
          TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r, epoch=999)
    assert rep.transfer_fee_config_authority is not None
    assert rep.withdraw_withheld_authority is not None
    assert rep.transfer_fee_config_authority != rep.withdraw_withheld_authority
    assert rep.withheld_amount_lamports == 987_654_321


def test_an_all_zero_authority_reads_as_absent_not_broken():
    """OptionalNonZeroPubkey: no tag byte, all-zero means None."""
    r = TF.StubAccountReader()
    r.set("M", _acct(_fee_payload(100, 1, 1, 200, 2, 700,
                                     auth=b"\x00" * 32, wdr=b"\x00" * 32)),
          TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r, epoch=999)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.PRESENT
    assert rep.transfer_fee_config_authority is None
    assert rep.withdraw_withheld_authority is None


# ---- UNKNOWN must stay UNKNOWN ---------------------------------------------


def test_malformed_extension_stays_UNKNOWN_and_never_zero():
    r = TF.StubAccountReader()
    # well-formed TLV header, but the payload is short
    bad = (bytes(165) + bytes([1])
           + (1).to_bytes(2, "little") + (200).to_bytes(2, "little")
           + b"\x01\x02")
    r.set("M", bad, TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.UNKNOWN
    assert rep.transfer_fee_bps is None
    assert "cannot be reported as fee-free" in " ".join(rep.warnings)


def test_a_truncated_payload_is_UNKNOWN():
    assert TF.decode_transfer_fee_config(b"\x00" * 107)["ok"] is False
    assert TF.decode_transfer_fee_config(bytes(108))["ok"] is True


def test_impossible_basis_points_are_rejected_not_clamped():
    """Above MAX_FEE_BASIS_POINTS the program itself refuses to compute a fee."""
    for bps in (10_001, 65_535):
        dec = TF.decode_transfer_fee_config(_fee_payload(0, 0, 1, bps, 1, 700))
        assert dec["ok"] is False
        assert dec["transfer_fee_bps"] is None
        assert "outside the valid range" in dec["error"]


def test_wrong_declared_extension_length_is_UNKNOWN():
    """A non-108-byte TransferFeeConfig is a layout this module cannot read."""
    acct = (bytes(165) + bytes([1])
            + (1).to_bytes(2, "little") + (90).to_bytes(2, "little")
            + bytes(90))
    r = TF.StubAccountReader()
    r.set("M", acct, TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.UNKNOWN
    assert "fixed at 108" in rep.reason


def test_an_account_that_is_not_a_mint_is_refused():
    """A 165-byte legacy token ACCOUNT must not be parsed as a mint."""
    acct = bytes(165) + bytes([TF.ACCOUNT_TYPE_ACCOUNT])
    r = TF.StubAccountReader()
    r.set("M", acct, TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.UNKNOWN
    assert "not 1 (mint)" in rep.reason


def test_a_non_zero_mint_padding_region_is_refused():
    r = TF.StubAccountReader()
    r.set("M", _acct(_fee_payload(0, 0, 1, 300, 5, 700), pad=b"\xff"),
          TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.UNKNOWN
    assert "padding" in rep.reason


def test_oversized_account_is_refused_not_parsed():
    r = TF.StubAccountReader()
    r.set("M", bytes(99_999), TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.UNKNOWN
    assert "refusing to parse" in rep.reason


def test_invalid_base64_is_UNKNOWN():
    class Raw:
        def get_account_info(self, address):
            return {"data": "!!!not base64!!!", "owner": TF.TOKEN_2022_PROGRAM_ID}

    rep = TF.inspect_transfer_fee("B", Raw())
    assert rep.transfer_fee_status is TF.TransferFeeStatus.UNKNOWN
    assert "base64" in rep.reason


def test_custom_program_is_UNKNOWN_and_marks_asymmetry_unprovable():
    r = TF.StubAccountReader()
    r.set("CUS", bytes(200), "SomeCustomProgram11111111111111111111111111")
    rep = TF.inspect_transfer_fee("CUS", r)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.UNKNOWN
    assert rep.token_program is TF.TokenProgram.UNKNOWN
    assert rep.asymmetric_unprovable is True
    assert rep.buy_fee_bps is None and rep.sell_fee_bps is None
    assert "never as zero" in " ".join(rep.warnings)


def test_unavailable_rpc_is_UNKNOWN_not_a_crash():
    class Dead:
        def get_account_info(self, address):
            raise ConnectionError("rpc unavailable")

    rep = TF.inspect_transfer_fee("X", Dead())
    assert rep.transfer_fee_status is TF.TransferFeeStatus.UNKNOWN
    assert "rpc unavailable" in rep.reason


def test_a_legacy_mint_length_short_of_82_is_UNKNOWN():
    r = TF.StubAccountReader()
    r.set("M", bytes(40), TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.UNKNOWN


def test_unknown_owner_is_UNKNOWN_never_NONE():
    """The status that matters most: an unrecognised program is not 'no fee'."""
    r = TF.StubAccountReader()
    r.set("M", bytes(82), "NotARealProgram")
    assert TF.inspect_transfer_fee("M", r).transfer_fee_status \
        is TF.TransferFeeStatus.UNKNOWN


# ---- walker + normalisation --------------------------------------------------


def test_extension_scan_passes_over_unknown_types():
    """A non-fee extension must be walked past, not mistaken for a fee."""
    other = _tlv(9, b"\x00" * 6)
    acct = (bytes(165) + bytes([1]) + other
            + _tlv(1, _fee_payload(0, 0, 1, 300, 5, 700)))
    r = TF.StubAccountReader()
    r.set("M", acct, TF.TOKEN_2022_PROGRAM_ID)
    rep = TF.inspect_transfer_fee("M", r, epoch=999)
    assert rep.transfer_fee_status is TF.TransferFeeStatus.PRESENT
    assert rep.transfer_fee_bps == 300


def test_the_account_type_byte_is_required():
    """Without the byte 165 == 1 there is no extension list to trust."""
    acct = bytes(166) + _tlv(1, _fee_payload(0, 0, 1, 300, 5, 700))
    r = TF.StubAccountReader()
    r.set("M", acct, TF.TOKEN_2022_PROGRAM_ID)
    assert TF.inspect_transfer_fee("M", r, epoch=999).transfer_fee_status \
        is TF.TransferFeeStatus.UNKNOWN


def test_decimal_normalisation():
    assert TF.normalise_bps(0) == 0
    assert TF.normalise_bps(10_000) == 10_000
    for bad in (None, True, -1, 10_001, 1.5, "500"):
        assert TF.normalise_bps(bad) is None, bad
    assert TF.normalise_lamports(0) == 0
    assert TF.normalise_lamports(-1) is None
    assert TF.normalise_lamports(None) is None


def test_batch_inspection_isolates_failures():
    r = TF.StubAccountReader()
    r.set("OK", _mint_with_fee(100, 1, 700), TF.TOKEN_2022_PROGRAM_ID)
    reps = TF.inspect_many(["OK", "MISSING", "OK"], r, epoch=999)
    assert [x.transfer_fee_status.value for x in reps] == [
        "PRESENT", "UNKNOWN", "PRESENT"]


def test_report_never_claims_an_estimate():
    r = TF.StubAccountReader()
    r.set("F", _mint_with_fee(500, 1, 700), TF.TOKEN_2022_PROGRAM_ID)
    blob = TF.inspect_transfer_fee("F", r, epoch=999).as_dict()
    assert "not an estimate" in blob["is_not_a_tax_estimate"]
    assert set(blob) >= {
        "transfer_fee_status", "buy_fee_bps", "sell_fee_bps",
        "max_fee_lamports", "token_program", "evidence_source", "state_slot",
        "timestamp", "older_transfer_fee_bps", "newer_transfer_fee_bps",
        "transfer_fee_config_authority", "withdraw_withheld_authority",
        "withheld_amount_lamports"}


# ===========================================================================
# 4. TASK 3 -- market timing
# ===========================================================================

from autopaper.forensics import market_timing as MT  # noqa: E402

BUDGET = 8_000.0


#: Fixed "now" so every classification is deterministic and does not depend on
#: the wall clock. 1000.5 is the quote instant, so nothing is stale by default.
REF = 1000.5


def _obs(**kw):
    base = dict(mint="M", dexscreener_pair_address="POOL1",
                observed_mark=2.0, observed_mark_timestamp=1000.0,
                quote_price=2.1, quote_timestamp=1000.5,
                state_fetch_timestamp=1000.4, quote_slot=200, state_slot=199,
                captured_at=REF)
    base.update(kw)
    return MT.build_observation(**base)


def _classify(obs, budget=BUDGET):
    """Always pin the reference time: the module falls back to time.time(), so
    a test that forgets this becomes a wall-clock test."""
    return MT.classify_timing(obs, freshness_budget_ms=budget,
                              reference_time=REF)


def test_same_window():
    d = _classify(_obs())
    assert d.mark_state_timing_status is MT.MarkStateTimingStatus.SAME_WINDOW
    assert d.timed_side == "none"
    assert d.as_dict()["is_a_trading_decision"] is False


def test_stale_mark_is_blamed_on_the_observed_side():
    d = _classify(_obs(observed_mark_timestamp=900.0))
    assert d.mark_state_timing_status is MT.MarkStateTimingStatus.STALE_MARK
    assert d.timed_side == "observed_mark"
    assert d.mark_age_ms == pytest.approx(100_500.0)


def test_stale_state_is_blamed_on_the_state_side():
    d = _classify(_obs(state_fetch_timestamp=900.0))
    assert d.mark_state_timing_status is MT.MarkStateTimingStatus.STALE_STATE
    assert d.timed_side == "on_chain_state"
    assert d.state_age_ms == pytest.approx(100_500.0)


def test_stale_quote_is_blamed_on_the_quote():
    d = _classify(_obs(quote_timestamp=900.0))
    assert d.mark_state_timing_status is MT.MarkStateTimingStatus.STALE_QUOTE
    assert d.timed_side == "quote"


def test_missing_timestamp_is_UNKNOWN_never_same_window():
    d = _classify(_obs(observed_mark_timestamp=None))
    assert d.mark_state_timing_status is MT.MarkStateTimingStatus.TIMING_UNKNOWN
    assert "observed_mark_timestamp" in d.missing_fields
    assert "absence of evidence" in d.reason


def test_no_budget_supplied_is_UNKNOWN_not_same_window():
    """This module must not be able to invent a threshold."""
    d = _classify(_obs(), budget=None)
    assert d.mark_state_timing_status is MT.MarkStateTimingStatus.TIMING_UNKNOWN
    assert "MarketConfig.stale_after_ms" in d.reason
    assert _classify(_obs(), budget=0).mark_state_timing_status \
        is MT.MarkStateTimingStatus.TIMING_UNKNOWN


def test_state_is_checked_before_mark():
    """A state older than the quote cannot be rescued by a fresh quote."""
    d = _classify(_obs(state_fetch_timestamp=900.0,
               observed_mark_timestamp=900.0))
    assert d.mark_state_timing_status is MT.MarkStateTimingStatus.STALE_STATE


def test_deltas_and_gap_are_computed():
    d = _classify(_obs())
    assert d.wall_clock_delta_ms == pytest.approx(500.0)
    assert d.slot_delta == 1
    assert d.price_gap_pct == pytest.approx((2.1 / 2.0 - 1) * 100)


def test_summary_counts_only_and_defines_no_threshold():
    pairs = [(o, _classify(o)) for o in
             (_obs(), _obs(observed_mark_timestamp=900.0),
              _obs(quote_timestamp=900.0))]
    s = MT.summarise(pairs)
    assert s["pairs"] == 3
    assert s["by_status"]["SAME_WINDOW"] == 1
    assert s["by_status"]["STALE_MARK"] == 1
    assert s["defines_a_threshold"] is False
    assert s["rejects_trades"] is False
    assert "NOT evidence of mispricing" in s["note"]


def test_provider_identity_rules_are_preserved_not_flattened():
    c = MT.identity_caveats()
    assert "multi-hop" in c["jupiter_quote_pool_key"]
    assert c["unknown_is_not_pass"].startswith("UNKNOWN identity stays UNKNOWN")


def test_real_same_market_identity_module_is_untouched_and_importable():
    """The extension must consume identity rules, never re-derive them."""
    from autopaper.same_market import identity
    assert hasattr(identity, "verify_pool_identity")
    assert hasattr(identity, "verify_identity")
    src = (SRC / "forensics" / "market_timing.py").read_text()
    assert "same_market" not in src or "identity_caveats" in src


# ===========================================================================
# 5. TASK 4 -- cost model
# ===========================================================================

from autopaper.forensics import cost_forensics as CF  # noqa: E402


def test_every_cost_component_is_traced_and_classified():
    comps = CF.traced_components()
    names = {c.name for c in comps}
    assert names >= {
        "trading_fee", "slippage_base", "size_impact", "staleness",
        "assumed_liquidity", "network_fee", "token_transfer_tax",
        "decision_latency"}
    for c in comps:
        assert c.kind in ("modeled", "assumed", "measured")
        assert c.source and c.note
    # the three biggest omissions must be classified as ASSUMED, not measured
    by = {c.name: c for c in comps}
    assert by["assumed_liquidity"].kind == "assumed"
    assert by["network_fee"].kind == "assumed"
    assert by["token_transfer_tax"].kind == "assumed"


def test_cost_formula_is_self_consistent():
    from autopaper.config import ExecutionModel
    from autopaper.execution.fill import impact_bps

    m = ExecutionModel()
    text = CF.cost_formula_text()
    assert f"{m.entry_fee_bps:g}" in text
    assert f"{m.entry_slippage_bps:g}" in text
    # and the arithmetic actually is 2*(fee+slip+impact) at a known size
    N, L = 10.0, 16_900.0
    imp = impact_bps(N, L, m)
    assert CF.compare_to_repo(N, L).repo_all_in_round_trip_bps == pytest.approx(
        2 * (m.entry_fee_bps + m.entry_slippage_bps + imp))


def test_amm_half_spread_matches_closed_form():
    for N, L in ((0.50, 16_900.0), (10.0, 16_900.0), (100.0, 10_000.0)):
        assert CF.amm_half_spread_bps(N, L) == pytest.approx(1e4 * N / (2 * L))
    assert CF.amm_half_spread_bps(10.0, 0.0) == 0.0
    assert CF.amm_half_spread_bps(0.0, 100.0) == 0.0


def test_sensitivity_grid_covers_every_requested_pair():
    grid = CF.sensitivity_grid()
    assert len(grid) == 18
    notionals = {r["notional_usd"] for r in grid}
    assert {0.5, 0.625, 1.0, 5.0, 10.0, 100.0} <= notionals
    liqs = {r["liquidity_usd"] for r in grid}
    assert {10_000.0, 16_900.0, 60_000.0} <= liqs
    # the assumed liquidity must be FLAGGED as assumed, never as observed
    for r in grid:
        assert r["liquidity_is_assumed"] == (r["liquidity_usd"] == 60_000.0)


def test_repo_cost_exceeds_amm_geometry_at_small_sizes():
    """The finding: the flat allowance is a large buffer, not an under-charge."""
    for r in CF.sensitivity_grid():
        assert r["repo_minus_amm_bps"] > 0, r


def test_small_bankroll_bias_is_quantified_not_asserted():
    b = CF.small_bankroll_bias()
    assert b["small_repo_impact_bps"] < b["large_repo_impact_bps"]
    assert b["impact_ratio_small_over_large"] < 1.0
    assert "FLATTERED" in b["verdict"]


def test_assumptions_register_labels_every_direction():
    a = CF.assumptions_register()
    assert {x["id"] for x in a} >= {"A1", "A2", "A3", "A4", "A5", "A6"}
    for x in a:
        assert x["assumption"] and x["reality"] and x["direction"]
    assert any("token transfer tax" in x["assumption"] for x in a)
    assert any("optimistic" in x["direction"] for x in a)


def test_cost_audit_declares_it_changes_nothing():
    a = CF.full_audit()
    assert a["changes_the_model"] is False
    assert a["pricing_label"] == "OBSERVATIONAL / MARK-BASED"


# ===========================================================================
# 6. TASK 5 -- minimum trade / rounding
# ===========================================================================

from autopaper.forensics import min_trade as MTD  # noqa: E402


def test_usd_to_base_units_truncates_and_reports_the_loss():
    a = MTD.usd_to_base_units(0.50, 2.234e-05, 6)
    assert a.integer_value == int(0.50 / 2.234e-05 * 1e6)
    assert a.truncation_loss >= 0.0
    assert "truncation" in a.note


def test_base_units_unbuyable_when_the_price_is_too_high():
    a = MTD.usd_to_base_units(0.50, 10.0, 0)
    assert a.integer_value == 0
    assert a.verdict is MTD.Verdict.BELOW_MINIMUM
    assert "UNBUYABLE" in a.note


def test_missing_price_is_UNKNOWN_not_zero():
    a = MTD.usd_to_base_units(0.50, None, 6)
    assert a.verdict is MTD.Verdict.UNKNOWN
    assert a.integer_value is None


def test_lamport_conversion_and_materiality():
    a = MTD.usd_to_lamports(0.50, 200.0)
    assert a.integer_value == 2_500_000
    m = MTD.lamport_truncation_is_material(0.50, 200.0)
    assert m["material"] is False
    assert "immaterial" in m["note"]


def test_unbuyable_boundary_scales_with_decimals():
    assert MTD.max_price_for_buyable(0.50, 0) == pytest.approx(0.50)
    assert MTD.max_price_for_buyable(0.50, 9) == pytest.approx(0.50 * 1e9)
    b = MTD.unbuyable_boundary()
    assert len(b["rows"]) == 12
    assert "cannot be collapsed to one number" in b["why_decimals_matters"]


def test_an_unverifiable_venue_floor_never_yields_a_pass():
    """
    The fabrication this audit exists to avoid.

    The floor is a constant nobody has checked against a live venue, so a
    position that clears it has NOT been shown to be tradeable.
    """
    for usd in (0.50, 1.0, 10.0, 1000.0):
        a = MTD.venue_minimum_status(usd)
        assert a.verdict is MTD.Verdict.UNKNOWN, usd
        assert a.clears_unverified_floor in (True, False)
        assert "NOT verifiable" in a.note
    assert MTD.KNOWN_VENUE_MINIMUMS[0]["verifiable_here"] is False
    assert MTD.venue_minimum_status(1000.0).clears_unverified_floor is True
    assert MTD.venue_minimum_status(0.01).clears_unverified_floor is False


def test_account_rent_is_unknown_without_a_balance():
    assert MTD.account_rent_status(None).verdict is MTD.Verdict.UNKNOWN
    ok = MTD.account_rent_status(10_000_000)
    assert ok.verdict is MTD.Verdict.OK
    bad = MTD.account_rent_status(1_000_000)
    assert bad.verdict is MTD.Verdict.BELOW_MINIMUM


def test_small_bankroll_audit_reports_unknowns_and_no_mutation():
    a = MTD.full_audit()
    assert a["changes_production_sizing"] is False
    assert "UNKNOWN, not guessed" in a["headline"]
    sb = a["small_bankroll"]
    assert len(sb["unknown_items"]) >= 5
    for r in sb["rows"]:
        assert r["any_unknown"] is True, r
    # nothing in the audit may claim a venue minimum it cannot verify
    for v in a["known_venue_minimums"]:
        assert v["verifiable_here"] is False


# ===========================================================================
# 7. determinism
# ===========================================================================


def test_all_audits_are_deterministic():
    assert CF.full_audit() == CF.full_audit()
    assert MTD.full_audit() == MTD.full_audit()
    r1 = TF.inspect_many(["A", "B"], TF.StubAccountReader())
    r2 = TF.inspect_many(["A", "B"], TF.StubAccountReader())
    assert [x.as_dict() for x in r1] == [x.as_dict() for x in r2]
    o = _obs()
    assert _classify(o).as_dict() == _classify(o).as_dict()


def test_repeated_runs_are_identical():
    for _ in range(3):
        assert CF.amm_half_spread_bps(10.0, 16_900.0) == pytest.approx(
            CF.amm_half_spread_bps(10.0, 16_900.0))
        assert len(CF.sensitivity_grid()) == 18
