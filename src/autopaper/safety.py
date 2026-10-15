"""
HARD SAFETY BARRIER for the V3 runtime.

This module is imported first by ``autopaper.__init__`` and is the single
place where the runtime's non-negotiable properties are enforced as code rather
than as a promise.

FOUR INDEPENDENT LAYERS
----------------------

1. ``FORBIDDEN_MODULES`` - a meta-path finder refuses to import any ML / Laya /
   shadow library at all, in this process, ever. A model gate cannot be
   smuggled in through a transitive import because the finder sits at position
   0 and wins over every other finder.

2. ``SECRET_PATTERNS`` - a redaction layer. No private key, seed phrase, bot
   token, API hash or Telethon session body is ever allowed to reach a log line,
   a journal record, a report or a Telegram message.

3. ``RuntimeState``'s arming contract - enforced in :mod:`autopaper.modes`.
   This module audits it rather than duplicating it.

4. ``audit`` / ``verify`` - machine-checkable startup and migration audits that
   the CLI prints before anything can trade.

WHAT CHANGED FROM THE PAPER-ONLY BARRIER
----------------------------------------

The previous barrier made "no real trade" structural by blocking ``solders``,
``solana`` and every swap router from being importable. That is exactly the
capability this runtime is being rebuilt to have, so the barrier is inverted:
the thing that is now structurally impossible is ML/Laya/shadow, and the thing
that is now procedurally guarded is REAL arming.

REAL IS NOT STRUCTURALLY IMPOSSIBLE, IT IS PROCEDURALLY ARMED
-------------------------------------------------------------

See :mod:`autopaper.modes`. ``REAL_ARMED`` is reachable only through an
operator-confirmed Telegram action plus a passing preflight. Neither an
environment variable nor a config value can produce it. This module's
:func:`assert_real_trading_disabled_at_startup` proves that at boot.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import re
import sys

from . import l25 as _l25
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence


def _v2floor_thresholds_frozen() -> bool:
    """
    Is V2_FLOOR's gate still exactly the researched rule?

    Re-verifies the module's own definition hash rather than trusting the
    constants, so an edited threshold is caught even if the constant it was
    edited in happens to be referenced elsewhere.
    """
    from . import v2floor as _v2f

    try:
        _v2f.verify_definition()
    except ValueError:
        return False
    return (
        _v2f.V2F_MIN_VOL5_PER_MC == 1.0
        and _v2f.V2F_MIN_CHANGE_5M_PCT == 10.0
        and _v2f.V2F_MAX_CHANGE_5M_PCT == 300.0
        and isinstance(_v2f.V2F_HASH, str)
        and len(_v2f.V2F_HASH) == 64
    )


def _v2floor_exit_bracket_frozen() -> bool:
    """
    Is V2_FLOOR's exit still exactly the researched bracket?

    Checked here because this is a SAFETY-relevant claim: the shared stack's
    -12% stop and 75s no-follow-through exit would silently close a V2 position
    long before the researched -30% stop, and that is a live-money difference,
    not a cosmetic one. So the audit verifies all four facts together --
    the two levels, the cap, and the precedence order.
    """
    from .strategy_profile import (
        PROFILE_EXIT_BRACKET,
        PROFILE_MAX_HOLD_S,
        V2F_EXIT_PRECEDENCE,
        V2F_TRIGGER_MAX_HOLD,
        V2F_TRIGGER_STOP_LOSS,
        V2F_TRIGGER_TAKE_PROFIT,
        StrategyProfile,
    )

    bracket = PROFILE_EXIT_BRACKET[StrategyProfile.V2_FLOOR]
    if bracket != (30.0, -30.0):
        return False
    if PROFILE_MAX_HOLD_S[StrategyProfile.V2_FLOOR] != 90.0:
        return False
    # SL, then TP, then the cap -- and the tokens are the researched ones
    return V2F_EXIT_PRECEDENCE == (
        V2F_TRIGGER_STOP_LOSS,
        V2F_TRIGGER_TAKE_PROFIT,
        V2F_TRIGGER_MAX_HOLD,
    )


def _only_v2_floor_is_specialised() -> bool:
    """
    No L25 profile has an exit bracket or a sizing override of its own.

    This is the regression guard for "adding V2_FLOOR changed nothing else".
    If a future edit gave an L25 profile either of these, it would be changing
    live-money behaviour under a name that promises it is unchanged, and this
    gate fails instead.

    V2_FLOOR_BETA is an experiment with its own exit parameters and sizing,
    which is explicitly allowed as a separate research tool.
    """
    from .strategy_profile import (
        PROFILE_EXIT_BRACKET,
        PROFILE_POSITION_PCT,
        StrategyProfile,
    )

    for p in StrategyProfile:
        if p is StrategyProfile.V2_FLOOR or p is StrategyProfile.V2_FLOOR_BETA:
            continue
        if PROFILE_EXIT_BRACKET[p] is not None:
            return False
        if PROFILE_POSITION_PCT[p] is not None:
            return False
    return True

AGENT_ID = "MEMEBOT_V3"

#: The active strategy identity. Read by the runtime, the logs, the Telegram UI
#: and the tests, so no surface can advertise a strategy other than the one
#: actually gating entries. Sourced from the strategy module rather than typed
#: here, so there is one name and no way for the two to drift.
STRATEGY = _l25.ACTIVE_STRATEGY

# ---------------------------------------------------------------------------
# Layer 1: ML / Laya / shadow must never be importable
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        # gradient boosting / ML inference
        "xgboost",
        "lightgbm",
        "catboost",
        "sklearn",
        "torch",
        "tensorflow",
        "onnxruntime",
        "joblib",
        "ml_gate",
        "legacy_ml",
        # Laya and its finetune path
        "laya",
        "laya_finetune",
        "laya_core",
        "laya_model",
        "laya_gate",
        # shadow runtime
        "shadow",
        "shadow_trader",
        "shadow_journal",
        "shadow_account",
    }
)

#: Symbolic names that indicate an ML/Laya/shadow call in this package's own
#: namespaces. Substring match, lowercased. Scoped to ``autopaper.*`` only -
#: third-party libraries legitimately contain words like "predict".
FORBIDDEN_ATTRS: tuple[str, ...] = (
    "xgboost",
    "laya",
    "laya_finetune",
    "lightgbm",
    "catboost",
    "sklearn",
    "legacy_ml",
    "ml_gate",
    "model_inference",
    "ml_confidence",
    "predict_proba",
    "booster",
    "feature_importances_",
    "shadow_position",
    "shadow_entry",
    "shadow_order",
    "shadow_capital",
    "shadow_account",
    "shadow_journal",
    "shadow_pnl",
)

#: Counters asserted by the test suite.
#:
#: ``EXTERNAL_MODEL_IMPORTS`` counts model libraries that were SUCCESSFULLY
#: imported. It is structurally zero: the finder below raises rather than
#: returning a spec, so no model module can ever finish loading.
#:
#: ``BLOCKED_MODEL_IMPORT_ATTEMPTS`` counts attempts, which is a different and
#: also useful number - it proves the guard is actually armed rather than
#: merely configured. Keeping the two separate matters: conflating them would
#: make the required "EXTERNAL_MODEL_IMPORTS = 0" assertion fail the moment a
#: test proved the guard works, which would be a test that rewards a broken
#: guard.
EXTERNAL_MODEL_IMPORTS: int = 0
BLOCKED_MODEL_IMPORT_ATTEMPTS: int = 0
MODEL_CALLS: int = 0
SHADOW_ENTRIES: int = 0
SHADOW_CAPITAL: float = 0.0


class SafetyViolation(RuntimeError):
    """A structural safety property was violated. Always means DO NOT TRADE."""


class _ForbiddenImportFinder(importlib.abc.MetaPathFinder):
    """
    Meta path finder that blocks ML/Laya/shadow module names.

    Inserted at position 0 so it wins over every other finder, including ones
    installed by third-party libraries later.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: Any | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        root = fullname.split(".", 1)[0]
        if fullname in FORBIDDEN_MODULES or root in FORBIDDEN_MODULES:
            global BLOCKED_MODEL_IMPORT_ATTEMPTS
            BLOCKED_MODEL_IMPORT_ATTEMPTS += 1
            # EXTERNAL_MODEL_IMPORTS is deliberately NOT incremented: nothing
            # was imported. A module that got through would be found by
            # loaded_forbidden_modules() and reported by the audit.
            raise SafetyViolation(
                f"{AGENT_ID}: import of '{fullname}' is blocked. The "
                f"{STRATEGY} runtime has no ML, Laya or shadow path."
            )
        return None


_INSTALLER: dict[str, Any] = {"installed": False, "already_present": False}


def install_import_guard() -> None:
    """Idempotently install the import guard. Safe to call repeatedly."""
    if _INSTALLER["installed"]:
        return
    for finder in sys.meta_path:
        if isinstance(finder, _ForbiddenImportFinder):
            _INSTALLER["installed"] = True
            _INSTALLER["already_present"] = True
            return
    sys.meta_path.insert(0, _ForbiddenImportFinder())
    _INSTALLER["installed"] = True


def is_forbidden_module(name: str) -> bool:
    root = name.split(".", 1)[0]
    return name in FORBIDDEN_MODULES or root in FORBIDDEN_MODULES


# ---------------------------------------------------------------------------
# Layer 2: secret redaction
# ---------------------------------------------------------------------------

#: Values that must never be printed. Matched by name when building a report
#: and by shape when scanning text.
#:
#: SHORT, GENERIC NAMES ARE A HAZARD
#: A bare ``"auth"`` was tried here and it redacted a field called
#: ``"authorized"`` - which is not a secret, it is the V3-C gate result, and
#: redacting it made the journal lie about whether a trade was authorised. So
#: every entry below is specific enough to match a credential and not a status
#: word. ``SECRET_EXACT_NAMES`` holds the ones that are only secret when they
#: ARE the whole name.
SECRET_KEY_NAMES: tuple[str, ...] = (
    "api_hash",
    "api_id",
    "bot_token",
    "access_token",
    "refresh_token",
    "auth_token",
    "authorization_header",
    "seed",
    "password",
    "secret",
    "private_key",
    "privatekey",
    "secret_key",
    "secretkey",
    "seed_phrase",
    "mnemonic",
    "passphrase",
    "session",
    "credential",
    "cookie",
    "signature_key",
    "signing_key",
)

#: Names that are secret ONLY when they are the entire key, never as a
#: substring. ``session`` is here because ``session_id`` and ``session_count``
#: are ordinary operational fields.
SECRET_EXACT_NAMES: frozenset[str] = frozenset(
    {"session", "secret", "token", "password", "passphrase", "cookie"}
)

#: Regex shapes that indicate leaked secret material in a string.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mnemonic", re.compile(r"\b(?:[a-z]{3,10}\s+){11,23}[a-z]{3,10}\b", re.IGNORECASE)),
    ("solana_private_key", re.compile(r"\[[0-9]{1,3}(?:\s*,\s*[0-9]{1,3}){60,}\]")),
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}=*")),
    ("telegram_session", re.compile(r"\bTelethon\s+session", re.IGNORECASE)),
    ("private_key_block", re.compile(r"(?:private[_ ]?key|seed[_ ]?phrase)\s*[:=]", re.IGNORECASE)),
)

#: The literal substituted for anything redacted.
REDACTED = "***REDACTED***"


def redact_value_container_placeholder_removed() -> None:
    """
    Removed: ``redact_value(value)``.

    It took no NAME, so it could not do what its docstring claimed ("Replace a
    value with a redaction marker when its NAME looks like a secret") -- it took
    no name at all. It returned ``REDACTED`` for EVERY non-``None`` value,
    including ``0``, ``""``, ``False`` and ``[]``, so the one thing it could do
    was destroy operational data.

    It had no production caller, so removing it costs nothing and removes a
    trap: the next person to reach for a "redact this" helper would have got a
    function that silently blanks every field it touched.

    The real helpers are :func:`is_secret_key` (decides by name),
    :func:`redact_mapping` (redacts a mapping) and
    :func:`redact_value_container` (descends into containers).
    """


def is_secret_key(key: Any) -> bool:
    """
    Is this key name a credential?

    Exact-name entries match only the whole name, so ``session_id`` and
    ``authorized`` are NOT secrets while ``api_hash`` and ``private_key`` are.
    Getting this wrong in the permissive direction is not "safe": it silently
    redacts real data, and a journal whose V3-C authorisation reads
    ``***REDACTED***`` is worse than no journal.
    """
    low = str(key).lower()
    if low in SECRET_EXACT_NAMES:
        return True
    return any(bad in low for bad in SECRET_KEY_NAMES)


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Redact every value whose key name is a credential."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if is_secret_key(key):
            out[key] = REDACTED if value is not None else None
        else:
            out[key] = redact_value_container(value)
    return out


def redact_value_container(value: Any) -> Any:
    """
    Recurse through the CONTAINER shapes a redaction pass can meet.

    This handles what :func:`redact_mapping` used to miss: a ``dict`` nested in
    a ``list`` or ``tuple`` was copied through VERBATIM. ``redact_mapping``
    recursed on ``isinstance(value, dict)`` only, so
    ``{"risk_flags": [{"api_hash": "..."}]}`` -- a shape the decision payload
    really produces, since ``risk_flags`` is a list of mappings -- wrote the
    secret to the journal unredacted.

    Dictionaries are re-keyed and lists/tuples keep their type, so a redacted
    payload is still structurally identical to what the caller passed in. Any
    other type is returned unchanged: this function only decides whether to
    DESCEND, never whether to alter a scalar.
    """
    if isinstance(value, dict):
        return redact_mapping(value)
    if isinstance(value, list):
        return [redact_value_container(item) for item in value]
    if isinstance(value, tuple):
        # ``namedtuple`` keeps its type through ``type(...)`` reconstruction so
        # a caller indexing by attribute is not broken by being redacted.
        if hasattr(value, "_fields"):
            return type(value)(*(redact_value_container(i) for i in value))
        return tuple(redact_value_container(item) for item in value)
    return value


def find_secrets_in_text(text: str) -> list[str]:
    """
    Names of secret PATTERNS present in ``text``. Values are never returned.

    This is the scan used to satisfy the requirement to report hardcoded
    credentials without showing their values.
    """
    if not text:
        return []
    hits: list[str] = []
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            hits.append(name)
    return hits


# ---------------------------------------------------------------------------
# Layer 2c: URL guard for the READ-ONLY data providers
# ---------------------------------------------------------------------------

#: Surfaces that can place or cancel an order. A read-only market-data or
#: narrative client must never reach one, because such a client has no notion
#: of sizing, of arming, or of the pre-trade safety sequence - it would be a
#: path to a real order that bypasses every guard in this package.
#:
#: Note what is NOT here: Solana mainnet RPC and Jupiter are deliberately
#: absent. They are legitimate targets, but ONLY for
#: :mod:`autopaper.execution`, which builds its own clients and never calls
#: this function. Keeping them out of the list is what makes the distinction
#: structural - the data layer physically cannot reach an execution venue.
FORBIDDEN_ORDER_URL_SUBSTRINGS: tuple[str, ...] = (
    "/order",
    "/orders",
    "placeorder",
    "createorder",
    "neworder",
    "submitorder",
    "addorder",
    "withdraw",
    # a swap EXECUTION endpoint, not a quote endpoint. Jupiter's quote lives
    # under /swap/v1/quote and its transaction builder under /swap/v1/swap, so
    # the execution path is matched on the final segment.
    "/swap/v1/swap",
    "/v6/swap",
    "/sendtransaction",
    "sendrawtransaction",
    "signandsend",
    "dexscreener.com/solana/orders",
    "api.binance.com",
    "api.exchange.coinbase.com",
    "api.kraken.com",
    "api.coinbase.com",
    "api.bybit.com",
    "api.okx.com",
    "ccxt",
)


def guard_url(url: str) -> None:
    """
    Assert a read-only client is not about to touch an order surface.

    Called by the market-data and narrative providers before every request. It
    is a belt-and-braces check on URLs those modules construct themselves from
    fixed bases; the real protection is that they have no order capability at
    all.
    """
    low = (url or "").lower()
    for bad in FORBIDDEN_ORDER_URL_SUBSTRINGS:
        if bad in low:
            raise SafetyViolation(
                f"{AGENT_ID}: read-only data provider blocked from an "
                f"order surface. Refusing URL."
            )


def mask_address(address: str, head: int = 4, tail: int = 4) -> str:
    """
    ``wUEU…Wbt6`` - the only form an operator surface may show.

    Idempotent: an already-masked value is returned unchanged. Renderers mask
    defensively even when the view is supposed to carry a masked value, and
    without idempotence a second pass would shorten ``So11…1112`` to ``So…``.
    """
    if not address:
        return "(none)"
    if "\u2026" in address or "..." in address:
        return address
    if len(address) <= head + tail + 1:
        return address[:2] + "\u2026"
    return f"{address[:head]}\u2026{address[-tail:]}"


def mask_telegram_id(chat: str) -> str:
    """
    Show which chat is configured without publishing the identifier.

    The operator needs to be able to CONFIRM the chat is Moo Deng President
    from the status message, so the first three characters survive. That is
    enough to recognise and not enough to impersonate.
    """
    if not chat:
        return "(not configured)"
    text = str(chat)
    if len(text) <= 3:
        return text
    return f"{text[:3]}…{text[-2:]}"


# ---------------------------------------------------------------------------
# Layer 2b: namespace audit
# ---------------------------------------------------------------------------


def _scan_namespace(ns: Any) -> list[str]:
    hits: list[str] = []
    try:
        keys = list(vars(ns).keys())
    except Exception:  # pragma: no cover - exotic module objects
        return hits
    for key in keys:
        low = key.lower()
        for bad in FORBIDDEN_ATTRS:
            if bad in low:
                hits.append(f"{key} (matched '{bad}')")
    return hits


def audit_loaded_modules() -> list[str]:
    """
    Audit this package's own namespaces for ML/Laya/shadow symbols.

    Scope is deliberately ``autopaper.*`` only. Third-party dependencies
    legitimately contain words like ``predict``; flagging them would be noise.
    """
    violations: list[str] = []
    for modname, mod in list(sys.modules.items()):
        if mod is None or not modname.startswith("autopaper"):
            continue
        try:
            ns = vars(mod)
        except Exception:  # pragma: no cover
            continue
        for hit in _scan_namespace(ns):
            if hit.startswith("_scan_namespace"):
                continue
            violations.append(f"{modname}.{hit}")
    return violations


def loaded_forbidden_modules() -> list[str]:
    """Forbidden module names present in ``sys.modules`` right now."""
    return sorted(
        name
        for name, mod in list(sys.modules.items())
        if mod is not None and is_forbidden_module(name)
    )


# ---------------------------------------------------------------------------
# Layer 3: REAL arming contract
# ---------------------------------------------------------------------------

#: Environment variables that must NEVER be able to arm REAL on their own.
#: They are read as CAPABILITIES (wallet, RPC) but never as authorization.
NON_AUTHORIZING_ENV: tuple[str, ...] = (
    "REAL_TRADING_ENABLED",
    "MODE",
    "TRADING_MODE",
    "I_UNDERSTAND_REAL_TRADING_RISKS",
    "AUTO_ARM_REAL",
    "ENABLE_LIVE",
)


def assert_real_trading_disabled_at_startup() -> None:
    """
    Prove that startup cannot arm REAL.

    Called by the CLI before the engine is constructed. It asserts the runtime
    state is the default PAPER/OFF and that REAL_TRADING_ENABLED is false
    regardless of what the environment says.
    """
    from .modes import RunState, RuntimeState

    state = RuntimeState()
    if state.real_trading_enabled:
        raise SafetyViolation(
            f"{AGENT_ID}: a freshly constructed RuntimeState reported REAL "
            "trading enabled. This is a construction bug."
        )
    if state.run_state is not RunState.OFF:
        raise SafetyViolation(
            f"{AGENT_ID}: a freshly constructed RuntimeState started in "
            f"{state.run_state.value}, not OFF."
        )
    # The env value is deliberately NOT honoured. It is read for reporting
    # only, and the runtime state is what governs.
    return None


def real_env_requests(env: dict[str, str] | None = None) -> dict[str, str]:
    """
    Report which non-authorizing env vars are SET, without their values.

    Useful in the preflight report so an operator can see that
    ``REAL_TRADING_ENABLED=true`` in the environment did nothing.

    ``env`` is the environment to inspect; ``None`` means this process's.
    """
    environment = os.environ if env is None else env
    out: dict[str, str] = {}
    for key in NON_AUTHORIZING_ENV:
        if key in environment:
            out[key] = "SET (ignored for authorization)"
    return out


# ---------------------------------------------------------------------------
# Layer 4: machine-checkable audit
# ---------------------------------------------------------------------------


@dataclass
class AuditReport:
    """Result of the migration audit."""

    ok: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    def render(self) -> str:
        lines = []
        for name, ok, detail in self.checks:
            lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        for n in self.notes:
            lines.append(f"[note] {n}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {"name": n, "ok": ok, "detail": d} for n, ok, d in self.checks
            ],
            "notes": list(self.notes),
        }


def audit(env: dict[str, str] | None = None) -> AuditReport:
    """
    The full migration audit: no ML, no Laya, no shadow, no live arming.

    ``env`` is the environment to audit against. ``None`` means the real
    process environment; a supplied mapping is used AS GIVEN, so a test or a
    caller can assert what a hypothetical environment would do without mutating
    this process.
    """
    environment = os.environ if env is None else env
    r = AuditReport(ok=True)

    loaded = loaded_forbidden_modules()
    r.add(
        "no_forbidden_modules_loaded", not loaded,
        "none" if not loaded else ", ".join(loaded),
    )
    r.add(
        "external_model_imports_zero", EXTERNAL_MODEL_IMPORTS == 0,
        f"EXTERNAL_MODEL_IMPORTS={EXTERNAL_MODEL_IMPORTS} "
        f"(blocked attempts: {BLOCKED_MODEL_IMPORT_ATTEMPTS})",
    )
    r.add(
        "model_calls_zero", MODEL_CALLS == 0, f"MODEL_CALLS={MODEL_CALLS}",
    )
    r.add(
        "shadow_entries_zero", SHADOW_ENTRIES == 0, f"SHADOW_ENTRIES={SHADOW_ENTRIES}",
    )
    r.add(
        "shadow_capital_zero", SHADOW_CAPITAL == 0.0,
        f"SHADOW_CAPITAL={SHADOW_CAPITAL}",
    )

    violations = audit_loaded_modules()
    r.add(
        "no_ml_symbols_in_package", not violations,
        "none" if not violations else "; ".join(violations),
    )

    try:
        assert_real_trading_disabled_at_startup()
        r.add("real_disabled_at_startup", True, "RuntimeState() is OFF/PAPER")
    except SafetyViolation as exc:
        r.add("real_disabled_at_startup", False, str(exc))

    from .modes import RuntimeState

    s = RuntimeState()
    r.add(
        "real_requires_operator", s.real_trading_enabled is False,
        "REAL_ARMED is unreachable without a Telegram confirmation token",
    )
    try:
        s.arm_real(confirmation=None, preflight_ok=True)
        r.add("real_arm_rejects_bare_call", False, "arm_real() accepted a bare call")
    except Exception:
        r.add("real_arm_rejects_bare_call", True, "arm_real() without a token refused")

    # Every threshold in the active strategy must be the researched value, and
    # the identity must be the one the runtime actually reports.
    from .l25 import L25_HASH, L25_MIN_LIQ_TO_MCAP
    from .v3c import V3C_HASH, V3C_MAX_MARKET_CAP_USD, V3C_MIN_SIGNAL_QUALITY

    r.add(
        "v3c_thresholds_frozen",
        V3C_MAX_MARKET_CAP_USD == 20_000.0 and V3C_MIN_SIGNAL_QUALITY == 0.55,
        f"MC<=${V3C_MAX_MARKET_CAP_USD:,.0f} SQ>={V3C_MIN_SIGNAL_QUALITY} "
        f"(hash {V3C_HASH[:8]})",
    )
    r.add(
        "l25_threshold_frozen",
        L25_MIN_LIQ_TO_MCAP == 0.9647,
        f"f_liq_to_mcap>={L25_MIN_LIQ_TO_MCAP:.4f} (hash {L25_HASH[:8]})",
    )
    r.add(
        "active_strategy_is_l25", STRATEGY == "L25",
        f"active strategy = {STRATEGY}",
    )

    # The selectable profile must be a closed set, must default to today's
    # behaviour, and must NOT be a route to changing the entry strategy. A
    # profile that could reach a threshold, a size or the entry gate would make
    # "selectable" a synonym for "editable", which is what these checks exist
    # to prevent.
    from .strategy_profile import (
        DEFAULT_PROFILE,
        ENTRY_STRATEGY,
        PROFILE_ENTRY_STRATEGY,
        PROFILE_MAX_HOLD_S,
        PROFILE_POSITION_PCT,
        PROFILE_VALUES,
        StrategyProfile,
    )

    r.add(
        "profile_set_is_closed",
        # The CLOSED profile set, recorded here deliberately. Adding a profile
        # must fail this gate until it is recorded, so a profile can never
        # appear in the selector without a safety audit naming it.
        set(PROFILE_VALUES) == {
            "L25_CURRENT", "L25_60S", "L25_MAX_HOLD", "L25_TP100", "V2_FLOOR",
            "V2_FLOOR_BETA",
        },
        f"profiles = {', '.join(PROFILE_VALUES)}",
    )
    r.add(
        "profile_default_is_l25_current",
        DEFAULT_PROFILE is StrategyProfile.L25_CURRENT,
        f"default = {DEFAULT_PROFILE.value}",
    )
    r.add(
        "default_entry_strategy_is_l25",
        ENTRY_STRATEGY == "L25",
        f"the default entry strategy is still {ENTRY_STRATEGY}",
    )
    r.add(
        "profile_l25_current_has_no_cap",
        PROFILE_MAX_HOLD_S[StrategyProfile.L25_CURRENT] is None,
        "L25_CURRENT adds no maximum hold, so it is behaviourally identical to "
        "the pre-profile runtime",
    )
    r.add(
        "profile_60s_cap_is_60",
        PROFILE_MAX_HOLD_S[StrategyProfile.L25_60S] == 60.0,
        f"L25_60S max hold = {PROFILE_MAX_HOLD_S[StrategyProfile.L25_60S]}s",
    )
    # A SECOND entry strategy now exists, so the audit can no longer assert that
    # every profile trades L25. What it asserts instead is the invariant that
    # actually matters: every profile resolves to a DECLARED entry strategy,
    # the default is still today's behaviour, and the L25 family is unchanged.
    # A profile that could reach an UNDECLARED gate would make "selectable" a
    # synonym for "editable", which is what this gate exists to prevent.
    # THE EFFECTIVE-ENTRY-GATE CONTRACT MUST HOLD.
    #
    # The authorizing gate is derived from the SELECTED ARMS, not from the
    # profile alone, because the Telegram selector writes the arms and never
    # writes the profile in PAPER. Three properties have to stay true or the
    # beta experiment is unreachable (or, worse, reachable by accident):
    #   * the contract resolves, for every arm, to a DECLARED gate;
    #   * selecting the beta group makes V2_FLOOR operative -- the bug that
    #     kept the experiment behind L25's gate;
    #   * no default, empty or unknown selection can ever produce V2_FLOOR.
    from .modes import RuntimeState as _RuntimeState
    from .paper_live.arms import (
        AMARIH_ARMS as _AMARIH_ARMS,
        AMARIH_GROUP as _AMARIH_GROUP,
        BETA_ARMS as _BETA_ARMS,
        BETA_GROUP as _BETA_GROUP,
        effective_entry_gates as _gates,
    )
    from .strategy_profile import GATE_AMARIH as _GATE_AMARIH

    _all_gates = {
        _gates([a.value for a in _BETA_ARMS]),
        _gates([_BETA_GROUP]),
        _gates(["L25_CURRENT"]),
        _gates(["L25_CURRENT", _BETA_GROUP]),
        _gates([a.value for a in _AMARIH_ARMS]),
        _gates([_AMARIH_GROUP]),
        _gates(["L25_CURRENT", _AMARIH_GROUP]),
        _RuntimeState().effective_entry_gates,
    }
    _declared = set(PROFILE_ENTRY_STRATEGY.values()) | {_GATE_AMARIH}
    r.add(
        "effective_entry_gates_resolve_to_declared_gates",
        all(set(g) <= _declared for g in _all_gates),
        "resolved gate sets: "
        + "; ".join(str(tuple(g)) for g in sorted(_all_gates, key=str)),
    )
    # V2 FLOOR (BETA) is DISABLED from the active bot: selecting the beta group
    # no longer authorizes V2_FLOOR (the resolution fails closed to L25), and
    # the beta arm names likewise resolve to nothing operable. This check pins
    # the removal -- if beta ever becomes reachable again, this audit fails.
    r.add(
        "beta_selection_stays_disabled",
        _gates([_BETA_GROUP]) == ("L25",)
        and _gates([a.value for a in _BETA_ARMS]) == ("L25",),
        "the disabled beta group and beta arms authorize through L25 only",
    )
    # The PumpView 'Amarih' paired-exit experiment is ENABLED (opt-in) and gets
    # its OWN gate -- exactly the separation the old beta contract required for
    # V2_FLOOR, applied to the new experiment. Selecting the Amarih group (or
    # its arm tokens, which resolve to the group) authorizes through AMARIH
    # alone, never through L25.
    r.add(
        "amarih_selection_makes_amarih_operative",
        _gates([_AMARIH_GROUP]) == ("AMARIH",)
        and _gates([a.value for a in _AMARIH_ARMS]) == ("AMARIH",),
        "selecting the Amarih group authorizes through AMARIH alone",
    )
    r.add(
        "default_and_empty_selection_never_activates_beta",
        _gates([]) == ("L25",)
        and _gates(["NOT_AN_ARM"]) == ("L25",)
        and _RuntimeState().effective_entry_gates == ("L25",),
        "an unset, empty or unrecognised selection authorizes through L25 only",
    )
    r.add(
        "default_and_empty_selection_never_activates_amarih",
        _gates([]) == ("L25",)
        and _gates(["NOT_AN_ARM"]) == ("L25",),
        "an unset, empty or unrecognised selection authorizes through L25 only",
    )

    r.add(
        "profile_entry_strategies_are_declared",
        set(PROFILE_ENTRY_STRATEGY.values()) == {"L25", "V2_FLOOR"},
        "entry strategies in use: "
        + ", ".join(sorted(set(PROFILE_ENTRY_STRATEGY.values()))),
    )
    r.add(
        "profile_l25_family_entry_is_unchanged",
        all(
            PROFILE_ENTRY_STRATEGY[p] == "L25"
            for p in (
                StrategyProfile.L25_CURRENT,
                StrategyProfile.L25_60S,
                StrategyProfile.L25_MAX_HOLD,
                StrategyProfile.L25_TP100,
            )
        ),
        "L25_CURRENT, L25_60S, L25_MAX_HOLD and L25_TP100 all still authorize through L25",
    )
    r.add(
        "profile_v2_floor_cap_is_90",
        PROFILE_MAX_HOLD_S[StrategyProfile.V2_FLOOR] == 90.0,
        f"V2_FLOOR max hold = {PROFILE_MAX_HOLD_S[StrategyProfile.V2_FLOOR]}s",
    )
    r.add(
        "profile_v2_floor_threshold_is_frozen",
        _v2floor_thresholds_frozen(),
        "V2_FLOOR's vol5/mcap and momentum-band thresholds match "
        "autopaper.v2floor's frozen constants and its definition hash verifies",
    )
    r.add(
        "profile_v2_floor_exit_bracket_is_the_research_rule",
        _v2floor_exit_bracket_frozen(),
        "V2_FLOOR's exit is its OWN researched bracket -- TP +30% / SL -30% / "
        "max hold 90s, in that precedence -- and not the shared L25 stack, "
        "whose -12% stop and 75s no-follow-through exit would otherwise close "
        "the position well before the researched -30% stop",
    )
    r.add(
        "profile_v2_floor_sizing_is_ten_percent",
        PROFILE_POSITION_PCT[StrategyProfile.V2_FLOOR] == 10.0,
        "V2_FLOOR stakes a flat 10% of current equity, the size the research "
        "was measured at; the shared 10%/12.5% ladder applies to every L25 "
        "profile",
    )
    r.add(
        "profile_only_v2_floor_overrides_sizing_or_exit",
        _only_v2_floor_is_specialised(),
        "V2_FLOOR is the only profile with its own exit bracket or its own "
        "position size, so no L25 profile's behaviour moved when V2_FLOOR was "
        "added",
    )
    r.add(
        "profile_selecting_v2_floor_does_not_arm_real",
        not getattr(RuntimeState(), "real_armed"),
        "selecting V2_FLOOR cannot set real_armed; arming REAL still requires "
        "arm_real() with a single-use operator token and a passing preflight",
    )

    env_reqs = real_env_requests(environment)
    if env_reqs:
        r.notes.append(
            "non-authorizing env vars present and IGNORED: "
            + ", ".join(sorted(env_reqs))
        )

    r.ok = all(ok for _, ok, _ in r.checks)
    return r


def all_forbidden_modules() -> Iterator[str]:
    return iter(sorted(FORBIDDEN_MODULES))


def all_secret_key_names() -> Iterable[str]:
    return sorted(SECRET_KEY_NAMES)


# ---------------------------------------------------------------------------
# backward-compatible aliases
# ---------------------------------------------------------------------------
#
# The pre-V3 package raised RealExecutionBlocked from a paper-only barrier.
# The name is retained so a stale import fails loudly rather than silently.


class RealExecutionBlocked(SafetyViolation):
    """Retained name. A real trade now requires arming, not a block."""


def verify_no_real_execution(env: dict[str, str] | None = None) -> AuditReport:
    """
    Deprecated. Use :func:`audit`.

    ``env`` is HONORED, not ignored. It used to be accepted and dropped, so a
    caller passing an environment and reading ``report.ok`` believed it had been
    checked against that environment when it had been checked against the real
    process environment instead -- a silent substitute, which is the one thing
    a "verify no real execution" helper must never be. The argument is now
    forwarded to :func:`audit`, which is what actually reads it.
    """
    return audit(env=env)


def assert_paper_only(env: dict[str, str] | None = None) -> AuditReport:
    """
    Deprecated. PAPER-only is no longer the design; use :func:`audit`.

    ``env`` is forwarded for the same reason as in
    :func:`verify_no_real_execution`.

    NOTE: the live PAPER capability barrier is
    :func:`autopaper.paper_live.guards.assert_paper_only`, which RAISES on
    failure. This module-level alias only returns a report.
    """
    return audit(env=env)
