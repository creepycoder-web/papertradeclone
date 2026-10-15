# PROJECT_CONTEXT.md

## Project Identity

**Name**: AUTONOMOUS_PAPER_AGENT_V1 / MEMEBOT V3 (autopaper)

**Purpose**: Autonomous Solana memecoin paper trader with optional REAL execution path that is procedurally guarded (operator-confirmed, preflight-required). The system is designed to observe Telegram alert feeds, evaluate candidates against the L25 strategy (and V2_FLOOR profile variant), and simulate/paper-trade with strict safety and accounting invariants.

**Scope**: Paper-trading simulation, market data ingestion (DexScreener primary, GeckoTerminal fallback), Telegram alert parsing/listening, strategy evaluation, position management, exits, accounting, and a hard safety barrier. Optional event-driven native monitoring via Solana WebSocket.

**Current Maturity**: Production-grade paper trading system with extensive test coverage (197+ tests), comprehensive safety guards, frozen strategy definitions, and detailed documentation. The real execution path exists as an implemented executor but is hard-disabled by default and requires explicit operator confirmation.

**What it does NOT promise**: It does not promise live trading safety without explicit operator action. It does not invent data when unavailable; it reports absence honestly. It explicitly avoids ML/Laya/shadow inference.

## Architecture and Execution Flow

### Entry Points
- **CLI** (`src/autopaper/cli.py`): `run`, `status`, `stop`, `flatten`, `resume`, `report`, `verify`, `config`, `login`. The `verify` command runs the full machine-checkable safety audit.
- **Engine** (`src/autopaper/engine.py`): Core autonomous loop that waits for alerts, parses, snapshots T0, observes market, evaluates NAR (when available), decides (BUY/WAIT/REJECT), executes paper fills, monitors positions, manages exits, records to ledger, updates account.
- **Safety barrier** (`src/autopaper/safety.py`): Imported first by `__init__.py`. Implements four independent layers (import guard, secret redaction, runtime state contract, audit/verify). Meta-path finder blocks forbidden modules at import time.

### Core Modules
- **modes.py**: Canonical `RuntimeState` (OFF/PAPER/REAL_ARMED/STOPPED/EMERGENCY_STOP), `TradeMode`, and strict REAL arming rules. REAL_ARMED is reachable only via operator-confirmed Telegram action plus passing preflight; env vars cannot auto-arm.
- **config.py**: Loads `.env.local` (overrides env overrides defaults). Documents all defaults, records `real.env_trading_enabled_request` as ignored for authorization.
- **models.py**: Frozen dataclasses (AlertSnapshot, Observation, Position, Trade, FillResult, etc.). Core data contracts.
- **clock.py**: Single time authority; `FakeClock` for deterministic tests.
- **logging_setup.py**: Tagged logging `[ALERT] [BUY] [PAPER_FILL] [EXIT] ...` with secret redaction.
- **safety.py**: FORBIDDEN_MODULES include xgboost/lightgbm/catboost/sklearn/torch/tensorflow/onnxruntime, laya/*, shadow/*. Scans for forbidden attributes and enforces startup invariants.

### Market Data Flow
- **market/base.py**: Base interfaces and freshness semantics (`MARKET_STALE_AFTER_MS`, `MARKET_UNUSABLE_AFTER_MS`).
- **market/dexscreener.py**: Primary provider with rate limiting and retry-after handling.
- **market/geckoterminal.py**: Fallback provider.
- **market/ratelimit.py**: Per-minute caps, exponential backoff, stale-but-not-invented on failure.
- **price_freshness.py**, **pricing.py**: Price normalization and execution price calculation.

### Telegram Integration
- **telegram/parser.py**: Pure alert parsing.
- **telegram/client.py**: Telethon user session listener with reconnect, dedupe (`(chat_id, message_id)`) at listener and candidate identity at engine. Backlog controlled by `TELEGRAM_ALLOW_BACKLOG`.
- **telegram/botapi.py**, **actions.py**, **notifier.py**, **ui.py**: Bot API for operator control/notifications with chat allowlist (integer compare).

### Strategy & Analysis
- **l25.py**: L25 strategy implementation and active strategy identity (`ACTIVE_STRATEGY = "L25"`).
- **v2floor.py**: V2_FLOOR profile variant with frozen thresholds and its own researched exit bracket.
- **strategy_profile.py**: Profiles (L25_CURRENT default, L25_60S, L25_MAX_HOLD, L25_TP100, V2_FLOOR, V2_FLOOR_BETA). Exit/sizing overrides scoped carefully; only V2_FLOOR overrides sizing/exit by design.
- **analysis/features.py**, **decision.py**, **routes.py**: Feature extraction, regime detection, route selection, decision engine. `confidence` is decision-strength [0,1], never presented as profit probability.

### Execution & Accounting
- **execution/fill.py**: Paper fill model with slippage (base + size impact + staleness penalty), fees, latency, unknown-liquidity conservative assumption, quote validity.
- **execution/quoting.py**, **planner.py**, **provider.py**, **adapters.py**: Quote logic, execution planning, provider abstractions, adapter interfaces.
- **execution/real.py**: Full REAL executor implementation (OnchainFill, PreflightResult, RealExecutor). Enforces pre-trade safety checks, broadcasts with intent persistence, confirms, reconciles actual amounts (separates swap leg from wallet debit). Reports AMBIGUOUS/CONFIRMED_UNRECONCILED correctly.
- **execution/real_emergency_adapter.py**, **adapters.py**, **emergency_service.py**, **emergency_close.py**: Emergency close infrastructure with adapter pattern (Paper vs Real) and strict gating.
- **exits.py**: Dynamic exit policy, runner mode, MFE/MAE tracking.
- **positions.py**: Lifecycle, MFE/MAE, DATA_UNCERTAIN handling (value at cost when unusable).
- **account.py**: Exact accounting with invariant: `cash_after − cash_before_entry == net_pnl` asserted on every closed trade. Fees/slippage booked when incurred.
- **ledger.py**: Four append-only JSONL streams (candidates, decisions, ledger, position_updates), atomic account state writes.
- **stats.py**: Performance stats and alert→outcome funnel.
- **paper_accounting.py**, **paper_live/**: Paper accounting utilities and live paper bridge.

### Native Monitoring
- **latency.py**, **price_freshness.py**: Timing/freshness utilities.
- **native_monitor_ws_config.py**, **price_layers.py**: Event-driven monitoring configuration and price layer separation (MARK ≠ NATIVE_REFERENCE ≠ EXECUTABLE_QUOTE ≠ PAPER_FILL).

### Data Persistence (gitignored)
- `var/`, `logs/`: State, logs, control.json (`stop`/`flatten`), paper_account.json, JSONL streams. `.env.local` also gitignored.

## Important Files and Directories

### Root
- `README.md`: Comprehensive project docs (safety, paper model, accounting, architecture).
- `pyproject.toml`: Python 3.11+, deps (telethon, httpx), pytest config.
- `uv.lock`: Lockfile.
- `src/autopaper/__init__.py`: Loads safety guard first, sets version 3.0.0.
- `.env.example`: Template (safe placeholders, no secrets). `.env.local`: Operator config (gitignored, may contain credentials/keys in practice).

### Source tree
- `src/autopaper/safety.py` (import guard, audits), `modes.py` (runtime state), `config.py` (env), `engine.py` (loop), `cli.py` (commands).
- `src/autopaper/execution/real.py` (real executor), `execution/adapters.py`, `emergency_service.py`.
- `src/autopaper/analysis/*`, `src/autopaper/market/*`, `src/autopaper/telegram/*`, `src/autopaper/forensics/*`, `src/autopaper/paper_live/*`.

### Tests
- `tests/`: 170+ Python test files, comprehensive coverage including safety, paper smoke, accounting, execution, strategy, etc. `tests/conftest.py` sets up test fixtures.

### Research/Data (may be tracked)
- `research/`, `reports/`, `var_alt/`, `session-*.md`, `freebuff-chat-*.md`, `paper.log`: Historical research, reports, runtime logs, sessions. Purpose varies; see audit section.

## Configuration

### Precedence
`.env.local` > process environment > built-in defaults (as implemented in config).

### Key Environment Variables (names only)
- Telegram: `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_CHAT`, `TELEGRAM_SESSION` (from login), `TELEGRAM_ALLOW_BACKLOG` (bool), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_CHAT`.
- Strategy: `STRATEGY_PROFILE` (L25_CURRENT default).
- NAR: `NAR_ENABLED`, `NAR_BOT`, `NAR_COMMAND`, `NAR_API_KEY`, `NAR_BASE_URL`, `NAR_TIMEOUT_S`.
- Market: `MARKET_POLL_INTERVAL_S`, `MARKET_OPEN_POLL_INTERVAL_S`, `MARKET_ALERT_SETTLE_S`, `MARKET_STALE_AFTER_MS`, `MARKET_UNUSABLE_AFTER_MS`, `MARKET_MAX_RPM`.
- Native monitor (optional): `NATIVE_MONITOR_WS_URL` (wss://), `NATIVE_MONITOR_ENABLED`, `NATIVE_MONITOR_STALE_AFTER_MS`, `NATIVE_MONITOR_OPEN_TIMEOUT_S`.
- Real/execution related (reported/used carefully): `WALLET_KEYPAIR_PATH`, `RPC_URL`, `I_UNDERSTAND_REAL_TRADING_RISKS`, `RAPTOR_ENABLED`, `REAL_TRADING_ENABLED` (recorded but ignored for authorization).
