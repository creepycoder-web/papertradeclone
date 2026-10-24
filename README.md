# AUTONOMOUS_PAPER_AGENT_V1

Autonomous Solana-memecoin **paper** trader.

> **PAPER ONLY.** There is no wallet, no private key, no RPC client, no swap
> router and no exchange order API anywhere in this package. Paper fills are
> computed locally. Real execution is not disabled by a flag — it does not
> exist. See [Safety barrier](#safety-barrier).

```
WAIT FOR ALERT → PARSE → T0 SNAPSHOT → OBSERVE MARKET → NAR LOOKUP
      → ANALYZE → BUY / WAIT / REJECT → PAPER FILL → MONITOR
      → HOLD / EXIT → RECORD → UPDATE ACCOUNT → WAIT FOR NEXT ALERT
```

---

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pytest -q          # 197 tests, ~0.3s, no network
.venv/bin/python -m autopaper.cli verify   # paper-only safety verification
.venv/bin/python -m autopaper.cli run      # start the autonomous loop
```

| command | effect |
|---|---|
| `run` | start / resume the autonomous paper loop |
| `status` | one-shot account, ledger and safety report |
| `stop` | **STOP** — no new positions; existing ones keep being managed |
| `health` | watchdog status; **exit 0 ok / 1 degraded / 2 stale** |
| `health --startup` | pre-flight ops checks, with the exact fix for each failure |
| `flatten` | **FLATTEN** — close open paper positions now |
| `resume` | clear a pending STOP/FLATTEN |
| `report` | full paper performance report, by route/regime/bucket |
| `verify` | paper-only safety verification |
| `config` | effective configuration |
| `login` | one-time Telegram user-session creation |

`stop` and `flatten` write to `var/control.json` and are picked up by the
running process, so the trader can be driven from another terminal.

**A stop is not a flatten.** Positions left open at shutdown are left open
deliberately and are resumed on the next `run`. Use `flatten` to close them.

---

## Current configuration status

`.env.local` as supplied contains:

| setting | state |
|---|---|
| `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` | present |
| `TELEGRAM_CHAT` | `MooDengPresidentCallers` |
| `TELEGRAM_ALLOW_BACKLOG` | `false` → only messages newer than boot |
| `TELEGRAM_SESSION` | **empty** → the listener is `AWAITING_LOGIN` |
| NAR source | **not configured** → every NAR-derived field is `null` |

Two consequences, stated plainly rather than papered over:

1. **Telegram cannot authenticate yet.** A user session must be created once:

   ```bash
   .venv/bin/python -m autopaper.cli login      # asks for phone + code
   # then put the printed TELEGRAM_SESSION=<string> into .env.local
   ```

   The engine keeps running and retries every 20s, so completing the login is
   picked up automatically with no restart.

2. **There is no NAR source configured.** The only configured source is the
   Telegram alert channel. `nar.NarSnapshot` therefore reports every
   narrative/security/holder/wallet-flow field as `null` with
   `available=False` and the reason. The decision engine downgrades confidence
   accordingly. **It never invents these values.** If a NAR endpoint is added
   later as `NAR_BASE_URL` / `NAR_API_KEY` in `.env.local`, the HTTP source
   activates with no code change.

---

## Safety barrier

`src/autopaper/safety.py` is imported before any other module and provides
three independent layers:

1. **Import guard** — a `sys.meta_path` finder refuses to import `solana`,
   `solders`, `base58`, `anchorpy`, `jupiter`, `raydium`, `pumpswap`, `ccxt`,
   `binance`, `okx`, `bybit`, `coinbase` and 12 more. The finder sits at
   position 0, so it beats any finder installed later.
2. **Symbol audit** — every namespace in this package is scanned for
   `sendTransaction`, `sendRawTransaction`, `sign_transaction`,
   `Keypair.from_secret_key`, `withdraw`, `place_order`, … Any hit refuses to
   arm.
3. **Startup verification** — `verify_no_real_execution()` produces a
   machine-checkable `SafetyReport` that the CLI prints and the engine requires
   before it will trade.

Environment handling is two-tiered and honest about the difference:

* **Blocking** — key material or exchange credentials (`PRIVATE_KEY`,
  `SEED_PHRASE`, `MNEMONIC`, `BINANCE_API_KEY`, …). Present → refuse to arm.
* **Reported, not blocking** — read-only RPC/indexer URLs. The shell this was
  built in exports `RPC_URL=https://api.mainnet.solana.com`. It is surfaced in
  the report and in the dashboard, but it is not a capability in this process:
  there is no RPC client, no signer, and no code path that reads it.
  `guard_url()` refuses to open one, so a future refactor cannot quietly
  acquire the capability.

The test suite audits this rather than asserting it in prose: a static scan of
the whole source tree, a walk of every public symbol in every module, and a
re-import attempt of each forbidden module. **The full suite also passes with
every outbound socket monkey-patched to raise**, which is the proof that no test
touches mainnet, Telegram, or any network.

---

## Paper execution model

Documented simulation. It is not a claim to reproduce any real historical
execution — it is a refusal to be optimistic.

```
quoted_price   = price visible at decision time
fill_price     = price visible at (decision_time + decision_latency),
                 always the WORSE of quote and current print
fill_price    += slippage:  base + size impact + staleness penalty
notional       = quantity × fill_price
fee            = notional × fee_bps
cash_out       = notional + fee
```

| parameter | value | rationale |
|---|---|---|
| `entry_fee_bps` / `exit_fee_bps` | 85 | DEX aggregator taker fee + priority |
| base `entry/exit_slippage_bps` | 120 | thin memecoin book |
| size impact | `sqrt(notional/liquidity)·90`, capped 90 bps | ~9 bps for 1% of the pool, ~45 bps for 25% |
| staleness penalty | 20 bps/s, capped 40 bps | charged when the print is not fresh |
| `decision_latency_ms` | 1200 | bot reaction + analysis; the market really moves in this window |
| unknown liquidity | assume a deliberately small $60k pool | missing data must never *reduce* cost |
| quote validity | 8000 ms | older than this the trade is **not filled** |

Direction: slippage is **added** on a buy and **subtracted** on a sell. Getting
that backwards hands the account free slippage on every exit.

Failure modes are recorded outcomes, not exceptions:

| condition | recorded as |
|---|---|
| no price at the execution timestamp | `UNFILLED / no_quote` |
| token gone | `UNFILLED / market_unavailable` |
| best print older than 8s | `UNFILLED / stale_quote` |
| exit quote unavailable | position stays **OPEN**, logged as an error |

Every one of these increments `unfilled_attempts` and appears in the ledger.
Nothing is hidden.

---

## Accounting

```
gross_pnl = (exit_reference − entry_reference) × quantity
net_pnl   = gross_pnl − entry_fee − exit_fee − entry_slippage − exit_slippage
equity    = cash + market_value(open positions)
```

`gross_pnl` uses the **pre-slippage market references**; slippage is then
deducted, which is what makes the identity above exact. Cash moves by the
**executed** prices. Those two facts are consistent by construction, and
`close_position` asserts the whole-trade invariant:

```
cash_after − cash_before_entry == net_pnl
```

That assertion caught a real double-count during development (the entry
reference price was not being stored, so `gross_pnl` used the already-slipped
fill *and* `net_pnl` subtracted the slippage again). It fires on every closed
trade, including a 300-case fuzz test.

Fees and slippage are booked **when incurred**, not when a trade closes, so an
open position's costs are visible on the dashboard.

Every hard cap is enforced against the **cash** a position consumes
(`notional × (1 + fee)`), and rounded **down** to cents. Capping the notional
while ignoring the fee lets real exposure drift above the limit on every single
trade — a bug this implementation had, and this module now prevents.

Mark-price convention: newest usable price. When a position's data is
`UNUSABLE` it is valued **at cost** and flagged `DATA_UNCERTAIN` — the equity
identity stays true without inventing a price.

---

## No-lookahead

Two gates, both mechanical:

* `ObservationSet.as_of(t)` is the only way market history is read, and it hard
  filters `observed_at <= t`.
* `ResearchMemory.visible_at(t)` is the only way memory is read, and it hard
  filters `exit_timestamp <= t`.

`AlertSnapshot` is a frozen dataclass whose `fields` and `provenance` maps are
read-only proxies, so "never backfill future information into the alert
snapshot" is a type-level guarantee.

The tests build a history containing a +750% rally and a subsequent −96% crash,
then assert the decision, regime, route, confidence and signal quality are
**bit-identical** with and without the future attached — in both directions (a
future crash must not cause a buy; a future moon must not inflate confidence).

---

## Strategy

### The anti-exhaustion ladder

The known failure mode is entering far above the first valid observed price.
`extension_from_first_valid_price` is anchored to the earliest price this agent
could actually have acted on, resolved by time: a registered anchor, else an
explicit alert-supplied first price, else whichever came first between the
alert's own price and the oldest print in our history.

Extension is a **permission ladder**, not a ban — because the biggest runners
come from exactly the momentum that a naive rule would reject:

| extension | requirement |
|---|---|
| ≤ 30% | freely tradeable |
| ≤ 60% | structurally sound: liq ≥ 0.55, activity ≥ 0.55, buy/sell ≥ 1.3 |
| ≤ 160% | very strong: liq ≥ 0.60, activity ≥ 0.60, buy/sell ≥ 2.0 |
| > 160% | refused |

An entry granted above the first rung gets **fast-failure protection and a
reduced size** regardless of how healthy the regime looks, because the thing
that makes it dangerous is the entry, not the current momentum. The confidence
bar drops accordingly (`min_confidence_to_buy_extended`), otherwise the normal
bar would cancel the ladder out and the upper rungs would be unreachable.

When no earlier price exists at all, extension is reported as `null`, **not**
zero, the entry-size band is not applied, and a `extension_unanchored` warning
is raised. A missing anchor must never score as a perfect entry.

### Regimes and routes

`VERY_EARLY · EARLY · ACCELERATING · EXPLOSIVE · EXTENDED · STABILIZING ·
WEAKENING · UNCERTAIN`

Routes: `FAST_ENTRY · PRECISION_ENTRY · CONFIRMATION_ENTRY ·
RECOVERY_RECLAIM · NO_TRADE`

A single print is `UNCERTAIN` — the momentum helpers otherwise echo the current
price back as 0%, which would hand every brand-new token a confident
`VERY_EARLY` off no evidence. The engine therefore waits `alert_settle_s`
(default 2s) for a second print before deciding. `RECOVERY_RECLAIM` is gated
behind four explicit preconditions because its history was weak; it is never
chosen merely because the pattern exists.

The momentum axis uses the longest window **actually available**, not the
longest one wished for. A token launched two minutes ago has no 5-minute
change, and a classifier that reads only fixed long windows calls every live
memecoin `UNCERTAIN`.

### Exits

Chosen from the entry regime, not fixed per account:

| style | behaviour |
|---|---|
| `FAST_FAILURE` | fragile/explosive entries: −12% stop, 75s fuse, no follow-through → out |
| `MOMENTUM_FAILURE` | −22% give-back from peak, momentum-collapse and structure-break detectors |
| `RUNNER` | no fixed target. Arms at +25%, then −18% give-back from the running peak, hard floor −8% |
| `TIME_DECAY` | weak confirmation: no progress in 240s → out, 900s cap |
| `FIXED_TARGET` | only where the thesis is explicitly a target |
| `DISASTER_CAP` | −45% simulation boundary. **Not** the strategy exit |
| `DATA_INVALID` | the market stopped telling us anything usable; exit to a known state |

Every trigger is declared in `exit_plan` **before the position exists**, and the
`exit_reason` string is the trigger name — so an after-the-fact excuse is
structurally impossible. A test sweeps every style × price × time × momentum
grid and asserts every emitted trigger is in the allowed set.

Runner mode is real: a token up +250% stays open. Stale data suppresses *new*
exits (panicking on a 20s-old print is acting on information the agent does not
have) and starts a 120s data-invalid timer that bounds the wait.

---

## Learning

`ResearchMemory` is a deterministic, readable estimator — no ML, no trained
model, no black box:

* Beta-Bernoulli shrinkage for hit rate (prior α=β=2)
* EWMA (α=0.25) of per-trade return
* shrinkage factor 0 below `MIN_SAMPLE=8`, ramping over the next 12 trades
* multiplier clamped to [0.55, 1.35]

A route with no history pulls exactly to 1.0 — a handful of trades cannot tilt
the agent. `lessons()` renders the current state, and the dashboard shows it.

---

## Data

* **DexScreener** primary — richest per-pair snapshot plus 5m/h1 buy/sell counts
* **GeckoTerminal** fallback + oldest-1m-candle anchor for first-valid-price

Rate limiting is a per-minute cap, `429` obeys `Retry-After` (defaulting to a
full 60s window when no header is sent) and is never retried immediately, and
transient failures back off exponentially. On failure the last good
observation is served **as STALE with the error attached** — never promoted to
fresh, never invented.

The agent quotes base-token venues and refuses stable/stable and stable/wrapped
pools. A memecoin quoted against USDC is kept — that is the deepest pool a new
launch has. When a mint has no usable pool, the provider says *why*
("appears only as a QUOTE token", "all base pairs are stable/stable") rather
than reporting a bare "no data".

---

## Ledger and recovery

Four append-only JSONL streams, never rewritten:

`candidates.jsonl` · `decisions.jsonl` · `ledger.jsonl` · `position_updates.jsonl`

plus `var/paper_account.json`, written atomically (temp file + `os.replace`) and
carrying account state, open positions, cooldowns, unfilled counts,
first-valid-price anchors and both dedupe sets.

### Observation capture

`var/observations/observations.jsonl` records **every** market observation the
agent received — for candidates it rejected as well as for positions it held —
captured at `MarketDataService._record`, the single point they all pass
through. That completeness is what makes the stream a valid input for replay:
a stream filtered by what the agent chose to act on would condition every
replayed decision on the live run's choices.

It is telemetry and is wired so it cannot be anything else. The capture call
runs *after* the observation is in history, inside its own guard, so a full
disk or a bug in the store costs a recorded row and never an observation the
agent needed in order to decide.

| key | default | effect |
|---|---|---|
| `OBS_ENABLED` | `true` | capture on. Off means replay has nothing to read. |
| `OBS_ROTATE_MB` | `64` | size at which the active file rotates |
| `OBS_MAX_MB` | `2000` | total disk cap for the stream, counting **compressed** rotated files |
| `OBS_FLUSH_S` | `5` | flush interval for the write buffer |
| `OBS_DISK_MARGIN_MB` | `500` | free disk required *on top of* `OBS_MAX_MB` before `health --startup` passes |

Each row carries `schema_version`, the `candidate_id` it was recorded under
(`null` for position monitoring, which has no candidate) and the observation's
own `received_at`. Rows are serialised from the `Observation` **dataclass
fields**, not from `as_dict()` — that reporting view omits `volume_24h_usd`, so
capturing through it silently dropped a field a replay would have decided on.

Rotated files are gzipped (`observations-<stamp>.jsonl.gz`, ~10x smaller), and
the cap counts the compressed size, so the same disk holds far more history.
Writes are buffered and flushed every `OBS_FLUSH_S` and on shutdown: an fsync
per observation is latency the trading loop waits behind, and losing a few
seconds of telemetry to a `kill -9` costs a replay nothing.

`load_observations(paths)` returns `{candidate_id: ObservationSet}`. A malformed
or truncated line is **skipped and counted**, never raised on — the writer can
be killed mid-row, so a truncated final line is the expected shape of a
crash-interrupted session, and a loader that raised would make every such
session unreadable.

At the cap the **oldest rotated** file is deleted first. The active file is
never deleted, and only files matching the stream's own
`observations-*.jsonl` pattern inside its own directory are ever considered —
the ledgers and `paper_account.json` live in `var/` too, and a disk-pressure
cleanup able to reach those could destroy the audit trail to save space. A test
asserts it cannot.

### Counterfactual tracking

Every gate trades false positives for false negatives, and a ledger of taken
trades can only show the first kind. So each **REJECT** and **WAIT** is followed
for `COUNTERFACTUAL_WINDOW_S` and the outcome it *would* have had is appended to
`var/counterfactual.jsonl`, against the blocking flag codes that refused it.
(The feature is "shadow tracking" in the brief; it is named *counterfactual*
throughout because `shadow` is reserved by the safety barrier for the banned
ML/shadow-inference subsystem — see `tests/test_no_ml_no_shadow.py`.) That is how
"which gate blocks winners?" becomes answerable.

The counterfactual runs through the *same* code the live path uses —
`execution.fill.simulate_fill` on both legs, `exits.ExitPolicy` for the exit,
and a real `PaperAccount` for the money. The account is a fresh private one per
candidate, so the isolation is structural: there is no object through which a
shadow's cash movement could reach the live book.

| key | default | effect |
|---|---|---|
| `COUNTERFACTUAL_ENABLED` | `true` | follow refused candidates |
| `COUNTERFACTUAL_WINDOW_S` | `1800` | how long each one is followed |
| `COUNTERFACTUAL_MAX_TRACKED` | `64` | concurrent trackings; every refusal past the cap is counted |

Three properties are enforced by tests rather than by convention:

* **No lookahead.** Every step reads `ObservationSet.as_of(t)`. A test appends a
  +400% print *after* the window and asserts the recorded row is byte-identical.
* **Nothing is written until it resolves.** An unresolved tracking lives in memory
  only; a half-finished counterfactual on disk would be indistinguishable from a
  resolved one that ended flat.
* **No path to a live decision.** The tracker holds no market service, so it
  cannot spend a request a live decision is waiting on — it is handed the
  history the monitor pass already fetched, on the same tick. It holds no
  `ResearchMemory`, so a tracked outcome cannot tilt route sizing.
  `CounterfactualTracker.needs()` states both.

On restart the engine reloads the bankroll, restores open positions with their
route/regime/exit style/MFE/MAE, rebuilds memory from **closed** trades only,
and resumes monitoring. **The bankroll is never reset** — `starting_bankroll_usd`
is written once and is what the return is measured against. A corrupt ledger
line is skipped, not fatal.

Duplicate suppression is two-layered: `(chat_id, message_id)` at the listener
and candidate identity at the engine, so a duplicate arriving through any path
still cannot open a second position.

---

## Confidence

`confidence` is a **decision-strength score** in [0, 1]. It is not a probability
of profit and is never described as one — a test greps the thesis and the full
serialised report for "probability of profit", "guaranteed", "risk-free" and
friends.

---

## Layout

```
src/autopaper/
  safety.py       import guard, symbol audit, verification report   <- imported first
  clock.py        the single time authority; FakeClock for tests
  config.py       .env.local loader; documents every default
  models.py       frozen AlertSnapshot, Observation, Position, Trade, FillResult
  logging_setup.py  tagged logging [ALERT] [BUY] [PAPER_FILL] [EXIT] …
  market/         DexScreener, GeckoTerminal, rate limiting, freshness
  nar/            NarSource; NullNarSource reports absence honestly
  telegram/       pure parser + Telethon listener with reconnect and dedupe
  analysis/       features, regime, routes, decision engine
  execution/      sizing caps, fill model
  exits.py        dynamic exit policy + runner mode
  positions.py    lifecycle, MFE/MAE, DATA_UNCERTAIN
  account.py      exact accounting + the drift assertion
  memory.py       time-correct paper memory + route adaptation
  ledger.py       append-only streams + durable state
  stats.py        performance statistics and the alert→outcome funnel
  dashboard.py    terminal status
  engine.py       the loop
  cli.py          run / status / stop / flatten / login / report / verify
tests/            197 offline tests
var/              state (gitignored)
logs/             autopaper.log (gitignored)
```

## Not implemented, deliberately

* **No re-entry** into a recently exited mint (`RiskLimits.reentry_enabled = False`)
* **No historical replay as live** — `TELEGRAM_ALLOW_BACKLOG=false`
* **No manufactured candidates** — with no alerts, the agent trades nothing
* **No live execution of any kind**


---

## Running unattended

`var/heartbeat.json` is written from the monitor tick every
`HEARTBEAT_INTERVAL_S` — from the tick itself, not from a timer of its own, so
a heartbeat cannot keep reporting OK while position management has stopped. It
is written atomically (temp file, `fsync`, `os.replace`): without the fsync a
crash can leave a zero-length file that reads as "no heartbeat", and a watchdog
would restart a healthy bot.

`autopaper.cli health` turns that into an exit code, and **the exit code is the
interface**:

| code | meaning | what a watchdog should do |
|---|---|---|
| 0 | heartbeat fresh, provider errors normal | nothing |
| 1 | heartbeat older than `HEARTBEAT_DEGRADED_S`, or provider error rate at or above `HEALTH_ERROR_RATE_DEGRADED` | warn only |
| 2 | heartbeat older than `HEARTBEAT_STALE_S`, missing or unparseable | restart |

| key | default | effect |
|---|---|---|
| `HEARTBEAT_INTERVAL_S` | `30` | how often the heartbeat is written |
| `HEARTBEAT_DEGRADED_S` | `90` | three missed beats → degraded |
| `HEARTBEAT_STALE_S` | `300` | ten missed beats → stale |
| `HEALTH_ERROR_RATE_DEGRADED` | `0.25` | provider error rate that counts as degraded |

A provider wobble is **degraded, never stale**: restarting on it would turn it
into a restart loop, and the runtime already marks data stale rather than
guessing. An absent heartbeat is reported as stale, not OK — missing is
indistinguishable from dead, so it takes the pessimistic reading.

`health --startup` checks the Telegram session and chat **key names** (never
values — this output reaches the journal), that `verify` passes, that the
run-path isolation check passes, that `var/` and `logs/` are writable, that free
disk exceeds `OBS_MAX_MB + OBS_DISK_MARGIN_MB`, and that `.env.local` is not
readable by group or others. Each failure prints its exact fix.
