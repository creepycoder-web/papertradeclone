# X1 — Winner-exit diagnosis

Read-only. Branch `round4`, head `19335e3`. Every statement carries file:line.
Nothing in this gate changed code.

## Summary

Round 3's finding is **correct in its conclusion but wrong about the mechanism**,
and the real mechanism is narrower and more fixable than reported.

A winning position on the paper-live path cannot exit at a profit, for **two
independent wiring defects**, not one:

1. the operator's take-profit snapshot is taken at entry and stored on the
   `PaperPosition`, but **never copied onto the production `Position`** the exit
   policy reads;
2. `momentum_30s_pct` / `structure_30s_pct` are computed and used on the REAL
   path, but the paper-live path passes `None`.

Round 3 also reported "the live monitor passes momentum and structure as None"
as if that were the whole story. It is not: even with those wired, a pure runner
still never exits (see the fixture table). The decisive fact is defect 1.

## 1. The exit decision, traced

### 1.1 The style dispatch is pinned to FAST_FAILURE

`PaperLiveEngine._production_position` (`paper_live/engine.py:5501-5564`) builds
the `Position` handed to the exit policy. It hardcodes
`exit_style=ExitStyle.FAST_FAILURE` (`paper_live/engine.py:5539`), with the
rationale at `:5527-5529`:

> FAST_FAILURE is the exit style every L25 profile uses; the exit policy reads
> it back off the position, so it is set here rather than passed in.

`ExitPolicy.assess` (`exits.py:96`) dispatches on `position.exit_style`
(`exits.py:108`). Pinning the style to FAST_FAILURE makes three whole branches
**unreachable on the paper-live path**:

| branch | file:line | upside exits it contains |
|---|---|---|
| `ExitStyle.RUNNER` | `exits.py:290-332` | `runner_giveback` (trail -18% from peak), `runner_floor`, `max_hold`, `pre_runner_giveback` |
| `ExitStyle.MOMENTUM_FAILURE` | `exits.py:334-363` | `peak_giveback`, `wide_giveback` |
| `ExitStyle.TIME_DECAY` | `exits.py:365-384` | `time_decay_no_progress`, `peak_giveback`, `max_hold` |

The runner-arming block at `exits.py:199-213` is also dead: it is guarded by
`if style is ExitStyle.RUNNER` (`exits.py:200`), so `runner_mode` is never set
and the -18% trailing stop at `exits.py:292-299` never evaluates.

The generic `wide_giveback` backstop at `exits.py:392-399` is explicitly scoped
to `MOMENTUM_FAILURE` and `TIME_DECAY` (`exits.py:393`), so it does not apply
either.

This pinning is a **documented decision**, not a defect, and it is defensible:
L25 is a fast-failure strategy. It is recorded here because it is the reason the
rich trailing machinery in `exits.py` is invisible on the live path.

### 1.2 What the FAST_FAILURE branch actually offers (`exits.py:226-289`)

In order:

| # | trigger | file:line | fires for a winner? |
|---|---|---|---|
| 1 | take-profit | `exits.py:252-260` | only if `_take_profit_for(position)` is not None — **see 1.3** |
| 2 | `fast_failure_stop` -12% | `exits.py:262-268` | no |
| 3 | `momentum_collapse` -18%/30s | `exits.py:269-275` | **never** — `_collapse` requires `momentum_30s_pct is not None` (`exits.py:719-723`) |
| 4 | `structure_break` -14%/45s | `exits.py:276-282` | **never** — same guard, `exits.py:726-730` |
| 5 | `no_follow_through` | `exits.py:283-289` | no — guarded by `ret < 3` |

Preceding the branch: the hard-hold gate (`exits.py:188-190`, L25_MAX_HOLD only),
the data-integrity gate, the V2_FLOOR bracket (`exits.py:192-197`, V2_FLOOR only)
and the -45% disaster cap (`exits.py:215-223`).
Following it: the profile-scoped max hold (`exits.py:414-423`), which reads
`_max_hold_for(position)` — **None for L25_CURRENT** (`strategy_profile.py:200`
shows the per-profile cap table; L25_CURRENT declares no cap).

So for L25_CURRENT the surviving exits are exactly: **-45% disaster cap, -12%
fast-failure stop, data-invalid timer**. Round 3's conclusion stands.

### 1.3 The take-profit snapshot — DEFECT 1

Take-profit is a *setting*, not a strategy arm
(`paper_live/take_profit.py:1-20`). The invariant is explicit
(`paper_live/take_profit.py:21-26`): the level is a SNAPSHOT taken when the
position opens, never re-read from the global setting.

The snapshot **is** taken. `PaperPosition.take_profit_pct` exists
(`paper_live/engine.py:732`) and is populated at entry from
`self._take_profit_snapshot()` at **two** entry sites:
`paper_live/engine.py:1856` and `paper_live/engine.py:3533`.

`_take_profit_for(position)` (`exits.py:537-549`) delegates to
`for_position` (`paper_live/take_profit.py:223-245`), which reads
`getattr(position, "take_profit_pct", None)` first (`:235`).

**But `_production_position` never copies it.** The `Position(...)` construction
at `paper_live/engine.py:5531-5564` sets `strategy_profile=position.production_profile`
(`:5563`) and ~30 other fields — and **omits `take_profit_pct`**. The production
`Position` therefore has no snapshot, `for_position` falls through to its legacy
profile fallback (`paper_live/take_profit.py:241-245`), and returns
`NO_TAKE_PROFIT` for every profile except `L25_TP100`.

Net effect: **an operator who sets a take-profit gets no take-profit.** The
setting is stored, displayed, persisted and silently ignored by the exit policy.
That is a wiring defect by any reading, and it is a one-line fix.

### 1.4 momentum / structure — DEFECT 2

The signals exist and are computed: `AutoTrader._market_derivatives`
(`engine.py:7632-7648`) returns `momentum_30s_pct` (`engine.py:7645`),
`structure_pct` and `liquidity_change_pct`.

They are passed to the **same** `ExitPolicy` on the REAL path
(`engine.py:7963-7971`), whose own comment at `engine.py:7960-7962` reads:

> The SAME ExitPolicy object the PAPER book uses. `RealPosition` exposes the
> fields the policy reads, so both modes share one definition of "when to exit".

The paper-live path passes `None` at all three of its `assess` call sites:
`paper_live/engine.py:4344-4345`, `:4381-4382`, `:5184-5185`.

So the two modes do **not** share one definition of when to exit, contrary to
that comment. This is a wiring defect, not an unfinished feature: the producer
exists, the consumer exists, one caller is wired and the other is not.

Caveat for costing option A: `_market_derivatives` lives on `AutoTrader`
(`engine.py:7632`), not on `PaperLiveEngine`. Wiring it means giving the paper
engine an equivalent computed from its own observation history — more than
forwarding an existing value.

### 1.5 Which exits can fire, by scenario (L25_CURRENT, today)

| position state | exits that can fire |
|---|---|
| up 20% | none |
| up 60% | none |
| up 150% | none |
| +80% then back to +5% | none (ret 5 > 3 defeats `no_follow_through`; no trail exists) |

It must fall to **-12%** to leave the book on price.

## 2. The other profiles

| profile | same gap? | why | evidence |
|---|---|---|---|
| L25_60S | **No** | declares a 60s max hold, caught by the profile-scoped cap | `exits.py:414-423`, cap table `strategy_profile.py:195-205` |
| L25_MAX_HOLD | **No** | true hard-hold; returns before every price trigger | `exits.py:188-190`, `_hard_hold_cap_for` `exits.py:572` |
| L25_TP100 | **No** | `for_position`'s legacy fallback maps the profile to +100% | `paper_live/take_profit.py:241-245`; profile is propagated at `paper_live/engine.py:5563` |
| V2_FLOOR | **No** | own bracket TP+30/SL-30/90s, evaluated before the shared stack | `exits.py:192-197`, `_v2floor_bracket_for` `exits.py:438` |
| V2_FLOOR_BETA | **No** | three arms, each with its own bracket | `strategy_profile.py:402-426` |

**The gap is specific to L25_CURRENT — the default.** Every other profile has
some upside or time exit. Defect 1 (the dropped snapshot), however, affects
**every** profile that relies on the operator's take-profit setting rather than
on the `L25_TP100` fallback.

## 3. What the research assumed

**`reports/` does not exist in this repository** — not `reports/research/`, not
the directory at all (`ls reports/` → `No such file or directory`). Round 2b
already recorded two further absent data files (`var/audit.md:134-142`).

Per the mission's instruction, I base the classification only on code and
docstrings, and infer nothing about the missing reports' contents.

What the docstrings claim:
- `strategy_profile.py:482` (L25_TP100) lists the preserved downside triggers,
  momentum collapse -18%/30s and structure break -14%/45s among them. On the
  paper-live path **both are inert**, for defect 2.
- `engine.py:7960-7962` claims paper and real share one exit definition. They
  do not, for defect 2.

So two documented claims are false on the live paper path. That is the
correctness argument, independent of any backtest.

## 4. Classification

| defect | classification | evidence |
|---|---|---|
| TP snapshot not propagated | **wiring defect** | producer `paper_live/engine.py:1856,3533`; consumer `paper_live/take_profit.py:235`; the copy is simply absent from `paper_live/engine.py:5531-5564` |
| momentum / structure `None` | **wiring defect** | producer `engine.py:7632-7648`; consumer `exits.py:719-730`; wired on the real path `engine.py:7963-7971`, not on the paper path `paper_live/engine.py:4344,4381,5184` |
| exit_style pinned FAST_FAILURE | **documented decision** | `paper_live/engine.py:5527-5529` |

No dated rationale exists anywhere for disabling momentum/structure on the paper
path. I searched for one and found only the comment that asserts the opposite.

## 5. Deployment interaction

`_deployment_id(strategy, threshold, strategy_hash)` =
`sha256(f"{strategy}|{threshold!r}|{strategy_hash}")[:16]`
(`forward_validation.py:251-253`). `strategy_hash` is `l25.L25_HASH`
(`engine.py:954`) — the **entry** strategy's hash.

Therefore the deployment id covers **the entry strategy, the L25 threshold and
the entry strategy's source hash. It does NOT cover exit wiring, the profile, or
the take-profit setting.**

| option | new deployment id? |
|---|---|
| A | **No** — automatic id is unchanged |
| B | **No** |
| C | **No** — the profile is not an input |
| E | n/a |

Two consequences, both yours to decide:

1. **A profile switch after deployment is silent.** Setting
   `STRATEGY_PROFILE=L25_TP100` on a running server keeps deployment id
   `ed885167c45c17d9`, so pre-switch and post-switch trades land in **one**
   PROSPECTIVE_PAPER dataset under two different exit behaviours. Nothing warns.
   This is exactly the mixing `<decided>` exists to prevent, and it is reachable
   today without touching any code.
2. **A boundary already exists in this checkout** (see the R0 report):
   `var/forward_validation/deployment.json`, id `ed885167c45c17d9`, with zero
   records beside it. `load_or_create_deployment` returns an existing boundary
   **UNCHANGED, even if the arguments differ** (`forward_validation.py:267-269`,
   `:277-279`). So any exit fix this round lands behind an id minted before it.

Proposed option **F** (separate, needs your approval): make the profile and the
take-profit setting inputs to `_deployment_id`, so changing either mints a new
boundary automatically instead of silently mixing. This touches
`forward_validation.py`, which `<decided>` freezes for statuses, policy values
and dataset separation — the id composition is none of those, but it is close
enough that I will not do it unasked.

## 6. Descriptive fixture comparison

Not a performance claim. It must not be used to rank options by profit: three
hand-drawn paths are not evidence about the strategy. It shows **which exit
fires and when**, nothing more.

Run against the real `ExitPolicy` with real `ExitConfig` defaults, driving
`assess()` tick by tick at 15s intervals. Full output:
`var/evidence/round4/x1/fixture_comparison.txt`. Momentum/structure under
option A are a stand-in (change over the trailing 30s), since
`_market_derivatives` is not reachable from the paper engine.

Config defaults in force: `fast_failure_stop_pct=-12.0`,
`disaster_cap_pct=-45.0`, `momentum_collapse_drop_pct=-18.0`,
`structure_break_pct=-14.0`, `fast_failure_window_s=75.0`,
`runner_arm_pct=25.0`, `runner_trail_pct=-18.0`, `momentum_trail_pct=-22.0`.

```
=== path: rise_to_150 (golden runner shape)  rets=[5, 20, 60, 150, 150, 150] ===
  CURRENT (live path today)          NO EXIT (held to end of path)
  A: momentum+structure wired        NO EXIT (held to end of path)
  B: TP snapshot propagated(+30)     t+45s at +60% -> tp30
  C: STRATEGY_PROFILE=L25_TP100      t+60s at +150% -> tp100

=== path: pump_to_80_fade_to_5  rets=[10, 80, 40, 5, 5, 5] ===
  CURRENT (live path today)          NO EXIT (held to end of path)
  A: momentum+structure wired        t+60s at +5% -> momentum_collapse
  B: TP snapshot propagated(+30)     t+30s at +80% -> tp30
  C: STRATEGY_PROFILE=L25_TP100      NO EXIT (held to end of path)

=== path: fast_failure  rets=[-5, -13, -13] ===
  CURRENT (live path today)          t+30s at -13% -> fast_failure_stop
  A: momentum+structure wired        t+30s at -13% -> fast_failure_stop
  B: TP snapshot propagated(+30)     t+30s at -13% -> fast_failure_stop
  C: STRATEGY_PROFILE=L25_TP100      t+30s at -13% -> fast_failure_stop
```

Readings, stated carefully:
- The downside path is **identical under every option**. No option weakens the stop.
- A alone never caps a pure runner; it is downside protection, not profit-taking.
- C only acts at +100%, so it does nothing for a position that fades from +80%.
- B acts on both winning paths. The +30% level shown is illustrative, not a
  recommendation; the level is an operator setting.

## 7. Options

| option | change | files | golden hash | deployment id | risks | $ |
|---|---|---|---|---|---|---|
| **A** | pass momentum/structure on the paper path | `paper_live/engine.py` (3 call sites) + a new derivative computation | **changes** (fade paths exit earlier) | unchanged | needs an observation-history derivative the paper engine lacks; must honour `as_of(t)` or it is lookahead | ~$3.5 |
| **B** | copy `take_profit_pct` onto the production `Position` | `paper_live/engine.py:5531-5564`, one line | **unchanged** unless a TP is set; golden sets none | unchanged | none found: restores a snapshot already taken, already persisted, already read by the policy | ~$1.5 |
| **C** | operator sets `STRATEGY_PROFILE=L25_TP100` in `.env.local` | none (config only) | unchanged | unchanged — **silently**, see §5 | only acts at +100%; no fade protection; a later profile switch mixes datasets | ~$0.8 |
| **D** | A + B | both above | **changes** | unchanged | union of A's and B's | ~$4.5 |
| **E** | document the gap only | `README.md`, `docs/RUNBOOK.md` | unchanged | unchanged | the default profile keeps running with no upside exit | ~$0.5 |
| **F** | *(additional)* profile + TP into the deployment id | `forward_validation.py` | unchanged | **new id on any change** | touches a frozen-ish module; needs your approval | ~$1.5 |

**Recommendation: B, and F alongside it if you want C to be safe later.**

B is the only option justified purely by correctness: the code already takes the
snapshot, already persists it and already reads it: the copy is simply missing,
so the operator's setting is silently discarded. Fixing it makes the program do
what it already says it does, and it is the one option whose downside behaviour
is provably byte-identical.

A is also a genuine defect and I would take it next, but it is a bigger change
with a real lookahead hazard, and on its own it does not let a winner take
profit. C is a setting, not a fix — it leaves the dropped-snapshot defect in
place for every other level.
