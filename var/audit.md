# G1 audit — real-execution reachability

Branch `round2`, head `e30c5e9`, tag `round2b-start`. Read-only; nothing
implemented. File:line evidence only, no inference presented as fact.

## Q1. What does `preflight` check and import?

*Confidence: high* — read directly from the function body.

`cli.py:377-419`. Lazily imports the real executor at `cli.py:389`
(`from .execution.real import RealExecutor`), constructs it at `cli.py:398`
and awaits `ex.preflight()`.

Refuses early when `cfg.real.missing_capabilities()` is non-empty
(`cli.py:391-396`, exit 2). Its own docstring (`cli.py:380-386`) states the
checks: the wallet loads, **signing works offline against a throwaway
instruction**, the RPC answers, the quote provider is reachable, the sizing
policy is verified as a pure function. Nothing is broadcast and it does not arm
REAL (`cli.py:416-418`).

So it is a genuine capability probe that **loads a key and signs**. Per
`<decided>` it is left untouched and never run.

## Q2. Where and why are the 5 missing packages used?

*Confidence: high* for solders; *high* for the four research packages (grep found no importer under src/).

**`solders` — INSIDE `src/autopaper`, imported lazily inside functions:**

| site | what |
|---|---|
| `execution/solana.py:327-330` | `from solders.keypair import Keypair` — loads a Solana keypair from file; raises "solders is not installed" |
| `execution/solana.py:346` | `from solders.pubkey import Pubkey` |
| `execution/solana.py:361-363` | `Hash`, `Message`, `Pubkey` — message construction for signing |
| `execution/real.py:554, 733` | `from solders.transaction import VersionedTransaction` |
| `execution/real.py:824-825` | `Pubkey`, `TransferParams`, `transfer` — a system transfer |

Every one is inside a function body, which is why the whole module tree imports
cleanly with `solders` absent. Its absence is an **environment** fact, not a
code guarantee.

**`l25x`, `savannah`, `randorder`, `r2search` — OUTSIDE `src/autopaper`.**
Referenced only from tests (`tests/test_l25x_*.py`, `tests/research/test_savannah*`,
`tests/test_l25.py:504` and `:558`). They are external research packages the
suite cross-checks the frozen strategy against. No module under
`src/autopaper` imports any of them.

## Q3. Can anything reachable from `run` build, sign or send a transaction, load a key, or call an order API?

*Confidence: high* on the import chain (module-level, read directly). *Medium* on "no env/config value alone reaches sign/send": the arming rules are asserted by verify, but I traced callers rather than proving unreachability exhaustively.

**YES — the code is imported unconditionally; calling it is gated procedurally, not structurally.**

Import chain, all module-level:

```
cli.cmd_run (cli.py:87)  ->  _trader(cfg)  ->  autopaper.engine
engine.py:90   from .execution.real import LiveSolanaBackend, RealExecutor
real.py:63     from .solana import (...)        # keypair loader, signer
engine.py:130  from .real_account import RealAccount
```

So `run` loads the full real-execution module tree into the process, including
the keypair loader (`execution/solana.py:327`) and the transaction builders
(`execution/real.py:554,733,824`).

What stops it being *called*:

* the executor is built lazily — `engine.py:1515-1527`, whose own docstring says
  "built on first use so no key is touched early";
* `real_capable()` requires wallet **and** RPC configured (`engine.py:1532-1535`);
* `observe_real_wallet()` (`engine.py:1299-1315`), the first thing that actually
  touches `real_backend.wallet_state()`, sits on the REAL BUY path and its
  docstring says that path is dead without an observation;
* reaching `REAL_ARMED` needs an operator-confirmed Telegram action plus a
  passing preflight; `safety.py:598-601` and `:713` ("real_requires_operator")
  assert that neither an env var nor a config value can produce it.

So **no config or env value alone** reaches build/sign/send. But two facts
remain, and both are honest readings rather than reassurance:

1. the capability is in-process and importable from `run`;
2. the only thing that would make a lazy `from solders...` fail **today** is
   that solders is not installed here. That is not a property of the code.

Under the mission's pessimistic-reading rule, Q3 is answered YES.

## Q4. Why does `verify` pass if that code exists?

*Confidence: high* — safety.py states it verbatim.

Because the barrier was **deliberately inverted**. `safety.py:29-36`, verbatim:

> The previous barrier made "no real trade" structural by blocking `solders`,
> `solana` and every swap router from being importable. That is exactly the
> capability this runtime is being rebuilt to have, so the barrier is inverted:
> the thing that is now structurally impossible is ML/Laya/shadow, and the thing
> that is now procedurally guarded is REAL arming.

`FORBIDDEN_MODULES` (`safety.py:152-175`) therefore lists only ML, Laya and
shadow names — **no `solana`, no `solders`, no swap router**. `verify` audits
the *arming rules* (`safety.py:713`), not the absence of real-execution code, so
it passes exactly as designed.

**The README is stale and contradicts this.** It still claims "There is no
wallet, no private key, no RPC client, no swap router and no exchange order API
anywhere in this package" and "Real execution is not disabled by a flag — it
does not exist." `PROJECT_CONTEXT.md` is the accurate document. Correcting the
README is a docs change for G6, not a code change.

## Q5. Does counterfactual.py honour the intent of test_no_ml_no_shadow.py?

*Confidence: medium* — a judgement about intent, not a fact.

The banned concept is a **second trading book** — shadow entries, shadow
capital, a shadow journal writer (`tests/test_no_ml_no_shadow.py:65-68,186-217`).
`counterfactual.py:346-352` does open a private `PaperAccount` per refused
candidate, which is book-shaped, so the name alone is not the answer.

It honours the intent in substance: the tracker holds no market service and no
`ResearchMemory` (`counterfactual.py` `needs()`), writes only to its own
`counterfactual.jsonl`, and touches neither the live ledger nor equity. But
"book-shaped and separate by construction" deserves an explicit test rather
than a reviewer's reading — proposed as part of G3a.

## Verdict

Q3 = YES. Per `<g1_audit>`, stopping with containment options and implementing
nothing.


---

## G2 addendum: additional missing packages

The context named five. Triage found **two more**, both outside `src/autopaper`
and both test-only: `l25r` (`tests/research/test_l25c_isolation.py`,
`tests/research/test_l25r_isolation.py`) and `research`
(`tests/test_l25_accounting_telemetry.py`). Three absent data files as well:
`reports/production_readiness.json`, `research/baseline/v1_manifest.json`, and
an absolute path from the original author's machine,
`/Users/lnwper/AI-paper-trade` (`tests/test_no_ml_no_shadow.py`).

## G2 addendum: one test reaches the network

`tests/test_paper_visibility_live_smoke.py` fails with
`httpcore.ProxyError: 403 Forbidden` — it attempts a real outbound HTTP call in
the default suite. Under the offline-tests rule that is a defect in its own
right, not just a failure. Pre-existing at `b508e2e`. Reported, not touched.
