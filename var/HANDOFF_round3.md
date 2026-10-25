{"gate":"R0 done","branch":"round3","head":"98de661","calls":{"R0":"9/6 (over: baseline rebuilt under the revised protocol)"},
 "baseline_failures":316,"baseline_ids":347,"flaky":"tests/test_persistence.py::test_poll_loop_survives_a_transient_error",
 "golden_hash":"not yet built","approved_behaviour_changes":[],
 "done":["R0 orientation","branch round3 created from main and pushed","baseline rebuilt per revised protocol"],
 "next":"G1 golden run - BLOCKED on the budget number and the A/B/C harness choice"}

## Measured at 98de661 (this session, not inherited)
- per-commit set: `649 passed, 4 skipped, 41 warnings in 9.64s` (matches the protocol's stated figure)
- baseline suite: `316 failed, 4653 passed, 53 skipped, 191 warnings, 31 errors in 323.57s`
- `var/base.txt`: 347 ids. Byte-identical to the round-2b baseline: `comm` both ways is empty.
- `verify` exit 0. Critical+isolation+obs+health green.

## Decisions
- Baseline rebuilt with the revised command (anchored `^(FAILED|ERROR) tests/`,
  `--continue-on-collection-errors`, no second `-q`). Old copy kept at
  var/evidence/round3/base_old.txt for the before/after record.
- Tags are NOT pushable with this credential: `git ls-remote --tags origin` is
  empty and every tag push reports "Everything up-to-date" without landing.
  Branch pushes work. Stopped retrying; tags are cosmetic here.

## Corrections to the round-3 prompt (verified against code)
- counterfactual.jsonl carries NO cost components, NO latency and NO deployment
  id. G5 cost-stress is therefore not computable on the refused side without an
  additive field; it must print "not computable: cost components missing".
- Counterfactual notional is fixed at COUNTERFACTUAL_NOTIONAL_USD = 100.0
  (counterfactual.py:90), not the live sizing policy. Gate value must difference
  mean net RETURN %, not dollars.

## Open questions (both block G1)
1. Budget is still `$___`; every gate cap scales from it.
2. Harness choice A/B/C for the golden run. My pick: A (test_paper_smoke.py
   style, green today), because test_end_to_end.py has 8 ids in the baseline
   failure set.

## Pushed commits
- (none yet this round; branch round3 pushed at 98de661)
