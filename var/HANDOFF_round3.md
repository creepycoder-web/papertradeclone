{"gate":"G1 done","branch":"round3","head":"7fba210",
 "calls":{"R0":"9/6","G1":"19/15"},
 "baseline_failures":316,"baseline_ids":347,
 "flaky":"tests/test_persistence.py::test_poll_loop_survives_a_transient_error",
 "golden_hash":"c7763e338fe1643258ca124119e129cd19c9555a2f03dce5c573a6b3f6fc297f",
 "approved_behaviour_changes":[],
 "done":["R0 orientation","branch round3 pushed","baseline rebuilt","G1 golden-run fingerprint"],
 "next":"G2 unattended ops (D5a log rotation -> D5b service -> D5c health timer -> D5d backup/restore). BLOCKED: budget number still unstated."}

## Measured this session at 98de661 / 7fba210
- per-commit set + golden: `652 passed, 4 skipped, 1 deselected, 45 warnings in 34.03s`
- golden, fresh process, incl. slow: `4 passed in 73.73s`
- golden, default (-m "not slow"): `3 passed, 1 deselected in 25.91s`
- baseline: `316 failed, 4653 passed, 53 skipped, 191 warnings, 31 errors in 323.57s`
- var/base.txt 347 ids, byte-identical to round 2b (comm empty both ways)

## Decisions
- Golden payload is an explicit allowlist projection; ids are uuid4 so whole
  records cannot be hashed. FakeClock times kept, paths/ids/pids excluded.
- Vanished-market scenario runs with provider retries off: each failing quote
  cost a real backoff sleep (93s of an initial 106s). The scenario is about how
  the ENGINE values absent data, not the provider retry count.
- Payload memoised per module; the two-independent-runs determinism check is
  marked `slow` because it cannot use the memo.
- Golden runtime is 26s, NOT the specified <5s. Reported, not hidden.

## Findings (recorded, not fixed)
- Under L25_CURRENT a WINNING position has no price exit: take-profit is an
  unset per-position snapshot, the profile declares no max hold, and the live
  monitor passes momentum/structure as None. Only the -45% disaster cap, the
  -12% stop and the data-invalid timer remain. Fingerprinted as-is.
- The forward-validation boundary guard fired unprompted during the golden run,
  refusing FakeClock-stamped records as prospective. Hard rule 7 holding on its
  own. Note: `_run` sets trader.forward = None, yet some path still builds a
  recorder - it REFUSES correctly, so behaviour is right; worth a look later.
- counterfactual.jsonl has no cost components, no latency, no deployment id;
  notional fixed at $100 (counterfactual.py:90). Limits G5 as reported.
- Tags are not pushable with this credential (`ls-remote --tags` empty; pushes
  report "Everything up-to-date" without landing). Branches push fine.

## Open question
Budget is still `$___`. Gate caps scale from it; R0 and G1 both ran over the
$20-reference caps (9/6 and 19/15) because of the protocol revision and the
runtime work.

## Pushed commits
- e343f30 https://github.com/creepycoder-web/papertradeclone/commit/e343f30
- 7fba210 https://github.com/creepycoder-web/papertradeclone/commit/7fba210
