# AGENTS.md

## Purpose

Persistent instructions for AI coding agents working in this repository. These rules supplement but do not override the safety rules in the task description.

## Core Principles

1. **Read first**: Before making substantial changes, read `README.md` and `PROJECT_CONTEXT.md`. Understand the actual implementation as the source of truth.
2. **Preserve existing work**: Do not rebuild, replace architecture, or make unrelated changes. The working tree may contain untracked and modified files - never discard them.
3. **Safety first**: This is a trading project with strict safety barriers. Real trading is procedurally guarded and requires explicit operator confirmation. Never bypass or weaken safety checks.

## Git Rules

- Inspect current Git state (`git status`, branch, remotes) before editing.
- Never run `git add .`, `git add -A`, `git clean`, `git reset --hard`, `git checkout -- .`, `git restore .`, mass deletion, stash, rebase, amend, history rewrite, branch changes, or auto-untrack files.
- Never commit or push. Stage only explicitly approved files. Report exact commands for review.
- Do not modify `.gitignore` automatically to resolve tracked-secrets; report findings instead.

## Security Rules

- Never read, overwrite, regenerate, delete, rename, or auto-modify `.env`, `.env.local`, credential files, private keys, wallet files, or secret-bearing config.
- Never print secret values in output, reports, or generated markdown.
- Check for secrets using safe, redacting methods. If a tracked file contains a real secret, flag as a release blocker - do not reproduce it.
- Generate `.env.example` only from verified variable names with fake placeholders; never copy real env files.
- Respect that `.gitignore` does not untrack already-tracked files.

## Trading Safety

- Treat as paper-trading/research unless implementation proves otherwise.
- Never execute real trades, sign/broadcast transactions, access wallet private keys, or activate live trading. The system's REAL mode requires explicit operator confirmation via Telegram plus passing preflight; you cannot auto-arm it.
- Do not run commands that trigger trading loops, order execution, or external financial operations without verifying side effects.
- Verify the safety verification passes (`python -m autopaper.cli verify`) conceptually - but never run trading operations.
- Note: env var `REAL_TRADING_ENABLED=true` is recorded but IGNORED for authorization (enforced in config/modes/engine).

## Development Rules

- **Minimal, focused changes**: Make the smallest change needed. Do not refactor unrelated subsystems to "clean up".
- **Source of truth**: If documentation conflicts with implementation, record the discrepancy. Prefer code inspection over assumptions.
- **Tests**: Add/update tests for behavioral changes where practical. Never fake test results or silently skip failures. When running tests, only run safe offline tests and report as passed/failed/skipped (never claim unexecuted pass).
- **Verify before documenting**: Every command, path, flag, or env var documented must be verified against actual files.
- **Preserve invariants**: Respect accounting invariants (e.g. `cash_after - cash_before_entry == net_pnl` on closes), safety barriers, and frozen strategy definitions. Do not weaken these.
- **Uncertainty**: Clearly distinguish facts, assumptions, and unverified results. Include file paths and class/function names so claims can be independently verified.
- **No comments unless asked**: Code style - do not add comments unless explicitly requested.
- **Update context**: When making important architectural/behavioral changes, update `PROJECT_CONTEXT.md` accordingly.

## File Handling

- Classify discovered files (essential source/tests, required config/lockfiles, useful docs/research, temp/logs/runtime state, sensitive, uncertain). Do not assume temp-like dirs (var/, var_alt/, reports/, research/) are disposable without inspection.
- Do not automatically stage `.gitignore` or other files; propose changes and let owner review.
- If an ignored file is already tracked, report it rather than relying on .gitignore.
- Never untrack files automatically; flag tracked secrets as blockers.

## What NOT To Do

- No history rewriting, rebasing, amending.
- No broad staging (`git add .` variants).
- No destructive git operations.
- No guessing URLs beyond what's needed to help with programming or those provided.
- No fabricating test results, features, APIs, CLI flags, or env vars not present in code.
- No copying real secrets into examples or reports.

## Verification Approach

Before finalizing, verify: diff is minimal and correct, no protected files changed, documented paths/commands exist, no secrets exposed (redacted), Markdown valid, and safety is not compromised. If safety cannot be established, flag as release blocker.

## Notes on This Codebase

- Safety verification: `python -m autopaper.cli verify` runs machine-checkable audits (import guard, real arm rules, frozen thresholds, profiles).
- Tests: Many tests exist; collection may fail if optional deps missing in some contexts, but core logic is testable. Focus on offline tests.
- REAL execution is implemented but gated; do not confuse implementation with enablement.
- Secret redaction is pervasive (logging, reports). Follow the same principle in any generated content.
