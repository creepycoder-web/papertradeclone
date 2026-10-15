# Contributing to AI-paper-trade (autopaper)

## Prerequisites

- Python 3.11+
- Git
- Access to the repository (read-only guidance unless explicitly authorized)

## Development Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q  # run tests
.venv/bin/python -m autopaper.cli verify  # run safety verification
```

## Safety & Security Requirements

This project has strict safety constraints:
- Paper-only by default. Real trading requires explicit operator confirmation via Telegram + passing preflight. `REAL_TRADING_ENABLED` env var is recorded but ignored for authorization.
- Never commit secrets (`.env.local`, private keys, wallet files, session files). These are gitignored.
- The import guard blocks ML/Laya/shadow libraries at import time.
- Respect accounting invariants and frozen strategy definitions.

## Workflow

1. **Read first**: Review `README.md`, `PROJECT_CONTEXT.md`, and `AGENTS.md` before making changes.
2. **Inspect Git state**: `git status --short --branch`, `git diff --name-only --cached`, `git diff --name-only`, `git ls-files --others --exclude-standard`.
3. **Make minimal, focused changes**. Do not refactor unrelated subsystems.
4. **Add/update tests** for behavioral changes where practical. Never fake test results or silently skip failures.
5. **Run safety verification**: `python -m autopaper.cli verify` and relevant tests.
6. **Update documentation** (`PROJECT_CONTEXT.md`) when important architecture/behavior changes.
7. **Never commit/push** unless explicitly requested by the maintainer.

## File Handling

- Do not modify `.gitignore` to untrack or hide files; report findings.
- If an ignored file is already tracked, flag it.
- Keep changes small and traceable. Include file paths/class/function names in discussions.

## Testing Guidelines

- Run offline tests only. Many tests monkeypatch sockets/network.
- Report results as passed/failed/skipped; never claim unexecuted tests pass.
- Do not run trading loops or external operations.

## Questions & Issues

- Check existing docs first. If behavior differs from docs, record the discrepancy.
- For security concerns (potential secrets, safety gaps), flag as release blockers.
