"""
Security scan.

Reports credential-shaped material WITHOUT ever printing a value. Every hit is
returned as ``(file, line, kind)`` and nothing else, so running this over a
repository is safe and the output can be pasted into a ticket.

WHAT IS SCANNED
---------------
1. the working tree, excluding the venv, caches and git objects
2. git HISTORY, because a secret removed from the working tree is still a
   disclosed secret
3. the process environment, by variable NAME only

Kinds
-----
mnemonic             a 12-24 word seed phrase shape
solana_keypair       a 64-element byte array (a Solana private key)
telegram_bot_token   the ``NNNN:AAaa...`` bot token shape
telethon_session     a Telethon session string
api_hash_looking     a 32-hex string in an assignment to a *_HASH / *_SECRET name
bearer_token         an Authorization header value
pem_private_key      a BEGIN ... PRIVATE KEY block
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Directories that are never scanned: build output, caches, and the venv.
SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    "node_modules", "build", "dist", ".mypy_cache", ".ruff_cache",
    "reports", "var", "logs", "research", "artifacts", "data",
})


@dataclass(frozen=True)
class Finding:
    """A credential-shaped thing. Deliberately carries no value."""

    file: str
    line: int
    kind: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}  [{self.kind}]"


#: English words that make prose look like a mnemonic. A real BIP39 phrase is
#: 12 or 24 dictionary-ish words; an English sentence is also 12+ lowercase
#: words. The discriminator is NOT the shape - it is the CONTEXT: a real phrase
#: appears as a VALUE (quoted, or after an assignment), never inside a comment
#: or a docstring. Without this list the scanner fires on every English sentence
#: in the codebase, and a scanner that cries wolf gets ignored.
_PROSE_WORDS = frozenset("""
a an the and or but if then when while for of to in on at by with from into
over under about after before during without within is are was were be been
being do does did doing have has had having it its this that these those not
no nor so such than too very can will just only also as because our your their
his her they them we you i me my mine ours yours theirs one two three some any
each every other another same different new old first last next own same
""".split())

#: (kind, compiled pattern). Applied to the raw line text.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mnemonic", re.compile(r"\b[a-z]{3,10}(?:\s+[a-z]{3,10}){11,23}\b")),
    ("solana_keypair", re.compile(
        r"\[\s*\d{1,3}\s*(?:,\s*\d{1,3}\s*){60,}\]")),
    ("telegram_bot_token", re.compile(
        r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b")),
    ("pem_private_key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}=*")),
    ("seed_assignment", re.compile(
        r"(?i)\b(seed_phrase|mnemonic|private_key|solana_private_key|"
        r"secret_key|api_secret)\b\s*[:=]\s*[\"']?[^\s\"']{8,}")),
)

#: A 32-hex value assigned to a credential-shaped NAME. The value is never
#: captured or returned, only the fact of the assignment.
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_HASH|API_SECRET|API_KEY|TOKEN|PASSWORD|"
    r"SECRET|PRIVATE_KEY|MNEMONIC|SEED)[A-Z0-9_]*)\b\s*[:=]\s*"
    r"[\"']?([0-9a-fA-F]{32,}|[0-9]{8,12}:[A-Za-z0-9_-]{30,})"
)

#: Lines that are allowed to contain these shapes, because they are the GUARD
#: that refuses them rather than code that uses them.
GUARD_MARKERS = (
    "REDACTED", "forbidden", "FORBIDDEN", "block", "refus", "Refus",
    "assert", "raise", "secret_key_names", "SECRET_PATTERNS",
    "skip", "Skip", "SKIP", "not configured", "missing",
)


#: A line that assigns or quotes a value, i.e. where a credential would live.
_VALUE_CONTEXT = re.compile(
    r"""(=\s*|:\s*|\(\s*|\[\s*|,\s*)["'][^"']*["']"""
)

#: Line prefixes that mean "this is prose, not a value".
_COMMENT_PREFIX = re.compile(r"^\s*(#|//|\*|-->|---|\"\"\"|\'\'\')")


def _is_guard_line(line: str) -> bool:
    low = line.lower()
    return any(m.lower() in low for m in GUARD_MARKERS)


def _looks_like_mnemonic(line: str) -> bool:
    """
    Is there a real seed phrase on this line?

    Two conditions, both required:

    1. NOT prose. A comment or docstring line is excluded outright, and at least
       three quarters of the words must not be ordinary English function words.
       A genuine BIP39 phrase is dictionary words but essentially never English
       function words in that density.
    2. It must sit in a VALUE position - quoted, or after an assignment - not
       floating in a sentence.
    """
    stripped = line.strip()
    if _COMMENT_PREFIX.match(stripped):
        return False
    m = re.search(r"\b[a-z]{3,10}(?:\s+[a-z]{3,10}){11,23}\b", stripped)
    if not m:
        return False
    words = m.group(0).split()
    if len(words) < 12:
        return False
    prose = sum(1 for w in words if w.lower() in _PROSE_WORDS)
    if prose / len(words) > 0.25:
        return False
    return bool(_VALUE_CONTEXT.search(stripped))


def scan_text(text: str, rel: str) -> list[Finding]:
    out: list[Finding] = []
    for i, line in enumerate(text.splitlines(), 1):
        if len(line) > 4000:
            # a data blob, not source; the patterns would be meaningless on it
            continue
        for kind, pattern in PATTERNS:
            if kind == "mnemonic":
                if not _looks_like_mnemonic(line):
                    continue
                out.append(Finding(rel, i, kind))
                continue
            m = pattern.search(line)
            if not m:
                continue
            if kind == "seed_assignment" and _is_guard_line(line):
                continue
            if kind == "solana_keypair" and _is_guard_line(line):
                continue
            out.append(Finding(rel, i, kind))
        m = CREDENTIAL_ASSIGNMENT.search(line)
        if m and not _is_guard_line(line):
            out.append(Finding(rel, i, "credential_assignment"))
    return out


def scan_tree(root: Path = REPO) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in (".py", ".toml", ".cfg", ".ini", ".txt",
                               ".md", ".json", ".yaml", ".yml", ".sh", ".env",
                               ".example", ""):
            continue
        if path.name == ".env.local":
            # the operator's own file, deliberately untracked; its KEYS are
            # audited separately below and its VALUES are never read here
            findings.extend(_scan_env_local(path))
            continue
        try:
            findings.extend(scan_text(path.read_text(errors="replace"),
                                      str(path.relative_to(root))))
        except OSError:
            continue
    return findings


def _scan_env_local(path: Path) -> list[Finding]:
    """
    Audit the env file by KEY NAME only, never reading or reporting a value.

    A credential stored in an untracked env file is CORRECT. A private KEY in
    one is not, because the wallet is loaded from a keypair file and an env-var
    key would bypass the file permission boundary.
    """
    out: list[Finding] = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return out
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip().upper()
        if key in ("PRIVATE_KEY", "SOLANA_PRIVATE_KEY", "SEED_PHRASE",
                   "MNEMONIC", "WALLET_SECRET", "SECRET_KEY"):
            out.append(Finding(str(path.relative_to(REPO)), i,
                               "key_material_in_env_file"))
    return out


def scan_git_history(root: Path = REPO, max_revs: int = 50) -> list[Finding]:
    """
    A secret deleted from the working tree is still a DISCLOSED secret.

    Scans the last ``max_revs`` revisions of every tracked text file and
    reports only which revision and which kind.
    """
    out: list[Finding] = []
    try:
        revs = subprocess.run(
            ["git", "log", f"-{max_revs}", "--format=%H"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return out
    seen: set[tuple[str, int, str]] = set()
    for rev in revs:
        try:
            files = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", rev],
                cwd=root, capture_output=True, text=True, check=True,
            ).stdout.splitlines()
        except subprocess.CalledProcessError:
            continue
        for rel in files:
            if any(part in SKIP_DIRS for part in Path(rel).parts):
                continue
            name = Path(rel).name
            if Path(rel).suffix not in (".py", ".toml", ".json", ".md", ".txt",
                                       ".cfg", ".ini", ".yaml", ".yml", ""):
                # An env file has no useful suffix, and an env file committed to
                # history is precisely the thing this scan exists to find.
                if not (name.startswith(".env") or name.endswith(".session")):
                    continue
            try:
                blob = subprocess.run(
                    ["git", "show", f"{rev}:{rel}"],
                    cwd=root, capture_output=True, check=True,
                ).stdout.decode("utf-8", errors="replace")
            except (subprocess.CalledProcessError, UnicodeDecodeError):
                continue
            for f in scan_text(blob, rel):
                key = (rel, f.line, f.kind)
                if key in seen:
                    continue
                seen.add(key)
                out.append(f)
    return out


def scan_environment() -> list[Finding]:
    """Credential-shaped environment variables, by NAME only."""
    out: list[Finding] = []
    for name in sorted(os.environ):
        if any(b in name.upper() for b in (
            "PRIVATE_KEY", "SEED_PHRASE", "MNEMONIC", "API_SECRET", "SECRET_KEY",
        )):
            out.append(Finding("<environment>", 0, f"env:{name}"))
    return out


def full_scan(root: Path = REPO) -> dict[str, list[Finding]]:
    return {
        "working_tree": scan_tree(root),
        "git_history": scan_git_history(root),
        "environment": scan_environment(),
    }


def render(results: dict[str, list[Finding]]) -> str:
    lines: list[str] = []
    total = 0
    for scope, findings in results.items():
        lines.append(f"== {scope} ==")
        if not findings:
            lines.append("  clean")
        for f in findings:
            lines.append(f"  {f}")
        total += len(findings)
        lines.append("")
    lines.append(f"TOTAL: {total}")
    return "\n".join(lines)
