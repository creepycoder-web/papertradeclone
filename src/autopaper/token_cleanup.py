"""
Safe empty token-account cleanup (rent recovery).

WHY THIS EXISTS
---------------
Every first buy of a mint creates an associated token account and locks
rent-exempt lamports in it. An L25 wallet churns through mints, and the rent is
never recovered because the bot never closes the account. Measured on the live
account: EIGHT empty Token-2022 accounts holding 12,110,720 lamports = 0.01211
SOL, which was 1.89x the wallet's entire liquid balance.

THE RULES
---------
Closing a token account is a WRITE to the chain, so it is treated like any other
write in this runtime: fail closed, and require explicit confirmation.

An account is only SAFE TO CLOSE when ALL of these hold:

  1. it is owned by the CONFIGURED wallet (never another owner's account);
  2. its token balance is EXACTLY ZERO, per a chain read of the account itself;
  3. no OPEN REAL or PAPER position in the runtime references its mint;
  4. no PENDING/unresolved transaction depends on its mint;
  5. it is a real, initialized token account under a known token program
     (classic or 2022), and the chain's owning program matches that program;
  6. the account address is present;
  7. its CLOSE AUTHORITY is the configured wallet and not ``None``;
  8. it carries no live delegate and is not frozen;
  9. no layer of the runtime still reserves the mint (unresolved execution,
     pending settlement, unresolved external close, expected arrival, journal);
 10. it is not the wrapped-SOL account, which is the pre-provisioned entry
     route and must stay available;
 11. the rent destination is the configured wallet itself.

Anything else is BLOCKED and named with the reason it was blocked, and anything
that cannot be PROVEN unreferenced is ``SKIPPED_AMBIGUOUS``. A blocked account
is never closed and never silently dropped -- it is reported.

DRY RUN BY DEFAULT
------------------
``plan_token_account_cleanup`` is pure and does no I/O: it turns a list of
accounts plus the runtime's open/pending/reserved mint sets into a plan and a
rent total. ``execute_cleanup`` performs the write and REQUIRES an explicit
confirmation token; without it, it refuses. Nothing here auto-closes anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional

from .execution.solana import (
    TOKEN_PROGRAMS,
    TOKEN_PROGRAM_ID,
    TOKEN_2022_PROGRAM_ID,
    WSOL_MINT,
)
from .wallet_assets import program_label

#: SPL Token ``CloseAccount`` instruction discriminator.
CLOSE_ACCOUNT_IX = 9

#: The literal that must be passed to actually close accounts. A boolean flag is
#: too easy to set by default and too easy to set twice.
CONFIRM_TOKEN = "CLOSE-EMPTY-TOKEN-ACCOUNTS"

#: SPL ``COption`` discriminant for the ``closeAuthority`` field.
#: ``0`` = None (nobody can close it), ``1`` = Some (an authority exists).
#: ``2`` does not occur on a normal token account; the mint close-authority
#: extension reuses the same enum, and ``None`` is still refused.
CLOSE_AUTHORITY_NONE = 0
CLOSE_AUTHORITY_SOME = 1

#: Verdict for an account the runtime could not prove safe to close.
SKIPPED_AMBIGUOUS = "SKIPPED_AMBIGUOUS"
BLOCKED = "BLOCKED"
ELIGIBLE = "ELIGIBLE"
#: The account is verified-empty and in the wallet, but the chain has not yet
#: confirmed that the close would succeed, so nothing is closed on inference.
AWAITING_SIMULATION = "AWAITING_SIMULATION"


@dataclass(frozen=True)
class CleanupCandidate:
    """An empty account that passed every safety rule."""

    token_account: str
    mint: str
    program: str = TOKEN_PROGRAM_ID
    lamports: int = 0
    reason: str = "empty, owned by this wallet, unreferenced"
    #: ``ELIGIBLE`` once the chain has proven the close, or the status under
    #: which the candidate is being HELD pending that proof.
    status: str = ELIGIBLE
    #: The on-chain ``closeAuthority`` address, or None when the account carries
    #: no close authority of its own.
    close_authority: Optional[str] = None
    #: The token-account owner as the chain reports it, distinct from the
    #: account's OWNING PROGRAM (which is the token program id).
    token_owner: str = ""
    #: The raw on-chain token amount, re-read from the account itself.
    on_chain_amount: Optional[int] = None

    def as_candidate_kwargs(self) -> dict[str, Any]:
        """Every field except the ones a caller re-states."""
        return {
            "token_account": self.token_account,
            "mint": self.mint,
            "program": self.program,
            "lamports": self.lamports,
            "reason": self.reason,
            "close_authority": self.close_authority,
            "token_owner": self.token_owner,
            "on_chain_amount": self.on_chain_amount,
        }

    @property
    def is_token_2022(self) -> bool:
        return self.program == TOKEN_2022_PROGRAM_ID

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_account": self.token_account,
            "mint": self.mint,
            "program": self.program,
            "program_label": program_label(self.program),
            "is_token_2022": self.is_token_2022,
            "rent_recoverable_lamports": self.lamports,
            "rent_recoverable_sol": self.lamports / 1_000_000_000,
            "reason": self.reason,
            "status": self.status,
            "token_owner": self.token_owner,
            "close_authority": self.close_authority,
            "on_chain_amount": self.on_chain_amount,
        }


@dataclass(frozen=True)
class BlockedAccount:
    """An empty-looking account that was NOT safe to close, and why."""

    token_account: str
    mint: str
    program: str = ""
    lamports: int = 0
    reason: str = ""
    #: ``ELIGIBLE`` or ``BLOCKED`` or ``SKIPPED_AMBIGUOUS``. Kept as a field so
    #: a report can bucket every candidate without re-deriving the verdict.
    status: str = BLOCKED
    #: The on-chain ``closeAuthority`` address, or None when the account has no
    #: close authority at all. An empty token account is NOT closable without it.
    close_authority: Optional[str] = None
    #: The token-account owner as the chain reports it, distinct from the
    #: account's OWNING PROGRAM (which is the token program id).
    token_owner: str = ""
    #: The raw on-chain token amount, re-read from the account itself.
    on_chain_amount: Optional[int] = None
    #: True when the runtime could not prove the account unreferenced. Never
    #: eligible; always reported.
    ambiguous: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_account": self.token_account,
            "mint": self.mint,
            "program": self.program,
            "program_label": program_label(self.program) if self.program else "",
            "rent_locked_lamports": self.lamports,
            "reason": self.reason,
            "status": self.status,
            "token_owner": self.token_owner,
            "close_authority": self.close_authority,
            "on_chain_amount": self.on_chain_amount,
        }


@dataclass
class CleanupPlan:
    """The result of a dry run. Nothing has been closed."""

    wallet: str = ""
    owner: str = ""
    rent_destination: str = ""
    accounts_found: int = 0
    safe_to_close: tuple[CleanupCandidate, ...] = ()
    blocked: tuple[BlockedAccount, ...] = ()
    #: Verified-empty accounts the planner is HOLDING because the chain has not
    #: yet proven the close would succeed. Never closed, never counted in the
    #: rent total, always reported.
    simulation_required: tuple[CleanupCandidate, ...] = ()
    estimated_rent_return_lamports: int = 0
    notes: tuple[str, ...] = ()

    @property
    def estimated_rent_return_sol(self) -> float:
        return self.estimated_rent_return_lamports / 1_000_000_000

    @property
    def has_safe_accounts(self) -> bool:
        return len(self.safe_to_close) > 0

    @property
    def skipped_ambiguous(self) -> tuple[BlockedAccount, ...]:
        return tuple(b for b in self.blocked if b.status == SKIPPED_AMBIGUOUS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "wallet": self.wallet,
            "owner": self.owner,
            "rent_destination": self.rent_destination,
            "rent_destination_is_configured_wallet":
                bool(self.owner) and self.rent_destination == self.owner,
            "accounts_found": self.accounts_found,
            "accounts_safe_to_close": len(self.safe_to_close),
            "accounts_skipped_ambiguous": len(self.skipped_ambiguous),
            "accounts_awaiting_simulation_proof": len(self.simulation_required),
            "estimated_rent_return_lamports": self.estimated_rent_return_lamports,
            "estimated_rent_return_sol": self.estimated_rent_return_sol,
            "safe_to_close": [c.as_dict() for c in self.safe_to_close],
            "blocked": [b.as_dict() for b in self.blocked],
            "simulation_required": [c.as_dict() for c in self.simulation_required],
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """A human diagnostic: mint, account, program, rent recoverable."""
        lines = [
            f"wallet            : {self.wallet}",
            f"rent destination  : {self.rent_destination} "
            f"({'== configured wallet' if self.rent_destination == self.owner else 'MISMATCH'})",
            f"accounts_found    : {self.accounts_found}",
            f"accounts_safe     : {len(self.safe_to_close)}",
            f"awaiting proof    : {len(self.simulation_required)}",
            f"blocked           : {len(self.blocked)}",
            f"rent recoverable  : {self.estimated_rent_return_lamports:,} lamports "
            f"({self.estimated_rent_return_sol:.9f} SOL)",
        ]
        if self.safe_to_close:
            lines.append("")
            lines.append("SAFE TO CLOSE:")
            for c in self.safe_to_close:
                lines.append(
                    f"  {c.mint[:20]:<20}  {c.token_account[:20]:<20}  "
                    f"{program_label(c.program):<10}  {c.lamports:,} lamports"
                )
        if self.simulation_required:
            lines.append("")
            lines.append("HELD (not closed; closability unproven):")
            for c in self.simulation_required:
                lines.append(
                    f"  {c.mint[:20]:<20}  {c.token_account[:20]:<20}  "
                    f"{c.reason}"
                )
        if self.blocked:
            lines.append("")
            lines.append("BLOCKED (NOT closed):")
            for b in self.blocked:
                lines.append(
                    f"  {b.mint[:20]:<20}  {b.token_account[:20]:<20}  "
                    f"{b.reason}"
                )
        if self.notes:
            lines.append("")
            for n in self.notes:
                lines.append(f"note: {n}")
        return "\n".join(lines)


def plan_token_account_cleanup(
    accounts: Iterable[dict],
    *,
    owner: str,
    open_mints: Iterable[str] = (),
    pending_mints: Iterable[str] = (),
    reserved_mints: Iterable[str] = (),
    rent_destination: Optional[str] = None,
    closability: Optional[dict[str, dict]] = None,
    wallet_masked: str = "",
    require_on_chain_state: bool = True,
) -> CleanupPlan:
    """
    Decide which empty token accounts are safe to close. Pure, no I/O.

    Every safety rule is evaluated from the DATA PASSED IN. The caller is
    responsible for having read the chain and the runtime state; this function
    never re-reads and never signs.

    WHY ``require_on_chain_state`` DEFAULTS TO TRUE
    ------------------------------------------------
    A ``CloseAccount`` is a WRITE, so "is this account empty and mine?" must be
    answered by the chain, not by a locally cached row. The row from
    ``getTokenAccountsByOwner`` carries the token balance but NOT
    ``closeAuthority`` / ``delegate`` / ``frozen``, and those decide whether a
    close can even happen. A row with no ``chain_state`` is refused rather than
    trusted: silently treating an unverified row as safe is the exact failure
    mode this guards.

    ``closability`` carries SIMULATION PROOFS keyed by token-account address,
    produced by :func:`simulate_close_for`. An account whose on-chain state
    leaves closability genuinely in question -- no close authority is set --
    lands in ``simulation_required`` instead of ``safe_to_close`` until the node
    itself says the close would succeed. Nothing is closed on inference.

    ``reserved_mints`` is the ambiguity union: a mint that ANY layer of the
    runtime still needs is refused as ``SKIPPED_AMBIGUOUS``, never closed on the
    strength of one layer's view (e.g. "no open position" while an unresolved
    execution still references the mint).
    """
    open_set = set(str(m) for m in open_mints if m)
    pending_set = set(str(m) for m in pending_mints if m)
    reserved_set = set(str(m) for m in reserved_mints if m)
    owner = str(owner or "")
    destination = str(rent_destination if rent_destination is not None
                      else owner)
    proofs = {str(k): (v if isinstance(v, dict) else {})
              for k, v in (closability or {}).items()}

    safe: list[CleanupCandidate] = []
    pending_sim: list[CleanupCandidate] = []
    blocked: list[BlockedAccount] = []
    found = 0
    notes: list[str] = []

    # -- rule 11: the rent must land back in the configured wallet -----------
    # A refund destination that is not the account owner would send the rent to a
    # third party. That is a whole-plan refusal, not a per-account skip.
    if destination != owner:
        blocked.append(BlockedAccount(
            token_account="", mint="", program="", lamports=0, status=BLOCKED,
            reason=(f"rent destination {destination[:12] or '(none)'} is not the "
                    f"configured wallet {owner[:12] or '(none)'}; refusing the "
                    f"whole plan")))
        notes.append("REFUSED: rent destination is not the configured wallet.")
        return CleanupPlan(
            wallet=wallet_masked, owner=owner, rent_destination=destination,
            accounts_found=0, safe_to_close=(), blocked=tuple(blocked),
            estimated_rent_return_lamports=0, notes=tuple(notes))

    for row in accounts or ():
        if not isinstance(row, dict):
            continue
        found += 1
        token_account = str(row.get("account") or "")
        mint = str(row.get("mint") or "")
        program = str(row.get("program") or "")
        try:
            amount = int(row.get("amount"))
        except (TypeError, ValueError):
            amount = None
        raw_lamports = row.get("lamports")
        try:
            lamports = int(raw_lamports) if raw_lamports is not None else 0
        except (TypeError, ValueError):
            lamports = 0
        # The chain's own reading of this account, when the caller fetched one.
        chain = row.get("chain_state")
        chain = chain if isinstance(chain, dict) else None

        def _block(reason: str, status: str = BLOCKED) -> None:
            blocked.append(BlockedAccount(
                token_account=token_account, mint=mint, program=program,
                lamports=lamports, reason=reason, status=status,
                close_authority=(str(chain.get("close_authority"))
                                 if chain and chain.get("close_authority")
                                 else None),
                token_owner=(str(chain.get("token_owner")) if chain
                             else str(row.get("owner") or "")),
                on_chain_amount=(int(chain["amount"])
                                 if chain and chain.get("amount") is not None
                                 else None),
                ambiguous=(status == SKIPPED_AMBIGUOUS)))

        # -- rule 6: a real account address --------------------------------
        if not token_account:
            _block("no token account address; refusing to close an unnamed "
                   "account")
            continue
        # -- rule 5: a known token program ---------------------------------
        if program not in TOKEN_PROGRAMS:
            _block(f"unknown token program {program or '(none)'}; refusing")
            continue

        # -- rule 12: the account really is a token account -----------------
        # Refuse a system account, a mint, or anything the node does not parse as
        # a token account. Only the chain can make this distinction, so an
        # unreadable account is refused rather than assumed.
        if require_on_chain_state and chain is None:
            _block("no on-chain state was read for this account; refusing to "
                   "close an unverified account")
            continue
        if chain is not None:
            if not chain.get("exists"):
                _block("account does not exist on chain; nothing to close")
                continue
            if not chain.get("is_token_account"):
                _block(f"not a token account (owned by "
                       f"{(chain.get('owner_program') or 'unknown')[:12]}...)")
                continue
            if chain.get("executable"):
                _block("account is executable; refusing")
                continue
            if chain.get("owner_program") not in TOKEN_PROGRAMS:
                _block(f"on-chain owning program "
                       f"{(chain.get('owner_program') or 'unknown')[:12]}... is "
                       f"not a token program")
                continue
            # The two token programs are DIFFERENT programs on chain. A row
            # that claims one program while the account is owned by the other
            # would build an instruction the wrong program rejects.
            if chain.get("owner_program") != program:
                _block(f"on-chain owning program "
                       f"{(chain.get('owner_program') or '?')[:12]}... does not "
                       f"match the row's program {program_label(program)}")
                continue

        # -- rule 1: owned by the configured wallet ------------------------
        row_owner = str(row.get("owner") or "")
        if owner and row_owner and row_owner != owner:
            _block(f"owned by {row_owner[:12]}..., not the configured wallet")
            continue
        if chain is not None:
            chain_owner = str(chain.get("token_owner") or "")
            if not chain_owner:
                _block("on-chain owner is unreadable; refusing to close")
                continue
            if owner and chain_owner != owner:
                _block(f"on-chain owner {chain_owner[:12]}... is not the "
                       f"configured wallet")
                continue
            # -- rule 4b: initialized, not frozen ---------------------------
            state = str(chain.get("state") or "")
            if state and state != "initialized":
                _block(f"token-account state is {state!r}, not 'initialized'")
                continue
            if chain.get("frozen"):
                _block("account is frozen; refusing to close")
                continue

        # -- rule 2: balance is EXACTLY zero -------------------------------
        if amount is None:
            _block("token balance unreadable; refusing to close")
            continue
        if amount != 0:
            _block(f"balance is {amount} raw units, not zero; NOT closed")
            continue
        # The on-chain read is authoritative over the listing row: if the two
        # disagree about the balance, the account is not provably empty.
        if chain is not None:
            chain_amount = chain.get("amount")
            if chain_amount is None:
                _block("on-chain token amount unreadable; refusing to close")
                continue
            if int(chain_amount) != 0:
                _block(f"on-chain balance is {int(chain_amount)} raw units, not "
                       f"zero; NOT closed")
                continue
            if int(chain_amount) != int(amount):
                _block(f"on-chain balance ({int(chain_amount)}) disagrees with "
                       f"the listed balance ({amount})")
                continue

        # -- rule 8: no live delegate ---------------------------------------
        # An outstanding allowance means somebody else can move tokens out of an
        # account this runtime believes is dead, so it is not provably empty.
        if chain is not None and chain.get("delegate"):
            _block(f"account carries a live delegate "
                   f"{str(chain['delegate'])[:12]}...")
            continue

        # -- rule 7: close authority ----------------------------------------
        # A close authority that is SOMEONE ELSE is a hard refusal: the token
        # program requires that exact key to sign, so this wallet can never close
        # the account and the attempt would fail on chain every time.
        #
        # ``close_authority is None`` is NOT a hard refusal, and refusing it on
        # the parsed field alone would be wrong twice over: Solana's Token
        # Interface lets the ACCOUNT OWNER close an account that carries no close
        # authority, so these accounts ARE closable by this wallet; and treating
        # them as unclosable would report a false blocker over genuinely
        # recoverable rent. They are instead held for SIMULATION PROOF -- the
        # node itself is asked whether the real close would succeed -- and are
        # eligible only once the node says yes.
        authority = str(chain.get("close_authority") or "") if chain else ""
        if authority and owner and authority != owner:
            _block(f"close authority is {authority[:12]}..., not the configured "
                   f"wallet")
            continue

        # -- rule 3: no open position --------------------------------------
        if mint in open_set:
            _block("an OPEN position references this mint")
            continue
        # -- rule 4: no pending transaction --------------------------------
        if mint in pending_set:
            _block("a PENDING transaction depends on this mint")
            continue
        # -- rule 10: no ambiguity anywhere in the runtime -----------------
        if mint in reserved_set:
            _block(f"the runtime still references this mint somewhere "
                   f"(unresolved execution, pending settlement, external-close, "
                   f"or journal) -- {SKIPPED_AMBIGUOUS}",
                   status=SKIPPED_AMBIGUOUS)
            continue
        # -- rule 10b: the wrapped-SOL account is trading infrastructure ----
        # It is empty whenever the bot sits between entries, but it is the
        # PRE-PROVISIONED account a real ENTRY swap routes through. Closing it
        # reclaims ~0.0015 SOL and costs the wallet its next entry.
        if mint == WSOL_MINT:
            _block("wrapped-SOL account; it is the pre-provisioned entry route "
                   "and must stay available")
            continue
        # -- rule 12c: the address must not be the mint itself --------------
        if token_account == mint:
            _block("address equals the mint; refusing")
            continue

        candidate = CleanupCandidate(
            token_account=token_account, mint=mint, program=program,
            lamports=lamports,
            reason="empty, owned by this wallet, initialized, no delegate, "
                   "unreferenced by the runtime",
            status=AWAITING_SIMULATION,
            close_authority=authority or None,
            token_owner=(str(chain.get("token_owner")) if chain else row_owner),
            on_chain_amount=(int(chain["amount"])
                             if chain and chain.get("amount") is not None
                             else None))

        # -- rule 7b: closability, proven by the node ----------------------
        if authority:
            # The close authority IS this wallet: no simulation needed to know
            # who may close it.
            candidate = replace(
                candidate, status=ELIGIBLE,
                reason=candidate.reason + "; close authority is this wallet")
            safe.append(candidate)
            continue
        proof = proofs.get(token_account)
        if proof is None:
            pending_sim.append(candidate)
            continue
        if proof.get("unproven"):
            # The node did not answer. That is NOT a refusal and must not be
            # reported as one: the account is still held, for the same reason it
            # was held before the probe.
            pending_sim.append(candidate)
            continue
        if not proof.get("closable"):
            _block(f"the chain does not accept the close: "
                   f"{str(proof.get('detail') or 'unknown failure')[:160]}")
            continue
        candidate = replace(
            candidate, status=ELIGIBLE,
            reason=(candidate.reason + "; no close authority is set, and the "
                    "chain confirms the account owner may close it"))
        safe.append(candidate)

    if not safe and not blocked and not pending_sim:
        notes.append("no token accounts were found for this wallet")
    if pending_sim:
        notes.append(
            f"{len(pending_sim)} account(s) have no close authority set. Solana "
            f"lets the account OWNER close such an account, but this is "
            f"confirmed by SIMULATION before anything is closed; run with "
            f"--verify or --execute to obtain that proof.")
    notes.append(
        "This is a DRY RUN. No account was closed and no transaction was "
        "broadcast. Closing requires an explicit confirmation.")
    return CleanupPlan(
        wallet=wallet_masked, owner=owner, rent_destination=destination,
        accounts_found=found,
        safe_to_close=tuple(safe), blocked=tuple(blocked),
        simulation_required=tuple(pending_sim),
        estimated_rent_return_lamports=sum(c.lamports for c in safe),
        notes=tuple(notes),
    )


async def _simulate_with_retry(rpc: Any, tx_b64: str, *, attempts: int = 4,
                               backoff_s: float = 2.0) -> tuple[Any, bool]:
    """
    Simulate, retrying a RATE-LIMITED node.

    Returns ``(outcome, unproven)``. ``unproven=True`` means the node never
    answered -- a transport fact, never a verdict about the transaction. Keeping
    that separate is the whole point: a 429 on a public endpoint must not be
    recorded as "the chain refuses this close", and must not be recorded as
    "verified" either.
    """
    import asyncio as _asyncio

    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return await rpc.simulate_transaction_detailed(tx_b64), False
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 >= max(1, attempts):
                break
            await _asyncio.sleep(backoff_s * (attempt + 1))
    return last, True


async def probe_closability(rpc: Any, wallet: Any, owner: str,
                            candidates: Iterable[Any]) -> dict[str, dict]:
    """
    Ask the node, by simulation, whether each held account can be closed.

    READ-ONLY. Simulation never puts anything on chain, so this is the safe way
    to resolve the one question the parsed account fields cannot answer for a
    Token-Interface account with no close authority. The private key signs
    transactions that are discarded; nothing is broadcast.

    ONE transaction holds EVERY candidate's close
    ------------------------------------------------
    Simulating each account separately costs one ``simulateTransaction`` per
    account, and a cleanup of a dozen accounts becomes a burst that a public
    rate-limited endpoint answers with 429s -- turning a routine verification
    into a stream of failures that each look like a chain-level refusal. One
    transaction containing every close is BOTH cheaper and stronger: if the batch
    simulates successfully then every individual close in it is proven, because
    the node executed all of them.

    When the batch is REFUSED, the accounts are probed one at a time so the
    failure is attributed to a specific account instead of being reported against
    all of them at once.
    """
    candidates = list(candidates)
    out: dict[str, dict] = {}
    if not candidates:
        return out
    unproven = {
        "token_account": "", "simulated": False, "closable": False,
        "unproven": True,
    }

    def _unproven_all(detail: str) -> dict[str, dict]:
        return {c.token_account: dict(unproven, token_account=c.token_account,
                                      detail=detail) for c in candidates}

    try:
        blockhash, _last_valid = await rpc.get_latest_blockhash()
    except Exception as exc:  # noqa: BLE001
        return _unproven_all(
            f"could not obtain a blockhash for the probe batch "
            f"({type(exc).__name__}); closability is UNPROVEN, not refused")

    batch_plan = CleanupPlan(
        wallet="", owner=owner, rent_destination=owner,
        safe_to_close=tuple(candidates),
        estimated_rent_return_lamports=sum(c.lamports for c in candidates))
    try:
        ixs = build_close_instructions(batch_plan)
    except Exception as exc:  # noqa: BLE001
        return _unproven_all(
            f"could not build the probe transaction ({type(exc).__name__}); "
            f"closability is UNPROVEN, not refused")
    import base64

    try:
        raw = wallet.sign_new_transaction(ixs, blockhash=blockhash)
    except Exception as exc:  # noqa: BLE001
        return _unproven_all(
            f"could not sign the probe ({type(exc).__name__}); closability is "
            f"UNPROVEN, not refused")
    b64 = base64.b64encode(raw).decode("ascii")

    outcome, failed_to_answer = await _simulate_with_retry(rpc, b64)
    if failed_to_answer:
        return _unproven_all(
            f"the node did not answer the batch simulation "
            f"({type(outcome).__name__}); closability is UNPROVEN, not refused")
    if outcome.ok:
        for c in candidates:
            out[c.token_account] = {
                "token_account": c.token_account, "simulated": True,
                "closable": True, "unproven": False,
                "detail": (f"{outcome.detail} (batch of "
                           f"{len(candidates)} closes in one transaction)"),
                "units_consumed": outcome.units_consumed,
            }
        return out

    # The batch was refused. Attribute the refusal per account rather than
    # charging the same unproven failure to every one of them.
    for candidate in candidates:
        out[candidate.token_account] = await simulate_close_for(
            rpc, wallet, owner, candidate.token_account, candidate.program,
            blockhash=blockhash)
    return out


def build_close_instructions(plan: CleanupPlan):
    """
    Build one SPL ``CloseAccount`` instruction per safe account.

    PURE AND OFFLINE. Delegates to the execution layer, which is the only layer
    permitted to import a chain library. Nothing is signed and nothing is sent.
    """
    from .execution.solana import close_account_instructions

    accounts = [(c.token_account, c.program) for c in plan.safe_to_close]
    return close_account_instructions(plan.owner, accounts)


def audit_close_instructions(instructions: Any, plan: CleanupPlan
                            ) -> list[str]:
    """
    Prove the cleanup transaction carries NOTHING but the intended closes.

    A cleanup transaction is the one write this runtime is authorized to make,
    so "only close instructions" is asserted structurally rather than trusted:
    every instruction is checked to be a ``CloseAccount`` (discriminator 9)
    addressed to one of the planned accounts under that account's OWN token
    program, with the planned wallet as both the rent destination and the
    signer. A swap, a transfer, an approval, a delegate change, a SystemProgram
    instruction or any non-token program id is a REFUSAL, not a warning --
    there is no path by which one of those is ever what the operator meant.

    Returns the list of violations; empty means the instruction set is exactly
    the planned closes. Pure: reads instructions, never builds or sends them.
    """
    expected = {c.token_account: c.program for c in plan.safe_to_close}
    problems: list[str] = []
    if not instructions:
        return ["no instructions were built"]
    for ix in instructions:
        program_id = str(ix.program_id)
        data = bytes(ix.data or b"")
        if program_id not in TOKEN_PROGRAMS:
            problems.append(f"instruction targets non-token program {program_id}")
            continue
        if len(data) != 1 or data[0] != CLOSE_ACCOUNT_IX:
            problems.append(
                f"instruction to {program_id} is not a single-byte "
                f"CloseAccount (discriminator {CLOSE_ACCOUNT_IX}); data="
                f"{data[:8].hex()}")
            continue
        metas = list(ix.accounts or ())
        if len(metas) != 3:
            problems.append(f"CloseAccount for {program_id} has {len(metas)} "
                            f"account metas, expected 3")
            continue
        target, destination, authority = (str(m.pubkey) for m in metas)
        if target not in expected:
            problems.append(f"CloseAccount targets unplanned account "
                            f"{target[:12]}...")
            continue
        if expected[target] != program_id:
            problems.append(f"account {target[:12]}... is owned by "
                            f"{program_label(expected[target])}, but the "
                            f"instruction uses {program_label(program_id)}")
            continue
        if destination != plan.rent_destination or authority != plan.owner:
            problems.append(f"rent destination {destination[:12]}... / authority "
                            f"{authority[:12]}... is not the configured wallet")
            continue
        if not any(m.is_signer and str(m.pubkey) == plan.owner
                   for m in metas):
            problems.append(f"CloseAccount for {target[:12]}... does not "
                            f"sign with the configured wallet")
    return problems


async def read_on_chain_states(rpc: Any, addresses: Iterable[str], *,
                               attempts: int = 3,
                               backoff_s: float = 1.5) -> dict[str, Optional[dict]]:
    """
    Read each account's authoritative state from the configured RPC.

    This is the ONLY place the cleanup path touches the chain before deciding
    anything. ``None`` means the account does not exist (already closed), which
    the planner reports rather than treating as an error.

    ONE batched call, with a bounded retry. The re-check immediately before a
    broadcast must not be a burst of N reads: on a rate-limited public endpoint
    the tail of that burst comes back HTTP 429, and a 429 inside a safety re-check
    is indistinguishable from "the chain says no". Retrying the batch as a unit,
    and raising if it never lands, keeps "the node would not answer" separate from
    "the node refused" -- and never lets an unreadable account read as verified.
    """
    wanted = [str(a) for a in addresses if a]
    if not wanted:
        return {}
    if hasattr(rpc, "get_token_account_states"):
        last: Exception | None = None
        for attempt in range(max(1, attempts)):
            try:
                states = await rpc.get_token_account_states(wanted)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt + 1 >= max(1, attempts):
                    break
                import asyncio as _asyncio

                await _asyncio.sleep(backoff_s * (attempt + 1))
                continue
            # A short answer means the node truncated the batch; per-account
            # reads then, so no address is silently unverified.
            if len(states) < len(wanted):
                missing = [a for a in wanted if a not in states]
                for address in missing:
                    states[address] = await rpc.get_token_account_state(address)
            return states
        raise RuntimeError(
            f"could not read {len(wanted)} token account(s) from the RPC after "
            f"{attempts} attempt(s) ({type(last).__name__ if last else 'unknown'}); "
            f"nothing was planned and nothing was signed")
    out: dict[str, Optional[dict]] = {}
    for address in wanted:
        out[address] = await rpc.get_token_account_state(address)
    return out


async def execute_cleanup(
    rpc: Any,
    wallet: Any,
    plan: CleanupPlan,
    *,
    confirm: str = "",
    simulate: bool = True,
    commitment: str = "confirmed",
    recheck: bool = True,
) -> dict[str, Any]:
    """
    Close the planned empty accounts. REQUIRES the explicit confirmation token.

    A WRITE, not a read. It refuses without ``confirm == CONFIRM_TOKEN``, refuses
    an empty plan, refuses a plan whose rent destination is not the configured
    wallet, audits the instruction set, re-reads every account from the chain,
    simulates, and only then broadcasts. One transaction is built and signed by
    the runtime wallet; the private key never leaves :class:`SolanaWallet`.

    WHY A RE-READ AFTER THE PLAN
    ----------------------------
    The plan and the broadcast are separated by operator time and by the plan's
    own ``--execute`` invocation. An account can be filled, delegated or closed
    in between. The account's on-chain state is therefore re-verified HERE,
    immediately before the transaction is built, and any account that is no
    longer provably empty-and-mine is dropped from the transaction rather than
    closed optimistically.

    A simulation failure is terminal: nothing is broadcast and the exact failure
    is returned verbatim.
    """
    if confirm != CONFIRM_TOKEN:
        return {
            "executed": False,
            "broadcast_attempted": False,
            "reason": (f"refusing to close accounts without the explicit "
                       f"confirmation token {CONFIRM_TOKEN!r}"),
            "closed": [],
        }
    if not plan.safe_to_close:
        return {"executed": False, "broadcast_attempted": False,
                "reason": "plan has no safe accounts", "closed": []}
    if plan.rent_destination != plan.owner:
        return {"executed": False, "broadcast_attempted": False,
                "reason": ("rent destination is not the configured wallet; "
                           "refusing to broadcast"),
                "closed": []}

    instructions = build_close_instructions(plan)
    problems = audit_close_instructions(instructions, plan)
    if problems:
        return {"executed": False, "broadcast_attempted": False,
                "reason": "instruction audit failed: " + "; ".join(problems),
                "closed": [], "audit_problems": problems}

    active_plan = plan
    recheck_report: list[dict[str, Any]] = []
    if recheck:
        planned = {c.token_account: c for c in plan.safe_to_close}
        states = await read_on_chain_states(
            rpc, list(planned))
        still_safe: list[CleanupCandidate] = []
        for address, candidate in planned.items():
            state = states.get(address)
            why = ""
            if state is None:
                why = "account no longer exists on chain"
            elif not state.get("is_token_account"):
                why = "account is no longer a token account"
            elif str(state.get("owner_program") or "") != candidate.program:
                why = "on-chain owning program changed"
            elif str(state.get("token_owner") or "") != plan.owner:
                why = "on-chain token owner is not the configured wallet"
            elif state.get("amount") is None:
                why = "on-chain token amount unreadable"
            elif int(state["amount"]) != 0:
                why = f"on-chain balance is {int(state['amount'])}, not zero"
            elif str(state.get("close_authority") or "") not in ("", plan.owner):
                # A close authority that is somebody else is terminal: that key
                # must sign. No close authority is NOT terminal -- Solana's Token
                # Interface lets the account owner close such an account, and
                # the pre-broadcast simulation below is what proves it.
                why = "close authority is not the configured wallet"
            elif state.get("delegate"):
                why = "account acquired a live delegate"
            elif state.get("frozen"):
                why = "account is frozen"
            recheck_report.append({
                "token_account": address, "eligible_after_recheck": not why,
                "reason": why or "still empty, owned, close-authority verified",
                "on_chain_amount": (None if state is None
                                    else state.get("amount")),
                "close_authority": (None if state is None
                                    else state.get("close_authority")),
                "lamports": (None if state is None else state.get("lamports")),
            })
            if why:
                recheck_report[-1]["status"] = BLOCKED
            else:
                recheck_report[-1]["status"] = ELIGIBLE
                still_safe.append(candidate)
        if not still_safe:
            return {"executed": False, "broadcast_attempted": False,
                    "reason": ("no account is still provably safe to close at "
                               "broadcast time"),
                    "closed": [], "recheck": recheck_report}
        if len(still_safe) != len(plan.safe_to_close):
            active_plan = CleanupPlan(
                wallet=plan.wallet, owner=plan.owner,
                rent_destination=plan.rent_destination,
                accounts_found=plan.accounts_found,
                safe_to_close=tuple(still_safe), blocked=plan.blocked,
                simulation_required=plan.simulation_required,
                estimated_rent_return_lamports=sum(
                    c.lamports for c in still_safe),
                notes=plan.notes + (
                    f"{len(plan.safe_to_close) - len(still_safe)} account(s) "
                    f"dropped at re-check; the chain no longer proves them safe",
                ))
            instructions = build_close_instructions(active_plan)
            problems = audit_close_instructions(instructions, active_plan)
            if problems:
                return {"executed": False, "broadcast_attempted": False,
                        "reason": "instruction audit failed: " + "; ".join(
                            problems), "closed": [],
                        "audit_problems": problems, "recheck": recheck_report}

    # A real recent blockhash. The zero hash is only valid for an offline
    # signing probe; a broadcast signed against it is rejected by the node.
    import base64

    try:
        blockhash, last_valid_block_height = await rpc.get_latest_blockhash()
    except Exception as exc:  # noqa: BLE001
        return {"executed": False, "broadcast_attempted": False,
                "reason": f"could not obtain a recent blockhash "
                          f"({type(exc).__name__}); nothing was signed",
                "closed": [], "recheck": recheck_report}

    # Build, sign, and (optionally) simulate, then broadcast ONE transaction.
    raw = wallet.sign_new_transaction(instructions, blockhash=blockhash)

    b64 = base64.b64encode(raw).decode("ascii")
    if simulate:
        # Retrying a rate-limited node is correct here and ONLY here: this is the
        # last gate before the one write this runtime is authorized to make, and
        # a 429 must not read as either "the chain refused" or "the chain
        # approved". A refusal that survives the retries is still terminal.
        outcome, failed_to_answer = await _simulate_with_retry(rpc, b64)
        if failed_to_answer:
            return {"executed": False, "broadcast_attempted": False,
                    "reason": (f"the node never answered the simulation "
                               f"({type(outcome).__name__}); the transaction was "
                               f"signed but NOT broadcast, and closability is "
                               f"UNPROVEN rather than refused"),
                    "closed": [], "recheck": recheck_report}
        ok = bool(getattr(outcome, "ok", False))
        detail = f"{getattr(outcome, 'status', '')}: {getattr(outcome, 'detail', '')}"
        simulation = (outcome.as_dict() if hasattr(outcome, "as_dict")
                      else {"ok": ok, "detail": detail})
        if not ok:
            return {"executed": False, "broadcast_attempted": False,
                    "reason": f"simulation refused: {detail}",
                    "simulation": simulation, "closed": [],
                    "recheck": recheck_report}
    else:
        simulation = {"ok": None, "detail": "simulation was NOT run"}

    signature = await rpc.send_transaction(b64)
    result = await rpc.wait_for_confirmation(
        signature, last_valid_block_height=last_valid_block_height,
        commitment=commitment)
    status = str((result or {}).get("status") or "")
    return {
        "executed": status in ("confirmed", "finalized"),
        "broadcast_attempted": True,
        "signature": signature,
        "status": status,
        "slot": (result or {}).get("slot"),
        "instruction_count": len(instructions),
        "simulation": simulation,
        "recheck": recheck_report,
        "rent_destination": active_plan.rent_destination,
        "closed": [c.token_account for c in active_plan.safe_to_close],
        "rent_recovered_lamports": active_plan.estimated_rent_return_lamports,
        "reason": "" if status in ("confirmed", "finalized")
                  else f"confirmation status {status!r}",
    }


async def simulate_close_for(rpc: Any, wallet: Any, owner: str,
                             token_account: str, program: str,
                             blockhash: Any = None) -> dict[str, Any]:
    """
    Ask the CHAIN whether one account can be closed, without broadcasting.

    ``simulateTransaction`` runs the real ``CloseAccount`` against real account
    state. The answer is the node's, not this module's inference from parsed
    fields: a foreign close authority, a delegate, or a non-zero balance fails
    here with the token program's own error. Nothing is sent; the signature that
    is produced is discarded with the object.

    Used as evidence, not as the decision: the planner refuses to build the
    instruction for an account it cannot prove safe, and this proves WHY.

    A transport failure returns ``closable=False`` with ``unproven=True``. That
    distinction matters: "the node could not answer" and "the node refused" must
    never collapse into the same verdict, or an RPC hiccup would masquerade as a
    chain-level rejection.
    """
    import base64

    from .execution.solana import close_account_instructions

    if blockhash is None:
        try:
            blockhash, _last_valid = await rpc.get_latest_blockhash()
        except Exception as exc:  # noqa: BLE001
            return {"token_account": token_account, "simulated": False,
                    "closable": False, "unproven": True,
                    "detail": (f"could not obtain a blockhash "
                               f"({type(exc).__name__}); closability is UNPROVEN, "
                               f"not refused")}
    instructions = close_account_instructions(owner, [(token_account, program)])
    try:
        raw = wallet.sign_new_transaction(instructions, blockhash=blockhash)
    except Exception as exc:  # noqa: BLE001
        return {"token_account": token_account, "simulated": False,
                "closable": False, "unproven": True,
                "detail": f"could not sign the probe ({type(exc).__name__})"}
    b64 = base64.b64encode(raw).decode("ascii")
    try:
        outcome = await rpc.simulate_transaction_detailed(b64)
    except Exception as exc:  # noqa: BLE001
        return {"token_account": token_account, "simulated": False,
                "closable": False, "unproven": True,
                "detail": f"simulation raised ({type(exc).__name__}); "
                          f"closability is UNPROVEN, not refused"}
    if outcome.ok:
        return {"token_account": token_account, "simulated": True,
                "closable": True, "unproven": False, "detail": outcome.detail,
                "units_consumed": outcome.units_consumed}
    return {"token_account": token_account, "simulated": True,
            "closable": False, "unproven": False, "detail": outcome.detail,
            "rpc_message": outcome.rpc_message,
            "programs_involved": programs_in_message(raw)}


def programs_in_message(raw: bytes) -> list[str]:
    """
    The program ids a built transaction actually references.

    Used to PROVE the absence of anything but the token programs in a cleanup
    transaction. A swap's router program, a SystemProgram transfer, an approval
    or a delegate change would each show up here as a program id that is not a
    token program.

    Delegates to the execution layer, which is the only layer permitted to
    import a chain library.
    """
    from .execution.solana import programs_in_transaction

    return programs_in_transaction(raw)


async def confirm_closures(rpc: Any, addresses: Iterable[str]) -> dict:
    """
    Prove, from the chain, that each planned account is GONE.

    "Broadcast OK" is not proof that rent moved. An account is closed when
    ``getAccountInfo`` no longer returns it -- the lamports are gone from that
    account and can only have gone to the rent destination the token program
    paid. Every address is re-read so the report distinguishes closed from still
    present from unreadable; an unreadable account is NOT reported as closed.
    """
    wanted = [str(a) for a in addresses if a]
    states: dict[str, Any] = {}
    read_error = ""
    if hasattr(rpc, "get_token_account_states"):
        # One batched read: N sequential getAccountInfo calls after a broadcast
        # is how a cleanup verification turns into a 429 and reports "unreadable"
        # for accounts that are demonstrably gone.
        try:
            states = await rpc.get_token_account_states(wanted)
        except Exception as exc:  # noqa: BLE001
            read_error = type(exc).__name__
        else:
            # A SHORT batch answer is not an answer. A load-balanced public node
            # can return fewer values than addresses asked for, and treating the
            # shortfall as "the accounts are still there" turns a completed
            # cleanup into a reported failure -- the one conclusion this whole
            # function exists to prevent.
            short = [a for a in wanted if a not in states]
            for address in short:
                try:
                    states[address] = await rpc.get_token_account_state(address)
                except Exception as exc:  # noqa: BLE001
                    states[address] = "unreadable"
                    read_error = read_error or type(exc).__name__
    else:
        for address in wanted:
            try:
                states[address] = await rpc.get_token_account_state(address)
            except Exception as exc:  # noqa: BLE001
                states[address] = "unreadable"
                read_error = read_error or type(exc).__name__
    results: list[dict[str, Any]] = []
    closed = 0
    for address in wanted:
        state = states.get(address, "unreadable")
        if isinstance(state, str):
            results.append({"token_account": address, "state": state,
                            "detail": read_error or "batched read incomplete",
                            "closed": False})
        elif state is None:
            closed += 1
            results.append({"token_account": address, "state": "does_not_exist",
                            "closed": True})
        else:
            results.append({
                "token_account": address, "state": "still_exists",
                "closed": False, "lamports": state.get("lamports"),
                "amount": state.get("amount"),
                "close_authority": state.get("close_authority"),
            })
    return {
        "accounts_checked": len(results),
        "accounts_closed": closed,
        "accounts_still_present": len(results) - closed,
        "results": results,
    }


async def resolve_broadcast_outcome(rpc: Any, signature: str, *,
                                    first_verdict: str = "",
                                    attempts: int = 5,
                                    backoff_s: float = 2.0
                                    ) -> dict[str, Any]:
    """
    Ask the chain what ACTUALLY happened to the cleanup transaction.

    WHY THIS EXISTS
    ---------------
    Confirmation is reported by polling ``getSignatureStatuses``, and that poll
    fails open on a rate-limited node: a 429 leaves the status list empty, and
    the waiter's only other move is to compare the current slot against the
    blockhash's ``lastValidBlockHeight``. On a load-balanced public endpoint
    those two numbers can come from DIFFERENT nodes whose slot counters are far
    apart, so a transaction that is already finalized gets reported as
    ``expired`` -- a failed cleanup, when in fact the rent is back.

    That is the worst possible error for this operation: an operator told "the
    rent is gone" who then re-runs the cleanup, or concludes the accounts are
    permanently closed, when the opposite is true. So an inconclusive first
    verdict is never reported as final. The chain is asked directly, with
    retries, for the transaction's own status, and THAT is what is reported.

    Returns the authoritative verdict plus what the first pass said, so the
    report can show that the two disagreed instead of quietly overwriting one.
    """
    import asyncio as _asyncio

    last_error = ""
    for attempt in range(max(1, attempts)):
        try:
            statuses = await rpc.get_signature_statuses(signature)
        except Exception as exc:  # noqa: BLE001
            last_error = type(exc).__name__
            statuses = []
        if statuses:
            st = statuses[0]
            if st is None:
                last_error = last_error or "signature not yet indexed"
            elif st.get("err"):
                return {
                    "authoritative_status": "failed", "err": st.get("err"),
                    "slot": st.get("slot"),
                    "superseded_first_verdict": first_verdict,
                    "note": "the chain reports this transaction FAILED",
                }
            else:
                conf = st.get("confirmationStatus")
                if conf in ("confirmed", "finalized"):
                    return {
                        "authoritative_status": conf, "err": None,
                        "slot": st.get("slot"),
                        "superseded_first_verdict": first_verdict,
                        "note": ("" if first_verdict in ("", conf) else
                                 f"the first confirmation pass reported "
                                 f"{first_verdict!r}; the chain says {conf}"),
                    }
                last_error = last_error or f"confirmationStatus={conf}"
        if attempt + 1 < max(1, attempts):
            await _asyncio.sleep(backoff_s * (attempt + 1))
    return {"authoritative_status": "unresolved", "err": None, "slot": None,
            "superseded_first_verdict": first_verdict,
            "note": (f"the node did not give a usable status for this "
                     f"signature ({last_error or 'no status'}); UNRESOLVED, not "
                     f"a failure and not a success")}


async def read_transaction_fee(rpc: Any, signature: str) -> Optional[int]:
    """
    The fee the chain charged for the cleanup transaction, in lamports.

    Read back from the confirmed transaction's meta, not estimated locally.
    ``None`` means the transaction could not be read; it is never reported as a
    zero fee, because a zero fee would silently inflate the recovered amount.
    """
    tx = await rpc.get_transaction(signature, commitment="confirmed")
    if not tx:
        return None
    meta = ((tx.get("meta") or {}))
    try:
        return int(meta["fee"])
    except (KeyError, TypeError, ValueError):
        return None


async def reconcile_balance(rpc: Any, address: str) -> dict:
    """Read the wallet's native SOL balance at a named instant."""
    lamports = await rpc.get_balance_lamports(address)
    return {
        "lamports": int(lamports),
        "sol": int(lamports) / 1_000_000_000,
    }
