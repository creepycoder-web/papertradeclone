"""
Tests for the REAL exit, settlement and recovery fixes.

Every test here is OFFLINE. No RPC, no broadcast, no wallet. Chain facts are
injected as fixtures, because the whole point of these code paths is that they
refuse to act on an unverified fact -- which is impossible to test if the test
itself depends on the chain answering.

The $SI constants are the real 2026-10-01 incident values, so a future reader can
tell whether a test is still about the actual failure or has drifted into
hypotheticals.
"""


from __future__ import annotations

import asyncio

import pytest

pytestmark = [
    pytest.mark.execution,
    pytest.mark.safety,
    pytest.mark.strategy,
]

SI_MINT = "552y8SENvNqmFH6MR4LNiWd89DX8av9sYBLDsnkDpump"
SI_TOKEN_AMOUNT = 4_061_052_725          # 4,061.052725 SI, 6 decimals
SI_ENTRY_COST_USD = 0.114_177
SI_DEST_OWNER = "Be1cPQUjoiM7nMnvxf2fauRZydFMWeNkF1BFfGChoGXGXTu"
WALLET = "wUEUbVmz2EfcLfchHmucTKhRpZgDpsBvo3FttZ4Wbt6"
SI_EXTERNAL_SIG = (
    "2aRDj8DagcpFPghTtiUWvwYatzT38Xb26ev3iL1MM51Nt5w3ptZgTrQGJqBFwFx"
    "MMqhjvJZ8rLtu4VcNFQWyfXRK"
)

USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
WSOL_MINT = "So11111111111111111111111111111111111111112"


# ===========================================================================
# helpers
# ===========================================================================


def _tx(moves, *, wallet_lamport_delta=0, err=None, wallet=WALLET):
    """
    A parsed transaction.

    ``moves`` is a list of ``(mint, owner, pre_amount, post_amount)``. Pre and
    post are both explicit because a DEBIT only exists as a difference: a
    transfer that sets post to zero while pre is also zero moves nothing, and a
    verifier keyed on post-amounts alone would be reading the wrong thing. The
    destination's movement, by contrast, has pre=0 and post=amount, which is what
    a real credit looks like.
    """
    pre = [{"accountIndex": 10 + i, "mint": m, "owner": o,
            "uiTokenAmount": {"amount": str(a), "decimals": "6"}}
           for i, (m, o, a, _b) in enumerate(moves)]
    post = [{"accountIndex": 10 + i, "mint": m, "owner": o,
             "uiTokenAmount": {"amount": str(b), "decimals": "6"}}
            for i, (m, o, _a, b) in enumerate(moves)]
    balances_pre = [0] * 20
    balances_post = [0] * 20
    if wallet_lamport_delta:
        balances_pre[1] = 1_000_000_000
        balances_post[1] = 1_000_000_000 + wallet_lamport_delta
    # The signer MUST appear in the account keys at index 1, because that is
    # how a native lamport movement is attributed to it rather than to a pool or
    # a fee payer. Index 0 is the fee payer, which is a different party in the
    # $SI incident and whose fee must not be counted as our proceeds.
    keys = [{"pubkey": "FeePayer111"}, {"pubkey": wallet}]
    keys += [{"pubkey": f"k{i}"} for i in range(18)]
    return {
        "meta": {
            "err": err,
            "fee": 15000,
            "preTokenBalances": pre,
            "postTokenBalances": post,
            "preBalances": balances_pre,
            "postBalances": balances_post,
        },
        "transaction": {"message": {"accountKeys": keys}},
    }


class _Chain:
    """A ChainFacts whose every answer is a fixture."""

    def __init__(self, *, balance=0, status=None, tx=None, raise_on=None):
        self.balance = balance
        self.status = status
        self.tx = tx
        self.raise_on = raise_on

    async def token_balance(self, owner, mint):
        if self.raise_on == "balance":
            raise RuntimeError("rpc down")
        return self.balance

    async def get_transaction(self, signature):
        if self.raise_on == "tx":
            raise RuntimeError("rpc down")
        return self.tx

    async def signature_status(self, signature):
        if self.raise_on == "status":
            raise RuntimeError("rpc down")
        return self.status


_CONFIRMED = {"err": None, "confirmationStatus": "confirmed"}


def _si_transfer_tx(*, debit_owner=WALLET, credit_owner=SI_DEST_OWNER,
                     amount=SI_TOKEN_AMOUNT):
    return _tx([(SI_MINT, debit_owner, amount, 0),
                (SI_MINT, credit_owner, 0, amount)])


def _verify(chain, **kw):
    from autopaper.external_close import verify_external_close
    return asyncio.run(verify_external_close(
        chain, wallet=WALLET, mint=SI_MINT, signature=SI_EXTERNAL_SIG,
        expected_amount=kw.pop("expected_amount", SI_TOKEN_AMOUNT), **kw))


# ===========================================================================
# 1. external close: every doubt is a refusal
# ===========================================================================


class TestExternalCloseRefusesOnDoubt:
    def test_a_wallet_that_still_holds_the_tokens_is_refused(self):
        """The position is not closed if the tokens are still there."""
        r = _verify(_Chain(balance=1, status=_CONFIRMED,
                           tx=_si_transfer_tx()))
        assert r.ok is False
        assert "NOT been closed externally" in r.reason

    def test_an_unknown_signature_is_refused(self):
        r = _verify(_Chain(balance=0, status=None))
        assert r.ok is False
        assert "not found" in r.reason

    def test_a_failed_transaction_is_refused(self):
        r = _verify(_Chain(balance=0,
                           status={"err": {"InstructionError": [0, "x"]},
                                   "confirmationStatus": "confirmed"}))
        assert r.ok is False
        assert "FAILED" in r.reason

    def test_an_unconfirmed_transaction_is_refused(self):
        r = _verify(_Chain(balance=0, status={"err": None,
                                              "confirmationStatus": "processed"}))
        assert r.ok is False
        assert "not confirmed" in r.reason

    def test_an_unreadable_balance_is_not_a_pass(self):
        r = _verify(_Chain(balance=None, status=_CONFIRMED))
        assert r.ok is False
        assert "unavailable" in r.reason

    def test_an_error_while_reading_is_not_a_pass(self):
        """A failed lookup is an unknown, not an absence."""
        r = _verify(_Chain(raise_on="balance"))
        assert r.ok is False
        assert "unreadable" in r.reason

    def test_a_missing_transaction_body_is_refused(self):
        r = _verify(_Chain(balance=0, status=_CONFIRMED, tx=None))
        assert r.ok is False
        assert "unavailable" in r.reason

    def test_a_partial_token_delta_is_refused(self):
        """A transaction that moved only part of the position does not close it."""
        r = _verify(_Chain(balance=0, status=_CONFIRMED,
                           tx=_si_transfer_tx(amount=12_345)))
        assert r.ok is False
        assert "does not equal the booked position amount" in r.reason
        assert r.debited_amount == 12_345
        assert r.expected_amount == SI_TOKEN_AMOUNT

    def test_a_debit_from_someone_elses_account_is_refused(self):
        """
        The chain must show OUR tokens leaving OUR account.

        A transfer of the same quantity from a third party's token account is
        not this position being closed, and recording it as closed would erase a
        position that is still held.
        """
        r = _verify(_Chain(balance=0, status=_CONFIRMED,
                           tx=_si_transfer_tx(debit_owner="SomebodyElse")))
        assert r.ok is False
        assert "owned by this wallet" in r.reason

    def test_a_debit_with_no_matching_credit_is_refused(self):
        """A debit with no destination is not a settlement."""
        tx = _tx([(SI_MINT, WALLET, SI_TOKEN_AMOUNT, 0),
                  (SI_MINT, SI_DEST_OWNER, 0, 1)])
        r = _verify(_Chain(balance=0, status=_CONFIRMED, tx=tx))
        assert r.ok is False
        assert "no matching destination" in r.reason


class TestExternalCloseAcceptsOnlyTheVerifiedIncident:
    def test_the_real_incident_transaction_is_accepted(self):
        r = _verify(_Chain(balance=0, status=_CONFIRMED,
                           tx=_si_transfer_tx()))
        assert r.ok is True, r.reason
        assert r.debited_amount == SI_TOKEN_AMOUNT
        assert r.destination_accounts

    def test_a_pure_transfer_credits_nothing(self):
        """
        A transfer that pays nothing is recorded as paying nothing.

        The wallet's native lamports were unchanged and no token was credited to
        it, so there are no credits at all. Inventing a sale price here would
        book a gain or a break-even on a position that was simply handed to
        someone else.
        """
        r = _verify(_Chain(balance=0, status=_CONFIRMED, tx=_si_transfer_tx()))
        assert r.ok is True
        assert r.proceeds_sol == 0.0
        assert r.credited_assets == ()
        assert r.settlement == "NO_PROCEEDS"

    def test_measured_native_proceeds_are_taken_from_the_chain(self):
        """If the wallet really was paid, that amount is used."""
        tx = _si_transfer_tx()
        tx["meta"]["preBalances"][1] = 1_000_000_000
        tx["meta"]["postBalances"][1] = 1_900_000_000
        r = _verify(_Chain(balance=0, status=_CONFIRMED, tx=tx))
        assert r.ok is True
        assert r.proceeds_sol == pytest.approx(0.9)
        assert r.settlement == "CONTROLLED"


class TestApplyExternalClose:
    def _pos(self):
        from autopaper.real_account import RealAccount
        acct = RealAccount()
        acct.register_entry(
            mint=SI_MINT, symbol="SI", fill=_entry_fill(),
            token_amount=SI_TOKEN_AMOUNT, entry_price_usd=0.000_000_0281,
            size_usd=SI_ENTRY_COST_USD, now=1_000.0, sol_price_usd=117.68,
        )
        return acct.positions[SI_MINT]

    def test_a_zero_proceeds_closure_is_a_total_loss_stated_plainly(self):
        """
        No proceeds means the whole cost basis is gone, and that is what it says.

        Not a break-even, not a partial recovery: a position transferred away
        with nothing received back has realized a loss equal to its cost, and
        recording anything else invents a result.
        """
        from autopaper.external_close import apply_external_close
        r = _verify(_Chain(balance=0, status=_CONFIRMED,
                           tx=_si_transfer_tx()))
        pos = self._pos()
        out = apply_external_close(pos, r, proceeds_usd=0.0, now=2000.0)

        from autopaper.real_account import cost_basis
        assert out["realized_pnl_usd"] == pytest.approx(-cost_basis(pos))
        assert out["proceeds_sol"] == 0.0
        assert out["exit_reason"] if "exit_reason" in out else True
        assert pos.status == "CLOSED"
        assert pos.exit_reason == "EXTERNALLY_CLOSED"
        assert pos.exit is not None
        assert pos.exit.side == "EXTERNAL", (
            "an external transfer must never look like a swap we broadcast")
        assert pos.exit.execution_price is None, (
            "no exit price may be invented for a transfer")
        assert "no asset was credited" in pos.ambiguity_reason

    def test_a_non_sol_credit_is_valued_and_realized(self):
        """
        THE $SI LESSON, AS A TEST.

        The reconciliation of 2026-10-02 first recorded this position as a
        "TOTAL LOSS of its cost basis". That was wrong. The chain shows the
        counterparty sent 0.170950 USDC into this wallet in the same transaction
        that removed the tokens -- a P2P settlement, not a giveaway. A path that
        measured only native lamports saw nothing arrive and asserted a loss on a
        trade that had actually been PAID.

        So proceeds are measured across every credited asset, and a profitable
        settlement books a gain.
        """
        from autopaper.external_close import apply_external_close
        tx = _tx([(SI_MINT, WALLET, SI_TOKEN_AMOUNT, 0),
                  (SI_MINT, SI_DEST_OWNER, 0, SI_TOKEN_AMOUNT),
                  (USDC_MINT, WALLET, 0, 170_950)])
        r = _verify(_Chain(balance=0, status=_CONFIRMED, tx=tx))
        assert r.ok is True
        assert r.proceeds_sol == 0.0, "no native SOL arrived"
        usdc = [c for c in r.credited_assets if c.mint == USDC_MINT]
        assert usdc, "the USDC credit must be measured, not ignored"
        assert usdc[0].amount == 170_950
        assert usdc[0].ui_amount == pytest.approx(0.170950)
        assert r.settlement == "SETTLED_NON_SOL"

        pos = self._pos()
        from autopaper.real_account import cost_basis
        out = apply_external_close(pos, r, proceeds_usd=0.170950,
                                   asset_prices_usd={USDC_MINT: 1.0},
                                   now=2000.0)
        assert out["pnl_stated"] is True
        assert out["realized_pnl_usd"] == pytest.approx(
            0.170950 - cost_basis(pos))
        assert out["realized_pnl_usd"] > 0, (
            "this settlement was a GAIN; recording it as a total loss is the "
            "bug this test exists to prevent")
        assert "credited" in pos.ambiguity_reason

    def test_an_unpriceable_credit_leaves_the_pnl_unstated(self):
        """
        Value received, price unreadable -> the PnL is not stated.

        Not zero, and not a loss. Both of those are guesses, and the whole point
        of the fail-closed rule is that a guess must not reach the ledger. The
        position still CLOSES, because the tokens really are gone.
        """
        from autopaper.external_close import apply_external_close
        tx = _tx([(SI_MINT, WALLET, SI_TOKEN_AMOUNT, 0),
                  (SI_MINT, SI_DEST_OWNER, 0, SI_TOKEN_AMOUNT),
                  (USDC_MINT, WALLET, 0, 170_950)])
        r = _verify(_Chain(balance=0, status=_CONFIRMED, tx=tx))
        pos = self._pos()
        out = apply_external_close(pos, r, proceeds_usd=None,
                                   asset_prices_usd={}, now=2000.0)
        assert pos.status == "CLOSED", "the sale is a fact regardless of price"
        assert out["pnl_stated"] is False
        assert out["realized_pnl_usd"] is None
        assert pos.realized_pnl_usd is None
        assert "NOT STATED" in pos.ambiguity_reason

    def test_the_audit_trail_records_the_external_signature(self):
        from autopaper.external_close import apply_external_close
        r = _verify(_Chain(balance=0, status=_CONFIRMED,
                           tx=_si_transfer_tx()))
        pos = self._pos()
        apply_external_close(pos, r, proceeds_usd=0.0, now=2000.0)
        assert pos.exit.signature == SI_EXTERNAL_SIG

    def test_an_unverified_closure_cannot_be_applied(self):
        from autopaper.external_close import (
            ExternalCloseRefused, apply_external_close)
        r = _verify(_Chain(balance=1, status=_CONFIRMED))
        pos = self._pos()
        with pytest.raises(ExternalCloseRefused):
            apply_external_close(pos, r, proceeds_usd=0.0)
        assert pos.status == "OPEN", "a refused closure must not mutate state"

    def test_a_closed_position_cannot_be_closed_twice(self):
        from autopaper.external_close import (
            ExternalCloseRefused, apply_external_close)
        r = _verify(_Chain(balance=0, status=_CONFIRMED,
                           tx=_si_transfer_tx()))
        pos = self._pos()
        apply_external_close(pos, r, proceeds_usd=0.0, now=2000.0)
        with pytest.raises(ExternalCloseRefused):
            apply_external_close(pos, r, proceeds_usd=0.0, now=2001.0)


def _entry_fill():
    from autopaper.real_account import RealFillRecord
    return RealFillRecord(
        signature="ENTRY-SIG", side="ENTRY", commitment="finalized",
        slot=1, confirmed_at=1_000.0, requested_lamports=970_000,
        actual_input_amount=970_000, actual_output_amount=SI_TOKEN_AMOUNT,
        fee_lamports=15000,
    )


# ===========================================================================
# 2. settlement: what did the exit ACTUALLY return?
# ===========================================================================


class TestSettlementClassification:
    def _c(self, tx, lam=0, wallet=WALLET):
        from autopaper.settlement import classify_settlement
        return classify_settlement(tx, wallet=wallet,
                                   native_lamports_delta=lam)

    def test_native_sol_to_the_signer_is_the_clean_case(self):
        from autopaper.settlement import Settlement
        r = self._c(_tx([(WSOL_MINT, WALLET, 0, 1)], wallet_lamport_delta=1_500_000_000),
                    lam=1_500_000_000)
        assert r.settlement == Settlement.SETTLED_SOL
        assert r.is_settled_sol and not r.is_problem
        assert r.output_amount == 1_500_000_000

    def test_wsol_settles_as_sol(self):
        from autopaper.settlement import Settlement
        r = self._c(_tx([(WSOL_MINT, WALLET, 0, 2_000_000_000)]))
        assert r.settlement == Settlement.SETTLED_SOL

    def test_a_usdc_payout_is_not_recorded_as_sol(self):
        """
        THE CASE THIS EXISTS FOR. A sell that returned USDC was previously
        booked as if it had returned SOL: the position closed, proceeds were
        attributed, and the money was simultaneously missing and marked
        received.
        """
        from autopaper.settlement import Settlement
        r = self._c(_tx([(USDC_MINT, WALLET, 0, 25_000_000)]))
        assert r.settlement == Settlement.SETTLED_SPL_NEEDS_NORMALIZATION
        assert r.output_mint == USDC_MINT
        assert r.output_amount == 25_000_000
        assert r.needs_normalization
        assert r.is_problem is False, (
            "the money IS ours; it is unfinished, not broken")

    def test_money_credited_to_another_account_is_never_proceeds(self):
        from autopaper.settlement import Settlement
        r = self._c(_tx([(USDC_MINT, "SomeOtherAccount", 0, 25_000_000)]))
        assert r.settlement == Settlement.UNCONTROLLED
        assert r.controlled is False
        assert r.is_problem
        assert "not spendable" in r.reason

    def test_nothing_arriving_is_not_a_sale(self):
        from autopaper.settlement import Settlement
        r = self._c(_tx([(WSOL_MINT, WALLET, 0, 0)]))
        assert r.settlement == Settlement.NO_PROCEEDS
        assert r.is_problem

    def test_an_unreadable_transaction_is_not_assumed_to_be_sol(self):
        """
        The two guesses have opposite consequences.

        Assuming SOL books money that is not there; assuming nothing reports a
        successful exit as unaccounted. Neither is safe, so the honest answer is
        UNREADABLE and the operator is told.
        """
        from autopaper.settlement import Settlement
        r = self._c(None)
        assert r.settlement == Settlement.UNREADABLE
        assert r.is_problem

    def test_a_failed_transaction_is_not_a_settlement(self):
        from autopaper.settlement import Settlement
        r = self._c(_tx([(USDC_MINT, WALLET, 0, 1)],
                        err={"InstructionError": [0, "boom"]}))
        assert r.settlement == Settlement.UNREADABLE


class TestNormalizationIsASeparateOrder:
    def test_a_usdc_settlement_permits_a_followup_conversion(self):
        from autopaper.settlement import (
            classify_settlement, normalize_intent_for)
        r = classify_settlement(_tx([(USDC_MINT, WALLET, 0, 25_000_000)]),
                                wallet=WALLET)
        i = normalize_intent_for(r, source_mint=r.output_mint,
                                 amount=r.output_amount)
        assert i is not None
        assert i["input_mint"] == USDC_MINT
        assert i["output_mint"] == WSOL_MINT
        assert i["input_amount"] == 25_000_000
        assert i["kind"] == "SETTLEMENT_NORMALIZATION"

    def test_an_unrecognised_asset_is_never_auto_swapped(self):
        """
        Automatically swapping a token nobody approved is how a bot ends up
        trading something the operator never agreed to. So the normalizable set
        is a closed allow-list and everything else goes to the operator.
        """
        from autopaper.settlement import (
            classify_settlement, normalize_intent_for)
        r = classify_settlement(_tx([("SomeUnknownMint", WALLET, 0, 7)]),
                                wallet=WALLET)
        assert r.needs_normalization is True, "it is still reported"
        assert normalize_intent_for(r, source_mint=r.output_mint,
                                    amount=r.output_amount) is None, (
            "but it is refused for automatic conversion")

    def test_a_clean_sol_settlement_needs_no_conversion(self):
        from autopaper.settlement import (
            classify_settlement, normalize_intent_for)
        r = classify_settlement(_tx([(WSOL_MINT, WALLET, 0, 1)],
                                    wallet_lamport_delta=1_000),
                                wallet=WALLET, native_lamports_delta=1_000)
        assert normalize_intent_for(r, source_mint=WSOL_MINT,
                                    amount=1) is None


# ===========================================================================
# 3. booking a non-SOL exit honestly
# ===========================================================================


class TestClosePositionSettlementAware:
    def _account(self):
        from autopaper.real_account import RealAccount
        acct = RealAccount()
        acct.register_entry(
            mint=SI_MINT, symbol="SI", fill=_entry_fill(),
            token_amount=SI_TOKEN_AMOUNT, entry_price_usd=0.000_000_0281,
            size_usd=SI_ENTRY_COST_USD, now=1_000.0, sol_price_usd=117.68,
        )
        return acct

    def _sell_fill(self, output_amount):
        from autopaper.real_account import RealFillRecord
        return RealFillRecord(
            signature="SELL-SIG", side="EXIT", commitment="confirmed",
            slot=2, confirmed_at=2_000.0, requested_lamports=SI_TOKEN_AMOUNT,
            actual_input_amount=SI_TOKEN_AMOUNT,
            actual_output_amount=output_amount, fee_lamports=15000,
        )

    def test_a_sol_settlement_behaves_exactly_as_before(self):
        from autopaper.real_account import RealAccount
        acct = self._account()
        p = acct.close_position(
            mint=SI_MINT, fill=self._sell_fill(500_000_000),
            exit_price_usd=0.000_000_0281, reason="tp", now=2_000.0,
            sol_price_usd=117.68)
        from autopaper.real_account import cost_basis
        assert p.status == "CLOSED"
        assert p.realized_pnl_usd == pytest.approx(0.5 * 117.68 - cost_basis(p))

    def test_a_priced_usdc_settlement_realizes_pnl_in_that_asset(self):
        acct = self._account()
        p = acct.close_position(
            mint=SI_MINT, fill=self._sell_fill(25_000_000),
            exit_price_usd=0.000_000_0281, reason="tp", now=2_000.0,
            sol_price_usd=117.68, output_mint=USDC_MINT, output_decimals=6,
            output_usd_price=1.0)
        from autopaper.real_account import cost_basis
        assert p.status == "CLOSED"
        assert p.realized_pnl_usd == pytest.approx(25.0 - cost_basis(p))

    def test_an_unpriceable_non_sol_settlement_states_no_pnl(self):
        """
        The sale is real, so the position CLOSES; the P&L is simply not stated.

        Refusing to book would leave the position OPEN forever while its tokens
        are gone. Computing a P&L from an unobserved price would fabricate a
        result that later reads as a measurement. Neither is done: the trade is
        recorded and the number is withheld.
        """
        acct = self._account()
        p = acct.close_position(
            mint=SI_MINT, fill=self._sell_fill(25_000_000),
            exit_price_usd=0.000_000_0281, reason="tp", now=2_000.0,
            sol_price_usd=117.68, output_mint=USDC_MINT, output_decimals=6,
            output_usd_price=None)
        assert p.status == "CLOSED", "a confirmed sale must be booked"
        assert p.realized_pnl_usd is None
        assert "not stated" in p.ambiguity_reason
        assert acct.realized_pnl_usd == 0.0, (
            "an unstated P&L must not be silently added to the total")

    def test_a_sol_settlement_still_requires_a_price(self):
        acct = self._account()
        with pytest.raises(ValueError, match="SOL/USD price"):
            acct.close_position(
                mint=SI_MINT, fill=self._sell_fill(500_000_000),
                exit_price_usd=0.000_000_0281, reason="tp", now=2_000.0,
                sol_price_usd=None)


# ===========================================================================
# 4. a blocked exit must be visible, and must not fake an intent
# ===========================================================================


class TestExitBlockedTelemetry:
    def _trader(self):
        from autopaper.config import build_config
        from autopaper.engine import PaperTrader
        return PaperTrader(build_config())

    def test_a_blocked_exit_notifies_with_its_occurrence_count(self):
        t = self._trader()
        sent = []
        t._notify = lambda text, *a, **k: sent.append(text)
        for _ in range(3):
            t._record_sell_block("SI", SI_MINT, "stale_market_data", "detail")
        assert len(sent) == 1, "the first occurrence notifies"
        assert "Occurrence: 1" in sent[0]
        assert t._sell_block_counts[(SI_MINT, "stale_market_data")] == 3

    def test_a_stuck_exit_keeps_announcing_itself(self):
        """
        One transient refusal and a permanently stuck exit produced identical
        silence. The count is what distinguishes them, so escalation is required
        and the number shown is always the TRUE count, not the notification
        count.
        """
        from autopaper.engine import _SELL_BLOCK_REPEAT_EVERY
        t = self._trader()
        sent = []
        t._notify = lambda text, *a, **k: sent.append(text)
        for _ in range(_SELL_BLOCK_REPEAT_EVERY * 3):
            t._record_sell_block("SI", SI_MINT, "stale_market_data", "d")
        assert len(sent) == 4, "first, then every Nth"
        assert f"Occurrence: {_SELL_BLOCK_REPEAT_EVERY * 3}" in sent[-1]
        assert t._sell_block_counts[(SI_MINT, "stale_market_data")] == 30

    def test_a_blocked_exit_creates_no_execution_intent(self):
        """
        A blocked exit broadcast nothing, so it must not leave an intent behind.

        An intent with no transaction is what makes the NEXT run believe an
        order is already in flight and refuse to try -- a self-inflicted
        deadlock on top of the original problem.
        """
        t = self._trader()
        t._notify = lambda *a, **k: None
        for _ in range(25):
            t._record_sell_block("SI", SI_MINT, "stale_market_data", "d")
        assert t.execution_intents.has_unresolved(mint=SI_MINT) is False

    def test_the_notified_reason_is_sanitized(self):
        from autopaper.engine import _sanitize_reason
        # a base64 transaction body must not reach a chat message
        assert "[redacted]" in _sanitize_reason(
            "failed tx " + "A" * 88)
        # a signed URL must not reach a chat message
        assert "[redacted]" in _sanitize_reason(
            "GET https://api.example.com/swap?key=abc123 failed")
        # a secret must not reach a chat message, not even truncated
        out = _sanitize_reason("api_key=sk-live-0123456789abcdef")
        assert "sk-live" not in out
        # newlines must not break the message layout
        assert "\n" not in _sanitize_reason("boom\nTraceback:\n  File x")
        assert len(_sanitize_reason("y" * 900)) <= 180


# ===========================================================================
# 5. status must distinguish armed from running
# ===========================================================================


class TestStatusSaysWhichRealState:
    def test_real_armed_and_running_reads_as_live(self):
        from autopaper.modes import RunState, RuntimeState, status_line
        s = RuntimeState()
        s.real_armed = True
        s.run_state = RunState.REAL_ARMED
        line = status_line(s)
        assert "REAL RUNNING" in line
        assert "NOT RUNNING" not in line

    def test_real_armed_but_stopped_is_not_reported_as_paper(self):
        """
        THE DANGEROUS CASE.

        ``trade_mode`` is derived and returns PAPER whenever the loop is not
        actively running in REAL_ARMED. So a stopped-but-armed account rendered
        as "PAPER", which reads as "no real authority here" while REAL is in fact
        still authorized and will trade the instant the loop restarts.
        """
        from autopaper.modes import RunState, RuntimeState, TradeMode, status_line
        s = RuntimeState()
        s.real_armed = True
        s.run_state = RunState.STOPPED
        assert s.trade_mode is TradeMode.PAPER, (
            "the derived mode is exactly the source of the ambiguity")
        line = status_line(s)
        assert "REAL ARMED, NOT RUNNING" in line
        assert "still AUTHORIZED" in line
        assert "PAPER" not in line

    def test_paper_running_and_paper_off_are_distinct(self):
        from autopaper.modes import RunState, RuntimeState, status_line
        s = RuntimeState()
        s.run_state = RunState.PAPER
        assert "PAPER RUNNING" in status_line(s)
        s.run_state = RunState.OFF
        assert "PAPER OFF" in status_line(s)

    def test_an_unarmed_account_never_mentions_real_running(self):
        from autopaper.modes import RunState, RuntimeState, status_line
        s = RuntimeState()
        s.run_state = RunState.PAPER
        assert "REAL" not in status_line(s)


class TestUnmeasurableEquityRendersAsUnavailable:
    def test_unknown_equity_never_invents_an_allocation_or_a_notional(self):
        """
        ``compute_position_size(None)`` did ``float(None)`` and raised, so a REAL
        wallet whose equity could not be valued would crash the status message.
        It must render as unavailable, and must not print a position size
        derived from no equity.
        """
        from autopaper.sizing import allocation_label, next_position_label
        assert "unavailable" in allocation_label(None)
        assert "unavailable" in next_position_label(None)
        assert "$" not in next_position_label(None)

    def test_a_measured_zero_still_renders_as_zero(self):
        """None and 0.0 must stay distinguishable."""
        from autopaper.sizing import allocation_label
        assert allocation_label(0.0) != allocation_label(None)
        assert "10%" in allocation_label(0.0)

    def test_the_account_aggregates_must_agree_with_the_row(self):
        """
        A ledger whose rows disagree with its total is worse than no ledger.

        The first version applied the external closure straight to the position
        object, so the row reported +$0.0567 while ``realized_pnl_usd`` said
        $0.00 and ``closed_count`` stayed 0. Every report that reads the total
        was quietly wrong, and nothing in the code would have complained.
        """
        from autopaper.real_account import RealAccount
        acct = RealAccount()
        acct.register_entry(
            mint=SI_MINT, symbol="SI", fill=_entry_fill(),
            token_amount=SI_TOKEN_AMOUNT, entry_price_usd=0.000_000_0281,
            size_usd=SI_ENTRY_COST_USD, now=1_000.0, sol_price_usd=117.68,
        )
        tx = _tx([(SI_MINT, WALLET, SI_TOKEN_AMOUNT, 0),
                  (SI_MINT, SI_DEST_OWNER, 0, SI_TOKEN_AMOUNT),
                  (USDC_MINT, WALLET, 0, 170_950)])
        r = _verify(_Chain(balance=0, status=_CONFIRMED, tx=tx))

        pos, out = acct.close_externally(
            SI_MINT, r, proceeds_usd=0.170922,
            asset_prices_usd={USDC_MINT: 0.999836}, now=2_000.0)

        assert pos.status == "CLOSED"
        assert acct.closed_count == 1
        assert acct.realized_pnl_usd == pytest.approx(pos.realized_pnl_usd), (
            "the account total must equal the row it aggregates")
        assert acct.realized_pnl_usd == pytest.approx(out["realized_pnl_usd"])

    def test_an_unstated_pnl_does_not_enter_the_account_total(self):
        """An unstated P&L is not zero, so it must not be added as zero."""
        from autopaper.real_account import RealAccount
        acct = RealAccount()
        acct.register_entry(
            mint=SI_MINT, symbol="SI", fill=_entry_fill(),
            token_amount=SI_TOKEN_AMOUNT, entry_price_usd=0.000_000_0281,
            size_usd=SI_ENTRY_COST_USD, now=1_000.0, sol_price_usd=117.68,
        )
        tx = _tx([(SI_MINT, WALLET, SI_TOKEN_AMOUNT, 0),
                  (SI_MINT, SI_DEST_OWNER, 0, SI_TOKEN_AMOUNT),
                  (USDC_MINT, WALLET, 0, 170_950)])
        r = _verify(_Chain(balance=0, status=_CONFIRMED, tx=tx))
        _pos, out = acct.close_externally(SI_MINT, r, proceeds_usd=None,
                                         asset_prices_usd={}, now=2_000.0)
        assert out["pnl_stated"] is False
        assert acct.realized_pnl_usd == 0.0
        assert acct.closed_count == 1


# ===========================================================================
# 6. a REFUSED broadcast is not an in-flight broadcast
# ===========================================================================


class TestSendFailureClassification:
    """
    ``UNKNOWN`` must mean "may be executing", not "something went wrong".

    An intent in UNKNOWN blocks EVERY REAL entry, process-wide. So a state that
    is actually safe must never be recorded as one that is dangerous, or the bot
    stops trading for a transaction that provably does not exist.
    """

    def test_a_node_preflight_refusal_is_provably_not_submitted(self):
        """
        THE 8HJY9 INCIDENT.

        ``skipPreflight: False`` means the node simulates the transaction before
        submitting it. RPC -32002 is that simulation failing, so the node
        refused to submit. One such intent blocked twenty REAL entries over three
        hours while the mint's token account did not exist on chain.
        """
        from autopaper.execution.solana import (
            RpcError, send_failure_may_have_submitted)
        exc = RpcError(
            "sendTransaction: RPC error -32002: Transaction simulation failed",
            code=-32002)
        assert send_failure_may_have_submitted(exc) is False, (
            "a node-side refusal with no signature is proof of non-submission")

    def test_a_lost_response_stays_unknown(self):
        """
        The genuinely ambiguous case, and the one UNKNOWN exists for.

        If the HTTP response is lost, the node may have submitted the
        transaction before the reply vanished. Releasing that intent would allow
        a second real order for a position that may already be open.
        """
        from autopaper.execution.solana import (
            RpcError, send_failure_may_have_submitted)
        for exc in (
            RpcError("sendTransaction: RPC timed out"),
            RpcError("sendTransaction: RPC transport error"),
            RpcError("sendTransaction: RPC HTTP 502"),
            RpcError("sendTransaction: RPC returned non-JSON"),
        ):
            assert send_failure_may_have_submitted(exc) is True

    def test_an_unrecognised_code_is_not_guessed(self):
        """
        The allow-list is closed on purpose.

        A coded error we do not understand is reported as possibly-submitted. The
        costs are asymmetric -- a wrong release risks a duplicate real order, a
        wrong hold costs a delay an operator can clear by hand -- so only
        well-understood refusals are trusted.
        """
        from autopaper.execution.solana import (
            RpcError, send_failure_may_have_submitted)
        assert send_failure_may_have_submitted(RpcError("x", code=-32000)) is True

    def test_a_non_rpc_exception_is_unknown(self):
        from autopaper.execution.solana import send_failure_may_have_submitted
        assert send_failure_may_have_submitted(ValueError("boom")) is True

    def test_a_node_refusal_becomes_a_terminal_intent_not_a_stuck_one(self):
        """End to end through the store: FAILED is terminal, UNKNOWN is not."""
        from autopaper.execution.intents import (
            UNRESOLVED_STATES, ExecutionIntentStore,
        )
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            store = ExecutionIntentStore(Path(td) / "i.jsonl")
            i = store.reserve(logical_key="ENTRY:" + SI_MINT, side="ENTRY",
                              mint=SI_MINT, requested_amount=1)
            # the real sequence the broadcast path walks
            store.transition(i.execution_id, "READY",
                             detail="raptor quote accepted")
            store.transition(i.execution_id, "SIGNING", detail="signing begins")
            store.transition(i.execution_id, "BROADCASTED",
                             detail="broadcast attempted")
            store.transition(i.execution_id, "FAILED",
                             detail="BROADCAST_REJECTED: RPC error -32002")
            cur = store.get(i.execution_id)
            assert cur.state == "FAILED"
            assert cur.state not in UNRESOLVED_STATES
            assert store.has_unresolved() is False, (
                "a provably unsubmitted order must not block the next entry")
