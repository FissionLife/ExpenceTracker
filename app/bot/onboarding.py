"""First contact: a new user is asked ONE question (where do you usually pay from?) and, optionally, a second
(how much is in it?). Everything else is learned from how they use the bot.

Never a trap: any message that is not an answer simply moves on (the first account is then chosen for them),
and an entry typed as the very first message is not lost, it is logged right after the answer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, select

from app.bot.textutil import parse_amount
from app.bot.ui import Btn, Reply
from app.config import settings
from app.db import SessionLocal
from app.models import Transaction, User
from app.services import ledger

STATE_TTL = timedelta(hours=1)
FALLBACK_ACCOUNT = "Bank"
OPENING_CATEGORY = "Opening Balance"
ADJUSTMENT_CATEGORY = "Balance Adjustment"

# What the first-question buttons and typed answers mean
CHOICES = {"bank": "Bank", "cash": "Cash", "upi": "UPI Wallet", "card": "Credit Card"}
_WORDS = {
    "bank": "bank",
    "debit": "bank",
    "account": "bank",
    "savings": "bank",
    "hdfc": "bank",
    "sbi": "bank",
    "icici": "bank",
    "axis": "bank",
    "kotak": "bank",
    "cash": "cash",
    "upi": "upi",
    "gpay": "upi",
    "phonepe": "upi",
    "paytm": "upi",
    "wallet": "upi",
    "card": "card",
    "credit": "card",
}

WELCOME = (
    "👋 Hi, I'm WalletLedger. Tell me what you spend and earn in plain words and I keep the books.\n\n"
    "One quick question so things land in the right place: where do you usually pay from?"
)


@dataclass
class State:
    step: str  # "account" | "balance"
    account: str = ""
    held: str | None = None  # the message they sent first, logged once the account is known
    at: datetime | None = None


_STATES: dict[str, State] = {}


def _account_rows() -> list[list[Btn]]:
    return [
        [Btn("🏦 Bank", "ob:acct:bank"), Btn("💵 Cash", "ob:acct:cash")],
        [Btn("📱 UPI / wallet", "ob:acct:upi"), Btn("💳 Credit card", "ob:acct:card")],
    ]


def _state(chat_id: str) -> State | None:
    state = _STATES.get(chat_id)
    if state and state.at and datetime.utcnow() - state.at > STATE_TTL:
        _STATES.pop(chat_id, None)
        return None
    return state


def enabled() -> bool:
    """ONBOARDING=false turns the first-contact questions off. The test suite runs with them off."""
    if settings.onboarding is not None:
        return settings.onboarding
    return not os.environ.get("WALLETLEDGER_TESTING")


def is_new_user(chat_id: str) -> bool:
    """No default account chosen yet and nothing ever recorded."""
    if not enabled():
        return False
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.telegram_chat_id == chat_id))
        if user is None:
            return True
        if user.default_account:
            return False
        has_any = db.scalar(
            select(func.count()).select_from(Transaction).where(Transaction.user_id == user.id)
        )
        return not has_any


def pick_account(text: str) -> str | None:
    """ "bank", "my hdfc account", "cash" -> a short account name. None if the text names none."""
    words = re.findall(r"[a-z]+", (text or "").lower())
    if not words or len(words) > 5:
        return None
    for word in words:
        kind = _WORDS.get(word)
        if kind:
            return CHOICES[kind]
    return None


def _set_default(chat_id: str, account: str) -> str:
    with SessionLocal() as db:
        user = ledger.get_or_create_user(db, chat_id)
        ledger.get_or_create_account(db, user.id, account)
        user.default_account = account
        db.commit()
    return account


def default_account(chat_id: str) -> str | None:
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.telegram_chat_id == chat_id))
        return user.default_account if user else None


async def needs_setup(chat_id: str) -> bool:
    from asyncio import to_thread

    return await to_thread(is_new_user, chat_id)


async def start(chat_id: str, held: str | None = None) -> Reply:
    """Greet a new user and ask where they pay from."""
    _STATES[chat_id] = State("account", held=held, at=datetime.utcnow())
    text = (
        WELCOME
        if not held
        else "👋 Hi! Got it, I'll log that in a second.\n\nFirst, where do you usually pay from?"
    )
    return Reply(text, rows=_account_rows())


async def intercept(chat_id: str, text: str) -> Reply | None:
    """Called for every non-command message. A Reply if onboarding handled it, else None (carry on as usual)."""
    state = _state(chat_id)
    if state is None:
        if not await needs_setup(chat_id):
            return None
        held = text if await _looks_like_entry(chat_id, text) else None
        return await start(chat_id, held)

    if state.step == "account":
        account = pick_account(text)
        if account is None:
            # not an answer: choose for them, forget the question, handle the message normally.
            # What they sent first is logged too, so nothing they typed is lost.
            _STATES.pop(chat_id, None)
            _set_default(chat_id, FALLBACK_ACCOUNT)
            if state.held:
                from app.bot import main as bot_main

                earlier = await bot_main.handle_incoming(chat_id, state.held)
                now = await bot_main.handle_incoming(chat_id, text)
                return Reply(f"{earlier.text}\n\n{now.text}", rows=now.rows or earlier.rows)
            return None
        return await answered(chat_id, account)

    # step == "balance"
    amount = parse_amount(text)
    if amount is None and re.fullmatch(
        r"\s*(skip|no|none|later|nah|zero|0)\s*[.!]?\s*", text or "", re.I
    ):
        _STATES.pop(chat_id, None)
        return Reply(_ready(state.account))
    if amount is None:
        _STATES.pop(chat_id, None)  # something else entirely: not a trap, just move on
        return None
    return await set_balance(chat_id, state.account, amount)


async def _looks_like_entry(chat_id: str, text: str) -> bool:
    """Any message with a number is held and logged once setup is done."""
    return bool(re.search(r"\d", text or ""))


async def answered(chat_id: str, account: str) -> Reply:
    """The account answer (typed or tapped): remember it, then log the held message or ask for a balance."""
    from asyncio import to_thread

    state = _STATES.pop(chat_id, None) or State("account")
    await to_thread(_set_default, chat_id, account)
    if state.held:
        from app.bot import main as bot_main

        logged = await bot_main.handle_incoming(chat_id, state.held)
        intro = f'✅ {account} is your main account (change it any time: "use cash by default").'
        return Reply(f"{intro}\n\n{logged.text}", rows=logged.rows)
    _STATES[chat_id] = State("balance", account=account, at=datetime.utcnow())
    return Reply(
        f"✅ {account} it is.\n\nRoughly how much is in {account} right now? "
        "Type an amount (like 25000) or skip, you can always tell me later.",
        rows=[[Btn("Skip for now", "ob:skip")]],
    )


async def skipped(chat_id: str) -> Reply:
    state = _STATES.pop(chat_id, None)
    account = state.account if state else (default_account(chat_id) or FALLBACK_ACCOUNT)
    return Reply(_ready(account))


async def set_balance(chat_id: str, account: str, amount: float) -> Reply:
    from asyncio import to_thread

    def run() -> float:
        with SessionLocal() as db:
            user = ledger.get_or_create_user(db, chat_id)
            result = ledger.record_transaction(
                db,
                user.id,
                amount,
                account,
                OPENING_CATEGORY,
                None,
                "income",
                "Opening balance",
                "manual",
            )
            return result["new_balance"]

    try:
        balance = await to_thread(run)
    except ValueError as exc:
        _STATES.pop(chat_id, None)
        return Reply(f"I couldn't use that amount ({exc}). You can tell me again any time.")
    _STATES.pop(chat_id, None)
    from app.bot import fmt

    return Reply(f"👍 {account} starts at {fmt.inr(balance)}.\n\n{_ready(account)}")


def _ready(account: str) -> str:
    return (
        "You're all set. Just tell me things like:\n"
        "• coffee 120\n"
        "• got salary 50000\n"
        "• paid 450 for dinner via cash\n\n"
        f'I\'ll use {account} unless you say otherwise. Ask me anything, like "how much did I spend this week?"'
    )


def reset(chat_id: str | None = None) -> None:
    """Tests: forget every unfinished onboarding (or one chat's)."""
    if chat_id is None:
        _STATES.clear()
    else:
        _STATES.pop(chat_id, None)
