"""The agent: one LLM planner reads the message, picks ledger tools, a guard asks before risky ones.

LangGraph StateGraph (checkpointed per chat, so a confirmation can pause and resume)::

    START -> ingress --(API key pasted)-------------------------------> renderer -> END
                |
                v
             planner --(final text / AI failure / step cap)-----------> renderer
                |  (tool_calls)
                v
              guard   -- interrupt({kind: confirm, id, description}) when a call needs a Yes
                |        (declined -> renderer with "Cancelled")
                v
             executor --(read/advice tool ran, or a recoverable error)-> planner (loop)
                |
                +--(only writes / insufficient balance)----------------> renderer

- ingress: key-paste short-circuit, /coach|/budgeter|/summary persona, loads accounts, budgets,
  today and the last HISTORY_LIMIT chat messages into an OpenAI-format conversation.
- planner: one ``ai_gateway.complete`` call with the tool schema -> tool_calls or the final text.
- guard: big transfers, deletes, balance resets, edits and over-budget spends pause the graph with
  ``interrupt()``; ``resume_pending`` continues it with ``Command(resume="yes"|"no")``.
- executor: runs the calls through ``tool_client`` (user_id injected), records each result.
- renderer: planner text, else fixed templates (replies.compact / tool_summary), plus budget note.

Money rules (overdraw, amount > 0, future dates) live in the ledger, never in the prompt.
Conversation history is saved in the database (encrypted, API keys redacted).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import select

from app import clock
from app.bot import fmt, replies
from app.bot import guardrails as gr
from app.bot.categories import EXPENSE_CATEGORIES, canonical_category
from app.bot.keys import save_keys
from app.db import SessionLocal
from app.models import ChatMessage, User
from app.services.ledger import get_or_create_user
from app.services_ai import ai_gateway as gw
from app.services_ai import tool_client
from app.services_ai.ai_gateway import extract_api_keys, mask_key
from app.services_ai.prompts import build_planner_prompt, get_persona, resolve_mode

logger = logging.getLogger(__name__)

# Transfer amount threshold that requires explicit human-in-the-loop confirmation
CONFIRMATION_TRANSFER_THRESHOLD = 5000.0

# A pending confirmation older than this is discarded, so a stale "yes" can never move money.
PENDING_ACTION_TTL = timedelta(minutes=5)

# Only the most recent N prior messages are sent to the LLM as conversation history.
HISTORY_LIMIT = 20

PERSONA_COMMAND = re.compile(r"^\s*/(coach|budgeter|summary)\b(.*)$", re.I | re.S)

MAX_STEPS = 5  # planner rounds per message
MAX_TOOL_CALLS = 12  # tool calls executed per planner round
TOOL_RESULT_CHARS = 4000  # a tool result is clipped to this before it goes back to the LLM

WRITE_TOOLS = {
    "log_expense",
    "log_income",
    "transfer_funds",
    "transfer_between_pipes",
    "edit_transaction",
    "delete_transaction",
    "set_category_budget",
    "set_default_account",
    "set_account_balance",
}
HIDDEN_TOOLS = {"save_user_api_key"}  # keys are saved by ingress, never by the LLM

NEEDS_KEY_TEXT = (
    "I need an AI key to understand free-form messages, so I didn't record anything.\n"
    "Add one with /setkey (a free Gemini key takes a minute: aistudio.google.com/apikey).\n"
    "Commands like /balance, /stats and /history work without it."
)
AI_BUSY_TEXT = "The AI is busy right now; nothing was recorded. Please try again."
TOO_MANY_STEPS_TEXT = (
    "That needed too many steps, so I stopped. Try splitting it into smaller messages."
)
ERROR_TEXT = "⚠️ Something went wrong on my side. Please try again in a moment."


def _amount_of(args: Any) -> float:
    """An amount from AI output: 7000, "7000", "7,000"; anything unreadable counts as 0 (the tool then reports it)."""
    try:
        return float(str(args.get("amount", 0)).replace(",", ""))
    except TypeError, ValueError, AttributeError:
        return 0.0


def confirmation_reason(tool_name: str | None, args: Any) -> str | None:
    """A sentence describing what needs the user's Yes, or None if the call can run straight away.

    A transfer of ₹5,000 or more, any edit or delete of an existing entry, and any balance reset.
    """
    if not isinstance(args, dict):
        return None
    if tool_name in ("transfer_funds", "transfer_between_pipes") and (
        _amount_of(args) >= CONFIRMATION_TRANSFER_THRESHOLD
    ):
        amount = _amount_of(args)
        shown = f"{amount:,.0f}" if amount == int(amount) else f"{amount:,.2f}"
        return (
            f"Transfer ₹{shown} from {args.get('from_account', 'Source')} "
            f"to {args.get('to_account', 'Destination')}"
        )
    if tool_name == "edit_transaction":
        changes = [
            f"{key.replace('_', ' ')} to {args[key]}"
            for key in (
                "amount",
                "category",
                "merchant",
                "account_name",
                "transaction_type",
                "transaction_date",
                "description",
            )
            if args.get(key) not in (None, "")
        ]
        return f"Change entry {str(args.get('transaction_id', ''))[:8]}: " + (
            ", ".join(changes) or "no changes"
        )
    if tool_name == "delete_transaction":
        return f"Delete entry {str(args.get('transaction_id', ''))[:8]} and restore its balance"
    if tool_name == "set_account_balance":
        try:
            shown = fmt.inr(float(str(args.get("balance", 0)).replace(",", "")))
        except TypeError, ValueError:
            shown = str(args.get("balance"))
        return (
            f"Set {args.get('account_name', 'the account')} balance to {shown} "
            "(the difference is recorded as an adjustment)"
        )
    return None


def _budget_overage_reason(chat_id: str, call: dict[str, Any]) -> str | None:
    """A confirmation sentence if this expense would push its category over budget, else None.

    Looks up the category's monthly budget and spend so far; only an expense that crosses 100%
    (and isn't already over) asks first. History-only entries and non-expenses never ask.
    """
    if call.get("tool_name") != "log_expense":
        return None
    args = call.get("tool_args") or {}
    if args.get("affects_balance") is False:
        return None
    try:
        amount = float(str(args.get("amount", 0)).replace(",", ""))
    except TypeError, ValueError:
        return None
    category = args.get("category")
    if not category or amount <= 0 or str(category).lower() == "uncategorized":
        return None
    canon = canonical_category(str(category))
    from app.services import analytics

    try:
        with SessionLocal() as db:
            user = _find_user(db, chat_id)
            if user is None:
                return None
            row = next(
                (
                    b
                    for b in analytics.budget_status(db, user.id)
                    if b["category"].lower() == canon.lower()
                ),
                None,
            )
    except Exception:
        return None
    if not row:
        return None
    already_over = row["spent"] > row["budget"]
    if already_over or row["spent"] + amount <= row["budget"]:
        return None  # already over (don't nag twice) or still within budget
    over_by = row["spent"] + amount - row["budget"]
    return (
        f"Log {fmt.inr(amount)} on {row['category']} — this exceeds your "
        f"{fmt.inr(row['budget'])} budget by {fmt.inr(over_by)} "
        f"({fmt.inr(row['left'])} left)"
    )


def _utc_now() -> datetime:
    """Helper to return current timezone-aware UTC datetime."""
    return datetime.now(UTC)


def _find_user(db: Any, chat_id: str) -> User | None:
    return db.get(User, str(chat_id)) or db.scalar(
        select(User).where(User.telegram_chat_id == str(chat_id))
    )


def redact_api_keys(text: str) -> str:
    """Replace every recognisable API key in ``text`` with its masked form."""
    for _, key in extract_api_keys(text):
        text = text.replace(key, mask_key(key))
    return text


# ---------------------------------------------------------------------------
# Database-Backed Persistent Conversation History
# ---------------------------------------------------------------------------


def get_db_conversation_history(user_id: str, limit: int | None = None) -> list[BaseMessage]:
    """Fetch a user's stored conversation (oldest first); ``limit`` keeps only the newest N."""
    db = SessionLocal()
    try:
        user = db.get(User, str(user_id))
        if user is None:
            user = db.scalar(select(User).where(User.telegram_chat_id == str(user_id)))
        if not user:
            return []

        stmt = select(ChatMessage).where(ChatMessage.user_id == user.id)
        if limit:
            stmt = stmt.order_by(ChatMessage.created_at.desc()).limit(limit)
            records = list(reversed(list(db.scalars(stmt))))
        else:
            records = list(db.scalars(stmt.order_by(ChatMessage.created_at.asc())))
        messages: list[BaseMessage] = []
        for r in records:
            if r.role == "user":
                messages.append(HumanMessage(content=r.content))
            elif r.role == "assistant":
                messages.append(AIMessage(content=r.content))
        return messages
    finally:
        db.close()


def save_db_chat_message(user_id: str, role: str, content: str) -> None:
    """Persists a chat message permanently into the database."""
    if not content or not content.strip():
        return
    db = SessionLocal()
    try:
        user = db.get(User, str(user_id))
        if user is None:
            user = db.scalar(select(User).where(User.telegram_chat_id == str(user_id)))
        if not user:
            user = get_or_create_user(db, telegram_chat_id=str(user_id))

        msg = ChatMessage(user_id=user.id, role=role, content=content.strip())
        db.add(msg)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving chat message to database: {e}", exc_info=True)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool Schemas for LLM Tool Calling (OpenAI / Gemini / LiteLLM format)
# ---------------------------------------------------------------------------

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "log_expense",
            "description": "Record an expense transaction and debit the specified payment account (pipe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The amount spent (must be greater than 0).",
                    },
                    "category": {
                        "type": "string",
                        "description": "The expense category (e.g. Food, Groceries, Transport, Shopping, Utilities, Subscriptions, Health, Personal Care, Entertainment, Rent, Gifts).",
                    },
                    "merchant": {
                        "type": "string",
                        "description": "The merchant, vendor, or payee name if identified (e.g. Swiggy, Uber, Starbucks, Amazon, Netflix, Poornima).",
                    },
                    "account_name": {
                        "type": "string",
                        "description": "The source payment account (pipe) used (e.g. Cash, Bank, UPI Wallet). Default to Cash if unspecified.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the expense item.",
                    },
                    "date": {
                        "type": "string",
                        "description": "YYYY-MM-DD, only when the user says it happened on another day (e.g. yesterday). Omit for today.",
                    },
                },
                "required": ["amount", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_income",
            "description": "Record an income deposit and credit the specified destination account (pipe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The income amount received (must be greater than 0).",
                    },
                    "source_account": {
                        "type": "string",
                        "description": "The destination account/pipe receiving the funds (e.g. Bank, Cash, UPI Wallet). Default to Bank.",
                    },
                    "category": {
                        "type": "string",
                        "description": "The income category (e.g. Salary, Freelance, Stipend, Cashback, Refund, Investment). Default to Income.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the income source.",
                    },
                },
                "required": ["amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_funds",
            "description": "Transfer funds between two account pipes without logging an expense.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The amount to transfer between pipes.",
                    },
                    "from_account": {
                        "type": "string",
                        "description": "The source account to debit (e.g. Bank, Cash, UPI Wallet).",
                    },
                    "to_account": {
                        "type": "string",
                        "description": "The destination account to credit (e.g. Cash, Bank, UPI Wallet).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description or note for the transfer.",
                    },
                },
                "required": ["amount", "from_account", "to_account"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipe_balances",
            "description": "Query liquid balances for all account pipes (Bank, Cash, Wallets) and total net worth.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_breakdown",
            "description": "Query spending totals grouped by category over a time period (week, month, quarter, year).",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["week", "month", "quarter", "year"],
                        "description": "Time period for spending summary. Default is month.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional specific category filter.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_budget_alert_check",
            "description": "Check if any category has crossed 80% or 100% of its monthly budget limit.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_accounts",
            "description": "List the user's accounts and current balances.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_categories",
            "description": "List the user's income or expense categories and monthly budgets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_type": {"type": "string", "enum": ["expense", "income"]}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transactions",
            "description": "Find prior ledger transactions by text, type, and ISO date range. Use this first to locate a record before editing it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "transaction_type": {
                        "type": "string",
                        "enum": ["expense", "income", "transfer"],
                    },
                    "start_date": {"type": "string", "description": "ISO date or datetime"},
                    "end_date": {"type": "string", "description": "ISO date or datetime"},
                    "limit": {"type": "integer", "description": "1 to 100, default 20"},
                    "offset": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_transaction",
            "description": "Correct a prior expense or income. Search first and use its exact transaction_id. Omit fields that should remain unchanged. Transfers cannot be edited.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "transaction_type": {"type": "string", "enum": ["expense", "income"]},
                    "account_name": {"type": "string"},
                    "category": {"type": "string", "description": "Empty string clears category"},
                    "merchant": {"type": "string", "description": "Empty string clears merchant"},
                    "description": {
                        "type": "string",
                        "description": "Empty string clears description",
                    },
                    "transaction_date": {
                        "type": "string",
                        "description": "New ISO-8601 transaction date or datetime",
                    },
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_category_budget",
            "description": "Set or clear a monthly expense budget for a category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "monthly_budget": {"type": ["number", "null"]},
                },
                "required": ["category", "monthly_budget"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_recurring_bill_detector",
            "description": "Detect repeating monthly bills, subscriptions, or recurring charges from transaction history.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_emergency_fund_calculator",
            "description": "Calculate emergency fund liquid runway (months) based on recent expense burn rate.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "financial_advice_bundle",
            "description": "Fetch holistic financial intelligence (recurring subscriptions, budget overages, emergency runway, and spending breakdown) to provide coaching and Reduce Money recommendations.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_snapshot",
            "description": "Plain summary of the user's finances for a period: income, spending, top categories and merchants, budgets, recurring bills, month-end projection, money now. Call this before giving advice or answering 'how am I doing'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "How many recent days to cover (default 30, max 365).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "savings_plan",
            "description": "Work out whether the user can save a target amount within some months, using their last 90 days of income and spending, and which discretionary categories to trim if not.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_amount": {"type": "number", "description": "Rupees to save in total."},
                    "months": {"type": "integer", "description": "Months available."},
                },
                "required": ["target_amount", "months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_transaction",
            "description": "Delete one of the user's existing entries (by its id or first 8+ characters, from search_transactions) and restore the account balance. Asks the user first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "Entry id (full or first 8+ characters).",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_default_account",
            "description": "Make an account the default one used when an entry names no account (created if new).",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_name": {"type": "string", "description": "e.g. Cash, Bank, UPI."}
                },
                "required": ["account_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_account_balance",
            "description": "Set how much money an account holds right now (opening balance or a correction). The difference is recorded as an adjustment entry. Asks the user first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_name": {"type": "string", "description": "e.g. Cash, Bank."},
                    "balance": {"type": "number", "description": "Rupees in the account now."},
                },
                "required": ["account_name", "balance"],
            },
        },
    },
]


def _pay_alternative(chat_id: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Shape the buttons for a blocked spend. Returns None when no spend was blocked.

    - A single expense blocked for lack of balance, with another pipe that could cover it:
      ``{"account", "balance", "args"}`` for a one-tap "pay from there instead" button
      next to a Cancel button.
    - A single expense blocked but no other pipe can cover it: ``{"cancel_only": True}`` so
      the reply still shows a Cancel button (nothing persists; the user just dismisses).
    """
    failed = [
        r
        for r in results
        if r.get("error")
        and "not enough balance" in str(r.get("error")).lower()
        and r.get("tool_name") == "log_expense"
    ]
    if len(failed) != 1 or any(r.get("result") for r in results):
        return None  # only offer it for a single clean "I used X" that overdrew
    r = failed[0]
    amount = r.get("tool_args", {}).get("amount")
    tried = str(r.get("tool_args", {}).get("account_name") or "").strip().lower()
    if not isinstance(amount, (int, float)):
        return {"cancel_only": True}
    with SessionLocal() as db:
        user = db.get(User, str(chat_id)) or db.scalar(
            select(User).where(User.telegram_chat_id == str(chat_id))
        )
        if user is None:
            return {"cancel_only": True}
        from app.models import Account

        best = None
        for a in db.scalars(select(Account).where(Account.user_id == user.id)):
            if a.name.strip().lower() == tried or a.balance < amount:
                continue
            if best is None or a.balance > best.balance:
                best = a
        if best is None:
            return {"cancel_only": True}
        args = dict(r.get("tool_args") or {})
        args["account_name"] = best.name
        return {"account": best.name, "balance": round(best.balance, 2), "args": args}


def _budget_note_for(state: dict[str, Any]) -> str | None:
    """One line about the budget of a category just spent on: used so far, left, and an alert at 80%+.

    Only for freshly logged expenses whose category has a budget. Returns None otherwise.
    """
    results = state.get("tool_results") or []
    spent_cats = {
        (r.get("result") or {}).get("category")
        for r in results
        if r.get("tool_name") == "log_expense" and r.get("result") and not r.get("error")
    }
    spent_cats.discard(None)
    spent_cats.discard("Uncategorized")
    if not spent_cats:
        return None
    from app.services import analytics

    chat_id = state.get("chat_id", "")
    with SessionLocal() as db:
        user = db.get(User, str(chat_id)) or db.scalar(
            select(User).where(User.telegram_chat_id == str(chat_id))
        )
        if user is None:
            return None
        rows = {b["category"].lower(): b for b in analytics.budget_status(db, user.id)}
    lines = []
    for cat in spent_cats:
        b = rows.get(str(cat).lower())
        if not b:
            continue
        used, limit, left, pct = b["spent"], b["budget"], b["left"], b["percent"]
        head = f"🎯 {b['category']} budget: {fmt.inr(used)} of {fmt.inr(limit)} used"
        if pct > 100:
            lines.append(f"🚨 {head} — over by {fmt.inr(used - limit)}")
        elif pct >= 80:
            lines.append(f"⚠️ {head} — nearly over, {fmt.inr(left)} left ({pct:.0f}%)")
        else:
            lines.append(f"{head} · {fmt.inr(left)} left")
    return "\n".join(lines) if lines else None


def _template_response_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Fixed Markdown templates for every tool result (also the offline fallback for advice)."""
    tool_name = state.get("tool_name")
    tool_result = state.get("tool_result")
    error = state.get("error")

    if error:
        return {
            "response_text": f"⚠️ **Ledger Notice:** Unable to complete request.\n*Details:* {error}"
        }

    if not tool_name or not tool_result:
        return {"response_text": "✨ Action completed successfully."}

    # 1. Expense Log
    if tool_name == "log_expense":
        amt = tool_result.get("amount", 0.0)
        acc = tool_result.get("account", "Cash")
        cat = tool_result.get("category", "General")
        merchant = tool_result.get("merchant", "N/A")
        new_bal = tool_result.get("new_balance", 0.0)
        return {
            "response_text": (
                f"💸 **Expense Recorded!**\n\n"
                f"• **Amount:** ₹{amt:,.2f}\n"
                f"• **Category:** {cat}\n"
                f"• **Merchant:** {merchant}\n"
                f"• **Paid From:** {acc} (New Balance: ₹{new_bal:,.2f})\n\n"
                f"✨ *Updated your Tank & Pipes ledger seamlessly.*"
            )
        }

    # 2. Income Log
    if tool_name == "log_income":
        amt = tool_result.get("amount", 0.0)
        acc = tool_result.get("account", "Bank")
        cat = tool_result.get("category", "Income")
        new_bal = tool_result.get("new_balance", 0.0)
        return {
            "response_text": (
                f"💰 **Income Credited!**\n\n"
                f"• **Amount:** ₹{amt:,.2f}\n"
                f"• **Category:** {cat}\n"
                f"• **Deposited To:** {acc} (New Balance: ₹{new_bal:,.2f})\n\n"
                f"📈 *Net worth increased in your Tank.*"
            )
        }

    # 3. Transfer
    if tool_name == "transfer_funds":
        amt = tool_result.get("transferred", 0.0)
        from_acc = tool_result.get("from_account", "Source")
        to_acc = tool_result.get("to_account", "Destination")
        from_bal = tool_result.get("from_balance", 0.0)
        to_bal = tool_result.get("to_balance", 0.0)
        return {
            "response_text": (
                f"🔄 **Pipe-to-Pipe Transfer Successful!**\n\n"
                f"• **Transferred:** ₹{amt:,.2f}\n"
                f"• **From:** {from_acc} (Balance: ₹{from_bal:,.2f})\n"
                f"• **To:** {to_acc} (Balance: ₹{to_bal:,.2f})\n\n"
                f"ℹ️ *This transfer was balanced internally and was not counted as an expense.*"
            )
        }

    # 4. Pipe Balances
    if tool_name == "get_pipe_balances":
        from app.bot import views

        return {"response_text": views.balance_text(tool_result)}

    # 5. Spending Breakdown
    if tool_name == "get_spending_breakdown":
        total = tool_result.get("total_expenses", 0.0)
        period = tool_result.get("period", "month")
        breakdown = tool_result.get("category_breakdown", {})
        if breakdown:
            lines = "\n".join([f"  • **{k}:** ₹{v:,.2f}" for k, v in breakdown.items()])
        else:
            lines = "  No recorded expenses for this period."
        return {
            "response_text": (
                f"📊 **Spending Breakdown ({period.title()}):**\n"
                f"💸 **Total Expenses:** ₹{total:,.2f}\n\n"
                f"**Categories:**\n{lines}"
            )
        }

    # 6. Budget Alert Skill
    if tool_name == "skill_budget_alert_check":
        month = tool_result.get("month", "")
        alerts = tool_result.get("alerts", [])
        if not alerts:
            return {
                "response_text": (
                    f"✅ **Budget Status ({month}):** All clear!\n"
                    f"None of your expense categories have crossed 80% of their monthly limit."
                )
            }
        lines = [
            f"{'🚨' if a.get('level') == 'exceeded' else '⚠️'} **{a['category']}**: ₹{a['spent']:,.2f} / ₹{a['budget']:,.2f} ({a['used_percent']}%)"
            for a in alerts
        ]
        return {
            "response_text": (
                f"⚠️ **Budget Alerts for {month}:**\n\n"
                + "\n".join(lines)
                + "\n\n💡 *Tip: Consider cutting discretionary spending in these categories.*"
            )
        }

    # 7. Recurring Bills Skill
    if tool_name == "skill_recurring_bill_detector":
        bills = tool_result.get("recurring_bills", [])
        if not bills:
            return {
                "response_text": (
                    "🔍 **Recurring Bills Detector:**\n"
                    "No repeating monthly subscriptions or bills detected in recent history."
                )
            }
        lines = [
            f"  • **{b['merchant']}**: ~₹{b['amount']:,.2f}/mo ({b['occurrences']} payments over {b['months']} months)"
            for b in bills
        ]
        return {
            "response_text": (
                "🔄 **Detected Recurring Bills & Subscriptions:**\n\n"
                + "\n".join(lines)
                + "\n\n💡 *Tip: Review these regularly to cancel unused memberships!*"
            )
        }

    # 8. Emergency Fund Skill
    if tool_name == "skill_emergency_fund_calculator":
        liquid = tool_result.get("liquid_balance", 0.0)
        burn = tool_result.get("average_monthly_expenses", 0.0)
        runway = tool_result.get("runway_months")
        target_3m = tool_result.get("target_3_month_fund", 0.0)
        shortfall = tool_result.get("shortfall_to_3_month_target", 0.0)
        runway_txt = f"{runway:.1f} months" if runway is not None else "N/A"
        health_icon = "🟢" if runway and runway >= 3 else "🟡" if runway and runway >= 1 else "🔴"
        return {
            "response_text": (
                f"🛡️ **Emergency Fund & Runway Analysis:**\n\n"
                f"• **Liquid Balances:** ₹{liquid:,.2f}\n"
                f"• **Monthly Burn Rate:** ₹{burn:,.2f}/mo\n"
                f"• {health_icon} **Estimated Runway:** **{runway_txt}**\n"
                f"• **Target (3 Months):** ₹{target_3m:,.2f}\n"
                f"• **Shortfall to Target:** ₹{shortfall:,.2f}\n\n"
                f"💡 *A solid emergency fund covers at least 3-6 months of essential burn.*"
            )
        }

    # 9. Financial Advice Bundle (Reduce Money Mode)
    if tool_name == "financial_advice_bundle":
        breakdown = tool_result.get("breakdown", {}).get("category_breakdown", {})
        recurring = tool_result.get("recurring", {}).get("recurring_bills", [])
        alerts = tool_result.get("budget", {}).get("alerts", [])
        emergency = tool_result.get("emergency", {})

        tips = []
        if alerts:
            tips.append(
                f"• **Cap Overages:** You are near/exceeding budget limits in {', '.join([a['category'] for a in alerts])}."
            )
        if recurring:
            tips.append(
                f"• **Audit Subscriptions:** You have {len(recurring)} recurring payments (e.g. {recurring[0]['merchant']}). Cancel any inactive ones."
            )
        if breakdown.get("Food", 0) > 3000:
            tips.append(
                f"• **Food & Dining:** ₹{breakdown.get('Food', 0):,.2f} spent. Cooking at home 2 days/week can save ~₹1,500/month."
            )
        if emergency.get("runway_months") and emergency.get("runway_months") < 3:
            tips.append(
                f"• **Boost Liquid Cushion:** Current runway is {emergency.get('runway_months'):.1f} months. Aim to reach 3.0 months."
            )
        if not tips:
            tips.append(
                "• **Maintain Tracking:** Keep logging every transaction to spot lifestyle creep early."
            )

        return {
            "response_text": (
                "🎯 **Fold-Style Financial Coach - Reduce Money Mode:**\n\n"
                "Here are actionable ways to optimize your finances:\n\n"
                + "\n".join(tips)
                + "\n\n💪 *Small daily optimizations compound into substantial wealth!*"
            )
        }

    # 10. Save API Key
    if tool_name == "save_user_api_key":
        provider = tool_result.get("provider", "LLM").title()
        return {
            "response_text": (
                f"🔑 **API Key Saved Successfully!**\n\n"
                f"• **Provider:** {provider}\n"
                f"• **Security:** Key stored encrypted in your private vault.\n"
                f"• **Rotation:** Round-robin key distribution enabled."
            )
        }

    if (
        tool_name == "savings_plan"
        and isinstance(tool_result, dict)
        and "needed_per_month" in tool_result
    ):
        from app.bot import views

        return {"response_text": views.plan_text(tool_result)}
    summary = (
        replies.tool_summary(tool_name, tool_result) if isinstance(tool_result, dict) else None
    )
    return {"response_text": summary or "✅ Done."}


# ---------------------------------------------------------------------------
# Graph state and nodes
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    chat_id: str
    user_text: str
    persona: str | None  # "coach" | "budgeter" | "summary" from /coach etc.
    convo: list[dict[str, Any]]  # OpenAI-format messages; overwritten by each node (no reducer)
    context: dict[str, Any]  # accounts, default account, budgets, today
    step: int  # planner rounds this turn
    proposed: list[dict[str, Any]]  # this round's tool calls: {id, name, args}
    declined: bool
    last_round: int  # how many tool_runs the executor just added
    tool_runs: list[dict[str, Any]]  # {tool_name, tool_args, result, error, error_kind}
    reply_text: str
    reply_source: str  # "planner" when reply_text is the LLM's own words
    pay_alternative: dict[str, Any] | None
    ai_failed: bool
    rejected: bool  # the tool guardrail threw every proposed call away this round
    off_topic_mixed: bool  # the message mixed money requests with an off-topic part
    off_topic_confirmed: bool  # the user already said Yes to the money part of such a message


def _load_context(chat_id: str) -> dict[str, Any]:
    """Live data the planner treats as the source of truth (balances, budgets, today)."""
    from app.models import Account
    from app.services import analytics

    context: dict[str, Any] = {"today": clock.today_local().isoformat()}
    try:
        with SessionLocal() as db:
            user = _find_user(db, chat_id)
            if user is None:
                return {**context, "accounts": [], "default_account": None, "budgets": []}
            accounts = db.scalars(select(Account).where(Account.user_id == user.id))
            context["accounts"] = [
                {"name": a.name, "balance": round(a.balance, 2)} for a in accounts
            ]
            context["default_account"] = user.default_account
            context["budgets"] = analytics.budget_status(db, user.id)
    except Exception:
        logger.warning("Agent: could not load context for %s", chat_id, exc_info=True)
    return context


def _history_dicts(chat_id: str, user_text: str) -> list[dict[str, Any]]:
    user_text = gr.mask_pii(redact_api_keys(user_text or ""))[0]
    history = get_db_conversation_history(chat_id, limit=HISTORY_LIMIT)
    out = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in history
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
    ]
    # The caller saves this turn's message before the graph runs; don't send it twice.
    if out and out[-1]["role"] == "user" and out[-1]["content"].strip() == user_text.strip():
        out.pop()
    return out


async def ingress_node(state: AgentState) -> dict[str, Any]:
    """No LLM: key paste short-circuit, persona command, context + history, per-turn reset."""
    chat_id = str(state.get("chat_id", ""))
    text = (state.get("user_text") or "").strip()
    reset: dict[str, Any] = {
        "step": 0,
        "proposed": [],
        "declined": False,
        "tool_runs": [],
        "reply_text": "",
        "reply_source": "",
        "pay_alternative": None,
        "ai_failed": False,
        "rejected": False,
        "off_topic_mixed": False,
        "off_topic_confirmed": False,
        "persona": None,
    }
    if extract_api_keys(text):
        return {**reset, "convo": [], "reply_text": await save_keys(chat_id, text)}
    text, flags = gr.check_input(text)  # input guardrail: length cap + PII masking
    if flags:
        logger.info("Input guardrail for %s: %s", chat_id, ", ".join(flags))
    persona = None
    match = PERSONA_COMMAND.match(text)
    if match:
        persona = resolve_mode(match.group(1).lower()).value
        text = match.group(2).strip() or "Review my finances."
    context = await asyncio.to_thread(_load_context, chat_id)
    history = await asyncio.to_thread(_history_dicts, chat_id, state.get("user_text") or "")
    system = build_planner_prompt(context, persona, EXPENSE_CATEGORIES)
    convo = [{"role": "system", "content": system}, *history, {"role": "user", "content": text}]
    return {**reset, "persona": persona, "context": context, "convo": convo, "user_text": text}


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        data = json.loads(raw or "{}")
    except TypeError, ValueError:
        return {}
    return data if isinstance(data, dict) else {}


async def planner_node(state: AgentState) -> dict[str, Any]:
    """One LLM call: conversation + tool schema -> tool calls (to guard) or final text."""
    chat_id = str(state.get("chat_id", ""))
    step = int(state.get("step") or 0)
    runs = state.get("tool_runs") or []
    wrote = any(r.get("tool_name") in WRITE_TOOLS and not r.get("error") for r in runs)
    if step >= MAX_STEPS:
        if runs:  # keep what was done; the renderer shows it from templates
            return {"proposed": [], "ai_failed": True}
        return {"proposed": [], "reply_text": TOO_MANY_STEPS_TEXT}
    schema = [
        t
        for t in tool_client.available_tools(TOOLS_SCHEMA)
        if t["function"]["name"] not in HIDDEN_TOOLS
    ] + [gr.OFF_TOPIC_TOOL]
    convo = list(state.get("convo") or [])
    try:
        response = await gw.complete(chat_id, convo, tools=schema)
        message = response.choices[0].message
    except gw.NoAPIKeyError:
        if wrote:
            return {"proposed": [], "ai_failed": True}
        return {"proposed": [], "reply_text": NEEDS_KEY_TEXT}
    except Exception as exc:
        logger.warning("Agent planner failed for %s: %s", chat_id, exc)
        if runs:
            return {"proposed": [], "ai_failed": True}
        return {"proposed": [], "reply_text": AI_BUSY_TEXT}

    tool_calls = list(getattr(message, "tool_calls", None) or [])
    off_topic = [
        tc
        for tc in tool_calls
        if getattr(tc, "function", None) and tc.function.name == "decline_off_topic"
    ]
    mixed = bool(state.get("off_topic_mixed"))
    if off_topic:
        tool_calls = [tc for tc in tool_calls if tc not in off_topic]
        if not tool_calls:
            logger.info("Off-topic declined for %s", chat_id)
            if runs or mixed:  # money part already handled: keep its reply, add the note
                return {"proposed": [], "off_topic_mixed": True}
            return {"proposed": [], "reply_text": gr.OFF_TOPIC_TEXT}
        logger.info("Off-topic part skipped for %s; money part kept", chat_id)
        mixed = True
    if not tool_calls:
        text = (getattr(message, "content", None) or "").strip()
        if not text and runs:
            return {"proposed": [], "ai_failed": True}
        return {
            "proposed": [],
            "reply_text": text or replies.NOT_UNDERSTOOD,
            "reply_source": "planner" if text else "",
        }
    proposed = []
    wire_calls = []
    for i, tc in enumerate(tool_calls[:MAX_TOOL_CALLS]):
        call_id = getattr(tc, "id", None) or f"call_{step}_{i}"
        name = tc.function.name
        raw_args = tc.function.arguments
        args = _parse_args(raw_args)
        args.pop("user_id", None)
        proposed.append({"id": call_id, "name": name, "args": args})
        wire_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": raw_args if isinstance(raw_args, str) else json.dumps(args),
                },
            }
        )
    convo.append(
        {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": wire_calls,
        }
    )
    return {
        "convo": convo,
        "proposed": proposed,
        "step": step + 1,
        "rejected": False,
        "off_topic_mixed": mixed,
    }


def _describe_call(call: dict[str, Any]) -> str:
    """A short human sentence for one write call, used in confirmation questions."""
    name, args = call["name"], call.get("args") or {}
    amount = fmt.inr(_amount_of(args))
    if name == "log_expense":
        bits = [args.get("category"), args.get("account_name")]
        return f"log {amount} expense" + (
            f" ({', '.join(str(b) for b in bits if b)})" if any(bits) else ""
        )
    if name == "log_income":
        bits = [args.get("category"), args.get("source_account")]
        return f"log {amount} income" + (
            f" ({', '.join(str(b) for b in bits if b)})" if any(bits) else ""
        )
    reason = confirmation_reason(name, args)
    if reason:
        return reason[0].lower() + reason[1:]
    return name.replace("_", " ")


def _pending_id(chat_id: str, calls: list[dict[str, Any]]) -> str:
    raw = json.dumps([chat_id, calls], sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:6]


def guard_node(state: AgentState) -> dict[str, Any]:
    """No LLM: pause with interrupt() when any proposed call needs the user's Yes.

    On resume LangGraph re-runs this node from the top: the id is deterministic, and the new
    created_at is never read (resume_pending checks the stored interrupt's created_at BEFORE
    resuming), so the second pass simply receives the answer from interrupt().
    """
    chat_id = str(state.get("chat_id", ""))
    calls = state.get("proposed") or []
    # --- tool guardrail: reject bad calls before anything runs; the planner sees the error ---
    schema = TOOLS_SCHEMA + [gr.OFF_TOPIC_TOOL]
    today = clock.today_local()
    ok_calls, rejects = [], []
    writes = 0
    for c in calls:
        error = gr.check_call(c["name"], c["args"], schema, today)
        if not error and c["name"] in WRITE_TOOLS:
            writes += 1
            if writes > gr.MAX_WRITES_PER_MESSAGE:
                error = f"too many writes in one message (limit {gr.MAX_WRITES_PER_MESSAGE})"
        (rejects if error else ok_calls).append((c, error))
    if rejects:
        convo = list(state.get("convo") or [])
        for c, error in rejects:
            logger.info("Tool guardrail rejected %s for %s: %s", c["name"], chat_id, error)
            convo.append(
                {"role": "tool", "tool_call_id": c["id"], "content": gr.as_json({"error": error})}
            )
        calls = [c for c, _ in ok_calls]
        if not calls:  # nothing survived: back to the planner to explain or ask
            return {"declined": False, "convo": convo, "proposed": [], "rejected": True}
        state = {**state, "convo": convo, "proposed": calls, "rejected": True}
    reasons = []
    for c in calls:
        reason = confirmation_reason(c["name"], c["args"]) or _budget_overage_reason(
            chat_id, {"tool_name": c["name"], "tool_args": c["args"]}
        )
        if reason:
            reasons.append(reason)
    rejected = bool(state.get("rejected"))
    mixed_writes = [c for c in calls if c["name"] in WRITE_TOOLS]
    ask_mixed = (
        bool(state.get("off_topic_mixed"))
        and not state.get("off_topic_confirmed")
        and bool(mixed_writes)
    )
    if ask_mixed:
        # The message also asked something off-topic: show what we understood and ask first.
        reasons = [gr.OFF_TOPIC_HITL_PREFIX + "; ".join(_describe_call(c) for c in mixed_writes)]
    if not reasons:
        return {
            "declined": False,
            "convo": state.get("convo"),
            "proposed": calls,
            "rejected": rejected,
        }
    description = " and ".join(reasons)
    others = len(calls) - len(reasons)
    if others > 0:
        description += (
            f" (and {others} other entr{'y' if others == 1 else 'ies'} from your message)"
        )
    if ask_mixed:
        description = reasons[0]
    answer = interrupt(
        {
            "kind": "confirm",
            "id": _pending_id(chat_id, calls),
            "created_at": _utc_now().isoformat(),
            "description": description,
            "calls": calls,
        }
    )
    if answer == "yes":
        return {
            "declined": False,
            "convo": state.get("convo"),
            "proposed": calls,
            "rejected": rejected,
            "off_topic_confirmed": bool(state.get("off_topic_confirmed")) or ask_mixed,
        }
    convo = list(state.get("convo") or [])
    for c in calls:
        convo.append({"role": "tool", "tool_call_id": c["id"], "content": "cancelled by the user"})
    return {"declined": True, "convo": convo, "proposed": [], "reply_text": replies.CANCELLED}


async def executor_node(state: AgentState) -> dict[str, Any]:
    """No LLM: run the approved calls (user_id injected), one tool message per call."""
    chat_id = str(state.get("chat_id", ""))
    convo = list(state.get("convo") or [])
    runs = list(state.get("tool_runs") or [])
    this_round: list[dict[str, Any]] = []
    for call in (state.get("proposed") or [])[:MAX_TOOL_CALLS]:
        name, args = call["name"], dict(call["args"])
        run: dict[str, Any] = {"tool_name": name, "tool_args": args, "result": None, "error": None}
        try:
            run["result"] = await tool_client.call_tool(name, **{**args, "user_id": chat_id})
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            run["error"] = error
            low = error.lower()
            run["error_kind"] = "insufficient_balance" if "not enough balance" in low else "error"
        this_round.append(run)
        payload = run["result"] if run["error"] is None else {"error": run["error"]}
        content = json.dumps(payload, default=str, ensure_ascii=False)[:TOOL_RESULT_CHARS]
        convo.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    return {
        "convo": convo,
        "tool_runs": runs + this_round,
        "proposed": [],
        "last_round": len(this_round),
        "pay_alternative": await asyncio.to_thread(_pay_alternative, chat_id, this_round),
    }


def route_after_ingress(state: AgentState) -> Literal["planner", "renderer"]:
    return "renderer" if state.get("reply_text") else "planner"


def route_after_planner(state: AgentState) -> Literal["guard", "renderer"]:
    return "guard" if state.get("proposed") else "renderer"


def route_after_guard(state: AgentState) -> Literal["executor", "planner", "renderer"]:
    if state.get("declined"):
        return "renderer"
    if state.get("rejected") and not state.get("proposed"):
        return "planner"  # every call was invalid; the planner sees the errors and retries or asks
    return "executor"


def route_after_executor(state: AgentState) -> Literal["planner", "renderer"]:
    """Read/advice results go back to the planner to be explained; plain writes are templated."""
    runs = state.get("tool_runs") or []
    if state.get("rejected"):
        return "planner"  # some calls were rejected by the tool guardrail: let the planner explain
    for r in runs[len(runs) - int(state.get("last_round") or 0) :]:
        if r["tool_name"] not in WRITE_TOOLS:
            return "planner"
        if r.get("error") and r.get("error_kind") != "insufficient_balance":
            return "planner"
    return "renderer"


def renderer_node(state: AgentState) -> dict[str, Any]:
    """No LLM: planner text, else fixed templates; then the persona header and budget note."""
    runs = state.get("tool_runs") or []
    text = state.get("reply_text") or ""
    from_planner = bool(text) and state.get("reply_source") == "planner"
    if not text:
        text = replies.compact({"tool_results": runs}) or ""
    if not text:
        summaries = [
            replies.tool_summary(r["tool_name"], r["result"])
            for r in runs
            if isinstance(r.get("result"), dict) and not r.get("error")
        ]
        text = "\n\n".join(s for s in summaries if s)
    if not text and runs:
        last = runs[-1]
        text = _template_response_dict(
            {"tool_name": last["tool_name"], "tool_result": last["result"], "error": last["error"]}
        )["response_text"]
    text = text or "Done."
    if from_planner:  # output guardrail: only LLM-written text can invent figures or leak
        chat_id = str(state.get("chat_id", ""))
        found = gr.leaks(text, chat_id)
        if found:
            logger.warning("Output guardrail blocked a leak for %s: %s", chat_id, found)
            text, from_planner = gr.LEAK_TEXT, False
        else:
            bad = gr.ungrounded_figures(
                text,
                state.get("context"),
                [{"args": r.get("tool_args"), "result": r.get("result")} for r in runs],
                state.get("user_text"),
            )
            if bad:
                logger.warning("Output guardrail: ungrounded figures %s for %s", bad, chat_id)
                fallback = replies.compact({"tool_results": runs}) or "\n\n".join(
                    s
                    for r in runs
                    if isinstance(r.get("result"), dict) and not r.get("error")
                    for s in [replies.tool_summary(r["tool_name"], r["result"])]
                    if s
                )
                if not fallback and runs and not runs[-1].get("error"):
                    last = runs[-1]
                    fallback = _template_response_dict(
                        {
                            "tool_name": last["tool_name"],
                            "tool_result": last["result"],
                            "error": None,
                        }
                    )["response_text"]
                text, from_planner = fallback or gr.GROUNDING_FALLBACK_TEXT, False
    text = gr.sanitize_reply(text)
    if state.get("off_topic_mixed") and not state.get("declined") and gr.OFF_TOPIC_NOTE not in text:
        text = (
            f"{text}\n\n{gr.OFF_TOPIC_NOTE}"
            if text.strip() and text != "Done."
            else gr.OFF_TOPIC_TEXT
        )
    persona = state.get("persona")
    if persona and from_planner:
        text = f"🧠 *{get_persona(persona).name}*\n\n{text}"
    if not state.get("declined") and any(r["tool_name"] == "log_expense" for r in runs):
        try:
            note = _budget_note_for({"chat_id": state.get("chat_id", ""), "tool_results": runs})
        except Exception:
            note = None
        if note:
            text = f"{text}\n{note}"
    return {"reply_text": text}


def build_agent_graph():
    """ingress -> planner -> guard (interrupt) -> executor -> planner|renderer, checkpointed."""
    workflow = StateGraph(AgentState)
    workflow.add_node("ingress", ingress_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("guard", guard_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("renderer", renderer_node)
    workflow.add_edge(START, "ingress")
    workflow.add_conditional_edges("ingress", route_after_ingress)
    workflow.add_conditional_edges("planner", route_after_planner)
    workflow.add_conditional_edges("guard", route_after_guard)
    workflow.add_conditional_edges("executor", route_after_executor)
    workflow.add_edge("renderer", END)
    # In-memory checkpoints (thread_id = chat_id): a pending Yes/No dies on restart, by design.
    return workflow.compile(checkpointer=InMemorySaver())


# Compiled LangGraph agent app
agent_app = build_agent_graph()


# ---------------------------------------------------------------------------
# Public Contract Interfaces
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """What the agent did, for the UI: the reply plus facts used to add buttons."""

    text: str
    transaction_ids: list[str] = field(default_factory=list)
    needs_confirmation: bool = False
    pending_id: str | None = None  # which question the Yes/Cancel buttons answer
    pay_alternative: dict[str, Any] | None = None  # {account, balance, args}: offer another account


def _config(chat_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": str(chat_id)}}


def _interrupt_value(items: Any) -> dict[str, Any] | None:
    for item in items or ():
        value = getattr(item, "value", None)
        if isinstance(value, dict) and value.get("kind") == "confirm":
            return value
    return None


async def _pending_payload(chat_id: str) -> dict[str, Any] | None:
    """The confirmation the graph is paused on for this chat, or None."""
    try:
        snapshot = await agent_app.aget_state(_config(chat_id))
    except Exception:
        return None
    value = _interrupt_value(getattr(snapshot, "interrupts", None))
    if value is None:
        for task in getattr(snapshot, "tasks", None) or ():
            value = _interrupt_value(getattr(task, "interrupts", None))
            if value:
                break
    return value


def _tx_ids(runs: list[dict[str, Any]]) -> list[str]:
    return [
        str(r["result"]["transaction_id"])
        for r in runs
        if isinstance(r.get("result"), dict) and r["result"].get("transaction_id")
    ]


async def _outcome_from(chat_id: str, result: Any) -> Outcome:
    """Turn a graph run (finished or paused on a confirmation) into an Outcome."""
    result = result if isinstance(result, dict) else {}
    runs = result.get("tool_runs") or []
    payload = _interrupt_value(result.get("__interrupt__")) or await _pending_payload(chat_id)
    if payload is not None:
        done = [r for r in runs if r.get("result") and not r.get("error")]
        prefix = replies.compact({"tool_results": done}) if done else None
        text = (f"{prefix}\n\n" if prefix else "") + replies.confirm_prompt(
            payload.get("description", "Go ahead")
        )
        return Outcome(text, _tx_ids(runs), True, payload.get("id"), None)
    text = result.get("reply_text") or "Done."
    return Outcome(text, _tx_ids(runs), False, None, result.get("pay_alternative"))


async def has_pending(chat_id: str) -> bool:
    """True while the graph for this chat waits for a Yes/Cancel."""
    return await _pending_payload(chat_id) is not None


async def cancel_pending(chat_id: str) -> None:
    """Answer the waiting confirmation with "no" and discard the reply."""
    if await _pending_payload(chat_id) is None:
        return
    try:
        await agent_app.ainvoke(Command(resume="no"), _config(chat_id))
    except Exception:
        logger.warning("Agent: could not cancel pending action for %s", chat_id, exc_info=True)


async def resume_pending(chat_id: str, yes: bool, pending_id: str | None = None) -> Outcome | None:
    """Continue a paused graph with the user's answer. None if nothing (or something else) waits.

    An answer older than PENDING_ACTION_TTL is refused and the stale action cancelled, so an old
    button can never move money.
    """
    payload = await _pending_payload(chat_id)
    if payload is None or (pending_id and payload.get("id") != pending_id):
        return None
    try:
        created = datetime.fromisoformat(payload["created_at"])
    except KeyError, TypeError, ValueError:
        created = None
    if created is None or _utc_now() - created > PENDING_ACTION_TTL:
        await cancel_pending(chat_id)
        return None
    try:
        result = await agent_app.ainvoke(Command(resume="yes" if yes else "no"), _config(chat_id))
        outcome = await _outcome_from(chat_id, result)
    except Exception as e:
        logger.error(f"Error resuming agent for {chat_id}: {e}", exc_info=True)
        outcome = Outcome(ERROR_TEXT)
    save_db_chat_message(chat_id, "assistant", outcome.text)
    return outcome


DECISION_PROMPT = (
    "You classify a chat reply. The bot asked the user to confirm this action: {description}.\n"
    "Read the user's reply (any language, slang or emoji) and answer with exactly one word:\n"
    "confirm - the user agrees to go ahead with that action\n"
    "cancel - the user refuses or wants to stop that action\n"
    "new - the user is saying or asking something else\n"
    "Answer only: confirm, cancel or new."
)


async def classify_decision(
    chat_id: str, text: str, description: str | None = None
) -> Literal["confirm", "cancel", "new"]:
    """One small LLM call: is this typed reply a yes, a no, or a new request? Unsure -> new.

    ``description`` names what is waiting; by default the agent's own pending action.
    """
    if description is None:
        payload = await _pending_payload(chat_id)
        description = (payload or {}).get("description", "the pending action")
    messages = [
        {"role": "system", "content": DECISION_PROMPT.format(description=description)},
        {"role": "user", "content": (text or "").strip()[:500]},
    ]
    try:
        response = await gw.complete(str(chat_id), messages)
    except Exception:
        return "new"
    if not isinstance(response, str):
        try:
            response = response.choices[0].message.content or ""
        except AttributeError, IndexError:
            return "new"
    word = re.sub(r"[^a-z]", " ", response.lower()).split()
    answer = word[0] if word else ""
    return answer if answer in ("confirm", "cancel", "new") else "new"  # type: ignore[return-value]


async def process_user_interaction(
    chat_id: str, text: str, attachments: list[Any] | None = None
) -> str:
    """Text-only entry point (kept for the dev endpoint and older callers)."""
    return (await process_user_interaction_ex(chat_id, text, attachments)).text


async def process_user_interaction_ex(
    chat_id: str, text: str, attachments: list[Any] | None = None
) -> Outcome:
    """Entry point for the Telegram bot: run (or resume) the graph and persist both messages."""
    chat_id = str(chat_id)
    clean_text = (text or "").strip()
    try:
        # Save user message to database (API keys masked, so they never reach chat history)
        save_db_chat_message(chat_id, "user", gr.mask_pii(redact_api_keys(clean_text))[0])
        if await has_pending(chat_id):
            decision = await classify_decision(chat_id, clean_text)
            if decision in ("confirm", "cancel"):
                outcome = await resume_pending(chat_id, decision == "confirm")
                if outcome is not None:
                    return outcome  # resume_pending saved the reply
            else:
                await cancel_pending(chat_id)
        result = await agent_app.ainvoke(
            {"chat_id": chat_id, "user_text": clean_text}, _config(chat_id)
        )
        outcome = await _outcome_from(chat_id, result)
    except Exception as e:
        logger.error(f"Error executing LLM Agent StateGraph: {e}", exc_info=True)
        outcome = Outcome(ERROR_TEXT)
    save_db_chat_message(chat_id, "assistant", outcome.text)
    return outcome
