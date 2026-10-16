"""Short, scannable replies for ledger actions.

    💸 −₹450 · Food & Dining
    Bank ₹84,550

Rules: one headline, one balance line, no repeated words, no lectures. Several entries in one message
become a small list. Errors say what to fix in one sentence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app import clock
from app.bot import fmt

LOGGING_TOOLS = {"log_expense", "log_income", "transfer_funds"}


def _when(args: dict[str, Any]) -> str:
    raw = (args or {}).get("date")
    if not raw:
        return ""
    try:
        d = datetime.fromisoformat(str(raw)[:10]).date()
    except ValueError:
        return ""
    return "" if d == clock.today_local() else f" · {fmt.day(d)}"


def _line(item: dict[str, Any]) -> str:
    """Headline for one logged entry."""
    name, res, args = item["tool_name"], item["result"], item.get("tool_args") or {}
    if name == "transfer_funds":
        return f"⇄ {fmt.inr(res['transferred'])} · {res['from_account']} → {res['to_account']}{_when(args)}"
    amount = fmt.inr(res["amount"], 2 if round(res["amount"], 2) % 1 else 0)
    cat = res.get("category") or "Uncategorized"
    merchant = res.get("merchant")
    detail = cat if not merchant or merchant == "N/A" else f"{merchant} · {cat}"
    if name == "log_income":
        return f"💰 +{amount} · {detail}{_when(args)}"
    return f"💸 −{amount} · {detail}{_when(args)}"


def _balance_line(items: list[dict[str, Any]]) -> str:
    """The latest balance of every account touched, in order of first appearance."""
    seen: dict[str, float] = {}
    for item in items:
        res = item["result"]
        if item["tool_name"] == "transfer_funds":
            seen[res["from_account"]] = res["from_balance"]
            seen[res["to_account"]] = res["to_balance"]
        elif res.get("account") is not None and res.get("new_balance") is not None:
            seen[res["account"]] = res["new_balance"]
    return " · ".join(f"{name} {fmt.inr(bal)}" for name, bal in seen.items())


def friendly_error(message: str) -> str:
    low = (message or "").lower()
    if "not enough balance" in low:
        return message  # already a clear sentence: "Not enough balance in Cash: you have 0..."
    if "greater than zero" in low:
        return "The amount has to be more than zero."
    if "same account" in low:
        return "Pick two different accounts for a transfer."
    if "too large" in low:
        return "That amount is too large."
    if "date is in the future" in low:
        return "That date is in the future."
    if "invalid date" in low:
        return "I couldn't read that date. Try 2026-09-30."
    return message or "Something went wrong."


def compact(state: dict[str, Any]) -> str | None:
    """Reply for logged entries (and their errors). None when this is not a logging action."""
    results = state.get("tool_results") or []
    if not results or any(r.get("tool_name") not in LOGGING_TOOLS for r in results):
        return None
    ok = [r for r in results if r.get("result") and not r.get("error")]
    failed = [r for r in results if r.get("error") or not r.get("result")]
    lines: list[str] = []
    if len(ok) == 1:
        lines.append(_line(ok[0]))
    elif ok:
        spent = sum(r["result"]["amount"] for r in ok if r["tool_name"] == "log_expense")
        earned = sum(r["result"]["amount"] for r in ok if r["tool_name"] == "log_income")
        parts = []
        if spent:
            parts.append(f"−{fmt.inr(spent, 2 if round(spent, 2) % 1 else 0)}")
        if earned:
            parts.append(f"+{fmt.inr(earned, 2 if round(earned, 2) % 1 else 0)}")
        head = f"✅ {len(ok)} entries"
        lines.append(f"{head} · {' · '.join(parts)}" if parts else head)
        lines += [f"• {_line(r)[2:].strip()}" for r in ok]
    balances = _balance_line(ok) if ok else ""
    if balances:
        lines.append(balances)
    for r in failed:
        what = r.get("tool_args", {}).get("amount")
        prefix = f"⚠️ ₹{what:g}: " if isinstance(what, (int, float)) else "⚠️ "
        lines.append(prefix + friendly_error(r.get("error") or "could not be saved"))
    return "\n".join(lines)


def _signed(kind: str, amount: float) -> str:
    text = fmt.inr(amount, 2 if round(amount, 2) % 1 else 0)
    return f"+{text}" if kind == "income" else f"−{text}" if kind == "expense" else text


def tool_summary(name: str, res: dict[str, Any]) -> str | None:
    """Plain text for tools the logging templates do not cover (edits, searches, lists, budgets)."""
    if name == "edit_transaction":
        where = f" · {res['account']}" if res.get("account") else ""
        return f"✏️ Updated · {_signed(res.get('type', ''), res['amount'])}{where}"
    if name == "search_transactions":
        rows = res.get("transactions") or []
        if not rows:
            return "I couldn't find anything like that."
        lines = [f"🧾 {len(rows)} found"]
        for tx in rows[:10]:
            what = tx.get("merchant") or tx.get("category") or tx.get("description") or ""
            lines.append(
                f"{fmt.day(tx['date'])}  {_signed(tx['type'], tx['amount'])}  {what}".rstrip()
            )
        if len(rows) > 10:
            lines.append(f"… and {len(rows) - 10} more (see /history)")
        return "\n".join(lines)
    if name == "list_accounts":
        accounts = res.get("accounts") or []
        if not accounts:
            return "No accounts yet."
        return "💰 Your accounts\n" + "\n".join(
            f"{a['name']}  {fmt.inr(a['balance'])}" for a in accounts
        )
    if name == "list_categories":
        cats = res.get("categories") or []
        if not cats:
            return "No categories yet."
        return "🏷 Categories\n" + "\n".join(
            f"{c['name']}"
            + (f"  (budget {fmt.inr(c['monthly_budget'])})" if c.get("monthly_budget") else "")
            for c in cats
        )
    if name == "set_category_budget":
        if res.get("monthly_budget"):
            return f"🎯 {res['category']} budget: {fmt.inr(res['monthly_budget'])} a month"
        return f"🎯 {res['category']} budget removed"
    if name == "get_financial_snapshot":
        lines = [
            f"📊 {res.get('period', 'Recently')}",
            f"Spent {fmt.inr(res.get('spent', 0))}  ·  Earned {fmt.inr(res.get('earned', 0))}",
        ]
        top = res.get("top_categories") or []
        if top:
            lines.append(
                "Top: " + " · ".join(f"{c['name']} {fmt.short(c['amount'])}" for c in top[:3])
            )
        for tip in (res.get("insights") or [])[:2]:
            lines.append(f"💡 {tip}")
        return "\n".join(lines)
    return None


def confirm_prompt(description: str) -> str:
    return f"⚠️ {description}?\nTap Yes to go ahead, or Cancel."


CANCELLED = "👍 Cancelled. Nothing changed."
NOTHING_PENDING = "👍 Nothing is waiting for an answer. Tell me what you spent, or try /menu."

GREETING = (
    "I log what you spend and earn.\n"
    "Try: Spent 450 on dinner, or Received 50000 salary.\n"
    "Then /stats, /balance, /history. More: /help"
)
HELLO = (
    'Hey! 👋 Tell me what you spent or earned, like "coffee 120" or "got salary 50000".\n'
    'Or ask me, like "how much did I spend this week?" or "what\'s my balance?"'
)
THANKS = "Anytime! 🙂"
BARE_AMOUNT = (
    "{amount}: did you spend it or receive it?\n"
    'Say it like "spent {plain} on lunch" or "received {plain} from Ravi".'
)
QUESTION_NEEDS_AI = (
    "That sounds like a question, so I didn't record anything.\n"
    "To answer questions like this I need an AI key: /setkey\n"
    "To log something, say it plainly: Spent 640 on dinner."
)
NOT_UNDERSTOOD = (
    "I couldn't read that as an entry, so I didn't record anything.\n"
    "Try: Spent 450 on dinner via Bank, or Received 60000 salary.\n"
    "To understand free-form messages, add your AI key: /setkey"
)
NO_AI_HINT = (
    "💡 For free-form questions and advice, add your own AI key: /setkey\n"
    "(a free Gemini key takes a minute: aistudio.google.com/apikey)"
)
