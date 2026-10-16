"""Typed financial tools exposed to the conversational agent and stdio MCP server.

The shared registry gives both integrations one stable dispatch point.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from datetime import datetime
from itertools import islice
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Account, ApiKey, Category, Merchant, Transaction, User
from app.services.ledger import (
    InsufficientBalanceError,
    get_account_balances,
    get_or_create_account,
    get_or_create_category,
    get_or_create_user,
    record_transaction,
    resolve_category,
    transfer_between_pipes,
    undo_transaction,
    validate_amount,
    validate_when,
)
from app.services.ledger import _adjust_balance as adjust_balance


def _with_db(fn: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    db = SessionLocal()
    try:
        return fn(db=db, **kwargs)
    finally:
        db.close()


def _resolve_user(db: Session, user_id: str) -> User:
    """Accept either the internal user UUID or the Telegram chat ID contract."""
    user = db.get(User, str(user_id))
    if user is None:
        user = db.scalar(select(User).where(User.telegram_chat_id == str(user_id)))
    return user or get_or_create_user(db, str(user_id))


def _positive_amount(amount: float) -> float:
    """The ledger's own check: a real, finite number above zero and not absurdly large."""
    return validate_amount(amount)


def log_expense(
    user_id: str,
    amount: float,
    category: str,
    merchant: str | None = None,
    account_name: str | None = None,
    description: str | None = None,
    source: str = "chat",
    date: str | None = None,
    affects_balance: bool = True,
    reference: str | None = None,
) -> dict[str, Any]:
    """Record an expense and debit its payment account (``source``: chat, pdf, sms...).

    ``date`` (YYYY-MM-DD) backdates it; leave empty for now. ``affects_balance=False`` records it as
    history only. ``reference`` is the bank's id for the payment (used to spot duplicates).
    """

    def run(db: Session, **values: Any) -> dict[str, Any]:
        user = _resolve_user(db, values.pop("user_id"))
        values["user_id"] = user.id
        values["amount"] = _positive_amount(values["amount"])
        if values.get("category_name"):  # "food"/"meals" -> the canonical "Food & Dining"
            from app.bot.categories import canonical_category

            values["category_name"] = canonical_category(values["category_name"])
        if not values.get("account_name"):  # none named: the user's default, else Cash
            values["account_name"] = user.default_account or "Cash"
        return record_transaction(db, tx_type="expense", **values)

    return _with_db(
        run,
        user_id=user_id,
        amount=amount,
        category_name=category,
        merchant_name=merchant,
        account_name=account_name,
        description=description,
        source=source,
        created_at=date,
        affects_balance=affects_balance,
        reference=reference,
    )


def log_income(
    user_id: str,
    amount: float,
    source_account: str = "Bank",
    category: str = "Income",
    description: str | None = None,
    source: str = "chat",
    date: str | None = None,
    affects_balance: bool = True,
    reference: str | None = None,
) -> dict[str, Any]:
    """Record income and credit its destination account (``source``: chat, pdf, sms...).

    ``date`` (YYYY-MM-DD) backdates it; leave empty for now. ``affects_balance=False`` records it as
    history only. ``reference`` is the bank's id for the payment (used to spot duplicates).
    """

    def run(db: Session, **values: Any) -> dict[str, Any]:
        user = _resolve_user(db, values.pop("user_id"))
        values["user_id"] = user.id
        values["amount"] = _positive_amount(values["amount"])
        return record_transaction(db, tx_type="income", **values)

    return _with_db(
        run,
        user_id=user_id,
        amount=amount,
        account_name=source_account,
        category_name=category,
        description=description,
        source=source,
        created_at=date,
        affects_balance=affects_balance,
        reference=reference,
    )


def transfer_funds(
    user_id: str,
    from_account: str,
    to_account: str,
    amount: float,
    description: str | None = None,
) -> dict[str, Any]:
    """Transfer money between accounts without recording an expense."""

    def run(db: Session, **values: Any) -> dict[str, Any]:
        user = _resolve_user(db, values.pop("user_id"))
        values["user_id"] = user.id
        values["amount"] = _positive_amount(values["amount"])
        return transfer_between_pipes(db, **values)

    return _with_db(
        run,
        user_id=user_id,
        from_account_name=from_account,
        to_account_name=to_account,
        amount=amount,
        description=description,
    )


def get_pipe_balances(user_id: str) -> dict[str, Any]:
    def run(db: Session, user_id: str) -> dict[str, Any]:
        return get_account_balances(db, _resolve_user(db, user_id).id)

    return _with_db(run, user_id=user_id)


def list_accounts(user_id: str) -> dict[str, Any]:
    """List the caller's accounts, including balances and account types."""

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        accounts = db.scalars(select(Account).where(Account.user_id == uid).order_by(Account.name))
        return {
            "accounts": [
                {"id": a.id, "name": a.name, "type": a.type, "balance": round(a.balance, 2)}
                for a in accounts
            ]
        }

    return _with_db(run, user_id=user_id)


def list_categories(user_id: str, transaction_type: str | None = None) -> dict[str, Any]:
    """List the caller's categories and configured budgets."""
    if transaction_type not in (None, "expense", "income"):
        raise ValueError("transaction_type must be expense or income")

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        stmt = select(Category).where(Category.user_id == uid)
        if transaction_type:
            stmt = stmt.where(Category.type == transaction_type)
        categories = db.scalars(stmt.order_by(Category.name))
        return {
            "categories": [
                {"id": c.id, "name": c.name, "type": c.type, "monthly_budget": c.budget_limit}
                for c in categories
            ]
        }

    return _with_db(run, user_id=user_id)


def search_transactions(
    user_id: str,
    query: str | None = None,
    transaction_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Search the caller's ledger, newest first. Dates use ISO-8601; limit is capped at 100."""
    if transaction_type not in (None, "expense", "income", "transfer"):
        raise ValueError("transaction_type must be expense, income, or transfer")
    if not 1 <= int(limit) <= 100 or int(offset) < 0:
        raise ValueError("limit must be 1..100 and offset must be non-negative")
    try:
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
    except ValueError as exc:
        raise ValueError("dates must be ISO-8601 date or datetime values") from exc
    if start and end and start > end:
        raise ValueError("start_date must not be after end_date")

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        stmt = (
            select(Transaction, Account.name, Category.name, Merchant.name)
            .outerjoin(Account, Transaction.account_id == Account.id)
            .outerjoin(Category, Transaction.category_id == Category.id)
            .outerjoin(Merchant, Transaction.merchant_id == Merchant.id)
            .where(Transaction.user_id == uid)
        )
        if transaction_type:
            stmt = stmt.where(Transaction.transaction_type == transaction_type)
        if start:
            stmt = stmt.where(Transaction.created_at >= start)
        if end:
            stmt = stmt.where(Transaction.created_at <= end)
        stmt = stmt.order_by(Transaction.created_at.desc())
        term = (query or "").strip().casefold()
        if term:
            # description is encrypted at rest (app/crypto.py), so SQL cannot search it: match
            # the decrypted rows here instead, then page.
            hits = (
                row
                for row in db.execute(stmt)
                if any(term in (text or "").casefold() for text in (row[0].description, *row[1:]))
            )
            rows = list(islice(hits, int(offset), int(offset) + int(limit)))
        else:
            rows = db.execute(stmt.offset(int(offset)).limit(int(limit))).all()
        return {
            "transactions": [
                {
                    "id": tx.id,
                    "date": tx.created_at.isoformat(),
                    "type": tx.transaction_type,
                    "amount": round(tx.amount, 2),
                    "account": account,
                    "category": category,
                    "merchant": merchant,
                    "description": tx.description,
                    "source": tx.source,
                }
                for tx, account, category, merchant in rows
            ],
            "limit": int(limit),
            "offset": int(offset),
        }

    return _with_db(run, user_id=user_id)


def edit_transaction(
    user_id: str,
    transaction_id: str,
    amount: float | None = None,
    transaction_type: str | None = None,
    account_name: str | None = None,
    category: str | None = None,
    merchant: str | None = None,
    description: str | None = None,
    transaction_date: str | None = None,
) -> dict[str, Any]:
    """Edit a prior non-transfer entry and reconcile affected account balances."""
    if amount is not None:
        amount = _positive_amount(amount)
    if transaction_type not in (None, "expense", "income"):
        raise ValueError("transaction_type may be expense or income")
    parsed_date = validate_when(transaction_date)  # same rules as logging: real, not in the future

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        tx = db.scalar(
            select(Transaction)
            .where(Transaction.id == transaction_id, Transaction.user_id == uid)
            .with_for_update()
        )
        if tx is None:
            raise ValueError("transaction not found")
        if tx.transaction_type == "transfer":
            raise ValueError(
                "transfer edits are not supported; record a correcting transfer instead"
            )
        old_account = db.get(Account, tx.account_id) if tx.account_id else None
        if old_account and old_account.user_id != uid:
            raise ValueError("transaction account does not belong to user")
        new_account = old_account
        if account_name is not None:
            if not account_name.strip():
                raise ValueError("account_name cannot be empty")
            new_account = get_or_create_account(db, uid, account_name.strip(), lock=True)
        old_type, old_amount = tx.transaction_type, float(tx.amount)
        new_type, new_amount = (
            transaction_type or old_type,
            amount if amount is not None else old_amount,
        )
        # Reverse the old posting, then apply the corrected posting.
        if old_account:
            adjust_balance(db, old_account.id, old_amount if old_type == "expense" else -old_amount)
        if new_account and new_type == "expense":
            # After the reversal above lands, block only if this would overdraw a funded account.
            db.flush()
            db.refresh(new_account)
            from app.services.ledger import _ensure_can_cover

            try:
                _ensure_can_cover(db, new_account, new_amount)
            except InsufficientBalanceError:
                db.rollback()
                raise
        if new_account:
            adjust_balance(db, new_account.id, -new_amount if new_type == "expense" else new_amount)
        tx.account_id, tx.transaction_type, tx.amount = (
            (new_account.id if new_account else None),
            new_type,
            new_amount,
        )
        if category is not None:
            if category.strip():
                tx.category_id = get_or_create_category(db, uid, category.strip(), new_type).id
            else:
                tx.category_id = None
        elif new_type != old_type:
            tx.category_id = None
        if merchant is not None:
            if merchant.strip():
                from app.services.ledger import get_or_create_merchant

                tx.merchant_id = get_or_create_merchant(
                    db, uid, merchant.strip(), tx.category_id
                ).id
            else:
                tx.merchant_id = None
        if description is not None:
            tx.description = description.strip() or None
        if parsed_date is not None:
            tx.created_at = parsed_date
        db.commit()
        return {
            "status": "success",
            "transaction_id": tx.id,
            "type": tx.transaction_type,
            "amount": round(tx.amount, 2),
            "account": new_account.name if new_account else None,
        }

    return _with_db(run, user_id=user_id)


def set_category_budget(
    user_id: str, category: str, monthly_budget: float | None
) -> dict[str, Any]:
    """Set or clear a monthly expense category budget (null clears it)."""
    if not category.strip():
        raise ValueError("category is required")
    if monthly_budget is not None and float(monthly_budget) <= 0:
        raise ValueError("monthly_budget must be greater than zero or null")

    def run(db: Session, user_id: str) -> dict[str, Any]:
        from app.bot.categories import canonical_category

        uid = _resolve_user(db, user_id).id
        # "food"/"meals" -> the canonical "Food & Dining" the expenses file under (same as /budget)
        cat = resolve_category(db, uid, canonical_category(category.strip()), "expense")
        cat.budget_limit = float(monthly_budget) if monthly_budget is not None else None
        db.commit()
        return {"status": "success", "category": cat.name, "monthly_budget": cat.budget_limit}

    return _with_db(run, user_id=user_id)


def delete_transaction(user_id: str, transaction_id: str) -> dict[str, Any]:
    """Delete one of the user's transactions (full id or first 8+ characters) and restore balances."""

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        return {**undo_transaction(db, uid, transaction_id), "status": "success"}

    return _with_db(run, user_id=user_id)


def set_default_account(user_id: str, account_name: str) -> dict[str, Any]:
    """Make `account_name` the account used when an entry names none (created if new)."""
    if not (account_name or "").strip():
        raise ValueError("account_name is required")

    def run(db: Session, user_id: str) -> dict[str, Any]:
        user = _resolve_user(db, user_id)
        account = get_or_create_account(db, user.id, account_name.strip())
        user.default_account = account.name
        db.commit()
        return {"status": "success", "default_account": account.name}

    return _with_db(run, user_id=user_id)


OPENING_BALANCE_CATEGORY = "Opening Balance"
BALANCE_ADJUSTMENT_CATEGORY = "Balance Adjustment"


def set_account_balance(user_id: str, account_name: str, balance: float) -> dict[str, Any]:
    """Make an account hold exactly `balance`: the difference is recorded as an opening balance
    (account has no entries yet) or a balance adjustment (income or expense)."""
    if not (account_name or "").strip():
        raise ValueError("account_name is required")
    target = float(balance)
    if target != target or target in (float("inf"), float("-inf")):
        raise ValueError("balance must be a real number")
    if target < 0:
        raise ValueError("balance must be zero or more")

    def run(db: Session, user_id: str) -> dict[str, Any]:
        user = _resolve_user(db, user_id)
        account = get_or_create_account(db, user.id, account_name.strip())
        name = account.name
        before = round(account.balance, 2)
        diff = round(target - before, 2)
        result_base = {"status": "success", "account": name, "previous_balance": before}
        if abs(diff) < 0.005:
            db.commit()
            return {**result_base, "new_balance": before, "transaction_id": None}
        has_entries = db.scalar(
            select(func.count())
            .select_from(Transaction)
            .where(Transaction.user_id == user.id, Transaction.account_id == account.id)
        )
        if diff > 0:
            category = BALANCE_ADJUSTMENT_CATEGORY if has_entries else OPENING_BALANCE_CATEGORY
            tx_type, amount = "income", diff
        else:
            category, tx_type, amount = BALANCE_ADJUSTMENT_CATEGORY, "expense", -diff
        result = record_transaction(
            db, user.id, amount, name, category, None, tx_type, "Balance set by you", "manual"
        )
        return {
            **result_base,
            "new_balance": result["new_balance"],
            "transaction_id": result["transaction_id"],
        }

    return _with_db(run, user_id=user_id)


def get_spending_breakdown(
    user_id: str, period: str = "month", category: str | None = None
) -> dict[str, Any]:
    """Return spending totals; period accepts week, month, quarter, or year (the last 7/30/90/365 days)."""
    from app.services import analytics

    periods = {"week": 7, "month": 30, "quarter": 90, "year": 365}
    if period not in periods:
        raise ValueError("period must be one of: week, month, quarter, year")

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        rows = analytics.load_rows(db, uid, analytics.default_query(periods[period]))
        breakdown = {
            c["category"]: c["amount"] for c in analytics.category_breakdown(rows, None, top_n=1000)
        }
        if category:
            breakdown = {
                name: total
                for name, total in breakdown.items()
                if name.casefold() == category.casefold()
            }
        return {
            "period": period,
            "period_days": periods[period],
            "total_expenses": round(sum(breakdown.values()), 2),
            "category_breakdown": breakdown,
        }

    return _with_db(run, user_id=user_id)


def skill_budget_alert_check(user_id: str) -> dict[str, Any]:
    """Report categories at or above 80% of their monthly budget (same numbers as /budget)."""
    from app.services import analytics

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        alerts = [
            {
                "category": b["category"],
                "spent": b["spent"],
                "budget": b["budget"],
                "used_percent": round(b["percent"], 1),
                "level": "exceeded" if b["level"] == "over" else "warning",
            }
            for b in analytics.budget_status(db, uid)
            if b["percent"] >= 80
        ]
        return {"month": analytics._now().strftime("%Y-%m"), "alerts": alerts}

    return _with_db(run, user_id=user_id)


def skill_recurring_bill_detector(user_id: str) -> dict[str, Any]:
    """Find merchants charged about every month for a near-constant amount (same rule as the dashboard)."""
    from app.services import analytics

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        rows = analytics.load_rows(db, uid, analytics.default_query(183))
        bills = []
        for item in analytics.recurring(rows):
            hits = [r for r in rows if r.kind == "expense" and r.merchant == item["merchant"]]
            bills.append(
                {
                    "merchant": item["merchant"],
                    "amount": item["amount"],
                    "occurrences": item["times"],
                    "months": len({(r.when.year, r.when.month) for r in hits}),
                }
            )
        return {"recurring_bills": bills}

    return _with_db(run, user_id=user_id)


def skill_emergency_fund_calculator(user_id: str) -> dict[str, Any]:
    """Estimate how long the money in cash/bank/wallet accounts lasts (same rule as the dashboard)."""
    from app.services import analytics

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        runway = analytics.runway(db, uid)
        burn = runway["monthly_spend"]
        liquid = runway["liquid_balance"]
        return {
            "liquid_balance": liquid,
            "average_monthly_expenses": burn,
            "runway_months": runway["months"],
            "target_3_month_fund": round(burn * 3, 2),
            "shortfall_to_3_month_target": round(max(0, burn * 3 - liquid), 2),
            "low_confidence": runway["low_confidence"],
        }

    return _with_db(run, user_id=user_id)


def save_user_api_key(user_id: str, provider: str, api_key: str) -> dict[str, Any]:
    """Encrypt and store a provider key using the application's configured secret."""
    if not api_key.strip() or not provider.strip():
        raise ValueError("provider and api_key are required")

    def run(db: Session, user_id: str) -> dict[str, Any]:
        from app.crypto import encrypt_vault  # same vault format as ai_gateway.encrypt_api_key

        uid = _resolve_user(db, user_id).id
        encrypted = encrypt_vault(api_key.strip())
        record = ApiKey(user_id=uid, provider=provider.strip().lower(), encrypted_key=encrypted)
        db.add(record)
        db.commit()
        return {"status": "success", "provider": record.provider, "key_id": record.id}

    return _with_db(run, user_id=user_id)


def get_financial_snapshot(user_id: str, days: int = 30) -> dict[str, Any]:
    """Compact summary of the user's money for a period (numbers for reasoning, not for display)."""
    from app.services import analytics

    days = max(1, min(int(days), 365))

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        return {"status": "success", **analytics.financial_snapshot(db, uid, days)}

    return _with_db(run, user_id=user_id)


def savings_plan(user_id: str, target_amount: float, months: int) -> dict[str, Any]:
    """Can the user save `target_amount` in `months`? Uses their last 90 days; suggests trims."""
    from app.services import analytics

    def run(db: Session, user_id: str) -> dict[str, Any]:
        uid = _resolve_user(db, user_id).id
        return {
            "status": "success",
            **analytics.savings_plan(db, uid, float(target_amount), int(months)),
        }

    return _with_db(run, user_id=user_id)


def financial_advice_bundle(user_id: str) -> dict[str, Any]:
    """Recurring bills, budget alerts, emergency runway and this month's spending in one call."""
    return {
        "recurring": skill_recurring_bill_detector(user_id),
        "budget": skill_budget_alert_check(user_id),
        "emergency": skill_emergency_fund_calculator(user_id),
        "breakdown": get_spending_breakdown(user_id, period="month"),
    }


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "financial_advice_bundle": financial_advice_bundle,
    "get_financial_snapshot": get_financial_snapshot,
    "savings_plan": savings_plan,
    "log_expense": log_expense,
    "log_income": log_income,
    "transfer_between_pipes": transfer_funds,
    "transfer_funds": transfer_funds,
    "get_pipe_balances": get_pipe_balances,
    "get_spending_breakdown": get_spending_breakdown,
    "save_user_api_key": save_user_api_key,
    "skill_budget_alert_check": skill_budget_alert_check,
    "skill_recurring_bill_detector": skill_recurring_bill_detector,
    "skill_emergency_fund_calculator": skill_emergency_fund_calculator,
    "list_accounts": list_accounts,
    "list_categories": list_categories,
    "search_transactions": search_transactions,
    "edit_transaction": edit_transaction,
    "set_category_budget": set_category_budget,
    "delete_transaction": delete_transaction,
    "set_default_account": set_default_account,
    "set_account_balance": set_account_balance,
}

mcp = MCPServer(
    name="WalletLedger",
    instructions=(
        "User-scoped financial ledger tools. Supply the user's Telegram chat ID or internal UUID."
    ),
)


def _expected_errors_visible(handler: Callable[..., dict[str, Any]]) -> Callable[..., dict]:
    """Send a tool's own ValueError text (e.g. "amount must be greater than zero") to MCP clients.

    MCPServer hides the text of unexpected exceptions; validation errors are meant for the caller.
    """

    @functools.wraps(handler)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return handler(*args, **kwargs)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


for name, handler in TOOL_REGISTRY.items():
    mcp.tool(name=name)(_expected_errors_visible(handler))


def execute_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch one registered agent tool and return a JSON-serializable result."""
    try:
        tool = TOOL_REGISTRY[tool_name]
    except KeyError as exc:
        raise ValueError(f"Unknown tool: {tool_name}") from exc
    return tool(**kwargs)


if __name__ == "__main__":
    mcp.run(transport="stdio")
