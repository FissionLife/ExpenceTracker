import re
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session, joinedload

from app import clock
from app.models import Account, Category, Merchant, Transaction, User

VALID_TX_TYPES = {"expense", "income"}
MAX_AMOUNT = 10_000_000_000.0


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _adjust_balance(db: Session, account_id: str, delta: float) -> None:
    """Add `delta` to an account's balance inside the database, in one statement.

    Reading the balance into Python, adding, and writing it back loses updates when two requests run at
    once (both read 100, both write 90). `balance = balance + delta` is applied by the database itself,
    so every change counts.
    """
    db.execute(
        update(Account)
        .where(Account.id == account_id)
        .values(balance=Account.balance + delta)
        .execution_options(synchronize_session=False)
    )


def parse_when(value: Any) -> datetime | None:
    """Accepts a datetime, an ISO date/datetime string, or None. Raises LedgerError if unreadable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "")
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise LedgerError(f"Invalid date: {value!r} (use YYYY-MM-DD)")


def _validate_date(value: Any) -> datetime | None:
    """A date or time typed by a person (their local time) -> the UTC time that is stored.

    A date without a time ("yesterday", a statement row) stays local midnight of that day, which converts
    back to the same local date (and tells the hour chart that the time of day is unknown).
    """
    local = parse_when(value)
    if local is None:
        return None
    local = local.replace(tzinfo=None)
    if local.year < 2000:
        raise LedgerError("Date is too far in the past")
    stored = clock.to_utc(local)
    if stored > _now() + timedelta(days=1):
        raise LedgerError("Date is in the future")
    return stored


def validate_when(value: Any) -> datetime | None:
    """A date from the user or the AI: readable, after 1999 and not in the future (or None)."""
    return _validate_date(value)


class LedgerError(ValueError):
    """Invalid ledger operation (bad amount or type, same-account transfer)."""


class InsufficientBalanceError(LedgerError):
    """A spend or transfer out would push an account below zero.

    Carries the account name, its current balance and the amount needed so the
    caller can show "Not enough in Cash" and offer the other account.
    """

    def __init__(self, account: str, available: float, needed: float) -> None:
        self.account = account
        self.available = round(available, 2)
        self.needed = round(needed, 2)
        super().__init__(
            f"Not enough balance in {account}: you have "
            f"{self.available:g} but this needs {self.needed:g}"
        )


def validate_amount(amount: float) -> float:
    """Returns the amount rounded to 2 decimals, or raises LedgerError (0, negative, NaN, inf)."""
    try:
        value = round(float(amount), 2)
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"Invalid amount: {amount!r}") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise LedgerError("Amount must be a finite number")
    if value <= 0:
        raise LedgerError("Amount must be greater than zero")
    if value > MAX_AMOUNT:
        raise LedgerError("Amount is too large")
    return value


def get_or_create_user(db: Session, telegram_chat_id: str) -> User:
    """Retrieves an existing user by Telegram chat ID or creates a new one."""
    stmt = select(User).where(User.telegram_chat_id == str(telegram_chat_id))
    user = db.scalar(stmt)
    if not user:
        user = User(telegram_chat_id=str(telegram_chat_id))
        db.add(user)
        db.commit()
        db.refresh(user)

        # Create default primary accounts (Pipes In)
        default_accounts = [
            Account(user_id=user.id, name="Cash", type="cash", balance=0.0),
            Account(user_id=user.id, name="Bank", type="bank", balance=0.0),
        ]
        db.add_all(default_accounts)
        db.commit()
    return user


def get_or_create_account(
    db: Session,
    user_id: str,
    account_name: str,
    account_type: str = "bank",
    *,
    lock: bool = False,
) -> Account:
    """Finds an account by name (case-insensitive) or creates it.

    ``lock=True`` takes a row lock (SELECT ... FOR UPDATE) so concurrent balance updates on
    PostgreSQL are serialised until the caller commits. SQLite ignores the lock (harmless).
    """
    stmt = select(Account).where(
        Account.user_id == user_id,
        func.lower(Account.name) == account_name.strip().lower(),
    )
    if lock:
        stmt = stmt.with_for_update()
    account = db.scalar(stmt)
    if not account:
        account = Account(
            user_id=user_id,
            name=account_name.strip(),
            type=account_type,
            balance=0.0,
        )
        db.add(account)
        db.commit()
        db.refresh(account)
    return account


def get_or_create_category(
    db: Session, user_id: str, category_name: str, cat_type: str = "expense"
) -> Category:
    """Finds a category by name or creates it."""
    stmt = select(Category).where(
        Category.user_id == user_id,
        func.lower(Category.name) == category_name.strip().lower(),
    )
    cat = db.scalar(stmt)
    if not cat:
        cat = Category(
            user_id=user_id,
            name=category_name.strip().title(),
            type=cat_type,
        )
        db.add(cat)
        db.commit()
        db.refresh(cat)
    return cat


def resolve_category(db: Session, user_id: str, name: str, cat_type: str = "expense") -> Category:
    """The category a person means by `name`: "food" is their "Food & Dining", not a second "Food".

    Order: exact name (any case) -> a category with that word in its name (only if exactly one) -> a new one.
    """
    key = name.strip().lower()
    mine = list(
        db.scalars(select(Category).where(Category.user_id == user_id, Category.type == cat_type))
    )
    exact = next((c for c in mine if c.name.lower() == key), None)
    if exact:
        return exact
    if len(key) >= 3:
        words = [c for c in mine if key in re.split(r"[^a-z0-9]+", c.name.lower())]
        if len(words) == 1:
            return words[0]
        partial = [c for c in mine if key in c.name.lower()]
        if len(partial) == 1:
            return partial[0]
    return get_or_create_category(db, user_id, name, cat_type)


def get_or_create_merchant(
    db: Session, user_id: str, merchant_name: str, default_category_id: str | None = None
) -> Merchant:
    """Finds a merchant by name or creates it."""
    stmt = select(Merchant).where(
        Merchant.user_id == user_id,
        func.lower(Merchant.name) == merchant_name.strip().lower(),
    )
    merchant = db.scalar(stmt)
    if not merchant:
        merchant = Merchant(
            user_id=user_id,
            name=merchant_name.strip().title(),
            default_category_id=default_category_id,
        )
        db.add(merchant)
        db.commit()
        db.refresh(merchant)
    return merchant


def _ensure_can_cover(db: Session, account: Account, amount: float) -> None:
    """Block a live spend that would push the account below zero.

    Strict wallet: every pipe is protected, including never-funded ones at zero. Called while
    the account row is locked, so the balance read here is the real current balance that the
    following _adjust_balance will change. ``db`` is unused but kept so the two call sites and
    the edit_transaction path stay symmetrical.
    """
    del db  # kept for signature symmetry
    if round(account.balance - amount, 2) < 0:
        raise InsufficientBalanceError(account.name, account.balance, amount)


def record_transaction(
    db: Session,
    user_id: str,
    amount: float,
    account_name: str,
    category_name: str | None = None,
    merchant_name: str | None = None,
    tx_type: str = "expense",
    description: str | None = None,
    source: str = "chat",
    created_at: datetime | None = None,
    affects_balance: bool = True,
    reference: str | None = None,
) -> dict[str, Any]:
    """
    Logs an expense or income and updates the corresponding account balance atomically.

    ``created_at`` backdates the transaction (statement imports, "yesterday I spent...").
    ``affects_balance=False`` records history only (reports count it, today's balance does not move).
    ``reference`` is the bank's own id for the payment, kept so the same payment is not imported twice.
    """
    when = _validate_date(created_at)
    if tx_type not in VALID_TX_TYPES:
        raise LedgerError(f"Invalid transaction type {tx_type!r}; use 'expense' or 'income'")
    amount = validate_amount(amount)
    account = get_or_create_account(db, user_id, account_name, lock=True)

    cat_id = None
    if category_name:
        cat = get_or_create_category(db, user_id, category_name, cat_type=tx_type)
        cat_id = cat.id

    merchant_id = None
    if merchant_name:
        merchant = get_or_create_merchant(db, user_id, merchant_name, default_category_id=cat_id)
        merchant_id = merchant.id
        if not cat_id and merchant.default_category_id:
            cat_id = merchant.default_category_id

    # Update balance according to Pipe flow (atomically, see _adjust_balance)
    if affects_balance:
        # Strict wallet: a live spend can't overdraw the account. Backdated writes (statement
        # imports, demo seed, "yesterday I spent...") reconstruct history, where the final balance
        # — not the order rows land in — is what matters, so they are exempt from the live check.
        if tx_type == "expense" and when is None:
            _ensure_can_cover(db, account, amount)
        _adjust_balance(db, account.id, -amount if tx_type == "expense" else amount)

    tx = Transaction(
        user_id=user_id,
        account_id=account.id,
        category_id=cat_id,
        merchant_id=merchant_id,
        amount=amount,
        transaction_type=tx_type,
        description=description,
        source=source,
        affects_balance=affects_balance,
        reference=(reference or None) and str(reference)[:80],
        **({"created_at": when} if when else {}),
    )
    db.add(tx)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(tx)

    return {
        "status": "success",
        "transaction_id": tx.id,
        "amount": tx.amount,
        "type": tx.transaction_type,
        "account": account.name,
        "new_balance": round(account.balance, 2),
        "category": category_name or "Uncategorized",
        "merchant": merchant_name or "N/A",
    }


def transfer_between_pipes(
    db: Session,
    user_id: str,
    from_account_name: str,
    to_account_name: str,
    amount: float,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Transfers funds from Pipe A to Pipe B without logging an external expense.
    """
    amount = validate_amount(amount)
    if from_account_name.strip().lower() == to_account_name.strip().lower():
        raise LedgerError("Cannot transfer to the same account")
    # Lock both rows in a stable order (by name) so two opposite transfers cannot deadlock.
    locked = {
        name.strip().lower(): get_or_create_account(db, user_id, name, lock=True)
        for name in sorted([from_account_name, to_account_name], key=lambda n: n.strip().lower())
    }
    from_account = locked[from_account_name.strip().lower()]
    to_account = locked[to_account_name.strip().lower()]

    _ensure_can_cover(db, from_account, amount)  # a transfer out cannot overdraw a funded source
    _adjust_balance(db, from_account.id, -amount)
    _adjust_balance(db, to_account.id, amount)

    tx = Transaction(
        user_id=user_id,
        account_id=from_account.id,
        to_account_id=to_account.id,
        amount=amount,
        transaction_type="transfer",
        description=description or f"Transfer from {from_account.name} to {to_account.name}",
        source="transfer",
    )
    db.add(tx)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(tx)

    return {
        "status": "success",
        "transaction_id": tx.id,
        "from_account": from_account.name,
        "from_balance": round(from_account.balance, 2),
        "to_account": to_account.name,
        "to_balance": round(to_account.balance, 2),
        "transferred": amount,
    }


def get_account_balances(db: Session, user_id: str) -> dict[str, Any]:
    """Fetches balances for all accounts associated with user and calculates net balance."""
    stmt = select(Account).where(Account.user_id == user_id)
    accounts = list(db.scalars(stmt))

    total = sum(acc.balance for acc in accounts)
    return {
        "total_net_worth": round(total, 2),
        "accounts": [
            {"id": acc.id, "name": acc.name, "type": acc.type, "balance": round(acc.balance, 2)}
            for acc in accounts
        ],
    }


def get_spending_summary(db: Session, user_id: str, days: int = 30) -> dict[str, Any]:
    """Aggregates recent spending by category and merchant."""
    start_date = datetime.utcnow() - timedelta(days=days)

    stmt = (
        select(Transaction)
        .options(joinedload(Transaction.category))
        .where(
            Transaction.user_id == user_id,
            Transaction.transaction_type == "expense",
            Transaction.created_at >= start_date,
        )
    )
    txs = list(db.scalars(stmt).unique())

    total_spent = sum(t.amount for t in txs)

    # Categories breakdown
    cat_totals: dict[str, float] = {}
    for t in txs:
        cat_name = t.category.name if t.category else "Uncategorized"
        cat_totals[cat_name] = cat_totals.get(cat_name, 0.0) + t.amount

    return {
        "period_days": days,
        "total_expenses": round(total_spent, 2),
        "transaction_count": len(txs),
        "category_breakdown": {k: round(v, 2) for k, v in cat_totals.items()},
    }


def undo_transaction(db: Session, user_id: str, transaction_id: str) -> dict[str, Any]:
    """Delete one of the user's transactions and put the balances back as they were.

    ``transaction_id`` may be the full id or its first 8 or more characters (as shown in buttons).
    """
    ident = (
        (transaction_id or "").strip().lower()
    )  # ids are lowercase UUIDs; LIKE is case-sensitive on Postgres
    if len(ident) < 6:
        raise LedgerError("Transaction id is too short")
    if not re.fullmatch(r"[0-9a-fA-F-]+", ident):  # ids are UUIDs; also keeps LIKE wildcards out
        raise LedgerError("That is not a transaction id")
    stmt = select(Transaction).where(Transaction.user_id == user_id)
    stmt = (
        stmt.where(Transaction.id == ident)
        if len(ident) >= 36
        else stmt.where(Transaction.id.like(f"{ident}%"))
    )
    matches = list(db.scalars(stmt.limit(2)))
    if not matches:
        raise LedgerError("That transaction was not found (maybe it was already undone)")
    if len(matches) > 1:
        raise LedgerError("That id matches more than one transaction; use a longer id")
    tx = matches[0]
    src = db.get(Account, tx.account_id) if tx.account_id else None
    dst = db.get(Account, tx.to_account_id) if tx.to_account_id else None
    kind, amount, applied = tx.transaction_type, tx.amount, tx.affects_balance is not False
    # Delete FIRST and look at the row count: a double tap (or two devices) must refund only once.
    removed = db.execute(
        delete(Transaction)
        .where(Transaction.id == tx.id, Transaction.user_id == user_id)
        .execution_options(synchronize_session=False)
    ).rowcount
    if removed != 1:
        db.rollback()
        raise LedgerError("That transaction was not found (maybe it was already undone)")
    if applied:
        if kind == "expense" and src:
            _adjust_balance(db, src.id, amount)
        elif kind == "income" and src:
            _adjust_balance(db, src.id, -amount)
        elif kind == "transfer":
            if src:
                _adjust_balance(db, src.id, amount)
            if dst:
                _adjust_balance(db, dst.id, -amount)
    db.flush()
    if src:
        db.refresh(src)  # the update above bypassed the object; read the real balance
    summary = {
        "status": "success",
        "transaction_id": tx.id,
        "type": tx.transaction_type,
        "amount": tx.amount,
        "account": src.name if src else None,
        "to_account": dst.name if dst else None,
        "new_balance": round(src.balance, 2) if src else None,
        "when": tx.created_at.date().isoformat(),
    }
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    return summary


def last_entered_transaction_id(db: Session, user_id: str) -> str | None:
    """The transaction the user entered most recently (not the one with the latest date).

    "Spent 20 on chai yesterday" is entered after "Spent 500 on rent today", so /undo must remove the
    chai. Rows from before `entered_at` existed fall back to their date.
    """
    return db.scalar(
        select(Transaction.id)
        .where(Transaction.user_id == user_id)
        .order_by(func.coalesce(Transaction.entered_at, Transaction.created_at).desc())
        .limit(1)
    )


def recent_transactions(
    db: Session, user_id: str, limit: int = 10, offset: int = 0, kind: str | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Newest first, with human-readable names. Returns (rows, total count)."""
    base = select(Transaction).where(Transaction.user_id == user_id)
    if kind:
        base = base.where(Transaction.transaction_type == kind)
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    stmt = (
        base.options(joinedload(Transaction.category), joinedload(Transaction.merchant))
        .order_by(Transaction.created_at.desc(), Transaction.id)
        .offset(offset)
        .limit(limit)
    )
    names = {a.id: a.name for a in db.scalars(select(Account).where(Account.user_id == user_id))}
    rows = [
        {
            "id": t.id,
            "date": t.created_at.date().isoformat(),
            "type": t.transaction_type,
            "amount": t.amount,
            "category": t.category.name if t.category else "",
            "merchant": t.merchant.name if t.merchant else "",
            "account": names.get(t.account_id, ""),
            "to_account": names.get(t.to_account_id, "") if t.to_account_id else "",
            "note": t.description or "",
        }
        for t in db.scalars(stmt).unique()
    ]
    return rows, total
