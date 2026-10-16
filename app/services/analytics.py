"""Statistics for one user, computed from the ledger.

One flexible `Query` (period + optional filters) is loaded once into light `Row`s, and small pure
functions turn those rows into numbers. The same `report()` feeds the Telegram messages, the chart
images and the web dashboard. Transfers between a user's own accounts are never counted as spending or
income. Amounts are rupees rounded to 2 decimals.
"""

from __future__ import annotations

import calendar
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, null, select
from sqlalchemy.orm import Session

from app import clock
from app.models import Account, Category, Merchant, Transaction

UNCATEGORIZED = "Uncategorized"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MAX_ROWS = 200_000  # safety valve for very large accounts


def _now() -> datetime:
    """Now in the user's timezone: every window, 'today' and day bucket here is a LOCAL day."""
    return clock.now_local()


def _r(x: float) -> float:
    return round(float(x), 2)


def _day_start(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


# =============================================================================== query


@dataclass(frozen=True)
class Query:
    """What to analyse: a time window plus optional filters."""

    start: datetime
    end: datetime  # exclusive
    label: str = "Last 30 days"
    accounts: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    merchants: tuple[str, ...] = ()  # substring match
    min_amount: float | None = None
    max_amount: float | None = None
    group: str = "auto"  # auto | day | week | month
    top_n: int = 8
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def days(self) -> int:
        return max(1, (self.end - self.start).days)

    def first_day(self) -> date | None:
        """The day of the user's very first entry (set by report()), or None if unknown."""
        value = self.extra.get("first_day")
        return value if isinstance(value, date) else None

    def effective_days(self, first_day: date | None = None) -> int:
        """Days the figures really cover: not past today, and not before the user's first entry.

        "all" starts in the year 2000, and a brand-new user has weeks of empty window, so dividing
        by the nominal length would make every daily average look tiny.
        """
        end = min(self.end.date(), _now().date() + timedelta(days=1))
        start = self.start.date()
        first_day = first_day or self.first_day()
        if first_day is not None and first_day > start:
            start = first_day
        return max(1, (end - start).days)

    @property
    def filtered(self) -> bool:
        return bool(
            self.accounts
            or self.categories
            or self.merchants
            or self.min_amount is not None
            or self.max_amount is not None
        )

    def previous(self) -> Query:
        """The period to compare with: the same stretch of the calendar one step earlier.

        This month so far (1-13 Oct) is compared with 1-13 Sep, this week so far with the same days of
        last week, this year so far with the same days last year. A fixed-length window (30 days) is
        compared with the 30 days before it.
        """
        tomorrow = _day_start(_now()) + timedelta(days=1)
        cur_end = min(self.end, tomorrow)  # a period still running only covers the days so far
        span = max(timedelta(days=1), cur_end - self.start)
        label = "previous period"
        month_aligned = self.start.day == 1 and (self.end.day == 1 or self.end == tomorrow)
        if month_aligned and span.days <= 31:
            prev_start = (self.start - timedelta(days=1)).replace(day=1)
            finished = self.end.day == 1 and self.end <= tomorrow  # a complete calendar month
            # a finished month is compared with the whole month before; one still running, with the same days
            prev_end = self.start if finished else min(prev_start + span, self.start)
            return replace(self, start=prev_start, end=prev_end, label=label)
        if self.label == "This week":
            return replace(
                self,
                start=self.start - timedelta(days=7),
                end=self.start - timedelta(days=7) + span,
                label=label,
            )
        if self.label.endswith("so far") and self.start.month == 1 and self.start.day == 1:
            prev_start = self.start.replace(year=self.start.year - 1)
            return replace(
                self, start=prev_start, end=min(prev_start + span, self.start), label=label
            )
        return replace(self, start=self.start - span, end=self.start, label=label)

    def describe(self) -> str:
        parts = [self.label]
        parts += list(self.categories) + list(self.accounts) + list(self.merchants)
        if self.min_amount is not None:
            parts.append(f"over {_inr(self.min_amount)}")
        if self.max_amount is not None:
            parts.append(f"under {_inr(self.max_amount)}")
        return " · ".join(parts)


_REL = re.compile(r"^(\d{1,4})\s*(d|day|days|w|week|weeks|m|month|months|y|year|years)$", re.I)
_RANGE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*(?:\.\.|to)\s*(\d{4}-\d{2}-\d{2})$", re.I)
_MONTH = re.compile(r"^(\d{4})-(\d{2})$")


def parse_period(text: str, now: datetime | None = None) -> Query | None:
    """Turn '30d', '6m', '1y', 'mtd', 'last month', '2025-09', 'ytd', 'all' or 'A..B' into a Query.

    Returns None if the text is not a period.
    """
    now = now or _now()
    t = (text or "").strip().lower()
    end = _day_start(now) + timedelta(days=1)
    if not t:
        return None
    m = _REL.match(t)
    if m:
        n, unit = int(m.group(1)), m.group(2)[0]
        days = {"d": 1, "w": 7, "m": 30, "y": 365}[unit] * n
        days = min(max(days, 1), 3650)
        return Query(end - timedelta(days=days), end, f"Last {days} day{'s' if days != 1 else ''}")
    if t in ("today",):
        return Query(end - timedelta(days=1), end, "Today")
    if t in ("yesterday",):
        return Query(end - timedelta(days=2), end - timedelta(days=1), "Yesterday")
    if t in ("week", "this week"):
        start = _day_start(now) - timedelta(days=now.weekday())
        return Query(start, end, "This week")
    if t in ("mtd", "month", "this month"):
        return Query(_day_start(now).replace(day=1), end, "This month")
    if t in ("last month", "lastmonth", "prev month"):
        first = _day_start(now).replace(day=1)
        prev_last = first - timedelta(days=1)
        return Query(prev_last.replace(day=1), first, prev_last.strftime("%B %Y"))
    if t in ("ytd", "year", "this year"):
        return Query(datetime(now.year, 1, 1), end, f"{now.year} so far")
    if t in ("all", "everything", "alltime", "all time"):
        return Query(datetime(2000, 1, 1), end, "All time", group="month", extra={"trim": True})
    m = _MONTH.match(t)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 2000 <= y <= now.year + 1:
            start = datetime(y, mo, 1)
            nxt = datetime(y + (mo == 12), 1 if mo == 12 else mo + 1, 1)
            return Query(start, nxt, start.strftime("%B %Y"))
    m = _RANGE.match(t)
    if m:
        try:
            a = datetime.strptime(m.group(1), "%Y-%m-%d")
            b = datetime.strptime(m.group(2), "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            return None
        if a < b:
            return Query(a, b, f"{m.group(1)} to {m.group(2)}")
    return None


def default_query(days: int = 30, now: datetime | None = None) -> Query:
    q = parse_period(f"{days}d", now)
    assert q is not None
    return q


def resolve_filters(
    db: Session, user_id: str, query: Query, words: list[str]
) -> tuple[Query, list[str]]:
    """Match leftover words ('food', 'hdfc', 'swiggy') to this user's categories/accounts/merchants.

    Returns the narrowed query and the words that matched nothing.
    """
    cats = {
        c.lower(): c for c in db.scalars(select(Category.name).where(Category.user_id == user_id))
    }
    accs = {
        a.lower(): a for a in db.scalars(select(Account.name).where(Account.user_id == user_id))
    }
    mers = {
        m.lower(): m for m in db.scalars(select(Merchant.name).where(Merchant.user_id == user_id))
    }
    got_cat: list[str] = []
    got_acc: list[str] = []
    got_mer: list[str] = []
    unmatched: list[str] = []
    for w in words:
        wl = w.strip().lower()
        if not wl:
            continue
        hit = next((v for k, v in cats.items() if wl == k or (len(wl) >= 3 and wl in k)), None)
        if hit:
            got_cat.append(hit)
            continue
        hit = next((v for k, v in accs.items() if wl == k or (len(wl) >= 3 and wl in k)), None)
        if hit:
            got_acc.append(hit)
            continue
        hit = next((v for k, v in mers.items() if wl == k or (len(wl) >= 3 and wl in k)), None)
        if hit:
            got_mer.append(hit)
            continue
        unmatched.append(w)
    return (
        replace(
            query,
            categories=tuple(dict.fromkeys(query.categories + tuple(got_cat))),
            accounts=tuple(dict.fromkeys(query.accounts + tuple(got_acc))),
            merchants=tuple(dict.fromkeys(query.merchants + tuple(got_mer))),
        ),
        unmatched,
    )


# =============================================================================== rows


@dataclass(frozen=True)
class Row:
    id: str
    when: datetime
    amount: float
    kind: str  # expense | income | transfer
    category: str
    merchant: str
    account: str
    note: str
    applied: bool = True  # False: imported history that never touched today's balances

    @property
    def day(self) -> date:
        return self.when.date()


def first_entry_day(db: Session, user_id: str) -> date | None:
    """The local day of the user's first transaction of any kind."""
    first = db.scalar(
        select(func.min(Transaction.created_at)).where(Transaction.user_id == user_id)
    )
    return clock.to_local(first).date() if first else None


def load_rows(db: Session, user_id: str, q: Query, notes: bool = False) -> list[Row]:
    """Transactions in the window that pass the filters, oldest first (one SQL query).

    The window is a window of LOCAL days (q.start/q.end); times in the rows are local too.
    Notes are encrypted, so they are only read (decrypted) when `notes=True`: lists and exports.
    If an account has more than MAX_ROWS in the window, the NEWEST rows are kept.
    """
    acc = Account.__table__.alias("a")
    stmt = (
        select(
            Transaction.id,
            Transaction.created_at,
            Transaction.amount,
            Transaction.transaction_type,
            Category.name,
            Merchant.name,
            acc.c.name,
            Transaction.description if notes else null(),
            Transaction.affects_balance,
        )
        .select_from(Transaction)
        .outerjoin(Category, Category.id == Transaction.category_id)
        .outerjoin(Merchant, Merchant.id == Transaction.merchant_id)
        .outerjoin(acc, acc.c.id == Transaction.account_id)
        .where(
            Transaction.user_id == user_id,
            Transaction.created_at >= clock.to_utc(q.start),
            Transaction.created_at < clock.to_utc(q.end),
        )
        .order_by(Transaction.created_at.desc())
        .limit(MAX_ROWS)
    )
    if q.categories:
        stmt = stmt.where(func.lower(Category.name).in_([c.lower() for c in q.categories]))
    if q.accounts:
        stmt = stmt.where(func.lower(acc.c.name).in_([a.lower() for a in q.accounts]))
    if q.min_amount is not None:
        stmt = stmt.where(Transaction.amount >= q.min_amount)
    if q.max_amount is not None:
        stmt = stmt.where(Transaction.amount <= q.max_amount)
    rows = [
        Row(
            i,
            clock.to_local(w),
            float(a),
            k,
            c or UNCATEGORIZED,
            mer or "",
            acct or "",
            n or "",
            f is not False,
        )
        for i, w, a, k, c, mer, acct, n, f in db.execute(stmt)
    ]
    rows.reverse()  # oldest first
    if q.merchants:
        needles = [m.lower() for m in q.merchants]
        rows = [r for r in rows if any(n in r.merchant.lower() for n in needles)]
    return rows


def _expenses(rows):
    return [r for r in rows if r.kind == "expense" and r.category != ADJUSTMENT]


def _inr(x: float) -> str:
    """₹1,50,500 (Indian grouping), for the plain-language insights."""
    whole = f"{abs(round(x)):d}"
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{'−' if round(x) < 0 else ''}₹{whole}"


OPENING = "Opening Balance"  # what the user already had: it is money, but not earnings
ADJUSTMENT = (
    "Balance Adjustment"  # "my bank balance is really 25000": fixes the total, is not spending
)
NOT_FLOW = (OPENING, ADJUSTMENT)


def _income(rows):
    return [r for r in rows if r.kind == "income" and r.category not in NOT_FLOW]


LEADING_FORMULA_CHARS = ("=", "+", "-", "@", chr(9), chr(13))


def csv_safe(value: str) -> str:
    """Spreadsheets run cells that start with = + - @ as formulas; a leading quote makes them text."""
    text = str(value or "")
    return "'" + text if text[:1] in LEADING_FORMULA_CHARS else text


def _change(now: float, before: float) -> float | None:
    """Percent change versus the previous period, or None when it would mislead.

    No previous spending means there is nothing to compare, and a jump above +400% usually means
    the previous period was nearly empty (a new user), so it is left out rather than shown as a
    scary number.
    """
    if before <= 0:
        return None
    pct = (now - before) / before * 100
    return None if pct > 400 else _r(pct)


# =============================================================================== sections


def summary(rows: list[Row], q: Query, prev_rows: list[Row] | None = None) -> dict[str, Any]:
    exp, inc = _expenses(rows), _income(rows)
    spent = sum(r.amount for r in exp)
    earned = sum(r.amount for r in inc)
    amounts = [r.amount for r in exp]
    active_days = len({r.day for r in exp})
    days = q.effective_days(None if q.first_day() else min((r.day for r in rows), default=None))
    out: dict[str, Any] = {
        "spent": _r(spent),
        "earned": _r(earned),
        "net": _r(earned - spent),
        "savings_rate": _r((earned - spent) / earned * 100) if earned > 0 else None,
        "transactions": len(exp) + len(inc),
        "expense_count": len(exp),
        "transfers": sum(1 for r in rows if r.kind == "transfer"),
        "avg_per_day": _r(spent / days),
        "avg_expense": _r(spent / len(exp)) if exp else 0.0,
        "median_expense": _r(statistics.median(amounts)) if amounts else 0.0,
        "largest_expense": _r(max(amounts)) if amounts else 0.0,
        "active_days": active_days,
        "no_spend_days": max(0, days - active_days),
    }
    if prev_rows is not None:
        pspent = sum(r.amount for r in _expenses(prev_rows))
        pearned = sum(r.amount for r in _income(prev_rows))
        out.update(
            prev_spent=_r(pspent),
            prev_earned=_r(pearned),
            spent_change_pct=_change(spent, pspent),
            earned_change_pct=_change(earned, pearned),
        )
    return out


def category_breakdown(
    rows: list[Row], prev_rows: list[Row] | None = None, top_n: int = 8
) -> list[dict]:
    """Spending per category (top N, the rest folded into 'Other'), with share and change."""
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in _expenses(rows):
        totals[r.category] += r.amount
        counts[r.category] += 1
    prev_totals: dict[str, float] = defaultdict(float)
    for r in _expenses(prev_rows or []):
        prev_totals[r.category] += r.amount
    grand = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    head, tail = ranked[:top_n], ranked[top_n:]
    out = [
        {
            "category": name,
            "amount": _r(amount),
            "share_pct": _r(amount / grand * 100),
            "count": counts[name],
            "avg": _r(amount / counts[name]),
            "change_pct": _change(amount, prev_totals.get(name, 0.0))
            if prev_rows is not None
            else None,
        }
        for name, amount in head
    ]
    if tail:
        other = sum(a for _, a in tail)
        out.append(
            {
                "category": "Other",
                "amount": _r(other),
                "share_pct": _r(other / grand * 100),
                "count": sum(counts[n] for n, _ in tail),
                "avg": _r(other / max(1, sum(counts[n] for n, _ in tail))),
                "change_pct": None,
            }
        )
    return out


def top_merchants(rows: list[Row], n: int = 8) -> list[dict]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in _expenses(rows):
        if r.merchant:
            totals[r.merchant] += r.amount
            counts[r.merchant] += 1
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])[:n]
    return [
        {"merchant": k, "amount": _r(v), "count": counts[k], "avg": _r(v / counts[k])}
        for k, v in ranked
    ]


def income_sources(rows: list[Row], n: int = 6) -> list[dict]:
    totals: dict[str, float] = defaultdict(float)
    for r in _income(rows):
        totals[r.merchant or r.category] += r.amount
    return [
        {"source": k, "amount": _r(v)} for k, v in sorted(totals.items(), key=lambda kv: -kv[1])[:n]
    ]


def by_account(rows: list[Row]) -> list[dict]:
    spent: dict[str, float] = defaultdict(float)
    earned: dict[str, float] = defaultdict(float)
    for r in rows:
        if r.kind == "expense":
            spent[r.account or "—"] += r.amount
        elif r.kind == "income" and r.category not in NOT_FLOW:
            earned[r.account or "—"] += r.amount
    names = set(spent) | set(earned)
    return sorted(
        ({"account": n, "spent": _r(spent[n]), "earned": _r(earned[n])} for n in names),
        key=lambda r: -r["spent"],
    )


def biggest_expenses(rows: list[Row], n: int = 5) -> list[dict]:
    top = sorted(_expenses(rows), key=lambda r: -r.amount)[:n]
    return [
        {
            "id": r.id,
            "date": r.day.isoformat(),
            "amount": _r(r.amount),
            "category": r.category,
            "merchant": r.merchant,
            "note": r.note[:60],
        }
        for r in top
    ]


def pick_group(q: Query) -> str:
    if q.group != "auto":
        return q.group
    d = q.days
    return "day" if d <= 45 else "week" if d <= 200 else "month"


def _bucket(d: date, group: str) -> date:
    if group == "week":
        return d - timedelta(days=d.weekday())
    if group == "month":
        return d.replace(day=1)
    return d


def _next_bucket(d: date, group: str) -> date:
    if group == "week":
        return d + timedelta(days=7)
    if group == "month":
        return date(d.year + (d.month == 12), 1 if d.month == 12 else d.month + 1, 1)
    return d + timedelta(days=1)


def series(rows: list[Row], q: Query) -> dict[str, Any]:
    """Spending and income per day/week/month (empty buckets included), with a trailing average."""
    group = pick_group(q)
    span = max(1, (q.end - q.start).days)
    if q.extra.get("trim") and rows:
        span = max(1, (q.end.date() - min(r.day for r in rows)).days)
    while (group == "day" and span > 700) or (group == "week" and span > 700 * 7):
        group = (
            "week" if group == "day" else "month"
        )  # more than ~700 points would be cut off at the end
    spent: dict[date, float] = defaultdict(float)
    earned: dict[date, float] = defaultdict(float)
    for r in rows:
        key = _bucket(r.day, group)
        if r.kind == "expense":
            spent[key] += r.amount
        elif r.kind == "income" and r.category not in NOT_FLOW:
            earned[key] += r.amount
    first = _bucket(q.start.date(), group)
    if q.extra.get("trim") and rows:  # "all time": begin at the first transaction, not in 2000
        first = _bucket(min(r.day for r in rows), group)
    last = min(_bucket((q.end - timedelta(days=1)).date(), group), _bucket(_now().date(), group))
    points, cur, cum, window = [], first, 0.0, []
    while cur <= last and len(points) < 800:
        window.append(spent[cur])
        cum += spent[cur]
        w = 7 if group == "day" else 4
        label = (
            cur.strftime("%d %b")
            if group != "month"
            else f"{calendar.month_abbr[cur.month]} {str(cur.year)[2:]}"
        )
        points.append(
            {
                "key": cur.isoformat(),
                "label": label,
                "spent": _r(spent[cur]),
                "earned": _r(earned[cur]),
                "net": _r(earned[cur] - spent[cur]),
                "avg": _r(sum(window[-w:]) / len(window[-w:])),
                "cumulative": _r(cum),
            }
        )
        cur = _next_bucket(cur, group)
    return {"group": group, "points": points}


def weekday_pattern(rows: list[Row], q: Query) -> list[dict]:
    per_day: dict[date, float] = defaultdict(float)
    for r in _expenses(rows):
        per_day[r.day] += r.amount
    counts: dict[int, int] = defaultdict(int)
    d = q.start.date()
    first = q.first_day() or (min((r.day for r in rows), default=None))
    if first is not None and first > d:
        d = first  # days before the user's first entry are not "zero spending days"
    end = min(q.end.date(), _now().date() + timedelta(days=1))
    while d < end:
        counts[d.weekday()] += 1
        d += timedelta(days=1)
    totals: dict[int, float] = defaultdict(float)
    for day, amount in per_day.items():
        totals[day.weekday()] += amount
    return [
        {
            "day": WEEKDAYS[i],
            "avg": _r(totals[i] / counts[i]) if counts[i] else 0.0,
            "total": _r(totals[i]),
        }
        for i in range(7)
    ]


def hour_pattern(rows: list[Row]) -> list[dict] | None:
    """Spending by hour of day. None if almost all times are midnight (imported statements)."""
    exp = _expenses(rows)
    if not exp or sum(1 for r in exp if r.when.hour or r.when.minute) < max(5, len(exp) * 0.3):
        return None
    totals: dict[int, float] = defaultdict(float)
    for r in exp:
        totals[r.when.hour] += r.amount
    return [{"hour": h, "total": _r(totals[h])} for h in range(24)]


def calendar_heatmap(rows: list[Row], weeks: int = 12) -> dict[str, Any]:
    """Spend per day for the last N weeks, laid out Monday..Sunday per column."""
    today = _now().date()
    start = today - timedelta(days=today.weekday()) - timedelta(weeks=weeks - 1)
    per_day: dict[date, float] = defaultdict(float)
    for r in _expenses(rows):
        per_day[r.day] += r.amount
    cells = []
    d = start
    while d <= today:
        cells.append(
            {
                "date": d.isoformat(),
                "week": (d - start).days // 7,
                "dow": d.weekday(),
                "spent": _r(per_day.get(d, 0.0)),
            }
        )
        d += timedelta(days=1)
    return {"weeks": weeks, "cells": cells, "max": max((c["spent"] for c in cells), default=0.0)}


def size_distribution(rows: list[Row]) -> list[dict]:
    """How many expenses fall into each size band."""
    bands = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 20000), (20000, float("inf"))]
    labels = ["<100", "100-500", "500-1k", "1k-5k", "5k-20k", "20k+"]
    counts = [0] * len(bands)
    totals = [0.0] * len(bands)
    for r in _expenses(rows):
        for i, (lo, hi) in enumerate(bands):
            if lo <= r.amount < hi:
                counts[i] += 1
                totals[i] += r.amount
                break
    return [
        {"band": labels[i], "count": counts[i], "total": _r(totals[i])} for i in range(len(bands))
    ]


def category_by_period(rows: list[Row], q: Query, top: int = 5) -> dict[str, Any]:
    """Stacked-chart data: for each period, spend per top category (+ Other)."""
    group = pick_group(q)
    totals: dict[str, float] = defaultdict(float)
    for r in _expenses(rows):
        totals[r.category] += r.amount
    names = [n for n, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:top]]
    per: dict[date, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in _expenses(rows):
        per[_bucket(r.day, group)][r.category if r.category in names else "Other"] += r.amount
    keys = sorted(per)
    cols = names + (["Other"] if any("Other" in per[k] for k in keys) else [])
    return {
        "group": group,
        "categories": cols,
        "periods": [
            {
                "key": k.isoformat(),
                "label": k.strftime("%d %b") if group != "month" else k.strftime("%b %y"),
                "values": {c: _r(per[k].get(c, 0.0)) for c in cols},
            }
            for k in keys
        ],
    }


def recurring(rows: list[Row], min_hits: int = 2) -> list[dict]:
    """Merchants charged repeatedly for a near-constant amount (subscriptions, rent, bills)."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in _expenses(rows):
        if r.merchant:
            groups[r.merchant].append(r)
    out = []
    for name, rs in groups.items():
        months = {(r.when.year, r.when.month) for r in rs}
        amounts = [r.amount for r in rs]
        days = sorted(r.when for r in rs)
        gaps = [(b - a).days for a, b in zip(days, days[1:], strict=False)]
        monthly = (
            bool(gaps) and 24 <= statistics.median(gaps) <= 38
        )  # weekly milk and two chai in a row are not bills
        if (
            len(rs) >= min_hits
            and len(months) >= 2
            and monthly
            and max(amounts) <= min(amounts) * 1.15
        ):
            avg = sum(amounts) / len(amounts)
            out.append(
                {
                    "merchant": name,
                    "amount": _r(avg),
                    "times": len(rs),
                    "category": rs[0].category,
                    "yearly": _r(avg * 12),
                }
            )
    return sorted(out, key=lambda r: -r["amount"])


def unusual(rows: list[Row], history: list[Row] | None = None, n: int = 5) -> list[dict]:
    """Expenses at least 3x the usual size for their category (and over 500)."""
    base = _expenses(history if history is not None else rows)
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in base:
        by_cat[r.category].append(r.amount)
    medians = {
        c: statistics.median(v) for c, v in by_cat.items() if len(v) >= 4
    }  # once per category
    flagged = []
    for r in _expenses(rows):
        med = medians.get(r.category)
        if med is None:
            continue
        if r.amount >= 3 * med and r.amount >= 500:
            flagged.append(
                {
                    "id": r.id,
                    "date": r.day.isoformat(),
                    "amount": _r(r.amount),
                    "category": r.category,
                    "merchant": r.merchant,
                    "usual": _r(med),
                }
            )
    return sorted(flagged, key=lambda r: -r["amount"])[:n]


# =============================================================================== whole-user views


def balances(db: Session, user_id: str) -> dict[str, Any]:
    accs = list(db.scalars(select(Account).where(Account.user_id == user_id)))
    return {
        "total": _r(sum(a.balance for a in accs)),
        "accounts": [{"name": a.name, "type": a.type, "balance": _r(a.balance)} for a in accs],
    }


def net_worth_series(db: Session, user_id: str, days: int = 60) -> list[dict[str, Any]]:
    """Total money over time, rebuilt by walking back from today's balances."""
    total = balances(db, user_id)["total"]
    today = _now().date()
    first = today - timedelta(days=days - 1)
    q = Query(datetime.combine(first, datetime.min.time()), _day_start(_now()) + timedelta(days=1))
    delta: dict[date, float] = defaultdict(float)
    for r in load_rows(db, user_id, q):
        if not r.applied:
            continue
        if r.kind == "expense":
            delta[r.day] -= r.amount
        elif r.kind == "income":
            delta[r.day] += r.amount
    out, running = [], total
    for i in range(days):
        d = today - timedelta(days=i)
        out.append({"date": d.isoformat(), "label": d.strftime("%d %b"), "total": _r(running)})
        running -= delta.get(d, 0.0)
    return list(reversed(out))


def budget_status(db: Session, user_id: str) -> list[dict[str, Any]]:
    """Each category with a monthly budget: spent so far this month and what is left."""
    now = _now()
    month_start = _day_start(now).replace(day=1)
    rows = load_rows(db, user_id, Query(month_start, _day_start(now) + timedelta(days=1)))
    spent: dict[str, float] = defaultdict(float)
    for r in _expenses(rows):
        spent[r.category.lower()] += r.amount
    out = []
    for c in db.scalars(
        select(Category).where(Category.user_id == user_id, Category.budget_limit.is_not(None))
    ):
        if not c.budget_limit or c.budget_limit <= 0:
            continue
        used = spent.get(c.name.lower(), 0.0)
        pct = used / c.budget_limit * 100
        out.append(
            {
                "category": c.name,
                "budget": _r(c.budget_limit),
                "spent": _r(used),
                "left": _r(c.budget_limit - used),
                "percent": _r(pct),
                "level": "over" if pct > 100 else "warn" if pct >= 80 else "ok",
            }
        )
    return sorted(out, key=lambda r: -r["percent"])


def month_pace(db: Session, user_id: str) -> dict[str, Any]:
    """This month's cumulative spend against last month's, plus a projection to month end."""
    now = _now()
    this_start = _day_start(now).replace(day=1)
    last_start = (this_start - timedelta(days=1)).replace(day=1)
    dim = calendar.monthrange(now.year, now.month)[1]
    cur = _expenses(load_rows(db, user_id, Query(this_start, _day_start(now) + timedelta(days=1))))
    prev = _expenses(load_rows(db, user_id, Query(last_start, this_start)))

    def cumulative(rs, start, days):
        per: dict[int, float] = defaultdict(float)
        for r in rs:
            per[(r.day - start.date()).days + 1] += r.amount
        total, out = 0.0, []
        for d in range(1, days + 1):
            total += per.get(d, 0.0)
            out.append(_r(total))
        return out

    this_curve = cumulative(cur, this_start, now.day)
    prev_days = calendar.monthrange(last_start.year, last_start.month)[1]
    spent = this_curve[-1] if this_curve else 0.0

    # Projection = spent so far + the regular daily pace for the days left + bills still due.
    # Big one-offs (rent, a laptop) are not extrapolated: they happened once.
    history = _expenses(load_rows(db, user_id, Query(this_start - timedelta(days=90), this_start)))
    typical = statistics.median([r.amount for r in history]) if history else 0.0
    big = max(5000.0, typical * 8)
    regular = sum(r.amount for r in cur if r.amount < big)
    daily_rate = regular / now.day
    charged_now = {r.merchant for r in cur if r.merchant}
    still_due = sum(
        item["amount"] for item in recurring(history) if item["merchant"] not in charged_now
    )
    # one or two days say nothing about a month: only add the known bills until there is a week of data
    projected = (
        spent + daily_rate * (dim - now.day) + still_due if now.day >= 5 else spent + still_due
    )
    return {
        "day": now.day,
        "days_in_month": dim,
        "days_left": dim - now.day,
        "spent": _r(spent),
        "daily_average": _r(spent / now.day),
        "regular_daily": _r(daily_rate),
        "bills_due": _r(still_due),
        "projected": _r(projected),
        "this_month": this_curve,
        "last_month": cumulative(prev, last_start, prev_days),
        "last_month_total": _r(sum(r.amount for r in prev)),
    }


_LIQUID_TYPES = {"cash", "bank", "wallet"}


def runway(db: Session, user_id: str, days: int = 90) -> dict[str, Any]:
    """How many months the money in cash/bank/wallet accounts lasts at the recent pace of spending.

    The pace is averaged over the days that really have history (at most `days`), so a new user with
    three days of data is not told something precise.
    """
    accounts = balances(db, user_id)["accounts"]
    total = balances(db, user_id)["total"]
    liquid = sum(a["balance"] for a in accounts if a["type"].lower() in _LIQUID_TYPES)
    q = default_query(days)
    first = first_entry_day(db, user_id)
    covered = q.effective_days(first)
    spent = sum(r.amount for r in _expenses(load_rows(db, user_id, q)))
    monthly = spent / covered * 30
    return {
        "balance": total,
        "liquid_balance": _r(liquid),
        "monthly_spend": _r(monthly),
        "months": _r(liquid / monthly) if monthly > 0 and liquid > 0 else None,
        "data_days": covered,
        "low_confidence": covered < 14,  # too little history to promise a number of months
    }


# =============================================================================== insights + report


def insights(rep: dict[str, Any]) -> list[str]:
    """A few short, plain sentences worth telling the user (empty list if no data)."""
    s = rep["summary"]
    if not s["transactions"]:
        return []
    out: list[str] = []
    cats = rep["categories"]
    if cats:
        top = cats[0]
        out.append(
            f"{top['category']} is your biggest spend: {_inr(top['amount'])} ({top['share_pct']:.0f}%)."
        )
    ch = s.get("spent_change_pct")
    if ch is not None and abs(ch) >= 5:
        out.append(
            f"Spending is {'up' if ch > 0 else 'down'} {abs(ch):.0f}% compared with the period before."
        )
    movers = [
        c for c in cats if c.get("change_pct") and c["change_pct"] >= 30 and c["amount"] >= 500
    ]
    if movers:
        m = max(movers, key=lambda c: c["change_pct"])
        out.append(f"{m['category']} jumped {m['change_pct']:.0f}% compared with before.")
    sr = s.get("savings_rate")
    if sr is not None:
        if sr >= 20:
            out.append(f"You kept {sr:.0f}% of what you earned.")
        elif sr < 0:
            out.append("You spent more than you earned in this period.")
    for b in rep.get("budgets", []):
        if b["level"] != "ok":
            out.append(
                f"{b['category']} is {'over' if b['level'] == 'over' else 'close to'} its budget ({b['percent']:.0f}% used)."
            )
    rec = rep.get("recurring", [])
    if rec:
        total = sum(r["amount"] for r in rec)
        out.append(f"{len(rec)} recurring payments add up to about {_inr(total)} a month.")
    if rep.get("unusual"):
        u = rep["unusual"][0]
        out.append(
            f"One {u['category']} expense of {_inr(u['amount'])} was far above your usual {_inr(u['usual'])}."
        )
    wd = rep.get("weekdays", [])
    if (
        wd and max(w["avg"] for w in wd) > 0 and rep["query"]["days"] >= 14
    ):  # a week says nothing about weekdays
        peak = max(wd, key=lambda w: w["avg"])
        out.append(f"{peak['day']} is your costliest day (about {_inr(peak['avg'])}).")
    pace = rep.get("pace")
    if pace and pace["day"] >= 5 and rep["query"]["days"] <= 62:
        out.append(f"At this pace the month ends near {_inr(pace['projected'])}.")
    return out[:7]


def report(
    db: Session, user_id: str, q: Query | None = None, compare: bool = True
) -> dict[str, Any]:
    """Everything the dashboard, chat messages and chart images need, for one query."""
    q = q or default_query(30)
    first = first_entry_day(db, user_id)
    q = replace(q, extra={**q.extra, "first_day": first})
    rows = load_rows(db, user_id, q)
    prev_rows = load_rows(db, user_id, q.previous()) if compare else None
    history = load_rows(db, user_id, replace(q, start=q.start - timedelta(days=90), end=q.end))
    rep: dict[str, Any] = {
        "query": {
            "label": q.label,
            "describe": q.describe(),
            "start": q.start.date().isoformat(),
            "end": (q.end - timedelta(days=1)).date().isoformat(),
            "days": q.days,
            "filtered": q.filtered,
        },
        "generated_at": _now().isoformat(timespec="seconds"),
        "summary": summary(rows, q, prev_rows),
        "categories": category_breakdown(rows, prev_rows, q.top_n),
        "merchants": top_merchants(rows, q.top_n),
        "income_sources": income_sources(rows),
        "accounts": by_account(rows),
        "biggest": biggest_expenses(rows),
        "series": series(rows, q),
        "stack": category_by_period(rows, q),
        "weekdays": weekday_pattern(rows, q),
        "hours": hour_pattern(rows),
        "sizes": size_distribution(rows),
        "recurring": recurring(history),
        "unusual": unusual(rows, history),
        # always the last 12 weeks up to today (whatever period is shown), with the same filters
        "heatmap": calendar_heatmap(
            load_rows(
                db,
                user_id,
                replace(
                    q,
                    start=_day_start(_now()) - timedelta(weeks=13),
                    end=_day_start(_now()) + timedelta(days=1),
                ),
            )
        ),
        "balances": balances(db, user_id),
        "budgets": budget_status(db, user_id),
        "pace": month_pace(db, user_id),
        "runway": runway(db, user_id),
        "net_worth": net_worth_series(db, user_id, min(max(q.days, 30), 365)),
    }
    rep["insights"] = insights(rep)
    return rep


# =============================================================================== planning

_ESSENTIAL_WORDS = {
    "rent", "bills", "utilities", "utility", "health", "medical", "medicine", "education", "school",
    "insurance", "loan", "emi", "groceries", "grocery", "tax", "taxes", "investments", "investment",
    "savings", "transfers", "transfer", "cash", "fees", "fee", "electricity", "water", "mortgage",
}  # fmt: skip


def _is_essential(category: str) -> bool:
    """True if any word of the category name is a need, an obligation or not really "spending"."""
    return any(word in _ESSENTIAL_WORDS for word in re.findall(r"[a-z]+", category.lower()))


def savings_plan(db: Session, user_id: str, target: float, months: int) -> dict[str, Any]:
    """Can the user save `target` rupees in `months` months, and where could the money come from?

    Uses the last 90 days: average monthly income and spending, so the answer rests on real numbers.
    Suggestions only trim discretionary categories (never rent, bills, health, groceries).
    """
    if target <= 0 or months <= 0:
        raise ValueError("target and months must both be greater than zero")
    q = default_query(90)
    covered_days = q.effective_days(first_entry_day(db, user_id))
    # two months of history or more counts as the full three-month window; less is averaged over what exists
    months_covered = 3.0 if covered_days >= 60 else max(covered_days / 30.4, 0.25)
    rows = load_rows(db, user_id, q)
    income = sum(r.amount for r in _income(rows)) / months_covered
    spend = sum(r.amount for r in _expenses(rows)) / months_covered
    surplus = income - spend
    needed = target / months
    gap = max(0.0, needed - max(0.0, surplus))
    cats = sorted(category_breakdown(rows, None, top_n=20), key=lambda c: -c["amount"])
    trim = []
    remaining = gap
    for c in cats:
        if remaining <= 0.5:
            break
        if _is_essential(c["category"]) or c["category"] in (UNCATEGORIZED, "Other"):
            continue
        monthly = c["amount"] / 3
        cut = min(monthly * 0.4, remaining)  # never ask for more than 40% of one category
        if cut >= 50:
            trim.append(
                {
                    "category": c["category"],
                    "monthly_spend": _r(monthly),
                    "cut": _r(cut),
                    "percent": _r(cut / monthly * 100),
                }
            )
            remaining -= cut
    return {
        "target": _r(target),
        "months": months,
        "needed_per_month": _r(needed),
        "avg_income": _r(income),
        "avg_spend": _r(spend),
        "avg_surplus": _r(surplus),
        "has_income_data": income > 0,
        "data_days": covered_days,
        "low_confidence": covered_days < 30,  # under a month of history: a rough guess
        "on_track": surplus >= needed,
        "gap_per_month": _r(gap),
        "suggested_cuts": trim,
        "still_short": _r(max(0.0, remaining)),
    }


def financial_snapshot(db: Session, user_id: str, days: int = 30) -> dict[str, Any]:
    """A compact, plain summary (for the AI to reason over): numbers only, no charts."""
    rep = report(db, user_id, default_query(days))
    s = rep["summary"]
    return {
        "period": rep["query"]["label"],
        "spent": s["spent"],
        "earned": s["earned"],
        "net": s["net"],
        "savings_rate_percent": s["savings_rate"],
        "spent_change_percent": s.get("spent_change_pct"),
        "top_categories": [
            {
                "name": c["category"],
                "amount": c["amount"],
                "share_percent": c["share_pct"],
                "change_percent": c["change_pct"],
            }
            for c in rep["categories"][:6]
        ],
        "top_merchants": [
            {"name": m["merchant"], "amount": m["amount"]} for m in rep["merchants"][:5]
        ],
        "budgets": [
            {
                "category": b["category"],
                "spent": b["spent"],
                "budget": b["budget"],
                "level": b["level"],
            }
            for b in rep["budgets"]
        ],
        "recurring": [
            {"name": r["merchant"], "monthly": r["amount"]} for r in rep["recurring"][:6]
        ],
        "month_end_projection": rep["pace"]["projected"],
        "money_now": rep["balances"]["total"],
        "months_of_runway": rep["runway"]["months"],
        "insights": rep["insights"],
    }
