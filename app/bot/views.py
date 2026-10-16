"""The screens of the bot: stats, charts, history, budgets, balance, undo, export, dashboard link.

Every function returns a `Reply` (see ui.py). Database and chart work runs in a worker thread so the
bot stays responsive. Messages are kept short: a headline, a few numbers, and buttons for the rest.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import re
from collections import OrderedDict

from app import clock
from app.bot import fmt, ui
from app.bot.categories import canonical_category
from app.bot.textutil import parse_amount, parse_goal
from app.bot.ui import Btn, Reply
from app.db import SessionLocal
from app.services import analytics, ledger
from app.web import tokens

logger = logging.getLogger(__name__)

HISTORY_PAGE = 8
_ICONS = {"cash": "💵", "bank": "🏦", "wallet": "👛", "credit_card": "💳"}

# Filters from /stats food hdfc are remembered under a short id so a button can refer to them
# (Telegram limits button data to 64 bytes). In memory only: after a restart the filters reset.
_ctx: OrderedDict[str, list[str]] = OrderedDict()


def _remember(words: list[str]) -> str:
    if not words:
        return ""
    key = format(abs(hash(tuple(words))) % (36**5), "x")
    _ctx[key] = words
    _ctx.move_to_end(key)
    while len(_ctx) > 500:
        _ctx.popitem(last=False)
    return key


def _words_for(ctx: str) -> list[str]:
    return list(_ctx.get(ctx, [])) if ctx else []


def parse_args(text: str) -> tuple[str, list[str]]:
    """'last month food hdfc' -> ('last month', ['food', 'hdfc']). Period defaults to 30d."""
    tokens_ = [t for t in (text or "").replace(",", " ").split() if t]
    for start in range(len(tokens_)):  # "food 7d" and "last month food" both work
        for n in (2, 1):
            chunk = tokens_[start : start + n]
            if len(chunk) == n and analytics.parse_period(" ".join(chunk)):
                return " ".join(chunk), tokens_[:start] + tokens_[start + n :]
    return "30d", tokens_


def _user_id(db, chat_id: str) -> str:
    return ledger.get_or_create_user(db, str(chat_id)).id


# ----------------------------------------------------------------------------------------- stats


def _compute(chat_id: str, period: str, words: list[str]):
    with SessionLocal() as db:
        uid = _user_id(db, chat_id)
        q = analytics.parse_period(period) or analytics.default_query(30)
        q, unmatched = analytics.resolve_filters(db, uid, q, words)
        return analytics.report(db, uid, q), unmatched


def caption(rep: dict, unmatched: list[str] | None = None) -> str:
    q, s = rep["query"], rep["summary"]
    lines = [f"📊 {q['describe']}"]
    if not s["transactions"]:
        lines.append("Nothing logged in this period yet.")
        lines.append("Try: Spent 450 on dinner via Bank")
    else:
        spent = f"Spent {fmt.inr(s['spent'])}"
        ch = fmt.change(s.get("spent_change_pct"))
        spent += f" {ch}" if ch else ""
        if q["filtered"]:  # a slice (one category, account...): earned/saved would be misleading
            lines.append(f"{spent}  ·  {fmt.plural(s['expense_count'], 'purchase')}")
            lines.append(
                f"Average {fmt.inr(s['avg_expense'])}  ·  largest {fmt.inr(s['largest_expense'])}"
            )
            top = rep["merchants"][:3]
            if top:
                lines.append(
                    "Top: " + " · ".join(f"{m['merchant']} {fmt.short(m['amount'])}" for m in top)
                )
        else:
            lines.append(f"{spent}  ·  Earned {fmt.inr(s['earned'])}")
            saved = (
                f"Saved {fmt.inr(s['net'])}"
                if s["net"] >= 0
                else f"Spent {fmt.inr(-s['net'])} more than earned"
            )
            if s["savings_rate"] is not None and s["net"] >= 0:
                saved += f" ({s['savings_rate']:.0f}%)"
            lines.append(f"{saved}  ·  {fmt.inr(s['avg_per_day'])} a day")
            top = rep["categories"][:3]
            if top:
                lines.append(
                    "Top: " + " · ".join(f"{c['category']} {fmt.short(c['amount'])}" for c in top)
                )
        if rep["insights"] and not q["filtered"]:
            lines.append("")
            lines.append("💡 " + rep["insights"][0])
    if unmatched:
        names = ", ".join(f"“{w}”" for w in unmatched[:3])
        lines.append(f"\nI couldn't match {names} to a category, account or merchant.")
    return "\n".join(lines)[:1000]


async def stats(chat_id: str, args: str = "", *, edit: bool = False) -> Reply:
    period, words = parse_args(args)
    return await stats_for(chat_id, period, words, edit=edit)


async def stats_for(chat_id: str, period: str, words: list[str], *, edit: bool = False) -> Reply:
    rep, unmatched = await asyncio.to_thread(_compute, chat_id, period, words)
    text = caption(rep, unmatched)
    ctx = _remember([w for w in words if w not in unmatched])
    rows = ui.stats_rows(period if period in dict(ui.PERIODS) else "30d", ctx)
    if not rep["summary"]["transactions"]:
        return Reply(text, rows=[period_only(period, ctx)], edit=edit)
    photo = await _render(rep, None)
    return Reply(text, photo=photo, rows=rows, edit=edit)


def period_only(period: str, ctx: str) -> list[Btn]:
    return ui.period_row(period, ctx)


async def _render(rep: dict, kind: str | None) -> bytes | None:
    try:
        from app import charts

        if kind is None:
            return await asyncio.to_thread(charts.render_dashboard, rep)
        return await asyncio.to_thread(charts.render_chart, rep, kind)
    except Exception:
        logger.exception("Chart rendering failed")
        return None


async def chart(chat_id: str, kind: str, period: str = "30d", ctx: str = "") -> Reply:
    rep, _ = await asyncio.to_thread(_compute, chat_id, period, _words_for(ctx))
    label = dict(ui.CHARTS).get(kind, kind)
    photo = await _render(rep, kind)
    text = f"{label} · {rep['query']['describe']}"
    if photo is None:
        text += "\nCouldn't draw the chart right now."
    return Reply(text, photo=photo, rows=ui.chart_nav_rows(kind, period, ctx), edit=True)


# --------------------------------------------------------------------------------------- balance


def _balance_text(chat_id: str) -> str:
    with SessionLocal() as db:
        data = ledger.get_account_balances(db, _user_id(db, chat_id))
    return balance_text(data)


def balance_text(data: dict) -> str:
    """The balance screen from get_account_balances(): one look, same wording everywhere."""
    lines = [f"💰 Total  {fmt.inr(data['total_net_worth'])}", ""]
    for a in sorted(data["accounts"], key=lambda a: -a["balance"]):
        icon = _ICONS.get(a["type"], "🏦")
        lines.append(f"{icon} {a['name']}  {fmt.inr(a['balance'])}")
    if not data["accounts"]:
        lines.append("No accounts yet. Say: Received 5000 in Bank")
    elif any(a["balance"] < 0 for a in data["accounts"]):
        lines.append("")
        lines.append(
            "💡 Low or negative? I only know what you've told me. Tell me the real amount, like “bank balance is 25000”."
        )
    return "\n".join(lines)


async def balance(chat_id: str) -> Reply:
    text = await asyncio.to_thread(_balance_text, chat_id)
    rows = [[Btn("📊 Stats", "go:stats"), Btn("🧾 History", "go:history")]]
    return Reply(text, rows=rows)


# --------------------------------------------------------------------------------------- history


def _history(chat_id: str, page: int):
    with SessionLocal() as db:
        uid = _user_id(db, chat_id)
        rows, total = ledger.recent_transactions(db, uid, HISTORY_PAGE, page * HISTORY_PAGE)
        return rows, total, ledger.last_entered_transaction_id(db, uid)


async def history(chat_id: str, page: int = 0, *, edit: bool = False) -> Reply:
    page = max(0, page)
    rows, total, last_id = await asyncio.to_thread(_history, chat_id, page)
    if not total:
        return Reply(
            "🧾 No transactions yet.\nTry: Spent 450 on dinner via Bank", rows=ui.back_home()
        )
    pages = max(1, -(-total // HISTORY_PAGE))
    if page >= pages:
        return await history(chat_id, pages - 1, edit=edit)
    lines = []
    for r in rows:
        if r["type"] == "transfer":
            sign, label = "⇄", f"{r['account']} → {r['to_account']}"
        else:
            sign = "−" if r["type"] == "expense" else "+"
            label = r["merchant"] or r["category"] or r["note"] or "—"
        amount = f"{sign}{fmt.inr(r['amount'])}"
        lines.append(f"{fmt.day(r['date'])} {amount:>11}  {label[:22]}")
    first = page * HISTORY_PAGE + 1
    head = f"🧾 Transactions {first}–{first + len(rows) - 1} of {total}"
    return Reply(
        f"{head}\n```\n" + "\n".join(lines) + "\n```",
        rows=ui.history_rows(page, pages, last_id),
        edit=edit,
    )


# ---------------------------------------------------------------------------------------- export


def _csv(chat_id: str, period: str, words: list[str] | None = None) -> tuple[bytes, int]:
    with SessionLocal() as db:
        uid = _user_id(db, chat_id)
        q = analytics.parse_period(period) or analytics.default_query(30)
        if words:
            q, _ = analytics.resolve_filters(db, uid, q, words)  # "/export food last month"
        rows = analytics.load_rows(db, uid, q, notes=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Date", "Type", "Amount", "Category", "Merchant", "Account", "Note"])
    for r in rows:
        w.writerow(
            [
                r.when.strftime("%Y-%m-%d"),
                r.kind,
                f"{r.amount:.2f}",
                *(analytics.csv_safe(v) for v in (r.category, r.merchant, r.account, r.note)),
            ]
        )
    return ("﻿" + buf.getvalue()).encode("utf-8"), len(rows)  # BOM so Excel reads ₹ and accents


async def export_csv(chat_id: str, period: str = "30d", words: list[str] | None = None) -> Reply:
    data, count = await asyncio.to_thread(_csv, chat_id, period, words or [])
    name = f"walletledger-{period}-{clock.now_local().strftime('%Y%m%d')}.csv"
    if not count:
        return Reply("Nothing to export in that period.")
    return Reply(f"📄 {fmt.plural(count, 'transaction')} exported", document=(data, name))


# --------------------------------------------------------------------------------------- budgets


def _set_budget(chat_id: str, category: str, amount: float | None) -> str:
    with SessionLocal() as db:
        uid = _user_id(db, chat_id)
        # "food"/"meals" mean the user's "Food & Dining": file the budget under the canonical
        # category name so later expenses count against it, even if no such category exists yet.
        cat = ledger.resolve_category(db, uid, canonical_category(category), "expense")
        cat.budget_limit = amount
        db.commit()
        return cat.name


def _budget_lines(chat_id: str) -> list[dict]:
    with SessionLocal() as db:
        return analytics.budget_status(db, _user_id(db, chat_id))


async def budgets(chat_id: str, args: str = "") -> Reply:
    words = (args or "").replace("₹", " ").split()
    if len(words) >= 2:
        name_words, last = words[:-1], words[-1].lower()
        if name_words and name_words[0].lower() in ("set", "to"):
            name_words = name_words[1:]
        category = " ".join(w for w in name_words if w.lower() not in ("budget", "to", "at"))
        try:
            if last in ("off", "none", "clear", "remove", "0"):
                name = await asyncio.to_thread(_set_budget, chat_id, category, None)
                return Reply(
                    f"🎯 Budget for {name} removed.", rows=[[Btn("🎯 Budgets", "go:budgets")]]
                )
            parsed = parse_amount(last)
            if parsed is None:
                raise ValueError("amount")
            amount = ledger.validate_amount(parsed)
            if category:
                name = await asyncio.to_thread(_set_budget, chat_id, category, amount)
                return Reply(
                    f"🎯 {name} budget set to {fmt.inr(amount)} a month.",
                    rows=[[Btn("🎯 Budgets", "go:budgets")]],
                )
        except ValueError:
            return Reply("Use: /budget food 6000   (or /budget food off to remove it)")
    rows = await asyncio.to_thread(_budget_lines, chat_id)
    if not rows:
        return Reply(
            "🎯 No budgets yet.\nSet one: /budget food 6000\nI'll warn you when you get close.",
            rows=ui.back_home(),
        )
    lines = ["🎯 Budgets this month", ""]
    for b in rows:
        icon = "🔴" if b["level"] == "over" else "🟠" if b["level"] == "warn" else "🟢"
        lines.append(f"{icon} {b['category']}  {fmt.short(b['spent'])} of {fmt.short(b['budget'])}")
        lines.append(f"   {fmt.bar(b['percent'])} {b['percent']:.0f}%")
    kb = [[Btn("🎯 Chart", "ch:budgets:30d"), Btn("📊 Stats", "go:stats")]]
    return Reply("\n".join(lines), rows=kb)


# ------------------------------------------------------------------------------------------ plan

PLAN_USAGE = "🎯 Savings plan\nTry: /plan 50000 6   (save ₹50,000 in 6 months)\nOr just say: I want to save 50k in 6 months"


def _plan(chat_id: str, target: float, months: int) -> dict:
    with SessionLocal() as db:
        return analytics.savings_plan(db, _user_id(db, chat_id), target, months)


def plan_text(p: dict) -> str:
    months = f"{p['months']} month{'s' if p['months'] != 1 else ''}"
    lines = [
        f"🎯 Save {fmt.inr(p['target'])} in {months}",
        f"Needed: {fmt.inr(p['needed_per_month'])} a month",
    ]
    if not p["has_income_data"]:
        lines.append(
            "I don't see income in your last 90 days, so I can't tell yet. Log your salary and ask again."
        )
        return "\n".join(lines)
    sign = "+" if p["avg_surplus"] >= 0 else "−"
    lines.append(
        f"You keep about {sign}{fmt.inr(abs(p['avg_surplus']))} a month "
        f"({fmt.short(p['avg_income'])} in, {fmt.short(p['avg_spend'])} out)"
    )
    if p.get("low_confidence"):
        lines.append(
            f"⚠️ Only {p['data_days']} days of history so far, so treat this as a rough guess."
        )
    if p["on_track"]:
        lines.append("✅ On track at your current pace.")
        return "\n".join(lines)
    lines.append(f"⚠️ Short by {fmt.inr(p['gap_per_month'])} a month. You could trim:")
    for c in p["suggested_cuts"]:
        lines.append(f"• {c['category']}  −{fmt.inr(c['cut'])} ({c['percent']:.0f}%)")
    if not p["suggested_cuts"]:
        lines.append("• nothing flexible enough to cut")
    if p["still_short"] > 0:
        lines.append(f"Still {fmt.inr(p['still_short'])} short: try more months or a lower target.")
    return "\n".join(lines)


async def plan(chat_id: str, args: str = "") -> Reply:
    goal = parse_goal(args)
    if goal is None:
        return Reply(PLAN_USAGE, rows=ui.back_home())
    result = await asyncio.to_thread(_plan, chat_id, *goal)
    return Reply(
        plan_text(result), rows=[[Btn("📊 Stats", "go:stats"), Btn("🎯 Budgets", "go:budgets")]]
    )


# ------------------------------------------------------------------------------------------ undo


def _undo(chat_id: str, ident: str) -> Reply:
    with SessionLocal() as db:
        uid = _user_id(db, chat_id)
        if ident in ("", "last"):
            latest = ledger.last_entered_transaction_id(db, uid)
            if not latest:
                return Reply("Nothing to undo yet.")
            ident = latest
        elif re.fullmatch(r"b[0-9a-f]{8}", ident):  # a big batch remembered by ui.undo_row
            batch = ui.undo_batch(ident[1:])
            if batch is None:
                return Reply(
                    "That undo button has expired. Use /history to remove entries.",
                    notice="Expired",
                )
            ident = ",".join(batch)
        done, failed = [], []
        for part in ident.split(","):
            try:
                done.append(ledger.undo_transaction(db, uid, part))
            except ledger.LedgerError as exc:
                failed.append(str(exc))
    if not done:
        return Reply(
            f"⚠️ {failed[0] if failed else 'Could not undo that.'}", notice="Could not undo"
        )
    lines = []
    for d in done:
        if d["type"] == "transfer":
            what = f"transfer {fmt.inr(d['amount'])} ({d['account']} → {d['to_account']})"
        else:
            what = f"{'expense' if d['type'] == 'expense' else 'income'} {fmt.inr(d['amount'])}"
        lines.append(f"↩️ Undone: {what}")
    last = done[-1]
    if last.get("new_balance") is not None and last["type"] != "transfer":
        lines.append(f"{last['account']} is back to {fmt.inr(last['new_balance'])}")
    if failed:
        lines.append(f"⚠️ {len(failed)} could not be undone (already removed?)")
    return Reply("\n".join(lines), notice="Undone", edit=True)


async def undo(chat_id: str, ident: str = "last") -> Reply:
    return await asyncio.to_thread(_undo, chat_id, ident)


# -------------------------------------------------------------------------------------- dashboard


async def dashboard(chat_id: str) -> Reply:
    url = tokens.dashboard_url(chat_id)
    minutes = _minutes()
    if tokens.is_https():
        rows = [[Btn("🖥 Open dashboard", web_app=url)]]
        text = f"🖥 Your dashboard is ready (link valid {minutes} min)."
    else:
        rows = []
        text = (
            f"🖥 Your dashboard (open on this computer, valid {minutes} min):\n{url}\n\n"
            "To open it inside Telegram, set PUBLIC_URL to an https address."
        )
    return Reply(text, rows=rows)


def _minutes() -> int:
    from app.config import settings

    return settings.dashboard_link_minutes


# ---------------------------------------------------------------------------------------- misc


def home() -> Reply:
    return Reply(
        "What would you like to do?\nOr just type, e.g. “Spent 450 on dinner”.", rows=ui.home_rows()
    )
