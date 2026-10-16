"""The web dashboard: a read-only page opened from a signed link (or as a Telegram Mini App).

    GET  /d/{token}                  the page (static HTML; the token is read by the page's script)
    GET  /api/dashboard              report as JSON      header: X-Dashboard-Token
    GET  /api/transactions           paged, searchable list
    GET  /api/export.csv             the filtered rows as CSV
    POST /api/webapp/auth            Telegram Mini App sign-in: {"initData": "..."} -> {"token": "..."}

Everything is scoped to the chat id inside the token, so nobody can see another user's data. The page
only reads; it cannot change anything. Responses are never cached.
"""

from __future__ import annotations

import asyncio
import csv
import io
import math
import time
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from app import clock
from app.config import bot_tokens, is_chat_allowed
from app.db import SessionLocal
from app.models import Account, Category, Merchant
from app.services import analytics, ledger
from app.web import tokens

router = APIRouter(tags=["Dashboard"])


STATIC = Path(__file__).parent / "static"

SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' https://telegram.org; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors https://web.telegram.org https://*.telegram.org https://t.me"
    ),
}

_hits: dict[str, deque] = defaultdict(deque)
RATE_LIMIT = 90  # requests per minute per user


def _limit(chat_id: str) -> None:
    now = time.monotonic()
    q = _hits[chat_id]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests, slow down a little")
    q.append(now)


def _chat_from(token: str | None) -> str:
    chat_id = tokens.read_token(token or "")
    if chat_id is None:
        raise HTTPException(
            status_code=401, detail="This link has expired. Ask the bot for /dashboard again."
        )
    if not is_chat_allowed(chat_id):
        raise HTTPException(status_code=403, detail="This bot is private.")
    _limit(chat_id)
    return chat_id


def _json(data, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status, headers=SECURITY_HEADERS)


def _csv_list(value: str | None) -> tuple[str, ...]:
    return tuple(v.strip() for v in (value or "").split(",") if v.strip())[:20]


def _float(value: str | None) -> float | None:
    try:
        number = float(value) if value not in (None, "") else None
    except ValueError:
        raise HTTPException(status_code=400, detail="Amounts must be numbers") from None
    if number is not None and not math.isfinite(number):  # "inf" and "nan" parse as floats
        raise HTTPException(status_code=400, detail="Amounts must be ordinary numbers")
    return number


def build_query(
    period: str = "30d",
    start: str | None = None,
    end: str | None = None,
    categories: str | None = None,
    accounts: str | None = None,
    merchants: str | None = None,
    min_amount: str | None = None,
    max_amount: str | None = None,
    group: str = "auto",
) -> analytics.Query:
    if start and end:
        q = analytics.parse_period(f"{start}..{end}")
        if q is None:
            raise HTTPException(status_code=400, detail="Use dates like 2026-09-01 and 2026-09-30")
    else:
        q = analytics.parse_period(period) or analytics.default_query(30)
    if group not in ("auto", "day", "week", "month"):
        group = "auto"
    return replace(
        q,
        categories=_csv_list(categories),
        accounts=_csv_list(accounts),
        merchants=_csv_list(merchants),
        min_amount=_float(min_amount),
        max_amount=_float(max_amount),
        group=group,
        top_n=10,
    )


def _query_from_request(request: Request) -> analytics.Query:
    p = request.query_params
    return build_query(
        period=p.get("period", "30d"),
        start=p.get("start"),
        end=p.get("end"),
        categories=p.get("categories"),
        accounts=p.get("accounts"),
        merchants=p.get("merchants"),
        min_amount=p.get("min"),
        max_amount=p.get("max"),
        group=p.get("group", "auto"),
    )


def _options(db, uid: str) -> dict:
    cats = sorted(db.scalars(select(Category.name).where(Category.user_id == uid)), key=str.lower)
    accs = sorted(db.scalars(select(Account.name).where(Account.user_id == uid)), key=str.lower)
    mers = sorted(db.scalars(select(Merchant.name).where(Merchant.user_id == uid)), key=str.lower)
    return {"categories": cats, "accounts": accs, "merchants": mers[:300]}


# ----------------------------------------------------------------------------------- the page


@router.get("/d/{token}", include_in_schema=False)
async def page(token: str):
    # The page itself holds no data, so it is served even for an expired link: the script then
    # shows a friendly "link expired" message when the API says 401.
    return FileResponse(STATIC / "dashboard.html", headers=SECURITY_HEADERS)


@router.get("/d", include_in_schema=False)
async def page_for_mini_app():
    """Telegram Mini App entry: the script signs in with Telegram's initData."""
    return FileResponse(STATIC / "dashboard.html", headers=SECURITY_HEADERS)


# ------------------------------------------------------------------------------------- the API


@router.get("/api/dashboard")
async def dashboard_data(request: Request, x_dashboard_token: str | None = Header(default=None)):
    chat_id = _chat_from(x_dashboard_token)
    q = _query_from_request(request)
    compare = request.query_params.get("compare", "1") != "0"

    def work() -> dict:
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, chat_id).id
            rep = analytics.report(db, uid, q, compare=compare)
            rep["options"] = _options(db, uid)
            return rep

    rep = await asyncio.to_thread(work)
    rep["applied"] = {
        "period": request.query_params.get("period", "30d"),
        "categories": list(q.categories),
        "accounts": list(q.accounts),
        "merchants": list(q.merchants),
        "min": q.min_amount,
        "max": q.max_amount,
        "group": q.group,
    }
    return _json(rep)


@router.get("/api/transactions")
async def transactions(request: Request, x_dashboard_token: str | None = Header(default=None)):
    chat_id = _chat_from(x_dashboard_token)
    q = _query_from_request(request)
    p = request.query_params
    search = (p.get("q") or "").strip().lower()
    sort = p.get("sort", "date")
    desc = p.get("dir", "desc") != "asc"
    try:
        page = max(1, int(p.get("page", "1")))
        per_page = min(100, max(5, int(p.get("per_page", "25"))))
    except ValueError:
        raise HTTPException(status_code=400, detail="page and per_page must be numbers") from None
    kind = p.get("type")

    def load() -> list:
        with SessionLocal() as db:
            return analytics.load_rows(db, ledger.get_or_create_user(db, chat_id).id, q, notes=True)

    rows = await asyncio.to_thread(load)
    if kind in ("expense", "income", "transfer"):
        rows = [r for r in rows if r.kind == kind]
    if search:
        rows = [
            r
            for r in rows
            if search
            in " ".join(
                [
                    r.merchant,
                    r.category,
                    r.note,
                    r.account,
                    f"{r.amount:.2f}".rstrip("0").rstrip("."),
                ]
            ).lower()
        ]
    keys = {
        "date": lambda r: r.when,
        "amount": lambda r: r.amount,
        "category": lambda r: r.category.lower(),
        "merchant": lambda r: r.merchant.lower(),
    }
    rows.sort(key=keys.get(sort, keys["date"]), reverse=desc)
    total = len(rows)
    sliced = rows[(page - 1) * per_page : page * per_page]
    return _json(
        {
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": max(1, -(-total // per_page)),
            "sum_expense": round(sum(r.amount for r in rows if r.kind == "expense"), 2),
            "sum_income": round(sum(r.amount for r in rows if r.kind == "income"), 2),
            "rows": [
                {
                    "id": r.id,
                    "date": r.when.strftime("%Y-%m-%d"),
                    "time": r.when.strftime("%H:%M") if (r.when.hour or r.when.minute) else "",
                    "type": r.kind,
                    "amount": r.amount,
                    "category": r.category,
                    "merchant": r.merchant,
                    "account": r.account,
                    "note": r.note,
                }
                for r in sliced
            ],
        }
    )


@router.get("/api/export.csv")
async def export_csv(request: Request, x_dashboard_token: str | None = Header(default=None)):
    chat_id = _chat_from(x_dashboard_token)
    q = _query_from_request(request)

    def load() -> list:
        with SessionLocal() as db:
            return analytics.load_rows(db, ledger.get_or_create_user(db, chat_id).id, q, notes=True)

    rows = await asyncio.to_thread(load)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Date", "Type", "Amount", "Category", "Merchant", "Account", "Note"])
    for r in reversed(rows):
        w.writerow(
            [
                r.when.strftime("%Y-%m-%d"),
                r.kind,
                f"{r.amount:.2f}",
                *(analytics.csv_safe(v) for v in (r.category, r.merchant, r.account, r.note)),
            ]
        )
    name = f"walletledger-{clock.now_local().strftime('%Y%m%d')}.csv"
    headers = {**SECURITY_HEADERS, "Content-Disposition": f'attachment; filename="{name}"'}
    return Response(("﻿" + buf.getvalue()).encode("utf-8"), media_type="text/csv", headers=headers)


class InitData(BaseModel):
    initData: str = Field(max_length=8192)  # Telegram's initData is well under 2 KB


@router.post("/api/webapp/auth")
async def webapp_auth(body: InitData):
    """Sign in from inside Telegram: verifies Telegram's signature and returns a short-lived token."""
    configured = bot_tokens()
    if not configured:
        raise HTTPException(status_code=404, detail="Telegram is not configured")
    # With several bots, the page was opened from one of them: its signature is the one that matches
    user = next(
        (u for u in (tokens.validate_init_data(body.initData, t) for t in configured) if u), None
    )
    if user is None:
        raise HTTPException(status_code=401, detail="Could not verify the Telegram sign-in")
    chat_id = str(
        user["id"]
    )  # in a private chat the chat id equals the user id (validate_init_data checks it is there)
    if not is_chat_allowed(chat_id):
        raise HTTPException(status_code=403, detail="This bot is private.")
    return _json({"token": tokens.make_token(chat_id, ttl_seconds=3600)})
