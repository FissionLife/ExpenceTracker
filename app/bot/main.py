"""Telegram entry point: commands, statement uploads, and the shared message handler.

Every channel uses the same logic so there is one code path:
    real Telegram (aiogram polling or webhook, see app/bot/telegram.py)
    /bot/webhook without a configured bot (dev curl testing)
    /bot/simulate_chat (teammates without a bot token)
        -> handle_incoming() / handle_document() -> the agent (agent.py) and the ledger.
"""

from __future__ import annotations

import asyncio
import difflib
import hmac
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select

from app import clock
from app.bot import agent, onboarding, ui, views
from app.bot.agent import process_user_interaction_ex
from app.bot.keys import KEY_COMMANDS, handle_admin_command, handle_key_command
from app.bot.ui import Reply
from app.config import dev_endpoints_enabled, is_chat_allowed
from app.db import SessionLocal
from app.health import user_status_text
from app.models import Merchant, Transaction
from app.services.ledger import get_or_create_user
from app.services_ai import tool_client
from app.services_ai.ai_gateway import extract_api_keys
from app.services_ai.transaction_parser import parse_statement_file, summarize_statement

PRIVATE_BOT_TEXT = "🔒 This bot is private."

logger = logging.getLogger(__name__)

bot_router = APIRouter(prefix="/bot", tags=["Telegram Bot"])

IMPORT_TTL = timedelta(minutes=10)
PREVIEW_ROWS = 5
IMPORT_ACCOUNT = "Bank"
SUPPORTED_UPLOADS = {".pdf": "pdf", ".txt": "text"}

# (command, short description). This list drives BOTH Telegram's "/" menu hints and /help.
COMMAND_MENU: list[tuple[str, str]] = [
    ("start", "Welcome and quick start"),
    ("menu", "Open the quick menu"),
    ("balance", "Your accounts and total money"),
    ("stats", "Spending stats and charts"),
    ("history", "Recent transactions"),
    ("undo", "Undo the last entry"),
    ("budget", "Monthly budgets: /budget food 6000"),
    ("plan", "Savings goal: /plan 50000 6"),
    ("dashboard", "Open your web dashboard"),
    ("export", "Download your data as CSV"),
    ("coach", "Friendly savings advice"),
    ("budgeter", "Strict budget advice"),
    ("summary", "Quick summary"),
    ("keys", "Manage your AI keys"),
    ("setkey", "Add your AI key"),
    ("status", "AI and storage health"),
    ("help", "How to use WalletLedger"),
]
# Commands that work but are not in the menu (the menu stays short; buttons cover most of these).
HIDDEN_COMMANDS = (
    "report",
    "chart",
    "testkey",
    "editkey",
    "delkey",
    "pausekey",
    "resumekey",
    "key",
    "admin",
)
KNOWN_COMMANDS = {f"/{name}" for name, _ in COMMAND_MENU} | {f"/{c}" for c in HIDDEN_COMMANDS}

HELP_TEXT = (
    "ℹ️ **WalletLedger**\n\n"
    "**Add** (just type)\n"
    "• Spent 450 on dinner via Bank\n"
    "• Received 60000 salary in Bank\n"
    "• Transferred 2000 from Bank to Cash\n"
    "• Send a PhonePe / GPay / bank PDF\n\n"
    "**See**\n"
    "/balance · /stats · /history · /budget · /plan · /dashboard · /export\n"
    "Try: /stats 90d food · /stats last month · /chart categories\n\n"
    "**Fix**\n"
    "/undo removes the last entry you made (or tap ↩️ Undo)\n\n"
    "**Advice**\n"
    "/coach · /budgeter · /summary\n\n"
    "**AI**\n"
    "/setkey <key> · /keys · /status\n"
    "Your own key is used first; a shared key is the backup.\n\n"
    "Tip: type / to see all commands."
)

START_TEXT = (
    "👋 **Welcome to WalletLedger**\n"
    "Tell me what you spend and earn, in plain words. I keep the books.\n\n"
    "Try: *Spent 450 on dinner via Bank*\n"
    "Or send a bank / PhonePe PDF to import it.\n\n"
    "For smart answers, add your AI key: /setkey"
)


def unknown_command_text(command: str) -> str:
    close = difflib.get_close_matches(command, sorted(KNOWN_COMMANDS), n=3, cutoff=0.5)
    hint = "Did you mean " + " or ".join(close) + "?" if close else "Try /help."
    return f"❓ I don't know `{command}`. {hint}"


# ---------------------------------------------------------------------------
# Pending statement imports (same yes/no idea as the agent's transfer confirmation)
# ---------------------------------------------------------------------------


@dataclass
class PendingImport:
    file_name: str
    rows: list[dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.utcnow)


_pending_imports: dict[str, PendingImport] = {}


def _get_pending_import(chat_id: str) -> PendingImport | None:
    pending = _pending_imports.get(chat_id)
    if pending and datetime.utcnow() - pending.created_at > IMPORT_TTL:
        _pending_imports.pop(chat_id, None)
        return None
    return pending


# ---------------------------------------------------------------------------
# Shared message handling
# ---------------------------------------------------------------------------


PERSONA_COMMANDS = {"/coach", "/budgeter", "/summary"}


def should_delete_message(text: str) -> bool:
    """True if the user's message contains an API key and should be removed from the chat."""
    clean = (text or "").strip()
    return clean.lower().startswith(("/setkey", "/key ", "/editkey")) or bool(
        extract_api_keys(clean)
    )


async def handle_incoming(chat_id: str, text: str) -> Reply:
    """Single entry point for a text message from any channel: returns text plus optional buttons."""
    chat_id = str(chat_id)
    clean = (text or "").strip()
    if not is_chat_allowed(chat_id):
        return Reply(PRIVATE_BOT_TEXT)

    pending = _get_pending_import(chat_id)
    if pending:
        decision = await agent.classify_decision(
            chat_id, clean, f"import {len(pending.rows)} rows from {pending.file_name}"
        )
        if decision != "new":
            return await answer_pending_import(
                chat_id, "confirmed" if decision == "confirm" else "cancelled"
            )
        _pending_imports.pop(chat_id, None)  # unrelated message: drop the pending import

    if clean.startswith("/"):
        # a command (even /balance) moves on: an old "transfer 7000?" must not be answered by a later "yes"
        await agent.cancel_pending(chat_id)
        parts = clean.split(None, 1)  # split on any whitespace (a newline or tab after the command)
        command, args = parts[0], (parts[1] if len(parts) > 1 else "")
        command = command.split("@", 1)[0].lower()  # "/balance@MyBot" in groups
        args = args.strip()
        if command == "/keys":
            from app.bot.callbacks import keys_screen

            return await keys_screen(chat_id)
        if command in KEY_COMMANDS:
            return Reply(await handle_key_command(chat_id, command, args))
        if command == "/status":
            return Reply(await user_status_text(chat_id), rows=ui.back_home())
        if command == "/admin":
            return Reply(await handle_admin_command(chat_id, args))
        reply = await _view_command(chat_id, command, args)
        if reply is not None:
            return reply
        if command not in PERSONA_COMMANDS:
            return Reply(unknown_command_text(command))

    if not extract_api_keys(clean) and not await agent.has_pending(chat_id):
        first = await onboarding.intercept(chat_id, clean)  # new user: ask where they pay from
        if first is not None:
            return first

    # Everything else in plain words (entries, questions, corrections, "bank balance is 25000",
    # "use cash by default", a typed yes/no) is understood by the AI agent, not by rules.
    return _reply_for(await process_user_interaction_ex(chat_id=chat_id, text=clean))


def _reply_for(outcome) -> Reply:
    """Text plus the right buttons: Yes/Cancel for a question, Undo for what was just recorded."""
    rows: list[list[ui.Btn]] = []
    if outcome.needs_confirmation:
        rows = ui.confirm_rows(outcome.pending_id)
        if outcome.transaction_ids:  # part of the message was already recorded: it can be undone
            rows.append(ui.undo_row(outcome.transaction_ids))
    elif outcome.transaction_ids:
        rows = [ui.undo_row(outcome.transaction_ids)]
    alt = getattr(outcome, "pay_alternative", None)
    if alt:
        if alt.get("cancel_only"):
            rows.append(ui.cancel_row())
        else:
            rows.append(ui.pay_alt_row(alt["account"], alt["balance"], alt["args"]))
    return Reply(outcome.text, rows=rows)


async def answer_pending_import(chat_id: str, decision: str) -> Reply:
    """Yes/No to a waiting statement import (typed or tapped). Never touches a pending transfer."""
    if _get_pending_import(chat_id) is None:
        return Reply(
            "⌛ That import has expired. Send the file again.",
            notice="Expired",
            rows=ui.back_home(),
        )
    done = await _finish_import(chat_id, decision)
    rows = [[ui.Btn("📊 Stats", "go:stats"), ui.Btn("🧾 History", "go:history")]]
    return Reply(done, rows=rows if "Imported" in done else [])


async def answer_pending_action(chat_id: str, yes: bool, pending_id: str | None = None) -> Reply:
    """Yes/Cancel to a waiting AI action or transfer (tapped). Only the question the button belongs to."""
    outcome = await agent.resume_pending(chat_id, yes, pending_id)
    if outcome is None:
        return Reply(
            "⌛ That question is no longer waiting, so nothing was changed.",
            notice="Expired",
            rows=ui.back_home(),
        )
    return _reply_for(outcome)


async def handle_text(chat_id: str, text: str) -> str:
    """Text-only wrapper (dev endpoint, tests): the reply text without buttons."""
    return (await handle_incoming(chat_id, text)).text


async def _view_command(chat_id: str, command: str, args: str) -> Reply | None:
    """Commands that open a screen (stats, history, ...). None if `command` is not one of them."""
    if command == "/start":
        _ensure_user(chat_id)
        if await onboarding.needs_setup(chat_id):
            return await onboarding.start(chat_id)
        return Reply(START_TEXT, rows=ui.home_rows())
    if command == "/menu":
        return views.home()
    if command == "/help":
        return Reply(HELP_TEXT, rows=ui.back_home())
    if command == "/balance":
        return await views.balance(chat_id)
    if command in ("/stats", "/report"):
        return await views.stats(chat_id, args)
    if command == "/chart":
        kind, _, rest = args.partition(" ")
        kind = kind.lower()
        if kind not in dict(ui.CHARTS) | {
            "monthly": "",
            "stack": "",
            "sizes": "",
            "hours": "",
            "accounts": "",
        }:
            names = ", ".join(k for k, _ in ui.CHARTS)
            return Reply(f"Which chart? /chart <name> [period]\nNames: {names}")
        period, words = views.parse_args(rest)
        return await views.chart(chat_id, kind, period, views._remember(words))
    if command == "/history":
        return await views.history(
            chat_id, int(args) - 1 if args.isdigit() and int(args) > 0 else 0
        )
    if command == "/undo":
        return await views.undo(chat_id, "last")
    if command == "/export":
        period, words = views.parse_args(args)
        return await views.export_csv(chat_id, period, words)
    if command == "/budget":
        return await views.budgets(chat_id, args)
    if command == "/dashboard":
        return await views.dashboard(chat_id)
    if command == "/plan":
        return await views.plan(chat_id, args)
    return None


def _ensure_user(chat_id: str) -> str:
    db = SessionLocal()
    try:
        return get_or_create_user(db, telegram_chat_id=chat_id).id
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Statement upload -> the parser (transaction_parser.py) -> preview -> the ledger on "yes"
# ---------------------------------------------------------------------------


def upload_type(file_name: str) -> str | None:
    """Map an uploaded file name to the parser's file type, or None if unsupported."""
    for ext, kind in SUPPORTED_UPLOADS.items():
        if (file_name or "").lower().endswith(ext):
            return kind
    return None


def _parse_upload(chat_id: str, file_bytes: bytes, kind: str) -> list[dict[str, Any]]:
    """Runs in a worker thread: its own database session, its own event loop for the parser."""
    with SessionLocal() as db:
        user_id = get_or_create_user(db, telegram_chat_id=chat_id).id
        return asyncio.run(parse_statement_file(file_bytes, kind, user_id=user_id, db=db))


async def handle_document(chat_id: str, file_name: str, file_bytes: bytes) -> str:
    """Parse an uploaded statement and ask the user to confirm the import."""
    chat_id = str(chat_id)
    if not is_chat_allowed(chat_id):
        return PRIVATE_BOT_TEXT
    kind = upload_type(file_name)
    if kind is None:
        return "📄 Please upload a PDF statement (or a .txt file with bank SMS lines)."

    try:
        # PDF reading is slow CPU work: do it in a worker thread so the bot keeps answering everyone else
        rows = await asyncio.to_thread(_parse_upload, chat_id, file_bytes, kind)
    except ValueError as e:
        return f"⚠️ Could not read `{file_name}`: {e}"
    except Exception as e:
        logger.error("Statement parsing failed for chat %s: %s", chat_id, e, exc_info=True)
        return f"⚠️ Could not read `{file_name}`. Please try another file."

    rows = [r for r in rows if float(r.get("amount") or 0) > 0]
    if not rows:
        return f"📄 I couldn't find any transactions in `{file_name}`."

    _pending_imports[chat_id] = PendingImport(file_name=file_name, rows=rows)
    return _import_preview(file_name, rows)


async def handle_document_rich(chat_id: str, file_name: str, file_bytes: bytes) -> Reply:
    """Like handle_document, plus Import / Skip buttons when a preview is waiting for an answer."""
    text = await handle_document(chat_id, file_name, file_bytes)
    rows = ui.import_rows() if _get_pending_import(str(chat_id)) else []
    return Reply(text, rows=rows)


def _import_preview(file_name: str, rows: list[dict[str, Any]]) -> str:
    summary = summarize_statement(rows)
    kind = f" ({summary['format_label']})" if summary["format_label"] else ""
    period = f" ({summary['period']})" if summary["period"] else ""
    top = " · ".join(f"{name} ₹{total:,.0f}" for name, total in summary["categories"][:3])
    sample = "\n".join(
        f"  • {r.get('date') or '—'}  {'+' if r.get('type') == 'income' else '-'}₹{r['amount']:,.2f}"
        f"  {r.get('merchant') or 'Unknown'} ({r.get('category') or 'Uncategorized'})"
        for r in rows[:PREVIEW_ROWS]
    )
    more = f"\n  …and {len(rows) - PREVIEW_ROWS} more" if len(rows) > PREVIEW_ROWS else ""
    net = summary["total_credit"] - summary["total_debit"]
    unclear = sum(1 for r in rows if "guessed" in str(r.get("note") or ""))
    warning = (
        f"\n⚠️ {unclear} row{'s' if unclear != 1 else ''} had no clear debit/credit marker and "
        "was read as an expense. Check them in /history afterwards.\n"
        if unclear
        else ""
    )
    sign = "−" if net < 0 else "+"
    return (
        f"📄 **Statement:** `{file_name}`{kind}\n"
        f"Found **{summary['count']} transactions**{period}: ₹{summary['total_debit']:,.2f} spent, "
        f"₹{summary['total_credit']:,.2f} received.\n"
        + (f"Top: {top}\n" if top else "")
        + f"\n{sample}{more}\n{warning}\n"
        f"They go into your **{IMPORT_ACCOUNT}** account.\n"
        f"📚 **History**: reports and charts count them; your balance stays as it is. (Reply 'yes')\n"
        f"💰 **History + balance**: also changes {IMPORT_ACCOUNT} by {sign}₹{abs(net):,.2f}. "
        "Pick this if the statement is the money you are tracking from now on.\n"
        "Reply 'no' to cancel."
    )


def _row_day(row: dict[str, Any]) -> str:
    """ISO date of a parsed statement row (today when the statement gave none)."""
    raw = str(row.get("date") or "")[:10]
    return raw if len(raw) == 10 else clock.today_local().isoformat()


def _merchant_match(a: str, b: str) -> bool:
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b or "unknown" in (a, b):
        return True  # one side has no merchant: amount and date have to carry it
    return a == b or a in b or b in a


def _drop_already_imported(chat_id: str, rows: list[dict[str, Any]]) -> tuple[list, int]:
    """Skip rows that are already in the ledger, so the same payment is never counted twice.

    A row is a duplicate when (1) it has the same bank reference (UTR) as an existing entry, or
    (2) an existing entry has the same type, amount and day (one day either way) and a matching
    merchant: that covers the same statement uploaded twice, the same payment in a bank statement AND
    a PhonePe statement, and an entry you already typed in chat. Each existing entry can match only
    one new row, so two genuine identical payments in one statement are both kept.
    Compared in Python because notes are stored encrypted.
    """
    if not rows:
        return rows, 0
    days = sorted(_row_day(r) for r in rows)
    start = datetime.fromisoformat(days[0]) - timedelta(days=1)
    end = datetime.fromisoformat(days[-1]) + timedelta(days=2)
    db = SessionLocal()
    try:
        user_id = get_or_create_user(db, telegram_chat_id=chat_id).id
        found = db.execute(
            select(
                Transaction.transaction_type,
                Transaction.amount,
                Transaction.created_at,
                Merchant.name,
                Transaction.reference,
            )
            .select_from(Transaction)
            .outerjoin(Merchant, Merchant.id == Transaction.merchant_id)
            .where(Transaction.user_id == user_id)
            .where(Transaction.transaction_type.in_(("expense", "income")))
            .where(Transaction.created_at >= clock.to_utc(start))
            .where(Transaction.created_at < clock.to_utc(end))
        ).all()
    finally:
        db.close()
    # a bank reference identifies a payment whatever its date: look the references up across all time
    wanted = {str(r.get("reference")).strip() for r in rows if r.get("reference")}
    refs: set[str] = set()
    if wanted:
        with SessionLocal() as db:
            refs = set(
                db.scalars(
                    select(Transaction.reference).where(
                        Transaction.user_id == user_id, Transaction.reference.in_(wanted)
                    )
                )
            )
    pool = [(t, round(a, 2), clock.to_local(c).date(), m or "") for t, a, c, m, _ in found]
    fresh, duplicates = [], 0
    for row in rows:
        ref = str(row.get("reference") or "").strip()
        if ref and ref in refs:
            duplicates += 1
            continue
        tx_type = "income" if row.get("type") == "income" else "expense"
        amount = round(float(row["amount"]), 2)
        day = datetime.fromisoformat(_row_day(row)).date()
        hit = next(
            (
                i
                for i, (t, a, d, m) in enumerate(pool)
                if t == tx_type
                and a == amount
                and abs((d - day).days) <= 1
                and _merchant_match(row.get("merchant") or "", m)
            ),
            None,
        )
        if hit is not None:
            pool.pop(hit)  # one existing entry matches one new row
            duplicates += 1
            continue
        fresh.append(
            row
        )  # (two rows of ONE statement may share a reference; only the ledger is checked)
    return fresh, duplicates


async def _finish_import(chat_id: str, decision: str) -> str:
    """decision: "history" (also what a typed "yes" means), "apply" (also change the balance), or cancel."""
    pending = _pending_imports.pop(chat_id, None)
    if pending is None:
        return "⌛ That import expired. Please upload the statement again."
    if decision not in ("history", "apply", "confirmed"):
        return "❌ **Import cancelled.** Nothing was added to your ledger."
    apply_balance = decision == "apply"

    rows, duplicates = _drop_already_imported(chat_id, pending.rows)
    imported, failed = 0, 0
    for row in rows:
        common = {
            "user_id": chat_id,
            "amount": row["amount"],
            "description": row.get("description"),
            "source": "pdf",
            "date": row.get("date"),
            "affects_balance": apply_balance,
            "reference": row.get("reference"),
        }
        try:
            if row.get("type") == "income":
                await tool_client.call_tool(
                    "log_income",
                    source_account=IMPORT_ACCOUNT,
                    category=row.get("category") or "Income",
                    **common,
                )
            else:
                await tool_client.call_tool(
                    "log_expense",
                    category=row.get("category") or "Uncategorized",
                    merchant=row.get("merchant") if row.get("merchant") != "Unknown" else None,
                    account_name=IMPORT_ACCOUNT,
                    **common,
                )
            imported += 1
        except Exception as e:
            failed += 1
            logger.warning("Import row failed for chat %s: %s", chat_id, e)

    noun = "transaction" if imported == 1 else "transactions"
    text = f"✅ **Imported {imported} {noun}** from `{pending.file_name}`."
    text += (
        f"\n💰 {IMPORT_ACCOUNT} balance updated."
        if apply_balance
        else f"\n📚 Added as history: your {IMPORT_ACCOUNT} balance did not change."
    )
    if duplicates:
        text += f"\n↩️ Skipped {duplicates} already in your ledger."
    if failed:
        text += f"\n⚠️ {failed} rows could not be imported."
    return text + "\nSee them in /stats or /history."


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


class SimulateMessageRequest(BaseModel):
    chat_id: str
    text: str


@bot_router.post("/webhook")
async def telegram_webhook(request: Request):
    return await _webhook(request, None)


@bot_router.post("/webhook/{bot_id}")
async def telegram_webhook_for_bot(bot_id: str, request: Request):
    """Same as /webhook, for one of several bots (each bot's own address is WEBHOOK_URL/<bot id>)."""
    return await _webhook(request, bot_id)


WEBHOOK_MAX_BYTES = 1_000_000  # real Telegram updates are a few KB


async def _json_body(request: Request) -> dict:
    try:
        data = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="invalid update")
    return data


async def _webhook(request: Request, bot_id: str | None):
    """Telegram webhook.

    Webhook mode with a bot token: verified with the secret header, then handled by aiogram,
    which sends the reply to Telegram. Otherwise (local dev / curl): the reply is returned
    as JSON so the endpoint stays testable without a token.
    """
    from app.bot import telegram

    if not telegram.webhook_active() and not dev_endpoints_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    if telegram.webhook_active():
        token = telegram.token_for(bot_id)
        if token is None and bot_id is not None:
            raise HTTPException(status_code=404, detail="Not found")
        header = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        # compare as bytes (a non-ASCII header must be a 403, not a crash) and BEFORE reading the body
        if not hmac.compare_digest(
            header.encode("utf-8", "replace"), telegram.webhook_secret(token).encode()
        ):
            raise HTTPException(status_code=403, detail="invalid webhook secret")
        if int(request.headers.get("content-length") or 0) > WEBHOOK_MAX_BYTES:
            raise HTTPException(status_code=413, detail="too large")
        update = await _json_body(request)
        await telegram.feed_webhook_update(update, bot_id)
        return {"ok": True}
    update = await _json_body(request)

    message = update.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    if not chat_id:
        return {"status": "ignored", "reason": "no message payload"}
    if message.get("document"):
        return {
            "status": "ok",
            "reply": "📄 File uploads need the real Telegram bot (set TELEGRAM_BOT_TOKEN).",
        }
    return {"status": "ok", "reply": await handle_text(chat_id, message.get("text") or "")}


@bot_router.post("/simulate_chat")
async def simulate_chat(req: SimulateMessageRequest):
    """Developer endpoint: same logic as Telegram, no bot token needed."""
    if not dev_endpoints_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    return {"status": "ok", "reply": await handle_text(req.chat_id, req.text)}
