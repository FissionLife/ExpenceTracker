"""What happens when someone taps an inline button (see ui.py for the data format)."""

from __future__ import annotations

import logging

from app.bot import keys as keys_mod
from app.bot import ui, views
from app.bot.ui import Reply
from app.config import is_chat_allowed

logger = logging.getLogger(__name__)

_KEY_COMMANDS = {"test": "/testkey", "pause": "/pausekey", "resume": "/resumekey", "del": "/delkey"}


async def keys_screen(chat_id: str) -> Reply:
    """The key list with a row of tap-to-manage buttons per key."""
    from app.services_ai import ai_gateway as gw

    text = keys_mod.keys_text(chat_id)
    return Reply(text, rows=ui.keys_rows([k["key_id"] for k in gw.list_api_keys(chat_id)]))


async def handle_callback(chat_id: str, data: str) -> Reply:
    """Turn a button tap into a Reply. Never raises: problems become a short message."""
    from app.bot import main as bot_main  # imported here: main imports this module's siblings

    chat_id = str(chat_id)
    if not is_chat_allowed(chat_id):
        return Reply(bot_main.PRIVATE_BOT_TEXT, notice="Private bot")
    action, _, rest = (data or "").partition(":")
    try:
        if action == "go":
            return await _go(chat_id, rest, bot_main)
        if action == "st":
            period, _, ctx = rest.partition(":")
            return await views.stats_for(chat_id, period or "30d", views._words_for(ctx), edit=True)
        if action == "ch":
            kind, _, tail = rest.partition(":")
            period, _, ctx = tail.partition(":")
            return await views.chart(chat_id, kind, period or "30d", ctx)
        if action == "hi":
            return await views.history(chat_id, int(rest or 0), edit=True)
        if action == "undo":
            return await views.undo(chat_id, rest or "last")
        if action == "ask":
            decision, _, pending_id = rest.partition(":")
            reply = await bot_main.answer_pending_action(
                chat_id, decision == "yes", pending_id or None
            )
            reply.edit = True
            reply.notice = reply.notice or ("Done" if decision == "yes" else "Cancelled")
            return reply
        if action == "ob":
            from app.bot import onboarding

            kind, _, which = rest.partition(":")
            if kind == "skip":
                return await onboarding.skipped(chat_id)
            account = onboarding.CHOICES.get(which)
            if kind != "acct" or account is None:
                return Reply(notice="That button is out of date")
            return await onboarding.answered(chat_id, account)
        if action == "imp":
            decision = {"hist": "history", "yes": "history", "apply": "apply"}.get(
                rest, "cancelled"
            )
            reply = await bot_main.answer_pending_import(chat_id, decision)
            reply.edit = True
            reply.notice = reply.notice or ("Skipped" if decision == "cancelled" else "Done")
            return reply
        if action == "key":
            sub, _, n = rest.partition(":")
            command = _KEY_COMMANDS.get(sub)
            if command is None:
                return Reply(notice="Unknown action")
            text = await keys_mod.handle_key_command(chat_id, command, "" if n == "all" else n)
            screen = await keys_screen(chat_id)
            screen.text = f"{text}\n\n{screen.text}"[:3900]
            screen.edit = True
            return screen
        if action == "ex":
            return await views.export_csv(chat_id, rest or "30d")
        if action == "pay":
            return await _pay_from_other(chat_id, rest)
    except Exception:
        logger.exception("Button action failed: %s", data)
        return Reply("⚠️ That didn't work. Please try again.", notice="Something went wrong")
    return Reply(notice="That button is out of date")


async def _go(chat_id: str, screen: str, bot_main) -> Reply:
    if screen == "home":
        reply = views.home()
        reply.edit = True
        return reply
    if screen == "stats":
        return await views.stats(chat_id, "")
    if screen == "balance":
        return await views.balance(chat_id)
    if screen == "history":
        return await views.history(chat_id, 0)
    if screen == "budgets":
        return await views.budgets(chat_id)
    if screen == "dashboard":
        return await views.dashboard(chat_id)
    if screen == "keys":
        return await keys_screen(chat_id)
    if screen == "status":
        from app.health import user_status_text

        return Reply(await user_status_text(chat_id), rows=ui.back_home())
    if screen == "help":
        return Reply(bot_main.HELP_TEXT, rows=ui.back_home())
    return Reply(notice="That button is out of date")


async def _pay_from_other(chat_id: str, key: str) -> Reply:
    """Retry a blocked spend on the account the user picked ('Use Bank instead')."""
    from app.services_ai import tool_client

    args = ui.pay_retry(key)
    if not args:
        return Reply(notice="That button is out of date")
    args = dict(args)
    args["user_id"] = chat_id
    try:
        res = await tool_client.call_tool("log_expense", **args)
    except Exception as exc:
        return Reply(f"⚠️ {views.replies.friendly_error(str(exc))}", notice="Couldn't log that")
    item = {"tool_name": "log_expense", "tool_args": args, "result": res, "error": None}
    text = views.replies.compact({"tool_results": [item]}) or "✅ Done."
    rows = [ui.undo_row([str(res["transaction_id"])])] if res.get("transaction_id") else []
    return Reply(text, rows=rows, edit=True, notice="Logged")
