"""Real Telegram connection via aiogram (polling for local dev, webhook for production).

Lifecycle (called from main.py's FastAPI lifespan):
    no bot token                 -> bot off; /bot/simulate_chat still works
    USE_WEBHOOK=false (default)  -> long polling in a background task
    USE_WEBHOOK=true             -> register WEBHOOK_URL with Telegram; /bot/webhook feeds aiogram

Several bots (TELEGRAM_BOT_TOKEN plus TELEGRAM_BOT_TOKENS) run side by side on one dispatcher and one
ledger. With webhooks each bot gets its own address: WEBHOOK_URL/<bot id> (one bot keeps WEBHOOK_URL).

All handlers delegate to app.bot.main (handle_text / handle_document) so Telegram, the dev
webhook and /bot/simulate_chat share one code path.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import logging
import re
import time
from typing import Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
    ReactionTypeEmoji,
    WebAppInfo,
)

from app.bot.callbacks import handle_callback
from app.bot.main import (
    COMMAND_MENU,
    handle_document_rich,
    handle_incoming,
    should_delete_message,
    upload_type,
)
from app.bot.profile import apply_profile
from app.bot.ui import Btn, Reply
from app.config import bot_tokens, settings

logger = logging.getLogger(__name__)


class _ConflictFilter(logging.Filter):
    """aiogram prints two lines every few seconds while another copy of the bot is polling the same token.
    Say it once in plain words (and again every 5 minutes), and hide the spam in between."""

    HINT = (
        "Another copy of this bot is polling the same token (another terminal, computer, server, or a "
        "teammate). Telegram gives each message to only ONE of them, so some of your messages will vanish. "
        "Stop the other copy, or give each copy its own bot (create one with @BotFather, /newbot)."
    )

    def __init__(self) -> None:
        super().__init__()
        self._last = 0.0

    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        if "TelegramConflictError" in text:
            now = time.monotonic()
            if now - self._last > 300:
                self._last = now
                logger.warning(self.HINT)
            return False
        return not ("Sleep for" in text and "try again" in text and self._last)


logging.getLogger("aiogram.dispatcher").addFilter(_ConflictFilter())

TELEGRAM_LIMIT = 4096
CHUNK_SIZE = 4000  # leave room so Markdown fallbacks never exceed the limit


_bots: dict[str, Bot] = {}  # bot id (the number before ":" in the token) -> Bot
_dispatcher: Dispatcher | None = None
_polling_task: asyncio.Task | None = None
_mode: str = "off"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_message(text: str, limit: int = CHUNK_SIZE) -> list[str]:
    """Split a long reply into Telegram-sized chunks, preferring paragraph/line breaks."""
    text = text or ""
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut <= 0:
            cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n ")
    if text.strip() or not chunks:
        chunks.append(text)
    return chunks


def to_telegram_markdown(text: str) -> str:
    """Convert **bold** (used across the codebase and by LLMs) to Telegram's legacy *bold*."""
    return re.sub(r"\*\*(.+?)\*\*", r"*\1*", text or "", flags=re.S)


def to_plain(text: str) -> str:
    """Telegram captions and fallbacks: drop Markdown markers."""
    return (text or "").replace("**", "").replace("`", "")


def keyboard(rows: list[list[Btn]]) -> InlineKeyboardMarkup | None:
    """Turn button rows (ui.Btn) into a Telegram inline keyboard."""
    if not rows:
        return None
    out = []
    for row in rows:
        buttons = []
        for b in row:
            if b.web_app:
                buttons.append(InlineKeyboardButton(text=b.text, web_app=WebAppInfo(url=b.web_app)))
            elif b.url:
                buttons.append(InlineKeyboardButton(text=b.text, url=b.url))
            else:
                buttons.append(
                    InlineKeyboardButton(text=b.text, callback_data=(b.data or "noop")[:64])
                )
        out.append(buttons)
    return InlineKeyboardMarkup(inline_keyboard=out)


async def send_reply(
    message: Message, text: str, reply_markup: InlineKeyboardMarkup | None = None
) -> None:
    """Send a (possibly long) Markdown reply; resend as plain text if Telegram rejects it.

    Buttons (if any) are attached to the last chunk.
    """
    chunks = split_message(text)
    for i, chunk in enumerate(chunks):
        markup = reply_markup if i == len(chunks) - 1 else None
        try:
            await message.answer(
                to_telegram_markdown(chunk), parse_mode=ParseMode.MARKDOWN, reply_markup=markup
            )
        except TelegramBadRequest as e:
            logger.info("Markdown rejected by Telegram (%s); sending plain text.", e)
            await message.answer(to_plain(chunk), parse_mode=None, reply_markup=markup)


async def deliver(message: Message, reply: Reply) -> None:
    """Send a Reply as new message(s): photo / file / text, with its buttons."""
    markup = keyboard(reply.rows)
    if reply.photo:
        await message.answer_photo(
            BufferedInputFile(reply.photo, "walletledger.png"),
            caption=to_plain(reply.text)[:1000],
            reply_markup=markup,
        )
    elif reply.document:
        data, name = reply.document
        await message.answer_document(
            BufferedInputFile(data, name), caption=to_plain(reply.text)[:1000], reply_markup=markup
        )
    elif reply.text:
        await send_reply(message, reply.text, markup)


async def update_message(message: Message, reply: Reply) -> None:
    """Show a Reply by editing the message a button was on (falls back to a new message)."""
    markup = keyboard(reply.rows)
    try:
        if message.photo and reply.photo:
            await message.edit_media(
                InputMediaPhoto(
                    media=BufferedInputFile(reply.photo, "walletledger.png"),
                    caption=to_plain(reply.text)[:1000],
                ),
                reply_markup=markup,
            )
            return
        if not message.photo and not message.document and not reply.photo and not reply.document:
            try:
                await message.edit_text(
                    to_telegram_markdown(reply.text),
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=markup,
                )
            except TelegramBadRequest as e:
                if "not modified" in str(e).lower():
                    return
                await message.edit_text(to_plain(reply.text), parse_mode=None, reply_markup=markup)
            return
    except TelegramBadRequest as e:
        if "not modified" in str(e).lower():
            return
        logger.info("Could not edit message (%s); sending a new one.", e)
    # Different kind of message (text <-> photo): send a new one and clear the old buttons.
    with contextlib.suppress(Exception):
        await message.edit_reply_markup(reply_markup=None)
    await deliver(message, reply)


def bot_id(token: str) -> str:
    """Telegram bot tokens look like 123456:ABC...; the number is the bot's id (not secret)."""
    return token.split(":", 1)[0]


def webhook_secret(token: str | None = None) -> str:
    """Secret Telegram must echo in X-Telegram-Bot-Api-Secret-Token on every webhook call.

    An explicit TELEGRAM_WEBHOOK_SECRET is used for every bot; otherwise each bot gets its own,
    derived from its token.
    """
    if settings.telegram_webhook_secret:
        return settings.telegram_webhook_secret
    token = token or (bot_tokens() or [""])[0]
    return hashlib.sha256(f"walletledger:{token}".encode()).hexdigest()[:48]


def webhook_url_for(token: str) -> str:
    """One bot: WEBHOOK_URL as given. Several: WEBHOOK_URL/<bot id>, so each bot has its own address."""
    base = settings.webhook_url.rstrip("/")
    return base if len(bot_tokens()) <= 1 else f"{base}/{bot_id(token)}"


def webhook_active() -> bool:
    return _mode == "webhook" and bool(_bots) and _dispatcher is not None


def token_for(bot_id_: str | None) -> str | None:
    """The configured token of this bot id (None = the first bot), or None if there is no such bot."""
    tokens = bot_tokens()
    if bot_id_ is None:
        return tokens[0] if tokens else None
    return next((t for t in tokens if bot_id(t) == bot_id_), None)


async def feed_webhook_update(update: dict[str, Any], bot_id_: str | None = None) -> None:
    if not webhook_active():
        raise RuntimeError("Telegram webhook mode is not running")
    token = token_for(bot_id_)
    bot = _bots.get(bot_id(token)) if token else None
    if bot is None:
        raise RuntimeError(f"No such bot: {bot_id_}")
    await _dispatcher.feed_webhook_update(bot, update)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def on_document(message: Message, bot: Bot) -> None:
    doc = message.document
    file_name = doc.file_name or "statement.pdf"
    if upload_type(file_name) is None:
        await send_reply(message, "📄 Please upload a PDF statement (or a .txt file of bank SMS).")
        return
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if doc.file_size and doc.file_size > max_bytes:
        await send_reply(message, f"📄 That file is too large (max {settings.max_upload_mb} MB).")
        return

    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    buffer = io.BytesIO()
    await bot.download(doc, destination=buffer)
    reply = await handle_document_rich(str(message.chat.id), file_name, buffer.getvalue())
    await deliver(message, reply)


_PICTURE_COMMANDS = ("/stats", "/report", "/chart")


async def on_text(message: Message, bot: Bot) -> None:
    text = message.text or ""
    if should_delete_message(text):
        # Remove the user's message so the API key does not stay in the chat history.
        with contextlib.suppress(Exception):
            await message.delete()
    wants_picture = text.lower().startswith(_PICTURE_COMMANDS)
    await bot.send_chat_action(
        message.chat.id, ChatAction.UPLOAD_PHOTO if wants_picture else ChatAction.TYPING
    )
    reply = await handle_incoming(str(message.chat.id), text)
    await deliver(message, reply)
    if any(b.data and b.data.startswith("undo:") for row in reply.rows for b in row):
        # A small tick on the user's message shows the entry was recorded.
        with contextlib.suppress(Exception):
            await bot.set_message_reaction(
                message.chat.id, message.message_id, [ReactionTypeEmoji(emoji="👍")]
            )


async def on_callback(callback: CallbackQuery, bot: Bot) -> None:
    """A tap on an inline button."""
    message = callback.message
    if message is None or not isinstance(message, Message):
        await callback.answer()
        return
    chat_id = str(message.chat.id)
    if message.chat.type == "private" and callback.from_user.id != message.chat.id:
        await callback.answer()  # a stranger cannot press buttons in someone else's chat
        return
    data = callback.data or ""
    try:
        if data.startswith(("st:", "ch:", "go:stats", "ex:")):
            with contextlib.suppress(Exception):
                await bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_PHOTO)
        reply = await handle_callback(chat_id, data)
    except Exception:
        logger.exception("Callback failed: %s", data)
        await callback.answer("Something went wrong", show_alert=False)
        return
    with contextlib.suppress(Exception):
        await callback.answer(reply.notice or None)
    if not (reply.text or reply.photo or reply.document):
        return
    if reply.edit:
        await update_message(message, reply)
    else:
        await deliver(message, reply)


async def on_other(message: Message) -> None:
    await send_reply(message, "I can read text messages and PDF statements. Try /help.")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def build_router() -> Router:
    """A fresh router per start(): aiogram routers can only be attached to one Dispatcher."""
    router = Router(name="walletledger")
    router.message.register(on_document, F.document)
    router.message.register(on_text, F.text)
    router.callback_query.register(on_callback, F.data)
    router.message.register(on_other)
    return router


async def _publish_profile(bot: Bot) -> None:
    """Refresh the command menu, descriptions and menu button on every start. Never fatal."""
    try:
        results = await apply_profile(bot)
    except Exception as exc:  # cosmetic: never stop the bot over its profile
        logger.warning("Could not publish the Telegram bot profile: %s", exc)
        return
    failed = {k: v for k, v in results.items() if v != "ok"}
    if failed:
        logger.warning("Bot profile partly applied: %s", failed)
    else:
        logger.info("Bot profile published (%d commands)", len(COMMAND_MENU))


async def start() -> str:
    """Start the bot(s) according to settings. Returns the mode: off, polling or webhook."""
    global _dispatcher, _polling_task, _mode
    tokens = bot_tokens()
    if not tokens:
        logger.info("No Telegram bot token set; Telegram bot disabled (/bot/simulate_chat works).")
        _mode = "off"
        return _mode

    _bots.clear()
    for token in tokens:
        _bots[bot_id(token)] = Bot(token=token)
    _dispatcher = Dispatcher()
    _dispatcher.include_router(build_router())
    names = {}
    for key, bot in list(_bots.items()):
        try:
            names[key] = (await bot.get_me()).username
        except Exception as exc:
            if len(tokens) == 1:
                raise  # the only bot is broken: say so loudly, as before
            logger.error(
                "Bot %s could not start (%s). Check its token. The other bots keep running.",
                key,
                type(exc).__name__,
            )
            with contextlib.suppress(Exception):
                await bot.session.close()
            _bots.pop(key)
            continue
        await _publish_profile(bot)
    if not _bots:
        raise RuntimeError("None of the configured Telegram bot tokens could be used.")
    tokens = [t for t in tokens if bot_id(t) in _bots]

    if settings.use_webhook:
        if not settings.webhook_url:
            raise RuntimeError(
                "USE_WEBHOOK=true requires WEBHOOK_URL (e.g. https://host/bot/webhook)"
            )
        for token in tokens:
            url = webhook_url_for(token)
            await _bots[bot_id(token)].set_webhook(
                url,
                secret_token=webhook_secret(token),
                allowed_updates=_dispatcher.resolve_used_update_types(),
                drop_pending_updates=False,
            )
            logger.info("Telegram bot @%s: webhook registered at %s", names[bot_id(token)], url)
        _mode = "webhook"
    else:
        # A webhook left over from production would block polling, so clear it first.
        for bot in _bots.values():
            await bot.delete_webhook(drop_pending_updates=False)
        _polling_task = asyncio.create_task(
            _dispatcher.start_polling(
                *_bots.values(), handle_signals=False, close_bot_session=False
            ),
            name="telegram-polling",
        )
        _mode = "polling"
        logger.info("Telegram polling started for %s", ", ".join(f"@{n}" for n in names.values()))
    return _mode


def running_bots() -> int:
    return len(_bots)


async def stop() -> None:
    """Stop polling and close the HTTP sessions. The webhooks stay registered on purpose."""
    global _dispatcher, _polling_task, _mode
    if _dispatcher is not None and _polling_task is not None:
        with contextlib.suppress(Exception):
            await _dispatcher.stop_polling()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(_polling_task, timeout=10)
    for bot in _bots.values():
        with contextlib.suppress(Exception):
            await bot.session.close()
    _bots.clear()
    _dispatcher, _polling_task, _mode = None, None, "off"
