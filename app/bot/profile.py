"""The bot's Telegram-side profile: command menu, descriptions and menu button.

One place defines it, and it is applied in two ways:
  * automatically every time the server starts (app/bot/telegram.py), and
  * on demand:  uv run python scripts/setup_bot.py   (see --help)

Every step is independent: one failing (for example a rate limit) never blocks the others.
"""

from __future__ import annotations

import logging

from aiogram import Bot
from aiogram.types import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeDefault,
    MenuButtonCommands,
)

from app.bot.main import COMMAND_MENU

logger = logging.getLogger(__name__)

# Shown at the top of an empty chat, before the user presses Start (Telegram limit: 512 chars).
DESCRIPTION = (
    "WalletLedger tracks your money in plain language.\n\n"
    '• Say "Spent 450 on dinner via Bank" to log it\n'
    "• Send a PhonePe / GPay / bank PDF to import transactions\n"
    "• Get balances, reports and savings advice\n"
    "• Bring your own AI key, or use the shared one\n\n"
    "Press START to begin."
)

# Shown on the bot's profile page and in shares (Telegram limit: 120 chars).
SHORT_DESCRIPTION = (
    "AI expense tracker: log spending in chat, import statements, get savings advice."
)

_SCOPES = (
    BotCommandScopeDefault(),
    BotCommandScopeAllPrivateChats(),
    BotCommandScopeAllGroupChats(),
)


def commands() -> list[BotCommand]:
    return [BotCommand(command=name, description=desc) for name, desc in COMMAND_MENU]


def validate() -> list[str]:
    """Local check of Telegram's limits, so mistakes show up before any network call."""
    problems = []
    if len(DESCRIPTION) > 512:
        problems.append(f"description is {len(DESCRIPTION)} chars (max 512)")
    if len(SHORT_DESCRIPTION) > 120:
        problems.append(f"short description is {len(SHORT_DESCRIPTION)} chars (max 120)")
    if len(COMMAND_MENU) > 100:
        problems.append("more than 100 commands")
    for name, desc in COMMAND_MENU:
        if (
            not (1 <= len(name) <= 32)
            or name != name.lower()
            or not name.replace("_", "").isalnum()
        ):
            problems.append(f"bad command name: {name!r}")
        if not (3 <= len(desc) <= 256):
            problems.append(f"/{name} description must be 3-256 chars")
    return problems


async def apply_profile(
    bot: Bot, *, name: str | None = None, reset: bool = False
) -> dict[str, str]:
    """Push the profile to Telegram. Returns {step: "ok" | "failed: reason"}."""
    problems = validate()
    if problems:
        raise ValueError("; ".join(problems))

    results: dict[str, str] = {}

    async def step(label: str, coro_factory) -> None:
        try:
            await coro_factory()
            results[label] = "ok"
        except Exception as exc:
            results[label] = f"failed: {type(exc).__name__}: {str(exc)[:120]}"
            logger.warning("Bot profile step %r failed: %s", label, exc)

    if reset:
        # Remove commands left behind in narrower scopes, which would override the default menu.
        for scope in _SCOPES:
            await step(
                f"clear commands ({scope.type})", lambda s=scope: bot.delete_my_commands(scope=s)
            )

    await step(
        "command menu", lambda: bot.set_my_commands(commands(), scope=BotCommandScopeDefault())
    )
    await step(
        "command menu (private chats)",
        lambda: bot.set_my_commands(commands(), scope=BotCommandScopeAllPrivateChats()),
    )
    await step("description", lambda: bot.set_my_description(description=DESCRIPTION))
    await step(
        "short description",
        lambda: bot.set_my_short_description(short_description=SHORT_DESCRIPTION),
    )
    await step("menu button", lambda: bot.set_chat_menu_button(menu_button=MenuButtonCommands()))
    if name:
        await step("display name", lambda: bot.set_my_name(name=name))
    return results


async def read_profile(bot: Bot) -> dict[str, object]:
    """What Telegram currently has for this bot (to compare with what we would publish)."""
    me = await bot.get_me()
    current = await bot.get_my_commands(scope=BotCommandScopeDefault())
    desc = await bot.get_my_description()
    short = await bot.get_my_short_description()
    name = await bot.get_my_name()
    menu = await bot.get_chat_menu_button()
    published = [(c.command, c.description) for c in current]
    return {
        "bot": f"@{me.username} (id {me.id})",
        "name": name.name,
        "description": desc.description,
        "short_description": short.short_description,
        "menu_button": menu.type,
        "commands": published,
        "in_sync": published == list(COMMAND_MENU)
        and desc.description == DESCRIPTION
        and short.short_description == SHORT_DESCRIPTION,
    }
