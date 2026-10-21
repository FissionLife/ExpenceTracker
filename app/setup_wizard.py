"""The setup wizard and the health check. Used by `scripts/setup.py` and by the first run of the server.

    uv run python scripts/setup.py              # guided setup (safe to re-run any time)
    uv run python scripts/setup.py --check      # health check only; changes nothing
    uv run python scripts/setup.py --defaults   # no questions: strong secret, SQLite, public bot

Rules this follows:
  * Safe to re-run: every question shows the current value, Enter keeps it.
  * The bot is PUBLIC by default (anyone may use it; each chat only ever sees its own data).
    "Private" (a list of allowed chat ids) is an explicit choice.
  * Every write to `.env` happens before the first import of `app.config`, so the settings that
    the server then reads are the final ones.
  * It can run inside `uvicorn` (see app/firstrun.py): questions are read from the console
    even when the reloader has closed stdin.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import secrets
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from app import envfile

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
EXAMPLE = ROOT / ".env.example"


class NoConsole(RuntimeError):
    """There is nobody to ask (no terminal)."""


class Console:
    """Questions and answers on the terminal. Replaceable in tests."""

    def __init__(self) -> None:
        self._stream = None  # opened lazily, only if stdin cannot be used

    def say(self, msg: str = "") -> None:
        print(msg, flush=True)

    def _readline(self, prompt: str) -> str:
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                return input(prompt)
        except ValueError, OSError:
            pass
        if self._stream is None:
            try:
                self._stream = open("CONIN$" if os.name == "nt" else "/dev/tty")  # noqa: SIM115
            except OSError as exc:
                raise NoConsole("no terminal to ask questions on") from exc
        print(prompt, end="", flush=True)
        line = self._stream.readline()
        if line == "":
            raise EOFError
        return line.rstrip("\r\n")

    def ask(self, prompt: str, default: str = "", secret: bool = False) -> str:
        suffix = (
            f" [{default}]" if default and not secret else (" [keep current]" if default else "")
        )
        try:
            raw = (
                getpass.getpass(f"{prompt}{suffix}: ")
                if secret
                else self._readline(f"{prompt}{suffix}: ")
            ).strip()
        except (EOFError, OSError) as exc:
            raise NoConsole("input ended") from exc
        return raw or default

    def yes(self, prompt: str, default: bool = True) -> bool:
        raw = self.ask(f"{prompt} [{'Y/n' if default else 'y/N'}]").lower()
        return default if not raw else raw.startswith("y")


# --------------------------------------------------------------------------------------- Telegram


async def check_token(token: str):
    from aiogram import Bot

    bot = Bot(token=token)
    try:
        return await bot.get_me()
    finally:
        await bot.session.close()


async def detect_chat_id(token: str, seconds: int = 60) -> str | None:
    """Wait for the user to message the bot and return that chat id. Only reads updates."""
    from aiogram import Bot

    bot = Bot(token=token)
    try:
        for _ in range(max(1, seconds // 3)):
            updates = await bot.get_updates(timeout=2, allowed_updates=["message"])
            for u in reversed(updates):
                if u.message and u.message.chat.type == "private":
                    return str(u.message.chat.id)
            await asyncio.sleep(1)
    finally:
        await bot.session.close()
    return None


def _token_problem(exc: Exception) -> str:
    """Plain words for why a token check failed: a wrong token is not the same as no internet."""
    name = type(exc).__name__
    if name in ("TelegramUnauthorizedError", "TelegramNotFound"):
        return "Telegram says that token is not valid. Copy it again from @BotFather."
    if name in (
        "TelegramNetworkError",
        "ClientConnectorError",
        "TimeoutError",
        "ServerDisconnectedError",
    ):
        return "Couldn't reach Telegram. Are you online?"
    return f"The check failed ({name})."


# ------------------------------------------------------------------------------------------ steps


def step_env_file(io: Console, path: Path) -> None:
    if not path.exists():
        if EXAMPLE.exists():
            shutil.copyfile(EXAMPLE, path)
        else:
            path.write_text("", encoding="utf-8")
        io.say("Created .env")


def step_secret(io: Console, path: Path) -> None:
    current = envfile.read_env(path).get("SECRET_KEY", "")
    if not envfile.is_weak_secret(current):
        io.say("SECRET_KEY: already set (kept).")
        return
    envfile.set_values(path, {"SECRET_KEY": secrets.token_urlsafe(48)})
    io.say("SECRET_KEY: generated a strong random one (it encrypts saved keys and notes).")


def _ask_token(io: Console, current: str, label: str) -> str:
    """Ask for one bot token, check it with Telegram, return it ('' = skipped)."""
    while True:
        token = io.ask(f"  {label}", current, secret=True)
        if not token:
            return ""
        if ":" not in token:
            io.say("  That doesn't look like a bot token (it has a colon, like 123456:ABC...).")
            if not io.yes("  Try again?"):
                return ""
            continue
        try:
            me = asyncio.run(check_token(token))
        except Exception as exc:
            io.say(f"  {_token_problem(exc)}")
            if type(exc).__name__ in (
                "TelegramNetworkError",
                "ClientConnectorError",
                "TimeoutError",
            ) and io.yes("  Save it anyway and check later?", False):
                return token
            if not io.yes("  Try again?"):
                return ""
            continue
        io.say(f"  OK: @{me.username}")
        return token


def step_telegram(io: Console, path: Path) -> list[str]:
    values = envfile.read_env(path)
    io.say("\nTelegram bot")
    io.say("  Create one in Telegram: open @BotFather, send /newbot, copy the token it gives you.")
    main = _ask_token(io, values.get("TELEGRAM_BOT_TOKEN", ""), "Bot token")
    if not main:
        io.say(
            "  Skipped: the bot stays off until you set TELEGRAM_BOT_TOKEN (the server still runs)."
        )
        return []
    envfile.set_values(path, {"TELEGRAM_BOT_TOKEN": main})
    tokens = [main]
    extra = [t.strip() for t in values.get("TELEGRAM_BOT_TOKENS", "").split(",") if t.strip()]
    if extra:
        io.say(f"  {len(extra)} extra bot(s) already configured (kept).")
        tokens += extra
    elif io.yes("  Run more bots on the same backend? (they share one ledger)", False):
        while True:
            more = _ask_token(io, "", f"Bot token {len(tokens) + 1} (Enter to stop)")
            if not more:
                break
            if more not in tokens:
                tokens.append(more)
    if len(tokens) > 1:
        envfile.set_values(path, {"TELEGRAM_BOT_TOKENS": ",".join(tokens[1:])})
    return tokens


def step_owner(io: Console, path: Path, tokens: list[str]) -> None:
    values = envfile.read_env(path)
    io.say("\nYou (the admin: manages shared AI keys and sees bot stats)")
    chat_id = values.get("ADMIN_CHAT_IDS", "")
    if (
        tokens
        and not chat_id
        and io.yes("  Detect your Telegram id now? (send any message to your bot)")
    ):
        io.say("  Waiting up to 60 seconds... open your bot in Telegram and send it any message.")
        try:
            found = asyncio.run(detect_chat_id(tokens[0]))
        except Exception as exc:
            found = None
            if type(exc).__name__ == "TelegramConflictError":
                io.say(
                    "  The bot is already running somewhere (stop that copy to detect, or type your id)."
                )
            else:
                io.say(f"  {_token_problem(exc)}")
        if found:
            io.say(f"  Found your id: {found}")
            chat_id = found
        elif found is None and chat_id == "":
            io.say("  Nothing received. You can type it in (ask @userinfobot for your id).")
    chat_id = io.ask("  Your Telegram id (Enter to skip)", chat_id)
    if chat_id:
        envfile.set_values(path, {"ADMIN_CHAT_IDS": chat_id})
    else:
        io.say("  Skipped: nobody can use /admin until you set ADMIN_CHAT_IDS.")

    allowed = values.get("ALLOWED_CHAT_IDS", "")
    io.say("\nWho can use the bot?")
    io.say("  public  = anyone who finds it (every chat only ever sees its own data)  <- default")
    io.say("  private = only the Telegram ids you list")
    mode = io.ask("  Choose public or private", "private" if allowed else "public").lower()
    if mode.startswith("priv"):
        ids = io.ask("  Allowed Telegram ids, separated by commas", allowed or chat_id)
        envfile.set_values(path, {"ALLOWED_CHAT_IDS": ids})
        io.say("  Private: only those ids can use the bot.")
    else:
        envfile.set_values(path, {"ALLOWED_CHAT_IDS": ""})
        io.say("  Public: anyone can use the bot.")


def step_database(io: Console, path: Path) -> None:
    io.say("\nDatabase")
    io.say(
        "  sqlite = a single file, no setup (recommended to start). postgres = a database server."
    )
    mode = io.ask(
        "  Choose sqlite or postgres", envfile.read_env(path).get("DB_MODE", "sqlite")
    ).lower()
    if mode not in ("sqlite", "postgres"):
        mode = "sqlite"
    envfile.set_values(path, {"DB_MODE": mode})
    if mode == "postgres":
        io.say(
            "  Make sure a PostgreSQL server is running (connection settings are in .env: POSTGRES_*)"
        )


def step_tool_mode(io: Console, path: Path) -> None:
    io.say("\nHow the AI calls ledger tools")
    io.say("  direct = plain function calls inside the bot (recommended, fastest).")
    io.say("  mcp    = through the real MCP server, started next to the bot (for demos).")
    current = envfile.read_env(path).get("TOOL_MODE", "direct") or "direct"
    mode = io.ask("  Choose direct or mcp", current).lower()
    envfile.set_values(path, {"TOOL_MODE": mode if mode in ("direct", "mcp") else "direct"})


def step_ai_key(io: Console) -> None:
    from app.services_ai import ai_gateway as gw

    io.say(
        "\nAI key (optional: unlocks free-form chat and advice for everyone as a shared fallback)"
    )
    names = ", ".join(gw.PROVIDER_LABELS.get(p, p) for p in gw.PROVIDERS)
    io.say(f"  Works with {names}. Users can also add their own with /setkey.")
    io.say("  Without any key the bot still works with built-in rules.")
    key = io.ask("  Paste a key to test and save as the shared key (Enter to skip)", secret=True)
    if not key:
        return
    from app.db import init_db

    found = gw.extract_api_keys(key)
    if not found:
        io.say(
            "  That doesn't look like a supported key. Skipped. (For Mistral use /setkey mistral <key> in the bot.)"
        )
        return
    provider, clean = found[0]
    io.say(f"  Testing your {gw.label(provider)} key...")
    check = asyncio.run(gw.validate_api_key(provider, clean))
    if not check.usable:
        io.say(f"  Not saved: {check.detail or check.status}")
        return
    init_db()
    result = gw.add_api_key(gw.SHARED_OWNER, provider, clean)
    io.say(
        f"  Saved ({result['status']}): {gw.label(provider)} {result['masked_key']} (stored encrypted)"
    )


def step_profile(io: Console) -> None:
    from app.config import bot_tokens

    if not bot_tokens():
        return
    io.say("\nPublishing the bot's command menu...")
    from aiogram import Bot

    from app.bot import profile

    async def go():
        out = {}
        for token in bot_tokens():
            bot = Bot(token=token)
            try:
                out.update(
                    {
                        f"{token.split(':')[0]}/{k}": v
                        for k, v in (await profile.apply_profile(bot)).items()
                    }
                )
            finally:
                await bot.session.close()
        return out

    try:
        results = asyncio.run(go())
    except Exception as exc:  # cosmetic: never fail setup over the menu
        io.say(
            f"  Skipped ({_token_problem(exc)}). Run `uv run python scripts/setup_bot.py` later."
        )
        return
    bad = {k: v for k, v in results.items() if v != "ok"}
    io.say("  Done." if not bad else f"  Partly done: {bad}")


# ------------------------------------------------------------------------------------- the wizard


def run_wizard(path: Path = ENV, *, first_run: bool = False, io: Console | None = None) -> None:
    io = io or Console()
    if first_run:
        io.say("WalletLedger first start: a one-minute setup, then the server starts.")
    else:
        io.say("WalletLedger setup.")
    io.say("Press Enter to keep a value or skip a question. Ctrl+C cancels.\n")
    step_env_file(io, path)
    step_secret(io, path)
    tokens = step_telegram(io, path)
    step_owner(io, path, tokens)
    step_database(io, path)
    step_tool_mode(io, path)
    envfile.set_values(
        path, {"SETUP_DONE": "true"}
    )  # all .env writes are finished before app.config loads
    step_ai_key(io)
    step_profile(io)
    if first_run:
        io.say("\nAll set. Starting the server...")
        io.say("(Run `uv run python scripts/setup.py` again any time to change something.)\n")
    else:
        io.say("\nAll set. Start the bot with:")
        io.say("    uv run uvicorn main:app --reload --port 8000")
        io.say(
            "(The first start asks nothing now. Check health any time: uv run python scripts/setup.py --check)"
        )


def apply_defaults(
    path: Path = ENV,
    *,
    token: str = "",
    admin_id: str = "",
    private: bool = False,
    io: Console | None = None,
) -> None:
    """No questions: for scripts and containers. A strong secret, SQLite, a public bot."""
    io = io or Console()
    step_env_file(io, path)
    step_secret(io, path)
    updates = {"SETUP_DONE": "true", "ALLOWED_CHAT_IDS": admin_id if (private and admin_id) else ""}
    if token:
        updates["TELEGRAM_BOT_TOKEN"] = token
    if admin_id:
        updates["ADMIN_CHAT_IDS"] = admin_id
    envfile.set_values(path, updates)
    io.say(f".env is ready ({'private' if private and admin_id else 'public'} bot, SQLite).")


# ----------------------------------------------------------------------------------- health check


def doctor(io: Console | None = None) -> int:
    """Health check: everything a first-time user can get wrong, in plain words."""
    io = io or Console()
    from app import health
    from app.config import bot_tokens, settings
    from app.db import describe_database

    io.say("WalletLedger health check\n")
    ok = True
    tokens = bot_tokens()
    for number, token in enumerate(tokens, start=1):
        label = "Telegram token" if len(tokens) == 1 else f"Telegram bot {number}"
        try:
            me = asyncio.run(check_token(token))
            io.say(f"OK    {label} works (@{me.username})")
        except Exception as exc:
            ok = False
            io.say(f"FAIL  {label}: {_token_problem(exc)}")
    for problem in health.setup_problems():
        ok = False
        io.say(f"WARN  {problem}")
    for note in health.setup_notes():
        io.say(f"INFO  {note}")
    io.say(f"OK    Storage: {describe_database()}")
    io.say(f"OK    Tool mode: {settings.tool_mode}")
    try:
        path = asyncio.run(health.ai_path("__doctor__"))
    except Exception:  # fresh database: tables are created the first time the server starts
        path = health.AIPath(0, 0, 0, None, None)
        io.say(
            "INFO  Database not initialised yet (it is created the first time the server starts)"
        )
    if path.has_ai:
        io.say(f"OK    AI available ({path.mode}); model {path.first_model}")
    else:
        io.say(
            "INFO  No AI key yet: basic commands work, free-form chat is off. Add one with the wizard"
        )
        io.say("      or send /admin addkey <key> to your bot.")
    io.say("\nAll good." if ok else "\nFix the items above, then run this check again.")
    return 0 if ok else 1


def main(
    argv: list[str] | None = None,
    *,
    env_path: Path = ENV,
    io_factory: Callable[[], Console] = Console,
) -> int:
    ap = argparse.ArgumentParser(
        prog="setup.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="health check only, changes nothing")
    ap.add_argument(
        "--defaults", action="store_true", help="no questions: strong secret, SQLite, public bot"
    )
    ap.add_argument("--token", default="", help="with --defaults: the bot token to save")
    ap.add_argument("--admin-id", default="", help="with --defaults: your Telegram id (the admin)")
    ap.add_argument(
        "--private", action="store_true", help="with --defaults: only --admin-id may use the bot"
    )
    args = ap.parse_args(argv)
    io = io_factory()
    if args.check:
        return doctor(io)
    if args.defaults:
        apply_defaults(
            env_path, token=args.token, admin_id=args.admin_id, private=args.private, io=io
        )
        return 0
    try:
        run_wizard(env_path, io=io)
    except NoConsole:
        io.say(
            "\nThere is no terminal to ask questions on. Use --defaults (see --help) or edit .env by hand."
        )
        return 2
    return 0
