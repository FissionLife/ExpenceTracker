#!/usr/bin/env python3
"""Set up (or refresh) the bot's Telegram profile: "/" command menu, description, short
description and menu button. Safe to run any time, also while the server is running.

    uv run python scripts/setup_bot.py                 # apply everything
    uv run python scripts/setup_bot.py --show          # what Telegram has vs what this code publishes
    uv run python scripts/setup_bot.py --reset         # clear old commands in every scope first
    uv run python scripts/setup_bot.py --name "WalletLedger"   # also change the display name
                                                       # (Telegram rate-limits name changes)

Needs TELEGRAM_BOT_TOKEN in .env. The same profile is also applied automatically on server start.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiogram import Bot  # noqa: E402

from app.bot import profile  # noqa: E402
from app.config import bot_tokens  # noqa: E402


def show(info: dict) -> None:
    print(f"Bot:               {info['bot']}")
    print(f"Display name:      {info['name']!r}")
    print(f"Menu button:       {info['menu_button']}")
    print(f"Short description: {info['short_description']!r}")
    print(f"Description:       {info['description']!r}")
    print(f"Commands ({len(info['commands'])}):")
    for name, desc in info["commands"]:
        print(f"  /{name:<10} {desc}")
    print(
        "In sync with this code:", "yes" if info["in_sync"] else "NO - run without --show to apply"
    )


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--show", action="store_true", help="only show the current profile")
    ap.add_argument("--reset", action="store_true", help="clear commands in all scopes first")
    ap.add_argument("--name", help="also set the bot's display name")
    args = ap.parse_args()

    tokens = bot_tokens()
    if not tokens:
        print("TELEGRAM_BOT_TOKEN is not set. Add it to .env first (token from @BotFather).")
        return 2

    problems = profile.validate()
    if problems:
        print("Profile is invalid:\n  - " + "\n  - ".join(problems))
        return 2

    if args.name and len(tokens) > 1:
        # one display name for every bot would make them indistinguishable (and Telegram rate-limits renames)
        print(
            "--name sets the same display name on every bot. Run it with a single bot configured."
        )
        return 2

    code = 0
    for token in tokens:
        bot = Bot(token=token)
        try:
            if len(tokens) > 1:
                print(f"\n=== bot {token.split(':')[0]} ===")
            if args.show:
                show(await profile.read_profile(bot))
                continue
            results = await profile.apply_profile(bot, name=args.name, reset=args.reset)
            for step, outcome in results.items():
                ok = outcome == "ok"
                print(f"{'OK  ' if ok else 'FAIL'} {step}" + ("" if ok else f"  -> {outcome}"))
            print()
            show(await profile.read_profile(bot))
            if not all(v == "ok" for v in results.values()):
                code = 1
        except (
            Exception
        ) as exc:  # a bad token or no network: say which bot and carry on with the next one
            print(f"Could not reach bot {token.split(':')[0]}: {type(exc).__name__}")
            code = 1
        finally:
            await bot.session.close()
    return code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
