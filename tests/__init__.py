"""Test package setup: runs before any test module imports app settings.

Tests must never reach the real Telegram API, even when a developer's .env has a bot token.
"""

import os

os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["TIMEZONE"] = (
    "UTC"  # tests compare dates; a fixed zone keeps them the same on every machine
)
os.environ["USE_WEBHOOK"] = "false"
