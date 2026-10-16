"""Plain-language health/status of the bot. Used by /status in Telegram and by scripts/setup.py."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.config import bot_tokens, settings
from app.crypto import weak_secret_key
from app.db import describe_database
from app.services_ai import ai_gateway as gw
from app.services_ai import model_catalog


@dataclass
class AIPath:
    own: int  # the user's active keys
    shared: int  # admin-managed shared keys (database)
    env: int  # keys from .env
    first_provider: str | None
    first_model: str | None

    @property
    def mode(self) -> str:
        if self.own:
            return "own key, then shared fallback" if (self.shared or self.env) else "own key only"
        if self.shared or self.env:
            return "shared key"
        return "offline (no AI key)"

    @property
    def has_ai(self) -> bool:
        return bool(self.own or self.shared or self.env)


async def ai_path(user_id: str) -> AIPath:
    """Which AI keys would be used for this user right now, and which model the first one gets."""
    candidates = await asyncio.to_thread(gw.build_candidates, user_id)
    own = sum(1 for c in candidates if c.source == "user")
    shared = sum(1 for c in candidates if c.source == "shared")
    env = sum(1 for c in candidates if c.source in ("server", "env"))
    first = next((c for c in candidates if c.api_key), None)
    model = None
    if first is not None:
        model = None
        if settings.dynamic_models:
            model = await model_catalog.best_model(first.provider, first.api_key)
        model = model or gw.default_model(first.provider)
    return AIPath(own, shared, env, first.provider if first else None, model)


async def user_status_text(chat_id: str) -> str:
    path = await ai_path(chat_id)
    lines = ["🩺 **Status**", ""]
    if path.has_ai:
        lines.append(f"🧠 AI: on ({path.mode})")
        lines.append(f"   model: {path.first_model} ({path.first_provider})")
    else:
        lines.append("🧠 AI: offline (basic commands only)")
        lines.append("   add a key with /setkey <key> to unlock free-form questions and advice")
    lines.append(f"🔑 Your keys: {path.own}  ·  shared fallback keys: {path.shared + path.env}")
    lines.append(f"💾 Storage: {describe_database().split(' (')[0]}")
    return "\n".join(lines)


def setup_problems() -> list[str]:
    """Things that make the bot unsafe or unusable, in plain words (empty list = healthy)."""
    problems = []
    if not bot_tokens():
        problems.append("TELEGRAM_BOT_TOKEN is empty, so the bot is off.")
    if weak_secret_key():
        problems.append(
            "SECRET_KEY is the public default or too short; saved AI keys and encrypted chat "
            "history are not safe."
        )
    if settings.use_webhook and not settings.webhook_url:
        problems.append("USE_WEBHOOK=true but WEBHOOK_URL is empty.")
    return problems


def setup_notes() -> list[str]:
    """Facts worth knowing that are not faults (a public bot is a normal, intended setup)."""
    notes = []
    if settings.allowed_chat_ids.strip():
        count = len([c for c in settings.allowed_chat_ids.split(",") if c.strip()])
        notes.append(f"Private bot: only {count} Telegram id(s) in ALLOWED_CHAT_IDS can use it.")
    else:
        notes.append("Public bot: anyone can use it; every chat only sees its own data.")
    if not settings.admin_chat_ids.strip():
        notes.append("ADMIN_CHAT_IDS is empty, so nobody can use /admin (shared keys, stats).")
    return notes
