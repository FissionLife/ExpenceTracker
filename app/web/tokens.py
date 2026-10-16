"""Signed links for the web dashboard (no passwords) and Telegram Mini App sign-in.

A dashboard link is  <public_url>/d/<token>  where token = base64(payload).signature.
The payload only holds the Telegram chat id and an expiry time; the signature (HMAC-SHA256 with a key
derived from SECRET_KEY) proves the server made it, so nobody can forge a link for another user.
Links expire (DASHBOARD_LINK_MINUTES) and are only ever sent to the chat they belong to.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from urllib.parse import parse_qsl

from app.config import settings


def _key() -> bytes:
    return hashlib.sha256(f"walletledger-dashboard:{settings.secret_key}".encode()).digest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(chat_id: str, ttl_seconds: int | None = None) -> str:
    ttl = ttl_seconds if ttl_seconds is not None else settings.dashboard_link_minutes * 60
    payload = _b64(
        json.dumps({"c": str(chat_id), "e": int(time.time()) + ttl}, separators=(",", ":")).encode()
    )
    sig = _b64(hmac.new(_key(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def read_token(token: str) -> str | None:
    """The chat id inside a valid, unexpired token; None for anything else."""
    try:
        payload, sig = token.split(".", 1)
        expected = _b64(hmac.new(_key(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(_unb64(payload))
        if int(data["e"]) < time.time():
            return None
        return str(data["c"])
    except Exception:
        return None


def base_url() -> str:
    return (settings.public_url or "http://localhost:8000").rstrip("/")


def is_https() -> bool:
    return base_url().startswith("https://")


def dashboard_url(chat_id: str) -> str:
    return f"{base_url()}/d/{make_token(chat_id)}"


def validate_init_data(init_data: str, bot_token: str, max_age_seconds: int = 86400) -> dict | None:
    """Verify the `initData` string a Telegram Mini App receives (Telegram's documented HMAC check).

    Returns the user dict ({"id": ..., ...}) if it is genuine and fresh, else None.
    """
    try:
        pairs = dict(parse_qsl(init_data, keep_blank_values=True))
        received = pairs.pop("hash")
        check = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
        secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        expected = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(received, expected):
            return None
        age = time.time() - int(pairs.get("auth_date", "0"))
        if (
            age > max_age_seconds or age < -300
        ):  # too old, or dated in the future (beyond clock drift)
            return None
        user = json.loads(pairs["user"])
        if (
            not isinstance(user, dict)
            or not isinstance(user.get("id"), int)
            or isinstance(user["id"], bool)
        ):
            return None
        return user
    except Exception:
        return None
