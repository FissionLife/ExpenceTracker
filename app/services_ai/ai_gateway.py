"""AI gateway: one entry point for every LLM call, whatever the provider.

One entry point for every LLM call in WalletLedger:

    reply = await ask_llm(user_id, system_prompt, user_prompt, model_preference=None)

What it does:

* **Provider-agnostic routing** via LiteLLM: Gemini, OpenAI, Claude (Anthropic), Groq, Ollama.
* **Key vault**: users' API keys are Fernet-encrypted in the ``api_keys`` table. The cipher is
  derived from ``settings.secret_key`` exactly like ``mcp_server.save_user_api_key``, so keys
  stored by either module are interchangeable.
* **Round-robin rotation**: consecutive requests for a user start from the next key, spreading
  load across free-tier keys (RPM / TPM limits).
* **Automatic fallback**: rate-limit (429), quota, auth and provider-outage errors move on to the
  next key / provider immediately. Rate-limited keys sit out a short cooldown.
* **Tool Calling Support**: seamlessly accepts standard OpenAI/Gemini function calling schemas.

Key sources, in order: the user's own active keys, then optional server keys from environment
variables (``GEMINI_API_KEYS``, ``OPENAI_API_KEYS``, ``ANTHROPIC_API_KEYS``, ``GROQ_API_KEYS``;
comma-separated). Ollama needs no key (``OLLAMA_API_BASE``, default http://localhost:11434).
Default models can be overridden with ``AI_MODEL_<PROVIDER>``, e.g. ``AI_MODEL_GEMINI``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from cryptography.fernet import InvalidToken, MultiFernet
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import crypto
from app.config import settings
from app.models import ApiKey, User
from app.services_ai import model_catalog

logger = logging.getLogger(__name__)

litellm.telemetry = False
litellm.suppress_debug_info = True

# --------------------------------------------------------------------------------------
# Providers & models
# --------------------------------------------------------------------------------------

DEFAULT_PROVIDER = "gemini"

# provider -> (default LiteLLM model, env var holding comma-separated server keys)
PROVIDERS: dict[str, tuple[str, str | None]] = {
    # "gemini-flash-latest" always points at Google's current Flash model; pinned versions
    # such as gemini-1.5-flash / gemini-2.5-flash are retired for new API keys (404).
    "gemini": ("gemini/gemini-flash-latest", "GEMINI_API_KEYS"),
    "openai": ("openai/gpt-4o-mini", "OPENAI_API_KEYS"),
    "anthropic": ("anthropic/claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEYS"),
    "groq": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEYS"),
    "openrouter": ("openrouter/openai/gpt-4o-mini", "OPENROUTER_API_KEYS"),
    "xai": ("xai/grok-3-mini", "XAI_API_KEYS"),
    "deepseek": ("deepseek/deepseek-chat", "DEEPSEEK_API_KEYS"),
    "mistral": ("mistral/mistral-small-latest", "MISTRAL_API_KEYS"),
    "ollama": ("ollama/llama3.1", None),
}

# Names shown to users (in /keys, /status and key tests)
PROVIDER_LABELS = {
    "gemini": "Gemini",
    "openai": "OpenAI",
    "anthropic": "Claude",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "xai": "xAI Grok",
    "deepseek": "DeepSeek",
    "mistral": "Mistral",
    "ollama": "Ollama",
}

# provider -> standard single-key env var (used after the *_API_KEYS list for that provider),
# e.g. GEMINI_API_KEYS -> GEMINI_API_KEY.
SINGLE_KEY_ENV: dict[str, str] = {
    provider: env_name.removesuffix("S")
    for provider, (_, env_name) in PROVIDERS.items()
    if env_name
}

# Every provider name maps to itself, plus common nicknames.
PROVIDER_ALIASES = {
    **{provider: provider for provider in PROVIDERS},
    "grok": "xai",
    "mistralai": "mistral",
    "google": "gemini",
    "gpt": "openai",
    "chatgpt": "openai",
    "claude": "anthropic",
}

# Recognisable key formats, used when a user pastes keys into chat.
KEY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic", re.compile(r"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("groq", re.compile(r"(?<![A-Za-z0-9_-])gsk_[A-Za-z0-9]{20,}")),
    ("openrouter", re.compile(r"(?<![A-Za-z0-9_-])sk-or-(?:v1-)?[A-Za-z0-9_\-]{20,}")),
    ("xai", re.compile(r"(?<![A-Za-z0-9_-])xai-[A-Za-z0-9]{20,}")),
    ("deepseek", re.compile(r"(?<![A-Za-z0-9_-])sk-[a-f0-9]{32}(?![A-Za-z0-9_\-])")),
    ("openai", re.compile(r"(?<![A-Za-z0-9_-])sk-(?!ant-|or-)[A-Za-z0-9_\-]{20,}")),
    ("gemini", re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_\-]{35}")),
    ("gemini", re.compile(r"(?<![A-Za-z0-9_-])AQ\.[0-9A-Za-z_\-.]{20,}")),
]

RATE_LIMIT_COOLDOWN_SECONDS = 60
AUTH_FAILURE_COOLDOWN_SECONDS = 15 * 60
DEFAULT_TIMEOUT_SECONDS = 15
FAST_ATTEMPT_SECONDS = 8  # per-model patience while other models are still available


def label(provider: str) -> str:
    """Friendly provider name for messages ("anthropic" -> "Claude")."""
    return PROVIDER_LABELS.get(provider, (provider or "").title())


def normalize_provider(provider: str) -> str:
    key = (provider or "").strip().lower()
    if key not in PROVIDER_ALIASES:
        raise ValueError(f"Unsupported provider {provider!r}. Supported: {', '.join(PROVIDERS)}")
    return PROVIDER_ALIASES[key]


def _env(name: str) -> str:
    """Read a setting: real environment variable first, then ``.env`` via app.config.

    pydantic-settings loads ``.env`` into ``settings`` but not into ``os.environ``, so
    reading only ``os.getenv`` would silently ignore keys placed in ``.env``.
    """
    value = os.getenv(name)
    if value is not None:
        return value
    return str(getattr(settings, name.lower(), "") or "")


def default_model(provider: str) -> str:
    provider = normalize_provider(provider)
    return (_env(f"AI_MODEL_{provider.upper()}") or None) or PROVIDERS[provider][0]


def resolve_model(model_preference: str | None) -> tuple[str, str]:
    """Turn a preference ("groq", "claude", "gemini/gemini-2.5-pro", None) into (provider, model)."""
    if not model_preference:
        return DEFAULT_PROVIDER, default_model(DEFAULT_PROVIDER)
    pref = model_preference.strip()
    if "/" in pref:
        prefix = pref.split("/", 1)[0].lower()
        if prefix in PROVIDER_ALIASES:
            return PROVIDER_ALIASES[prefix], pref
        raise ValueError(f"Unsupported model {model_preference!r}")
    lowered = pref.lower()
    if lowered in PROVIDER_ALIASES:
        provider = PROVIDER_ALIASES[lowered]
        return provider, default_model(provider)
    for prefix, provider in (
        ("gemini", "gemini"),
        ("gpt", "openai"),
        ("o1", "openai"),
        ("o3", "openai"),
        ("o4", "openai"),
        ("claude", "anthropic"),
    ):
        if lowered.startswith(prefix):
            return provider, f"{provider}/{pref}"
    raise ValueError(
        f"Cannot infer a provider for model {model_preference!r}; use 'provider/model'"
    )


def detect_provider(api_key: str) -> str | None:
    """Guess the provider from a key's format (None if unknown)."""
    for provider, pattern in KEY_PATTERNS:
        if pattern.fullmatch(api_key.strip()):
            return provider
    return None


def extract_api_keys(text: str) -> list[tuple[str, str]]:
    """Find every recognisable API key in a chat message -> [(provider, key), ...]."""
    found: list[tuple[int, str, str]] = []
    taken: list[tuple[int, int]] = []
    for provider, pattern in KEY_PATTERNS:
        for match in pattern.finditer(text or ""):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue
            taken.append(span)
            found.append((span[0], provider, match.group(0)))
    return [(provider, key) for _, provider, key in sorted(found)]


def parse_provider_and_key(text: str) -> tuple[str, str] | None:
    """'/setkey mistral abc123...' style: a provider name followed by a key with no known prefix."""
    tokens = (text or "").split()
    if len(tokens) >= 2 and tokens[0].lower() in PROVIDER_ALIASES:
        key = tokens[1].strip()
        if len(key) >= 16 and not re.search(r"\s", key):
            return PROVIDER_ALIASES[tokens[0].lower()], key
    return None


_SECRET_IN_URL = re.compile(
    r"(?i)\b(key|api[_-]?key|token|access_token|authorization)=([^&\s\"']+)"
)


def redact_secrets(text: str) -> str:
    """Hide API keys in text that is logged or shown: recognisable keys and `?key=...` style parameters."""
    for _, key in extract_api_keys(text or ""):
        text = text.replace(key, mask_key(key))
    return _SECRET_IN_URL.sub(lambda m: f"{m.group(1)}=***", text or "")


def mask_key(api_key: str) -> str:
    key = api_key.strip()
    return f"{key[:4]}…{key[-4:]}" if len(key) > 10 else "****"


# --------------------------------------------------------------------------------------
# Encryption
# --------------------------------------------------------------------------------------


def _fernet() -> MultiFernet:
    # HKDF-derived vault key (app/crypto.py): writes use SECRET_KEY, reads also SECRET_KEY_OLD.
    return crypto.vault_fernet()


def encrypt_api_key(api_key: str) -> str:
    return _fernet().encrypt(api_key.strip().encode("utf-8")).decode("utf-8")


def decrypt_api_key(encrypted_key: str) -> str:
    # New format first, then the legacy sha256(SECRET_KEY) format. Raises InvalidToken.
    return crypto.decrypt_vault(encrypted_key)


# --------------------------------------------------------------------------------------
# Key vault (api_keys table)
# --------------------------------------------------------------------------------------


def _session() -> Session:
    from app.db import SessionLocal

    return SessionLocal()


def _find_user(db: Session, user_id: str) -> User | None:
    """Accept either the internal user UUID or the Telegram chat id."""
    user = db.get(User, str(user_id))
    if user is None:
        user = db.scalar(select(User).where(User.telegram_chat_id == str(user_id)))
    return user


def _find_or_create_user(db: Session, user_id: str) -> User:
    user = _find_user(db, user_id)
    if user is None:
        from app.services.ledger import get_or_create_user

        user = get_or_create_user(db, str(user_id))
    return user


def _owned_key(db: Session, user_id: str, key_id: str) -> ApiKey | None:
    """The ApiKey row ``key_id`` if it belongs to ``user_id``, else None."""
    user = _find_user(db, user_id)
    record = db.get(ApiKey, key_id)
    if user is None or record is None or record.user_id != user.id:
        return None
    return record


def add_api_key(user_id: str, provider: str, api_key: str) -> dict[str, Any]:
    """Encrypt and store one key. Re-adding an existing key just re-activates it."""
    provider = normalize_provider(provider)
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("api_key is required")
    with _session() as db:
        user = _find_or_create_user(db, user_id)
        for record in db.scalars(
            select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.provider == provider)
        ):
            try:
                same = decrypt_api_key(record.encrypted_key) == api_key
            except InvalidToken:
                continue
            if same:
                record.is_active = True
                db.commit()
                return {
                    "status": "exists",
                    "key_id": record.id,
                    "provider": provider,
                    "masked_key": mask_key(api_key),
                }
        record = ApiKey(user_id=user.id, provider=provider, encrypted_key=encrypt_api_key(api_key))
        db.add(record)
        db.commit()
        return {
            "status": "added",
            "key_id": record.id,
            "provider": provider,
            "masked_key": mask_key(api_key),
        }


def add_api_keys_from_text(
    user_id: str, text: str, default_provider: str | None = None
) -> list[dict[str, Any]]:
    """Store every key found in a chat message (users may paste several Gemini keys at once)."""
    keys = extract_api_keys(text)
    if not keys and default_provider:
        tokens = [t for t in re.split(r"[\s,;]+", text or "") if len(t) >= 20]
        keys = [(normalize_provider(default_provider), token) for token in tokens]
    return [add_api_key(user_id, provider, key) for provider, key in keys]


def list_api_keys(user_id: str) -> list[dict[str, Any]]:
    """Masked view of a user's keys - never returns the plain key."""
    with _session() as db:
        user = _find_user(db, user_id)
        if user is None:
            return []
        rows = db.scalars(
            select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at, ApiKey.id)
        )
        result = []
        for record in rows:
            try:
                masked = mask_key(decrypt_api_key(record.encrypted_key))
            except InvalidToken:
                masked = "<undecryptable>"
            result.append(
                {
                    "key_id": record.id,
                    "provider": record.provider,
                    "masked_key": masked,
                    "is_active": record.is_active,
                }
            )
        return result


def _set_key_active(user_id: str, key_id: str, active: bool) -> bool:
    with _session() as db:
        record = _owned_key(db, user_id, key_id)
        if record is None:
            return False
        record.is_active = active
        db.commit()
        return True


def deactivate_api_key(user_id: str, key_id: str) -> bool:
    return _set_key_active(user_id, key_id, False)


def activate_api_key(user_id: str, key_id: str) -> bool:
    return _set_key_active(user_id, key_id, True)


def delete_api_key(user_id: str, key_id: str) -> bool:
    with _session() as db:
        record = _owned_key(db, user_id, key_id)
        if record is None:
            return False
        db.delete(record)
        db.commit()
        return True


# --------------------------------------------------------------------------------------
# Rotation state
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyCandidate:
    provider: str
    api_key: str = field(repr=False)
    source: str  # "user" or "server"
    key_id: str | None = None

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:16]


class _RotationState:
    """Thread-safe round-robin cursors and per-key cooldowns (in memory, per process)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cursors: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}

    def next_offset(self, pool: str, size: int) -> int:
        if size <= 0:
            return 0
        with self._lock:
            offset = self._cursors.get(pool, 0) % size
            self._cursors[pool] = offset + 1
            return offset

    def cool_down(self, fingerprint: str, seconds: float) -> None:
        with self._lock:
            self._cooldown_until[fingerprint] = time.monotonic() + seconds

    def is_cooling(self, fingerprint: str) -> bool:
        with self._lock:
            until = self._cooldown_until.get(fingerprint)
            if until is None:
                return False
            if until <= time.monotonic():
                del self._cooldown_until[fingerprint]
                return False
            return True

    def reset(self) -> None:
        with self._lock:
            self._cursors.clear()
            self._cooldown_until.clear()


_rotation = _RotationState()


def reset_rotation_state() -> None:
    _rotation.reset()


def _rotate(items: list[KeyCandidate], pool: str) -> list[KeyCandidate]:
    if not items:
        return items
    offset = _rotation.next_offset(pool, len(items))
    return items[offset:] + items[:offset]


def _server_keys(provider: str) -> list[str]:
    env_name = PROVIDERS[provider][1]
    if not env_name:
        return []
    raw = _env(env_name)
    return [k.strip() for k in re.split(r"[\s,;]+", raw) if k.strip()]


def _single_env_key(provider: str) -> str | None:
    """Standard single-key variable (GEMINI_API_KEY, OPENAI_API_KEY, ...), if set."""
    env_name = SINGLE_KEY_ENV.get(provider)
    value = _env(env_name).strip() if env_name else ""
    return value or None


SHARED_OWNER = "__shared__"  # pseudo-user that owns the admin-managed shared keys


def _load_user_keys(user_id: str, source: str = "user") -> list[KeyCandidate]:
    with _session() as db:
        user = _find_user(db, user_id)
        if user is None:
            return []
        rows = db.scalars(
            select(ApiKey)
            .where(ApiKey.user_id == user.id, ApiKey.is_active.is_(True))
            .order_by(ApiKey.created_at, ApiKey.id)
        )
        keys = []
        for record in rows:
            try:
                provider = normalize_provider(record.provider)
                keys.append(
                    KeyCandidate(provider, decrypt_api_key(record.encrypted_key), source, record.id)
                )
            except InvalidToken, ValueError:
                logger.warning("Skipping unusable api key %s", record.id)
        return keys


def get_user_api_keys(user_id: str) -> list[tuple[str, str]]:
    """Convenience helper: returns active (provider, plain_key) pairs."""
    candidates = build_candidates(user_id)
    return [(c.provider, c.api_key) for c in candidates if c.api_key]


def _grouped_rotated(
    keys: list[KeyCandidate], preferred: str, pool_prefix: str
) -> list[KeyCandidate]:
    """Group keys by provider (preferred provider first), each group round-robin rotated.

    The rotation pool for a provider is ``f"{pool_prefix}:{provider}"``.
    """
    by_provider: dict[str, list[KeyCandidate]] = {}
    for key in keys:
        by_provider.setdefault(key.provider, []).append(key)
    order = [preferred] + [p for p in by_provider if p != preferred]
    result: list[KeyCandidate] = []
    for provider in order:
        result += _rotate(by_provider.get(provider, []), f"{pool_prefix}:{provider}")
    return result


def build_candidates(
    user_id: str, preferred_provider: str = DEFAULT_PROVIDER
) -> list[KeyCandidate]:
    """Ordered list of keys to try for one request (round-robin rotated, cooling keys last).

    The user's own keys come first. The team's server keys always follow as a fallback, so a
    user key that is wrong, revoked or out of quota never leaves the user without AI.
    """
    candidates = _grouped_rotated(_load_user_keys(user_id), preferred_provider, f"user:{user_id}")

    # Shared keys managed by the admin from Telegram (stored encrypted in the database).
    candidates += _grouped_rotated(
        _load_user_keys(SHARED_OWNER, source="shared"), preferred_provider, "shared"
    )

    # Server keys from .env, grouped by provider so the preferred provider is always tried first.
    # Within a provider: the rotated GEMINI_API_KEYS-style list, then the single
    # GEMINI_API_KEY-style variable (skipped if it is already in the list).
    order = [preferred_provider] + [p for p in PROVIDERS if p != preferred_provider]
    for provider in order:
        listed = _server_keys(provider)
        server = [KeyCandidate(provider, k, "server") for k in listed]
        candidates += _rotate(server, f"server:{provider}")
        single = _single_env_key(provider)
        if single and single not in listed:
            candidates.append(KeyCandidate(provider, single, "env"))

    seen: set[tuple[str, str]] = set()
    unique = []
    for c in candidates:
        if (c.provider, c.api_key) not in seen:
            seen.add((c.provider, c.api_key))
            unique.append(c)

    ready: list[KeyCandidate] = []
    cooling: list[KeyCandidate] = []
    for c in unique:
        (cooling if _rotation.is_cooling(c.fingerprint) else ready).append(c)
    return ready + cooling


def has_server_keys() -> bool:
    """True if shared fallback keys exist, in .env or the admin-managed store (never reveals them)."""
    if any(_server_keys(p) or _single_env_key(p) for p in PROVIDERS):
        return True
    return bool(_load_user_keys(SHARED_OWNER, source="shared"))


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class AIGatewayError(RuntimeError):
    """Base error raised by the gateway."""


class NoAPIKeyError(AIGatewayError):
    """The user has no usable key and no server key is configured."""


class AllKeysExhaustedError(AIGatewayError):
    """Every candidate key/provider failed with a retryable error."""

    def __init__(self, attempts: list[dict[str, str]]):
        self.attempts = attempts
        summary = "; ".join(f"{a['provider']} {a['key']}: {a['error']}" for a in attempts)
        super().__init__(f"All {len(attempts)} key(s) failed. {summary}")


_RATE_LIMIT_ERRORS = (litellm.RateLimitError,)
_AUTH_ERRORS = (litellm.AuthenticationError, litellm.PermissionDeniedError)
_TRANSIENT_ERRORS = (
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.BadGatewayError,
    litellm.NotFoundError,
)
_KEY_HINTS = (
    "api key",
    "api_key",
    "apikey",
    "quota",
    "credential",
    "unauthorized",
    "permission",
    "access denied",
    "no access",
    "forbidden",
    "model not found",
    "not found or no access",
)


def _classify(exc: Exception) -> str | None:
    """'rate_limit' | 'auth' | 'transient' -> try next key; None -> not a key problem, raise."""
    if isinstance(exc, _RATE_LIMIT_ERRORS):
        return "rate_limit"
    if isinstance(exc, _AUTH_ERRORS):
        return "auth"
    if isinstance(exc, _TRANSIENT_ERRORS):
        return "transient"
    if isinstance(exc, (litellm.BadRequestError, litellm.APIError)):
        message = str(exc).lower()
        if "429" in message or "resource_exhausted" in message or "rate limit" in message:
            return "rate_limit"
        if any(hint in message for hint in _KEY_HINTS):
            return "auth"
    return None


def _short_error(exc: Exception) -> str:
    return redact_secrets(_short_error_raw(exc))


def _short_error_raw(exc: Exception) -> str:
    text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return f"{type(exc).__name__}: {text[:160]}"


# --------------------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------------------


def _model_for(candidate: KeyCandidate, provider: str, model: str) -> str:
    return model if candidate.provider == provider else default_model(candidate.provider)


def _wants_default_model(model_preference: str | None) -> bool:
    """True when the caller did not name a specific model ("groq" or nothing, not "groq/x")."""
    pref = (model_preference or "").strip().lower()
    return not pref or pref in PROVIDER_ALIASES


def _is_model_problem(exc: Exception) -> bool:
    """True when the MODEL (not the key) is the problem: retired, busy, slow or unknown."""
    if isinstance(
        exc,
        (
            litellm.NotFoundError,
            litellm.Timeout,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
            litellm.BadGatewayError,
        ),
    ):
        return True
    text = str(exc).lower()
    return any(
        hint in text
        for hint in (
            "no longer available",
            "high demand",
            "overloaded",
            "model not found",
            "is not found",
        )
    )


async def _candidate_models(
    candidate: KeyCandidate, provider: str, model: str, dynamic: bool
) -> list[str]:
    """Models to try for this key, best first (one entry when the caller pinned a model)."""
    if dynamic and settings.dynamic_models and not _env(f"AI_MODEL_{candidate.provider.upper()}"):
        base = _env("OLLAMA_API_BASE") or "http://localhost:11434"
        chain = await model_catalog.model_chain(candidate.provider, candidate.api_key, base)
        if chain:
            return chain
    return [_model_for(candidate, provider, model)]


def _extract_text(response: Any) -> str:
    content = response.choices[0].message.content
    return (content or "").strip()


async def complete(
    user_id: str,
    messages: list[dict[str, Any]],
    model_preference: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Run a chat completion with rotation + fallback.
    Returns plain text string if no tools, or full LiteLLM response object if tools are used.
    """
    provider, model = resolve_model(model_preference)
    dynamic = _wants_default_model(model_preference)
    candidates = await asyncio.to_thread(build_candidates, user_id, provider)
    if provider == "ollama" or not candidates:
        if provider == "ollama":
            candidates = [KeyCandidate("ollama", "", "local")]
        else:
            raise NoAPIKeyError(
                "No AI API key found. Send your key in chat (e.g. 'my Gemini key is AIza...') "
                "or configure GEMINI_API_KEYS on the server."
            )
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT_SECONDS)

    attempts: list[dict[str, str]] = []
    for candidate in candidates:
        models = await _candidate_models(candidate, provider, model, dynamic)
        call_kwargs = dict(kwargs)
        if candidate.provider == "ollama":
            call_kwargs.setdefault("api_base", _env("OLLAMA_API_BASE") or "http://localhost:11434")
        else:
            call_kwargs["api_key"] = candidate.api_key

        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"

        masked = mask_key(candidate.api_key) if candidate.api_key else "local"
        for position, call_model in enumerate(models):
            attempt_kwargs = dict(call_kwargs)
            if position < len(models) - 1:  # a slow model should not stall the chat: try the next
                attempt_kwargs["timeout"] = min(
                    float(call_kwargs.get("timeout", DEFAULT_TIMEOUT_SECONDS)), FAST_ATTEMPT_SECONDS
                )
            try:
                response = await litellm.acompletion(
                    model=call_model, messages=messages, num_retries=0, **attempt_kwargs
                )
            except Exception as exc:
                kind = _classify(exc)
                attempts.append(
                    {
                        "provider": candidate.provider,
                        "key": masked,
                        "model": call_model,
                        "error": _short_error(exc),
                    }
                )
                if _is_model_problem(exc):
                    # This model is retired for this key or busy right now: try the next model
                    # on the SAME key before giving up on the key.
                    model_catalog.forget_working(candidate.provider, candidate.api_key)
                    if position < len(models) - 1:
                        logger.info(
                            "AI gateway: %s unavailable for key %s, next model", call_model, masked
                        )
                        continue
                    kind = kind or "transient"
                if kind is None:
                    raise
                logger.warning(
                    "AI gateway: %s key %s failed (%s), trying next",
                    candidate.provider,
                    masked,
                    kind,
                )
                if kind == "rate_limit":
                    _rotation.cool_down(candidate.fingerprint, RATE_LIMIT_COOLDOWN_SECONDS)
                elif kind == "auth":
                    _rotation.cool_down(candidate.fingerprint, AUTH_FAILURE_COOLDOWN_SECONDS)
                break  # next key

            if dynamic and candidate.provider != "ollama":
                model_catalog.remember_working(
                    candidate.provider, candidate.api_key, call_model.split("/", 1)[-1]
                )
            return response if tools else _extract_text(response)

    raise AllKeysExhaustedError(attempts)


# --------------------------------------------------------------------------------------
# Key testing & editing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyCheck:
    """Result of a live key test. ``usable`` keys may be saved."""

    status: str  # "valid" | "rate_limited" | "invalid" | "unverified"
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status in ("valid", "rate_limited")


def _explain_rejection(provider: str, exc: Exception) -> str:
    """Turn a provider's auth error into something a user can act on."""
    text = str(exc)
    if "ACCESS_TOKEN_TYPE_UNSUPPORTED" in text:
        return (
            "Google says this key type isn't accepted by the Gemini API "
            "(ACCESS_TOKEN_TYPE_UNSUPPORTED). Create a fresh key at aistudio.google.com/apikey"
        )
    if "API_KEY_SERVICE_BLOCKED" in text:
        return (
            "the key's project blocks the Gemini API (API_KEY_SERVICE_BLOCKED). Enable the "
            "Generative Language API for that project, or use a key from another project"
        )
    if "API key not valid" in text or "API_KEY_INVALID" in text:
        return "Google says this API key is not valid (typo, revoked, or copied incompletely)"
    if "insufficient_quota" in text or "billing" in text.lower():
        return "the account has no quota or billing set up"
    return "the provider rejected this key"


async def validate_api_key(provider: str, api_key: str) -> KeyCheck:
    """Test a key with a tiny real request, so a dead key is never silently saved.

    valid        - the provider answered
    rate_limited - the key is real but out of quota/rate right now (still worth saving)
    invalid      - rejected (revoked, mistyped, wrong provider, no access)
    unverified   - could not tell (offline, provider outage); caller decides
    """
    provider = normalize_provider(provider)
    api_key = (api_key or "").strip()
    if not api_key:
        return KeyCheck("invalid", "empty key")
    models = [default_model(provider)]
    if settings.dynamic_models and not _env(f"AI_MODEL_{provider.upper()}"):
        models = (await model_catalog.model_chain(provider, api_key))[:3] or models
    last_exc: Exception | None = None
    for model in models:
        try:
            await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word OK."}],
                api_key=api_key,
                max_tokens=32,
                timeout=12,
                num_retries=0,
            )
        except Exception as exc:
            kind = _classify(exc)
            if kind == "rate_limit":
                return KeyCheck("rate_limited", "quota or rate limit reached right now")
            # A retired/unknown model (e.g. "model not found", which _KEY_HINTS also matches)
            # says nothing about the key: try the next model. Real auth error classes still
            # mean the key itself was rejected.
            if _is_model_problem(exc) and not isinstance(exc, _AUTH_ERRORS):
                last_exc = exc  # this model is busy/retired for this key: try the next one
                continue
            if kind == "auth":
                return KeyCheck("invalid", _explain_rejection(provider, exc))
            return KeyCheck("unverified", _short_error(exc))
        model_catalog.remember_working(provider, api_key, model.split("/", 1)[-1])
        return KeyCheck("valid")
    # Every model was busy or unavailable. If the provider still lets this key list its models,
    # the key itself is genuine, so don't throw it away because of a temporary outage.
    if await model_catalog.list_models(provider, api_key):
        return KeyCheck("valid", "key accepted; the AI models are busy right now")
    return KeyCheck("unverified", _short_error(last_exc) if last_exc else "no answer")


async def add_validated_keys(user_id: str, text: str) -> list[dict[str, Any]]:
    """Find keys in ``text``, test each live, and save only the ones that work.

    Each result has ``status``: added | exists | rejected, plus provider, masked_key, check.
    """
    explicit = parse_provider_and_key(text)
    found = list(dict.fromkeys(([explicit] if explicit else []) + extract_api_keys(text)))
    checks = await asyncio.gather(*(validate_api_key(p, k) for p, k in found))
    results: list[dict[str, Any]] = []
    for (provider, key), check in zip(found, checks, strict=True):
        if not check.usable:
            results.append(
                {
                    "status": "rejected",
                    "provider": provider,
                    "masked_key": mask_key(key),
                    "check": check,
                }
            )
            continue
        saved = await asyncio.to_thread(add_api_key, user_id, provider, key)
        results.append({**saved, "check": check})
    return results


def replace_api_key(user_id: str, key_id: str, provider: str, api_key: str) -> bool:
    """Swap the stored secret of an existing key (edit in place). False if it is not theirs."""
    provider = normalize_provider(provider)
    with _session() as db:
        record = _owned_key(db, user_id, key_id)
        if record is None:
            return False
        record.provider = provider
        record.encrypted_key = encrypt_api_key(api_key.strip())
        record.is_active = True
        db.commit()
        return True


def get_plain_key(user_id: str, key_id: str) -> tuple[str, str] | None:
    """(provider, plain key) for one of the user's own keys, or None. For re-testing only."""
    with _session() as db:
        record = _owned_key(db, user_id, key_id)
        if record is None:
            return None
        try:
            return record.provider, decrypt_api_key(record.encrypted_key)
        except InvalidToken:
            return None


async def ask_llm(
    user_id: str,
    system_prompt: str,
    user_prompt: str,
    model_preference: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    messages_history: list[dict[str, str]] | None = None,
) -> Any:
    """Contract 2 (Agent -> Gateway): pull the user's round-robin key and run via LiteLLM."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if messages_history:
        messages.extend(messages_history)
    messages.append({"role": "user", "content": user_prompt})
    return await complete(user_id, messages, model_preference=model_preference, tools=tools)


async def ask_persona(
    user_id: str,
    user_prompt: str,
    mode: str | None = None,
    context: Any = None,
    model_preference: str | None = None,
) -> str:
    """Convenience: pick a persona (auto-detected from the message if ``mode`` is None)."""
    from app.services_ai.prompts import build_system_prompt, detect_mode

    chosen = mode or detect_mode(user_prompt)
    return await ask_llm(
        user_id, build_system_prompt(chosen, context), user_prompt, model_preference
    )
