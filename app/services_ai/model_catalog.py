"""Dynamic model discovery, so model names never need manual updates.

For each provider we ask its own "list models" endpoint which models the given API key can use,
pick the best fit by rule (see ``_RULES``), and cache the answer. If the lookup fails (offline,
bad key, unknown response) the caller falls back to the static default in ai_gateway.PROVIDERS.

    model = await best_model("gemini", api_key)      # e.g. "gemini/gemini-flash-latest" or None
    invalidate("gemini", api_key)                    # call after a "model not found" error
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

SUCCESS_TTL_SECONDS = 6 * 3600
FAILURE_TTL_SECONDS = 5 * 60
REQUEST_TIMEOUT_SECONDS = 6.0

_EXCLUDE_COMMON = re.compile(
    r"embed|audio|realtime|transcribe|whisper|tts|image|vision-preview|moderation|guard|"
    r"search|instruct|codex|live|robotics|computer-use|aqa|imagen|veo|learnlm|gemma|"
    r"thinking|exp\b|-exp-|preview",
    re.I,
)

_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}


def _fingerprint(api_key: str) -> str:
    return hashlib.sha256((api_key or "").encode()).hexdigest()[:16]


def invalidate(provider: str, api_key: str = "") -> None:
    _cache.pop((provider, _fingerprint(api_key)), None)


def clear_cache() -> None:
    _cache.clear()
    _working.clear()


# ----------------------------- listing endpoints ---------------------------------------


async def _get_json(url: str, headers: dict[str, str]) -> dict:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _fetch_gemini(api_key: str, _base: str) -> list[str]:
    data = await _get_json(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        {"x-goog-api-key": api_key},
    )
    return [
        m["name"].removeprefix("models/")
        for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


async def _fetch_openai_style(url: str, api_key: str) -> list[str]:
    data = await _get_json(url, {"Authorization": f"Bearer {api_key}"})
    return [m["id"] for m in data.get("data", [])]


async def _fetch_openai(api_key: str, _base: str) -> list[str]:
    return await _fetch_openai_style("https://api.openai.com/v1/models", api_key)


async def _fetch_groq(api_key: str, _base: str) -> list[str]:
    return await _fetch_openai_style("https://api.groq.com/openai/v1/models", api_key)


async def _fetch_xai(api_key: str, _base: str) -> list[str]:
    return await _fetch_openai_style("https://api.x.ai/v1/models", api_key)


async def _fetch_deepseek(api_key: str, _base: str) -> list[str]:
    return await _fetch_openai_style("https://api.deepseek.com/models", api_key)


async def _fetch_mistral(api_key: str, _base: str) -> list[str]:
    return await _fetch_openai_style("https://api.mistral.ai/v1/models", api_key)


async def _fetch_anthropic(api_key: str, _base: str) -> list[str]:
    data = await _get_json(
        "https://api.anthropic.com/v1/models?limit=1000",
        {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
    )
    return [m["id"] for m in data.get("data", [])]


async def _fetch_ollama(_api_key: str, base: str) -> list[str]:
    data = await _get_json(f"{base.rstrip('/')}/api/tags", {})
    return [m["name"] for m in data.get("models", [])]


_FETCHERS: dict[str, Callable] = {
    "gemini": _fetch_gemini,
    "openai": _fetch_openai,
    "anthropic": _fetch_anthropic,
    "groq": _fetch_groq,
    "xai": _fetch_xai,
    "deepseek": _fetch_deepseek,
    "mistral": _fetch_mistral,
    "ollama": _fetch_ollama,
}


# ----------------------------- choosing the best ---------------------------------------


def _version_key(model_id: str) -> tuple:
    """Higher = newer. Version numbers compare first, date stamps (>= 10000) break ties."""
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", model_id)]
    return tuple(n for n in nums if n < 10000), tuple(n for n in nums if n >= 10000)


def _newest(ids: list[str], include: str, exclude: str = "") -> str | None:
    inc = re.compile(include, re.I)
    exc = re.compile(exclude, re.I) if exclude else None
    pool = [
        i
        for i in ids
        if inc.search(i) and not _EXCLUDE_COMMON.search(i) and not (exc and exc.search(i))
    ]
    return max(pool, key=_version_key) if pool else None


def _pick_gemini(ids: list[str]) -> str | None:
    if "gemini-flash-latest" in ids:  # Google's moving alias: always the current Flash
        return "gemini-flash-latest"
    return _newest(ids, r"^gemini-[\d.]+-flash$") or _newest(ids, r"^gemini-.*flash", r"lite")


def _pick_openai(ids: list[str]) -> str | None:
    return _newest(ids, r"^gpt-[\d.]+-mini$") or _newest(ids, r"^gpt-[\d.]+o?-mini")


def _pick_anthropic(ids: list[str]) -> str | None:
    return _newest(ids, r"^claude-.*sonnet") or _newest(ids, r"^claude-.*haiku")


def _pick_groq(ids: list[str]) -> str | None:
    return _newest(ids, r"llama.*versatile", r"distil") or _newest(ids, r"^llama")


def _pick_ollama(ids: list[str]) -> str | None:
    llama = [i for i in ids if i.lower().startswith("llama3")]
    return max(llama, key=_version_key) if llama else (ids[0] if ids else None)


def _pick_xai(ids: list[str]) -> str | None:
    return _newest(ids, r"^grok-[\d.]+-mini$") or _newest(ids, r"^grok-[\d.]+$")


def _pick_deepseek(ids: list[str]) -> str | None:
    return "deepseek-chat" if "deepseek-chat" in ids else None


def _pick_mistral(ids: list[str]) -> str | None:
    for name in ("mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"):
        if name in ids:
            return name
    return None


_RULES: dict[str, Callable[[list[str]], str | None]] = {
    "xai": _pick_xai,
    "deepseek": _pick_deepseek,
    "mistral": _pick_mistral,
    "gemini": _pick_gemini,
    "openai": _pick_openai,
    "anthropic": _pick_anthropic,
    "groq": _pick_groq,
    "ollama": _pick_ollama,
}


def pick_best(provider: str, model_ids: list[str]) -> str | None:
    """Pure function: choose the best chat model id for a provider from its listed ids."""
    rule = _RULES.get(provider)
    return rule(model_ids) if rule else None


# ----------------------------- public API ----------------------------------------------


async def list_models(provider: str, api_key: str = "", base_url: str = "") -> list[str]:
    """Model ids the key can use ([] when lookup fails). Cached; failures are cached briefly."""
    fetch = _FETCHERS.get(provider)
    if fetch is None:
        return []
    cache_key = (provider, _fingerprint(api_key))
    hit = _cache.get(cache_key)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    try:
        ids = await fetch(api_key, base_url)
        ttl = SUCCESS_TTL_SECONDS if ids else FAILURE_TTL_SECONDS
    except Exception as exc:  # network, 401/403, bad JSON: fall back to static defaults
        logger.info("Model discovery failed for %s (%s)", provider, type(exc).__name__)
        ids, ttl = [], FAILURE_TTL_SECONDS
    _cache[cache_key] = (time.monotonic() + ttl, ids)
    return ids


async def best_model(provider: str, api_key: str = "", base_url: str = "") -> str | None:
    """LiteLLM model name (``provider/id``) discovered for this key, or None to use the default."""
    ids = await list_models(provider, api_key, base_url)
    best = pick_best(provider, ids)
    return f"{provider}/{best}" if best else None


# ----------------------------- ranked chain + "what worked" memory ---------------------

_WORKING_TTL_SECONDS = 6 * 3600
_working: dict[tuple[str, str], tuple[float, str]] = {}


def _newest_all(ids: list[str], include: str, exclude: str = "", limit: int = 3) -> list[str]:
    inc = re.compile(include, re.I)
    exc = re.compile(exclude, re.I) if exclude else None
    pool = [
        i
        for i in ids
        if inc.search(i) and not _EXCLUDE_COMMON.search(i) and not (exc and exc.search(i))
    ]
    return sorted(pool, key=_version_key, reverse=True)[:limit]


def rank_models(provider: str, ids: list[str]) -> list[str]:
    """Up to ~5 chat models in the order worth trying (the first that answers is used).

    A model can be listed yet still fail for one key (retired for new projects, or overloaded), so
    callers try down this list and remember the one that works.
    """
    out: list[str] = []

    def add(*names: str) -> None:
        for n in names:
            if n and n in ids and n not in out:
                out.append(n)

    if provider == "gemini":
        add("gemini-flash-latest")
        add(*_newest_all(ids, r"^gemini-[\d.]+-flash$", limit=2))
        add("gemini-flash-lite-latest")
        add(*_newest_all(ids, r"^gemini-[\d.]+-flash-lite$", limit=1))
        # previews are excluded by _EXCLUDE_COMMON; allow the newest as a last resort
        previews = [i for i in ids if re.match(r"^gemini-[\d.]+-flash(-lite)?-preview", i)]
        add(*sorted(previews, key=_version_key, reverse=True)[:1])
    elif provider == "openai":
        add(*_newest_all(ids, r"^gpt-[\d.]+-mini$", limit=2))
        add(*_newest_all(ids, r"^gpt-[\d.]+o?-mini", limit=1))
        add(*_newest_all(ids, r"^gpt-[\d.]+o?$", limit=1))
    elif provider == "anthropic":
        add(*_newest_all(ids, r"^claude-.*sonnet", limit=2))
        add(*_newest_all(ids, r"^claude-.*haiku", limit=1))
    elif provider == "groq":
        add(*_newest_all(ids, r"llama.*versatile", r"distil", limit=2))
        add(*_newest_all(ids, r"^llama.*instant", limit=1))
    else:
        best = pick_best(provider, ids)
        add(best or "")
    if not out:
        best = pick_best(provider, ids)
        if best:
            out.append(best)
    return out[:5]


def remember_working(provider: str, api_key: str, model_id: str) -> None:
    _working[(provider, _fingerprint(api_key))] = (
        time.monotonic() + _WORKING_TTL_SECONDS,
        model_id,
    )


def forget_working(provider: str, api_key: str) -> None:
    _working.pop((provider, _fingerprint(api_key)), None)


def _working_model(provider: str, api_key: str) -> str | None:
    hit = _working.get((provider, _fingerprint(api_key)))
    if hit and hit[0] > time.monotonic():
        return hit[1]
    return None


async def model_chain(provider: str, api_key: str = "", base_url: str = "") -> list[str]:
    """LiteLLM names (``provider/id``) to try for this key: the one that last worked, then the ranking."""
    ids = await list_models(provider, api_key, base_url)
    ranked = rank_models(provider, ids)
    worked = _working_model(provider, api_key)
    if worked and worked in ranked:
        ranked.remove(worked)
        ranked.insert(0, worked)
    elif worked and worked not in ranked and ids and worked in ids:
        ranked.insert(0, worked)
    return [f"{provider}/{m}" for m in ranked]
