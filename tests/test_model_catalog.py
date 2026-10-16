"""Dynamic model discovery: picking, caching, failure fallback and gateway integration.

Offline only: provider list endpoints and litellm.acompletion are mocked.
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-catalog-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "true"
for _var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "AI_MODEL_GEMINI", "AI_MODEL_GROQ"):
    os.environ.pop(_var, None)

import asyncio  # noqa: E402
import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

from app.db import init_db  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402
from app.services_ai import model_catalog as mc  # noqa: E402

init_db()
GEMINI_KEY = "AIza" + "Z" * 35


def run(coro):
    return asyncio.run(coro)


def ok_response(text="hi"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class PickBestTests(unittest.TestCase):
    def test_gemini_prefers_latest_alias_then_newest_stable_flash(self):
        ids = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-pro"]
        self.assertEqual(mc.pick_best("gemini", ids), "gemini-flash-latest")
        ids = [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-3-flash",
            "gemini-3-flash-preview",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-image",
            "gemini-3-pro",
            "gemini-embedding-001",
        ]
        self.assertEqual(mc.pick_best("gemini", ids), "gemini-3-flash")

    def test_openai_picks_newest_mini_and_skips_special_models(self):
        ids = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-5-mini",
            "gpt-5-mini-audio-preview",
            "gpt-4o-mini-realtime-preview",
            "text-embedding-3-small",
        ]
        self.assertEqual(mc.pick_best("openai", ids), "gpt-5-mini")

    def test_anthropic_prefers_newest_sonnet_then_haiku(self):
        ids = [
            "claude-3-5-sonnet-20241022",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-1-20250805",
        ]
        self.assertEqual(mc.pick_best("anthropic", ids), "claude-sonnet-4-5-20250929")
        self.assertEqual(
            mc.pick_best("anthropic", ["claude-haiku-4-5-20251001"]), "claude-haiku-4-5-20251001"
        )

    def test_groq_and_ollama(self):
        ids = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "whisper-large-v3",
            "llama-guard-4-12b",
        ]
        self.assertEqual(mc.pick_best("groq", ids), "llama-3.3-70b-versatile")
        self.assertEqual(
            mc.pick_best("ollama", ["mistral:7b", "llama3.1:8b", "llama3.2:3b"]), "llama3.2:3b"
        )
        self.assertEqual(mc.pick_best("ollama", ["mistral:7b"]), "mistral:7b")

    def test_nothing_usable_returns_none(self):
        self.assertIsNone(mc.pick_best("gemini", ["gemini-embedding-001"]))
        self.assertIsNone(mc.pick_best("openai", []))
        self.assertIsNone(mc.pick_best("unknown", ["x"]))


class ListingTests(unittest.TestCase):
    def setUp(self):
        mc.clear_cache()

    def test_cached_and_prefixed(self):
        fetch = AsyncMock(return_value=["gemini-flash-latest"])
        with patch.dict(mc._FETCHERS, {"gemini": fetch}):
            self.assertEqual(run(mc.best_model("gemini", "k1")), "gemini/gemini-flash-latest")
            run(mc.best_model("gemini", "k1"))
            self.assertEqual(fetch.await_count, 1)  # second call served from cache
            run(mc.best_model("gemini", "k2"))
            self.assertEqual(fetch.await_count, 2)  # cache is per key
            mc.invalidate("gemini", "k1")
            run(mc.best_model("gemini", "k1"))
            self.assertEqual(fetch.await_count, 3)

    def test_failure_returns_none_and_is_cached_briefly(self):
        fetch = AsyncMock(side_effect=RuntimeError("offline"))
        with patch.dict(mc._FETCHERS, {"gemini": fetch}):
            self.assertIsNone(run(mc.best_model("gemini", "k")))
            self.assertIsNone(run(mc.best_model("gemini", "k")))
            self.assertEqual(fetch.await_count, 1)


class GatewayIntegrationTests(unittest.TestCase):
    def setUp(self):
        mc.clear_cache()
        gw.reset_rotation_state()
        # a developer's local .env may pin models; these tests are about discovery
        for name in ("ai_model_gemini", "ai_model_groq"):
            patcher = patch.object(gw.settings, name, "")
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(gw.settings, "dynamic_models", True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.chat = f"catalog-{id(self)}"
        gw.add_api_key(self.chat, "gemini", GEMINI_KEY)

    def complete(self, pref=None, discovered=("gemini-3-flash",)):
        fetch = AsyncMock(return_value=list(discovered))
        call = AsyncMock(return_value=ok_response())
        with (
            patch.dict(mc._FETCHERS, {"gemini": fetch}),
            patch.object(gw.litellm, "acompletion", call),
        ):
            run(gw.complete(self.chat, [{"role": "user", "content": "hi"}], pref))
        return call.await_args.kwargs["model"], fetch

    def test_uses_discovered_model_by_default(self):
        model, _ = self.complete()
        self.assertEqual(model, "gemini/gemini-3-flash")
        model, _ = self.complete("gemini")
        self.assertEqual(model, "gemini/gemini-3-flash")

    def test_explicit_model_is_never_overridden(self):
        model, fetch = self.complete("gemini/gemini-2.5-pro")
        self.assertEqual(model, "gemini/gemini-2.5-pro")
        fetch.assert_not_awaited()

    def test_env_override_wins(self):
        with patch.dict(os.environ, {"AI_MODEL_GEMINI": "gemini/my-pinned"}):
            model, fetch = self.complete()
        self.assertEqual(model, "gemini/my-pinned")
        fetch.assert_not_awaited()

    def test_falls_back_to_static_default_when_discovery_finds_nothing(self):
        model, _ = self.complete(discovered=())
        self.assertEqual(model, gw.PROVIDERS["gemini"][0])

    def test_can_be_switched_off(self):
        with patch.object(gw.settings, "dynamic_models", False):
            model, fetch = self.complete()
        self.assertEqual(model, gw.PROVIDERS["gemini"][0])
        fetch.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
