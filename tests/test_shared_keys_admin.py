"""Shared (admin-managed) keys, /admin, /status and Gemini AQ. keys. Offline."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-shared-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
for _var in (
    "GEMINI_API_KEYS",
    "GEMINI_API_KEY",
    "OPENAI_API_KEYS",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEYS",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEYS",
    "GROQ_API_KEY",
    "AI_MODEL_GEMINI",
):
    os.environ.pop(_var, None)

import asyncio  # noqa: E402
import itertools  # noqa: E402
import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402

import litellm  # noqa: E402

from app.bot import main as bot  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402

init_db()
_ids = itertools.count(1)
AQ = "AQ.Ab8RN6" + "q" * 30 + ".x_y-Z"
USER_KEY = "AIza" + "U" * 35
SHARED_KEY = "AIza" + "S" * 35


def run(coro):
    return asyncio.run(coro)


def ok_response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])


def provider(good):
    async def _call(**kw):
        if kw.get("api_key") in good:
            return ok_response()
        raise litellm.AuthenticationError("bad", llm_provider="gemini", model="m")

    return _call


class Base(unittest.TestCase):
    def setUp(self):
        gw.reset_rotation_state()
        self.chat = f"shared-{next(_ids)}"
        self._admins = settings.admin_chat_ids
        settings.admin_chat_ids = "ADMIN1"

    def tearDown(self):
        settings.admin_chat_ids = self._admins
        for k in gw.list_api_keys(gw.SHARED_OWNER):  # the shared store is global: keep tests apart
            gw.delete_api_key(gw.SHARED_OWNER, k["key_id"])

    def say(self, text, chat=None):
        return run(bot.handle_text(chat or self.chat, text))


class AQKeyTests(Base):
    def test_aq_keys_with_dots_are_detected_everywhere(self):
        self.assertEqual(gw.detect_provider(AQ), "gemini")
        self.assertEqual(gw.extract_api_keys(f"my key is {AQ}, thanks"), [("gemini", AQ)])
        self.assertTrue(bot.should_delete_message(f"/setkey {AQ}"))
        self.assertNotIn(AQ, gw.mask_key(AQ))

    def test_aq_key_is_sent_unchanged_and_saved_when_it_works(self):
        seen = {}

        async def call(**kw):
            seen.update(kw)
            return ok_response()

        with patch.object(gw.litellm, "acompletion", call):
            reply = self.say(f"/setkey {AQ}")
        self.assertEqual(seen["api_key"], AQ)
        self.assertIn("works. Saved", reply)
        self.assertEqual(gw.get_user_api_keys(self.chat)[0], ("gemini", AQ))

    def test_rejection_reasons_are_actionable(self):
        def rejected(text):
            return AsyncMock_raise(litellm.AuthenticationError(text, llm_provider="g", model="m"))

        for text, hint in (
            ("401 ACCESS_TOKEN_TYPE_UNSUPPORTED", "aistudio.google.com/apikey"),
            ("403 API_KEY_SERVICE_BLOCKED", "Generative Language API"),
            ("API key not valid. Please pass a valid API key.", "not valid"),
        ):
            with patch.object(gw.litellm, "acompletion", rejected(text)):
                reply = self.say(f"/setkey {AQ}")
            self.assertIn(hint, reply, text)
            self.assertIn("Not saved", reply)
            self.assertEqual(gw.get_user_api_keys(self.chat), [])


def AsyncMock_raise(exc):
    async def _call(**kw):
        raise exc

    return _call


class SharedKeyTests(Base):
    def add_shared(self):
        with patch.object(gw.litellm, "acompletion", provider({SHARED_KEY})):
            return self.say(f"/admin addkey {SHARED_KEY}", chat="ADMIN1")

    def test_only_admins_can_use_admin(self):
        self.assertIn("Admin only", self.say("/admin keys"))
        self.assertIn("Admin only", self.say(f"/admin addkey {SHARED_KEY}"))
        self.assertEqual(gw.list_api_keys(gw.SHARED_OWNER), [])
        settings.admin_chat_ids = ""
        self.assertIn("Admin only", self.say("/admin keys", chat="ADMIN1"))

    def test_admin_adds_tests_lists_and_removes_shared_keys(self):
        self.assertIn("works. Saved", self.add_shared())
        listing = self.say("/admin keys", chat="ADMIN1")
        self.assertIn("1. Gemini", listing)
        self.assertNotIn(SHARED_KEY, listing)
        with patch.object(gw.litellm, "acompletion", provider(set())):
            self.assertIn("invalid", self.say("/admin testkeys", chat="ADMIN1"))
        self.assertIn("paused", self.say("/admin pausekey 1", chat="ADMIN1"))
        self.assertFalse(gw.has_server_keys())
        self.assertIn("active again", self.say("/admin resumekey 1", chat="ADMIN1"))
        self.assertTrue(gw.has_server_keys())
        self.assertIn("removed", self.say("/admin delkey 1", chat="ADMIN1"))
        self.assertEqual(gw.list_api_keys(gw.SHARED_OWNER), [])

    def test_dead_shared_key_is_rejected(self):
        with patch.object(gw.litellm, "acompletion", provider(set())):
            reply = self.say(f"/admin addkey {SHARED_KEY}", chat="ADMIN1")
        self.assertIn("Not saved", reply)
        self.assertEqual(gw.list_api_keys(gw.SHARED_OWNER), [])

    def test_order_is_own_then_shared_then_env_and_shared_keys_stay_private(self):
        self.add_shared()
        with patch.object(gw.litellm, "acompletion", provider({USER_KEY})):
            self.say(f"/setkey {USER_KEY}")
        with patch.dict(os.environ, {"GEMINI_API_KEYS": "env-key-1"}):
            order = [(c.source, c.api_key) for c in gw.build_candidates(self.chat)]
        self.assertEqual(
            order, [("user", USER_KEY), ("shared", SHARED_KEY), ("server", "env-key-1")]
        )
        # users never see shared keys in their own /keys list
        own_listing = self.say("/keys")
        self.assertIn(gw.mask_key(USER_KEY), own_listing)
        self.assertNotIn(gw.mask_key(SHARED_KEY), own_listing)
        self.assertEqual(len(gw.list_api_keys(self.chat)), 1)

    def test_user_without_keys_gets_ai_through_the_shared_key(self):
        self.add_shared()
        with patch.object(gw.litellm, "acompletion", provider({SHARED_KEY})):
            self.assertEqual(run(gw.ask_llm(self.chat, "sys", "hi")), "OK")

    def test_admin_stats(self):
        self.assertIn("Users:", self.say("/admin stats", chat="ADMIN1"))
        self.assertIn("/admin keys", self.say("/admin", chat="ADMIN1"))


class StatusTests(Base):
    def test_status_offline(self):
        reply = self.say("/status")
        self.assertIn("offline", reply)
        self.assertIn("/setkey", reply)

    def test_status_with_own_and_shared_keys(self):
        with patch.object(gw.litellm, "acompletion", provider({SHARED_KEY, USER_KEY})):
            self.say(f"/admin addkey {SHARED_KEY}", chat="ADMIN1")
            self.say(f"/setkey {USER_KEY}")
        reply = self.say("/status")
        self.assertIn("own key, then shared fallback", reply)
        self.assertIn("Your keys: 1", reply)
        self.assertIn("shared fallback keys: 1", reply)
        self.assertNotIn(USER_KEY, reply)
        self.assertNotIn(SHARED_KEY, reply)

    def test_status_is_in_the_menu(self):
        self.assertIn("status", [n for n, _ in bot.COMMAND_MENU])


class HealthTests(Base):
    def test_setup_problems_are_plain_language(self):
        from app import health

        with (
            patch.object(settings, "allowed_chat_ids", ""),
            patch.object(settings, "admin_chat_ids", ""),
            patch.object(settings, "telegram_bot_token", ""),
        ):
            problems = " ".join(health.setup_problems())
            notes = " ".join(
                health.setup_notes()
            )  # a public bot is intended: a note, not a problem
        self.assertIn("TELEGRAM_BOT_TOKEN", problems)
        self.assertNotIn("ALLOWED_CHAT_IDS", problems)
        self.assertIn("Public bot", notes)
        self.assertIn("ADMIN_CHAT_IDS", notes)


if __name__ == "__main__":
    unittest.main()
