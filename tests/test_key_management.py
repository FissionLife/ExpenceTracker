"""Key management: live test before save, edit/remove/pause/test commands, server-key fallback,
command hints. Offline: litellm and provider model lists are mocked."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-keys-test-")
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
from unittest.mock import AsyncMock, patch  # noqa: E402

import litellm  # noqa: E402

from app.bot import main as bot  # noqa: E402
from app.db import init_db  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402

init_db()
_ids = itertools.count(1)
GOOD = "AIza" + "G" * 35
GOOD2 = "AIza" + "H" * 35
BAD = "AIza" + "B" * 35


def run(coro):
    return asyncio.run(coro)


def ok_response(text="hi"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def auth_error():
    return litellm.AuthenticationError("bad key", llm_provider="gemini", model="m")


def rate_error():
    return litellm.RateLimitError("429 rate limit", llm_provider="gemini", model="m")


def fake_provider(good_keys):
    """acompletion stand-in: keys in good_keys work, everything else is rejected."""

    async def _call(**kwargs):
        if kwargs.get("api_key") in good_keys:
            return ok_response("OK")
        raise auth_error()

    return _call


class KeyTestCase(unittest.TestCase):
    def setUp(self):
        gw.reset_rotation_state()
        self.chat = f"keys-{next(_ids)}"

    def say(self, text):
        return run(bot.handle_text(self.chat, text))

    def saved(self):
        return gw.get_user_api_keys(self.chat)


class ValidateBeforeSaveTests(KeyTestCase):
    def test_working_key_is_saved(self):
        with patch.object(gw.litellm, "acompletion", fake_provider({GOOD})):
            reply = self.say(f"/setkey {GOOD}")
        self.assertIn("works. Saved", reply)
        self.assertNotIn(GOOD, reply)
        self.assertEqual(self.saved(), [("gemini", GOOD)])

    def test_dead_key_is_rejected_and_not_saved(self):
        with patch.object(gw.litellm, "acompletion", fake_provider({GOOD})):
            reply = self.say(f"/setkey {BAD}")
        self.assertIn("Not saved", reply)
        self.assertIn("rejected", reply)
        self.assertEqual(self.saved(), [])

    def test_mixed_keys_save_only_the_working_one(self):
        with patch.object(gw.litellm, "acompletion", fake_provider({GOOD})):
            reply = self.say(f"/setkey {GOOD} {BAD}")
        self.assertEqual(self.saved(), [("gemini", GOOD)])
        self.assertIn("works. Saved", reply)
        self.assertIn("Not saved", reply)

    def test_rate_limited_key_is_still_saved(self):
        with patch.object(gw.litellm, "acompletion", AsyncMock(side_effect=rate_error())):
            reply = self.say(f"/setkey {GOOD}")
        self.assertIn("Saved", reply)
        self.assertEqual(self.saved(), [("gemini", GOOD)])

    def test_unverifiable_key_is_not_saved(self):
        boom = AsyncMock(side_effect=litellm.APIConnectionError("net down", "gemini", "m"))
        with patch.object(gw.litellm, "acompletion", boom):
            reply = self.say(f"/setkey {GOOD}")
        self.assertIn("couldn't verify", reply)
        self.assertEqual(self.saved(), [])

    def test_conversational_key_is_validated_too(self):
        with patch.object(gw.litellm, "acompletion", fake_provider(set())):
            reply = self.say(f"my gemini key is {BAD}")
        self.assertIn("Not saved", reply)
        self.assertEqual(self.saved(), [])


class ManageKeysTests(KeyTestCase):
    def setUp(self):
        super().setUp()
        with patch.object(gw.litellm, "acompletion", fake_provider({GOOD, GOOD2})):
            self.say(f"/setkey {GOOD} {GOOD2}")

    def test_keys_are_numbered_and_masked(self):
        reply = self.say("/keys")
        self.assertIn("1. Gemini", reply)
        self.assertIn("2. Gemini", reply)
        self.assertNotIn(GOOD, reply)

    def test_edit_replaces_only_after_new_key_passes(self):
        new = "AIza" + "N" * 35
        with patch.object(gw.litellm, "acompletion", fake_provider(set())):
            reply = self.say(f"/editkey 1 {new}")
        self.assertIn("didn't pass the test", reply)
        self.assertIn(GOOD, [k for _, k in self.saved()])
        with patch.object(gw.litellm, "acompletion", fake_provider({new})):
            reply = self.say(f"/editkey 1 {new}")
        self.assertIn("replaced", reply)
        keys = [k for _, k in self.saved()]
        self.assertIn(new, keys)
        self.assertNotIn(GOOD, keys)
        self.assertEqual(len(keys), 2)

    def test_delete_pause_resume(self):
        self.assertIn("paused", self.say("/pausekey 1"))
        self.assertEqual([k for _, k in self.saved()], [GOOD2])
        self.assertIn("active again", self.say("/resumekey 1"))
        self.assertEqual(len(self.saved()), 2)
        self.assertIn("removed", self.say("/delkey 2"))
        self.assertEqual(len(self.saved()), 1)

    def test_bad_numbers_get_helpful_errors(self):
        for cmd in ("/delkey 9", "/delkey abc", "/pausekey", "/editkey 7 " + GOOD):
            self.assertIn("1–2", self.say(cmd), cmd)
        self.assertIn("Usage", self.say("/editkey"))

    def test_cannot_touch_another_users_keys(self):
        other = f"keys-{next(_ids)}"
        reply = run(bot.handle_text(other, "/delkey 1"))
        self.assertIn("no saved keys", reply)
        self.assertEqual(len(self.saved()), 2)

    def test_testkey_reports_each_key(self):
        with patch.object(gw.litellm, "acompletion", fake_provider({GOOD})):
            reply = self.say("/testkey")
        self.assertIn("1.", reply)
        self.assertIn("valid", reply)
        self.assertIn("invalid", reply)
        self.assertIn("/delkey", reply)


class FallbackTests(KeyTestCase):
    def test_server_keys_follow_user_keys_and_are_deduped(self):
        with patch.object(gw.litellm, "acompletion", fake_provider({GOOD})):
            self.say(f"/setkey {GOOD}")
        with patch.dict(os.environ, {"GEMINI_API_KEYS": f"server-1,{GOOD}"}):
            order = [(c.source, c.api_key) for c in gw.build_candidates(self.chat)]
        self.assertEqual(order, [("user", GOOD), ("server", "server-1")])

    def test_broken_user_key_falls_back_to_server_key(self):
        with patch.object(gw.litellm, "acompletion", fake_provider({GOOD})):
            self.say(f"/setkey {GOOD}")
        with (
            patch.dict(os.environ, {"GEMINI_API_KEYS": "server-1"}),
            patch.object(gw.litellm, "acompletion", fake_provider({"server-1"})),
        ):
            text = run(gw.ask_llm(self.chat, "sys", "hello"))
        self.assertEqual(text, "OK")

    def test_has_server_keys(self):
        self.assertFalse(gw.has_server_keys())
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_x"}):
            self.assertTrue(gw.has_server_keys())

    def test_user_with_no_keys_uses_server_keys(self):
        with (
            patch.dict(os.environ, {"GEMINI_API_KEYS": "server-1"}),
            patch.object(gw.litellm, "acompletion", fake_provider({"server-1"})),
        ):
            self.assertEqual(run(gw.ask_llm(self.chat, "sys", "hello")), "OK")


class CommandHintTests(KeyTestCase):
    def test_menu_covers_every_key_command(self):
        names = {name for name, _ in bot.COMMAND_MENU}
        for cmd in ("setkey", "keys"):
            self.assertIn(cmd, names)
        for cmd in ("editkey", "delkey", "pausekey", "resumekey", "testkey"):
            self.assertIn("/" + cmd, bot.KNOWN_COMMANDS)  # hidden from the menu, still work
        self.assertTrue(all(desc and len(desc) <= 256 for _, desc in bot.COMMAND_MENU))
        self.assertLessEqual(len(bot.COMMAND_MENU), 100)

    def test_help_mentions_menu_and_keys(self):
        reply = self.say("/help")
        for word in ("/setkey", "/keys", "backup"):
            self.assertIn(word, reply)

    def test_unknown_command_suggests_close_matches(self):
        reply = self.say("/balanse")
        self.assertIn("/balance", reply)
        self.assertIn("don't know", reply)
        self.assertIn("/help", self.say("/zzzzzz"))

    def test_persona_commands_still_reach_the_agent(self):
        self.assertNotIn("don't know", self.say("/coach"))

    def test_edit_message_is_deleted_from_chat(self):
        self.assertTrue(bot.should_delete_message(f"/editkey 1 {GOOD}"))
        self.assertTrue(bot.should_delete_message("/key abc"))


if __name__ == "__main__":
    unittest.main()
