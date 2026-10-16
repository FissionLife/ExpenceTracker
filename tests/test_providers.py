"""More AI providers and the bring-your-own-key onboarding. Offline."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-providers-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
for _var in (
    "GEMINI_API_KEYS", "GEMINI_API_KEY", "OPENAI_API_KEYS", "OPENAI_API_KEY", "ANTHROPIC_API_KEYS",
    "GROQ_API_KEYS", "OPENROUTER_API_KEYS", "XAI_API_KEYS", "DEEPSEEK_API_KEYS", "MISTRAL_API_KEYS",
    "MISTRAL_API_KEY",
):  # fmt: skip
    os.environ[_var] = ""

import asyncio  # noqa: E402
import itertools  # noqa: E402
import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app.bot import main as bot  # noqa: E402
from app.db import init_db  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402
from app.services_ai import model_catalog as mc  # noqa: E402
from app.services_ai.mcp_server import execute_tool  # noqa: E402

try:
    from tests.fake_llm import FakeLLM, tool  # noqa: E402
except ImportError:  # discover -s tests (no -t .) puts tests/ on sys.path directly
    from fake_llm import FakeLLM, tool  # noqa: E402

init_db()
_ids = itertools.count(1)

KEYS = {
    "gemini": "AIza" + "A" * 35,
    "gemini-aq": "AQ.Ab8RN6" + "z" * 30,
    "anthropic": "sk-ant-" + "a" * 30,
    "groq": "gsk_" + "b" * 30,
    "openrouter": "sk-or-v1-" + "c" * 40,
    "xai": "xai-" + "d" * 30,
    "deepseek": "sk-" + "e" * 32,
    "openai": "sk-proj-" + "f" * 40,
}


def run(coro):
    return asyncio.run(coro)


def ok_response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])


class DetectionTests(unittest.TestCase):
    def test_every_provider_format_is_recognised(self):
        expected = {
            "gemini": "gemini", "gemini-aq": "gemini", "anthropic": "anthropic", "groq": "groq",
            "openrouter": "openrouter", "xai": "xai", "deepseek": "deepseek", "openai": "openai",
        }  # fmt: skip
        for name, provider in expected.items():
            found = gw.extract_api_keys(f"here is my key {KEYS[name]} thanks")
            self.assertEqual(found, [(provider, KEYS[name])], name)
            self.assertEqual(gw.detect_provider(KEYS[name]), provider)

    def test_similar_prefixes_do_not_get_confused(self):
        self.assertEqual(gw.detect_provider("sk-" + "1" * 48), "openai")  # long sk- key
        self.assertEqual(gw.detect_provider("sk-or-v1-" + "9" * 40), "openrouter")
        self.assertEqual(gw.detect_provider("sk-ant-" + "9" * 30), "anthropic")
        self.assertIsNone(gw.detect_provider("not-a-key"))

    def test_several_keys_in_one_message(self):
        text = f"{KEYS['gemini']} and {KEYS['groq']}, plus {KEYS['xai']}"
        self.assertEqual([p for p, _ in gw.extract_api_keys(text)], ["gemini", "groq", "xai"])

    def test_explicit_provider_for_keys_without_a_prefix(self):
        key = "abcdefghijklmnopqrstuvwxyz123456"
        self.assertEqual(gw.parse_provider_and_key(f"mistral {key}"), ("mistral", key))
        self.assertEqual(gw.parse_provider_and_key(f"Grok {KEYS['xai']}"), ("xai", KEYS["xai"]))
        self.assertIsNone(gw.parse_provider_and_key("mistral short"))
        self.assertIsNone(gw.parse_provider_and_key("hello world"))
        self.assertIsNone(gw.parse_provider_and_key(key))

    def test_aliases_labels_and_defaults(self):
        for alias, provider in {
            "grok": "xai",
            "Mistral": "mistral",
            "mistralai": "mistral",
            "claude": "anthropic",
            "google": "gemini",
        }.items():
            self.assertEqual(gw.normalize_provider(alias), provider)
        self.assertEqual(gw.label("anthropic"), "Claude")
        self.assertEqual(gw.label("xai"), "xAI Grok")
        self.assertEqual(gw.label("unknown"), "Unknown")
        for p in ("openrouter", "xai", "deepseek", "mistral"):
            self.assertIn("/", gw.default_model(p))
        with self.assertRaises(ValueError):
            gw.normalize_provider("skynet")

    def test_models_pick_for_the_new_providers(self):
        self.assertEqual(
            mc.pick_best("deepseek", ["deepseek-reasoner", "deepseek-chat"]), "deepseek-chat"
        )
        self.assertEqual(
            mc.pick_best("mistral", ["mistral-large-latest", "mistral-small-latest"]),
            "mistral-small-latest",
        )
        self.assertEqual(
            mc.pick_best("xai", ["grok-3", "grok-4-mini", "grok-3-mini"]), "grok-4-mini"
        )
        self.assertIsNone(mc.pick_best("deepseek", ["something-else"]))
        for p in ("xai", "deepseek", "mistral"):
            self.assertIn(p, mc._FETCHERS)


class SetKeyFlowTests(unittest.TestCase):
    def setUp(self):
        gw.reset_rotation_state()
        self.chat = f"prov-{next(_ids)}"

    def say(self, text):
        return run(bot.handle_text(self.chat, text))

    def test_each_provider_key_is_tested_with_its_own_model_and_saved(self):
        seen = []

        async def ok(**kw):
            seen.append((kw["model"], kw["api_key"]))
            return ok_response()

        for name in ("openrouter", "xai", "deepseek", "groq", "anthropic"):
            with patch.object(gw.litellm, "acompletion", ok):
                reply = self.say(f"/setkey {KEYS[name]}")
            self.assertIn("works. Saved", reply, name)
            self.assertIn(gw.label(name), reply)
            self.assertNotIn(KEYS[name], reply)
        models = {m.split("/")[0] for m, _ in seen}
        self.assertTrue({"openrouter", "xai", "deepseek", "groq", "anthropic"} <= models)
        self.assertEqual(len(gw.list_api_keys(self.chat)), 5)

    def test_mistral_needs_the_provider_word_and_works_with_it(self):
        key = "abcdefghijklmnopqrstuvwxyz123456"

        async def ok(**kw):
            return ok_response()

        self.assertIn("couldn't find a supported API key", self.say(f"/setkey {key}"))
        with patch.object(gw.litellm, "acompletion", ok):
            reply = self.say(f"/setkey mistral {key}")
        self.assertIn("Mistral", reply)
        self.assertIn("works. Saved", reply)
        self.assertEqual(gw.get_user_api_keys(self.chat), [("mistral", key)])

    def test_usage_text_lists_the_options(self):
        usage = self.say("/setkey")
        for word in ("aistudio.google.com/apikey", "OpenRouter", "DeepSeek", "/setkey mistral"):
            self.assertIn(word, usage)

    def test_server_keys_for_new_providers_are_used(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEYS": "ds-1,ds-2", "MISTRAL_API_KEY": "mi-1"}):
            order = [(c.provider, c.api_key) for c in gw.build_candidates(self.chat)]
        self.assertIn(("deepseek", "ds-1"), order)
        self.assertIn(("deepseek", "ds-2"), order)
        self.assertIn(("mistral", "mi-1"), order)
        with patch.dict(os.environ, {"XAI_API_KEYS": "x-1"}):
            self.assertTrue(gw.has_server_keys())


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        gw.reset_rotation_state()
        self.chat = f"onb-{next(_ids)}"

    def say(self, text):
        return run(bot.handle_text(self.chat, text))

    def test_user_without_a_key_is_told_how_to_add_one(self):
        reply = self.say("hello there")
        self.assertIn("/setkey", reply)
        self.assertIn("aistudio.google.com/apikey", reply)

    def test_clear_messages_never_mention_keys(self):
        """With a working AI, a logged entry's reply does not nag about keys."""
        execute_tool("set_account_balance", user_id=self.chat, account_name="Bank", balance=5000)
        with FakeLLM(
            tool("log_expense", amount=450, category="Food & Dining", account_name="Bank")
        ):
            reply = self.say("Spent 450 on dinner via Bank")
        self.assertIn("450", reply)
        self.assertNotIn("setkey", reply)
        self.assertNotIn("AI key", reply)

    def test_free_text_without_a_key_is_not_logged(self):
        with FakeLLM(no_key=True):
            reply = self.say("Spent 450 on dinner via Bank")
        self.assertIn("/setkey", reply)

    def test_user_with_a_key_is_not_nagged(self):
        async def ok(**kw):
            return ok_response()

        with patch.object(gw.litellm, "acompletion", ok):
            self.say(f"/setkey {KEYS['gemini']}")
            reply = self.say("hello there")
        self.assertNotIn("aistudio.google.com", reply)

    def test_status_says_how_the_ai_works(self):
        status = self.say("/status")
        self.assertIn("offline", status)
        self.assertIn("/setkey", status)


if __name__ == "__main__":
    unittest.main()
