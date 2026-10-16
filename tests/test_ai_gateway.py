"""Tests for Role 3: AI gateway (key vault, rotation, fallback) and mode personas.

Offline only - litellm.acompletion is mocked, no real LLM calls are made.
Run with:  uv run python -m unittest discover -s tests -v
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

# Point the app at a throwaway SQLite DB *before* any app module is imported.
_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-test-")
os.environ["DYNAMIC_MODELS"] = "false"  # tests never hit provider model-list APIs
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
_KEY_VARS = (
    "GEMINI_API_KEYS",
    "OPENAI_API_KEYS",
    "ANTHROPIC_API_KEYS",
    "GROQ_API_KEYS",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "AI_MODEL_GEMINI",
)
for _var in _KEY_VARS:
    os.environ.pop(_var, None)

import asyncio  # noqa: E402
import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

import litellm  # noqa: E402

from app.db import SessionLocal, engine, init_db  # noqa: E402
from app.models import ApiKey  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402
from app.services_ai import prompts  # noqa: E402

GEMINI_KEYS = [f"AIza{c * 35}" for c in "ABC"]


def fake_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def rate_limit_error() -> Exception:
    return litellm.RateLimitError("429 quota exceeded", llm_provider="gemini", model="gemini")


def auth_error() -> Exception:
    return litellm.AuthenticationError("invalid api key", llm_provider="gemini", model="gemini")


def run(coro):
    return asyncio.run(coro)


class GatewayTestCase(unittest.TestCase):
    def setUp(self):
        init_db()
        db = SessionLocal()
        db.query(ApiKey).delete()
        db.commit()
        db.close()
        gw.reset_rotation_state()
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for var in _KEY_VARS:
            os.environ.pop(var, None)

    def tearDown(self):
        self.env.stop()

    @classmethod
    def tearDownClass(cls):
        engine.dispose()


class TestEncryptionAndVault(GatewayTestCase):
    def test_encrypt_roundtrip_and_ciphertext_hides_key(self):
        token = gw.encrypt_api_key(GEMINI_KEYS[0])
        self.assertNotIn(GEMINI_KEYS[0], token)
        self.assertEqual(gw.decrypt_api_key(token), GEMINI_KEYS[0])

    def test_compatible_with_mcp_save_user_api_key(self):
        from app.services_ai.mcp_server import save_user_api_key

        save_user_api_key("chat-mcp", "Gemini", GEMINI_KEYS[1])
        keys = gw._load_user_keys("chat-mcp")
        self.assertEqual([(k.provider, k.api_key) for k in keys], [("gemini", GEMINI_KEYS[1])])

    def test_add_list_dedupe_deactivate_delete(self):
        first = gw.add_api_key("chat-1", "google", GEMINI_KEYS[0])
        self.assertEqual(first["status"], "added")
        self.assertEqual(gw.add_api_key("chat-1", "gemini", GEMINI_KEYS[0])["status"], "exists")
        gw.add_api_key("chat-1", "claude", "sk-ant-" + "x" * 30)

        listed = gw.list_api_keys("chat-1")
        self.assertEqual(len(listed), 2)
        self.assertEqual({k["provider"] for k in listed}, {"gemini", "anthropic"})
        self.assertTrue(all("…" in k["masked_key"] for k in listed))
        self.assertNotIn(GEMINI_KEYS[0], str(listed))

        db = SessionLocal()
        stored = db.query(ApiKey).all()
        db.close()
        self.assertTrue(all(GEMINI_KEYS[0] not in r.encrypted_key for r in stored))

        self.assertTrue(gw.deactivate_api_key("chat-1", first["key_id"]))
        self.assertEqual([k.provider for k in gw._load_user_keys("chat-1")], ["anthropic"])
        self.assertFalse(gw.delete_api_key("someone-else", first["key_id"]))
        self.assertTrue(gw.delete_api_key("chat-1", first["key_id"]))
        self.assertEqual(len(gw.list_api_keys("chat-1")), 1)

    def test_add_keys_from_chat_text(self):
        text = f"here are my gemini keys {GEMINI_KEYS[0]}, {GEMINI_KEYS[1]} and gsk_{'g' * 30}"
        results = gw.add_api_keys_from_text("chat-2", text)
        self.assertEqual([r["provider"] for r in results], ["gemini", "gemini", "groq"])
        self.assertEqual(len(gw.list_api_keys("chat-2")), 3)

    def test_unknown_provider_rejected(self):
        with self.assertRaises(ValueError):
            gw.add_api_key("chat-1", "myspace", "x" * 30)


class TestRoutingHelpers(GatewayTestCase):
    def test_resolve_model(self):
        self.assertEqual(gw.resolve_model(None), ("gemini", "gemini/gemini-flash-latest"))
        self.assertEqual(gw.resolve_model("claude")[0], "anthropic")
        self.assertEqual(
            gw.resolve_model("groq/llama-3.1-8b-instant"), ("groq", "groq/llama-3.1-8b-instant")
        )
        self.assertEqual(gw.resolve_model("gpt-4o"), ("openai", "openai/gpt-4o"))
        with self.assertRaises(ValueError):
            gw.resolve_model("mystery-model")

    def test_detect_provider(self):
        self.assertEqual(gw.detect_provider(GEMINI_KEYS[0]), "gemini")
        self.assertEqual(gw.detect_provider("sk-ant-" + "a" * 30), "anthropic")
        self.assertEqual(gw.detect_provider("sk-" + "a" * 40), "openai")
        self.assertEqual(gw.detect_provider("gsk_" + "a" * 40), "groq")
        self.assertIsNone(gw.detect_provider("hello"))


class TestRotationAndFallback(GatewayTestCase):
    def _add_user_keys(self, chat_id="chat-rr"):
        for key in GEMINI_KEYS:
            gw.add_api_key(chat_id, "gemini", key)
        return chat_id

    def test_round_robin_across_consecutive_requests(self):
        chat_id = self._add_user_keys()
        mock = AsyncMock(return_value=fake_response("ok"))
        with patch.object(gw.litellm, "acompletion", mock):
            for _ in range(4):
                self.assertEqual(run(gw.ask_llm(chat_id, "sys", "hi")), "ok")
        used = [c.kwargs["api_key"] for c in mock.call_args_list]
        self.assertEqual(used, [GEMINI_KEYS[0], GEMINI_KEYS[1], GEMINI_KEYS[2], GEMINI_KEYS[0]])

    def test_fallback_on_rate_limit_then_cooldown(self):
        chat_id = self._add_user_keys()
        mock = AsyncMock(side_effect=[rate_limit_error(), fake_response("second key worked")])
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(run(gw.ask_llm(chat_id, "sys", "hi")), "second key worked")
        self.assertEqual([c.kwargs["api_key"] for c in mock.call_args_list], GEMINI_KEYS[:2])

        # The rate-limited key is cooling down and moves to the back of the queue.
        order = [c.api_key for c in gw.build_candidates(chat_id)]
        self.assertEqual(order[-1], GEMINI_KEYS[0])

    def test_fallback_on_auth_error_and_quota_bad_request(self):
        chat_id = self._add_user_keys()
        quota = litellm.BadRequestError(
            "RESOURCE_EXHAUSTED: quota", model="g", llm_provider="gemini"
        )
        mock = AsyncMock(side_effect=[auth_error(), quota, fake_response("third")])
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(run(gw.ask_llm(chat_id, "sys", "hi")), "third")
        self.assertEqual(mock.await_count, 3)

    def test_all_keys_exhausted(self):
        chat_id = self._add_user_keys()
        mock = AsyncMock(side_effect=[rate_limit_error() for _ in GEMINI_KEYS])
        with (
            patch.object(gw.litellm, "acompletion", mock),
            self.assertRaises(gw.AllKeysExhaustedError) as ctx,
        ):
            run(gw.ask_llm(chat_id, "sys", "hi"))
        self.assertEqual(len(ctx.exception.attempts), 3)
        self.assertNotIn(GEMINI_KEYS[0], str(ctx.exception))

    def test_non_key_errors_are_not_swallowed(self):
        chat_id = self._add_user_keys()
        bad = litellm.BadRequestError(
            "messages must not be empty", model="g", llm_provider="gemini"
        )
        mock = AsyncMock(side_effect=bad)
        with (
            patch.object(gw.litellm, "acompletion", mock),
            self.assertRaises(litellm.BadRequestError),
        ):
            run(gw.ask_llm(chat_id, "sys", "hi"))
        self.assertEqual(mock.await_count, 1)

    def test_cross_provider_fallback_uses_provider_default_model(self):
        chat_id = "chat-multi"
        gw.add_api_key(chat_id, "gemini", GEMINI_KEYS[0])
        gw.add_api_key(chat_id, "groq", "gsk_" + "q" * 30)
        mock = AsyncMock(side_effect=[rate_limit_error(), fake_response("from groq")])
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(run(gw.ask_llm(chat_id, "sys", "hi")), "from groq")
        models = [c.kwargs["model"] for c in mock.call_args_list]
        self.assertEqual(models, ["gemini/gemini-flash-latest", "groq/llama-3.3-70b-versatile"])

    def test_server_keys_used_when_user_has_none(self):
        os.environ["GEMINI_API_KEYS"] = "server-key-1, server-key-2"
        mock = AsyncMock(return_value=fake_response("ok"))
        with patch.object(gw.litellm, "acompletion", mock):
            run(gw.ask_llm("brand-new-chat", "sys", "hi"))
            run(gw.ask_llm("brand-new-chat", "sys", "hi"))
        self.assertEqual(
            [c.kwargs["api_key"] for c in mock.call_args_list], ["server-key-1", "server-key-2"]
        )

    def test_server_key_list_beats_unrelated_single_env_keys(self):
        # Regression: a stray OPENAI_API_KEY in the shell must not be tried before Gemini keys.
        os.environ["OPENAI_API_KEY"] = "sk-unrelated-company-key"
        os.environ["GEMINI_API_KEYS"] = "gem-1,gem-2"
        os.environ["GEMINI_API_KEY"] = "gem-single"
        order = [(c.provider, c.api_key) for c in gw.build_candidates("fresh-chat")]
        self.assertEqual(
            order[:3], [("gemini", "gem-1"), ("gemini", "gem-2"), ("gemini", "gem-single")]
        )
        self.assertEqual(order[-1], ("openai", "sk-unrelated-company-key"))

    def test_single_env_key_not_duplicated(self):
        os.environ["GEMINI_API_KEYS"] = "gem-1"
        os.environ["GEMINI_API_KEY"] = "gem-1"
        keys = [c.api_key for c in gw.build_candidates("fresh-chat")]
        self.assertEqual(keys.count("gem-1"), 1)

    def test_single_env_key_alone_works(self):
        os.environ["GEMINI_API_KEY"] = "gem-only"
        mock = AsyncMock(return_value=fake_response("ok"))
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(run(gw.ask_llm("fresh-chat", "sys", "hi")), "ok")
        self.assertEqual(mock.call_args.kwargs["api_key"], "gem-only")

    def test_access_denied_moves_to_next_key(self):
        # Regression: gateway proxies reply "Access denied: model ... not found or no access".
        os.environ["GEMINI_API_KEYS"] = "gem-1,gem-2"
        denied = litellm.BadRequestError(
            "OpenAIException - Access denied: model 'x' not found or no access.",
            model="g",
            llm_provider="gemini",
        )
        mock = AsyncMock(side_effect=[denied, fake_response("second worked")])
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(run(gw.ask_llm("fresh-chat", "sys", "hi")), "second worked")
        self.assertEqual([c.kwargs["api_key"] for c in mock.call_args_list], ["gem-1", "gem-2"])

    def test_tools_return_full_response_and_history_is_sent(self):
        os.environ["GEMINI_API_KEYS"] = "gem-1"
        response = fake_response("tool reply")
        mock = AsyncMock(return_value=response)
        tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]
        history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
        with patch.object(gw.litellm, "acompletion", mock):
            result = run(
                gw.ask_llm("fresh-chat", "sys", "now", tools=tools, messages_history=history)
            )
        self.assertIs(result, response)
        self.assertEqual(mock.call_args.kwargs["tools"], tools)
        self.assertEqual(
            [m["content"] for m in mock.call_args.kwargs["messages"]],
            ["sys", "earlier", "ok", "now"],
        )

    def test_no_keys_raises_helpful_error(self):
        with self.assertRaises(gw.NoAPIKeyError):
            run(gw.ask_llm("nobody", "sys", "hi"))

    def test_messages_shape_and_system_prompt(self):
        chat_id = self._add_user_keys()
        mock = AsyncMock(return_value=fake_response("  trimmed  "))
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(run(gw.ask_llm(chat_id, "be nice", "hello")), "trimmed")
        self.assertEqual(
            mock.call_args.kwargs["messages"],
            [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_ask_persona_uses_detected_mode(self):
        chat_id = self._add_user_keys()
        mock = AsyncMock(return_value=fake_response("cut it"))
        with patch.object(gw.litellm, "acompletion", mock):
            run(gw.ask_persona(chat_id, "I overspent again, be ruthless", context={"food": 9000}))
        system = mock.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Ruthless Budgeter", system)
        self.assertIn('"food": 9000', system)


class TestPrompts(unittest.TestCase):
    def test_three_personas_exist(self):
        self.assertEqual({m.value for m in prompts.PERSONAS}, {"coach", "budgeter", "summary"})
        self.assertEqual(len(prompts.list_modes()), 3)

    def test_detect_mode(self):
        cases = {
            "How can I reduce my expenses?": prompts.Mode.BUDGETER,
            "I overspent on Swiggy again": prompts.Mode.BUDGETER,
            "Give me a quick summary of this month": prompts.Mode.SUMMARY,
            "where did my money go": prompts.Mode.SUMMARY,
            "Any tips to build an emergency fund?": prompts.Mode.COACH,
            "/summary": prompts.Mode.SUMMARY,
            "/ruthless how am I doing": prompts.Mode.BUDGETER,
            "hello there": prompts.Mode.COACH,
            "": prompts.Mode.COACH,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(prompts.detect_mode(text), expected)

    def test_resolve_mode_aliases(self):
        self.assertEqual(prompts.resolve_mode("Ruthless Budgeter"), prompts.Mode.BUDGETER)
        self.assertEqual(prompts.resolve_mode("quick-summary"), prompts.Mode.SUMMARY)
        self.assertEqual(prompts.resolve_mode("/coach"), prompts.Mode.COACH)
        self.assertEqual(prompts.resolve_mode(None), prompts.Mode.COACH)
        with self.assertRaises(ValueError):
            prompts.resolve_mode("pirate")

    def test_build_system_prompt(self):
        text = prompts.build_system_prompt("summary", context={"total_expenses": 1200})
        self.assertIn("Tank & Pipes", text)
        self.assertIn("Quick Summary", text)
        self.assertIn('"total_expenses": 1200', text)
        self.assertNotIn("financial data", prompts.build_system_prompt("coach"))

    def test_personas_carry_no_example_rupee_amounts(self):
        # the eval found the Coach quoting savings targets that are not in the user's data; its own
        # example ("saves ~₹1,600/month") invited that. Examples may use shares and percentages only.
        for mode, persona in prompts.PERSONAS.items():
            with self.subTest(mode=mode.value):
                self.assertNotRegex(persona.prompt, r"₹\s*\d", persona.name)


if __name__ == "__main__":
    unittest.main()
