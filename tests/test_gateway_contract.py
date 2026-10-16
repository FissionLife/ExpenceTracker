"""Contract tests for app/services_ai/ai_gateway.py (exact observable behaviour).

These pin down what the gateway does today so a refactor (e.g. splitting the module) can be
proven behaviour-preserving: provider tables, key ordering, rotation, vault ownership, key
validation statuses and the shape of completion calls.

Offline only - litellm.acompletion is mocked, no real LLM calls are made.
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

# Point the app at a throwaway SQLite DB *before* any app module is imported.
_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-test-")
os.environ["DYNAMIC_MODELS"] = "false"  # tests never hit provider model-list APIs
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")

import asyncio  # noqa: E402
import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

import litellm  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import SessionLocal, engine, init_db  # noqa: E402
from app.models import ApiKey, User  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402

_PROVIDER_NAMES = (
    "gemini",
    "openai",
    "anthropic",
    "groq",
    "openrouter",
    "xai",
    "deepseek",
    "mistral",
    "ollama",
)
# Every environment variable that can feed a key, a model override or a base URL into the gateway.
_ENV_VARS = tuple(
    sorted(
        {f"{p.upper()}_API_KEYS" for p in _PROVIDER_NAMES}
        | {f"{p.upper()}_API_KEY" for p in _PROVIDER_NAMES}
        | {f"AI_MODEL_{p.upper()}" for p in _PROVIDER_NAMES}
        | {"OLLAMA_API_BASE", "OPENAI_BASE_URL"}
    )
)

GEM = [f"AIza{c * 35}" for c in "DEFGH"]
GROQ = "gsk_" + "r" * 30
CLAUDE = "sk-ant-" + "c" * 30


def fake_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def rate_limit_error():
    return litellm.RateLimitError("429 quota exceeded", llm_provider="gemini", model="gemini")


def auth_error(message="invalid api key"):
    return litellm.AuthenticationError(message, llm_provider="gemini", model="gemini")


def run(coro):
    return asyncio.run(coro)


class ContractTestCase(unittest.TestCase):
    """Empty key table, fresh rotation state, no provider keys in env or settings."""

    def setUp(self):
        init_db()
        db = SessionLocal()
        db.query(ApiKey).delete()  # also clears the shared (SHARED_OWNER) store
        db.commit()
        db.close()
        gw.reset_rotation_state()

        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for var in _ENV_VARS:
            os.environ.pop(var, None)

        # settings may have been built (by another test module) while the developer's shell
        # still exported real keys: blank every attribute the gateway can fall back to.
        for var in _ENV_VARS:
            attr = var.lower()
            if hasattr(settings, attr):
                p = patch.object(settings, attr, "")
                p.start()
                self.addCleanup(p.stop)
        p = patch.object(settings, "dynamic_models", False)
        p.start()
        self.addCleanup(p.stop)

    @classmethod
    def tearDownClass(cls):
        engine.dispose()


# ----------------------------------------------------------------------------------------
# Provider tables and constants
# ----------------------------------------------------------------------------------------


class TestProviderTables(unittest.TestCase):
    def test_provider_aliases_snapshot(self):
        self.assertEqual(
            gw.PROVIDER_ALIASES,
            {
                "openrouter": "openrouter",
                "xai": "xai",
                "grok": "xai",
                "deepseek": "deepseek",
                "mistral": "mistral",
                "mistralai": "mistral",
                "google": "gemini",
                "gemini": "gemini",
                "openai": "openai",
                "gpt": "openai",
                "chatgpt": "openai",
                "anthropic": "anthropic",
                "claude": "anthropic",
                "groq": "groq",
                "ollama": "ollama",
            },
        )

    def test_single_key_env_snapshot(self):
        self.assertEqual(
            gw.SINGLE_KEY_ENV,
            {
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "groq": "GROQ_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "xai": "XAI_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "mistral": "MISTRAL_API_KEY",
            },
        )

    def test_providers_snapshot_and_key_order(self):
        self.assertEqual(list(gw.PROVIDERS), list(_PROVIDER_NAMES))
        self.assertEqual(
            gw.PROVIDERS,
            {
                "gemini": ("gemini/gemini-flash-latest", "GEMINI_API_KEYS"),
                "openai": ("openai/gpt-4o-mini", "OPENAI_API_KEYS"),
                "anthropic": ("anthropic/claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEYS"),
                "groq": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEYS"),
                "openrouter": ("openrouter/openai/gpt-4o-mini", "OPENROUTER_API_KEYS"),
                "xai": ("xai/grok-3-mini", "XAI_API_KEYS"),
                "deepseek": ("deepseek/deepseek-chat", "DEEPSEEK_API_KEYS"),
                "mistral": ("mistral/mistral-small-latest", "MISTRAL_API_KEYS"),
                "ollama": ("ollama/llama3.1", None),
            },
        )

    def test_every_alias_targets_a_provider_and_labels_cover_all(self):
        self.assertTrue(set(gw.PROVIDER_ALIASES.values()) <= set(gw.PROVIDERS))
        self.assertEqual(set(gw.PROVIDER_LABELS), set(gw.PROVIDERS))
        self.assertEqual(set(gw.SINGLE_KEY_ENV), set(gw.PROVIDERS) - {"ollama"})

    def test_constants(self):
        self.assertEqual(gw.DEFAULT_PROVIDER, "gemini")
        self.assertEqual(gw.SHARED_OWNER, "__shared__")
        self.assertEqual(gw.RATE_LIMIT_COOLDOWN_SECONDS, 60)
        self.assertEqual(gw.AUTH_FAILURE_COOLDOWN_SECONDS, 900)
        self.assertEqual(gw.DEFAULT_TIMEOUT_SECONDS, 15)
        self.assertEqual(gw.FAST_ATTEMPT_SECONDS, 8)

    def test_error_hierarchy(self):
        self.assertTrue(issubclass(gw.AIGatewayError, RuntimeError))
        self.assertTrue(issubclass(gw.NoAPIKeyError, gw.AIGatewayError))
        self.assertTrue(issubclass(gw.AllKeysExhaustedError, gw.AIGatewayError))
        err = gw.AllKeysExhaustedError(
            [{"provider": "gemini", "key": "AIza…DDDD", "model": "m", "error": "boom"}]
        )
        self.assertEqual(str(err), "All 1 key(s) failed. gemini AIza…DDDD: boom")
        self.assertEqual(err.attempts[0]["model"], "m")

    def test_keycheck_usable(self):
        self.assertTrue(gw.KeyCheck("valid").usable)
        self.assertTrue(gw.KeyCheck("rate_limited").usable)
        self.assertFalse(gw.KeyCheck("invalid").usable)
        self.assertFalse(gw.KeyCheck("unverified").usable)
        self.assertEqual(gw.KeyCheck("valid").detail, "")


# ----------------------------------------------------------------------------------------
# Small pure helpers
# ----------------------------------------------------------------------------------------


class TestParseAndMask(unittest.TestCase):
    def test_parse_provider_and_key_edges(self):
        k16 = "a" * 16
        self.assertEqual(gw.parse_provider_and_key(f"mistral {k16}"), ("mistral", k16))
        self.assertIsNone(gw.parse_provider_and_key("mistral " + "a" * 15))
        self.assertEqual(gw.parse_provider_and_key(f"  GPT   {k16}  extra"), ("openai", k16))
        self.assertEqual(gw.parse_provider_and_key(f"chatgpt\n{k16}"), ("openai", k16))
        self.assertEqual(gw.parse_provider_and_key(f"google {k16}"), ("gemini", k16))
        self.assertEqual(gw.parse_provider_and_key(f"ollama {k16}"), ("ollama", k16))
        self.assertIsNone(gw.parse_provider_and_key("mistral"))
        self.assertIsNone(gw.parse_provider_and_key(""))
        self.assertIsNone(gw.parse_provider_and_key(None))
        self.assertIsNone(gw.parse_provider_and_key(f"/setkey mistral {k16}"))
        self.assertIsNone(gw.parse_provider_and_key(f"skynet {k16}"))

    def test_mask_key_edges(self):
        self.assertEqual(gw.mask_key(""), "****")
        self.assertEqual(gw.mask_key("abc"), "****")
        self.assertEqual(gw.mask_key("a" * 10), "****")
        self.assertEqual(gw.mask_key("abcdefghijk"), "abcd…hijk")
        self.assertEqual(gw.mask_key("  abcdefghijk \n"), "abcd…hijk")
        self.assertEqual(gw.mask_key("   short   "), "****")


# ----------------------------------------------------------------------------------------
# Candidate ordering and rotation
# ----------------------------------------------------------------------------------------


class TestBuildCandidates(ContractTestCase):
    def pairs(self, user="chat-order", preferred="gemini"):
        return [(c.source, c.provider, c.api_key) for c in gw.build_candidates(user, preferred)]

    def test_no_keys_anywhere_gives_empty_list(self):
        self.assertEqual(gw.build_candidates("nobody"), [])
        self.assertEqual(gw.get_user_api_keys("nobody"), [])
        self.assertFalse(gw.has_server_keys())

    def test_full_order_user_shared_server_env(self):
        gw.add_api_key("chat-order", "gemini", GEM[0])
        gw.add_api_key("chat-order", "claude", CLAUDE)
        gw.add_api_key("chat-order", "groq", GROQ)
        gw.add_api_key(gw.SHARED_OWNER, "openai", "shared-openai-key-123")
        gw.add_api_key(gw.SHARED_OWNER, "gemini", GEM[1])
        os.environ.update(
            {
                "GROQ_API_KEYS": "gq1, gq2",
                "GROQ_API_KEY": "gq-single",
                "GEMINI_API_KEYS": "g1",
                "MISTRAL_API_KEY": "m1",
            }
        )
        self.assertEqual(
            self.pairs(preferred="groq"),
            [
                ("user", "groq", GROQ),
                ("user", "gemini", GEM[0]),
                ("user", "anthropic", CLAUDE),
                ("shared", "openai", "shared-openai-key-123"),
                ("shared", "gemini", GEM[1]),
                ("server", "groq", "gq1"),
                ("server", "groq", "gq2"),
                ("env", "groq", "gq-single"),
                ("server", "gemini", "g1"),
                ("env", "mistral", "m1"),
            ],
        )
        self.assertTrue(gw.has_server_keys())

    def test_preferred_provider_defaults_to_gemini(self):
        gw.add_api_key("chat-order", "groq", GROQ)
        gw.add_api_key("chat-order", "gemini", GEM[0])
        order = [(c.provider, c.api_key) for c in gw.build_candidates("chat-order")]
        self.assertEqual(order, [("gemini", GEM[0]), ("groq", GROQ)])

    def test_server_list_split_on_commas_semicolons_whitespace(self):
        os.environ["OPENAI_API_KEYS"] = " o1;o2 ,, o3\no4 "
        keys = [c.api_key for c in gw.build_candidates("x", "openai")]
        self.assertEqual(keys, ["o1", "o2", "o3", "o4"])

    def test_settings_used_when_env_var_absent(self):
        with patch.object(settings, "deepseek_api_keys", "ds1,ds2"):
            order = self.pairs(user="x")
        self.assertEqual(order, [("server", "deepseek", "ds1"), ("server", "deepseek", "ds2")])

    def test_dedupe_identical_provider_and_key_keeps_first_source(self):
        gw.add_api_key("chat-order", "gemini", GEM[0])
        gw.add_api_key(gw.SHARED_OWNER, "gemini", GEM[0])
        os.environ["GEMINI_API_KEYS"] = f"{GEM[0]},g2"
        os.environ["GEMINI_API_KEY"] = GEM[0]
        self.assertEqual(self.pairs(), [("user", "gemini", GEM[0]), ("server", "gemini", "g2")])

    def test_same_key_under_different_providers_is_not_deduped(self):
        os.environ["GEMINI_API_KEYS"] = "same-key"
        os.environ["OPENAI_API_KEYS"] = "same-key"
        self.assertEqual(
            self.pairs(user="x"),
            [("server", "gemini", "same-key"), ("server", "openai", "same-key")],
        )

    def test_inactive_user_and_shared_keys_are_skipped(self):
        own = gw.add_api_key("chat-order", "gemini", GEM[0])
        shared = gw.add_api_key(gw.SHARED_OWNER, "gemini", GEM[1])
        gw.deactivate_api_key("chat-order", own["key_id"])
        gw.deactivate_api_key(gw.SHARED_OWNER, shared["key_id"])
        self.assertEqual(self.pairs(), [])
        self.assertFalse(gw.has_server_keys())

    def test_has_server_keys_true_with_only_a_shared_key(self):
        gw.add_api_key(gw.SHARED_OWNER, "gemini", GEM[1])
        self.assertTrue(gw.has_server_keys())

    def test_get_user_api_keys_returns_provider_key_pairs(self):
        gw.add_api_key("chat-order", "groq", GROQ)
        os.environ["GEMINI_API_KEYS"] = "g1"
        self.assertEqual(gw.get_user_api_keys("chat-order"), [("groq", GROQ), ("gemini", "g1")])

    def test_candidates_carry_key_id_only_for_stored_keys(self):
        own = gw.add_api_key("chat-order", "gemini", GEM[0])
        os.environ["GEMINI_API_KEYS"] = "g1"
        user_c, server_c = gw.build_candidates("chat-order")
        self.assertEqual(user_c.key_id, own["key_id"])
        self.assertIsNone(server_c.key_id)
        self.assertNotIn(GEM[0], repr(user_c))  # the key is hidden from repr
        self.assertEqual(len(user_c.fingerprint), 16)

    def test_rotation_pools_advance_independently_per_call(self):
        gw.add_api_key("chat-rot", "gemini", GEM[0])
        gw.add_api_key("chat-rot", "gemini", GEM[1])
        os.environ["GEMINI_API_KEYS"] = "s1,s2,s3"
        os.environ["GEMINI_API_KEY"] = "single"
        seen = [[c.api_key for c in gw.build_candidates("chat-rot")] for _ in range(4)]
        self.assertEqual(
            seen,
            [
                [GEM[0], GEM[1], "s1", "s2", "s3", "single"],
                [GEM[1], GEM[0], "s2", "s3", "s1", "single"],
                [GEM[0], GEM[1], "s3", "s1", "s2", "single"],
                [GEM[1], GEM[0], "s1", "s2", "s3", "single"],
            ],
        )

    def test_user_pool_is_per_user_server_pool_is_global(self):
        gw.add_api_key("alice", "gemini", GEM[0])
        gw.add_api_key("alice", "gemini", GEM[1])
        gw.add_api_key("bob", "gemini", GEM[2])
        gw.add_api_key("bob", "gemini", GEM[3])
        os.environ["GEMINI_API_KEYS"] = "s1,s2"
        first_alice = [c.api_key for c in gw.build_candidates("alice")]
        first_bob = [c.api_key for c in gw.build_candidates("bob")]
        self.assertEqual(first_alice, [GEM[0], GEM[1], "s1", "s2"])
        # bob's own pool is fresh, but the server pool already moved on
        self.assertEqual(first_bob, [GEM[2], GEM[3], "s2", "s1"])

    def test_reset_rotation_state_restarts_cursors(self):
        os.environ["GEMINI_API_KEYS"] = "s1,s2"
        gw.build_candidates("x")
        gw.reset_rotation_state()
        self.assertEqual([c.api_key for c in gw.build_candidates("x")], ["s1", "s2"])

    def test_cooling_key_is_moved_last(self):
        os.environ["GEMINI_API_KEYS"] = "a1,b2,c3"
        mock = AsyncMock(side_effect=[rate_limit_error(), fake_response("ok")])
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(run(gw.complete("x", [{"role": "user", "content": "hi"}])), "ok")
        self.assertEqual([c.kwargs["api_key"] for c in mock.call_args_list], ["a1", "b2"])
        # rotation: call 2 starts at b2 -> b2,c3,a1 ; call 3 starts at c3 -> c3,a1,b2,
        # but a1 is cooling, so it is moved behind b2.
        self.assertEqual([c.api_key for c in gw.build_candidates("x")], ["b2", "c3", "a1"])
        self.assertEqual([c.api_key for c in gw.build_candidates("x")], ["c3", "b2", "a1"])
        gw.reset_rotation_state()  # also clears cooldowns
        self.assertEqual([c.api_key for c in gw.build_candidates("x")], ["a1", "b2", "c3"])


# ----------------------------------------------------------------------------------------
# Vault ownership
# ----------------------------------------------------------------------------------------


class TestVaultOwnership(ContractTestCase):
    def setUp(self):
        super().setUp()
        self.mine = gw.add_api_key("owner", "gemini", GEM[0])["key_id"]
        gw.add_api_key("intruder", "groq", GROQ)  # intruder exists as a user

    def test_other_users_cannot_touch_the_key(self):
        for who in ("intruder", "ghost-user"):
            with self.subTest(who=who):
                self.assertFalse(gw.activate_api_key(who, self.mine))
                self.assertFalse(gw.deactivate_api_key(who, self.mine))
                self.assertFalse(gw.delete_api_key(who, self.mine))
                self.assertFalse(gw.replace_api_key(who, self.mine, "groq", "x" * 30))
                self.assertIsNone(gw.get_plain_key(who, self.mine))
        # untouched
        self.assertEqual(gw.get_plain_key("owner", self.mine), ("gemini", GEM[0]))
        self.assertEqual(
            [(k["provider"], k["is_active"]) for k in gw.list_api_keys("owner")],
            [("gemini", True)],
        )

    def test_unknown_key_ids(self):
        for kid in ("no-such-id", ""):
            with self.subTest(kid=kid):
                self.assertFalse(gw.activate_api_key("owner", kid))
                self.assertFalse(gw.deactivate_api_key("owner", kid))
                self.assertFalse(gw.delete_api_key("owner", kid))
                self.assertFalse(gw.replace_api_key("owner", kid, "gemini", GEM[1]))
                self.assertIsNone(gw.get_plain_key("owner", kid))

    def test_deactivate_then_activate(self):
        self.assertTrue(gw.deactivate_api_key("owner", self.mine))
        self.assertFalse(gw.list_api_keys("owner")[0]["is_active"])
        self.assertEqual(gw.build_candidates("owner"), [])
        self.assertTrue(gw.activate_api_key("owner", self.mine))
        self.assertTrue(gw.list_api_keys("owner")[0]["is_active"])
        self.assertEqual([c.api_key for c in gw.build_candidates("owner")], [GEM[0]])

    def test_delete_removes_the_key(self):
        self.assertTrue(gw.delete_api_key("owner", self.mine))
        self.assertEqual(gw.list_api_keys("owner"), [])
        self.assertFalse(gw.delete_api_key("owner", self.mine))

    def test_replace_reactivates_and_changes_provider(self):
        gw.deactivate_api_key("owner", self.mine)
        self.assertTrue(gw.replace_api_key("owner", self.mine, "claude", f"  {CLAUDE}  "))
        listed = gw.list_api_keys("owner")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["key_id"], self.mine)
        self.assertEqual(listed[0]["provider"], "anthropic")
        self.assertTrue(listed[0]["is_active"])
        self.assertEqual(listed[0]["masked_key"], gw.mask_key(CLAUDE))
        self.assertEqual(gw.get_plain_key("owner", self.mine), ("anthropic", CLAUDE))

    def test_replace_with_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            gw.replace_api_key("owner", self.mine, "skynet", GEM[1])
        self.assertEqual(gw.get_plain_key("owner", self.mine), ("gemini", GEM[0]))

    def test_get_plain_key_round_trip_and_storage_is_encrypted(self):
        self.assertEqual(gw.get_plain_key("owner", self.mine), ("gemini", GEM[0]))
        db = SessionLocal()
        try:
            record = db.get(ApiKey, self.mine)
            self.assertNotIn(GEM[0], record.encrypted_key)
            self.assertEqual(gw.decrypt_api_key(record.encrypted_key), GEM[0])
        finally:
            db.close()

    def test_internal_user_uuid_is_accepted_like_chat_id(self):
        db = SessionLocal()
        try:
            internal_id = db.query(User).filter(User.telegram_chat_id == "owner").one().id
        finally:
            db.close()
        self.assertEqual(gw.get_plain_key(internal_id, self.mine), ("gemini", GEM[0]))
        self.assertEqual(len(gw.list_api_keys(internal_id)), 1)

    def test_add_existing_key_reactivates_and_reports_exists(self):
        gw.deactivate_api_key("owner", self.mine)
        result = gw.add_api_key("owner", "google", f" {GEM[0]} ")
        self.assertEqual(
            result,
            {
                "status": "exists",
                "key_id": self.mine,
                "provider": "gemini",
                "masked_key": gw.mask_key(GEM[0]),
            },
        )
        self.assertTrue(gw.list_api_keys("owner")[0]["is_active"])

    def test_add_empty_key_raises_and_unknown_user_lists_nothing(self):
        with self.assertRaises(ValueError):
            gw.add_api_key("owner", "gemini", "   ")
        self.assertEqual(gw.list_api_keys("ghost-user"), [])


# ----------------------------------------------------------------------------------------
# Live key validation (mocked)
# ----------------------------------------------------------------------------------------


class TestValidateApiKey(ContractTestCase):
    def test_valid_and_call_shape(self):
        mock = AsyncMock(return_value=fake_response("OK"))
        with patch.object(gw.litellm, "acompletion", mock):
            check = run(gw.validate_api_key("claude", f"  {CLAUDE} "))
        self.assertEqual(check, gw.KeyCheck("valid"))
        mock.assert_awaited_once()
        self.assertEqual(
            mock.call_args.kwargs,
            {
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                "api_key": CLAUDE,
                "max_tokens": 32,
                "timeout": 12,
                "num_retries": 0,
            },
        )

    def test_model_override_env_is_used(self):
        os.environ["AI_MODEL_GROQ"] = "groq/custom-model"
        mock = AsyncMock(return_value=fake_response("OK"))
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(run(gw.validate_api_key("groq", GROQ)).status, "valid")
        self.assertEqual(mock.call_args.kwargs["model"], "groq/custom-model")

    def test_rate_limited(self):
        mock = AsyncMock(side_effect=rate_limit_error())
        with patch.object(gw.litellm, "acompletion", mock):
            check = run(gw.validate_api_key("gemini", GEM[2]))
        self.assertEqual(
            check, gw.KeyCheck("rate_limited", "quota or rate limit reached right now")
        )
        self.assertTrue(check.usable)

    def test_invalid_auth_error(self):
        mock = AsyncMock(side_effect=auth_error())
        with patch.object(gw.litellm, "acompletion", mock):
            check = run(gw.validate_api_key("gemini", GEM[2]))
        self.assertEqual(check, gw.KeyCheck("invalid", "the provider rejected this key"))
        self.assertFalse(check.usable)

    def test_invalid_from_quota_free_bad_request_with_key_hint(self):
        bad = litellm.BadRequestError("API key not valid", model="g", llm_provider="gemini")
        with patch.object(gw.litellm, "acompletion", AsyncMock(side_effect=bad)):
            check = run(gw.validate_api_key("gemini", GEM[2]))
        self.assertEqual(check.status, "invalid")
        self.assertIn("not valid", check.detail)

    def test_empty_key_is_invalid_without_a_call(self):
        mock = AsyncMock()
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(
                run(gw.validate_api_key("gemini", "  ")), gw.KeyCheck("invalid", "empty key")
            )
        mock.assert_not_awaited()

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            run(gw.validate_api_key("skynet", "x" * 30))

    def test_unverified_on_unclassified_error(self):
        mock = AsyncMock(side_effect=ValueError("boom\nsecond line"))
        with patch.object(gw.litellm, "acompletion", mock):
            check = run(gw.validate_api_key("gemini", GEM[2]))
        self.assertEqual(check, gw.KeyCheck("unverified", "ValueError: boom"))

    def test_unverified_on_connection_error(self):
        err = litellm.APIConnectionError("network down", llm_provider="gemini", model="g")
        with patch.object(gw.litellm, "acompletion", AsyncMock(side_effect=err)):
            check = run(gw.validate_api_key("gemini", GEM[2]))
        self.assertEqual(check.status, "unverified")
        self.assertTrue(check.detail.startswith("APIConnectionError: "))


# ----------------------------------------------------------------------------------------
# complete(): call shape
# ----------------------------------------------------------------------------------------


MESSAGES = [{"role": "user", "content": "hi"}]


class TestCompleteShape(ContractTestCase):
    def test_no_tools_returns_stripped_text_and_call_kwargs(self):
        os.environ["GEMINI_API_KEYS"] = "gem-1"
        mock = AsyncMock(return_value=fake_response("  hello \n"))
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(run(gw.complete("x", MESSAGES)), "hello")
        self.assertEqual(
            mock.call_args.kwargs,
            {
                "model": "gemini/gemini-flash-latest",
                "messages": MESSAGES,
                "num_retries": 0,
                "timeout": 15,
                "api_key": "gem-1",
            },
        )

    def test_none_content_becomes_empty_string(self):
        # LiteLLM always returns a ModelResponse object; dict responses are not supported.
        os.environ["GEMINI_API_KEYS"] = "gem-1"
        with patch.object(gw.litellm, "acompletion", AsyncMock(return_value=fake_response(None))):
            self.assertEqual(run(gw.complete("x", MESSAGES)), "")

    def test_tools_return_full_response_with_tool_choice_auto(self):
        os.environ["GEMINI_API_KEYS"] = "gem-1"
        response = fake_response("  not stripped  ")
        tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]
        mock = AsyncMock(return_value=response)
        with patch.object(gw.litellm, "acompletion", mock):
            result = run(gw.complete("x", MESSAGES, tools=tools))
        self.assertIs(result, response)
        self.assertEqual(mock.call_args.kwargs["tools"], tools)
        self.assertEqual(mock.call_args.kwargs["tool_choice"], "auto")

    def test_caller_kwargs_pass_through_and_timeout_override(self):
        os.environ["GEMINI_API_KEYS"] = "gem-1"
        mock = AsyncMock(return_value=fake_response("ok"))
        with patch.object(gw.litellm, "acompletion", mock):
            run(gw.complete("x", MESSAGES, timeout=3, temperature=0.1))
        self.assertEqual(mock.call_args.kwargs["timeout"], 3)
        self.assertEqual(mock.call_args.kwargs["temperature"], 0.1)

    def test_explicit_model_preference(self):
        os.environ["GROQ_API_KEYS"] = "gq1"
        os.environ["GEMINI_API_KEYS"] = "gem-1"
        mock = AsyncMock(return_value=fake_response("ok"))
        with patch.object(gw.litellm, "acompletion", mock):
            run(gw.complete("x", MESSAGES, model_preference="groq/llama-3.1-8b-instant"))
        self.assertEqual(mock.call_args.kwargs["model"], "groq/llama-3.1-8b-instant")
        self.assertEqual(mock.call_args.kwargs["api_key"], "gq1")

    def test_ollama_uses_default_api_base_and_no_api_key(self):
        os.environ["GEMINI_API_KEYS"] = "gem-1"  # must be ignored on the ollama path
        mock = AsyncMock(return_value=fake_response("local"))
        with patch.object(gw.litellm, "acompletion", mock):
            self.assertEqual(run(gw.complete("x", MESSAGES, model_preference="ollama")), "local")
        mock.assert_awaited_once()
        kwargs = mock.call_args.kwargs
        self.assertNotIn("api_key", kwargs)
        self.assertEqual(kwargs["api_base"], "http://localhost:11434")
        self.assertEqual(kwargs["model"], "ollama/llama3.1")
        self.assertEqual(kwargs["timeout"], 15)

    def test_ollama_api_base_env_and_caller_override(self):
        os.environ["OLLAMA_API_BASE"] = "http://gpu-box:11434"
        mock = AsyncMock(return_value=fake_response("ok"))
        with patch.object(gw.litellm, "acompletion", mock):
            run(gw.complete("x", MESSAGES, model_preference="ollama"))
            self.assertEqual(mock.call_args.kwargs["api_base"], "http://gpu-box:11434")
            run(gw.complete("x", MESSAGES, model_preference="ollama", api_base="http://other"))
            self.assertEqual(mock.call_args.kwargs["api_base"], "http://other")

    def test_ollama_failure_raises_all_keys_exhausted_with_local_marker(self):
        err = litellm.APIConnectionError("refused", llm_provider="ollama", model="ollama/x")
        with (
            patch.object(gw.litellm, "acompletion", AsyncMock(side_effect=err)),
            self.assertRaises(gw.AllKeysExhaustedError) as ctx,
        ):
            run(gw.complete("x", MESSAGES, model_preference="ollama"))
        self.assertEqual(len(ctx.exception.attempts), 1)
        self.assertEqual(ctx.exception.attempts[0]["provider"], "ollama")
        self.assertEqual(ctx.exception.attempts[0]["key"], "local")

    def test_attempt_records_hold_masked_key_and_model(self):
        os.environ["GEMINI_API_KEYS"] = f"{GEM[3]}"
        with (
            patch.object(gw.litellm, "acompletion", AsyncMock(side_effect=rate_limit_error())),
            self.assertRaises(gw.AllKeysExhaustedError) as ctx,
        ):
            run(gw.complete("x", MESSAGES))
        attempt = ctx.exception.attempts[0]
        self.assertEqual(set(attempt), {"provider", "key", "model", "error"})
        self.assertEqual(attempt["key"], gw.mask_key(GEM[3]))
        self.assertEqual(attempt["model"], "gemini/gemini-flash-latest")
        self.assertTrue(attempt["error"].startswith("RateLimitError: "))

    def test_no_keys_raises_no_api_key_error(self):
        with (
            patch.object(gw.litellm, "acompletion", AsyncMock()) as mock,
            self.assertRaises(gw.NoAPIKeyError),
        ):
            run(gw.complete("x", MESSAGES))
        mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
