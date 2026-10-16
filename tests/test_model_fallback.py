"""Per-key model fallback: retired/overloaded models are skipped, the working one is remembered,
and a genuine key is never rejected just because Gemini is busy. Offline (litellm is mocked)."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-fallback-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "true"
for _var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "AI_MODEL_GEMINI"):
    os.environ[_var] = ""

import asyncio  # noqa: E402
import itertools  # noqa: E402
import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

import litellm  # noqa: E402

from app.db import init_db  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402
from app.services_ai import model_catalog as mc  # noqa: E402

init_db()
_ids = itertools.count(1)

# The model list a real new-format (AQ.) Gemini key returns, trimmed.
AQ_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]
KEY1 = "AQ.Ab8RN6" + "a" * 30
KEY2 = "AIza" + "B" * 35


def run(coro):
    return asyncio.run(coro)


def ok(text="hi"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def overloaded():
    return litellm.ServiceUnavailableError("503 high demand", llm_provider="gemini", model="m")


def retired():
    return litellm.NotFoundError(
        "404 This model is no longer available to new users", llm_provider="gemini", model="m"
    )


def rate_limited():
    return litellm.RateLimitError("429 quota", llm_provider="gemini", model="m")


class Base(unittest.TestCase):
    def setUp(self):
        mc.clear_cache()
        gw.reset_rotation_state()
        self.chat = f"fallback-{next(_ids)}"
        for patcher in (
            patch.object(gw.settings, "ai_model_gemini", ""),
            patch.object(gw.settings, "dynamic_models", True),
            patch.dict(mc._FETCHERS, {"gemini": AsyncMock(return_value=AQ_MODELS)}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def add_keys(self, *keys):
        for k in keys:
            gw.add_api_key(self.chat, "gemini", k)

    def ask(self, behaviour):
        """behaviour(model, api_key) -> response or raises; returns (text, calls)."""
        calls = []

        async def fake(**kw):
            calls.append((kw["model"].split("/", 1)[1], kw["api_key"]))
            return behaviour(*calls[-1])

        with patch.object(gw.litellm, "acompletion", fake):
            text = run(gw.ask_llm(self.chat, "sys", "hello"))
        return text, calls


class RankingTests(unittest.TestCase):
    def test_ranking_for_a_real_aq_key(self):
        ranked = mc.rank_models("gemini", AQ_MODELS)
        self.assertEqual(ranked[0], "gemini-flash-latest")
        self.assertIn("gemini-flash-lite-latest", ranked)
        self.assertLessEqual(len(ranked), 5)
        self.assertNotIn("gemini-2.5-pro", ranked)  # pro models are not part of the default chain

    def test_remembered_model_goes_first_and_can_be_forgotten(self):
        mc.clear_cache()
        with patch.dict(mc._FETCHERS, {"gemini": AsyncMock(return_value=AQ_MODELS)}):
            mc.remember_working("gemini", KEY1, "gemini-flash-lite-latest")
            chain = run(mc.model_chain("gemini", KEY1))
            self.assertEqual(chain[0], "gemini/gemini-flash-lite-latest")
            self.assertEqual(len(chain), len(set(chain)))
            mc.forget_working("gemini", KEY1)
            self.assertEqual(run(mc.model_chain("gemini", KEY1))[0], "gemini/gemini-flash-latest")

    def test_other_providers_have_chains(self):
        self.assertTrue(mc.rank_models("openai", ["gpt-4o-mini", "gpt-5-mini", "gpt-4o"]))
        self.assertTrue(mc.rank_models("anthropic", ["claude-sonnet-4-5-20250929"]))
        self.assertTrue(mc.rank_models("groq", ["llama-3.3-70b-versatile"]))
        self.assertEqual(mc.rank_models("gemini", []), [])


class CompletionFallbackTests(Base):
    def test_overloaded_then_retired_then_working_model(self):
        self.add_keys(KEY1)

        def behaviour(model, key):
            if model == "gemini-flash-latest":
                raise overloaded()
            if model == "gemini-3.8-flash":
                raise retired()
            return ok("answered")

        text, calls = self.ask(behaviour)
        self.assertEqual(text, "answered")
        self.assertEqual(
            [m for m, _ in calls][:3],
            ["gemini-flash-latest", "gemini-3.8-flash", "gemini-3.7-flash"],
        )
        self.assertTrue(all(k == KEY1 for _, k in calls))  # stayed on the same key

    def test_working_model_is_remembered_and_tried_first_next_time(self):
        self.add_keys(KEY1)

        def behaviour(model, key):
            if model != "gemini-flash-lite-latest":
                raise overloaded()
            return ok()

        self.ask(behaviour)
        _, calls = self.ask(lambda model, key: ok())
        self.assertEqual(calls[0][0], "gemini-flash-lite-latest")

    def test_all_models_busy_moves_to_the_next_key(self):
        self.add_keys(KEY1, KEY2)

        def behaviour(model, key):
            if key == KEY1:
                raise overloaded()
            return ok("second key")

        text, calls = self.ask(behaviour)
        self.assertEqual(text, "second key")
        self.assertEqual({k for _, k in calls[:-1]}, {KEY1})
        self.assertEqual(calls[-1][1], KEY2)

    def test_rate_limit_skips_straight_to_the_next_key(self):
        self.add_keys(KEY1, KEY2)

        def behaviour(model, key):
            if key == KEY1:
                raise rate_limited()
            return ok("fine")

        text, calls = self.ask(behaviour)
        self.assertEqual(text, "fine")
        self.assertEqual([k for _, k in calls], [KEY1, KEY2])  # one try on key 1, no model hopping

    def test_pinned_model_is_never_replaced(self):
        self.add_keys(KEY1)
        seen = []

        async def fake(**kw):
            seen.append(kw["model"])
            raise overloaded()

        with (
            patch.object(gw.litellm, "acompletion", fake),
            self.assertRaises(gw.AllKeysExhaustedError),
        ):
            run(gw.ask_llm(self.chat, "sys", "hi", model_preference="gemini/gemini-2.5-pro"))
        self.assertEqual(seen, ["gemini/gemini-2.5-pro"])


class ValidationTests(Base):
    def check(self, behaviour, listing=AQ_MODELS):
        calls = []

        async def fake(**kw):
            calls.append(kw["model"])
            return behaviour(kw["model"])

        mc.clear_cache()
        with (
            patch.object(gw.litellm, "acompletion", fake),
            patch.dict(mc._FETCHERS, {"gemini": AsyncMock(return_value=listing)}),
        ):
            return run(gw.validate_api_key("gemini", KEY1)), calls

    def test_works_on_a_later_model(self):
        def behaviour(model):
            if "flash-latest" in model and "lite" not in model:
                raise overloaded()
            return ok()

        result, calls = self.check(behaviour)
        self.assertEqual(result.status, "valid")
        self.assertGreaterEqual(len(calls), 2)

    def test_genuine_key_is_accepted_when_gemini_is_busy(self):
        def busy(model):
            raise overloaded()

        result, calls = self.check(busy)
        self.assertEqual(result.status, "valid")
        self.assertIn("busy", result.detail)
        self.assertLessEqual(len(calls), 3)

    def test_unverified_when_busy_and_the_key_cannot_even_list_models(self):
        def busy(model):
            raise overloaded()

        result, _ = self.check(busy, listing=[])
        self.assertEqual(result.status, "unverified")

    def test_rejected_key_stops_immediately(self):
        def bad(model):
            raise litellm.AuthenticationError("API key not valid", llm_provider="gemini", model="m")

        result, calls = self.check(bad)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(len(calls), 1)

    def test_model_not_found_is_not_a_bad_key(self):
        # "model not found" also matches the auth hints; a retired first model must not get a
        # valid key rejected. Both error classes litellm uses for it are covered.
        for error_cls in (litellm.BadRequestError, litellm.NotFoundError):
            with self.subTest(error=error_cls.__name__):
                first = []

                def behaviour(model, error_cls=error_cls, first=first):
                    if not first:
                        first.append(model)
                        raise error_cls(
                            "400 model not found or no access", llm_provider="gemini", model=model
                        )
                    return ok()

                result, calls = self.check(behaviour)
                self.assertEqual(result.status, "valid")
                self.assertEqual(len(calls), 2)

    def test_auth_error_mentioning_model_is_still_invalid(self):
        def bad(model):
            raise litellm.AuthenticationError(
                "API key not valid: model not found", llm_provider="gemini", model="m"
            )

        result, calls = self.check(bad)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(len(calls), 1)

    def test_rate_limited_key_is_usable(self):
        def limited(model):
            raise rate_limited()

        result, _ = self.check(limited)
        self.assertEqual(result.status, "rate_limited")
        self.assertTrue(result.usable)


if __name__ == "__main__":
    unittest.main()
