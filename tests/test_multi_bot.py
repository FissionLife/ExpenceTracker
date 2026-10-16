"""Several Telegram bots on one backend. Offline: Telegram is faked."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-multibot-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["TELEGRAM_BOT_TOKENS"] = ""
for _var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "OPENAI_API_KEYS", "GROQ_API_KEYS"):
    os.environ[_var] = ""

import asyncio  # noqa: E402
import hashlib  # noqa: E402
import hmac  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402
from urllib.parse import urlencode  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.bot import main as bot_main  # noqa: E402
from app.bot import telegram  # noqa: E402
from app.config import bot_tokens, dev_endpoints_enabled, settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.services import ledger  # noqa: E402
from app.web import tokens  # noqa: E402
from main import app  # noqa: E402

try:
    from tests.fake_llm import FakeLLM, tool  # noqa: E402
except ImportError:  # discover -s tests (no -t .) puts tests/ on sys.path directly
    from fake_llm import FakeLLM, tool  # noqa: E402

init_db()

ONE = "1111:AAA-first"
TWO = "2222:BBB-second"
THREE = "3333:CCC-third"


def configure(primary="", extra=""):
    return (
        patch.object(settings, "telegram_bot_token", primary),
        patch.object(settings, "telegram_bot_tokens", extra),
    )


class TokenListTests(unittest.TestCase):
    def test_primary_then_extras_without_duplicates_or_blanks(self):
        a, b = configure(ONE, f" {TWO} , {ONE},, {THREE} ")
        with a, b:
            self.assertEqual(bot_tokens(), [ONE, TWO, THREE])

    def test_extras_alone_are_enough(self):
        a, b = configure("", TWO)
        with a, b:
            self.assertEqual(bot_tokens(), [TWO])
            self.assertFalse(dev_endpoints_enabled())

    def test_no_tokens_means_dev_mode(self):
        a, b = configure("", "")
        with a, b:
            self.assertEqual(bot_tokens(), [])
            self.assertTrue(dev_endpoints_enabled())

    def test_bot_id_is_the_number_before_the_colon(self):
        self.assertEqual(telegram.bot_id(ONE), "1111")


class WebhookAddressTests(unittest.TestCase):
    def test_one_bot_keeps_the_plain_address(self):
        a, b = configure(ONE, "")
        with a, b, patch.object(settings, "webhook_url", "https://h.example/bot/webhook"):
            self.assertEqual(telegram.webhook_url_for(ONE), "https://h.example/bot/webhook")

    def test_several_bots_each_get_their_own_address(self):
        a, b = configure(ONE, TWO)
        with a, b, patch.object(settings, "webhook_url", "https://h.example/bot/webhook/"):
            self.assertEqual(telegram.webhook_url_for(ONE), "https://h.example/bot/webhook/1111")
            self.assertEqual(telegram.webhook_url_for(TWO), "https://h.example/bot/webhook/2222")

    def test_each_bot_has_its_own_secret_unless_one_is_set(self):
        with patch.object(settings, "telegram_webhook_secret", ""):
            self.assertNotEqual(telegram.webhook_secret(ONE), telegram.webhook_secret(TWO))
        with patch.object(settings, "telegram_webhook_secret", "shared"):
            self.assertEqual(telegram.webhook_secret(ONE), telegram.webhook_secret(TWO))


class WebhookRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def post(self, path, secret):
        return self.client.post(
            path, json={"update_id": 1}, headers={"X-Telegram-Bot-Api-Secret-Token": secret}
        )

    def test_each_bot_only_accepts_its_own_secret(self):
        a, b = configure(ONE, TWO)
        with (
            a,
            b,
            patch.object(settings, "telegram_webhook_secret", ""),
            patch.object(telegram, "webhook_active", return_value=True),
            patch.object(telegram, "feed_webhook_update", AsyncMock()) as feed,
        ):
            ok = self.post("/bot/webhook/2222", telegram.webhook_secret(TWO))
            wrong_bot = self.post("/bot/webhook/2222", telegram.webhook_secret(ONE))
            unknown = self.post("/bot/webhook/9999", telegram.webhook_secret(ONE))
            plain = self.post("/bot/webhook", telegram.webhook_secret(ONE))  # the first bot
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(wrong_bot.status_code, 403)
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(plain.status_code, 200)
        self.assertEqual([c.args[1] for c in feed.await_args_list], ["2222", None])


class FakeBot:
    """Stands in for aiogram's Bot: records what the server asks Telegram to do."""

    instances: list = []

    def __init__(self, token):
        self.token = token
        self.calls = []
        self.session = MagicMock(close=AsyncMock())
        FakeBot.instances.append(self)

    async def get_me(self):
        return MagicMock(username=f"bot{self.token.split(':')[0]}")

    async def set_webhook(self, url, **kw):
        self.calls.append(("set_webhook", url, kw["secret_token"]))

    async def delete_webhook(self, **kw):
        self.calls.append(("delete_webhook",))


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        FakeBot.instances = []

    def run_start(self, **settings_):
        a, b = configure(ONE, TWO)
        with (
            a,
            b,
            patch.object(telegram, "Bot", FakeBot),
            patch.object(telegram, "_publish_profile", AsyncMock()) as publish,
            patch.object(settings, "telegram_webhook_secret", ""),
            patch.object(settings, "use_webhook", settings_.get("webhook", False)),
            patch.object(settings, "webhook_url", "https://h.example/bot/webhook"),
            patch.object(telegram.Dispatcher, "start_polling", AsyncMock()) as polling,
        ):

            async def go():
                mode = await telegram.start()
                running = telegram.running_bots()
                await asyncio.sleep(0)
                await telegram.stop()
                return mode, running

            return (*asyncio.run(go()), publish, polling)

    def test_webhook_mode_registers_every_bot_at_its_own_address(self):
        mode, running, publish, _ = self.run_start(webhook=True)
        self.assertEqual((mode, running), ("webhook", 2))
        self.assertEqual(publish.await_count, 2)  # every bot gets the menu
        seen = {b.token: b.calls for b in FakeBot.instances}
        self.assertEqual(seen[ONE][0][:2], ("set_webhook", "https://h.example/bot/webhook/1111"))
        self.assertEqual(seen[TWO][0][:2], ("set_webhook", "https://h.example/bot/webhook/2222"))
        self.assertNotEqual(seen[ONE][0][2], seen[TWO][0][2])

    def test_polling_mode_serves_all_bots_from_one_dispatcher(self):
        mode, running, _, polling = self.run_start()
        self.assertEqual((mode, running), ("polling", 2))
        self.assertEqual(len(polling.await_args.args), 2)  # both bots passed to one start_polling
        for bot in FakeBot.instances:
            self.assertIn(("delete_webhook",), bot.calls)

    def test_stop_closes_every_bot_session(self):
        self.run_start()
        for bot in FakeBot.instances:
            bot.session.close.assert_awaited()
        self.assertEqual(telegram.running_bots(), 0)


class SharedBackendTests(unittest.TestCase):
    def test_a_person_sees_one_ledger_through_any_bot(self):
        chat = "777001"
        run = asyncio.run
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, chat).id
            ledger.record_transaction(db, uid, 1200, "Cash", "Seed", tx_type="income")
        with FakeLLM(
            tool("log_expense", amount=100, category="Food & Dining", account_name="Cash")
        ):
            run(bot_main.handle_text(chat, "Spent 100 on chai via Cash"))  # "through bot 1"
        reply = run(bot_main.handle_text(chat, "/balance"))  # "through bot 2": same chat id
        self.assertIn("₹1,100", reply)
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, chat).id
            self.assertEqual(ledger.recent_transactions(db, uid, 5)[1], 2)

    def test_different_people_stay_separate(self):
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, "777002").id
            ledger.record_transaction(db, uid, 1000, "Cash", "Seed", tx_type="income")
        with FakeLLM(tool("log_expense", amount=55, category="Food & Dining", account_name="Cash")):
            asyncio.run(bot_main.handle_text("777002", "Spent 55 on tea via Cash"))
        reply = asyncio.run(bot_main.handle_text("777003", "/history"))
        self.assertNotIn("55", reply)


class MiniAppSecondBotTests(unittest.TestCase):
    def init_data(self, bot_token, user_id):
        fields = {
            "auth_date": str(int(time.time())),
            "query_id": "AAH",
            "user": json.dumps({"id": user_id}),
        }
        check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
        secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        return urlencode(
            {**fields, "hash": hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()}
        )

    def test_sign_in_from_the_second_bot_works_and_a_stranger_bot_fails(self):
        client = TestClient(app)
        a, b = configure(ONE, TWO)
        with a, b:
            ok = client.post("/api/webapp/auth", json={"initData": self.init_data(TWO, 5150)})
            stranger = client.post(
                "/api/webapp/auth", json={"initData": self.init_data(THREE, 5150)}
            )
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(tokens.read_token(ok.json()["token"]), "5150")
        self.assertEqual(stranger.status_code, 401)


if __name__ == "__main__":
    unittest.main()
