"""Regression tests for the security and setup audit: settings parsing, webhook, sign-in, scripts. Offline."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-security-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
for _var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "OPENAI_API_KEYS", "GROQ_API_KEYS"):
    os.environ[_var] = ""

import hashlib  # noqa: E402
import hmac  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402
from urllib.parse import urlencode  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app import envfile  # noqa: E402
from app.bot import telegram  # noqa: E402
from app.config import ROOT, Settings, dev_endpoints_enabled, settings  # noqa: E402
from app.services_ai import transaction_parser as tp  # noqa: E402
from app.web import tokens  # noqa: E402
from main import app  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))

GOOD_TOKEN = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"


def cfg(**kw):
    return Settings(_env_file=None, database_url="", **kw)


class SettingsParsingTests(unittest.TestCase):
    def test_a_blank_secret_key_is_the_flagged_placeholder(self):
        self.assertTrue(envfile.is_weak_secret(cfg(secret_key="").secret_key))
        self.assertEqual(cfg(secret_key="").secret_key, envfile.INSECURE_DEV_KEY)

    def test_stray_spaces_do_not_break_booleans(self):
        self.assertTrue(cfg(use_webhook=" true ").use_webhook)

    def test_empty_environment_values_mean_not_set(self):
        with patch.dict(os.environ, {"USE_WEBHOOK": "", "PORT": ""}):
            self.assertFalse(Settings(_env_file=None).use_webhook)

    def test_a_relative_sqlite_path_is_next_to_the_project_not_the_shell(self):
        s = cfg(sqlite_path="data/x.db")
        self.assertEqual(s.database_url, f"sqlite:///{(ROOT / 'data/x.db').as_posix()}")

    def test_an_unknown_database_scheme_is_refused(self):
        with self.assertRaises(ValueError):
            Settings(_env_file=None, database_url="mysql://u:p@h/db")

    def test_bot_tokens_split_on_commas_semicolons_and_spaces(self):
        other = "654321:ZYX-WVU9876tsRqpo-nml43K2j1h098gf22"
        with (
            patch.object(settings, "telegram_bot_token", GOOD_TOKEN),
            patch.object(settings, "telegram_bot_tokens", f"{other}; {GOOD_TOKEN}"),
        ):
            from app.config import bot_tokens

            self.assertEqual(bot_tokens(), [GOOD_TOKEN, other])

    def test_the_simulator_is_off_on_a_public_https_address_even_without_a_token(self):
        with (
            patch.object(settings, "enable_dev_endpoints", None),
            patch.object(settings, "telegram_bot_token", ""),
            patch.object(settings, "telegram_bot_tokens", ""),
        ):
            with patch.object(settings, "public_url", ""):
                self.assertTrue(dev_endpoints_enabled())
            with patch.object(settings, "public_url", "https://bot.example.com"):
                self.assertFalse(dev_endpoints_enabled())


class EnvFileTests(unittest.TestCase):
    def test_inline_comments_are_not_part_of_the_value(self):
        path = Path(_TMP_DIR) / "inline.env"
        path.write_text('A=1 # note\nB="x # y"\nC=http://h/#frag\n', encoding="utf-8")
        got = envfile.read_env(path)
        self.assertEqual((got["A"], got["B"], got["C"]), ("1", "x # y", "http://h/#frag"))

    def test_the_backup_is_written_next_to_the_file(self):
        path = Path(_TMP_DIR) / "backup.env"
        path.write_text("A=1\n", encoding="utf-8")
        envfile.set_values(path, {"A": "2"})
        self.assertEqual((path.with_name("backup.env.bak")).read_text(encoding="utf-8"), "A=1\n")


class WebhookTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def post(self, secret, **kw):
        return self.client.post(
            "/bot/webhook", headers={"X-Telegram-Bot-Api-Secret-Token": secret}, **kw
        )

    def test_a_wrong_secret_is_refused_before_the_body_is_looked_at(self):
        with (
            patch.object(settings, "telegram_bot_token", GOOD_TOKEN),
            patch.object(telegram, "webhook_active", return_value=True),
            patch.object(telegram, "feed_webhook_update", AsyncMock()) as feed,
        ):
            r = self.post("nope", content=b"{ this is not json")
        self.assertEqual(r.status_code, 403)
        feed.assert_not_awaited()

    def test_a_non_ascii_secret_is_a_403_not_a_crash(self):
        with (
            patch.object(settings, "telegram_bot_token", GOOD_TOKEN),
            patch.object(telegram, "webhook_active", return_value=True),
        ):
            r = self.client.post(
                "/bot/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "café".encode("latin-1")},
                json={"update_id": 1},
            )
        self.assertEqual(r.status_code, 403)

    def test_a_good_secret_with_a_broken_body_is_a_400(self):
        with (
            patch.object(settings, "telegram_bot_token", GOOD_TOKEN),
            patch.object(telegram, "webhook_active", return_value=True),
        ):
            secret = telegram.webhook_secret(GOOD_TOKEN)
            broken = self.post(secret, content=b"{ nope")
            not_an_object = self.post(secret, content=b"[1, 2]")
        self.assertEqual((broken.status_code, not_an_object.status_code), (400, 400))


def signed(fields: dict, bot_token: str = GOOD_TOKEN) -> str:
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    return urlencode(
        {**fields, "hash": hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()}
    )


class MiniAppSignInTests(unittest.TestCase):
    def fields(self, **over):
        base = {"auth_date": str(int(time.time())), "user": json.dumps({"id": 42})}
        return {**base, **over}

    def test_a_genuine_fresh_sign_in_works(self):
        self.assertEqual(tokens.validate_init_data(signed(self.fields()), GOOD_TOKEN)["id"], 42)

    def test_a_sign_in_dated_in_the_future_is_refused(self):
        future = self.fields(auth_date=str(int(time.time()) + 3600))
        self.assertIsNone(tokens.validate_init_data(signed(future), GOOD_TOKEN))

    def test_a_user_that_is_not_an_object_with_a_numeric_id_is_refused(self):
        for user in ("[1]", "123", '{"id": "42"}', '{"id": true}', "{}", "not json"):
            self.assertIsNone(
                tokens.validate_init_data(signed(self.fields(user=user)), GOOD_TOKEN), user
            )

    def test_an_oversized_sign_in_is_refused_by_the_endpoint(self):
        r = TestClient(app).post("/api/webapp/auth", json={"initData": "a" * 9000})
        self.assertEqual(r.status_code, 422)


class DashboardHeadersTests(unittest.TestCase):
    def test_the_page_can_only_be_framed_by_telegram(self):
        from app.web.dashboard import SECURITY_HEADERS

        csp = SECURITY_HEADERS["Content-Security-Policy"]
        self.assertIn("frame-ancestors https://web.telegram.org", csp)
        self.assertNotIn("frame-ancestors *", csp)
        self.assertIn("base-uri 'none'", csp)
        self.assertIn("form-action 'none'", csp)


class StatementLimitsTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_huge_file_is_refused(self):
        with self.assertRaises(ValueError):
            await tp.parse_statement_file(b"x" * (tp.MAX_STATEMENT_BYTES + 1), "txt")

    async def test_absurdly_long_lines_are_skipped_not_parsed(self):
        junk = ("1/1/2026 " + "9" * 5000 + " Swiggy\n").encode()
        good = b"01/01/2026,Swiggy,250.00\n"
        rows = await tp.parse_statement_file(junk + good, "txt")
        self.assertLessEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
