"""Regression tests: ledger input validation / category summary, and bot access control.

Offline only. Run with:  uv run python -m unittest discover -s tests -v
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-ledger-test-")
os.environ["DYNAMIC_MODELS"] = "false"  # tests never hit provider model-list APIs
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["USE_WEBHOOK"] = "false"
os.environ.pop("ENABLE_DEV_ENDPOINTS", None)
os.environ.pop("ALLOWED_CHAT_IDS", None)

import asyncio  # noqa: E402
import itertools  # noqa: E402
import unittest  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.bot import main as bot  # noqa: E402
from app.config import dev_endpoints_enabled, is_chat_allowed, settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.services import ledger  # noqa: E402
from main import app  # noqa: E402

init_db()
_ids = itertools.count(1)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.db = SessionLocal()
        self.user = ledger.get_or_create_user(self.db, f"ledger-{next(_ids)}")

    def tearDown(self):
        self.db.close()

    def total(self):
        return ledger.get_account_balances(self.db, self.user.id)["total_net_worth"]

    def test_invalid_amounts_rejected_and_nothing_changes(self):
        for bad in (0, -50, float("nan"), float("inf"), "abc", None):
            with self.assertRaises(ledger.LedgerError):
                ledger.record_transaction(self.db, self.user.id, bad, "Cash")
        self.assertEqual(self.total(), 0)
        # the session is still usable after the rejections (fund first; strict wallet blocks overdraw)
        ledger.record_transaction(self.db, self.user.id, 100, "Cash", tx_type="income")
        ledger.record_transaction(self.db, self.user.id, 10, "Cash")
        self.assertEqual(self.total(), 90)

    def test_invalid_type_and_same_account_transfer(self):
        with self.assertRaises(ledger.LedgerError):
            ledger.record_transaction(self.db, self.user.id, 5, "Cash", tx_type="transfer")
        with self.assertRaises(ledger.LedgerError):
            ledger.transfer_between_pipes(self.db, self.user.id, "Cash", " cash ", 5)
        with self.assertRaises(ledger.LedgerError):
            ledger.transfer_between_pipes(self.db, self.user.id, "Cash", "Bank", -5)
        self.assertEqual(self.total(), 0)

    def test_ledger_error_is_a_value_error(self):
        # callers (agent, MCP tools, bot) already catch ValueError
        self.assertTrue(issubclass(ledger.LedgerError, ValueError))

    def test_transfer_is_not_spending(self):
        ledger.record_transaction(self.db, self.user.id, 5000, "Bank", tx_type="income")
        ledger.transfer_between_pipes(self.db, self.user.id, "Bank", "Cash", 2000)
        self.assertEqual(self.total(), 5000)
        self.assertEqual(ledger.get_spending_summary(self.db, self.user.id)["total_expenses"], 0)

    def test_summary_reports_real_categories(self):
        ledger.record_transaction(self.db, self.user.id, 1000, "Cash", tx_type="income")  # fund it
        ledger.record_transaction(self.db, self.user.id, 100, "Cash", "Food", "Swiggy")
        ledger.record_transaction(self.db, self.user.id, 300, "Cash", "Shopping", "Amazon")
        ledger.record_transaction(self.db, self.user.id, 50, "Cash")
        summary = ledger.get_spending_summary(self.db, self.user.id)
        self.assertEqual(
            summary["category_breakdown"],
            {"Food": 100.0, "Shopping": 300.0, "Uncategorized": 50.0},
        )

    def test_users_are_isolated(self):
        other = ledger.get_or_create_user(self.db, f"ledger-{next(_ids)}")
        ledger.record_transaction(self.db, other.id, 1500, "Cash", tx_type="income")  # fund it
        ledger.record_transaction(self.db, other.id, 999, "Cash", "Food")
        self.assertEqual(ledger.get_spending_summary(self.db, self.user.id)["total_expenses"], 0)


class AccessControlTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self._saved = (
            settings.telegram_bot_token,
            settings.enable_dev_endpoints,
            settings.allowed_chat_ids,
        )

    def tearDown(self):
        (
            settings.telegram_bot_token,
            settings.enable_dev_endpoints,
            settings.allowed_chat_ids,
        ) = self._saved

    def test_dev_endpoints_default_to_token_less_only(self):
        settings.enable_dev_endpoints = None
        settings.telegram_bot_token = ""
        self.assertTrue(dev_endpoints_enabled())
        settings.telegram_bot_token = "123:abc"
        self.assertFalse(dev_endpoints_enabled())
        settings.enable_dev_endpoints = True
        self.assertTrue(dev_endpoints_enabled())

    def test_simulate_chat_closed_when_dev_endpoints_disabled(self):
        settings.enable_dev_endpoints = False
        body = {"chat_id": "1", "text": "/help"}
        self.assertEqual(self.client.post("/bot/simulate_chat", json=body).status_code, 404)
        upd = {"update_id": 1, "message": {"chat": {"id": 1}, "text": "/help"}}
        self.assertEqual(self.client.post("/bot/webhook", json=upd).status_code, 404)

    def test_allow_list(self):
        settings.allowed_chat_ids = ""
        self.assertTrue(is_chat_allowed("42"))
        settings.allowed_chat_ids = "1, 2"
        self.assertTrue(is_chat_allowed("2"))
        self.assertFalse(is_chat_allowed("99"))
        self.assertIn("private", asyncio.run(bot.handle_text("99", "/balance")).lower())
        self.assertIn("private", asyncio.run(bot.handle_document("99", "a.pdf", b"x")).lower())
        self.assertNotIn("private", asyncio.run(bot.handle_text("1", "/help")).lower())


if __name__ == "__main__":
    unittest.main()
