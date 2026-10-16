"""Balance overdraw guard and budget tracking/overage behavior. Offline.

Covers the fixes for: no account can go negative on a live spend or transfer (strict wallet);
budgets attach to the canonical category and track spend; and an overspend of a budget asks
the user first. Chat tests script the AI agent with FakeLLM.
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-guard-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""

import asyncio  # noqa: E402
import itertools  # noqa: E402
import unittest  # noqa: E402

from sqlalchemy import select  # noqa: E402

from app.bot import agent  # noqa: E402
from app.bot import main as bot_main  # noqa: E402
from app.bot.categories import canonical_category  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.models import Account  # noqa: E402
from app.services import ledger  # noqa: E402
from app.services_ai.mcp_server import execute_tool  # noqa: E402

try:
    from tests.fake_llm import FakeLLM, tool  # noqa: E402
except ImportError:  # run as: unittest discover -s tests (no top-level dir)
    from fake_llm import FakeLLM, tool  # noqa: E402

init_db()
_ids = itertools.count(1)


def new_user():
    chat = f"guard-{next(_ids)}"
    with SessionLocal() as db:
        return chat, ledger.get_or_create_user(db, chat).id


def fund(chat, account, amount):
    """Give an account a real opening balance via income (so it counts as 'funded')."""
    execute_tool("log_income", user_id=chat, amount=amount, source_account=account)


def balance_of(uid, name):
    with SessionLocal() as db:
        acc = db.scalar(select(Account).where(Account.user_id == uid, Account.name == name))
        return acc.balance if acc else None


class BlockedSpendButtonsTests(unittest.TestCase):
    def test_use_other_pipe_button_appears_when_an_alt_can_cover(self):
        chat, uid = new_user()
        fund(chat, "Bank", 8500)
        # Cash is 0, Bank has enough: "420 paid via cash" should offer Bank
        with FakeLLM() as llm:
            llm.add(tool("log_expense", amount=420, category="Shopping", account_name="Cash"))
            reply = asyncio.run(bot_main.handle_incoming(chat, "paid 420 via cash"))
        self.assertEqual(balance_of(uid, "Cash"), 0)
        self.assertEqual(balance_of(uid, "Bank"), 8500)
        # error text + two buttons on the alt row (switch + cancel)
        self.assertIn("not enough balance", reply.text.lower())
        flat = [b for row in reply.rows for b in row]
        datas = [b.data for b in flat]
        self.assertTrue(any(d and d.startswith("pay:") for d in datas), datas)
        self.assertIn("go:home", datas)  # Cancel

    def test_cancel_only_when_no_other_pipe_has_funds(self):
        chat, uid = new_user()
        # No pipe funded at all: just a Cancel button, no pay:
        with FakeLLM() as llm:
            llm.add(tool("log_expense", amount=420, category="Shopping", account_name="Cash"))
            reply = asyncio.run(bot_main.handle_incoming(chat, "paid 420 via cash"))
        self.assertIn("not enough balance", reply.text.lower())
        flat = [b for row in reply.rows for b in row]
        datas = [b.data for b in flat]
        self.assertIn("go:home", datas)
        self.assertFalse(any(d and d.startswith("pay:") for d in datas), datas)


class OverdrawGuardTests(unittest.TestCase):
    def test_funded_account_cannot_go_negative_on_a_live_spend(self):
        chat, uid = new_user()
        fund(chat, "Cash", 500)
        with self.assertRaises(Exception) as ctx:
            execute_tool(
                "log_expense", user_id=chat, amount=700, category="Food", account_name="Cash"
            )
        self.assertIn("not enough balance", str(ctx.exception).lower())
        self.assertEqual(balance_of(uid, "Cash"), 500)  # nothing deducted

    def test_drained_account_blocks_the_next_spend(self):
        # Kevin's case: funded, drained to 0, then overspent.
        chat, uid = new_user()
        fund(chat, "Cash", 1000)
        execute_tool("log_expense", user_id=chat, amount=1000, category="Food", account_name="Cash")
        self.assertEqual(balance_of(uid, "Cash"), 0)
        with self.assertRaises(Exception) as ctx:
            execute_tool(
                "log_expense", user_id=chat, amount=2000, category="Food", account_name="Cash"
            )
        self.assertIn("not enough balance", str(ctx.exception).lower())

    def test_a_never_funded_pipe_also_cannot_go_negative(self):
        # Strict wallet: every pipe is protected. A brand-new ₹0 Cash account can't be spent from.
        chat, uid = new_user()
        with self.assertRaises(Exception) as ctx:
            execute_tool(
                "log_expense", user_id=chat, amount=300, category="Food", account_name="Cash"
            )
        self.assertIn("not enough balance", str(ctx.exception).lower())
        self.assertEqual(balance_of(uid, "Cash"), 0)  # nothing deducted

    def test_transfer_out_cannot_overdraw_a_funded_source(self):
        chat, uid = new_user()
        fund(chat, "Bank", 100)
        with self.assertRaises(Exception) as ctx:
            execute_tool(
                "transfer_funds", user_id=chat, from_account="Bank", to_account="Cash", amount=500
            )
        self.assertIn("not enough balance", str(ctx.exception).lower())

    def test_a_spend_within_balance_is_allowed(self):
        chat, uid = new_user()
        fund(chat, "Cash", 500)
        execute_tool("log_expense", user_id=chat, amount=200, category="Food", account_name="Cash")
        self.assertEqual(balance_of(uid, "Cash"), 300)


class BudgetTrackingTests(unittest.TestCase):
    def test_budget_sets_on_canonical_category_and_tracks_spend(self):
        chat, uid = new_user()
        fund(chat, "Bank", 5000)
        res = execute_tool(
            "set_category_budget", user_id=chat, category="food", monthly_budget=1000
        )
        self.assertEqual(res["category"], "Food & Dining")
        # a spend on "food" counts against the same canonical category
        execute_tool("log_expense", user_id=chat, amount=400, category="food", account_name="Bank")
        from app.services import analytics

        with SessionLocal() as db:
            rows = analytics.budget_status(db, uid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category"], "Food & Dining")
        self.assertEqual(rows[0]["spent"], 400)
        self.assertEqual(rows[0]["left"], 600)

    def test_meals_maps_to_food_and_dining(self):
        self.assertEqual(canonical_category("meals"), "Food & Dining")
        self.assertEqual(canonical_category("food"), "Food & Dining")


class BudgetOverageConfirmTests(unittest.TestCase):
    def setUp(self):
        self.chat, self.uid = new_user()
        fund(self.chat, "Bank", 50000)
        execute_tool("set_category_budget", user_id=self.chat, category="food", monthly_budget=1000)

    def ask(self, amount):
        with FakeLLM() as llm:
            llm.add(tool("log_expense", amount=amount, category="meals", account_name="Bank"))
            return asyncio.run(
                agent.process_user_interaction_ex(self.chat, f"I ate {amount} for meals")
            )

    def test_overspending_a_budget_asks_first(self):
        execute_tool(
            "log_expense", user_id=self.chat, amount=700, category="food", account_name="Bank"
        )
        # An expense that pushes Food over 1000 should require confirmation, not run immediately.
        outcome = self.ask(600)
        self.assertTrue(outcome.needs_confirmation)
        self.assertIn("budget", outcome.text.lower())
        self.assertEqual(balance_of(self.uid, "Bank"), 49300)  # nothing deducted yet
        done = asyncio.run(agent.resume_pending(self.chat, True, outcome.pending_id))
        self.assertFalse(done.needs_confirmation)
        self.assertEqual(balance_of(self.uid, "Bank"), 48700)

    def test_declining_the_overspend_records_nothing(self):
        execute_tool(
            "log_expense", user_id=self.chat, amount=700, category="food", account_name="Bank"
        )
        outcome = self.ask(600)
        asyncio.run(agent.resume_pending(self.chat, False, outcome.pending_id))
        self.assertEqual(balance_of(self.uid, "Bank"), 49300)

    def test_a_spend_within_budget_does_not_ask(self):
        outcome = self.ask(200)
        self.assertFalse(outcome.needs_confirmation)
        self.assertEqual(balance_of(self.uid, "Bank"), 49800)


class BudgetChatTests(unittest.TestCase):
    """Budget requests in plain words: the AI decides; the tool lands it on the canonical category."""

    def budgets(self, uid):
        from app.services import analytics

        with SessionLocal() as db:
            return {r["category"]: r["budget"] for r in analytics.budget_status(db, uid)}

    def test_a_complaint_is_not_a_budget_command(self):
        chat, uid = new_user()
        text = (
            "Hey I told you I have allotted budget of 1k for food but I overused "
            "why didn't you give a reminder"
        )
        with FakeLLM() as llm:
            llm.add("Sorry about that. You set a ₹1,000 food budget; I'll flag it from now on.")
            reply = asyncio.run(bot_main.handle_incoming(chat, text))
        self.assertIn("food budget", reply.text)
        self.assertEqual(self.budgets(uid), {})

    def test_a_budget_request_sets_the_canonical_category(self):
        chat, uid = new_user()
        with FakeLLM() as llm:
            llm.add(tool("set_category_budget", category="food", monthly_budget=2000))
            reply = asyncio.run(bot_main.handle_incoming(chat, "budget for food 2k"))
        self.assertIn("Food & Dining", reply.text)
        self.assertEqual(self.budgets(uid), {"Food & Dining": 2000})


if __name__ == "__main__":
    unittest.main()
