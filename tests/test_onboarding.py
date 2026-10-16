"""First contact and everyday chat: the one setup question, default account, corrections, balances.

Offline: the AI agent is scripted with FakeLLM (the fake decides the tool call; the test checks the
ledger, the reply and the confirmation flow).
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-onboarding-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
for _var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "OPENAI_API_KEYS", "GROQ_API_KEYS"):
    os.environ[_var] = ""

import asyncio  # noqa: E402
import itertools  # noqa: E402
import unittest  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app.bot import agent, onboarding  # noqa: E402
from app.bot import main as bot_main  # noqa: E402
from app.bot.callbacks import handle_callback  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.services import ledger  # noqa: E402

try:
    from tests.fake_llm import FakeLLM, tool  # noqa: E402
except ImportError:  # run as: unittest discover -s tests (no top-level dir)
    from fake_llm import FakeLLM, tool  # noqa: E402

_ids = itertools.count(700000)


def say(chat, text):
    return asyncio.run(bot_main.handle_incoming(chat, text))


def tap(chat, data):
    return asyncio.run(handle_callback(chat, data))


def yes(chat):
    """Tap Yes on the question the agent is waiting on."""
    return asyncio.run(bot_main.answer_pending_action(chat, True))


def last_ids(chat, n=10):
    with SessionLocal() as db:
        uid = ledger.get_or_create_user(db, chat).id
        rows, _ = ledger.recent_transactions(db, uid, n)
    return rows


def balances(chat):
    with SessionLocal() as db:
        user = ledger.get_or_create_user(db, chat)
        data = ledger.get_account_balances(db, user.id)
    return {a["name"]: a["balance"] for a in data["accounts"]}


class NewUserTests(unittest.TestCase):
    def setUp(self):
        init_db()
        self.chat = str(next(_ids))
        onboarding.reset()
        patcher = patch.object(settings, "onboarding", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_start_asks_one_question_with_buttons(self):
        r = say(self.chat, "/start")
        self.assertIn("where do you usually pay from", r.text)
        labels = [b.text for row in r.rows for b in row]
        self.assertTrue(any("Bank" in x for x in labels) and any("Cash" in x for x in labels))

    def test_answering_then_giving_a_balance_finishes_setup(self):
        say(self.chat, "/start")
        r = tap(self.chat, "ob:acct:bank")
        self.assertIn("how much is in Bank", r.text)
        r = say(self.chat, "25000")
        self.assertIn("starts at ₹25,000", r.text)
        self.assertEqual(balances(self.chat)["Bank"], 25000)
        # the opening balance is money, not earnings
        stats = say(self.chat, "/stats")
        self.assertNotIn("25,000", stats.text)

    def test_skip_button_and_typed_skip_both_work(self):
        say(self.chat, "/start")
        tap(self.chat, "ob:acct:cash")
        self.assertIn("all set", tap(self.chat, "ob:skip").text)
        other = str(next(_ids))
        say(other, "/start")
        say(other, "bank")
        self.assertIn("all set", say(other, "skip").text)

    def test_the_default_account_takes_expenses_that_name_none(self):
        say(self.chat, "/start")
        say(self.chat, "cash")
        say(self.chat, "skip")
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, self.chat).id
            ledger.record_transaction(db, uid, 1000, "Cash", "Seed", tx_type="income")
        with FakeLLM() as llm:  # the AI names no account: the user's default is used
            llm.add(tool("log_expense", amount=20, category="Food & Dining"))
            say(self.chat, "chai 20")
        self.assertEqual(balances(self.chat)["Cash"], 980)
        self.assertEqual(balances(self.chat).get("Bank", 0), 0)

    def test_a_first_message_that_is_an_entry_is_logged_after_the_answer(self):
        with FakeLLM() as llm:
            r = say(self.chat, "spent 450 on dinner")
            self.assertIn("where do you usually pay from", r.text)
            self.assertEqual(llm.calls, [])  # held by onboarding, the AI is not asked yet
            self.assertEqual(balances(self.chat).get("Bank", 0), 0)  # nothing yet
            llm.add(tool("log_expense", amount=450, category="Food & Dining", account_name="Bank"))
            r = tap(self.chat, "ob:acct:bank")
            self.assertEqual(len(llm.calls), 1)  # the held message reached the AI after the answer
        self.assertIn("not enough balance", r.text.lower())  # strict wallet: Bank is ₹0
        self.assertEqual(balances(self.chat)["Bank"], 0)

    def test_ignoring_the_question_is_never_a_dead_end(self):
        with FakeLLM() as llm:
            say(self.chat, "spent 450 on dinner")
            llm.add(tool("log_expense", amount=450, category="Food & Dining"))
            llm.add(tool("log_income", amount=50000, source_account="Bank", category="Salary"))
            r = say(self.chat, "also got salary 50000")  # not an answer
            self.assertIn("not enough balance", r.text.lower())
            self.assertIn("+₹50,000", r.text)
            self.assertEqual(balances(self.chat)["Bank"], 50000)
            # and the next message is plain chat again, not another question
            llm.add(tool("log_expense", amount=100, category="Food & Dining"))
            r = say(self.chat, "coffee 100")
        self.assertNotIn("where do you usually pay", r.text)
        self.assertEqual(balances(self.chat)["Bank"], 49900)

    def test_a_returning_user_is_not_asked(self):
        say(self.chat, "/start")
        tap(self.chat, "ob:acct:bank")
        say(self.chat, "skip")
        self.assertNotIn("where do you usually pay", say(self.chat, "/start").text)

    def test_commands_work_for_a_new_user_without_the_question(self):
        self.assertIn("Total", say(self.chat, "/balance").text)

    def test_default_can_be_changed_in_chat(self):
        say(self.chat, "/start")
        say(self.chat, "bank")
        say(self.chat, "skip")
        with FakeLLM() as llm:
            llm.add(tool("set_default_account", account_name="Cash"))
            say(self.chat, "use cash by default")
            self.assertEqual(onboarding.default_account(self.chat), "Cash")
            with SessionLocal() as db:
                uid = ledger.get_or_create_user(db, self.chat).id
                ledger.record_transaction(db, uid, 1000, "Cash", "Seed", tx_type="income")
            llm.add(tool("log_expense", amount=20, category="Food & Dining"))
            say(self.chat, "chai 20")
        self.assertEqual(balances(self.chat)["Cash"], 980)

    def test_a_stale_button_does_not_crash(self):
        self.assertTrue(tap(self.chat, "ob:acct:nonsense").notice)


class EverydayChatTests(unittest.TestCase):
    """The new user is set up; these are the things people actually type (the AI is scripted)."""

    def setUp(self):
        init_db()
        self.chat = str(next(_ids))
        onboarding.reset()
        with patch.object(settings, "onboarding", True):
            say(self.chat, "/start")
            say(self.chat, "bank")
            say(self.chat, "skip")

    def seed(self, amount=100000):
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, self.chat).id
            ledger.record_transaction(db, uid, amount, "Bank", "Seed", tx_type="income")

    def tx_id(self, amount):
        return next(r["id"] for r in last_ids(self.chat) if r.get("amount") == amount)

    def test_set_the_real_balance_in_words_asks_first(self):
        with FakeLLM() as llm:
            llm.add(tool("set_account_balance", account_name="Bank", balance=25000))
            r = say(self.chat, "bank balance is 25000")
            self.assertTrue(r.rows)  # Yes / Cancel
            self.assertIn("Bank balance to ₹25,000", r.text)
            self.assertEqual(balances(self.chat).get("Bank", 0), 0)  # nothing until Yes
            yes(self.chat)
            self.assertEqual(balances(self.chat)["Bank"], 25000)
            llm.add(tool("set_account_balance", account_name="Cash", balance=5000))
            say(self.chat, "I have 5000 in cash")
            yes(self.chat)
        self.assertEqual(balances(self.chat)["Cash"], 5000)

    def test_setting_a_balance_is_not_spending_or_earning(self):
        with FakeLLM() as llm:
            for amount in (25000, 20000):
                llm.add(tool("set_account_balance", account_name="Bank", balance=amount))
                say(self.chat, f"bank balance is {amount}")
                yes(self.chat)
        self.assertEqual(balances(self.chat)["Bank"], 20000)
        stats = say(self.chat, "/stats").text
        self.assertNotIn("Spent ₹5,000", stats)
        self.assertNotIn("Earned ₹25,000", stats)

    def test_correcting_an_entry_asks_then_moves_the_balance(self):
        self.seed()
        with FakeLLM() as llm:
            llm.add(tool("log_expense", amount=120, category="Coffee", account_name="Bank"))
            say(self.chat, "coffee 120")
            llm.add(tool("log_expense", amount=450, category="Dinner", account_name="Bank"))
            say(self.chat, "paid 450 for dinner")
            dinner = self.tx_id(450)
            llm.add(tool("edit_transaction", transaction_id=dinner, amount=540))
            r = say(self.chat, "oops the dinner was actually 540 not 450")
            self.assertIn("Change entry", r.text)
            self.assertEqual(balances(self.chat)["Bank"], 100000 - 570)  # not yet
            done = yes(self.chat)
            self.assertIn("540", done.text)
            self.assertEqual(balances(self.chat)["Bank"], 100000 - 660)
            llm.add(tool("edit_transaction", transaction_id=dinner, amount=600))
            say(self.chat, "change the last one to 600")
            yes(self.chat)
        self.assertEqual(balances(self.chat)["Bank"], 100000 - 720)

    def test_cancelling_a_correction_changes_nothing(self):
        self.seed()
        with FakeLLM() as llm:
            llm.add(tool("log_expense", amount=450, category="Dinner", account_name="Bank"))
            say(self.chat, "paid 450 for dinner")
            llm.add(tool("edit_transaction", transaction_id=self.tx_id(450), amount=540))
            say(self.chat, "the dinner was 540")
            r = asyncio.run(bot_main.answer_pending_action(self.chat, False))
        self.assertIn("Cancelled", r.text)
        self.assertEqual(balances(self.chat)["Bank"], 100000 - 450)

    def test_a_plain_statement_is_a_new_entry_not_a_correction(self):
        self.seed()
        with FakeLLM() as llm:
            llm.add(tool("log_expense", amount=450, category="Dinner", account_name="Bank"))
            say(self.chat, "paid 450 for dinner")
            llm.add(tool("log_expense", amount=400, category="Dinner", account_name="Bank"))
            r = say(self.chat, "dinner was 400")
        self.assertFalse(agent_waits(self.chat))
        self.assertIn("−₹400", r.text)
        self.assertEqual(balances(self.chat)["Bank"], 100000 - 850)

    def test_a_correction_with_nothing_to_correct_changes_nothing(self):
        with FakeLLM() as llm:
            llm.add(tool("search_transactions", query="pizza"))
            llm.add("I couldn't find a pizza entry, so nothing changed.")
            r = say(self.chat, "delete the pizza")
            self.assertEqual(len(llm.calls), 2)  # looked it up, then answered
            tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
            self.assertIn("transactions", tool_msgs[-1]["content"])
        self.assertIn("couldn't find", r.text)
        self.assertEqual(last_ids(self.chat), [])

    def test_deleting_an_entry_by_name(self):
        self.seed()
        with FakeLLM() as llm:
            llm.add(tool("log_expense", amount=120, category="Coffee", account_name="Bank"))
            say(self.chat, "coffee 120")
            llm.add(tool("log_expense", amount=250, category="Transport", account_name="Bank"))
            say(self.chat, "uber 250")
            llm.add(tool("delete_transaction", transaction_id=self.tx_id(120)))
            r = say(self.chat, "delete the coffee")
            self.assertIn("Delete entry", r.text)
            self.assertEqual(balances(self.chat)["Bank"], 100000 - 370)  # not before Yes
            yes(self.chat)
        self.assertEqual(balances(self.chat)["Bank"], 100000 - 250)

    def test_chat_without_an_entry_gets_the_ai_reply_and_logs_nothing(self):
        with FakeLLM() as llm:
            llm.add("Hey! Tell me what you spent.")
            self.assertIn("Hey", say(self.chat, "hi").text)
            llm.add("450: did you spend it or receive it?")
            r = say(self.chat, "450")
        self.assertIn("spend it or receive it", r.text)
        self.assertEqual(last_ids(self.chat), [])
        self.assertEqual(balances(self.chat).get("Bank", 0), 0)


def agent_waits(chat):
    return asyncio.run(agent.has_pending(chat))


if __name__ == "__main__":
    unittest.main()
