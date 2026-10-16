"""Statement import logic: history vs balance, duplicates across sources, off the event loop. Offline."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-import-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""

import asyncio  # noqa: E402
import itertools  # noqa: E402
import threading  # noqa: E402
import unittest  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app.bot import main as bot  # noqa: E402
from app.bot import ui  # noqa: E402
from app.bot.callbacks import handle_callback  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.services import analytics, ledger  # noqa: E402
from app.services_ai.mcp_server import execute_tool  # noqa: E402

init_db()
_ids = itertools.count(1)


def run(coro):
    return asyncio.run(coro)


def new_chat():
    return f"imp-{next(_ids)}"


def day(n):
    return (datetime.now() - timedelta(days=n)).date().isoformat()


def rows(*items):
    return [
        {
            "date": d,
            "type": t,
            "amount": a,
            "merchant": m,
            "category": "Food & Dining",
            "description": desc,
            "reference": ref,
        }
        for d, t, a, m, desc, ref in items
    ]


def stage(chat, parsed):
    bot._pending_imports[chat] = bot.PendingImport(file_name="s.pdf", rows=parsed)


def bank(chat):
    with SessionLocal() as db:
        uid = ledger.get_or_create_user(db, chat).id
        accounts = ledger.get_account_balances(db, uid)["accounts"]
        return next((a["balance"] for a in accounts if a["name"] == "Bank"), None)


def fund(chat, amount):
    execute_tool("log_income", user_id=chat, amount=amount, source_account="Bank")


def count(chat):
    with SessionLocal() as db:
        return ledger.recent_transactions(db, ledger.get_or_create_user(db, chat).id, 1)[1]


class BalanceChoiceTests(unittest.TestCase):
    def test_history_only_leaves_the_balance_alone_but_reports_count_it(self):
        chat = new_chat()
        fund(chat, 10000)  # today's income: Bank holds 10000
        stage(chat, rows((day(40), "expense", 450.0, "Swiggy", "UPI/Swiggy", "R1")))
        reply = run(bot.answer_pending_import(chat, "history"))
        self.assertIn("did not change", reply.text)
        self.assertEqual(bank(chat), 10000.0)
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, chat).id
            rep = analytics.report(db, uid, analytics.parse_period("90d"))
            self.assertEqual(rep["summary"]["spent"], 450.0)  # counted in reports
            net = rep["net_worth"]
            self.assertEqual(net[-1]["total"], 10000.0)
            # the line is flat before today's income: the imported history never moved it
            self.assertEqual({p["total"] for p in net[:-1]}, {0.0})

    def test_apply_changes_the_balance_and_undo_puts_it_back(self):
        chat = new_chat()
        fund(chat, 10000)  # today's income: Bank holds 10000
        stage(chat, rows((day(40), "expense", 450.0, "Swiggy", "UPI/Swiggy", "R1")))
        reply = run(bot.answer_pending_import(chat, "apply"))
        self.assertIn("balance updated", reply.text)
        self.assertEqual(bank(chat), 9550.0)
        run(bot.handle_text(chat, "/undo"))
        self.assertEqual(bank(chat), 10000.0)

    def test_undoing_history_does_not_touch_the_balance(self):
        chat = new_chat()
        fund(chat, 10000)  # today's income: Bank holds 10000
        stage(chat, rows((day(40), "expense", 450.0, "Swiggy", "UPI/Swiggy", "R1")))
        run(bot.answer_pending_import(chat, "history"))
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, chat).id
            history = next(
                t for t in ledger.recent_transactions(db, uid, 5)[0] if t["amount"] == 450.0
            )
            ledger.undo_transaction(db, uid, history["id"])
        self.assertEqual(bank(chat), 10000.0)  # not 10450

    def test_buttons_offer_both_and_the_callbacks_work(self):
        data = [b.data for row in ui.import_rows() for b in row]
        self.assertEqual(data, ["imp:hist", "imp:apply", "imp:no"])
        chat = new_chat()
        stage(chat, rows((day(5), "income", 700.0, "Employer", "SALARY", "R9")))
        reply = run(handle_callback(chat, "imp:apply"))
        self.assertIn("Imported 1 transaction", reply.text)
        self.assertEqual(bank(chat), 700.0)

    def test_preview_explains_both_choices(self):
        parsed = rows((day(5), "expense", 450.0, "Swiggy", "x", "R1"))
        parsed.append(
            {
                "date": day(4),
                "type": "expense",
                "amount": 10.0,
                "merchant": "A",
                "category": "X",
                "note": "debit/credit guessed",
            }
        )
        text = bot._import_preview("s.pdf", parsed)
        self.assertIn("History", text)
        self.assertIn("balance", text)
        self.assertIn("no clear debit/credit marker", text)


class DuplicateTests(unittest.TestCase):
    def import_rows(self, chat, parsed, decision="history"):
        stage(chat, parsed)
        return run(bot.answer_pending_import(chat, decision)).text

    def test_the_same_statement_twice_adds_nothing(self):
        chat = new_chat()
        parsed = rows(
            (day(10), "expense", 100.0, "Chai Point", "a", "U1"),
            (day(10), "expense", 100.0, "Chai Point", "a", "U2"),
        )
        self.assertIn("Imported 2", self.import_rows(chat, parsed))
        self.assertIn("Imported 0", self.import_rows(chat, parsed))
        self.assertEqual(count(chat), 2)

    def test_two_genuine_identical_payments_in_one_statement_are_both_kept(self):
        chat = new_chat()
        parsed = rows(
            (day(10), "expense", 20.0, "Chai Point", "x", ""),
            (day(10), "expense", 20.0, "Chai Point", "x", ""),
        )
        self.assertIn("Imported 2", self.import_rows(chat, parsed))

    def test_the_same_payment_in_a_second_statement_format_is_recognised(self):
        chat = new_chat()
        self.import_rows(chat, rows((day(12), "expense", 450.0, "Swiggy", "UPI-SWIGGY-12345", "")))
        other_format = rows((day(11), "expense", 450.0, "Swiggy", "Paid to Swiggy DEBIT", ""))
        self.assertIn(
            "Imported 0", self.import_rows(chat, other_format)
        )  # a day off, other wording
        self.assertEqual(count(chat), 1)

    def test_same_bank_reference_is_the_same_payment(self):
        chat = new_chat()
        self.import_rows(chat, rows((day(12), "expense", 450.0, "Swiggy", "one", "UTR123456")))
        again = self.import_rows(
            chat, rows((day(30), "expense", 450.0, "Something else", "two", "UTR123456"))
        )
        self.assertIn("Skipped 1", again)

    def test_a_payment_already_typed_in_chat_is_not_added_again(self):
        chat = new_chat()
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, chat).id
            when = datetime.now() - timedelta(days=3)
            ledger.record_transaction(
                db, uid, 450, "Bank", "Food & Dining", "Swiggy", created_at=when
            )
        again = self.import_rows(chat, rows((day(3), "expense", 450.0, "Swiggy", "UPI", "R7")))
        self.assertIn("Skipped 1", again)

    def test_different_amounts_or_merchants_are_not_duplicates(self):
        chat = new_chat()
        self.import_rows(chat, rows((day(8), "expense", 450.0, "Swiggy", "a", "")))
        more = self.import_rows(
            chat,
            rows(
                (day(8), "expense", 451.0, "Swiggy", "b", ""),
                (day(8), "expense", 450.0, "Zomato", "c", ""),
            ),
        )
        self.assertIn("Imported 2", more)


class NotBlockingTests(unittest.TestCase):
    def test_parsing_runs_in_a_worker_thread(self):
        seen = {}

        def fake_parse(chat_id, file_bytes, kind):
            seen["thread"] = threading.current_thread().name
            return rows((day(2), "expense", 10.0, "Shop", "x", ""))

        async def go():
            with patch.object(bot, "_parse_upload", fake_parse):
                text = await bot.handle_document(new_chat(), "s.txt", b"x")
            return threading.current_thread().name, text

        main_thread, text = asyncio.run(go())
        self.assertNotEqual(seen["thread"], main_thread)
        self.assertIn("Found **1 transactions**", text)


if __name__ == "__main__":
    unittest.main()
