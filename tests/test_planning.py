"""Savings plans, the /plan command and the multi-step agent loop. Offline: the AI is FakeLLM."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-planning-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
for _var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "OPENAI_API_KEYS", "GROQ_API_KEYS"):
    os.environ[_var] = ""

import asyncio  # noqa: E402
import itertools  # noqa: E402
import unittest  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from app.bot import agent  # noqa: E402
from app.bot import main as bot  # noqa: E402
from app.bot.textutil import parse_goal  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.services import analytics, ledger  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402
from app.services_ai.mcp_server import execute_tool  # noqa: E402

try:
    from tests.fake_llm import FakeLLM, tool  # noqa: E402
except ImportError:  # run as: unittest discover -s tests (no top-level dir)
    from fake_llm import FakeLLM, tool  # noqa: E402

init_db()
_ids = itertools.count(1)


def run(coro):
    return asyncio.run(coro)


def seed(chat: str, income=60000.0, spend_cats=None):
    """Three months of salary and spending, so the plan has real numbers."""
    spend_cats = spend_cats or {"Shopping": 9000.0, "Entertainment": 6000.0, "Rent": 15000.0}
    with SessionLocal() as db:
        uid = ledger.get_or_create_user(db, chat).id
        for m in range(3):
            when = datetime.now() - timedelta(days=10 + 30 * m)
            if income:
                ledger.record_transaction(
                    db, uid, income, "Bank", "Salary", tx_type="income", created_at=when
                )
            for cat, amount in spend_cats.items():
                ledger.record_transaction(
                    db, uid, amount, "Bank", cat, tx_type="expense", created_at=when
                )
    return uid


class GoalParsingTests(unittest.TestCase):
    """parse_goal still reads the /plan command arguments."""

    def test_goals_are_understood(self):
        cases = {
            "50000 in 6 months": (50000, 6),
            "50k 6": (50000, 6),
            "2 lakh in 1 year": (200000, 12),
            "save ₹1,20,000 in 8 months": (120000, 8),
            "30000 in 12 weeks": (30000, 3),
        }
        for text, expected in cases.items():
            self.assertEqual(parse_goal(text), expected, text)

    def test_unclear_goals_are_not_guessed(self):
        for text in ("", "hello", "6 months", "50000", "50000 600000"):
            self.assertIsNone(parse_goal(text), text)


class QuestionTests(unittest.TestCase):
    def test_no_key_question_gets_a_hint_not_a_log(self):
        chat = f"q-{next(_ids)}"
        with FakeLLM(no_key=True):
            reply = run(bot.handle_text(chat, "is 640 too much for dinner?"))
        self.assertIn("didn't record anything", reply)
        self.assertIn("/setkey", reply)
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, chat).id
            rows, _ = ledger.recent_transactions(db, uid, 5)
        self.assertEqual(rows, [])

    def test_a_question_the_ai_answers_in_words_is_never_logged(self):
        chat = f"q-{next(_ids)}"
        with FakeLLM() as llm:
            llm.add("₹640 is a bit above a usual dinner, but fine once in a while.")
            reply = run(bot.handle_incoming(chat, "is 640 too much for dinner?"))
        self.assertIn("usual dinner", reply.text)
        self.assertFalse(reply.rows)
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, chat).id
            self.assertEqual(ledger.recent_transactions(db, uid, 5)[0], [])


class SavingsPlanTests(unittest.TestCase):
    def setUp(self):
        self.chat = f"plan-{next(_ids)}"

    def test_on_track(self):
        uid = seed(self.chat)  # 60000 in, 30000 out -> +30000 a month
        with SessionLocal() as db:
            p = analytics.savings_plan(db, uid, 60000, 6)
        self.assertTrue(p["on_track"])
        self.assertEqual(p["needed_per_month"], 10000)
        self.assertAlmostEqual(p["avg_surplus"], 30000, delta=1)
        self.assertEqual(p["suggested_cuts"], [])

    def test_short_gets_cuts_from_flexible_categories_only(self):
        uid = seed(self.chat, income=32000)  # +2000 a month
        with SessionLocal() as db:
            p = analytics.savings_plan(db, uid, 60000, 6)  # needs 10000
        self.assertFalse(p["on_track"])
        self.assertAlmostEqual(p["gap_per_month"], 8000, delta=1)
        names = {c["category"] for c in p["suggested_cuts"]}
        self.assertTrue(names <= {"Shopping", "Entertainment"})
        self.assertNotIn("Rent", names)
        for c in p["suggested_cuts"]:
            self.assertLessEqual(c["percent"], 40.01)
        self.assertGreater(p["still_short"], 0)

    def test_no_income_is_said_plainly(self):
        uid = seed(self.chat, income=0)
        with SessionLocal() as db:
            p = analytics.savings_plan(db, uid, 1000, 2)
        self.assertFalse(p["has_income_data"])
        self.assertIn("income", run(bot.handle_text(self.chat, "/plan 1000 2")))

    def test_bad_input_is_refused(self):
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, self.chat).id
            for target, months in ((0, 3), (100, 0), (-5, 3)):
                with self.assertRaises(ValueError):
                    analytics.savings_plan(db, uid, target, months)

    def test_snapshot_has_the_basics(self):
        uid = seed(self.chat)
        with SessionLocal() as db:
            s = analytics.financial_snapshot(db, uid, 60)
        for key in ("spent", "earned", "top_categories", "budgets", "money_now", "insights"):
            self.assertIn(key, s)
        self.assertGreater(s["spent"], 0)


class PlanCommandTests(unittest.TestCase):
    def setUp(self):
        self.chat = f"plancmd-{next(_ids)}"

    def say(self, text):
        return run(bot.handle_incoming(self.chat, text))

    def test_command_shows_the_plan(self):
        seed(self.chat, income=32000)
        by_command = self.say("/plan 60000 6")
        self.assertIn("Save ₹60,000 in 6 months", by_command.text)
        self.assertIn("Short by", by_command.text)
        self.assertIn("Shopping", by_command.text)

    def test_a_saving_sentence_runs_the_plan_tool_for_the_ai(self):
        seed(self.chat, income=32000)
        with FakeLLM() as llm:
            llm.add(tool("savings_plan", target_amount=60000, months=6))
            llm.add("You're short: trim Shopping and Entertainment.")
            reply = self.say("I want to save 60000 in 6 months")
            tool_msg = [m for m in llm.calls[1] if m["role"] == "tool"][-1]["content"]
        self.assertEqual(reply.text, "You're short: trim Shopping and Entertainment.")
        self.assertIn('"on_track": false', tool_msg)
        self.assertIn("Shopping", tool_msg)
        self.assertFalse(reply.rows)  # nothing recorded, nothing to undo

    def test_on_track_message(self):
        seed(self.chat)
        self.assertIn("On track", self.say("/plan 60000 6").text)

    def test_usage_when_unclear(self):
        self.assertIn("/plan 50000 6", self.say("/plan").text)
        self.assertIn("/plan 50000 6", self.say("/plan soon").text)

    def test_plan_is_in_the_menu(self):
        self.assertIn("plan", dict(bot.COMMAND_MENU))
        self.assertLessEqual(len(bot.COMMAND_MENU), 100)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.chat = f"tools-{next(_ids)}"
        seed(self.chat)

    def test_plan_tool(self):
        res = execute_tool("savings_plan", user_id=self.chat, target_amount=30000, months=3)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["on_track"])

    def test_snapshot_tool_clamps_days(self):
        res = execute_tool("get_financial_snapshot", user_id=self.chat, days=99999)
        self.assertEqual(res["status"], "success")

    def test_both_tools_are_offered_to_the_ai(self):
        names = {t["function"]["name"] for t in agent.TOOLS_SCHEMA}
        self.assertTrue({"savings_plan", "get_financial_snapshot"} <= names)


class AgentLoopTests(unittest.TestCase):
    """The planner -> guard -> executor loop, with the AI scripted."""

    def setUp(self):
        gw.reset_rotation_state()
        self.chat = f"loop-{next(_ids)}"
        self.uid = seed(self.chat)

    def say(self, text):
        return run(bot.handle_incoming(self.chat, text))

    def balances(self):
        with SessionLocal() as db:
            data = ledger.get_account_balances(db, self.uid)["accounts"]
        return {a["name"]: a["balance"] for a in data}

    def count(self, amount):
        from sqlalchemy import text

        with SessionLocal() as db:
            return db.scalar(
                text("select count(*) from transactions where amount=:a and user_id=:u"),
                {"a": amount, "u": self.uid},
            )

    def test_plain_answer_needs_no_tools(self):
        with FakeLLM("Sure, ask away.") as llm:
            reply = self.say("Can I ask you something?")
        self.assertEqual(reply.text, "Sure, ask away.")
        self.assertEqual(len(llm.calls), 1)

    def test_looks_up_then_answers(self):
        with FakeLLM() as llm:
            llm.add(tool("savings_plan", target_amount=50000, months=5))
            llm.add("Yes. You keep about ₹30,000 a month.")
            reply = self.say("Can I afford a 50000 laptop in 5 months?")
        self.assertEqual(reply.text, "Yes. You keep about ₹30,000 a month.")
        self.assertEqual(len(llm.calls), 2)
        tool_msg = [m for m in llm.calls[1] if m["role"] == "tool"][0]
        self.assertIn("needed_per_month", tool_msg["content"])
        self.assertEqual(self.balances()["Bank"], 90000)  # nothing written

    def test_writes_run_for_this_user_even_if_the_ai_names_another(self):
        other = f"other-{next(_ids)}"
        with FakeLLM() as llm:
            llm.add(
                tool(
                    "log_expense",
                    amount=120,
                    category="Food & Dining",
                    account_name="Bank",
                    user_id=other,
                )
            )
            reply = self.say("lunch 120")
        self.assertIn("−₹120", reply.text)
        self.assertEqual(self.count(120), 1)
        self.assertEqual(self.balances()["Bank"], 90000 - 120)
        with SessionLocal() as db:
            other_uid = ledger.get_or_create_user(db, other).id
            self.assertEqual(ledger.recent_transactions(db, other_uid, 5)[0], [])

    def test_key_saving_tool_is_never_available(self):
        with FakeLLM("ok") as llm:
            self.say("hello there")
        self.assertNotIn("save_user_api_key", str(llm.calls[0]))
        offered = {t["function"]["name"] for t in agent.TOOLS_SCHEMA} - agent.HIDDEN_TOOLS
        self.assertNotIn("save_user_api_key", offered)

    def test_big_transfer_waits_for_yes(self):
        with FakeLLM() as llm:
            llm.add(tool("transfer_funds", from_account="Bank", to_account="Cash", amount=9000))
            reply = self.say("move 9000 to cash")
        self.assertIn("Transfer ₹9,000", reply.text)
        self.assertTrue(reply.rows)  # Yes / Cancel
        self.assertEqual(self.balances().get("Cash", 0), 0)
        self.assertEqual(self.balances()["Bank"], 90000)

    def test_small_transfer_runs(self):
        with FakeLLM() as llm:
            llm.add(tool("transfer_funds", from_account="Bank", to_account="Cash", amount=500))
            reply = self.say("move 500 to cash")
        self.assertFalse(run(agent.has_pending(self.chat)))
        self.assertIn("500", reply.text)
        self.assertEqual(self.balances()["Cash"], 500)
        self.assertEqual(self.balances()["Bank"], 89500)

    def test_step_limit(self):
        with FakeLLM() as llm:
            for _ in range(agent.MAX_STEPS + 2):
                llm.add(tool("get_pipe_balances"))
            reply = self.say("check my balance again and again")
        self.assertEqual(len(llm.calls), agent.MAX_STEPS)
        self.assertTrue(reply.text)
        self.assertEqual(self.balances()["Bank"], 90000)

    def test_ai_failure_before_writing_records_nothing(self):
        with FakeLLM(gw.AllKeysExhaustedError([])):
            reply = self.say("How am I doing this month?")
        self.assertIn("nothing was recorded", reply.text)
        self.assertEqual(self.balances()["Bank"], 90000)

    def test_ai_failure_after_writing_keeps_the_result(self):
        with FakeLLM() as llm:
            llm.add(tool("log_expense", amount=77, category="Food & Dining", account_name="Bank"))
            llm.add(tool("get_pipe_balances"))  # a read after the write sends it back to the AI
            llm.add(gw.AllKeysExhaustedError([]))
            reply = self.say("lunch 77 and show my balance")
        self.assertIn("77", reply.text)
        self.assertEqual(self.count(77), 1)
        self.assertEqual(self.balances()["Bank"], 90000 - 77)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        gw.reset_rotation_state()
        self.chat = f"e2e-{next(_ids)}"
        seed(self.chat)

    def say(self, text):
        return run(bot.handle_incoming(self.chat, text))

    def test_complex_question_uses_the_agent_and_replies(self):
        with FakeLLM() as llm:
            llm.add(tool("get_financial_snapshot", days=30))
            llm.add("You are doing fine: you keep about half of what you earn.")
            reply = self.say("How am I doing this month, should I worry about anything?")
        self.assertIn("doing fine", reply.text)
        self.assertEqual(len(llm.calls), 2)

    def test_agent_logs_show_the_compact_line_and_an_undo_button(self):
        with FakeLLM() as llm:
            llm.add(tool("log_expense", amount=640, category="Food & Dining", account_name="Bank"))
            reply = self.say("I just ate out for 640, is that too much compared to my usual?")
        self.assertIn("−₹640", reply.text)
        self.assertTrue(any(b.data.startswith("undo:") for row in reply.rows for b in row))

    def test_big_transfer_asks_for_confirmation_then_runs(self):
        with FakeLLM() as llm:
            llm.add(tool("transfer_funds", from_account="Bank", to_account="Cash", amount=9000))
            first = self.say("Please move some money to cash so I can afford the trip next week")
            self.assertIn("Transfer ₹9,000", first.text)
            llm.add("confirm")  # the typed "yes" is read by the AI
            done = self.say("yes")
        self.assertIn("9,000", done.text)
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, self.chat).id
            accounts = {
                a["name"]: a["balance"] for a in ledger.get_account_balances(db, uid)["accounts"]
            }
        self.assertEqual(accounts["Cash"], 9000)
        self.assertEqual(accounts["Bank"], 81000)

    def test_no_key_never_logs_free_text(self):
        with FakeLLM(no_key=True):
            reply = self.say("Spent 450 on dinner via Bank")
        self.assertIn("/setkey", reply.text)
        self.assertNotIn("−₹450", reply.text)


if __name__ == "__main__":
    unittest.main()
