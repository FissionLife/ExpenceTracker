"""The flow display: one plain line per agent step, in the terminal and (on request) in Telegram.

Offline only: the AI is a scripted stand-in (tests/fake_llm.py).
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-flow-test-")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ.setdefault(
    "DATABASE_URL", "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
)
os.environ["TELEGRAM_BOT_TOKEN"] = ""

import asyncio  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app.bot import agent, flow  # noqa: E402
from app.bot import main as bot  # noqa: E402
from app.db import engine, init_db  # noqa: E402
from app.services_ai.mcp_server import execute_tool  # noqa: E402

try:
    from tests.fake_llm import FakeLLM, tool  # noqa: E402
except ImportError:  # discover -s tests (no -t .) puts tests/ on sys.path directly
    from fake_llm import FakeLLM, tool  # noqa: E402


def run(coro):
    return asyncio.run(coro)


NODES = ["ingress", "planner", "guard", "executor", "renderer"]


class FlowTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    @classmethod
    def tearDownClass(cls):
        engine.dispose()

    def setUp(self):
        self.chat = f"flow-{uuid.uuid4().hex[:8]}"
        execute_tool("log_income", user_id=self.chat, amount=20000, source_account="Bank")
        execute_tool("log_income", user_id=self.chat, amount=5000, source_account="Cash")

    def tearDown(self):
        flow.set_enabled(self.chat, False)

    def ask(self, text, *answers):
        with FakeLLM(*answers):
            return run(agent.process_user_interaction_ex(self.chat, text))


class TraceTests(FlowTestCase):
    def test_a_plain_entry_shows_all_five_steps_and_what_the_database_did(self):
        outcome = self.ask(
            "spent 450 on dinner",
            tool("log_expense", amount=450, category="Food & Dining", account_name="Cash"),
        )
        steps = [line.split(":")[0] for line in outcome.trace]
        self.assertEqual(steps, NODES)
        text = "\n".join(outcome.trace)
        self.assertIn("log_expense(amount=450", text)  # what the AI picked
        self.assertIn("inserted 1 transaction (expense Rs 450", text)  # what the database did
        self.assertIn("Cash balance now Rs 4,550", text)
        self.assertIn("all checks passed", text)

    def test_a_risky_transfer_pauses_and_the_trace_says_so_then_resumes(self):
        paused = self.ask(
            "move 6000 from bank to cash",
            tool("transfer_funds", from_account="Bank", to_account="Cash", amount=6000),
        )
        self.assertTrue(paused.needs_confirmation)
        self.assertIn("paused, waiting for your Yes/Cancel", paused.trace[-1])
        self.assertNotIn("executor", "\n".join(paused.trace))  # nothing ran yet
        done = run(agent.resume_pending(self.chat, True))
        text = "\n".join(done.trace)
        self.assertIn("you said Yes", text)
        self.assertIn("moved Rs 6,000 Bank -> Cash", text)

    def test_saying_no_changes_nothing_and_the_trace_says_so(self):
        self.ask(
            "move 6000 from bank to cash",
            tool("transfer_funds", from_account="Bank", to_account="Cash", amount=6000),
        )
        done = run(agent.resume_pending(self.chat, False))
        self.assertIn("you said No: nothing was changed", "\n".join(done.trace))

    def test_without_an_ai_key_the_trace_says_nothing_was_sent(self):
        with FakeLLM(no_key=True):
            outcome = run(agent.process_user_interaction_ex(self.chat, "spent 450 on dinner"))
        text = "\n".join(outcome.trace)
        self.assertIn("no AI key is set", text)
        self.assertNotIn("executor", text)

    def test_a_question_reads_the_database_and_the_trace_says_nothing_changed(self):
        outcome = self.ask(
            "how much did I spend on food?",
            tool("get_spending_breakdown", period="month"),
            "You have not spent anything on food yet.",
        )
        text = "\n".join(outcome.trace)
        self.assertIn("read-only query, nothing changed", text)
        self.assertIn("the AI read the tool results and wrote the answer", text)

    def test_the_terminal_gets_the_same_lines(self):
        with self.assertLogs("flow", level="INFO") as logs:
            self.ask(
                "spent 100 on tea",
                tool("log_expense", amount=100, category="Food & Dining", account_name="Cash"),
            )
        joined = "\n".join(logs.output)
        self.assertIn(f"[chat {self.chat}] planner:", joined)
        self.assertIn("executor:", joined)

    def test_showing_the_flow_can_never_break_a_message(self):
        with patch.object(flow, "describe", side_effect=RuntimeError("boom")):
            outcome = self.ask(
                "spent 100 on tea",
                tool("log_expense", amount=100, category="Food & Dining", account_name="Cash"),
            )
        self.assertIn("100", outcome.text)  # the entry still worked
        self.assertEqual(outcome.trace, [])


class CodeLocationTests(FlowTestCase):
    """Each step also says which function ran, in which file and line, who sent the graph there."""

    def entry(self):
        return self.ask(
            "spent 450 on dinner",
            tool("log_expense", amount=450, category="Food & Dining", account_name="Cash"),
        )

    def test_every_step_has_a_code_location(self):
        outcome = self.entry()
        self.assertEqual(len(outcome.trace_at), len(outcome.trace))
        for node, at in zip(NODES, outcome.trace_at, strict=True):
            self.assertIn(f"{node}_node()", at)  # e.g. planner_node()
            self.assertIn("app/bot/agent.py:", at)

    def test_who_sent_the_graph_to_each_step(self):
        at = self.entry().trace_at
        self.assertIn("on_text()", at[0])  # the entry path: Telegram -> handle_incoming -> agent
        self.assertIn("handle_incoming()", at[0])
        self.assertIn("process_user_interaction_ex()", at[0])
        self.assertIn("route_after_ingress()", at[1])
        self.assertIn("route_after_planner()", at[2])
        self.assertIn("route_after_guard()", at[3])
        self.assertIn("route_after_executor()", at[4])

    def test_the_executor_shows_the_whole_chain_down_to_the_ledger(self):
        executor = self.entry().trace_at[3]
        for name in ("call_tool()", "execute_tool()", "log_expense()", "record_transaction()"):
            self.assertIn(name, executor)
        self.assertLess(executor.index("call_tool()"), executor.index("record_transaction()"))

    def test_the_line_number_points_at_the_def_not_the_decorator(self):
        at = self.entry().trace_at[1]
        line = int(at.split("agent.py:")[1].split(" ")[0])
        source = (Path(agent.__file__)).read_text(encoding="utf-8").splitlines()
        self.assertTrue(source[line - 1].lstrip().startswith("async def planner_node"))

    def test_the_terminal_prints_the_code_lines_too(self):
        with self.assertLogs("flow", level="INFO") as logs:
            self.entry()
        joined = "\n".join(logs.output)
        self.assertIn("at    app/bot/agent.py:", joined)
        self.assertIn("from  app/bot/agent.py:", joined)


class TelegramTests(FlowTestCase):
    def test_flow_command_switches_the_footer_on_and_off(self):
        answer = tool("log_expense", amount=100, category="Food & Dining", account_name="Cash")
        with FakeLLM(answer):
            plain = run(bot.handle_incoming(self.chat, "spent 100 on tea")).text
        self.assertNotIn("How I got here", plain)

        self.assertIn("ON", run(bot.handle_incoming(self.chat, "/flow on")).text)
        with FakeLLM(answer):
            shown = run(bot.handle_incoming(self.chat, "spent 100 on tea")).text
        self.assertIn("How I got here:", shown)
        self.assertIn("1. ingress:", shown)
        self.assertIn("5. renderer:", shown)

        self.assertIn("OFF", run(bot.handle_incoming(self.chat, "/flow off")).text)
        with FakeLLM(answer):
            hidden = run(bot.handle_incoming(self.chat, "spent 100 on tea")).text
        self.assertNotIn("How I got here", hidden)

    def test_flow_code_adds_the_file_and_line_under_each_step(self):
        answer = tool("log_expense", amount=100, category="Food & Dining", account_name="Cash")
        run(bot.handle_incoming(self.chat, "/flow on"))
        with FakeLLM(answer):
            plain = run(bot.handle_incoming(self.chat, "spent 100 on tea")).text
        self.assertNotIn("app/bot/agent.py", plain)

        self.assertIn("CODE", run(bot.handle_incoming(self.chat, "/flow code")).text)
        with FakeLLM(answer):
            coded = run(bot.handle_incoming(self.chat, "spent 100 on tea")).text
        self.assertIn("planner_node()", coded)
        self.assertIn("record_transaction()", coded)

    def test_flow_alone_toggles(self):
        run(bot.handle_incoming(self.chat, "/flow"))
        self.assertTrue(flow.enabled(self.chat))
        run(bot.handle_incoming(self.chat, "/flow"))
        self.assertFalse(flow.enabled(self.chat))


if __name__ == "__main__":
    unittest.main()
