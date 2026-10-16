"""Tests for Role 2: the LLM-native LangGraph agent.

Offline only: the single LLM entry point ``ai_gateway.complete`` is replaced by ``FakeLLM`` (see
``tests/fake_llm.py``), so each test states exactly what the "AI" decided (a tool call or text) and
asserts the resulting ledger state, reply, and confirmation behaviour.

Run with:  uv run python -m unittest discover -s tests -v
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

# Point the app at a throwaway SQLite DB *before* any app module is imported.
_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-agent-test-")
os.environ["DYNAMIC_MODELS"] = "false"  # tests never hit provider model-list APIs
os.environ.setdefault(
    "DATABASE_URL", "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
)
_KEY_VARS = (
    "GEMINI_API_KEYS",
    "OPENAI_API_KEYS",
    "ANTHROPIC_API_KEYS",
    "GROQ_API_KEYS",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
)

import asyncio  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

from app.bot import agent  # noqa: E402
from app.db import SessionLocal, engine, init_db  # noqa: E402
from app.models import ApiKey, ChatMessage, User  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402
from app.services_ai.mcp_server import execute_tool  # noqa: E402

try:
    from tests.fake_llm import FakeLLM, tool
except ImportError:  # discover -s tests (no -t .) puts tests/ on sys.path directly
    from fake_llm import FakeLLM, tool

AQ_KEY = "AQ.Ab8RN6" + "x" * 40 + "YOQw"
AQ_KEY_2 = "AQ.Ab8RN6" + "y" * 40 + "eDzg"
AIZA_KEY = "AIza" + "B" * 35


def run(coro):
    return asyncio.run(coro)


def chat(chat_id: str, text: str) -> str:
    """Send a message through the agent and return the reply text."""
    return run(agent.process_user_interaction(chat_id, text))


class AgentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    @classmethod
    def tearDownClass(cls):
        engine.dispose()

    def setUp(self):
        self.chat_id = f"test-{uuid.uuid4().hex[:10]}"
        gw.reset_rotation_state()
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for var in _KEY_VARS:
            os.environ.pop(var, None)
        # Strict wallet: seed funded balances so later spends/transfers are allowed.
        execute_tool("log_income", user_id=self.chat_id, amount=50000, source_account="Bank")

    def tearDown(self):
        self.env.stop()

    def add_key(self):
        gw.add_api_key(self.chat_id, "gemini", AIZA_KEY)

    def balances(self) -> dict[str, float]:
        result = execute_tool("get_pipe_balances", user_id=self.chat_id)
        return {a["name"]: a["balance"] for a in result["accounts"]}


class TestConfirmationFlow(AgentTestCase):
    """A big transfer pauses for a Yes; a stale or unrelated reply never moves money."""

    def test_yes_executes_transfer(self):
        with FakeLLM() as llm:
            llm.add(tool("transfer_funds", from_account="Bank", to_account="Cash", amount=10000))
            reply = chat(self.chat_id, "Transfer 10000 from Bank to Cash")
        self.assertIn("Tap Yes", reply)
        self.assertEqual(self.balances()["Bank"], 50000)
        # A typed "yes" is classified by the LLM, then resume runs the paused graph.
        with FakeLLM("confirm") as llm:
            reply = chat(self.chat_id, "yes")
        self.assertIn("Bank → Cash", reply)
        self.assertEqual(self.balances()["Bank"], 40000)
        self.assertEqual(self.balances()["Cash"], 10000)

    def test_no_cancels_transfer(self):
        with FakeLLM(tool("transfer_funds", from_account="Bank", to_account="Cash", amount=10000)):
            chat(self.chat_id, "Transfer 10000 from Bank to Cash")
        with FakeLLM("cancel"):
            reply = chat(self.chat_id, "No")
        self.assertIn("Cancelled", reply)
        self.assertEqual(self.balances()["Bank"], 50000)
        self.assertFalse(run(agent.has_pending(self.chat_id)))

    def test_yes_reply_does_not_replan(self):
        """A yes/no reply is answered by the classifier + resume, not a fresh planner round."""
        with FakeLLM(
            tool("transfer_funds", from_account="Bank", to_account="Cash", amount=10000)
        ) as llm:
            reply = chat(self.chat_id, "move ten thousand from bank to cash")
            self.assertIn("Tap Yes", reply)
            self.assertEqual(len(llm.calls), 1)
        with FakeLLM("confirm") as llm:
            reply = chat(self.chat_id, "Yes!")
            # Exactly one call: the decision classifier. The transfer is NOT re-planned.
            self.assertEqual(len(llm.calls), 1)
        self.assertIn("Bank → Cash", reply)
        self.assertEqual(self.balances()["Bank"], 40000)

    def test_unrelated_message_cancels_pending(self):
        with FakeLLM(tool("transfer_funds", from_account="Bank", to_account="Cash", amount=10000)):
            chat(self.chat_id, "Transfer 10000 from Bank to Cash")
        # "what is my balance" classifies as 'new', so the pending transfer is dropped.
        with FakeLLM("new", tool("get_pipe_balances")):
            chat(self.chat_id, "what is my balance")
        self.assertFalse(run(agent.has_pending(self.chat_id)))
        self.assertEqual(self.balances()["Bank"], 50000)

    def test_expired_confirmation_is_ignored(self):
        with FakeLLM(tool("transfer_funds", from_account="Bank", to_account="Cash", amount=10000)):
            chat(self.chat_id, "Transfer 10000 from Bank to Cash")
        # resume_pending refuses an answer older than PENDING_ACTION_TTL.
        from datetime import timedelta

        real_now = agent._utc_now()
        with patch.object(agent, "_utc_now", lambda: real_now + timedelta(minutes=6)):
            outcome = run(agent.resume_pending(self.chat_id, True))
        self.assertIsNone(outcome)
        self.assertEqual(self.balances()["Bank"], 50000)
        self.assertFalse(run(agent.has_pending(self.chat_id)))

    def test_small_transfer_runs_immediately(self):
        with FakeLLM(tool("transfer_funds", from_account="Bank", to_account="Cash", amount=500)):
            reply = chat(self.chat_id, "Transfer 500 from Bank to Cash")
        self.assertIn("Bank → Cash", reply)
        self.assertFalse(run(agent.has_pending(self.chat_id)))
        self.assertEqual(self.balances()["Cash"], 500)

    def test_classify_decision_reads_yes_no_and_other(self):
        with FakeLLM("confirm"):
            self.assertEqual(
                run(agent.classify_decision(self.chat_id, "go ahead", "do X")), "confirm"
            )
        with FakeLLM("cancel"):
            self.assertEqual(run(agent.classify_decision(self.chat_id, "nope", "do X")), "cancel")
        with FakeLLM("new"):
            self.assertEqual(
                run(agent.classify_decision(self.chat_id, "what's my balance", "do X")), "new"
            )


class TestApiKeysInChat(AgentTestCase):
    """AQ./AIza keys pasted in chat are stored encrypted and never sent to an AI."""

    def test_aq_key_saved_masked_and_encrypted(self):
        with (
            FakeLLM() as llm,
            patch.object(gw, "validate_api_key", AsyncMock(return_value=gw.KeyCheck("valid"))),
        ):
            reply = chat(self.chat_id, f"my gemini key is {AQ_KEY}")
            # A pasted key short-circuits in ingress: the planner LLM is never called.
            self.assertEqual(len(llm.calls), 0)
        self.assertIn("Saved", reply)
        self.assertIn(gw.mask_key(AQ_KEY), reply)
        self.assertNotIn(AQ_KEY, reply)
        self.assertEqual(gw.get_user_api_keys(self.chat_id), [("gemini", AQ_KEY)])

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.telegram_chat_id == self.chat_id).one()
            stored = db.query(ApiKey).filter(ApiKey.user_id == user.id).one()
            self.assertNotIn(AQ_KEY, stored.encrypted_key)
            history = " ".join(m.content for m in db.query(ChatMessage).filter_by(user_id=user.id))
            self.assertNotIn(AQ_KEY, history, "raw key must not be kept in chat history")
        finally:
            db.close()

    def test_multiple_keys_and_no_expense_logged(self):
        with (
            FakeLLM(),
            patch.object(gw, "validate_api_key", AsyncMock(return_value=gw.KeyCheck("valid"))),
        ):
            reply = chat(self.chat_id, f"keys: {AQ_KEY}, {AQ_KEY_2}")
        self.assertEqual(reply.count("Saved"), 2)
        self.assertEqual(len(gw.get_user_api_keys(self.chat_id)), 2)
        self.assertEqual(self.balances()["Bank"], 50000)

    def test_next_message_uses_the_saved_key(self):
        with (
            FakeLLM(),
            patch.object(gw, "validate_api_key", AsyncMock(return_value=gw.KeyCheck("valid"))),
        ):
            chat(self.chat_id, AQ_KEY)
        # With a key saved, a free-text message reaches the planner and is answered.
        with FakeLLM("Hi from AI") as llm:
            reply = chat(self.chat_id, "hello there")
            self.assertEqual(len(llm.calls), 1)
        self.assertIn("Hi from AI", reply)


class TestPersonas(AgentTestCase):
    """Advice/report personas are triggered by /coach|/budgeter|/summary and render the LLM's words."""

    def test_summary_request_uses_summary_persona(self):
        self.add_key()
        with FakeLLM() as llm:
            llm.add(tool("get_financial_snapshot", days=30))
            llm.add("• 💸 Spent ₹0 this month")
            reply = chat(self.chat_id, "/summary")
        self.assertIn("Spent ₹0", reply)
        self.assertIn("🧠", reply)  # the persona header heads the planner's own words
        # The second (text) call received the tool result as context in the conversation.
        tool_msgs = [m for m in llm.calls[1] if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertNotIn("error", tool_msgs[0]["content"])
        self.assertIn("spent", tool_msgs[0]["content"])

    def test_budgeter_command_skips_to_the_persona(self):
        self.add_key()
        with FakeLLM("1. Cut Swiggy") as llm:
            reply = chat(self.chat_id, "/budgeter I overspent")
            self.assertEqual(len(llm.calls), 1, "/budgeter goes straight to one LLM call")
        self.assertIn("1. Cut Swiggy", reply)
        # The persona name heads the reply.
        self.assertIn("🧠", reply)

    def test_no_key_gives_the_setkey_hint(self):
        with FakeLLM(no_key=True):
            reply = chat(self.chat_id, "How can I save money?")
        self.assertIn("/setkey", reply)


class TestHistoryAndNoKey(AgentTestCase):
    """No duplicate message, bounded history, and an honest no-key hint."""

    def test_ai_is_called_with_message_once(self):
        self.add_key()
        with FakeLLM("ok", "ok") as llm:
            chat(self.chat_id, "first message")
            chat(self.chat_id, "second message")
        self.assertEqual(len(llm.calls), 2, "the planner is called once per free-text message")
        last = llm.calls[-1]
        contents = [m.get("content") for m in last if isinstance(m.get("content"), str)]
        self.assertEqual(contents.count("second message"), 1)
        self.assertEqual(contents[-1], "second message")
        self.assertTrue(any("first message" in c for c in contents))

    def test_ai_failure_records_nothing_and_says_so(self):
        self.add_key()
        with FakeLLM(RuntimeError("provider down")), self.assertLogs("app.bot.agent", "WARNING"):
            reply = chat(self.chat_id, "spent 200 on chai")
        self.assertEqual(reply, agent.AI_BUSY_TEXT)
        self.assertEqual(self.balances()["Bank"], 50000)

    def test_history_is_limited(self):
        for i in range(30):
            agent.save_db_chat_message(self.chat_id, "user", f"old message {i}")
        history = agent.get_db_conversation_history(self.chat_id, limit=agent.HISTORY_LIMIT)
        self.assertEqual(len(history), agent.HISTORY_LIMIT)
        self.assertEqual(history[-1].content, "old message 29")

    def test_no_key_free_text_is_not_logged(self):
        with FakeLLM(no_key=True):
            reply = chat(self.chat_id, "spent 200 on chai")
        self.assertIn("/setkey", reply)
        self.assertEqual(self.balances()["Bank"], 50000)  # nothing recorded without a key


class TestLedgerActionsFromLLM(AgentTestCase):
    """The ledger outcome of what the LLM decides: expenses, income, multi-tool messages."""

    def test_single_expense_to_default_account(self):
        execute_tool("set_default_account", user_id=self.chat_id, account_name="Bank")
        self.add_key()
        with FakeLLM(
            tool("log_expense", amount=200, category="Food & Dining", account_name="Bank")
        ):
            reply = chat(self.chat_id, "spent 200 on food")
        self.assertIn("−₹200", reply)
        self.assertIn("Food", reply)
        self.assertAlmostEqual(self.balances()["Bank"], 50000 - 200, delta=1)

    def test_multi_tool_batch_from_one_message(self):
        self.add_key()
        with FakeLLM() as llm:
            llm.add(
                [
                    tool("log_income", amount=15000, category="Stipend", source_account="Bank"),
                    tool(
                        "log_expense",
                        amount=190,
                        category="Food & Dining",
                        merchant="Poornima",
                        account_name="Bank",
                    ),
                    tool(
                        "log_expense",
                        amount=80,
                        category="Food & Dining",
                        merchant="Neeraj",
                        account_name="Bank",
                    ),
                ]
            )
            reply = chat(self.chat_id, "stipend and snacks")
            self.assertEqual(len(llm.calls), 1)
        self.assertIn("3 entries", reply)
        self.assertIn("+₹15,000", reply)
        self.assertAlmostEqual(self.balances()["Bank"], 50000 + 15000 - 190 - 80, delta=1)

    def test_spending_query_renders_the_breakdown(self):
        self.add_key()
        with FakeLLM() as llm:
            llm.add(tool("get_spending_breakdown", period="month"))
            llm.add("You've spent ₹0 this month.")
            reply = chat(self.chat_id, "how much total did i spend")
        self.assertIn("spent", reply.lower())


class TestAccountSelection(AgentTestCase):
    """The LLM picks the account; the ledger enforces the balance. No dead-end prompts."""

    def test_expense_to_named_default_account(self):
        execute_tool("set_default_account", user_id=self.chat_id, account_name="Bank")
        self.add_key()
        with FakeLLM(tool("log_expense", amount=3000, category="Rent", account_name="Bank")):
            reply = chat(self.chat_id, "I spent 3k for rent")
        self.assertNotIn("which account", reply.lower(), f"Should not ask: {reply}")
        self.assertAlmostEqual(self.balances().get("Bank", 0), 50000 - 3000, delta=1)

    def test_explicit_account_runs_immediately(self):
        self.add_key()
        with FakeLLM(
            tool("log_expense", amount=500, category="Food & Dining", account_name="Bank")
        ):
            reply = chat(self.chat_id, "spent 500 on food via bank")
        self.assertNotIn("which account", reply.lower(), f"Should not prompt: {reply}")
        self.assertAlmostEqual(self.balances().get("Bank", 0), 50000 - 500, delta=1)

    def test_cash_expense_blocked_when_cash_is_empty(self):
        """Strict wallet: spending from a ₹0 Cash pipe is blocked, not silently logged."""
        self.add_key()
        with FakeLLM(
            tool("log_expense", amount=200, category="Food & Dining", account_name="Cash")
        ):
            reply = chat(self.chat_id, "spent 200 on chai from cash")
        # Cash was never funded, so the spend fails and says why; Bank is offered instead.
        self.assertIn("not enough balance", reply.lower())
        self.assertEqual(self.balances().get("Cash", 0.0), 0.0)
        self.assertEqual(self.balances()["Bank"], 50000)

    def test_blocked_spend_offers_another_account(self):
        self.add_key()
        with FakeLLM(
            tool("log_expense", amount=200, category="Food & Dining", account_name="Cash")
        ):
            outcome = run(agent.process_user_interaction_ex(self.chat_id, "chai 200 from cash"))
        self.assertIsNotNone(outcome.pay_alternative)
        self.assertEqual(outcome.pay_alternative.get("account"), "Bank")
        self.assertEqual(outcome.transaction_ids, [])

    def test_changing_the_default_moves_later_expenses(self):
        self.add_key()
        execute_tool(
            "transfer_funds",
            user_id=self.chat_id,
            from_account="Bank",
            to_account="Cash",
            amount=1000,
        )
        with FakeLLM(tool("set_default_account", account_name="Cash")):
            chat(self.chat_id, "use cash by default")
        with SessionLocal() as db:
            user = db.query(User).filter(User.telegram_chat_id == self.chat_id).one()
            self.assertEqual(user.default_account, "Cash")
        with FakeLLM(tool("log_expense", amount=20, category="Food & Dining", account_name="Cash")):
            chat(self.chat_id, "chai 20")
        self.assertAlmostEqual(self.balances()["Cash"], 1000 - 20, delta=1)


if __name__ == "__main__":
    unittest.main()
