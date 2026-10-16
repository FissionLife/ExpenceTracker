"""Regression tests for the chat-logic audit: confirmations, buttons, keys, startup. Offline.

Free text is read by the LLM agent, so every chat-driven test scripts the decision with
``FakeLLM`` (tests/fake_llm.py) and asserts the ledger/reply/button outcome.
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-chatfix-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
for _var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "OPENAI_API_KEYS", "GROQ_API_KEYS"):
    os.environ[_var] = ""

import asyncio  # noqa: E402
import itertools  # noqa: E402
import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app.bot import agent, ui  # noqa: E402
from app.bot import main as bot  # noqa: E402
from app.bot.callbacks import handle_callback  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.services import ledger  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402

try:
    from tests.fake_llm import FakeLLM, tool
except ImportError:  # discover -s tests (no -t .) puts tests/ on sys.path directly
    from fake_llm import FakeLLM, tool

init_db()
_ids = itertools.count(1)


def run(coro):
    return asyncio.run(coro)


def chat():
    return f"cf-{next(_ids)}"


def balance(chat_id, account):
    with SessionLocal() as db:
        uid = ledger.get_or_create_user(db, chat_id).id
        for a in ledger.get_account_balances(db, uid)["accounts"]:
            if a["name"] == account:
                return a["balance"]
    return None


def seed(chat_id, account="Bank", amount=30000.0):
    """Fund an account directly (also marks the user as set up, so onboarding stays out of the way)."""
    with SessionLocal() as db:
        user = ledger.get_or_create_user(db, chat_id)
        ledger.record_transaction(db, user.id, amount, account, "Seed", tx_type="income")
        user.default_account = account
        db.commit()


class KeyDetectionTests(unittest.TestCase):
    def test_hyphenated_words_are_not_api_keys(self):
        for text in ("paid 500 for desk-organizer-with-drawers", "ask-me-anything-about-spending"):
            self.assertEqual(gw.extract_api_keys(text), [], text)
        self.assertEqual(len(gw.extract_api_keys("my key sk-proj-" + "a" * 40)), 1)

    def test_secrets_are_hidden_in_logs_and_errors(self):
        text = "failed: https://x/y?key=AIza" + "B" * 35 + "&alt=1 and sk-ant-" + "c" * 30
        hidden = gw.redact_secrets(text)
        self.assertNotIn("B" * 20, hidden)
        self.assertNotIn("c" * 20, hidden)
        self.assertIn("key=***", hidden)


class ConfirmationTests(unittest.TestCase):
    def say(self, chat_id, text, *answers):
        with FakeLLM(*answers):
            return run(bot.handle_incoming(chat_id, text))

    def setUp(self):
        self.chat = chat()
        seed(self.chat)

    def pending(self):
        """Ask for a transfer big enough to need a Yes; returns the Reply with its buttons."""
        return self.say(
            self.chat,
            "transfer 7000 from bank to cash",
            tool("transfer_funds", from_account="Bank", to_account="Cash", amount=7000),
        )

    def test_a_typed_yes_confirms_the_waiting_transfer(self):
        self.pending()
        reply = self.say(self.chat, "haan kar do", "confirm")  # the LLM reads the yes
        self.assertIn("7,000", reply.text)
        self.assertEqual(balance(self.chat, "Cash"), 7000.0)

    def test_a_typed_no_cancels_and_moves_nothing(self):
        self.pending()
        reply = self.say(self.chat, "nahi, rehne do", "cancel")
        self.assertIn("Cancelled", reply.text)
        self.assertEqual(balance(self.chat, "Cash"), 0.0)
        self.assertFalse(run(agent.has_pending(self.chat)))

    def test_buttons_carry_the_id_of_their_question(self):
        reply = self.pending()
        data = [b.data for row in reply.rows for b in row]
        self.assertTrue(all(d.count(":") == 2 for d in data[:2]), data)

    def test_an_old_yes_button_cannot_confirm_a_newer_question(self):
        old = [b.data for row in self.pending().rows for b in row][0]
        self.say(  # a newer, different question replaces the old one
            self.chat,
            "transfer 6000 from bank to cash",
            tool("transfer_funds", from_account="Bank", to_account="Cash", amount=6000),
        )
        stale = run(handle_callback(self.chat, old))
        self.assertIn("no longer waiting", stale.text)
        self.assertEqual(balance(self.chat, "Cash"), 0.0)

    def test_the_right_button_confirms(self):
        data = [b.data for row in self.pending().rows for b in row][0]
        done = run(handle_callback(self.chat, data))
        self.assertIn("7,000", done.text)
        self.assertEqual(balance(self.chat, "Cash"), 7000.0)

    def test_the_cancel_button_moves_nothing(self):
        data = [b.data for row in self.pending().rows for b in row][1]
        done = run(handle_callback(self.chat, data))
        self.assertIn("Cancelled", done.text)
        self.assertEqual(balance(self.chat, "Cash"), 0.0)

    def test_a_command_in_between_cancels_the_question(self):
        self.pending()
        self.say(self.chat, "/balance")
        self.assertFalse(run(agent.has_pending(self.chat)))
        reply = self.say(self.chat, "yes", "Nothing is waiting for an answer.")
        self.assertNotIn("7,000", reply.text)
        self.assertEqual(balance(self.chat, "Cash"), 0.0)

    def test_commands_split_on_any_whitespace(self):
        reply = self.say(self.chat, "/history\t1")
        self.assertNotIn("don't know", reply.text)

    def test_risky_tool_calls_wait_for_a_yes(self):
        reason = agent.confirmation_reason(
            "edit_transaction", {"transaction_id": "abcdef123456", "amount": 5}
        )
        self.assertIn("Change entry abcdef12", reason)
        self.assertIn(
            "Delete entry abcdef12",
            agent.confirmation_reason("delete_transaction", {"transaction_id": "abcdef123456"}),
        )
        self.assertIsNotNone(
            agent.confirmation_reason(
                "set_account_balance", {"account_name": "Cash", "balance": 500}
            )
        )
        self.assertIsNone(agent.confirmation_reason("log_expense", {"amount": 9000}))
        self.assertIsNotNone(
            agent.confirmation_reason(
                "transfer_funds", {"amount": "7,000", "from_account": "A", "to_account": "B"}
            )
        )
        self.assertIsNone(agent.confirmation_reason("transfer_funds", "not a dict"))
        self.assertIsNone(  # below the threshold: runs straight away
            agent.confirmation_reason(
                "transfer_funds", {"amount": 4999, "from_account": "A", "to_account": "B"}
            )
        )

    def test_the_ai_cannot_save_keys(self):
        self.assertNotIn("save_user_api_key", {t["function"]["name"] for t in agent.TOOLS_SCHEMA})
        self.assertIn("save_user_api_key", agent.HIDDEN_TOOLS)


class PlannerLoopTests(unittest.TestCase):
    """The planner/guard/executor loop: a write runs once, and risky calls all wait together."""

    def setUp(self):
        self.chat = chat()
        seed(self.chat, "Bank", 30000)
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, self.chat).id
            ledger.record_transaction(db, uid, 10000, "Cash", "Seed", tx_type="income")

    def test_a_write_with_an_empty_final_answer_is_not_run_twice(self):
        with FakeLLM() as llm:
            llm.add(
                [
                    tool("log_expense", amount=200, category="Food & Dining", account_name="Cash"),
                    tool("get_pipe_balances"),
                ]
            )
            llm.add("")  # the planner's final answer is empty: the renderer falls back to templates
            reply = run(bot.handle_incoming(self.chat, "log 200 lunch and show balances please"))
        self.assertTrue(reply.text.strip())
        self.assertEqual(balance(self.chat, "Cash"), 9800.0)  # once, not twice

    def test_two_big_transfers_both_wait_and_nothing_runs_first(self):
        with FakeLLM() as llm:
            llm.add(
                [
                    tool("log_expense", amount=50, category="Food & Dining", account_name="Bank"),
                    tool("transfer_funds", from_account="Bank", to_account="Cash", amount=6000),
                    tool("transfer_funds", from_account="Bank", to_account="Savings", amount=7000),
                ]
            )
            outcome = run(agent.process_user_interaction_ex(self.chat, "move money and note chai"))
        self.assertTrue(outcome.needs_confirmation)
        self.assertIn("6,000", outcome.text)
        self.assertIn("7,000", outcome.text)
        # Nothing ran yet, not even the small expense that shares the message.
        self.assertEqual(balance(self.chat, "Bank"), 30000.0)
        self.assertEqual(balance(self.chat, "Savings"), None)

    def test_confirming_runs_every_call_from_that_message(self):
        with FakeLLM() as llm:
            llm.add(
                [
                    tool("log_expense", amount=50, category="Food & Dining", account_name="Bank"),
                    tool("transfer_funds", from_account="Bank", to_account="Cash", amount=6000),
                ]
            )
            outcome = run(agent.process_user_interaction_ex(self.chat, "chai 50 and move 6000"))
            self.assertTrue(outcome.needs_confirmation)
            done = run(agent.resume_pending(self.chat, True, outcome.pending_id))
        self.assertIsNotNone(done)
        self.assertEqual(balance(self.chat, "Bank"), 30000.0 - 50 - 6000)
        self.assertEqual(balance(self.chat, "Cash"), 16000.0)


class ButtonTests(unittest.TestCase):
    def test_key_buttons_carry_the_key_id_not_its_position(self):
        rows = ui.keys_rows(["11111111-aaaa", "22222222-bbbb"])
        data = [b.data for row in rows for b in row]
        self.assertIn("key:del:k11111111", data)
        self.assertIn("key:del:k22222222", data)
        self.assertNotIn("key:del:1", data)

    def test_deleting_by_id_twice_removes_only_one_key(self):
        from app.bot import keys as keys_mod

        chat_id = chat()
        with patch.object(gw, "add_validated_keys"):  # keys are inserted directly: no live test
            for provider, key in (
                ("gemini", "AIza" + "A" * 35),
                ("groq", "gsk_" + "b" * 30),
                ("openai", "sk-proj-" + "c" * 40),
            ):
                gw.add_api_key(chat_id, provider, key)
        before = gw.list_api_keys(chat_id)
        target = "k" + before[0]["key_id"][:8]
        run(keys_mod.handle_key_command(chat_id, "/delkey", target))
        again = run(keys_mod.handle_key_command(chat_id, "/delkey", target))  # a stale second tap
        self.assertEqual(len(gw.list_api_keys(chat_id)), 2)
        self.assertNotIn("removed", again)

    def test_history_undo_button_names_the_entry_so_a_double_tap_is_harmless(self):
        chat_id = chat()
        seed(chat_id, "Cash", 10000)
        with FakeLLM(
            tool("log_expense", amount=100, category="Food & Dining", account_name="Cash"),
            tool("log_expense", amount=200, category="Transport", account_name="Cash"),
        ):
            run(bot.handle_incoming(chat_id, "Spent 100 on chai via Cash"))
            run(bot.handle_incoming(chat_id, "Spent 200 on auto via Cash"))
        history = run(bot.handle_incoming(chat_id, "/history"))
        undo = next(b.data for row in history.rows for b in row if b.data.startswith("undo:"))
        self.assertNotEqual(undo, "undo:last")
        run(handle_callback(chat_id, undo))
        run(handle_callback(chat_id, undo))  # second tap on the same message
        self.assertEqual(balance(chat_id, "Cash"), 9900.0)  # only the auto was undone

    def test_an_expired_import_button_says_so(self):
        reply = run(bot.answer_pending_import(chat(), "confirmed"))
        self.assertIn("expired", reply.text)

    def test_an_expired_action_button_says_so(self):
        reply = run(bot.answer_pending_action(chat(), True, "deadbe"))
        self.assertIn("no longer waiting", reply.text)


class StartupTests(unittest.TestCase):
    def test_one_unreachable_bot_does_not_stop_the_others(self):
        from app.bot import telegram
        from app.config import settings

        class FakeBot:
            def __init__(self, token):
                self.token = token
                self.session = SimpleNamespace(close=lambda: asyncio.sleep(0))

            async def get_me(self):
                if self.token.startswith("2222"):
                    raise ConnectionError("no route")
                return SimpleNamespace(username="ok")

            async def delete_webhook(self, **kw):
                pass

        async def go():
            with (
                patch.object(settings, "telegram_bot_token", "1111:AAA"),
                patch.object(settings, "telegram_bot_tokens", "2222:BBB"),
                patch.object(settings, "use_webhook", False),
                patch.object(telegram, "Bot", FakeBot),
                patch.object(telegram, "_publish_profile"),
                patch.object(
                    telegram.Dispatcher, "start_polling", lambda *a, **k: asyncio.sleep(0)
                ),
            ):
                mode = await telegram.start()
                running = telegram.running_bots()
                await telegram.stop()
                return mode, running

        self.assertEqual(asyncio.run(go()), ("polling", 1))


if __name__ == "__main__":
    unittest.main()
