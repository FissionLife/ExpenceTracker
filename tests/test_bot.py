"""Tests for Role 1: Telegram entry point, /setkey, statement import, settings, ledger locks.

Offline only: no Telegram network calls and no real LLM calls.
Run with:  uv run python -m unittest discover -s tests -v
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-bot-test-")
os.environ["DYNAMIC_MODELS"] = "false"  # tests never hit provider model-list APIs
os.environ.setdefault(
    "DATABASE_URL", "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
)
# Never reach real Telegram from tests, even if a developer's .env has a token.
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["USE_WEBHOOK"] = "false"
for _var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "OPENAI_API_KEYS", "OPENAI_API_KEY"):
    os.environ.pop(_var, None)

import asyncio  # noqa: E402
import io  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

from aiogram.exceptions import TelegramBadRequest  # noqa: E402
from aiogram.methods import SendMessage  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.bot import main as bot  # noqa: E402
from app.bot import telegram  # noqa: E402
from app.config import Settings, settings  # noqa: E402
from app.db import SessionLocal, engine, init_db  # noqa: E402
from app.models import Transaction, User  # noqa: E402
from app.services_ai import ai_gateway as gw  # noqa: E402
from app.services_ai.mcp_server import execute_tool  # noqa: E402

try:
    from tests.fake_llm import FakeLLM, tool  # noqa: E402
except ImportError:  # discover -s tests (no -t .) puts tests/ on sys.path directly
    from fake_llm import FakeLLM, tool  # noqa: E402

AQ_KEY = "AQ.Ab8RN6" + "z" * 40 + "Q1w2"


def run(coro):
    return asyncio.run(coro)


def make_pdf(lines: list[str]) -> bytes:
    """Build a minimal text-based PDF (one line per row) without extra dependencies."""
    content = "BT /F1 10 Tf 40 800 Td 14 TL " + " ".join(
        "(" + line.replace("(", "[").replace(")", "]") + ") '" for line in lines
    )
    content += " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1"))
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
    out.write(f"startxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


STATEMENT_LINES = [
    "02/10/2026 Paid to Swiggy Rs. 450.50 debited",
    "03/10/2026 Paid to Uber Rs. 220.00 debited",
    "05/10/2026 Salary credited Rs. 50,000.00 received from ACME",
]


class BotTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    @classmethod
    def tearDownClass(cls):
        engine.dispose()

    def setUp(self):
        self.chat_id = str(uuid.uuid4().int)[:9]
        bot._pending_imports.clear()
        gw.reset_rotation_state()

    def balances(self) -> dict[str, float]:
        result = execute_tool("get_pipe_balances", user_id=self.chat_id)
        return {a["name"]: a["balance"] for a in result["accounts"]}


class TestCommands(BotTestCase):
    def test_start_help_balance_report(self):
        self.assertIn("Welcome to WalletLedger", run(bot.handle_text(self.chat_id, "/start")))
        self.assertIn("/setkey", run(bot.handle_text(self.chat_id, "/help")))
        self.assertIn("Total", run(bot.handle_text(self.chat_id, "/balance")))
        self.assertIn("Last 30 days", run(bot.handle_text(self.chat_id, "/report")))

    def test_command_with_bot_suffix(self):
        self.assertIn("Total", run(bot.handle_text(self.chat_id, "/balance@MyBot")))

    def test_setkey_saves_and_masks(self):
        with patch.object(gw, "validate_api_key", AsyncMock(return_value=gw.KeyCheck("valid"))):
            reply = run(bot.handle_text(self.chat_id, f"/setkey {AQ_KEY}"))
        self.assertIn("works. Saved", reply)
        self.assertIn(gw.mask_key(AQ_KEY), reply)
        self.assertNotIn(AQ_KEY, reply)
        self.assertEqual(gw.get_user_api_keys(self.chat_id), [("gemini", AQ_KEY)])
        keys = run(bot.handle_text(self.chat_id, "/keys"))
        self.assertIn(gw.mask_key(AQ_KEY), keys)

    def test_setkey_usage_and_invalid(self):
        self.assertIn("Add your AI key", run(bot.handle_text(self.chat_id, "/setkey")))
        self.assertIn(
            "couldn't find a supported API key", run(bot.handle_text(self.chat_id, "/setkey hello"))
        )

    def test_should_delete_message(self):
        self.assertTrue(bot.should_delete_message(f"/setkey {AQ_KEY}"))
        self.assertTrue(bot.should_delete_message(f"my key is {AQ_KEY}"))
        self.assertFalse(bot.should_delete_message("spent 200 on food"))

    def test_other_text_goes_to_agent(self):
        with FakeLLM(tool("log_income", amount=1000, source_account="Bank", category="Salary")):
            reply = run(bot.handle_text(self.chat_id, "Received 1000 salary in Bank"))
        self.assertIn("+₹1,000", reply)
        self.assertEqual(self.balances()["Bank"], 1000)

    def test_free_text_without_an_ai_key_records_nothing(self):
        with FakeLLM(no_key=True):
            reply = run(bot.handle_text(self.chat_id, "Received 1000 salary in Bank"))
        self.assertIn("/setkey", reply)
        self.assertEqual(self.balances().get("Bank", 0), 0)


class TestStatementImport(BotTestCase):
    def test_pdf_preview_then_yes_imports_with_pdf_source(self):
        reply = run(bot.handle_document(self.chat_id, "phonepe.pdf", make_pdf(STATEMENT_LINES)))
        self.assertIn("Found **3 transactions**", reply)
        self.assertIn("₹670.50 spent", reply)
        self.assertIn("₹50,000.00 received", reply)
        self.assertEqual(self.balances()["Bank"], 0)

        with FakeLLM("confirm"):  # the typed yes is classified by the AI
            reply = run(bot.handle_text(self.chat_id, "yes"))
        self.assertIn("Imported 3 transactions", reply)
        self.assertEqual(
            self.balances()["Bank"], 0
        )  # "yes" imports history: today's balance is not rewritten
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.telegram_chat_id == self.chat_id).one()
            txs = list(db.query(Transaction).filter_by(user_id=user.id))
            sources = {t.source for t in txs}
            dates = sorted(t.created_at.date().isoformat() for t in txs)
        finally:
            db.close()
        self.assertEqual(sources, {"pdf"})
        # imported rows keep the statement's own dates (not the day of the import)
        self.assertEqual(dates, ["2026-10-02", "2026-10-03", "2026-10-05"])
        self.assertIn("₹670", run(bot.handle_text(self.chat_id, "/report")))

    def test_no_cancels_import(self):
        run(bot.handle_document(self.chat_id, "s.pdf", make_pdf(STATEMENT_LINES)))
        with FakeLLM("cancel"):
            self.assertIn("cancelled", run(bot.handle_text(self.chat_id, "no")))
        self.assertEqual(self.balances()["Bank"], 0)

    def test_expired_import_is_not_applied(self):
        run(bot.handle_document(self.chat_id, "s.pdf", make_pdf(STATEMENT_LINES)))
        bot._pending_imports[self.chat_id].created_at = datetime.utcnow() - timedelta(minutes=11)
        with FakeLLM("confirm"):
            run(bot.handle_text(self.chat_id, "yes"))
        self.assertEqual(self.balances()["Bank"], 0)

    def test_bad_files(self):
        self.assertIn("PDF", run(bot.handle_document(self.chat_id, "photo.jpg", b"x")))
        self.assertIn("Could not read", run(bot.handle_document(self.chat_id, "a.pdf", b"nope")))
        empty = run(bot.handle_document(self.chat_id, "e.pdf", make_pdf(["hello world"])))
        self.assertIn("couldn't find any transactions", empty)


class TestHttpEndpoints(BotTestCase):
    def setUp(self):
        super().setUp()
        from main import app

        # Settings may already be loaded from a developer's .env: force the bot off.
        for name, value in (("telegram_bot_token", ""), ("use_webhook", False)):
            patcher = patch.object(settings, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(app)

    def test_simulate_chat(self):
        with self.client as client:
            res = client.post(
                "/bot/simulate_chat", json={"chat_id": self.chat_id, "text": "/balance"}
            )
            self.assertEqual(res.status_code, 200)
            self.assertIn("Total", res.json()["reply"])
            self.assertEqual(client.get("/health").json()["telegram"], "off")

    def test_dev_webhook_returns_reply(self):
        update = {"update_id": 1, "message": {"chat": {"id": int(self.chat_id)}, "text": "/help"}}
        with self.client as client:
            res = client.post("/bot/webhook", json=update)
        self.assertIn("just type", res.json()["reply"])

    def test_webhook_rejects_wrong_secret(self):
        with (
            patch.object(telegram, "webhook_active", return_value=True),
            patch.object(telegram, "feed_webhook_update", AsyncMock()) as feed,
            self.client as client,
        ):
            bad = client.post("/bot/webhook", json={"update_id": 1})
            good = client.post(
                "/bot/webhook",
                json={"update_id": 2},
                headers={"X-Telegram-Bot-Api-Secret-Token": telegram.webhook_secret()},
            )
        self.assertEqual(bad.status_code, 403)
        self.assertEqual(good.status_code, 200)
        feed.assert_awaited_once()


class TestTelegramHelpers(unittest.TestCase):
    def test_split_message(self):
        text = ("line of text\n" * 1000).strip()
        chunks = telegram.split_message(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= telegram.TELEGRAM_LIMIT for c in chunks))
        self.assertEqual("".join(chunks).count("line of text"), 1000)
        self.assertEqual(telegram.split_message("short"), ["short"])

    def test_markdown_conversion(self):
        self.assertEqual(telegram.to_telegram_markdown("**Total:** ₹5"), "*Total:* ₹5")

    def test_send_reply_falls_back_to_plain_text(self):
        error = TelegramBadRequest(SendMessage(chat_id=1, text="x"), "can't parse entities")
        message = SimpleNamespace(answer=AsyncMock(side_effect=[error, None]))
        run(telegram.send_reply(message, "**bad _markdown"))
        self.assertEqual(message.answer.await_count, 2)
        self.assertIsNone(message.answer.await_args.kwargs["parse_mode"])

    def test_restart_builds_fresh_router(self):
        """start() after stop() in the same process must not reuse an attached router."""
        fake_bot = SimpleNamespace(
            get_me=AsyncMock(return_value=SimpleNamespace(username="test_bot")),
            delete_webhook=AsyncMock(),
            session=SimpleNamespace(close=AsyncMock()),
        )

        async def cycle():
            with (
                patch.object(settings, "telegram_bot_token", "123:abc"),
                patch.object(telegram, "Bot", return_value=fake_bot),
                patch("aiogram.Dispatcher.start_polling", AsyncMock()),
            ):
                for _ in range(2):
                    self.assertEqual(await telegram.start(), "polling")
                    await telegram.stop()

        run(cycle())

    def test_start_without_token_is_off(self):
        with patch.object(settings, "telegram_bot_token", ""):
            self.assertEqual(run(telegram.start()), "off")


class TestSettingsAndLedger(unittest.TestCase):
    def test_unknown_env_lines_are_ignored_and_ai_settings_load(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as fh:
            fh.write("SOME_UNKNOWN_SETTING=1\nGEMINI_API_KEYS=k1,k2\nAI_MODEL_GEMINI=gemini/x\n")
        try:
            loaded = Settings(_env_file=fh.name)
        finally:
            os.unlink(fh.name)
        self.assertEqual(loaded.gemini_api_keys, "k1,k2")
        self.assertEqual(loaded.ai_model_gemini, "gemini/x")

    def test_gateway_reads_keys_from_settings(self):
        os.environ.pop("GEMINI_API_KEYS", None)
        key = "AIza" + "S" * 35
        with patch.object(settings, "gemini_api_keys", key):
            self.assertEqual(gw._server_keys("gemini"), [key])
        with patch.dict(os.environ, {"GEMINI_API_KEYS": "envkey"}):
            self.assertEqual(gw._server_keys("gemini"), ["envkey"], "real env var wins")

    def test_ledger_locks_account_rows(self):
        from sqlalchemy import event

        statements: list[str] = []
        chat_id = str(uuid.uuid4().int)[:9]

        def capture(conn, cursor, statement, *args):
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            from app.services.ledger import record_transaction, transfer_between_pipes

            db = SessionLocal()
            try:
                user = User(telegram_chat_id=chat_id)
                db.add(user)
                db.commit()
                record_transaction(db, user.id, 1000, "Bank", tx_type="income")
                transfer_between_pipes(db, user.id, "Bank", "Cash", 10)
            finally:
                db.close()
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        from sqlalchemy import select
        from sqlalchemy.dialects import postgresql

        from app.models import Account

        sql = str(select(Account).with_for_update().compile(dialect=postgresql.dialect()))
        self.assertIn("FOR UPDATE", sql)
        self.assertTrue(any("FROM accounts" in s for s in statements))


if __name__ == "__main__":
    unittest.main()
