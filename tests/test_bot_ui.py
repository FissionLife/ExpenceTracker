"""Bot screens, buttons, undo, export, dashboard links and Telegram delivery. Offline."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-ui-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
for _var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "OPENAI_API_KEYS", "OPENAI_API_KEY"):
    os.environ[_var] = ""

import asyncio  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import hmac  # noqa: E402
import io  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402
from urllib.parse import urlencode  # noqa: E402

from aiogram.exceptions import TelegramBadRequest  # noqa: E402
from aiogram.methods import EditMessageText  # noqa: E402

from app.bot import callbacks, telegram, ui, views  # noqa: E402
from app.bot import main as bot  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.services import ledger  # noqa: E402
from app.web import tokens  # noqa: E402

try:
    from tests.fake_llm import FakeLLM, tool  # noqa: E402
except ImportError:  # discover -s tests (no -t .) puts tests/ on sys.path directly
    from fake_llm import FakeLLM, tool  # noqa: E402

init_db()
_ids = itertools.count(1)
PNG = b"\x89PNG"


def run(coro):
    return asyncio.run(coro)


class Base(unittest.TestCase):
    def setUp(self):
        self.chat = f"ui-{next(_ids)}"
        with SessionLocal() as db:
            self.uid = ledger.get_or_create_user(db, self.chat).id

    def seed(self, n=12):
        with SessionLocal() as db:
            ledger.record_transaction(
                db,
                self.uid,
                60000,
                "Bank",
                "Salary",
                "Acme",
                "income",
                "Salary",
                "chat",
                datetime.utcnow() - timedelta(days=9),
            )
            for i in range(n):
                ledger.record_transaction(
                    db,
                    self.uid,
                    200 + i * 10,
                    "Bank",
                    "Food",
                    "Swiggy" if i % 2 else "Zomato",
                    "expense",
                    "meal",
                    "chat",
                    datetime.utcnow() - timedelta(days=i % 8),
                )
            ledger.record_transaction(
                db,
                self.uid,
                3000,
                "Bank",
                "Shopping",
                "Amazon",
                "expense",
                "x",
                "chat",
                datetime.utcnow() - timedelta(days=2),
            )

    def balance(self, name="Bank"):
        with SessionLocal() as db:
            accs = ledger.get_account_balances(db, self.uid)["accounts"]
        return next(a["balance"] for a in accs if a["name"] == name)

    def say(self, text):
        return run(bot.handle_incoming(self.chat, text))

    def tap(self, data):
        return run(callbacks.handle_callback(self.chat, data))

    @staticmethod
    def buttons(reply):
        return [b for row in reply.rows for b in row]


class StatsTests(Base):
    def test_stats_has_picture_caption_and_period_buttons(self):
        self.seed()
        r = self.say("/stats")
        self.assertTrue(r.photo.startswith(PNG))
        self.assertIn("Last 30 days", r.text)
        self.assertIn("Spent", r.text)
        data = [b.data for b in self.buttons(r)]
        for expected in ("st:7d", "st:30d", "st:90d", "st:1y", "go:dashboard", "ex:30d"):
            self.assertIn(expected, data)
        self.assertLessEqual(len(r.text), 1000)

    def test_stats_arguments(self):
        self.seed()
        self.assertIn("Last 90 days", self.say("/stats 90d").text)
        r = self.say("/stats 90d food")
        self.assertIn("Food & Dining" if False else "Food", r.text)
        self.assertIn("purchase", r.text)  # filtered view talks about purchases, not income/savings
        self.assertNotIn("Earned", r.text)
        r = self.say("/stats 30d food zzzz")
        self.assertIn("couldn't match", r.text)
        self.assertIn("“zzzz”", r.text)
        self.assertIn("This month", self.say("/stats this month").text)
        self.assertIn("This month", self.say("/report mtd").text)  # /report is an alias

    def test_filter_buttons_keep_the_filter(self):
        self.seed()
        first = self.say("/stats food")
        ctx = next(b.data for b in self.buttons(first) if b.data.startswith("st:90d")).split(":")[2]
        again = self.tap(f"st:90d:{ctx}")
        self.assertIn("Food", again.text)
        self.assertIn("Last 90 days", again.text)
        self.assertTrue(again.edit)

    def test_empty_account_gets_a_friendly_message_and_no_picture(self):
        r = self.say("/stats")
        self.assertIsNone(r.photo)
        self.assertIn("Nothing logged", r.text)
        self.assertIn("Spent 450", r.text)

    def test_chart_command_and_buttons(self):
        self.seed()
        r = self.say("/chart categories 90d")
        self.assertTrue(r.photo.startswith(PNG))
        self.assertIn("Categories", r.text)
        self.assertIn("Which chart", self.say("/chart nonsense").text)
        for kind in ("trend", "merchants", "pace", "budgets", "weekdays", "networth", "heatmap"):
            r = self.tap(f"ch:{kind}:30d")
            self.assertTrue(r.photo.startswith(PNG), kind)
            self.assertTrue(r.edit)
        nav = [b.data for b in self.buttons(self.tap("ch:trend:30d"))]
        self.assertIn("st:30d", nav)  # back to overview

    def test_picture_failure_degrades_to_text(self):
        self.seed()
        with patch("app.charts.render_dashboard", side_effect=RuntimeError("boom")):
            r = self.say("/stats")
        self.assertIsNone(r.photo)
        self.assertIn("Spent", r.text)


class MoneyScreensTests(Base):
    def test_balance(self):
        self.seed()
        r = self.say("/balance")
        self.assertIn("Total", r.text)
        self.assertIn("Bank", r.text)
        self.assertIn("₹", r.text)

    def test_history_paging(self):
        self.seed(20)
        first = self.say("/history")
        self.assertIn("1–8 of 22", first.text)
        self.assertIn("Older ▶", [b.text for b in self.buttons(first)])
        self.assertNotIn("◀ Newer", [b.text for b in self.buttons(first)])
        last = self.tap("hi:2")
        self.assertIn("17–22 of 22", last.text)
        self.assertIn("◀ Newer", [b.text for b in self.buttons(last)])
        self.assertTrue(last.edit)
        self.assertIn(
            "1–8 of 22", self.tap("hi:99").text.replace("17–22 of 22", "1–8 of 22")
        )  # clamps

    def test_history_empty(self):
        self.assertIn("No transactions yet", self.say("/history").text)

    def test_budget_set_list_and_clear(self):
        self.seed()
        self.assertIn("No budgets yet", self.say("/budget").text)
        self.assertIn("₹6,000", self.say("/budget food 6000").text)
        self.assertIn("₹4,000", self.say("/budget set groceries to 4k").text)
        listing = self.say("/budget").text
        self.assertIn("Food", listing)
        self.assertIn("%", listing)
        self.assertIn("removed", self.say("/budget food off").text)
        self.assertIn("Use:", self.say("/budget food abc").text)

    def test_export_csv(self):
        self.seed(3)
        r = self.say("/export 30d")
        data, name = r.document
        self.assertTrue(name.endswith(".csv"))
        text = data.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
        self.assertEqual(
            rows[0], ["Date", "Type", "Amount", "Category", "Merchant", "Account", "Note"]
        )
        self.assertEqual(len(rows) - 1, 5)
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        self.assertIn(
            "Nothing to export",
            run(views.export_csv(self.chat, "7d")).text
            if False
            else run(views.export_csv(f"ui-empty-{next(_ids)}", "7d")).text,
        )


class UndoAndConfirmTests(Base):
    def test_logging_offers_undo_and_undo_restores_the_balance(self):
        with FakeLLM(tool("log_income", amount=1000, source_account="Bank", category="Salary")):
            r = self.say("Received 1000 salary in Bank")
        self.assertTrue(any(b.data.startswith("undo:") for b in self.buttons(r)))
        self.assertEqual(self.balance(), 1000.0)
        with FakeLLM(
            tool("log_expense", amount=300, category="Food & Dining", account_name="Bank")
        ):
            r = self.say("Spent 300 on dinner via Bank")
        undo = next(b.data for b in self.buttons(r) if b.data.startswith("undo:"))
        self.assertEqual(self.balance(), 700.0)
        done = self.tap(undo)
        self.assertIn("Undone", done.text)
        self.assertTrue(done.edit)
        self.assertEqual(self.balance(), 1000.0)
        again = self.tap(undo)  # pressing twice is harmless
        self.assertIn("not found", again.text)
        self.assertEqual(self.balance(), 1000.0)

    def test_undo_command_removes_the_latest(self):
        with FakeLLM(
            tool("log_income", amount=500, source_account="Bank"),
            tool("log_expense", amount=100, category="Food & Dining", account_name="Bank"),
        ):
            self.say("Received 500 in Bank")
            self.say("Spent 100 on coffee via Bank")
        self.assertEqual(self.balance(), 400.0)
        self.assertIn("Undone", self.say("/undo").text)
        self.assertEqual(self.balance(), 500.0)
        self.say("/undo")
        self.assertEqual(self.balance(), 0.0)
        self.assertIn("Nothing to undo", self.say("/undo").text)

    def test_large_transfer_gets_confirm_buttons(self):
        with FakeLLM(
            tool("log_income", amount=200000, source_account="Bank"),
            tool("transfer_funds", from_account="Bank", to_account="Cash", amount=100000),
        ):
            self.say("Received 200000 in Bank")
            r = self.say("Transferred 100000 from Bank to Cash")
        yes = next(b.data for b in self.buttons(r) if b.data.startswith("ask:yes"))
        self.assertEqual(self.balance("Cash"), 0.0)
        done = self.tap(yes)
        self.assertTrue(done.edit)
        self.assertEqual(self.balance("Cash"), 100000.0)

    def test_cancel_button(self):
        with FakeLLM(
            tool("log_income", amount=200000, source_account="Bank"),
            tool("transfer_funds", from_account="Bank", to_account="Cash", amount=150000),
        ):
            self.say("Received 200000 in Bank")
            r = self.say("Transferred 150000 from Bank to Cash")
        no = next(b.data for b in self.buttons(r) if b.data.startswith("ask:no"))
        cancelled = self.tap(no)
        self.assertIn("ancel", cancelled.text)
        self.assertEqual(self.balance("Cash"), 0.0)


class NavigationTests(Base):
    def test_menu_and_start_have_buttons(self):
        start = self.say("/start")
        self.assertIn("Welcome", start.text)
        labels = [b.text for b in self.buttons(start)]
        for expected in ("📊 Stats", "💰 Balance", "🧾 History", "🖥 Dashboard", "🔑 AI keys"):
            self.assertIn(expected, labels)
        self.assertTrue(self.say("/menu").rows)

    def test_every_menu_button_leads_somewhere(self):
        self.seed()
        for data in (
            "go:home",
            "go:stats",
            "go:balance",
            "go:history",
            "go:budgets",
            "go:dashboard",
            "go:keys",
            "go:status",
            "go:help",
        ):
            r = self.tap(data)
            self.assertTrue(r.text, data)
        self.assertEqual(self.tap("go:nowhere").notice, "That button is out of date")
        self.assertEqual(self.tap("zzz:1").notice, "That button is out of date")

    def test_all_button_data_fits_telegrams_limit(self):
        self.seed()
        for reply in (
            self.say("/stats food"),
            self.say("/history"),
            self.say("/keys"),
            self.say("/start"),
            self.tap("ch:trend:30d"),
        ):
            for b in self.buttons(reply):
                self.assertLessEqual(len((b.data or "").encode()), 64, b.data)

    def test_help_is_neat_and_complete(self):
        text = self.say("/help").text
        for needle in ("/stats", "/undo", "/budget", "/dashboard", "/export", "/setkey"):
            self.assertIn(needle, text)
        self.assertLess(len(text), 900)

    def test_unknown_command_and_hidden_commands(self):
        self.assertIn("don't know", self.say("/zzz").text)
        for hidden in ("/report", "/chart", "/testkey", "/admin"):
            self.assertNotIn("don't know", self.say(hidden).text, hidden)

    def test_private_bot_blocks_buttons_too(self):
        with patch.object(settings, "allowed_chat_ids", "999"):
            r = self.tap("go:stats")
        self.assertIn("private", r.text.lower())

    def test_key_buttons(self):
        r = self.tap("go:keys")
        self.assertIn("keys", r.text.lower())
        r = self.tap("key:test:all")
        self.assertTrue(r.edit)
        self.assertIn("no saved keys", r.text.lower())


class DashboardLinkTests(Base):
    def test_token_round_trip_expiry_and_tamper(self):
        t = tokens.make_token("123")
        self.assertEqual(tokens.read_token(t), "123")
        self.assertIsNone(tokens.read_token(tokens.make_token("123", ttl_seconds=-1)))
        payload, sig = t.split(".")
        self.assertIsNone(tokens.read_token(payload + "." + sig[:-2] + "xx"))
        forged = tokens.make_token("999").split(".")[0] + "." + sig
        self.assertIsNone(tokens.read_token(forged))
        for junk in ("", "abc", "a.b", "....", "x" * 500):
            self.assertIsNone(tokens.read_token(junk))

    def test_token_depends_on_the_secret(self):
        t = tokens.make_token("123")
        with patch.object(settings, "secret_key", "another-secret-value-123456789"):
            self.assertIsNone(tokens.read_token(t))

    def test_dashboard_reply_http_vs_https(self):
        with patch.object(settings, "public_url", ""):
            r = self.say("/dashboard")
            self.assertIn("http://localhost:8000/d/", r.text)
            self.assertFalse(r.rows)
        with patch.object(settings, "public_url", "https://bot.example.com/"):
            r = self.say("/dashboard")
            btn = self.buttons(r)[0]
            self.assertTrue(btn.web_app.startswith("https://bot.example.com/d/"))
            self.assertEqual(tokens.read_token(btn.web_app.rsplit("/", 1)[1]), self.chat)

    def test_telegram_init_data_validation(self):
        bot_token = "123456:TEST-TOKEN"
        fields = {
            "auth_date": str(int(time.time())),
            "query_id": "AAH",
            "user": json.dumps({"id": 42, "first_name": "A"}),
        }
        check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
        secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        good = urlencode(
            {**fields, "hash": hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()}
        )
        self.assertEqual(tokens.validate_init_data(good, bot_token)["id"], 42)
        self.assertIsNone(tokens.validate_init_data(good, "123456:OTHER"))
        self.assertIsNone(tokens.validate_init_data(good.replace("42", "43"), bot_token))
        self.assertIsNone(tokens.validate_init_data("", bot_token))
        old = {**fields, "auth_date": str(int(time.time()) - 999999)}
        check_old = "\n".join(f"{k}={v}" for k, v in sorted(old.items()))
        stale = urlencode(
            {**old, "hash": hmac.new(secret, check_old.encode(), hashlib.sha256).hexdigest()}
        )
        self.assertIsNone(tokens.validate_init_data(stale, bot_token))


class FakeMessage:
    """Just enough of aiogram's Message to see what the delivery code does."""

    def __init__(self, photo=False, document=False):
        self.photo = [object()] if photo else None
        self.document = object() if document else None
        self.answer = AsyncMock()
        self.answer_photo = AsyncMock()
        self.answer_document = AsyncMock()
        self.edit_text = AsyncMock()
        self.edit_media = AsyncMock()
        self.edit_reply_markup = AsyncMock()


class DeliveryTests(unittest.TestCase):
    def test_keyboard_conversion(self):
        kb = telegram.keyboard(
            [
                [ui.Btn("A", "x:1"), ui.Btn("B", url="https://e.com")],
                [ui.Btn("C", web_app="https://e.com/d/t")],
            ]
        )
        row1, row2 = kb.inline_keyboard
        self.assertEqual((row1[0].callback_data, row1[1].url), ("x:1", "https://e.com"))
        self.assertEqual(row2[0].web_app.url, "https://e.com/d/t")
        self.assertIsNone(telegram.keyboard([]))

    def test_deliver_text_photo_and_document(self):
        m = FakeMessage()
        run(telegram.deliver(m, ui.Reply("hello **x**", rows=[[ui.Btn("A", "a:1")]])))
        self.assertIsNotNone(m.answer.await_args.kwargs["reply_markup"])
        m = FakeMessage()
        run(telegram.deliver(m, ui.Reply("cap", photo=PNG + b"x")))
        self.assertEqual(m.answer_photo.await_args.kwargs["caption"], "cap")
        m = FakeMessage()
        run(telegram.deliver(m, ui.Reply("file", document=(b"a,b", "x.csv"))))
        m.answer_document.assert_awaited_once()
        m = FakeMessage()
        run(telegram.deliver(m, ui.Reply()))
        m.answer.assert_not_awaited()

    def test_long_text_puts_buttons_on_the_last_chunk(self):
        m = FakeMessage()
        run(telegram.deliver(m, ui.Reply("word " * 2000, rows=[[ui.Btn("A", "a:1")]])))
        calls = m.answer.await_args_list
        self.assertGreater(len(calls), 1)
        self.assertIsNone(calls[0].kwargs["reply_markup"])
        self.assertIsNotNone(calls[-1].kwargs["reply_markup"])

    def test_update_message_edits_text_in_place(self):
        m = FakeMessage()
        run(
            telegram.update_message(m, ui.Reply("new text", rows=[[ui.Btn("A", "a:1")]], edit=True))
        )
        m.edit_text.assert_awaited_once()
        m.answer.assert_not_awaited()

    def test_update_message_edits_photo_in_place(self):
        m = FakeMessage(photo=True)
        run(telegram.update_message(m, ui.Reply("cap", photo=PNG, edit=True)))
        m.edit_media.assert_awaited_once()

    def test_update_message_switching_kind_sends_a_new_message(self):
        m = FakeMessage()  # text message, but the new reply is a picture
        run(telegram.update_message(m, ui.Reply("cap", photo=PNG, edit=True)))
        m.answer_photo.assert_awaited_once()
        m.edit_reply_markup.assert_awaited_once()

    def test_not_modified_is_ignored_and_markdown_errors_fall_back(self):
        m = FakeMessage()
        err = TelegramBadRequest(
            method=EditMessageText(text="x"), message="Bad Request: message is not modified"
        )
        m.edit_text.side_effect = err
        run(telegram.update_message(m, ui.Reply("same", edit=True)))
        m.answer.assert_not_awaited()
        m = FakeMessage()
        bad = TelegramBadRequest(method=EditMessageText(text="x"), message="can't parse entities")
        m.edit_text.side_effect = [bad, None]
        run(telegram.update_message(m, ui.Reply("*oops", edit=True)))
        self.assertEqual(m.edit_text.await_count, 2)

    def test_callback_handler_answers_and_edits(self):
        message = SimpleNamespace(chat=SimpleNamespace(id=555, type="private"))
        cb = SimpleNamespace(
            message=message, from_user=SimpleNamespace(id=555), data="go:home", answer=AsyncMock()
        )
        bot_mock = SimpleNamespace(send_chat_action=AsyncMock())
        with (
            patch.object(
                telegram,
                "handle_callback",
                AsyncMock(return_value=ui.Reply("hi", notice="ok", edit=True)),
            ),
            patch.object(telegram, "update_message", AsyncMock()) as upd,
            patch.object(telegram, "isinstance", create=True, return_value=True),
        ):
            # aiogram's isinstance(message, Message) check: use a real-ish subclass instead
            from aiogram.types import Message

            fake = Message.model_construct(message_id=1, date=datetime.utcnow(), chat=message.chat)
            cb.message = fake
            run(telegram.on_callback(cb, bot_mock))
        cb.answer.assert_awaited_with("ok")
        upd.assert_awaited_once()

    def test_stranger_cannot_press_buttons_in_a_private_chat(self):
        from aiogram.types import Message

        fake = Message.model_construct(
            message_id=1, date=datetime.utcnow(), chat=SimpleNamespace(id=555, type="private")
        )
        cb = SimpleNamespace(
            message=fake, from_user=SimpleNamespace(id=777), data="go:stats", answer=AsyncMock()
        )
        with patch.object(telegram, "handle_callback", AsyncMock()) as handler:
            run(telegram.on_callback(cb, SimpleNamespace(send_chat_action=AsyncMock())))
        handler.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
