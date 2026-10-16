"""The web dashboard API: signed links, isolation between users, filters, security headers. Offline."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-dashweb-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""

import csv  # noqa: E402
import hashlib  # noqa: E402
import hmac  # noqa: E402
import io  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from unittest.mock import patch  # noqa: E402
from urllib.parse import urlencode  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.services import ledger  # noqa: E402
from app.web import dashboard, tokens  # noqa: E402
from main import app  # noqa: E402

init_db()
_ids = itertools.count(1)


class Base(unittest.TestCase):
    def setUp(self):
        dashboard._hits.clear()
        self.client = TestClient(app)
        self.chat = f"web-{next(_ids)}"
        self.token = tokens.make_token(self.chat)
        self.h = {"X-Dashboard-Token": self.token}
        with SessionLocal() as db:
            uid = ledger.get_or_create_user(db, self.chat).id
            now = datetime.utcnow()
            ledger.record_transaction(
                db,
                uid,
                50000,
                "Bank",
                "Salary",
                "Acme",
                "income",
                "pay",
                "chat",
                now - timedelta(days=8),
            )
            for i in range(10):
                ledger.record_transaction(
                    db,
                    uid,
                    100 + i * 50,
                    "Bank" if i % 2 else "Cash",
                    "Food" if i % 3 else "Transport",
                    "Swiggy" if i % 2 else "Uber",
                    "expense",
                    f"note {i}",
                    "chat",
                    now - timedelta(days=i),
                )

    def get(self, path, **params):
        return self.client.get(path, params=params, headers=self.h)


class PageAndAuthTests(Base):
    def test_page_is_served_with_security_headers_and_no_data(self):
        r = self.client.get(f"/d/{self.token}")
        self.assertEqual(r.status_code, 200)
        self.assertIn("WalletLedger dashboard", r.text)
        self.assertNotIn(self.chat, r.text)
        self.assertEqual(r.headers["cache-control"], "no-store")
        self.assertEqual(r.headers["referrer-policy"], "no-referrer")
        self.assertIn("default-src 'self'", r.headers["content-security-policy"])
        self.assertNotIn(
            "unsafe-inline",
            r.headers["content-security-policy"].split("script-src")[1].split(";")[0],
        )
        self.assertEqual(self.client.get("/d").status_code, 200)  # Mini App entry

    def test_static_assets(self):
        for name, kind in (("dashboard.js", "javascript"), ("dashboard.css", "css")):
            r = self.client.get(f"/static/{name}")
            self.assertEqual(r.status_code, 200)
            self.assertIn(kind, r.headers["content-type"])

    def test_api_requires_a_valid_token(self):
        for headers in (
            {},
            {"X-Dashboard-Token": "garbage"},
            {"X-Dashboard-Token": tokens.make_token(self.chat, ttl_seconds=-5)},
        ):
            for path in ("/api/dashboard", "/api/transactions", "/api/export.csv"):
                self.assertEqual(
                    self.client.get(path, headers=headers).status_code, 401, (path, headers)
                )

    def test_token_in_the_query_string_is_not_accepted(self):
        self.assertEqual(
            self.client.get("/api/dashboard", params={"token": self.token}).status_code, 401
        )

    def test_private_bot_blocks_the_dashboard(self):
        with patch.object(settings, "allowed_chat_ids", "999"):
            self.assertEqual(self.get("/api/dashboard").status_code, 403)

    def test_rate_limit(self):
        with patch.object(dashboard, "RATE_LIMIT", 3):
            codes = [self.get("/api/dashboard", period="7d").status_code for _ in range(5)]
        self.assertEqual(codes[:3], [200, 200, 200])
        self.assertEqual(codes[3:], [429, 429])

    def test_users_only_see_their_own_data(self):
        other = f"web-{next(_ids)}"
        with SessionLocal() as db:
            other_uid = ledger.get_or_create_user(db, other).id
            ledger.record_transaction(db, other_uid, 10000, "Bank", "Salary", tx_type="income")
            ledger.record_transaction(
                db,
                other_uid,
                777,
                "Bank",
                "Secret",
                "Hidden",
                "expense",
                "private note",
            )
        mine = self.get("/api/dashboard").json()
        self.assertNotIn("Secret", json.dumps(mine))
        self.assertNotIn("Hidden", json.dumps(mine))
        theirs = self.client.get(
            "/api/dashboard", headers={"X-Dashboard-Token": tokens.make_token(other)}
        ).json()
        self.assertEqual(theirs["summary"]["spent"], 777.0)
        rows = self.get("/api/transactions", period="1y").json()["rows"]
        self.assertTrue(all(r["merchant"] != "Hidden" for r in rows))


class ReportTests(Base):
    def test_dashboard_report_shape(self):
        d = self.get("/api/dashboard", period="30d").json()
        for key in (
            "summary",
            "categories",
            "merchants",
            "series",
            "stack",
            "weekdays",
            "heatmap",
            "budgets",
            "pace",
            "net_worth",
            "insights",
            "options",
            "applied",
            "query",
            "recurring",
            "unusual",
        ):
            self.assertIn(key, d)
        self.assertEqual(d["summary"]["earned"], 50000.0)
        self.assertEqual(d["summary"]["expense_count"], 10)
        self.assertIn("Food", d["options"]["categories"])
        self.assertEqual(sorted(d["options"]["accounts"]), ["Bank", "Cash"])

    def test_filters_and_periods(self):
        base = self.get("/api/dashboard", period="30d").json()["summary"]["spent"]
        food = self.get("/api/dashboard", period="30d", categories="Food").json()
        self.assertLess(food["summary"]["spent"], base)
        self.assertTrue(food["query"]["filtered"])
        cash = self.get("/api/dashboard", period="30d", accounts="Cash").json()["summary"]["spent"]
        bank = self.get("/api/dashboard", period="30d", accounts="Bank").json()["summary"]["spent"]
        self.assertAlmostEqual(cash + bank, base, places=2)
        swiggy = self.get("/api/dashboard", period="30d", merchants="swig").json()["summary"][
            "spent"
        ]
        self.assertGreater(swiggy, 0)
        big = self.get("/api/dashboard", period="30d", min="400").json()["summary"]["expense_count"]
        self.assertLess(big, 10)
        week = self.get("/api/dashboard", period="7d").json()
        self.assertEqual(week["series"]["group"], "day")
        self.assertEqual(self.get("/api/dashboard", period="1y").json()["series"]["group"], "month")
        self.assertEqual(
            self.get("/api/dashboard", period="90d", group="day").json()["series"]["group"], "day"
        )

    def test_custom_range_and_bad_input(self):
        today = datetime.utcnow().date()
        a, b = (today - timedelta(days=3)).isoformat(), today.isoformat()
        d = self.get("/api/dashboard", start=a, end=b).json()
        self.assertEqual(d["query"]["start"], a)
        self.assertEqual(self.get("/api/dashboard", start="nope", end="nada").status_code, 400)
        self.assertEqual(self.get("/api/dashboard", min="abc").status_code, 400)
        self.assertEqual(
            self.get("/api/dashboard", period="garbage").status_code, 200
        )  # falls back to 30d
        self.assertEqual(
            self.get("/api/dashboard", group="evil").json()["applied"]["group"], "auto"
        )

    def test_compare_toggle(self):
        self.assertIsNotNone(
            self.get("/api/dashboard", compare="1").json()["summary"]["prev_spent"] is not None
        )
        self.assertNotIn("prev_spent", self.get("/api/dashboard", compare="0").json()["summary"])

    def test_injection_style_input_is_harmless(self):
        d = self.get(
            "/api/dashboard", categories="x' OR '1'='1", merchants="%; DROP TABLE users;--"
        )
        self.assertEqual(d.status_code, 200)
        self.assertEqual(d.json()["summary"]["spent"], 0)


class TransactionsAndExportTests(Base):
    def test_paging_sorting_search(self):
        d = self.get("/api/transactions", period="30d", per_page=5, page=1).json()
        self.assertEqual((d["total"], len(d["rows"]), d["pages"]), (11, 5, 3))
        self.assertEqual(
            self.get("/api/transactions", period="30d", per_page=5, page=3)
            .json()["rows"]
            .__len__(),
            1,
        )
        by_amount = self.get("/api/transactions", period="30d", sort="amount", dir="desc").json()[
            "rows"
        ]
        self.assertEqual(by_amount[0]["amount"], 50000.0)
        asc = self.get("/api/transactions", period="30d", sort="amount", dir="asc").json()["rows"]
        self.assertLessEqual(asc[0]["amount"], asc[1]["amount"])
        found = self.get("/api/transactions", period="30d", q="uber").json()
        self.assertTrue(found["total"] > 0 and all("Uber" in r["merchant"] for r in found["rows"]))
        income = self.get("/api/transactions", period="30d", type="income").json()
        self.assertEqual(income["total"], 1)
        self.assertEqual(self.get("/api/transactions", per_page="x").status_code, 400)
        self.assertLessEqual(len(self.get("/api/transactions", per_page=9999).json()["rows"]), 100)

    def test_csv_export_matches_filters(self):
        r = self.get("/api/export.csv", period="30d", categories="Food")
        self.assertEqual(r.status_code, 200)
        self.assertIn("attachment", r.headers["content-disposition"])
        rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
        self.assertEqual(rows[0][0], "Date")
        self.assertTrue(all(row[3] == "Food" for row in rows[1:]))
        self.assertGreater(len(rows), 1)


class MiniAppAuthTests(Base):
    BOT = "123456:TEST-TOKEN"

    def init_data(self, user_id, age=0, tamper=False):
        fields = {
            "auth_date": str(int(time.time()) - age),
            "query_id": "AAH",
            "user": json.dumps({"id": user_id}),
        }
        check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
        secret = hmac.new(b"WebAppData", self.BOT.encode(), hashlib.sha256).digest()
        digest = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
        data = urlencode({**fields, "hash": digest})
        return data.replace(str(user_id), str(user_id + 1)) if tamper else data

    def test_valid_sign_in_returns_a_working_token_for_that_user_only(self):
        with patch.object(settings, "telegram_bot_token", self.BOT):
            r = self.client.post("/api/webapp/auth", json={"initData": self.init_data(4242)})
        self.assertEqual(r.status_code, 200)
        token = r.json()["token"]
        self.assertEqual(tokens.read_token(token), "4242")

    def test_forged_stale_or_unconfigured_sign_in_is_rejected(self):
        with patch.object(settings, "telegram_bot_token", self.BOT):
            self.assertEqual(
                self.client.post(
                    "/api/webapp/auth", json={"initData": self.init_data(4242, tamper=True)}
                ).status_code,
                401,
            )
            self.assertEqual(
                self.client.post(
                    "/api/webapp/auth", json={"initData": self.init_data(4242, age=10**6)}
                ).status_code,
                401,
            )
            self.assertEqual(
                self.client.post("/api/webapp/auth", json={"initData": "nonsense"}).status_code, 401
            )
        with patch.object(settings, "telegram_bot_token", ""):
            self.assertEqual(
                self.client.post(
                    "/api/webapp/auth", json={"initData": self.init_data(4242)}
                ).status_code,
                404,
            )

    def test_private_bot_blocks_mini_app_sign_in(self):
        with (
            patch.object(settings, "telegram_bot_token", self.BOT),
            patch.object(settings, "allowed_chat_ids", "1"),
        ):
            self.assertEqual(
                self.client.post(
                    "/api/webapp/auth", json={"initData": self.init_data(4242)}
                ).status_code,
                403,
            )


if __name__ == "__main__":
    unittest.main()
