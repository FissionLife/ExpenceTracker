"""Analytics engine, number formatting, and ledger undo / backdating. Offline, temp SQLite."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-analytics-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"

import itertools  # noqa: E402
import unittest  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from app.bot import fmt  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.services import analytics as A  # noqa: E402
from app.services import ledger  # noqa: E402

init_db()
_ids = itertools.count(1)


def ago(days, hour=12):
    return (datetime.utcnow() - timedelta(days=days)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )


class Base(unittest.TestCase):
    def setUp(self):
        self.db = SessionLocal()
        self.user = ledger.get_or_create_user(self.db, f"analytics-{next(_ids)}")
        self.uid = self.user.id

    def tearDown(self):
        self.db.close()

    def spend(
        self, amount, days_ago=1, cat="Food", merchant=None, account="Bank", kind="expense", hour=12
    ):
        return ledger.record_transaction(
            self.db,
            self.uid,
            amount,
            account,
            cat,
            merchant,
            kind,
            f"{merchant or cat}",
            "chat",
            ago(days_ago, hour),
        )


class FormatTests(unittest.TestCase):
    def test_indian_grouping(self):
        self.assertEqual(fmt.inr(0), "₹0")
        self.assertEqual(fmt.inr(950), "₹950")
        self.assertEqual(fmt.inr(12345), "₹12,345")
        self.assertEqual(fmt.inr(123456), "₹1,23,456")
        self.assertEqual(fmt.inr(12345678), "₹1,23,45,678")
        self.assertEqual(fmt.inr(-1234.5, 2), "−₹1,234.50")
        self.assertEqual(fmt.inr(1000.5, 2), "₹1,000.50")

    def test_short_money_and_helpers(self):
        self.assertEqual(fmt.short(850), "₹850")
        self.assertEqual(fmt.short(12500), "₹12.5k")
        self.assertEqual(fmt.short(3000), "₹3k")
        self.assertEqual(fmt.short(320000), "₹3.2L")
        self.assertEqual(fmt.short(14000000), "₹1.4Cr")
        self.assertEqual(fmt.change(27.4), "▲27%")
        self.assertEqual(fmt.change(-6.2), "▼6%")
        self.assertEqual(fmt.change(0.4), "")
        self.assertEqual(fmt.change(None), "")
        self.assertEqual(fmt.bar(50, 10), "▰▰▰▰▰▱▱▱▱▱")
        self.assertEqual(fmt.bar(500, 4), "▰▰▰▰")
        self.assertEqual(fmt.plural(1, "purchase"), "1 purchase")
        self.assertEqual(fmt.plural(3, "purchase"), "3 purchases")
        self.assertEqual(fmt.day("2026-10-05"), "05 Oct")


class PeriodTests(unittest.TestCase):
    NOW = datetime(2026, 10, 11, 15, 30)

    def p(self, text):
        return A.parse_period(text, self.NOW)

    def test_relative_periods(self):
        q = self.p("30d")
        self.assertEqual((q.days, q.label), (30, "Last 30 days"))
        self.assertEqual(self.p("2w").days, 14)
        self.assertEqual(self.p("6m").days, 180)
        self.assertEqual(self.p("1y").days, 365)
        self.assertEqual(self.p("9999d").days, 3650)  # capped

    def test_named_periods(self):
        self.assertEqual(self.p("today").label, "Today")
        self.assertEqual(self.p("mtd").start, datetime(2026, 10, 1))
        last = self.p("last month")
        self.assertEqual(
            (last.start, last.end, last.label),
            (datetime(2026, 9, 1), datetime(2026, 10, 1), "September 2026"),
        )
        self.assertEqual(self.p("ytd").start, datetime(2026, 1, 1))
        self.assertEqual(self.p("this week").start, datetime(2026, 10, 5))  # Monday
        self.assertTrue(self.p("all").extra.get("trim"))

    def test_month_and_range(self):
        m = self.p("2026-02")
        self.assertEqual((m.start, m.end), (datetime(2026, 2, 1), datetime(2026, 3, 1)))
        dec = self.p("2025-12")
        self.assertEqual(dec.end, datetime(2026, 1, 1))
        r = self.p("2026-09-01..2026-09-15")
        self.assertEqual((r.start, r.end), (datetime(2026, 9, 1), datetime(2026, 9, 16)))
        self.assertEqual(self.p("2026-09-01 to 2026-09-15").days, 15)

    def test_rejects_non_periods(self):
        for bad in ("", "food", "2026-13", "2026-09-15..2026-09-01", "30", "hello world"):
            self.assertIsNone(self.p(bad), bad)

    def test_previous_period_is_the_same_length_just_before(self):
        q = self.p("30d")
        prev = q.previous()
        self.assertEqual(prev.end, q.start)
        self.assertEqual(prev.days, q.days)


class ReportTests(Base):
    def test_empty_account_is_safe_everywhere(self):
        rep = A.report(self.db, self.uid)
        self.assertEqual(rep["summary"]["spent"], 0)
        self.assertEqual(rep["categories"], [])
        self.assertEqual(rep["insights"], [])
        self.assertIsNone(rep["summary"]["savings_rate"])
        self.assertEqual(len(rep["series"]["points"]), 30)
        self.assertIsNone(rep["hours"])
        self.assertEqual(rep["recurring"], [])

    def test_summary_numbers_and_transfers_do_not_count(self):
        self.spend(60000, 5, "Salary", "Acme", kind="income")
        self.spend(1000, 2, "Food", "Swiggy")
        self.spend(500, 3, "Food", "Zomato")
        ledger.transfer_between_pipes(self.db, self.uid, "Bank", "Cash", 2000)
        s = A.report(self.db, self.uid, A.parse_period("30d"))["summary"]
        self.assertEqual((s["spent"], s["earned"], s["net"]), (1500.0, 60000.0, 58500.0))
        self.assertEqual(s["expense_count"], 2)
        self.assertEqual(s["transfers"], 1)
        self.assertEqual(s["avg_expense"], 750.0)
        self.assertEqual(s["largest_expense"], 1000.0)
        self.assertEqual(s["savings_rate"], 97.5)

    def test_comparison_with_previous_period(self):
        self.spend(1000, 40, "Food")  # previous 30 days
        self.spend(1500, 5, "Food")
        s = A.report(self.db, self.uid, A.parse_period("30d"))["summary"]
        self.assertEqual(s["prev_spent"], 1000.0)
        self.assertEqual(s["spent_change_pct"], 50.0)

    def test_categories_top_n_and_other(self):
        for i, cat in enumerate(["A", "B", "C", "D", "E"], start=1):
            self.spend(i * 100, 2, cat)
        q = A.parse_period("30d")
        rows = A.category_breakdown(A.load_rows(self.db, self.uid, q), None, top_n=3)
        self.assertEqual([r["category"] for r in rows], ["E", "D", "C", "Other"])
        self.assertEqual(rows[-1]["amount"], 300.0)
        self.assertAlmostEqual(sum(r["share_pct"] for r in rows), 100.0, delta=0.1)

    def test_filters_narrow_everything(self):
        self.spend(400, 2, "Food", "Swiggy", account="Bank")
        self.spend(600, 2, "Food", "Zomato", account="Cash")
        self.spend(900, 2, "Shopping", "Amazon", account="Bank")
        base = A.parse_period("30d")
        q, unmatched = A.resolve_filters(self.db, self.uid, base, ["food", "bank", "nonsense"])
        self.assertEqual(unmatched, ["nonsense"])
        self.assertEqual((q.categories, q.accounts), (("Food",), ("Bank",)))
        self.assertTrue(q.filtered)
        self.assertEqual(A.report(self.db, self.uid, q)["summary"]["spent"], 400.0)
        by_merchant, _ = A.resolve_filters(self.db, self.uid, base, ["zomato"])
        self.assertEqual(A.report(self.db, self.uid, by_merchant)["summary"]["spent"], 600.0)

    def test_amount_filters(self):
        self.spend(100, 2, "Food")
        self.spend(5000, 2, "Food")
        from dataclasses import replace

        q = replace(A.parse_period("30d"), min_amount=1000)
        self.assertEqual(A.report(self.db, self.uid, q)["summary"]["spent"], 5000.0)

    def test_series_grouping_adapts_to_the_range(self):
        self.spend(100, 2)
        self.assertEqual(
            A.report(self.db, self.uid, A.parse_period("30d"))["series"]["group"], "day"
        )
        self.assertEqual(
            A.report(self.db, self.uid, A.parse_period("90d"))["series"]["group"], "week"
        )
        self.assertEqual(
            A.report(self.db, self.uid, A.parse_period("1y"))["series"]["group"], "month"
        )
        pts = A.report(self.db, self.uid, A.parse_period("7d"))["series"]["points"]
        self.assertEqual(len(pts), 7)
        self.assertEqual(sum(p["spent"] for p in pts), 100.0)

    def test_all_time_starts_at_the_first_transaction(self):
        self.spend(100, 70)
        pts = A.report(self.db, self.uid, A.parse_period("all"))["series"]["points"]
        self.assertLessEqual(len(pts), 4)
        self.assertGreater(len(pts), 0)

    def test_recurring_budgets_unusual_and_weekdays(self):
        for months_ago in range(3):
            self.spend(649, 3 + 31 * months_ago, "Entertainment", "Netflix")
        for i in range(6):
            self.spend(300, 2 + i, "Shopping", "Amazon")
        self.spend(9000, 1, "Shopping", "Croma")
        ledger.get_or_create_category(self.db, self.uid, "Shopping").budget_limit = 5000.0
        self.db.commit()
        rep = A.report(self.db, self.uid, A.parse_period("30d"))
        self.assertEqual([r["merchant"] for r in rep["recurring"]], ["Netflix"])
        self.assertEqual([u["merchant"] for u in rep["unusual"]], ["Croma"])
        self.assertEqual(rep["budgets"][0]["level"], "over")
        self.assertEqual(len(rep["weekdays"]), 7)
        self.assertTrue(any("budget" in line for line in rep["insights"]))

    def test_hour_pattern_only_when_times_are_known(self):
        for i in range(8):
            self.spend(100, 2 + i, "Food", hour=9 + i)
        self.assertIsNotNone(A.report(self.db, self.uid)["hours"])
        other = ledger.get_or_create_user(self.db, f"analytics-{next(_ids)}").id
        for i in range(8):
            ledger.record_transaction(
                self.db, other, 100, "Bank", "Food", None, "expense", "x", "pdf", ago(2 + i, 0)
            )
        self.assertIsNone(A.report(self.db, other)["hours"])

    def test_month_pace_projection_ignores_one_off_big_purchases(self):
        self.spend(200, 0, "Food")
        self.spend(50000, 0, "Shopping", "Croma")  # a one-off: must not be extrapolated
        pace = A.month_pace(self.db, self.uid)
        self.assertGreater(pace["spent"], 50000)
        self.assertEqual(pace["regular_daily"], round(200 / pace["day"], 2))
        # naive extrapolation would be spent/day*days_in_month; ours adds only the regular pace
        naive = pace["spent"] / pace["day"] * pace["days_in_month"]
        self.assertLessEqual(pace["projected"], max(naive, pace["spent"] + 1))
        if pace["day"] > 1:
            self.assertLess(pace["projected"], naive)

    def test_net_worth_series_ends_at_todays_balance(self):
        self.spend(10000, 6, "Salary", "Acme", kind="income")
        self.spend(2500, 3, "Food")
        series = A.net_worth_series(self.db, self.uid, 10)
        self.assertEqual(series[-1]["total"], 7500.0)
        self.assertEqual(series[0]["total"], 0.0)
        self.assertEqual(len(series), 10)

    def test_users_are_isolated(self):
        self.spend(999, 2, "Food")
        other = ledger.get_or_create_user(self.db, f"analytics-{next(_ids)}").id
        self.assertEqual(A.report(self.db, other)["summary"]["spent"], 0)

    def test_heatmap_and_sizes(self):
        self.spend(50, 2)
        self.spend(30000, 3)
        rep = A.report(self.db, self.uid)
        self.assertEqual(rep["heatmap"]["max"], 30000.0)
        bands = {b["band"]: b["count"] for b in rep["sizes"]}
        self.assertEqual((bands["<100"], bands["20k+"]), (1, 1))


class LedgerUndoAndDateTests(Base):
    def balance(self, name="Bank"):
        accs = ledger.get_account_balances(self.db, self.uid)["accounts"]
        return next(a["balance"] for a in accs if a["name"] == name)

    def test_backdated_transaction_gets_its_date(self):
        ledger.record_transaction(
            self.db, self.uid, 100, "Bank", "Food", None, "expense", "x", "pdf", "2026-09-05"
        )
        rows, _ = ledger.recent_transactions(self.db, self.uid, 5)
        self.assertEqual(rows[0]["date"], "2026-09-05")

    def test_bad_dates_are_rejected(self):
        for bad in ("yesterday", "2999-01-01", "1990-01-01", "2026-13-40"):
            with self.assertRaises(ledger.LedgerError):
                ledger.record_transaction(self.db, self.uid, 100, "Bank", created_at=bad)

    def test_undo_expense_income_and_transfer(self):
        income = self.spend(1000, 1, "Salary", kind="income")
        expense = self.spend(300, 1, "Food")
        self.assertEqual(self.balance(), 700.0)
        ledger.undo_transaction(self.db, self.uid, expense["transaction_id"][:8])
        self.assertEqual(self.balance(), 1000.0)
        ledger.undo_transaction(self.db, self.uid, income["transaction_id"])
        self.assertEqual(self.balance(), 0.0)
        self.spend(500, 1, "Salary", kind="income")
        t = ledger.transfer_between_pipes(self.db, self.uid, "Bank", "Cash", 200)
        self.assertEqual((self.balance("Bank"), self.balance("Cash")), (300.0, 200.0))
        ledger.undo_transaction(self.db, self.uid, t["transaction_id"])
        self.assertEqual((self.balance("Bank"), self.balance("Cash")), (500.0, 0.0))

    def test_undo_errors(self):
        tx = self.spend(100, 1)
        for bad in ("", "abc", "zzzzzzzz"):
            with self.assertRaises(ledger.LedgerError):
                ledger.undo_transaction(self.db, self.uid, bad)
        ledger.undo_transaction(self.db, self.uid, tx["transaction_id"])
        with self.assertRaises(ledger.LedgerError):  # already undone
            ledger.undo_transaction(self.db, self.uid, tx["transaction_id"])

    def test_cannot_undo_someone_elses_transaction(self):
        tx = self.spend(100, 1)
        other = ledger.get_or_create_user(self.db, f"analytics-{next(_ids)}").id
        with self.assertRaises(ledger.LedgerError):
            ledger.undo_transaction(self.db, other, tx["transaction_id"])
        self.assertEqual(self.balance(), -100.0)

    def test_recent_transactions_paging(self):
        for i in range(5):
            self.spend(10 + i, i + 1)
        rows, total = ledger.recent_transactions(self.db, self.uid, 2, 0)
        self.assertEqual((len(rows), total), (2, 5))
        rows2, _ = ledger.recent_transactions(self.db, self.uid, 2, 4)
        self.assertEqual(len(rows2), 1)


if __name__ == "__main__":
    unittest.main()
