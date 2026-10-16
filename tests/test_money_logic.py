"""Regression tests for the money-logic audit: time zones, comparisons, averages, tools. Offline."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-money-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""

import itertools  # noqa: E402
import threading  # noqa: E402
import unittest  # noqa: E402
from datetime import date, datetime, timedelta  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app import clock  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.services import analytics, ledger  # noqa: E402
from app.services_ai.mcp_server import execute_tool  # noqa: E402

init_db()
_ids = itertools.count(1)

NOW = datetime(2026, 10, 13, 15, 0)  # a fixed "now" (local time) so periods are predictable


def frozen(now=NOW):
    """Freeze the clock: local 'now' is `now` (treated as UTC when TIMEZONE=UTC)."""
    return patch.object(clock, "now_local", return_value=now)


def user():
    chat = f"money-{next(_ids)}"
    with SessionLocal() as db:
        return chat, ledger.get_or_create_user(db, chat).id


def spend(
    uid, amount, days_ago, category="Food & Dining", merchant=None, kind="expense", account="Bank"
):
    with SessionLocal() as db:
        ledger.record_transaction(
            db,
            uid,
            amount,
            account,
            category,
            merchant,
            tx_type=kind,
            created_at=NOW - timedelta(days=days_ago),
        )


class TimezoneTests(unittest.TestCase):
    def test_a_late_night_spend_counts_on_the_local_day(self):
        _, uid = user()
        # 01:30 on 1 October in India (UTC+5:30) is 20:00 on 30 September in UTC
        with patch.object(settings, "timezone", "Asia/Kolkata"):
            stored = ledger._validate_date("2026-10-01T01:30:00")
            self.assertEqual(stored, datetime(2026, 9, 30, 20, 0))
            with SessionLocal() as db:
                ledger.record_transaction(
                    db, uid, 1500, "Bank", "Food", created_at="2026-10-01T01:30:00"
                )
            with frozen(datetime(2026, 10, 13, 9, 0)), SessionLocal() as db:
                october = analytics.report(db, uid, analytics.parse_period("2026-10"))
                september = analytics.report(db, uid, analytics.parse_period("2026-09"))
        self.assertEqual(october["summary"]["spent"], 1500.0)
        self.assertEqual(september["summary"]["spent"], 0.0)

    def test_a_date_without_a_time_stays_on_that_day_everywhere(self):
        for zone in ("Asia/Kolkata", "America/New_York", "UTC"):
            with patch.object(settings, "timezone", zone):
                stored = ledger._validate_date("2026-10-05")
                self.assertEqual(clock.to_local(stored).date(), date(2026, 10, 5), zone)

    def test_unknown_timezone_falls_back_to_utc(self):
        with patch.object(settings, "timezone", "Mars/Olympus"):
            clock._zone.cache_clear()
            self.assertEqual(clock.to_local(datetime(2026, 1, 1, 10)), datetime(2026, 1, 1, 10))
        clock._zone.cache_clear()


class ComparisonTests(unittest.TestCase):
    def test_month_so_far_is_compared_with_the_same_days_last_month(self):
        with frozen():
            q = analytics.parse_period("mtd").previous()
        self.assertEqual((q.start, q.end), (datetime(2026, 9, 1), datetime(2026, 9, 14)))

    def test_a_partial_named_month_is_clamped_too(self):
        with frozen():
            q = analytics.parse_period("2026-10").previous()
        self.assertEqual((q.start, q.end), (datetime(2026, 9, 1), datetime(2026, 9, 14)))

    def test_a_finished_month_is_compared_with_the_whole_month_before(self):
        with frozen():
            q = analytics.parse_period("2026-09").previous()
        self.assertEqual((q.start, q.end), (datetime(2026, 8, 1), datetime(2026, 9, 1)))

    def test_this_week_so_far_is_compared_with_the_same_days_last_week(self):
        with frozen():  # Tuesday 13 Oct
            q = analytics.parse_period("week").previous()
        self.assertEqual((q.start, q.end), (datetime(2026, 10, 5), datetime(2026, 10, 7)))

    def test_a_rolling_window_is_compared_with_the_window_before_it(self):
        with frozen():
            now_q = analytics.parse_period("30d")
            prev = now_q.previous()
        self.assertEqual(prev.end, now_q.start)
        self.assertEqual(prev.end - prev.start, now_q.end - now_q.start)

    def test_year_so_far_is_compared_with_the_same_days_last_year(self):
        with frozen():
            q = analytics.parse_period("ytd").previous()
        self.assertEqual(q.start, datetime(2025, 1, 1))
        self.assertEqual(q.end, datetime(2025, 10, 14))


class AverageTests(unittest.TestCase):
    def test_daily_average_uses_the_first_entry_ever_not_the_window(self):
        _, uid = user()
        spend(uid, 3000, 1)
        spend(uid, 100, 400)  # old history exists, so the 30-day window really is 30 days
        with frozen(), SessionLocal() as db:
            rep = analytics.report(db, uid, analytics.parse_period("30d"))
        self.assertAlmostEqual(rep["summary"]["avg_per_day"], 100.0, delta=1)

    def test_a_new_user_is_not_divided_by_empty_days(self):
        _, uid = user()
        spend(uid, 1000, 9)
        with frozen(), SessionLocal() as db:
            rep = analytics.report(db, uid, analytics.parse_period("all"))
        self.assertAlmostEqual(rep["summary"]["avg_per_day"], 100.0, delta=12)  # 1000 over ~10 days

    def test_weekday_averages_count_only_days_that_exist(self):
        _, uid = user()
        for weeks in range(5):  # a Saturday every week for five weeks
            spend(uid, 500, 3 + 7 * weeks)  # 10 Oct 2026 is a Saturday
        with frozen(), SessionLocal() as db:
            rep = analytics.report(db, uid, analytics.parse_period("1y"))
        sat = next(w for w in rep["weekdays"] if w["day"] == "Sat")
        self.assertEqual(sat["avg"], 500.0)  # not 500 spread over 365 days

    def test_exactly_hitting_the_budget_is_not_over_it(self):
        chat, uid = user()
        with SessionLocal() as db:
            cat = ledger.get_or_create_category(db, uid, "Food & Dining")
            cat.budget_limit = 1000.0
            db.commit()
        spend(uid, 1000, 2)
        with frozen(), SessionLocal() as db:
            level = analytics.budget_status(db, uid)[0]["level"]
        self.assertEqual(level, "warn")


class PlanningTests(unittest.TestCase):
    def test_runway_and_plan_admit_when_there_is_little_history(self):
        _, uid = user()
        spend(uid, 30000, 3, "Salary", kind="income")
        spend(uid, 6000, 2)
        with frozen(), SessionLocal() as db:
            runway = analytics.runway(db, uid)
            plan = analytics.savings_plan(db, uid, 60000, 6)
        self.assertTrue(runway["low_confidence"])
        self.assertTrue(plan["low_confidence"])
        self.assertLessEqual(runway["data_days"], 4)
        self.assertGreater(
            plan["avg_income"], 30000
        )  # averaged over days that exist, not diluted over 90

    def test_loans_and_investments_are_never_suggested_cuts(self):
        for name in (
            "Loan/Emi",
            "Investments",
            "Mutual Fund Investment",
            "Rent",
            "Bills & Utilities",
        ):
            self.assertTrue(analytics._is_essential(name), name)
        for name in ("Shopping", "Entertainment", "Food & Dining"):
            self.assertFalse(analytics._is_essential(name), name)

    def test_the_month_projection_does_not_extrapolate_from_two_days(self):
        _, uid = user()
        with frozen(datetime(2026, 11, 2, 12, 0)):
            spend(uid, 400, 0)
            with SessionLocal() as db:
                pace = analytics.month_pace(db, uid)
        self.assertLess(pace["projected"], 1000)  # not 400 x 30 = 12000


class RecurringTests(unittest.TestCase):
    def rows(self, spec):
        return [
            analytics.Row(str(i), when, amt, "expense", "Food", merchant, "Bank", "")
            for i, (merchant, when, amt) in enumerate(spec)
        ]

    def test_monthly_bills_are_recurring(self):
        rows = self.rows([("Netflix", datetime(2026, m, 5), 649.0) for m in (7, 8, 9)])
        self.assertEqual([r["merchant"] for r in analytics.recurring(rows)], ["Netflix"])

    def test_weekly_milk_and_two_chai_in_a_row_are_not_bills(self):
        milk = [("Milkman", datetime(2026, 9, 1) + timedelta(days=7 * i), 60.0) for i in range(6)]
        chai = [("Chai", datetime(2026, 9, 30), 255.0), ("Chai", datetime(2026, 10, 1), 255.0)]
        self.assertEqual(analytics.recurring(self.rows(milk + chai)), [])


class HeatmapAndTruncationTests(unittest.TestCase):
    def test_heatmap_is_the_last_weeks_whatever_period_is_shown(self):
        _, uid = user()
        spend(uid, 700, 3)
        with frozen(), SessionLocal() as db:
            rep = analytics.report(
                db, uid, analytics.parse_period("2026-08")
            )  # a month with no spending
        self.assertGreater(rep["heatmap"]["max"], 0)

    def test_long_ranges_group_coarser_instead_of_cutting_off_the_newest_days(self):
        rows = []
        q = analytics.Query(datetime(2022, 1, 1), datetime(2026, 10, 14), group="day")
        series = analytics.series(rows, q)
        self.assertNotEqual(series["group"], "day")
        self.assertEqual(series["points"][-1]["key"][:4], "2026")


class ToolsAgreeWithTheDashboardTests(unittest.TestCase):
    def test_runway_budget_and_recurring_tools_use_the_same_definitions(self):
        chat, uid = user()
        spend(uid, 50000, 80, "Salary", kind="income")
        for m in range(3):
            spend(uid, 649, 60 - 30 * m, "Entertainment", "Netflix")
        spend(uid, 3000, 5)
        with SessionLocal() as db:
            cat = ledger.get_or_create_category(db, uid, "Entertainment")
            cat.budget_limit = 700.0
            db.commit()
        # Both sides must see the same "today", or the 90-day runway window differs.
        with frozen():
            with SessionLocal() as db:
                rep = analytics.report(db, uid, analytics.parse_period("30d"))
            runway = execute_tool("skill_emergency_fund_calculator", user_id=chat)
            bills = execute_tool("skill_recurring_bill_detector", user_id=chat)["recurring_bills"]
            spending = execute_tool("get_spending_breakdown", user_id=chat, period="month")
        self.assertEqual(runway["runway_months"], rep["runway"]["months"])
        self.assertEqual([b["merchant"] for b in bills], [r["merchant"] for r in rep["recurring"]])
        self.assertIn("Food & Dining", spending["category_breakdown"])


class CategoryResolverTests(unittest.TestCase):
    def test_one_resolver_for_the_command_and_the_ai(self):
        chat, uid = user()
        with SessionLocal() as db:
            ledger.get_or_create_category(db, uid, "Food & Dining")
            ledger.get_or_create_category(db, uid, "Seafood Market")
        result = execute_tool(
            "set_category_budget", user_id=chat, category="food", monthly_budget=4000
        )
        self.assertEqual(
            result["category"], "Food & Dining"
        )  # the word, not a substring of "Seafood"
        with SessionLocal() as db:
            self.assertEqual(
                ledger.resolve_category(db, uid, "FOOD & dining").name, "Food & Dining"
            )


class UndoRaceTests(unittest.TestCase):
    def test_two_simultaneous_undos_refund_only_once(self):
        chat, uid = user()
        spend(uid, 1000, 1, "Salary", kind="income")
        with SessionLocal() as db:
            tx = ledger.record_transaction(db, uid, 400, "Bank", "Food")["transaction_id"]
        results = []

        def undo():
            try:
                with SessionLocal() as db:
                    ledger.undo_transaction(db, uid, tx)
                results.append("ok")
            except ledger.LedgerError:
                results.append("already")

        threads = [threading.Thread(target=undo) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(results), ["already", "ok"])
        with SessionLocal() as db:
            balance = next(
                a["balance"]
                for a in ledger.get_account_balances(db, uid)["accounts"]
                if a["name"] == "Bank"
            )
        self.assertEqual(balance, 1000.0)


class FormattingTests(unittest.TestCase):
    def test_no_negative_zero_and_one_kind_of_minus(self):
        from app.bot import fmt

        self.assertEqual(fmt.inr(-0.0000001), "₹0")
        self.assertEqual(fmt.inr(-5000), "−₹5,000")
        self.assertEqual(fmt.short(-0.2), "₹0")

    def test_amounts_for_budgets_understand_k_and_lakh(self):
        from app.bot import textutil

        self.assertEqual(textutil.parse_amount("6k"), 6000.0)
        self.assertEqual(textutil.parse_amount("2l"), 200000.0)
        self.assertEqual(textutil.parse_amount("₹6,000"), 6000.0)
        self.assertIsNone(textutil.parse_amount("food"))

    def test_stats_arguments_may_put_the_period_anywhere(self):
        from app.bot import views

        self.assertEqual(views.parse_args("food 7d"), ("7d", ["food"]))
        self.assertEqual(views.parse_args("hdfc last month food"), ("last month", ["hdfc", "food"]))
        self.assertEqual(views.parse_args("food"), ("30d", ["food"]))

    def test_a_loss_is_not_called_savings(self):
        from app.bot import views

        rep = {
            "query": {"describe": "Last 30 days", "filtered": False},
            "summary": {
                "transactions": 2,
                "spent": 9000.0,
                "earned": 4000.0,
                "net": -5000.0,
                "savings_rate": -125.0,
                "avg_per_day": 300.0,
            },
            "categories": [],
            "merchants": [],
            "insights": [],
        }
        text = views.caption(rep)
        self.assertNotIn("Saved", text)
        self.assertIn("more than earned", text)


if __name__ == "__main__":
    unittest.main()
