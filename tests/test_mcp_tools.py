"""Tests for the registry-backed ledger tools and their MCP surface."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)

import asyncio
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.services_ai import mcp_server


class MCPToolTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False)
        self.session_patch = patch.object(mcp_server, "SessionLocal", self.session_factory)
        self.session_patch.start()
        self.user_id = "mcp-test-user"

    def tearDown(self):
        self.session_patch.stop()
        self.engine.dispose()

    def balances(self):
        return {
            account["name"]: account["balance"]
            for account in mcp_server.list_accounts(self.user_id)["accounts"]
        }

    def test_search_and_edit_reconcile_balances_and_metadata(self):
        mcp_server.log_income(self.user_id, 10000, source_account="Cash")
        mcp_server.log_income(self.user_id, 10000, source_account="Bank")
        created = mcp_server.log_expense(
            self.user_id, 100, "Food", merchant="Cafe", account_name="Cash", description="lunch"
        )
        transaction_id = created["transaction_id"]

        matches = mcp_server.search_transactions(self.user_id, query="Cafe")
        self.assertEqual(matches["transactions"][0]["id"], transaction_id)

        mcp_server.edit_transaction(
            self.user_id,
            transaction_id,
            amount=60,
            account_name="Bank",
            category="Dining",
            merchant="Cafe Two",
            description="team lunch",
            transaction_date="2026-01-02T12:00:00",
        )
        self.assertEqual(self.balances(), {"Bank": 9940.0, "Cash": 10000.0})
        edited = mcp_server.search_transactions(self.user_id, query="team lunch")["transactions"][0]
        self.assertEqual(edited["merchant"], "Cafe Two")
        self.assertEqual(edited["category"], "Dining")
        self.assertTrue(edited["date"].startswith("2026-01-02"))

    def test_edit_can_correct_expense_to_income(self):
        mcp_server.log_income(self.user_id, 10000, source_account="Cash")
        created = mcp_server.log_expense(self.user_id, 40, "Food", account_name="Cash")
        mcp_server.edit_transaction(
            self.user_id, created["transaction_id"], transaction_type="income"
        )
        self.assertEqual(self.balances()["Cash"], 10040.0)

    def test_budget_tools_set_and_clear_monthly_limit(self):
        result = mcp_server.set_category_budget(self.user_id, "Food", 250)
        self.assertEqual(result["monthly_budget"], 250)
        categories = mcp_server.list_categories(self.user_id, "expense")["categories"]
        self.assertEqual(categories[0]["monthly_budget"], 250)
        self.assertIsNone(
            mcp_server.set_category_budget(self.user_id, "Food", None)["monthly_budget"]
        )

    def test_rejects_transfer_edit_cross_user_edit_and_bad_search(self):
        mcp_server.log_income(self.user_id, 10000, source_account="Bank")
        transfer = mcp_server.transfer_funds(self.user_id, "Bank", "Cash", 10)
        with self.assertRaisesRegex(ValueError, "transfer edits are not supported"):
            mcp_server.edit_transaction(self.user_id, transfer["transaction_id"], amount=11)
        with self.assertRaisesRegex(ValueError, "transaction not found"):
            mcp_server.edit_transaction("another-user", transfer["transaction_id"], amount=11)
        with self.assertRaisesRegex(ValueError, "limit"):
            mcp_server.search_transactions(self.user_id, limit=101)

    def test_mcp_transport_registers_registry_tools(self):
        from app.services_ai.mcp_transport import mcp

        self.assertIs(mcp, mcp_server.mcp)
        tools = asyncio.run(mcp.list_tools())
        names = {tool.name for tool in tools}
        self.assertTrue(set(mcp_server.TOOL_REGISTRY).issubset(names))


if __name__ == "__main__":
    unittest.main()
