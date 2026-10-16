"""TOOL_MODE switch: direct Python calls vs the real MCP server over stdio.

The MCP tests start the actual server process (python -m app.services_ai.mcp_transport) against a
throwaway SQLite file. No network, no LLM.
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-toolmode-test-")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ.setdefault(
    "DATABASE_URL", "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
)

import asyncio  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app.bot.agent import TOOLS_SCHEMA  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import engine, init_db  # noqa: E402
from app.services_ai import tool_client  # noqa: E402
from app.services_ai.mcp_server import TOOL_REGISTRY  # noqa: E402


class ToolModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    @classmethod
    def tearDownClass(cls):
        engine.dispose()

    def setUp(self):
        self.user = f"tm-{uuid.uuid4().hex[:10]}"

    def test_default_mode_is_direct(self):
        self.assertEqual(type(settings).model_fields["tool_mode"].default, "direct")

    def test_every_agent_tool_exists_in_the_registry(self):
        names = {t["function"]["name"] for t in TOOLS_SCHEMA}
        self.assertEqual(names - set(TOOL_REGISTRY), set())

    def test_direct_mode_calls_functions(self):
        async def go():
            with patch.object(settings, "tool_mode", "direct"):
                self.assertEqual(await tool_client.start(), "direct")
                self.assertFalse(tool_client.is_connected())
                await tool_client.call_tool(
                    "log_income", user_id=self.user, amount=1000, source_account="Cash"
                )
                await tool_client.call_tool(
                    "log_expense", user_id=self.user, amount=40, category="Food"
                )
                bundle = await tool_client.call_tool("financial_advice_bundle", user_id=self.user)
                self.assertEqual(set(bundle), {"recurring", "budget", "emergency", "breakdown"})
                with self.assertRaises(ValueError):
                    await tool_client.call_tool(
                        "log_expense", user_id=self.user, amount=-1, category="Food"
                    )

        asyncio.run(go())

    def test_mcp_mode_without_server_fails_clearly(self):
        async def go():
            with (
                patch.object(settings, "tool_mode", "mcp"),
                self.assertRaisesRegex(RuntimeError, "not running"),
            ):
                await tool_client.call_tool("get_pipe_balances", user_id=self.user)

        asyncio.run(go())

    def test_mcp_mode_round_trip_over_stdio(self):
        async def go():
            with patch.object(settings, "tool_mode", "mcp"):
                self.assertEqual(await tool_client.start(), "mcp")
                try:
                    self.assertTrue(tool_client.is_connected())
                    self.assertTrue(set(TOOL_REGISTRY).issubset(tool_client.server_tools()))
                    self.assertEqual(
                        len(tool_client.available_tools(TOOLS_SCHEMA)), len(TOOLS_SCHEMA)
                    )
                    await tool_client.call_tool(
                        "log_income", user_id=self.user, amount=1000, source_account="Bank"
                    )
                    await tool_client.call_tool(
                        "log_expense",
                        user_id=self.user,
                        amount=250,
                        category="Food",
                        account_name="Bank",
                    )
                    res = await tool_client.call_tool("get_pipe_balances", user_id=self.user)
                    balances = {a["name"]: a["balance"] for a in res["accounts"]}
                    self.assertEqual(balances["Bank"], 750.0)
                    bundle = await tool_client.call_tool(
                        "financial_advice_bundle", user_id=self.user
                    )
                    self.assertIn("breakdown", bundle)
                    with self.assertRaisesRegex(tool_client.ToolCallError, "greater than zero"):
                        await tool_client.call_tool(
                            "log_expense", user_id=self.user, amount=-5, category="Food"
                        )
                finally:
                    await tool_client.stop()
                self.assertFalse(tool_client.is_connected())

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
