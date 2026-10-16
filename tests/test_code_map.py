"""scripts/code_map.py: the function-by-function map finds the real calls and the LangGraph wiring."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
import sys  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import code_map  # noqa: E402


class CodeMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = code_map.Project()

    def callees(self, fid):
        return {code_map.short(self.p, t) for t in self.p.calls.get(fid, {})}

    def callers(self, fid):
        return {code_map.short(self.p, c) for c, _ in self.p.callers(fid)}

    def test_the_planner_calls_the_one_ai_call_and_is_reached_by_three_routes(self):
        planner = "app.bot.agent.planner_node"
        self.assertIn("ai_gateway.complete", self.callees(planner))
        self.assertTrue(
            {"agent.route_after_ingress", "agent.route_after_guard", "agent.route_after_executor"}
            <= self.callers(planner)
        )

    def test_decorators_are_not_counted_as_calls(self):
        # @flow.traced("planner") runs when the file loads, not when the planner runs
        self.assertNotIn("flow.traced", self.callees("app.bot.agent.planner_node"))

    def test_the_wiring_is_read_from_the_route_functions(self):
        edges = {(a, b) for a, b, _ in self.p.wiring}
        for edge in [
            ("ingress", "planner"),
            ("planner", "guard"),
            ("planner", "renderer"),
            ("guard", "executor"),
            ("executor", "planner"),
            ("executor", "renderer"),
        ]:
            self.assertIn(edge, edges)

    def test_the_ledger_write_is_reachable_from_the_tool_dispatcher(self):
        self.assertIn(
            "mcp_server.log_expense", self.callees("app.services_ai.mcp_server.execute_tool")
        )
        self.assertIn(
            "ledger.record_transaction", self.callees("app.services_ai.mcp_server.log_expense")
        )

    def test_routes_are_labelled(self):
        self.assertEqual(self.p.fns["main.health"].route, "GET /health")

    def test_the_document_covers_every_function(self):
        text = code_map.build_doc(self.p)
        for name in ("planner_node", "record_transaction", "handle_incoming", "init_db"):
            self.assertIn(f"`{name}`", text)
        self.assertIn("## The spine", text)


if __name__ == "__main__":
    unittest.main()
