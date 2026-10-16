"""See the agent flow and the database, with no AI key and no Telegram.

    uv run python scripts/show_flow.py            # demo: three messages, flow + database after each
    uv run python scripts/show_flow.py --graph    # the real LangGraph as a Mermaid diagram

The demo runs the REAL graph, guardrails, ledger and database (a throwaway SQLite file). Only the
AI is replaced by a fixed script that answers like a good model would, so it is repeatable.
It never reads your .env.
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Before any app import: a throwaway database, and never the real .env or real keys.
os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(tempfile.mkdtemp()) / "demo.db").replace(
    "\\", "/"
)
os.environ["TELEGRAM_BOT_TOKEN"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.bot import agent  # noqa: E402
from app.db import engine, init_db  # noqa: E402
from app.services_ai.mcp_server import execute_tool  # noqa: E402
from tests.fake_llm import FakeLLM, tool  # noqa: E402

CHAT = "demo-chat"


def show_tables(title: str) -> None:
    """The raw rows, exactly as stored (note the encrypted description)."""
    with engine.connect() as db:
        accounts = db.execute(text("SELECT name, balance FROM accounts ORDER BY name")).all()
        txs = db.execute(
            text(
                "SELECT t.transaction_type, t.amount, a.name, t.description "
                "FROM transactions t LEFT JOIN accounts a ON a.id = t.account_id ORDER BY t.created_at"
            )
        ).all()
    print(f"\n   DATABASE ({title})")
    print("   accounts      : " + ", ".join(f"{n} = {b:,.0f}" for n, b in accounts))
    for kind, amount, account, description in txs:
        stored = (description or "-")[:34] + (
            "..." if description and len(description) > 34 else ""
        )
        print(
            f"   transactions  : {kind:<9} {amount:>8,.0f}  {account or '-':<5} description={stored}"
        )


def step(number: int, user_text: str, *answers):
    print(f"\n{'=' * 78}\n{number}. USER SAYS: {user_text!r}\n{'=' * 78}")
    with FakeLLM(*answers):
        outcome = asyncio.run(agent.process_user_interaction_ex(CHAT, user_text))
    for i, line in enumerate(outcome.trace, 1):
        print(f"   {i}. {line}")
    print(f"   -> reply: {outcome.text.splitlines()[0] if outcome.text else ''}")
    if outcome.needs_confirmation:
        print("   -> the graph is PAUSED here (interrupt), waiting for the user's Yes/Cancel")
    return outcome


def demo() -> None:
    init_db()
    execute_tool(
        "log_income", user_id=CHAT, amount=20000, source_account="Bank", description="salary"
    )
    execute_tool(
        "log_income", user_id=CHAT, amount=1000, source_account="Cash", description="cash in hand"
    )
    show_tables("before")

    step(
        1,
        "spent 450 on dinner",
        tool("log_expense", amount=450, category="Food & Dining", account_name="Cash"),
    )
    show_tables("after 1: one new transaction, Cash balance changed together")

    paused = step(
        2,
        "move 6000 from bank to cash",
        tool("transfer_funds", from_account="Bank", to_account="Cash", amount=6000),
    )
    show_tables("after 2: unchanged, the graph is waiting")

    if paused.needs_confirmation:
        print(f"\n{'=' * 78}\n3. USER TAPS YES\n{'=' * 78}")
        outcome = asyncio.run(agent.resume_pending(CHAT, True))
        for i, line in enumerate(outcome.trace, 1):
            print(f"   {i}. {line}")
        show_tables("after 3: two balances moved, one transfer row added")


def graph() -> None:
    print(agent.agent_app.get_graph().draw_mermaid())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--graph", action="store_true", help="print the real graph as Mermaid")
    graph() if parser.parse_args().graph else demo()
