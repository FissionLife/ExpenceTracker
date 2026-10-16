"""One door for every ledger tool call: direct Python, or the real MCP server.

TOOL_MODE=direct (default)  call the functions in ``mcp_server.TOOL_REGISTRY`` in-process.
TOOL_MODE=mcp               talk to the MCP server (``python -m app.services_ai.mcp_transport``)
                            over stdio. It is started once at app startup (``main.py`` lifespan)
                            and closed at shutdown.

Callers (the agent and PDF import) only use ``call_tool()`` and
``available_tools()``; they never need to know which mode is on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from app.config import settings
from app.services_ai.mcp_server import execute_tool

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SERVER_MODULE = "app.services_ai.mcp_transport"
START_TIMEOUT_S = 60.0
CALL_TIMEOUT_S = 60.0


class ToolCallError(RuntimeError):
    """A tool failed inside the MCP server. The message is the tool's own error text."""


_stack: AsyncExitStack | None = None
_session: Any = None  # mcp.ClientSession while connected
_server_tools: list[str] = []


def mode() -> str:
    """The configured mode: ``"mcp"`` or ``"direct"``."""
    return "mcp" if settings.tool_mode == "mcp" else "direct"


def is_connected() -> bool:
    return _session is not None


def server_tools() -> list[str]:
    """Tool names the MCP server advertised at startup (empty in direct mode)."""
    return list(_server_tools)


def _server_params():
    from mcp import StdioServerParameters

    env = dict(os.environ)  # same SECRET_KEY, AI settings etc. as this process
    env["DATABASE_URL"] = settings.database_url  # the exact database this process uses
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    # Same working directory as this process, so the server reads the same .env file.
    return StdioServerParameters(
        command=sys.executable, args=["-m", SERVER_MODULE], env=env, cwd=os.getcwd()
    )


async def start() -> str:
    """Connect to the MCP server when TOOL_MODE=mcp. Returns the active mode.

    Must be awaited and later ``stop()``-ed from the same task (the FastAPI lifespan does this).
    Raises if the server cannot be started, so a broken MCP demo fails loudly at startup.
    """
    global _stack, _session, _server_tools
    if mode() != "mcp":
        return "direct"
    if _session is not None:
        return "mcp"

    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    stack = AsyncExitStack()
    try:
        read, write = await stack.enter_async_context(stdio_client(_server_params()))
        session = await stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), START_TIMEOUT_S)
        listed = await asyncio.wait_for(session.list_tools(), START_TIMEOUT_S)
    except BaseException:
        await stack.aclose()
        raise
    _stack, _session = stack, session
    _server_tools = sorted(t.name for t in listed.tools)
    logger.info("MCP tool server connected over stdio: %d tools", len(_server_tools))
    return "mcp"


async def stop() -> None:
    """Close the MCP session and the server process (no-op in direct mode)."""
    global _stack, _session, _server_tools
    stack, _stack, _session, _server_tools = _stack, None, None, []
    if stack is not None:
        await stack.aclose()
        logger.info("MCP tool server stopped")


def _error_text(name: str, result: Any) -> str:
    text = " ".join(getattr(c, "text", "") for c in result.content or []).strip()
    prefix = f"Error executing tool {name}: "
    return text[len(prefix) :] if text.startswith(prefix) else (text or f"{name} failed")


def _result_data(result: Any) -> dict[str, Any]:
    if isinstance(result.structured_content, dict):
        return result.structured_content
    text = "".join(getattr(c, "text", "") for c in result.content or [])
    try:
        data = json.loads(text)
    except ValueError:
        return {"status": "success", "text": text}
    return data if isinstance(data, dict) else {"status": "success", "result": data}


async def call_tool(name: str, **args: Any) -> dict[str, Any]:
    """Run one ledger tool and return its result dict. Raises on tool failure in both modes."""
    if mode() != "mcp":
        return await asyncio.to_thread(execute_tool, name, **args)
    if _session is None:
        raise RuntimeError(
            "TOOL_MODE=mcp but the MCP tool server is not running (it starts with the app)"
        )
    result = await _session.call_tool(name, args, read_timeout_seconds=CALL_TIMEOUT_S)
    if result.is_error:
        raise ToolCallError(_error_text(name, result))
    return _result_data(result)


def available_tools(schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The LLM tool schema, limited (in MCP mode) to tools the server actually offers.

    The descriptions stay the agent's prompt-tuned ones; the server decides which tools exist.
    """
    if mode() != "mcp" or _session is None:
        return schema
    offered = set(_server_tools)
    return [t for t in schema if t["function"]["name"] in offered]
