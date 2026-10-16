"""Compatibility entry point for the stdio MCP server in ``mcp_server``.

Run with ``uv run python -m app.services_ai.mcp_transport``.
"""

from app.services_ai.mcp_server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
