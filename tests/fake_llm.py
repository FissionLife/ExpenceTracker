"""Scripted stand-in for the LLM, for offline tests of the AI agent.

The agent calls ``ai_gateway.complete``; ``FakeLLM`` replaces it and returns queued answers shaped
like LiteLLM responses, so a test states exactly what the "AI" decided:

    with FakeLLM() as llm:
        llm.add(tool("log_expense", amount=200, category="Food & Dining"))
        reply = asyncio.run(bot_main.handle_text(chat, "spent 200 on chai"))

Each queued item answers one call (planner rounds, the yes/no classifier). A finished queue answers
with plain text "OK" so a read tool's follow-up round still gets a reply. ``llm.calls`` holds the
messages of every call for assertions. ``FakeLLM(no_key=True)`` behaves like a user with no AI key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import patch

from app.services_ai import ai_gateway


def tool(name: str, **args: Any) -> dict[str, Any]:
    """One tool call the AI makes."""
    return {"name": name, "args": args}


def text_response(content: str) -> Any:
    return NS(choices=[NS(message=NS(content=content, tool_calls=None))])


def tools_response(*calls: dict[str, Any]) -> Any:
    tool_calls = [
        NS(id=f"call_{i}", function=NS(name=c["name"], arguments=json.dumps(c["args"])))
        for i, c in enumerate(calls)
    ]
    return NS(choices=[NS(message=NS(content=None, tool_calls=tool_calls))])


class FakeLLM:
    def __init__(self, *answers: Any, no_key: bool = False, default: str = "OK") -> None:
        self.queue: list[Any] = []
        self.calls: list[list[dict[str, Any]]] = []
        self.no_key = no_key
        self.default = default
        for a in answers:
            self.add(a)
        self._patch = patch.object(ai_gateway, "complete", self._complete)

    def add(self, *items: Any) -> FakeLLM:
        """Queue one answer: a tool(...) dict, a list of tool(...) dicts, text, or an exception."""
        if len(items) > 1:
            self.queue.append(tools_response(*items))
            return self
        item = items[0]
        if isinstance(item, dict):
            item = tools_response(item)
        elif isinstance(item, list):
            item = tools_response(*item)
        elif isinstance(item, str):
            item = text_response(item)
        self.queue.append(item)
        return self

    async def _complete(self, user_id: str, messages: list[dict[str, Any]], *a: Any, **kw: Any):
        self.calls.append(messages)
        if self.no_key:
            raise ai_gateway.NoAPIKeyError("No AI API key found.")
        item = self.queue.pop(0) if self.queue else text_response(self.default)
        if isinstance(item, BaseException):
            raise item
        return item

    def __enter__(self) -> FakeLLM:
        self._patch.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._patch.stop()
