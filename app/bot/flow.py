"""Shows what the agent is doing, one plain line per step.

Every node of the LangGraph agent is wrapped with ``@traced("name")``. After the node runs, one line
describing what it decided or did is (1) printed to the terminal through the ``flow`` logger and
(2) kept in ``state["trace"]``, so Telegram can show the same lines under the reply (``/flow on``
or ``SHOW_FLOW=true``). The wrapper only reads the node's input and output: it never changes what
the node does. Lines are plain ASCII on purpose, so they are safe in any terminal.

Example, "spent 450 on dinner" (the at / from / calls lines say where in the code that step is)::

    planner: round 1: the AI picked 1 tool(s): log_expense(amount=450, category=Food & Dining)
        at    app/bot/agent.py:1030 planner_node()
        from  app/bot/agent.py:1262 route_after_ingress()  (it sent the graph here)
        calls app/services_ai/ai_gateway.py:696 complete()  (the one AI call)
    guard: all checks passed, no approval needed
    executor: ran 1 tool(s): log_expense: inserted 1 transaction (...); Cash balance now Rs 550
        at    app/bot/agent.py:1231 executor_node()
        from  app/bot/agent.py:1268 route_after_guard()
        calls tool_client.call_tool() -> mcp_server.execute_tool() -> mcp_server.log_expense()
              -> ledger.record_transaction()
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

from langgraph.errors import GraphInterrupt

from app.config import settings

flow_log = logging.getLogger("flow")

ROOT = Path(__file__).resolve().parents[2]
_chat_mode: dict[str, str] = {}  # per-chat /flow off|on|code; lost on restart (a display setting)


def mode(chat_id: str) -> str:
    """off = nothing under replies, on = the steps, code = the steps plus where each lives in the code."""
    return _chat_mode.get(str(chat_id), "on" if settings.show_flow else "off")


def enabled(chat_id: str) -> bool:
    """Should this chat's replies carry the "how I got here" lines?"""
    return mode(chat_id) != "off"


def set_mode(chat_id: str, value: str) -> None:
    _chat_mode[str(chat_id)] = value


def set_enabled(chat_id: str, on: bool) -> None:
    set_mode(chat_id, "on" if on else "off")


def footer(trace: list[str], trace_at: list[str] | None = None, code: bool = False) -> str:
    """The lines as a short block under a Telegram reply (with the code locations when ``code``)."""
    if not trace:
        return ""
    lines = []
    for i, line in enumerate(trace, 1):
        lines.append(f"{i}. {line}")
        if code and trace_at and i <= len(trace_at) and trace_at[i - 1]:
            lines.extend("      " + part for part in trace_at[i - 1].splitlines())
    return "\n\nHow I got here:\n" + "\n".join(lines)


# ------------------------------------------------------------------------------ wording


def _rs(value: Any) -> str:
    try:
        number = float(value)
    except TypeError, ValueError:
        return str(value)
    return f"Rs {number:,.0f}" if number == int(number) else f"Rs {number:,.2f}"


def _call(call: dict[str, Any]) -> str:
    """log_expense(amount=450, category=Food & Dining): the first few arguments, kept short."""
    args = call.get("args") if "args" in call else call.get("tool_args")
    bits = [
        f"{k}={str(v)[:18]}"
        for k, v in (args or {}).items()
        if v is not None and k not in ("user_id", "description", "note")
    ][:4]
    name = call.get("name") or call.get("tool_name")
    return f"{name}({', '.join(bits)})"


def db_effect(run: dict[str, Any]) -> str:
    """What one tool call did to the database, in words."""
    from app.bot.agent import WRITE_TOOLS

    name, result = run["tool_name"], run.get("result") or {}
    if run.get("error"):
        return f"{name} failed ({str(run['error'])[:60]}); nothing was written"
    if name in ("log_expense", "log_income"):
        kind = "expense" if name == "log_expense" else "income"
        return (
            f"{name}: inserted 1 transaction ({kind} {_rs(result.get('amount'))}, "
            f"{result.get('category')}); {result.get('account')} balance now {_rs(result.get('new_balance'))}"
        )
    if name in ("transfer_funds", "transfer_between_pipes"):
        return (
            f"{name}: moved {_rs(result.get('transferred'))} {result.get('from_account')} -> "
            f"{result.get('to_account')}; 2 balances updated, 1 transfer row inserted"
        )
    if name in WRITE_TOOLS:
        return f"{name}: changed the database"
    return f"{name}: read-only query, nothing changed"


def describe(node: str, before: dict[str, Any], after: dict[str, Any]) -> str:
    """One line saying what this node just did (``before`` = its input, ``after`` = its update)."""
    from app.bot import agent
    from app.bot import guardrails as gr

    if node == "ingress":
        if after.get("reply_text") and not after.get("convo"):
            return "an API key was pasted: saved it, it is never sent to the AI"
        accounts = len((after.get("context") or {}).get("accounts") or [])
        history = max(len(after.get("convo") or []) - 2, 0)
        persona = f"; persona {after['persona']}" if after.get("persona") else ""
        return (
            f"loaded {accounts} account(s) and {history} earlier message(s); "
            f"card/ID numbers masked{persona}"
        )

    if node == "planner":
        proposed = after.get("proposed") or []
        if proposed:
            n = len(proposed)
            rnd = after.get("step") or before.get("step") or 1
            return f"round {rnd}: the AI picked {n} tool(s): " + "; ".join(
                _call(c) for c in proposed
            )
        text = after.get("reply_text") or ""
        if after.get("ai_failed"):
            return "the AI was unavailable, so the finished steps are used as they are"
        if text == agent.NEEDS_KEY_TEXT:
            return "no AI key is set, so nothing was sent to a model"
        if text == agent.AI_BUSY_TEXT:
            return "the AI did not answer; nothing was recorded"
        if text == agent.TOO_MANY_STEPS_TEXT:
            return f"stopped after {agent.MAX_STEPS} rounds"
        if text == gr.OFF_TOPIC_TEXT:
            return "the AI marked the message as off-topic and declined"
        if after.get("reply_source") == "planner":
            if before.get("tool_runs"):
                return "the AI read the tool results and wrote the answer"
            return "the AI answered directly, no tool needed"
        return "the AI gave no usable answer"

    if node == "guard":
        if after.get("declined"):
            return "you said No: nothing was changed"
        if after.get("asked"):
            return "this needed your approval; you said Yes, so it goes ahead"
        if after.get("rejected") and not after.get("proposed"):
            return "rejected the AI's call(s) as invalid; the AI will try again"
        return "all checks passed, no approval needed"

    if node == "executor":
        count = int(after.get("last_round") or 0)
        runs = (after.get("tool_runs") or [])[-count:] if count else []
        return f"ran {len(runs)} tool(s): " + " | ".join(db_effect(r) for r in runs)

    if node == "renderer":
        return "reply written from " + (after.get("render_how") or "a fixed template")
    return ""


# ------------------------------------------------------------------------------ where in the code

_MODULES = {
    "agent": "app.bot.agent",
    "gr": "app.bot.guardrails",
    "keys": "app.bot.keys",
    "main": "app.bot.main",
    "telegram": "app.bot.telegram",
    "replies": "app.bot.replies",
    "gw": "app.services_ai.ai_gateway",
    "prompts": "app.services_ai.prompts",
    "tool_client": "app.services_ai.tool_client",
    "mcp": "app.services_ai.mcp_server",
    "ledger": "app.services.ledger",
}

# For each graph node: its function, and the main functions it calls (read it top to bottom).
_NODES: dict[str, dict[str, Any]] = {
    "ingress": {
        "fn": "agent.ingress_node",
        "calls": [
            "keys.save_keys",
            "gr.check_input",
            "agent._load_context",
            "agent._history_dicts",
            "prompts.build_planner_prompt",
        ],
    },
    "planner": {
        "fn": "agent.planner_node",
        "calls": ["tool_client.available_tools", "gw.complete"],
    },
    "guard": {
        "fn": "agent.guard_node",
        "calls": ["gr.check_call", "agent.confirmation_reason", "agent._budget_overage_reason"],
    },
    "executor": {"fn": "agent.executor_node", "calls": ["tool_client.call_tool"]},
    "renderer": {
        "fn": "agent.renderer_node",
        "calls": ["replies.compact", "gr.leaks", "gr.ungrounded_figures", "gr.sanitize_reply"],
    },
}
_ENTRY = ["telegram.on_text", "main.handle_incoming", "agent.process_user_interaction_ex"]

# The ledger function behind each tool that changes data.
_LEDGER = {
    "log_expense": "record_transaction",
    "log_income": "record_transaction",
    "set_account_balance": "record_transaction",
    "transfer_funds": "transfer_between_pipes",
    "transfer_between_pipes": "transfer_between_pipes",
    "delete_transaction": "undo_transaction",
}


def where(target: Any) -> str:
    """A name like "agent.planner_node" (or the function itself) -> "app/bot/agent.py:1030 planner_node()"."""
    try:
        if isinstance(target, str):
            alias, _, name = target.partition(".")
            obj = getattr(importlib.import_module(_MODULES[alias]), name)
        else:
            obj = target
        obj = inspect.unwrap(obj)
        path = Path(inspect.getsourcefile(obj)).resolve()
        lines, start = inspect.getsourcelines(obj)
        # a decorated function starts at its decorator: point at the def line instead
        offset = next(
            (i for i, s in enumerate(lines) if s.lstrip().startswith(("def ", "async def "))), 0
        )
        return f"{path.relative_to(ROOT).as_posix()}:{start + offset} {obj.__name__}()"
    except Exception:
        return str(target)


def _tool_chain(name: str) -> str:
    """tool_client.call_tool -> execute_tool -> the tool's own function -> the ledger function."""
    from app.services_ai import mcp_server, tool_client

    parts = [where("tool_client.call_tool")]
    if tool_client.mode() == "mcp":
        parts.append("MCP server process (python -m app.services_ai.mcp_transport)")
    else:
        parts.append(where("mcp.execute_tool"))
    handler = mcp_server.TOOL_REGISTRY.get(name)
    if handler is not None:
        parts.append(where(handler))
    if name in _LEDGER:
        parts.append(where(f"ledger.{_LEDGER[name]}"))
    return " -> ".join(parts)


def code_for(node: str, before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Where this step lives in the code: the function, who sent the graph here, what it calls."""
    spec = _NODES[node]
    lines = [f"at    {where(spec['fn'])}"]
    if node == "ingress":
        lines.append("from  " + " -> ".join(where(r) for r in _ENTRY))
    else:
        previous = (before.get("trace") or [""])[-1].split(":")[0] or "ingress"
        lines.append(f"from  {where(f'agent.route_after_{previous}')}  (it sent the graph here)")
    if node == "executor":
        count = int(after.get("last_round") or 0)
        runs = (after.get("tool_runs") or [])[-count:] if count else []
        for run in runs:
            lines.append(f"calls {_tool_chain(run['tool_name'])}")
    else:
        lines.append("calls " + ", ".join(where(c) for c in spec["calls"]))
    return lines


# ------------------------------------------------------------------------------ decorator


def _record(node: str, state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    try:
        line = f"{node}: {describe(node, state, update)}"
        flow_log.info("[chat %s] %s", state.get("chat_id", "?"), line)
        at = ""
        try:
            at = "\n".join(code_for(node, state, update))
            if settings.flow_code:
                for part in at.splitlines():
                    flow_log.info("[chat %s]     %s", state.get("chat_id", "?"), part)
        except Exception:
            flow_log.debug("could not locate %s", node, exc_info=True)
        base = [] if node == "ingress" else list(state.get("trace") or [])
        base_at = [] if node == "ingress" else list(state.get("trace_at") or [])
        return {**update, "trace": [*base, line], "trace_at": [*base_at, at]}
    except Exception:  # showing the flow must never break a message
        flow_log.debug("could not describe %s", node, exc_info=True)
        return update


def traced(node: str):
    """Wrap a graph node so each run is logged and added to ``state["trace"]``."""

    def decorate(fn):
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def wrapper(state):
                try:
                    update = await fn(state)
                except GraphInterrupt:
                    flow_log.info(
                        "[chat %s] %s: paused, waiting for Yes/Cancel", state.get("chat_id"), node
                    )
                    raise
                return _record(node, state, update)

        else:

            @functools.wraps(fn)
            def wrapper(state):
                try:
                    update = fn(state)
                except GraphInterrupt:
                    flow_log.info(
                        "[chat %s] %s: paused, waiting for Yes/Cancel", state.get("chat_id"), node
                    )
                    raise
                return _record(node, state, update)

        return wrapper

    return decorate
