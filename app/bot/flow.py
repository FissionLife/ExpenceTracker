"""Shows what the agent is doing, one plain line per step.

Every node of the LangGraph agent is wrapped with ``@traced("name")``. After the node runs, one line
describing what it decided or did is (1) printed to the terminal through the ``flow`` logger and
(2) kept in ``state["trace"]``, so Telegram can show the same lines under the reply (``/flow on``
or ``SHOW_FLOW=true``). The wrapper only reads the node's input and output: it never changes what
the node does. Lines are plain ASCII on purpose, so they are safe in any terminal.

Example, "spent 450 on dinner"::

    ingress : loaded 2 accounts and 6 earlier messages; card/ID numbers masked
    planner : round 1: the AI picked 1 tool: log_expense(amount=450, category=Food & Dining)
    guard   : all checks passed, no approval needed
    executor: log_expense: inserted 1 transaction (expense Rs 450, Food & Dining); Cash balance now Rs 550
    renderer: reply written from a fixed template
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any

from langgraph.errors import GraphInterrupt

from app.config import settings

flow_log = logging.getLogger("flow")

_chat_setting: dict[str, bool] = {}  # per-chat /flow on|off; lost on restart (a display setting)


def enabled(chat_id: str) -> bool:
    """Should this chat's replies carry the "how I got here" lines?"""
    return _chat_setting.get(str(chat_id), settings.show_flow)


def set_enabled(chat_id: str, on: bool) -> None:
    _chat_setting[str(chat_id)] = on


def footer(trace: list[str]) -> str:
    """The lines as a short block under a Telegram reply."""
    if not trace:
        return ""
    return "\n\nHow I got here:\n" + "\n".join(f"{i}. {line}" for i, line in enumerate(trace, 1))


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


# ------------------------------------------------------------------------------ decorator


def _record(node: str, state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    try:
        line = f"{node}: {describe(node, state, update)}"
        flow_log.info("[chat %s] %s", state.get("chat_id", "?"), line)
        base = [] if node == "ingress" else list(state.get("trace") or [])
        return {**update, "trace": [*base, line]}
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
