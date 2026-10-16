# WalletLedger

A Telegram bot that tracks money from plain sentences ("spent 450 on dinner"). An LLM agent reads the
message, a small set of ledger tools acts on a SQL database, and deterministic guardrails check every
action and every reply.

**Technical Q&A sheet (database, FastAPI, MCP, LangGraph, architecture, why Telegram): [docs/TECH_QA.md](docs/TECH_QA.md)**

## Run it

```bash
uv sync
uv run uvicorn main:app --reload --port 8000   # first start asks a few setup questions
```

Settings live in `.env` (see `.env.example`): `TELEGRAM_BOT_TOKEN`, `SECRET_KEY`, `DATABASE_URL`
(SQLite by default, PostgreSQL for production), `TOOL_MODE` (`direct` or `mcp`).
Users bring their own AI key with `/setkey`.

## Layout

| Path | What |
|---|---|
| `main.py` | FastAPI app and startup |
| `app/bot/` | Telegram adapter, message router, the LangGraph agent, guardrails, screens |
| `app/services/` | `ledger.py` (the only code that changes balances), `analytics.py` |
| `app/services_ai/` | tool registry and MCP server, tool client, LLM gateway, prompts, statement parser |
| `app/web/` | signed-link dashboard and its static files |
| `app/db.py`, `app/models.py` | database (tables are created at startup) |
| `app/crypto.py` | encryption at rest |
| `scripts/` | `setup.py`, `setup_bot.py`, `protect_data.py` (key rotation) |
