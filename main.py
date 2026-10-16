import logging
from contextlib import asynccontextmanager

from app import firstrun

# First start: if the bot is not set up yet, ask the few questions right here and carry on.
# This must run before anything imports app.config (that is what reads .env).
firstrun.ensure_setup()

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.bot import telegram  # noqa: E402
from app.bot.main import bot_router  # noqa: E402
from app.config import INSECURE_DEV_KEY, settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.services_ai import tool_client  # noqa: E402
from app.web.dashboard import STATIC  # noqa: E402
from app.web.dashboard import router as dashboard_router  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    init_db()
    from app import crypto
    from app.web import tokens

    if crypto.weak_secret_key() and tokens.is_https():
        # Dashboard links are signed with SECRET_KEY. A public placeholder means anyone could sign one
        # for any chat id, and the address is reachable from the internet. Refuse to run that way.
        raise RuntimeError(
            "SECRET_KEY is empty, a placeholder or too short, but PUBLIC_URL is a public https address. "
            'Set a long random SECRET_KEY first: python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
    if settings.secret_key == INSECURE_DEV_KEY:
        logging.getLogger(__name__).warning(
            "SECRET_KEY is the public default: set a random one before storing real user API keys"
        )
    # Ledger tools: direct Python calls, or the MCP server over stdio when TOOL_MODE=mcp
    app.state.tool_mode = await tool_client.start()
    # Telegram bot: off without a token, polling locally, webhook when USE_WEBHOOK=true
    app.state.telegram_mode = await telegram.start()
    try:
        yield
    finally:
        await telegram.stop()
        await tool_client.stop()


app = FastAPI(
    title="WalletLedger API",
    version="0.1.0",
    description="AI-powered Telegram financial assistant & personal expense tracker",
    lifespan=lifespan,
)

# Include Telegram Bot Router
app.include_router(bot_router)

# Web dashboard (signed links, read-only)
app.include_router(dashboard_router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def root():
    return {
        "project": "WalletLedger",
        "status": "online",
        "description": "Telegram AI expense tracker & ledger",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "telegram": getattr(app.state, "telegram_mode", "off"),
        "tools": getattr(app.state, "tool_mode", "direct"),
    }


def start_dev():
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    start_dev()
