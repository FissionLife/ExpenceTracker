import os
import re
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.envfile import INSECURE_DEV_KEY  # noqa: F401  (also used by main.py and crypto.py)

ROOT = (
    Path(__file__).resolve().parent.parent
)  # the project folder: .env and the SQLite file live here


class Settings(BaseSettings):
    """Application configuration for WalletLedger.

    Values come from real environment variables first, then the ``.env`` file.
    See ``.env.example`` for every setting with a short explanation.
    """

    model_config = SettingsConfigDict(
        # The test-suite sets WALLETLEDGER_TESTING so it never reads (or spends) a developer's real .env
        # always the project's .env, whichever folder the program is started from
        env_file=None
        if os.environ.get("WALLETLEDGER_TESTING", "").lower() in ("1", "true", "yes")
        else ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # unknown lines in .env must not stop the app
        env_ignore_empty=True,  # "SETTING=" (an unset ${VAR} in compose, a blank line) means "not set", not a crash
    )

    @field_validator("*", mode="before")
    @classmethod
    def _strip_text(cls, value):
        """ "USE_WEBHOOK=true " (a stray space) must work like "true"."""
        return value.strip() if isinstance(value, str) else value

    # Database: pick ONE backend with DB_MODE, or give a full DATABASE_URL (which wins).
    #   DB_MODE=sqlite   -> a local file (SQLITE_PATH). Zero setup; best for dev and single-user.
    #   DB_MODE=postgres -> POSTGRES_* settings (a PostgreSQL server you run).
    db_mode: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_path: str = "./walletledger.db"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "appdb"
    # Full SQLAlchemy URL. Empty = build it from DB_MODE. If set, DB_MODE is inferred from it.
    database_url: str = ""

    # How the AI's ledger tools are called (app/services_ai/tool_client.py):
    #   direct -> plain Python function calls in this process (default, fastest).
    #   mcp    -> through the real MCP server, started as a child process over stdio.
    tool_mode: Literal["direct", "mcp"] = "direct"

    # Telegram bot (token from BotFather). Empty token = bot disabled, /bot/simulate_chat still works.
    telegram_bot_token: str = ""
    # More bots on the same backend: comma-separated tokens, in addition to TELEGRAM_BOT_TOKEN.
    # Every bot shares one ledger: a person's data follows their Telegram chat id, whichever bot they use.
    telegram_bot_tokens: str = ""
    # False = long polling (local dev). True = webhook (production, needs a public HTTPS URL).
    use_webhook: bool = False
    # Show the agent's step-by-step flow under every Telegram reply (a chat can switch it with /flow).
    # The terminal always prints the flow.
    show_flow: bool = False
    # In the terminal, also print where each step lives in the code (file:line, who called it).
    flow_code: bool = True
    # Full public URL Telegram should call, e.g. https://example.com/bot/webhook
    webhook_url: str = ""
    # Secret Telegram echoes back on every webhook call. Empty = derived from the bot token.
    telegram_webhook_secret: str = ""
    # Largest statement upload accepted from chat, in megabytes.
    max_upload_mb: int = 10

    # Only these Telegram chat ids may use the bot (comma-separated). Empty = anyone.
    allowed_chat_ids: str = ""
    # Chat ids allowed to run /admin (manage shared AI keys, see bot stats). Empty = no admins.
    admin_chat_ids: str = ""
    # /bot/simulate_chat and the token-less /bot/webhook let the caller act as ANY chat id.
    # Unset = enabled only while no bot token is configured; true/false overrides.
    enable_dev_endpoints: bool | None = None
    onboarding: bool | None = (
        None  # ask a new user where they pay from; None = on (off in the test suite)
    )

    # The timezone people live in. Times are stored in UTC; "today", daily totals and budget months follow this.
    timezone: str = (
        "UTC"
        if os.environ.get("WALLETLEDGER_TESTING", "").lower() in ("1", "true", "yes")
        else "Asia/Kolkata"
    )

    # Public address of this server (https://...), used for dashboard links and the Telegram Mini App.
    # Empty = http://localhost:8000 (opens in a browser on this computer only).
    public_url: str = ""
    # How long a dashboard link stays valid, in minutes.
    dashboard_link_minutes: int = 120

    # Security key for encrypting user API keys, chat history and transaction notes (app/crypto.py)
    secret_key: str = (
        INSECURE_DEV_KEY  # a blank value counts as the placeholder (see _blank_secret)
    )
    # Previous SECRET_KEY values (comma-separated), still accepted for reading during a rotation.
    secret_key_old: str = ""

    # Optional server-side AI keys, used only when a user has not added their own key.
    # *_API_KEYS accept several keys separated by commas (round-robin rotation).
    gemini_api_keys: str = ""
    openai_api_keys: str = ""
    anthropic_api_keys: str = ""
    groq_api_keys: str = ""
    openrouter_api_keys: str = ""
    xai_api_keys: str = ""
    deepseek_api_keys: str = ""
    mistral_api_keys: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    xai_api_key: str = ""
    deepseek_api_key: str = ""
    mistral_api_key: str = ""

    # Optional model overrides (LiteLLM names, e.g. gemini/gemini-flash-latest)
    ai_model_gemini: str = ""
    ai_model_openai: str = ""
    ai_model_anthropic: str = ""
    ai_model_groq: str = ""
    ai_model_openrouter: str = ""
    ai_model_xai: str = ""
    ai_model_deepseek: str = ""
    ai_model_mistral: str = ""
    ai_model_ollama: str = ""
    ollama_api_base: str = ""
    # Ask each provider which models a key can use and pick the best one automatically, so model
    # names never need manual updates. Falls back to the static defaults if the lookup fails.
    dynamic_models: bool = True

    @field_validator("secret_key", mode="after")
    @classmethod
    def _blank_secret(cls, value: str) -> str:
        """An empty SECRET_KEY must not become an empty encryption key: it is the (flagged) placeholder."""
        return value or INSECURE_DEV_KEY

    @model_validator(mode="after")
    def _resolve_database(self) -> Settings:
        url = self.database_url.strip()
        if url:
            # Heroku-style / bare postgres URLs -> the psycopg2 driver we ship
            for prefix in ("postgres://", "postgresql://"):
                if url.startswith(prefix):
                    url = "postgresql+psycopg2://" + url[len(prefix) :]
            if not re.match(r"^(sqlite|postgresql)(\+\w+)?://", url):
                raise ValueError(
                    "DATABASE_URL must start with sqlite:// or postgresql:// (or postgres://)"
                )
            self.database_url = url
            self.db_mode = "sqlite" if url.startswith("sqlite") else "postgres"
        elif self.db_mode == "sqlite":
            path = Path(self.sqlite_path.replace("\\", "/"))
            if not path.is_absolute():
                path = ROOT / path  # not "wherever the shell happens to be"
            self.database_url = f"sqlite:///{path.as_posix()}"
        else:
            self.database_url = (
                f"postgresql+psycopg2://{quote_plus(self.postgres_user)}:"
                f"{quote_plus(self.postgres_password)}@{self.postgres_host}:"
                f"{self.postgres_port}/{self.postgres_db}"
            )
        return self


settings = Settings()


def is_chat_allowed(chat_id: str) -> bool:
    allowed = {c.strip() for c in settings.allowed_chat_ids.split(",") if c.strip()}
    return not allowed or str(chat_id) in allowed


def is_admin(chat_id: str) -> bool:
    admins = {c.strip() for c in settings.admin_chat_ids.split(",") if c.strip()}
    return str(chat_id) in admins


def bot_tokens() -> list[str]:
    """Every configured bot token (TELEGRAM_BOT_TOKEN first, then TELEGRAM_BOT_TOKENS), no duplicates."""
    found: list[str] = []
    extra = re.split(r"[,;\s]+", settings.telegram_bot_tokens)  # commas, semicolons or spaces
    for raw in [settings.telegram_bot_token, *extra]:
        token = raw.strip()
        if token and token not in found:
            found.append(token)
    return found


def dev_endpoints_enabled() -> bool:
    if settings.enable_dev_endpoints is not None:
        return settings.enable_dev_endpoints
    # The simulator acts as ANY chat id, so it stays off on a public address even if no token is set
    return not bot_tokens() and not settings.public_url.strip().lower().startswith("https://")
