"""First-run check: is the bot set up? If not, ask right here, then carry on starting.

`main.py` calls `ensure_setup()` before anything reads the settings, so these all just work:

    uv run uvicorn main:app --reload --port 8000     # first time: asks a few questions, then starts
    uv run uvicorn main:app --reload --port 8000     # every time after: starts straight away

"Set up" means: a bot token is present and SECRET_KEY is a real random value, or the wizard has been
completed once (it writes SETUP_DONE=true, so skipping the bot token on purpose is remembered).

It never prompts when nobody can answer (Docker, CI, a service) or when told not to:
    SETUP_SKIP=1 (environment or .env)   never ask
    WALLETLEDGER_TESTING=1               the test suite
In those cases it prints one line about what is missing and the server starts as before.

This module imports nothing from the app except `envfile`, on purpose: importing app.config would read
the settings before the wizard has written them.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app import envfile

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


@dataclass
class SetupState:
    done: bool
    missing: list[str] = field(default_factory=list)


def _values(env_path: Path, environ: dict[str, str] | None) -> dict[str, str]:
    """`.env` overlaid with real environment variables (those win, exactly like the app's settings)."""
    merged = dict(envfile.read_env(env_path))
    source = os.environ if environ is None else environ
    for key in (
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_BOT_TOKENS",
        "SECRET_KEY",
        "SETUP_DONE",
        "SETUP_SKIP",
    ):
        if key in source:
            merged[key] = source[key]
    return merged


def setup_state(env_path: Path | None = None, environ: dict[str, str] | None = None) -> SetupState:
    values = _values(env_path or ROOT / ".env", environ)
    if envfile.truthy(values.get("SETUP_DONE")):
        return SetupState(True)
    missing = []
    if not (
        values.get("TELEGRAM_BOT_TOKEN", "").strip()
        or values.get("TELEGRAM_BOT_TOKENS", "").strip()
    ):
        missing.append("a Telegram bot token")
    if envfile.is_weak_secret(values.get("SECRET_KEY")):
        missing.append("a SECRET_KEY")
    return SetupState(not missing, missing)


def console_available() -> bool:
    """Can we ask the person at the keyboard? True for a terminal, even when uvicorn's reloader
    runs the app in a child process whose stdin is closed (we then talk to the console directly)."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return True
    except ValueError, OSError:
        pass
    try:
        # the controlling terminal: CONIN$ on Windows, /dev/tty elsewhere
        with open("CONIN$" if os.name == "nt" else "/dev/tty") as _:
            return True
    except OSError:
        return False


def ensure_setup(
    env_path: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    interactive: bool | None = None,
) -> str:
    """Returns "ok" (already set up), "ran" (the wizard ran) or "skipped" (could not or should not ask)."""
    source = os.environ if environ is None else environ
    path = env_path or ROOT / ".env"
    if source.get("WALLETLEDGER_TESTING") or envfile.truthy(
        _values(path, environ).get("SETUP_SKIP")
    ):
        return "skipped"
    state = setup_state(path, environ)
    if state.done:
        return "ok"
    can_ask = console_available() if interactive is None else interactive
    if not can_ask:
        print(
            f"WalletLedger is not set up yet (missing {' and '.join(state.missing)}). "
            "Run `uv run python scripts/setup.py` in a terminal, or set TELEGRAM_BOT_TOKEN and SECRET_KEY "
            "as environment variables. Starting anyway.",
            file=sys.stderr,
        )
        return "skipped"
    from app import setup_wizard

    try:
        setup_wizard.run_wizard(path, first_run=True)
    except setup_wizard.NoConsole:
        # the terminal went away mid-question: keep what was saved (without SETUP_DONE, so we ask again
        # next time) and start the server anyway
        print(
            "\nNo terminal to ask on. Starting anyway; finish with `uv run python scripts/setup.py`.",
            file=sys.stderr,
        )
        return "skipped"
    return "ran"
