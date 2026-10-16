#!/usr/bin/env python3
"""WalletLedger setup wizard: answer a few questions, get a working bot.

    uv run python scripts/setup.py              # guided setup (safe to re-run any time)
    uv run python scripts/setup.py --check      # health check only; changes nothing
    uv run python scripts/setup.py --defaults   # no questions: strong secret, SQLite, public bot

You do not have to run this first: `uv run uvicorn main:app --reload --port 8000` asks the same
questions on the very first start, and starts straight away afterwards. The code lives in
app/setup_wizard.py; this file only launches it. See --help for all options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.setup_wizard import main  # noqa: E402

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
