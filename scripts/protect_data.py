#!/usr/bin/env python3
"""Encrypt WalletLedger user data at rest, check progress, and rotate SECRET_KEY.

    uv run python scripts/protect_data.py status              # counts only, changes nothing
    uv run python scripts/protect_data.py encrypt --dry-run   # what would be encrypted
    uv run python scripts/protect_data.py encrypt             # encrypt old plaintext rows
    uv run python scripts/protect_data.py rotate              # re-encrypt with the newest key

New data is encrypted automatically by the app. ``encrypt`` is only needed once, for rows saved
before encryption existed. It is safe to run again (already-encrypted rows are skipped).

To change SECRET_KEY: put the old value in SECRET_KEY_OLD, the new one in SECRET_KEY, restart the
app, run ``rotate``, then remove SECRET_KEY_OLD.

BACK UP THE DATABASE FIRST (copy walletledger.db, or pg_dump). Restoring the backup undoes
everything this script does. It never prints user text. Works with SQLite and PostgreSQL through
the app's own database settings. See docs/TECH_QA.md (section 1, encryption at rest).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlalchemy as sa  # noqa: E402
from cryptography.fernet import InvalidToken  # noqa: E402

from app import crypto  # noqa: E402

# Free text the user typed, encrypted with the field key.
FIELDS = (("chat_messages", "content"), ("transactions", "description"))
# Users' AI keys, encrypted with the vault key (always encrypted; only ``rotate`` touches them).
VAULT = ("api_keys", "encrypted_key")
BATCH = 500


def _table(name: str, column: str) -> sa.TableClause:
    # Plain Text columns on purpose: we want the raw stored value, not the app's decryption.
    return sa.table(name, sa.column("id", sa.String), sa.column(column, sa.Text))


def _batches(engine: sa.Engine, name: str, column: str, size: int = BATCH):
    """Yield (connection, table, rows) per batch, ordered by id; each batch is one transaction."""
    tbl = _table(name, column)
    last = ""
    while True:
        with engine.begin() as conn:
            rows = conn.execute(
                sa.select(tbl.c.id, tbl.c[column])
                .where(tbl.c.id > last)
                .order_by(tbl.c.id)
                .limit(size)
            ).all()
            if not rows:
                return
            yield conn, tbl, rows
        last = rows[-1][0]


def _existing(engine: sa.Engine, items):
    inspector = sa.inspect(engine)
    return [(t, c) for t, c in items if inspector.has_table(t)]


def classify_field(value: str | None) -> str:
    if value is None or value == "":
        return "empty"
    if not crypto.is_encrypted(value):
        return "plaintext"
    if not crypto.needs_rotation(value):
        return "encrypted (current key)"
    try:
        crypto.decrypt_text_strict(value)
        return "encrypted (old key)"
    except InvalidToken, ValueError:
        return "unreadable"


def classify_vault(token: str | None) -> str:
    if not token:
        return "empty"
    if not crypto.vault_needs_rotation(token):
        return "encrypted (current key)"
    try:
        crypto.vault_fernet().decrypt(token.encode("utf-8"))
        return "encrypted (old key)"
    except InvalidToken:
        pass
    try:
        crypto.legacy_vault_fernet().decrypt(token.encode("utf-8"))
        return "encrypted (old format)"
    except InvalidToken:
        return "unreadable"


def status(engine: sa.Engine) -> dict[str, Counter]:
    """Per column: how many rows are plaintext / encrypted / unreadable. Never returns content."""
    report: dict[str, Counter] = {}
    for name, column in _existing(engine, FIELDS):
        counts = Counter()
        for _conn, _tbl, rows in _batches(engine, name, column):
            counts.update(classify_field(value) for _id, value in rows)
        report[f"{name}.{column}"] = counts
    for name, column in _existing(engine, [VAULT]):
        counts = Counter()
        for _conn, _tbl, rows in _batches(engine, name, column):
            counts.update(classify_vault(value) for _id, value in rows)
        report[f"{name}.{column}"] = counts
    return report


def encrypt(engine: sa.Engine, dry_run: bool = False, size: int = BATCH) -> dict[str, int]:
    """Encrypt every plaintext value in FIELDS. Idempotent. Returns rows changed per column."""
    changed: dict[str, int] = {}
    for name, column in _existing(engine, FIELDS):
        count = 0
        for conn, tbl, rows in _batches(engine, name, column, size):
            for row_id, value in rows:
                if classify_field(value) != "plaintext":
                    continue
                count += 1
                if not dry_run:
                    conn.execute(
                        sa.update(tbl)
                        .where(tbl.c.id == row_id)
                        .values({column: crypto.encrypt_text(value)})
                    )
        changed[f"{name}.{column}"] = count
    return changed


def rotate(engine: sa.Engine, dry_run: bool = False, size: int = BATCH) -> dict[str, Counter]:
    """Re-encrypt everything the newest key cannot read yet (old key or old vault format)."""
    report: dict[str, Counter] = {}
    for name, column in _existing(engine, FIELDS):
        counts = Counter()
        for conn, tbl, rows in _batches(engine, name, column, size):
            for row_id, value in rows:
                if not crypto.needs_rotation(value or ""):
                    continue
                try:
                    new = crypto.rotate_text(value)
                except InvalidToken, ValueError:
                    counts["unreadable"] += 1
                    continue
                counts["rotated"] += 1
                if not dry_run:
                    conn.execute(sa.update(tbl).where(tbl.c.id == row_id).values({column: new}))
        report[f"{name}.{column}"] = counts
    for name, column in _existing(engine, [VAULT]):
        counts = Counter()
        for conn, tbl, rows in _batches(engine, name, column, size):
            for row_id, token in rows:
                if not token or not crypto.vault_needs_rotation(token):
                    continue
                try:
                    new = crypto.encrypt_vault(crypto.decrypt_vault(token))
                except InvalidToken:
                    counts["unreadable"] += 1
                    continue
                counts["rotated"] += 1
                if not dry_run:
                    conn.execute(sa.update(tbl).where(tbl.c.id == row_id).values({column: new}))
        report[f"{name}.{column}"] = counts
    return report


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def _confirm(args: argparse.Namespace, action: str) -> bool:
    print(f"About to {action}.")
    print("BACK UP THE DATABASE FIRST (copy the .db file, or pg_dump). The backup is the undo.")
    if args.yes:
        return True
    try:
        if sys.stdin.isatty():
            return input("Type 'yes' to continue: ").strip().lower() == "yes"
    except EOFError:
        print()
    print("Not confirmed: re-run with --yes, or type 'yes' at the prompt.")
    return False


def _key_ok(args: argparse.Namespace) -> bool:
    if crypto.weak_secret_key() and not args.allow_insecure_key:
        print(
            "Refusing: SECRET_KEY is the public default or too short, so encryption would protect\n"
            "nothing. Set a long random SECRET_KEY (scripts/setup.py does it) and restart, or pass\n"
            "--allow-insecure-key for a throwaway dev database."
        )
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="count encrypted vs plaintext rows (changes nothing)")
    for name, text in (
        ("encrypt", "encrypt rows saved before encryption existed"),
        ("rotate", "re-encrypt everything with the newest SECRET_KEY"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--dry-run", action="store_true", help="only count, change nothing")
        p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
        p.add_argument("--allow-insecure-key", action="store_true", help="dev databases only")
        p.add_argument("--batch-size", type=int, default=BATCH)
    args = parser.parse_args(argv)

    from app.db import describe_database, engine

    print(f"Database: {describe_database()}")
    if args.command == "status":
        report = status(engine)
        if not report:
            print("  No WalletLedger tables in this database (has the app run here yet?).")
        for column, counts in report.items():
            parts = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no rows"
            print(f"  {column:<28} {parts}")
        if crypto.weak_secret_key():
            print("Warning: SECRET_KEY is weak or the public default; set a long random one.")
        return 0

    if not _key_ok(args):
        return 1
    if args.command == "encrypt":
        if not args.dry_run and not _confirm(args, "encrypt plaintext chat history and notes"):
            print("Cancelled.")
            return 1
        result = encrypt(engine, dry_run=args.dry_run, size=args.batch_size)
        verb = "would encrypt" if args.dry_run else "encrypted"
        for column, count in result.items():
            print(f"  {column:<28} {verb} {count} row(s)")
        return 0

    if not args.dry_run and not _confirm(args, "re-encrypt data with the newest SECRET_KEY"):
        print("Cancelled.")
        return 1
    result = rotate(engine, dry_run=args.dry_run, size=args.batch_size)
    verb = "would re-encrypt" if args.dry_run else "re-encrypted"
    unreadable = 0
    for column, counts in result.items():
        unreadable += counts["unreadable"]
        extra = f", {counts['unreadable']} unreadable" if counts["unreadable"] else ""
        print(f"  {column:<28} {verb} {counts['rotated']} row(s){extra}")
    if unreadable:
        print(
            "Some rows match no key in SECRET_KEY / SECRET_KEY_OLD and were left as they are.\n"
            "Add the secret they were written with to SECRET_KEY_OLD and run rotate again."
        )
        return 1  # not finished: scripts and CI must not treat this as success
    elif not args.dry_run:
        print("Done. You can now remove SECRET_KEY_OLD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
