"""SQLite / PostgreSQL mode selection and SQLite hardening (no live Postgres needed)."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-dbmode-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"

import unittest  # noqa: E402

from sqlalchemy import select, text  # noqa: E402

from app import db  # noqa: E402
from app.config import ROOT, Settings  # noqa: E402
from app.models import Account, User  # noqa: E402
from app.services import ledger  # noqa: E402


def cfg(**kw):
    return Settings(_env_file=None, database_url="", **kw)


class ModeSelectionTests(unittest.TestCase):
    def test_default_is_sqlite_file(self):
        s = cfg()
        self.assertEqual(s.db_mode, "sqlite")
        self.assertEqual(
            s.database_url, f"sqlite:///{(ROOT / 'walletledger.db').as_posix()}"
        )  # not cwd-relative

    def test_sqlite_path_setting(self):
        s = cfg(db_mode="sqlite", sqlite_path=r"data\app.db")
        self.assertEqual(s.database_url, f"sqlite:///{(ROOT / 'data/app.db').as_posix()}")

    def test_postgres_mode_builds_url_and_escapes_password(self):
        s = cfg(
            db_mode="postgres",
            postgres_password="p@ss/w:rd",
            postgres_host="db",
            postgres_port=5432,
        )
        self.assertEqual(s.db_mode, "postgres")
        self.assertEqual(
            s.database_url, "postgresql+psycopg2://postgres:p%40ss%2Fw%3Ard@db:5432/appdb"
        )

    def test_explicit_url_wins_and_sets_mode(self):
        s = Settings(_env_file=None, db_mode="sqlite", database_url="postgres://u:p@h:5432/d")
        self.assertEqual(s.database_url, "postgresql+psycopg2://u:p@h:5432/d")
        self.assertEqual(s.db_mode, "postgres")
        s = Settings(_env_file=None, db_mode="postgres", database_url="sqlite:///x.db")
        self.assertEqual(s.db_mode, "sqlite")

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            Settings(_env_file=None, database_url="", db_mode="mysql")


class SqliteHardeningTests(unittest.TestCase):
    def test_active_backend_is_sqlite_and_described_without_secrets(self):
        self.assertTrue(db.IS_SQLITE)
        self.assertTrue(db.describe_database().startswith("SQLite"))

    def test_pragmas_applied(self):
        db.init_db()
        with db.engine.connect() as conn:
            self.assertEqual(conn.execute(text("PRAGMA foreign_keys")).scalar(), 1)
            self.assertEqual(conn.execute(text("PRAGMA journal_mode")).scalar().lower(), "wal")

    def test_cascade_delete_works_on_sqlite(self):
        db.init_db()
        with db.SessionLocal() as s:
            user = ledger.get_or_create_user(s, "cascade-1")
            ledger.record_transaction(s, user.id, 1000, "Cash", "Salary", tx_type="income")
            ledger.record_transaction(s, user.id, 10, "Cash", "Food")
            uid = user.id
            s.delete(s.get(User, uid))
            s.commit()
            self.assertEqual(s.scalars(select(Account).where(Account.user_id == uid)).all(), [])

    def test_creates_missing_parent_directory(self):
        target = os.path.join(_TMP_DIR, "nested", "dir", "x.db").replace("\\", "/")
        eng = None
        try:
            from app.config import settings

            old = settings.database_url
            settings.database_url = f"sqlite:///{target}"
            eng = db._build_engine()
            self.assertTrue(os.path.isdir(os.path.dirname(target)))
        finally:
            settings.database_url = old
            if eng is not None:
                eng.dispose()


if __name__ == "__main__":
    unittest.main()
