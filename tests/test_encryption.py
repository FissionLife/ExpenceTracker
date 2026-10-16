"""Encryption at rest (app/crypto.py), the API-key vault upgrade and scripts/protect_data.py.

Offline; every test uses its own temporary SQLite file.
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)

import base64
import hashlib
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app import crypto
from app.bot import agent
from app.config import settings
from app.models import Base, ChatMessage, Transaction, User
from app.services import ledger
from app.services_ai import ai_gateway as gw
from app.services_ai import mcp_server

ROOT = Path(__file__).resolve().parent.parent
KEY_A = "a" * 20 + "-first-strong-secret-key-for-tests"
KEY_B = "b" * 20 + "-second-strong-secret-key-for-tests"

_spec = importlib.util.spec_from_file_location("protect_data", ROOT / "scripts" / "protect_data.py")
protect_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(protect_data)


def legacy_vault_token(secret: str, plaintext: str) -> str:
    """How API keys were encrypted before HKDF: Fernet(sha256(SECRET_KEY))."""
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key).encrypt(plaintext.encode()).decode()


class EncryptedDbCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="walletledger-enc-test-")
        url = "sqlite:///" + os.path.join(self.dir, "enc.db").replace("\\", "/")
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False)
        self.keys = patch.multiple(settings, secret_key=KEY_A, secret_key_old="")
        self.keys.start()
        crypto._warned_unreadable = False

    def tearDown(self):
        self.keys.stop()
        self.engine.dispose()
        shutil.rmtree(self.dir, ignore_errors=True)

    def use_keys(self, new: str, old: str = ""):
        settings.secret_key = new
        settings.secret_key_old = old

    def raw(self, sql: str, **params):
        with self.engine.connect() as conn:
            return conn.execute(text(sql), params).scalar()

    def add_user(self) -> str:
        with self.Session() as db:
            user = User(telegram_chat_id="42")
            db.add(user)
            db.commit()
            return user.id

    def add_message(self, user_id: str, content: str, role: str = "user") -> str:
        with self.Session() as db:
            msg = ChatMessage(user_id=user_id, role=role, content=content)
            db.add(msg)
            db.commit()
            return msg.id

    def read_message(self, msg_id: str) -> str:
        with self.Session() as db:
            return db.get(ChatMessage, msg_id).content


class CryptoModuleTests(EncryptedDbCase):
    def test_round_trip_and_self_describing_prefix(self):
        token = crypto.encrypt_text("Spent 450 on dinner")
        self.assertTrue(token.startswith("enc1:"))
        self.assertTrue(crypto.is_encrypted(token))
        self.assertNotIn("dinner", token)
        self.assertEqual(crypto.decrypt_text(token), "Spent 450 on dinner")
        self.assertNotEqual(crypto.encrypt_text("x"), crypto.encrypt_text("x"))  # random IV

    def test_legacy_plaintext_passes_through(self):
        self.assertFalse(crypto.is_encrypted("hello"))
        self.assertEqual(crypto.decrypt_text("hello"), "hello")
        self.assertEqual(crypto.decrypt_text("enc1:not-a-token"), "enc1:not-a-token")

    def test_purpose_keys_are_distinct(self):
        self.assertNotEqual(
            crypto.derive_key(KEY_A, crypto.FIELD_CONTEXT),
            crypto.derive_key(KEY_A, crypto.VAULT_CONTEXT),
        )
        field_token = crypto.encrypt_text("secret")
        with self.assertRaises(InvalidToken):
            crypto.vault_fernet().decrypt(field_token[len("enc1:") :].encode())

    def test_weak_secret_detection(self):
        self.assertTrue(crypto.weak_secret_key("short"))
        self.assertTrue(crypto.weak_secret_key("change-me-to-a-long-random-string"))
        self.assertFalse(crypto.weak_secret_key(KEY_A))


class ColumnEncryptionTests(EncryptedDbCase):
    def test_ciphertext_in_database_plaintext_in_python(self):
        uid = self.add_user()
        msg_id = self.add_message(uid, "my salary is 90000")
        stored = self.raw("SELECT content FROM chat_messages WHERE id = :i", i=msg_id)
        self.assertTrue(stored.startswith("enc1:"))
        self.assertNotIn("salary", stored)
        self.assertEqual(self.read_message(msg_id), "my salary is 90000")

        with self.Session() as db:
            ledger.get_or_create_account(db, uid, "Cash")
            db.execute(text("UPDATE accounts SET balance = 1000 WHERE user_id = :u"), {"u": uid})
            db.commit()
        with self.Session() as db:
            ledger.record_transaction(db, uid, 120, "Cash", "Food", "Cafe", description="birthday")
        stored = self.raw("SELECT description FROM transactions")
        self.assertTrue(stored.startswith("enc1:"))
        self.assertNotIn("birthday", stored)
        with self.Session() as db:
            self.assertEqual(db.query(Transaction).one().description, "birthday")

        # Columns used for grouping and filtering stay plaintext on purpose.
        self.assertEqual(self.raw("SELECT name FROM merchants"), "Cafe")
        self.assertEqual(self.raw("SELECT amount FROM transactions"), 120)

    def test_null_description_stays_null(self):
        uid = self.add_user()
        with self.Session() as db:
            ledger.get_or_create_account(db, uid, "Cash")
            db.execute(text("UPDATE accounts SET balance = 1000 WHERE user_id = :u"), {"u": uid})
            db.commit()
        with self.Session() as db:
            ledger.record_transaction(db, uid, 5, "Cash", "Food")
        self.assertIsNone(self.raw("SELECT description FROM transactions"))

    def test_legacy_plaintext_row_still_readable(self):
        uid = self.add_user()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO chat_messages (id, user_id, role, content, created_at) "
                    "VALUES ('old', :u, 'user', 'typed before encryption', CURRENT_TIMESTAMP)"
                ),
                {"u": uid},
            )
        self.assertEqual(self.read_message("old"), "typed before encryption")

    def test_wrong_key_gives_placeholder_and_one_warning(self):
        uid = self.add_user()
        first = self.add_message(uid, "one")
        second = self.add_message(uid, "two")
        self.use_keys(KEY_B)
        with self.assertLogs("app.crypto", level="WARNING") as logs:
            self.assertEqual(self.read_message(first), crypto.UNREADABLE)
            self.assertEqual(self.read_message(second), crypto.UNREADABLE)
        self.assertEqual(len(logs.records), 1)

    def test_rotation_with_secret_key_old(self):
        uid = self.add_user()
        old_id = self.add_message(uid, "written with A")
        self.use_keys(KEY_B, old=f" , {KEY_A}")
        self.assertEqual(self.read_message(old_id), "written with A")
        new_id = self.add_message(uid, "written with B")

        self.use_keys(KEY_B)  # without the old key only the new row is readable
        self.assertEqual(self.read_message(new_id), "written with B")
        self.assertEqual(self.read_message(old_id), crypto.UNREADABLE)

        self.use_keys(KEY_B, old=KEY_A)
        result = protect_data.rotate(self.engine)
        self.assertEqual(result["chat_messages.content"]["rotated"], 1)
        self.assertEqual(protect_data.rotate(self.engine)["chat_messages.content"]["rotated"], 0)
        self.use_keys(KEY_B)
        self.assertEqual(self.read_message(old_id), "written with A")


class VaultTests(EncryptedDbCase):
    def test_new_keys_round_trip_and_use_hkdf_key(self):
        token = gw.encrypt_api_key("  AIza-test-key-123  ")
        self.assertEqual(gw.decrypt_api_key(token), "AIza-test-key-123")
        self.assertFalse(crypto.vault_needs_rotation(token))
        with self.assertRaises(InvalidToken):
            crypto.legacy_vault_fernet().decrypt(token.encode())

    def test_legacy_sha256_key_still_decrypts(self):
        token = legacy_vault_token(KEY_A, "AIza-legacy")
        self.assertEqual(gw.decrypt_api_key(token), "AIza-legacy")
        self.assertTrue(crypto.vault_needs_rotation(token))

    def test_legacy_key_under_old_secret_and_rotate_upgrades_it(self):
        uid = self.add_user()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO api_keys (id, user_id, provider, encrypted_key, is_active, created_at) "
                    "VALUES ('k1', :u, 'gemini', :t, 1, CURRENT_TIMESTAMP)"
                ),
                {"u": uid, "t": legacy_vault_token(KEY_A, "AIza-old")},
            )
        self.use_keys(KEY_B, old=KEY_A)
        self.assertEqual(
            protect_data.status(self.engine)["api_keys.encrypted_key"],
            {"encrypted (old format)": 1},
        )
        self.assertEqual(protect_data.rotate(self.engine)["api_keys.encrypted_key"]["rotated"], 1)
        self.use_keys(KEY_B)
        stored = self.raw("SELECT encrypted_key FROM api_keys")
        self.assertEqual(gw.decrypt_api_key(stored), "AIza-old")

    def test_wrong_key_still_raises_invalid_token(self):
        token = gw.encrypt_api_key("AIza-x")
        self.use_keys(KEY_B)
        with self.assertRaises(InvalidToken):
            gw.decrypt_api_key(token)


class ScriptTests(EncryptedDbCase):
    def seed_plaintext(self):
        uid = self.add_user()
        with self.engine.begin() as conn:
            for i in range(7):
                conn.execute(
                    text(
                        "INSERT INTO chat_messages (id, user_id, role, content, created_at) "
                        "VALUES (:i, :u, 'user', :c, CURRENT_TIMESTAMP)"
                    ),
                    {"i": f"m{i}", "u": uid, "c": f"private note {i}"},
                )
            conn.execute(
                text(
                    "INSERT INTO transactions (id, user_id, amount, transaction_type, description,"
                    " source, created_at) VALUES"
                    " ('t1', :u, 10, 'expense', 'gift for mum', 'chat', CURRENT_TIMESTAMP),"
                    " ('t2', :u, 20, 'expense', NULL, 'chat', CURRENT_TIMESTAMP)"
                ),
                {"u": uid},
            )
        self.add_message(uid, "already encrypted")
        return uid

    def test_status_counts_without_content(self):
        self.seed_plaintext()
        report = protect_data.status(self.engine)
        self.assertEqual(
            report["chat_messages.content"], {"plaintext": 7, "encrypted (current key)": 1}
        )
        self.assertEqual(report["transactions.description"], {"plaintext": 1, "empty": 1})
        self.assertNotIn("private", repr(report))

    def test_encrypt_dry_run_then_real_then_idempotent(self):
        self.seed_plaintext()
        dry = protect_data.encrypt(self.engine, dry_run=True, size=3)
        self.assertEqual(dry, {"chat_messages.content": 7, "transactions.description": 1})
        self.assertEqual(protect_data.status(self.engine)["chat_messages.content"]["plaintext"], 7)

        self.assertEqual(protect_data.encrypt(self.engine, size=3), dry)
        self.assertEqual(
            protect_data.encrypt(self.engine),
            {"chat_messages.content": 0, "transactions.description": 0},
        )
        report = protect_data.status(self.engine)
        self.assertEqual(report["chat_messages.content"], {"encrypted (current key)": 8})
        leaked = self.raw("SELECT COUNT(*) FROM chat_messages WHERE content LIKE '%private%'")
        self.assertEqual(leaked, 0)
        self.assertEqual(self.read_message("m3"), "private note 3")
        with self.Session() as db:
            self.assertEqual(db.get(Transaction, "t1").description, "gift for mum")

    def test_cli_refuses_insecure_key_and_needs_confirmation(self):
        with (
            patch.object(protect_data, "encrypt") as enc,
            patch("builtins.print"),
            patch.object(protect_data.sys.stdin, "isatty", return_value=False),
        ):
            self.use_keys("walletledger-insecure-dev-key-change-in-prod")
            self.assertEqual(protect_data.main(["encrypt", "--yes"]), 1)
            self.use_keys(KEY_A)
            self.assertEqual(protect_data.main(["encrypt"]), 1)  # no --yes, not interactive
            enc.assert_not_called()
            enc.return_value = {}
            self.assertEqual(protect_data.main(["encrypt", "--yes"]), 0)
            enc.assert_called_once()


class AppIntegrationTests(EncryptedDbCase):
    def test_chat_history_reaches_agent_as_plaintext(self):
        with patch.object(agent, "SessionLocal", self.Session):
            agent.save_db_chat_message("42", "user", "Spent 450 on dinner")
            agent.save_db_chat_message("42", "assistant", "Logged it.")
            history = agent.get_db_conversation_history("42", limit=10)
        self.assertEqual([m.content for m in history], ["Spent 450 on dinner", "Logged it."])
        self.assertEqual(
            self.raw("SELECT COUNT(*) FROM chat_messages WHERE content LIKE 'enc1:%'"), 2
        )

    def test_mcp_search_and_edit_work_on_encrypted_descriptions(self):
        with patch.object(mcp_server, "SessionLocal", self.Session):
            # Pre-fund Cash so the three live expenses don't hit the strict-balance guard.
            with self.Session() as _db:
                from app.models import User as _User

                _u = _db.query(_User).filter_by(telegram_chat_id="42").one_or_none()
                if _u is None:
                    _u = _User(telegram_chat_id="42")
                    _db.add(_u)
                    _db.commit()
                from app.services.ledger import get_or_create_account

                get_or_create_account(_db, _u.id, "Cash")
                _db.execute(
                    text("UPDATE accounts SET balance = 1000 WHERE user_id = :u"), {"u": _u.id}
                )
                _db.commit()
            first = mcp_server.log_expense("42", 100, "Food", merchant="Cafe", description="Lunch")
            mcp_server.log_expense("42", 50, "Travel", merchant="Uber", description="airport")
            mcp_server.log_expense("42", 30, "Food", merchant="Bakery", description="lunch snack")

            found = mcp_server.search_transactions("42", query="LUNCH")["transactions"]
            self.assertEqual(sorted(t["description"] for t in found), ["Lunch", "lunch snack"])
            paged = mcp_server.search_transactions("42", query="lunch", limit=1, offset=1)
            self.assertEqual(len(paged["transactions"]), 1)
            self.assertEqual(
                mcp_server.search_transactions("42", query="uber")["transactions"][0]["merchant"],
                "Uber",
            )
            self.assertEqual(len(mcp_server.search_transactions("42")["transactions"]), 3)

            mcp_server.edit_transaction("42", first["transaction_id"], description="team lunch")
            edited = mcp_server.search_transactions("42", query="team")["transactions"]
            self.assertEqual([t["description"] for t in edited], ["team lunch"])
        stored = self.raw(
            "SELECT description FROM transactions WHERE id = :i", i=first["transaction_id"]
        )
        self.assertTrue(stored.startswith("enc1:"))

    def test_mcp_saved_api_key_uses_new_vault(self):
        with patch.object(mcp_server, "SessionLocal", self.Session):
            mcp_server.save_user_api_key("42", "gemini", "AIza-from-mcp")
        stored = self.raw("SELECT encrypted_key FROM api_keys")
        self.assertFalse(crypto.vault_needs_rotation(stored))
        self.assertEqual(gw.decrypt_api_key(stored), "AIza-from-mcp")


if __name__ == "__main__":
    unittest.main()
