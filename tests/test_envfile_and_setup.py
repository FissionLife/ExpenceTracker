"""The .env editor and the setup helpers. Offline; never touches a real .env."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app import envfile

ROOT = Path(__file__).resolve().parent.parent


class EnvFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="walletledger-env-test-"))
        self.path = self.dir / ".env"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")

    def test_replaces_in_place_and_keeps_comments_and_order(self):
        self.write("# header\nA=1\n# note\nB=2\nC=3\n")
        changed = envfile.set_values(self.path, {"B": "two"})
        self.assertEqual(changed, ["B"])
        self.assertEqual(self.path.read_text(), "# header\nA=1\n# note\nB=two\nC=3\n")

    def test_revives_commented_template_line(self):
        self.write("# DB_MODE=sqlite\nX=1\n")
        envfile.set_values(self.path, {"DB_MODE": "postgres"})
        self.assertEqual(envfile.read_env(self.path)["DB_MODE"], "postgres")
        self.assertEqual(self.path.read_text().count("DB_MODE"), 1)

    def test_appends_new_keys(self):
        self.write("A=1\n")
        envfile.set_values(self.path, {"NEW": "v"})
        self.assertEqual(envfile.read_env(self.path), {"A": "1", "NEW": "v"})

    def test_backup_created_and_no_change_means_no_write(self):
        self.write("A=1\n")
        self.assertEqual(envfile.set_values(self.path, {"A": "1"}), [])
        self.assertFalse(self.path.with_name(".env.bak").exists())
        envfile.set_values(self.path, {"A": "2"})
        self.assertEqual(self.path.with_name(".env.bak").read_text(), "A=1\n")

    def test_duplicate_active_lines_do_not_shadow_the_new_value(self):
        self.write("A=1\nB=x\nA=3\n")
        envfile.set_values(self.path, {"A": "9"})
        self.assertEqual(self.path.read_text().count("A="), 1)
        self.assertEqual(envfile.read_env(self.path)["A"], "9")

    def test_values_with_spaces_are_quoted_and_round_trip(self):
        envfile.set_values(self.path, {"NAME": "Wallet Ledger"}, backup=False)
        self.assertEqual(envfile.read_env(self.path)["NAME"], "Wallet Ledger")

    def test_missing_file_is_created(self):
        envfile.set_values(self.path, {"A": "1"})
        self.assertEqual(envfile.read_env(self.path), {"A": "1"})


class SetupScriptTests(unittest.TestCase):
    def test_check_mode_runs_and_reports_without_changing_anything(self):
        work = tempfile.mkdtemp(prefix="walletledger-setup-test-")  # no .env here
        env = {
            **os.environ,
            "TELEGRAM_BOT_TOKEN": "",
            "DATABASE_URL": "sqlite:///" + os.path.join(work, "t.db").replace("\\", "/"),
            "DYNAMIC_MODELS": "false",
            "ALLOWED_CHAT_IDS": "",
            "ADMIN_CHAT_IDS": "",
        }
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "setup.py"), "--check"],
            capture_output=True,
            text=True,
            env=env,
            cwd=work,
            timeout=120,
        )
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn("TELEGRAM_BOT_TOKEN", out.stdout)
        self.assertIn("Fix the items above", out.stdout)
        self.assertFalse(os.path.exists(os.path.join(work, ".env")))


if __name__ == "__main__":
    unittest.main()
