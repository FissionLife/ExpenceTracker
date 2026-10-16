"""Bot profile (command menu, descriptions, menu button) and the setup script. Offline."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import subprocess
import sys
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-profile-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ["TELEGRAM_BOT_TOKEN"] = ""

import asyncio  # noqa: E402
import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

from app.bot import profile, telegram  # noqa: E402
from app.bot.main import COMMAND_MENU  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def fake_bot(**overrides):
    names = [
        "set_my_commands",
        "delete_my_commands",
        "set_my_description",
        "set_my_short_description",
        "set_chat_menu_button",
        "set_my_name",
    ]
    bot = SimpleNamespace(**{n: AsyncMock() for n in names})
    for k, v in overrides.items():
        setattr(bot, k, v)
    return bot


class ValidationTests(unittest.TestCase):
    def test_shipped_profile_is_valid(self):
        self.assertEqual(profile.validate(), [])

    def test_limits_are_enforced(self):
        with patch.object(profile, "DESCRIPTION", "x" * 513):
            self.assertTrue(any("description" in p for p in profile.validate()))
        with patch.object(profile, "SHORT_DESCRIPTION", "x" * 121):
            self.assertTrue(any("short" in p for p in profile.validate()))
        with patch.object(profile, "COMMAND_MENU", [("Bad-Name", "ok description")]):
            self.assertTrue(any("bad command name" in p for p in profile.validate()))
        with patch.object(profile, "COMMAND_MENU", [("fine", "x")]):
            self.assertTrue(any("3-256" in p for p in profile.validate()))

    def test_invalid_profile_never_reaches_telegram(self):
        bot = fake_bot()
        with patch.object(profile, "SHORT_DESCRIPTION", "x" * 200), self.assertRaises(ValueError):
            run(profile.apply_profile(bot))
        bot.set_my_commands.assert_not_awaited()

    def test_menu_has_the_key_and_help_commands(self):
        names = [c.command for c in profile.commands()]
        self.assertEqual(names, [n for n, _ in COMMAND_MENU])
        for must in ("help", "balance", "stats", "setkey", "keys"):
            self.assertIn(must, names)


class ApplyTests(unittest.TestCase):
    def test_applies_every_step(self):
        bot = fake_bot()
        results = run(profile.apply_profile(bot))
        self.assertTrue(all(v == "ok" for v in results.values()), results)
        self.assertEqual(bot.set_my_commands.await_count, 2)  # default + private chats
        bot.set_my_description.assert_awaited_once()
        bot.set_my_short_description.assert_awaited_once()
        bot.set_chat_menu_button.assert_awaited_once()
        bot.set_my_name.assert_not_awaited()  # only on request (rate limited by Telegram)
        bot.delete_my_commands.assert_not_awaited()

    def test_one_failing_step_does_not_block_the_rest(self):
        bot = fake_bot(set_my_description=AsyncMock(side_effect=RuntimeError("flood")))
        results = run(profile.apply_profile(bot))
        self.assertTrue(results["description"].startswith("failed"))
        self.assertEqual(results["command menu"], "ok")
        self.assertEqual(results["menu button"], "ok")
        bot.set_chat_menu_button.assert_awaited_once()

    def test_reset_clears_every_scope_first(self):
        bot = fake_bot()
        run(profile.apply_profile(bot, reset=True))
        self.assertEqual(bot.delete_my_commands.await_count, 3)

    def test_name_is_set_only_when_given(self):
        bot = fake_bot()
        run(profile.apply_profile(bot, name="WalletLedger"))
        bot.set_my_name.assert_awaited_once_with(name="WalletLedger")


class StartupHookTests(unittest.TestCase):
    def test_start_hook_applies_profile(self):
        bot = fake_bot()
        run(telegram._publish_profile(bot))
        self.assertEqual(bot.set_my_commands.await_count, 2)

    def test_start_hook_is_never_fatal(self):
        run(telegram._publish_profile(fake_bot(set_my_commands=AsyncMock(side_effect=OSError))))
        with patch.object(profile, "SHORT_DESCRIPTION", "x" * 200):
            run(telegram._publish_profile(fake_bot()))  # invalid profile: logged, not raised


class ReadProfileTests(unittest.TestCase):
    def reader(self, commands, description, short):
        return SimpleNamespace(
            get_me=AsyncMock(return_value=SimpleNamespace(username="b", id=1)),
            get_my_commands=AsyncMock(
                return_value=[SimpleNamespace(command=c, description=d) for c, d in commands]
            ),
            get_my_description=AsyncMock(return_value=SimpleNamespace(description=description)),
            get_my_short_description=AsyncMock(
                return_value=SimpleNamespace(short_description=short)
            ),
            get_my_name=AsyncMock(return_value=SimpleNamespace(name="Bot")),
            get_chat_menu_button=AsyncMock(return_value=SimpleNamespace(type="commands")),
        )

    def test_in_sync_detection(self):
        synced = self.reader(COMMAND_MENU, profile.DESCRIPTION, profile.SHORT_DESCRIPTION)
        self.assertTrue(run(profile.read_profile(synced))["in_sync"])
        stale = self.reader(COMMAND_MENU[:-1], profile.DESCRIPTION, profile.SHORT_DESCRIPTION)
        self.assertFalse(run(profile.read_profile(stale))["in_sync"])


class ScriptTests(unittest.TestCase):
    def test_script_exits_cleanly_without_a_token(self):
        env = {**os.environ, "TELEGRAM_BOT_TOKEN": ""}
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(
            [sys.executable, os.path.join(root, "scripts", "setup_bot.py"), "--show"],
            capture_output=True,
            text=True,
            env=env,
            cwd=_TMP_DIR,  # no .env here, so the token really is empty
            timeout=60,
        )
        self.assertEqual(out.returncode, 2)
        self.assertIn("TELEGRAM_BOT_TOKEN is not set", out.stdout)


if __name__ == "__main__":
    unittest.main()
