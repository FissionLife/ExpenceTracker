"""First-run setup: the server asks once, then starts as usual. Offline: Telegram is faked."""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

from app import envfile, firstrun, setup_wizard  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOOD_TOKEN = "1234567:AAA-good-token"
STRONG = "x" * 48


class FakeConsole(setup_wizard.Console):
    """Answers questions from a dict of {part of the question: answer}; anything else keeps the default."""

    def __init__(self, answers=None, yes_answers=None):
        self.answers = answers or {}
        self.yes_answers = yes_answers or {}
        self.said = []
        self.asked = []

    def say(self, msg=""):
        self.said.append(msg)

    def ask(self, prompt, default="", secret=False):
        self.asked.append(prompt)
        for key, value in self.answers.items():
            if key in prompt:
                return value or default
        return default

    def yes(self, prompt, default=True):
        self.asked.append(prompt)
        for key, value in self.yes_answers.items():
            if key in prompt:
                return value
        return default

    @property
    def text(self):
        return "\n".join(self.said)


def me(name="mybot"):
    return types.SimpleNamespace(username=name)


def tmp_env(content=None):
    path = Path(tempfile.mkdtemp(prefix="walletledger-firstrun-")) / ".env"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    return path


class SetupStateTests(unittest.TestCase):
    def state(self, content=None, environ=None):
        return firstrun.setup_state(tmp_env(content), environ or {})

    def test_no_env_file_means_not_set_up(self):
        state = self.state()
        self.assertFalse(state.done)
        self.assertEqual(len(state.missing), 2)

    def test_token_and_strong_secret_mean_set_up(self):
        self.assertTrue(self.state(f"TELEGRAM_BOT_TOKEN={GOOD_TOKEN}\nSECRET_KEY={STRONG}\n").done)

    def test_extra_bots_alone_count_as_a_token(self):
        self.assertTrue(self.state(f"TELEGRAM_BOT_TOKENS={GOOD_TOKEN}\nSECRET_KEY={STRONG}\n").done)

    def test_a_placeholder_or_short_secret_is_not_enough(self):
        for secret in ("change-me-to-a-long-random-string", "short", envfile.INSECURE_DEV_KEY):
            state = self.state(f"TELEGRAM_BOT_TOKEN={GOOD_TOKEN}\nSECRET_KEY={secret}\n")
            self.assertFalse(state.done, secret)
            self.assertEqual(state.missing, ["a SECRET_KEY"])

    def test_skipping_the_token_on_purpose_is_remembered(self):
        self.assertTrue(self.state("SETUP_DONE=true\n").done)

    def test_real_environment_variables_count_like_the_app_treats_them(self):
        env = {"TELEGRAM_BOT_TOKEN": GOOD_TOKEN, "SECRET_KEY": STRONG}
        self.assertTrue(self.state(None, env).done)  # Docker, cloud: no .env file at all


class EnsureSetupTests(unittest.TestCase):
    def test_does_nothing_when_already_set_up(self):
        path = tmp_env("SETUP_DONE=true\n")
        with patch.object(setup_wizard, "run_wizard") as wizard:
            self.assertEqual(firstrun.ensure_setup(path, environ={}, interactive=True), "ok")
        wizard.assert_not_called()

    def test_asks_when_not_set_up_and_somebody_can_answer(self):
        path = tmp_env()
        with patch.object(setup_wizard, "run_wizard") as wizard:
            self.assertEqual(firstrun.ensure_setup(path, environ={}, interactive=True), "ran")
        wizard.assert_called_once()
        self.assertTrue(wizard.call_args.kwargs["first_run"])

    def test_a_terminal_that_vanishes_mid_question_does_not_crash_the_server(self):
        with patch.object(setup_wizard, "run_wizard", side_effect=setup_wizard.NoConsole("gone")):
            self.assertEqual(
                firstrun.ensure_setup(tmp_env(), environ={}, interactive=True), "skipped"
            )

    def test_never_blocks_when_nobody_can_answer(self):
        path = tmp_env()
        with patch.object(setup_wizard, "run_wizard") as wizard:
            self.assertEqual(firstrun.ensure_setup(path, environ={}, interactive=False), "skipped")
        wizard.assert_not_called()

    def test_tests_and_setup_skip_never_ask(self):
        path = tmp_env()
        with patch.object(setup_wizard, "run_wizard") as wizard:
            self.assertEqual(
                firstrun.ensure_setup(
                    path, environ={"WALLETLEDGER_TESTING": "1"}, interactive=True
                ),
                "skipped",
            )
            self.assertEqual(
                firstrun.ensure_setup(path, environ={"SETUP_SKIP": "1"}, interactive=True),
                "skipped",
            )
        wizard.assert_not_called()

    def test_importing_main_in_tests_does_not_prompt(self):
        # main.py calls ensure_setup() first; under WALLETLEDGER_TESTING it must return at once
        self.assertEqual(firstrun.ensure_setup(tmp_env()), "skipped")


class WizardTests(unittest.TestCase):
    def run_wizard(self, io, path=None, check=None):
        path = path or tmp_env()
        with (
            patch.object(setup_wizard, "check_token", check or AsyncMock(return_value=me())),
            patch.object(setup_wizard, "detect_chat_id", AsyncMock(return_value="424242")),
            patch.object(setup_wizard, "step_profile"),
            patch.object(setup_wizard, "step_ai_key"),
        ):
            setup_wizard.run_wizard(path, io=io)
        return path, envfile.read_env(path)

    def test_default_answers_give_a_working_public_bot(self):
        io = FakeConsole({"Bot token": GOOD_TOKEN})
        _, env = self.run_wizard(io)
        self.assertEqual(env["TELEGRAM_BOT_TOKEN"], GOOD_TOKEN)
        self.assertFalse(envfile.is_weak_secret(env["SECRET_KEY"]))
        self.assertEqual(env["ADMIN_CHAT_IDS"], "424242")  # detected
        self.assertEqual(
            env["ALLOWED_CHAT_IDS"], "", "the bot must be PUBLIC unless asked otherwise"
        )
        self.assertEqual(env["SETUP_DONE"], "true")
        self.assertIn("Public: anyone can use the bot", io.text)

    def test_private_is_an_explicit_choice(self):
        io = FakeConsole(
            {"Bot token": GOOD_TOKEN, "public or private": "private", "Allowed Telegram ids": "1,2"}
        )
        _, env = self.run_wizard(io)
        self.assertEqual(env["ALLOWED_CHAT_IDS"], "1,2")

    def test_rerunning_keeps_values_and_can_switch_back_to_public(self):
        path = tmp_env(
            f"TELEGRAM_BOT_TOKEN={GOOD_TOKEN}\nSECRET_KEY={STRONG}\nALLOWED_CHAT_IDS=9\n"
        )
        io = FakeConsole({"public or private": "public"})
        _, env = self.run_wizard(io, path)
        self.assertEqual(env["SECRET_KEY"], STRONG)  # never regenerated
        self.assertEqual(env["ALLOWED_CHAT_IDS"], "")

    def test_a_wrong_token_is_not_saved_and_the_reason_is_plain(self):
        class Unauthorized(Exception):
            pass

        Unauthorized.__name__ = "TelegramUnauthorizedError"
        io = FakeConsole({"Bot token": "9:bad"}, {"Try again": False})
        _, env = self.run_wizard(io, check=AsyncMock(side_effect=Unauthorized()))
        self.assertEqual(env.get("TELEGRAM_BOT_TOKEN", ""), "")
        self.assertIn("not valid", io.text)
        self.assertEqual(env["SETUP_DONE"], "true")  # skipping the token is remembered

    def test_offline_is_told_apart_from_a_wrong_token(self):
        class Offline(Exception):
            pass

        Offline.__name__ = "TelegramNetworkError"
        io = FakeConsole({"Bot token": GOOD_TOKEN}, {"Save it anyway": True})
        _, env = self.run_wizard(io, check=AsyncMock(side_effect=Offline()))
        self.assertIn("online", io.text)
        self.assertEqual(env["TELEGRAM_BOT_TOKEN"], GOOD_TOKEN)

    def test_a_token_without_a_colon_is_caught_before_asking_telegram(self):
        check = AsyncMock(return_value=me())
        io = FakeConsole({"Bot token": "notatoken"}, {"Try again": False})
        self.run_wizard(io, check=check)
        check.assert_not_awaited()
        self.assertIn("doesn't look like a bot token", io.text)

    def test_extra_bots_are_saved_next_to_the_main_one(self):
        answers = iter(["2222:BBB-second", ""])  # one more bot, then Enter to stop
        io = FakeConsole({"Bot token": GOOD_TOKEN}, {"more bots": True})
        original = io.ask

        def ask(prompt, default="", secret=False):
            if "Bot token 2" in prompt or "Bot token 3" in prompt:
                return next(answers)
            return original(prompt, default, secret)

        io.ask = ask
        _, env = self.run_wizard(io)
        self.assertEqual(env["TELEGRAM_BOT_TOKENS"], "2222:BBB-second")

    def test_skipping_everything_still_leaves_a_runnable_setup(self):
        io = FakeConsole({"Bot token": ""})
        _, env = self.run_wizard(io)
        self.assertEqual(env.get("TELEGRAM_BOT_TOKEN", ""), "")
        self.assertFalse(envfile.is_weak_secret(env["SECRET_KEY"]))
        self.assertEqual(env["SETUP_DONE"], "true")
        self.assertTrue(firstrun.setup_state(tmp_env("SETUP_DONE=true\n"), {}).done)


class DefaultsAndScriptTests(unittest.TestCase):
    def test_defaults_mode_asks_nothing(self):
        path = tmp_env()
        io = FakeConsole()
        code = setup_wizard.main(
            ["--defaults", "--token", GOOD_TOKEN, "--admin-id", "77"],
            env_path=path,
            io_factory=lambda: io,
        )
        env = envfile.read_env(path)
        self.assertEqual(code, 0)
        self.assertEqual(io.asked, [])
        self.assertEqual(
            (env["TELEGRAM_BOT_TOKEN"], env["ADMIN_CHAT_IDS"], env["ALLOWED_CHAT_IDS"]),
            (GOOD_TOKEN, "77", ""),
        )
        self.assertTrue(firstrun.setup_state(path, {}).done)

    def test_defaults_private_needs_an_admin_id(self):
        path = tmp_env()
        setup_wizard.main(
            ["--defaults", "--private", "--admin-id", "77"], env_path=path, io_factory=FakeConsole
        )
        self.assertEqual(envfile.read_env(path)["ALLOWED_CHAT_IDS"], "77")

    def test_no_terminal_gives_a_clear_message_not_a_traceback(self):
        class Dead(FakeConsole):
            def ask(self, *a, **k):
                raise setup_wizard.NoConsole("no terminal")

            yes = ask

        io = Dead()
        code = setup_wizard.main([], env_path=tmp_env(), io_factory=lambda: io)
        self.assertEqual(code, 2)
        self.assertIn("--defaults", io.text)

    def test_the_script_launches_and_defaults_work_from_the_command_line(self):
        work = Path(tempfile.mkdtemp(prefix="walletledger-setup-cli-"))
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "setup.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=work,
            timeout=60,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--defaults", out.stdout)


class HealthNotesTests(unittest.TestCase):
    def test_a_public_bot_is_a_note_not_a_problem(self):
        from app import health
        from app.config import settings

        with (
            patch.object(settings, "allowed_chat_ids", ""),
            patch.object(settings, "telegram_bot_token", GOOD_TOKEN),
            patch.object(settings, "telegram_bot_tokens", ""),
            patch.object(settings, "secret_key", STRONG),
        ):
            self.assertEqual(health.setup_problems(), [])
            self.assertIn("Public bot", " ".join(health.setup_notes()))
        with patch.object(settings, "allowed_chat_ids", "1,2"):
            self.assertIn("2 Telegram id", " ".join(health.setup_notes()))


if __name__ == "__main__":
    unittest.main()
