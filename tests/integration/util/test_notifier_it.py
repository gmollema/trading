from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from dotenv import load_dotenv

import pytest

# Force load the .env file explicitly from the absolute project root
# before importing notifier, so the environment variables are guaranteed to be there.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

from trading_bot.util import notifier
from trading_bot.util import logger


@pytest.mark.integration
class TestNotifyLiveIntegration(unittest.TestCase):
    """Integration test verifying real notification delivery to your actual channels."""

    def setUp(self):
        """Redirect the logger's paths into a temporary directory.

        logger.LOGS_DIR is the repo-relative Path("logs"), so deleting it
        here -- as this test used to -- destroyed the live bots' logs on
        every full test run, and left scheduled tasks appending into a
        directory that no longer existed. Same disk I/O, different place.
        """
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        logs_dir = Path(tmp.name) / "logs"
        for attr, value in (("LOGS_DIR", logs_dir),
                            ("NOTIFY_ERRORS_LOG", logs_dir / "notify_errors.log")):
            p = patch.object(logger, attr, value)
            p.start()
            self.addCleanup(p.stop)

    def test_live_notification_delivery(self):
        """Fires live alerts using the real credentials loaded from your local .env file."""
        # Print confirmation details to your terminal so you can see exactly what was found
        print(f"\n[IT Status] Searching for .env at: {DOTENV_PATH.resolve()}")
        print(f"[IT Status] Loaded Telegram Token: {'✓ Found' if notifier.TELEGRAM_BOT_TOKEN else '✗ MISSING'}")
        print(f"[IT Status] Loaded Telegram Chat ID: {'✓ Found' if notifier.TELEGRAM_CHAT_ID else '✗ MISSING'}")
        print(f"[IT Status] Loaded ntfy URL: {'✓ Found' if notifier.NOTIFY_URL else '✗ MISSING'}")

        # Ensure you actually have credentials before running the test
        if not (notifier.TELEGRAM_BOT_TOKEN and notifier.TELEGRAM_CHAT_ID):
            self.skipTest("Missing real Telegram credentials in your local environment setup.")

        # Execute network calls over active web connections to send a real message
        try:
            notifier.notify(
                title="🚀 Live Integration Test Successful",
                body="If you see this, your actual Telegram pipeline is working perfectly with the trading app!",
                priority="high"
            )
        except Exception as ex:
            self.fail(f"notify() leaked an exception out into the open loop: {ex}")

        # Since notify() swallows errors, let's verify if anything broke behind the scenes
        if logger.NOTIFY_ERRORS_LOG.exists():
            log_data = logger.NOTIFY_ERRORS_LOG.read_text(encoding="utf-8")
            self.fail(
                f"A notification request failed and logged an error behind the scenes! "
                f"Check your credentials. Log data:\n{log_data}"
            )


if __name__ == "__main__":
    unittest.main()