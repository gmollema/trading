import unittest
from unittest.mock import patch, MagicMock
import requests

from trading_bot.util import notifier


class TestNotify(unittest.TestCase):
    """Isolated tests for the notification router orchestration logic."""

    def setUp(self):
        """Force mock environmental configurations inside the notifier module context."""
        self.patch_env = patch.multiple(
            notifier,
            TELEGRAM_BOT_TOKEN="mock_telegram_token",
            TELEGRAM_CHAT_ID="mock_chat_id",
            NOTIFY_URL="https://ntfy.sh/mock_topic",
        )
        self.patch_env.start()

    def tearDown(self):
        """Stop environmental patches."""
        self.patch_env.stop()

    @patch("trading_bot.util.notifier.requests.post")
    def test_notify_success_both_channels(self, mock_post):
        """Verify both channels receive network POST calls under normal conditions."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        notifier.notify("Test Title", "Test Body", priority="high")

        # requests.post should be hit exactly twice (once for Telegram, once for ntfy)
        self.assertEqual(mock_post.call_count, 2)

    @patch("trading_bot.util.notifier.log_error")
    @patch("trading_bot.util.notifier.requests.post")
    def test_notify_routes_failures_to_logger(self, mock_post, mock_log_error):
        """Confirm that HTTP network exceptions are caught and correctly piped to the logger."""
        # Simulate an untrusted network timeout across all network calls
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        try:
            notifier.notify("Network Breakdown", "Should be caught gracefully")
        except Exception as e:
            self.fail(f"notify() leaked a network exception: {e} - violating its core contract.")

        # Ensure our separate logger module was handed the problems cleanly with proper contexts
        self.assertEqual(mock_log_error.call_count, 2)
        mock_log_error.assert_any_call("telegram", mock_post.side_effect)
        mock_log_error.assert_any_call("ntfy", mock_post.side_effect)

    @patch("trading_bot.util.notifier.requests.post")
    def test_skips_when_credentials_are_missing(self, mock_post):
        """Verify notification endpoints are skipped entirely if credentials are missing or blank."""
        with patch.multiple(notifier, TELEGRAM_BOT_TOKEN="", NOTIFY_URL=""):
            notifier.notify("Title", "Body")

            # Since tokens are missing, no HTTP requests should fire at all
            mock_post.assert_not_called()

    @patch("trading_bot.util.notifier.requests.post")
    def test_html_escaping_for_telegram(self, mock_post):
        """Ensure special characters convert cleanly to safe HTML entities for Telegram."""
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Pass symbols that could break Telegram markdown/HTML parsing
        notifier.notify("BTC_USDT Breakout!", "Target > $100K & < $105K")

        # Inspect arguments passed into the Telegram call (the first item in call_args_list)
        tg_call_args = mock_post.call_args_list[0]
        payload = tg_call_args[1].get("json", {})

        # Confirm strings are converted to HTML-safe equivalents
        self.assertIn("BTC_USDT Breakout!", payload["text"])
        self.assertIn("&gt;", payload["text"])
        self.assertIn("&lt;", payload["text"])

    @patch("trading_bot.util.notifier.requests.post")
    def test_default_priority_gets_robot_icon(self, mock_post):
        mock_post.return_value = MagicMock()

        notifier.notify("BUY AAPL", "@ $100.00", priority="default")

        tg_payload = mock_post.call_args_list[0][1]["json"]
        self.assertTrue(tg_payload["text"].startswith(f"<b>{notifier.ICON_DEFAULT} BUY AAPL"))
        ntfy_headers = mock_post.call_args_list[1][1]["headers"]
        self.assertTrue(ntfy_headers["Title"].startswith(notifier.ICON_DEFAULT))

    @patch("trading_bot.util.notifier.requests.post")
    def test_high_priority_gets_warning_icon(self, mock_post):
        mock_post.return_value = MagicMock()

        notifier.notify("Cycle CRASHED", "boom", priority="high")

        tg_payload = mock_post.call_args_list[0][1]["json"]
        self.assertTrue(tg_payload["text"].startswith(f"<b>{notifier.ICON_HIGH_PRIORITY} Cycle CRASHED"))
        ntfy_headers = mock_post.call_args_list[1][1]["headers"]
        self.assertTrue(ntfy_headers["Title"].startswith(notifier.ICON_HIGH_PRIORITY))


if __name__ == "__main__":
    unittest.main()