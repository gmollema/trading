import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trading_bot.util import logger


class TestLogger(unittest.TestCase):
    """Isolated tests for the hardened filesystem logger component."""

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

    def test_log_error_writes_successfully(self):
        """Verify that log_error safely dumps exception stack traces to disk."""
        try:
            raise ValueError("Simulated trading engine glitch")
        except ValueError as e:
            logger.log_error("order_execution", e)

        # Check that the log file was actually created
        self.assertTrue(logger.NOTIFY_ERRORS_LOG.exists())

        # Read the file content and verify the context and traceback are present
        log_text = logger.NOTIFY_ERRORS_LOG.read_text(encoding="utf-8")
        self.assertIn("[order_execution]", log_text)
        self.assertIn("ValueError: Simulated trading engine glitch", log_text)

    @patch("trading_bot.util.logger.Path.open")
    def test_logger_catastrophe_fails_silently(self, mock_open):
        """Ensure that even if the filesystem goes read-only, the logger never leaks exceptions."""
        # Force the filesystem to throw an error when attempting to open a file
        mock_open.side_effect = OSError("Disk space completely full or read-only filesystem")

        try:
            logger.log_error("critical_context", Exception("System Boom"))
        except Exception as e:
            self.fail(f"logger.log_error leaked an exception during total disk failure: {e}")


if __name__ == "__main__":
    unittest.main()