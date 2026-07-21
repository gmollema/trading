import unittest
import shutil

import pytest

from trading_bot.util import logger


@pytest.mark.integration
class TestLoggerIntegration(unittest.TestCase):
    """Integration test verifying true file-system manipulation on the storage drive."""

    def setUp(self):
        """Ensure a clean, predictable state before running local disk actions."""
        if logger.NOTIFY_ERRORS_LOG.exists():
            logger.NOTIFY_ERRORS_LOG.unlink()
        if logger.LOGS_DIR.exists():
            shutil.rmtree(logger.LOGS_DIR)

    def tearDown(self):
        """Recursively clears out both files and directories created during runtime."""
        if logger.LOGS_DIR.exists():
            shutil.rmtree(logger.LOGS_DIR)

    def test_actual_file_creation_and_append_behavior(self):
        """Verifies that files are physically generated on disk and handle successive drops."""
        # Verify clean slate
        self.assertFalse(logger.NOTIFY_ERRORS_LOG.exists())
        self.assertFalse(logger.LOGS_DIR.exists())

        # Trigger first physical write
        try:
            raise RuntimeError("Primary live error trace")
        except RuntimeError as e:
            logger.log_error("it_context_primary", e)

        # 1. Assert file physically exists on the machine
        self.assertTrue(logger.NOTIFY_ERRORS_LOG.exists())

        # Trigger secondary append write
        try:
            raise KeyError("Secondary live key collision")
        except KeyError as e:
            logger.log_error("it_context_secondary", e)

        # 2. Read back real data to confirm clean multi-write execution
        log_content = logger.NOTIFY_ERRORS_LOG.read_text(encoding="utf-8")

        self.assertIn("[it_context_primary]", log_content)
        self.assertIn("RuntimeError: Primary live error trace", log_content)
        self.assertIn("[it_context_secondary]", log_content)
        self.assertIn("Secondary live key collision", log_content)


if __name__ == "__main__":
    unittest.main()