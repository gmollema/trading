import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from trading_bot.util import logger


@pytest.mark.integration
class TestLoggerIntegration(unittest.TestCase):
    """Integration test verifying true file-system manipulation on the storage drive.

    The paths are redirected into a temporary directory rather than used
    as they ship. logger.LOGS_DIR is the repo-relative Path("logs"), so
    this test used to rmtree the REAL log directory in setUp and again in
    tearDown -- meaning any full test run silently destroyed the live
    bots' logs, and any scheduled task appending into that directory
    started writing into nothing. The disk I/O being exercised is
    identical either way; only the location changes.
    """

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # A directory that does NOT exist yet, so the test still proves
        # log_error creates its own parent rather than assuming one.
        logs_dir = Path(tmp.name) / "logs"
        for attr, value in (("LOGS_DIR", logs_dir),
                            ("NOTIFY_ERRORS_LOG", logs_dir / "notify_errors.log")):
            p = patch.object(logger, attr, value)
            p.start()
            self.addCleanup(p.stop)

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

    def test_the_real_log_directory_is_never_touched(self):
        """Regression guard for the incident described in the class docstring."""
        self.assertNotEqual(logger.LOGS_DIR.resolve(), Path("logs").resolve())
        try:
            raise RuntimeError("stays in the tmpdir")
        except RuntimeError as e:
            logger.log_error("it_isolation", e)
        self.assertFalse((Path("logs") / "notify_errors.log").exists()
                         and "stays in the tmpdir"
                         in (Path("logs") / "notify_errors.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
