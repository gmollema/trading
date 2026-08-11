import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_bot.cli.heartbeat_monitor import _in_check_window, check_one
from trading_bot.util.heartbeat import write_heartbeat

ET = ZoneInfo("America/New_York")


class TestInCheckWindow(unittest.TestCase):
    def test_weekday_within_window(self):
        # 2026-08-11 is a Tuesday.
        self.assertTrue(_in_check_window(datetime(2026, 8, 11, 12, 0, tzinfo=ET)))

    def test_weekday_before_window(self):
        self.assertFalse(_in_check_window(datetime(2026, 8, 11, 10, 0, tzinfo=ET)))

    def test_weekday_after_window(self):
        self.assertFalse(_in_check_window(datetime(2026, 8, 11, 16, 30, tzinfo=ET)))

    def test_saturday_excluded(self):
        # 2026-08-15 is a Saturday.
        self.assertFalse(_in_check_window(datetime(2026, 8, 15, 12, 0, tzinfo=ET)))

    def test_sunday_excluded(self):
        # 2026-08-16 is a Sunday.
        self.assertFalse(_in_check_window(datetime(2026, 8, 16, 12, 0, tzinfo=ET)))


class TestCheckOne(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "heartbeat.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_fresh_heartbeat_returns_none(self):
        write_heartbeat(self.path, "ok")
        self.assertIsNone(check_one("TestBot", self.path))

    def test_missing_heartbeat_returns_alert(self):
        result = check_one("TestBot", self.path)
        self.assertIsNotNone(result)
        self.assertIn("TestBot", result)
        self.assertIn("no heartbeat", result)

    def test_stale_heartbeat_returns_alert(self):
        import json
        from datetime import timedelta, timezone

        old_ts = datetime.now(timezone.utc) - timedelta(minutes=30)
        self.path.write_text(json.dumps({"timestamp_iso": old_ts.isoformat(), "status": "ok"}))
        result = check_one("TestBot", self.path)
        self.assertIsNotNone(result)
        self.assertIn("TestBot", result)
        self.assertIn("30 min ago", result)


if __name__ == "__main__":
    unittest.main()
