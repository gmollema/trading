import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.util.heartbeat import read_heartbeat_age_minutes, write_heartbeat


class TestHeartbeat(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "heartbeat.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_write_then_read_is_fresh(self):
        write_heartbeat(self.path, "ok")
        age = read_heartbeat_age_minutes(self.path)
        self.assertIsNotNone(age)
        self.assertLess(age, 1.0)

    def test_write_records_status(self):
        write_heartbeat(self.path, "force_close")
        payload = json.loads(self.path.read_text())
        self.assertEqual(payload["status"], "force_close")

    def test_missing_file_returns_none(self):
        self.assertIsNone(read_heartbeat_age_minutes(self.path))

    def test_stale_timestamp_reports_correct_age(self):
        old_ts = datetime.now(timezone.utc) - timedelta(minutes=45)
        self.path.write_text(json.dumps({"timestamp_iso": old_ts.isoformat(), "status": "ok"}))
        age = read_heartbeat_age_minutes(self.path)
        self.assertGreater(age, 44)
        self.assertLess(age, 46)

    def test_corrupt_file_returns_none(self):
        self.path.write_text("not json")
        self.assertIsNone(read_heartbeat_age_minutes(self.path))

    def test_missing_timestamp_key_returns_none(self):
        self.path.write_text(json.dumps({"status": "ok"}))
        self.assertIsNone(read_heartbeat_age_minutes(self.path))

    def test_write_swallows_errors_on_unwritable_path(self):
        # A path under a nonexistent directory can't be written -- must
        # not raise, since a heartbeat write is never allowed to break the
        # actual trading cycle that calls it.
        bad_path = Path(self.tmpdir.name) / "nonexistent_dir" / "heartbeat.json"
        write_heartbeat(bad_path, "ok")  # should not raise
        self.assertFalse(bad_path.exists())


if __name__ == "__main__":
    unittest.main()
