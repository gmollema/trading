"""Unit tests for trading_bot.cli.rotate_logs."""

import csv
import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from trading_bot.cli import rotate_logs

ET = ZoneInfo("America/New_York")
TRADES_FIELDNAMES = ["timestamp_iso", "symbol", "side", "size", "fill_price", "order_id", "status"]


def set_mtime(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


class RotateLogsTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        base = Path(self._tmpdir.name)
        self.logs_dir = base / "logs"
        self.archive_dir = self.logs_dir / "archive"
        self.trades_path = base / "trades.csv"
        self.safety_log_path = base / "safety-check-log.json"

        self.patchers = [
            patch.object(rotate_logs, "LOGS_DIR", self.logs_dir),
            patch.object(rotate_logs, "ARCHIVE_DIR", self.archive_dir),
            patch.object(rotate_logs, "TRADES_CSV_PATH", self.trades_path),
            patch.object(rotate_logs, "SAFETY_LOG_PATH", self.safety_log_path),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        self._tmpdir.cleanup()


class TestRotateGenericLogs(RotateLogsTestBase):
    def test_missing_logs_dir_returns_zero(self):
        self.assertEqual(rotate_logs.rotate_generic_logs(), 0)

    def test_old_log_file_is_archived(self):
        self.logs_dir.mkdir(parents=True)
        old_file = self.logs_dir / "cycle_errors.log"
        old_file.write_text("old stuff")
        yesterday = datetime.now(ET) - timedelta(days=1)
        set_mtime(old_file, yesterday)

        count = rotate_logs.rotate_generic_logs()

        self.assertEqual(count, 1)
        self.assertFalse(old_file.exists())
        date_str = yesterday.strftime("%Y-%m-%d")
        self.assertTrue((self.archive_dir / date_str / "cycle_errors.log").exists())

    def test_todays_log_file_is_not_moved(self):
        self.logs_dir.mkdir(parents=True)
        today_file = self.logs_dir / "cycle_errors.log"
        today_file.write_text("fresh")

        count = rotate_logs.rotate_generic_logs()

        self.assertEqual(count, 0)
        self.assertTrue(today_file.exists())

    def test_non_log_extension_is_ignored(self):
        self.logs_dir.mkdir(parents=True)
        other_file = self.logs_dir / "notes.txt"
        other_file.write_text("keep me")
        set_mtime(other_file, datetime.now(ET) - timedelta(days=5))

        count = rotate_logs.rotate_generic_logs()

        self.assertEqual(count, 0)
        self.assertTrue(other_file.exists())

    def test_directories_are_skipped(self):
        self.logs_dir.mkdir(parents=True)
        (self.logs_dir / "subdir.log").mkdir()  # a directory, oddly named .log

        count = rotate_logs.rotate_generic_logs()

        self.assertEqual(count, 0)

    def test_name_collision_gets_suffixed_not_clobbered(self):
        self.logs_dir.mkdir(parents=True)
        yesterday = datetime.now(ET) - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")
        dest_dir = self.archive_dir / date_str
        dest_dir.mkdir(parents=True)
        (dest_dir / "cycle_errors.log").write_text("already archived")

        new_old_file = self.logs_dir / "cycle_errors.log"
        new_old_file.write_text("today's archival batch")
        set_mtime(new_old_file, yesterday)

        count = rotate_logs.rotate_generic_logs()

        self.assertEqual(count, 1)
        archived_files = list(dest_dir.iterdir())
        self.assertEqual(len(archived_files), 2)  # original + suffixed


class TestRotateTradesCsv(RotateLogsTestBase):
    def _write(self, rows):
        with self.trades_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADES_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_missing_file_returns_zero(self):
        self.assertEqual(rotate_logs.rotate_trades_csv(), 0)

    def test_empty_file_returns_zero(self):
        self.trades_path.write_text("")

        self.assertEqual(rotate_logs.rotate_trades_csv(), 0)

    def test_all_recent_rows_returns_zero_and_keeps_file_untouched(self):
        recent = datetime.now(timezone.utc).isoformat()
        self._write([{"timestamp_iso": recent, "symbol": "AAPL", "side": "BUY",
                      "size": 1, "fill_price": 100, "order_id": 1, "status": "Filled"}])

        self.assertEqual(rotate_logs.rotate_trades_csv(), 0)
        with self.trades_path.open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 1)

    def test_old_rows_are_archived_and_removed(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        recent_ts = datetime.now(timezone.utc).isoformat()
        self._write(
            [
                {"timestamp_iso": old_ts, "symbol": "OLD", "side": "BUY",
                 "size": 1, "fill_price": 50, "order_id": 1, "status": "Filled"},
                {"timestamp_iso": recent_ts, "symbol": "NEW", "side": "BUY",
                 "size": 1, "fill_price": 100, "order_id": 2, "status": "Filled"},
            ]
        )

        result = rotate_logs.rotate_trades_csv()

        self.assertEqual(result, 1)
        with self.trades_path.open() as f:
            remaining = list(csv.DictReader(f))
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["symbol"], "NEW")

        today_str = datetime.now(ET).strftime("%Y%m%d")
        archive_path = self.archive_dir / f"trades_{today_str}.csv"
        with archive_path.open() as f:
            archived = list(csv.DictReader(f))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0]["symbol"], "OLD")

    def test_unparseable_timestamp_row_is_kept_not_lost(self):
        self._write([{"timestamp_iso": "not-a-timestamp", "symbol": "WEIRD", "side": "BUY",
                      "size": 1, "fill_price": 10, "order_id": 1, "status": "Filled"}])

        result = rotate_logs.rotate_trades_csv()

        self.assertEqual(result, 0)
        with self.trades_path.open() as f:
            remaining = list(csv.DictReader(f))
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["symbol"], "WEIRD")


class TestRotateSafetyLog(RotateLogsTestBase):
    def test_missing_file_returns_zero(self):
        self.assertEqual(rotate_logs.rotate_safety_log(), 0)

    def test_small_file_is_not_rotated(self):
        self.safety_log_path.write_text("small content")

        self.assertEqual(rotate_logs.rotate_safety_log(), 0)
        self.assertTrue(self.safety_log_path.exists())

    @patch("trading_bot.cli.rotate_logs.SAFETY_LOG_MAX_BYTES", 10)
    def test_oversized_file_is_archived_and_replaced_with_fresh_empty_file(self):
        self.safety_log_path.write_text("this content exceeds ten bytes easily")

        result = rotate_logs.rotate_safety_log()

        self.assertEqual(result, 1)
        self.assertTrue(self.safety_log_path.exists())
        self.assertEqual(self.safety_log_path.read_text(), "")
        today_str = datetime.now(ET).strftime("%Y%m%d")
        archived = self.archive_dir / f"safety-check-log_{today_str}.json"
        self.assertTrue(archived.exists())
        self.assertIn("exceeds ten bytes", archived.read_text())


class TestMain(RotateLogsTestBase):
    def test_reports_total_rotated_count(self):
        self.logs_dir.mkdir(parents=True)
        old_file = self.logs_dir / "cycle_errors.log"
        old_file.write_text("old")
        set_mtime(old_file, datetime.now(ET) - timedelta(days=1))

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = rotate_logs.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Rotated 1 files", buf.getvalue())

    def test_nothing_to_rotate_reports_zero(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = rotate_logs.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Rotated 0 files", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
