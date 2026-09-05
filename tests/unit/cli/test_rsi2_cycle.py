"""Unit tests for trading_bot.cli.rsi2_cycle.

This file exists as much to IMPORT the module as to assert about it. It
had no test of its own, so a syntax error in it survived a fully green
966-test run and only surfaced when the scheduled task failed -- the
module is an entrypoint nothing else imports.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trading_bot.cli import rsi2_cycle


class TestParseArgs(unittest.TestCase):
    def test_defaults_are_safe(self):
        args = rsi2_cycle.parse_args([])
        self.assertFalse(args.arm)
        self.assertFalse(args.ignore_window)

    def test_arm_and_ignore_window(self):
        args = rsi2_cycle.parse_args(["--arm", "--ignore-window"])
        self.assertTrue(args.arm)
        self.assertTrue(args.ignore_window)


class TestLogEvent(unittest.TestCase):
    def test_writes_one_json_line_and_creates_the_directory(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "cycle.log"
            rsi2_cycle.log_event({"event": "decision", "action": "hold"}, path=path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        row = json.loads(lines[0])
        self.assertEqual(row["event"], "decision")
        self.assertIn("ts", row)

    def test_appends_rather_than_truncating(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cycle.log"
            rsi2_cycle.log_event({"event": "one"}, path=path)
            rsi2_cycle.log_event({"event": "two"}, path=path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(x)["event"] for x in lines], ["one", "two"])

    def test_survives_no_console(self):
        """The scheduled task runs pythonw, where sys.stdout is None and
        print() raises. The file write must still happen."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cycle.log"
            with patch.object(sys, "stdout", None):
                rsi2_cycle.log_event({"event": "headless"}, path=path)
            self.assertIn("headless", path.read_text(encoding="utf-8"))

    def test_an_unwritable_log_does_not_take_the_cycle_down(self):
        with TemporaryDirectory() as tmp:
            # A file where the directory needs to be: mkdir must fail.
            blocker = Path(tmp) / "blocker"
            blocker.write_text("not a directory")
            rsi2_cycle.log_event({"event": "decision"}, path=blocker / "cycle.log")


class TestRecordFill(unittest.TestCase):
    class _Status:
        status = "Filled"
        avgFillPrice = 5432.25

    class _Order:
        orderId = 77

    class _Trade:
        orderStatus = None
        order = None

    class _Contract:
        symbol = "MES"
        localSymbol = "MESZ6"

    def _trade(self, avg_price):
        t = self._Trade()
        s = self._Status()
        s.avgFillPrice = avg_price
        t.orderStatus = s
        t.order = self._Order()
        return t

    def test_row_shape_matches_the_csv_header(self):
        row = rsi2_cycle.record_fill(self._trade(5432.25), self._Contract(),
                                     "BUY", 1, "rsi2_dip_1", "2026-09-03")
        from trading_bot import rsi2_live
        self.assertEqual(set(row), set(rsi2_live.TRADES_CSV_HEADER))
        self.assertEqual(row["fill_price"], 5432.25)
        self.assertEqual(row["local_symbol"], "MESZ6")

    def test_unfilled_order_records_zero_rather_than_guessing(self):
        row = rsi2_cycle.record_fill(self._trade(0.0), self._Contract(),
                                     "BUY", 1, "rsi2_dip_1", "2026-09-03")
        self.assertEqual(row["fill_price"], 0.0)


if __name__ == "__main__":
    unittest.main()
