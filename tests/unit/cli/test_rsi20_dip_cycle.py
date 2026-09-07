"""Unit tests for trading_bot.cli.rsi20_dip_cycle.

This file exists as much to IMPORT the module as to assert about it.
Entrypoints nothing else imports are exactly where a syntax error
survives a fully green suite -- that happened to rsi2_cycle on
2026-09-05 and only surfaced when the scheduled task failed.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trading_bot.cli import rsi20_dip_cycle as cycle


class TestParseArgs(unittest.TestCase):
    def test_defaults_are_safe(self):
        args = cycle.parse_args([])
        self.assertFalse(args.arm)
        self.assertFalse(args.ignore_window)

    def test_arm_and_ignore_window(self):
        args = cycle.parse_args(["--arm", "--ignore-window"])
        self.assertTrue(args.arm)
        self.assertTrue(args.ignore_window)


class TestClientIdIsolation(unittest.TestCase):
    def test_does_not_collide_with_the_other_bots(self):
        """Sharing a client id makes TWS drop one of the connections."""
        from trading_bot.cli import rsi2_cycle
        self.assertNotEqual(cycle.DEFAULT_CLIENT_ID, rsi2_cycle.DEFAULT_CLIENT_ID)
        self.assertNotIn(cycle.DEFAULT_CLIENT_ID, (0, 4, 95))


class TestLogEvent(unittest.TestCase):
    def test_writes_one_json_line_and_creates_the_directory(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "cycle.log"
            cycle.log_event({"event": "decision", "action": "hold"}, path=path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        row = json.loads(lines[0])
        self.assertEqual(row["event"], "decision")
        self.assertIn("ts", row)

    def test_appends_rather_than_truncating(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cycle.log"
            cycle.log_event({"event": "one"}, path=path)
            cycle.log_event({"event": "two"}, path=path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(x)["event"] for x in lines], ["one", "two"])

    def test_survives_no_console(self):
        """The scheduled task runs pythonw, where sys.stdout is None and
        print() raises. The file write must still happen, so it goes
        first."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cycle.log"
            with patch.object(sys, "stdout", None):
                cycle.log_event({"event": "headless"}, path=path)
            self.assertIn("headless", path.read_text(encoding="utf-8"))

    def test_an_unwritable_log_does_not_take_the_cycle_down(self):
        with TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("not a directory")
            cycle.log_event({"event": "decision"}, path=blocker / "cycle.log")

    def test_its_log_path_is_not_rsi2s(self):
        from trading_bot.cli import rsi2_cycle
        self.assertNotEqual(cycle.CYCLE_LOG_PATH, rsi2_cycle.CYCLE_LOG_PATH)


class TestRecordFill(unittest.TestCase):
    class _Status:
        status = "Filled"
        avgFillPrice = 5432.25

    class _Order:
        orderId = 91

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
        row = cycle.record_fill(self._trade(5432.25), self._Contract(),
                                "BUY", 1, "rsi_dip", "2026-09-04")
        from trading_bot import rsi20_dip_live
        self.assertEqual(set(row), set(rsi20_dip_live.TRADES_CSV_HEADER))
        self.assertEqual(row["fill_price"], 5432.25)
        self.assertEqual(row["local_symbol"], "MESZ6")

    def test_unfilled_order_records_zero_rather_than_guessing(self):
        row = cycle.record_fill(self._trade(0.0), self._Contract(),
                                "BUY", 1, "rsi_dip", "2026-09-04")
        self.assertEqual(row["fill_price"], 0.0)


class TestOutsideWindowShortCircuits(unittest.TestCase):
    def test_a_weekend_run_returns_without_connecting(self):
        """The gate must be checked BEFORE the broker import, so a
        mis-scheduled run costs nothing and cannot touch TWS."""
        with TemporaryDirectory() as tmp:
            with patch.object(cycle, "CYCLE_LOG_PATH", Path(tmp) / "c.log"):
                with patch.object(cycle.rsi20_dip_live, "in_decision_window",
                                  return_value=False):
                    with patch.object(cycle, "load_dotenv",
                                      side_effect=AssertionError("must not get this far")):
                        self.assertEqual(cycle.main([]), 0)
                rows = [json.loads(x) for x in
                        (Path(tmp) / "c.log").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["event"], "outside_decision_window")


if __name__ == "__main__":
    unittest.main()
