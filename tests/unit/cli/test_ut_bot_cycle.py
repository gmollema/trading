"""Unit tests for trading_bot.cli.ut_bot_cycle's broker-free pieces.

The broker-coupled flows (orders, stop management) follow cycle.py's/
smc_cycle.py's already-reviewed patterns and get exercised by paper
trading itself; these tests cover the pure logic: trade-log round trips
and bar freshness -- same scope as test_smc_cycle.py."""

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from trading_bot import ut_bot_live
from trading_bot.cli import ut_bot_cycle


class TestTradeLog(unittest.TestCase):
    def test_append_writes_header_then_rows(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ut_bot_trades.csv"
            with patch.object(ut_bot_live, "UT_BOT_TRADES_CSV_PATH", csv_path):
                ut_bot_cycle.append_trade_row("USDJPY", "BUY", 1000, 150.123456, 1, "Filled", "entry")
                ut_bot_cycle.append_trade_row("USDJPY", "SELL", 1000, 151.654321, 2, "Filled", "sell_signal")

                lines = csv_path.read_text().splitlines()
                self.assertEqual(lines[0], ",".join(ut_bot_live.TRADES_CSV_HEADER))
                self.assertEqual(len(lines), 3)
                self.assertIn("150.123456", lines[1])  # FX precision preserved, not rounded to 2dp


class TestBarsAreFresh(unittest.TestCase):
    def test_recent_last_bar_is_fresh(self):
        now = datetime.now(timezone.utc)
        idx = pd.date_range(now - timedelta(hours=2), now - timedelta(minutes=30), freq="h", tz="UTC")
        bars = pd.DataFrame({"High": [1.0] * len(idx)}, index=idx)
        self.assertTrue(ut_bot_cycle._bars_are_fresh(bars))

    def test_stale_last_bar_is_not_fresh(self):
        now = datetime.now(timezone.utc)
        idx = pd.date_range(now - timedelta(hours=5), periods=3, freq="h", tz="UTC")
        bars = pd.DataFrame({"High": [1.0] * 3}, index=idx)
        self.assertFalse(ut_bot_cycle._bars_are_fresh(bars))


class TestFastExitCheck(unittest.TestCase):
    def test_matches_get_market_status_for_a_handful_of_times(self):
        # Cross-check the duplicated fast-path logic against
        # ut_bot_live.get_market_status for the same instants, so the two
        # can't silently drift apart.
        cases = [
            ("2024-01-02 03:00", "ok"),  # Tuesday
            ("2024-01-05 16:59", "ok"),  # Friday before close
            ("2024-01-05 17:00", "closed"),  # Friday at close
            ("2024-01-06 12:00", "closed"),  # Saturday
            ("2024-01-07 16:59", "closed"),  # Sunday before reopen
            ("2024-01-07 17:00", "ok"),  # Sunday at reopen
        ]
        for ts, expected in cases:
            now_et = datetime.fromisoformat(ts)
            expected_fast = None if expected == "ok" else "closed"
            with patch("trading_bot.cli.ut_bot_cycle.datetime") as mock_dt:
                mock_dt.now.return_value = now_et
                self.assertEqual(ut_bot_cycle._fast_exit_check(), expected_fast, ts)
            self.assertEqual(ut_bot_live.get_market_status(now_et), expected, ts)


if __name__ == "__main__":
    unittest.main()
