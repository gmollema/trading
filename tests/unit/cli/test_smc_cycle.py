"""Unit tests for trading_bot.cli.smc_cycle's broker-free pieces.

The broker-coupled flows (orders, stop management) follow cycle.py's
already-reviewed patterns and get exercised by paper trading itself;
these tests cover the pure logic: trade-log round trips, daily BUY
counting, entry-bar index mapping, and bar freshness."""

import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from trading_bot import smc_live
from trading_bot.cli import smc_cycle

ET = ZoneInfo("America/New_York")


class TestTradeLog(unittest.TestCase):
    def test_append_creates_header_then_counts_todays_buys(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "smc_trades.csv"
            with patch.object(smc_live, "SMC_TRADES_CSV_PATH", csv_path):
                smc_cycle.append_trade_row("NVDA", "BUY", 10, 100.5, 1, "Filled", "entry")
                smc_cycle.append_trade_row("NVDA", "SELL", 10, 101.5, 2, "Filled", "tp1")
                smc_cycle.append_trade_row("AMD", "BUY", 5, 50.0, 3, "Filled", "entry")

                lines = csv_path.read_text().splitlines()
                self.assertEqual(lines[0], ",".join(smc_live.TRADES_CSV_HEADER))
                self.assertEqual(len(lines), 4)
                self.assertEqual(smc_cycle.count_today_buys(), 2)

    def test_count_today_buys_missing_file_is_zero(self):
        with TemporaryDirectory() as tmp:
            with patch.object(smc_live, "SMC_TRADES_CSV_PATH", Path(tmp) / "none.csv"):
                self.assertEqual(smc_cycle.count_today_buys(), 0)


def _today_bars(n: int, start_hh=9, start_mm=30):
    today = datetime.now(ET).date()
    start = datetime(today.year, today.month, today.day, start_hh, start_mm, tzinfo=ET)
    idx = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({"High": [10.0] * n}, index=idx)


class TestEntryBarIndex(unittest.TestCase):
    def test_exact_match(self):
        bars = _today_bars(5)
        self.assertEqual(smc_cycle._entry_bar_index(bars, bars.index[2].isoformat()), 2)

    def test_first_bar_at_or_after_when_no_exact_match(self):
        bars = _today_bars(5)
        between = (bars.index[2] + timedelta(minutes=2)).isoformat()
        self.assertEqual(smc_cycle._entry_bar_index(bars, between), 3)

    def test_unparseable_or_future_returns_none(self):
        bars = _today_bars(5)
        self.assertIsNone(smc_cycle._entry_bar_index(bars, "not-a-date"))
        after_all = (bars.index[-1] + timedelta(hours=1)).isoformat()
        self.assertIsNone(smc_cycle._entry_bar_index(bars, after_all))


class TestBarsAreFresh(unittest.TestCase):
    def test_recent_last_bar_is_fresh(self):
        now = datetime.now(ET)
        idx = pd.date_range(now - timedelta(minutes=30), now - timedelta(minutes=5), freq="5min")
        bars = pd.DataFrame({"High": [1.0] * len(idx)}, index=idx)
        self.assertTrue(smc_cycle._bars_are_fresh(bars))

    def test_stale_last_bar_is_not_fresh(self):
        now = datetime.now(ET)
        idx = pd.date_range(now - timedelta(hours=3), periods=5, freq="5min")
        bars = pd.DataFrame({"High": [1.0] * 5}, index=idx)
        self.assertFalse(smc_cycle._bars_are_fresh(bars))

    def test_yesterdays_bars_are_not_fresh(self):
        now = datetime.now(ET)
        idx = pd.date_range(now - timedelta(days=1, minutes=10), periods=3, freq="5min")
        bars = pd.DataFrame({"High": [1.0] * 3}, index=idx)
        self.assertFalse(smc_cycle._bars_are_fresh(bars))


if __name__ == "__main__":
    unittest.main()
