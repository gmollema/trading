"""Unit tests for trading_bot.cli.smc_cycle's broker-free pieces.

The broker-coupled flows (orders, stop management) follow cycle.py's
already-reviewed patterns and get exercised by paper trading itself;
these tests cover the pure logic: trade-log round trips, daily BUY
counting, entry-bar index mapping, and bar freshness."""

import json
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
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


class TestGet5mBarsLogging(unittest.TestCase):
    """get_5m_bars returns None down three separate paths and callers see
    only that None, so each path has to say why it bailed."""

    def _fetch(self, side_effect=None, frame=None):
        """Run get_5m_bars against a stubbed yfinance, returning
        (result, logged_events)."""
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "smc-safety-check-log.json"
            ticker = MagicMock()
            if side_effect is not None:
                ticker.history.side_effect = side_effect
            else:
                ticker.history.return_value = frame
            with patch.object(smc_live, "SMC_SAFETY_LOG_PATH", log_path), \
                    patch.object(smc_cycle, "yf") as fake_yf:
                fake_yf.Ticker.return_value = ticker
                result = smc_cycle.get_5m_bars("NVDA", context="entry_scan")
            events = []
            if log_path.exists():
                events = [json.loads(line) for line in log_path.read_text().splitlines()]
            return result, events

    def test_yfinance_error_logs_cause_and_message(self):
        result, events = self._fetch(side_effect=RuntimeError("boom"))
        self.assertIsNone(result)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "bars_unavailable")
        self.assertEqual(events[0]["cause"], "yfinance_error")
        self.assertEqual(events[0]["symbol"], "NVDA")
        self.assertEqual(events[0]["context"], "entry_scan")
        # The bare `except Exception` used to discard this entirely.
        self.assertIn("boom", events[0]["error"])

    def test_empty_response_logs_its_own_cause(self):
        result, events = self._fetch(frame=pd.DataFrame())
        self.assertIsNone(result)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["cause"], "empty_response")

    def test_bars_entirely_outside_rth_log_their_own_cause(self):
        idx = pd.date_range("2026-08-24 04:00", periods=3, freq="5min", tz=ET)
        result, events = self._fetch(frame=pd.DataFrame({"High": [1.0] * 3}, index=idx))
        self.assertIsNone(result)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["cause"], "no_rth_rows")

    def test_rth_bars_are_returned_and_log_nothing(self):
        idx = pd.date_range("2026-08-24 09:55", periods=3, freq="5min", tz=ET)
        result, events = self._fetch(frame=pd.DataFrame({"High": [1.0] * 3}, index=idx))
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertEqual(events, [])


def _ohlc_bars(highs, closes=None):
    """Today's 5-min bars with the given highs, indexed from 09:30 ET."""
    today = datetime.now(ET).date()
    start = datetime(today.year, today.month, today.day, 9, 30, tzinfo=ET)
    idx = pd.date_range(start, periods=len(highs), freq="5min")
    return pd.DataFrame(
        {"High": [float(h) for h in highs], "Close": [float(c) for c in (closes or highs)]},
        index=idx,
    )


class TestTp1Touched(unittest.TestCase):
    """TP1 arms off bar highs at/after the entry bar, not a polled quote."""

    def test_post_entry_high_reaching_target_triggers(self):
        bars = _ohlc_bars([180.0, 181.0, 183.5, 182.0])
        self.assertTrue(smc_cycle._tp1_touched(bars, 1, 183.32))

    def test_post_entry_highs_below_target_do_not_trigger(self):
        bars = _ohlc_bars([180.0, 182.0, 182.96, 182.63])
        self.assertFalse(smc_cycle._tp1_touched(bars, 1, 183.32))

    def test_target_reached_only_before_entry_does_not_trigger(self):
        # The KLAC 2026-08-25 case: the stock printed 185.99 pre-entry and
        # never exceeded 182.96 afterwards, but a delayed quote still fired
        # TP1. Pre-entry highs must not count.
        bars = _ohlc_bars([185.99, 184.85, 182.86, 182.96, 182.63])
        self.assertFalse(smc_cycle._tp1_touched(bars, 2, 183.32))

    def test_entry_on_final_bar_still_evaluates_that_bar(self):
        bars = _ohlc_bars([180.0, 181.0, 183.4])
        self.assertTrue(smc_cycle._tp1_touched(bars, 2, 183.32))
        self.assertFalse(smc_cycle._tp1_touched(bars, 2, 184.0))

    def test_exact_touch_counts(self):
        bars = _ohlc_bars([180.0, 183.32])
        self.assertTrue(smc_cycle._tp1_touched(bars, 1, 183.32))


class TestLastBarClose(unittest.TestCase):
    """The force-close / new-high fill-price fallback reads a real traded
    price, not get_current_price's delayed quote."""

    def test_returns_latest_close(self):
        bars = _ohlc_bars([181.0, 182.0, 183.0], closes=[180.5, 181.5, 182.75])
        with patch.object(smc_cycle, "get_5m_bars", return_value=bars):
            self.assertAlmostEqual(smc_cycle._last_bar_close("KLAC"), 182.75)

    def test_missing_bars_return_none_so_caller_can_fall_through(self):
        with patch.object(smc_cycle, "get_5m_bars", return_value=None):
            self.assertIsNone(smc_cycle._last_bar_close("KLAC"))

    def test_empty_frame_returns_none(self):
        with patch.object(smc_cycle, "get_5m_bars", return_value=pd.DataFrame({"Close": []})):
            self.assertIsNone(smc_cycle._last_bar_close("KLAC"))


class _FakeOrderStatus:
    def __init__(self, statuses):
        self._statuses = list(statuses)

    @property
    def status(self):
        # Last value repeats once the script is exhausted.
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]


class _FakeTrade:
    def __init__(self, statuses):
        self.orderStatus = _FakeOrderStatus(statuses)


class _FakeIB:
    def __init__(self, trade):
        self._trade = trade
        self.sleeps = 0

    def placeOrder(self, contract, order):
        return self._trade

    def sleep(self, _secs):
        self.sleeps += 1

    def qualifyContracts(self, stock):
        return [stock]


class TestMarketOrderWaitsForFill(unittest.TestCase):
    """_market_order must wait for a settled status, not for "Submitted"."""

    def _run(self, statuses, timeout=1.0):
        ib = _FakeIB(_FakeTrade(statuses))
        with patch.object(smc_cycle, "ORDER_FILL_TIMEOUT_SECS", timeout):
            trade = smc_cycle._market_order(ib, "NVDA", "BUY", 10)
        return ib, trade

    def test_returns_once_filled(self):
        ib, _ = self._run(["PendingSubmit", "Submitted", "Filled"])
        # Broke out on Filled rather than spinning to the timeout.
        self.assertEqual(ib.sleeps, 3)

    def test_does_not_break_early_on_submitted(self):
        # The regression guard: "Submitted" arrives before any fill, so
        # breaking on it left avgFillPrice empty and callers recording a
        # theoretical price instead of the real one.
        ib, _ = self._run(["Submitted"], timeout=0.6)
        self.assertGreater(ib.sleeps, 1)

    def test_returns_on_terminal_non_fill(self):
        ib, _ = self._run(["Submitted", "Cancelled"])
        self.assertEqual(ib.sleeps, 2)


if __name__ == "__main__":
    unittest.main()
