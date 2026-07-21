"""Unit tests for trading_bot.cli.morning_prefilter.

yf.download is always mocked -- no real network calls. SP500_TICKERS is
patched to a small deterministic list per test class rather than the real
500-ticker universe. notify is always patched too, since .env has real
Telegram credentials configured for this project.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_bot.cli import morning_prefilter as prefilter


def make_yf_frame(ticker_data: dict) -> pd.DataFrame:
    """Build a 2-row (yesterday, today) MultiIndex-column frame shaped like
    yf.download(group_by='ticker') returns, so data[ticker] indexing behaves
    the same way it would against the real thing."""
    frames = {ticker: pd.DataFrame(cols) for ticker, cols in ticker_data.items()}
    return pd.concat(frames, axis=1)


def bar_pair(yesterday_close, today_open, today_close, today_high=None, today_low=None):
    return {
        "Open": [yesterday_close, today_open],
        "High": [yesterday_close, today_high if today_high is not None else today_close],
        "Low": [yesterday_close, today_low if today_low is not None else today_close],
        "Close": [yesterday_close, today_close],
    }


class TestIbkrToYahoo(unittest.TestCase):
    def test_class_share_space_becomes_hyphen(self):
        self.assertEqual(prefilter.ibkr_to_yahoo("BRK B"), "BRK-B")

    def test_plain_ticker_unchanged(self):
        self.assertEqual(prefilter.ibkr_to_yahoo("AAPL"), "AAPL")


@patch("trading_bot.cli.morning_prefilter.SP500_TICKERS", ["AAPL", "MSFT", "GOOG", "BADTICKER"])
class TestScreen(unittest.TestCase):
    def test_download_exception_returns_failure(self):
        with patch("trading_bot.cli.morning_prefilter.yf.download", side_effect=ConnectionError("boom")):
            result = prefilter.screen(3.0, 3.0)

        self.assertFalse(result["success"])
        self.assertIn("boom", result["error"])
        self.assertEqual(result["total_screened"], 4)

    def test_empty_dataframe_returns_failure(self):
        with patch("trading_bot.cli.morning_prefilter.yf.download", return_value=pd.DataFrame()):
            result = prefilter.screen(3.0, 3.0)

        self.assertFalse(result["success"])
        self.assertIn("empty dataframe", result["error"])

    def test_categorizes_gap_price_and_missing_data_correctly(self):
        frame = make_yf_frame(
            {
                "AAPL": bar_pair(yesterday_close=100.0, today_open=106.0, today_close=106.0),  # gap 6% -> survivor
                "MSFT": bar_pair(yesterday_close=100.0, today_open=101.0, today_close=101.0),  # gap 1% -> below_gap
                "GOOG": bar_pair(yesterday_close=2.0, today_open=2.05, today_close=2.05),  # $2.05 -> below_price
                # BADTICKER intentionally absent from the returned frame -> KeyError -> failed
            }
        )
        with patch("trading_bot.cli.morning_prefilter.yf.download", return_value=frame):
            result = prefilter.screen(min_gap_pct=3.0, min_price=3.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["total_screened"], 4)
        self.assertEqual(result["below_gap"], 1)
        self.assertEqual(result["below_price"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["survivors_count"], 1)
        self.assertEqual(result["survivors"][0]["ticker"], "AAPL")
        self.assertAlmostEqual(result["survivors"][0]["gap_pct"], 6.0)

    def test_nan_gap_is_failed_not_survivor(self):
        """NaN/inf guard: a NaN gap must not silently slip past both the
        below_price and below_gap checks and get treated as a survivor."""
        frame = make_yf_frame(
            {
                "AAPL": bar_pair(yesterday_close=float("nan"), today_open=106.0, today_close=106.0),
                "MSFT": bar_pair(yesterday_close=100.0, today_open=110.0, today_close=110.0),
                "GOOG": bar_pair(yesterday_close=100.0, today_open=110.0, today_close=110.0),
                "BADTICKER": bar_pair(yesterday_close=100.0, today_open=110.0, today_close=110.0),
            }
        )
        with patch("trading_bot.cli.morning_prefilter.yf.download", return_value=frame):
            result = prefilter.screen(min_gap_pct=3.0, min_price=3.0)

        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["survivors_count"], 3)
        self.assertNotIn("AAPL", [s["ticker"] for s in result["survivors"]])


@patch("trading_bot.cli.morning_prefilter.SP500_TICKERS", ["A", "B", "C"])
class TestScreenCapping(unittest.TestCase):
    @patch("trading_bot.cli.morning_prefilter.MAX_SURVIVORS", 2)
    def test_survivors_sorted_descending_and_capped(self):
        frame = make_yf_frame(
            {
                "A": bar_pair(yesterday_close=100.0, today_open=105.0, today_close=105.0),  # +5%
                "B": bar_pair(yesterday_close=100.0, today_open=110.0, today_close=110.0),  # +10%
                "C": bar_pair(yesterday_close=100.0, today_open=104.0, today_close=104.0),  # +4%
            }
        )
        with patch("trading_bot.cli.morning_prefilter.yf.download", return_value=frame):
            result = prefilter.screen(min_gap_pct=3.0, min_price=3.0)

        self.assertEqual(result["survivors_before_cap"], 3)
        self.assertEqual(result["survivors_count"], 2)
        self.assertEqual([s["ticker"] for s in result["survivors"]], ["B", "A"])  # descending gap


class TestWriteWatchlist(unittest.TestCase):
    """write_watchlist: ticker-line formatting and the cap-visibility note
    (added so a big-gap day silently truncated to MAX_SURVIVORS doesn't
    read as if only that many candidates ever existed)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.watchlist_path = Path(self._tmpdir.name) / "watchlist.txt"
        self.patcher = patch.object(prefilter, "WATCHLIST_PATH", self.watchlist_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def _survivor(self, ticker="AAPL", gap_pct=5.0, today_open=105.0, yesterday_close=100.0):
        return {"ticker": ticker, "gap_pct": gap_pct, "today_open": today_open, "yesterday_close": yesterday_close}

    def test_writes_ticker_lines(self):
        prefilter.write_watchlist([self._survivor()], 3.0, 3.0, 500, survivors_before_cap=1)

        content = self.watchlist_path.read_text()
        self.assertIn("AAPL  # gap +5.00%  open $105.00  prev $100.00", content)

    def test_no_cap_note_when_not_capped(self):
        prefilter.write_watchlist([self._survivor()], 3.0, 3.0, 500, survivors_before_cap=1)

        content = self.watchlist_path.read_text()
        self.assertIn("# Survivors: 1 (screened 500 tickers)", content)
        self.assertNotIn("capped at", content)

    def test_cap_note_shown_when_truncated(self):
        prefilter.write_watchlist([self._survivor()], 3.0, 3.0, 500, survivors_before_cap=47)

        content = self.watchlist_path.read_text()
        self.assertIn("1 of 47 that passed (capped at 20)", content)


@patch("trading_bot.cli.morning_prefilter.notify")
@patch("trading_bot.cli.morning_prefilter.write_watchlist")
@patch("trading_bot.cli.morning_prefilter.screen")
class TestMain(unittest.TestCase):
    """main: dry-run gating of notify/write_watchlist, degradation alerts,
    and the truncation note reaching the Telegram body."""

    def _run_main(self, argv):
        buf = io.StringIO()
        with patch("sys.argv", ["morning_prefilter.py"] + argv), redirect_stdout(buf):
            try:
                prefilter.main()
                exit_code = None
            except SystemExit as e:
                exit_code = e.code
        return buf.getvalue(), exit_code

    def _success_result(self, survivors=None, survivors_before_cap=None, failed=0, total=500):
        survivors = survivors or []
        return {
            "success": True,
            "total_screened": total,
            "survivors": survivors,
            "survivors_count": len(survivors),
            "survivors_before_cap": survivors_before_cap if survivors_before_cap is not None else len(survivors),
            "below_gap": 0,
            "below_price": 0,
            "failed": failed,
            "elapsed_seconds": 1.23,
        }

    def test_failure_result_notifies_and_exits_1(self, mock_screen, mock_write, mock_notify):
        mock_screen.return_value = {
            "success": False,
            "error": "boom",
            "total_screened": 500,
            "elapsed_seconds": 0.1,
        }

        _, exit_code = self._run_main([])

        self.assertEqual(exit_code, 1)
        mock_notify.assert_called_once()
        self.assertIn("FAILED", mock_notify.call_args[0][0])

    def test_failure_result_dry_run_does_not_notify(self, mock_screen, mock_write, mock_notify):
        mock_screen.return_value = {
            "success": False,
            "error": "boom",
            "total_screened": 500,
            "elapsed_seconds": 0.1,
        }

        _, exit_code = self._run_main(["--dry-run"])

        self.assertEqual(exit_code, 1)
        mock_notify.assert_not_called()

    def test_success_writes_watchlist_and_notifies(self, mock_screen, mock_write, mock_notify):
        survivors = [{"ticker": "AAPL", "gap_pct": 5.0, "today_open": 105.0, "yesterday_close": 100.0}]
        mock_screen.return_value = self._success_result(survivors=survivors)

        out, exit_code = self._run_main([])

        self.assertIsNone(exit_code)
        mock_write.assert_called_once()
        mock_notify.assert_called_once()
        summary = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(summary["survivors_count"], 1)
        self.assertIsNotNone(summary["watchlist_path"])

    def test_dry_run_skips_watchlist_and_notify(self, mock_screen, mock_write, mock_notify):
        mock_screen.return_value = self._success_result(survivors=[])

        out, exit_code = self._run_main(["--dry-run"])

        self.assertIsNone(exit_code)
        mock_write.assert_not_called()
        mock_notify.assert_not_called()
        summary = json.loads(out.strip().splitlines()[-1])
        self.assertIsNone(summary["watchlist_path"])

    def test_severe_degradation_triggers_alert(self, mock_screen, mock_write, mock_notify):
        mock_screen.return_value = self._success_result(survivors=[], failed=480, total=500)

        self._run_main([])

        titles = [c[0][0] for c in mock_notify.call_args_list]
        self.assertTrue(any("ALERT" in t for t in titles))

    def test_cap_note_included_in_notify_body_when_truncated(self, mock_screen, mock_write, mock_notify):
        survivors = [{"ticker": "AAPL", "gap_pct": 5.0, "today_open": 105.0, "yesterday_close": 100.0}]
        mock_screen.return_value = self._success_result(survivors=survivors, survivors_before_cap=47)

        self._run_main([])

        body = mock_notify.call_args_list[-1][0][1]
        self.assertIn("47 passed, capped", body)


if __name__ == "__main__":
    unittest.main()
