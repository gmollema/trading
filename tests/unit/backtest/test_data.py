"""Unit tests for trading_bot.backtest.data.

Uses small synthetic CSV fixtures (not the real backtest_data/ cache) via
tempfile.TemporaryDirectory(), passed as explicit daily_dir/intraday_dir
overrides -- load_daily/load_intraday/build_symbol_frame/available_tickers
all accept these, so no module-level Path patching is needed here (unlike
cycle.py's tests, which patch module constants because cycle.py's own
functions don't take a dir argument).
"""

import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading_bot.backtest import data


def _write_daily_csv(path: Path, dates: list[str], closes: list[float]):
    lines = ["Date,Open,High,Low,Close,Volume,Dividends,Stock Splits"]
    for d, c in zip(dates, closes):
        lines.append(f"{d},{c},{c + 1},{c - 1},{c},1000,0.0,0.0")
    path.write_text("\n".join(lines) + "\n")


def _write_intraday_csv(path: Path, rows: list[tuple[str, float, float, float, float, int]]):
    lines = ["date,open,high,low,close,volume"]
    for ts, o, h, low, c, v in rows:
        lines.append(f"{ts},{o},{h},{low},{c},{v}")
    path.write_text("\n".join(lines) + "\n")


def _daily_dates(n: int) -> list[str]:
    # One row per calendar day (weekends included -- irrelevant to the pure
    # date-shift math being tested here) starting 2024-01-01.
    return [d.strftime("%Y-%m-%d 00:00:00-05:00") for d in pd.date_range("2024-01-01", periods=n, freq="D")]


class TestLoadDaily(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daily_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_missing_file_returns_none(self):
        self.assertIsNone(data.load_daily("NOPE", self.daily_dir))

    def test_loads_tz_aware_and_sorted(self):
        _write_daily_csv(self.daily_dir / "AAPL.csv", _daily_dates(3), [10.0, 11.0, 12.0])

        df = data.load_daily("AAPL", self.daily_dir)

        self.assertEqual(len(df), 3)
        self.assertEqual(str(df["Date"].dt.tz), "America/New_York")
        self.assertTrue(df["Date"].is_monotonic_increasing)

    def test_class_share_ticker_uses_underscore_filename(self):
        _write_daily_csv(self.daily_dir / "BRK_B.csv", _daily_dates(1), [300.0])

        df = data.load_daily("BRK B", self.daily_dir)

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)


class TestLoadIntraday(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.intraday_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_missing_file_returns_none(self):
        self.assertIsNone(data.load_intraday("NOPE", self.intraday_dir))

    def test_loads_tz_aware_and_sorted(self):
        _write_intraday_csv(
            self.intraday_dir / "AAPL.csv",
            [
                ("2024-07-15 09:35:00-04:00", 10, 10.5, 9.5, 10.2, 100),
                ("2024-07-15 09:30:00-04:00", 10, 10.5, 9.5, 10.0, 100),
            ],
        )

        df = data.load_intraday("AAPL", self.intraday_dir)

        self.assertEqual(len(df), 2)
        self.assertTrue(df["date"].is_monotonic_increasing)
        self.assertEqual(str(df["date"].dt.tz), "America/New_York")


class TestComputeDailyContext(unittest.TestCase):
    def test_first_row_has_no_prior_context(self):
        daily = pd.DataFrame(
            {"Date": pd.to_datetime(_daily_dates(3), utc=True), "High": [11, 12, 13], "Close": [10, 11, 12]}
        )

        out = data.compute_daily_context(daily)

        self.assertTrue(math.isnan(out["prior_day_high"].iloc[0]))
        self.assertTrue(math.isnan(out["prior_day_close"].iloc[0]))

    def test_prior_day_values_are_a_one_row_shift(self):
        daily = pd.DataFrame(
            {"Date": pd.to_datetime(_daily_dates(3), utc=True), "High": [11, 12, 13], "Close": [10, 11, 12]}
        )

        out = data.compute_daily_context(daily)

        self.assertEqual(out["prior_day_high"].iloc[1], 11)
        self.assertEqual(out["prior_day_close"].iloc[1], 10)
        self.assertEqual(out["prior_day_high"].iloc[2], 12)
        self.assertEqual(out["prior_day_close"].iloc[2], 11)

    def test_sma200_nan_before_200_prior_days_available(self):
        n = 205
        daily = pd.DataFrame(
            {
                "Date": pd.to_datetime(_daily_dates(n), utc=True),
                "High": [float(i) + 1 for i in range(n)],
                "Close": [float(i) for i in range(n)],
            }
        )

        out = data.compute_daily_context(daily)

        self.assertTrue(math.isnan(out["sma200"].iloc[199]))
        self.assertFalse(math.isnan(out["sma200"].iloc[200]))

    def test_sma200_value_excludes_todays_own_close(self):
        n = 205
        daily = pd.DataFrame(
            {
                "Date": pd.to_datetime(_daily_dates(n), utc=True),
                "High": [float(i) + 1 for i in range(n)],
                "Close": [float(i) for i in range(n)],  # Close[i] == i
            }
        )

        out = data.compute_daily_context(daily)

        # sma200 at row 200 = mean(Close[0:200]) = mean(0..199) = 99.5
        self.assertAlmostEqual(out["sma200"].iloc[200], 99.5)

    def test_no_lookahead_truncating_future_rows_leaves_past_context_unchanged(self):
        n = 210
        closes = [float(i) for i in range(n)]
        highs = [c + 1 for c in closes]
        dates = _daily_dates(n)
        full = pd.DataFrame({"Date": pd.to_datetime(dates, utc=True), "High": highs, "Close": closes})
        truncated = full.iloc[:205].copy()

        out_full = data.compute_daily_context(full)
        out_truncated = data.compute_daily_context(truncated)

        last_idx = 204
        self.assertEqual(out_full["prior_day_high"].iloc[last_idx], out_truncated["prior_day_high"].iloc[last_idx])
        self.assertEqual(out_full["prior_day_close"].iloc[last_idx], out_truncated["prior_day_close"].iloc[last_idx])
        self.assertEqual(out_full["sma200"].iloc[last_idx], out_truncated["sma200"].iloc[last_idx])


class TestComputeIntradayContext(unittest.TestCase):
    """3 days, 2 bars/day (09:30, 09:35), rvol_lookback_days=2.

    volumes: day1 [100, 50], day2 [200, 100], day3 [300, 150]
    cum vols: day1 [100, 150], day2 [200, 300], day3 [300, 450]
    """

    def _make_intraday_df(self):
        rows = [
            ("2024-07-15 09:30:00-04:00", 100),
            ("2024-07-15 09:35:00-04:00", 50),
            ("2024-07-16 09:30:00-04:00", 200),
            ("2024-07-16 09:35:00-04:00", 100),
            ("2024-07-17 09:30:00-04:00", 300),
            ("2024-07-17 09:35:00-04:00", 150),
        ]
        return pd.DataFrame(
            {
                "date": pd.to_datetime([r[0] for r in rows], utc=True),
                "open": 10.0,
                "high": [11.0] * 6,
                "low": [9.0] * 6,
                "close": [10.0] * 6,
                "volume": [r[1] for r in rows],
            }
        )

    def test_running_cum_vol_resets_each_day(self):
        out = data.compute_intraday_context(self._make_intraday_df(), rvol_lookback_days=2)

        self.assertEqual(out["running_cum_vol"].tolist(), [100, 150, 200, 300, 300, 450])

    def test_running_hod_lod_per_day(self):
        df = self._make_intraday_df()
        df.loc[0, "high"] = 12.0  # day1 09:30 makes a fresh high
        df.loc[1, "high"] = 11.0  # day1 09:35 lower high -- running_hod stays 12.0

        out = data.compute_intraday_context(df, rvol_lookback_days=2)

        self.assertEqual(out["running_hod"].iloc[0], 12.0)
        self.assertEqual(out["running_hod"].iloc[1], 12.0)

    def test_first_day_rvol_is_nan_no_history(self):
        out = data.compute_intraday_context(self._make_intraday_df(), rvol_lookback_days=2)

        self.assertTrue(math.isnan(out["rvol"].iloc[0]))
        self.assertTrue(math.isnan(out["rvol"].iloc[1]))

    def test_second_day_rvol_uses_single_prior_day(self):
        out = data.compute_intraday_context(self._make_intraday_df(), rvol_lookback_days=2)

        # day2 09:30: denom = day1's cum vol @ 09:30 = 100 -> rvol = 200/100 = 2.0
        self.assertAlmostEqual(out["rvol"].iloc[2], 2.0)
        # day2 09:35: denom = day1's cum vol @ 09:35 = 150 -> rvol = 300/150 = 2.0
        self.assertAlmostEqual(out["rvol"].iloc[3], 2.0)

    def test_third_day_rvol_averages_two_prior_days(self):
        out = data.compute_intraday_context(self._make_intraday_df(), rvol_lookback_days=2)

        # day3 09:30: denom = mean(100, 200) = 150 -> rvol = 300/150 = 2.0
        self.assertAlmostEqual(out["rvol"].iloc[4], 2.0)
        # day3 09:35: denom = mean(150, 300) = 225 -> rvol = 450/225 = 2.0
        self.assertAlmostEqual(out["rvol"].iloc[5], 2.0)

    def test_no_lookahead_truncating_future_days_leaves_past_context_unchanged(self):
        full = self._make_intraday_df()
        truncated = full.iloc[:4].copy()  # drop day3 entirely

        out_full = data.compute_intraday_context(full, rvol_lookback_days=2)
        out_truncated = data.compute_intraday_context(truncated, rvol_lookback_days=2)

        for i in range(4):
            self.assertEqual(out_full["running_cum_vol"].iloc[i], out_truncated["running_cum_vol"].iloc[i])
            full_rvol, trunc_rvol = out_full["rvol"].iloc[i], out_truncated["rvol"].iloc[i]
            if math.isnan(full_rvol):
                self.assertTrue(math.isnan(trunc_rvol))
            else:
                self.assertAlmostEqual(full_rvol, trunc_rvol)


class TestBuildSymbolFrame(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daily_dir = Path(self._tmpdir.name) / "daily"
        self.intraday_dir = Path(self._tmpdir.name) / "intraday_5m"
        self.daily_dir.mkdir()
        self.intraday_dir.mkdir()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_missing_daily_returns_none(self):
        _write_intraday_csv(
            self.intraday_dir / "AAPL.csv", [("2024-07-15 09:30:00-04:00", 10, 10.5, 9.5, 10.2, 100)]
        )

        result = data.build_symbol_frame("AAPL", 14, self.daily_dir, self.intraday_dir)

        self.assertIsNone(result)

    def test_missing_intraday_returns_none(self):
        _write_daily_csv(self.daily_dir / "AAPL.csv", _daily_dates(1), [10.0])

        result = data.build_symbol_frame("AAPL", 14, self.daily_dir, self.intraday_dir)

        self.assertIsNone(result)

    def test_merges_daily_context_onto_matching_trading_date(self):
        _write_daily_csv(self.daily_dir / "AAPL.csv", _daily_dates(3), [10.0, 11.0, 12.0])
        third_day = _daily_dates(3)[2][:10]  # just the date part, e.g. "2024-01-03"
        _write_intraday_csv(
            self.intraday_dir / "AAPL.csv",
            [(f"{third_day} 09:30:00-05:00", 12, 12.5, 11.5, 12.2, 500)],
        )

        result = data.build_symbol_frame("AAPL", 14, self.daily_dir, self.intraday_dir)

        self.assertIsNotNone(result)
        self.assertEqual(result["symbol"].iloc[0], "AAPL")
        self.assertEqual(result["prior_day_high"].iloc[0], 12.0)  # High of the 2nd daily row (Close 11 + 1)
        self.assertEqual(result["prior_day_close"].iloc[0], 11.0)


class TestAvailableTickers(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daily_dir = Path(self._tmpdir.name) / "daily"
        self.intraday_dir = Path(self._tmpdir.name) / "intraday_5m"
        self.daily_dir.mkdir()
        self.intraday_dir.mkdir()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_only_tickers_with_both_files_are_kept_in_order(self):
        for ticker in ("AAPL", "MSFT", "NVDA"):
            (self.daily_dir / f"{ticker}.csv").write_text("x")
        for ticker in ("AAPL", "NVDA"):  # MSFT missing its intraday file
            (self.intraday_dir / f"{ticker}.csv").write_text("x")

        result = data.available_tickers(["MSFT", "AAPL", "NVDA", "TSLA"], self.daily_dir, self.intraday_dir)

        self.assertEqual(result, ["AAPL", "NVDA"])


if __name__ == "__main__":
    unittest.main()
