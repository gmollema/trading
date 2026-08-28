"""Unit tests for trading_bot.cli.merge_backtest_data.

Uses small synthetic CSV fixtures in tempfile.TemporaryDirectory() rather
than the real backtest_data/ cache, in the same spirit as
tests/unit/backtest/test_data.py.

The cases that matter here are the two the merge exists to get right:
newest-source-wins on an overlapping timestamp (so a partial final bar is
replaced by its settled version), and refusing to splice sources that sit
on different price-adjustment bases.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading_bot.cli import merge_backtest_data as m


def _write_intraday(path: Path, rows: list[tuple[str, float, float]]):
    """rows are (timestamp, close, volume); OHL are derived off close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["date,open,high,low,close,volume"]
    for ts, c, v in rows:
        lines.append(f"{ts},{c},{c + 1},{c - 1},{c},{v}")
    path.write_text("\n".join(lines) + "\n")


def _write_daily(path: Path, rows: list[tuple[str, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Date,Open,High,Low,Close,Volume,Dividends,Stock Splits"]
    for d, c in rows:
        lines.append(f"{d},{c},{c + 1},{c - 1},{c},1000,0.0,0.0")
    path.write_text("\n".join(lines) + "\n")


def _daily_dates(n: int) -> list[str]:
    """`n` consecutive calendar days as UTC midnight stamps.

    Weekends are included -- irrelevant to the pure timestamp-keyed merge
    logic under test, and it keeps the count above MIN_OVERLAP_ROWS.
    """
    return [d.strftime("%Y-%m-%d 00:00:00+00:00") for d in pd.date_range("2024-01-02", periods=n)]


class DetectDateColTest(unittest.TestCase):
    def test_distinguishes_daily_from_intraday_by_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_daily(d / "a.csv", [("2024-01-02 00:00:00-05:00", 10.0)])
            _write_intraday(d / "b.csv", [("2024-01-02 09:30:00-05:00", 10.0, 100)])
            (d / "junk.csv").write_text("foo,bar\n1,2\n")
            self.assertEqual(m.detect_date_col(d / "a.csv"), "Date")
            self.assertEqual(m.detect_date_col(d / "b.csv"), "date")
            self.assertIsNone(m.detect_date_col(d / "junk.csv"))


class MergeTickerTest(unittest.TestCase):
    def test_newest_source_wins_on_overlapping_timestamp(self):
        # The shared 09:35 bar is partial in the old source (half the
        # volume, stale close) and settled in the new one.
        old = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-02 14:30:00+00:00", "2024-01-02 14:35:00+00:00"], utc=True
                ),
                "close": [10.0, 10.5],
                "volume": [100.0, 50.0],
            }
        )
        new = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-02 14:35:00+00:00", "2024-01-02 14:40:00+00:00"], utc=True
                ),
                "close": [10.9, 11.0],
                "volume": [220.0, 300.0],
            }
        )
        out = m.merge_ticker([old, new], "date")
        self.assertEqual(len(out), 3)
        settled = out[out["date"] == pd.Timestamp("2024-01-02 14:35:00+00:00")]
        self.assertEqual(settled["close"].iloc[0], 10.9)
        self.assertEqual(settled["volume"].iloc[0], 220.0)

    def test_keeps_history_only_the_older_source_has(self):
        old = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02 14:30:00+00:00"], utc=True),
                "close": [10.0],
                "volume": [100.0],
            }
        )
        new = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-06-03 14:30:00+00:00"], utc=True),
                "close": [20.0],
                "volume": [200.0],
            }
        )
        out = m.merge_ticker([old, new], "date")
        self.assertEqual(len(out), 2)
        self.assertEqual(out["date"].min(), pd.Timestamp("2024-01-02 14:30:00+00:00"))

    def test_output_is_sorted_ascending(self):
        new = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-06-03 14:30:00+00:00"], utc=True),
                "close": [20.0],
                "volume": [200.0],
            }
        )
        old = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02 14:30:00+00:00"], utc=True),
                "close": [10.0],
                "volume": [100.0],
            }
        )
        # Sources passed oldest-first; merge must still sort by time.
        out = m.merge_ticker([old, new], "date")
        self.assertTrue(out["date"].is_monotonic_increasing)

    def test_mixed_offset_and_utc_sources_align_on_the_same_instant(self):
        # The fetch step writes ET offsets, previously merged dirs hold
        # UTC; the same instant written both ways must dedupe to one row.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_intraday(d / "utc.csv", [("2024-01-02 14:30:00+00:00", 10.0, 100)])
            _write_intraday(d / "et.csv", [("2024-01-02 09:30:00-05:00", 11.0, 200)])
            a = m.read_cache_csv(d / "utc.csv", "date")
            b = m.read_cache_csv(d / "et.csv", "date")
            out = m.merge_ticker([a, b], "date")
            self.assertEqual(len(out), 1)
            self.assertEqual(out["close"].iloc[0], 11.0)


class CompareOverlapTest(unittest.TestCase):
    def _frames(self, old_closes, new_closes):
        ts = pd.date_range("2024-01-02 14:30:00+00:00", periods=len(old_closes), freq="D")
        old = pd.DataFrame({"date": ts, "close": old_closes})
        new = pd.DataFrame({"date": ts, "close": new_closes})
        return old, new

    def test_identical_sources_report_no_disagreement(self):
        old, new = self._frames([10.0, 11.0, 12.0], [10.0, 11.0, 12.0])
        n_overlap, n_diff, median_shift = m.compare_overlap(old, new, "date")
        self.assertEqual((n_overlap, n_diff), (3, 0))
        self.assertEqual(median_shift, 0.0)

    def test_wholesale_rescale_shows_a_consistent_signed_shift(self):
        old, new = self._frames([10.0, 11.0, 12.0], [9.9, 10.89, 11.88])
        n_overlap, n_diff, median_shift = m.compare_overlap(old, new, "date")
        self.assertEqual((n_overlap, n_diff), (3, 3))
        self.assertAlmostEqual(median_shift, -0.01, places=6)

    def test_balanced_noise_cancels_to_about_zero(self):
        # Sub-cent corrections scattered in both directions -- the shape
        # IBKR intraday refetches actually produce. Many rows differ, but
        # the signed median stays far below the rescale threshold.
        n = 100
        old_closes = [100.0] * n
        new_closes = [100.0 + (0.01 if i % 2 else -0.01) for i in range(n)]
        old, new = self._frames(old_closes, new_closes)
        _, n_diff, median_shift = m.compare_overlap(old, new, "date")
        self.assertEqual(n_diff, n)
        self.assertLess(abs(median_shift), m.RESCALE_MEDIAN_REL)

    def test_median_is_taken_over_the_oldest_slice(self):
        # A dividend part-way through the overlap: the oldest rows are
        # rescaled, recent ones untouched. Judging on the whole overlap
        # would wash this out; judging on the oldest slice must catch it.
        n = 100
        old_closes = [100.0] * n
        new_closes = [99.0] * 20 + [100.0] * 80
        old, new = self._frames(old_closes, new_closes)
        _, _, median_shift = m.compare_overlap(old, new, "date")
        self.assertAlmostEqual(median_shift, -0.01, places=6)
        self.assertGreater(abs(median_shift), m.RESCALE_MEDIAN_REL)

    def test_disjoint_sources_report_zero_overlap(self):
        old = pd.DataFrame(
            {"date": pd.to_datetime(["2024-01-02 14:30:00+00:00"], utc=True), "close": [10.0]}
        )
        new = pd.DataFrame(
            {"date": pd.to_datetime(["2024-06-03 14:30:00+00:00"], utc=True), "close": [20.0]}
        )
        self.assertEqual(m.compare_overlap(old, new, "date"), (0, 0, 0.0))


class MainTest(unittest.TestCase):
    def _run(self, argv):
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", ["merge_backtest_data"] + argv):
            return m.main()

    def _intraday_bars(self, start_min: int, count: int, closes=None):
        """`count` consecutive 5-min bars from 14:30 UTC + start_min."""
        base = pd.Timestamp("2024-01-02 14:30:00+00:00")
        out = []
        for i in range(count):
            ts = base + pd.Timedelta(minutes=5 * (start_min + i))
            c = 10.0 + (start_min + i) * 0.01 if closes is None else closes[i]
            out.append((ts.strftime("%Y-%m-%d %H:%M:%S%z"), round(c, 4), 100 + i))
        return out

    def test_merges_intraday_and_writes_output(self):
        # A realistic shape: a long clean overlap plus one late-corrected
        # bar, which must NOT trip the adjustment-basis guard.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            old_rows = self._intraday_bars(0, 40)
            new_rows = self._intraday_bars(20, 40)
            # The bar at index 20 (old) / 0 (new) came back corrected by a cent.
            corrected = (new_rows[0][0], new_rows[0][1] + 0.01, new_rows[0][2])
            new_rows[0] = corrected
            _write_intraday(d / "old" / "AAPL.csv", old_rows)
            _write_intraday(d / "new" / "AAPL.csv", new_rows)

            rc = self._run([str(d / "old"), str(d / "new"), "--out", str(d / "out")])
            self.assertEqual(rc, 0)

            got = pd.read_csv(d / "out" / "AAPL.csv")
            got["date"] = pd.to_datetime(got["date"], utc=True)
            # 40 old + 40 new sharing 20 timestamps => 60 distinct bars.
            self.assertEqual(len(got), 60)
            self.assertTrue(got["date"].is_monotonic_increasing)
            # The corrected bar carries the newer source's close.
            seam = got[got["date"] == pd.Timestamp(corrected[0])]
            self.assertAlmostEqual(seam["close"].iloc[0], corrected[1], places=4)

    def test_long_clean_overlap_is_not_flagged_as_a_rescale(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rows = self._intraday_bars(0, 50)
            _write_intraday(d / "old" / "AAPL.csv", rows)
            _write_intraday(d / "new" / "AAPL.csv", rows)
            rc = self._run([str(d / "old"), str(d / "new"), "--out", str(d / "out")])
            self.assertEqual(rc, 0)
            self.assertTrue((d / "out" / "AAPL.csv").exists())

    def test_widespread_balanced_noise_does_not_block_the_merge(self):
        # The real intraday case: a long overlap where ~half the bars
        # differ by a cent in either direction. Counting differing rows
        # would refuse this merge; the signed median must not.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rows = self._intraday_bars(0, 60)
            noisy = [
                (ts, c + (0.01 if i % 2 else -0.01), v) for i, (ts, c, v) in enumerate(rows)
            ]
            _write_intraday(d / "old" / "AAPL.csv", rows)
            _write_intraday(d / "new" / "AAPL.csv", noisy)
            rc = self._run([str(d / "old"), str(d / "new"), "--out", str(d / "out")])
            self.assertEqual(rc, 0)
            self.assertTrue((d / "out" / "AAPL.csv").exists())

    def test_short_overlap_is_below_the_detection_floor(self):
        # Documents the known limitation: fewer than MIN_OVERLAP_ROWS
        # shared rows is too little signal to call a rescale, so the merge
        # proceeds rather than false-positiving on a corrected bar.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_intraday(d / "old" / "AAPL.csv", self._intraday_bars(0, 3))
            _write_intraday(d / "new" / "AAPL.csv", self._intraday_bars(2, 3))
            rc = self._run([str(d / "old"), str(d / "new"), "--out", str(d / "out")])
            self.assertEqual(rc, 0)

    def test_carries_forward_a_ticker_only_the_older_source_has(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_intraday(d / "old" / "GONE.csv", [("2024-01-02 14:30:00+00:00", 10.0, 100)])
            _write_intraday(d / "new" / "AAPL.csv", [("2024-01-02 14:30:00+00:00", 20.0, 100)])
            rc = self._run([str(d / "old"), str(d / "new"), "--out", str(d / "out")])
            self.assertEqual(rc, 0)
            self.assertTrue((d / "out" / "GONE.csv").exists())
            self.assertTrue((d / "out" / "AAPL.csv").exists())

    def test_refuses_to_splice_different_adjustment_bases(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            dates = _daily_dates(30)
            _write_daily(d / "old" / "PG.csv", [(x, 100.0 + i) for i, x in enumerate(dates)])
            # Same days, every close rescaled by a dividend adjustment.
            _write_daily(
                d / "new" / "PG.csv", [(x, (100.0 + i) * 0.993) for i, x in enumerate(dates)]
            )
            rc = self._run([str(d / "old"), str(d / "new"), "--out", str(d / "out")])
            self.assertEqual(rc, 1)
            self.assertFalse((d / "out" / "PG.csv").exists())

    def test_on_rescale_warn_merges_anyway(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            dates = _daily_dates(30)
            _write_daily(d / "old" / "PG.csv", [(x, 100.0 + i) for i, x in enumerate(dates)])
            _write_daily(
                d / "new" / "PG.csv", [(x, (100.0 + i) * 0.993) for i, x in enumerate(dates)]
            )
            rc = self._run(
                [str(d / "old"), str(d / "new"), "--out", str(d / "out"), "--on-rescale", "warn"]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((d / "out" / "PG.csv").exists())

    def test_on_rescale_skip_omits_the_ticker(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            dates = _daily_dates(30)
            _write_daily(d / "old" / "PG.csv", [(x, 100.0 + i) for i, x in enumerate(dates)])
            _write_daily(
                d / "new" / "PG.csv", [(x, (100.0 + i) * 0.993) for i, x in enumerate(dates)]
            )
            _write_daily(d / "old" / "OK.csv", [(x, 50.0) for x in dates])
            _write_daily(d / "new" / "OK.csv", [(x, 50.0) for x in dates])
            rc = self._run(
                [str(d / "old"), str(d / "new"), "--out", str(d / "out"), "--on-rescale", "skip"]
            )
            self.assertEqual(rc, 0)
            self.assertFalse((d / "out" / "PG.csv").exists())
            self.assertTrue((d / "out" / "OK.csv").exists())

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_intraday(d / "old" / "AAPL.csv", [("2024-01-02 14:30:00+00:00", 10.0, 100)])
            _write_intraday(d / "new" / "AAPL.csv", [("2024-01-03 14:30:00+00:00", 11.0, 100)])
            rc = self._run([str(d / "old"), str(d / "new"), "--out", str(d / "out"), "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertFalse((d / "out").exists())

    def test_refuses_to_overwrite_a_populated_out_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_intraday(d / "old" / "AAPL.csv", [("2024-01-02 14:30:00+00:00", 10.0, 100)])
            _write_intraday(d / "new" / "AAPL.csv", [("2024-01-03 14:30:00+00:00", 11.0, 100)])
            _write_intraday(d / "out" / "AAPL.csv", [("2020-01-02 14:30:00+00:00", 1.0, 1)])
            rc = self._run([str(d / "old"), str(d / "new"), "--out", str(d / "out")])
            self.assertEqual(rc, 2)
            # Pre-existing content left untouched.
            self.assertEqual(len(pd.read_csv(d / "out" / "AAPL.csv")), 1)

    def test_refuses_when_out_is_also_a_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_intraday(d / "old" / "AAPL.csv", [("2024-01-02 14:30:00+00:00", 10.0, 100)])
            _write_intraday(d / "new" / "AAPL.csv", [("2024-01-03 14:30:00+00:00", 11.0, 100)])
            rc = self._run([str(d / "old"), str(d / "new"), "--out", str(d / "old")])
            self.assertEqual(rc, 2)

    def test_requires_at_least_two_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_intraday(d / "old" / "AAPL.csv", [("2024-01-02 14:30:00+00:00", 10.0, 100)])
            rc = self._run([str(d / "old"), "--out", str(d / "out")])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
