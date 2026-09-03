"""Unit tests for the universe aggregation in cli.rsi2_universe.

The log-space-per-symbol construction here replaced a first version that
compared a SUM of simple per-trade returns against a COMPOUNDED
buy-and-hold total return. That is not a like-for-like ratio: over 16
years a name that went up 15x has a benchmark no arithmetic sum of
bounded trade returns can reach, so the ratio understated the strategy
roughly fourfold. These tests pin the corrected quantity.
"""

import math
import unittest

from trading_bot.cli.rsi2_universe import summarize


def sym(strat_ret, bh_ret, held, days):
    return {"strat_ret": strat_ret, "bh_ret": bh_ret, "held": held, "days": days}


class TestSummarizeRatio(unittest.TestCase):
    def test_equal_daily_log_rates_give_a_ratio_of_one(self):
        # Half the exposure and the matching square-root-of-compounding
        # return: 1.21 = 1.1^2, so log(1.21)/100 == log(1.1)/50.
        stats = [sym(0.1, 0.21, 50, 100)]
        s = summarize([100.0], {"A": [100.0]}, {"2020-01-02": [100.0]}, stats)
        self.assertAlmostEqual(s["median_ratio_per_day"], 1.0, places=2)

    def test_beating_the_benchmark_per_day_is_detected(self):
        # Same return as the benchmark in a tenth of the time.
        stats = [sym(0.5, 0.5, 10, 100)]
        s = summarize([1.0], {"A": [1.0]}, {"2020-01-02": [1.0]}, stats)
        self.assertAlmostEqual(s["median_ratio_per_day"], 10.0, places=2)
        self.assertEqual(s["symbols_beating_bh_per_day_pct"], 100.0)

    def test_pools_by_median_not_mean(self):
        """One 100-bagger must not carry the pooled figure. The mean of
        these ratios is far above 1; the median is exactly 1."""
        stats = [sym(0.1, 0.21, 50, 100) for _ in range(9)]
        stats.append(sym(99.0, 0.21, 50, 100))
        s = summarize([1.0], {"A": [1.0]}, {"2020-01-02": [1.0]}, stats)
        self.assertAlmostEqual(s["median_ratio_per_day"], 1.0, places=2)

    def test_log_space_is_used_not_simple_ratios(self):
        stats = [sym(1.0, 3.0, 100, 200)]
        s = summarize([1.0], {"A": [1.0]}, {"2020-01-02": [1.0]}, stats)
        expected = (math.log1p(1.0) / 100) / (math.log1p(3.0) / 200)
        self.assertAlmostEqual(s["median_ratio_per_day"], round(expected, 2), places=2)
        # A naive simple-return ratio would give 1.0/3.0 * 2 = 0.67 instead.
        self.assertNotAlmostEqual(s["median_ratio_per_day"], 0.67, places=2)

    def test_total_wipeout_symbols_are_excluded_rather_than_crashing(self):
        stats = [sym(-1.0, 0.5, 10, 100), sym(0.5, 0.5, 10, 100)]
        s = summarize([1.0], {"A": [1.0]}, {"2020-01-02": [1.0]}, stats)
        self.assertAlmostEqual(s["median_ratio_per_day"], 10.0, places=2)

    def test_zero_exposure_symbols_are_excluded(self):
        stats = [sym(0.0, 0.5, 0, 100), sym(0.5, 0.5, 10, 100)]
        s = summarize([1.0], {"A": [1.0]}, {"2020-01-02": [1.0]}, stats)
        self.assertEqual(s["symbols_beating_bh_per_day_pct"], 100.0)


class TestSummarizeConcentration(unittest.TestCase):
    def test_top_ten_date_share_is_reported(self):
        by_date = {f"2020-01-{d:02d}": [100.0] for d in range(1, 21)}
        by_date["2020-02-01"] = [2000.0]
        rets = [v for vals in by_date.values() for v in vals]
        stats = [sym(0.5, 0.5, 10, 100)]
        s = summarize(rets, {"A": rets}, by_date, stats)
        # 2000 + nine 100s = 2900 of 4000 total.
        self.assertAlmostEqual(s["top10_dates_share_pct"], 72.5, places=1)
        self.assertEqual(s["entry_dates"], 21)

    def test_no_trades_returns_a_stub(self):
        self.assertEqual(summarize([], {}, {}, []), {"trades": 0})


if __name__ == "__main__":
    unittest.main()
