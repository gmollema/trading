"""Unit tests for trading_bot.backtest.filters.evaluate_entry.

Mirrors tests/unit/cli/test_cycle.py's TestEvaluateEntryFilters, adapted to
evaluate_entry's precomputed-context API and the absence of I1 (the cached
intraday bars are RTH-only, so premarket high can't be computed -- see the
module docstring in filters.py).
"""

import unittest

from trading_bot.backtest import filters


def make_rules(**overrides):
    rules = {
        "daily_filters": {
            "D1_above_prior_day_high": True,
            "D2_prior_close_above_sma200": True,
            "D3_min_gap_pct_from_prior_close": 3.0,
        },
        "intraday_filters": {
            "I2_above_today_hod": True,
            "I3_rvol_min": 2.0,
            "I3_rvol_lookback_days": 14,
        },
    }
    rules.update(overrides)
    return rules


def make_daily_ctx(**overrides):
    ctx = {"prior_day_high": 100.0, "prior_day_close": 95.0, "sma200": 90.0}
    ctx.update(overrides)
    return ctx


def make_intraday_ctx(**overrides):
    ctx = {"today_hod": 103.0, "rvol": 3.0}
    ctx.update(overrides)
    return ctx


PRICE = 104.0  # above all thresholds in the defaults above


class TestEvaluateEntry(unittest.TestCase):
    def test_all_filters_pass(self):
        passed, reasons = filters.evaluate_entry(make_daily_ctx(), make_intraday_ctx(), PRICE, make_rules())

        self.assertTrue(passed)
        self.assertEqual(reasons, [])

    def test_insufficient_daily_data_fails_closed(self):
        passed, reasons = filters.evaluate_entry(None, make_intraday_ctx(), PRICE, make_rules())

        self.assertFalse(passed)
        self.assertEqual(reasons, ["insufficient daily data"])

    def test_missing_sma200_fails_closed(self):
        passed, reasons = filters.evaluate_entry(
            make_daily_ctx(sma200=None), make_intraday_ctx(), PRICE, make_rules()
        )

        self.assertFalse(passed)
        self.assertEqual(reasons, ["insufficient daily data"])

    def test_nan_sma200_fails_closed(self):
        passed, reasons = filters.evaluate_entry(
            make_daily_ctx(sma200=float("nan")), make_intraday_ctx(), PRICE, make_rules()
        )

        self.assertFalse(passed)
        self.assertEqual(reasons, ["insufficient daily data"])

    def test_insufficient_intraday_data_fails_closed(self):
        passed, reasons = filters.evaluate_entry(make_daily_ctx(), None, PRICE, make_rules())

        self.assertFalse(passed)
        self.assertEqual(reasons, ["insufficient intraday data"])

    def test_d1_fail_price_not_above_prior_day_high(self):
        passed, reasons = filters.evaluate_entry(
            make_daily_ctx(prior_day_high=110.0), make_intraday_ctx(), PRICE, make_rules()
        )

        self.assertFalse(passed)
        self.assertIn("D1 fail: price not above prior day high", reasons)

    def test_d2_fail_prior_close_not_above_sma200(self):
        passed, reasons = filters.evaluate_entry(
            make_daily_ctx(sma200=110.0), make_intraday_ctx(), PRICE, make_rules()
        )

        self.assertFalse(passed)
        self.assertIn("D2 fail: prior close not above SMA200", reasons)

    def test_d3_fail_gap_below_threshold(self):
        passed, reasons = filters.evaluate_entry(
            make_daily_ctx(prior_day_close=102.0), make_intraday_ctx(), PRICE, make_rules()  # gap ~1.96% < 3.0%
        )

        self.assertFalse(passed)
        self.assertTrue(any(r.startswith("D3 fail") for r in reasons))

    def test_i2_fail_price_below_today_hod(self):
        passed, reasons = filters.evaluate_entry(
            make_daily_ctx(), make_intraday_ctx(today_hod=110.0), PRICE, make_rules()
        )

        self.assertFalse(passed)
        self.assertIn("I2 fail: price not at/above today HOD", reasons)

    def test_i3_fail_rvol_below_minimum(self):
        passed, reasons = filters.evaluate_entry(
            make_daily_ctx(), make_intraday_ctx(rvol=1.0), PRICE, make_rules()
        )

        self.assertFalse(passed)
        self.assertIn("I3 fail: rvol 1.00 < 2.0", reasons)

    def test_i3_fail_when_rvol_missing(self):
        passed, reasons = filters.evaluate_entry(
            make_daily_ctx(), make_intraday_ctx(rvol=None), PRICE, make_rules()
        )

        self.assertFalse(passed)
        self.assertIn("I3 fail: rvol N/A < 2.0", reasons)

    def test_i3_fail_when_rvol_nan(self):
        passed, reasons = filters.evaluate_entry(
            make_daily_ctx(), make_intraday_ctx(rvol=float("nan")), PRICE, make_rules()
        )

        self.assertFalse(passed)
        self.assertIn("I3 fail: rvol N/A < 2.0", reasons)

    def test_multiple_failures_all_reported(self):
        passed, reasons = filters.evaluate_entry(
            make_daily_ctx(prior_day_high=110.0), make_intraday_ctx(today_hod=110.0), PRICE, make_rules()
        )

        self.assertFalse(passed)
        self.assertIn("D1 fail: price not above prior day high", reasons)
        self.assertIn("I2 fail: price not at/above today HOD", reasons)


if __name__ == "__main__":
    unittest.main()
