"""Unit tests for trading_bot.cli.rsi20_dip_etf_backtest.

The signal is already covered by test_rsi20_dip_signals; what is tested
here is the exposure model, and specifically the property that motivated
writing a bar-by-bar walk instead of scaling each trade's return:
volatility decay. If simulate_leveraged ever starts agreeing with
`N x trade pct`, the leveraged numbers become fiction and this suite is
what should notice.
"""

import unittest
from datetime import date, timedelta

from trading_bot.cli import rsi20_dip_etf_backtest as bt


def bars_from_closes(closes, start=date(2024, 1, 1)):
    """Bars whose opens equal the previous close, so a full-period hold
    has no partial-day effects to reason about."""
    n = len(closes)
    opens = [closes[0]] + [closes[i - 1] for i in range(1, n)]
    return {"date": [start + timedelta(days=i) for i in range(n)],
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes)],
            "low": [min(o, c) for o, c in zip(opens, closes)],
            "close": list(closes)}


def hold_everything(bars):
    return [{"entry_idx": 0, "exit_idx": len(bars["close"]) - 1}]


class TestLeverageIsAppliedDaily(unittest.TestCase):
    def test_a_straight_line_up_earns_about_n_times(self):
        """With no volatility there is no decay, so 3x a steady rise is
        close to 3x the move -- the sanity floor for the model.

        The comparison is against closes[-2], not closes[-1]: the exit
        fills at the last bar's OPEN, which in this fixture equals the
        previous close. The final bar's own move is never earned.
        """
        closes = [100.0 * (1.01 ** k) for k in range(11)]
        bars = bars_from_closes(closes)
        one = bt.simulate_leveraged(bars, hold_everything(bars), 1.0, 0.0, 0.0)
        three = bt.simulate_leveraged(bars, hold_everything(bars), 3.0, 0.0, 0.0)
        self.assertAlmostEqual(one["final"], closes[-2] / closes[0], places=6)
        self.assertGreater(three["final"] - 1.0, 2.9 * (one["final"] - 1.0))

    def test_volatility_decay_makes_3x_worse_than_3x(self):
        """+10%/-10% alternating leaves the index at 0.99 and a 3x fund
        at 0.91. Scaling a trade's total return by 3 would report a small
        LOSS instead, and would overstate every leveraged result."""
        closes = [100.0]
        for k in range(10):
            closes.append(closes[-1] * (1.10 if k % 2 == 0 else 0.90))
        bars = bars_from_closes(closes)
        one = bt.simulate_leveraged(bars, hold_everything(bars), 1.0, 0.0, 0.0)
        three = bt.simulate_leveraged(bars, hold_everything(bars), 3.0, 0.0, 0.0)
        naive = 1.0 + 3.0 * (one["final"] - 1.0)
        self.assertLess(three["final"], one["final"])
        self.assertLess(three["final"], naive)
        # 5 up days and 4 down days are earned (the 10th is the exit
        # bar's open): 1.3^5 x 0.7^4 = 0.89147 at 3x, against
        # 1.1^5 x 0.9^4 = 1.05665 at 1x. Same nine index moves, and the
        # leveraged fund ends 11% down while the index ends 6% up.
        self.assertAlmostEqual(three["final"], 0.89147, places=5)
        self.assertAlmostEqual(one["final"], 1.05666, places=4)

    def test_a_3x_fund_can_be_wiped_out_by_a_move_the_index_survives(self):
        """A 34% single-day index drop is -102% at 3x. Unfloored, the
        product goes negative and the next down day multiplies two
        negatives back into a profit."""
        # Crash must land on a CLOSE while held: entry bar 0, -34% close
        # on bar 1, exit at bar 2's open.
        bars = {"date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "open": [100.0, 100.0, 66.0], "high": [100.0, 100.0, 66.0],
                "low": [100.0, 66.0, 66.0], "close": [100.0, 66.0, 66.0]}
        sim = bt.simulate_leveraged(bars, [{"entry_idx": 0, "exit_idx": 2}],
                                    3.0, 0.0, 0.0)
        self.assertEqual(sim["final"], 0.0)

    def test_zero_equity_is_absorbing(self):
        """Once wiped out, no subsequent rally brings the fund back."""
        bars = {"date": [date(2024, 1, k + 1) for k in range(4)],
                "open": [100.0, 100.0, 66.0, 100.0], "high": [100.0] * 4,
                "low": [66.0] * 4, "close": [100.0, 66.0, 100.0, 150.0]}
        sim = bt.simulate_leveraged(bars, [{"entry_idx": 0, "exit_idx": 3}],
                                    3.0, 0.0, 0.0)
        self.assertEqual(sim["final"], 0.0)


class TestCostsAccrueOnlyWhileHeld(unittest.TestCase):
    def _flat(self, n=253):
        return bars_from_closes([100.0] * n)

    def test_holding_a_flat_index_loses_exactly_the_carry(self):
        """expense + (N-1) x financing, on a flat market, over ~a year."""
        bars = self._flat()
        sim = bt.simulate_leveraged(bars, hold_everything(bars), 3.0, 0.01, 0.04)
        # 1% expense + 2 x 4% financing = 9%/yr. Charged on all 253 bars:
        # both partial days carry cost, which is what a real fund does.
        self.assertAlmostEqual(sim["final"], (1 - 0.09 / 252) ** 253, places=6)

    def test_no_trades_means_no_cost_at_all(self):
        """The strategy is out of the market ~50% of the time, and that
        is where its cost advantage over holding the fund comes from."""
        bars = self._flat()
        sim = bt.simulate_leveraged(bars, [], 3.0, 0.01, 0.04)
        self.assertEqual(sim["final"], 1.0)
        self.assertEqual(sim["days_held"], 0)

    def test_1x_pays_no_financing(self):
        bars = self._flat()
        sim = bt.simulate_leveraged(bars, hold_everything(bars), 1.0, 0.0, 0.04)
        self.assertAlmostEqual(sim["final"], 1.0, places=9)

    def test_higher_financing_never_helps(self):
        bars = bars_from_closes([100.0 * (1.0005 ** k) for k in range(253)])
        trades = hold_everything(bars)
        cheap = bt.simulate_leveraged(bars, trades, 3.0, 0.0091, 0.0)
        dear = bt.simulate_leveraged(bars, trades, 3.0, 0.0091, 0.06)
        self.assertLess(dear["final"], cheap["final"])


class TestPartialDays(unittest.TestCase):
    def test_entry_earns_from_the_open_not_the_previous_close(self):
        """Fills happen at the open, so an overnight gap before the entry
        bar must NOT be credited to the position."""
        # Bar 1 gaps up 10% overnight, then falls back to flat by close.
        bars = {"date": [date(2024, 1, 1), date(2024, 1, 2)],
                "open": [100.0, 110.0], "high": [100.0, 110.0],
                "low": [100.0, 100.0], "close": [100.0, 100.0]}
        sim = bt.simulate_leveraged(bars, [{"entry_idx": 1, "exit_idx": 1}], 3.0, 0.0, 0.0)
        # Entered and exited on the same bar: no exposure, no gap credit.
        self.assertAlmostEqual(sim["final"], 1.0, places=9)

    def test_exit_earns_up_to_the_open_not_the_close(self):
        """The exit fills at the open, so a same-day collapse after the
        open belongs to nobody."""
        bars = {"date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "open": [100.0, 100.0, 110.0], "high": [100.0, 100.0, 110.0],
                "low": [100.0, 100.0, 50.0], "close": [100.0, 100.0, 50.0]}
        sim = bt.simulate_leveraged(bars, [{"entry_idx": 1, "exit_idx": 2}], 3.0, 0.0, 0.0)
        # Held from bar 1's open (100) to bar 2's open (110): +10% index,
        # +30% at 3x. The crash to 50 on bar 2's close is not ours.
        self.assertAlmostEqual(sim["final"], 1.30, places=6)


class TestCurveStats(unittest.TestCase):
    def test_drawdown_is_measured_from_the_running_peak(self):
        stats = bt.curve_stats([1.0, 2.0, 1.0, 1.5], 1.0)
        self.assertAlmostEqual(stats["max_dd"], 50.0)

    def test_cagr_compounds_over_the_year_count(self):
        stats = bt.curve_stats([1.0, 4.0], 2.0)
        self.assertAlmostEqual(stats["cagr"], 100.0, places=6)

    def test_an_empty_curve_does_not_raise(self):
        stats = bt.curve_stats([], 1.0)
        self.assertEqual(stats["final"], 1.0)


class TestArgParsing(unittest.TestCase):
    def test_defaults(self):
        args = bt.parse_args([])
        self.assertEqual(args.symbol, "^GSPC")
        self.assertEqual(args.start_year, 2008)
        self.assertFalse(args.windows)
        self.assertEqual(args.entry_level, 60)
        self.assertEqual(args.exit_level, 65)

    def test_multipliers_and_rates_parse_as_lists(self):
        args = bt.parse_args(["--multipliers", "2,3", "--finance-rates", "0,5"])
        self.assertEqual([float(m) for m in args.multipliers.split(",")], [2.0, 3.0])
        self.assertEqual([float(r) / 100 for r in args.finance_rates.split(",")],
                         [0.0, 0.05])


if __name__ == "__main__":
    unittest.main()
