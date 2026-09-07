"""Unit tests for trading_bot.backtest.rsi20_dip_signals.

The fixture is hand-traced rather than sampled. Its shape is forced by
the strategy: RSI(20) only reaches 60 after a sustained rise is
interrupted by a real pullback, and the same bar must still close above
a 200-bar SMA -- so the series has to climb for 250 bars, drop hard for
six, then recover. The exact arithmetic is spelled out in
TestFixtureTiming and a regression there is a behaviour change, not a
brittle fixture.
"""

import unittest

from trading_bot.backtest import rsi20_dip_signals as dip
from trading_bot.backtest.rsi2_signals import simple_moving_average, wilder_rsi


def make_bars(closes):
    """Opens sit half a point above the previous close, so a fill at the
    open is always distinguishable from a fill at either close."""
    n = len(closes)
    if n == 0:
        return {k: [] for k in ("date", "open", "high", "low", "close")}
    opens = [closes[0]] + [round(closes[i - 1] + 0.5, 4) for i in range(1, n)]
    return {
        "date": list(range(n)),
        "open": opens,
        "high": [max(o, c) + 0.25 for o, c in zip(opens, closes)],
        "low": [min(o, c) - 0.25 for o, c in zip(opens, closes)],
        "close": list(closes),
    }


def rising_then_dip(n_rise=250, rise=1.0, drop=6.0, n_drop=6, rally=4.0, n_rally=24):
    c = [100 + rise * k for k in range(n_rise)]
    for _ in range(n_drop):
        c.append(c[-1] - drop)
    for _ in range(n_rally):
        c.append(c[-1] + rally)
    return c


def rising_then_crash(n_rise=250, rise=1.0, drop=20.0, n_drop=12):
    c = [100 + rise * k for k in range(n_rise)]
    for _ in range(n_drop):
        c.append(c[-1] - drop)
    return c


CLOSES = rising_then_dip()
BARS = make_bars(CLOSES)


class TestValidation(unittest.TestCase):
    def test_equal_levels_are_refused(self):
        """The failure that made the original reconstruction churn."""
        with self.assertRaises(ValueError):
            dip.find_rsi20_dip_trades(BARS, entry_level=60, exit_level=60)

    def test_inverted_levels_are_refused(self):
        with self.assertRaises(ValueError):
            dip.find_rsi20_dip_trades(BARS, entry_level=70, exit_level=60)

    def test_bad_periods_are_refused(self):
        for kwargs in ({"rsi_period": 0}, {"ma_period": 0}, {"offset_bars": -1}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                dip.find_rsi20_dip_trades(BARS, **kwargs)

    def test_empty_input(self):
        self.assertEqual(dip.find_rsi20_dip_trades(make_bars([])), [])

    def test_too_little_history_yields_no_trades(self):
        self.assertEqual(dip.find_rsi20_dip_trades(make_bars([100.0] * 50)), [])


class TestShiftedTrendMa(unittest.TestCase):
    def test_none_before_the_offset_can_see(self):
        out = dip.shifted_trend_ma([float(i) for i in range(20)], ma_period=5, offset_bars=3)
        self.assertEqual(out[:3], [None, None, None])

    def test_equals_the_sma_from_offset_bars_ago(self):
        closes = [float(i) for i in range(40)]
        sma = simple_moving_average(closes, 5)
        out = dip.shifted_trend_ma(closes, ma_period=5, offset_bars=3)
        self.assertEqual(out[20], sma[17])
        self.assertEqual(out[39], sma[36])

    def test_zero_offset_is_the_plain_sma(self):
        closes = [float(i) for i in range(40)]
        self.assertEqual(dip.shifted_trend_ma(closes, 5, 0), simple_moving_average(closes, 5))


class TestFixtureTiming(unittest.TestCase):
    """RSI(20) on this fixture reads 60.67 at bar 251 and 50.05 at bar
    252, so the crossing DOWN through 60 completes on bar 252 and the
    fill belongs on bar 253's open. It reads 63.49 at bar 263 and 65.97
    at bar 264, so the crossing UP through 65 completes on 264 and that
    fill belongs on 265's open."""

    def setUp(self):
        self.trades = dip.find_rsi20_dip_trades(BARS)
        self.rsi = wilder_rsi(CLOSES, 20)

    def test_the_fixture_produces_exactly_one_trade(self):
        self.assertEqual(len(self.trades), 1)

    def test_rsi_brackets_the_entry_crossing(self):
        self.assertGreaterEqual(self.rsi[251], 60)
        self.assertLess(self.rsi[252], 60)

    def test_entry_fills_at_the_open_after_the_signal_bar(self):
        t = self.trades[0]
        self.assertEqual(t["entry_idx"], 253)
        self.assertEqual(t["entry_price"], BARS["open"][253])

    def test_rsi_brackets_the_exit_crossing(self):
        self.assertLessEqual(self.rsi[263], 65)
        self.assertGreater(self.rsi[264], 65)

    def test_exit_fills_at_the_open_after_the_signal_bar(self):
        t = self.trades[0]
        self.assertEqual(t["exit_idx"], 265)
        self.assertEqual(t["exit_price"], BARS["open"][265])
        self.assertEqual(t["reason"], dip.REASON_RSI_EXIT)

    def test_pct_and_points_agree_with_the_fills(self):
        t = self.trades[0]
        self.assertAlmostEqual(t["points"], t["exit_price"] - t["entry_price"])
        self.assertAlmostEqual(t["pct"], t["exit_price"] / t["entry_price"] - 1)
        self.assertEqual(t["bars_held"], 265 - 253)

    def test_a_run_of_sub_level_bars_gives_one_entry_not_four(self):
        """Bars 252-255 all sit below 60. A state test rather than a
        crossing test would open four trades here."""
        below = [i for i in range(252, 256) if self.rsi[i] < 60]
        self.assertEqual(len(below), 4)
        self.assertEqual(len(self.trades), 1)


class TestTrendBreakExit(unittest.TestCase):
    def setUp(self):
        self.closes = rising_then_crash()
        self.bars = make_bars(self.closes)
        self.trades = dip.find_rsi20_dip_trades(self.bars)

    def test_a_crash_through_the_ma_closes_the_trade(self):
        self.assertTrue(self.trades)
        self.assertEqual(self.trades[0]["reason"], dip.REASON_TREND_BREAK)

    def test_the_exit_bar_is_below_the_unshifted_ma(self):
        """The exit uses the UNSHIFTED MA deliberately -- in an uptrend it
        sits above the shifted one, so it fires earlier."""
        t = self.trades[0]
        sma = simple_moving_average(self.closes, 200)
        signal_bar = t["exit_idx"] - 1
        self.assertLess(self.closes[signal_bar], sma[signal_bar])


class TestPositionDiscipline(unittest.TestCase):
    def test_trades_never_overlap(self):
        closes = rising_then_dip(n_rally=8) + rising_then_dip(n_rise=40)[:0]
        trades = dip.find_rsi20_dip_trades(make_bars(closes))
        for earlier, later in zip(trades, trades[1:]):
            self.assertLessEqual(earlier["exit_idx"], later["entry_idx"])

    def test_entry_always_precedes_its_exit(self):
        for t in dip.find_rsi20_dip_trades(BARS):
            self.assertLess(t["entry_idx"], t["exit_idx"])

    def test_an_open_trade_is_marked_to_the_final_close(self):
        truncated = {k: v[:258] for k, v in BARS.items()}
        trades = dip.find_rsi20_dip_trades(truncated)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["reason"], dip.REASON_END_OF_DATA)
        self.assertEqual(t["exit_price"], truncated["close"][-1])


class TestEquityCurve(unittest.TestCase):
    def test_curve_covers_every_bar_from_the_start_index(self):
        trades = dip.find_rsi20_dip_trades(BARS)
        curve = dip.equity_curve(BARS, trades, start_idx=200)
        self.assertEqual(len(curve), len(BARS["close"]) - 200)
        self.assertEqual(curve[0]["equity"], 1.0)

    def test_a_winning_trade_compounds_into_the_final_equity(self):
        trades = dip.find_rsi20_dip_trades(BARS)
        curve = dip.equity_curve(BARS, trades)
        self.assertAlmostEqual(curve[-1]["equity"], 1.0 + trades[0]["pct"], places=6)

    def test_open_positions_are_marked_to_market(self):
        """A closed-trade-only curve would report equity 1.0 throughout
        the hold and hide the drawdown the account actually took."""
        trades = dip.find_rsi20_dip_trades(BARS)
        curve = dip.equity_curve(BARS, trades)
        held = [p for p in curve if p["in_market"]]
        self.assertTrue(held)
        self.assertTrue(any(p["equity"] != 1.0 for p in held))

    def test_drawdown_sees_the_dip_inside_an_open_trade(self):
        trades = dip.find_rsi20_dip_trades(BARS)
        curve = dip.equity_curve(BARS, trades)
        self.assertGreater(dip.max_drawdown_pct(curve), 0.0)

    def test_drawdown_of_a_monotonic_curve_is_zero(self):
        curve = [{"equity": 1.0 + i * 0.01, "in_market": False} for i in range(10)]
        self.assertEqual(dip.max_drawdown_pct(curve), 0.0)

    def test_drawdown_is_measured_from_the_peak(self):
        curve = [{"equity": e, "in_market": False} for e in (1.0, 2.0, 1.0, 1.5)]
        self.assertAlmostEqual(dip.max_drawdown_pct(curve), 50.0)


class TestAsymmetryMatters(unittest.TestCase):
    """The finding the module exists to record: a wider band trades less
    and holds longer. Pinned so a future edit cannot quietly restore the
    churning behaviour."""

    def test_a_wider_band_holds_longer(self):
        narrow = dip.find_rsi20_dip_trades(BARS, entry_level=60, exit_level=62)
        wide = dip.find_rsi20_dip_trades(BARS, entry_level=60, exit_level=75)
        self.assertTrue(narrow and wide)
        self.assertLess(narrow[0]["bars_held"], wide[0]["bars_held"])


if __name__ == "__main__":
    unittest.main()
