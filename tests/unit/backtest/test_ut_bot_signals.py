"""Unit tests for trading_bot.backtest.ut_bot_signals.

The lifecycle fixture in TestFindUtBotLongTrades was hand-traced against
the module's own docstring derivation of the Pine Script before being
encoded here (see that test's docstring) -- a regression here is a real
behavior change, not a fragile fixture.
"""

import unittest

from trading_bot.backtest import ut_bot_signals as ut


class TestTrueRange(unittest.TestCase):
    def test_first_bar_has_no_prior_close(self):
        self.assertEqual(ut.true_range(10, 8, None), 2)

    def test_gap_up_widens_range(self):
        # high-low=3, |high-prevClose|=4, |low-prevClose|=1 -> max is 4
        self.assertEqual(ut.true_range(12, 9, 8), 4)

    def test_gap_down_widens_range(self):
        # high-low=3, |high-prevClose|=2, |low-prevClose|=5 -> max is 5
        self.assertEqual(ut.true_range(9, 6, 11), 5)


class TestWilderAtr(unittest.TestCase):
    HIGHS = [10, 12, 9]
    LOWS = [8, 9, 5]
    CLOSES = [9, 11, 6]

    def test_period_one_equals_true_range_exactly(self):
        # TR: bar0=2 (no prior close), bar1=max(3,3,0)=3, bar2=max(4,2,6)=6
        self.assertEqual(ut.wilder_atr(self.HIGHS, self.LOWS, self.CLOSES, 1), [2.0, 3.0, 6.0])

    def test_period_two_seeds_then_recurses(self):
        # seed = mean(TR[0:2]) = (2+3)/2 = 2.5, held for bars 0-1;
        # bar2 = (2.5*(2-1) + 6) / 2 = 4.25
        self.assertEqual(ut.wilder_atr(self.HIGHS, self.LOWS, self.CLOSES, 2), [2.5, 2.5, 4.25])

    def test_empty_input(self):
        self.assertEqual(ut.wilder_atr([], [], [], 1), [])


class TestAtrTrailingStop(unittest.TestCase):
    def test_matches_hand_traced_sequence(self):
        # See TestFindUtBotLongTrades for the full derivation of this
        # fixture; values here were independently hand-computed bar by
        # bar from the module docstring's branch logic.
        h = [10, 11, 9, 8, 12, 11.5, 10, 8]
        l = [8, 9, 6, 6, 7, 9, 7, 5]
        c = [9, 10, 7, 7.5, 11, 9.5, 7.2, 5.5]
        stop = ut.atr_trailing_stop(h, l, c, key_value=1.0, atr_period=1)
        self.assertEqual(stop, [7.0, 8.0, 11.0, 9.5, 6.0, 7.0, 7.0, 8.5])

    def test_first_bar_flips_up_when_price_positive(self):
        # nz(stop[-1], 0) = 0 for bar 0; any positive close satisfies the
        # "src > prevStop" branch, giving stop = src - nLoss.
        stop = ut.atr_trailing_stop([10], [8], [9], key_value=1.0, atr_period=1)
        self.assertEqual(stop, [9 - 2])  # nLoss = 1.0 * TR(10,8,None) = 2


class TestCrossoverCrossunder(unittest.TestCase):
    def test_crossover_fires_exactly_on_the_crossing_bar(self):
        series = [5, 6, 8]
        reference = [7, 7, 7]
        self.assertEqual(ut.crossover(series, reference), [False, False, True])

    def test_crossunder_fires_exactly_on_the_crossing_bar(self):
        series = [8, 7, 5]
        reference = [6, 6, 6]
        self.assertEqual(ut.crossunder(series, reference), [False, False, True])

    def test_first_bar_is_never_a_cross(self):
        self.assertEqual(ut.crossover([100], [1]), [False])
        self.assertEqual(ut.crossunder([1], [100]), [False])


def _bars(h, l, c):
    return {"high": h, "low": l, "close": c, "date": list(range(len(c)))}


class TestFindUtBotLongTrades(unittest.TestCase):
    """Hand-traced 8-bar fixture (key_value=1.0, atr_period=1 for
    arithmetic simplicity -- the crossover mechanics being tested don't
    depend on the specific default parameter values):

    idx  high  low   close
    0    10    8     9
    1    11    9     10
    2    9     6     7
    3    8     6     7.5
    4    12    7     11     <- stop dips to 6 the bar before, close(11)
                               crosses above it -> BUY here
    5    11.5  9     9.5
    6    10    7     7.2
    7    8     5     5.5    <- stop holds near 7-8.5, close crosses
                               below it -> SELL here

    Full bar-by-bar stop/ATR derivation lives in
    TestAtrTrailingStop.test_matches_hand_traced_sequence and this
    module's docstring; this test only checks the resulting trade.
    """

    H = [10, 11, 9, 8, 12, 11.5, 10, 8]
    L = [8, 9, 6, 6, 7, 9, 7, 5]
    C = [9, 10, 7, 7.5, 11, 9.5, 7.2, 5.5]

    def test_full_round_trip(self):
        trades = ut.find_ut_bot_long_trades(_bars(self.H, self.L, self.C), key_value=1.0, atr_period=1)
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade["entry_idx"], 4)
        self.assertEqual(trade["entry_price"], 11)
        self.assertAlmostEqual(trade["stop_at_entry"], 6.0)
        self.assertEqual(trade["exit_idx"], 7)
        self.assertEqual(trade["exit_price"], 5.5)
        self.assertEqual(trade["reason"], "sell_signal")

    def test_still_open_position_closes_at_end_of_data(self):
        # Truncate right at the entry bar -- no sell signal ever arrives.
        truncated = _bars(self.H[:5], self.L[:5], self.C[:5])
        trades = ut.find_ut_bot_long_trades(truncated, key_value=1.0, atr_period=1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["reason"], "end_of_data")
        self.assertEqual(trades[0]["exit_idx"], 4)

    def test_sell_signal_ignored_while_flat(self):
        # A sell crossunder happens at idx 2 in the full fixture while no
        # position is open yet (entry is later, at idx 4) -- must not
        # produce a phantom trade.
        trades = ut.find_ut_bot_long_trades(_bars(self.H, self.L, self.C), key_value=1.0, atr_period=1)
        self.assertTrue(all(t["entry_idx"] != 2 for t in trades))

    def test_empty_bars_returns_no_trades(self):
        self.assertEqual(ut.find_ut_bot_long_trades(_bars([], [], [])), [])

    def test_no_regime_filter_by_default(self):
        # vol_filter_lookback defaults to None -- identical to
        # test_full_round_trip's result even though the parameter now exists.
        trades = ut.find_ut_bot_long_trades(_bars(self.H, self.L, self.C), key_value=1.0, atr_period=1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["entry_idx"], 4)

    def test_regime_filter_blocks_every_entry_without_enough_history(self):
        # lookback (1000) far exceeds the 8-bar fixture, so
        # volatility_regime_ok is False for every bar (no trailing average
        # ever exists) -- the entry at idx 4 must be suppressed entirely.
        trades = ut.find_ut_bot_long_trades(
            _bars(self.H, self.L, self.C), key_value=1.0, atr_period=1,
            vol_filter_lookback=1000, vol_filter_atr_period=1,
        )
        self.assertEqual(trades, [])

    def test_regime_filter_lets_entry_through_when_volatility_is_calm(self):
        # lookback=2 gives a valid trailing average from bar 1 onward, and
        # this fixture's volatility never spikes relative to its own
        # recent history (verified: volatility_regime_ok is only False on
        # bar 0, True everywhere else) -- the idx 4 entry still fires.
        trades = ut.find_ut_bot_long_trades(
            _bars(self.H, self.L, self.C), key_value=1.0, atr_period=1,
            vol_filter_lookback=2, vol_filter_max_ratio=1.5, vol_filter_atr_period=1,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["entry_idx"], 4)


class TestTrailingAverage(unittest.TestCase):
    def test_none_until_lookback_bars_exist(self):
        self.assertEqual(ut._trailing_average([1, 2, 3, 4, 5], lookback=3), [None, None, 2.0, 3.0, 4.0])


class TestVolatilityRegimeOk(unittest.TestCase):
    def test_detects_a_volatility_spike_relative_to_trailing_history(self):
        # closes held constant at 100 so true_range is driven entirely by
        # each bar's own high-low (no gap component): TR = [2, 2, 2, 20, 2]
        # at atr_period=1 (TR passes straight through, see TestWilderAtr).
        # With lookback=3, max_ratio=1.5: bars 0-1 have no trailing average
        # yet (False); bar 2's average (0.02) matches its own TR (0.02,
        # True); bar 3 spikes to TR=20 (atr_pct=0.20) against a trailing
        # average of 0.08 -- 0.20 > 0.08*1.5=0.12, so False; bar 4 is back
        # to normal (0.02 <= 0.08*1.5) even though the average is still
        # spike-elevated, so True again.
        highs = [101, 101, 101, 110, 101]
        lows = [99, 99, 99, 90, 99]
        closes = [100, 100, 100, 100, 100]
        result = ut.volatility_regime_ok(highs, lows, closes, atr_period=1, lookback=3, max_ratio=1.5)
        self.assertEqual(result, [False, False, True, False, True])


class TestFindUtBotConfirmedTrades(unittest.TestCase):
    """User-specified variant (not part of the original Pine Script):
    entries need a cross AND a sloping stop in the trade's direction,
    confirmed only if the VERY NEXT bar still clears the line; exits stay
    the original unconfirmed crossunder/crossover. Long+short. Each
    fixture below was constructed and verified by hand (bar-by-bar
    stop/slope/cross printout) before being encoded as a test."""

    def test_confirmed_long_round_trip(self):
        # Arms at idx 5 (buy crossover with stop numerically rising --
        # the rare case where the "trend flip" branch's src-nLoss still
        # exceeds the prior stop); confirmed at idx 6 (close still above
        # the line); exits at idx 7 on the plain sell crossunder.
        h = [10.0, 9.6, 9.2, 9.0, 8.9, 9.05, 9.15, 8.9]
        l = [9.0, 8.8, 8.4, 8.2, 8.1, 8.90, 8.95, 8.6]
        c = [9.5, 9.0, 8.6, 8.3, 8.15, 9.00, 9.10, 8.75]
        bars = _bars(h, l, c)

        trades = ut.find_ut_bot_confirmed_trades(bars, key_value=0.3, atr_period=1)

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade["side"], "long")
        self.assertEqual(trade["entry_idx"], 6)
        self.assertEqual(trade["entry_price"], 9.10)
        self.assertAlmostEqual(trade["stop_at_entry"], 9.04)
        self.assertEqual(trade["exit_idx"], 7)
        self.assertEqual(trade["exit_price"], 8.75)
        self.assertEqual(trade["reason"], "sell_signal")

    def test_confirmed_short_round_trip(self):
        # Arms at idx 7 (sell crossunder with stop numerically falling);
        # confirmed at idx 8; exits at idx 11 on the plain buy crossover.
        h = [10.0, 9.6, 9.2, 8.9, 9.3, 9.6, 9.7, 9.5, 9.2, 8.8, 8.5, 8.7, 8.5]
        l = [9.0, 8.8, 8.4, 8.1, 8.6, 9.1, 9.3, 8.9, 8.6, 8.2, 7.9, 8.2, 8.0]
        c = [9.5, 9.0, 8.6, 8.3, 9.1, 9.5, 9.65, 9.0, 8.7, 8.4, 8.0, 8.6, 8.1]
        bars = _bars(h, l, c)

        trades = ut.find_ut_bot_confirmed_trades(bars, key_value=0.5, atr_period=1)

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade["side"], "short")
        self.assertEqual(trade["entry_idx"], 8)
        self.assertEqual(trade["entry_price"], 8.7)
        self.assertAlmostEqual(trade["stop_at_entry"], 9.0)
        self.assertEqual(trade["exit_idx"], 11)
        self.assertEqual(trade["exit_price"], 8.6)
        self.assertEqual(trade["reason"], "buy_signal")

    def test_failed_confirmation_produces_no_trade_and_does_not_linger(self):
        # idx 5 arms a long (buy crossover, stop rising) but idx 6's
        # close falls back below the line -- confirmation fails, and the
        # arm is discarded rather than re-checked at idx 7. idx 6 ALSO
        # happens to arm a short (sell crossunder, stop falling), which
        # idx 7 also fails to confirm. Net result: zero trades from
        # either attempt.
        h = [10.0, 9.6, 9.2, 9.0, 8.9, 9.05, 8.6, 8.9]
        l = [9.0, 8.8, 8.4, 8.2, 8.1, 8.90, 8.2, 8.5]
        c = [9.5, 9.0, 8.6, 8.3, 8.15, 9.00, 8.4, 8.7]
        bars = _bars(h, l, c)

        trades = ut.find_ut_bot_confirmed_trades(bars, key_value=0.3, atr_period=1)
        self.assertEqual(trades, [])

    def test_cross_with_wrong_slope_never_arms(self):
        # idx 1 in the short fixture is a sell crossunder while the stop
        # is RISING (not falling) -- must not arm a short.
        h = [10.0, 9.6, 9.2, 8.9]
        l = [9.0, 8.8, 8.4, 8.1]
        c = [9.5, 9.0, 8.6, 8.3]
        bars = _bars(h, l, c)
        trades = ut.find_ut_bot_confirmed_trades(bars, key_value=0.5, atr_period=1)
        self.assertEqual(trades, [])

    def test_too_few_bars_returns_no_trades(self):
        self.assertEqual(ut.find_ut_bot_confirmed_trades(_bars([10], [8], [9])), [])


if __name__ == "__main__":
    unittest.main()
