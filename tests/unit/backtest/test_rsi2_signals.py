"""Unit tests for trading_bot.backtest.rsi2_signals.

The shared fixture below is hand-traced, not sampled from real data. Its
shape is forced by the strategy itself: RSI(2) can only reach single
digits when a drop dwarfs the recent daily gains, while the trend filter
demands the same bar still close above its SMA -- so a synthetic series
has to rise steadily for a long stretch and then drop sharply but not
too far. The arithmetic that makes RSI(2) = 7.69 and SMA(21) = 91.28 on
the signal bar is spelled out in TestRsi2EntrySignals; a regression there
is a real behavior change, not a fragile fixture.
"""

import unittest

from trading_bot.backtest import rsi2_signals as rsi2


def make_bars(closes, opens=None, highs=None, lows=None):
    """Bars with sensible defaults: open at the previous close, and a
    high/low that bracket the bar without triggering anything. Dates are
    plain ints -- the module treats them as opaque labels."""
    n = len(closes)
    opens = list(closes) if opens is None else opens
    highs = [max(o, c) for o, c in zip(opens, closes)] if highs is None else highs
    lows = [min(o, c) for o, c in zip(opens, closes)] if lows is None else lows
    return {"date": list(range(n)), "open": opens, "high": highs, "low": lows, "close": closes}


# closes 80..100 inclusive (idx 0-20), then two sharp down bars.
RISING = list(range(80, 101))
SIGNAL_CLOSES = RISING + [96, 92]  # idx 21, 22
SIGNAL_IDX = 22
SMA_PERIOD = 21


class TestWilderRsi(unittest.TestCase):
    def test_hand_computed_two_period(self):
        # changes: +1, +1, -12. Seed at idx2: avg_gain=(1+1)/2=1,
        # avg_loss=0 -> RSI 100. idx3: avg_gain=(1*1+0)/2=0.5,
        # avg_loss=(0*1+12)/2=6 -> rs=1/12 -> RSI=100-100/(1+1/12)=7.69.
        out = rsi2.wilder_rsi([100, 101, 102, 90], 2)
        self.assertEqual(out[:2], [None, None])
        self.assertEqual(out[2], 100.0)
        self.assertAlmostEqual(out[3], 7.6923, places=3)

    def test_shorter_than_period_is_all_none(self):
        self.assertEqual(rsi2.wilder_rsi([100, 101], 2), [None, None])
        self.assertEqual(rsi2.wilder_rsi([], 2), [])

    def test_monotone_rise_is_one_hundred(self):
        out = rsi2.wilder_rsi([10, 11, 12, 13, 14], 2)
        self.assertEqual(out[2:], [100.0, 100.0, 100.0])

    def test_monotone_fall_is_zero(self):
        out = rsi2.wilder_rsi([14, 13, 12, 11, 10], 2)
        self.assertEqual(out[2:], [0.0, 0.0, 0.0])

    def test_flat_series_is_fifty(self):
        # Neither gains nor losses: the 0/0 convention, not a crash.
        self.assertEqual(rsi2.wilder_rsi([5, 5, 5, 5], 2)[2:], [50.0, 50.0])


class TestSimpleMovingAverage(unittest.TestCase):
    def test_none_until_window_is_full_then_trails(self):
        self.assertEqual(rsi2.simple_moving_average([1, 2, 3, 4], 3), [None, None, 2.0, 3.0])

    def test_includes_the_current_bar(self):
        # Deliberately NOT shifted -- see the module docstring on SMA
        # alignment. mean(2,3,4) uses bar 3's own value.
        self.assertEqual(rsi2.simple_moving_average([1, 2, 3, 4], 3)[3], 3.0)


class TestRsi2EntrySignals(unittest.TestCase):
    def test_fires_exactly_once_on_the_crossing_bar(self):
        # RSI(2) over SIGNAL_CLOSES: a 20-bar run of +1 leaves avg_gain=1,
        # avg_loss=0 (RSI 100). idx21 (-4): avg_gain=0.5, avg_loss=2 ->
        # RSI 20. idx22 (-4): avg_gain=0.25, avg_loss=3 -> RSI 7.69.
        # So 20 -> 7.69 is the downward crossing of the 10 level.
        rsi = rsi2.wilder_rsi(SIGNAL_CLOSES, 2)
        self.assertAlmostEqual(rsi[21], 20.0, places=6)
        self.assertAlmostEqual(rsi[SIGNAL_IDX], 7.6923, places=3)
        # SMA(21) at idx22 = mean(closes[2:23]) = (82..100) + 96 + 92
        # = 1729 + 188 = 1917 / 21 = 91.29, and the close of 92 clears it.
        sma = rsi2.simple_moving_average(SIGNAL_CLOSES, SMA_PERIOD)
        self.assertAlmostEqual(sma[SIGNAL_IDX], 91.2857, places=3)

        signals = rsi2.rsi2_entry_signals(SIGNAL_CLOSES, 2, 10.0, SMA_PERIOD)
        self.assertEqual([i for i, s in enumerate(signals) if s], [SIGNAL_IDX])

    def test_staying_below_the_level_does_not_re_signal(self):
        # A third down bar keeps RSI under 10 but is not a fresh crossing.
        closes = SIGNAL_CLOSES + [91]
        signals = rsi2.rsi2_entry_signals(closes, 2, 10.0, SMA_PERIOD)
        self.assertTrue(signals[SIGNAL_IDX])
        self.assertFalse(signals[SIGNAL_IDX + 1])

    def test_close_below_sma_blocks_the_signal(self):
        # Same crossing, but a deeper drop puts the close under its SMA.
        closes = RISING + [96, 70]
        rsi = rsi2.wilder_rsi(closes, 2)
        self.assertLess(rsi[SIGNAL_IDX], 10.0)
        signals = rsi2.rsi2_entry_signals(closes, 2, 10.0, SMA_PERIOD)
        self.assertEqual([i for i, s in enumerate(signals) if s], [])


class TestFindRsi2LongTrades(unittest.TestCase):
    """All fixtures extend SIGNAL_CLOSES, so the single entry always fills
    at the open of bar 23 (SIGNAL_IDX + 1)."""

    ENTRY_IDX = SIGNAL_IDX + 1

    def _run(self, closes, opens=None, highs=None, lows=None, **kwargs):
        kwargs.setdefault("sma_period", SMA_PERIOD)
        kwargs.setdefault("min_hold_days", 1)
        return rsi2.find_rsi2_long_trades(make_bars(closes, opens, highs, lows), **kwargs)

    def _extend(self, tail_closes, tail_opens):
        """SIGNAL_CLOSES plus a controlled tail; returns (closes, opens)."""
        closes = SIGNAL_CLOSES + tail_closes
        opens = list(SIGNAL_CLOSES) + tail_opens
        return closes, opens

    def test_entry_fills_at_the_next_bars_open(self):
        closes, opens = self._extend([95], [93])
        trades = self._run(closes, opens, stop_points=None)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["entry_idx"], self.ENTRY_IDX)
        # Not the signal bar's close of 92, and not bar 23's close of 95.
        self.assertEqual(trades[0]["entry_price"], 93)

    def test_first_profitable_close_exits_on_the_entry_bar_when_undelayed(self):
        closes, opens = self._extend([95], [93])
        trades = self._run(closes, opens, stop_points=None, min_hold_days=1)
        self.assertEqual(trades[0]["exit_idx"], self.ENTRY_IDX)
        self.assertEqual(trades[0]["reason"], "first_profitable_close")
        self.assertEqual(trades[0]["points"], 2)
        self.assertEqual(trades[0]["bars_held"], 1)

    def test_day_delay_holds_through_profitable_closes(self):
        # Profitable on all three bars; a delay of 3 must ignore the first
        # two and exit on the third (the entry bar counts as day 1).
        closes, opens = self._extend([95, 96, 97], [93, 93, 93])
        trades = self._run(closes, opens, stop_points=None, min_hold_days=3)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_idx"], self.ENTRY_IDX + 2)
        self.assertEqual(trades[0]["bars_held"], 3)
        self.assertEqual(trades[0]["points"], 4)

    def test_delay_expiry_still_requires_profit(self):
        # Delay satisfied on bar 24, but the close is under water there;
        # the exit waits for bar 25, which is not.
        closes, opens = self._extend([94, 92, 96], [93, 93, 93])
        trades = self._run(closes, opens, stop_points=None, min_hold_days=2)
        self.assertEqual(trades[0]["exit_idx"], self.ENTRY_IDX + 2)
        self.assertEqual(trades[0]["points"], 3)

    def test_exit_timing_next_open_defers_the_fill(self):
        closes, opens = self._extend([95, 99], [93, 94])
        trades = self._run(closes, opens, stop_points=None, exit_timing="next_open")
        self.assertEqual(trades[0]["exit_idx"], self.ENTRY_IDX + 1)
        # Bar 24's open of 94, not bar 23's close of 95 or bar 24's 99.
        self.assertEqual(trades[0]["exit_price"], 94)

    def test_stop_fills_intrabar_at_the_stop_price(self):
        closes, opens = self._extend([94, 90], [93, 93])
        lows = [min(o, c) for o, c in zip(opens, closes)]
        lows[self.ENTRY_IDX + 1] = 87  # pierces a stop at 93 - 5 = 88
        trades = self._run(closes, opens, lows=lows, stop_points=5, min_hold_days=99)
        self.assertEqual(trades[0]["reason"], "stop_loss")
        self.assertEqual(trades[0]["exit_price"], 88)
        self.assertEqual(trades[0]["points"], -5)

    def test_gap_below_the_stop_fills_at_the_open(self):
        # There is no liquidity between the prior close and a gapped
        # open, so a stop at 88 cannot fill at 88 -- it fills at 85.
        closes, opens = self._extend([94, 84], [93, 85])
        trades = self._run(closes, opens, stop_points=5, min_hold_days=99)
        self.assertEqual(trades[0]["reason"], "stop_loss")
        self.assertEqual(trades[0]["exit_price"], 85)
        self.assertEqual(trades[0]["points"], -8)

    def test_stop_wins_a_tie_with_the_exit_signal(self):
        # Bar 24 both pierces the stop and closes in profit. A daily bar
        # cannot tell us the low came after the close, so the resting
        # stop is the honest fill.
        closes, opens = self._extend([94, 99], [93, 93])
        lows = [min(o, c) for o, c in zip(opens, closes)]
        lows[self.ENTRY_IDX + 1] = 87
        trades = self._run(closes, opens, lows=lows, stop_points=5, min_hold_days=2)
        self.assertEqual(trades[0]["reason"], "stop_loss")

    def test_rsi_exit_mode_uses_the_overbought_level(self):
        # Two strong up bars drive RSI(2) back above 70; the exit is
        # signalled at that close and fills on the following open.
        closes, opens = self._extend([96, 104, 105], [93, 100, 101])
        trades = self._run(
            closes, opens, stop_points=None,
            exit_mode=rsi2.EXIT_MODE_RSI, exit_timing="next_open",
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["reason"], "rsi_exit")
        rsi = rsi2.wilder_rsi(closes, 2)
        self.assertGreater(rsi[trades[0]["exit_idx"] - 1], 70.0)

    def test_mae_records_the_worst_excursion_while_held(self):
        closes, opens = self._extend([94, 92, 96], [93, 93, 93])
        lows = [min(o, c) for o, c in zip(opens, closes)]
        lows[self.ENTRY_IDX + 1] = 89  # 4 points under the 93 entry
        trades = self._run(closes, opens, lows=lows, stop_points=None, min_hold_days=3)
        self.assertEqual(trades[0]["mae_points"], 4)

    def test_mae_excludes_the_low_of_a_bar_exited_at_its_open(self):
        closes, opens = self._extend([95, 99], [93, 94])
        lows = [min(o, c) for o, c in zip(opens, closes)]
        lows[self.ENTRY_IDX + 1] = 80  # after we are already out at 94
        trades = self._run(closes, opens, lows=lows, stop_points=None, exit_timing="next_open")
        self.assertEqual(trades[0]["mae_points"], 0.0)

    def test_unstopped_first_profitable_close_can_only_produce_winners(self):
        """Not an edge -- a tautology, and the reason this strategy's win
        rate is not evidence of anything. With no stop, the only exit
        that can fire is a profitable one, so every closed trade wins by
        construction however deep it went first."""
        closes, opens = self._extend([80, 70, 60, 50, 96], [93, 93, 93, 93, 93])
        trades = self._run(closes, opens, stop_points=None, min_hold_days=1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["reason"], "first_profitable_close")
        self.assertGreater(trades[0]["points"], 0)
        # The risk it took is invisible in the P&L and visible only here.
        self.assertEqual(trades[0]["mae_points"], 43)

    def test_open_position_at_end_of_data_is_closed_at_the_last_close(self):
        closes, opens = self._extend([90, 88], [93, 91])
        trades = self._run(closes, opens, stop_points=None, min_hold_days=99)
        self.assertEqual(trades[0]["reason"], "end_of_data")
        self.assertEqual(trades[0]["exit_idx"], len(closes) - 1)
        self.assertEqual(trades[0]["exit_price"], 88)

    def test_trades_never_overlap_on_a_long_noisy_series(self):
        """The one-position-at-a-time invariant (the video is explicit
        that an in-trade RSI dip is ignored, not stacked), asserted over a
        deterministic pseudo-random walk long enough to generate dozens of
        signals rather than a hand-picked one."""
        closes = []
        price = 1000.0
        state = 12345
        for _ in range(3000):
            state = (1103515245 * state + 12345) % (2**31)
            price *= 1 + ((state / (2**31)) - 0.49) * 0.02
            closes.append(price)
        trades = rsi2.find_rsi2_long_trades(make_bars(closes), sma_period=200, min_hold_days=5)
        self.assertGreater(len(trades), 10)
        for earlier, later in zip(trades, trades[1:]):
            self.assertLessEqual(earlier["exit_idx"], later["entry_idx"])
            self.assertLessEqual(earlier["entry_idx"], earlier["exit_idx"])

    def test_rejects_contradictory_stop_configuration(self):
        with self.assertRaises(ValueError):
            rsi2.find_rsi2_long_trades(make_bars(SIGNAL_CLOSES), stop_points=200, stop_pct=3)

    def test_rejects_unknown_modes(self):
        with self.assertRaises(ValueError):
            rsi2.find_rsi2_long_trades(make_bars(SIGNAL_CLOSES), exit_mode="nope")
        with self.assertRaises(ValueError):
            rsi2.find_rsi2_long_trades(make_bars(SIGNAL_CLOSES), exit_timing="nope")

    def test_empty_input(self):
        self.assertEqual(rsi2.find_rsi2_long_trades(make_bars([])), [])


if __name__ == "__main__":
    unittest.main()
