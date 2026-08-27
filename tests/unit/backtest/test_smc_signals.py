"""Unit tests for trading_bot.backtest.smc_signals.

The integration test (TestFindSmcLongTrades) uses a hand-traced synthetic
bar sequence -- every swing high/low, order block, FVG, entry, and exit in
it was verified by hand against the algorithm before being encoded here
(see the docstring on test_full_trade_lifecycle for the trace), so a
regression here is a real behavior change, not a fragile fixture.
"""

import unittest
from datetime import datetime

from trading_bot.backtest import smc_signals as smc


class TestCandleDirection(unittest.TestCase):
    def test_bullish_when_close_above_open(self):
        self.assertEqual(smc.candle_direction(10, 12), "bullish")

    def test_bearish_when_close_below_open(self):
        self.assertEqual(smc.candle_direction(12, 10), "bearish")

    def test_doji_counts_as_bullish(self):
        self.assertEqual(smc.candle_direction(10, 10), "bullish")


class TestFindSwingHighs(unittest.TestCase):
    def test_single_swing_high_detected(self):
        highs = [10, 10, 12, 10, 10]
        self.assertEqual(smc.find_swing_highs(highs), [(2, 12)])

    def test_monotonic_series_has_no_swing_high(self):
        highs = [10, 11, 12, 13, 14]
        self.assertEqual(smc.find_swing_highs(highs), [])

    def test_equal_neighbor_does_not_count(self):
        highs = [10, 11, 12, 11, 10]  # 12 > both sides -- should count
        self.assertEqual(smc.find_swing_highs(highs), [(2, 12)])
        tie = [10, 12, 12, 12, 10]  # no strict max -- should not count
        self.assertEqual(smc.find_swing_highs(tie), [])

    def test_too_few_bars_returns_empty(self):
        self.assertEqual(smc.find_swing_highs([10, 11, 12, 11]), [])


class TestFindSwingLows(unittest.TestCase):
    def test_single_swing_low_detected(self):
        lows = [10, 8, 3, 8, 10]
        self.assertEqual(smc.find_swing_lows(lows), [(2, 3)])

    def test_monotonic_series_has_no_swing_low(self):
        self.assertEqual(smc.find_swing_lows([10, 9, 8, 7, 6]), [])


class TestHasFvg(unittest.TestCase):
    def test_gap_present(self):
        highs = [11, 17, 20]
        lows = [8, 12, 16]
        self.assertTrue(smc.has_fvg(highs, lows, 0))  # highs[0]=11 < lows[2]=16

    def test_no_gap_when_overlapping(self):
        highs = [15, 17, 20]
        lows = [8, 12, 14]
        self.assertFalse(smc.has_fvg(highs, lows, 0))  # highs[0]=15 >= lows[2]=14

    def test_out_of_range_returns_false(self):
        self.assertFalse(smc.has_fvg([1, 2], [1, 2], 0))


def _bars(rows: list[tuple[float, float, float, float]]) -> dict:
    """rows of (open, high, low, close); date is just the row index."""
    return {
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
        "date": list(range(len(rows))),
    }


class TestFindSmcLongTrades(unittest.TestCase):
    def test_too_few_bars_returns_no_trades(self):
        bars = _bars([(10, 10, 9, 10)] * 5)
        self.assertEqual(smc.find_smc_long_trades(bars), [])

    def test_full_trade_lifecycle(self):
        """Hand-traced scenario (see module docstring header for the full
        derivation): a swing high forms at idx 2 (12), a bearish OB candle
        at idx 5 (high=11, low=8) is followed by a bullish impulse that
        both creates a 3-candle FVG (highs[5]=11 < lows[7]=16) and closes
        above the swing high at idx 6 (ChoCh) -- confirming the OB.
        Price pulls back to touch the OB's top (11) at idx 8, entering
        long at 11 with a stop at 8. No bearish OB ever forms, so TP1 is
        None. A later swing high confirms at idx 11 (25), so the position
        fully exits there at its close (17) once confirmed at idx 13.
        """
        rows = [
            (10, 10, 9, 10),    # 0
            (10, 10, 9, 10),    # 1
            (12, 12, 11, 12),   # 2 -- swing high (12), confirmed at idx 4
            (10, 10, 9, 10),    # 3
            (10, 10, 9, 10),    # 4
            (11, 11, 8, 9),     # 5 -- bearish OB candle (high=11, low=8)
            (9, 17, 12, 16),    # 6 -- bullish impulse; closes(16) > swing high(12) -> ChoCh
            (16, 20, 16, 19),   # 7 -- FVG confirmed: highs[5]=11 < lows[7]=16
            (19, 19, 10, 13),   # 8 -- pullback touches OB top (11) -> ENTRY @ 11
            (13, 16, 13, 15),   # 9
            (15, 19, 15, 18),   # 10
            (18, 25, 17, 17),   # 11 -- new swing high (25), confirmed at idx 13
            (17, 20, 15, 16),   # 12
            (16, 18, 13, 14),   # 13 -- exit fires here (confirmation lag)
        ]
        bars = _bars(rows)

        trades = smc.find_smc_long_trades(bars)

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade["entry_idx"], 8)
        self.assertEqual(trade["entry_price"], 11)
        self.assertEqual(trade["stop_price"], 8)
        self.assertEqual(trade["ob_idx"], 5)
        self.assertIsNone(trade["tp1_price"])

        self.assertEqual(len(trade["fills"]), 1)
        fill = trade["fills"][0]
        self.assertEqual(fill["idx"], 11)
        self.assertEqual(fill["price"], 17)
        self.assertEqual(fill["qty_fraction"], 1.0)
        self.assertEqual(fill["reason"], "new_high_exit")

    def test_stop_out_before_any_exit_signal(self):
        """Same setup as the lifecycle test through entry, but price drops
        straight through the stop instead of continuing up."""
        rows = [
            (10, 10, 9, 10), (10, 10, 9, 10), (12, 12, 11, 12),
            (10, 10, 9, 10), (10, 10, 9, 10),
            (11, 11, 8, 9), (9, 17, 12, 16), (16, 20, 16, 19),
            (19, 19, 10, 13),   # entry @ 11
            (13, 13, 5, 6),     # low(5) <= stop(8) -> stopped out
        ]
        bars = _bars(rows)

        trades = smc.find_smc_long_trades(bars)

        self.assertEqual(len(trades), 1)
        fill = trades[0]["fills"][0]
        self.assertEqual(fill["reason"], "stop")
        self.assertEqual(fill["price"], 8)
        self.assertEqual(fill["qty_fraction"], 1.0)


class TestRequireConfirmedTrend(unittest.TestCase):
    def test_raw_choch_ob_is_skipped_when_confirmation_required(self):
        """The lifecycle scenario's OB (idx 5) forms at idx 6, the very
        break that first flips trend from "none" to "up" -- a raw,
        unconfirmed ChoCh. With require_confirmed_trend=True, that OB
        must be skipped entirely (no continuation BoS ever follows in
        this fixture), so no trade should be taken at all."""
        rows = [
            (10, 10, 9, 10), (10, 10, 9, 10), (12, 12, 11, 12), (10, 10, 9, 10), (10, 10, 9, 10),
            (11, 11, 8, 9), (9, 17, 12, 16), (16, 20, 16, 19), (19, 19, 10, 13),
            (13, 16, 13, 15), (15, 19, 15, 18), (18, 25, 17, 17), (17, 20, 15, 16), (16, 18, 13, 14),
        ]
        bars = _bars(rows)

        trades_without_confirmation = smc.find_smc_long_trades(bars, require_confirmed_trend=False)
        trades_with_confirmation = smc.find_smc_long_trades(bars, require_confirmed_trend=True)

        self.assertEqual(len(trades_without_confirmation), 1)  # sanity check against the baseline test
        self.assertEqual(trades_with_confirmation, [])


def _bars_with_day_boundary(rows: list[tuple[float, float, float, float]], day_boundary_idx: int) -> dict:
    """Same shape as _bars, but with real datetimes so dates[i].date() is
    meaningful: bars before day_boundary_idx fall on 2024-01-01, bars from
    day_boundary_idx onward fall on 2024-01-02 -- needed to exercise
    force_close_same_day, which _bars' plain integer "date" stand-ins
    can't support (force_close_same_day=False, the default, never calls
    .date() on them, which is why the other tests get away with ints)."""
    day_a = datetime(2024, 1, 1, 9, 30)
    day_b = datetime(2024, 1, 2, 9, 30)
    dates = [day_a if i < day_boundary_idx else day_b for i in range(len(rows))]
    return {
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
        "date": dates,
    }


class TestForceCloseSameDay(unittest.TestCase):
    """Same hand-traced lifecycle fixture as TestFindSmcLongTrades (entry
    at idx 8 @ 11, stop @ 8, next confirmed swing high -- and thus the
    ordinary new_high_exit -- at idx 11), just with real dates layered on
    so a day boundary can be placed relative to the entry."""

    LIFECYCLE_ROWS = [
        (10, 10, 9, 10), (10, 10, 9, 10), (12, 12, 11, 12), (10, 10, 9, 10), (10, 10, 9, 10),
        (11, 11, 8, 9), (9, 17, 12, 16), (16, 20, 16, 19), (19, 19, 10, 13),
        (13, 16, 13, 15), (15, 19, 15, 18), (18, 25, 17, 17), (17, 20, 15, 16), (16, 18, 13, 14),
    ]

    def test_forces_exit_at_entry_days_last_bar_instead_of_riding_to_next_day(self):
        """Day boundary falls right after idx 9 (entry @ idx 8's own day
        runs through idx 9), so with force_close_same_day the position
        should be cut off at idx 9's close instead of surviving to see
        idx 11's new_high_exit."""
        bars = _bars_with_day_boundary(self.LIFECYCLE_ROWS, day_boundary_idx=10)

        baseline = smc.find_smc_long_trades(bars, force_close_same_day=False)
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]["fills"][0]["reason"], "new_high_exit")
        self.assertEqual(baseline[0]["fills"][0]["idx"], 11)

        forced = smc.find_smc_long_trades(bars, force_close_same_day=True)
        self.assertEqual(len(forced), 1)
        trade = forced[0]
        self.assertEqual(trade["entry_idx"], 8)
        self.assertEqual(len(trade["fills"]), 1)
        fill = trade["fills"][0]
        self.assertEqual(fill["idx"], 9)
        self.assertEqual(fill["price"], self.LIFECYCLE_ROWS[9][3])  # that day's close
        self.assertEqual(fill["qty_fraction"], 1.0)
        self.assertEqual(fill["reason"], "same_day_force_close")

    def test_skips_entry_that_would_land_on_last_bar_of_a_day(self):
        """Day boundary falls right after idx 8 -- the bar the OB retest
        would otherwise enter on is the LAST bar of its day, leaving no
        room to force-close same-day. force_close_same_day should skip
        taking that entry at all (rather than let it carry into the next
        day for even one bar), and since no later bar ever re-touches the
        OB's top (11) in this fixture, no trade is taken at all."""
        bars = _bars_with_day_boundary(self.LIFECYCLE_ROWS, day_boundary_idx=9)

        baseline = smc.find_smc_long_trades(bars, force_close_same_day=False)
        self.assertEqual(len(baseline), 1)  # sanity check against the ordinary lifecycle test

        forced = smc.find_smc_long_trades(bars, force_close_same_day=True)
        self.assertEqual(forced, [])


class TestLatestEntrySignal(unittest.TestCase):
    """Live-polling adapter over the hand-traced lifecycle fixture, padded
    with 2 extra flat leading bars so truncating at the entry bar still
    clears find_smc_long_trades' n >= 10 minimum (live bar windows are
    hundreds of bars, so the guard never matters there -- only in this
    fixture). With the padding, the entry triggers on idx 10: a live
    recompute whose LAST bar is idx 10 should report the signal, and one
    ending a bar earlier should not."""

    PADDED_ROWS = [(10, 10, 9, 10)] * 2 + [
        (10, 10, 9, 10), (10, 10, 9, 10), (12, 12, 11, 12), (10, 10, 9, 10), (10, 10, 9, 10),
        (11, 11, 8, 9), (9, 17, 12, 16), (16, 20, 16, 19), (19, 19, 10, 13),
        (13, 16, 13, 15), (15, 19, 15, 18), (18, 25, 17, 17), (17, 20, 15, 16), (16, 18, 13, 14),
    ]
    ENTRY_IDX = 10

    def test_signal_fires_when_entry_bar_is_last(self):
        bars = _bars(self.PADDED_ROWS[:self.ENTRY_IDX + 1])  # ends on the entry bar
        signal = smc.latest_entry_signal(bars)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["entry_idx"], self.ENTRY_IDX)
        self.assertEqual(signal["entry_price"], 11)
        self.assertEqual(signal["stop_price"], 8)
        self.assertIsNone(signal["tp1_price"])

    def test_no_signal_one_bar_before_entry(self):
        bars = _bars(self.PADDED_ROWS[:self.ENTRY_IDX])
        self.assertIsNone(smc.latest_entry_signal(bars))

    def test_no_signal_after_entry_bar_has_passed(self):
        # A cycle running one bar late must NOT re-report the same entry
        # (the OB was mitigated on the entry bar; the next bar is not an
        # entry bar).
        bars = _bars(self.PADDED_ROWS[:self.ENTRY_IDX + 2])
        self.assertIsNone(smc.latest_entry_signal(bars))

    def test_empty_bars_returns_none(self):
        self.assertIsNone(smc.latest_entry_signal(_bars([])))


class TestConfirmedNewHighExit(unittest.TestCase):
    HIGHS = [10, 10, 12, 10, 10, 11, 17, 20, 19, 16, 19, 25, 20, 18]  # swing high @ idx 11 (25)

    def test_exit_once_post_entry_swing_high_is_confirmed(self):
        # entry @ idx 8; swing high @ 11 needs 2 bars after it (window=2),
        # so it's confirmed once idx 13 exists.
        self.assertTrue(smc.confirmed_new_high_exit(self.HIGHS, entry_idx=8, swing_window=2))

    def test_no_exit_before_confirmation_lag_elapses(self):
        self.assertFalse(smc.confirmed_new_high_exit(self.HIGHS[:13], entry_idx=8, swing_window=2))

    def test_pre_entry_swing_highs_do_not_trigger(self):
        # Swing high @ idx 2 (12) is before an entry at idx 8 -- with the
        # post-entry data truncated so idx 11's high isn't confirmed, the
        # pre-entry pivot alone must not fire the exit.
        self.assertFalse(smc.confirmed_new_high_exit(self.HIGHS[:12], entry_idx=8, swing_window=2))


class TestSlippage(unittest.TestCase):
    """Slippage must move fills adversely without changing WHICH bar triggers
    them -- a level is still touched at exactly the same index, only the
    recorded price differs. The lifecycle fixture (entry @ 11 on idx 8,
    new_high_exit @ 17 on idx 11) is reused as the reference."""

    LIFECYCLE_ROWS = [
        (10, 10, 9, 10), (10, 10, 9, 10), (12, 12, 11, 12), (10, 10, 9, 10), (10, 10, 9, 10),
        (11, 11, 8, 9), (9, 17, 12, 16), (16, 20, 16, 19), (19, 19, 10, 13),
        (13, 16, 13, 15), (15, 19, 15, 18), (18, 25, 17, 17), (17, 20, 15, 16), (16, 18, 13, 14),
    ]
    STOP_OUT_ROWS = [
        (10, 10, 9, 10), (10, 10, 9, 10), (12, 12, 11, 12), (10, 10, 9, 10), (10, 10, 9, 10),
        (11, 11, 8, 9), (9, 17, 12, 16), (16, 20, 16, 19), (19, 19, 10, 13),
        (13, 13, 5, 6),
    ]

    def test_none_and_zero_match_the_frictionless_baseline(self):
        bars = _bars(self.LIFECYCLE_ROWS)
        baseline = smc.find_smc_long_trades(bars)
        for spec in (None, 0, 0.0, {}, {"entry": 0.0}):
            self.assertEqual(smc.find_smc_long_trades(bars, slippage_bps=spec), baseline, msg=f"spec={spec!r}")

    def test_buy_slips_up_and_sell_slips_down(self):
        bars = _bars(self.LIFECYCLE_ROWS)
        trades = smc.find_smc_long_trades(bars, slippage_bps={"entry": 100, "new_high_exit": 100})
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade["entry_idx"], 8)  # same bar as the frictionless run
        self.assertAlmostEqual(trade["entry_price"], 11 * 1.01)  # buy fills HIGHER
        self.assertEqual(trade["fills"][0]["idx"], 11)
        self.assertAlmostEqual(trade["fills"][0]["price"], 17 * 0.99)  # sell fills LOWER

    def test_initial_stop_price_is_never_slipped(self):
        """Sizing reads initial_stop_price, so it must stay the raw OB low --
        slipping the entry widens real risk rather than hiding it."""
        bars = _bars(self.LIFECYCLE_ROWS)
        trade = smc.find_smc_long_trades(bars, slippage_bps={"entry": 250})[0]
        self.assertEqual(trade["initial_stop_price"], 8)
        self.assertGreater(trade["entry_price"], 11)

    def test_stop_triggers_on_the_true_level_but_fills_worse(self):
        bars = _bars(self.STOP_OUT_ROWS)
        trades = smc.find_smc_long_trades(bars, slippage_bps={"stop": 50})
        fill = trades[0]["fills"][0]
        self.assertEqual(fill["reason"], "stop")
        self.assertEqual(fill["idx"], 9)  # unchanged trigger bar
        self.assertAlmostEqual(fill["price"], 8 * 0.995)

    def test_scalar_applies_to_every_leg(self):
        bars = _bars(self.LIFECYCLE_ROWS)
        trade = smc.find_smc_long_trades(bars, slippage_bps=100)[0]
        self.assertAlmostEqual(trade["entry_price"], 11 * 1.01)
        self.assertAlmostEqual(trade["fills"][0]["price"], 17 * 0.99)

    def test_unknown_reason_key_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            smc.find_smc_long_trades(_bars(self.LIFECYCLE_ROWS), slippage_bps={"stop_loss": 10})
        self.assertIn("stop_loss", str(ctx.exception))

    def test_breakeven_stop_becomes_a_loss(self):
        """The point of the whole feature: post-TP1 the stop sits at
        entry_price, so with zero slippage the runner leg is exactly $0
        every time. With slippage it must come back BELOW entry."""
        entry = 100.0
        self.assertEqual(smc._slipped(entry, 0, "sell"), entry)
        self.assertLess(smc._slipped(entry, 10, "sell"), entry)
        self.assertAlmostEqual(smc._slipped(entry, 10, "sell"), 99.9)


if __name__ == "__main__":
    unittest.main()
