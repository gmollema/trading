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

        trades = smc.find_smc_long_trades(bars, entry_fill="level", exit_fill="level")

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

        trades = smc.find_smc_long_trades(bars, entry_fill="level", exit_fill="level")

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

        baseline = smc.find_smc_long_trades(bars, force_close_same_day=False, entry_fill="level", exit_fill="level")
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]["fills"][0]["reason"], "new_high_exit")
        self.assertEqual(baseline[0]["fills"][0]["idx"], 11)

        forced = smc.find_smc_long_trades(bars, force_close_same_day=True, entry_fill="level", exit_fill="level")
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

        baseline = smc.find_smc_long_trades(bars, force_close_same_day=False, entry_fill="level", exit_fill="level")
        self.assertEqual(len(baseline), 1)  # sanity check against the ordinary lifecycle test

        forced = smc.find_smc_long_trades(bars, force_close_same_day=True, entry_fill="level", exit_fill="level")
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
        trades = smc.find_smc_long_trades(bars, entry_fill="level", exit_fill="level", slippage_bps={"entry": 100, "new_high_exit": 100})
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
        trades = smc.find_smc_long_trades(bars, entry_fill="level", exit_fill="level", slippage_bps={"stop": 50})
        fill = trades[0]["fills"][0]
        self.assertEqual(fill["reason"], "stop")
        self.assertEqual(fill["idx"], 9)  # unchanged trigger bar
        self.assertAlmostEqual(fill["price"], 8 * 0.995)

    def test_scalar_applies_to_every_leg(self):
        bars = _bars(self.LIFECYCLE_ROWS)
        trade = smc.find_smc_long_trades(bars, entry_fill="level", exit_fill="level", slippage_bps=100)[0]
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


class TestPostTp1StopFraction(unittest.TestCase):
    """A trade that reaches TP1 and then retraces: where the stop sits after
    TP1 decides whether it scratches or keeps its original room. The fixture
    enters at 11 with an initial stop at 8 and a TP1 at 15 (a bearish OB low
    above entry), then drops to 9 -- below entry but above the initial stop --
    so fraction 1.0 stops out while fraction 0.0 survives."""

    ROWS = [
        (10, 10, 9, 10), (10, 10, 9, 10), (12, 12, 11, 12), (10, 10, 9, 10), (10, 10, 9, 10),
        (11, 11, 8, 9), (9, 17, 12, 16), (16, 20, 16, 19), (19, 19, 10, 13),
        (13, 16, 13, 15), (15, 19, 15, 18), (18, 25, 17, 17), (17, 20, 15, 16), (16, 18, 13, 14),
    ]

    def test_default_is_breakeven(self):
        bars = _bars(self.ROWS)
        self.assertEqual(smc.DEFAULT_POST_TP1_STOP_FRACTION, 1.0)
        default = smc.find_smc_long_trades(bars)
        explicit = smc.find_smc_long_trades(bars, post_tp1_stop_fraction=1.0)
        self.assertEqual(default, explicit)

    def test_fraction_interpolates_between_initial_stop_and_entry(self):
        """entry 11, initial stop 8 -> the post-TP1 stop must land on
        8 + f*(11-8) for each f. Checked directly on the mutation so the
        assertion does not depend on which bar happens to trigger."""
        for f, expected in ((0.0, 8.0), (0.5, 9.5), (1.0, 11.0), (1.25, 11.75)):
            pos = {"initial_stop_price": 8.0, "entry_price": 11.0}
            got = pos["initial_stop_price"] + f * (pos["entry_price"] - pos["initial_stop_price"])
            self.assertAlmostEqual(got, expected, msg=f"fraction={f}")

    def test_zero_fraction_leaves_the_original_stop_in_place(self):
        bars = _bars(self.ROWS)
        trades = smc.find_smc_long_trades(bars, post_tp1_stop_fraction=0.0)
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        reasons = [f["reason"] for f in trade["fills"]]
        if "tp1" in reasons:
            # After TP1 the stop must still be the untouched initial stop.
            self.assertEqual(trade["stop_price"], trade["initial_stop_price"])

    def test_post_tp1_stop_is_clamped_to_the_last_traded_price(self):
        """A stop must never be placed above the bar close that set it --
        otherwise a large fraction books exits at prices the market never
        printed, and the sweep rewards impossible fills."""
        bars = _bars(self.ROWS)
        for frac in (1.25, 2.0, 5.0, 50.0):
            trades = smc.find_smc_long_trades(bars, post_tp1_stop_fraction=frac)
            for tr in trades:
                tp1_fill = next((f for f in tr["fills"] if f["reason"] == "tp1"), None)
                if tp1_fill is None:
                    continue
                close_at_tp1 = self.ROWS[tp1_fill["idx"]][3]
                self.assertLessEqual(tr["stop_price"], close_at_tp1, msg=f"fraction={frac}")

    def test_huge_fractions_stop_improving_once_clamped(self):
        """Sanity guard on the clamp: absurd fractions must converge rather
        than keep paying more."""
        bars = _bars(self.ROWS)
        a = smc.find_smc_long_trades(bars, post_tp1_stop_fraction=10.0)
        b = smc.find_smc_long_trades(bars, post_tp1_stop_fraction=1000.0)
        self.assertEqual(a, b)

    def test_fraction_above_one_puts_the_stop_past_entry(self):
        bars = _bars(self.ROWS)
        trade = smc.find_smc_long_trades(bars, post_tp1_stop_fraction=2.0)[0]
        if any(f["reason"] == "tp1" for f in trade["fills"]):
            self.assertGreater(trade["stop_price"], trade["entry_price"])


class TestExitFullyAtTp1(unittest.TestCase):
    """The whole position leaves at tp1_price; there is no runner afterwards,
    so no breakeven stop and no new-high exit can fire."""

    ROWS = TestPostTp1StopFraction.ROWS

    def test_default_is_off(self):
        bars = _bars(self.ROWS)
        self.assertEqual(
            smc.find_smc_long_trades(bars),
            smc.find_smc_long_trades(bars, exit_fully_at_tp1=False),
        )

    def test_single_tp1_fill_closes_the_trade(self):
        bars = _bars(self.ROWS)
        trades = smc.find_smc_long_trades(bars, exit_fully_at_tp1=True)
        for tr in trades:
            reasons = [f["reason"] for f in tr["fills"]]
            if "tp1" not in reasons:
                continue
            self.assertEqual(reasons, ["tp1"], msg="tp1 must be the only, final fill")
            self.assertEqual(tr["fills"][0]["qty_fraction"], 1.0)
            self.assertAlmostEqual(tr["remaining_fraction"], 0.0)

    def test_fills_at_the_tp1_level_not_a_later_close(self):
        bars = _bars(self.ROWS)
        for tr in smc.find_smc_long_trades(bars, exit_fully_at_tp1=True):
            for f in tr["fills"]:
                if f["reason"] == "tp1":
                    self.assertEqual(f["price"], tr["tp1_price"])

    def test_slippage_still_applies_to_the_full_exit(self):
        bars = _bars(self.ROWS)
        clean = smc.find_smc_long_trades(bars, exit_fully_at_tp1=True)
        slipped = smc.find_smc_long_trades(bars, exit_fully_at_tp1=True, slippage_bps={"tp1": 100})
        for a, b in zip(clean, slipped):
            for fa, fb in zip(a["fills"], b["fills"]):
                if fa["reason"] == "tp1":
                    self.assertAlmostEqual(fb["price"], fa["price"] * 0.99)

    def test_trades_without_a_tp1_level_are_unaffected(self):
        """tp1_price is None when no bearish OB sits above entry -- those
        trades must still run to their new-high exit."""
        bars = _bars(TestSlippage.LIFECYCLE_ROWS)
        trades = smc.find_smc_long_trades(bars, exit_fully_at_tp1=True)
        self.assertEqual(len(trades), 1)
        self.assertIsNone(trades[0]["tp1_price"])
        self.assertEqual([f["reason"] for f in trades[0]["fills"]], ["new_high_exit"])


if __name__ == "__main__":
    unittest.main()


class TestEntryFill(unittest.TestCase):
    """The reachable entry specs, over the same hand-traced lifecycle
    fixture. Its retest bar is idx 8 (19, 19, 10, 13): its low (10) reaches
    the OB high (11), and the bar after it opens at 13 and ranges up to 16.
    So the three specs price the identical signal at 11 (unreachable), 13
    (next_open) and 16 (next_high)."""

    LIFECYCLE_ROWS = [
        (10, 10, 9, 10), (10, 10, 9, 10), (12, 12, 11, 12), (10, 10, 9, 10), (10, 10, 9, 10),
        (11, 11, 8, 9), (9, 17, 12, 16), (16, 20, 16, 19), (19, 19, 10, 13),
        (13, 16, 13, 15), (15, 19, 15, 18), (18, 25, 17, 17), (17, 20, 15, 16), (16, 18, 13, 14),
    ]

    def test_the_default_is_the_reachable_spec(self):
        """Flipped on 2026-08-30. A default nobody can execute is how this
        repo came to publish +97.2% for a strategy worth about +0.2%, so
        forgetting to pass a spec now yields a fill an order could get."""
        self.assertEqual(smc.DEFAULT_ENTRY_FILL, "next_open")
        self.assertEqual(smc.DEFAULT_EXIT_FILL, "next_open")
        trades = smc.find_smc_long_trades(_bars(self.LIFECYCLE_ROWS))
        self.assertEqual(trades[0]["entry_idx"], 9)
        self.assertEqual(trades[0]["entry_price"], 13)

    def test_level_still_available_when_asked_for(self):
        trades = smc.find_smc_long_trades(_bars(self.LIFECYCLE_ROWS), entry_fill="level")
        self.assertEqual(trades[0]["entry_idx"], 8)
        self.assertEqual(trades[0]["entry_price"], 11)

    def test_next_open_fills_on_the_following_bars_open(self):
        trades = smc.find_smc_long_trades(_bars(self.LIFECYCLE_ROWS), entry_fill="next_open")
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade["entry_idx"], 9)
        self.assertEqual(trade["entry_date"], 9)
        self.assertEqual(trade["entry_price"], 13)  # opens[9]
        self.assertEqual(trade["stop_price"], 8)    # still the OB's low
        self.assertEqual(trade["fills"][0]["reason"], "new_high_exit")

    def test_next_high_fills_at_the_worst_price_of_the_fill_bar(self):
        trades = smc.find_smc_long_trades(_bars(self.LIFECYCLE_ROWS), entry_fill="next_high")
        self.assertEqual(trades[0]["entry_idx"], 9)
        self.assertEqual(trades[0]["entry_price"], 16)  # highs[9]

    def test_signal_bar_and_level_are_recorded_separately_from_the_fill(self):
        """The OB high is the trigger, not the fill -- both are kept, so a
        run can be compared against the unreachable spec it replaces."""
        trades = smc.find_smc_long_trades(_bars(self.LIFECYCLE_ROWS), entry_fill="next_open")
        trade = trades[0]
        self.assertEqual(trade["signal_idx"], 8)
        self.assertEqual(trade["signal_price"], 11)
        self.assertNotEqual(trade["entry_price"], trade["signal_price"])

    def test_position_is_managed_from_the_fill_bar(self):
        """A stop breach on the FILL bar counts: the position exists from
        that bar's open onward."""
        rows = list(self.LIFECYCLE_ROWS)
        rows[9] = (13, 16, 7, 15)  # dips to 7, under the stop at 8
        trades = smc.find_smc_long_trades(_bars(rows), entry_fill="next_open")
        self.assertEqual(len(trades), 1)
        fill = trades[0]["fills"][0]
        self.assertEqual(fill["reason"], "stop")
        self.assertEqual(fill["idx"], 9)

    def test_no_trade_when_the_signal_lands_on_the_last_bar(self):
        """Nothing to fill on -- the bar after the signal is the one that
        does not exist. Padded with 2 flat leading bars so truncating at
        the retest still clears find_smc_long_trades' n >= 10 minimum."""
        padded = [(10, 10, 9, 10)] * 2 + list(self.LIFECYCLE_ROWS)
        rows = padded[:11]  # ends on the retest bar (idx 8 + 2 padding)
        self.assertEqual(smc.find_smc_long_trades(_bars(rows), entry_fill="next_open"), [])
        self.assertEqual(len(smc.find_smc_long_trades(_bars(rows), entry_fill="level")), 1)

    def test_no_trade_across_a_session_boundary(self):
        """An order placed at a day's close is not live at the next open,
        and this strategy holds nothing overnight."""
        bars = _bars_with_day_boundary(self.LIFECYCLE_ROWS, day_boundary_idx=9)
        self.assertEqual(smc.find_smc_long_trades(bars, entry_fill="next_open"), [])

    def test_fill_bar_opening_below_the_stop_is_dropped(self):
        """No positive risk-per-share to size against. Documented in the
        entry block as the one place this model flatters itself: live it
        would be a fill and an immediate stop-out, not a skipped trade.
        The OB is consumed, so the drop is not undone by re-entering a bar
        later at whatever price the recovery offers."""
        rows = list(self.LIFECYCLE_ROWS)
        rows[9] = (7, 16, 6, 15)  # opens at 7, below the OB low (8)
        self.assertEqual(smc.find_smc_long_trades(_bars(rows), entry_fill="next_open"), [])

    def test_slippage_still_applies_on_top_of_the_fill_price(self):
        trades = smc.find_smc_long_trades(
            _bars(self.LIFECYCLE_ROWS), entry_fill="next_open", slippage_bps={"entry": 100.0},
        )
        self.assertAlmostEqual(trades[0]["entry_price"], 13 * 1.01)
        self.assertEqual(trades[0]["signal_price"], 11)  # the level is never slipped

    def test_unknown_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            smc.find_smc_long_trades(_bars(self.LIFECYCLE_ROWS), entry_fill="mid")


class TestRequireObReclaim(unittest.TestCase):
    """The signal-side filter: the retest bar must CLOSE back above the OB
    high. Same fixture, with the retest bar's close moved either side of
    the level (11) -- nothing else about the structure depends on it."""

    ROWS_RECLAIMED = TestEntryFill.LIFECYCLE_ROWS                      # idx 8 closes at 13
    ROWS_REJECTED = (
        TestEntryFill.LIFECYCLE_ROWS[:8]
        + [(19, 19, 10, 10.5)]                                          # idx 8 closes at 10.5
        + TestEntryFill.LIFECYCLE_ROWS[9:]
    )

    def test_bar_closing_back_above_the_level_still_trades(self):
        trades = smc.find_smc_long_trades(_bars(self.ROWS_RECLAIMED), require_ob_reclaim=True, entry_fill="level", exit_fill="level")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["entry_idx"], 8)

    def test_bar_closing_below_the_level_is_skipped(self):
        bars = _bars(self.ROWS_REJECTED)
        self.assertEqual(len(smc.find_smc_long_trades(bars, entry_fill="level")), 1)
        self.assertEqual(smc.find_smc_long_trades(bars, require_ob_reclaim=True), [])

    def test_a_rejected_retest_still_mitigates_the_order_block(self):
        """"Only the FIRST retest counts" survives the filter: a second
        touch of the same OB must not become a second chance at it."""
        rows = list(self.ROWS_REJECTED) + [
            (14, 14, 10, 13),   # 14 -- touches the OB high (11) again, and reclaims
            (13, 15, 13, 14),   # 15
        ]
        trades = smc.find_smc_long_trades(_bars(rows), require_ob_reclaim=True)
        self.assertEqual(trades, [])

    def test_composes_with_a_reachable_fill(self):
        trades = smc.find_smc_long_trades(
            _bars(self.ROWS_RECLAIMED), entry_fill="next_open", require_ob_reclaim=True,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["entry_idx"], 9)
        self.assertEqual(trades[0]["entry_price"], 13)

    def test_live_adapter_applies_the_same_filter(self):
        """The reclaim rule is signal-side, so smc_cycle must see exactly
        what the backtest scored -- unlike entry_fill, which cannot apply
        live (see latest_entry_signal's docstring)."""
        padded = [(10, 10, 9, 10)] * 2 + list(self.ROWS_REJECTED)
        bars = _bars(padded[:11])  # ends on the retest bar (idx 8 + 2 padding)
        self.assertIsNotNone(smc.latest_entry_signal(bars))
        self.assertIsNone(smc.latest_entry_signal(bars, require_ob_reclaim=True))


class TestExitFill(unittest.TestCase):
    """The exits carried the same defect the entry did: each booked its
    trigger as its fill. The new-high exit was the worst -- it fires only
    once a pivot is CONFIRMED, swing_window bars later, but filled at the
    pivot bar's own close.

    All of these use the hand-traced lifecycle fixture (entry at idx 8 on
    the OB high of 11, stop at 8) with swing_window=2, so the new-high
    pivot at idx 11 confirms at idx 13.
    """

    LIFECYCLE_ROWS = TestEntryFill.LIFECYCLE_ROWS

    def test_new_high_exit_fills_after_confirmation_not_at_the_pivot(self):
        rows = list(self.LIFECYCLE_ROWS) + [(14, 15, 13, 14)]  # idx 14, the fill bar
        bars = _bars(rows)

        level = smc.find_smc_long_trades(bars, exit_fill="level", entry_fill="level")[0]["fills"][0]
        self.assertEqual(level["idx"], 11)              # the pivot bar
        self.assertEqual(level["price"], 17)            # its close

        reachable = smc.find_smc_long_trades(bars, exit_fill="next_open", entry_fill="level")[0]["fills"][0]
        self.assertEqual(reachable["reason"], "new_high_exit")
        self.assertEqual(reachable["idx"], 14)          # the bar after confirmation at 13
        self.assertEqual(reachable["price"], 14)        # opens[14]

    def test_new_high_exit_falls_back_to_the_confirming_close(self):
        """No bar left to fill on: the market order still executes, at the
        close of the bar that revealed the exit. An exit is not optional
        the way an entry is."""
        bars = _bars(self.LIFECYCLE_ROWS)  # confirmation lands on the last bar, idx 13
        fill = smc.find_smc_long_trades(bars, exit_fill="next_open")[0]["fills"][0]
        self.assertEqual(fill["reason"], "new_high_exit")
        self.assertEqual(fill["idx"], 13)
        self.assertEqual(fill["price"], self.LIFECYCLE_ROWS[13][3])

    def test_stop_still_fills_at_its_level(self):
        """It rests at the broker and triggers intrabar, so unlike the
        others it earns its level."""
        rows = list(self.LIFECYCLE_ROWS)
        rows[9] = (13, 13, 5, 6)  # trades down through the stop at 8, opening above it
        fill = smc.find_smc_long_trades(_bars(rows), exit_fill="next_open", entry_fill="level")[0]["fills"][0]
        self.assertEqual(fill["reason"], "stop")
        self.assertEqual(fill["price"], 8)

    def test_stop_that_gaps_through_fills_at_the_open(self):
        """A bar that OPENED under the stop never offered the level."""
        rows = list(self.LIFECYCLE_ROWS)
        rows[9] = (6, 7, 5, 6)  # opens at 6, below the stop at 8
        bars = _bars(rows)

        self.assertEqual(smc.find_smc_long_trades(bars, entry_fill="level", exit_fill="level")[0]["fills"][0]["price"], 8)
        gapped = smc.find_smc_long_trades(bars, exit_fill="next_open", entry_fill="level")[0]["fills"][0]
        self.assertEqual(gapped["reason"], "stop")
        self.assertEqual(gapped["price"], 6)

    def test_force_close_fills_at_the_bars_open(self):
        """The bot fires at 15:51 ET, inside the bar before the day's
        last, so that last bar's OPEN is the nearest reachable price --
        its close is nine minutes of hindsight."""
        bars = _bars_with_day_boundary(self.LIFECYCLE_ROWS, day_boundary_idx=10)

        level = smc.find_smc_long_trades(bars, force_close_same_day=True, entry_fill="level", exit_fill="level")[0]["fills"][0]
        self.assertEqual(level["reason"], "same_day_force_close")
        self.assertEqual(level["price"], self.LIFECYCLE_ROWS[9][3])  # closes[9]

        reachable = smc.find_smc_long_trades(
            bars, force_close_same_day=True, exit_fill="next_open", entry_fill="level",
        )[0]["fills"][0]
        self.assertEqual(reachable["reason"], "same_day_force_close")
        self.assertEqual(reachable["idx"], 9)
        self.assertEqual(reachable["price"], self.LIFECYCLE_ROWS[9][0])  # opens[9]

    def test_unknown_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            smc.find_smc_long_trades(_bars(self.LIFECYCLE_ROWS), exit_fill="market")


class TestTp1ExitFill(unittest.TestCase):
    """TP1 needs a bearish order block above entry to exist at all, so it
    gets its own fixture: the lifecycle scenario with a downside break
    added before the entry, which registers a bearish OB whose low becomes
    the target.
    """

    ROWS = [
        (10, 10, 9, 10),    # 0
        (10, 10, 9, 10),    # 1
        (12, 12, 11, 12),   # 2  swing high (12), confirmed at idx 4
        (10, 10, 9, 10),    # 3
        (10, 10, 9, 10),    # 4
        (11, 11, 8, 9),     # 5  bullish OB candle (high=11, low=8)
        (9, 17, 12, 16),    # 6  impulse closes above the swing high -> ChoCh
        (16, 20, 16, 19),   # 7  FVG confirmed: highs[5]=11 < lows[7]=16
        (18, 19, 17, 18),   # 8
        (18, 18, 15, 16),   # 9  swing low (15), confirmed at idx 11
        (16, 19, 16, 18),   # 10
        (18, 20, 17, 19),   # 11
        (17, 19, 16, 18),   # 12 bullish -- becomes the BEARISH OB (low=16)
        (16, 16, 13, 14),   # 13 closes below the swing low (15) -> bearish break
        (14, 14, 10, 11),   # 14 pullback touches the OB top (11) -> ENTRY @ 11
        (11, 13, 11, 13),   # 15
        (13, 17, 13, 16),   # 16 high (17) reaches TP1 at the bearish OB low (16)
        (15, 16, 14, 15),   # 17 the bar TP1 fills on under next_open
        (15, 16, 14, 15),   # 18
        (15, 16, 14, 15),   # 19
    ]
    ENTRY_IDX = 14
    TP1_TOUCH_IDX = 16

    def _trades(self, **kwargs):
        # entry pinned to "level" so the fixture's traced entry index and
        # price stay the subject; this suite is about the EXIT legs.
        kwargs.setdefault("entry_fill", "level")
        kwargs.setdefault("exit_fill", "level")
        return smc.find_smc_long_trades(_bars(self.ROWS), **kwargs)

    def test_fixture_produces_a_tp1(self):
        trade = self._trades()[0]
        self.assertIsNotNone(trade["tp1_price"])
        self.assertEqual([f["reason"] for f in trade["fills"]][0], "tp1")

    def test_level_fills_at_the_target(self):
        fill = self._trades()[0]["fills"][0]
        trade = self._trades()[0]
        self.assertEqual(fill["price"], trade["tp1_price"])

    def test_next_open_fills_a_bar_later(self):
        trade = self._trades(exit_fill="next_open")[0]
        fill = trade["fills"][0]
        self.assertEqual(fill["reason"], "tp1")
        touch_idx = self._trades()[0]["fills"][0]["idx"]
        self.assertEqual(fill["idx"], touch_idx + 1)
        self.assertEqual(fill["price"], self.ROWS[touch_idx + 1][0])

    def test_resting_limit_keeps_the_target(self):
        """A sell limit above the market is not adversely selected the way
        a buy limit at the entry level was: it fills exactly when price
        reaches the target, which is the event being traded."""
        trade = self._trades(exit_fill="next_open", tp1_resting_limit=True)[0]
        fill = trade["fills"][0]
        self.assertEqual(fill["reason"], "tp1")
        self.assertEqual(fill["price"], trade["tp1_price"])
        self.assertEqual(fill["idx"], self._trades()[0]["fills"][0]["idx"])

    def test_breakeven_stop_is_clamped_to_what_tp1_actually_got(self):
        """The stop is placed right after the TP1 leg fills, so that fill
        is the market reference -- not a close the bot never traded at."""
        trade = self._trades(exit_fill="next_open", post_tp1_stop_fraction=5.0)[0]
        tp1_fill = trade["fills"][0]
        self.assertLessEqual(trade["stop_price"], tp1_fill["price"])


class TestEntryAllowed(unittest.TestCase):
    """The live bot only scans for entries between time_filter's bounds.
    The backtest scanned every bar, so it opened positions at times the
    bot does not even look at."""

    LIFECYCLE_ROWS = TestEntryFill.LIFECYCLE_ROWS
    ENTRY_IDX = 8

    def _mask(self, blocked: set[int]) -> list[bool]:
        return [i not in blocked for i in range(len(self.LIFECYCLE_ROWS))]

    def test_none_allows_every_bar(self):
        trades = smc.find_smc_long_trades(_bars(self.LIFECYCLE_ROWS), entry_allowed=None)
        self.assertEqual(len(trades), 1)

    def test_allowed_entry_bar_still_trades(self):
        trades = smc.find_smc_long_trades(
            _bars(self.LIFECYCLE_ROWS), entry_allowed=self._mask(set()), entry_fill="level",
        )
        self.assertEqual(trades[0]["entry_idx"], self.ENTRY_IDX)

    def test_blocked_entry_bar_takes_no_trade(self):
        trades = smc.find_smc_long_trades(
            _bars(self.LIFECYCLE_ROWS), entry_allowed=self._mask({self.ENTRY_IDX}),
        )
        self.assertEqual(trades, [])

    def test_a_blocked_retest_still_mitigates_the_order_block(self):
        """What the live bot does: smc_cycle re-runs the whole signal pass
        each cycle, so a touch outside the window has already consumed the
        block by the time the next in-window cycle looks at it. Leaving it
        pending would invent an entry the bot cannot take."""
        rows = list(self.LIFECYCLE_ROWS) + [
            (14, 14, 10, 13),   # 14 -- touches the OB high (11) again
            (13, 15, 13, 14),   # 15
        ]
        mask = [i != self.ENTRY_IDX for i in range(len(rows))]
        self.assertEqual(smc.find_smc_long_trades(_bars(rows), entry_allowed=mask), [])

    def test_mask_length_must_match_the_bars(self):
        """A short mask would silently gate the wrong bars."""
        with self.assertRaises(ValueError):
            smc.find_smc_long_trades(_bars(self.LIFECYCLE_ROWS), entry_allowed=[True] * 3)
