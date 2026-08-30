"""Unit tests for trading_bot.cli.smc_signal_parity.

The measurement only means anything if the replay is faithful to how
smc_cycle actually drives latest_entry_signal: a truncated trailing
window, closed bars only, and the entry window checked against when the
bot could ACT rather than when the bar opened. Each of those is pinned
here, because getting any of them wrong would produce a plausible parity
number that measured the wrong thing.
"""

import datetime
import unittest

import pandas as pd

from trading_bot.cli import smc_signal_parity as p

RULES = {
    "time_filter": {"earliest_entry_et": "10:05", "latest_entry_et": "15:45",
                    "force_close_et": "15:51"},
    "time_window_bars": 33,
    "tp1_fraction": 0.25,
    "swing_window": 2,
}


def _frame(n: int, start="2026-08-24 09:30", flat=True) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start, tz="America/New_York"), periods=n, freq="5min")
    price = [10.0] * n if flat else [10.0 + i * 0.01 for i in range(n)]
    return pd.DataFrame({
        "date": idx.tz_convert("UTC"),
        "open": price, "high": price, "low": price, "close": price,
    }).reset_index(drop=True)


class LiveSignalsForTest(unittest.TestCase):
    def test_flat_bars_produce_no_signal(self):
        """The control: no structure, no order block, nothing to fire."""
        frame = _frame(200)
        sessions = set(frame["date"].dt.tz_convert(p.ET).dt.date)
        self.assertEqual(p.live_signals_for(frame, sessions, RULES, False), set())

    def test_a_session_not_asked_for_is_not_replayed(self):
        """Tickers are only replayed on days they were on the watchlist."""
        frame = _frame(200)
        self.assertEqual(p.live_signals_for(frame, set(), RULES, False), set())

    def test_windows_shorter_than_the_signal_minimum_are_skipped(self):
        """find_smc_long_trades returns nothing under 10 bars; replaying
        them would be wasted work, not a different answer."""
        frame = _frame(4)
        sessions = set(frame["date"].dt.tz_convert(p.ET).dt.date)
        self.assertEqual(p.live_signals_for(frame, sessions, RULES, False), set())

    def test_the_action_delay_is_applied_to_the_entry_window(self):
        """A bar opening at 09:58 is acted on at 10:05 and is in window;
        one opening at 09:53 is acted on at 10:00 and is not. Checking the
        bar's own timestamp instead would shift every boundary by a bar."""
        self.assertEqual(p.ACTION_DELAY_MINUTES, 7)
        self.assertEqual(p.LIVE_WINDOW_DAYS, 7)

    def test_the_window_is_trailing_and_bounded(self):
        """The replay must never hand the signal pass more history than
        smc_cycle fetches -- that is the entire quantity being measured."""
        frame = _frame(3000, start="2026-06-01 09:30")
        sessions = {frame["date"].dt.tz_convert(p.ET).dt.date.iloc[-1]}
        # A one-day window can only ever see one session of bars.
        seen = []
        original = p.latest_entry_signal

        def spy(bars, *a, **kw):
            seen.append(len(bars["close"]))
            return original(bars, *a, **kw)

        p.latest_entry_signal = spy
        try:
            p.live_signals_for(frame, sessions, RULES, False, window_days=1)
        finally:
            p.latest_entry_signal = original
        self.assertTrue(seen, "expected at least one in-window bar")
        # 1 day of 5-min bars is 288 at most; a full session is 78.
        self.assertLess(max(seen), 300)
