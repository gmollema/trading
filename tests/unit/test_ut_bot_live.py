"""Unit tests for trading_bot.ut_bot_live (pure live-plumbing helpers for
the UT Bot paper-trading cycle)."""

import unittest
from datetime import datetime

import pandas as pd

from trading_bot import ut_bot_live


def _dt(hh, mm, date="2024-01-02"):  # 2024-01-02 is a Tuesday
    return datetime.fromisoformat(f"{date} {hh:02d}:{mm:02d}:00")


class TestGetMarketStatus(unittest.TestCase):
    def test_saturday_is_closed_all_day(self):
        self.assertEqual(ut_bot_live.get_market_status(_dt(0, 0, "2024-01-06")), "closed")
        self.assertEqual(ut_bot_live.get_market_status(_dt(23, 59, "2024-01-06")), "closed")

    def test_sunday_before_reopen_is_closed(self):
        self.assertEqual(ut_bot_live.get_market_status(_dt(16, 59, "2024-01-07")), "closed")

    def test_sunday_at_and_after_reopen_is_ok(self):
        self.assertEqual(ut_bot_live.get_market_status(_dt(17, 0, "2024-01-07")), "ok")
        self.assertEqual(ut_bot_live.get_market_status(_dt(23, 0, "2024-01-07")), "ok")

    def test_friday_before_close_is_ok(self):
        self.assertEqual(ut_bot_live.get_market_status(_dt(16, 59, "2024-01-05")), "ok")

    def test_friday_at_and_after_close_is_closed(self):
        self.assertEqual(ut_bot_live.get_market_status(_dt(17, 0, "2024-01-05")), "closed")
        self.assertEqual(ut_bot_live.get_market_status(_dt(23, 0, "2024-01-05")), "closed")

    def test_weekday_any_time_is_ok(self):
        self.assertEqual(ut_bot_live.get_market_status(_dt(3, 0)), "ok")
        self.assertEqual(ut_bot_live.get_market_status(_dt(23, 59)), "ok")


class TestBarsFrameToDict(unittest.TestCase):
    def test_roundtrip_shape(self):
        idx = pd.date_range("2024-01-02 09:00", periods=3, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {"High": [1.5, 2.5, 3.5], "Low": [0.5, 1.5, 2.5], "Close": [1.2, 2.2, 3.2]},
            index=idx,
        )
        bars = ut_bot_live.bars_frame_to_dict(frame)
        self.assertEqual(bars["high"], [1.5, 2.5, 3.5])
        self.assertEqual(bars["low"], [0.5, 1.5, 2.5])
        self.assertEqual(bars["close"], [1.2, 2.2, 3.2])
        self.assertEqual(list(bars["date"]), list(idx))


if __name__ == "__main__":
    unittest.main()
