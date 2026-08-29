"""Unit tests for trading_bot.live (pure live-plumbing helpers for the
gap-and-go paper-trading cycle).

The counterpart to test_smc_live.py. These used to live in
tests/unit/cli/test_cycle.py, because the functions used to live in
cli/cycle.py -- which is exactly what made importing them fatal outside
market hours. cli/cycle.py's own tests keep the gate, the broker
plumbing, and everything that needs a mocked IB.
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trading_bot import live

ET = ZoneInfo("America/New_York")
MONDAY = "2024-01-08"
SATURDAY = "2024-01-06"
SUNDAY = "2024-01-07"


def et(day: str, hh: int, mm: int) -> datetime:
    return datetime.fromisoformat(f"{day} {hh:02d}:{mm:02d}:00").replace(tzinfo=ET)


class TestLoadRules(unittest.TestCase):
    def test_reads_the_shipped_file(self):
        rules = live.load_rules()
        self.assertIn("time_filter", rules)

    def test_reads_an_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.json"
            path.write_text(json.dumps({"time_filter": {}}))
            self.assertEqual(live.load_rules(path), {"time_filter": {}})


class TestGetMarketStatus(unittest.TestCase):
    """get_market_status: weekday/time-of-day -> status, boundary-exact.

    Boundaries come from rules.json's time_filter now, so these pass one
    in explicitly rather than asserting against values baked into the
    function. RULES matches the shipped file, which is why every boundary
    below is unchanged from when they were hardcoded -- the config always
    stated the same times it was being ignored in favour of.
    """

    RULES = {"time_filter": {"earliest_entry_et": "10:05", "latest_entry_et": "15:30",
                             "force_close_et": "15:51"}}

    def _status(self, day, hh, mm, rules=None):
        return live.get_market_status(et(day, hh, mm), rules or self.RULES)

    def test_saturday_is_weekend(self):
        self.assertEqual(self._status(SATURDAY, 12, 0), "weekend")

    def test_sunday_is_weekend(self):
        self.assertEqual(self._status(SUNDAY, 12, 0), "weekend")

    def test_before_the_open_is_too_early(self):
        self.assertEqual(self._status(MONDAY, 9, 29), "too_early")

    def test_from_the_open_positions_are_managed(self):
        """Was "too_early" until 10:00, which exits the whole cycle -- so a
        position surviving a failed force-close went unmanaged through the
        first half hour."""
        self.assertEqual(self._status(MONDAY, 9, 30), "manage_only")
        self.assertEqual(self._status(MONDAY, 9, 59), "manage_only")

    def test_10am_exactly_is_manage_only(self):
        self.assertEqual(self._status(MONDAY, 10, 0), "manage_only")

    def test_1004_is_manage_only(self):
        self.assertEqual(self._status(MONDAY, 10, 4), "manage_only")

    def test_1005_exactly_is_ok(self):
        self.assertEqual(self._status(MONDAY, 10, 5), "ok")

    def test_1529_is_ok(self):
        self.assertEqual(self._status(MONDAY, 15, 29), "ok")

    def test_1530_exactly_is_manage_only(self):
        """Entry window closes at 15:30 -- that minute itself is manage_only, not ok."""
        self.assertEqual(self._status(MONDAY, 15, 30), "manage_only")

    def test_1550_is_manage_only(self):
        self.assertEqual(self._status(MONDAY, 15, 50), "manage_only")

    def test_1551_exactly_is_force_close(self):
        self.assertEqual(self._status(MONDAY, 15, 51), "force_close")

    def test_1600_exactly_is_force_close(self):
        self.assertEqual(self._status(MONDAY, 16, 0), "force_close")

    def test_1601_is_closed(self):
        self.assertEqual(self._status(MONDAY, 16, 1), "closed")

    def test_the_config_actually_moves_the_boundaries(self):
        """The regression this replaces: all four times were hardcoded, so
        rules.json's time_filter read as authoritative while changing it
        did nothing whatsoever."""
        shifted = {"time_filter": {"earliest_entry_et": "09:45", "latest_entry_et": "14:00",
                                   "force_close_et": "15:00"}}
        self.assertEqual(self._status(MONDAY, 9, 45, shifted), "ok")
        self.assertEqual(self._status(MONDAY, 13, 59, shifted), "ok")
        self.assertEqual(self._status(MONDAY, 14, 0, shifted), "manage_only")
        self.assertEqual(self._status(MONDAY, 15, 0, shifted), "force_close")

    def test_the_session_still_bounds_the_config(self):
        """An entry window opening before the bell means "from the bell"."""
        early = {"time_filter": {"earliest_entry_et": "08:00", "latest_entry_et": "15:30",
                                 "force_close_et": "15:51"}}
        self.assertEqual(self._status(MONDAY, 9, 0, early), "too_early")
        self.assertEqual(self._status(MONDAY, 9, 30, early), "ok")

    def test_it_matches_the_shipped_rules_file(self):
        """The values under test are the ones actually in use."""
        shipped = json.loads(Path("rules.json").read_text())
        for hh, mm in ((9, 29), (9, 30), (10, 5), (15, 30), (15, 51), (16, 1)):
            self.assertEqual(
                live.get_market_status(et(MONDAY, hh, mm), shipped),
                self._status(MONDAY, hh, mm),
                f"{hh}:{mm:02d}",
            )


class TestComputeSwingLows(unittest.TestCase):
    """compute_swing_lows: a bar is a swing low iff lower than the 2 bars
    before AND the 2 bars after it (strict <)."""

    def _bars(self, lows):
        return pd.DataFrame({"Low": lows})

    def test_single_swing_low_detected(self):
        bars = self._bars([10, 8, 3, 8, 10])
        self.assertEqual(live.compute_swing_lows(bars), [3.0])

    def test_monotonic_series_has_no_swing_low(self):
        bars = self._bars([10, 9, 8, 7, 6])
        self.assertEqual(live.compute_swing_lows(bars), [])

    def test_equal_neighbor_does_not_count_as_swing_low(self):
        """Strict '<' -- a tie with a neighbor's min does not qualify."""
        bars = self._bars([10, 9, 5, 9, 10, 9, 4, 9, 10])
        self.assertEqual(live.compute_swing_lows(bars), [5.0, 4.0])

    def test_too_few_bars_returns_empty(self):
        bars = self._bars([10, 9, 8, 7])
        self.assertEqual(live.compute_swing_lows(bars), [])

    def test_results_are_plain_floats(self):
        bars = self._bars([10, 8, 3, 8, 10])
        for value in live.compute_swing_lows(bars):
            self.assertIsInstance(value, float)


class TestReadWatchlist(unittest.TestCase):
    """read_watchlist: comment/blank-line stripping, inline-comment
    trailing text, class-share tickers with embedded spaces."""

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(live.read_watchlist(Path("does_not_exist_watchlist.txt")), [])

    def test_parses_tickers_skipping_blank_and_comment_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "watchlist.txt"
            path.write_text(
                "# header comment\n"
                "\n"
                "AAPL  # gap +5.00%\n"
                "MSFT\n"
                "  \n"
                "# another comment\n"
                "NVDA # trailing comment\n"
            )
            self.assertEqual(live.read_watchlist(path), ["AAPL", "MSFT", "NVDA"])

    def test_class_share_ticker_with_space_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "watchlist.txt"
            path.write_text("BRK B  # class share\nTRGP\n")
            self.assertEqual(live.read_watchlist(path), ["BRK B", "TRGP"])
