"""Unit tests for trading_bot.strategy.

The pure functions (has_open_position, within_entry_window, clean_price)
are tested directly with plain values -- no mocks needed. The evaluate()
orchestrator is tested with a mocked IB instance and patched I/O helpers,
so no filesystem or broker is touched.
"""

import math
import unittest
from unittest.mock import MagicMock, patch

from trading_bot import strategy


def make_position(symbol: str, size: float) -> MagicMock:
    """Build a minimal stand-in for an ib_async position entry."""
    pos = MagicMock()
    pos.contract.symbol = symbol
    pos.position = size
    return pos


class TestHasOpenPosition(unittest.TestCase):
    """has_open_position: long-only position detection."""

    def test_empty_positions_is_false(self):
        self.assertFalse(strategy.has_open_position([], "AAPL"))

    def test_matching_long_position_is_true(self):
        positions = [make_position("AAPL", 100)]
        self.assertTrue(strategy.has_open_position(positions, "AAPL"))

    def test_other_symbol_is_false(self):
        positions = [make_position("MSFT", 100)]
        self.assertFalse(strategy.has_open_position(positions, "AAPL"))

    def test_short_position_does_not_count(self):
        """size < 0 is a short; the long-only gate ignores it."""
        positions = [make_position("AAPL", -100)]
        self.assertFalse(strategy.has_open_position(positions, "AAPL"))

    def test_flat_position_does_not_count(self):
        """size == 0 entries (closed-out) do not block entry."""
        positions = [make_position("AAPL", 0)]
        self.assertFalse(strategy.has_open_position(positions, "AAPL"))

    def test_match_among_many(self):
        positions = [
            make_position("MSFT", 50),
            make_position("AAPL", 10),
            make_position("NVDA", -5),
        ]
        self.assertTrue(strategy.has_open_position(positions, "AAPL"))


class TestWithinEntryWindow(unittest.TestCase):
    """within_entry_window: inclusive HH:MM window with strict input format."""

    def test_inside_window(self):
        self.assertTrue(strategy.within_entry_window("12:00", "10:05", "15:30"))

    def test_exactly_at_earliest_is_inside(self):
        self.assertTrue(strategy.within_entry_window("10:05", "10:05", "15:30"))

    def test_exactly_at_latest_is_inside(self):
        self.assertTrue(strategy.within_entry_window("15:30", "10:05", "15:30"))

    def test_one_minute_before_earliest_is_outside(self):
        self.assertFalse(strategy.within_entry_window("10:04", "10:05", "15:30"))

    def test_one_minute_after_latest_is_outside(self):
        self.assertFalse(strategy.within_entry_window("15:31", "10:05", "15:30"))

    def test_non_zero_padded_input_raises(self):
        """'9:35' would compare wrong lexicographically -> must be rejected."""
        with self.assertRaises(ValueError):
            strategy.within_entry_window("9:35", "10:05", "15:30")

    def test_garbage_input_raises(self):
        with self.assertRaises(ValueError):
            strategy.within_entry_window("12:00", "start", "15:30")

    def test_out_of_range_time_raises(self):
        with self.assertRaises(ValueError):
            strategy.within_entry_window("25:00", "10:05", "15:30")


class TestCleanPrice(unittest.TestCase):
    """clean_price: primary/fallback selection with NaN/None/<=0 handling."""

    def test_valid_market_price_wins(self):
        self.assertEqual(strategy.clean_price(101.5, 99.0), 101.5)

    def test_nan_market_price_falls_back_to_last(self):
        self.assertEqual(strategy.clean_price(math.nan, 99.0), 99.0)

    def test_none_market_price_falls_back_to_last(self):
        self.assertEqual(strategy.clean_price(None, 99.0), 99.0)

    def test_zero_market_price_falls_back_to_last(self):
        self.assertEqual(strategy.clean_price(0.0, 99.0), 99.0)

    def test_negative_market_price_falls_back_to_last(self):
        self.assertEqual(strategy.clean_price(-1.0, 99.0), 99.0)

    def test_both_unusable_returns_zero(self):
        self.assertEqual(strategy.clean_price(math.nan, math.nan), 0.0)
        self.assertEqual(strategy.clean_price(None, None), 0.0)
        self.assertEqual(strategy.clean_price(0.0, 0.0), 0.0)

    def test_result_is_plain_float(self):
        """Guards the json-encoder issue: numpy floats must come out as float."""
        self.assertIsInstance(strategy.clean_price(101.5, None), float)


@patch("trading_bot.strategy._log_result")  # keep the filesystem out of unit tests
class TestEvaluate(unittest.TestCase):
    """evaluate: orchestration of the pure checks with a mocked IB."""

    def setUp(self):
        self.ib = MagicMock()
        self.ib.positions.return_value = []
        self.rules = {"time_filter": {"earliest_entry_et": "10:05",
                                      "latest_entry_et": "15:30"}}

    def test_fails_when_already_in_position(self, mock_log):
        self.ib.positions.return_value = [make_position("AAPL", 10)]

        result = strategy.evaluate("AAPL", self.ib)

        self.assertFalse(result["pass"])
        self.assertEqual(result["reasons"], ["already in position"])
        self.assertEqual(result["price"], 0.0)
        mock_log.assert_called_once()
        # short-circuit: rules/broker must not be touched after the position gate
        self.ib.qualifyContracts.assert_not_called()

    def test_fails_outside_entry_window(self, mock_log):
        with patch("trading_bot.strategy._load_rules", return_value=self.rules), \
             patch("trading_bot.strategy.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "09:00"

            result = strategy.evaluate("AAPL", self.ib)

        self.assertFalse(result["pass"])
        self.assertEqual(result["reasons"], ["outside entry window 10:05-15:30"])
        self.ib.qualifyContracts.assert_not_called()

    def test_passes_inside_window_with_price(self, mock_log):
        ticker = MagicMock()
        ticker.marketPrice.return_value = 123.45
        ticker.last = 123.40
        self.ib.reqMktData.return_value = ticker
        self.ib.qualifyContracts.return_value = [MagicMock()]

        with patch("trading_bot.strategy._load_rules", return_value=self.rules), \
             patch("trading_bot.strategy.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "12:00"

            result = strategy.evaluate("AAPL", self.ib)

        self.assertTrue(result["pass"])
        self.assertEqual(result["price"], 123.45)
        self.assertIn("time gate ok", result["reasons"])
        self.assertIn("no existing position", result["reasons"])
        # market data subscription must always be cleaned up
        self.ib.cancelMktData.assert_called_once()
        mock_log.assert_called_once()

    def test_pass_with_unusable_price_returns_zero(self, mock_log):
        """A pass without a usable tick yields price 0.0 -- callers must check."""
        ticker = MagicMock()
        ticker.marketPrice.return_value = math.nan
        ticker.last = math.nan
        self.ib.reqMktData.return_value = ticker
        self.ib.qualifyContracts.return_value = [MagicMock()]

        with patch("trading_bot.strategy._load_rules", return_value=self.rules), \
             patch("trading_bot.strategy.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "12:00"

            result = strategy.evaluate("AAPL", self.ib)

        self.assertTrue(result["pass"])
        self.assertEqual(result["price"], 0.0)

    def test_market_data_cancelled_even_if_sleep_raises(self, mock_log):
        """The finally-block must cancel the subscription on errors too."""
        ticker = MagicMock()
        self.ib.reqMktData.return_value = ticker
        self.ib.qualifyContracts.return_value = [MagicMock()]
        self.ib.sleep.side_effect = ConnectionError("socket dropped")

        with patch("trading_bot.strategy._load_rules", return_value=self.rules), \
             patch("trading_bot.strategy.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "12:00"

            with self.assertRaises(ConnectionError):
                strategy.evaluate("AAPL", self.ib)

        self.ib.cancelMktData.assert_called_once()


if __name__ == "__main__":
    unittest.main()
