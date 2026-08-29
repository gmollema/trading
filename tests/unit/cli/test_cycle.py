"""Unit tests for the pure/near-pure helper functions in trading_bot.cli.cycle.

cycle.py runs a market-status gate at MODULE IMPORT TIME (by design -- see
its module docstring -- so Task Scheduler runs exit in well under a second
outside market hours, before the heavy numpy/yfinance/ib_async imports).
That means a plain `import` of this module during a test run outside
10:00-16:00 ET on a weekday would call sys.exit(0) mid-import and abort
collection. We neutralize sys.exit for just that one import statement so
the module always loads fully regardless of when the suite runs.

Covers: get_market_status, _cast_bools, ibkr_to_yahoo, compute_swing_lows,
evaluate_entry_filters (get_daily_context/get_intraday_context patched
out, since those hit yfinance), _cancel_stop, manage_position,
force_close_all, and entry_scan's concurrent-position cap. Order
placement/cancellation is mocked at the _cancel_stop/_sell_market/
_place_stop boundary rather than simulated against a fake ib_async IB, to
keep these tests focused on cycle.py's own branching/sequencing logic.

log_event and notify are patched in every test that exercises them:
log_event would otherwise append to the real safety-check-log.json, and
notify would otherwise fire a real Telegram message via the credentials
in .env.
"""

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

with patch("sys.exit"):
    from trading_bot.cli import cycle

ET = ZoneInfo("America/New_York")

# 2026-07-13 is a Monday; 07-11/07-12 are the preceding Sat/Sun.
MONDAY = (2026, 7, 13)
SATURDAY = (2026, 7, 11)
SUNDAY = (2026, 7, 12)


def et(date: tuple[int, int, int], hour: int, minute: int) -> datetime:
    year, month, day = date
    return datetime(year, month, day, hour, minute, tzinfo=ET)


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
        return cycle.get_market_status(et(day, hh, mm), rules or self.RULES)

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
                cycle.get_market_status(et(MONDAY, hh, mm), shipped),
                self._status(MONDAY, hh, mm),
                f"{hh}:{mm:02d}",
            )


class TestFastExitCheck(unittest.TestCase):
    """The pre-import gate may only know the session, never the entry
    window -- exiting here skips position management too."""

    def _at(self, day, hh, mm):
        with patch("trading_bot.cli.cycle.datetime") as fake:
            fake.now.return_value = et(day, hh, mm)
            return cycle._fast_exit_check()

    def test_before_the_open_exits(self):
        self.assertEqual(self._at(MONDAY, 9, 29), "too_early")

    def test_from_the_open_it_does_not_exit(self):
        for hh, mm in ((9, 30), (9, 45), (10, 0)):
            self.assertIsNone(self._at(MONDAY, hh, mm))

    def test_after_the_close_exits(self):
        self.assertEqual(self._at(MONDAY, 16, 1), "closed")

    def test_weekend_exits(self):
        self.assertEqual(self._at(SATURDAY, 12, 0), "weekend")

    def test_it_agrees_with_get_market_status_on_the_session_bounds(self):
        """Two gates, one session: disagreement means the cycle either
        dies early or wakes up with nothing to do."""
        shipped = json.loads(Path("rules.json").read_text())
        for hh, mm in ((9, 29), (9, 30), (16, 0), (16, 1)):
            exits_late = cycle.get_market_status(et(MONDAY, hh, mm), shipped) in ("too_early", "closed")
            self.assertEqual(self._at(MONDAY, hh, mm) is not None, exits_late, f"{hh}:{mm:02d}")


class TestCastBools(unittest.TestCase):
    """_cast_bools: recursively normalize bool/np.bool_ for json.dumps."""

    def test_plain_bool_passes_through(self):
        self.assertIs(cycle._cast_bools(True), True)

    def test_numpy_bool_becomes_plain_bool(self):
        result = cycle._cast_bools(np.bool_(True))
        self.assertIs(type(result), bool)
        self.assertTrue(result)

    def test_non_bool_values_are_unchanged(self):
        self.assertEqual(cycle._cast_bools(3.14), 3.14)
        self.assertEqual(cycle._cast_bools("AAPL"), "AAPL")
        self.assertIsNone(cycle._cast_bools(None))

    def test_recurses_into_dict_and_list(self):
        nested = {
            "flag": np.bool_(False),
            "items": [np.bool_(True), 1, "x"],
        }
        result = cycle._cast_bools(nested)

        self.assertIs(type(result["flag"]), bool)
        self.assertFalse(result["flag"])
        self.assertIs(type(result["items"][0]), bool)
        self.assertEqual(result["items"][1:], [1, "x"])


class TestIbkrToYahoo(unittest.TestCase):
    """ibkr_to_yahoo: class-share space -> hyphen for yfinance lookups."""

    def test_class_share_space_becomes_hyphen(self):
        self.assertEqual(cycle.ibkr_to_yahoo("BRK B"), "BRK-B")

    def test_plain_ticker_is_unchanged(self):
        self.assertEqual(cycle.ibkr_to_yahoo("AAPL"), "AAPL")


class TestComputeSwingLows(unittest.TestCase):
    """compute_swing_lows: a bar is a swing low iff lower than the 2 bars
    before AND the 2 bars after it (strict <)."""

    def _bars(self, lows):
        return pd.DataFrame({"Low": lows})

    def test_single_swing_low_detected(self):
        bars = self._bars([10, 8, 3, 8, 10])
        self.assertEqual(cycle.compute_swing_lows(bars), [3.0])

    def test_monotonic_series_has_no_swing_low(self):
        bars = self._bars([10, 9, 8, 7, 6])
        self.assertEqual(cycle.compute_swing_lows(bars), [])

    def test_equal_neighbor_does_not_count_as_swing_low(self):
        """Strict '<' -- a tie with a neighbor's min does not qualify."""
        bars = self._bars([10, 9, 5, 9, 10, 9, 4, 9, 10])
        self.assertEqual(cycle.compute_swing_lows(bars), [5.0, 4.0])

    def test_too_few_bars_returns_empty(self):
        bars = self._bars([10, 9, 8, 7])
        self.assertEqual(cycle.compute_swing_lows(bars), [])

    def test_results_are_plain_floats(self):
        bars = self._bars([10, 8, 3, 8, 10])
        for value in cycle.compute_swing_lows(bars):
            self.assertIsInstance(value, float)


def make_rules(**overrides):
    rules = {
        "daily_filters": {
            "D1_above_prior_day_high": True,
            "D2_prior_close_above_sma200": True,
            "D3_min_gap_pct_from_prior_close": 3.0,
        },
        "intraday_filters": {
            "I1_above_premarket_high": True,
            "I2_above_today_hod": True,
            "I3_rvol_min": 2.0,
            "I3_rvol_lookback_days": 14,
        },
    }
    rules.update(overrides)
    return rules


def make_daily_ctx(**overrides):
    ctx = {"prior_day_high": 100.0, "prior_day_close": 95.0, "sma200": 90.0}
    ctx.update(overrides)
    return ctx


def make_intraday_ctx(**overrides):
    ctx = {
        "premarket_high": 101.0,
        "today_hod": 103.0,
        "today_lod": 98.0,
        "latest_price": 104.0,  # price used throughout: above all thresholds above
        "rvol": 3.0,
    }
    ctx.update(overrides)
    return ctx


@patch("trading_bot.cli.cycle.get_intraday_context")
@patch("trading_bot.cli.cycle.get_daily_context")
class TestEvaluateEntryFilters(unittest.TestCase):
    """evaluate_entry_filters: D1-D3 / I1-I3 gate, with yfinance-backed
    context builders patched out."""

    def test_all_filters_pass(self, mock_daily, mock_intraday):
        mock_daily.return_value = make_daily_ctx()
        mock_intraday.return_value = make_intraday_ctx()

        passed, reasons, details = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertEqual(details["price"], 104.0)

    def test_insufficient_daily_data_fails_closed(self, mock_daily, mock_intraday):
        mock_daily.return_value = None

        passed, reasons, details = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertFalse(passed)
        self.assertEqual(reasons, ["insufficient daily data"])
        self.assertEqual(details, {})
        mock_intraday.assert_not_called()

    def test_insufficient_intraday_data_fails_closed(self, mock_daily, mock_intraday):
        mock_daily.return_value = make_daily_ctx()
        mock_intraday.return_value = None

        passed, reasons, details = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertFalse(passed)
        self.assertEqual(reasons, ["insufficient intraday data"])
        self.assertEqual(details, {})

    def test_d1_fail_price_not_above_prior_day_high(self, mock_daily, mock_intraday):
        mock_daily.return_value = make_daily_ctx(prior_day_high=110.0)
        mock_intraday.return_value = make_intraday_ctx()

        passed, reasons, _ = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertFalse(passed)
        self.assertIn("D1 fail: price not above prior day high", reasons)

    def test_d2_fail_prior_close_not_above_sma200(self, mock_daily, mock_intraday):
        mock_daily.return_value = make_daily_ctx(sma200=110.0)
        mock_intraday.return_value = make_intraday_ctx()

        passed, reasons, _ = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertFalse(passed)
        self.assertIn("D2 fail: prior close not above SMA200", reasons)

    def test_d3_fail_gap_below_threshold(self, mock_daily, mock_intraday):
        mock_daily.return_value = make_daily_ctx(prior_day_close=102.0)  # gap ~1.96% < 3.0%
        mock_intraday.return_value = make_intraday_ctx()

        passed, reasons, _ = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertFalse(passed)
        self.assertTrue(any(r.startswith("D3 fail") for r in reasons))

    def test_i1_fail_price_not_above_premarket_high(self, mock_daily, mock_intraday):
        mock_daily.return_value = make_daily_ctx()
        mock_intraday.return_value = make_intraday_ctx(premarket_high=105.0)

        passed, reasons, _ = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertFalse(passed)
        self.assertIn("I1 fail: price not above premarket high", reasons)

    def test_i1_fail_when_premarket_high_missing(self, mock_daily, mock_intraday):
        mock_daily.return_value = make_daily_ctx()
        mock_intraday.return_value = make_intraday_ctx(premarket_high=None)

        passed, reasons, _ = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertFalse(passed)
        self.assertIn("I1 fail: price not above premarket high", reasons)

    def test_i2_fail_price_below_today_hod(self, mock_daily, mock_intraday):
        mock_daily.return_value = make_daily_ctx()
        mock_intraday.return_value = make_intraday_ctx(today_hod=110.0)

        passed, reasons, _ = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertFalse(passed)
        self.assertIn("I2 fail: price not at/above today HOD", reasons)

    def test_i3_fail_rvol_below_minimum(self, mock_daily, mock_intraday):
        mock_daily.return_value = make_daily_ctx()
        mock_intraday.return_value = make_intraday_ctx(rvol=1.0)

        passed, reasons, _ = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertFalse(passed)
        self.assertIn("I3 fail: rvol 1.00 < 2.0", reasons)

    def test_i3_fail_when_rvol_missing(self, mock_daily, mock_intraday):
        mock_daily.return_value = make_daily_ctx()
        mock_intraday.return_value = make_intraday_ctx(rvol=None)

        passed, reasons, _ = cycle.evaluate_entry_filters("AAPL", make_rules())

        self.assertFalse(passed)
        self.assertIn("I3 fail: rvol N/A < 2.0", reasons)


def make_order(order_id):
    order = MagicMock()
    order.orderId = order_id
    return order


class TestCancelStop(unittest.TestCase):
    """_cancel_stop: cancel + confirm the resting stop is actually gone
    before the caller is allowed to treat it as safe to replace."""

    def setUp(self):
        self.ib = MagicMock()

    @patch("trading_bot.cli.cycle.log_event")
    def test_no_stop_order_id_returns_true_without_ib_calls(self, mock_log):
        self.assertTrue(cycle._cancel_stop(self.ib, None))
        self.ib.reqOpenOrders.assert_not_called()

    @patch("trading_bot.cli.cycle.log_event")
    def test_stop_not_in_open_orders_returns_true(self, mock_log):
        """Already gone (filled or previously cancelled) -- nothing to do."""
        self.ib.openOrders.return_value = []

        self.assertTrue(cycle._cancel_stop(self.ib, 111))
        self.ib.cancelOrder.assert_not_called()

    @patch("trading_bot.cli.cycle.log_event")
    def test_confirmed_cancel_returns_true(self, mock_log):
        order = make_order(111)
        # First lookup finds the order; the confirm-loop lookup finds it gone.
        self.ib.openOrders.side_effect = [[order], []]

        result = cycle._cancel_stop(self.ib, 111)

        self.assertTrue(result)
        self.ib.cancelOrder.assert_called_once_with(order)

    @patch("trading_bot.cli.cycle.log_event")
    def test_cancel_order_raises_returns_false(self, mock_log):
        order = make_order(111)
        self.ib.openOrders.return_value = [order]
        self.ib.cancelOrder.side_effect = ConnectionError("boom")

        result = cycle._cancel_stop(self.ib, 111)

        self.assertFalse(result)
        mock_log.assert_called_once()

    @patch("trading_bot.cli.cycle.CANCEL_CONFIRM_TIMEOUT_SECS", 0.05)
    @patch("trading_bot.cli.cycle.log_event")
    def test_unconfirmed_cancel_times_out_returns_false(self, mock_log):
        order = make_order(111)
        self.ib.openOrders.return_value = [order]  # always present -- never confirms

        result = cycle._cancel_stop(self.ib, 111)

        self.assertFalse(result)
        mock_log.assert_called_once()


def make_pos(**overrides):
    pos = {
        "symbol": "AAPL",
        "entry_price": 100.0,
        "qty": 30,
        "initial_stop": 95.0,
        "current_stop_price": 95.0,
        "stop_order_id": 111,
        "state": "pre_breakeven",
        "R": 5.0,
    }
    pos.update(overrides)
    return pos


def make_exit_rules(partial_trigger_R=0.75, breakeven_trigger_R=1.0):
    return {
        "exit": {
            "breakeven_trigger_R": breakeven_trigger_R,
            "partial_profit_trigger_R": partial_trigger_R,
        }
    }


def make_trade(status):
    trade = MagicMock()
    trade.orderStatus.status = status
    return trade


@patch("trading_bot.cli.cycle.notify")
@patch("trading_bot.cli.cycle.log_event")
@patch("trading_bot.cli.cycle._place_stop")
@patch("trading_bot.cli.cycle._sell_market")
@patch("trading_bot.cli.cycle._cancel_stop")
@patch("trading_bot.cli.cycle.get_current_price")
class TestManagePositionPartialProfit(unittest.TestCase):
    """manage_position: pre_breakeven -> partial-profit branch. Covers the
    cancel-before-sell ordering fix and the failed-sell stop-restoration
    safety net."""

    def test_cancel_before_sell_ordering(
        self, mock_price, mock_cancel, mock_sell, mock_place_stop, mock_log, mock_notify
    ):
        """The old full-qty stop must be cancelled (and confirmed) BEFORE
        the partial market sell fires -- selling first would leave the old
        stop resting alongside the market sell, risking an oversell if both
        fire."""
        call_order = []
        mock_cancel.side_effect = lambda *a, **k: call_order.append("cancel") or True
        mock_sell.side_effect = lambda *a, **k: call_order.append("sell") or make_trade("Filled")
        mock_price.return_value = 103.75  # entry + 0.75*R -- partial trigger only
        mock_place_stop.return_value = 222

        cycle.manage_position(MagicMock(), make_pos(), make_exit_rules())

        self.assertEqual(call_order, ["cancel", "sell"])

    def test_partial_skipped_when_cancel_not_confirmed(
        self, mock_price, mock_cancel, mock_sell, mock_place_stop, mock_log, mock_notify
    ):
        mock_cancel.return_value = False
        mock_price.return_value = 103.75

        result = cycle.manage_position(MagicMock(), make_pos(), make_exit_rules())

        mock_sell.assert_not_called()
        mock_place_stop.assert_not_called()
        self.assertEqual(result["state"], "pre_breakeven")
        self.assertEqual(result["qty"], 30)

    def test_partial_taken_on_success(
        self, mock_price, mock_cancel, mock_sell, mock_place_stop, mock_log, mock_notify
    ):
        mock_cancel.return_value = True
        mock_sell.return_value = make_trade("Filled")
        mock_price.return_value = 103.75  # only the partial trigger reached
        mock_place_stop.return_value = 222

        result = cycle.manage_position(MagicMock(), make_pos(), make_exit_rules())

        self.assertEqual(result["state"], "post_breakeven_partial_done")
        self.assertEqual(result["qty"], 20)  # 30 - ceil(30/3)=10
        self.assertAlmostEqual(result["current_stop_price"], 99.0)  # entry*0.99
        self.assertEqual(result["stop_order_id"], 222)

    def test_fast_move_past_both_thresholds_gets_breakeven_stop(
        self, mock_price, mock_cancel, mock_sell, mock_place_stop, mock_log, mock_notify
    ):
        """Price jumping straight past both the partial and breakeven
        triggers within one poll must still take the partial exit AND get
        the full breakeven stop, not silently just the discounted 0.99x
        stop -- otherwise the scheduled partial exit is effectively lost."""
        mock_cancel.return_value = True
        mock_sell.return_value = make_trade("Filled")
        mock_price.return_value = 105.5  # entry + 1.0*R = 105.0 -- past breakeven too
        mock_place_stop.return_value = 222

        result = cycle.manage_position(MagicMock(), make_pos(), make_exit_rules())

        self.assertEqual(result["state"], "post_breakeven_partial_done")
        self.assertEqual(result["current_stop_price"], 100.0)  # full entry, not *0.99

    def test_failed_sell_restores_full_stop(
        self, mock_price, mock_cancel, mock_sell, mock_place_stop, mock_log, mock_notify
    ):
        """If the partial sell is rejected after the old stop was already
        cancelled, the position must not be left unprotected."""
        mock_cancel.return_value = True
        mock_sell.return_value = make_trade("Rejected")
        mock_price.return_value = 103.75
        mock_place_stop.return_value = 333
        ib = MagicMock()

        result = cycle.manage_position(ib, make_pos(), make_exit_rules())

        self.assertEqual(result["qty"], 30)  # unchanged -- sell never went through
        self.assertEqual(result["state"], "pre_breakeven")  # not advanced
        self.assertEqual(result["stop_order_id"], 333)
        mock_place_stop.assert_called_once_with(ib, "AAPL", 30, 95.0)
        mock_notify.assert_called_once()
        self.assertIn("PARTIAL FAILED", mock_notify.call_args[0][0])


@patch("trading_bot.cli.cycle.notify")
@patch("trading_bot.cli.cycle.log_event")
@patch("trading_bot.cli.cycle._place_stop")
@patch("trading_bot.cli.cycle._sell_market")
@patch("trading_bot.cli.cycle._cancel_stop")
@patch("trading_bot.cli.cycle.get_current_price")
class TestManagePositionBreakevenOnlyFallback(unittest.TestCase):
    """Defensive elif branch: only reachable if rules.json is ever
    configured with breakeven_trigger_R < partial_profit_trigger_R."""

    def test_breakeven_only_when_partial_trigger_is_higher(
        self, mock_price, mock_cancel, mock_sell, mock_place_stop, mock_log, mock_notify
    ):
        mock_cancel.return_value = True
        mock_price.return_value = 105.0  # reaches breakeven (1.0R) but not partial (1.5R)
        mock_place_stop.return_value = 999

        rules = make_exit_rules(partial_trigger_R=1.5, breakeven_trigger_R=1.0)
        result = cycle.manage_position(MagicMock(), make_pos(), rules)

        mock_sell.assert_not_called()
        self.assertEqual(result["state"], "post_breakeven_no_partial")
        self.assertEqual(result["current_stop_price"], 100.0)
        self.assertEqual(result["qty"], 30)  # unchanged, no partial sold

    def test_breakeven_skipped_when_cancel_not_confirmed(
        self, mock_price, mock_cancel, mock_sell, mock_place_stop, mock_log, mock_notify
    ):
        mock_cancel.return_value = False
        mock_price.return_value = 105.0

        rules = make_exit_rules(partial_trigger_R=1.5, breakeven_trigger_R=1.0)
        result = cycle.manage_position(MagicMock(), make_pos(), rules)

        mock_place_stop.assert_not_called()
        self.assertEqual(result["state"], "pre_breakeven")


@patch("trading_bot.cli.cycle.notify")
@patch("trading_bot.cli.cycle.log_event")
@patch("trading_bot.cli.cycle._place_stop")
@patch("trading_bot.cli.cycle._cancel_stop")
@patch("trading_bot.cli.cycle.get_5m_bars_today")
@patch("trading_bot.cli.cycle.get_current_price")
class TestManagePositionTrailingStop(unittest.TestCase):
    """manage_position: post_breakeven -> swing-low trailing stop ratchet."""

    def _bars_with_swing_low(self, low_value):
        # low_value must be lower than both neighboring pairs to register as
        # a swing low (see TestComputeSwingLows).
        return pd.DataFrame({"Low": [110, 105, low_value, 105, 110]})

    def test_ratchet_skipped_when_cancel_not_confirmed(
        self, mock_price, mock_bars, mock_cancel, mock_place_stop, mock_log, mock_notify
    ):
        mock_price.return_value = 110.0
        mock_bars.return_value = self._bars_with_swing_low(100.5)  # candidate 100.49 > current 99.0
        mock_cancel.return_value = False

        pos = make_pos(state="post_breakeven_partial_done", current_stop_price=99.0, qty=20)
        result = cycle.manage_position(MagicMock(), pos, make_exit_rules())

        mock_place_stop.assert_not_called()
        self.assertEqual(result["current_stop_price"], 99.0)  # unchanged

    def test_ratchet_applied_on_confirmed_cancel(
        self, mock_price, mock_bars, mock_cancel, mock_place_stop, mock_log, mock_notify
    ):
        mock_price.return_value = 110.0
        mock_bars.return_value = self._bars_with_swing_low(100.5)
        mock_cancel.return_value = True
        mock_place_stop.return_value = 777

        pos = make_pos(state="post_breakeven_partial_done", current_stop_price=99.0, qty=20)
        result = cycle.manage_position(MagicMock(), pos, make_exit_rules())

        self.assertAlmostEqual(result["current_stop_price"], 100.49)
        self.assertEqual(result["stop_order_id"], 777)


@patch("trading_bot.cli.cycle.notify")
@patch("trading_bot.cli.cycle.log_event")
@patch("trading_bot.cli.cycle._sell_market")
@patch("trading_bot.cli.cycle._cancel_stop")
class TestForceCloseAll(unittest.TestCase):
    """force_close_all: a rejected/failed EOD sell must not be silently
    forgotten -- it should stay tracked and raise a high-priority alert."""

    def test_successful_close_removes_position(self, mock_cancel, mock_sell, mock_log, mock_notify):
        mock_cancel.return_value = True
        mock_sell.return_value = make_trade("Filled")

        remaining = cycle.force_close_all(MagicMock(), [make_pos()])

        self.assertEqual(remaining, [])
        mock_notify.assert_not_called()

    def test_failed_close_keeps_position_and_alerts(self, mock_cancel, mock_sell, mock_log, mock_notify):
        mock_cancel.return_value = True
        mock_sell.return_value = make_trade("Rejected")
        pos = make_pos()

        remaining = cycle.force_close_all(MagicMock(), [pos])

        self.assertEqual(remaining, [pos])
        mock_notify.assert_called_once()
        self.assertIn("FORCE CLOSE FAILED", mock_notify.call_args[0][0])
        self.assertEqual(mock_notify.call_args[0][2], "high")

    def test_mixed_results_only_keeps_the_failed_one(self, mock_cancel, mock_sell, mock_log, mock_notify):
        mock_cancel.return_value = True
        good = make_pos(symbol="AAPL")
        bad = make_pos(symbol="MSFT")
        mock_sell.side_effect = [make_trade("Filled"), make_trade("Cancelled")]

        remaining = cycle.force_close_all(MagicMock(), [good, bad])

        self.assertEqual(remaining, [bad])


def make_position_stub(symbol):
    p = MagicMock()
    p.contract.symbol = symbol
    return p


@patch("trading_bot.cli.cycle.notify")
@patch("trading_bot.cli.cycle.log_event")
@patch("trading_bot.cli.cycle._place_stop")
@patch("trading_bot.cli.cycle.subprocess.run")
@patch("trading_bot.cli.cycle.evaluate_entry_filters")
@patch("trading_bot.cli.cycle.read_watchlist")
@patch("trading_bot.cli.cycle.count_today_buys")
class TestEntryScanConcurrencyCap(unittest.TestCase):
    """entry_scan: rules.json's risk.max_concurrent_positions must actually
    cap how many positions can be open at once, not just dedup by symbol."""

    def _make_ib(self, held_symbols):
        ib = MagicMock()
        ib.positions.return_value = [make_position_stub(s) for s in held_symbols]
        return ib

    def test_scan_skipped_when_already_at_cap(
        self, mock_count, mock_watchlist, mock_evaluate, mock_subprocess, mock_place_stop, mock_log, mock_notify
    ):
        mock_count.return_value = 0
        mock_watchlist.return_value = ["AAPL", "MSFT"]
        ib = self._make_ib(["AAPL", "MSFT"])  # already 2 held, cap is 2
        rules = {"risk": {"max_concurrent_positions": 2}}
        env = {"max_trades_per_day": 10}

        result = cycle.entry_scan(ib, rules, env)

        self.assertEqual(result, [])
        mock_evaluate.assert_not_called()

    def test_scan_stops_once_cap_reached_mid_loop(
        self, mock_count, mock_watchlist, mock_evaluate, mock_subprocess, mock_place_stop, mock_log, mock_notify
    ):
        mock_count.return_value = 0
        mock_watchlist.return_value = ["AAA", "BBB", "CCC"]
        ib = self._make_ib(["ZZZ"])  # 1 already held, cap is 2 -> room for exactly 1 more
        rules = {"risk": {"max_concurrent_positions": 2}}
        env = {"max_trades_per_day": 10, "portfolio_value_usd": 100000, "max_risk_per_trade_pct": 1.0}

        mock_evaluate.return_value = (
            True,
            [],
            {"price": 100.0, "today_lod": 95.0, "today_hod": 101.0, "gap_pct": 5.0},
        )
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_place_stop.return_value = 42

        result = cycle.entry_scan(ib, rules, env)

        self.assertEqual(len(result), 1)  # cap=2, 1 already held -> only 1 new entry allowed
        self.assertEqual(mock_evaluate.call_count, 1)  # loop stopped after hitting the cap

    def test_no_cap_configured_does_not_restrict(
        self, mock_count, mock_watchlist, mock_evaluate, mock_subprocess, mock_place_stop, mock_log, mock_notify
    ):
        """rules.json without a risk.max_concurrent_positions key must not
        block entries (back-compat with configs that omit it)."""
        mock_count.return_value = 0
        mock_watchlist.return_value = []
        ib = self._make_ib(["A", "B", "C", "D", "E"])
        rules = {"risk": {}}
        env = {"max_trades_per_day": 10}

        result = cycle.entry_scan(ib, rules, env)

        self.assertEqual(result, [])  # empty watchlist -- just confirming no early skip/exception


def make_fill(order_id, price, fill_time=None, side="SLD"):
    fill = MagicMock()
    fill.execution.orderId = order_id
    fill.execution.price = price
    fill.execution.side = side
    fill.time = fill_time
    return fill


@patch("trading_bot.cli.cycle.notify")
@patch("trading_bot.cli.cycle.log_event")
class TestCheckStopOuts(unittest.TestCase):
    """check_stop_outs: match fills to positions by stop_order_id (NOT
    quantity -- quantity matching causes false stop-outs after partials),
    scoped to the last hour, with naive fill timestamps normalized to UTC."""

    def test_no_positions_returns_empty(self, mock_log, mock_notify):
        ib = MagicMock()
        ib.fills.return_value = []

        remaining, events = cycle.check_stop_outs(ib, [])

        self.assertEqual(remaining, [])
        self.assertEqual(events, [])

    def test_fills_query_failure_is_logged_and_treated_as_no_fills(self, mock_log, mock_notify):
        ib = MagicMock()
        ib.fills.side_effect = ConnectionError("boom")
        pos = make_pos()

        remaining, events = cycle.check_stop_outs(ib, [pos])

        self.assertEqual(remaining, [pos])
        self.assertEqual(events, [])
        mock_log.assert_called_once()
        self.assertEqual(mock_log.call_args[0][0]["event"], "fills_query_failed")

    def test_matching_recent_fill_marks_position_stopped_out(self, mock_log, mock_notify):
        now = datetime.now(timezone.utc)
        ib = MagicMock()
        ib.fills.return_value = [make_fill(order_id=111, price=94.5, fill_time=now)]
        pos = make_pos(stop_order_id=111, entry_price=100.0, qty=30)

        remaining, events = cycle.check_stop_outs(ib, [pos])

        self.assertEqual(remaining, [])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "stopped_out")
        self.assertEqual(events[0]["fill_price"], 94.5)
        self.assertAlmostEqual(events[0]["pnl"], (94.5 - 100.0) * 30)
        mock_notify.assert_called_once()

    def test_non_matching_fill_leaves_position_in_remaining(self, mock_log, mock_notify):
        now = datetime.now(timezone.utc)
        ib = MagicMock()
        ib.fills.return_value = [make_fill(order_id=999, price=94.5, fill_time=now)]
        pos = make_pos(stop_order_id=111)

        remaining, events = cycle.check_stop_outs(ib, [pos])

        self.assertEqual(remaining, [pos])
        self.assertEqual(events, [])
        mock_notify.assert_not_called()

    def test_stale_fill_older_than_one_hour_is_ignored(self, mock_log, mock_notify):
        stale_time = datetime.now(timezone.utc) - timedelta(hours=2)
        ib = MagicMock()
        ib.fills.return_value = [make_fill(order_id=111, price=94.5, fill_time=stale_time)]
        pos = make_pos(stop_order_id=111)

        remaining, events = cycle.check_stop_outs(ib, [pos])

        self.assertEqual(remaining, [pos])
        self.assertEqual(events, [])

    def test_naive_datetime_fill_time_is_normalized_to_utc(self, mock_log, mock_notify):
        """A naive fill.time (no tzinfo) must be treated as UTC, not raise
        on comparison with the aware one_hour_ago cutoff."""
        naive_recent = datetime.now(timezone.utc).replace(tzinfo=None)
        ib = MagicMock()
        ib.fills.return_value = [make_fill(order_id=111, price=94.5, fill_time=naive_recent)]
        pos = make_pos(stop_order_id=111)

        remaining, events = cycle.check_stop_outs(ib, [pos])

        self.assertEqual(remaining, [])
        self.assertEqual(len(events), 1)

    def test_fill_with_missing_time_is_treated_as_recent(self, mock_log, mock_notify):
        ib = MagicMock()
        ib.fills.return_value = [make_fill(order_id=111, price=94.5, fill_time=None)]
        pos = make_pos(stop_order_id=111)

        remaining, events = cycle.check_stop_outs(ib, [pos])

        self.assertEqual(remaining, [])
        self.assertEqual(len(events), 1)

    def test_position_without_stop_order_id_never_matches(self, mock_log, mock_notify):
        now = datetime.now(timezone.utc)
        ib = MagicMock()
        # A fill whose orderId happens to be None too -- must still not match.
        ib.fills.return_value = [make_fill(order_id=None, price=94.5, fill_time=now)]
        pos = make_pos(stop_order_id=None)

        remaining, events = cycle.check_stop_outs(ib, [pos])

        self.assertEqual(remaining, [pos])
        self.assertEqual(events, [])

    def test_mixed_positions_only_matching_one_is_removed(self, mock_log, mock_notify):
        now = datetime.now(timezone.utc)
        ib = MagicMock()
        ib.fills.return_value = [make_fill(order_id=111, price=94.5, fill_time=now)]
        stopped = make_pos(symbol="AAPL", stop_order_id=111)
        still_open = make_pos(symbol="MSFT", stop_order_id=222)

        remaining, events = cycle.check_stop_outs(ib, [stopped, still_open])

        self.assertEqual(remaining, [still_open])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["symbol"], "AAPL")

    def test_buy_fill_sharing_the_stop_order_id_is_not_treated_as_stopped_out(self, mock_log, mock_notify):
        """Regression: two independent IBKR client connections (this
        process placing the stop vs. trade.py's subprocess placing the
        entry BUY) can end up assigning the SAME order ID to two DIFFERENT
        orders. A BUY fill sharing that ID must never be mistaken for the
        stop-loss having been hit -- a stop-out is always a SELL."""
        now = datetime.now(timezone.utc)
        ib = MagicMock()
        ib.fills.return_value = [make_fill(order_id=4, price=207.44, fill_time=now, side="BOT")]
        pos = make_pos(symbol="CRWD", stop_order_id=4, entry_price=207.43, qty=48)

        remaining, events = cycle.check_stop_outs(ib, [pos])

        self.assertEqual(remaining, [pos])
        self.assertEqual(events, [])
        mock_notify.assert_not_called()


class TestLoadSavePositions(unittest.TestCase):
    """load_positions / save_positions: JSON round-trip, corrupt-file
    fallback, and atomic (tmp + os.replace) writes."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.positions_path = Path(self._tmpdir.name) / "open_positions.json"
        self.patcher = patch.object(cycle, "POSITIONS_PATH", self.positions_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_load_missing_file_returns_empty_list(self):
        self.assertEqual(cycle.load_positions(), [])

    def test_load_after_save_round_trips(self):
        positions = [make_pos(symbol="AAPL"), make_pos(symbol="MSFT")]

        cycle.save_positions(positions)

        self.assertEqual(cycle.load_positions(), positions)

    def test_load_corrupt_json_returns_empty_list(self):
        self.positions_path.write_text("{not valid json")

        self.assertEqual(cycle.load_positions(), [])

    def test_save_casts_numpy_bools(self):
        cycle.save_positions([make_pos(some_flag=np.bool_(True))])

        raw = json.loads(self.positions_path.read_text())
        self.assertIs(raw[0]["some_flag"], True)

    def test_save_leaves_no_leftover_tmp_file(self):
        cycle.save_positions([make_pos()])

        self.assertFalse(self.positions_path.with_suffix(".json.tmp").exists())
        self.assertTrue(self.positions_path.exists())


class TestReadWatchlist(unittest.TestCase):
    """read_watchlist: comment/blank-line stripping, inline-comment
    trailing text, class-share tickers with embedded spaces."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.watchlist_path = Path(self._tmpdir.name) / "watchlist.txt"
        self.patcher = patch.object(cycle, "WATCHLIST_PATH", self.watchlist_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(cycle.read_watchlist(), [])

    def test_parses_tickers_skipping_blank_and_comment_lines(self):
        self.watchlist_path.write_text(
            "# header comment\n"
            "\n"
            "AAPL  # gap +5.00%\n"
            "MSFT\n"
            "  \n"
            "# another comment\n"
            "NVDA # trailing comment\n"
        )

        self.assertEqual(cycle.read_watchlist(), ["AAPL", "MSFT", "NVDA"])

    def test_class_share_ticker_with_space_preserved(self):
        self.watchlist_path.write_text("BRK B  # some note\n")

        self.assertEqual(cycle.read_watchlist(), ["BRK B"])


class TestCountTodayBuys(unittest.TestCase):
    """count_today_buys: only BUY rows whose timestamp_iso date matches
    "today" (ET) count."""

    FIELDNAMES = ["timestamp_iso", "symbol", "side", "size", "fill_price", "order_id", "status"]

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.trades_path = Path(self._tmpdir.name) / "trades.csv"
        self.patcher = patch.object(cycle, "TRADES_CSV_PATH", self.trades_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def _write_trades(self, rows):
        with self.trades_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_missing_file_returns_zero(self):
        self.assertEqual(cycle.count_today_buys(), 0)

    def test_counts_only_todays_buys(self):
        self._write_trades(
            [
                {"timestamp_iso": "2026-07-13T14:00:00+00:00", "symbol": "AAPL", "side": "BUY",
                 "size": 10, "fill_price": 100, "order_id": 1, "status": "Filled"},
                {"timestamp_iso": "2026-07-13T15:00:00+00:00", "symbol": "MSFT", "side": "BUY",
                 "size": 5, "fill_price": 200, "order_id": 2, "status": "Filled"},
                {"timestamp_iso": "2026-07-13T15:30:00+00:00", "symbol": "MSFT", "side": "SELL",
                 "size": 5, "fill_price": 210, "order_id": 3, "status": "Filled"},
                {"timestamp_iso": "2026-07-12T14:00:00+00:00", "symbol": "NVDA", "side": "BUY",
                 "size": 3, "fill_price": 300, "order_id": 4, "status": "Filled"},
            ]
        )

        with patch("trading_bot.cli.cycle.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-07-13"
            count = cycle.count_today_buys()

        self.assertEqual(count, 2)


class TestLogEvent(unittest.TestCase):
    """log_event: appends a JSON-lines record, stamping a timestamp and
    casting numpy bools so json.dumps never chokes on them."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.safety_log_path = Path(self._tmpdir.name) / "safety-check-log.json"
        self.patcher = patch.object(cycle, "SAFETY_LOG_PATH", self.safety_log_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_appends_jsonl_line_with_timestamp(self):
        cycle.log_event({"event": "test_event", "symbol": "AAPL"})

        lines = self.safety_log_path.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["event"], "test_event")
        self.assertIn("timestamp_iso", record)

    def test_casts_numpy_bool_before_writing(self):
        cycle.log_event({"event": "test", "flag": np.bool_(True)})

        record = json.loads(self.safety_log_path.read_text().splitlines()[0])
        self.assertIs(record["flag"], True)

    def test_appends_multiple_events(self):
        cycle.log_event({"event": "one"})
        cycle.log_event({"event": "two"})

        lines = self.safety_log_path.read_text().splitlines()
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()