"""Unit tests for trading_bot.cli.smc_cycle's broker-free pieces.

The broker-coupled flows (orders, stop management) follow cycle.py's
already-reviewed patterns and get exercised by paper trading itself;
these tests cover the pure logic: trade-log round trips, daily BUY
counting, entry-bar index mapping, and bar freshness."""

import json
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

from trading_bot import smc_live
from trading_bot.cli import smc_cycle

ET = ZoneInfo("America/New_York")


class TestTradeLog(unittest.TestCase):
    def test_append_creates_header_then_counts_todays_buys(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "smc_trades.csv"
            with patch.object(smc_live, "SMC_TRADES_CSV_PATH", csv_path):
                smc_cycle.append_trade_row("NVDA", "BUY", 10, 100.5, 1, "Filled", "entry")
                smc_cycle.append_trade_row("NVDA", "SELL", 10, 101.5, 2, "Filled", "tp1")
                smc_cycle.append_trade_row("AMD", "BUY", 5, 50.0, 3, "Filled", "entry")

                lines = csv_path.read_text().splitlines()
                self.assertEqual(lines[0], ",".join(smc_live.TRADES_CSV_HEADER))
                self.assertEqual(len(lines), 4)
                self.assertEqual(smc_cycle.count_today_buys(), 2)

    def test_count_today_buys_missing_file_is_zero(self):
        with TemporaryDirectory() as tmp:
            with patch.object(smc_live, "SMC_TRADES_CSV_PATH", Path(tmp) / "none.csv"):
                self.assertEqual(smc_cycle.count_today_buys(), 0)


def _today_bars(n: int, start_hh=9, start_mm=30):
    today = datetime.now(ET).date()
    start = datetime(today.year, today.month, today.day, start_hh, start_mm, tzinfo=ET)
    idx = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({"High": [10.0] * n}, index=idx)


class TestEntryBarIndex(unittest.TestCase):
    def test_exact_match(self):
        bars = _today_bars(5)
        self.assertEqual(smc_cycle._entry_bar_index(bars, bars.index[2].isoformat()), 2)

    def test_first_bar_at_or_after_when_no_exact_match(self):
        bars = _today_bars(5)
        between = (bars.index[2] + timedelta(minutes=2)).isoformat()
        self.assertEqual(smc_cycle._entry_bar_index(bars, between), 3)

    def test_unparseable_or_future_returns_none(self):
        bars = _today_bars(5)
        self.assertIsNone(smc_cycle._entry_bar_index(bars, "not-a-date"))
        after_all = (bars.index[-1] + timedelta(hours=1)).isoformat()
        self.assertIsNone(smc_cycle._entry_bar_index(bars, after_all))


class TestBarsAreFresh(unittest.TestCase):
    def test_recent_last_bar_is_fresh(self):
        now = datetime.now(ET)
        idx = pd.date_range(now - timedelta(minutes=30), now - timedelta(minutes=5), freq="5min")
        bars = pd.DataFrame({"High": [1.0] * len(idx)}, index=idx)
        self.assertTrue(smc_cycle._bars_are_fresh(bars))

    def test_stale_last_bar_is_not_fresh(self):
        now = datetime.now(ET)
        idx = pd.date_range(now - timedelta(hours=3), periods=5, freq="5min")
        bars = pd.DataFrame({"High": [1.0] * 5}, index=idx)
        self.assertFalse(smc_cycle._bars_are_fresh(bars))

    def test_yesterdays_bars_are_not_fresh(self):
        now = datetime.now(ET)
        idx = pd.date_range(now - timedelta(days=1, minutes=10), periods=3, freq="5min")
        bars = pd.DataFrame({"High": [1.0] * 3}, index=idx)
        self.assertFalse(smc_cycle._bars_are_fresh(bars))


class TestGet5mBarsLogging(unittest.TestCase):
    """get_5m_bars returns None down three separate paths and callers see
    only that None, so each path has to say why it bailed."""

    def _fetch(self, side_effect=None, frame=None):
        """Run get_5m_bars against a stubbed yfinance, returning
        (result, logged_events)."""
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "smc-safety-check-log.json"
            ticker = MagicMock()
            if side_effect is not None:
                ticker.history.side_effect = side_effect
            else:
                ticker.history.return_value = frame
            with patch.object(smc_live, "SMC_SAFETY_LOG_PATH", log_path), \
                    patch.object(smc_cycle, "yf") as fake_yf:
                fake_yf.Ticker.return_value = ticker
                result = smc_cycle.get_5m_bars("NVDA", context="entry_scan")
            events = []
            if log_path.exists():
                events = [json.loads(line) for line in log_path.read_text().splitlines()]
            return result, events

    def test_yfinance_error_logs_cause_and_message(self):
        result, events = self._fetch(side_effect=RuntimeError("boom"))
        self.assertIsNone(result)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "bars_unavailable")
        self.assertEqual(events[0]["cause"], "yfinance_error")
        self.assertEqual(events[0]["symbol"], "NVDA")
        self.assertEqual(events[0]["context"], "entry_scan")
        # The bare `except Exception` used to discard this entirely.
        self.assertIn("boom", events[0]["error"])

    def test_empty_response_logs_its_own_cause(self):
        result, events = self._fetch(frame=pd.DataFrame())
        self.assertIsNone(result)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["cause"], "empty_response")

    def test_bars_entirely_outside_rth_log_their_own_cause(self):
        idx = pd.date_range("2026-08-24 04:00", periods=3, freq="5min", tz=ET)
        result, events = self._fetch(frame=pd.DataFrame({"High": [1.0] * 3}, index=idx))
        self.assertIsNone(result)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["cause"], "no_rth_rows")

    def test_rth_bars_are_returned_and_log_nothing(self):
        idx = pd.date_range("2026-08-24 09:55", periods=3, freq="5min", tz=ET)
        result, events = self._fetch(frame=pd.DataFrame({"High": [1.0] * 3}, index=idx))
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertEqual(events, [])


def _ohlc_bars(highs, closes=None):
    """Today's 5-min bars with the given highs, indexed from 09:30 ET."""
    today = datetime.now(ET).date()
    start = datetime(today.year, today.month, today.day, 9, 30, tzinfo=ET)
    idx = pd.date_range(start, periods=len(highs), freq="5min")
    return pd.DataFrame(
        {"High": [float(h) for h in highs], "Close": [float(c) for c in (closes or highs)]},
        index=idx,
    )


class TestTp1Touched(unittest.TestCase):
    """TP1 arms off bar highs at/after the entry bar, not a polled quote."""

    def test_post_entry_high_reaching_target_triggers(self):
        bars = _ohlc_bars([180.0, 181.0, 183.5, 182.0])
        self.assertTrue(smc_cycle._tp1_touched(bars, 1, 183.32))

    def test_post_entry_highs_below_target_do_not_trigger(self):
        bars = _ohlc_bars([180.0, 182.0, 182.96, 182.63])
        self.assertFalse(smc_cycle._tp1_touched(bars, 1, 183.32))

    def test_target_reached_only_before_entry_does_not_trigger(self):
        # The KLAC 2026-08-25 case: the stock printed 185.99 pre-entry and
        # never exceeded 182.96 afterwards, but a delayed quote still fired
        # TP1. Pre-entry highs must not count.
        bars = _ohlc_bars([185.99, 184.85, 182.86, 182.96, 182.63])
        self.assertFalse(smc_cycle._tp1_touched(bars, 2, 183.32))

    def test_entry_on_final_bar_still_evaluates_that_bar(self):
        bars = _ohlc_bars([180.0, 181.0, 183.4])
        self.assertTrue(smc_cycle._tp1_touched(bars, 2, 183.32))
        self.assertFalse(smc_cycle._tp1_touched(bars, 2, 184.0))

    def test_exact_touch_counts(self):
        bars = _ohlc_bars([180.0, 183.32])
        self.assertTrue(smc_cycle._tp1_touched(bars, 1, 183.32))


class TestTp1TouchDetail(unittest.TestCase):
    """The touch detail feeds the execution-quality record: which bar armed
    TP1, how high it printed, and how many bars ago that was."""

    def test_returns_first_touching_bar_not_the_highest(self):
        bars = _ohlc_bars([180.0, 183.5, 190.0, 182.0])
        self.assertEqual(smc_cycle._tp1_touch_detail(bars, 1, 183.32), (1, 183.5))

    def test_returns_none_when_never_touched(self):
        bars = _ohlc_bars([180.0, 182.0, 182.96])
        self.assertIsNone(smc_cycle._tp1_touch_detail(bars, 1, 183.32))

    def test_pre_entry_touch_is_ignored(self):
        """KLAC 2026-08-25 again: 185.99 printed pre-entry must not arm TP1
        or be reported as the trigger bar."""
        bars = _ohlc_bars([185.99, 184.85, 182.86, 182.96])
        self.assertIsNone(smc_cycle._tp1_touch_detail(bars, 2, 183.32))

    def test_index_is_absolute_not_relative_to_entry(self):
        bars = _ohlc_bars([180.0, 180.0, 180.0, 183.4, 182.0])
        self.assertEqual(smc_cycle._tp1_touch_detail(bars, 2, 183.32), (3, 183.4))

    def test_staleness_is_derivable_from_the_index(self):
        """bars_since_touch in the log event is (len-1) - touch_idx: the
        touch can be several cycles old before a fetch reveals it."""
        bars = _ohlc_bars([180.0, 183.4, 182.0, 181.0, 181.5])
        touch_idx, _ = smc_cycle._tp1_touch_detail(bars, 0, 183.32)
        self.assertEqual(touch_idx, 1)
        self.assertEqual((len(bars) - 1) - touch_idx, 3)

    def test_touched_wrapper_agrees_with_detail(self):
        for highs, entry_idx, target in (
            ([180.0, 183.5], 1, 183.32),
            ([180.0, 182.0], 1, 183.32),
            ([185.99, 182.0], 1, 183.32),
        ):
            bars = _ohlc_bars(highs)
            self.assertEqual(
                smc_cycle._tp1_touched(bars, entry_idx, target),
                smc_cycle._tp1_touch_detail(bars, entry_idx, target) is not None,
            )

    def test_slippage_bps_formula_matches_the_klac_arithmetic(self):
        """The number the record exists to collect: positive means the fill
        came back BELOW the level."""
        level, fill = 183.32, 182.43
        self.assertEqual(round((level - fill) / level * 10_000, 1), 48.5)
        self.assertEqual(round((level - level) / level * 10_000, 1), 0.0)


class TestLastBarClose(unittest.TestCase):
    """The force-close / new-high fill-price fallback reads a real traded
    price, not get_current_price's delayed quote."""

    def test_returns_latest_close(self):
        bars = _ohlc_bars([181.0, 182.0, 183.0], closes=[180.5, 181.5, 182.75])
        with patch.object(smc_cycle, "get_5m_bars", return_value=bars):
            self.assertAlmostEqual(smc_cycle._last_bar_close("KLAC"), 182.75)

    def test_missing_bars_return_none_so_caller_can_fall_through(self):
        with patch.object(smc_cycle, "get_5m_bars", return_value=None):
            self.assertIsNone(smc_cycle._last_bar_close("KLAC"))

    def test_empty_frame_returns_none(self):
        with patch.object(smc_cycle, "get_5m_bars", return_value=pd.DataFrame({"Close": []})):
            self.assertIsNone(smc_cycle._last_bar_close("KLAC"))


class _FakeOrderStatus:
    def __init__(self, statuses):
        self._statuses = list(statuses)

    @property
    def status(self):
        # Last value repeats once the script is exhausted.
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]


class _FakeTrade:
    def __init__(self, statuses):
        self.orderStatus = _FakeOrderStatus(statuses)


class _FakeIB:
    def __init__(self, trade):
        self._trade = trade
        self.sleeps = 0

    def placeOrder(self, contract, order):
        return self._trade

    def sleep(self, _secs):
        self.sleeps += 1

    def qualifyContracts(self, stock):
        return [stock]


class _FakeOrder:
    def __init__(self, order_id):
        self.orderId = order_id


class _StopTrade:
    def __init__(self, statuses, order_id=77, avg_fill=0.0):
        self.orderStatus = _FakeOrderStatus(statuses)
        # manage_position reads avgFillPrice off the sell it just placed.
        self.orderStatus.avgFillPrice = avg_fill
        self.order = _FakeOrder(order_id)


class TestPlaceStopConfirmsPlacement(unittest.TestCase):
    """A rejected stop must not look like a live one. Previously _place_stop
    returned the orderId unconditionally, so the position carried on with a
    stop_order_id set and nothing actually resting behind it."""

    def _run(self, statuses):
        ib = _FakeIB(_StopTrade(statuses))
        events, notes = [], []
        with patch.object(smc_cycle, "_qualify", side_effect=lambda _ib, sym: sym),              patch.object(smc_cycle, "log_event", side_effect=events.append),              patch.object(smc_cycle, "notify", side_effect=lambda *a, **k: notes.append(a)):
            result = smc_cycle._place_stop(ib, "KLAC", 40, 182.0512)
        return result, events, notes

    def test_rejected_stop_returns_none(self):
        result, events, notes = self._run(["PendingSubmit", "Rejected"])
        self.assertIsNone(result)
        self.assertEqual([e["event"] for e in events], ["place_stop_failed"])
        self.assertEqual(events[0]["stop_price"], 182.05)  # rounded to a tick
        self.assertEqual(notes[0][2], "high")  # paged, not a routine note

    def test_presubmitted_is_a_resting_stop(self):
        result, events, _ = self._run(["PendingSubmit", "PreSubmitted"])
        self.assertEqual(result, 77)
        self.assertEqual(events, [])

    def test_submitted_is_accepted_too(self):
        result, _, _ = self._run(["Submitted"])
        self.assertEqual(result, 77)

    def test_still_pending_keeps_the_id_but_flags_it(self):
        """Returning None here would orphan an order that may be live and
        let a caller place a second one for the same shares."""
        with patch.object(smc_cycle, "STOP_CONFIRM_TIMEOUT_SECS", 2):
            result, events, _ = self._run(["PendingSubmit"])
        self.assertEqual(result, 77)
        self.assertEqual([e["event"] for e in events], ["place_stop_unconfirmed"])

    def test_cancelled_counts_as_failure(self):
        result, events, _ = self._run(["Cancelled"])
        self.assertIsNone(result)
        self.assertEqual(events[0]["status"], "Cancelled")


class TestBreakevenStopIsPlaceable(unittest.TestCase):
    """TP1 arms off a bar high that can be several bars old, so price may sit
    below entry when the cycle acts. A sell stop above the last traded price
    is not a resting order, so breakeven has to be clamped under the market."""

    RULES = {"tp1_fraction": 0.25, "swing_window": 20}

    def _run_tp1(self, last_close):
        bars = _ohlc_bars([182.10, 183.40, last_close], closes=[182.00, 183.00, last_close])
        pos = {
            "symbol": "KLAC", "entry_price": 182.05, "qty": 54, "original_qty": 54,
            "stop_price": 180.44, "current_stop_price": 180.44, "stop_order_id": 9,
            "tp1_price": 183.32, "tp1_done": False,
            "entry_bar_iso": bars.index[0].isoformat(),
        }
        placed, events = [], []
        with patch.object(smc_cycle, "get_5m_bars", return_value=bars),              patch.object(smc_cycle, "_cancel_stop", return_value=True),              patch.object(smc_cycle, "_limit_order",
                          return_value=(_StopTrade(["Filled"], order_id=12, avg_fill=183.30), 13, 183.30)),              patch.object(smc_cycle, "_place_stop",
                          side_effect=lambda _ib, _s, _q, price: placed.append(price) or 55),              patch.object(smc_cycle, "append_trade_row"),              patch.object(smc_cycle, "notify"),              patch.object(smc_cycle, "log_event", side_effect=events.append):
            out = smc_cycle.manage_position(object(), pos, self.RULES)
        return out, placed, events

    def test_stop_is_clamped_below_market_when_price_fell_under_entry(self):
        _, placed, events = self._run_tp1(last_close=181.50)
        ceiling = 181.50 * (1 - smc_cycle.STOP_BELOW_MARKET_BUFFER_BPS / 10_000)
        self.assertAlmostEqual(placed[-1], ceiling)
        self.assertLess(placed[-1], 182.05)  # not the unplaceable breakeven
        clamped = [e for e in events if e["event"] == "tp1_stop_clamped"]
        self.assertEqual(len(clamped), 1)
        self.assertEqual(clamped[0]["wanted"], 182.05)

    def test_breakeven_is_used_unchanged_when_price_is_above_entry(self):
        out, placed, events = self._run_tp1(last_close=184.00)
        self.assertAlmostEqual(placed[-1], 182.05)
        self.assertEqual([e for e in events if e["event"] == "tp1_stop_clamped"], [])
        self.assertTrue(out["tp1_done"])


class TestUnprotectedPositionIsRepaired(unittest.TestCase):
    """A position whose stop placement failed must get another attempt on the
    next cycle rather than running naked for the rest of the trade."""

    RULES = {"tp1_fraction": 0.25, "swing_window": 20}

    def _run(self, stop_order_id, place_returns):
        pos = {
            "symbol": "KLAC", "entry_price": 182.05, "qty": 54, "original_qty": 54,
            "stop_price": 180.44, "current_stop_price": 180.44,
            "stop_order_id": stop_order_id, "tp1_price": None, "tp1_done": True,
            "entry_bar_iso": "not-a-timestamp",
        }
        placed, events = [], []
        with patch.object(smc_cycle, "get_5m_bars", return_value=None),              patch.object(smc_cycle, "_place_stop",
                          side_effect=lambda _ib, _s, _q, price: placed.append(price) or place_returns),              patch.object(smc_cycle, "log_event", side_effect=events.append):
            out = smc_cycle.manage_position(object(), pos, self.RULES)
        return out, placed, events

    def test_missing_stop_triggers_a_replacement(self):
        out, placed, events = self._run(stop_order_id=None, place_returns=61)
        self.assertEqual(placed, [180.44])
        self.assertEqual(out["stop_order_id"], 61)
        self.assertIn("stop_replaced", [e["event"] for e in events])

    def test_existing_stop_is_left_alone(self):
        _, placed, events = self._run(stop_order_id=9, place_returns=61)
        self.assertEqual(placed, [])
        self.assertNotIn("stop_replaced", [e["event"] for e in events])

    def test_failed_replacement_leaves_it_unprotected_and_silent_about_success(self):
        out, placed, events = self._run(stop_order_id=None, place_returns=None)
        self.assertEqual(placed, [180.44])
        self.assertIsNone(out["stop_order_id"])
        self.assertNotIn("stop_replaced", [e["event"] for e in events])


class TestMarketOrderWaitsForFill(unittest.TestCase):
    """_market_order must wait for a settled status, not for "Submitted"."""

    def _run(self, statuses, timeout=1.0):
        ib = _FakeIB(_FakeTrade(statuses))
        with patch.object(smc_cycle, "ORDER_FILL_TIMEOUT_SECS", timeout):
            trade = smc_cycle._market_order(ib, "NVDA", "BUY", 10)
        return ib, trade

    def test_returns_once_filled(self):
        ib, _ = self._run(["PendingSubmit", "Submitted", "Filled"])
        # Broke out on Filled rather than spinning to the timeout.
        self.assertEqual(ib.sleeps, 3)

    def test_does_not_break_early_on_submitted(self):
        # The regression guard: "Submitted" arrives before any fill, so
        # breaking on it left avgFillPrice empty and callers recording a
        # theoretical price instead of the real one.
        ib, _ = self._run(["Submitted"], timeout=0.6)
        self.assertGreater(ib.sleeps, 1)

    def test_returns_on_terminal_non_fill(self):
        ib, _ = self._run(["Submitted", "Cancelled"])
        self.assertEqual(ib.sleeps, 2)


if __name__ == "__main__":
    unittest.main()


class TestAdverseSlippageBps(unittest.TestCase):
    """Entry slippage was the one execution leg the safety log could not
    reconstruct after the fact -- entry_opened recorded the fill under the
    name entry_price and dropped the signal level entirely.
    """

    def test_buy_filling_above_the_level_is_adverse(self):
        # 100.00 signal, 100.50 fill -> paid 50 bps more than the backtest
        # assumes it would.
        self.assertAlmostEqual(
            smc_cycle.adverse_slippage_bps(100.0, 100.5, "BUY"), 50.0, places=6
        )

    def test_buy_filling_below_the_level_is_favourable(self):
        self.assertAlmostEqual(
            smc_cycle.adverse_slippage_bps(100.0, 99.5, "BUY"), -50.0, places=6
        )

    def test_sell_sign_is_inverted(self):
        # Selling BELOW the level is the adverse direction, so it must come
        # back positive the same way an over-paid buy does.
        self.assertAlmostEqual(
            smc_cycle.adverse_slippage_bps(100.0, 99.5, "SELL"), 50.0, places=6
        )
        self.assertAlmostEqual(
            smc_cycle.adverse_slippage_bps(100.0, 100.5, "SELL"), -50.0, places=6
        )

    def test_exact_fill_is_zero(self):
        self.assertEqual(smc_cycle.adverse_slippage_bps(100.0, 100.0, "BUY"), 0.0)

    def test_reproduces_the_measured_live_tp1_slippage(self):
        # KLAC's real TP1: target 183.32, filled 182.43 -- the 49 bps the
        # slippage note in smc_signals.py records.
        self.assertAlmostEqual(
            smc_cycle.adverse_slippage_bps(183.320007, 182.43, "SELL"), 48.5, places=1
        )

    def test_non_positive_signal_price_returns_none_rather_than_dividing(self):
        self.assertIsNone(smc_cycle.adverse_slippage_bps(0.0, 100.0, "BUY"))
        self.assertIsNone(smc_cycle.adverse_slippage_bps(-1.0, 100.0, "BUY"))
        self.assertIsNone(smc_cycle.adverse_slippage_bps(None, 100.0, "BUY"))


class TestRoundedHelper(unittest.TestCase):
    def test_rounds_a_number(self):
        self.assertEqual(smc_cycle._rounded(1.23456), 1.23)

    def test_passes_none_through(self):
        # An unmeasurable slippage must log as null, not crash the entry.
        self.assertIsNone(smc_cycle._rounded(None))


class _LimitOrderStatus:
    """Own status class, NOT a patched _FakeOrderStatus -- attaching a
    `filled` property to that shared type leaks into every other test
    using it."""

    def __init__(self, statuses, filled, avg_fill):
        self._statuses = list(statuses)
        self._filled = filled
        self.avgFillPrice = avg_fill

    @property
    def status(self):
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    @property
    def filled(self):
        return self._filled


class _LimitTrade:
    def __init__(self, statuses, filled, avg_fill=0.0, order_id=99):
        self.orderStatus = _LimitOrderStatus(statuses, filled, avg_fill)
        self.order = _FakeOrder(order_id)


class _LimitIB(_FakeIB):
    def __init__(self, trade):
        super().__init__(trade)
        self.cancelled = []
        self.placed_orders = []

    def placeOrder(self, contract, order):
        self.placed_orders.append(order)
        return self._trade

    def cancelOrder(self, order):
        self.cancelled.append(order)


class TestLimitPriceFor(unittest.TestCase):
    def test_zero_through_sits_exactly_at_the_level(self):
        self.assertEqual(smc_cycle.limit_price_for(100.0, "BUY", 0.0), 100.0)
        self.assertEqual(smc_cycle.limit_price_for(100.0, "SELL", 0.0), 100.0)

    def test_through_moves_toward_the_filling_side(self):
        # A buy pays up, a sell accepts less -- both more likely to fill.
        self.assertEqual(smc_cycle.limit_price_for(100.0, "BUY", 50.0), 100.5)
        self.assertEqual(smc_cycle.limit_price_for(100.0, "SELL", 50.0), 99.5)

    def test_rounds_to_a_cent(self):
        self.assertEqual(smc_cycle.limit_price_for(343.829987, "BUY", 0.0), 343.83)

    def test_default_through_is_zero_so_the_level_is_honoured(self):
        self.assertEqual(smc_cycle.LIMIT_THROUGH_BPS, 0.0)
        self.assertEqual(smc_cycle.limit_price_for(50.0, "BUY"), 50.0)


class TestLimitOrder(unittest.TestCase):
    """The unfilled remainder MUST be cancelled: a limit left resting can
    fill between cycles and desynchronise the position state that every
    stop this module places is sized against."""

    def test_full_fill_returns_quantity_and_price(self):
        ib = _LimitIB(_LimitTrade(["Submitted", "Filled"], 10, avg_fill=100.25))
        trade, filled, avg = smc_cycle._limit_order(ib, "AAA", "BUY", 10, 100.0, timeout_secs=1)
        self.assertEqual(filled, 10)
        self.assertEqual(avg, 100.25)
        self.assertEqual(ib.cancelled, [])

    def test_unfilled_limit_is_cancelled(self):
        ib = _LimitIB(_LimitTrade(["Submitted"], 0, avg_fill=0.0))
        _, filled, _ = smc_cycle._limit_order(ib, "AAA", "BUY", 10, 100.0, timeout_secs=1)
        self.assertEqual(filled, 0)
        self.assertEqual(len(ib.cancelled), 1)

    def test_partial_fill_is_reported_and_remainder_cancelled(self):
        ib = _LimitIB(_LimitTrade(["Submitted"], 4, avg_fill=100.0))
        _, filled, _ = smc_cycle._limit_order(ib, "AAA", "BUY", 10, 100.0, timeout_secs=1)
        self.assertEqual(filled, 4)
        self.assertEqual(len(ib.cancelled), 1)

    def test_settled_order_is_not_cancelled_again(self):
        # Already Cancelled by IBKR -- re-cancelling would be a spurious API call.
        ib = _LimitIB(_LimitTrade(["Cancelled"], 0))
        _, filled, _ = smc_cycle._limit_order(ib, "AAA", "BUY", 10, 100.0, timeout_secs=1)
        self.assertEqual(filled, 0)
        self.assertEqual(ib.cancelled, [])

    def test_places_a_limit_at_the_signal_level(self):
        ib = _LimitIB(_LimitTrade(["Filled"], 10, avg_fill=100.0))
        smc_cycle._limit_order(ib, "AAA", "BUY", 10, 100.0, timeout_secs=1)
        self.assertEqual(len(ib.placed_orders), 1)
        self.assertEqual(ib.placed_orders[0].lmtPrice, 100.0)
        self.assertEqual(ib.placed_orders[0].action, "BUY")


class TestTp1LimitNotFilled(unittest.TestCase):
    """The dangerous limit-order path: TP1 cancels the resting stop BEFORE
    selling, so a limit that never fills would leave a live position with
    no stop behind it. The stop must go back and tp1_done must stay False
    so the next cycle retries -- the touch that armed it is still true."""

    RULES = {"tp1_fraction": 0.25, "swing_window": 20}

    def _run(self, filled_qty):
        bars = _ohlc_bars([182.10, 183.40, 184.00], closes=[182.00, 183.00, 184.00])
        pos = {
            "symbol": "KLAC", "entry_price": 182.05, "qty": 54, "original_qty": 54,
            "stop_price": 180.44, "current_stop_price": 180.44, "stop_order_id": 9,
            "tp1_price": 183.32, "tp1_done": False,
            "entry_bar_iso": bars.index[0].isoformat(),
        }
        placed, events, rows = [], [], []
        trade = _StopTrade(["Submitted"], order_id=12, avg_fill=0.0)
        with patch.object(smc_cycle, "get_5m_bars", return_value=bars), \
             patch.object(smc_cycle, "_cancel_stop", return_value=True), \
             patch.object(smc_cycle, "_limit_order", return_value=(trade, filled_qty, 183.32)), \
             patch.object(smc_cycle, "_place_stop",
                          side_effect=lambda _ib, _s, _q, price: placed.append(price) or 55), \
             patch.object(smc_cycle, "append_trade_row", side_effect=lambda *a, **k: rows.append(a)), \
             patch.object(smc_cycle, "notify"), \
             patch.object(smc_cycle, "log_event", side_effect=events.append):
            out = smc_cycle.manage_position(object(), pos, self.RULES)
        return out, placed, events, rows

    def test_unfilled_tp1_restores_the_stop_and_keeps_position_intact(self):
        out, placed, events, rows = self._run(filled_qty=0)
        self.assertFalse(out["tp1_done"])          # retried next cycle
        self.assertEqual(out["qty"], 54)           # nothing sold
        self.assertEqual(placed, [180.44])         # original stop back on
        self.assertEqual(out["stop_order_id"], 55)
        self.assertEqual(rows, [])                 # no phantom trade row
        self.assertEqual(len([e for e in events if e["event"] == "tp1_not_filled"]), 1)

    def test_partial_tp1_fill_sells_only_what_filled(self):
        out, _, events, rows = self._run(filled_qty=5)
        self.assertEqual(out["qty"], 49)           # 54 - 5, not 54 - 13
        self.assertTrue(out["tp1_done"])
        self.assertEqual(len([e for e in events if e["event"] == "tp1_partial_fill"]), 1)
        self.assertEqual(rows[0][2], 5)            # trade row records 5 shares
