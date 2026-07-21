"""Unit tests for trading_bot.cli.compute_perf.

notify is always patched -- .env has real Telegram credentials configured
for this project. Functions that need a controlled "today" call
datetime.now(ET) directly without mocking the datetime class (mocking it
would also break _parse_ts's use of datetime.fromisoformat elsewhere in
the same call chain), so timestamps are built from the real current time.
"""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from trading_bot.cli import compute_perf as perf

TRADES_FIELDNAMES = ["timestamp_iso", "symbol", "side", "size", "fill_price", "order_id", "status"]


def make_trades_df(rows):
    return pd.DataFrame(rows)


def make_pair(symbol, pnl_usd):
    return {"symbol": symbol, "pnl_usd": pnl_usd}


class TestLoadTodayTrades(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.trades_path = Path(self._tmpdir.name) / "trades.csv"
        self.patcher = patch.object(perf, "TRADES_CSV_PATH", self.trades_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def _write(self, rows):
        with self.trades_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADES_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_missing_file_returns_empty_frame_with_expected_columns(self):
        df = perf.load_today_trades()
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), TRADES_FIELDNAMES)

    def test_filters_to_todays_rows_only(self):
        self._write(
            [
                {"timestamp_iso": "2026-07-13T14:00:00+00:00", "symbol": "AAPL", "side": "BUY",
                 "size": 10, "fill_price": 100, "order_id": 1, "status": "Filled"},
                {"timestamp_iso": "2026-07-12T14:00:00+00:00", "symbol": "NVDA", "side": "BUY",
                 "size": 3, "fill_price": 300, "order_id": 2, "status": "Filled"},
            ]
        )

        with patch("trading_bot.cli.compute_perf.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-07-13"
            df = perf.load_today_trades()

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "AAPL")

    def test_coerces_numeric_columns(self):
        self._write(
            [
                {"timestamp_iso": "2026-07-13T14:00:00+00:00", "symbol": "AAPL", "side": "BUY",
                 "size": "10", "fill_price": "100.5", "order_id": 1, "status": "Filled"},
            ]
        )

        with patch("trading_bot.cli.compute_perf.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-07-13"
            df = perf.load_today_trades()

        self.assertEqual(df.iloc[0]["size"], 10.0)
        self.assertEqual(df.iloc[0]["fill_price"], 100.5)


class TestPairTradesFifo(unittest.TestCase):
    def test_empty_frame_returns_empty_list(self):
        self.assertEqual(perf.pair_trades_fifo(pd.DataFrame()), [])

    def test_simple_buy_sell_pair(self):
        df = make_trades_df(
            [
                {"timestamp_iso": "2026-07-13T14:00:00+00:00", "symbol": "AAPL", "side": "BUY",
                 "size": 10, "fill_price": 100.0},
                {"timestamp_iso": "2026-07-13T14:30:00+00:00", "symbol": "AAPL", "side": "SELL",
                 "size": 10, "fill_price": 105.0},
            ]
        )

        pairs = perf.pair_trades_fifo(df)

        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertEqual(p["symbol"], "AAPL")
        self.assertEqual(p["qty"], 10.0)
        self.assertAlmostEqual(p["pnl_usd"], 50.0)
        self.assertAlmostEqual(p["pnl_pct"], 5.0)
        self.assertAlmostEqual(p["hold_minutes"], 30.0)

    def test_partial_sell_splits_buy_lot_fifo(self):
        """A single BUY lot split across two SELL fills (e.g. partial profit
        then a later stop-out) must FIFO-match correctly."""
        df = make_trades_df(
            [
                {"timestamp_iso": "2026-07-13T10:10:00+00:00", "symbol": "AAPL", "side": "BUY",
                 "size": 30, "fill_price": 100.0},
                {"timestamp_iso": "2026-07-13T10:40:00+00:00", "symbol": "AAPL", "side": "SELL",
                 "size": 10, "fill_price": 110.0},
                {"timestamp_iso": "2026-07-13T11:10:00+00:00", "symbol": "AAPL", "side": "SELL",
                 "size": 20, "fill_price": 95.0},
            ]
        )

        pairs = perf.pair_trades_fifo(df)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["qty"], 10.0)
        self.assertAlmostEqual(pairs[0]["pnl_usd"], 100.0)  # (110-100)*10
        self.assertEqual(pairs[1]["qty"], 20.0)
        self.assertAlmostEqual(pairs[1]["pnl_usd"], -100.0)  # (95-100)*20

    def test_sell_spans_multiple_buy_lots_oldest_first(self):
        df = make_trades_df(
            [
                {"timestamp_iso": "2026-07-13T10:00:00+00:00", "symbol": "MSFT", "side": "BUY",
                 "size": 5, "fill_price": 200.0},
                {"timestamp_iso": "2026-07-13T10:05:00+00:00", "symbol": "MSFT", "side": "BUY",
                 "size": 5, "fill_price": 210.0},
                {"timestamp_iso": "2026-07-13T10:30:00+00:00", "symbol": "MSFT", "side": "SELL",
                 "size": 8, "fill_price": 220.0},
            ]
        )

        pairs = perf.pair_trades_fifo(df)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["qty"], 5.0)
        self.assertEqual(pairs[0]["buy_price"], 200.0)  # oldest lot consumed first
        self.assertEqual(pairs[1]["qty"], 3.0)
        self.assertEqual(pairs[1]["buy_price"], 210.0)

    def test_incomplete_rows_are_skipped(self):
        df = make_trades_df(
            [
                {"timestamp_iso": "2026-07-13T10:00:00+00:00", "symbol": "AAPL", "side": "BUY",
                 "size": float("nan"), "fill_price": 200.0},
                {"timestamp_iso": "2026-07-13T10:05:00+00:00", "symbol": "MSFT", "side": "BUY",
                 "size": 5, "fill_price": float("nan")},
            ]
        )

        self.assertEqual(perf.pair_trades_fifo(df), [])

    def test_unparseable_timestamp_is_skipped(self):
        df = make_trades_df(
            [{"timestamp_iso": "not-a-timestamp", "symbol": "AAPL", "side": "BUY", "size": 5, "fill_price": 100.0}]
        )

        self.assertEqual(perf.pair_trades_fifo(df), [])

    def test_leftover_sell_with_no_matching_buy_is_ignored(self):
        df = make_trades_df(
            [{"timestamp_iso": "2026-07-13T10:00:00+00:00", "symbol": "AAPL", "side": "SELL",
              "size": 10, "fill_price": 100.0}]
        )

        self.assertEqual(perf.pair_trades_fifo(df), [])


class TestAggregate(unittest.TestCase):
    def test_zero_trades(self):
        result = perf.aggregate([])

        self.assertEqual(result["total_trades"], 0)
        self.assertEqual(result["profit_factor"], "n/a")
        self.assertIsNone(result["largest_winner"])
        self.assertIsNone(result["largest_loser"])

    def test_largest_winner_and_loser_drawn_from_correct_side(self):
        pairs = [make_pair("AAPL", 100.0), make_pair("MSFT", -50.0), make_pair("NVDA", 30.0), make_pair("TSLA", -10.0)]

        result = perf.aggregate(pairs)

        self.assertEqual(result["largest_winner"]["symbol"], "AAPL")
        self.assertEqual(result["largest_winner"]["pnl_usd"], 100.0)
        self.assertEqual(result["largest_loser"]["symbol"], "MSFT")
        self.assertEqual(result["largest_loser"]["pnl_usd"], -50.0)

    def test_all_losing_day_has_no_largest_winner(self):
        """Regression: previously max() ran over ALL pairs (not just wins),
        so an all-losing day would label its least-bad loss 'largest_winner'."""
        pairs = [make_pair("AAPL", -5.0), make_pair("MSFT", -20.0)]

        result = perf.aggregate(pairs)

        self.assertIsNone(result["largest_winner"])
        self.assertEqual(result["largest_loser"]["symbol"], "MSFT")

    def test_all_winning_day_has_no_largest_loser(self):
        """Regression: previously min() ran over ALL pairs, so an
        all-winning day would label its smallest win 'largest_loser'."""
        pairs = [make_pair("AAPL", 5.0), make_pair("MSFT", 20.0)]

        result = perf.aggregate(pairs)

        self.assertIsNone(result["largest_loser"])
        self.assertEqual(result["largest_winner"]["symbol"], "MSFT")

    def test_breakeven_only_day_has_neither(self):
        pairs = [make_pair("AAPL", 0.0), make_pair("MSFT", 0.0)]

        result = perf.aggregate(pairs)

        self.assertIsNone(result["largest_winner"])
        self.assertIsNone(result["largest_loser"])
        self.assertEqual(result["profit_factor"], "n/a")

    def test_profit_factor_infinite_when_no_losses(self):
        pairs = [make_pair("AAPL", 100.0), make_pair("MSFT", 50.0)]

        result = perf.aggregate(pairs)

        self.assertEqual(result["profit_factor"], "inf")

    def test_profit_factor_computed_normally(self):
        pairs = [make_pair("AAPL", 100.0), make_pair("MSFT", -50.0)]

        result = perf.aggregate(pairs)

        self.assertEqual(result["profit_factor"], 2.0)

    def test_win_rate_and_counts(self):
        pairs = [make_pair("A", 10.0), make_pair("B", -5.0), make_pair("C", 10.0)]

        result = perf.aggregate(pairs)

        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["win_rate_pct"], 66.67)


class TestBuildNotificationBody(unittest.TestCase):
    def test_zero_trades_message(self):
        summary = perf.aggregate([])

        self.assertEqual(perf.build_notification_body(summary), "No closed trades today.")

    def test_includes_key_stats(self):
        pairs = [make_pair("AAPL", 100.0), make_pair("MSFT", -50.0)]
        summary = perf.aggregate(pairs)

        body = perf.build_notification_body(summary)

        self.assertIn("Trades: 2", body)
        self.assertIn("Best: AAPL", body)
        self.assertIn("Worst: MSFT", body)
        self.assertIn("PF: 2.0", body)


class TestNative(unittest.TestCase):
    def test_casts_numpy_types(self):
        self.assertIs(type(perf._native(np.bool_(True))), bool)
        self.assertIs(type(perf._native(np.int64(5))), int)
        self.assertIs(type(perf._native(np.float64(1.5))), float)

    def test_passes_through_plain_types(self):
        self.assertEqual(perf._native("x"), "x")
        self.assertEqual(perf._native(5), 5)
        self.assertIsNone(perf._native(None))


class TestLoadOpenPositions(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.positions_path = Path(self._tmpdir.name) / "open_positions.json"
        self.patcher = patch.object(perf, "POSITIONS_PATH", self.positions_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(perf.load_open_positions(), [])

    def test_valid_list_round_trips(self):
        self.positions_path.write_text(json.dumps([{"symbol": "AAPL"}]))

        self.assertEqual(perf.load_open_positions(), [{"symbol": "AAPL"}])

    def test_non_list_json_returns_empty_list(self):
        self.positions_path.write_text(json.dumps({"not": "a list"}))

        self.assertEqual(perf.load_open_positions(), [])

    def test_corrupt_json_returns_empty_list(self):
        self.positions_path.write_text("{not valid")

        self.assertEqual(perf.load_open_positions(), [])


class TestGetInitialStopForPair(unittest.TestCase):
    def test_matches_open_position_by_entry_price(self):
        pair = {"symbol": "AAPL", "buy_price": 100.0}
        by_symbol = {"AAPL": [{"entry_price": 100.005, "initial_stop": 97.0}]}

        self.assertEqual(perf.get_initial_stop_for_pair(pair, by_symbol), 97.0)

    def test_falls_back_to_99pct_of_buy_price(self):
        pair = {"symbol": "AAPL", "buy_price": 100.0}

        self.assertAlmostEqual(perf.get_initial_stop_for_pair(pair, {}), 99.0)

    def test_no_matching_entry_price_falls_back(self):
        pair = {"symbol": "AAPL", "buy_price": 100.0}
        by_symbol = {"AAPL": [{"entry_price": 50.0, "initial_stop": 48.0}]}  # different entry -- no match

        self.assertAlmostEqual(perf.get_initial_stop_for_pair(pair, by_symbol), 99.0)


class TestComputeRForPairs(unittest.TestCase):
    def test_r_multiple_computed(self):
        pairs = [{"symbol": "AAPL", "buy_price": 100.0, "sell_price": 110.0}]
        open_positions = [{"symbol": "AAPL", "entry_price": 100.0, "initial_stop": 95.0}]

        result = perf.compute_r_for_pairs(pairs, open_positions)

        self.assertAlmostEqual(result[0]["r_multiple"], 2.0)  # (110-100)/(100-95)
        self.assertEqual(result[0]["initial_stop"], 95.0)

    def test_zero_risk_gives_none_r_multiple(self):
        pairs = [{"symbol": "AAPL", "buy_price": 100.0, "sell_price": 110.0}]
        open_positions = [{"symbol": "AAPL", "entry_price": 100.0, "initial_stop": 100.0}]

        result = perf.compute_r_for_pairs(pairs, open_positions)

        self.assertIsNone(result[0]["r_multiple"])


class TestRBucketLabel(unittest.TestCase):
    def test_boundaries(self):
        cases = [
            (-3, 0), (-2, 0),
            (-1.5, 1), (-1, 1),
            (-0.5, 2), (0, 2),
            (0.5, 3), (1, 3),
            (1.5, 4), (2, 4),
            (2.5, 5), (3, 5),
            (3.5, 6),
        ]
        for r, idx in cases:
            with self.subTest(r=r):
                self.assertEqual(perf.r_bucket_label(r), perf.R_BUCKET_LABELS[idx])


class TestBuildRHistogram(unittest.TestCase):
    def test_counts_and_skips_none(self):
        pairs = [{"r_multiple": 0.5}, {"r_multiple": -3.0}, {"r_multiple": None}]

        hist = perf.build_r_histogram(pairs)

        self.assertEqual(hist[perf.R_BUCKET_LABELS[3]], 1)
        self.assertEqual(hist[perf.R_BUCKET_LABELS[0]], 1)
        self.assertEqual(sum(hist.values()), 2)


class TestGetLastPrice(unittest.TestCase):
    def test_uses_1m_data_when_available(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [101.0, 102.5]})

        with patch("trading_bot.cli.compute_perf.yf.Ticker", return_value=mock_ticker):
            price = perf.get_last_price("AAPL")

        self.assertEqual(price, 102.5)

    def test_falls_back_to_daily_when_1m_empty(self):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = [pd.DataFrame(), pd.DataFrame({"Close": [99.0]})]

        with patch("trading_bot.cli.compute_perf.yf.Ticker", return_value=mock_ticker):
            price = perf.get_last_price("AAPL")

        self.assertEqual(price, 99.0)

    def test_returns_none_when_both_empty(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("trading_bot.cli.compute_perf.yf.Ticker", return_value=mock_ticker):
            price = perf.get_last_price("AAPL")

        self.assertIsNone(price)

    def test_returns_none_on_exception(self):
        with patch("trading_bot.cli.compute_perf.yf.Ticker", side_effect=ConnectionError("boom")):
            price = perf.get_last_price("AAPL")

        self.assertIsNone(price)

    def test_class_share_ticker_converted_for_yahoo(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [100.0]})

        with patch("trading_bot.cli.compute_perf.yf.Ticker", return_value=mock_ticker) as mock_yf_ticker:
            perf.get_last_price("BRK B")

        mock_yf_ticker.assert_called_once_with("BRK-B")


@patch("trading_bot.cli.compute_perf.get_last_price")
class TestBuildOpenPositionsView(unittest.TestCase):
    def test_computes_unrealized_r(self, mock_price):
        mock_price.return_value = 110.0
        positions = [{"symbol": "AAPL", "entry_price": 100.0, "qty": 10, "current_stop_price": 95.0, "R": 5.0}]

        view = perf.build_open_positions_view(positions)

        self.assertEqual(view[0]["unrealized_r"], 2.0)

    def test_zero_r_denom_skips_unrealized_r(self, mock_price):
        mock_price.return_value = 110.0
        positions = [{"symbol": "AAPL", "entry_price": 100.0, "qty": 10, "R": 0}]

        view = perf.build_open_positions_view(positions)

        self.assertIsNone(view[0]["unrealized_r"])

    def test_missing_price_skips_unrealized_r(self, mock_price):
        mock_price.return_value = None
        positions = [{"symbol": "AAPL", "entry_price": 100.0, "qty": 10, "R": 5.0}]

        view = perf.build_open_positions_view(positions)

        self.assertIsNone(view[0]["unrealized_r"])


class TestGetLastCycleInfo(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.safety_log_path = Path(self._tmpdir.name) / "safety-check-log.json"
        self.patcher = patch.object(perf, "SAFETY_LOG_PATH", self.safety_log_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_missing_file(self):
        info = perf.get_last_cycle_info()

        self.assertEqual(info["status"], "no cycle data yet")
        self.assertIsNone(info["timestamp"])

    def test_returns_last_line(self):
        self.safety_log_path.write_text(
            json.dumps({"timestamp_iso": "2026-07-13T10:00:00+00:00", "event": "entry_opened"}) + "\n"
            + json.dumps({"timestamp_iso": "2026-07-13T10:05:00+00:00", "event": "moved_to_breakeven"}) + "\n"
        )

        info = perf.get_last_cycle_info()

        self.assertEqual(info["status"], "moved_to_breakeven")
        self.assertEqual(info["timestamp"], "2026-07-13T10:05:00+00:00")

    def test_corrupt_last_line_falls_back(self):
        self.safety_log_path.write_text("not valid json\n")

        info = perf.get_last_cycle_info()

        self.assertEqual(info["status"], "no cycle data yet")


class TestRenderDashboardHtmlSmoke(unittest.TestCase):
    def test_renders_without_error_and_contains_key_sections(self):
        summary = perf.aggregate([make_pair("AAPL", 50.0)])
        pairs_with_r = [
            {
                "symbol": "AAPL", "buy_price": 100.0, "sell_price": 105.0, "pnl_usd": 50.0,
                "r_multiple": 1.0, "hold_minutes": 12.0, "sell_ts": "2026-07-13T14:00:00+00:00",
            }
        ]
        histogram = perf.build_r_histogram(pairs_with_r)
        positions_view = [
            {"symbol": "MSFT", "qty": 5, "entry": 50.0, "stop": 48.0, "current_price": 52.0, "unrealized_r": 1.0}
        ]
        last_cycle = {"timestamp": "2026-07-13T14:00:00+00:00", "status": "ok"}

        html_doc = perf.render_dashboard_html(summary, pairs_with_r, histogram, positions_view, last_cycle, "2026-07-13")

        self.assertIn("<html", html_doc)
        self.assertIn("AAPL", html_doc)
        self.assertIn("MSFT", html_doc)


class TestWriteDashboard(unittest.TestCase):
    def test_creates_dir_and_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dashboard_dir = Path(tmpdir) / "dashboard"
            dashboard_path = dashboard_dir / "index.html"
            with patch.object(perf, "DASHBOARD_DIR", dashboard_dir), patch.object(perf, "DASHBOARD_PATH", dashboard_path):
                perf.write_dashboard("<html>hi</html>")

            self.assertTrue(dashboard_path.exists())
            self.assertEqual(dashboard_path.read_text(encoding="utf-8"), "<html>hi</html>")


class TestMain(unittest.TestCase):
    """main: end-to-end wiring, including the largest_winner/largest_loser
    fix reaching the printed summary and the Telegram body."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        base = Path(self._tmpdir.name)
        self.trades_path = base / "trades.csv"
        self.positions_path = base / "open_positions.json"
        self.safety_log_path = base / "safety-check-log.json"
        self.dashboard_dir = base / "dashboard"
        self.dashboard_path = self.dashboard_dir / "index.html"

        self.patchers = [
            patch.object(perf, "TRADES_CSV_PATH", self.trades_path),
            patch.object(perf, "POSITIONS_PATH", self.positions_path),
            patch.object(perf, "SAFETY_LOG_PATH", self.safety_log_path),
            patch.object(perf, "DASHBOARD_DIR", self.dashboard_dir),
            patch.object(perf, "DASHBOARD_PATH", self.dashboard_path),
            patch("trading_bot.cli.compute_perf.notify"),
        ]
        self.mocks = [p.start() for p in self.patchers]
        self.mock_notify = self.mocks[-1]

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        self._tmpdir.cleanup()

    def _write_trades(self, rows):
        with self.trades_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADES_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_main_prints_summary_notifies_and_writes_dashboard(self):
        # Real (unmocked) "now" so the trades land in "today" regardless of
        # when the suite runs -- mocking datetime.now would also break
        # _parse_ts's datetime.fromisoformat call in the same chain.
        from datetime import datetime as real_datetime, timezone as real_timezone

        now = real_datetime.now(real_timezone.utc)
        self._write_trades(
            [
                {"timestamp_iso": now.isoformat(), "symbol": "AAPL", "side": "BUY",
                 "size": 10, "fill_price": 100, "order_id": 1, "status": "Filled"},
                {"timestamp_iso": (now + timedelta(minutes=30)).isoformat(), "symbol": "AAPL", "side": "SELL",
                 "size": 10, "fill_price": 110, "order_id": 2, "status": "Filled"},
            ]
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            perf.main()

        summary = json.loads(buf.getvalue().strip())
        self.assertEqual(summary["total_trades"], 1)
        self.assertEqual(summary["wins"], 1)
        self.mock_notify.assert_called_once()
        self.assertTrue(self.dashboard_path.exists())

    def test_main_handles_zero_trades(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            perf.main()

        summary = json.loads(buf.getvalue().strip())
        self.assertEqual(summary["total_trades"], 0)
        self.mock_notify.assert_called_once()
        body = self.mock_notify.call_args[0][1]
        self.assertEqual(body, "No closed trades today.")


if __name__ == "__main__":
    unittest.main()
