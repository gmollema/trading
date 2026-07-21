"""Unit tests for trading_bot.cli.bot.

IBKRClient, strategy.evaluate, and subprocess.run are always mocked -- no
real broker connection or subprocess spawn.
"""

import csv
import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from trading_bot.cli import bot

CSV_HEADER = ["timestamp_iso", "symbol", "side", "size", "fill_price", "order_id", "status"]

BASE_ENV = {
    "IBKR_HOST": "127.0.0.1",
    "IBKR_PORT": "7497",
    "IBKR_CLIENT_ID": "2",
    "PAPER_TRADING": "true",
    "MAX_TRADE_SIZE_USD": "1000",
    "PORTFOLIO_VALUE_USD": "10000",
    "MAX_TRADES_PER_DAY": "5",
}


class TestEnsureTradesCsv(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.trades_path = Path(self._tmpdir.name) / "trades.csv"
        self.patcher = patch.object(bot, "TRADES_CSV", self.trades_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_creates_file_with_header_if_missing(self):
        bot.ensure_trades_csv()

        with self.trades_path.open() as f:
            self.assertEqual(next(csv.reader(f)), CSV_HEADER)

    def test_does_not_overwrite_existing_file(self):
        self.trades_path.write_text("existing content\n")

        bot.ensure_trades_csv()

        self.assertEqual(self.trades_path.read_text(), "existing content\n")


class TestCountTodayBuys(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.trades_path = Path(self._tmpdir.name) / "trades.csv"
        self.patcher = patch.object(bot, "TRADES_CSV", self.trades_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def _write(self, rows):
        with self.trades_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_counts_only_todays_buys(self):
        self._write(
            [
                {"timestamp_iso": "2026-07-13T14:00:00+00:00", "symbol": "AAPL", "side": "BUY",
                 "size": 10, "fill_price": 100, "order_id": 1, "status": "Filled"},
                {"timestamp_iso": "2026-07-12T14:00:00+00:00", "symbol": "NVDA", "side": "BUY",
                 "size": 3, "fill_price": 300, "order_id": 2, "status": "Filled"},
            ]
        )

        with patch("trading_bot.cli.bot.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-07-13"
            count = bot.count_today_buys()

        self.assertEqual(count, 1)


def _run_main(argv, env_overrides=None):
    env = dict(BASE_ENV)
    if env_overrides:
        env.update(env_overrides)
    buf = io.StringIO()
    with patch("sys.argv", ["bot.py"] + argv), patch.dict(os.environ, env, clear=True), redirect_stdout(buf):
        try:
            bot.main()
            code = None
        except SystemExit as e:
            code = e.code
    return buf.getvalue(), code


@patch("trading_bot.cli.bot.load_dotenv")
class TestMainGuards(unittest.TestCase):
    """Paper/live port guard and the daily trade-count gate."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.trades_path = Path(self._tmpdir.name) / "trades.csv"
        self.patcher = patch.object(bot, "TRADES_CSV", self.trades_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_paper_flag_with_live_port_aborts(self, mock_load_dotenv):
        _, code = _run_main(["--symbol", "AAPL"], {"PAPER_TRADING": "true", "IBKR_PORT": "7496"})

        self.assertIn("ABORT", str(code))

    def test_live_flag_with_paper_port_aborts(self, mock_load_dotenv):
        _, code = _run_main(["--symbol", "AAPL"], {"PAPER_TRADING": "false", "IBKR_PORT": "7497"})

        self.assertIn("ABORT", str(code))

    def test_max_trades_per_day_reached_exits_without_connecting(self, mock_load_dotenv):
        with patch("trading_bot.cli.bot.IBKRClient") as mock_client_cls:
            _, code = _run_main(["--symbol", "AAPL"], {"MAX_TRADES_PER_DAY": "0"})

        self.assertEqual(code, 0)
        mock_client_cls.assert_not_called()

    def test_ibkr_connection_failure_exits_1(self, mock_load_dotenv):
        with patch("trading_bot.cli.bot.IBKRClient", side_effect=ConnectionError("boom")):
            out, code = _run_main(["--symbol", "AAPL"])

        self.assertEqual(code, 1)
        self.assertIn("EXCEPTION", out)


@patch("trading_bot.cli.bot.load_dotenv")
@patch("trading_bot.cli.bot.IBKRClient")
class TestMainTradingFlow(unittest.TestCase):
    """The evaluate -> size -> spawn trade.py -> report flow, including the
    returncode-checking fix (a rejected order must not fall back to a stale
    trades.csv row)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.trades_path = Path(self._tmpdir.name) / "trades.csv"
        self.patcher = patch.object(bot, "TRADES_CSV", self.trades_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def _write_trades(self, rows):
        with self.trades_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_check_only_mode_does_not_place_order(self, mock_client_cls, mock_load_dotenv):
        mock_client_cls.return_value = MagicMock()

        with patch("trading_bot.cli.bot.strategy.evaluate",
                    return_value={"pass": True, "reasons": [], "price": 100.0}):
            with patch("trading_bot.cli.bot.subprocess.run") as mock_run:
                _, code = _run_main(["--symbol", "AAPL", "--check-only"])

        self.assertEqual(code, 0)
        mock_run.assert_not_called()

    def test_strategy_fail_does_not_place_order(self, mock_client_cls, mock_load_dotenv):
        mock_client_cls.return_value = MagicMock()

        with patch("trading_bot.cli.bot.strategy.evaluate",
                    return_value={"pass": False, "reasons": ["outside window"], "price": 0.0}):
            with patch("trading_bot.cli.bot.subprocess.run") as mock_run:
                _, code = _run_main(["--symbol", "AAPL"])

        self.assertEqual(code, 0)
        mock_run.assert_not_called()

    def test_invalid_price_does_not_place_order(self, mock_client_cls, mock_load_dotenv):
        mock_client_cls.return_value = MagicMock()

        with patch("trading_bot.cli.bot.strategy.evaluate",
                    return_value={"pass": True, "reasons": [], "price": 0.0}):
            with patch("trading_bot.cli.bot.subprocess.run") as mock_run:
                _, code = _run_main(["--symbol", "AAPL"])

        self.assertEqual(code, 0)
        mock_run.assert_not_called()

    def test_position_too_small_does_not_place_order(self, mock_client_cls, mock_load_dotenv):
        mock_client_cls.return_value = MagicMock()

        with patch("trading_bot.cli.bot.strategy.evaluate",
                    return_value={"pass": True, "reasons": [], "price": 1_000_000.0}):
            with patch("trading_bot.cli.bot.subprocess.run") as mock_run:
                _, code = _run_main(["--symbol", "AAPL"])

        self.assertEqual(code, 0)
        mock_run.assert_not_called()

    def test_successful_trade_reports_success(self, mock_client_cls, mock_load_dotenv):
        mock_client_cls.return_value = MagicMock()
        self._write_trades(
            [{"timestamp_iso": "2026-07-13T14:00:00+00:00", "symbol": "AAPL", "side": "BUY",
              "size": 1, "fill_price": 100, "order_id": 1, "status": "Filled"}]
        )

        with patch("trading_bot.cli.bot.strategy.evaluate",
                    return_value={"pass": True, "reasons": [], "price": 100.0}):
            with patch("trading_bot.cli.bot.subprocess.run",
                        return_value=MagicMock(returncode=0, stdout="ok", stderr="")):
                out, code = _run_main(["--symbol", "AAPL"])

        self.assertIsNone(code)
        self.assertIn("SUCCESS", out)

    def test_failed_subprocess_reports_failure_ignoring_stale_csv_row(self, mock_client_cls, mock_load_dotenv):
        """Regression: trade.py exits nonzero (writing NO row) on a
        rejected order -- bot.py must not fall back to trusting a stale
        trades.csv row left over from an earlier, unrelated trade."""
        mock_client_cls.return_value = MagicMock()
        self._write_trades(
            [{"timestamp_iso": "2026-07-13T10:00:00+00:00", "symbol": "MSFT", "side": "BUY",
              "size": 1, "fill_price": 200, "order_id": 1, "status": "Filled"}]
        )

        with patch("trading_bot.cli.bot.strategy.evaluate",
                    return_value={"pass": True, "reasons": [], "price": 100.0}):
            with patch(
                "trading_bot.cli.bot.subprocess.run",
                return_value=MagicMock(returncode=1, stdout="", stderr="ORDER NOT SUCCESSFUL (status=Rejected)"),
            ):
                out, code = _run_main(["--symbol", "AAPL"])

        self.assertEqual(code, 1)
        self.assertIn("FAILURE", out)
        self.assertNotIn("SUCCESS:", out)  # not "SUCCESS" -- would false-match "NOT SUCCESSFUL"

    def test_subprocess_timeout_exits_1(self, mock_client_cls, mock_load_dotenv):
        mock_client_cls.return_value = MagicMock()

        with patch("trading_bot.cli.bot.strategy.evaluate",
                    return_value={"pass": True, "reasons": [], "price": 100.0}):
            with patch(
                "trading_bot.cli.bot.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="trade.py", timeout=30),
            ):
                _, code = _run_main(["--symbol", "AAPL"])

        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
