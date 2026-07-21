"""Unit tests for trading_bot.cli.trade. IBKRClient is always mocked."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from trading_bot.cli import trade

CSV_HEADER = ["timestamp_iso", "symbol", "side", "size", "fill_price", "order_id", "status"]


def make_trade(status, fill_price=100.0, order_id=1):
    t = MagicMock()
    t.orderStatus.status = status
    t.orderStatus.avgFillPrice = fill_price
    t.order.orderId = order_id
    return t


class TestPrintTradeLogTail(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmpdir.name) / "trade.log"
        self.patcher = patch.object(trade, "TRADE_LOG", self.log_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_missing_file_prints_placeholder(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            trade.print_trade_log_tail()

        self.assertIn("no trade.log found", buf.getvalue())

    def test_prints_last_n_lines_only(self):
        self.log_path.write_text("\n".join(f"line{i}" for i in range(30)) + "\n")

        buf = io.StringIO()
        with redirect_stdout(buf):
            trade.print_trade_log_tail(n=5)

        out = buf.getvalue()
        self.assertIn("line29", out)
        self.assertNotIn("line24", out)


class TestEnsureTradesCsv(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.trades_path = Path(self._tmpdir.name) / "trades.csv"
        self.patcher = patch.object(trade, "TRADES_CSV", self.trades_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_creates_file_with_header(self):
        trade.ensure_trades_csv()

        with self.trades_path.open() as f:
            self.assertEqual(next(csv.reader(f)), CSV_HEADER)

    def test_does_not_overwrite_existing_file(self):
        self.trades_path.write_text("existing\n")

        trade.ensure_trades_csv()

        self.assertEqual(self.trades_path.read_text(), "existing\n")


class TestAppendTradeRow(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.trades_path = Path(self._tmpdir.name) / "trades.csv"
        self.patcher = patch.object(trade, "TRADES_CSV", self.trades_path)
        self.patcher.start()
        trade.ensure_trades_csv()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_appends_row(self):
        row = {"timestamp_iso": "2026-07-13T10:00:00+00:00", "symbol": "AAPL", "side": "BUY",
               "size": 1, "fill_price": 100.0, "order_id": 1, "status": "Filled"}

        trade.append_trade_row(row)

        with self.trades_path.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "AAPL")

    def test_retries_once_then_gives_up_on_permission_error(self):
        # Path.open() doesn't dispatch through builtins.open at the Python
        # level, so patching Path.open directly is what's needed to force
        # the PermissionError branch.
        row = {"timestamp_iso": "x", "symbol": "AAPL", "side": "BUY",
               "size": 1, "fill_price": 100.0, "order_id": 1, "status": "Filled"}

        with patch.object(Path, "open", side_effect=PermissionError("locked")):
            with patch("trading_bot.cli.trade.time.sleep"):
                with self.assertRaises(SystemExit):
                    trade.append_trade_row(row)


def _run_main(argv):
    buf = io.StringIO()
    with patch("sys.argv", ["trade.py"] + argv), redirect_stdout(buf):
        try:
            trade.main()
            code = None
        except SystemExit as e:
            code = e.code
    return buf.getvalue(), code


@patch("trading_bot.cli.trade.load_dotenv")
class TestMain(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.trades_path = Path(self._tmpdir.name) / "trades.csv"
        self.patcher = patch.object(trade, "TRADES_CSV", self.trades_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self._tmpdir.cleanup()

    def test_connection_failure_exits_1(self, mock_load_dotenv):
        with patch("trading_bot.cli.trade.IBKRClient", side_effect=ConnectionError("boom")):
            out, code = _run_main(["--symbol", "AAPL", "--side", "BUY", "--size", "1"])

        self.assertEqual(code, 1)
        self.assertIn("EXCEPTION", out)

    def test_successful_order_appends_row(self, mock_load_dotenv):
        mock_client = MagicMock()
        mock_client.place_order.return_value = make_trade("Filled")

        with patch("trading_bot.cli.trade.IBKRClient", return_value=mock_client):
            out, code = _run_main(["--symbol", "AAPL", "--side", "BUY", "--size", "1"])

        self.assertIsNone(code)
        with self.trades_path.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "Filled")
        mock_client.disconnect.assert_called_once()

    def test_rejected_order_does_not_append_row(self, mock_load_dotenv):
        """Regression: 'Rejected' must be treated as a failure -- it was
        previously missing from FAILED_STATUSES, so a rejected order would
        fall through and get written to trades.csv as if it succeeded."""
        mock_client = MagicMock()
        mock_client.place_order.return_value = make_trade("Rejected")

        with patch("trading_bot.cli.trade.IBKRClient", return_value=mock_client):
            out, code = _run_main(["--symbol", "AAPL", "--side", "BUY", "--size", "1"])

        self.assertEqual(code, 1)
        self.assertIn("ORDER NOT SUCCESSFUL", out)
        with self.trades_path.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 0)

    def test_cancelled_order_does_not_append_row(self, mock_load_dotenv):
        mock_client = MagicMock()
        mock_client.place_order.return_value = make_trade("Cancelled")

        with patch("trading_bot.cli.trade.IBKRClient", return_value=mock_client):
            out, code = _run_main(["--symbol", "AAPL", "--side", "BUY", "--size", "1"])

        self.assertEqual(code, 1)
        with self.trades_path.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 0)

    def test_unexpected_exception_during_place_order_exits_1(self, mock_load_dotenv):
        mock_client = MagicMock()
        mock_client.place_order.side_effect = RuntimeError("qualification failed")

        with patch("trading_bot.cli.trade.IBKRClient", return_value=mock_client):
            out, code = _run_main(["--symbol", "AAPL", "--side", "BUY", "--size", "1"])

        self.assertEqual(code, 1)
        self.assertIn("EXCEPTION", out)
        mock_client.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
