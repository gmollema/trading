"""Unit tests for trading_bot.cli.close_one. ib_async.IB is always mocked."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, mock_open, patch

from trading_bot.cli import close_one


def make_position(symbol, size):
    p = MagicMock()
    p.contract.symbol = symbol
    p.position = size
    return p


def make_trade_result(status, filled=None, avg_fill_price=None, fills=None):
    t = MagicMock()
    t.orderStatus.status = status
    t.orderStatus.avgFillPrice = avg_fill_price
    t.orderStatus.filled = filled
    t.fills = fills or []
    return t


class TestPrintTradeLogTail(unittest.TestCase):
    def test_missing_file_prints_placeholder(self):
        buf = io.StringIO()
        with patch("builtins.open", side_effect=FileNotFoundError()):
            with redirect_stdout(buf):
                close_one.print_trade_log_tail()

        self.assertIn("no trade.log found", buf.getvalue())

    def test_prints_last_n_lines_only(self):
        content = "".join(f"line{i}\n" for i in range(30))
        with patch("builtins.open", mock_open(read_data=content)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                close_one.print_trade_log_tail(n=5)

        out = buf.getvalue()
        self.assertIn("line29", out)
        self.assertNotIn("line24", out)


def _run_main(argv):
    buf = io.StringIO()
    with patch("sys.argv", ["close_one.py"] + argv), redirect_stdout(buf):
        try:
            close_one.main()
            code = None
        except SystemExit as e:
            code = e.code
    return buf.getvalue(), code


class TestMain(unittest.TestCase):
    def test_connection_failure_exits_1(self):
        mock_ib = MagicMock()
        mock_ib.connect.side_effect = ConnectionError("boom")

        with patch("trading_bot.cli.close_one.IB", return_value=mock_ib):
            out, code = _run_main(["--symbol", "MU"])

        self.assertEqual(code, 1)
        self.assertIn("CONNECTION FAILED", out)

    def test_not_connected_after_connect_exits_1(self):
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = False

        with patch("trading_bot.cli.close_one.IB", return_value=mock_ib):
            out, code = _run_main(["--symbol", "MU"])

        self.assertEqual(code, 1)

    def test_no_position_for_symbol_exits_0(self):
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.positions.return_value = [make_position("AAPL", 10)]

        with patch("trading_bot.cli.close_one.IB", return_value=mock_ib):
            out, code = _run_main(["--symbol", "MU"])

        self.assertEqual(code, 0)
        self.assertIn("NO POSITION TO CLOSE", out)

    def test_flat_position_exits_0(self):
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.positions.return_value = [make_position("MU", 0)]

        with patch("trading_bot.cli.close_one.IB", return_value=mock_ib):
            out, code = _run_main(["--symbol", "MU"])

        self.assertEqual(code, 0)
        self.assertIn("NO POSITION TO CLOSE", out)

    def test_qualify_failure_exits_1(self):
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.positions.return_value = [make_position("MU", 10)]
        mock_ib.qualifyContracts.return_value = []

        with patch("trading_bot.cli.close_one.IB", return_value=mock_ib):
            out, code = _run_main(["--symbol", "MU"])

        self.assertEqual(code, 1)
        self.assertIn("Could not qualify", out)

    def test_long_position_sells_and_succeeds(self):
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.positions.return_value = [make_position("MU", 100)]
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.placeOrder.return_value = make_trade_result("Filled", filled=100, avg_fill_price=95.5)

        with patch("trading_bot.cli.close_one.IB", return_value=mock_ib):
            out, code = _run_main(["--symbol", "MU"])

        self.assertIsNone(code)
        self.assertIn("Status: Filled", out)
        placed_order = mock_ib.placeOrder.call_args[0][1]
        self.assertEqual(placed_order.action, "SELL")
        self.assertEqual(placed_order.totalQuantity, 100)

    def test_short_position_buys_to_cover(self):
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.positions.return_value = [make_position("MU", -50)]
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.placeOrder.return_value = make_trade_result("Filled", filled=50, avg_fill_price=95.5)

        with patch("trading_bot.cli.close_one.IB", return_value=mock_ib):
            _run_main(["--symbol", "MU"])

        placed_order = mock_ib.placeOrder.call_args[0][1]
        self.assertEqual(placed_order.action, "BUY")
        self.assertEqual(placed_order.totalQuantity, 50)

    def test_rejected_close_exits_1(self):
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.positions.return_value = [make_position("MU", 10)]
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.placeOrder.return_value = make_trade_result("Rejected")

        with patch("trading_bot.cli.close_one.IB", return_value=mock_ib):
            out, code = _run_main(["--symbol", "MU"])

        self.assertEqual(code, 1)
        self.assertIn("CLOSE NOT SUCCESSFUL", out)

    def test_unexpected_exception_exits_1(self):
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.positions.side_effect = RuntimeError("boom")

        with patch("trading_bot.cli.close_one.IB", return_value=mock_ib):
            out, code = _run_main(["--symbol", "MU"])

        self.assertEqual(code, 1)
        self.assertIn("EXCEPTION", out)
        mock_ib.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
