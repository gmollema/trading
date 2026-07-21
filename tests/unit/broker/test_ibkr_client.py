import unittest
from unittest.mock import MagicMock, patch

from trading_bot.broker.ibkr_client import IBKRClient


class TestIBKRClient(unittest.TestCase):

    def setUp(self):
        patcher = patch("trading_bot.broker.ibkr_client.IB")
        mock_ib_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_ib = mock_ib_class.return_value
        self.client = IBKRClient("127.0.0.1", 7497, 1)

    def test_init_connects(self):
        self.mock_ib.connect.assert_called_once_with("127.0.0.1", 7497, 1)

    def test_init_requests_delayed_market_data(self):
        """Paper accounts generally have no live data entitlement -- without
        this, every price fetch silently falls back to 0.0."""
        self.mock_ib.reqMarketDataType.assert_called_once_with(3)

    def test_place_order_success(self):
        mock_contract = MagicMock()
        self.mock_ib.qualifyContracts.return_value = [mock_contract]

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Submitted"
        self.mock_ib.placeOrder.return_value = mock_trade

        trade = self.client.place_order("AAPL", "BUY", 100)

        self.mock_ib.qualifyContracts.assert_called_once()
        self.mock_ib.placeOrder.assert_called_once()
        self.assertEqual(trade, mock_trade)

    def test_place_order_fails_qualification(self):
        """Test that the system raises an error if the stock symbol is invalid."""
        self.mock_ib.qualifyContracts.return_value = []

        with self.assertRaises(RuntimeError):
            self.client.place_order("FAKE_SYMBOL", "BUY", 10)

    def test_place_order_rejects_invalid_side(self):
        """Een ongeldige side moet direct falen, vóór enig contact met de broker."""
        with self.assertRaises(ValueError):
            self.client.place_order("AAPL", "BYU", 100)
        self.mock_ib.qualifyContracts.assert_not_called()

    def test_place_order_settles_on_filled(self):
        """Test dat de loop stopt zodra de status 'Filled' is."""
        self.mock_ib.qualifyContracts.return_value = [MagicMock()]

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        self.mock_ib.placeOrder.return_value = mock_trade

        trade = self.client.place_order("AAPL", "BUY", 100)

        self.assertEqual(trade.orderStatus.status, "Filled")
        # De loop moet minstens één keer gedraaid hebben
        self.assertTrue(self.mock_ib.sleep.called)

    def test_place_order_accepts_presubmitted(self):
        """PreSubmitted (buiten RTH) telt als geaccepteerd — geen timeout."""
        self.mock_ib.qualifyContracts.return_value = [MagicMock()]

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "PreSubmitted"
        self.mock_ib.placeOrder.return_value = mock_trade

        trade = self.client.place_order("AAPL", "BUY", 100)

        self.assertEqual(trade, mock_trade)

    def test_place_order_timeout(self):
        """Status blijft 'PendingSubmit': na de deadline moet TimeoutError volgen."""
        self.mock_ib.qualifyContracts.return_value = [MagicMock()]

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "PendingSubmit"
        self.mock_ib.placeOrder.return_value = mock_trade

        with patch("trading_bot.broker.ibkr_client.time.time") as mock_time:
            # deadline = 1000 + 10 = 1010; checks: 1005 (loop draait), 1011 (deadline verstreken)
            mock_time.side_effect = [1000, 1005, 1011, 1020, 1030]

            with self.assertRaises(TimeoutError):
                self.client.place_order("AAPL", "BUY", 100)


if __name__ == "__main__":
    unittest.main()