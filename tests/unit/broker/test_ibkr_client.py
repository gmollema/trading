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


class TestQualifyStock(unittest.TestCase):
    """qualifyContracts returns one slot PER REQUESTED CONTRACT, with None
    in the slot for an unknown or ambiguous one -- so `[None]` is truthy
    and a bare truthiness check lets a None contract through."""

    def setUp(self):
        patcher = patch("trading_bot.broker.ibkr_client.IB")
        mock_ib_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_ib = mock_ib_class.return_value
        self.client = IBKRClient("127.0.0.1", 7497, 1)

    def test_a_none_slot_raises_rather_than_returning_none(self):
        """The regression: 3USL requested on 'LSE' (its venue is 'LSEETF')
        returned [None], which surfaced later as AttributeError
        'NoneType' has no attribute 'includeExpired'."""
        self.mock_ib.qualifyContracts.return_value = [None]
        with self.assertRaises(RuntimeError) as ctx:
            self.client.qualify_stock("3USL", "LSE", "USD")
        self.assertIn("LSEETF", str(ctx.exception))

    def test_an_empty_list_also_raises(self):
        self.mock_ib.qualifyContracts.return_value = []
        with self.assertRaises(RuntimeError):
            self.client.qualify_stock("NOPE", "LSEETF", "USD")

    def test_a_single_match_is_returned(self):
        contract = MagicMock()
        self.mock_ib.qualifyContracts.return_value = [contract]
        self.assertIs(self.client.qualify_stock("3USL", "LSEETF", "USD"), contract)

    def test_multiple_matches_raise_and_name_the_venues(self):
        a, b = MagicMock(), MagicMock()
        a.primaryExchange, b.primaryExchange = "LSEETF", "BVME.ETF"
        self.mock_ib.qualifyContracts.return_value = [a, b]
        with self.assertRaises(RuntimeError) as ctx:
            self.client.qualify_stock("3USL", "SMART", "USD")
        self.assertIn("BVME.ETF", str(ctx.exception))

    def test_place_order_no_longer_orders_against_a_none_contract(self):
        """Same latent bug in the pre-existing stock path: a None slot
        would have reached placeOrder."""
        self.mock_ib.qualifyContracts.return_value = [None]
        with self.assertRaises(RuntimeError):
            self.client.place_order("NOPE", "BUY", 1)
        self.mock_ib.placeOrder.assert_not_called()


class TestPlaceStockOrder(unittest.TestCase):
    def setUp(self):
        patcher = patch("trading_bot.broker.ibkr_client.IB")
        mock_ib_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_ib = mock_ib_class.return_value
        self.client = IBKRClient("127.0.0.1", 7497, 1)
        self.trade = MagicMock()
        self.trade.orderStatus.status = "Submitted"
        self.mock_ib.placeOrder.return_value = self.trade

    def test_defaults_to_regular_hours_only(self):
        """The opposite of place_order's True: a leveraged ETP's spread
        outside its home session is wider than the edge collected."""
        self.client.place_stock_order(MagicMock(), "BUY", 4)
        order = self.mock_ib.placeOrder.call_args[0][1]
        self.assertFalse(order.outsideRth)

    def test_rejects_bad_side_and_quantity(self):
        for side, qty in (("HOLD", 1), ("BUY", 0), ("BUY", -3)):
            with self.assertRaises(ValueError):
                self.client.place_stock_order(MagicMock(), side, qty)
        self.mock_ib.placeOrder.assert_not_called()


class TestNetLiquidation(unittest.TestCase):
    def setUp(self):
        patcher = patch("trading_bot.broker.ibkr_client.IB")
        mock_ib_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_ib = mock_ib_class.return_value
        self.client = IBKRClient("127.0.0.1", 7497, 1)

    def _row(self, tag, value, currency):
        row = MagicMock()
        row.tag, row.value, row.currency = tag, value, currency
        return row

    def test_returns_the_value_with_its_currency(self):
        """Returned with the currency because a small account is very
        likely not denominated in the currency of what it buys."""
        self.mock_ib.accountSummary.return_value = [
            self._row("BuyingPower", "1234", "EUR"),
            self._row("NetLiquidation", "512.34", "EUR"),
        ]
        self.assertEqual(self.client.net_liquidation(), (512.34, "EUR"))

    def test_missing_row_raises(self):
        self.mock_ib.accountSummary.return_value = []
        with self.assertRaises(RuntimeError):
            self.client.net_liquidation()
