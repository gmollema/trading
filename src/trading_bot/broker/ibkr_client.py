"""Thin wrapper around ib_async's IB client for connect / order / disconnect."""

from __future__ import annotations

import time

from ib_async import IB, MarketOrder, Stock, Trade

SETTLED_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"}
ACCEPTED_STATUSES = {"Submitted", "PreSubmitted"}
ORDER_SETTLE_TIMEOUT_SECS = 10

# 1 = live, 2 = frozen, 3 = delayed, 4 = delayed-frozen. Paper-trading
# accounts generally have no live data entitlements, so every price fetch
# would otherwise silently get NaN/no ticks and fall back to 0.0. Delayed
# (~15-20 min lagged) data requires no subscription. Reconsider this if
# this client is ever pointed at a live account with a real-time data
# subscription -- you'd want MARKET_DATA_TYPE = 1 there instead.
MARKET_DATA_TYPE = 3


class IBKRClient:
    """Small convenience wrapper around ib_async.IB for this bot's needs."""

    def __init__(self, host: str, port: int, client_id: int) -> None:
        self.ib = IB()
        self.ib.connect(host, port, client_id)
        self.ib.reqMarketDataType(MARKET_DATA_TYPE)

    def place_order(self, symbol: str, side: str, quantity: int) -> Trade:
        """Qualify the contract, place a market order, and wait for the
        order status to settle (i.e. move past PendingSubmit), up to
        ORDER_SETTLE_TIMEOUT_SECS seconds.

        Raises:
            ValueError: if side is not BUY or SELL.
            RuntimeError: if the contract cannot be qualified.
            TimeoutError: if the order does not settle within the timeout.
        """
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid order side: {side!r} (expected 'BUY' or 'SELL')")

        stock = Stock(symbol, "SMART", "USD")
        qualified = self.ib.qualifyContracts(stock)
        if not qualified:
            raise RuntimeError(f"Could not qualify contract for symbol: {symbol}")
        contract = qualified[0]

        order = MarketOrder(side, quantity)
        order.outsideRth = True

        trade = self.ib.placeOrder(contract, order)

        deadline = time.time() + ORDER_SETTLE_TIMEOUT_SECS
        while time.time() < deadline:
            self.ib.sleep(0.25)
            status = trade.orderStatus.status
            if status in SETTLED_STATUSES or status in ACCEPTED_STATUSES:
                break
        else:
            raise TimeoutError(
                f"Order for {symbol} did not settle within {ORDER_SETTLE_TIMEOUT_SECS}s "
                f"(status={trade.orderStatus.status})"
            )

        return trade

    def disconnect(self) -> None:
        self.ib.disconnect()