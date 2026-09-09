"""Thin wrapper around ib_async's IB client for connect / order / disconnect."""

from __future__ import annotations

import time

from datetime import date, datetime

from ib_async import IB, ContFuture, Future, MarketOrder, Stock, Trade

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

    def front_future(self, symbol: str, exchange: str, min_days_to_expiry: int) -> Future:
        """The nearest ES/MES contract with at least `min_days_to_expiry`
        days of life left, as a fully qualified Future.

        Resolved via ContFuture, which is IBKR's own pointer at the
        current front month -- qualifying it yields a conId that maps
        straight back to a concrete, tradeable Future. ContFuture ITSELF
        must never be sent as an order: it is a data construct stitched
        across expiries and has no deliverable.

        `min_days_to_expiry` is what stops the bot buying a contract that
        expires mid-trade. A position held across expiry is either
        auto-liquidated by the broker or, worse, settled -- neither is
        something the backtest models. When the front month is inside the
        window, the next expiry is used instead, which costs a small
        amount of basis and no correctness.

        Raises:
            RuntimeError: if no contract can be qualified.
        """
        cont = ContFuture(symbol, exchange)
        if not self.ib.qualifyContracts(cont):
            raise RuntimeError(f"could not qualify ContFuture {symbol} on {exchange}")

        details = self.ib.reqContractDetails(Future(symbol=symbol, exchange=exchange))
        candidates = []
        for d in details:
            raw = d.contract.lastTradeDateOrContractMonth
            if not raw or len(raw) < 8:
                continue
            expiry = datetime.strptime(raw[:8], "%Y%m%d").date()
            days = (expiry - date.today()).days
            if days >= min_days_to_expiry:
                candidates.append((expiry, d.contract))
        if not candidates:
            raise RuntimeError(
                f"no {symbol} contract with >= {min_days_to_expiry} days to expiry")
        candidates.sort(key=lambda c: c[0])
        contract = candidates[0][1]
        if not self.ib.qualifyContracts(contract):
            raise RuntimeError(f"could not qualify {symbol} {contract.lastTradeDateOrContractMonth}")
        return contract

    def place_futures_order(self, contract: Future, side: str, quantity: int) -> Trade:
        """Market order on an already-qualified futures contract.

        Takes a contract rather than a symbol precisely so the caller has
        to have resolved the expiry first -- see front_future. Unlike the
        stock path, outsideRth is left at its default: ES trades nearly
        around the clock, so "outside regular hours" is not a meaningful
        distinction, and this bot deliberately fills at the RTH open.
        """
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid order side: {side!r} (expected 'BUY' or 'SELL')")
        if quantity < 1:
            raise ValueError(f"Invalid quantity: {quantity!r}")

        trade = self.ib.placeOrder(contract, MarketOrder(side, quantity))
        deadline = time.time() + ORDER_SETTLE_TIMEOUT_SECS
        while time.time() < deadline:
            self.ib.sleep(0.25)
            status = trade.orderStatus.status
            if status in SETTLED_STATUSES or status in ACCEPTED_STATUSES:
                break
        else:
            raise TimeoutError(
                f"Order for {contract.localSymbol or contract.symbol} did not settle within "
                f"{ORDER_SETTLE_TIMEOUT_SECS}s (status={trade.orderStatus.status})"
            )
        return trade

    def qualify_stock(self, symbol: str, exchange: str = "SMART",
                      currency: str = "USD", primary_exchange: str = "") -> Stock:
        """A fully qualified Stock/ETF contract.

        `exchange` and `currency` are parameters, unlike place_order's
        hardcoded SMART/USD, because a EU-accessible leveraged ETP is not
        a US-listed one: the UCITS S&P 500 3x products trade on LSE and
        Euronext, and SMART/USD would either fail to qualify or -- worse
        -- resolve an ambiguous ticker to the wrong listing.
        `primary_exchange` disambiguates tickers that exist on several
        venues; pass it whenever qualification returns more than one.

        Raises:
            RuntimeError: if the contract cannot be qualified, or if it is
                ambiguous and no primary_exchange was given.
        """
        stock = Stock(symbol, exchange, currency)
        if primary_exchange:
            stock.primaryExchange = primary_exchange
        qualified = self.ib.qualifyContracts(stock)
        if not qualified:
            raise RuntimeError(
                f"could not qualify {symbol} on {exchange} in {currency} -- check the "
                f"ticker, and that the account has trading permissions for that venue")
        if len(qualified) > 1 and not primary_exchange:
            venues = ",".join(sorted({c.primaryExchange or c.exchange for c in qualified}))
            raise RuntimeError(
                f"{symbol} on {exchange} is ambiguous across {venues} -- "
                f"set primary_exchange to pick one")
        return qualified[0]

    def place_stock_order(self, contract: Stock, side: str, quantity: int,
                          outside_rth: bool = False) -> Trade:
        """Market order on an already-qualified stock/ETF contract.

        outside_rth defaults to FALSE here, the opposite of place_order.
        That method's True is right for liquid US large caps; it is wrong
        for a leveraged ETP, whose spread outside its home session widens
        to multiples of the edge this strategy is trying to collect. A
        rejected order is recoverable; a 1% slip on entry is not.

        Takes a contract rather than a symbol so the caller has to have
        resolved the venue first -- same reasoning as place_futures_order.
        """
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid order side: {side!r} (expected 'BUY' or 'SELL')")
        if quantity < 1:
            raise ValueError(f"Invalid quantity: {quantity!r}")

        order = MarketOrder(side, quantity)
        order.outsideRth = outside_rth
        trade = self.ib.placeOrder(contract, order)

        deadline = time.time() + ORDER_SETTLE_TIMEOUT_SECS
        while time.time() < deadline:
            self.ib.sleep(0.25)
            status = trade.orderStatus.status
            if status in SETTLED_STATUSES or status in ACCEPTED_STATUSES:
                break
        else:
            raise TimeoutError(
                f"Order for {contract.symbol} did not settle within "
                f"{ORDER_SETTLE_TIMEOUT_SECS}s (status={trade.orderStatus.status})"
            )
        return trade

    def net_liquidation(self) -> tuple[float, str]:
        """(equity, currency) from the account summary, in the account's
        BASE currency.

        Returned with its currency rather than as a bare float because a
        small account is very likely not denominated in the currency of
        the thing it is buying, and silently treating EUR equity as USD
        would oversize every order by the FX rate.

        Raises:
            RuntimeError: if the summary has no NetLiquidation row.
        """
        for row in self.ib.accountSummary():
            if row.tag == "NetLiquidation":
                return float(row.value), row.currency
        raise RuntimeError("account summary returned no NetLiquidation row")

    def disconnect(self) -> None:
        self.ib.disconnect()