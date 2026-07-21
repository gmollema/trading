# tests/integration/test_ibkr_client_it.py
from datetime import datetime
import time

import pytest
import pytz
from ib_async import Stock, LimitOrder

DEAD_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"}
CANCEL_CONFIRMED = {"Cancelled", "ApiCancelled"}


def is_us_market_open():
    """Controleert of de Amerikaanse markt (RTH) momenteel open is.
    NB: houdt geen rekening met beursvakanties of halve handelsdagen."""
    us_tz = pytz.timezone("US/Eastern")
    now_us = datetime.now(us_tz)

    if now_us.weekday() > 4:  # 5 = zaterdag, 6 = zondag
        return False

    market_start = now_us.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = now_us.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_start <= now_us <= market_end


def close_open_orders(paper_client):
    for trade in paper_client.ib.trades():
        if trade.orderStatus.status not in DEAD_STATUSES:
            paper_client.ib.cancelOrder(trade.order)


@pytest.mark.integration
def test_real_paper_trade(paper_client):
    """Test de volledige flow met dynamische status-verificatie op basis van markturen."""

    # 0. Pre-test cleanup
    close_open_orders(paper_client)

    # 1. Define contract and qualify it
    symbol = "AAPL"
    contract = Stock(symbol, "SMART", "USD")
    paper_client.ib.qualifyContracts(contract)

    # 2. Limit order ver onder de marktprijs: wordt geaccepteerd maar nooit gevuld
    order = LimitOrder("BUY", 1, lmtPrice=100.0, outsideRth=True)
    trade = paper_client.ib.placeOrder(contract, order)

    try:
        # 3. Verwachte statussen op basis van markturen
        if is_us_market_open():
            expected_statuses = {"PreSubmitted", "Submitted", "Filled"}
        else:
            expected_statuses = {"PendingSubmit", "PreSubmitted"}

        # 4. Wacht max 3 seconden tot de status klopt
        deadline = time.time() + 3.0
        while trade.orderStatus.status not in expected_statuses:
            if time.time() > deadline:
                context = "binnen" if is_us_market_open() else "buiten"
                pytest.fail(
                    f"Status was {trade.orderStatus.status} ({context} reguliere handelsuren), "
                    f"verwacht één van: {expected_statuses}"
                )
            paper_client.ib.sleep(0.1)

    finally:
        # 5. Teardown: annuleer de order, ook als de test hierboven faalt
        if trade.orderStatus.status not in DEAD_STATUSES:
            paper_client.ib.cancelOrder(trade.order)
            cancel_deadline = time.time() + 5.0
            while trade.orderStatus.status not in CANCEL_CONFIRMED:
                if time.time() > cancel_deadline:
                    break
                paper_client.ib.sleep(0.1)

    # 6. Bevestig buiten de finally dat de cancel gelukt is (alleen als niet gevuld)
    if trade.orderStatus.status != "Filled":
        assert trade.orderStatus.status in CANCEL_CONFIRMED, (
            f"Order niet netjes geannuleerd, status: {trade.orderStatus.status}"
        )


if __name__ == "__main__":
    pass