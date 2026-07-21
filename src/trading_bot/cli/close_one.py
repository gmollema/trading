#!/usr/bin/env python3
"""
Manually flatten a single stuck paper-trading position via TWS.

Run this on the machine where TWS (paper trading) is running with the API
enabled on port 7497.

Usage:
    python -m trading_bot.cli.close_one --symbol MU
"""

import argparse
import sys
import time
import logging

# --- Logging setup: capture ib_async's internal log (includes rejection reasons) ---
logging.basicConfig(
    filename="trade.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

try:
    from ib_async import IB, MarketOrder, Stock
except ImportError:
    print("ERROR: ib_async is not installed.")
    print("Run: pip install ib_async")
    sys.exit(1)


HOST = "127.0.0.1"
PORT = 7497  # paper trading
CLIENT_ID = 11  # different from buy_one.py's 10, to avoid collision
TIMEOUT_SECS = 10


def print_trade_log_tail(n=20):
    try:
        with open("trade.log", "r") as f:
            lines = f.readlines()
        print("---- trade.log (last {} lines) ----".format(n))
        for line in lines[-n:]:
            print(line.rstrip())
        print("------------------------------------")
    except FileNotFoundError:
        print("(no trade.log found)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    symbol = args.symbol

    ib = IB()

    try:
        try:
            ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
        except Exception as e:
            print(f"CONNECTION FAILED: {e}")
            print("Check that TWS is running, paper trading mode is active,")
            print("API connections are enabled (File > Global Configuration > API > Settings),")
            print(f"and the socket port matches {PORT}.")
            sys.exit(1)

        if not ib.isConnected():
            print("CONNECTION FAILED: ib.isConnected() returned False after connect().")
            sys.exit(1)

        # Find the position
        positions = ib.positions()
        target_position = None
        for pos in positions:
            if pos.contract.symbol == symbol:
                target_position = pos
                break

        if target_position is None:
            print("NO POSITION TO CLOSE")
            sys.exit(0)

        position_size = target_position.position  # positive = long, negative = short

        # IMPORTANT: do NOT reuse target_position.contract directly. IBKR
        # reports the position's contract with its actual settlement
        # exchange (e.g. NASDAQ), and placing an order against that contract
        # routes it DIRECTLY to that exchange - which trips IBKR's
        # "Precautionary Settings" block on direct-routed orders (Error
        # 10311) and the order gets silently Cancelled. Rebuild a
        # SMART-routed contract instead, same as buy_one.py does.
        stock = Stock(symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(stock)
        if not qualified:
            print(f"ERROR: Could not qualify SMART contract for {symbol}.")
            print_trade_log_tail()
            sys.exit(1)
        contract = qualified[0]

        if position_size == 0:
            print("NO POSITION TO CLOSE")
            sys.exit(0)

        # Determine order direction and quantity to flatten the position
        action = "SELL" if position_size > 0 else "BUY"
        qty = abs(position_size)

        order = MarketOrder(action, qty)
        order.outsideRth = True

        trade = ib.placeOrder(contract, order)

        # Wait up to TIMEOUT_SECS for status to settle
        deadline = time.time() + TIMEOUT_SECS
        settled_statuses = {
            "Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"
        }
        while time.time() < deadline:
            ib.sleep(0.25)
            status = trade.orderStatus.status
            if status in settled_statuses or status == "Submitted":
                break

        status = trade.orderStatus.status
        avg_fill_price = trade.orderStatus.avgFillPrice
        filled_qty = trade.orderStatus.filled

        fill_price_str = "pending"
        if trade.fills:
            last_fill = trade.fills[-1]
            fill_price_str = str(last_fill.execution.price)
        elif avg_fill_price and avg_fill_price > 0:
            fill_price_str = str(avg_fill_price)

        sold_qty_str = str(filled_qty) if filled_qty else str(qty) + " (requested)"

        print(f"Sold quantity: {sold_qty_str}")
        print(f"Fill price: {fill_price_str}")
        print(f"Status: {status}")

        if status in ("Rejected", "ApiCancelled", "Inactive", "Cancelled"):
            print(f"CLOSE NOT SUCCESSFUL (status={status}). Showing trade.log for details:")
            print_trade_log_tail()
            sys.exit(1)

    except Exception as e:
        print(f"EXCEPTION: {e}")
        print_trade_log_tail()
        sys.exit(1)
    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()