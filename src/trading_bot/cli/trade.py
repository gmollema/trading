"""Order-execution CLI, invoked as a subprocess by bot.py.

Usage:
    python -m trading_bot.cli.trade --symbol NVDA --side BUY --size 3
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from trading_bot.broker.ibkr_client import IBKRClient

TRADES_CSV = Path("trades.csv")
TRADE_LOG = Path("trade.log")
CSV_HEADER = ["timestamp_iso", "symbol", "side", "size", "fill_price", "order_id", "status"]
FAILED_STATUSES = {"Cancelled", "ApiCancelled", "Inactive", "Rejected"}

# Capture ib_async's internal log (includes rejection reasons) to trade.log,
# so print_trade_log_tail() has something to show on a rejected order.
logging.basicConfig(
    filename=str(TRADE_LOG),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def print_trade_log_tail(n: int = 20) -> None:
    if TRADE_LOG.exists():
        lines = TRADE_LOG.read_text().splitlines()
        print(f"---- trade.log (last {n} lines) ----")
        for line in lines[-n:]:
            print(line)
        print("------------------------------------")
    else:
        print("(no trade.log found)")


def get_current_price(ibkr: IBKRClient, contract) -> float | None:
    """Best-effort current price for `contract`, for use as a fill-price
    stand-in when an order's avgFillPrice isn't populated yet (see
    append_trade_row's caller below). Mirrors smc_cycle.get_current_price."""
    ticker = ibkr.ib.reqMktData(contract, "", False, False)
    ibkr.ib.sleep(2)
    price = ticker.marketPrice()
    if price is None or price != price or price <= 0:  # NaN / missing
        last = ticker.last
        price = last if last and last == last and last > 0 else None
    ibkr.ib.cancelMktData(contract)
    return float(price) if price else None


def ensure_trades_csv() -> None:
    if not TRADES_CSV.exists():
        with TRADES_CSV.open("w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_trade_row(row: dict, _retried: bool = False) -> None:
    try:
        with TRADES_CSV.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)
    except (PermissionError, OSError):
        if _retried:
            print("trades.csv is locked by another process; retry failed.")
            sys.exit(1)
        time.sleep(1)
        append_trade_row(row, _retried=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", required=True, choices=["BUY", "SELL"])
    parser.add_argument("--size", required=True, type=int)
    args = parser.parse_args()

    load_dotenv()

    ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.environ.get("IBKR_PORT", "7497"))
    ibkr_exec_client_id = int(os.environ.get("IBKR_EXEC_CLIENT_ID", "3"))

    ensure_trades_csv()

    ibkr = None
    try:
        try:
            ibkr = IBKRClient(ibkr_host, ibkr_port, ibkr_exec_client_id)
        except Exception as e:
            print(f"EXCEPTION connecting to IBKR: {e}")
            sys.exit(1)

        trade = ibkr.place_order(args.symbol, args.side, args.size)

        status = trade.orderStatus.status
        # avgFillPrice is only populated once the fill actually settles --
        # IBKRClient.place_order's wait loop can return as soon as the
        # order merely reaches "Submitted", before that happens. Falling
        # back to 0 (as this used to) silently corrupts every downstream
        # P&L calc from trades.csv (confirmed in practice: two live rows,
        # CRWD and CTAS, both status "Submitted" fill_price 0).
        fill_price = trade.orderStatus.avgFillPrice or get_current_price(ibkr, trade.contract) or 0
        order_id = trade.order.orderId

        if status in FAILED_STATUSES:
            print(f"ORDER NOT SUCCESSFUL (status={status})")
            print_trade_log_tail()
            sys.exit(1)

        row = {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "symbol": args.symbol,
            "side": args.side,
            "size": args.size,
            "fill_price": fill_price,
            "order_id": order_id,
            "status": status,
        }
        append_trade_row(row)

        print(f"Order ID: {order_id}")
        print(f"Fill price: {fill_price}")
        print(f"Status: {status}")

    except Exception as e:
        print(f"EXCEPTION: {e}")
        sys.exit(1)
    finally:
        if ibkr is not None:
            ibkr.disconnect()


if __name__ == "__main__":
    main()
