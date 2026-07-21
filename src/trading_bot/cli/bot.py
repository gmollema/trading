"""Single-symbol dev CLI for the IBKR paper-trading bot.

Usage:
    python -m trading_bot.cli.bot --symbol NVDA --check-only
    python -m trading_bot.cli.bot --symbol NVDA
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from trading_bot import strategy
from trading_bot.broker.ibkr_client import IBKRClient

TRADES_CSV = Path("trades.csv")
CSV_HEADER = ["timestamp_iso", "symbol", "side", "size", "fill_price", "order_id", "status"]
ET_ZONE = ZoneInfo("America/New_York")
LIVE_PORTS = {7496, 4001}
SUBPROCESS_TIMEOUT_SECS = 30


def log(msg: str) -> None:
    now_et = datetime.now(ET_ZONE)
    print(f"[{now_et.strftime('%H:%M:%S')} ET] {msg}")


def ensure_trades_csv() -> None:
    if not TRADES_CSV.exists():
        with TRADES_CSV.open("w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def count_today_buys() -> int:
    today_et = datetime.now(ET_ZONE).strftime("%Y-%m-%d")
    count = 0
    with TRADES_CSV.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("side") == "BUY" and row.get("timestamp_iso", "").startswith(today_et):
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    # 1. Load .env
    load_dotenv()

    ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.environ.get("IBKR_PORT", "7497"))
    ibkr_client_id = int(os.environ.get("IBKR_CLIENT_ID", "2"))
    paper_trading = os.environ.get("PAPER_TRADING", "true").strip().lower() == "true"
    max_trade_size_usd = float(os.environ.get("MAX_TRADE_SIZE_USD", "0"))
    portfolio_value_usd = float(os.environ.get("PORTFOLIO_VALUE_USD", "0"))
    max_trades_per_day = int(os.environ.get("MAX_TRADES_PER_DAY", "0"))

    # 2. Hard guard: paper flag vs live port must agree
    if paper_trading and ibkr_port in LIVE_PORTS:
        sys.exit("ABORT: paper flag but live port")
    if not paper_trading and ibkr_port not in LIVE_PORTS:
        sys.exit("ABORT: live flag but paper port")

    # 3. trades.csv daily-limit gate
    ensure_trades_csv()
    today_buy_count = count_today_buys()
    if today_buy_count >= max_trades_per_day:
        log(f"MAX_TRADES_PER_DAY ({max_trades_per_day}) already reached today. Skipping.")
        sys.exit(0)

    # 4. Connect
    try:
        ibkr = IBKRClient(ibkr_host, ibkr_port, ibkr_client_id)
    except Exception as e:
        log(f"EXCEPTION connecting to IBKR: {e}")
        sys.exit(1)

    try:
        # 5. Evaluate strategy
        result = strategy.evaluate(args.symbol, ibkr.ib)
        log(f"strategy.evaluate result: {result}")

        # 6. check-only mode
        if args.check_only:
            sys.exit(0)

        # 7. Not pass
        if not result["pass"]:
            log(f"Reasons: {result['reasons']}")
            sys.exit(0)

        # 8. Size position
        price = result["price"]
        if price <= 0:
            log("Invalid price from strategy evaluation, cannot size position.")
            sys.exit(0)

        budget = min(max_trade_size_usd, portfolio_value_usd * 0.10)
        quantity = int(budget / price)
        if quantity < 1:
            log("position too small")
            sys.exit(0)

        # 9. Spawn trade.py subprocess
        log(f"Placing BUY order for {quantity} share(s) of {args.symbol} via trade.py")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m", "trading_bot.cli.trade",
                    "--symbol", args.symbol,
                    "--side", "BUY",
                    "--size", str(quantity),
                ],
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECS,
            )
        except subprocess.TimeoutExpired:
            log("trade.py subprocess exceeded 30s timeout. Killed.")
            sys.exit(1)

        print(proc.stdout)
        if proc.stderr:
            log(f"trade.py stderr: {proc.stderr}")

        # 10. trade.py exits nonzero (and writes NO row) on a rejected/
        # cancelled order -- check that FIRST, before trusting trades.csv's
        # last row, which would otherwise be a stale row from an earlier,
        # unrelated trade.
        if proc.returncode != 0:
            log(f"FAILURE: trade.py exited with code {proc.returncode}")
            sys.exit(1)

        with TRADES_CSV.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            log("No rows found in trades.csv after trade attempt.")
            sys.exit(1)

        last_row = rows[-1]
        if last_row.get("status") in {"Filled", "Submitted"}:
            log(f"SUCCESS: {last_row}")
        else:
            log(f"FAILURE: {last_row}")

    finally:
        ibkr.disconnect()


if __name__ == "__main__":
    main()
