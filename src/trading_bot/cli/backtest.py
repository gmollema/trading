"""Backtest CLI for the Trend Join Long strategy (rules.json) against cached
history in backtest_data/ (see backtest_fetch_data.py).

KNOWN LIMITATION: the cached intraday bars are RTH-only (useRTH=True), so
the I1 "above premarket high" filter cannot be computed and is always
treated as passing. See src/trading_bot/backtest/filters.py.

Usage:
    python -m trading_bot.cli.backtest --tickers AAPL,MSFT
    python -m trading_bot.cli.backtest --start-date 2025-01-01 --end-date 2025-12-31
    python -m trading_bot.cli.backtest   # full cached S&P 500 universe
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from trading_bot.backtest.data import available_tickers
from trading_bot.backtest.engine import run_backtest
from trading_bot.cli.compute_perf import aggregate, pair_trades_fifo
from trading_bot.cli.trade import CSV_HEADER
from trading_bot.data.sp500_tickers import SP500_TICKERS

RULES_PATH = Path("rules.json")
TRADES_OUT_PATH = Path("backtest_trades.csv")
TRADES_DETAILED_OUT_PATH = Path("backtest_trades_detailed.csv")
EQUITY_OUT_PATH = Path("backtest_equity_curve.csv")
DETAILED_CSV_HEADER = CSV_HEADER + ["reason", "r_multiple"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated override list")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--initial-capital", type=float, default=None)
    return parser.parse_args(argv)


def write_trades_csv(trades: list[dict], path: Path = TRADES_OUT_PATH) -> None:
    """Write the compute_perf.py-compatible schema (trade.CSV_HEADER exactly).

    Trade dicts carry extra analysis-only fields (reason, r_multiple) not in
    CSV_HEADER -- extrasaction="ignore" drops them here rather than raising.
    """
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)


def write_trades_detailed_csv(trades: list[dict], path: Path = TRADES_DETAILED_OUT_PATH) -> None:
    """Write the same trades with the extra `reason`/`r_multiple` fields,
    for post-hoc analysis of what's driving wins/losses."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAILED_CSV_HEADER)
        writer.writeheader()
        writer.writerows(trades)


def write_equity_curve_csv(equity_curve: list[dict], path: Path = EQUITY_OUT_PATH) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "equity"])
        writer.writeheader()
        writer.writerows(equity_curve)


def main(argv=None) -> int:
    args = parse_args(argv)
    load_dotenv()

    rules = json.loads(RULES_PATH.read_text())

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = available_tickers(SP500_TICKERS)

    initial_capital = args.initial_capital
    if initial_capital is None:
        initial_capital = float(os.environ.get("PORTFOLIO_VALUE_USD", "100000"))

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    print(f"Backtesting {len(tickers)} tickers, initial capital ${initial_capital:,.2f}...")
    result = run_backtest(tickers, rules, initial_capital, start_date=start_date, end_date=end_date)
    trades = result["trades"]
    equity_curve = result["equity_curve"]

    write_trades_csv(trades)
    write_trades_detailed_csv(trades)
    write_equity_curve_csv(equity_curve)

    closed_pairs = pair_trades_fifo(pd.DataFrame(trades)) if trades else []
    summary = aggregate(closed_pairs)
    final_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital
    summary["initial_capital"] = round(initial_capital, 2)
    summary["final_equity"] = round(final_equity, 2)
    summary["total_return_pct"] = round((final_equity / initial_capital - 1) * 100, 2) if initial_capital else 0.0
    summary["tickers_backtested"] = len(tickers)

    print(json.dumps(summary, indent=2))
    print(
        f"Trades written to {TRADES_OUT_PATH} ({TRADES_DETAILED_OUT_PATH} with reason/R detail), "
        f"equity curve to {EQUITY_OUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
