"""Fetch and cache historical hourly FX bars for the UT Bot backtest.

Data source: yfinance, "{pair}=X" tickers, interval="1h". Unlike
backtest_fetch_data.py's equity fetchers, this ALWAYS overwrites any
existing cache file rather than skipping it -- yfinance's 1h-interval
history is a rolling window anchored to "now" (documented as capped at
730 days, though it has returned further back than that in practice), so
re-running this is how the window rolls forward to include today;
skip-if-cached would defeat that entirely.

Usage:
    python -m trading_bot.cli.ut_bot_fetch_data
    python -m trading_bot.cli.ut_bot_fetch_data --pairs USDJPY,GBPUSD
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yfinance as yf

DATA_DIR = Path("backtest_data/fx_1h")
DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]


def fetch(pair: str, data_dir: Path = DATA_DIR) -> str:
    """Fetch and cache `pair`'s hourly bars. Returns a human-readable status
    string; never raises (a failure/empty response is reported, not thrown),
    so one bad pair can't abort the rest of a multi-pair run."""
    try:
        hist = yf.Ticker(f"{pair}=X").history(period="730d", interval="1h", auto_adjust=True)
    except Exception as e:
        return f"failed ({e})"
    if hist is None or hist.empty:
        return "failed (empty response)"

    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{pair}.csv"
    tmp_path = path.with_suffix(".csv.tmp")
    hist.to_csv(tmp_path)
    tmp_path.replace(path)
    return f"fetched ({len(hist)} rows, {hist.index[0].date()} .. {hist.index[-1].date()})"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=str, default=None, help="comma-separated override list")
    parser.add_argument("--data-dir", type=str, default=None, help="override the output directory")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pairs = [p.strip().upper() for p in args.pairs.split(",")] if args.pairs else DEFAULT_PAIRS
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    for pair in pairs:
        print(f"{pair}: {fetch(pair, data_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
