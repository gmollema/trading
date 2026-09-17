"""Fetch 4-hour bars for RSI2 strategy intraday backtest.

yfinance has limited historical intraday data (~1-2 years), so this fetches
recent 4-hour bars for proof-of-concept vs daily bars.

Usage:
    python -m trading_bot.cli.rsi2_fetch_4h_data --symbols ^GSPC,^IXIC
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

DATA_DIR = Path("backtest_data/4h_index")


def fetch_4h_bars(symbols: list[str], days: int = 365) -> dict[str, pd.DataFrame]:
    """Fetch 4-hour bars for symbols over past N days.

    Returns dict of {symbol: dataframe with OHLC}.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    results = {}
    for symbol in symbols:
        print(f"Fetching 4-hour bars for {symbol} ({days} days)...", end=" ")
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="4h",
                auto_adjust=False,
                progress=False,
                threads=False
            )
            if len(df) > 0:
                df = df.dropna()
                df.index.name = "Date"
                results[symbol] = df[["Open", "High", "Low", "Close"]]
                print(f"✓ {len(df)} bars")
            else:
                print("✗ No data")
        except Exception as e:
            print(f"✗ Error: {e}")

    return results


def save_4h_bars(data: dict[str, pd.DataFrame]) -> None:
    """Save 4-hour bars to CSV files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol, df in data.items():
        safe_name = symbol.lstrip("^").replace("=", "_") + ".csv"
        path = DATA_DIR / safe_name
        df_copy = df.reset_index()
        df_copy.columns = ["Date", "Open", "High", "Low", "Close"]
        df_copy.to_csv(path, index=False)
        print(f"Saved: {path} ({len(df)} bars)")


def load_4h_bars(symbol: str) -> dict:
    """Load 4-hour bars from CSV into backtest format."""
    path = DATA_DIR / (symbol.lstrip("^").replace("=", "_") + ".csv")
    if not path.exists():
        raise FileNotFoundError(f"No 4-hour data at {path} — run fetch first")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date").reset_index(drop=True)

    return {
        "date": list(df["Date"]),
        "open": df["Open"].tolist(),
        "high": df["High"].tolist(),
        "low": df["Low"].tolist(),
        "close": df["Close"].tolist(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=str, default="^GSPC,^IXIC",
                        help="comma-separated symbols to fetch")
    parser.add_argument("--days", type=int, default=365,
                        help="how many days of history to fetch")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",")]

    print(f"Fetching {args.days} days of 4-hour bars for {symbols}\n")
    data = fetch_4h_bars(symbols, args.days)

    if data:
        save_4h_bars(data)
        print(f"\nSaved to {DATA_DIR}")
        return 0
    else:
        print("No data fetched")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
