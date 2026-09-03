"""Fetch and cache daily index bars for the 2-period RSI backtest.

Writes to its own directory (backtest_data/daily_index/) rather than
backtest_data/daily/, which holds the 500-odd S&P constituents on a
rolling 2-year yfinance window that the gap-and-go and SMC backtests
depend on. This strategy needs one symbol with two decades of history
instead, and the two caches have no business sharing a namespace.

Source is yfinance, same as backtest_fetch_data.py's daily leg. IBKR is
not used here: index history that far back is not something the paper TWS
connection reliably serves, and the strategy trades off daily closes, so
there is nothing IBKR's intraday depth would add.

Note on auto_adjust: passed False deliberately. ^GSPC pays no dividends
and has no splits, so adjustment is a no-op for it -- but for a tradeable
proxy like SPY it is NOT, and back-adjusted prices would silently rescale
the whole series and make the strategy's 200-POINT stop meaningless. The
video reports results in raw index points, so raw is what we cache.

Usage:
    python -m trading_bot.cli.rsi2_fetch_data
    python -m trading_bot.cli.rsi2_fetch_data --symbols ^GSPC,SPY --start 1993-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yfinance as yf

DAILY_INDEX_DIR = Path("backtest_data/daily_index")
# 2006, not 2008: the 200-day SMA filter needs 200 bars of warmup before
# the first tradeable day, and the video's test window opens in 2008.
DEFAULT_START = "2006-01-01"
DEFAULT_SYMBOLS = ["^GSPC"]


def safe_filename(symbol: str) -> str:
    """^GSPC -> GSPC.csv. The caret is legal on POSIX but not on Windows,
    and this repo runs on both."""
    return symbol.lstrip("^").replace("=", "_") + ".csv"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=None, help="default: today")
    parser.add_argument("--out-dir", type=str, default=str(DAILY_INDEX_DIR))
    parser.add_argument("--force", action="store_true", help="refetch symbols already cached")
    return parser.parse_args(argv)


def fetch_one(symbol: str, out_dir: Path, start: str, end: str | None, force: bool) -> str:
    """Returns 'cached' | 'fetched' | 'failed'."""
    path = out_dir / safe_filename(symbol)
    if path.exists() and not force:
        return "cached"
    try:
        hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=False)
    except Exception as exc:  # noqa: BLE001 - network/provider errors are all equally fatal here
        print(f"{symbol}: fetch failed ({exc})")
        return "failed"
    if hist.empty:
        print(f"{symbol}: fetch returned no rows")
        return "failed"
    hist.index.name = "Date"
    hist[["Open", "High", "Low", "Close"]].to_csv(path)
    print(f"{symbol}: {len(hist)} bars -> {path} ({hist.index[0].date()} .. {hist.index[-1].date()})")
    return "fetched"


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    results = [fetch_one(s, out_dir, args.start, args.end, args.force) for s in symbols]
    print(f"done: {results.count('fetched')} fetched, {results.count('cached')} cached, {results.count('failed')} failed")
    return 1 if "failed" in results else 0


if __name__ == "__main__":
    raise SystemExit(main())
