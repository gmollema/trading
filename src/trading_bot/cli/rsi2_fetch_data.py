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


DAILY_LONG_DIR = Path("backtest_data/daily_long")
SP500_START = "2010-01-01"
BATCH = 50


def fetch_sp500_long(out_dir: Path, start: str, end: str | None, force: bool) -> int:
    """Batch-fetch deep history for the whole S&P constituent list.

    Separate from the single-symbol path for two reasons. yfinance's
    batch `download` does 50 tickers per request instead of 502 separate
    ones, which is the difference between a minute and half an hour. And
    `auto_adjust=True` is mandatory here, unlike for the index: raw stock
    prices contain splits, and a 2-for-1 split is a 50% single-day drop
    that RSI(2) would read as the mother of all dips. Adjusted prices are
    also what backtest_data/daily/ already holds, so the two agree.

    Writes to its own directory. backtest_data/daily/ is a rolling 2-year
    window the gap-and-go and SMC backtests depend on, and must not be
    overwritten with a different span.
    """
    import yfinance as yf

    from trading_bot.data.sp500_tickers import SP500_TICKERS

    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in SP500_TICKERS
            if force or not (out_dir / (t.replace(" ", "_") + ".csv")).exists()]
    if not todo:
        print(f"all {len(SP500_TICKERS)} tickers already cached in {out_dir}")
        return 0

    written = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        yahoo = {t: t.replace(" ", "-") for t in chunk}
        try:
            data = yf.download(list(yahoo.values()), start=start, end=end, interval="1d",
                               auto_adjust=True, group_by="ticker", progress=False,
                               threads=True)
        except Exception as exc:  # noqa: BLE001
            print(f"batch {i // BATCH + 1}: failed ({exc})")
            continue
        for ibkr, ysym in yahoo.items():
            try:
                df = data[ysym] if len(yahoo) > 1 else data
            except KeyError:
                continue
            df = df.dropna(subset=["Close"])
            if len(df) < 260:
                continue
            df.index.name = "Date"
            df[["Open", "High", "Low", "Close"]].to_csv(out_dir / (ibkr.replace(" ", "_") + ".csv"))
            written += 1
        print(f"batch {i // BATCH + 1}/{(len(todo) + BATCH - 1) // BATCH}: {written} written so far")
    print(f"done: {written} of {len(todo)} tickers -> {out_dir}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sp500", action="store_true",
                        help=f"batch-fetch deep history for every S&P constituent into {DAILY_LONG_DIR}")
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
    if args.sp500:
        start = SP500_START if args.start == DEFAULT_START else args.start
        return fetch_sp500_long(DAILY_LONG_DIR, start, args.end, args.force)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    results = [fetch_one(s, out_dir, args.start, args.end, args.force) for s in symbols]
    print(f"done: {results.count('fetched')} fetched, {results.count('cached')} cached, {results.count('failed')} failed")
    return 1 if "failed" in results else 0


if __name__ == "__main__":
    raise SystemExit(main())
