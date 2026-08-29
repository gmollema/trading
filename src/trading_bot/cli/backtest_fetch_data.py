"""Fetch and cache historical daily + 5-min intraday bars for backtesting.

Resumable by design: each ticker's data is written to disk immediately
after fetching, and any ticker whose cache file already exists is skipped.
An interrupted run can simply be re-invoked to continue where it left off,
without losing or re-fetching completed tickers.

Data sources:
  - Daily bars (2y, for D1-D3's prior-day-high / SMA200 / gap%): yfinance.
  - 5-min intraday bars (1y, for I1-I3's premarket-high / today-HOD / RVOL):
    IBKR's own reqHistoricalData, via the same paper TWS connection the
    live bot uses. This goes back ~1 year vs yfinance's ~60-day cap for
    5-min bars, and needs no separate data-provider signup.

Usage:
    python -m trading_bot.cli.backtest_fetch_data
    python -m trading_bot.cli.backtest_fetch_data --tickers AAPL,MSFT
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from ib_async import Stock

from trading_bot.backtest.data import DAILY_DIR, INTRADAY_DIR, safe_filename
from trading_bot.broker.ibkr_client import IBKRClient
from trading_bot.data.sp500_tickers import SP500_TICKERS


INTRADAY_PREMARKET_DIR = Path("backtest_data/intraday_5m_premarket")

DAILY_PERIOD = "2y"
INTRADAY_DURATION = "1 Y"
# IBKR enforces max 60 historical-data requests per 10 min per connection.
# 12s spacing (~50/10min) left too little headroom once retries during a
# bad stretch pushed the effective rate over the limit, and TWS dropped
# the connection outright (confirmed empirically). 18s (~33/10min) leaves
# real margin even with retries.
IBKR_PACING_SLEEP_SECS = 18
FETCH_CLIENT_ID = 95


def ibkr_to_yahoo(ticker: str) -> str:
    return ticker.replace(" ", "-")


def fetch_daily(ticker_ibkr: str, daily_dir: Path = DAILY_DIR, end_date: str | None = None) -> str:
    """Returns 'cached' | 'fetched' | 'failed'.

    `end_date` (YYYY-MM-DD) fetches a 2y window ENDING there instead of
    yfinance's default period="2y" (which is always anchored to "now") --
    needed to backtest a different historical year than the one already
    cached in DAILY_DIR."""
    path = daily_dir / safe_filename(ticker_ibkr)
    if path.exists():
        return "cached"

    ticker_yahoo = ibkr_to_yahoo(ticker_ibkr)
    try:
        if end_date is not None:
            end = pd.Timestamp(end_date)
            start = end - pd.Timedelta(days=730)
            hist = yf.Ticker(ticker_yahoo).history(
                start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=True,
            )
        else:
            hist = yf.Ticker(ticker_yahoo).history(period=DAILY_PERIOD, interval="1d", auto_adjust=True)
    except Exception:
        return "failed"
    if hist is None or hist.empty:
        return "failed"

    daily_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    hist.to_csv(tmp_path)
    tmp_path.replace(path)
    return "fetched"


def fetch_intraday(
    ib, ticker_ibkr: str, intraday_dir: Path = INTRADAY_DIR, use_rth: bool = True, end_date: str | None = None
) -> str:
    """Returns 'cached' | 'fetched' | 'failed'.

    IBKR's historical data farms occasionally take a while to respond to
    the first request for a symbol -- a single retry with a longer timeout
    clears up most of these transient hiccups (confirmed empirically: a
    60s timeout failed twice in a row, a 90s timeout then succeeded).

    `use_rth=False` (paired with `intraday_dir=INTRADAY_PREMARKET_DIR`)
    additionally fetches premarket bars, for gap-risk analysis that the
    default RTH-only cache can't support (see backtest/filters.py's I1
    limitation note).

    `end_date` (YYYY-MM-DD) fetches the 1y window ENDING there instead of
    "now" -- needed to backtest a different historical year than the one
    already cached in INTRADAY_DIR."""
    path = intraday_dir / safe_filename(ticker_ibkr)
    if path.exists():
        return "cached"

    try:
        stock = Stock(ticker_ibkr, "SMART", "USD")
        qualified = ib.qualifyContracts(stock)
        if not qualified:
            return "failed"
        contract = qualified[0]
    except Exception:
        return "failed"

    end_date_time = f"{end_date.replace('-', '')} 23:59:59" if end_date else ""

    bars = None
    for timeout_secs in (60, 90):
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_date_time,
                durationStr=INTRADAY_DURATION,
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
                timeout=timeout_secs,
            )
        except Exception:
            bars = None
        if bars:
            break

    if not bars:
        return "failed"

    df = pd.DataFrame(
        [
            {
                "date": b.date,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
    )
    intraday_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    return "fetched"


def _reconnect_with_retries(attempts: int = 3, delay_secs: float = 10) -> "IBKRClient":
    """Reconnect with a few retries rather than letting a transient
    ConnectionRefusedError (observed in practice when TWS briefly drops
    the API port during a bad pacing stretch) kill an hours-long fetch."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return IBKRClient("127.0.0.1", 7497, FETCH_CLIENT_ID)
        except Exception as e:
            last_exc = e
            print(f"  reconnect attempt {attempt}/{attempts} failed: {e}")
            time.sleep(delay_secs)
    raise last_exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated override list")
    parser.add_argument(
        "--premarket",
        action="store_true",
        help="fetch premarket+RTH intraday bars into backtest_data/intraday_5m_premarket/ "
        "instead of the default RTH-only backtest_data/intraday_5m/ (skips the daily-bar step)",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="YYYY-MM-DD: fetch the 2y daily / 1y intraday windows ENDING here instead of 'now' "
        "(for backtesting a different historical year); writes to backtest_data/{daily,intraday_5m}_<end-date>/ "
        "to avoid clobbering the default cache",
    )
    parser.add_argument("--daily-dir", type=str, default=None, help="override the daily output directory")
    parser.add_argument("--intraday-dir", type=str, default=None, help="override the intraday output directory")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)
    total = len(tickers)

    if args.daily_dir:
        daily_dir = Path(args.daily_dir)
    elif args.end_date:
        daily_dir = Path(f"backtest_data/daily_{args.end_date}")
    else:
        daily_dir = DAILY_DIR

    if args.intraday_dir:
        intraday_dir = Path(args.intraday_dir)
    elif args.premarket:
        intraday_dir = INTRADAY_PREMARKET_DIR
    elif args.end_date:
        intraday_dir = Path(f"backtest_data/intraday_5m_{args.end_date}")
    else:
        intraday_dir = INTRADAY_DIR
    use_rth = not args.premarket

    if not args.premarket:
        print(f"=== Daily bars ({DAILY_PERIOD if not args.end_date else f'2y ending {args.end_date}'}) "
              f"for {total} tickers -> {daily_dir} ===")
        daily_counts = {"cached": 0, "fetched": 0, "failed": 0}
        for i, ticker in enumerate(tickers, 1):
            result = fetch_daily(ticker, daily_dir, args.end_date)
            daily_counts[result] += 1
            if result != "cached":
                print(f"[{i}/{total}] daily {ticker}: {result}")
        print(f"Daily done: {daily_counts}")

    label = "premarket+RTH" if args.premarket else "RTH-only"
    print(f"\n=== Intraday 5-min bars ({label}, {INTRADAY_DURATION}"
          f"{f' ending {args.end_date}' if args.end_date else ''}) for {total} tickers -> {intraday_dir} ===")
    already_cached = sum(1 for t in tickers if (intraday_dir / safe_filename(t)).exists())
    print(f"{already_cached}/{total} already cached; fetching the rest "
          f"(~{IBKR_PACING_SLEEP_SECS}s/ticker due to IBKR pacing limits)...")

    ib = IBKRClient("127.0.0.1", 7497, FETCH_CLIENT_ID).ib
    intraday_counts = {"cached": 0, "fetched": 0, "failed": 0}
    try:
        for i, ticker in enumerate(tickers, 1):
            path = intraday_dir / safe_filename(ticker)
            if path.exists():
                intraday_counts["cached"] += 1
                continue

            if not ib.isConnected():
                # IBKR's historical-data pacing limit (60 requests / 10 min
                # per connection) can get tripped by retries during a bad
                # stretch, and TWS drops the connection outright when it
                # does -- reconnect rather than let every remaining ticker
                # fail silently against a dead session.
                print("  connection lost -- reconnecting...")
                try:
                    ib.disconnect()
                except Exception:
                    pass
                ib = _reconnect_with_retries().ib

            result = fetch_intraday(ib, ticker, intraday_dir, use_rth, args.end_date)
            intraday_counts[result] += 1
            print(f"[{i}/{total}] intraday {ticker}: {result}")
            time.sleep(IBKR_PACING_SLEEP_SECS)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    print(f"\nIntraday done: {intraday_counts}")
    print("=== Fetch complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
