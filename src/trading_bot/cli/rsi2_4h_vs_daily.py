"""Compare RSI2 performance on 4-hour bars vs daily bars.

Tests the same strategy on both timeframes over the same date range
to measure if intraday trading improves ROI.

Usage:
    python -m trading_bot.cli.rsi2_4h_vs_daily --symbol ^GSPC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

from trading_bot.backtest.rsi2_signals import find_rsi2_long_trades
from trading_bot.cli.rsi2_fetch_data import DAILY_INDEX_DIR, safe_filename
from trading_bot.cli.rsi2_fetch_4h_data import load_4h_bars


def load_bars(symbol: str, data_dir: Path) -> dict:
    """Load daily bars from CSV."""
    path = data_dir / safe_filename(symbol)
    if not path.exists():
        raise FileNotFoundError(f"no cached bars at {path}")
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


def get_daily_bars_for_period(symbol: str, start_date: datetime, end_date: datetime) -> dict:
    """Load daily bars from cached data, filtered to date range."""
    daily_bars = load_bars(symbol, DAILY_INDEX_DIR)

    # Filter to the date range
    mask = [
        start_date <= ts <= end_date
        for ts in daily_bars["date"]
    ]

    return {
        "date": [d for d, m in zip(daily_bars["date"], mask) if m],
        "open": [o for o, m in zip(daily_bars["open"], mask) if m],
        "high": [h for h, m in zip(daily_bars["high"], mask) if m],
        "low": [l for l, m in zip(daily_bars["low"], mask) if m],
        "close": [c for c, m in zip(daily_bars["close"], mask) if m],
    }


def backtest_timeframe(bars: dict, timeframe: str) -> dict:
    """Run RSI2 backtest on bars."""
    trades = find_rsi2_long_trades(
        bars,
        rsi_period=2,
        entry_level=10.0,
        exit_level=70.0,
        sma_period=200,
        stop_points=200.0,
        exit_mode="first_profitable_close",
        min_hold_days=12,
        exit_timing="close",
    )

    if not trades:
        return {
            "timeframe": timeframe,
            "trades": 0,
            "net_points": 0,
            "win_pct": 0,
            "bars": len(bars["date"]),
            "period": f"{bars['date'][0]} to {bars['date'][-1]}",
        }

    wins = [t for t in trades if t["points"] > 0]
    net = sum(t["points"] for t in trades)

    return {
        "timeframe": timeframe,
        "trades": len(trades),
        "net_points": round(net, 1),
        "win_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_points": round(net / len(trades), 1) if trades else 0,
        "bars": len(bars["date"]),
        "period": f"{bars['date'][0].date()} to {bars['date'][-1].date()}",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=str, default="^GSPC")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"Comparing RSI2 strategy: 4-hour vs daily bars on {args.symbol}\n")

    # Load 4-hour bars
    try:
        bars_4h = load_4h_bars(args.symbol)
        start_date = bars_4h["date"][0]
        end_date = bars_4h["date"][-1]
        print(f"4-hour bars: {len(bars_4h['date'])} bars from {start_date.date()} to {end_date.date()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Load daily bars for same period
    bars_daily = get_daily_bars_for_period(args.symbol, start_date, end_date)
    print(f"Daily bars: {len(bars_daily['date'])} bars from {bars_daily['date'][0].date()} to {bars_daily['date'][-1].date()}\n")

    # Backtest both
    print("Backtesting...")
    result_4h = backtest_timeframe(bars_4h, "4-hour")
    result_daily = backtest_timeframe(bars_daily, "daily")

    # Compare
    print("\n" + "="*70)
    print(f"RESULTS for {args.symbol}")
    print("="*70)
    print(f"\nPeriod: {result_4h['period']}")
    print(f"\n{'Metric':<25} {'Daily':<20} {'4-Hour':<20} {'Difference'}")
    print("-" * 70)

    trades_diff = result_4h['trades'] - result_daily['trades']
    points_diff = result_4h['net_points'] - result_daily['net_points']
    win_diff = result_4h['win_pct'] - result_daily['win_pct']

    print(f"{'Trades':<25} {result_daily['trades']:<20} {result_4h['trades']:<20} {trades_diff:+.0f} ({trades_diff/result_daily['trades']*100 if result_daily['trades'] else 0:+.0f}%)")
    print(f"{'Net points':<25} {result_daily['net_points']:<20} {result_4h['net_points']:<20} {points_diff:+.1f}")
    print(f"{'Win %':<25} {result_daily['win_pct']:<20} {result_4h['win_pct']:<20} {win_diff:+.1f}pp")
    print(f"{'Avg per trade':<25} {result_daily['avg_points']:<20} {result_4h['avg_points']:<20} {result_4h['avg_points'] - result_daily['avg_points']:+.1f}")
    print(f"{'OHLC bars':<25} {result_daily['bars']:<20} {result_4h['bars']:<20}")

    # ROI estimation (rough)
    if result_daily['trades'] > 0 and result_4h['trades'] > 0:
        roi_daily = (result_daily['net_points'] / 60) / 500 * 100  # Rough estimate
        roi_4h = (result_4h['net_points'] / 60) / 500 * 100
        print(f"\nEstimated annual ROI (annualized from {result_4h['period']}):")
        print(f"  Daily: {roi_daily:.1f}%")
        print(f"  4-hour: {roi_4h:.1f}%")
        print(f"  Improvement: {roi_4h - roi_daily:+.1f}pp")

    print("\n" + "="*70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
