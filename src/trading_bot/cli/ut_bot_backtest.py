"""Backtest CLI for the UT Bot ATR trailing-stop FX strategy against cached
hourly bars in backtest_data/fx_1h/ (see ut_bot_fetch_data.py).

Defaults to the previously-validated "best combo" -- IBKR IDEALPRO
commission, a typical bid/ask spread, and the volatility regime filter,
all enabled (see ut_bot_engine.run_ut_bot_backtest's docstring for what
each models) -- since that combo, not the bare zero-cost/no-filter
defaults, is what's actually been validated as tradeable. Use
--no-commission/--no-spread/--no-vol-filter to compare against the
baseline.

Usage:
    python -m trading_bot.cli.ut_bot_backtest                    # every cached pair
    python -m trading_bot.cli.ut_bot_backtest --pairs USDJPY
    python -m trading_bot.cli.ut_bot_backtest --pairs USDJPY --no-vol-filter
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from trading_bot.backtest.portfolio import FX_COMMISSION_BPS, FX_TYPICAL_SPREAD_PIPS
from trading_bot.backtest.ut_bot_engine import run_ut_bot_backtest
from trading_bot.cli.compute_perf import aggregate, pair_trades_fifo
from trading_bot.cli.trade import CSV_HEADER

DATA_DIR = Path("backtest_data/fx_1h")
# See ut_bot_signals.DEFAULT_VOL_FILTER_LOOKBACK -- the validated regime-filter
# setting, not exposed as a CLI flag since it's a fixed part of the "best combo".
VOL_FILTER_LOOKBACK = 500


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=str, default=None, help="comma-separated override list; default = every cached pair")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=".", help="directory for per-pair trades/equity-curve CSVs")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--risk-pct", type=float, default=1.0)
    parser.add_argument("--no-commission", action="store_true", help="disable IBKR IDEALPRO commission modeling")
    parser.add_argument("--no-spread", action="store_true", help="disable bid/ask spread modeling")
    parser.add_argument("--no-vol-filter", action="store_true", help="disable the volatility regime filter")
    return parser.parse_args(argv)


def load_bars(pair: str, data_dir: Path) -> dict:
    df = pd.read_csv(data_dir / f"{pair}.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.sort_values("Datetime").reset_index(drop=True)
    return {
        "date": list(df["Datetime"]),
        "high": df["High"].tolist(),
        "low": df["Low"].tolist(),
        "close": df["Close"].tolist(),
    }


def max_drawdown_pct(equity_curve: list[dict]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]["equity"]
    worst = 0.0
    for point in equity_curve:
        peak = max(peak, point["equity"])
        if peak:
            worst = max(worst, (peak - point["equity"]) / peak * 100)
    return worst


def write_trades_csv(trades: list[dict], path: Path) -> None:
    """Write the compute_perf.py-compatible schema (trade.CSV_HEADER exactly).

    Trade dicts carry an extra analysis-only `reason` field not in
    CSV_HEADER -- extrasaction="ignore" drops it here rather than raising,
    same convention as cli/backtest.py's write_trades_csv."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)


def write_equity_curve_csv(equity_curve: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "equity"])
        writer.writeheader()
        writer.writerows(equity_curve)


def run_one(pair: str, args: argparse.Namespace, data_dir: Path) -> dict:
    bars = load_bars(pair, data_dir)
    result = run_ut_bot_backtest(
        bars,
        args.initial_capital,
        symbol=pair,
        risk_pct=args.risk_pct,
        fx_commission_bps=None if args.no_commission else FX_COMMISSION_BPS,
        spread_pips=None if args.no_spread else FX_TYPICAL_SPREAD_PIPS.get(pair),
        vol_filter_lookback=None if args.no_vol_filter else VOL_FILTER_LOOKBACK,
    )
    trades = result["trades"]
    equity_curve = result["equity_curve"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_trades_csv(trades, out_dir / f"ut_bot_trades_{pair}.csv")
    write_equity_curve_csv(equity_curve, out_dir / f"ut_bot_equity_curve_{pair}.csv")

    closed_pairs = pair_trades_fifo(pd.DataFrame(trades)) if trades else []
    summary = aggregate(closed_pairs)
    final_equity = equity_curve[-1]["equity"] if equity_curve else args.initial_capital
    summary["bars"] = len(bars["date"])
    summary["initial_capital"] = round(args.initial_capital, 2)
    summary["final_equity"] = round(final_equity, 2)
    summary["total_return_pct"] = (
        round((final_equity / args.initial_capital - 1) * 100, 2) if args.initial_capital else 0.0
    )
    summary["max_drawdown_pct"] = round(max_drawdown_pct(equity_curve), 2)
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    if args.pairs:
        pairs = [p.strip().upper() for p in args.pairs.split(",")]
    else:
        pairs = sorted(p.stem for p in data_dir.glob("*.csv"))

    for pair in pairs:
        if not (data_dir / f"{pair}.csv").exists():
            print(f"{pair}: no cached data in {data_dir}, skipping (run ut_bot_fetch_data first)")
            continue
        summary = run_one(pair, args, data_dir)
        print(f"{pair}: {json.dumps(summary)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
