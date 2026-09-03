"""RSI(2) across the S&P constituent universe, one name at a time.

Answers whether breadth rescues the strategy: a rare-but-good setup like
"only buy the second dip" has too few instances on a single index to
judge, so this runs it across ~500 names and pools the trades.

Three measurement rules this enforces, each of which has already
produced a wrong answer in this repo when skipped:

  Open trades count. Positions still open at the end of the data are
  marked to the final close and included. Excluding them is catastrophic
  for exits that only close winners -- it deletes the entire losing tail
  and turned a real 62%-of-symbols result into a fake 100%.

  Buy-and-hold is the benchmark, per day held. A long-only dip-buyer in a
  rising market makes money with no edge at all. Every figure is stated
  against holding the same names over the same window, divided by days
  actually exposed.

  Trades are not independent observations. Names dip together, so the
  pooled per-trade t-statistic is inflated by clustering. This reports
  the number of distinct entry DATES and how much of the total P&L a
  handful of them carry -- the effective sample is days, not trades.

Survivorship: the ticker list is a static snapshot of CURRENT index
members, so names that were removed over the window are absent and every
figure here is biased upward. Treat results as an upper bound.

Usage:
    python -m trading_bot.cli.rsi2_universe
    python -m trading_bot.cli.rsi2_universe --first-dips 1,2,3 --stop-pct 8
    python -m trading_bot.cli.rsi2_universe --out-csv rsi2_universe.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

from trading_bot.backtest.rsi2_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_SMA_PERIOD,
    find_rsi2_scale_in_trades,
)

DAILY_LONG_DIR = Path("backtest_data/daily_long")
DEFAULT_COST_BPS = 5.0
MIN_BARS = 400


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=str(DAILY_LONG_DIR))
    parser.add_argument("--first-dips", type=str, default="1,2,3",
                        help="comma-separated dip numbers to test, each with a single position")
    parser.add_argument("--entry-level", type=float, default=DEFAULT_ENTRY_LEVEL)
    parser.add_argument("--exit-level", type=float, default=DEFAULT_EXIT_LEVEL)
    parser.add_argument("--sma-period", type=int, default=DEFAULT_SMA_PERIOD)
    parser.add_argument("--stop-pct", type=float, default=None)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help="round-trip cost in basis points of notional")
    parser.add_argument("--start", type=str, default=None, help="ignore entries before this date")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None)
    return parser.parse_args(argv)


def load_bars(path: Path) -> dict | None:
    df = pd.read_csv(path)
    if len(df) < MIN_BARS:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return {"date": list(df["Date"]), "open": df["Open"].tolist(), "high": df["High"].tolist(),
            "low": df["Low"].tolist(), "close": df["Close"].tolist()}


def run_one(bars: dict, args: argparse.Namespace, first_dip: int) -> list[dict]:
    return find_rsi2_scale_in_trades(
        bars,
        entry_level=args.entry_level,
        exit_level=args.exit_level,
        sma_period=args.sma_period,
        max_positions=1,
        first_dip=first_dip,
        stop_pct=args.stop_pct,
    )


def summarize(rets: list[float], by_symbol: dict, by_date: dict, per_symbol_stats: list[dict]) -> dict:
    """`per_symbol_stats` carries one {strat_ret, bh_ret, held, days} per
    symbol, compounded within that symbol.

    The strategy-vs-benchmark comparison is done in LOG space per symbol
    and only then pooled. Comparing a sum of simple per-trade returns
    against a compounded buy-and-hold total return is not a like-for-like
    ratio -- over 16 years a name that went up 15x has a compounded
    benchmark no arithmetic sum of bounded trade returns can approach, so
    that construction understates the strategy by a wide margin. log(1+r)
    divided by days is additive and compounding-consistent on both sides,
    which makes "return per day of exposure" a fair quantity to divide.

    Pooling is by MEDIAN symbol, not by mean: a handful of 100-baggers in
    a survivorship-selected list dominates any mean.
    """
    n = len(rets)
    if not n:
        return {"trades": 0}
    mean = statistics.mean(rets)
    sd = statistics.stdev(rets) if n > 1 else 0.0
    total = sum(rets)
    day_tot = {d: sum(v) for d, v in by_date.items()}
    ranked = sorted(day_tot, key=lambda d: day_tot[d], reverse=True)
    top10 = sum(day_tot[d] for d in ranked[:10])
    sym_means = [statistics.mean(v) for v in by_symbol.values()]

    usable = [s for s in per_symbol_stats
              if s["held"] > 0 and s["days"] > 0 and s["strat_ret"] > -1 and s["bh_ret"] > -1]
    strat_per_day = [math.log1p(s["strat_ret"]) / s["held"] for s in usable]
    bh_per_day = [math.log1p(s["bh_ret"]) / s["days"] for s in usable]
    ratios = [a / b for a, b in zip(strat_per_day, bh_per_day) if b > 0]
    beat = sum(1 for a, b in zip(strat_per_day, bh_per_day) if a > b)

    return {
        "trades": n,
        "symbols": len(by_symbol),
        "entry_dates": len(by_date),
        "mean_bps": round(mean, 1),
        "win_pct": round(sum(1 for r in rets if r > 0) / n * 100, 1),
        # Reported, but see the module docstring: clustering inflates this.
        "naive_t": round(mean / (sd / math.sqrt(n)), 2) if sd else 0.0,
        "symbols_positive_pct": round(sum(1 for v in sym_means if v > 0) / len(sym_means) * 100, 1),
        # Median symbol, compounded, log-return per day.
        "median_strat_ret_pct": round(statistics.median(s["strat_ret"] for s in usable) * 100, 1),
        "median_bh_ret_pct": round(statistics.median(s["bh_ret"] for s in usable) * 100, 1),
        "median_exposure_pct": round(statistics.median(s["held"] / s["days"] for s in usable) * 100, 1),
        "median_ratio_per_day": round(statistics.median(ratios), 2) if ratios else None,
        "symbols_beating_bh_per_day_pct": round(beat / len(usable) * 100, 1) if usable else None,
        "entry_dates_positive_pct": round(sum(1 for v in day_tot.values() if v > 0) / len(day_tot) * 100, 1),
        "top10_dates_share_pct": round(top10 / total * 100, 1) if total > 0 else None,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise SystemExit(f"no cached bars in {data_dir} -- run: "
                         f"python -m trading_bot.cli.rsi2_fetch_data --sp500")
    dips = [int(x) for x in args.first_dips.split(",") if x.strip()]

    start_ts = pd.Timestamp(args.start) if args.start else None
    end_ts = pd.Timestamp(args.end) if args.end else None

    def keep(ts) -> bool:
        return not ((start_ts is not None and ts < start_ts) or (end_ts is not None and ts > end_ts))

    loaded = [(p.stem, load_bars(p)) for p in paths]
    loaded = [(s, b) for s, b in loaded if b is not None]
    span = f"{min(b['date'][0] for _, b in loaded).date()} .. {max(b['date'][-1] for _, b in loaded).date()}"
    print(f"{len(loaded)} symbols from {data_dir} ({span}), "
          f"cost {args.cost_bps}bps round trip, "
          f"stop {'none' if args.stop_pct is None else f'{args.stop_pct:g}%'}")
    print(f"RSI(2) < {args.entry_level:g} entry, > {args.exit_level:g} exit, "
          f"SMA{args.sma_period}; open trades marked to final close and INCLUDED\n")

    rows = []
    for dip in dips:
        rets, by_symbol, by_date = [], {}, defaultdict(list)
        per_symbol_stats = []
        for symbol, bars in loaded:
            per, held, compounded = [], 0, 1.0
            for t in run_one(bars, args, dip):
                if not keep(t["entry_date"]):
                    continue
                r = t["points"] / t["entry_price"] - args.cost_bps / 10000
                per.append(r * 10000)
                by_date[str(t["entry_date"])[:10]].append(r * 10000)
                held += t["bars_held"]
                compounded *= 1 + r
            if per:
                by_symbol[symbol] = per
                rets += per
            lo = args.sma_period
            while lo < len(bars["date"]) and not keep(bars["date"][lo]):
                lo += 1
            if per and lo < len(bars["date"]) - 1:
                per_symbol_stats.append({
                    "strat_ret": compounded - 1,
                    "bh_ret": bars["close"][-1] / bars["open"][lo] - 1,
                    "held": held,
                    "days": len(bars["date"]) - lo,
                })
        s = summarize(rets, by_symbol, by_date, per_symbol_stats)
        rows.append({"first_dip": dip, **s})
        print(f"dip{dip}_only  {json.dumps(s)}")

    if args.out_csv:
        path = Path(args.out_csv)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
