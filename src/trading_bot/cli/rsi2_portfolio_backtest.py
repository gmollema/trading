"""Portfolio backtest CLI: RSI(2) dip buying across the S&P universe.

Turns the per-symbol expectancy result into a portfolio return, or fails
to. See backtest/rsi2_portfolio.py for the allocation rules and why each
was chosen.

The benchmark is an equal-weight buy-and-hold of the SAME names over the
SAME window, with no costs -- deliberately generous, so beating it means
something. Both carry identical survivorship bias (the ticker list is a
snapshot of current members), which is the point of comparing them to
each other rather than to an index.

Usage:
    python -m trading_bot.cli.rsi2_portfolio_backtest
    python -m trading_bot.cli.rsi2_portfolio_backtest --first-dips 1,2,3 --slots 10,20,40
    python -m trading_bot.cli.rsi2_portfolio_backtest --cash-yield-pct 4 --priority symbol
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from trading_bot.backtest.rsi2_portfolio import (
    DEFAULT_MAX_SLOTS,
    DEFAULT_SLIPPAGE_BPS,
    cagr_pct,
    equal_weight_buy_hold,
    max_drawdown_pct,
    prepare,
    run_portfolio,
)
from trading_bot.backtest.rsi2_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_SMA_PERIOD,
)
from trading_bot.cli.rsi2_universe import DAILY_LONG_DIR, load_bars


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=str(DAILY_LONG_DIR))
    parser.add_argument("--first-dips", type=str, default="1,2,3")
    parser.add_argument("--slots", type=str, default=str(DEFAULT_MAX_SLOTS))
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--entry-level", type=float, default=DEFAULT_ENTRY_LEVEL)
    parser.add_argument("--exit-level", type=float, default=DEFAULT_EXIT_LEVEL)
    parser.add_argument("--sma-period", type=int, default=DEFAULT_SMA_PERIOD)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--cash-yield-pct", type=float, default=0.0)
    parser.add_argument("--priority", type=str, default="rsi", choices=("rsi", "symbol"))
    parser.add_argument("--entry-next-open", action="store_true")
    parser.add_argument("--full-history-only", action="store_true", default=True,
                        help="restrict to names present on the first calendar date (default)")
    parser.add_argument("--out-csv", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise SystemExit(f"no cached bars in {data_dir} -- run: "
                         f"python -m trading_bot.cli.rsi2_fetch_data --sp500")

    loaded = []
    for p in paths:
        bars = load_bars(p)
        if bars is not None:
            loaded.append((p.stem, bars))

    calendar = sorted({d for _, bars in loaded for d in bars["date"]})
    if args.full_history_only:
        first = calendar[0]
        loaded = [(s, b) for s, b in loaded if b["date"][0] == first]
        calendar = sorted({d for _, bars in loaded for d in bars["date"]})

    data = [prepare(s, b, entry_level=args.entry_level, exit_level=args.exit_level,
                    sma_period=args.sma_period) for s, b in loaded]
    years = (calendar[-1] - calendar[args.sma_period]).days / 365.25

    print(f"{len(data)} symbols, {len(calendar)} calendar dates, "
          f"{calendar[args.sma_period].date()} .. {calendar[-1].date()} ({years:.1f}y)")
    print(f"capital ${args.initial_capital:,.0f}, slippage {args.slippage_bps}bps/side, "
          f"IBKR tiered commission, cash yield {args.cash_yield_pct}%, "
          f"priority {args.priority}, entry on {'next open' if args.entry_next_open else 'close'}\n")

    bench = equal_weight_buy_hold(data, calendar, args.initial_capital, args.sma_period)
    bench_cagr = cagr_pct(args.initial_capital, bench[-1]["equity"], years) if bench else 0.0
    bench_dd = max_drawdown_pct(bench)
    print(f"benchmark  equal-weight buy & hold, no costs: "
          f"final ${bench[-1]['equity']:,.0f}  CAGR {bench_cagr:.2f}%  "
          f"maxDD {bench_dd:.1f}%  CAGR/DD {bench_cagr / bench_dd:.2f}\n")

    rows = []
    for slots in [int(x) for x in args.slots.split(",") if x.strip()]:
        for dip in [int(x) for x in args.first_dips.split(",") if x.strip()]:
            result = run_portfolio(
                data, calendar,
                initial_capital=args.initial_capital,
                max_slots=slots,
                first_dip=dip,
                slippage_bps=args.slippage_bps,
                cash_yield_pct=args.cash_yield_pct,
                priority=args.priority,
                entry_next_open=args.entry_next_open,
                warmup=args.sma_period,
            )
            curve = result["equity_curve"]
            final = result["final_equity"]
            trades = result["trades"]
            wins = [t for t in trades if t["pnl"] > 0]
            cagr = cagr_pct(args.initial_capital, final, years)
            dd = max_drawdown_pct(curve)
            summary = {
                "slots": slots,
                "first_dip": dip,
                "trades": len(trades),
                "final_equity": round(final),
                "cagr_pct": round(cagr, 2),
                "max_dd_pct": round(dd, 1),
                "cagr_over_dd": round(cagr / dd, 2) if dd else None,
                "win_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
                "avg_positions": round(statistics.mean(p["positions"] for p in curve), 1),
                "avg_deployed_pct": round(statistics.mean(p["deployed_pct"] for p in curve), 1),
                "pct_days_flat": round(sum(1 for p in curve if p["positions"] == 0) / len(curve) * 100, 1),
                "cap_bound_days": result["stats"]["cap_bound_days"],
                "signal_days": result["stats"]["signal_days"],
                "skipped_no_slot": result["stats"]["skipped_no_slot"],
                "beats_benchmark_cagr": cagr > bench_cagr,
                "beats_benchmark_ratio": (cagr / dd) > (bench_cagr / bench_dd) if dd and bench_dd else None,
            }
            rows.append(summary)
            print(f"slots={slots:>3d} dip{dip}  {json.dumps(summary)}")

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
