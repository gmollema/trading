"""Backtest CLI for the RSI(2) scale-into-weakness variant
(see backtest/rsi2_signals.find_rsi2_scale_in_trades).

Reports the video's own headline metric -- net profit divided by maximum
drawdown -- on both bases it distinguishes:

  closed   Drawdown measured on realized, closed-out trades only. The
           video calls this "close to close" and says it is "the one to
           take most notice of".
  open     Drawdown on daily mark-to-market equity, so a stack sitting
           underwater counts. The video calls its equivalent "intraday";
           this version marks at daily closes rather than intrabar, so it
           sits between the video's two columns and is the more
           conservative of the pair reported here.

Both matter for this strategy specifically, because it has no stop and
scales INTO losses: the closed-trade figure cannot see the worst of a
three-contract stack bought on the way down.

Windows: `video` is the January 2005 - August 2023 span the video tests,
`after_video` is everything since, which did not exist when it was made.

Usage:
    python -m trading_bot.cli.rsi2_scale_in
    python -m trading_bot.cli.rsi2_scale_in --entry-level 5 --max-sweep 6
    python -m trading_bot.cli.rsi2_scale_in --contract MES --stop-pct 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from trading_bot.backtest.rsi2_engine import CONTRACTS, ES
from trading_bot.backtest.rsi2_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_RSI_PERIOD,
    DEFAULT_SMA_PERIOD,
    find_rsi2_scale_in_trades,
)
from trading_bot.cli.rsi2_backtest import in_window, load_bars
from trading_bot.cli.rsi2_fetch_data import DAILY_INDEX_DIR

WINDOWS = {
    "video": ("2005-01-01", "2023-08-31"),
    "after_video": ("2023-09-01", None),
    "full": ("2005-01-01", None),
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=str, default="ES=F")
    parser.add_argument("--data-dir", type=str, default=str(DAILY_INDEX_DIR))
    parser.add_argument("--contract", type=str, default="ES", choices=sorted(CONTRACTS))
    parser.add_argument("--rsi-period", type=int, default=DEFAULT_RSI_PERIOD)
    parser.add_argument("--entry-level", type=float, default=DEFAULT_ENTRY_LEVEL)
    parser.add_argument("--exit-level", type=float, default=DEFAULT_EXIT_LEVEL)
    parser.add_argument("--sma-period", type=int, default=DEFAULT_SMA_PERIOD)
    parser.add_argument("--max-sweep", type=int, default=6,
                        help="sweep max_positions from 1 to this (the video's own table goes to 6)")
    parser.add_argument("--stop-pct", type=float, default=None,
                        help="per-position stop; the video uses none")
    parser.add_argument("--entry-timing", type=str, default="close", choices=("close", "next_open"))
    parser.add_argument("--exit-timing", type=str, default="close", choices=("close", "next_open"))
    parser.add_argument("--slippage-ticks", type=float, default=0.0,
                        help="ticks of adverse fill per leg per contract (0 = the video's assumption)")
    parser.add_argument("--out-csv", type=str, default=None)
    return parser.parse_args(argv)


def equity_curves(positions: list[dict], bars: dict, multiplier: float,
                  cost_per_contract: float) -> tuple[list[float], list[float]]:
    """(closed_equity, open_equity) in dollars, one point per bar.

    closed_equity steps only when a position is realized; open_equity also
    carries the running mark on everything still held. Both start at 0 --
    these are P&L curves, not account curves, since the video reports
    net profit against drawdown rather than a return on capital.
    """
    n = len(bars["close"])
    closes = bars["close"]
    realized_at = [0.0] * n
    mark = [0.0] * n

    for pos in positions:
        net = pos["points"] * multiplier - cost_per_contract
        realized_at[pos["exit_idx"]] += net
        for i in range(pos["entry_idx"], pos["exit_idx"]):
            mark[i] += (closes[i] - pos["entry_price"]) * multiplier

    closed, opened = [], []
    run = 0.0
    for i in range(n):
        run += realized_at[i]
        closed.append(run)
        opened.append(run + mark[i])
    return closed, opened


def max_dd(curve: list[float]) -> float:
    peak = 0.0
    worst = 0.0
    for v in curve:
        peak = max(peak, v)
        worst = max(worst, peak - v)
    return worst


def summarize(positions: list[dict], bars: dict, window, multiplier: float,
              cost_per_contract: float) -> dict:
    """Positions are filtered by ENTRY date; the curves are then built on
    the window's own bars so a drawdown is measured over the same span."""
    start, end = window
    idx = [i for i, ts in enumerate(bars["date"]) if in_window(ts, start, end)]
    if not idx or not positions:
        return {"contracts": 0, "campaigns": 0, "net_pnl": 0.0}
    lo, hi = idx[0], idx[-1]
    sub = {k: bars[k][lo : hi + 1] for k in ("date", "open", "high", "low", "close")}
    shifted = [{**p, "entry_idx": p["entry_idx"] - lo, "exit_idx": min(p["exit_idx"] - lo, hi - lo)}
               for p in positions]

    closed, opened = equity_curves(shifted, sub, multiplier, cost_per_contract)
    net = closed[-1]
    dd_closed = max_dd(closed)
    dd_open = max_dd(opened)
    wins = [p for p in positions if p["points"] * multiplier - cost_per_contract > 0]
    campaigns = len({p["campaign"] for p in positions})
    held = sum(p["bars_held"] for p in positions)

    return {
        "contracts": len(positions),
        "campaigns": campaigns,
        "net_pnl": round(net),
        "avg_trade": round(net / len(positions)),
        "pct_profitable": round(len(wins) / len(positions) * 100, 1),
        "dd_closed": round(dd_closed),
        "dd_open": round(dd_open),
        # The video's headline ratio, as a multiple rather than its percent.
        "net_over_dd_closed": round(net / dd_closed, 2) if dd_closed else None,
        "net_over_dd_open": round(net / dd_open, 2) if dd_open else None,
        "avg_contracts_per_campaign": round(len(positions) / campaigns, 2) if campaigns else 0.0,
        "worst_mae_pts": round(max(p["mae_points"] for p in positions), 1),
        "contract_days": held,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = CONTRACTS[args.contract]
    bars = load_bars(args.symbol, Path(args.data_dir))
    cost = 2 * spec.commission_per_side + 2 * args.slippage_ticks * spec.tick_size * spec.multiplier

    print(f"{args.symbol}: {len(bars['date'])} daily bars, "
          f"{bars['date'][0].date()} .. {bars['date'][-1].date()}")
    print(f"{spec.name} @ ${spec.multiplier:g}/point, ${cost:.2f} cost per contract round trip, "
          f"stop {'none' if args.stop_pct is None else f'{args.stop_pct:g}%'}, "
          f"entry/exit on {args.entry_timing}/{args.exit_timing}")
    print(f"RSI({args.rsi_period}) < {args.entry_level:g} entry, > {args.exit_level:g} exit, "
          f"SMA{args.sma_period} trend filter\n")

    rows = []
    for window_name, window in WINDOWS.items():
        print(f"--- {window_name} "
              f"({window[0]} .. {window[1] or bars['date'][-1].date()}) ---")
        for maxpos in range(1, args.max_sweep + 1):
            positions = find_rsi2_scale_in_trades(
                bars,
                rsi_period=args.rsi_period,
                entry_level=args.entry_level,
                exit_level=args.exit_level,
                sma_period=args.sma_period,
                max_positions=maxpos,
                stop_pct=args.stop_pct,
                entry_timing=args.entry_timing,
                exit_timing=args.exit_timing,
            )
            windowed = [p for p in positions if in_window(p["entry_date"], *window)]
            s = summarize(windowed, bars, window, spec.multiplier, cost)
            rows.append({"window": window_name, "max_positions": maxpos, **s})
            print(f"  max_pos={maxpos}  {json.dumps(s)}")
        print()

    if args.out_csv:
        path = Path(args.out_csv)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
