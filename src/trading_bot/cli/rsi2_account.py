"""Account-level backtest of the settled rsi2 baseline: strategy P&L plus
the yield on collateral.

Two modeling points that decide the answer:

  Collateral earns interest CONTINUOUSLY, not only while flat. A futures
  position consumes margin, not cash, and at IBKR that margin can sit in
  cash or T-bills and keep earning. So the yield accrues on the whole
  balance every day, including days a contract is open. This is not
  double counting: a futures price embeds the cost of carry, so long-
  futures P&L is already an excess-return-over-financing stream. Adding
  the short rate back on the full balance is exactly how managed-futures
  returns decompose (collateral yield + trading P&L), and leaving it out
  understates any futures strategy.

  A flat rate across 2005-2026 is fiction. Short rates were ~5% in
  2006-07, essentially zero from 2009 through 2015 and again in 2020-21,
  and back near 5% in 2023-24. Assuming a constant 4% hands the strategy
  roughly a decade of interest that did not exist. --flat-yield is
  offered because it is what gets asked for; HISTORICAL_SHORT_RATE is the
  default and is what the conclusion should rest on.

Sizing is a fixed contract count by default, which isolates the cash-yield
effect. --margin-scaled instead scales contracts with equity so the book
compounds, capped by --max-margin-pct.

Usage:
    python -m trading_bot.cli.rsi2_account
    python -m trading_bot.cli.rsi2_account --flat-yield 4 --capitals 50000,100000
    python -m trading_bot.cli.rsi2_account --margin-scaled --contract MES
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_bot.backtest.rsi2_engine import CONTRACTS, cagr_pct
from trading_bot.backtest.rsi2_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_SMA_PERIOD,
    find_rsi2_scale_in_trades,
)
from trading_bot.cli.rsi2_backtest import in_window, load_bars
from trading_bot.cli.rsi2_fetch_data import DAILY_INDEX_DIR

# Approximate ANNUAL AVERAGE 3-month T-bill yields, in percent. These are
# order-of-magnitude figures for a sensitivity test, not quoted data --
# individual years may be off by tens of basis points. What matters for
# the conclusion is the shape, which is not in doubt: near-zero through
# 2009-2015 and 2020-21, ~5% in 2006-07 and 2023-24.
HISTORICAL_SHORT_RATE = {
    2005: 3.2, 2006: 4.8, 2007: 4.4, 2008: 1.4, 2009: 0.15, 2010: 0.14,
    2011: 0.05, 2012: 0.09, 2013: 0.06, 2014: 0.03, 2015: 0.05, 2016: 0.32,
    2017: 0.93, 2018: 1.94, 2019: 2.06, 2020: 0.37, 2021: 0.04, 2022: 2.02,
    2023: 5.07, 2024: 5.0, 2025: 4.2, 2026: 3.8,
}
TRADING_DAYS = 252


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=str, default="ES=F")
    parser.add_argument("--data-dir", type=str, default=str(DAILY_INDEX_DIR))
    parser.add_argument("--contract", type=str, default="ES", choices=sorted(CONTRACTS))
    parser.add_argument("--capitals", type=str, default="50000,100000,250000")
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--margin-scaled", action="store_true",
                        help="scale contracts with equity instead of holding a fixed count")
    parser.add_argument("--max-margin-pct", type=float, default=25.0)
    parser.add_argument("--flat-yield", type=float, default=None,
                        help="constant annual yield in percent; overrides the historical path")
    parser.add_argument("--no-yield", action="store_true", help="zero collateral yield")
    parser.add_argument("--entry-level", type=float, default=DEFAULT_ENTRY_LEVEL)
    parser.add_argument("--exit-level", type=float, default=DEFAULT_EXIT_LEVEL)
    parser.add_argument("--sma-period", type=int, default=DEFAULT_SMA_PERIOD)
    parser.add_argument("--start", type=str, default="2005-01-01")
    parser.add_argument("--slippage-ticks", type=float, default=0.0)
    parser.add_argument("--out-csv", type=str, default=None)
    return parser.parse_args(argv)


def rate_for(year: int, args: argparse.Namespace) -> float:
    if args.no_yield:
        return 0.0
    if args.flat_yield is not None:
        return args.flat_yield
    return HISTORICAL_SHORT_RATE.get(year, 0.0)


def run_account(trades: list[dict], bars: dict, capital: float, spec, args: argparse.Namespace) -> dict:
    """Daily equity walk: collateral yield on the full balance every day,
    plus mark-to-market on any open position, plus realized P&L at exits.
    """
    cost = 2 * spec.commission_per_side + 2 * args.slippage_ticks * spec.tick_size * spec.multiplier
    entry_at = {t["entry_idx"]: t for t in trades}
    exit_at: dict[int, list[dict]] = {}
    for t in trades:
        exit_at.setdefault(t["exit_idx"], []).append(t)

    equity = capital
    open_trade = None
    open_contracts = 0
    curve = []
    interest_total = 0.0
    peak = capital
    worst_dd = 0.0
    contract_days = 0

    lo = next(i for i, ts in enumerate(bars["date"]) if in_window(ts, args.start, None))
    for i in range(lo, len(bars["date"])):
        year = bars["date"][i].year
        daily = (rate_for(year, args) / 100.0) / TRADING_DAYS
        interest = equity * daily
        equity += interest
        interest_total += interest

        for t in exit_at.get(i, []):
            if open_trade is t:
                equity += t["points"] * spec.multiplier * open_contracts - cost * open_contracts
                open_trade, open_contracts = None, 0

        if i in entry_at and open_trade is None:
            t = entry_at[i]
            if args.margin_scaled:
                n = int(equity * (args.max_margin_pct / 100.0) // spec.margin_per_contract)
            else:
                n = args.contracts
            if n >= 1:
                open_trade, open_contracts = t, n

        mark = 0.0
        if open_trade is not None:
            contract_days += open_contracts
            mark = (bars["close"][i] - open_trade["entry_price"]) * spec.multiplier * open_contracts
        marked = equity + mark
        peak = max(peak, marked)
        worst_dd = max(worst_dd, (peak - marked) / peak * 100 if peak > 0 else 0.0)
        curve.append({"date": bars["date"][i], "equity": marked})

    years = (bars["date"][-1] - bars["date"][lo]).days / 365.25
    return {
        "equity_curve": curve,
        "final_equity": round(equity),
        "total_return_pct": round((equity / capital - 1) * 100, 1),
        "cagr_pct": round(cagr_pct(capital, equity, years), 2),
        "max_dd_pct": round(worst_dd, 1),
        "interest_earned": round(interest_total),
        "interest_share_pct": round(interest_total / (equity - capital) * 100, 1) if equity > capital else None,
        "contract_days": contract_days,
        "years": round(years, 1),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = CONTRACTS[args.contract]
    bars = load_bars(args.symbol, Path(args.data_dir))
    trades = [t for t in find_rsi2_scale_in_trades(
        bars, entry_level=args.entry_level, exit_level=args.exit_level,
        sma_period=args.sma_period, max_positions=1)
        if in_window(t["entry_date"], args.start, None)]

    if args.no_yield:
        label = "no collateral yield"
    elif args.flat_yield is not None:
        label = f"FLAT {args.flat_yield:g}% collateral yield (see module docstring: not realistic)"
    else:
        label = "historical short-rate path (approximate annual averages)"
    sizing = (f"margin-scaled, {args.max_margin_pct:g}% cap" if args.margin_scaled
              else f"fixed {args.contracts} contract(s)")
    print(f"{args.symbol}  {spec.name} @ ${spec.multiplier:g}/pt  {len(trades)} trades from {args.start}")
    print(f"{label}; sizing: {sizing}\n")

    rows = []
    for capital in [float(x) for x in args.capitals.split(",") if x.strip()]:
        s = run_account(trades, bars, capital, spec, args)
        rows.append({"capital": capital, **{k: v for k, v in s.items() if k != "equity_curve"}})
        print(f"${capital:>10,.0f}  {json.dumps({k: v for k, v in s.items() if k != 'equity_curve'})}")

    if args.out_csv:
        import csv
        path = Path(args.out_csv)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
