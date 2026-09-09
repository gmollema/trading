"""Backtest the RSI(20) dip signal traded through a leveraged ETF.

The signal is the one in backtest/rsi20_dip_signals.py, unchanged and
computed on the INDEX. What this CLI adds is the exposure model: instead
of assuming the account earns the index's move, it walks a SYNTHETIC Nx
daily-rebalanced fund bar by bar.

WHY A BAR-BY-BAR WALK IS NECESSARY
----------------------------------
A leveraged ETF's return is path-dependent. The fund resets to Nx
leverage at every close, so it earns Nx the DAILY return, not Nx the
trade's return, and the difference is volatility decay: +10%/-10% on
consecutive days leaves the index at 0.99 and a 3x fund at 0.91. Scaling
find_rsi20_dip_trades' per-trade `pct` by N would miss that entirely and
overstate 3x badly. Hence the daily walk, with entry and exit days
handled as partial periods since fills happen at the open.

COSTS
-----
    expense ratio + (N - 1) x financing rate, accrued ONLY on days held.
The financing term is the borrowing cost on the leveraged portion and is
the single biggest unknown here: a real ETP borrows at roughly the short
rate plus a spread, and its swap costs sit above the plain rate. Sweep
--finance-rates rather than trusting one number. Days-held matters
because the strategy is in the market only ~50% of the time, which
halves the drag relative to holding the fund.

WHAT IT MEASURES, AND THE CAVEAT
--------------------------------
Full window the leveraged version wins; split by window it mostly does
not. Run --windows and read that table before funding anything. Neither
commission nor spread is modelled: at ~4.4 round trips a year on a small
account, IBKR's per-order minimum is a further ~0.5%/yr of drag.

Usage:
    python -m trading_bot.cli.rsi20_dip_etf_backtest
    python -m trading_bot.cli.rsi20_dip_etf_backtest --windows
    python -m trading_bot.cli.rsi20_dip_etf_backtest --symbol ES=F --multipliers 2,3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trading_bot.backtest.rsi20_dip_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_MA_PERIOD,
    DEFAULT_OFFSET_BARS,
    DEFAULT_RSI_PERIOD,
    find_rsi20_dip_trades,
)
from trading_bot.cli.rsi2_backtest import load_bars
from trading_bot.cli.rsi2_fetch_data import DAILY_INDEX_DIR

TRADING_DAYS = 252.0
DEFAULT_START_YEAR = 2008
# Expense ratios: 1x is a plain tracker, 2x/3x are typical leveraged ETP
# figures. The UCITS 3x S&P products sit near 0.75-0.95%.
DEFAULT_EXPENSE_RATIOS = {1.0: 0.0003, 2.0: 0.0089, 3.0: 0.0091}
DEFAULT_MULTIPLIERS = (1.0, 2.0, 3.0)
DEFAULT_FINANCE_RATES = (0.0, 0.02, 0.04)
# Sub-periods for --windows. Deliberately not tuned: two halves would
# hide that the full-sample win comes from one crash.
WINDOWS = ((2008, 2013), (2013, 2018), (2018, 2022), (2022, 2027), (2015, 2027))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=str, default="^GSPC")
    parser.add_argument("--data-dir", type=str, default=str(DAILY_INDEX_DIR))
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--multipliers", type=str, default="1,2,3",
                        help="comma-separated fund leverage factors")
    parser.add_argument("--finance-rates", type=str, default="0,2,4",
                        help="comma-separated annual financing rates, in percent")
    parser.add_argument("--expense-ratio", type=float, default=None,
                        help="override the per-multiplier default, as a fraction")
    parser.add_argument("--windows", action="store_true",
                        help="also report each sub-period separately")
    parser.add_argument("--rsi-length", type=int, default=DEFAULT_RSI_PERIOD)
    parser.add_argument("--ma-length", type=int, default=DEFAULT_MA_PERIOD)
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET_BARS)
    parser.add_argument("--entry-level", type=float, default=DEFAULT_ENTRY_LEVEL)
    parser.add_argument("--exit-level", type=float, default=DEFAULT_EXIT_LEVEL)
    return parser.parse_args(argv)


def _step(equity: float, fund_return: float, daily_cost: float) -> float:
    """One day of compounding, floored at zero.

    The floor is not cosmetic. A 3x fund needs only a 33.4% single-day
    index drop to be worth nothing, and the naive product goes NEGATIVE
    there -- after which the next down day multiplies a negative equity
    by a negative return and the curve recovers into profit. A real fund
    cannot go below zero: it rebalances intraday and, at that point, is
    wound up. Zero is absorbing, so once wiped out the curve stays there.

    2008-2026 never reaches it (3x bottoms out at a 44% drawdown), so
    this changes no reported figure -- it stops a more violent window
    from reporting a fictional one.
    """
    return max(equity * (1.0 + fund_return - daily_cost), 0.0)


def simulate_leveraged(bars: dict, trades: list[dict], mult: float,
                       expense_ratio: float, finance_rate: float,
                       start_idx: int = 0, end_idx: int | None = None) -> dict:
    """Equity curve of an Nx daily-rebalanced fund traded on `trades`.

    100% of equity per trade, marked to the close every bar so an open
    losing position shows up in the drawdown -- the open-equity basis
    used across the rsi2 and rsi20_dip work.

    Fills follow the signal module: an entry earns Nx the entry bar's
    OPEN-to-close move, full days earn Nx close-to-close, and an exit
    earns Nx the previous close to the exit bar's OPEN. Cost accrues on
    every day any exposure is held, including both partial days.

    Returns {"curve", "days_held", "bars", "final"}.
    """
    opens, closes = bars["open"], bars["close"]
    last = len(closes) - 1 if end_idx is None else end_idx
    entry_at = {t["entry_idx"]: t for t in trades}
    exit_at = {t["exit_idx"] for t in trades}
    daily_cost = (expense_ratio + (mult - 1.0) * finance_rate) / TRADING_DAYS

    eq, held, days_held = 1.0, False, 0
    curve = []
    for i in range(start_idx, last + 1):
        if held and i in exit_at:
            eq = _step(eq, mult * (opens[i] / closes[i - 1] - 1.0), daily_cost)
            held = False
        elif held:
            eq = _step(eq, mult * (closes[i] / closes[i - 1] - 1.0), daily_cost)
            days_held += 1
        if not held and i in entry_at:
            if entry_at[i]["exit_idx"] == i:  # entered and exited on one bar
                eq = _step(eq, 0.0, daily_cost)
            else:
                eq = _step(eq, mult * (closes[i] / opens[i] - 1.0), daily_cost)
                held = True
                days_held += 1
        curve.append(eq)
    return {"curve": curve, "days_held": days_held, "bars": len(curve),
            "final": curve[-1] if curve else 1.0}


def curve_stats(curve: list[float], years: float) -> dict:
    """CAGR, max drawdown and terminal multiple of an equity curve."""
    if not curve or years <= 0:
        return {"cagr": 0.0, "max_dd": 0.0, "final": 1.0}
    peak, worst = curve[0], 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak * 100.0)
    return {"cagr": (curve[-1] ** (1.0 / years) - 1.0) * 100.0,
            "max_dd": worst, "final": curve[-1]}


def buy_and_hold(bars: dict, start_idx: int, end_idx: int) -> list[float]:
    base = bars["close"][start_idx]
    return [bars["close"][i] / base for i in range(start_idx, end_idx + 1)]


def _expense_for(mult: float, override: float | None) -> float:
    if override is not None:
        return override
    return DEFAULT_EXPENSE_RATIOS.get(mult, 0.0091)


def _row(label: str, stats: dict, in_market: float) -> str:
    return (f"{label:26} {stats['cagr']:7.2f}% {stats['max_dd']:7.1f}% "
            f"{stats['final']:8.2f}x {in_market:7.1f}%")


def report_period(bars: dict, trades: list[dict], start_idx: int, end_idx: int,
                  args: argparse.Namespace, mults: list[float],
                  rates: list[float]) -> None:
    dates = bars["date"]
    years = (dates[end_idx] - dates[start_idx]).days / 365.25
    n_bars = end_idx - start_idx + 1
    in_window = [t for t in trades
                 if t["entry_idx"] >= start_idx and t["exit_idx"] <= end_idx]

    print(f"\n{dates[start_idx].date()} -> {dates[end_idx].date()}  "
          f"({years:.1f}y, {n_bars} bars, {len(in_window)} trades)")
    print(f"{'':26} {'CAGR':>8} {'maxDD':>8} {'x money':>9} {'in mkt':>8}")
    bh = curve_stats(buy_and_hold(bars, start_idx, end_idx), years)
    print(_row("buy & hold index", bh, 100.0))

    for rate in rates:
        print()
        for mult in mults:
            sim = simulate_leveraged(bars, in_window, mult,
                                     _expense_for(mult, args.expense_ratio),
                                     rate, start_idx, end_idx)
            stats = curve_stats(sim["curve"], years)
            label = f"{mult:.0f}x  finance {rate * 100:.0f}%"
            print(_row(label, stats, sim["days_held"] / n_bars * 100.0))

    # Holding the leveraged fund throughout, at the highest financing rate
    # tested: the comparison that shows the timing is doing the work.
    if rates:
        print()
        for mult in [m for m in mults if m > 1.0]:
            always = [{"entry_idx": start_idx, "exit_idx": end_idx}]
            sim = simulate_leveraged(bars, always, mult,
                                     _expense_for(mult, args.expense_ratio),
                                     max(rates), start_idx, end_idx)
            stats = curve_stats(sim["curve"], years)
            print(_row(f"HOLD {mult:.0f}x fund, fin {max(rates) * 100:.0f}%",
                       stats, sim["days_held"] / n_bars * 100.0))


def main(argv=None) -> int:
    args = parse_args(argv)
    mults = [float(m) for m in args.multipliers.split(",") if m.strip()]
    rates = [float(r) / 100.0 for r in args.finance_rates.split(",") if r.strip()]

    bars = load_bars(args.symbol, Path(args.data_dir))
    dates = bars["date"]
    trades = find_rsi20_dip_trades(
        bars, rsi_period=args.rsi_length, ma_period=args.ma_length,
        offset_bars=args.offset, entry_level=args.entry_level,
        exit_level=args.exit_level)

    def idx_from(year: int) -> int | None:
        return next((i for i, d in enumerate(dates) if d.year >= year), None)

    start_idx = idx_from(args.start_year)
    if start_idx is None:
        raise SystemExit(f"no bars at or after {args.start_year} in {args.symbol}")

    print(f"{args.symbol}  RSI({args.rsi_length}) dip through an Nx daily-rebalanced fund")
    report_period(bars, trades, start_idx, len(dates) - 1, args, mults, rates)

    if args.windows:
        for y0, y1 in WINDOWS:
            i0 = idx_from(y0)
            if i0 is None:
                continue
            i1 = max((i for i, d in enumerate(dates) if d.year < y1), default=None)
            if i1 is None or i1 <= i0:
                continue
            report_period(bars, trades, i0, i1, args, mults, rates)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
