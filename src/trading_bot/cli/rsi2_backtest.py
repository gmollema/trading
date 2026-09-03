"""Backtest CLI for the 2-period RSI strategy and the video's
first-profitable-close exit modification (see backtest/rsi2_signals.py).

Reports in INDEX POINTS, matching how the video reports, so its numbers
can be compared line for line against ours. No position sizing, no
commission, no spread: the video models none of those either, and adding
them here would make the comparison unfalsifiable rather than more
realistic. What that omission is worth is quantified by --cost-points.

Three windows are reported by default, and the split matters:
  in_sample    2008-01-01..2019-12-31  the window the video optimizes the
               day delay over. Its headline numbers describe THIS.
  video_oos    2020-01-01..2021-06-30  the video's own held-out window.
               18 months and 5-ish trades, which is not enough to
               confirm or refute anything on its own.
  post_video   2021-07-01..present     data that did not exist when the
               video was made. This is the only window nobody could have
               fitted to, and is the one worth believing.

A buy-and-hold point total is printed for each window alongside, because
"net profit in points" for a long-only strategy in a bull market is not
by itself evidence of an edge -- it has to beat sitting in the index for
the same period, and the strategy is only exposed a fraction of the time.

Usage:
    python -m trading_bot.cli.rsi2_backtest
    python -m trading_bot.cli.rsi2_backtest --sweep-delay
    python -m trading_bot.cli.rsi2_backtest --stop-pct 3 --cost-points 1
    python -m trading_bot.cli.rsi2_backtest --out-csv rsi2_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from dataclasses import replace

from trading_bot.backtest.rsi2_engine import (
    CONTRACTS,
    DEFAULT_MAX_MARGIN_PCT,
    DEFAULT_RISK_PCT,
    DEFAULT_SLIPPAGE_TICKS,
    ContractSpec,
    cagr_pct,
    max_drawdown_pct,
    run_rsi2_futures_backtest,
)
from trading_bot.backtest.rsi2_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_MIN_HOLD_DAYS,
    DEFAULT_RSI_PERIOD,
    DEFAULT_SMA_PERIOD,
    DEFAULT_STOP_POINTS,
    EXIT_MODE_FIRST_PROFITABLE_CLOSE,
    EXIT_MODE_RSI,
    find_rsi2_long_trades,
)
from trading_bot.cli.rsi2_fetch_data import DAILY_INDEX_DIR, safe_filename

WINDOWS = {
    "in_sample": ("2008-01-01", "2019-12-31"),
    "video_oos": ("2020-01-01", "2021-06-30"),
    "post_video": ("2021-07-01", None),
}
DELAY_SWEEP = range(1, 21)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=str, default="^GSPC")
    parser.add_argument("--data-dir", type=str, default=str(DAILY_INDEX_DIR))
    parser.add_argument("--rsi-period", type=int, default=DEFAULT_RSI_PERIOD)
    parser.add_argument("--entry-level", type=float, default=DEFAULT_ENTRY_LEVEL)
    parser.add_argument("--exit-level", type=float, default=DEFAULT_EXIT_LEVEL)
    parser.add_argument("--sma-period", type=int, default=DEFAULT_SMA_PERIOD)
    parser.add_argument("--min-hold-days", type=int, default=DEFAULT_MIN_HOLD_DAYS)
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument("--stop-points", type=float, default=DEFAULT_STOP_POINTS)
    stop.add_argument("--stop-pct", type=float, default=None, help="percentage stop instead of a flat point stop")
    stop.add_argument("--no-stop", action="store_true", help="Connors' documented original: no stop at all")
    parser.add_argument("--cost-points", type=float, default=0.0,
                        help="points deducted per round trip for spread/commission (0 = the video's assumption)")
    parser.add_argument("--sweep-delay", action="store_true", help="reproduce the video's 1..20 day-delay optimization")
    parser.add_argument("--dollars", action="store_true",
                        help="also size in whole futures contracts and report dollar P&L (see backtest/rsi2_engine.py)")
    parser.add_argument("--contract", type=str, default="MES", choices=sorted(CONTRACTS))
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT)
    parser.add_argument("--max-margin-pct", type=float, default=DEFAULT_MAX_MARGIN_PCT)
    parser.add_argument("--slippage-ticks", type=float, default=DEFAULT_SLIPPAGE_TICKS)
    parser.add_argument("--margin-per-contract", type=float, default=None,
                        help="override the contract's assumed margin (it moves with volatility; see rsi2_engine)")
    parser.add_argument("--out-csv", type=str, default=None)
    return parser.parse_args(argv)


def resolve_spec(args: argparse.Namespace) -> ContractSpec:
    spec = CONTRACTS[args.contract]
    if args.margin_per_contract is not None:
        spec = replace(spec, margin_per_contract=args.margin_per_contract)
    return spec


def dollar_summary(point_trades: list[dict], bars: dict, window: tuple[str | None, str | None],
                   args: argparse.Namespace, spec: ContractSpec) -> dict:
    """Contract-sized dollar result for one variant over one window.

    The index return is the benchmark rather than a dollar buy-and-hold
    figure: comparing dollars would require inventing a leverage choice
    for the benchmark, and any such choice would decide the comparison
    by itself.
    """
    result = run_rsi2_futures_backtest(
        point_trades, bars, args.initial_capital, spec,
        risk_pct=args.risk_pct, max_margin_pct=args.max_margin_pct,
        slippage_ticks=args.slippage_ticks,
    )
    taken = result["trades"]
    final = result["final_equity"]

    start, end = window
    idx = [i for i, ts in enumerate(bars["date"]) if in_window(ts, start, end)]
    if idx:
        span_years = (bars["date"][idx[-1]] - bars["date"][idx[0]]).days / 365.25
        index_return = (bars["close"][idx[-1]] / bars["open"][idx[0]] - 1) * 100
    else:
        span_years, index_return = 0.0, 0.0

    costs = sum(t["commission"] + t["slippage"] for t in taken)
    return {
        "trades_taken": len(taken),
        "trades_skipped": result["skipped_trades"],
        "net_pnl": round(final - args.initial_capital, 0),
        "return_pct": round((final / args.initial_capital - 1) * 100, 1),
        "cagr_pct": round(cagr_pct(args.initial_capital, final, span_years), 1),
        "max_dd_pct": round(max_drawdown_pct(result["equity_curve"]), 1),
        "avg_contracts": round(sum(t["contracts"] for t in taken) / len(taken), 1) if taken else 0.0,
        "max_contracts": max((t["contracts"] for t in taken), default=0),
        "costs_paid": round(costs, 0),
        "costs_pct_of_gross": round(costs / sum(t["gross_pnl"] for t in taken) * 100, 1)
        if taken and sum(t["gross_pnl"] for t in taken) > 0 else 0.0,
        "index_return_pct": round(index_return, 1),
        "locked_out_from": str(result["locked_out_from"].date()) if result["locked_out_from"] is not None else None,
    }


def load_bars(symbol: str, data_dir: Path) -> dict:
    path = data_dir / safe_filename(symbol)
    if not path.exists():
        raise SystemExit(f"no cached bars at {path} -- run: python -m trading_bot.cli.rsi2_fetch_data --symbols {symbol}")
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


def in_window(ts, start: str | None, end: str | None) -> bool:
    if start is not None and ts < pd.Timestamp(start, tz="UTC"):
        return False
    if end is not None and ts > pd.Timestamp(end, tz="UTC"):
        return False
    return True


def max_drawdown_points(trades: list[dict]) -> float:
    """Peak-to-trough of the CLOSED-TRADE cumulative point curve.

    Closed-trade, not intraday: it is the same basis the video reads off
    its equity curve, and it understates the real drawdown, since a
    position sitting 190 points underwater on its way to a profitable
    close never shows up here at all.
    """
    peak = 0.0
    equity = 0.0
    worst = 0.0
    for trade in trades:
        equity += trade["net_points"]
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def open_equity_drawdown_points(trades: list[dict], bars: dict) -> float:
    """Peak-to-trough of the MARK-TO-MARKET point curve: closed P&L plus
    the open position's unrealized move, valued at every bar's close.

    This is the drawdown a real account would have lived through, and it
    is the number max_drawdown_points hides. A first-profitable-close
    exit deliberately refuses to realize losers, so its closed-trade
    curve is almost monotone by construction while the open position can
    be arbitrarily deep underwater -- with the stop disabled, the
    closed-trade drawdown is exactly zero and this figure is the only one
    that says anything at all.

    The exit bar itself is not marked: its P&L arrives realized, at the
    actual fill, one line below.
    """
    equity = 0.0
    peak = 0.0
    worst = 0.0
    closes = bars["close"]
    for trade in trades:
        for i in range(trade["entry_idx"], trade["exit_idx"]):
            mtm = equity + (closes[i] - trade["entry_price"])
            peak = max(peak, mtm)
            worst = max(worst, peak - mtm)
        equity += trade["net_points"]
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def summarize(trades: list[dict], bars: dict, window: tuple[str | None, str | None]) -> dict:
    """Point-based stats plus the exposure and buy-and-hold context the
    video omits."""
    start, end = window
    bar_dates = bars["date"]
    idx_in_window = [i for i, ts in enumerate(bar_dates) if in_window(ts, start, end)]
    bars_in_window = len(idx_in_window)

    wins = [t for t in trades if t["net_points"] > 0]
    losses = [t for t in trades if t["net_points"] <= 0]
    net = sum(t["net_points"] for t in trades)
    days_exposed = sum(t["bars_held"] for t in trades)

    if idx_in_window:
        first, last = idx_in_window[0], idx_in_window[-1]
        buy_hold = bars["close"][last] - bars["open"][first]
    else:
        buy_hold = 0.0

    return {
        "trades": len(trades),
        "net_points": round(net, 1),
        "win_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "avg_points": round(net / len(trades), 1) if trades else 0.0,
        "gross_win_points": round(sum(t["net_points"] for t in wins), 1),
        "gross_loss_points": round(sum(t["net_points"] for t in losses), 1),
        "largest_loss_points": round(min((t["net_points"] for t in trades), default=0.0), 1),
        "max_drawdown_points": round(max_drawdown_points(trades), 1),
        "open_equity_dd_points": round(open_equity_drawdown_points(trades, bars), 1),
        "worst_mae_points": round(max((t["mae_points"] for t in trades), default=0.0), 1),
        "median_mae_points": round(_median([t["mae_points"] for t in trades]), 1),
        "avg_bars_held": round(days_exposed / len(trades), 1) if trades else 0.0,
        "exposure_pct": round(days_exposed / bars_in_window * 100, 1) if bars_in_window else 0.0,
        "buy_hold_points": round(buy_hold, 1),
        # Points earned per day actually holding a position, against the
        # same figure for simply owning the index. The strategy is only
        # exposed a third of the time, so absolute net points understate
        # it and this is the fairer comparison.
        "points_per_exposed_day": round(net / days_exposed, 2) if days_exposed else 0.0,
        "buy_hold_points_per_day": round(buy_hold / bars_in_window, 2) if bars_in_window else 0.0,
        # Concentration: net points with the five biggest winners removed.
        # Five days carried the whole SMC result; this is the same check.
        "net_points_ex_top5": round(net - sum(sorted((t["net_points"] for t in trades), reverse=True)[:5]), 1),
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def run_variant(bars: dict, args: argparse.Namespace, exit_mode: str, min_hold_days: int,
                exit_timing: str) -> list[dict]:
    """One full walk over the whole cached series. Windowing is applied
    afterwards, by entry date, rather than by slicing the bars first --
    slicing would fabricate an artificial end-of-data exit at every
    window boundary and would strip the 200 bars of SMA warmup."""
    trades = find_rsi2_long_trades(
        bars,
        rsi_period=args.rsi_period,
        entry_level=args.entry_level,
        exit_level=args.exit_level,
        sma_period=args.sma_period,
        stop_points=None if (args.no_stop or args.stop_pct is not None) else args.stop_points,
        stop_pct=args.stop_pct,
        exit_mode=exit_mode,
        min_hold_days=min_hold_days,
        exit_timing=exit_timing,
    )
    for trade in trades:
        trade["net_points"] = trade["points"] - args.cost_points
    return trades


def variants(args: argparse.Namespace) -> list[tuple[str, str, int, str]]:
    """(label, exit_mode, min_hold_days, exit_timing) for the headline table."""
    return [
        ("baseline_rsi70", EXIT_MODE_RSI, 0, "next_open"),
        ("fpc_no_delay", EXIT_MODE_FIRST_PROFITABLE_CLOSE, 1, "close"),
        (f"fpc_delay{args.min_hold_days}", EXIT_MODE_FIRST_PROFITABLE_CLOSE, args.min_hold_days, "close"),
        (f"fpc_delay{args.min_hold_days}_next_open", EXIT_MODE_FIRST_PROFITABLE_CLOSE, args.min_hold_days, "next_open"),
    ]


def _write_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    bars = load_bars(args.symbol, Path(args.data_dir))
    print(f"{args.symbol}: {len(bars['date'])} daily bars, "
          f"{bars['date'][0].date()} .. {bars['date'][-1].date()}")
    stop_desc = "none" if args.no_stop else (f"{args.stop_pct}%" if args.stop_pct is not None else f"{args.stop_points} pts")
    print(f"stop: {stop_desc}   cost per round trip: {args.cost_points} pts\n")

    rows: list[dict] = []
    by_variant: dict[str, list[dict]] = {}
    for label, exit_mode, min_hold, timing in variants(args):
        all_trades = run_variant(bars, args, exit_mode, min_hold, timing)
        by_variant[label] = all_trades
        for window_name, window in WINDOWS.items():
            windowed = [t for t in all_trades if in_window(t["entry_date"], *window)]
            summary = {"variant": label, "window": window_name, **summarize(windowed, bars, window)}
            rows.append(summary)
            print(f"{label:32s} {window_name:11s} {json.dumps({k: v for k, v in summary.items() if k not in ('variant', 'window')})}")
        print()

    dollar_rows: list[dict] = []
    if args.dollars:
        spec = resolve_spec(args)
        risk_per_contract = (args.stop_points * spec.multiplier) if not (args.no_stop or args.stop_pct is not None) else None
        print(f"{spec.name}: ${spec.multiplier:g}/point, ${spec.commission_per_side}/side, "
              f"${spec.margin_per_contract:g} margin (assumed)   capital ${args.initial_capital:,.0f}, "
              f"risk {args.risk_pct}%, slippage {args.slippage_ticks} tick/side")
        if risk_per_contract:
            floor = risk_per_contract / (args.risk_pct / 100.0)
            print(f"  a {args.stop_points:g}-point stop risks ${risk_per_contract:,.0f} per contract, so "
                  f"{args.risk_pct}% risk needs ~${floor:,.0f} of equity for the FIRST contract")
        print()
        for label, all_trades in by_variant.items():
            for window_name, window in WINDOWS.items():
                windowed = [t for t in all_trades if in_window(t["entry_date"], *window)]
                summary = {"variant": label, "window": window_name,
                           **dollar_summary(windowed, bars, window, args, spec)}
                dollar_rows.append(summary)
                print(f"{label:32s} {window_name:11s} "
                      f"{json.dumps({k: v for k, v in summary.items() if k not in ('variant', 'window')})}")
            print()

    if args.sweep_delay:
        print("day-delay sweep (the video's optimization, run on each window separately)")
        for delay in DELAY_SWEEP:
            all_trades = run_variant(bars, args, EXIT_MODE_FIRST_PROFITABLE_CLOSE, delay, "close")
            cells = []
            for window_name, window in WINDOWS.items():
                windowed = [t for t in all_trades if in_window(t["entry_date"], *window)]
                summary = summarize(windowed, bars, window)
                rows.append({"variant": f"sweep_delay{delay}", "window": window_name, **summary})
                cells.append(f"{window_name}={summary['net_points']:>8.1f} ({summary['trades']:>3d}t, {summary['win_pct']:>5.1f}%)")
            print(f"  delay={delay:2d}  " + "  ".join(cells))

    if args.out_csv:
        path = Path(args.out_csv)
        _write_rows(rows, path)
        print(f"\nwrote {len(rows)} rows to {path}")
        if dollar_rows:
            # Separate file, not merged: the dollar and point summaries
            # have disjoint columns and jamming them into one table would
            # leave half of every row empty.
            dollar_path = path.with_name(f"{path.stem}_dollars{path.suffix}")
            _write_rows(dollar_rows, dollar_path)
            print(f"wrote {len(dollar_rows)} rows to {dollar_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
