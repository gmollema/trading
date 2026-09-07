"""Backtest CLI for the RSI(20) dip strategy (see backtest/rsi20_dip_signals.py).

Reports in PERCENT and compounds 100% of equity per trade, which is what
the TradingView reference does, so the numbers here are comparable to its
Strategy Tester line for line. No commission or slippage is modelled for
the same reason -- the reference models none either, and the point of
this CLI is the comparison. At ~4.5 trades a year friction is not what
decides this strategy.

Buy-and-hold is printed alongside every result, on the same window and
the same open-to-close basis. It is not decoration: the strategy loses to
it on every instrument tested, and any run that omits the benchmark will
read as a success when it is not.

Usage:
    python -m trading_bot.cli.rsi20_dip_backtest
    python -m trading_bot.cli.rsi20_dip_backtest --symbols "^GSPC,NQ=F,^GDAXI"
    python -m trading_bot.cli.rsi20_dip_backtest --entry-level 55 --exit-level 80
    python -m trading_bot.cli.rsi20_dip_backtest --grid
    python -m trading_bot.cli.rsi20_dip_backtest --out-csv rsi20_dip_results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from trading_bot.backtest.rsi20_dip_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_MA_PERIOD,
    DEFAULT_OFFSET_BARS,
    DEFAULT_RSI_PERIOD,
    REASON_END_OF_DATA,
    equity_curve,
    find_rsi20_dip_trades,
    max_drawdown_pct,
)
from trading_bot.cli.rsi2_backtest import load_bars
from trading_bot.cli.rsi2_fetch_data import DAILY_INDEX_DIR

DEFAULT_SYMBOLS = "^GSPC,ES=F,NQ=F"
DEFAULT_START_YEAR = 2008
# The reference's own window, so its Strategy Tester numbers line up.
GRID_ENTRIES = (40, 45, 50, 55, 60, 65, 70)
GRID_EXITS = (55, 60, 65, 70, 75, 80)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=str, default=DEFAULT_SYMBOLS)
    parser.add_argument("--data-dir", type=str, default=str(DAILY_INDEX_DIR))
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--rsi-length", type=int, default=DEFAULT_RSI_PERIOD)
    parser.add_argument("--ma-length", type=int, default=DEFAULT_MA_PERIOD)
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET_BARS)
    parser.add_argument("--entry-level", type=float, default=DEFAULT_ENTRY_LEVEL)
    parser.add_argument("--exit-level", type=float, default=DEFAULT_EXIT_LEVEL)
    parser.add_argument("--grid", action="store_true",
                        help="sweep entry x exit levels instead of a single run")
    parser.add_argument("--out-csv", type=str, default=None)
    return parser.parse_args(argv)


def _start_index(bars: dict, start_year: int) -> int | None:
    return next((i for i, d in enumerate(bars["date"]) if d.year >= start_year), None)


def evaluate(bars: dict, args, entry_level: float, exit_level: float) -> dict | None:
    """One run's statistics, or None if the symbol has too little history."""
    lo = _start_index(bars, args.start_year)
    if lo is None or len(bars["close"]) - lo < args.ma_length + 2:
        return None

    trades = [t for t in find_rsi20_dip_trades(
        bars, rsi_period=args.rsi_length, ma_period=args.ma_length,
        offset_bars=args.offset, entry_level=entry_level, exit_level=exit_level)
        if t["entry_idx"] >= lo]
    if not trades:
        return None

    curve = equity_curve(bars, trades, start_idx=lo)
    final = curve[-1]["equity"]
    years = (bars["date"][-1] - bars["date"][lo]).days / 365.25
    rets = [t["pct"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    exposed = sum(1 for p in curve if p["in_market"])

    entry_px = bars["open"][lo]
    bh_final = bars["close"][-1] / entry_px
    bh_peak, bh_dd = 0.0, 0.0
    for i in range(lo, len(bars["close"])):
        bh_peak = max(bh_peak, bars["close"][i])
        bh_dd = max(bh_dd, (bh_peak - bars["close"][i]) / bh_peak * 100.0)

    dd = max_drawdown_pct(curve)
    cagr = (final ** (1 / years) - 1) * 100 if years > 0 else 0.0
    return {
        "entry_level": entry_level,
        "exit_level": exit_level,
        "years": round(years, 1),
        "trades": len(trades),
        "open_at_end": sum(1 for t in trades if t["reason"] == REASON_END_OF_DATA),
        "total_return_pct": round((final - 1) * 100, 1),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct": round(dd, 1),
        "return_per_dd": round(cagr / dd, 2) if dd > 0 else None,
        "win_pct": round(len(wins) / len(trades) * 100, 1),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "avg_win_pct": round(sum(wins) / len(wins) * 100, 2) if wins else 0.0,
        "avg_loss_pct": round(sum(losses) / len(losses) * 100, 2) if losses else 0.0,
        "expectancy_pct": round(sum(rets) / len(rets) * 100, 2),
        "avg_bars_held": round(sum(t["bars_held"] for t in trades) / len(trades), 1),
        "exposure_pct": round(exposed / len(curve) * 100, 0),
        "buy_hold_return_pct": round((bh_final - 1) * 100, 1),
        "buy_hold_cagr_pct": round((bh_final ** (1 / years) - 1) * 100, 2) if years > 0 else 0.0,
        "buy_hold_max_dd_pct": round(bh_dd, 1),
        "beat_buy_hold": cagr > (bh_final ** (1 / years) - 1) * 100 if years > 0 else False,
    }


def print_single(symbol: str, r: dict) -> None:
    bh_ratio = r["buy_hold_cagr_pct"] / r["buy_hold_max_dd_pct"] if r["buy_hold_max_dd_pct"] else 0
    print(f"\n=== {symbol} === {r['years']} years, {r['trades']} trades "
          f"({r['trades'] / r['years']:.1f}/yr, {r['open_at_end']} open at end)")
    print(f"  return {r['total_return_pct']:+.0f}%   CAGR {r['cagr_pct']:.2f}%   "
          f"max DD {r['max_dd_pct']:.1f}%   return/DD {r['return_per_dd']}")
    print(f"  win {r['win_pct']:.1f}%   PF {r['profit_factor']}   "
          f"avg win {r['avg_win_pct']:+.2f}%   avg loss {r['avg_loss_pct']:+.2f}%   "
          f"expectancy {r['expectancy_pct']:+.2f}%")
    print(f"  held {r['avg_bars_held']:.1f} bars, in market {r['exposure_pct']:.0f}% of days")
    print(f"  BUY AND HOLD  {r['buy_hold_return_pct']:+.0f}%   "
          f"CAGR {r['buy_hold_cagr_pct']:.2f}%   max DD {r['buy_hold_max_dd_pct']:.1f}%   "
          f"return/DD {bh_ratio:.2f}")
    verdict = "STRATEGY" if r["beat_buy_hold"] else "buy and hold"
    print(f"  higher CAGR: {verdict}"
          f"   |   better return/DD: "
          f"{'STRATEGY' if (r['return_per_dd'] or 0) > bh_ratio else 'buy and hold'}")


def main(argv=None) -> int:
    args = parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    data_dir = Path(args.data_dir)
    rows: list[dict] = []

    for symbol in symbols:
        try:
            bars = load_bars(symbol, data_dir)
        except SystemExit as exc:
            print(f"{symbol}: {exc}")
            continue

        if not args.grid:
            r = evaluate(bars, args, args.entry_level, args.exit_level)
            if r is None:
                print(f"{symbol}: not enough history from {args.start_year}")
                continue
            print_single(symbol, r)
            rows.append({"symbol": symbol, **r})
            continue

        print(f"\n=== {symbol} === CAGR % by entry (rows) x exit (cols)")
        print("      " + "".join(f"{x:>9}" for x in GRID_EXITS))
        for el in GRID_ENTRIES:
            cells = ""
            for xl in GRID_EXITS:
                if xl <= el:
                    cells += f"{'-':>9}"
                    continue
                r = evaluate(bars, args, float(el), float(xl))
                cells += f"{'n/a':>9}" if r is None else f"{r['cagr_pct']:>8.2f}%"
                if r is not None:
                    rows.append({"symbol": symbol, **r})
            print(f" <{el:>3} {cells}")
        bh = next((x for x in rows if x["symbol"] == symbol), None)
        if bh:
            print(f"      buy and hold CAGR {bh['buy_hold_cagr_pct']:.2f}%, "
                  f"max DD {bh['buy_hold_max_dd_pct']:.1f}%")

    if args.out_csv and rows:
        path = Path(args.out_csv)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {path}")

    beat = sum(1 for r in rows if r["beat_buy_hold"])
    if rows and not args.grid:
        print(f"\nbeat buy-and-hold on CAGR: {beat} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
