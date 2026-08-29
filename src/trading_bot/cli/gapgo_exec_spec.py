"""Price the gap-and-go strategy at fills an order can actually get.

The same treatment backtest/smc_signals.py went through on 2026-08-29,
applied to the other bot. Two defects, identical in kind:

  the entry books the signal bar's own CLOSE. cycle.py computes the
      D1-D3/I1-I3 filters from the latest 5-minute bar and then sends a
      market order, which fills on the NEXT bar. The close it was decided
      on is not a price it can trade at.

  the market-order exits book their trigger. The partial profit at 0.75R,
      the max-hold exit and the daily force-close are all MarketOrders
      sent once a bar has closed and told the bot something happened.
      The stop is the exception and nearly earns its level -- it rests at
      IBKR as a real StopOrder -- except on a bar that OPENED through it,
      where the level was never on offer.

And no cost model at all: portfolio.py's own docstring said it "ignores
slippage", and run_backtest charged no commission, so every gap-and-go
figure in this repo is frictionless as well as unreachable.

A caveat that does NOT apply to the SMC equivalent, and matters: this bot
has never produced a measured fill. trades.csv holds two BUYs, both
Submitted with fill_price 0. Every slippage rate here is therefore
borrowed from the SMC bot's live fills -- same broker, same order types,
different symbols and times -- and is an assumption rather than a
measurement. The stop rate is the better-founded of the two (a resting
StopOrder measured 0-5.9 bps over 7 fills); the market-leg rate is the
48-79 bps those TP1 fills gave up, which under a next-bar fill is already
priced by the bars, leaving spread and impact.

Usage:
    python -m trading_bot.cli.gapgo_exec_spec --out gapgo_exec_spec_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trading_bot import live
from trading_bot.backtest.engine import _build_universe, build_day_frames, run_backtest
from trading_bot.cli.smc_full_backtest import (
    MARKET_LEG_SLIPPAGE_BPS,
    RESIDUAL_ENTRY_SLIPPAGE_BPS,
    STOP_SLIPPAGE_BPS,
    TIERED,
    FIXED,
)
from trading_bot.cli.smc_rr_walkforward import expanding_folds, summarize
from trading_bot.data.sp500_tickers import SP500_TICKERS

DEFAULT_INITIAL_CAPITAL = 100_000.0

# The gap-and-go cache reaches back further than the SMC one, so these are
# its own rather than smc_rr_walkforward's.
DEFAULT_FIT_START = "2025-07-15"
DEFAULT_BOUNDARIES = ["2026-03-14", "2026-05-14", "2026-07-14", "2026-08-26"]

FILL_SPECS = ("level", "next_open")


def leg_slippage(fill_spec: str) -> dict:
    """Per-leg rates for a fill spec.

    The chase is either modelled by the fill bar or charged as basis
    points, never both: the market-leg figure IS the delay between a level
    being touched and an order arriving, so charging it on top of a
    next-bar fill would bill the same lag twice. The stop never chases --
    it rests at the broker -- and what does beat it, a gap through the
    level, is priced from the bar under fill_spec="next_open".
    """
    market_rate = MARKET_LEG_SLIPPAGE_BPS if fill_spec == "level" else RESIDUAL_ENTRY_SLIPPAGE_BPS
    return {
        "entry": market_rate,
        "stop": STOP_SLIPPAGE_BPS,
        "partial_profit": market_rate,
        "force_close": market_rate,
        "max_hold_reached": market_rate,
        "end_of_data": 0.0,
    }


def cost_bases(fill_spec: str) -> list[dict]:
    slippage = leg_slippage(fill_spec)
    return [
        {"name": "zero_cost", "slippage": None, "commission": None},
        {"name": "commission_tiered", "slippage": None, "commission": TIERED},
        {"name": "commission_fixed", "slippage": None, "commission": FIXED},
        {"name": "realistic_tiered", "slippage": slippage, "commission": TIERED},
        {"name": "realistic_fixed", "slippage": slippage, "commission": FIXED},
    ]


REPORTED_BASES = ("zero_cost", "commission_tiered", "realistic_tiered")


def run_one(tickers, rules, universe, day_frames, initial_capital, fill_spec, basis) -> dict | None:
    """One (fill spec, cost basis) over an already-windowed universe."""
    per_share, minimum = basis["commission"] if basis["commission"] else (None, 1.0)
    result = run_backtest(
        tickers,
        rules,
        initial_capital,
        universe=universe,
        day_frames=day_frames,
        fill_spec=fill_spec,
        slippage_bps=basis["slippage"],
        commission_per_share=per_share,
        commission_min=minimum,
    )
    if not result.get("equity_curve"):
        return None
    stats = summarize(result, initial_capital)
    stats["buys"] = sum(1 for t in result["trades"] if t["side"] == "BUY")
    return stats


def windows_for(universe, day_frames, folds) -> list[tuple]:
    """(scope, fold, universe, day_frames) per evaluation window.

    Built once and shared across every (fill_spec, basis) pair: narrowing
    the universe forces day_frames to be rebuilt with it, and doing that
    inside the cost loop would repeat the expensive half of the job forty
    times over for ten answers.
    """
    out = [("full_period", 0, universe, day_frames)]
    for fold in folds:
        lo, hi = fold["fit_end"].date(), fold["test_end"].date()
        window = universe[(universe["trading_date"] >= lo) & (universe["trading_date"] <= hi)]
        if window.empty:
            continue
        out.append(("test", fold["fold"], window, build_day_frames(window)))
    return out


def sweep(tickers, rules, universe, day_frames, folds, initial_capital) -> pd.DataFrame:
    rows = []
    for scope, fold_no, window, frames in windows_for(universe, day_frames, folds):
        for fill_spec in FILL_SPECS:
            for basis in cost_bases(fill_spec):
                stats = run_one(tickers, rules, window, frames, initial_capital, fill_spec, basis)
                if not stats:
                    continue
                rows.append({"fill_spec": fill_spec, "basis": basis["name"],
                             "scope": scope, "fold": fold_no, **stats})
                if scope == "full_period":
                    print(f"  {fill_spec:10s} {basis['name']:18s} ret {stats['ret_pct']:+8.3f}%  "
                          f"pf {stats['pf']}  buys {stats['buys']}", flush=True)
    return pd.DataFrame(rows)


def oos_table(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.scope == "test"].groupby(["fill_spec", "basis"], sort=False).agg(
        trades=("trades", "sum"), ret_pct=("ret_pct", "sum"),
        worst_dd_pct=("max_dd_pct", "min"), pf=("pf", "mean"), win_rate_pct=("win_rate_pct", "mean"),
    ).round(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rules", type=Path, default=Path("rules.json"))
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fit-start", type=str, default=DEFAULT_FIT_START)
    parser.add_argument("--boundaries", type=str, default=",".join(DEFAULT_BOUNDARIES))
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated override list")
    parser.add_argument("--out", type=Path, default=Path("gapgo_exec_spec_results.csv"))
    args = parser.parse_args()

    rules = live.load_rules(args.rules)
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)
    folds = expanding_folds(args.fit_start, [b.strip() for b in args.boundaries.split(",")])

    print(f"building the universe across {len(tickers)} tickers...", flush=True)
    universe = _build_universe(tickers, rules)
    day_frames = build_day_frames(universe)
    print(f"  {len(universe)} bars, {int(universe.entry_signal.sum())} entry signals", flush=True)

    df = sweep(tickers, rules, universe, day_frames, folds, args.initial_capital)
    if df.empty:
        print("no results")
        return 1
    df.to_csv(args.out, index=False)

    shown = df[df.basis.isin(REPORTED_BASES)]
    print("\n===== FULL PERIOD =====")
    fp = shown[shown.scope == "full_period"]
    print(fp[["fill_spec", "basis", "buys", "trades", "ret_pct",
              "max_dd_pct", "pf", "win_rate_pct"]].to_string(index=False))

    print("\n===== OUT-OF-SAMPLE (test folds summed) =====")
    print(oos_table(shown).to_string())

    print("\nlevel is what every gap-and-go figure in this repo was scored on: the entry books the "
          "signal bar's own close and the market exits book their trigger. Nothing executes either.")
    print("Every slippage rate here is borrowed from the SMC bot -- this one has never produced a "
          "measured fill (trades.csv: two BUYs, both Submitted at fill_price 0).")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
