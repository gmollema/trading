"""How much entry slippage the SMC strategy can absorb before it loses.

smc_fill_audit will report a measured entry rate once enough live fills
accumulate. This is the threshold to judge it against, and without it that
measurement is a number with nothing to compare to.

The rate the backtest currently charges the entry -- 2.0 bps -- is not a
measurement. It was borrowed from the stop leg, which rests at IBKR and so
never chases, on the reasoning that a next-bar fill already prices the
chase from the bars and only spread and impact remain. The entry is a
market order against the same book, so the borrow is defensible, but it
has never been checked against a single live entry fill.

The commit history quotes "break-even sits around 15 bps", and that figure
does NOT transfer: it was computed on the old spec, where the entry filled
at the order block's high on the touching bar. That fill does not exist,
the strategy has since been respecified, and the number went with it.

So: hold everything at the configured basis -- both fill specs, the live
force-close, entry window and daily watchlist -- and vary only the entry
rate. Exits keep theirs, because the question is what the ENTRY can cost,
not what every leg together can.

Each rate is a real rebuild rather than a re-simulation. Entry slippage
moves entry_price, which feeds position sizing and the guard that drops a
fill landing at or below the stop, so it changes which trades exist and
not merely what they are worth.

Usage:
    python -m trading_bot.cli.smc_entry_breakeven --out smc_entry_breakeven_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trading_bot import smc_live
from trading_bot.backtest.data import DAILY_DIR
from trading_bot.backtest.smc_engine import build_smc_candidates, watchlist_from_rules
from trading_bot.cli.smc_entry_spec import run_one
from trading_bot.cli.smc_full_backtest import (
    DEFAULT_INITIAL_CAPITAL,
    FIXED,
    TIERED,
    leg_slippage,
)
from trading_bot.cli.smc_rr_walkforward import (
    DEFAULT_BOUNDARIES,
    DEFAULT_FIT_START,
    align_tz,
    expanding_folds,
)
from trading_bot.data.sp500_tickers import SP500_TICKERS

# Dense where the crossing is expected. On this module's basis -- every
# leg slipped -- the configured 2.0 bps returns +0.23% out of sample, and
# the mean trade clears only a few bps of entry notional, so a handful of
# basis points is the whole margin. (The +0.93% quoted elsewhere is
# smc_full_backtest's commission_tiered row, which leaves fills costless;
# the two are not the same basis and mixing them up is exactly the
# confusion the COMMISSIONS rename below is guarding against.)
DEFAULT_RATES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 25.0]

# Named "realistic_*" to match smc_full_backtest's vocabulary, because
# that is what these are: every leg carries slippage, and only the
# commission schedule varies. Calling the middle row "commission_tiered"
# would read as commission-on-costless-fills, which is a different and
# much kinder basis. Cross-check: at entry=2.0 bps, realistic_tiered here
# reproduces smc_full_backtest's realistic_tiered exactly.
COMMISSIONS = [
    {"name": "realistic_no_commission", "commission": None},
    {"name": "realistic_tiered", "commission": TIERED},
    {"name": "realistic_fixed", "commission": FIXED},
]

# The one that decides it: tiered is what this account actually pays.
DECIDING_BASIS = "realistic_tiered"


def slippage_at(rate: float, entry_fill: str, exit_fill: str, tp1_resting_limit: bool) -> dict:
    """The configured per-leg rates with only the entry's replaced."""
    return {**leg_slippage(entry_fill, exit_fill, tp1_resting_limit), "entry": rate}


def build_all(tickers, intraday_dir: Path, rules: dict, watchlist: dict, rates: list[float]) -> dict:
    entry, exit_ = smc_live.entry_rules(rules), smc_live.exit_rules(rules)
    tf = rules.get("time_filter") or {}
    out = {}
    for rate in rates:
        print(f"building entry_slippage={rate} bps", flush=True)
        out[rate] = build_smc_candidates(
            tickers,
            intraday_dir=intraday_dir,
            time_window_bars=rules["time_window_bars"],
            tp1_fraction=rules["tp1_fraction"],
            swing_window=rules["swing_window"],
            slippage_bps=slippage_at(rate, entry["fill"], exit_["fill"], exit_["tp1_resting_limit"]),
            entry_fill=entry["fill"],
            require_ob_reclaim=entry["require_ob_reclaim"],
            exit_fill=exit_["fill"],
            tp1_resting_limit=exit_["tp1_resting_limit"],
            force_close_same_day=True,
            entry_window_et=(tf.get("earliest_entry_et"), tf.get("latest_entry_et")),
            daily_watchlist=watchlist,
        )
        print(f"  {len(out[rate])} candidates", flush=True)
    return out


def sweep(candidates_by_rate, folds, rules, initial_capital) -> pd.DataFrame:
    rows = []
    for rate, cands in candidates_by_rate.items():
        ref = cands[0][0] if cands else None
        for basis in COMMISSIONS:
            common = {"entry_slippage_bps": rate, "basis": basis["name"]}
            full = run_one(cands, rules, initial_capital, basis["commission"])
            if full:
                rows.append({**common, "scope": "full_period", "fold": 0, **full})
            for fold in folds:
                stats = run_one(
                    cands, rules, initial_capital, basis["commission"],
                    align_tz(fold["fit_end"], ref), align_tz(fold["test_end"], ref),
                )
                if stats:
                    rows.append({**common, "scope": "test", "fold": fold["fold"], **stats})
    return pd.DataFrame(rows)


def oos_by_rate(df: pd.DataFrame, basis: str) -> pd.DataFrame:
    scoped = df[(df.scope == "test") & (df.basis == basis)]
    return scoped.groupby("entry_slippage_bps", sort=True).agg(
        trades=("trades", "sum"), ret_pct=("ret_pct", "sum"),
        worst_dd_pct=("max_dd_pct", "min"), pf=("pf", "mean"),
    ).round(3)


def crossing(curve: pd.DataFrame) -> float | None:
    """Where summed out-of-sample return crosses zero, interpolated.

    Linear between the bracketing rates: the curve is smooth enough over a
    few basis points for that, and quoting a bracket rather than a fitted
    point would invite reading precision into a grid that has none. None
    if it never crosses within the swept range -- worth distinguishing
    from "crosses at the last point tested".
    """
    rates = list(curve.index)
    rets = list(curve.ret_pct)
    for (r0, v0), (r1, v1) in zip(zip(rates, rets), zip(rates[1:], rets[1:])):
        if v0 >= 0 >= v1 and v0 != v1:
            return r0 + (r1 - r0) * (v0 / (v0 - v1))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intraday-dir", type=Path, default=Path("backtest_data/intraday_5m"))
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--rules", type=Path, default=Path("smc_rules.json"))
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fit-start", type=str, default=DEFAULT_FIT_START)
    parser.add_argument("--boundaries", type=str, default=",".join(DEFAULT_BOUNDARIES))
    parser.add_argument("--rates", type=str, default=",".join(str(r) for r in DEFAULT_RATES))
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("smc_entry_breakeven_results.csv"))
    args = parser.parse_args()

    rules = json.loads(args.rules.read_text())
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)
    folds = expanding_folds(args.fit_start, [b.strip() for b in args.boundaries.split(",")])
    rates = [float(r) for r in args.rates.split(",")]

    entry, exit_ = smc_live.entry_rules(rules), smc_live.exit_rules(rules)
    print(f"basis: entry_fill={entry['fill']} reclaim={entry['require_ob_reclaim']} "
          f"exit_fill={exit_['fill']}, live constraints on", flush=True)
    print("rebuilding the daily watchlist...", flush=True)
    watchlist = watchlist_from_rules(list(SP500_TICKERS), args.daily_dir, rules)

    candidates_by_rate = build_all(tickers, args.intraday_dir, rules, watchlist, rates)
    df = sweep(candidates_by_rate, folds, rules, args.initial_capital)
    if df.empty:
        print("no results")
        return 1
    df.to_csv(args.out, index=False)

    for basis in (b["name"] for b in COMMISSIONS):
        curve = oos_by_rate(df, basis)
        print(f"\n===== OUT-OF-SAMPLE by entry slippage ({basis}) =====")
        print(curve.to_string())
        point = crossing(curve)
        if point is None:
            sign = "still positive" if curve.ret_pct.iloc[-1] > 0 else "already negative"
            print(f"  no zero crossing within {rates[0]}-{rates[-1]} bps ({sign} at the far end)")
        else:
            print(f"  break-even at ~{point:.1f} bps of entry slippage")

    print("\nExits keep their configured rate throughout: this is what the ENTRY can cost, "
          "not what every leg together can.")
    print(f"Compare against smc_fill_audit's measured entry median. The configured "
          f"{leg_slippage(entry['fill'], exit_['fill'], exit_['tp1_resting_limit'])['entry']} bps "
          "is an assumption borrowed from the stop leg, not a measurement.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
