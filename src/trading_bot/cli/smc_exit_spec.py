"""Compare SMC exit specifications that an order can actually reach.

The companion to smc_entry_spec, for the other side of the trade. The
entry was respecified on 2026-08-29 because it filled at a trigger no
order type reaches; all four exits had the same defect, and one of them
far worse than the entry ever did:

  new_high_exit  fires when a swing high is CONFIRMED, which takes
      swing_window bars -- 20 in smc_rules.json, or 100 minutes on 5-min
      bars -- and then fills at that pivot bar's own close. It sold at a
      local peak chosen with a hundred minutes of hindsight. This is the
      leg to watch in the table below.

  tp1  fires when a bar's high reaches the target and fills AT the
      target. The bot reads bar highs after the close and sends a market
      order, so it gets the next bar, not the level. It could rest a
      limit there instead -- unlike the entry's buy limit, a sell limit
      above the market is not adversely selected, since it fills exactly
      when price reaches the target, which is the event being traded.
      That needs OCA bracketing live (TP1 sells part of the position
      while the stop covers all of it), so it is modelled here before
      being built.

  stop  nearly earns its level: it rests at IBKR and triggers intrabar.
      What it does not survive is a bar that OPENED through it, where
      the level was never on offer. Priced from the bar under next_open.

  same_day_force_close  inert in these runs (the harnesses leave
      force_close_same_day off -- see the caveat at the bottom of the
      output), but respecified for consistency.

The entry is held fixed at whatever smc_rules.json now specifies, so this
isolates the exit question rather than re-deciding a settled one.

Usage:
    python -m trading_bot.cli.smc_exit_spec \
        --intraday-dir backtest_data/intraday_5m \
        --out smc_exit_spec_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trading_bot import smc_live
from trading_bot.backtest.smc_engine import build_smc_candidates
from trading_bot.cli.smc_entry_spec import run_one
from trading_bot.cli.smc_full_backtest import COST_BASES, DEFAULT_INITIAL_CAPITAL, leg_slippage
from trading_bot.cli.smc_rr_walkforward import (
    DEFAULT_BOUNDARIES,
    DEFAULT_FIT_START,
    align_tz,
    expanding_folds,
    summarize,
)
from trading_bot.data.sp500_tickers import SP500_TICKERS

EXIT_SPECS = [
    {"name": "level", "exit_fill": "level", "tp1_resting_limit": False},
    {"name": "next_open", "exit_fill": "next_open", "tp1_resting_limit": False},
    {"name": "next_open_tp1_limit", "exit_fill": "next_open", "tp1_resting_limit": True},
]

# Printed in full; the CSV carries every basis in COST_BASES.
REPORTED_BASES = ("zero_cost", "commission_tiered", "realistic_tiered")


def build_all(tickers, intraday_dir: Path, rules: dict, entry: dict, specs=EXIT_SPECS, bases=COST_BASES) -> dict:
    """Candidates keyed by (exit spec name, cost basis name).

    Signal generation depends on the execution spec and the slippage dict
    but never on commission, which is applied at portfolio level and
    cannot change which signals exist -- so bases sharing a slippage
    setting share one build.
    """
    out = {}
    for spec in specs:
        seen = {}
        for basis in bases:
            slippage = None if basis["slippage"] is None else leg_slippage(
                entry["fill"], spec["exit_fill"], spec["tp1_resting_limit"],
            )
            key = json.dumps(slippage, sort_keys=True)
            if key not in seen:
                print(f"building {spec['name']:20s} slippage={key}", flush=True)
                seen[key] = build_smc_candidates(
                    tickers,
                    intraday_dir=intraday_dir,
                    time_window_bars=rules["time_window_bars"],
                    tp1_fraction=rules["tp1_fraction"],
                    swing_window=rules["swing_window"],
                    slippage_bps=slippage,
                    entry_fill=entry["fill"],
                    require_ob_reclaim=entry["require_ob_reclaim"],
                    exit_fill=spec["exit_fill"],
                    tp1_resting_limit=spec["tp1_resting_limit"],
                )
                print(f"  {len(seen[key])} candidates", flush=True)
            out[(spec["name"], basis["name"])] = seen[key]
    return out


def oos_table(df: pd.DataFrame) -> pd.DataFrame:
    """Test folds summed per (exit spec, basis) -- the out-of-sample read."""
    return df[df.scope == "test"].groupby(["exit_spec", "basis"], sort=False).agg(
        trades=("trades", "sum"), ret_pct=("ret_pct", "sum"),
        worst_dd_pct=("max_dd_pct", "min"), pf=("pf", "mean"), win_rate_pct=("win_rate_pct", "mean"),
    ).round(3)


def sweep(candidates_by, specs, bases, folds, rules, initial_capital) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for basis in bases:
            cands = candidates_by[(spec["name"], basis["name"])]
            ref = cands[0][0] if cands else None
            common = {"exit_spec": spec["name"], "basis": basis["name"]}

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intraday-dir", type=Path, default=Path("backtest_data/intraday_5m"))
    parser.add_argument("--rules", type=Path, default=Path("smc_rules.json"))
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fit-start", type=str, default=DEFAULT_FIT_START)
    parser.add_argument("--boundaries", type=str, default=",".join(DEFAULT_BOUNDARIES))
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated override list")
    parser.add_argument("--out", type=Path, default=Path("smc_exit_spec_results.csv"))
    args = parser.parse_args()

    if not args.intraday_dir.is_dir():
        print(f"error: no such intraday dir: {args.intraday_dir}")
        return 2

    rules = json.loads(args.rules.read_text())
    entry = smc_live.entry_rules(rules)
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)
    folds = expanding_folds(args.fit_start, [b.strip() for b in args.boundaries.split(",")])

    print(f"entry held at: fill={entry['fill']} require_ob_reclaim={entry['require_ob_reclaim']}", flush=True)
    candidates_by = build_all(tickers, args.intraday_dir, rules, entry)
    df = sweep(candidates_by, EXIT_SPECS, COST_BASES, folds, rules, args.initial_capital)
    if df.empty:
        print("no results")
        return 1
    df.insert(0, "entry_spec", f"{entry['fill']}{'_reclaim' if entry['require_ob_reclaim'] else ''}")
    df.to_csv(args.out, index=False)

    shown = df[df.basis.isin(REPORTED_BASES)]
    print("\n===== FULL PERIOD =====")
    fp = shown[shown.scope == "full_period"]
    print(fp[["exit_spec", "basis", "signals", "trades", "ret_pct",
              "max_dd_pct", "pf", "win_rate_pct"]].to_string(index=False))

    print("\n===== OUT-OF-SAMPLE (test folds summed) =====")
    print(oos_table(shown).to_string())

    print("\nlevel is the historical spec: every exit booked its own trigger as its fill, "
          "and new_high_exit filled at a pivot close swing_window bars before it could be known.")
    print("next_open_tp1_limit describes a bot that does not exist yet -- it needs OCA bracketing, "
          "since TP1 sells part of a position the stop covers all of.")
    print("Caveat unchanged from smc_entry_spec: these runs leave force_close_same_day off and "
          "skip the daily_trend_filter, while the live bot does both.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
