"""What the SMC backtest was scoring versus what the bot actually does.

The execution specs were fixed on 2026-08-29 (smc_entry_spec,
smc_exit_spec). Three constraints the live bot obeys were still missing
from every harness, and they are not details:

  force_close_same_day  smc_cycle flattens everything at 15:51 ET
      (time_filter.force_close_et). The harnesses ran with it OFF, so
      they held positions overnight and collected gaps the bot never sees
      -- in either direction.

  entry window  the live cycle only scans between time_filter's
      earliest_entry_et and latest_entry_et (10:05-15:30 ET). The
      backtest opened positions at 09:35 and 15:45, times the bot does
      not look at. This is the big one by signal count: order blocks form
      on the open's volatility, so a large share of them retest before
      the bot is awake.

  daily watchlist  cli/smc_prefilter.py writes at most
      universe.max_watchlist_size (40) names each morning -- prior close
      above SMA200, at or above min_price_usd, ranked by 20-day dollar
      volume -- and smc_cycle scans only those. The backtest scanned all
      503. daily_trend_filter reproduced the SMA200 screen alone, so even
      with it on the universe was still several times too large, and
      biased: the cap keeps the most liquid names, where fills are best
      and 5-minute bars most trustworthy.

Applied cumulatively, so each row is one step from the number the repo
published to the one the bot can earn. Nothing here changes the strategy;
it changes which trades the bot was ever in a position to take.

Usage:
    python -m trading_bot.cli.smc_live_parity \
        --intraday-dir backtest_data/intraday_5m \
        --out smc_live_parity_results.csv
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
from trading_bot.cli.smc_full_backtest import COST_BASES, DEFAULT_INITIAL_CAPITAL, leg_slippage
from trading_bot.cli.smc_rr_walkforward import (
    DEFAULT_BOUNDARIES,
    DEFAULT_FIT_START,
    align_tz,
    expanding_folds,
)
from trading_bot.data.sp500_tickers import SP500_TICKERS

REPORTED_BASES = ("zero_cost", "commission_tiered", "realistic_tiered")


def parity_steps(rules: dict, watchlist: dict) -> list[dict]:
    """Cumulative constraint sets, from the old harness basis to live."""
    tf = rules.get("time_filter") or {}
    window = (tf.get("earliest_entry_et"), tf.get("latest_entry_et"))
    steps = [{"name": "harness", "kwargs": {}}]
    steps.append({"name": "+force_close", "kwargs": {**steps[-1]["kwargs"],
                                                     "force_close_same_day": True}})
    steps.append({"name": "+entry_window", "kwargs": {**steps[-1]["kwargs"],
                                                      "entry_window_et": window}})
    steps.append({"name": "+watchlist", "kwargs": {**steps[-1]["kwargs"],
                                                   "daily_watchlist": watchlist}})
    return steps


def build_all(tickers, intraday_dir: Path, rules: dict, steps: list[dict], bases=COST_BASES) -> dict:
    """Candidates keyed by (step name, basis name), one build per distinct
    (step, slippage) -- commission is portfolio-level and cannot change
    which signals exist."""
    entry = smc_live.entry_rules(rules)
    exit_ = smc_live.exit_rules(rules)
    out = {}
    for step in steps:
        seen = {}
        for basis in bases:
            slippage = None if basis["slippage"] is None else leg_slippage(
                entry["fill"], exit_["fill"], exit_["tp1_resting_limit"],
            )
            key = json.dumps(slippage, sort_keys=True)
            if key not in seen:
                print(f"building {step['name']:15s} slippage={key}", flush=True)
                seen[key] = build_smc_candidates(
                    tickers,
                    intraday_dir=intraday_dir,
                    time_window_bars=rules["time_window_bars"],
                    tp1_fraction=rules["tp1_fraction"],
                    swing_window=rules["swing_window"],
                    slippage_bps=slippage,
                    entry_fill=entry["fill"],
                    require_ob_reclaim=entry["require_ob_reclaim"],
                    exit_fill=exit_["fill"],
                    tp1_resting_limit=exit_["tp1_resting_limit"],
                    **step["kwargs"],
                )
                print(f"  {len(seen[key])} candidates", flush=True)
            out[(step["name"], basis["name"])] = seen[key]
    return out


def sweep(candidates_by, steps, bases, folds, rules, initial_capital) -> pd.DataFrame:
    rows = []
    for step in steps:
        for basis in bases:
            cands = candidates_by[(step["name"], basis["name"])]
            ref = cands[0][0] if cands else None
            common = {"step": step["name"], "basis": basis["name"]}

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


def oos_table(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.scope == "test"].groupby(["step", "basis"], sort=False).agg(
        trades=("trades", "sum"), ret_pct=("ret_pct", "sum"),
        worst_dd_pct=("max_dd_pct", "min"), pf=("pf", "mean"), win_rate_pct=("win_rate_pct", "mean"),
    ).round(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intraday-dir", type=Path, default=Path("backtest_data/intraday_5m"))
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--rules", type=Path, default=Path("smc_rules.json"))
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fit-start", type=str, default=DEFAULT_FIT_START)
    parser.add_argument("--boundaries", type=str, default=",".join(DEFAULT_BOUNDARIES))
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated override list")
    parser.add_argument("--out", type=Path, default=Path("smc_live_parity_results.csv"))
    args = parser.parse_args()

    for directory in (args.intraday_dir, args.daily_dir):
        if not directory.is_dir():
            print(f"error: no such directory: {directory}")
            return 2

    rules = json.loads(args.rules.read_text())
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)
    folds = expanding_folds(args.fit_start, [b.strip() for b in args.boundaries.split(",")])

    # The watchlist is screened over the WHOLE index regardless of which
    # tickers are being simulated: the 40-name cap is a global ranking, and
    # scoring a subset against a cap computed from that subset would hand
    # it slots the live bot never had.
    print("rebuilding the daily watchlist...", flush=True)
    watchlist = watchlist_from_rules(list(SP500_TICKERS), args.daily_dir, rules)
    sized = [len(v) for v in watchlist.values() if v]
    print(f"  {len(watchlist)} dates, median {int(pd.Series(sized).median())} names", flush=True)

    steps = parity_steps(rules, watchlist)
    candidates_by = build_all(tickers, args.intraday_dir, rules, steps)
    df = sweep(candidates_by, steps, COST_BASES, folds, rules, args.initial_capital)
    if df.empty:
        print("no results")
        return 1
    df.to_csv(args.out, index=False)

    shown = df[df.basis.isin(REPORTED_BASES)]
    print("\n===== FULL PERIOD =====")
    fp = shown[shown.scope == "full_period"]
    print(fp[["step", "basis", "signals", "trades", "ret_pct",
              "max_dd_pct", "pf", "win_rate_pct"]].to_string(index=False))

    print("\n===== OUT-OF-SAMPLE (test folds summed) =====")
    print(oos_table(shown).to_string())

    print("\nsteps are cumulative: +watchlist is the live bot's actual constraint set.")
    print("'harness' is what every SMC figure in this repo was scored on before today.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
