"""Is the SMC entry window worth changing?

smc_live_parity found time_filter's 10:05-15:30 ET scan window to be the
single most expensive constraint in the strategy: it removes 44% of all
signals, and out-of-sample return fell 5.9pp when it was applied. That
raises an obvious question and a trap. The obvious question is whether
the window should be widened. The trap is that the 5.9pp was measured
BEFORE the daily watchlist was applied, over all 503 tickers -- and the
watchlist was the one constraint that helped. The cohort the window
discards may look quite different once only 40 liquid names remain.

So everything here runs at the full live basis (reachable fills,
force_close_same_day, daily watchlist) and varies nothing but the window.

Two views, because they answer different questions:

  cohorts   every signal the strategy generates with NO window, bucketed
      by the ET half-hour the bot would act in. Says where the edge sits
      on the clock, per signal, uncontaminated by position sizing or the
      concurrency cap.

  windows   the actual walk-forward under each candidate window. Says
      what the portfolio does, which is not the same thing: two positions
      at a time means a good cohort can be crowded out by an earlier
      mediocre one, and a window that admits more signals does not
      necessarily bank more of them.

Each window is a real rebuild rather than a filter over one candidate
set. Filtering afterwards would leave every out-of-window position still
occupying its symbol's one slot, understating any wider window -- the
same subtlety that makes entry_allowed a signal-layer parameter instead
of a post-hoc drop.

What is actually changeable, before reading anything into the numbers:

  * The morning watchlist does not exist until HT_SMC_Prefilter runs at
    09:40 ET, so nothing before roughly 09:45 is reachable at all,
    whatever the table says about it. Those rows are diagnostic.
  * HT_SMC_Cycle starts at 10:02 ET and is staggered 2 minutes behind
    HT_Cycle on purpose. Moving the window earlier means rescheduling
    that task, and a stagger of 0 puts two bots' yfinance and IBKR bursts
    on top of each other.
  * The late bound trades against force_close_et (15:51): an entry at
    15:45 has six minutes to work before it is flattened at market.

Usage:
    python -m trading_bot.cli.smc_entry_window \
        --intraday-dir backtest_data/intraday_5m \
        --out smc_entry_window_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trading_bot import smc_live
from trading_bot.backtest.data import DAILY_DIR
from trading_bot.backtest.smc_engine import (
    DEFAULT_ACTION_DELAY_MINUTES,
    build_smc_candidates,
    watchlist_from_rules,
)
from trading_bot.cli.smc_entry_spec import run_one
from trading_bot.cli.smc_full_backtest import COST_BASES, DEFAULT_INITIAL_CAPITAL, leg_slippage
from trading_bot.cli.smc_rr_walkforward import (
    DEFAULT_BOUNDARIES,
    DEFAULT_FIT_START,
    align_tz,
    expanding_folds,
)
from trading_bot.data.sp500_tickers import SP500_TICKERS

# (name, earliest_et, latest_et). "unreachable" flags windows the current
# schedule cannot serve -- kept because the cohort question is worth
# answering even where the answer cannot be acted on directly.
CANDIDATE_WINDOWS = [
    {"name": "none", "window": (None, None), "unreachable": True},
    {"name": "0930-1530", "window": ("09:30", "15:30"), "unreachable": True},
    {"name": "0945-1530", "window": ("09:45", "15:30"), "unreachable": False},
    {"name": "1005-1530", "window": ("10:05", "15:30"), "unreachable": False},
    {"name": "1035-1530", "window": ("10:35", "15:30"), "unreachable": False},
    {"name": "1005-1545", "window": ("10:05", "15:45"), "unreachable": False},
    {"name": "0945-1545", "window": ("09:45", "15:45"), "unreachable": False},
]

REPORTED_BASES = ("commission_tiered", "realistic_tiered")
CURRENT_WINDOW_NAME = "1005-1530"


def action_times(dates, delay_minutes: int = DEFAULT_ACTION_DELAY_MINUTES):
    """When the bot could act on each bar, in ET -- the clock reading the
    window is compared against (see engine.entry_window_mask)."""
    return dates.dt.tz_convert("America/New_York") + pd.Timedelta(minutes=delay_minutes)


def trade_return_pct(trade: dict) -> float:
    """Net return over entry notional, summed across the fill ladder."""
    entry = trade["entry_price"]
    if entry <= 0:
        return 0.0
    return sum(f["qty_fraction"] * (f["price"] - entry) / entry for f in trade["fills"]) * 100


def cohort_table(candidates: list[tuple], freq: str = "30min") -> pd.DataFrame:
    """Per-signal outcomes bucketed by the half-hour the bot would act in.

    Deliberately per SIGNAL, not per taken position: sizing and the
    concurrency cap decide which signals become trades, and mixing that in
    would answer "which hours got slots" rather than "which hours are
    worth a slot".
    """
    if not candidates:
        return pd.DataFrame()
    rows = []
    for entry_date, symbol, trade in candidates:
        rows.append({
            "acted_at": entry_date,
            "ret_pct": trade_return_pct(trade),
            "hit_tp1": any(f["reason"] == "tp1" for f in trade["fills"]),
            "stopped": bool(trade["fills"]) and trade["fills"][0]["reason"] == "stop",
        })
    df = pd.DataFrame(rows)
    df["bucket"] = action_times(df["acted_at"]).dt.floor(freq).dt.strftime("%H:%M")
    grouped = df.groupby("bucket", sort=True).agg(
        signals=("ret_pct", "size"),
        mean_ret_pct=("ret_pct", "mean"),
        win_pct=("ret_pct", lambda s: (s > 0).mean() * 100),
        tp1_pct=("hit_tp1", "mean"),
        stopped_pct=("stopped", "mean"),
    )
    grouped["tp1_pct"] *= 100
    grouped["stopped_pct"] *= 100
    grouped["total_ret_pct"] = grouped.signals * grouped.mean_ret_pct
    return grouped.round(4)


def build_all(tickers, intraday_dir: Path, rules: dict, watchlist: dict,
              windows=CANDIDATE_WINDOWS, bases=COST_BASES) -> dict:
    """Candidates per (window name, basis name). One build per distinct
    (window, slippage): commission is portfolio-level and cannot change
    which signals exist."""
    entry = smc_live.entry_rules(rules)
    exit_ = smc_live.exit_rules(rules)
    out = {}
    for spec in windows:
        seen = {}
        for basis in bases:
            slippage = None if basis["slippage"] is None else leg_slippage(
                entry["fill"], exit_["fill"], exit_["tp1_resting_limit"],
            )
            key = json.dumps(slippage, sort_keys=True)
            if key not in seen:
                print(f"building {spec['name']:12s} slippage={key}", flush=True)
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
                    force_close_same_day=True,
                    daily_watchlist=watchlist,
                    entry_window_et=spec["window"],
                )
                print(f"  {len(seen[key])} candidates", flush=True)
            out[(spec["name"], basis["name"])] = seen[key]
    return out


def sweep(candidates_by, windows, bases, folds, rules, initial_capital) -> pd.DataFrame:
    rows = []
    for spec in windows:
        for basis in bases:
            cands = candidates_by[(spec["name"], basis["name"])]
            ref = cands[0][0] if cands else None
            common = {"window": spec["name"], "reachable": not spec["unreachable"],
                      "basis": basis["name"]}

            full = run_one(cands, rules, initial_capital, basis["commission"])
            if full:
                rows.append({**common, "scope": "full_period", "fold": 0, **full})
            for fold in folds:
                fit_start = align_tz(fold["fit_start"], ref)
                fit_end = align_tz(fold["fit_end"], ref)
                test_end = align_tz(fold["test_end"], ref)
                # Fit rows exist so a window can be CHOSEN without looking
                # at the period it is then judged on -- see nested_choice.
                for scope, lo, hi in (("fit", fit_start, fit_end), ("test", fit_end, test_end)):
                    stats = run_one(cands, rules, initial_capital, basis["commission"], lo, hi)
                    if stats:
                        rows.append({**common, "scope": scope, "fold": fold["fold"], **stats})
    return pd.DataFrame(rows)


def nested_choice(df: pd.DataFrame, basis: str, metric: str = "ret_pct",
                  reachable_only: bool = True) -> pd.DataFrame:
    """Pick each fold's window on its FIT period, score it on the test one.

    Reading the out-of-sample table and taking the best row is selection
    on the data that row is supposed to validate: with seven windows and
    three folds, some window wins by chance. This is the honest version --
    the choice sees only the fit period, so the test column is what
    re-tuning the window each fold would actually have earned, and it is
    directly comparable to leaving the window alone.
    """
    rows = []
    scoped = df[df.basis == basis]
    if reachable_only:
        scoped = scoped[scoped.reachable]
    for fold in sorted(scoped.fold.unique()):
        if fold == 0:
            continue
        fit = scoped[(scoped.fold == fold) & (scoped.scope == "fit")]
        test = scoped[(scoped.fold == fold) & (scoped.scope == "test")]
        if fit.empty or test.empty:
            continue
        chosen = fit.loc[fit[metric].idxmax(), "window"]
        chosen_test = test[test.window == chosen]
        rows.append({
            "fold": fold,
            "chosen_on_fit": chosen,
            "fit_ret_pct": round(float(fit[metric].max()), 3),
            "test_ret_pct": round(float(chosen_test[metric].iloc[0]), 3) if not chosen_test.empty else float("nan"),
            "best_test_in_hindsight": test.loc[test[metric].idxmax(), "window"],
            "best_test_ret_pct": round(float(test[metric].max()), 3),
        })
    return pd.DataFrame(rows)


def oos_table(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.scope == "test"].groupby(["window", "basis"], sort=False).agg(
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
    parser.add_argument("--out", type=Path, default=Path("smc_entry_window_results.csv"))
    parser.add_argument("--cohorts-out", type=Path, default=Path("smc_entry_window_cohorts.csv"))
    args = parser.parse_args()

    for directory in (args.intraday_dir, args.daily_dir):
        if not directory.is_dir():
            print(f"error: no such directory: {directory}")
            return 2

    rules = json.loads(args.rules.read_text())
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)
    folds = expanding_folds(args.fit_start, [b.strip() for b in args.boundaries.split(",")])

    print("rebuilding the daily watchlist...", flush=True)
    watchlist = watchlist_from_rules(list(SP500_TICKERS), args.daily_dir, rules)

    candidates_by = build_all(tickers, args.intraday_dir, rules, watchlist)

    cohorts = cohort_table(candidates_by[("none", "zero_cost")])
    if not cohorts.empty:
        cohorts.to_csv(args.cohorts_out)
        print("\n===== SIGNAL COHORTS BY ACTING HALF-HOUR (no window, zero cost) =====")
        print(cohorts.to_string())
        print("mean_ret_pct is per signal over entry notional; total_ret_pct is that "
              "times the count, i.e. what the bucket contributes in aggregate.")

    df = sweep(candidates_by, CANDIDATE_WINDOWS, COST_BASES, folds, rules, args.initial_capital)
    if df.empty:
        print("no results")
        return 1
    df.to_csv(args.out, index=False)

    shown = df[df.basis.isin(REPORTED_BASES)]
    print("\n===== FULL PERIOD =====")
    fp = shown[shown.scope == "full_period"]
    print(fp[["window", "reachable", "basis", "signals", "trades", "ret_pct",
              "max_dd_pct", "pf", "win_rate_pct"]].to_string(index=False))

    print("\n===== OUT-OF-SAMPLE (test folds summed) =====")
    print(oos_table(shown).to_string())

    for basis in REPORTED_BASES:
        nested = nested_choice(df, basis)
        if nested.empty:
            continue
        current = df[(df.basis == basis) & (df.scope == "test") & (df.window == CURRENT_WINDOW_NAME)]
        print(f"\n===== RE-TUNING THE WINDOW EACH FOLD ({basis}, reachable only) =====")
        print(nested.to_string(index=False))
        print(f"  re-tuned each fold: {nested.test_ret_pct.sum():+.3f}%   "
              f"left at {CURRENT_WINDOW_NAME}: {current.ret_pct.sum():+.3f}%   "
              f"best in hindsight: {nested.best_test_ret_pct.sum():+.3f}%")
    print("\nThe hindsight column is not achievable -- it is there to show how much of any "
          "apparent gain is just picking the winner after the fact.")

    unreachable = [w["name"] for w in CANDIDATE_WINDOWS if w["unreachable"]]
    print(f"\nreachable=False rows ({', '.join(unreachable)}) need a watchlist that does not exist "
          "until HT_SMC_Prefilter runs at 09:40 ET. They are diagnostic, not proposals.")
    print("Moving the early bound also means rescheduling HT_SMC_Cycle, which starts at 10:02 and "
          "is staggered 2 minutes off HT_Cycle to keep their data bursts apart.")
    print(f"\nwrote {args.out} and {args.cohorts_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
