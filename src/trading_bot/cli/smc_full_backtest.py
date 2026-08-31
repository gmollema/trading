"""Full SMC backtest across cost bases, in-sample and walk-forward.

Every SMC figure in this repo predating 2026-08-28 was produced on the
engine's zero-cost defaults (slippage_bps 0.0 for every leg,
commission_per_share None). That is not a survivable basis for this
strategy: the mean trade returns ~32 bps of entry notional and the median
stop sits 17.3 bps from entry, while live market-order fills measured 48.5
and 79.4 bps adverse on TP1. Costs here are the same order as the edge, so
they decide the answer rather than trimming it.

So this runs the SAME data across several cost bases side by side, on the
execution spec smc_rules.json configures. Note what zero_cost means now
that the fills themselves are reachable: it keeps the configured fill
PRICES and drops only commission and the residual spread, so the gap to
the realistic rows is the cost overlay by itself. Under the old spec that
row also assumed a fill price nothing could get -- a limit at ob_high
fills 31-42% of signals and almost only when the setup is failing
(smc_fill_model) -- which made it uninterpretable rather than merely
optimistic.

Slippage is applied per leg rather than as one rate, because the legs
differ by order type. The stop rests at IBKR as a real StopOrder and
measured 0-5.9 bps over 7 live fills; the non-stop exits are MarketOrders
sent after a level is touched, and the two TP1 fills measured 48.5 and
79.4.

The ENTRY leg's rate depends on the entry spec, which is read from the
rules file the live bot uses (see smc_live.entry_rules) rather than fixed
here. Under the old "level" spec it borrowed TP1's 64 bps, since it was
the same order type chasing the same kind of level. Under a next-bar fill
the chase is read off the bars instead, so only spread and impact are left
to charge -- see RESIDUAL_ENTRY_SLIPPAGE_BPS. Charging both would bill the
same delay twice. The exit legs are set the same way from the rules
file's exit spec -- they had the same defect and it is fixed the same
way, with the new-high exit the worst of the three (it fired on a
confirmed pivot but filled at that pivot's own close, swing_window bars
earlier). See leg_slippage.

It also runs under the constraints the live bot actually operates within,
which no harness in this repo did before 2026-08-29: positions flattened
same-day (time_filter.force_close_et), entries only inside the scan window
(time_filter.earliest/latest_entry_et), and a universe limited to the
morning watchlist cli/smc_prefilter.py would have written that day
(universe.*, ~40 names out of 503). Those three cost more than every
execution-spec correction put together -- see smc_live_parity, which walks
them one at a time. --harness-basis drops them, for reproducing the older
figures rather than for producing new ones.

Candidates are rebuilt only when slippage changes, since commission is
applied at portfolio level and cannot affect signal generation.

Usage:
    python -m trading_bot.cli.smc_full_backtest \
        --intraday-dir backtest_data/intraday_5m \
        --out smc_full_backtest_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trading_bot import smc_live
from trading_bot.backtest.data import DAILY_DIR
from trading_bot.backtest.smc_engine import (
    build_smc_candidates,
    simulate_smc_portfolio,
    watchlist_from_rules,
)
from trading_bot.backtest.smc_signals import DEFAULT_EXIT_FILL
from trading_bot.cli.smc_rr_walkforward import (
    DEFAULT_BOUNDARIES,
    DEFAULT_FIT_START,
    align_tz,
    expanding_folds,
    summarize,
)
from trading_bot.data.sp500_tickers import SP500_TICKERS

DEFAULT_INITIAL_CAPITAL = 100_000.0

# Measured on live fills 2026-08-28; see the module docstring.
STOP_SLIPPAGE_BPS = 2.0
MARKET_LEG_SLIPPAGE_BPS = 64.0
ENTRY_SLIPPAGE_BPS = MARKET_LEG_SLIPPAGE_BPS  # assumed, not measured

# What is left on the entry once the fill bar itself prices the chase. The
# 64 bps above is (fill - level)/level for an order sent up to a bar after
# the level was touched: it IS the chase, and charging it on top of a
# next-bar fill would bill the same delay twice. The stop leg is the
# evidence for the remainder -- a market execution that does not chase,
# measured 0-5.9 bps over 7 live fills, median 2.0.
RESIDUAL_ENTRY_SLIPPAGE_BPS = STOP_SLIPPAGE_BPS


def leg_slippage(entry_fill: str, exit_fill: str, tp1_resting_limit: bool = False) -> dict:
    """Per-leg slippage for the execution spec in force.

    Each leg is charged the rate its own order type gives up, and only
    what the fill model has not already accounted for. A "level" spec
    books the trigger as the fill, so the whole chase has to be charged as
    basis points; a next-bar spec reads the chase off the bars, leaving
    spread and impact. Charging both would bill the same delay twice.

    The stop is the exception in both directions: it rests at the broker,
    so it never chases, and the one thing that does beat it -- a bar
    opening straight through the level -- is priced from the bar under
    exit_fill="next_open" rather than as a rate.

    A resting TP1 limit gives up nothing: a sell limit fills at its price
    or better, which is what makes it worth the OCA plumbing it needs.
    """
    entry_rate = MARKET_LEG_SLIPPAGE_BPS if entry_fill == "level" else RESIDUAL_ENTRY_SLIPPAGE_BPS
    exit_rate = MARKET_LEG_SLIPPAGE_BPS if exit_fill == "level" else RESIDUAL_ENTRY_SLIPPAGE_BPS
    return {
        "entry": entry_rate,
        "stop": STOP_SLIPPAGE_BPS,
        "tp1": 0.0 if tp1_resting_limit else exit_rate,
        "new_high_exit": exit_rate,
        "same_day_force_close": exit_rate,
        "end_of_data": 0.0,
    }


# The original all-market-legs basis, kept as a named constant because the
# historical figures were produced against it.
MEASURED_SLIPPAGE = leg_slippage("level", "level")


# IBKR US-stock schedules. Tiered is the cheaper base rate; Fixed is the
# conservative bound. The per-ORDER minimum is what bites at this
# strategy's sizes -- a 9-share lot pays the minimum, not the per-share
# rate, and a round trip through TP1 pays it three times.
TIERED = (0.0035, 0.35)
FIXED = (0.005, 1.00)

COST_BASES = [
    {"name": "zero_cost", "slippage": None, "commission": None},
    {"name": "commission_tiered", "slippage": None, "commission": TIERED},
    {"name": "commission_fixed", "slippage": None, "commission": FIXED},
    {"name": "realistic_tiered", "slippage": MEASURED_SLIPPAGE, "commission": TIERED},
    {"name": "realistic_fixed", "slippage": MEASURED_SLIPPAGE, "commission": FIXED},
]


def entry_slippage_bps(entry_fill: str) -> float:
    """The entry rate that goes with a given fill spec."""
    return leg_slippage(entry_fill, DEFAULT_EXIT_FILL)["entry"]


def cost_bases_for(
    entry_fill: str,
    exit_fill: str = DEFAULT_EXIT_FILL,
    tp1_resting_limit: bool = False,
    bases: list[dict] | None = None,
) -> list[dict]:
    """COST_BASES with every leg's rate matched to the spec in force.

    The costless bases keep no slippage at all: they exist to isolate what
    the fill respec alone does from what the cost overlay adds, and
    charging them a rate would defeat that.
    """
    slippage = leg_slippage(entry_fill, exit_fill, tp1_resting_limit)
    return [
        {**b, "slippage": None if b["slippage"] is None else slippage}
        for b in (COST_BASES if bases is None else bases)
    ]


def slippage_key(slippage: dict | None) -> str:
    """Stable identity for a slippage setting, so candidate generation --
    the expensive step -- is shared by every cost basis using it."""
    if slippage is None:
        return "none"
    return json.dumps(slippage, sort_keys=True)


def group_by_slippage(cost_bases: list[dict]) -> dict[str, dict]:
    """Map slippage identity -> the setting itself, deduplicated."""
    return {slippage_key(c["slippage"]): c["slippage"] for c in cost_bases}


def run_one(candidates, rules, initial_capital, commission, lo=None, hi=None) -> dict:
    """Simulate one cost basis over an optional [lo, hi) entry-date window."""
    window = candidates
    if lo is not None and hi is not None:
        window = [c for c in candidates if lo <= c[0] < hi]
    if not window:
        return None
    risk = rules["risk"]
    per_share, minimum = (commission if commission else (None, 1.0))
    result = simulate_smc_portfolio(
        window,
        initial_capital,
        risk_pct=risk["max_risk_per_trade_pct"],
        max_position_pct=risk["max_position_size_pct_of_portfolio"],
        max_concurrent_positions=risk["max_concurrent_positions"],
        commission_per_share=per_share,
        commission_min=minimum,
    )
    stats = summarize(result, initial_capital)
    stats["signals"] = len(window)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intraday-dir", type=Path, default=Path("backtest_data/intraday_5m"))
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--rules", type=Path, default=Path("smc_rules.json"))
    parser.add_argument("--early-retest", choices=["skip", "defer"], default=None,
                        help="override smc_rules.json's entry.early_retest, for A/B runs")
    parser.add_argument("--harness-basis", action="store_true",
                        help="drop the live bot's same-day close, entry window and daily "
                             "watchlist -- the pre-2026-08-29 basis, for reproducing old figures")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fit-start", type=str, default=DEFAULT_FIT_START)
    parser.add_argument("--boundaries", type=str, default=",".join(DEFAULT_BOUNDARIES))
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("smc_full_backtest_results.csv"))
    args = parser.parse_args()

    if not args.intraday_dir.is_dir():
        print(f"error: no such intraday dir: {args.intraday_dir}")
        return 2

    rules = json.loads(args.rules.read_text())
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)
    folds = expanding_folds(args.fit_start, [b.strip() for b in args.boundaries.split(",")])

    # The entry spec comes from the rules file the live bot reads, so this
    # scores the entry the bot is actually trying to get rather than a
    # basis chosen here. It also sets the entry slippage rate, since the
    # chase is either modelled by the fill bar or charged as bps, never both.
    entry = smc_live.entry_rules(rules)
    exit_ = smc_live.exit_rules(rules)
    cost_bases = cost_bases_for(entry["fill"], exit_["fill"], exit_["tp1_resting_limit"])
    print(f"entry spec: fill={entry['fill']} require_ob_reclaim={entry['require_ob_reclaim']}", flush=True)
    print(f"exit spec:  fill={exit_['fill']} tp1_resting_limit={exit_['tp1_resting_limit']}", flush=True)
    print(f"slippage:   {leg_slippage(entry['fill'], exit_['fill'], exit_['tp1_resting_limit'])}", flush=True)

    # The watchlist is screened over the WHOLE index even when --tickers
    # narrows the simulation: the 40-name cap is a global ranking, and
    # computing it from a subset would hand that subset slots the live bot
    # never had.
    live_constraints = {}
    if not args.harness_basis:
        tf = rules.get("time_filter") or {}
        print("rebuilding the daily watchlist...", flush=True)
        live_constraints = {
            "force_close_same_day": True,
            "entry_window_et": (tf.get("earliest_entry_et"), tf.get("latest_entry_et")),
            "daily_watchlist": watchlist_from_rules(list(SP500_TICKERS), args.daily_dir, rules),
        }
        print(f"live basis: force-close on, entries {tf.get('earliest_entry_et')}-"
              f"{tf.get('latest_entry_et')} ET, "
              f"{len(live_constraints['daily_watchlist'])} watchlist dates", flush=True)
    else:
        print("harness basis: no same-day close, no entry window, full index", flush=True)

    # Build once per distinct slippage setting, not once per cost basis.
    candidates_by_slippage = {}
    for key, slippage in group_by_slippage(cost_bases).items():
        print(f"building candidates (slippage={key})...", flush=True)
        candidates_by_slippage[key] = build_smc_candidates(
            tickers,
            intraday_dir=args.intraday_dir,
            time_window_bars=rules["time_window_bars"],
            tp1_fraction=rules["tp1_fraction"],
            swing_window=rules["swing_window"],
            slippage_bps=slippage,
            entry_fill=entry["fill"],
            require_ob_reclaim=entry["require_ob_reclaim"],
            exit_fill=exit_["fill"],
            tp1_resting_limit=exit_["tp1_resting_limit"],
            early_retest=args.early_retest or entry["early_retest"],
            **live_constraints,
        )
        print(f"  {len(candidates_by_slippage[key])} candidates", flush=True)

    rows = []
    for basis in cost_bases:
        cands = candidates_by_slippage[slippage_key(basis["slippage"])]
        ref = cands[0][0] if cands else None

        tagged = {"entry_fill": entry["fill"], "require_ob_reclaim": entry["require_ob_reclaim"],
                  "exit_fill": exit_["fill"], "tp1_resting_limit": exit_["tp1_resting_limit"],
                  "live_constraints": bool(live_constraints), "basis": basis["name"]}

        full = run_one(cands, rules, args.initial_capital, basis["commission"])
        if full:
            rows.append({**tagged, "scope": "full_period", "fold": 0, **full})
            print(f"{basis['name']:20s} full: {full['trades']} trades  ret {full['ret_pct']}%  "
                  f"pf {full['pf']}  dd {full['max_dd_pct']}%", flush=True)

        for fold in folds:
            fit_end = align_tz(fold["fit_end"], ref)
            test_end = align_tz(fold["test_end"], ref)
            stats = run_one(cands, rules, args.initial_capital, basis["commission"], fit_end, test_end)
            if stats:
                rows.append({**tagged, "scope": "test", "fold": fold["fold"], **stats})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n===== FULL PERIOD =====")
    fp = df[df.scope == "full_period"]
    print(fp[["basis", "trades", "ret_pct", "max_dd_pct", "pf", "win_rate_pct"]].to_string(index=False))

    print("\n===== OUT-OF-SAMPLE (test folds summed) =====")
    oos = df[df.scope == "test"].groupby("basis", sort=False).agg(
        trades=("trades", "sum"), ret_pct=("ret_pct", "sum"),
        max_dd_pct=("max_dd_pct", "min"), pf=("pf", "mean"), win_rate_pct=("win_rate_pct", "mean"),
    ).round(3)
    print(oos.to_string())
    print(f"\nzero_cost keeps the {entry['fill']}/{exit_['fill']} FILLS and drops only commission "
          "and the residual spread, so the gap to the realistic rows is the cost overlay alone. "
          "It is not the old zero-cost basis, which also assumed unreachable fill PRICES.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
