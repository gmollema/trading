"""Compare SMC entry specifications that an order can actually reach.

The strategy's entry was specified as "fill at the order block's high the
moment price retests it". Nothing executes that. smc_fill_model (2026-08-28)
showed a limit resting at that level fills 31-42% of signals and almost
exclusively when the setup is failing -- filled trades returned -0.05% mean
at an 18% win rate, the ones it declined +0.49% at 62% -- while a market
order sent once the touch is known measured 48-79 bps of slippage on the
comparable leg, against a 17.3 bps median stop distance. The entry price
every figure in this repo was scored on is not a property of the market;
it is an artifact of filling at a trigger.

So this respecifies the entry as something a market order gets, and
measures what is left. Two axes:

  entry_fill: where the fill lands. "level" is the old unreachable spec,
      kept as the reference row. "next_open" and "next_high" decide the
      signal on the CLOSED touching bar and fill on the next one, which is
      what an order sent at that close achieves; they bracket where inside
      that bar the fill lands (its first print, and the worst price it ever
      offered). The live cycle fires ~2 minutes into the fill bar, so its
      true fill sits between them.

  require_ob_reclaim: whether the touching bar must also CLOSE back above
      the OB high. This is the cohort split smc_fill_model found, turned
      into a rule the bot can apply -- it is decidable at the bar's close,
      before the order goes in.

Entry slippage is set per spec, not shared, because the same 64 bps cannot
be charged twice. That figure is (fill - level)/level for a market order
sent up to a full bar after the level was touched: it IS the chase. Under
the reachable specs the chase is read off the bars instead, so what remains
on top is spread and impact, for which the best evidence in this repo is
the stop leg -- a market execution with no chase, measured 0-5.9 bps over 7
live fills, median 2.0.

The exit legs are NOT respecified here and still carry the full market-leg
rate in the realistic bases. TP1 and the new-high exit have exactly the
same defect as the old entry -- they fill at a level the bot only learns
about after the bar closes -- so those rows stay conservative on the exits
while the entry question is settled. That is deliberately left frozen even
though the exits HAVE since been respecified (smc_exit_spec): this module
holds every other axis fixed at what it was when its results file was
written, so re-running it reproduces those numbers rather than silently
answering a different question.

Usage:
    python -m trading_bot.cli.smc_entry_spec \
        --intraday-dir backtest_data/intraday_5m \
        --out smc_entry_spec_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trading_bot.backtest.smc_engine import build_smc_candidates, simulate_smc_portfolio
from trading_bot.cli.smc_full_backtest import (
    COST_BASES,
    DEFAULT_INITIAL_CAPITAL,
    MARKET_LEG_SLIPPAGE_BPS,
    RESIDUAL_ENTRY_SLIPPAGE_BPS,
)
from trading_bot.cli.smc_rr_walkforward import (
    DEFAULT_BOUNDARIES,
    DEFAULT_FIT_START,
    align_tz,
    expanding_folds,
    summarize,
)
from trading_bot.data.sp500_tickers import SP500_TICKERS

ENTRY_SPECS = [
    {"name": "level", "entry_fill": "level", "reclaim": False,
     "entry_slippage_bps": MARKET_LEG_SLIPPAGE_BPS},
    {"name": "level_reclaim", "entry_fill": "level", "reclaim": True,
     "entry_slippage_bps": MARKET_LEG_SLIPPAGE_BPS},
    {"name": "next_open", "entry_fill": "next_open", "reclaim": False,
     "entry_slippage_bps": RESIDUAL_ENTRY_SLIPPAGE_BPS},
    {"name": "next_open_reclaim", "entry_fill": "next_open", "reclaim": True,
     "entry_slippage_bps": RESIDUAL_ENTRY_SLIPPAGE_BPS},
    {"name": "next_high", "entry_fill": "next_high", "reclaim": False,
     "entry_slippage_bps": RESIDUAL_ENTRY_SLIPPAGE_BPS},
    {"name": "next_high_reclaim", "entry_fill": "next_high", "reclaim": True,
     "entry_slippage_bps": RESIDUAL_ENTRY_SLIPPAGE_BPS},
]

# Printed in full; the CSV carries every basis in COST_BASES.
REPORTED_BASES = ("zero_cost", "commission_tiered", "realistic_tiered")


def spec_slippage(basis_slippage: dict | None, spec: dict) -> dict | None:
    """The basis's slippage with this spec's entry rate substituted in.

    A basis with no slippage at all keeps none: zero_cost exists to isolate
    the fill respec from the cost overlay, and charging it an entry rate
    would defeat that.
    """
    if basis_slippage is None:
        return None
    return {**basis_slippage, "entry": spec["entry_slippage_bps"]}


def run_one(candidates, rules, initial_capital, commission, lo=None, hi=None) -> dict | None:
    """Simulate one (spec, basis) over an optional [lo, hi) entry-date window."""
    window = candidates
    if lo is not None and hi is not None:
        window = [c for c in candidates if lo <= c[0] < hi]
    if not window:
        return None
    risk = rules["risk"]
    per_share, minimum = commission if commission else (None, 1.0)
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


def build_all(tickers, intraday_dir: Path, rules: dict, specs=ENTRY_SPECS, bases=COST_BASES) -> dict:
    """Candidates keyed by (spec name, slippage identity).

    Signal generation is the expensive step and depends only on the entry
    spec and the slippage dict, never on commission -- which is applied at
    portfolio level and cannot change which signals exist. So bases sharing
    a slippage setting share one build.
    """
    out = {}
    for spec in specs:
        seen = {}
        for basis in bases:
            slippage = spec_slippage(basis["slippage"], spec)
            key = json.dumps(slippage, sort_keys=True)
            if key in seen:
                continue
            print(f"building {spec['name']:18s} slippage={key}", flush=True)
            seen[key] = build_smc_candidates(
                tickers,
                intraday_dir=intraday_dir,
                time_window_bars=rules["time_window_bars"],
                tp1_fraction=rules["tp1_fraction"],
                swing_window=rules["swing_window"],
                slippage_bps=slippage,
                entry_fill=spec["entry_fill"],
                require_ob_reclaim=spec["reclaim"],
                # Frozen at "level" -- see the note above about holding
                # every other axis where it was when the results file was
                # written. This used to come from the default.
                exit_fill="level",
            )
            print(f"  {len(seen[key])} candidates", flush=True)
        for basis in bases:
            key = json.dumps(spec_slippage(basis["slippage"], spec), sort_keys=True)
            out[(spec["name"], basis["name"])] = seen[key]
    return out


def sweep(candidates_by, specs, bases, folds, rules, initial_capital) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for basis in bases:
            cands = candidates_by[(spec["name"], basis["name"])]
            ref = cands[0][0] if cands else None
            common = {"entry_spec": spec["name"], "basis": basis["name"]}

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
    """Test folds summed per (spec, basis) -- the out-of-sample read."""
    return df[df.scope == "test"].groupby(["entry_spec", "basis"], sort=False).agg(
        trades=("trades", "sum"), ret_pct=("ret_pct", "sum"),
        worst_dd_pct=("max_dd_pct", "min"), pf=("pf", "mean"), win_rate_pct=("win_rate_pct", "mean"),
    ).round(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intraday-dir", type=Path, default=Path("backtest_data/intraday_5m"))
    parser.add_argument("--rules", type=Path, default=Path("smc_rules.json"))
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fit-start", type=str, default=DEFAULT_FIT_START)
    parser.add_argument("--boundaries", type=str, default=",".join(DEFAULT_BOUNDARIES))
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated override list")
    parser.add_argument("--out", type=Path, default=Path("smc_entry_spec_results.csv"))
    args = parser.parse_args()

    if not args.intraday_dir.is_dir():
        print(f"error: no such intraday dir: {args.intraday_dir}")
        return 2

    rules = json.loads(args.rules.read_text())
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)
    folds = expanding_folds(args.fit_start, [b.strip() for b in args.boundaries.split(",")])

    candidates_by = build_all(tickers, args.intraday_dir, rules)
    df = sweep(candidates_by, ENTRY_SPECS, COST_BASES, folds, rules, args.initial_capital)
    if df.empty:
        print("no results")
        return 1
    df.to_csv(args.out, index=False)

    shown = df[df.basis.isin(REPORTED_BASES)]
    print("\n===== FULL PERIOD =====")
    fp = shown[shown.scope == "full_period"]
    print(fp[["entry_spec", "basis", "signals", "trades", "ret_pct",
              "max_dd_pct", "pf", "win_rate_pct"]].to_string(index=False))

    print("\n===== OUT-OF-SAMPLE (test folds summed) =====")
    print(oos_table(shown).to_string())

    print("\nlevel is the historical spec and is unreachable by any order type "
          "(smc_fill_model); it is here as the reference the other rows replace.")
    print("next_open and next_high bracket where in the fill bar a market order "
          "sent at the signal bar's close actually lands.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
