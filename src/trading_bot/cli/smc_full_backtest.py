"""Full SMC backtest across cost bases, in-sample and walk-forward.

Every SMC figure in this repo predating 2026-08-28 was produced on the
engine's zero-cost defaults (slippage_bps 0.0 for every leg,
commission_per_share None). That is not a survivable basis for this
strategy: the mean trade returns ~32 bps of entry notional and the median
stop sits 17.3 bps from entry, while live market-order fills measured 48.5
and 79.4 bps adverse on TP1. Costs here are the same order as the edge, so
they decide the answer rather than trimming it.

So this runs the SAME data across several cost bases side by side. The
zero-cost row is kept only for comparability with the historical numbers,
NOT because any order type can achieve it -- see smc_fill_model, which
found a limit at ob_high fills just 31-42% of signals and almost only when
the setup is failing.

Slippage is applied per leg rather than as one rate, because the legs
differ by order type. The stop rests at IBKR as a real StopOrder and
measured 0-5.9 bps over 7 live fills; entries and non-stop exits are
MarketOrders sent after a level is touched, and the two TP1 fills measured
48.5 and 79.4. ENTRY_SLIPPAGE_BPS is an assumption, not a measurement --
it borrows TP1's figure on the grounds that it is the same order type
against the same book. smc_cycle now logs entry_slippage_bps per fill, so
it can be replaced with a measured number once enough entries accumulate.

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

from trading_bot.backtest.smc_engine import build_smc_candidates, simulate_smc_portfolio
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

MEASURED_SLIPPAGE = {
    "entry": ENTRY_SLIPPAGE_BPS,
    "stop": STOP_SLIPPAGE_BPS,
    "tp1": MARKET_LEG_SLIPPAGE_BPS,
    "new_high_exit": MARKET_LEG_SLIPPAGE_BPS,
    "same_day_force_close": MARKET_LEG_SLIPPAGE_BPS,
    "end_of_data": 0.0,
}

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
    parser.add_argument("--rules", type=Path, default=Path("smc_rules.json"))
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

    # Build once per distinct slippage setting, not once per cost basis.
    candidates_by_slippage = {}
    for key, slippage in group_by_slippage(COST_BASES).items():
        print(f"building candidates (slippage={key})...", flush=True)
        candidates_by_slippage[key] = build_smc_candidates(
            tickers,
            intraday_dir=args.intraday_dir,
            time_window_bars=rules["time_window_bars"],
            tp1_fraction=rules["tp1_fraction"],
            swing_window=rules["swing_window"],
            slippage_bps=slippage,
        )
        print(f"  {len(candidates_by_slippage[key])} candidates", flush=True)

    rows = []
    for basis in COST_BASES:
        cands = candidates_by_slippage[slippage_key(basis["slippage"])]
        ref = cands[0][0] if cands else None

        full = run_one(cands, rules, args.initial_capital, basis["commission"])
        if full:
            rows.append({"basis": basis["name"], "scope": "full_period", "fold": 0, **full})
            print(f"{basis['name']:20s} full: {full['trades']} trades  ret {full['ret_pct']}%  "
                  f"pf {full['pf']}  dd {full['max_dd_pct']}%", flush=True)

        for fold in folds:
            fit_end = align_tz(fold["fit_end"], ref)
            test_end = align_tz(fold["test_end"], ref)
            stats = run_one(cands, rules, args.initial_capital, basis["commission"], fit_end, test_end)
            if stats:
                rows.append({"basis": basis["name"], "scope": "test", "fold": fold["fold"], **stats})

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
    print("\nzero_cost is for comparability with the repo's historical figures only; "
          "no order type achieves it (see smc_fill_model).")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
