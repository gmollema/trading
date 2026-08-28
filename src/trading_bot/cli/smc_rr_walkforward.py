"""Walk-forward sweep of a minimum-R:R entry filter for the SMC strategy.

Asks whether refusing signals whose TP1 sits too close to the stop --
relative to the risk taken -- improves out-of-sample results. It does not:
the first run of this sweep (2026-08-28, 3 expanding folds over the
2025-07-15 cache) found every threshold monotonically WORSE than no filter
at all out of sample, with drawdown flat across the grid. The reason is in
the per-bucket numbers: mean return per trade is nearly constant across
R:R, because a low R:R means TP1 is close and therefore hit ~79% of the
time, which exactly offsets the poor ratio. R:R-to-TP1 carries almost no
expectancy information, so filtering on it just discards positive-
expectancy trades. Signals with no TP1 at all are the single best cohort
(pure runners, nothing capping them), so dropping those hurts most.

Kept as a committed harness rather than deleted with its negative result:
the repo had no walk-forward script in version control (the one behind
smc_derisk_walkforward_results_v5.csv was lost and reconstructed from its
own output header), and the fold construction, round-trip pairing and
metrics here are the reusable parts of any future sweep.

R:R is measured at signal time from entry_price, initial_stop_price and
tp1_price -- all known before the bot commits, so no lookahead.

Usage:
    python -m trading_bot.cli.smc_rr_walkforward \
        --intraday-dir backtest_data/intraday_5m_merged_2026-08-27 \
        --out smc_min_rr_walkforward_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trading_bot.backtest.smc_engine import build_smc_candidates, simulate_smc_portfolio
from trading_bot.data.sp500_tickers import SP500_TICKERS

DEFAULT_INTRADAY_DIR = Path("backtest_data/intraday_5m_merged_2026-08-27")
DEFAULT_INITIAL_CAPITAL = 100_000.0

# Anchor and boundaries matching smc_derisk_walkforward_results_v5.csv's
# expanding-window convention, so runs stay comparable with it.
DEFAULT_FIT_START = "2025-07-15 14:15:00-04:00"
DEFAULT_BOUNDARIES = [
    "2026-03-14 16:00:00-04:00",
    "2026-05-14 16:00:00-04:00",
    "2026-07-14 16:00:00-04:00",
    "2026-08-26 16:00:00-04:00",
]
DEFAULT_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def signal_rr(trade: dict) -> float | None:
    """Reward-to-risk to TP1, or None when it is undefined.

    None means the signal has no TP1 (no bearish order block resting above
    entry) -- a runner with no target, not a bad trade. Callers must decide
    explicitly whether to keep those; see filter_by_rr's drop_no_tp1.
    """
    entry = trade.get("entry_price")
    stop = trade.get("initial_stop_price")
    tp1 = trade.get("tp1_price")
    if entry is None or stop is None or tp1 is None:
        return None
    if entry <= stop:
        return None
    return (tp1 - entry) / (entry - stop)


def filter_by_rr(candidates: list[tuple], min_rr: float, drop_no_tp1: bool) -> list[tuple]:
    """Keep candidates meeting `min_rr`; no-TP1 signals per `drop_no_tp1`."""
    kept = []
    for cand in candidates:
        rr = signal_rr(cand[2])
        if rr is None:
            if not drop_no_tp1:
                kept.append(cand)
        elif rr >= min_rr:
            kept.append(cand)
    return kept


def expanding_folds(fit_start, boundaries: list) -> list[dict]:
    """Expanding-window folds: fit always starts at `fit_start`, and each
    fold tests on the span between consecutive boundaries.

    With boundaries [b0, b1, b2], fold 1 fits fit_start->b0 and tests
    b0->b1; fold 2 fits fit_start->b1 and tests b1->b2. So n boundaries
    yield n-1 folds, and every test window is strictly after its own fit
    window.
    """
    fit_start = pd.Timestamp(fit_start)
    bounds = [pd.Timestamp(b) for b in boundaries]
    return [
        {"fold": i + 1, "fit_start": fit_start, "fit_end": bounds[i], "test_end": bounds[i + 1]}
        for i in range(len(bounds) - 1)
    ]


def round_trip_pnls(trades: list[dict]) -> list[float]:
    """Realised P&L per round trip from a flat list of fills.

    simulate_smc_portfolio returns individual fills, so a TP1 partial and
    the exit that closes the remainder are separate rows; both belong to
    one round trip. Lots are matched FIFO per symbol and a lot is only
    booked once its quantity is fully sold, which keeps a partial TP1 from
    being counted as a completed winner on its own.
    """
    pnls: list[float] = []
    lots: dict[str, list[dict]] = {}
    for row in trades:
        symbol = row["symbol"]
        qty = float(row["size"])
        price = float(row["fill_price"])
        if row["side"] == "BUY":
            lots.setdefault(symbol, []).append({"px": price, "qty": qty, "pnl": 0.0})
            continue
        queue = lots.get(symbol) or []
        while qty > 1e-9 and queue:
            lot = queue[0]
            take = min(qty, lot["qty"])
            lot["pnl"] += (price - lot["px"]) * take
            lot["qty"] -= take
            qty -= take
            if lot["qty"] <= 1e-9:
                pnls.append(lot["pnl"])
                queue.pop(0)
    return pnls


def summarize(result: dict, initial_capital: float) -> dict:
    """Return/drawdown from the equity curve, PF/win rate from round trips."""
    curve = pd.Series([float(p["equity"]) for p in result.get("equity_curve", [])])
    if curve.empty:
        return {"trades": 0, "ret_pct": 0.0, "max_dd_pct": 0.0,
                "pf": float("nan"), "win_rate_pct": float("nan")}

    pnls = pd.Series(round_trip_pnls(result.get("trades", [])), dtype=float)
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    return {
        "trades": len(pnls),
        "ret_pct": round((curve.iloc[-1] / initial_capital - 1) * 100, 4),
        "max_dd_pct": round(((curve / curve.cummax()) - 1).min() * 100, 4),
        "pf": round(wins / losses, 3) if losses > 0 else float("inf"),
        "win_rate_pct": round((pnls > 0).mean() * 100, 2) if len(pnls) else float("nan"),
    }


def align_tz(ts: pd.Timestamp, reference: pd.Timestamp) -> pd.Timestamp:
    """Put `ts` on the same tz-awareness footing as `reference`.

    Candidate dates are always tz-aware (they come from IBKR bars), while a
    fold boundary typed on the command line usually is not -- comparing the
    two raises "Cannot compare tz-naive and tz-aware timestamps". Rather
    than make the caller match the cache's timezone by hand, adopt the
    reference's: a naive boundary is read as a wall-clock time in the
    candidates' own timezone, which is what someone writing --fit-start
    2025-07-15 means.
    """
    if reference is None:
        return ts
    if reference.tz is not None and ts.tz is None:
        return ts.tz_localize(reference.tz)
    if reference.tz is None and ts.tz is not None:
        return ts.tz_localize(None)
    return ts


def _window(candidates: list[tuple], lo, hi) -> list[tuple]:
    return [c for c in candidates if lo <= c[0] < hi]


def run_sweep(
    candidates: list[tuple],
    folds: list[dict],
    thresholds: list[float],
    rules: dict,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    drop_no_tp1_options: tuple[bool, ...] = (False, True),
    verbose: bool = True,
) -> pd.DataFrame:
    """Simulate every (threshold, drop_no_tp1, fold, phase) combination.

    `candidates` is built once by the caller and reused across the whole
    grid -- build_smc_candidates is the expensive step and is independent
    of every dimension swept here.
    """
    risk = rules["risk"]
    reference = candidates[0][0] if candidates else None
    rows = []
    for drop_no_tp1 in drop_no_tp1_options:
        for threshold in thresholds:
            kept = filter_by_rr(candidates, threshold, drop_no_tp1)
            for fold in folds:
                fit_start = align_tz(fold["fit_start"], reference)
                fit_end = align_tz(fold["fit_end"], reference)
                test_end = align_tz(fold["test_end"], reference)
                phases = (
                    ("fit", fit_start, fit_end),
                    ("test", fit_end, test_end),
                )
                for phase, lo, hi in phases:
                    window = _window(kept, lo, hi)
                    if not window:
                        continue
                    result = simulate_smc_portfolio(
                        window,
                        initial_capital,
                        risk_pct=risk["max_risk_per_trade_pct"],
                        max_position_pct=risk["max_position_size_pct_of_portfolio"],
                        max_concurrent_positions=risk["max_concurrent_positions"],
                    )
                    stats = summarize(result, initial_capital)
                    rows.append({"min_rr": threshold, "drop_no_tp1": drop_no_tp1,
                                 "fold": fold["fold"], "phase": phase,
                                 "signals": len(window), **stats})
                    if verbose:
                        print(f"  rr>={threshold} drop_no_tp1={drop_no_tp1} "
                              f"fold{fold['fold']} {phase}: {stats['trades']} trades "
                              f"ret {stats['ret_pct']}% dd {stats['max_dd_pct']}% "
                              f"pf {stats['pf']}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intraday-dir", type=Path, default=DEFAULT_INTRADAY_DIR)
    parser.add_argument("--rules", type=Path, default=Path("smc_rules.json"))
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fit-start", type=str, default=DEFAULT_FIT_START)
    parser.add_argument(
        "--boundaries", type=str, default=",".join(DEFAULT_BOUNDARIES),
        help="comma-separated fold boundaries; n boundaries give n-1 expanding folds",
    )
    parser.add_argument(
        "--thresholds", type=str, default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="comma-separated minimum R:R values to sweep",
    )
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated override list")
    parser.add_argument("--out", type=Path, default=Path("smc_min_rr_walkforward_results.csv"))
    args = parser.parse_args()

    if not args.intraday_dir.is_dir():
        print(f"error: no such intraday dir: {args.intraday_dir}")
        return 2

    rules = json.loads(args.rules.read_text())
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)
    folds = expanding_folds(args.fit_start, [b.strip() for b in args.boundaries.split(",")])
    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(f"building candidates over {len(tickers)} tickers (once, reused across the grid)...")
    candidates = build_smc_candidates(
        tickers,
        intraday_dir=args.intraday_dir,
        time_window_bars=rules["time_window_bars"],
        tp1_fraction=rules["tp1_fraction"],
        swing_window=rules["swing_window"],
    )
    no_tp1 = sum(1 for c in candidates if signal_rr(c[2]) is None)
    print(f"{len(candidates)} candidates; {no_tp1} ({no_tp1 / max(len(candidates), 1) * 100:.1f}%) "
          "have no TP1 (undefined R:R)")

    df = run_sweep(candidates, folds, thresholds, rules, args.initial_capital)
    df.to_csv(args.out, index=False)

    test = df[df.phase == "test"]
    if not test.empty:
        print("\n===== OUT-OF-SAMPLE (test folds only) =====")
        summary = test.pivot_table(
            index=["drop_no_tp1", "min_rr"],
            values=["trades", "ret_pct", "max_dd_pct", "pf", "win_rate_pct"],
            aggfunc={"trades": "sum", "ret_pct": "sum", "max_dd_pct": "min",
                     "pf": "mean", "win_rate_pct": "mean"},
        )
        print(summary[["trades", "ret_pct", "max_dd_pct", "pf", "win_rate_pct"]].round(3).to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
