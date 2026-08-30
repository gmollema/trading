"""Is the SMC result a broad edge, or a handful of good days?

A walk-forward return says how much was made. It does not say whether it
came from a hundred small wins or from five days that broke the right
way, and those are different strategies to own. This answers that, on
whatever basis smc_rules.json currently configures.

The test is to remove the best N trading days and look at what is left.
If dropping five out of two hundred flips the sign, the return was those
five days -- the rest of the sample was noise around zero, and forward
performance depends on catching days like them again.

The worst-N-removed column is reported beside it and is a DIFFERENT
question: it measures what the tail costs, and it can only ever flatter a
positive result, because removing losers always does. The two get
confused easily -- this file's first draft printed only the worst-removed
column under a heading asking about concentration, which could not have
answered it. Both are shown so the asymmetry is visible.

What it found on 2026-08-30, over 236 trading days on the live basis
(flatten daily, tiered commission, all legs slipped):

    days cut   best removed
           0        +2.325%
           3        +0.404%
           5        -0.344%

Five days out of 236 carry the whole result. The same test on the
overnight-holding variant, which returns roughly triple, collapses the
same way (+5.692% -> -0.680% at five days cut) -- so the extra return
there was five days of luck rather than a better strategy, and that is
why it was not adopted.

Usage:
    python -m trading_bot.cli.smc_concentration
"""

from __future__ import annotations

import argparse
import collections
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
from trading_bot.cli.smc_full_backtest import DEFAULT_INITIAL_CAPITAL, TIERED, leg_slippage
from trading_bot.data.sp500_tickers import SP500_TICKERS

DEFAULT_DROPS = (0, 1, 3, 5, 10)


def daily_pnl(result: dict) -> pd.DataFrame:
    """Realised P&L per calendar date, FIFO-matched from the fill ladder.

    simulate_smc_portfolio emits individual fills, so a TP1 partial and
    the exit closing the remainder are separate rows belonging to one
    round trip; lots are matched per symbol so a partial is not scored as
    a completed trade. Attributed to the date of the SELL, which is when
    the money is actually made or lost.
    """
    lots: dict[str, list[dict]] = collections.defaultdict(list)
    rows = []
    for t in result["trades"]:
        day = pd.Timestamp(t["timestamp_iso"]).date()
        qty, px = float(t["size"]), float(t["fill_price"])
        if t["side"] == "BUY":
            lots[t["symbol"]].append({"px": px, "qty": qty})
            continue
        remaining, pnl = qty, 0.0
        while remaining > 1e-9 and lots[t["symbol"]]:
            lot = lots[t["symbol"]][0]
            take = min(remaining, lot["qty"])
            pnl += (px - lot["px"]) * take
            lot["qty"] -= take
            remaining -= take
            if lot["qty"] <= 1e-9:
                lots[t["symbol"]].pop(0)
        rows.append({"date": day, "symbol": t["symbol"], "pnl": pnl, "reason": t.get("reason")})
    return pd.DataFrame(rows)


def dependence(pnl: pd.DataFrame, initial_capital: float, drops=DEFAULT_DROPS) -> pd.DataFrame:
    """Total return with the best N days removed, and with the worst N.

      best removed   the robustness test. If it turns negative, the result
          was a few good days rather than an edge.
      worst removed  what the tail costs. Cannot flip a positive result,
          so it measures exposure and not quality.

    Days are aggregated before ranking: two trades closing on one date are
    one day of risk, not two.
    """
    by_day = pnl.groupby("date").pnl.sum().sort_values()
    out = []
    for n in drops:
        worst_removed = by_day.iloc[n:] if n else by_day
        best_removed = by_day.iloc[:len(by_day) - n] if n else by_day
        out.append({
            "n": n,
            "best_removed_ret": best_removed.sum() / initial_capital * 100,
            "worst_removed_ret": worst_removed.sum() / initial_capital * 100,
            "days_left": len(by_day) - n,
        })
    return pd.DataFrame(out)


def top_days(pnl: pd.DataFrame, initial_capital: float, n: int = 5) -> pd.DataFrame:
    by_day = pnl.groupby("date").pnl.sum()
    best = by_day.nlargest(n).rename("pnl").reset_index()
    best["pct_of_capital"] = best.pnl / initial_capital * 100
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intraday-dir", type=Path, default=Path("backtest_data/intraday_5m"))
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--rules", type=Path, default=Path("smc_rules.json"))
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("smc_concentration_daily_pnl.csv"))
    args = parser.parse_args()

    rules = json.loads(args.rules.read_text())
    entry, exit_ = smc_live.entry_rules(rules), smc_live.exit_rules(rules)
    tf = rules.get("time_filter") or {}
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)

    watchlist = watchlist_from_rules(list(SP500_TICKERS), args.daily_dir, rules)
    candidates = build_smc_candidates(
        tickers, intraday_dir=args.intraday_dir,
        time_window_bars=rules["time_window_bars"], tp1_fraction=rules["tp1_fraction"],
        swing_window=rules["swing_window"],
        slippage_bps=leg_slippage(entry["fill"], exit_["fill"], exit_["tp1_resting_limit"]),
        entry_fill=entry["fill"], require_ob_reclaim=entry["require_ob_reclaim"],
        exit_fill=exit_["fill"], tp1_resting_limit=exit_["tp1_resting_limit"],
        force_close_same_day=True,
        entry_window_et=(tf.get("earliest_entry_et"), tf.get("latest_entry_et")),
        daily_watchlist=watchlist,
    )
    risk = rules["risk"]
    result = simulate_smc_portfolio(
        candidates, args.initial_capital,
        risk_pct=risk["max_risk_per_trade_pct"],
        max_position_pct=risk["max_position_size_pct_of_portfolio"],
        max_concurrent_positions=risk["max_concurrent_positions"],
        commission_per_share=TIERED[0], commission_min=TIERED[1],
    )

    pnl = daily_pnl(result)
    if pnl.empty:
        print("no closed trades")
        return 1
    pnl.to_csv(args.out, index=False)

    by_day = pnl.groupby("date").pnl.sum()
    total = by_day.sum() / args.initial_capital * 100
    print(f"{len(candidates)} candidates, {len(pnl)} fills, {len(by_day)} trading days with a close")
    print(f"basis: entry={entry['fill']} exit={exit_['fill']} reclaim={entry['require_ob_reclaim']}, "
          f"tiered commission, all legs slipped")
    print(f"total: {total:+.3f}% on ${args.initial_capital:,.0f}\n")

    print(f"  {'days cut':>8s} {'best removed':>14s} {'worst removed':>15s}")
    for _, r in dependence(pnl, args.initial_capital).iterrows():
        print(f"  {int(r.n):>8d} {r.best_removed_ret:>13.3f}% {r.worst_removed_ret:>14.3f}%")

    print("\n  best days:")
    for _, r in top_days(pnl, args.initial_capital).iterrows():
        print(f"    {r.date}  {r.pnl:>+9,.0f}  {r.pct_of_capital:>+6.2f}% of capital")

    print("\n'best removed' turning negative means the return was those days, not an edge.")
    print("'worst removed' only ever flatters a positive result -- it measures the tail.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
