"""Do the live bot and the backtest agree on which signals exist?

They do not, and this measures how often. Found on 2026-08-30 while
pre-flighting the live path: of 4 backtest signals over two sessions, the
live path reproduced 3.

The cause is that find_smc_long_trades is a stateful forward pass with no
warm-up convention. trend, last_swing_high, the swing-high pointer and
pending_bull_obs all accumulate from the first bar of whatever data it is
handed, so the same bar reaches different state depending on where the
data starts. smc_cycle hands it roughly 7 days (yfinance period="7d");
the backtest hands it the whole cache. Verified not to converge with more
history -- 7, 10, 14, 21, 30, 45 and 60 day windows all miss the same
signal that full history finds -- so this is not a warm-up length that
can simply be raised until it agrees.

Neither side is "correct". The backtest's state depends on where the
cache happens to begin, which is no less arbitrary than seven days. What
matters is the size and direction of the disagreement, because every
figure this repo publishes is computed on the backtest's side of it.

Replays the live path exactly as smc_cycle drives it -- same 7-day
window, same closed-bars-only rule, same require_ob_reclaim -- for every
in-window bar of every watchlist ticker over a date range, and compares
against build_smc_candidates over full history.

Usage:
    python -m trading_bot.cli.smc_signal_parity --start 2026-06-01 --end 2026-08-28
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
from pathlib import Path

import pandas as pd

from trading_bot import smc_live
from trading_bot.backtest.data import DAILY_DIR, load_intraday
from trading_bot.backtest.smc_engine import build_smc_candidates, watchlist_from_rules
from trading_bot.backtest.smc_signals import latest_entry_signal
from trading_bot.cli.smc_full_backtest import leg_slippage
from trading_bot.data.sp500_tickers import SP500_TICKERS

ET = "America/New_York"
# smc_cycle.get_5m_bars fetches yfinance period="7d".
LIVE_WINDOW_DAYS = 7
# Bar close + the cycle's 2-minute stagger; see engine.entry_window_mask.
ACTION_DELAY_MINUTES = 7


def live_signals_for(frame: pd.DataFrame, sessions: set, rules: dict, reclaim: bool,
                     window_days: int = LIVE_WINDOW_DAYS) -> set:
    """Every (bar timestamp) the live path would report a signal on.

    Deliberately re-runs the whole pass per bar rather than once: that is
    what smc_cycle does every five minutes, and the point here is
    fidelity to the live caller, not speed.
    """
    tf = rules["time_filter"]
    et = frame["date"].dt.tz_convert(ET)
    out = set()
    for i in frame.index[et.dt.date.isin(sessions)]:
        acted = (et.iloc[i] + pd.Timedelta(minutes=ACTION_DELAY_MINUTES)).strftime("%H:%M")
        if not (tf["earliest_entry_et"] <= acted <= tf["latest_entry_et"]):
            continue
        lo = frame["date"].iloc[i] - pd.Timedelta(days=window_days)
        win = frame[(frame["date"] > lo) & (frame.index <= i)]
        if len(win) < 10:
            continue
        bars = {
            "open": win["open"].tolist(), "high": win["high"].tolist(),
            "low": win["low"].tolist(), "close": win["close"].tolist(),
            "date": win["date"].tolist(),
        }
        if latest_entry_signal(bars, rules["time_window_bars"], rules["tp1_fraction"],
                               rules["swing_window"], False, reclaim) is not None:
            out.add(frame["date"].iloc[i])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intraday-dir", type=Path, default=Path("backtest_data/intraday_5m"))
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--rules", type=Path, default=Path("smc_rules.json"))
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--window-days", type=int, default=LIVE_WINDOW_DAYS,
                        help="what the live bot fetches; raising it does not make the two agree")
    parser.add_argument("--out", type=Path, default=Path("smc_signal_parity.csv"))
    args = parser.parse_args()

    rules = json.loads(args.rules.read_text())
    entry, exit_ = smc_live.entry_rules(rules), smc_live.exit_rules(rules)
    tf = rules["time_filter"]
    start = datetime.date.fromisoformat(args.start)
    end = datetime.date.fromisoformat(args.end)

    watchlist = watchlist_from_rules(list(SP500_TICKERS), args.daily_dir, rules)
    sessions = {d for d in watchlist if start <= d <= end and watchlist[d]}
    tickers = sorted(set().union(*(watchlist[d] for d in sessions))) if sessions else []
    if not tickers:
        print("no sessions in range")
        return 1
    print(f"{len(sessions)} sessions, {len(tickers)} distinct watchlist tickers", flush=True)

    candidates = build_smc_candidates(
        tickers, intraday_dir=args.intraday_dir,
        time_window_bars=rules["time_window_bars"], tp1_fraction=rules["tp1_fraction"],
        swing_window=rules["swing_window"],
        slippage_bps=leg_slippage(entry["fill"], exit_["fill"], exit_["tp1_resting_limit"]),
        entry_fill=entry["fill"], require_ob_reclaim=entry["require_ob_reclaim"],
        exit_fill=exit_["fill"], tp1_resting_limit=exit_["tp1_resting_limit"],
        force_close_same_day=True,
        entry_window_et=(tf["earliest_entry_et"], tf["latest_entry_et"]),
        daily_watchlist=watchlist,
    )

    frames = {t: load_intraday(t, args.intraday_dir) for t in tickers}
    backtest = set()
    for _, sym, tr in candidates:
        frame = frames.get(sym)
        if frame is None:
            continue
        ts = frame["date"].iloc[tr["signal_idx"]]
        if ts.tz_convert(ET).date() in sessions:
            backtest.add((sym, ts))
    print(f"backtest signals in range: {len(backtest)}", flush=True)

    live = set()
    for n, t in enumerate(tickers, 1):
        frame = frames.get(t)
        if frame is None or frame.empty:
            continue
        # Only replay days this ticker was actually on the watchlist for.
        mine = {d for d in sessions if t in watchlist[d]}
        for ts in live_signals_for(frame, mine, rules, entry["require_ob_reclaim"], args.window_days):
            live.add((t, ts))
        if n % 10 == 0:
            print(f"  replayed {n}/{len(tickers)} tickers, {len(live)} live signals so far", flush=True)

    both, only_live, only_bt = live & backtest, live - backtest, backtest - live
    rows = ([{"symbol": s, "bar": t, "seen": "both"} for s, t in sorted(both)]
            + [{"symbol": s, "bar": t, "seen": "live_only"} for s, t in sorted(only_live)]
            + [{"symbol": s, "bar": t, "seen": "backtest_only"} for s, t in sorted(only_bt)])
    pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"\n===== SIGNAL PARITY, {args.start} to {args.end} =====")
    print(f"  backtest signals : {len(backtest)}")
    print(f"  live signals     : {len(live)}")
    print(f"  agree            : {len(both)}")
    print(f"  backtest only    : {len(only_bt)}  (live misses these)")
    print(f"  live only        : {len(only_live)}  (backtest never scores these)")
    if backtest:
        print(f"\n  live reproduces {len(both) / len(backtest) * 100:.0f}% of backtest signals")
    if live:
        print(f"  {len(only_live) / len(live) * 100:.0f}% of live signals are absent from the backtest")
    by_month = collections.Counter(t.tz_convert(ET).strftime("%Y-%m") for _, t in only_bt)
    if by_month:
        print(f"\n  misses by month: {dict(sorted(by_month.items()))}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
