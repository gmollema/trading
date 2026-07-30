"""Pure/live-logic helpers for the UT Bot paper-trading cycle
(cli/ut_bot_cycle.py).

Kept separate from the scheduled entrypoint so everything here is
importable and unit-testable without a broker connection and without
cycle-style import-time market gates -- same split as smc_live.py /
cli/smc_cycle.py, for the same reason (cli/ut_bot_cycle.py's own fast
exit gate is guarded inside `if __name__ == "__main__":`, so importing
this module never triggers it). Signal logic itself lives in
backtest/ut_bot_signals.py (latest_long_entry_signal / latest_sell_signal)
-- the live bot re-runs the exact code the backtest validated; this
module only adds the live plumbing around it: config, per-pair state
file constants, bar-frame conversion, and position sizing.

IMPORTANT, read before enabling more than USDJPY: ut_bot_rules.json's
"pairs" block carries a `validated` flag per pair. Only USDJPY's
combination of parameters has actually cleared a walk-forward validation
(see ut_bot_engine.py's module history) -- EURUSD and USDCAD are included
because they had the next-best backtested returns, but that edge only
showed up in a narrow corner of their own parameter grids, which is
exactly the pattern that turned out to be curve-fitting for every OTHER
pair tested besides USDJPY. Trading them live is a real bet that their
narrow-corner result is genuine, not backtest noise -- it hasn't been
confirmed the way USDJPY's has.

Also worth knowing before running this for real: PORTFOLIO_VALUE_USD is
the same env var the gap-and-go and SMC bots already read, and each bot
sizes new positions against that FULL value independently -- adding a
third bot on the same env var compounds this codebase's existing (pre-
existing, not introduced here) over-allocation risk if all three end up
holding positions at once.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

UT_BOT_RULES_PATH = Path("ut_bot_rules.json")
UT_BOT_POSITIONS_PATH = Path("ut_bot_open_positions.json")
UT_BOT_TRADES_CSV_PATH = Path("ut_bot_trades.csv")
UT_BOT_SAFETY_LOG_PATH = Path("ut-bot-safety-check-log.json")

TRADES_CSV_HEADER = ["timestamp_iso", "symbol", "side", "size", "fill_price", "order_id", "status", "reason"]


def load_ut_bot_rules(path: Path = UT_BOT_RULES_PATH) -> dict:
    return json.loads(path.read_text())


def get_market_status(now_et: datetime) -> str:
    """FX trades ~24/5 on IDEALPRO (Sunday ~17:00 ET open through Friday
    ~17:00 ET close) -- no intraday force-close the way day-trading
    equities need (UT Bot holds positions across many bars/days by
    design), so this only distinguishes weekend-closed from ok, unlike
    cycle.py/smc_live.py's richer manage_only/force_close states."""
    weekday = now_et.weekday()  # Monday=0 .. Sunday=6
    hm = now_et.strftime("%H:%M")
    if weekday == 5:  # Saturday: closed all day
        return "closed"
    if weekday == 6 and hm < "17:00":  # Sunday before reopen
        return "closed"
    if weekday == 4 and hm >= "17:00":  # Friday after close
        return "closed"
    return "ok"


def bars_frame_to_dict(bars: pd.DataFrame) -> dict:
    """Convert a yfinance-style OHLC frame (High/Low/Close columns,
    DatetimeIndex) into the plain-lists bars dict ut_bot_signals expects."""
    return {
        "high": bars["High"].astype(float).tolist(),
        "low": bars["Low"].astype(float).tolist(),
        "close": bars["Close"].astype(float).tolist(),
        "date": list(bars.index),
    }
