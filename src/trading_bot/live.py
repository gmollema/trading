"""Pure/live-logic helpers for the gap-and-go paper-trading cycle
(cli/cycle.py). The counterpart to smc_live.py, which does the same job
for the SMC bot.

Kept separate from the scheduled entrypoint so everything here is
importable and unit-testable without a broker connection -- and, more
pointedly, without the entrypoint. backtest/engine.py and
backtest/portfolio.py both need functions that used to live in cycle.py,
so importing them dragged ib_async, yfinance and numpy into the backtest
behind a market-status gate that called sys.exit() outside trading hours.
Two production modules were wrapping their imports in
`patch("sys.exit")` to get around it.

Deliberately stdlib-only, unlike smc_live (which needs pandas for
bars_frame_to_dict). cli/cycle.py imports this ABOVE its fast time gate,
so anything heavy in here would cost every no-op cycle the sub-second
exit the gate exists to provide.

All gap-and-go state files are separate from the SMC bot's
(smc_watchlist.txt / smc_open_positions.json / smc_trades.csv), so either
bot can be enabled, disabled, or inspected without touching the other.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from trading_bot.util.market_hours import MARKET_CLOSE_ET, MARKET_OPEN_ET

RULES_PATH = Path("rules.json")
WATCHLIST_PATH = Path("watchlist.txt")
POSITIONS_PATH = Path("open_positions.json")
TRADES_CSV_PATH = Path("trades.csv")
SAFETY_LOG_PATH = Path("safety-check-log.json")


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def get_market_status(now_et: datetime, rules: dict) -> str:
    """Returns one of: weekend, too_early, closed, manage_only, force_close, ok.

    Boundaries come from rules.json's time_filter. They used to be
    hardcoded here -- all four of them -- which made that whole config
    block decorative: it happened to state the same values the code used,
    so it read as authoritative while changing it did nothing at all.

    The only times still fixed in code are the session's own
    (util.market_hours), because those are a fact about the exchange
    rather than a choice this strategy makes. An earliest_entry_et before
    the open therefore means "from the open", not an error.

    The old floor also sat ABOVE earliest_entry_et at 10:00, and
    "too_early" exits the entire cycle rather than just declining to
    enter -- so a position that survived a failed force-close went
    unmanaged, with no stop repair and no exit, through the first half
    hour of the session.
    """
    if now_et.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return "weekend"

    tf = rules["time_filter"]
    earliest_entry = tf["earliest_entry_et"]
    latest_entry = tf["latest_entry_et"]
    force_close = tf["force_close_et"]

    hm = now_et.strftime("%H:%M")

    if hm < MARKET_OPEN_ET:
        return "too_early"
    if hm > MARKET_CLOSE_ET:
        return "closed"
    if hm < earliest_entry:
        return "manage_only"
    if hm < latest_entry:
        return "ok"
    if hm < force_close:
        return "manage_only"
    if hm <= MARKET_CLOSE_ET:
        return "force_close"
    return "closed"  # unreachable, defensive fallback


def read_watchlist(path: Path = WATCHLIST_PATH) -> list[str]:
    if not path.exists():
        return []
    tickers = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ticker = line.split("#")[0].strip()
        if ticker:
            tickers.append(ticker)
    return tickers


def compute_swing_lows(bars_5m) -> list[float]:
    """A bar's low is a swing low if it's lower than the 2 bars before AND
    the 2 bars after it. Returns swing lows in chronological order.

    Takes anything with a "Low" column exposing .tolist() -- a pandas
    frame in every real caller -- rather than importing pandas to say so,
    which keeps this module cheap enough for cli/cycle.py's fast gate.
    """
    lows = bars_5m["Low"].tolist()
    swing_lows = []
    for i in range(2, len(lows) - 2):
        before = lows[i - 2:i]
        after = lows[i + 1:i + 3]
        if lows[i] < min(before) and lows[i] < min(after):
            swing_lows.append(float(lows[i]))
    return swing_lows
