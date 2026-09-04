"""Pure live-logic helpers for the rsi2 daily cycle (cli/rsi2_cycle.py).

Kept separate from the scheduled entrypoint so all of it is importable and
unit-testable without a broker connection. Signal logic itself is NOT
reimplemented here -- it comes from backtest/rsi2_signals.py, so the live
bot runs the exact code the backtest validated. This module adds only the
plumbing: config, state IO, and the decision function that turns a bar
history plus a held position into an action.

State files are deliberately separate from every other bot's, so rsi2 can
be enabled, disabled or inspected without touching gap-and-go or SMC.

THE FILL SPEC, and why it is what it is
---------------------------------------
rsi2 is a DAILY strategy whose signal needs the day's CLOSE: RSI(2) below
10 is only knowable once that close exists. A bot cannot therefore trade
the close it computed from -- that fill is unreachable, and chasing it is
what invalidated the SMC strategy in this repo.

So this bot implements the reachable spec instead, measured as
equivalent (6.43% CAGR vs 6.36% at $75k, and marginally better):

  - decide from the LAST COMPLETED daily bar
  - fill with a market order at the next regular-session open

which is `entry_timing="next_open", exit_timing="next_open"` in
find_rsi2_scale_in_trades. The cycle is therefore scheduled once per
trading day, a couple of minutes before 09:30 ET, and does nothing at any
other time.

Consequences worth stating plainly:

  - Today's partial bar is never used for a decision. A cycle that runs
    mid-session must refuse to act, because the bar it would read is
    incomplete and RSI(2) on a partial bar is a different indicator.
  - There is one decision per day. Missing a day means missing that
    day's entry or exit outright; there is no catch-up, because entering
    a day late is a different trade from the one that was backtested.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from trading_bot.backtest.rsi2_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_RSI_PERIOD,
    DEFAULT_SMA_PERIOD,
    rsi2_dip_sequence,
    wilder_rsi,
)

RSI2_RULES_PATH = Path("rsi2_rules.json")
RSI2_POSITIONS_PATH = Path("rsi2_open_positions.json")
RSI2_TRADES_CSV_PATH = Path("rsi2_trades.csv")
RSI2_HEARTBEAT_PATH = Path("rsi2_heartbeat.json")

TRADES_CSV_HEADER = [
    "timestamp_iso", "symbol", "local_symbol", "side", "size", "fill_price",
    "order_id", "status", "reason", "signal_bar_date",
]

# The window, in ET minutes-from-midnight, in which the daily cycle is
# allowed to act: a few minutes before the RTH open through a short grace
# period after it. Outside this the bot exits without touching anything --
# it is not an intraday strategy and a mid-session run would be reading a
# partial bar.
DECISION_WINDOW_START_ET = "09:25"
DECISION_WINDOW_END_ET = "09:45"

DEFAULT_RULES = {
    "symbol": "ES",
    "exchange": "CME",
    "contracts": 1,
    "rsi_period": DEFAULT_RSI_PERIOD,
    "entry_level": DEFAULT_ENTRY_LEVEL,
    "exit_level": DEFAULT_EXIT_LEVEL,
    "sma_period": DEFAULT_SMA_PERIOD,
    "first_dip": 1,
    "min_days_to_expiry": 10,
    "max_contracts": 1,
}


def load_rules(path: Path = RSI2_RULES_PATH) -> dict:
    """Rules from disk with defaults filled in, validated.

    `max_contracts` is a hard ceiling checked here rather than trusted
    from the config: this is the one knob whose typo costs real money, and
    a 10x fat-finger on a $50-a-point contract is not something to
    discover from a fill report.
    """
    raw = json.loads(path.read_text()) if path.exists() else {}
    rules = {**DEFAULT_RULES, **raw}
    if rules["contracts"] < 1:
        raise ValueError(f"contracts must be >= 1, got {rules['contracts']}")
    if rules["contracts"] > rules["max_contracts"]:
        raise ValueError(
            f"contracts ({rules['contracts']}) exceeds max_contracts "
            f"({rules['max_contracts']}) -- raise the ceiling deliberately if intended")
    if not 0 < rules["entry_level"] < rules["exit_level"] < 100:
        raise ValueError(f"need 0 < entry_level < exit_level < 100, got "
                         f"{rules['entry_level']} / {rules['exit_level']}")
    if rules["first_dip"] < 1:
        raise ValueError(f"first_dip must be >= 1, got {rules['first_dip']}")
    if rules["min_days_to_expiry"] < 1:
        raise ValueError("min_days_to_expiry must be >= 1")
    return rules


def in_decision_window(now_et: datetime) -> bool:
    """True only inside the daily decision window on a weekday.

    Deliberately narrow. rsi2 acts once a day at the open; a run at any
    other time is either a scheduling mistake or a manual invocation, and
    in both cases doing nothing is correct.
    """
    if now_et.weekday() >= 5:
        return False
    hhmm = now_et.strftime("%H:%M")
    return DECISION_WINDOW_START_ET <= hhmm <= DECISION_WINDOW_END_ET


def load_positions(path: Path = RSI2_POSITIONS_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_positions(positions: list[dict], path: Path = RSI2_POSITIONS_PATH) -> None:
    """Atomic write -- a crash mid-save must not leave a truncated file
    that reads back as "flat" and desynchronises the bot from the broker."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(positions, indent=2))
    tmp.replace(path)


def append_trade(row: dict, path: Path = RSI2_TRADES_CSV_PATH) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADES_CSV_HEADER, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_bars(bars: dict, now_et: datetime) -> dict:
    """`bars` with any bar dated today dropped.

    IBKR's daily bar for the current session is a PARTIAL bar and its
    close is just the last trade. Feeding it to RSI(2) produces a
    different indicator from the one that was backtested, so it is
    removed rather than trusted. This is the single most important guard
    in the module: everything downstream assumes the last bar is final.
    """
    today = now_et.date()
    keep = [i for i, d in enumerate(bars["date"]) if _as_date(d) < today]
    return {k: [bars[k][i] for i in keep] for k in ("date", "open", "high", "low", "close")}


def _as_date(value):
    return value.date() if hasattr(value, "date") else value


def decide(bars: dict, position: dict | None, rules: dict) -> dict:
    """What to do at today's open, given completed daily bars.

    Returns {"action": "buy"|"sell"|"hold", "reason", "rsi", "dip",
    "signal_bar_date", "contracts"}.

    Mirrors find_rsi2_scale_in_trades' per-bar order exactly: an
    overbought close exits before anything else is considered, and only
    then can a fresh crossing open a position. `first_dip` uses the shared
    rsi2_dip_sequence so live and backtest cannot disagree about what
    "the second dip" means.
    """
    closes = bars["close"]
    n = len(closes)
    need = max(rules["sma_period"], rules["rsi_period"] + 1)
    if n < need:
        return {"action": "hold", "reason": f"need {need} bars, have {n}",
                "rsi": None, "dip": 0, "signal_bar_date": None, "contracts": 0}

    rsi = wilder_rsi(closes, rules["rsi_period"])
    dips = rsi2_dip_sequence(closes, rules["rsi_period"], rules["entry_level"],
                             rules["exit_level"], rules["sma_period"])
    i = n - 1
    last_rsi = rsi[i]
    bar_date = str(_as_date(bars["date"][i]))
    base = {"rsi": last_rsi, "dip": dips[i], "signal_bar_date": bar_date}

    if position is not None:
        if last_rsi is not None and last_rsi > rules["exit_level"]:
            return {**base, "action": "sell", "reason": "rsi_exit",
                    "contracts": position.get("contracts", rules["contracts"])}
        return {**base, "action": "hold", "reason": "holding", "contracts": 0}

    if dips[i] >= rules["first_dip"] and dips[i] > 0:
        return {**base, "action": "buy", "reason": f"rsi2_dip_{dips[i]}",
                "contracts": rules["contracts"]}
    return {**base, "action": "hold", "reason": "no_signal", "contracts": 0}


def expiry_action(position: dict, today, min_days: int) -> str:
    """"roll_out" when a held contract is inside its expiry window, else "".

    The backtest has no concept of expiry; a real position must be closed
    before the contract dies or the broker decides for you. Closing early
    deviates from the backtest by at most a few days of a trade, which is
    strictly preferable to being auto-liquidated at whatever price the
    liquidation gets.
    """
    raw = position.get("expiry")
    if not raw:
        return ""
    try:
        expiry = datetime.strptime(str(raw)[:8], "%Y%m%d").date()
    except ValueError:
        return ""
    return "roll_out" if (expiry - today).days <= min_days else ""
