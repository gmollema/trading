"""Pure live-logic helpers for the RSI(20) dip daily cycle
(cli/rsi20_dip_cycle.py).

Same shape as rsi2_live: no broker connection here, so all of it is
importable and unit-testable. Signal logic is NOT reimplemented -- the
crossings, the shifted trend MA and the RSI all come from
backtest/rsi20_dip_signals.py, so the live bot runs the code the backtest
validated. This module adds only config, state IO, and the decision
function that turns a bar history plus a held position into an action.

The generic daily-cycle plumbing -- the decision window, dropping today's
partial bar, and the positions/trades file IO -- is imported from
rsi2_live rather than copied. Those are the parts that must not diverge
between two strategies that both act once a day at the open, and
`completed_bars` in particular is the guard the whole design rests on. If
a third daily strategy appears, move them to util/ rather than copying
them a third time.

THE FILL SPEC
-------------
Identical to rsi2's, and for the same reason: the signal needs the day's
CLOSE, so the close it computed from is unreachable. The bot therefore
decides from the LAST COMPLETED daily bar and fills with a market order
at the next regular-session open, which is exactly what
find_rsi20_dip_trades models. The cycle runs once per trading day a few
minutes before 09:30 ET and does nothing at any other time.

WHAT THIS STRATEGY IS WORTH, STATED PLAINLY
-------------------------------------------
It loses to buy-and-hold on all eight instruments tested -- 6.55% CAGR
against the S&P's 9.33% over 2008-2026, and 45-55% of the index's annual
return on every market and every window tried, including 41 years of
Nasdaq and 57 of DAX. What it has is a better return per unit of
drawdown (0.39 vs 0.18 on the S&P) and per unit of time invested (13.1%
vs 9.33%, being in the market ~50% of days). Converting that into more
money than holding requires leverage.

It is wired live because it was asked for, not because the numbers argue
for it. Read backtest/rsi20_dip_signals.py's docstring before funding it,
and note that `entry_level` is fitted to US large-cap behaviour -- the
55-60 ridge does not exist on the DAX, FTSE, Nikkei or Euro Stoxx.

DO NOT RUN THIS AND rsi2 ON THE SAME CONTRACT AT ONCE
-----------------------------------------------------
Both default to MES. IBKR nets positions per contract, so two long-only
bots each holding "one" MES leaves a net position of two and neither bot
knows: margin doubles, and each one's JSON says it holds a single
contract. They happen not to corrupt each other -- both are long-only, so
each SELL reduces the net by one -- but the exposure is twice what either
backtest assumed. rsi2's scheduled task is disabled as of 2026-09-07, so
there is no live conflict today. Change `symbol` here before enabling
both, or accept 2x size deliberately.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from trading_bot.backtest.rsi20_dip_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_MA_PERIOD,
    DEFAULT_OFFSET_BARS,
    DEFAULT_RSI_PERIOD,
    REASON_RSI_EXIT,
    REASON_TREND_BREAK,
    crossed_down,
    crossed_up,
    shifted_trend_ma,
)
from trading_bot.backtest.rsi2_signals import simple_moving_average, wilder_rsi

# Shared, deliberately -- see the module docstring.
from trading_bot.rsi2_live import (  # noqa: F401 - re-exported for the cycle
    DECISION_WINDOW_END_ET,
    DECISION_WINDOW_START_ET,
    TRADES_CSV_HEADER,
    append_trade,
    completed_bars,
    in_decision_window,
    load_positions,
    save_positions,
)

RULES_PATH = Path("rsi20_dip_rules.json")
POSITIONS_PATH = Path("rsi20_dip_open_positions.json")
TRADES_CSV_PATH = Path("rsi20_dip_trades.csv")
HEARTBEAT_PATH = Path("rsi20_dip_heartbeat.json")

DEFAULT_RULES = {
    "symbol": "MES",
    "exchange": "CME",
    "contracts": 1,
    "rsi_period": DEFAULT_RSI_PERIOD,
    "ma_period": DEFAULT_MA_PERIOD,
    "offset_bars": DEFAULT_OFFSET_BARS,
    "entry_level": DEFAULT_ENTRY_LEVEL,
    "exit_level": DEFAULT_EXIT_LEVEL,
    "min_days_to_expiry": 10,
    "max_contracts": 1,
}


def load_rules(path: Path = RULES_PATH) -> dict:
    """Rules from disk with defaults filled in, validated.

    `max_contracts` is a hard ceiling checked here rather than trusted
    from the file: it is the one knob whose typo costs real money.
    `exit_level > entry_level` is checked for a different reason -- equal
    levels are not a sizing mistake but a LOGIC mistake that silently
    turns the strategy into a churning machine, which is precisely how
    the original reconstruction went wrong.
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
        raise ValueError(
            f"need 0 < entry_level < exit_level < 100, got {rules['entry_level']} / "
            f"{rules['exit_level']} -- equal levels exit on the mirror of the entry")
    if rules["rsi_period"] < 1 or rules["ma_period"] < 1:
        raise ValueError("rsi_period and ma_period must be >= 1")
    if rules["offset_bars"] < 0:
        raise ValueError("offset_bars must be >= 0")
    if rules["min_days_to_expiry"] < 1:
        raise ValueError("min_days_to_expiry must be >= 1")
    return rules


def bars_needed(rules: dict) -> int:
    """Completed bars required before the strategy has an opinion.

    The trend filter reads SMA(ma_period) as it stood offset_bars ago, so
    the first bar with a value is ma_period - 1 + offset_bars; the
    crossings need the bar before that as well. Reported rather than
    silently held on, so a "hold" in the log always means the market
    offered nothing -- never that the bot could not see.
    """
    return max(rules["ma_period"] + rules["offset_bars"], rules["rsi_period"] + 1) + 1


def decide(bars: dict, position: dict | None, rules: dict) -> dict:
    """What to do at today's open, given completed daily bars.

    Returns {"action": "buy"|"sell"|"hold", "reason", "rsi", "trend_ok",
    "signal_bar_date", "contracts"}.

    Mirrors find_rsi20_dip_trades' per-bar order exactly: an exit is
    considered before any entry, and the RSI exit is tested before the
    trend break so the logged reason matches the backtest's.
    """
    closes = bars["close"]
    n = len(closes)
    need = bars_needed(rules)
    if n < need:
        return {"action": "hold", "reason": f"need {need} bars, have {n}",
                "rsi": None, "trend_ok": None, "signal_bar_date": None, "contracts": 0}

    rsi = wilder_rsi(closes, rules["rsi_period"])
    trend = simple_moving_average(closes, rules["ma_period"])
    shifted = shifted_trend_ma(closes, rules["ma_period"], rules["offset_bars"])
    i = n - 1
    bar_date = str(_as_date(bars["date"][i]))
    base = {
        "rsi": rsi[i],
        "trend_ok": None if shifted[i] is None else closes[i] > shifted[i],
        "signal_bar_date": bar_date,
    }

    if rsi[i] is None or trend[i] is None or shifted[i] is None:
        return {**base, "action": "hold", "reason": "indicators_not_ready", "contracts": 0}

    if position is not None:
        held = position.get("contracts", rules["contracts"])
        if crossed_up(rsi, rules["exit_level"], i):
            return {**base, "action": "sell", "reason": REASON_RSI_EXIT, "contracts": held}
        if closes[i] < trend[i]:
            return {**base, "action": "sell", "reason": REASON_TREND_BREAK, "contracts": held}
        return {**base, "action": "hold", "reason": "holding", "contracts": 0}

    if closes[i] > shifted[i] and crossed_down(rsi, rules["entry_level"], i):
        return {**base, "action": "buy", "reason": "rsi_dip", "contracts": rules["contracts"]}
    return {**base, "action": "hold", "reason": "no_signal", "contracts": 0}


def expiry_action(position: dict, today, min_days: int) -> str:
    """"roll_out" when a held contract is inside its expiry window.

    The backtest has no concept of expiry; a real position must be closed
    before the contract dies or the broker decides for you. Closing a few
    days early deviates from the backtest by a fraction of one trade,
    which beats being auto-liquidated at whatever price that gets.
    """
    raw = position.get("expiry")
    if not raw:
        return ""
    try:
        expiry = datetime.strptime(str(raw)[:8], "%Y%m%d").date()
    except ValueError:
        return ""
    return "roll_out" if (expiry - today).days <= min_days else ""


def _as_date(value):
    return value.date() if hasattr(value, "date") else value
