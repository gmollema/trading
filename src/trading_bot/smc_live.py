"""Pure/live-logic helpers for the SMC paper-trading cycle (cli/smc_cycle.py).

Kept separate from the scheduled entrypoint so everything here is
importable and unit-testable without a broker connection and without
cycle-style import-time market gates. Signal logic itself lives in
backtest/smc_signals.py (latest_entry_signal / confirmed_new_high_exit)
-- the live bot re-runs the exact code the backtest validated; this
module only adds the live plumbing around it: config, watchlist IO,
bar-frame conversion, and the reactive position-size dampener computed
from the SMC bot's own realized trade history.

All SMC state files are deliberately separate from the gap-and-go bot's
(watchlist.txt / open_positions.json / trades.csv), so either bot can be
enabled, disabled, or inspected without touching the other.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from trading_bot.backtest import portfolio
from trading_bot.backtest.smc_signals import (
    DEFAULT_ENTRY_FILL,
    DEFAULT_EXIT_FILL,
    ENTRY_FILLS,
    EXIT_FILLS,
)

SMC_RULES_PATH = Path("smc_rules.json")
SMC_WATCHLIST_PATH = Path("smc_watchlist.txt")
SMC_POSITIONS_PATH = Path("smc_open_positions.json")
SMC_TRADES_CSV_PATH = Path("smc_trades.csv")
SMC_SAFETY_LOG_PATH = Path("smc-safety-check-log.json")

TRADES_CSV_HEADER = ["timestamp_iso", "symbol", "side", "size", "fill_price", "order_id", "status", "reason"]


def load_smc_rules(path: Path = SMC_RULES_PATH) -> dict:
    return json.loads(path.read_text())


# What smc_rules.json's "entry" block means when it is absent or partial:
# the original unreachable spec, so an old rules file keeps describing the
# behaviour it actually described.
DEFAULT_ENTRY_RULES = {"fill": DEFAULT_ENTRY_FILL, "require_ob_reclaim": False}


def entry_rules(rules: dict) -> dict:
    """The entry specification from a rules dict, defaults filled in.

    Shared by the live bot and every backtest CLI so they cannot drift:
    the whole point of respecifying the entry was that the backtest was
    scoring a fill the bot could not get, and two independent readings of
    the config is the obvious way to reintroduce that.

    `fill` is validated here even though only the backtests consume it --
    live it is implicit (a market order sent the moment a closed bar
    produces a signal, which is what "next_open" models). A typo would
    otherwise sit in the config unnoticed until a backtest run picked it
    up, describing a live bot that had been doing something else.
    """
    spec = {**DEFAULT_ENTRY_RULES, **(rules.get("entry") or {})}
    if spec["fill"] not in ENTRY_FILLS:
        raise ValueError(f"unknown entry fill: {spec['fill']!r}; expected one of {list(ENTRY_FILLS)}")
    return {"fill": spec["fill"], "require_ob_reclaim": bool(spec["require_ob_reclaim"])}


DEFAULT_EXIT_RULES = {"fill": DEFAULT_EXIT_FILL, "tp1_resting_limit": False}


def exit_rules(rules: dict) -> dict:
    """The exit specification from a rules dict, defaults filled in.

    Backtest-side only, unlike entry_rules: every leg it describes is
    already what the live bot does. The stop rests at IBKR, and TP1 and
    the new-high exit are market orders sent once a closed bar reveals
    the trigger. What was wrong was the backtest, which booked each
    trigger as its own fill -- so this configures the model to match the
    bot rather than asking the bot to change.

    tp1_resting_limit is the one exception and is the reason this is
    configurable at all: it describes a bot that does NOT exist yet, one
    resting a bracketed take-profit at the target. Setting it true without
    building that would put the backtest back to scoring fills nobody
    gets, which is the whole failure being corrected here.
    """
    spec = {**DEFAULT_EXIT_RULES, **(rules.get("exit") or {})}
    if spec["fill"] not in EXIT_FILLS:
        raise ValueError(f"unknown exit fill: {spec['fill']!r}; expected one of {list(EXIT_FILLS)}")
    return {"fill": spec["fill"], "tp1_resting_limit": bool(spec["tp1_resting_limit"])}


def get_market_status(now_et: datetime, rules: dict) -> str:
    """Same shape as cycle.get_market_status (weekend / too_early / closed /
    manage_only / force_close / ok), but with the window boundaries taken
    from smc_rules.json's time_filter instead of hardcoded -- duplicated
    rather than imported because cycle.py sys.exit()s at import time
    outside market hours."""
    if now_et.weekday() >= 5:
        return "weekend"

    tf = rules["time_filter"]
    earliest_entry = tf["earliest_entry_et"]
    latest_entry = tf["latest_entry_et"]
    force_close = tf["force_close_et"]

    hm = now_et.strftime("%H:%M")
    if hm < "10:00":
        return "too_early"
    if hm > "16:00":
        return "closed"
    if hm < earliest_entry:
        return "manage_only"
    if hm < latest_entry:
        return "ok"
    if hm < force_close:
        return "manage_only"
    if hm <= "16:00":
        return "force_close"
    return "closed"  # unreachable, defensive fallback


def read_watchlist(path: Path = SMC_WATCHLIST_PATH) -> list[str]:
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


def bars_frame_to_dict(bars: pd.DataFrame) -> dict:
    """Convert a yfinance-style OHLC frame (Open/High/Low/Close columns,
    DatetimeIndex) into the plain-lists bars dict smc_signals expects."""
    return {
        "open": bars["Open"].astype(float).tolist(),
        "high": bars["High"].astype(float).tolist(),
        "low": bars["Low"].astype(float).tolist(),
        "close": bars["Close"].astype(float).tolist(),
        "date": list(bars.index),
    }


def reactive_size_multiplier(
    trades_csv_path: Path = SMC_TRADES_CSV_PATH,
    window: int = 200,
    pf_threshold: float = 0.8,
    size_mult: float = 0.3,
) -> float:
    """Live port of smc_engine's reactive-derisk sizing dampener: trailing
    profit factor over the last `window` CLOSED round-trip trades from the
    SMC bot's own trade log; below `pf_threshold`, scale new entries by
    `size_mult`. Until `window` round trips exist (including a missing or
    empty log -- e.g. the bot's first weeks), sizing is unaffected, same
    as the backtest's warm-up behavior."""
    if not trades_csv_path.exists():
        return 1.0
    try:
        df = pd.read_csv(trades_csv_path, dtype=str)
    except Exception:
        return 1.0
    if df.empty:
        return 1.0
    for col in ("size", "fill_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Imported lazily: compute_perf pulls in yfinance at module import,
    # which this otherwise-pure module shouldn't force on its importers.
    from trading_bot.cli.compute_perf import pair_trades_fifo

    closed_pairs = pair_trades_fifo(df)
    if len(closed_pairs) < window:
        return 1.0

    recent = closed_pairs[-window:]
    gains = sum(p["pnl_usd"] for p in recent if p["pnl_usd"] > 0)
    losses = -sum(p["pnl_usd"] for p in recent if p["pnl_usd"] < 0)
    profit_factor = float("inf") if losses == 0 else gains / losses
    return size_mult if profit_factor < pf_threshold else 1.0


def entry_size(
    portfolio_value: float,
    entry_price: float,
    stop_price: float,
    rules: dict,
    trades_csv_path: Path = SMC_TRADES_CSV_PATH,
) -> int:
    """Risk-based share count for a new SMC entry: the same
    portfolio.position_size math the backtest engine uses, scaled by the
    reactive-derisk multiplier from realized SMC trade history."""
    risk = rules["risk"]
    size = portfolio.position_size(
        portfolio_value,
        risk["max_risk_per_trade_pct"],
        entry_price,
        stop_price,
        risk["max_position_size_pct_of_portfolio"],
    )
    derisk = rules.get("reactive_derisk")
    if derisk:
        size = int(
            size
            * reactive_size_multiplier(
                trades_csv_path, derisk["window"], derisk["pf_threshold"], derisk["size_mult"]
            )
        )
    return int(size)
