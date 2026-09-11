"""Pure live-logic helpers for the RSI(20) dip strategy traded directly
through SPY with a small cash allocation.

This variant is deliberately SIMPLE:
    - $500 configured account
    - 5% allocation per trade = $25 target
    - SPY signal
    - SPY trade
    - 1x exposure
    - no leveraged ETP
    - no broker margin
    - fractional SPY shares where IBKR/API permissions allow them

The RSI/trend signal logic is NOT reimplemented here. It continues to
come from backtest/rsi20_dip_signals.py through rsi20_dip_live.py, so the
live bot uses the same signal implementation that the backtest validated.

FILL SPEC
---------
The signal needs the day's CLOSE. That close is unavailable at the
decision point, so the bot decides from the LAST COMPLETED daily bar and
fills at the next regular-session open.

The cycle runs once per trading day inside the shared decision window
before the US regular session open.

SIZING
------
The account is intentionally treated as a small trading account:

    configured capital:       $500
    position fraction:          5%
    target position:           $25
    sizing buffer:              97%
    order budget:            $24.25

The buffer leaves room for the next-open price to differ from the signal
bar's close.

The bot does NOT borrow from IBKR and does NOT use broker margin.

FRACTIONAL SHARES
-----------------
SPY is intended to be traded fractionally. `size_increment` defaults to
0.0001 shares.

Whether fractional orders are accepted depends on the IBKR account,
contract and API/order path. The old 3USL-specific claim that all
fractional API orders are rejected is intentionally NOT carried over
here. Verify SPY fractional orders with whatIfOrder before enabling live
submission.

HARD POSITION LIMIT
-------------------
MAX_POSITION_VALUE is an additional safety ceiling of $25 by default.

This is independent of position_fraction. If configuration accidentally
requests more than the hard limit, load_rules() rejects it.

This file only sizes the strategy. The actual IBKR execution layer should
also enforce the same ceiling immediately before submitting a BUY order.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from trading_bot.backtest.rsi20_dip_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_MA_PERIOD,
    DEFAULT_OFFSET_BARS,
    DEFAULT_RSI_PERIOD,
)

# Keep the validated signal path shared with the futures implementation.
from trading_bot.rsi20_dip_live import bars_needed, decide  # noqa: F401
from trading_bot.rsi20_dip_live import decide as _decide_contracts

from trading_bot.rsi2_live import (
    DECISION_WINDOW_END_ET,
    DECISION_WINDOW_START_ET,
    append_trade,
    completed_bars,
    in_decision_window,
    load_positions,
    save_positions,
)


RULES_PATH = Path("rsi20_dip_etf_rules.json")
POSITIONS_PATH = Path("rsi20_dip_etf_open_positions.json")
TRADES_CSV_PATH = Path("rsi20_dip_etf_trades.csv")
HEARTBEAT_PATH = Path("rsi20_dip_etf_heartbeat.json")

# Absolute safety ceiling for this small-account version.
MAX_POSITION_VALUE = 25.00


DEFAULT_RULES = {
    # ---------------------------------------------------------------
    # Signal
    # ---------------------------------------------------------------
    "signal_symbol": "SPY",
    "signal_exchange": "SMART",
    "signal_currency": "USD",
    "signal_primary_exchange": "ARCA",

    # ---------------------------------------------------------------
    # Trade
    # ---------------------------------------------------------------
    # We now trade the same instrument on which the signal is based.
    "trade_symbol": "SPY",
    "trade_exchange": "SMART",
    "trade_currency": "USD",
    "trade_primary_exchange": "ARCA",

    # No leverage.
    "trade_leverage": 1,

    # Fractional SPY shares.
    "size_increment": 0.0001,

    # ---------------------------------------------------------------
    # Account / sizing
    # ---------------------------------------------------------------
    "capital": 500.0,
    "capital_currency": "USD",
    "max_capital": 500.0,

    # 5% of $500 = $25 target position.
    "position_fraction": 0.05,

    # Size slightly below target because the order fills at the next
    # open rather than the signal bar close.
    "sizing_buffer": 0.97,

    # Extra defensive ceiling.
    "max_position_value": MAX_POSITION_VALUE,

    "max_shares": 100,

    # ---------------------------------------------------------------
    # Strategy parameters -- unchanged
    # ---------------------------------------------------------------
    "rsi_period": DEFAULT_RSI_PERIOD,
    "ma_period": DEFAULT_MA_PERIOD,
    "offset_bars": DEFAULT_OFFSET_BARS,
    "entry_level": DEFAULT_ENTRY_LEVEL,
    "exit_level": DEFAULT_EXIT_LEVEL,
}


def load_rules(path: Path = RULES_PATH) -> dict:
    """Load rules from disk, fill defaults and validate them.

    This version intentionally permits signal_symbol == trade_symbol
    because the strategy now computes RSI/trend on SPY and trades SPY.

    The important money-safety checks are:
        - capital <= max_capital
        - position_fraction <= 5%
        - position value <= hard $25 ceiling
        - no leverage
        - valid fractional-share increment
        - USD/USD currency match
    """
    raw = json.loads(path.read_text()) if path.exists() else {}
    rules = {**DEFAULT_RULES, **raw}

    if rules["capital"] <= 0:
        raise ValueError(
            f"capital must be > 0, got {rules['capital']}"
        )

    if rules["capital"] > rules["max_capital"]:
        raise ValueError(
            f"capital ({rules['capital']}) exceeds max_capital "
            f"({rules['max_capital']}) -- raise the ceiling deliberately "
            f"if intended"
        )

    if rules["capital_currency"] != rules["trade_currency"]:
        raise ValueError(
            f"capital is in {rules['capital_currency']} but "
            f"{rules['trade_symbol']} trades in {rules['trade_currency']} "
            f"-- convert the cash in IBKR; this bot does not do FX"
        )

    if not 0 < rules["position_fraction"] <= 0.05:
        raise ValueError(
            f"position_fraction must be in (0, 0.05], got "
            f"{rules['position_fraction']} -- this small-account version "
            f"is capped at 5% per trade"
        )

    if not 0 < rules["sizing_buffer"] <= 1:
        raise ValueError(
            f"sizing_buffer must be in (0, 1], got "
            f"{rules['sizing_buffer']}"
        )

    if rules["max_shares"] <= 0:
        raise ValueError(
            f"max_shares must be > 0, got {rules['max_shares']}"
        )

    # This version is intentionally unleveraged.
    if rules["trade_leverage"] != 1:
        raise ValueError(
            f"trade_leverage must be exactly 1 for this version, "
            f"got {rules['trade_leverage']} -- broker leverage is disabled"
        )

    if not rules["signal_symbol"] or not rules["trade_symbol"]:
        raise ValueError(
            "signal_symbol and trade_symbol must both be set"
        )

    # signal_symbol == trade_symbol is VALID here.
    #
    # The old 3USL version rejected this because RSI on a leveraged ETP
    # is materially different from RSI on the index. We now trade SPY
    # directly, so there is no such mismatch.

    if rules["size_increment"] <= 0:
        raise ValueError(
            f"size_increment must be > 0, got "
            f"{rules['size_increment']}"
        )

    max_position_value = float(rules["max_position_value"])

    if max_position_value <= 0:
        raise ValueError(
            f"max_position_value must be > 0, got "
            f"{max_position_value}"
        )

    # This strategy is deliberately capped at $25.
    if max_position_value > MAX_POSITION_VALUE:
        raise ValueError(
            f"max_position_value ({max_position_value}) exceeds the "
            f"hard ${MAX_POSITION_VALUE:.2f} safety ceiling"
        )

    target_position = rules["capital"] * rules["position_fraction"]

    if target_position > max_position_value + 1e-9:
        raise ValueError(
            f"target position ${target_position:.2f} exceeds hard "
            f"position limit ${max_position_value:.2f}"
        )

    if not 0 < rules["entry_level"] < rules["exit_level"] < 100:
        raise ValueError(
            f"need 0 < entry_level < exit_level < 100, got "
            f"{rules['entry_level']} / {rules['exit_level']}"
        )

    if rules["rsi_period"] < 1 or rules["ma_period"] < 1:
        raise ValueError(
            "rsi_period and ma_period must be >= 1"
        )

    if rules["offset_bars"] < 0:
        raise ValueError(
            "offset_bars must be >= 0"
        )

    return rules


def decide_action(
        bars: dict,
        position: dict | None,
        rules: dict,
) -> dict:
    """Run the shared RSI20 decision logic and adapt contracts -> shares.

    The shared futures decision function expects a `contracts` field.
    Supplying a placeholder keeps the signal implementation identical.

    For a BUY, actual share sizing is performed separately by shares_for().
    For a SELL, the complete currently-held share count is returned.
    """
    out = _decide_contracts(
        bars,
        position,
        {**rules, "contracts": 1},
    )

    held = (
        float(position.get("shares", 0.0))
        if position
        else 0.0
    )

    shares = held if out["action"] == "sell" else 0.0

    return {
        k: v
        for k, v in out.items()
        if k != "contracts"
    } | {
        "shares": shares,
    }


def usable_capital(
        rules: dict,
        account_equity: float | None,
        account_currency: str = "",
) -> float:
    """Return the capital available for the strategy.

    The configured allocation is 5% of the configured capital.

    If IBKR reports valid equity in the same currency, the allocation is
    also constrained by 5% of actual account equity.

    A missing equity value does NOT cause a larger position to be taken;
    the configured allocation remains the fallback.
    """
    allocation = (
            rules["capital"]
            * rules["position_fraction"]
    )

    if account_equity is None or account_equity <= 0:
        return allocation

    if (
            account_currency
            and account_currency != rules["trade_currency"]
    ):
        return allocation

    return min(
        allocation,
        account_equity * rules["position_fraction"],
        )


def shares_for(
        capital: float,
        price: float,
        rules: dict,
) -> dict:
    """Calculate the fractional SPY position.

    Returns:

        {
            "shares": ...,
            "notional": ...,
            "idle_cash": ...,
            "idle_pct": ...
        }

    The target is 5% of capital, subject to the configured hard
    $25 position ceiling and the sizing buffer.

    Quantity is rounded DOWN to the configured fractional increment.
    """
    if capital <= 0:
        raise ValueError(
            f"capital must be > 0 to size a position, got {capital}"
        )

    if price <= 0:
        raise ValueError(
            f"price must be > 0 to size a position, got {price}"
        )

    increment = float(rules["size_increment"])

    # The caller normally passes the already-allocated capital
    # (approximately $25 for a $500 account).
    budget = capital * rules["sizing_buffer"]

    # Never exceed the hard $25 notional limit.
    budget = min(
        budget,
        float(rules["max_position_value"])
        * rules["sizing_buffer"],
        )

    # Floor to the instrument's quantity increment.
    units = math.floor(
        budget / price / increment + 1e-9
    )

    shares = (
            max(units, 0)
            * increment
    )

    shares = min(
        shares,
        float(rules["max_shares"]),
    )

    # Re-quantise after max_shares.
    shares = round(
        math.floor(
            shares / increment + 1e-9
        )
        * increment,
        8,
        )

    notional = shares * price

    # Final defensive check.
    if notional > float(rules["max_position_value"]) + 1e-8:
        raise ValueError(
            f"calculated position ${notional:.4f} exceeds hard "
            f"limit ${rules['max_position_value']:.2f}"
        )

    idle = max(capital - notional, 0.0)

    return {
        "shares": shares,
        "notional": notional,
        "idle_cash": idle,
        "idle_pct": (
            idle / capital * 100.0
            if capital > 0
            else 0.0
        ),
    }


def effective_leverage(
        sized: dict,
        capital: float,
        rules: dict,
) -> float:
    """Return actual exposure as a multiple of account capital.

    Because this version trades SPY at 1x, this is simply:

        position_notional / account_capital

    For example:

        $25 SPY position / $500 account = 0.05x

    This is NOT broker margin leverage.
    """
    if capital <= 0:
        return 0.0

    return (
            sized["notional"] / capital
    )


__all__ = [
    "DECISION_WINDOW_END_ET",
    "DECISION_WINDOW_START_ET",
    "DEFAULT_RULES",
    "HEARTBEAT_PATH",
    "MAX_POSITION_VALUE",
    "POSITIONS_PATH",
    "RULES_PATH",
    "TRADES_CSV_PATH",
    "append_trade",
    "bars_needed",
    "completed_bars",
    "decide",
    "decide_action",
    "effective_leverage",
    "in_decision_window",
    "load_positions",
    "load_rules",
    "save_positions",
    "shares_for",
    "usable_capital",
]