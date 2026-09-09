"""Pure live-logic helpers for the RSI(20) dip strategy expressed through
a LEVERAGED ETF instead of a futures contract
(cli/rsi20_dip_etf_cycle.py).

WHY THIS VARIANT EXISTS
-----------------------
One MES contract is ~$38,800 of index. A EUR 500 account cannot post the
~$2,000 overnight margin, and at 70x leverage a 1.4% adverse move would
liquidate it. A daily-rebalanced 3x ETF gets leveraged exposure with no
margin, no liquidation and no expiry, sized to whatever cash is there.

THE SIGNAL IS COMPUTED ON THE INDEX, NOT ON THE ETP
---------------------------------------------------
This is the load-bearing design decision. RSI(20) on a 3x fund is NOT
RSI(20) on its index: tripling every daily return pushes the oscillator
to more extreme values, so the 60/65 levels -- already only fitted to US
large-cap behaviour -- would mean something different applied to the
fund's own price. `signal_symbol` is therefore the index proxy (SPY) and
`trade_symbol` the ETP, and they are deliberately separate keys.

WHAT LEVERAGE ACTUALLY DID, MEASURED
------------------------------------
Signals on ^GSPC 2008-2026, exposure through a synthetic Nx
daily-rebalanced fund, costs = expense ratio + (N-1) x financing on days
held only, no commission:

    variant                   CAGR    maxDD    x money
    buy & hold index         9.41%    53.3%      5.36x
    strategy 1x              6.48%    16.6%      3.23x
    strategy 2x, fin 4%      9.60%    31.7%      5.53x
    strategy 3x, fin 4%     12.24%    44.2%      8.64x
    HOLD 3x fund always      5.82%    95.0%      2.87x

The last row is the point: permanently holding a 3x fund loses to the
plain index, because volatility decay compounds against it. The strategy
beats it only by being out of the market ~50% of the time, and out
during the worst of it.

AND THE HONEST CAVEAT, ONCE
---------------------------
Split by window, the advantage is not stable. 2015-2026: 3x returns
12.96% against the index's 12.04% while carrying a 44% drawdown against
its 34%. 2x LOSES to the index over that window (9.99%). The full-sample
win leans on sitting out 2008-09. Treat this as roughly a coin flip
against a plain index fund with more volatility, not as an edge.

EU ACCESS: NOT UPRO
-------------------
UPRO and SSO are US-domiciled, have no PRIIPs KID, and are blocked for
EU retail accounts. The default `trade_symbol` is a UCITS-wrapped LSE
listing instead. Verify it qualifies in YOUR account before trusting it
-- run the cycle with --check, which resolves both contracts and prints
the sizing without placing anything. Leveraged ETPs also usually require
a broker appropriateness test before the permission is granted.

FX IS NOT HANDLED, DELIBERATELY
-------------------------------
`capital_currency` must equal `trade_currency` or load_rules raises.
A EUR account buying a USD-denominated ETP either converts first or lets
IBKR lend the USD at its margin rate -- which on a small balance is
several percent a year against a strategy whose whole margin over an
index fund is about one point. Converting once, by hand, is both cheaper
and the reason this is a hard error rather than an FX lookup.
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

# The decision function, the bar-completeness guard and the state IO are
# imported, not reimplemented: this variant differs from the futures one
# in its instrument and its sizing, and in nothing else. A second copy of
# `decide` is a second thing to keep in step with the backtest.
from trading_bot.rsi20_dip_live import bars_needed, decide  # noqa: F401 - re-exported
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

DEFAULT_RULES = {
    # Index proxy the signal is computed from. SPY rather than ^SPX
    # because index data needs a separate IBKR entitlement and an ETF's
    # TRADES bars do not.
    "signal_symbol": "SPY",
    "signal_exchange": "SMART",
    "signal_currency": "USD",
    "signal_primary_exchange": "ARCA",
    # The 3x S&P 500 ETP actually traded. VERIFY THIS QUALIFIES FIRST.
    "trade_symbol": "3USL",
    "trade_exchange": "LSE",
    "trade_currency": "USD",
    "trade_primary_exchange": "",
    "trade_leverage": 3,
    # The allocation, in trade_currency. Capped by real account equity at
    # runtime, so this can safely sit inside a bigger account.
    "capital": 500.0,
    "capital_currency": "USD",
    "max_capital": 500.0,
    "position_fraction": 1.0,
    # Size off 97% of capital: the order fills at the next open, which can
    # gap above the close it was sized from, and an order for one share
    # more than the cash covers is simply rejected.
    "sizing_buffer": 0.97,
    "max_shares": 100,
    "rsi_period": DEFAULT_RSI_PERIOD,
    "ma_period": DEFAULT_MA_PERIOD,
    "offset_bars": DEFAULT_OFFSET_BARS,
    "entry_level": DEFAULT_ENTRY_LEVEL,
    "exit_level": DEFAULT_EXIT_LEVEL,
}


def load_rules(path: Path = RULES_PATH) -> dict:
    """Rules from disk with defaults filled in, validated.

    The checks are the ones whose absence costs money rather than
    correctness: `max_capital` is a ceiling verified here instead of
    trusted from the file, `capital_currency == trade_currency` refuses
    to guess at FX (see the module docstring), and
    `exit_level > entry_level` catches the logic error that turns this
    strategy into a churning machine.

    `signal_symbol == trade_symbol` is rejected outright: computing
    RSI(20) on a 3x fund's own price silently changes what the fitted
    60/65 levels mean.
    """
    raw = json.loads(path.read_text()) if path.exists() else {}
    rules = {**DEFAULT_RULES, **raw}

    if rules["capital"] <= 0:
        raise ValueError(f"capital must be > 0, got {rules['capital']}")
    if rules["capital"] > rules["max_capital"]:
        raise ValueError(
            f"capital ({rules['capital']}) exceeds max_capital "
            f"({rules['max_capital']}) -- raise the ceiling deliberately if intended")
    if rules["capital_currency"] != rules["trade_currency"]:
        raise ValueError(
            f"capital is in {rules['capital_currency']} but {rules['trade_symbol']} "
            f"trades in {rules['trade_currency']} -- convert the cash in IBKR and set "
            f"capital_currency to {rules['trade_currency']}; this bot does not do FX")
    if not 0 < rules["position_fraction"] <= 1:
        raise ValueError(
            f"position_fraction must be in (0, 1], got {rules['position_fraction']}")
    if not 0 < rules["sizing_buffer"] <= 1:
        raise ValueError(
            f"sizing_buffer must be in (0, 1], got {rules['sizing_buffer']}")
    if rules["max_shares"] < 1:
        raise ValueError(f"max_shares must be >= 1, got {rules['max_shares']}")
    if rules["trade_leverage"] < 1:
        raise ValueError(f"trade_leverage must be >= 1, got {rules['trade_leverage']}")
    if not rules["signal_symbol"] or not rules["trade_symbol"]:
        raise ValueError("signal_symbol and trade_symbol must both be set")
    if rules["signal_symbol"] == rules["trade_symbol"]:
        raise ValueError(
            "signal_symbol must differ from trade_symbol -- the RSI levels are fitted "
            "to the INDEX, and a leveraged fund's own RSI is a different series")
    if not 0 < rules["entry_level"] < rules["exit_level"] < 100:
        raise ValueError(
            f"need 0 < entry_level < exit_level < 100, got {rules['entry_level']} / "
            f"{rules['exit_level']} -- equal levels exit on the mirror of the entry")
    if rules["rsi_period"] < 1 or rules["ma_period"] < 1:
        raise ValueError("rsi_period and ma_period must be >= 1")
    if rules["offset_bars"] < 0:
        raise ValueError("offset_bars must be >= 0")
    return rules


def decide_action(bars: dict, position: dict | None, rules: dict) -> dict:
    """`decide` with the futures-only `contracts` key adapted away.

    The shared decide() sizes its own answer in CONTRACTS, reading
    rules["contracts"] -- a key this variant does not have, because its
    size is not a constant: a buy is sized from capital and the ETP's
    price, and a sell is whatever share count is held. So a placeholder
    is supplied and the returned `contracts` is dropped in favour of
    `shares`, set by the caller.

    This adapter exists so the signal path stays the single shared
    function rather than a second copy that has to be kept in step with
    the backtest. Without it, an ETF cycle raises KeyError('contracts')
    the first time it holds a position -- which is how this was found.
    """
    out = _decide_contracts(bars, position, {**rules, "contracts": 1})
    held = int(position.get("shares", 0)) if position else 0
    shares = held if out["action"] == "sell" else 0
    return {k: v for k, v in out.items() if k != "contracts"} | {"shares": shares}


def usable_capital(rules: dict, account_equity: float | None,
                   account_currency: str = "") -> float:
    """The smaller of the configured allocation and real account equity.

    Account equity in a currency other than trade_currency is ignored
    rather than converted, for the same reason load_rules refuses to do
    FX -- but the allocation still applies, so the bot stays sized by the
    figure a human wrote down. `None` equity (summary unavailable) is
    treated the same way: fall back to the allocation, never to nothing.
    """
    allocation = rules["capital"] * rules["position_fraction"]
    if account_equity is None or account_equity <= 0:
        return allocation
    if account_currency and account_currency != rules["trade_currency"]:
        return allocation
    return min(allocation, account_equity * rules["position_fraction"])


def shares_for(capital: float, price: float, rules: dict) -> dict:
    """How many whole shares to buy, and how much of the capital that
    leaves unused.

    Returns {"shares", "notional", "idle_cash", "idle_pct"}.

    The idle figure is returned rather than merely tolerated because at
    this account size it is the dominant sizing error: 500 of capital
    against a ~$100 ETP buys 4 shares and leaves ~20% of the account in
    cash, which silently turns a 3x strategy into a 2.4x one. A caller
    that finds idle_pct high should either raise the capital or pick a
    lower-priced listing -- it is not a bug to be rounded away.

    Raises:
        ValueError: if price is not positive. A zero price means the
            quote failed, and sizing off it would ask for a colossal
            position.
    """
    if price <= 0:
        raise ValueError(f"price must be > 0 to size a position, got {price}")
    budget = capital * rules["sizing_buffer"]
    shares = min(int(math.floor(budget / price)), rules["max_shares"])
    notional = shares * price
    idle = max(capital - notional, 0.0)
    return {
        "shares": shares,
        "notional": notional,
        "idle_cash": idle,
        "idle_pct": (idle / capital * 100.0) if capital > 0 else 0.0,
    }


def effective_leverage(sized: dict, capital: float, rules: dict) -> float:
    """Real exposure as a multiple of capital, after share rounding.

    A 3x fund held with 20% of the account idle is 2.4x exposure. Logged
    every cycle so the number in the log is the position actually taken,
    not the one the config asked for.
    """
    if capital <= 0:
        return 0.0
    return sized["notional"] / capital * rules["trade_leverage"]


__all__ = [
    "DECISION_WINDOW_END_ET",
    "DECISION_WINDOW_START_ET",
    "DEFAULT_RULES",
    "HEARTBEAT_PATH",
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
