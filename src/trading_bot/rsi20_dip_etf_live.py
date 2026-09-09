"""Pure live-logic helpers for the RSI(20) dip strategy expressed through
a LEVERAGED ETF instead of a futures contract
(cli/rsi20_dip_etf_cycle.py).

WHY THIS VARIANT EXISTS
-----------------------
One MES contract is ~$38,800 of index. A EUR 500 account cannot post the
~$2,000 overnight margin, and at 70x leverage a 1.4% adverse move would
liquidate it. A daily-rebalanced leveraged ETF gets leveraged exposure
with no margin, no liquidation and no expiry, sized to whatever cash is
there.

THE SIGNAL IS COMPUTED ON THE INDEX, NOT ON THE ETP
---------------------------------------------------
This is the load-bearing design decision. RSI(20) on a 2x fund is NOT
RSI(20) on its index: doubling every daily return pushes the oscillator
to more extreme values, so the 60/65 levels -- already only fitted to US
large-cap behaviour -- would mean something different applied to the
fund's own price. `signal_symbol` is therefore the index proxy (SPY) and
`trade_symbol` the ETP, and they are deliberately separate keys.
load_rules rejects setting them equal.

WHAT LEVERAGE ACTUALLY DID, MEASURED
------------------------------------
Signals on ^GSPC 2008-2026, exposure through a synthetic Nx
daily-rebalanced fund, costs = expense ratio + (N-1) x financing on days
held only, no commission. Reproduce with cli/rsi20_dip_etf_backtest.py:

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

FRACTIONAL ORDERS ARE IMPOSSIBLE VIA THE API
-------------------------------------------
Checked with whatIfOrder against a live TWS on 2026-09-09. Any
non-integer quantity, on either candidate instrument, is refused:

    Error 10243: Fractional-sized order cannot be placed via API.
                 Please use desktop version to place this order.

Whole quantities pass the same pre-check (only a harmless 10349 "TIF was
set to DAY" note). This is an API-level restriction, NOT the account
permission and NOT the contract: IBKR reports sizeIncrement 0.0001 for
XS2D and will honour it for a human clicking in TWS desktop, but never
for an order sent over the wire. Enabling fractional trading on the
account does not change it.

So this bot buys whole shares, and `size_increment` is validated as a
whole number for that reason. Sizing an order at 1.3378 shares produces
a rejection, not a small position.

WHICH INSTRUMENT, GIVEN WHOLE SHARES ONLY
-----------------------------------------
The share price relative to the capital decides it, because whatever
rounding leaves behind sits in cash earning nothing and dilutes the
leverage. At 500 of capital:

    XS2D  2x at $362.51 -> 1 share,  72.5% invested -> 1.45x effective
    3USL  3x at $188.15 -> 2 shares, 75.3% invested -> 2.26x effective

Walking the account as cash + position (no intra-trade rebalance):

    2008-2026                CAGR    maxDD      2015-2026    CAGR    maxDD
    buy & hold index        9.41%    53.3%      index       12.04%   33.9%
    XS2D 2x, 1 share        7.34%    23.9%      XS2D 1 sh    7.60%   23.9%
    3USL 3x, 2 shares       9.91%    35.2%      3USL 2 sh   10.40%   35.2%

3USL wins on both windows, so it is the default: a cheaper share deploys
more of a small account, and that matters more here than the difference
between a 2x and a 3x fund. It is an ETN, so it carries issuer credit
risk that XS2D (a swap-based UCITS ETF) does not -- accepted knowingly,
because 1.45x of a 2x fund is the worse trade.

Raising the capital is what improves this, not switching instrument: ten
3USL shares is ~$1,900 at 91.7% invested and 2.75x.

AND THE HONEST CAVEAT, ONCE
---------------------------
At 500 of capital the default returns 10.40% against the index's 12.04%
over 2015-2026, with a 35.2% drawdown against its 33.9% -- less return
AND slightly more drawdown. Over the full sample it is 9.91% against
9.41% at 35.2% against 53.3%, which is the better trade, but that edge
leans on sitting out 2008-09. On this data a plain index fund is the
better bet at this size. It is wired because it was asked for.

EU ACCESS: NOT UPRO
-------------------
UPRO and SSO are US-domiciled, have no PRIIPs KID, and are blocked for
EU retail accounts. Both instruments above are European listings, on
venue LSEETF -- NOT 'LSE', which returns no security definition.
Leveraged ETPs usually require a broker appropriateness test, and
fractional orders require fractional trading to be enabled on the
account. Run --check after any change to the instrument; it prints the
increment-aware sizing and the leverage actually taken on.

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
    # The leveraged ETP actually traded. Resolved against a live TWS on
    # 2026-09-09: WisdomTree S&P 500 3x Daily Leveraged, conId 118833789,
    # ~$188. The venue is LSEETF, NOT 'LSE' -- 'LSE' returns no security
    # definition. An ETN, so it carries issuer credit risk.
    #
    # 3x rather than the 2x XS2D because fractional orders are refused
    # over the API (see the docstring): at whole shares a $188 price
    # deploys 75% of a 500 account for 2.26x, while XS2D's $362 deploys
    # 72% of it for only 1.45x.
    "trade_symbol": "3USL",
    "trade_exchange": "LSEETF",
    "trade_currency": "USD",
    "trade_primary_exchange": "",
    "trade_leverage": 3,
    # Order-size granularity. Must be a WHOLE number: IBKR error 10243
    # refuses any fractional quantity sent over the API, whatever
    # sizeIncrement the contract advertises.
    "size_increment": 1.0,
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
    increment = rules["size_increment"]
    if increment < 1 or increment != int(increment):
        raise ValueError(
            f"size_increment must be a whole number >= 1, got {increment} -- IBKR "
            f"error 10243 refuses any fractional quantity sent over the API, "
            f"whatever sizeIncrement the contract advertises. A lot-size "
            f"instrument may legitimately want 10 or 100")
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
    held = float(position.get("shares", 0.0)) if position else 0.0
    shares = held if out["action"] == "sell" else 0.0
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
    """How much to buy, rounded DOWN to the instrument's order-size
    increment, and how much of the capital that leaves unused.

    Returns {"shares", "notional", "idle_cash", "idle_pct"}.

    `size_increment` is the order granularity, and it must be a whole
    number: IBKR error 10243 refuses fractional quantities over the API
    regardless of the contract's advertised sizeIncrement. It stays a
    parameter rather than a hardcoded 1 because lot-traded instruments
    legitimately need 10 or 100, and because writing the constraint down
    is what stops 0.0001 being tried again.

    The idle figure is returned rather than merely tolerated, because
    whenever the increment is coarse relative to the capital it is the
    dominant sizing error and it silently reduces leverage.

    Raises:
        ValueError: if price is not positive. A zero price means the
            quote failed, and sizing off it would ask for a colossal
            position.
    """
    if price <= 0:
        raise ValueError(f"price must be > 0 to size a position, got {price}")
    increment = rules["size_increment"]
    budget = capital * rules["sizing_buffer"]
    # Floor to a whole number of increments. The epsilon absorbs binary
    # representation error -- without it a budget that divides exactly
    # loses a full increment, since 485/357.33/0.0001 evaluates to
    # 13572.999999... rather than 13573.
    units = math.floor(budget / price / increment + 1e-9)
    shares = min(max(units, 0) * increment, float(rules["max_shares"]))
    # Re-quantise after the max_shares clamp, which can land off-grid,
    # and drop float dust that would otherwise reach the order as
    # 1.3572000000000002.
    shares = round(math.floor(shares / increment + 1e-9) * increment, 8)
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
