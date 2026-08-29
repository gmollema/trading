"""Pure position-sizing and exit-state-machine logic for the backtest engine.

Ports cycle.entry_scan's risk-based sizing math (cycle.py:763-766) and
cycle.manage_position's pre_breakeven -> post_breakeven exit state machine
(cycle.py:351-496), adapted from a live poll (current market price, real
broker stop orders) to a bar-close simulation: a stop-loss "fill" is
simulated whenever a bar's Low <= the current stop price, at the stop price
itself -- a standard backtest simplification that ignores slippage/
gap-through risk.
"""

from __future__ import annotations

import math

import pandas as pd

from trading_bot.cli.cycle import compute_swing_lows


DEFAULT_COMMISSION_PER_SHARE = 0.005
DEFAULT_COMMISSION_MIN = 1.0

# IBKR's TIERED plan base rate (monthly volume <= 300k shares):
# $0.0035/share with a $0.35 per-order minimum. Exchange/clearing/
# regulatory pass-throughs come on top, so the real all-in cost lands
# between this and roughly $0.005/share with a $0.50 minimum -- pass the
# conservative pair when bracketing. Why it matters for THIS codebase:
# the small-account failure mode measured for the SMC strategy is the
# flat per-order minimum dominating tiny fills, so Fixed -> Tiered
# roughly HALVES the viable capital floor (backtested 2026-07: ~$10K ->
# ~$5K at a 30% position cap, positive on all three cached years even
# under the conservative variant; $1K stays unviable under every
# schedule tested, including zero commission).
TIERED_COMMISSION_PER_SHARE = 0.0035
TIERED_COMMISSION_MIN = 0.35


def commission(qty: float, per_share: float = DEFAULT_COMMISSION_PER_SHARE, minimum: float = DEFAULT_COMMISSION_MIN) -> float:
    """IBKR's standard WHOLE-SHARE US-stock commission schedule: $/share
    with a per-order minimum (defaults: $0.005/share, $1.00 minimum,
    IBKR's Fixed plan). Applies once per FILL (each BUY or SELL
    execution), not once per round-trip trade -- a position closed via a
    partial + a final fill pays this twice on the way out, matching real
    order-by-order billing. Do NOT use this for fractional-share fills --
    see fractional_commission, which follows IBKR's actual (different)
    published schedule for those."""
    return max(qty * per_share, minimum)


DEFAULT_FRACTIONAL_COMMISSION_PCT = 0.01
DEFAULT_FRACTIONAL_COMMISSION_MIN = 0.01


def fractional_commission(
    qty: float,
    price: float,
    pct_of_notional: float = DEFAULT_FRACTIONAL_COMMISSION_PCT,
    minimum: float = DEFAULT_FRACTIONAL_COMMISSION_MIN,
) -> float:
    """IBKR's ACTUAL published fractional-share commission schedule --
    confirmed via IBKR's own commissions pages (2026-04): the greater of
    1% of trade value or $0.01 per fill. This is a completely different
    structure from the whole-share per-share/flat-minimum schedule above
    (which is what `commission()` models) -- fractional fills are NOT
    just "the per-share rate applied to a non-integer qty"."""
    return max(qty * price * pct_of_notional, minimum)


# IBKR IDEALPRO forex commission, Tier I (retail; applies up to $1B of
# combined monthly spot-currency trade value, which covers every account
# this codebase is built for): 0.20 basis points of trade notional, with
# a $2.00 minimum per order. Verified against IBKR's published forex
# commission schedule (2026-07). This is a completely different cost
# structure from the per-share equity schedules above (commission() /
# fractional_commission()) -- FX commission is priced off notional trade
# value, not share count, so those must not be reused for FX fills.
FX_COMMISSION_BPS = 0.20
FX_COMMISSION_MIN_USD = 2.00


def fx_commission(notional: float, bps: float = FX_COMMISSION_BPS, minimum: float = FX_COMMISSION_MIN_USD) -> float:
    """IBKR IDEALPRO's forex commission: bps/10000 * notional, or
    `minimum`, whichever is greater. Applies once per FILL (each BUY or
    SELL execution), same per-execution billing as commission()/
    fractional_commission() above.

    `notional` should be qty * price in USD. For a USD-quote pair
    (EURUSD, GBPUSD, AUDUSD, NZDUSD -- price is USD per unit of base
    currency) qty * price IS the USD notional exactly. For a USD-BASE
    pair (USDJPY, USDCAD, USDCHF -- price is quote-currency per USD),
    qty * price is actually quote-currency notional, not USD -- true to a
    real IBKR fill this would need converting at the prevailing USD/quote
    rate, which this backtest's engines don't do (same simplification
    ut_bot_engine's position sizing already makes when it treats risk
    distance in "price units" as USD regardless of quote convention)."""
    return max(notional * bps / 10000, minimum)


FX_PIP_SIZE = 0.0001
FX_PIP_SIZE_JPY = 0.01

# UNLIKE FX_COMMISSION_BPS above, this is NOT a published, verified
# schedule -- IBKR IDEALPRO passes through live interbank quotes, so the
# spread actually paid on any given fill varies continuously with
# liquidity, time of day, and volatility; there is no fixed rate card for
# it the way there is for commission. These are commonly-cited ballpark
# figures for institutional/ECN-tier pricing (the tier IDEALPRO
# approximates), cross-referenced across several public spread-comparison
# sources (2026-07) -- a reasonable planning assumption for "is this
# strategy's edge bigger than a plausible spread cost", NOT a guarantee
# of what any specific fill will actually cost. Treat real execution data
# as authoritative over this table whenever it's available.
FX_TYPICAL_SPREAD_PIPS = {
    "EURUSD": 0.2,
    "GBPUSD": 0.4,
    "USDJPY": 0.25,
    "AUDUSD": 0.5,
    "USDCAD": 0.6,
    "USDCHF": 0.6,
    "NZDUSD": 0.8,
}


def fx_pip_size(symbol: str) -> float:
    """0.01 for JPY-quoted pairs (e.g. USDJPY), 0.0001 for every other
    major -- the standard FX convention for what one "pip" is worth in
    price terms."""
    return FX_PIP_SIZE_JPY if symbol.upper().endswith("JPY") else FX_PIP_SIZE


def fx_half_spread_price(symbol: str, spread_pips: float) -> float:
    """Half of `spread_pips` converted to price units for `symbol` -- the
    per-side cost an aggressive fill pays crossing the spread (see
    fx_fill_price). `spread_pips` is always caller-supplied (e.g. from
    FX_TYPICAL_SPREAD_PIPS, or real observed spread data) rather than
    defaulted here, since -- unlike commission -- there's no single
    "correct" number to fall back to silently."""
    return spread_pips * fx_pip_size(symbol) / 2


def fx_fill_price(mid_price: float, side: str, half_spread: float) -> float:
    """The realistic fill price for an aggressive/market order crossing
    the spread: a BUY fills at the ask (half_spread ABOVE mid), a SELL
    fills at the bid (half_spread BELOW mid). `side` is the order's own
    direction (BUY/SELL), not "entry vs exit" or "long vs short" -- a
    short's opening SELL crosses down to the bid exactly like a long's
    closing SELL does."""
    return mid_price + half_spread if side == "BUY" else mid_price - half_spread


def position_size(
    portfolio_value: float,
    risk_pct: float,
    price: float,
    initial_stop: float,
    max_position_pct: float,
    allow_fractional: bool = False,
) -> float:
    """Risk-based share count, capped at `max_position_pct` of portfolio value.

    Mirrors cycle.entry_scan: size = min(risk_dollars / R, max_position_pct%
    of portfolio_value / price). Returns 0 if R <= 0 (non-positive risk
    distance) or the capped size rounds/clips to <= 0.

    allow_fractional: when False (the default -- matches the live bot's
        actual current capability, whole-share orders only), the result
        is floored to a whole share count (an int). When True, returns
        the raw fractional share count instead, for brokers/backtests
        that support fractional-share orders -- e.g. IBKR fractional
        shares, which sidestep small accounts rounding most signals down
        to 0 shares in the first place.
    """
    r_per_share = price - initial_stop
    if r_per_share <= 0:
        return 0
    risk_dollars = portfolio_value * (risk_pct / 100)
    size_by_risk = risk_dollars / r_per_share
    size_by_cap = portfolio_value * (max_position_pct / 100) / price
    size = max(0.0, min(size_by_risk, size_by_cap))
    return size if allow_fractional else math.floor(size)


def initial_stop_from_lod(today_lod: float) -> float:
    """rules.json's exit.initial_stop_rule: "lod_minus_1pct"."""
    return today_lod * 0.99


def open_position(symbol: str, price: float, today_lod: float, qty: int) -> dict:
    """Build a new open-position record.

    Initial stop at LOD-1% (rules.json's exit.initial_stop_rule:
    "lod_minus_1pct"); state machine starts at pre_breakeven, matching
    cycle.entry_scan's new_pos shape.
    """
    initial_stop = initial_stop_from_lod(today_lod)
    r = price - initial_stop
    return {
        "symbol": symbol,
        "entry_price": price,
        "qty": qty,
        "initial_stop": initial_stop,
        "current_stop_price": initial_stop,
        "state": "pre_breakeven",
        "R": r,
    }


def manage_position(pos: dict, bar: dict, recent_lows: list[float], exit_cfg: dict) -> tuple[dict | None, list[dict]]:
    """Evaluate one bar (open/high/low/close) against an open position.

    Args:
        pos: an open-position record (see `open_position`).
        bar: {"low", "close"} for the current bar.
        recent_lows: that symbol's Low prices from today's session open
            through this bar inclusive -- used only in the post_breakeven
            trailing-stop branch (compute_swing_lows needs 2 bars on each
            side of a candidate low, so this list can lag by up to 2 bars,
            exactly like the live bot's own bars-fetched-so-far behavior).
        exit_cfg: rules.json's "exit" section.

    Returns:
        (updated_pos_or_None, fills) -- pos is None once qty reaches 0
        (fully closed this bar); fills is a list of {"qty", "price",
        "reason"} dicts for any SELL simulated on this bar.
    """
    fills: list[dict] = []

    stop_price = pos["current_stop_price"]
    if bar["low"] <= stop_price:
        fills.append({"qty": pos["qty"], "price": stop_price, "reason": "stop"})
        return None, fills

    entry_price = pos["entry_price"]
    r = pos["R"]
    state = pos["state"]
    price = bar["close"]

    if state == "pre_breakeven":
        partial_trigger_r = exit_cfg["partial_profit_trigger_R"]
        breakeven_trigger_r = exit_cfg["breakeven_trigger_R"]
        partial_fraction = exit_cfg.get("partial_profit_fraction", 1 / 3)

        reached_partial = price >= entry_price + partial_trigger_r * r
        reached_breakeven = price >= entry_price + breakeven_trigger_r * r

        if reached_partial:
            partial_qty = min(math.ceil(pos["qty"] * partial_fraction), pos["qty"])
            fills.append({"qty": partial_qty, "price": price, "reason": "partial_profit"})
            remaining_qty = pos["qty"] - partial_qty
            if remaining_qty <= 0:
                return None, fills
            # A fast move that jumps past both triggers within one bar still
            # gets the better (breakeven) stop, not just the discounted one.
            new_stop = entry_price if reached_breakeven else entry_price * 0.99
            pos = {
                **pos,
                "qty": remaining_qty,
                "current_stop_price": new_stop,
                "state": "post_breakeven_partial_done",
            }
        elif reached_breakeven:
            # Only reachable if rules.json ever configures
            # breakeven_trigger_R < partial_profit_trigger_R -- kept as a
            # defensive fallback for that ordering, matching cycle.py.
            pos = {**pos, "current_stop_price": entry_price, "state": "post_breakeven_no_partial"}

    elif state.startswith("post_breakeven"):
        swing_lows = compute_swing_lows(pd.DataFrame({"Low": recent_lows}))
        if swing_lows:
            candidate_stop = swing_lows[-1] - 0.01
            if candidate_stop > pos["current_stop_price"]:
                pos = {**pos, "current_stop_price": candidate_stop}

    return pos, fills


def close_position(pos: dict, price: float, reason: str = "force_close") -> dict:
    """Force-close the remaining quantity of `pos` at `price` (e.g. the
    15:51 ET EOD flatten, or the end of the available data)."""
    return {"qty": pos["qty"], "price": price, "reason": reason}
