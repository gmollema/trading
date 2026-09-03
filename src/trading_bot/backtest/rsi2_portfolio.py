"""Equal-weight multi-symbol portfolio simulation for RSI(2) dip buying.

This is the step that turns the universe expectancy result into a return.
The per-symbol study found that buying only the second or third dip earns
1.94x / 3.84x buy-and-hold's rate PER DAY OF EXPOSURE, but is invested
only 3% / 0.5% of days -- so on one name it compounds to nearly nothing.
Concurrency across ~500 names is what could convert that rate into a
portfolio return, and whether it does depends entirely on things a
per-symbol study cannot see: how much the signals overlap in time, how
often the position cap binds, and how much capital sits idle.

Design decisions that materially affect the answer, all explicit:

  Capital per slot. Equity is divided into `max_slots` equal notional
  slots, sized off equity at the moment of entry. A slot is idle cash
  when unused. This is deliberately the simple rule -- no volatility
  targeting, no Kelly -- because the question is whether the edge
  survives basic portfolio mechanics, not whether it can be optimised.

  Whole shares. `floor(alloc / price)`, and a signal that cannot afford
  one share is skipped. Fractional shares would flatter small accounts.

  Cap contention. When more signals arrive than there are free slots,
  `priority` decides. "rsi" takes the most oversold first, which is
  motivated by this strategy's own finding that deeper is better but is
  a choice fitted to that finding; "symbol" takes them alphabetically,
  which is arbitrary but unbiased. Run both -- if they disagree, the
  result depends on the selection rule rather than the signal.

  No lookahead. A signal is computed from bars up to and including the
  entry bar and filled at that bar's close, which is a market-on-close
  order and is what the source video specifies. `entry_next_open` defers
  every fill to the symbol's following open instead; the difference
  between the two is a direct measure of how much the result leans on
  MOC timing.

  Carry-forward marking. Symbols do not all trade every date on the
  master calendar. An open position with no bar on a given date is marked
  at its last known close rather than dropped from equity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from trading_bot.backtest.portfolio import TIERED_COMMISSION_MIN, TIERED_COMMISSION_PER_SHARE
from trading_bot.backtest.rsi2_signals import (
    DEFAULT_ENTRY_LEVEL,
    DEFAULT_EXIT_LEVEL,
    DEFAULT_RSI_PERIOD,
    DEFAULT_SMA_PERIOD,
    rsi2_dip_sequence,
    wilder_rsi,
)

DEFAULT_MAX_SLOTS = 20
DEFAULT_SLIPPAGE_BPS = 2.5  # per side
TRADING_DAYS = 252


@dataclass
class SymbolData:
    """One symbol's bars plus everything precomputed from them."""

    symbol: str
    dates: list
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    dip: list[int]
    overbought: list[bool]
    index_of: dict = field(default_factory=dict)


def prepare(symbol: str, bars: dict, rsi_period: int = DEFAULT_RSI_PERIOD,
            entry_level: float = DEFAULT_ENTRY_LEVEL, exit_level: float = DEFAULT_EXIT_LEVEL,
            sma_period: int = DEFAULT_SMA_PERIOD) -> SymbolData:
    closes = bars["close"]
    rsi = wilder_rsi(closes, rsi_period)
    return SymbolData(
        symbol=symbol,
        dates=bars["date"],
        opens=bars["open"],
        highs=bars["high"],
        lows=bars["low"],
        closes=closes,
        dip=rsi2_dip_sequence(closes, rsi_period, entry_level, exit_level, sma_period),
        overbought=[r is not None and r > exit_level for r in rsi],
        index_of={d: i for i, d in enumerate(bars["date"])},
    )


def commission(shares: int) -> float:
    """IBKR tiered US-stock schedule, charged per fill."""
    return max(shares * TIERED_COMMISSION_PER_SHARE, TIERED_COMMISSION_MIN)


def run_portfolio(
    data: list[SymbolData],
    calendar: list,
    initial_capital: float = 100_000.0,
    max_slots: int = DEFAULT_MAX_SLOTS,
    first_dip: int = 1,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    cash_yield_pct: float = 0.0,
    priority: str = "rsi",
    entry_next_open: bool = False,
    warmup: int = DEFAULT_SMA_PERIOD,
) -> dict:
    """Walk the master calendar once, holding at most `max_slots` names.

    Returns {"equity_curve", "trades", "final_equity", "stats"} where
    equity_curve is one {date, equity, positions, deployed_pct} per
    calendar date.
    """
    if priority not in ("rsi", "symbol"):
        raise ValueError(f"unknown priority {priority!r}")
    if max_slots < 1:
        raise ValueError("max_slots must be at least 1")

    cash = initial_capital
    open_pos: dict[str, dict] = {}
    pending: list[tuple[str, int]] = []
    trades: list[dict] = []
    curve: list[dict] = []
    last_close: dict[str, float] = {}
    slip = slippage_bps / 10_000.0
    daily_cash_rate = (cash_yield_pct / 100.0) / TRADING_DAYS
    cap_bound_days = 0
    signal_days = 0
    skipped_no_slot = 0
    skipped_too_small = 0

    by_symbol = {d.symbol: d for d in data}

    def slot_value() -> float:
        """Notional per slot, off CURRENT equity so the book compounds.

        Equity here is cash plus open positions marked at their last known
        close -- not `cash` alone, which would shrink the slot as the book
        fills up and make position size depend on entry order.
        """
        held = sum(p["shares"] * last_close.get(s, p["entry_price"]) for s, p in open_pos.items())
        return (cash + held) / max_slots

    def _try_open(sd: SymbolData, i: int, price: float, dip_num: int, date) -> None:
        nonlocal cash, skipped_too_small
        fill = price * (1 + slip)
        shares = math.floor(slot_value() / fill)
        if shares < 1:
            skipped_too_small += 1
            return
        fee = commission(shares)
        cost = shares * fill + fee
        if cost > cash:
            shares = math.floor((cash - TIERED_COMMISSION_MIN) / fill)
            if shares < 1:
                skipped_too_small += 1
                return
            fee = commission(shares)
            cost = shares * fill + fee
        cash -= cost
        last_close[sd.symbol] = price
        open_pos[sd.symbol] = {
            "shares": shares,
            "entry_price": fill,
            "entry_date": date,
            "entry_fee": fee,
            "dip": dip_num,
            "bars_held": 1,
        }

    def close_position(sd: SymbolData, i: int, price: float, reason: str, date) -> None:
        nonlocal cash
        pos = open_pos.pop(sd.symbol)
        fill = price * (1 - slip)
        fee = commission(pos["shares"])
        cash += pos["shares"] * fill - fee
        trades.append({
            "symbol": sd.symbol,
            "entry_date": pos["entry_date"],
            "exit_date": date,
            "entry_price": pos["entry_price"],
            "exit_price": fill,
            "shares": pos["shares"],
            "dip": pos["dip"],
            "bars_held": pos["bars_held"],
            "pnl": pos["shares"] * (fill - pos["entry_price"]) - fee - pos["entry_fee"],
            "reason": reason,
        })

    for date in calendar:
        if cash_yield_pct:
            cash *= 1 + daily_cash_rate

        # --- 1. exits, and pending fills from the previous bar ---
        for symbol in list(open_pos):
            sd = by_symbol[symbol]
            i = sd.index_of.get(date)
            if i is None:
                continue
            open_pos[symbol]["bars_held"] += 1
            if sd.overbought[i]:
                close_position(sd, i, sd.closes[i], "rsi_exit", date)

        filled_pending: list[tuple[str, int]] = []
        for symbol, dip_num in pending:
            sd = by_symbol[symbol]
            i = sd.index_of.get(date)
            if i is None:
                continue
            filled_pending.append((symbol, dip_num))
            if symbol in open_pos or len(open_pos) >= max_slots:
                skipped_no_slot += 1
                continue
            _try_open(sd, i, sd.opens[i], dip_num, date)
        pending = [p for p in pending if p not in filled_pending]

        # --- 2. mark to market ---
        equity = cash
        deployed = 0.0
        for symbol, pos in open_pos.items():
            sd = by_symbol[symbol]
            i = sd.index_of.get(date)
            price = sd.closes[i] if i is not None else last_close.get(symbol, pos["entry_price"])
            last_close[symbol] = price
            value = pos["shares"] * price
            equity += value
            deployed += value

        # --- 3. new signals ---
        candidates = []
        for sd in data:
            i = sd.index_of.get(date)
            if i is None or i < warmup or sd.symbol in open_pos:
                continue
            if sd.dip[i] >= first_dip and sd.dip[i] > 0:
                candidates.append((sd, i))

        if candidates:
            signal_days += 1
            if priority == "rsi":
                # Deeper dip number first, then the lower close-to-SMA
                # ratio as a stable tiebreak; symbol last so the order is
                # fully deterministic.
                candidates.sort(key=lambda c: (-c[0].dip[c[1]], c[0].symbol))
            else:
                candidates.sort(key=lambda c: c[0].symbol)

            free = max_slots - len(open_pos)
            if len(candidates) > free:
                cap_bound_days += 1
            for sd, i in candidates:
                if len(open_pos) >= max_slots:
                    skipped_no_slot += 1
                    continue
                if entry_next_open:
                    pending.append((sd.symbol, sd.dip[i]))
                else:
                    _try_open(sd, i, sd.closes[i], sd.dip[i], date)

        curve.append({
            "date": date,
            "equity": equity,
            "positions": len(open_pos),
            "deployed_pct": (deployed / equity * 100) if equity > 0 else 0.0,
        })

    # close anything still open, at its last known price
    for symbol in list(open_pos):
        sd = by_symbol[symbol]
        price = last_close.get(symbol, open_pos[symbol]["entry_price"])
        close_position(sd, len(sd.closes) - 1, price, "end_of_data", calendar[-1])

    final = cash + sum(
        pos["shares"] * last_close.get(s, pos["entry_price"]) for s, pos in open_pos.items()
    )
    return {
        "equity_curve": curve,
        "trades": trades,
        "final_equity": final,
        "stats": {
            "cap_bound_days": cap_bound_days,
            "signal_days": signal_days,
            "skipped_no_slot": skipped_no_slot,
            "skipped_too_small": skipped_too_small,
        },
    }


def max_drawdown_pct(curve: list[dict]) -> float:
    if not curve:
        return 0.0
    peak = curve[0]["equity"]
    worst = 0.0
    for p in curve:
        peak = max(peak, p["equity"])
        if peak > 0:
            worst = max(worst, (peak - p["equity"]) / peak * 100)
    return worst


def cagr_pct(initial: float, final: float, years: float) -> float:
    if initial <= 0 or final <= 0 or years <= 0:
        return 0.0
    return ((final / initial) ** (1 / years) - 1) * 100


def equal_weight_buy_hold(data: list[SymbolData], calendar: list, initial_capital: float,
                          warmup: int = DEFAULT_SMA_PERIOD) -> list[dict]:
    """Benchmark: split capital equally across every symbol that has a bar
    at the first tradeable date, buy at that date's close, hold to the end.

    No rebalancing, whole shares, no costs -- a deliberately generous
    benchmark, so beating it means something.
    """
    if len(calendar) <= warmup:
        return []
    start_date = calendar[warmup]
    eligible = [sd for sd in data if start_date in sd.index_of]
    if not eligible:
        return []
    alloc = initial_capital / len(eligible)
    holdings = {}
    cash = initial_capital
    for sd in eligible:
        price = sd.closes[sd.index_of[start_date]]
        shares = math.floor(alloc / price)
        if shares > 0:
            holdings[sd.symbol] = shares
            cash -= shares * price

    by_symbol = {sd.symbol: sd for sd in data}
    last = {}
    curve = []
    for date in calendar[warmup:]:
        equity = cash
        for symbol, shares in holdings.items():
            sd = by_symbol[symbol]
            i = sd.index_of.get(date)
            price = sd.closes[i] if i is not None else last.get(symbol)
            if price is None:
                continue
            last[symbol] = price
            equity += shares * price
        curve.append({"date": date, "equity": equity})
    return curve
