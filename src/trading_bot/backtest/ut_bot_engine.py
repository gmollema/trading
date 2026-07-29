"""Single-symbol backtest simulation for the UT Bot ATR trailing-stop
strategy (see ut_bot_signals.py for the signal logic itself).

Simpler than smc_engine.py/engine.py on purpose: this strategy trades ONE
instrument with at most one open position at a time (the signal walk in
ut_bot_signals.find_ut_bot_long_trades already enforces that), so there
is no portfolio-level chronological merge to do -- just risk-based sizing
per trade, reusing the exact same portfolio.position_size math the other
two strategies use.

Commission: modeled via portfolio.fx_commission (IBKR IDEALPRO's actual
published forex schedule -- 0.20 bps of notional, $2 minimum per order),
NOT the per-share schedules portfolio.commission / .fractional_commission
implement for equities -- FX commission is priced off notional trade
value, not share count, so reusing those would silently apply the wrong
cost model. Off by default (fx_commission_bps=None), matching how SMC's
commission modeling defaults off too -- every result reported for this
strategy before this was added is on a zero-cost basis.
"""

from __future__ import annotations

from trading_bot.backtest import portfolio
from trading_bot.backtest.ut_bot_signals import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_KEY_VALUE,
    find_ut_bot_confirmed_trades,
    find_ut_bot_long_trades,
)

DEFAULT_MAX_POSITION_PCT = 100.0  # single instrument, no diversification to preserve


def _trade_row(timestamp, symbol: str, side: str, qty: float, price: float, order_id: int, reason: str) -> dict:
    size = qty if isinstance(qty, float) and not qty.is_integer() else int(qty)
    return {
        "timestamp_iso": timestamp.tz_convert("UTC").isoformat(),
        "symbol": symbol,
        "side": side,
        "size": size,
        "fill_price": round(float(price), 6),  # FX rates need more precision than 2dp
        "order_id": order_id,
        "status": "Filled",
        "reason": reason,
    }


def run_ut_bot_backtest(
    bars: dict,
    initial_capital: float,
    symbol: str = "GBPUSD",
    risk_pct: float = 1.0,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    key_value: float = DEFAULT_KEY_VALUE,
    atr_period: int = DEFAULT_ATR_PERIOD,
    allow_fractional_shares: bool = False,
    fx_commission_bps: float | None = None,
    fx_commission_min: float = portfolio.FX_COMMISSION_MIN_USD,
) -> dict:
    """Simulate the UT Bot strategy over one symbol's bars (see
    ut_bot_signals.find_ut_bot_long_trades for `bars`' shape). Returns
    {"trades": [...], "equity_curve": [...]}; `trades` uses the same
    schema as trade.CSV_HEADER plus a `reason` field, matching the other
    two strategies' engines so cli/compute_perf.pair_trades_fifo works
    unchanged.

    Each trade's risk distance (R) is entry_price - stop_at_entry (the
    ATR trailing-stop line's value at the moment of entry) -- the
    trailing stop itself IS this strategy's exit mechanism, so there is
    no separate stop-loss order to size against.

    fx_commission_bps: IBKR IDEALPRO's forex commission (see
        portfolio.fx_commission), charged once per FILL (the entry BUY
        and the exit SELL, so every round trip pays it twice) against
        size * fill_price as the notional. Left None (the default), no
        commission is modeled at all -- matches every result already
        reported for this strategy and mirrors smc_engine.py's
        commission_per_share=None default.
    """
    signal_trades = find_ut_bot_long_trades(bars, key_value, atr_period)

    equity = float(initial_capital)
    trades_out: list[dict] = []
    equity_curve: list[dict] = []
    order_id = 0

    for trade in signal_trades:
        entry_price = trade["entry_price"]
        stop_price = trade["stop_at_entry"]
        size = portfolio.position_size(equity, risk_pct, entry_price, stop_price, max_position_pct, allow_fractional_shares)
        min_size = 1e-9 if allow_fractional_shares else 1
        if size < min_size:
            continue

        order_id += 1
        if fx_commission_bps is not None:
            equity -= portfolio.fx_commission(size * entry_price, fx_commission_bps, fx_commission_min)
        trades_out.append(_trade_row(trade["entry_date"], symbol, "BUY", size, entry_price, order_id, "entry"))
        equity_curve.append({"timestamp": trade["entry_date"].isoformat(), "equity": equity})

        equity += (trade["exit_price"] - entry_price) * size
        if fx_commission_bps is not None:
            equity -= portfolio.fx_commission(size * trade["exit_price"], fx_commission_bps, fx_commission_min)
        order_id += 1
        trades_out.append(
            _trade_row(trade["exit_date"], symbol, "SELL", size, trade["exit_price"], order_id, trade["reason"])
        )
        equity_curve.append({"timestamp": trade["exit_date"].isoformat(), "equity": equity})

    return {"trades": trades_out, "equity_curve": equity_curve}


def _position_size_for_side(
    equity: float, risk_pct: float, entry_price: float, stop_price: float, max_position_pct: float,
    allow_fractional_shares: bool,
) -> float:
    """portfolio.position_size assumes a long convention (stop below
    price); a short's stop sits ABOVE price, which would make
    entry_price - stop_price negative and always size to 0. Only the
    RISK DISTANCE matters to that function, not direction, so pass a
    synthetic stop the same distance below price to get correct sizing
    for either side without duplicating the risk/cap math here."""
    distance = abs(entry_price - stop_price)
    synthetic_stop = entry_price - distance
    return portfolio.position_size(equity, risk_pct, entry_price, synthetic_stop, max_position_pct, allow_fractional_shares)


def run_ut_bot_confirmed_backtest(
    bars: dict,
    initial_capital: float,
    symbol: str = "GBPUSD",
    risk_pct: float = 1.0,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    key_value: float = DEFAULT_KEY_VALUE,
    atr_period: int = DEFAULT_ATR_PERIOD,
    allow_fractional_shares: bool = False,
    fx_commission_bps: float | None = None,
    fx_commission_min: float = portfolio.FX_COMMISSION_MIN_USD,
) -> dict:
    """Long+short simulation of the user-specified "confirmed entry" UT
    Bot variant (see ut_bot_signals.find_ut_bot_confirmed_trades).

    Returns {"trades": [...], "equity_curve": [...], "summary": {...}}.
    `trades` uses the trade.CSV_HEADER-compatible schema (BUY opens a
    long / covers a short's... no: BUY opens a long, SELL closes it;
    SELL opens a short, BUY closes it -- side reflects the actual order
    direction, not "entry vs exit"), for logging/live-parity. `summary`
    is computed DIRECTLY from the signal-level trade list, not via
    cli.compute_perf.pair_trades_fifo -- that helper assumes a long-only
    FIFO (every BUY opens, every SELL closes) and would silently
    mispair a short's opening SELL against an unrelated long.

    fx_commission_bps: see run_ut_bot_backtest's identical parameter.
        Charged once per fill against size * fill_price notional and
        netted directly into each trade's closed pnl (so `summary`'s
        gross_pnl_usd/profit_factor/etc. are cost-inclusive), not just
        deducted from equity on the side.
    """
    signal_trades = find_ut_bot_confirmed_trades(bars, key_value, atr_period)

    equity = float(initial_capital)
    trades_out: list[dict] = []
    equity_curve: list[dict] = []
    closed_pnls: list[float] = []
    order_id = 0

    for trade in signal_trades:
        entry_price = trade["entry_price"]
        stop_price = trade["stop_at_entry"]
        side = trade["side"]
        size = _position_size_for_side(equity, risk_pct, entry_price, stop_price, max_position_pct, allow_fractional_shares)
        min_size = 1e-9 if allow_fractional_shares else 1
        if size < min_size:
            continue

        entry_side, exit_side = ("BUY", "SELL") if side == "long" else ("SELL", "BUY")
        order_id += 1
        trades_out.append(_trade_row(trade["entry_date"], symbol, entry_side, size, entry_price, order_id, f"entry_{side}"))
        equity_curve.append({"timestamp": trade["entry_date"].isoformat(), "equity": equity})

        pnl = (trade["exit_price"] - entry_price) * size if side == "long" else (entry_price - trade["exit_price"]) * size
        if fx_commission_bps is not None:
            # Netted once, in full, at exit -- NOT also deducted from
            # equity at entry -- so it's counted exactly once per round
            # trip (this variant's summary is built from closed_pnls, not
            # equity deltas, so double-booking it here would silently
            # double-charge every trade). The entry equity_curve point
            # above therefore doesn't yet reflect this trade's cost; it
            # only shows up when the round trip closes.
            round_trip_commission = (
                portfolio.fx_commission(size * entry_price, fx_commission_bps, fx_commission_min)
                + portfolio.fx_commission(size * trade["exit_price"], fx_commission_bps, fx_commission_min)
            )
            pnl -= round_trip_commission
        equity += pnl
        closed_pnls.append(pnl)
        order_id += 1
        trades_out.append(
            _trade_row(trade["exit_date"], symbol, exit_side, size, trade["exit_price"], order_id, trade["reason"])
        )
        equity_curve.append({"timestamp": trade["exit_date"].isoformat(), "equity": equity})

    wins = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p < 0]
    gross_pnl = sum(closed_pnls)
    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    profit_factor = ("inf" if sum_wins > 0 else "n/a") if sum_losses == 0 else round(sum_wins / sum_losses, 2)
    summary = {
        "total_trades": len(closed_pnls),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(closed_pnls) * 100, 2) if closed_pnls else 0.0,
        "gross_pnl_usd": round(gross_pnl, 2),
        "avg_winner": round(sum_wins / len(wins), 2) if wins else 0.0,
        "avg_loser": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "profit_factor": profit_factor,
        "final_equity": round(equity, 2),
        "total_return_pct": round((equity / initial_capital - 1) * 100, 2) if initial_capital else 0.0,
    }

    return {"trades": trades_out, "equity_curve": equity_curve, "summary": summary}
