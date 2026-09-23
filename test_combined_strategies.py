"""Backtest RSI2 + MA 30/90 strategies combined."""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from trading_bot.backtest.rsi2_signals import (
    find_rsi2_long_trades, simple_moving_average, get_optimal_entry_rsi,
    get_optimal_sma, get_optimal_stop
)
from trading_bot.backtest.position_sizing import annual_dollars, position_size, window_years
from trading_bot.cli.rsi2_backtest import load_bars, in_window

def find_ma_trades(bars, ma_short=30, ma_long=90):
    """Find MA crossover trades."""
    closes = bars["close"]
    dates = bars["date"]
    ma_short_vals = simple_moving_average(closes, ma_short)
    ma_long_vals = simple_moving_average(closes, ma_long)

    trades = []
    in_trade = False
    entry_idx = None
    entry_price = None

    for i in range(1, len(closes)):
        if (ma_short_vals[i] is not None and ma_long_vals[i] is not None and
            ma_short_vals[i-1] is not None and ma_long_vals[i-1] is not None):

            if (not in_trade and ma_short_vals[i-1] <= ma_long_vals[i-1] and
                ma_short_vals[i] > ma_long_vals[i]):
                in_trade = True
                entry_idx = i
                entry_price = closes[i]
            elif (in_trade and ma_short_vals[i-1] >= ma_long_vals[i-1] and
                  ma_short_vals[i] < ma_long_vals[i]):
                exit_price = closes[i]
                points = exit_price - entry_price
                trades.append({
                    "entry_date": dates[entry_idx],
                    "entry_price": entry_price,
                    "exit_date": dates[i],
                    "exit_price": exit_price,
                    "points": points,
                    "strategy": "MA",
                })
                in_trade = False

    return trades

def test_symbol(symbol, label):
    bars = load_bars(symbol, Path("backtest_data/daily_index"))

    print(f"\n{'='*80}")
    print(f"{label} ({symbol})")
    print(f"{'='*80}\n")

    # RSI2 trades
    rsi2_trades = find_rsi2_long_trades(
        bars,
        rsi_period=2,
        entry_level=get_optimal_entry_rsi(symbol),
        exit_level=70.0,
        sma_period=get_optimal_sma(symbol),
        stop_points=get_optimal_stop(symbol),
        exit_mode="first_profitable_close",
        min_hold_days=12,
        exit_timing="close",
    )
    rsi2_windowed = [t for t in rsi2_trades if in_window(t["entry_date"], "2021-07-01", None)]
    for t in rsi2_windowed:
        t["strategy"] = "RSI2"

    # MA trades
    ma_trades = find_ma_trades(bars)
    ma_windowed = [t for t in ma_trades if in_window(t["entry_date"], "2021-07-01", None)]
    for t in ma_windowed:
        t["strategy"] = "MA"

    # Combined
    all_trades = sorted(rsi2_windowed + ma_windowed, key=lambda t: t["entry_date"])

    # Calculate stats
    def calc_stats(trades, name):
        if not trades:
            return
        net = sum(t["points"] for t in trades)
        wins = len([t for t in trades if t["points"] > 0])
        win_pct = (wins / len(trades) * 100) if trades else 0
        avg = net / len(trades)
        years = window_years("2021-07-01", bars["date"][-1])
        yearly = len(trades) / years
        annual = annual_dollars(trades, symbol, years)

        print(f"{name}:")
        print(f"  Trades: {len(trades)}, Net: {net:.0f} pts, Win%: {win_pct:.0f}%, Avg: {avg:.0f} pts/trade")
        print(f"  Annual: {yearly:.1f} trades/yr = ${annual:.2f}/yr at ${position_size(symbol):.2f} per trade\n")

        return annual

    rsi2_annual = calc_stats(rsi2_windowed, "RSI2")
    ma_annual = calc_stats(ma_windowed, "MA 30/90")
    combined_annual = calc_stats(all_trades, "COMBINED")

    print(f"Summary:")
    print(f"  RSI2 alone: ${rsi2_annual:.2f}/yr")
    print(f"  MA alone: ${ma_annual:.2f}/yr")
    print(f"  BOTH together: ${combined_annual:.2f}/yr")
    print(f"  Combined benefit: ${combined_annual - rsi2_annual - ma_annual:+.2f}/yr")

# Test both indices
test_symbol("^GSPC", "S&P 500")
test_symbol("^IXIC", "Nasdaq 100")
