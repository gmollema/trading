"""Backtest Moving Average Crossover strategy (50/200 MA).

Entry: 50-MA crosses above 200-MA
Exit: 50-MA crosses below 200-MA
"""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from trading_bot.backtest.rsi2_signals import simple_moving_average
from trading_bot.backtest.position_sizing import annual_dollars, position_size, window_years
from trading_bot.cli.rsi2_backtest import load_bars, in_window

def find_ma_crossover_trades(bars, ma_short=50, ma_long=200):
    """Find trades based on MA crossover."""
    closes = bars["close"]
    dates = bars["date"]

    # Calculate moving averages
    ma_short_vals = simple_moving_average(closes, ma_short)
    ma_long_vals = simple_moving_average(closes, ma_long)

    trades = []
    in_trade = False
    entry_idx = None
    entry_price = None

    for i in range(1, len(closes)):
        if (ma_short_vals[i] is not None and ma_long_vals[i] is not None and
            ma_short_vals[i-1] is not None and ma_long_vals[i-1] is not None):

            # Entry: 50-MA crosses above 200-MA
            if (not in_trade and
                ma_short_vals[i-1] <= ma_long_vals[i-1] and
                ma_short_vals[i] > ma_long_vals[i]):
                in_trade = True
                entry_idx = i
                entry_price = closes[i]

            # Exit: 50-MA crosses below 200-MA
            elif (in_trade and
                  ma_short_vals[i-1] >= ma_long_vals[i-1] and
                  ma_short_vals[i] < ma_long_vals[i]):
                exit_price = closes[i]
                points = exit_price - entry_price
                trades.append({
                    "entry_idx": entry_idx,
                    "entry_date": dates[entry_idx],
                    "entry_price": entry_price,
                    "exit_idx": i,
                    "exit_date": dates[i],
                    "exit_price": exit_price,
                    "points": points,
                    "days_held": (dates[i] - dates[entry_idx]).days,
                })
                in_trade = False

    return trades

def test_symbol(symbol: str, label: str):
    bars = load_bars(symbol, Path("backtest_data/daily_index"))

    print(f"\n{'='*80}")
    print(f"{label} ({symbol}) - Moving Average Crossover (50/200)")
    print(f"Post-video period (2021-07-01 onwards)\n")

    trades = find_ma_crossover_trades(bars)
    windowed = [t for t in trades if in_window(t["entry_date"], "2021-07-01", None)]

    if not windowed:
        print("No trades found")
        return

    # Manual summary
    net_points = sum(t["points"] for t in windowed)
    wins = [t for t in windowed if t["points"] > 0]
    win_pct = (len(wins) / len(windowed) * 100) if windowed else 0
    avg_points = net_points / len(windowed) if windowed else 0
    avg_days_held = sum(t["days_held"] for t in windowed) / len(windowed) if windowed else 0

    print(f"Trades: {len(windowed)}")
    print(f"Net Points: {net_points:.1f}")
    print(f"Win %: {win_pct:.1f}%")
    print(f"Avg/Trade: {avg_points:.1f} pts")
    print(f"Avg Days Held: {avg_days_held:.1f} days")

    # Calculate annual profit at the size actually bought per trade
    years = window_years("2021-07-01", bars["date"][-1])
    yearly_trades = len(windowed) / years
    annual_profit = annual_dollars(windowed, symbol, years)

    print(f"\nAnnual Estimate (${position_size(symbol):.2f} per trade, no costs):")
    print(f"  Trades/year: {yearly_trades:.1f}")
    print(f"  Annual profit: ${annual_profit:.2f}/year")

# Test both indices
test_symbol("^GSPC", "S&P 500")
test_symbol("^IXIC", "Nasdaq 100")
