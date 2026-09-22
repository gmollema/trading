"""Backtest simplified Bollinger Band Breakout strategy.

Entry: Close breaks above upper Bollinger Band (after low volatility)
Exit: Close below 20-MA or 25 days held
"""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from trading_bot.backtest.rsi2_signals import simple_moving_average
from trading_bot.cli.rsi2_backtest import load_bars, in_window

def find_bb_trades(bars):
    """Find Bollinger Band breakout trades."""
    closes = bars["close"]
    dates = bars["date"]

    # Calculate 20-MA and simple volatility
    ma20 = simple_moving_average(closes, 20)

    # Simple volatility: average of last 20 bars range
    volatility = []
    for i in range(len(closes)):
        if i < 20:
            volatility.append(None)
        else:
            high_low_diffs = []
            for j in range(i-19, i+1):
                if j < len(bars["high"]) and j < len(bars["low"]):
                    high_low_diffs.append(bars["high"][j] - bars["low"][j])
            avg_vol = sum(high_low_diffs) / len(high_low_diffs) if high_low_diffs else 0
            volatility.append(avg_vol)

    # Bollinger Bands: MA +/- (2 * volatility)
    upper_band = [ma20[i] + (2 * volatility[i]) if ma20[i] and volatility[i] else None for i in range(len(ma20))]
    lower_band = [ma20[i] - (2 * volatility[i]) if ma20[i] and volatility[i] else None for i in range(len(ma20))]

    trades = []
    in_trade = False
    entry_idx = None
    entry_price = None
    hold_days = 0

    for i in range(21, len(closes) - 1):
        if not in_trade and upper_band[i] is not None:
            # Entry: close breaks above upper band
            if closes[i] > upper_band[i]:
                in_trade = True
                entry_idx = i
                entry_price = closes[i]
                hold_days = 0

        elif in_trade:
            hold_days += 1

            # Exit: close below 20-MA or 25 days
            should_exit = False
            if ma20[i] is not None and closes[i] < ma20[i]:
                should_exit = True
            elif hold_days >= 25:
                should_exit = True

            if should_exit:
                exit_price = closes[i]
                points = exit_price - entry_price
                trades.append({
                    "entry_date": dates[entry_idx],
                    "entry_price": entry_price,
                    "exit_date": dates[i],
                    "exit_price": exit_price,
                    "points": points,
                    "days_held": (dates[i] - dates[entry_idx]).days,
                })
                in_trade = False

    return trades

def test_symbol(symbol, label):
    bars = load_bars(symbol, Path("backtest_data/daily_index"))

    print(f"\n{'='*80}")
    print(f"{label} ({symbol}) - Bollinger Band Breakout")
    print(f"Post-video period (2021-07-01 onwards)\n")

    trades = find_bb_trades(bars)
    windowed = [t for t in trades if in_window(t["entry_date"], "2021-07-01", None)]

    if not windowed:
        print("No trades found")
        return 0

    # Manual summary
    net_points = sum(t["points"] for t in windowed)
    wins = [t for t in windowed if t["points"] > 0]
    win_pct = (len(wins) / len(windowed) * 100) if windowed else 0
    avg_points = net_points / len(windowed) if windowed else 0

    print(f"Trades: {len(windowed)}")
    print(f"Net Points: {net_points:.1f}")
    print(f"Win %: {win_pct:.1f}%")
    print(f"Avg/Trade: {avg_points:.1f} pts")

    # Calculate annual profit
    yearly_trades = len(windowed) / 5.2
    if symbol == "^GSPC":
        dollar_per_pt = 0.111  # $25 / 225 stop
    else:
        dollar_per_pt = 0.10   # $12.50 / 125 stop

    annual_profit = yearly_trades * (avg_points * dollar_per_pt)

    print(f"\nAnnual Estimate:")
    print(f"  Trades/year: {yearly_trades:.1f}")
    print(f"  Annual profit: ${annual_profit:.0f}/year")

    return annual_profit

# Test both indices
print(f"{'='*80}")
print(f"BOLLINGER BAND BREAKOUT STRATEGY")
print(f"{'='*80}")

annual_sp = test_symbol("^GSPC", "S&P 500")
annual_nd = test_symbol("^IXIC", "Nasdaq 100")

print(f"\n{'='*80}")
print(f"COMBINED ANNUAL PROFIT: ${annual_sp + annual_nd:.0f}/year")
print(f"{'='*80}\n")
