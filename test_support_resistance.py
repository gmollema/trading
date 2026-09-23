"""Backtest Support/Resistance Bounce strategy.

Entry: Price bounces off support level (after 5-10% drop)
Exit: Reaches prior resistance or 30 days held
"""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from trading_bot.backtest.position_sizing import annual_dollars, position_size, window_years
from trading_bot.cli.rsi2_backtest import load_bars, in_window

def find_support_resistance_trades(bars, lookback=50):
    """Find S/R bounce trades."""
    closes = bars["close"]
    dates = bars["date"]
    highs = bars["high"]
    lows = bars["low"]

    trades = []
    in_trade = False
    entry_idx = None
    entry_price = None
    support_level = None
    resistance_level = None
    hold_days = 0

    for i in range(lookback, len(closes)):
        # Find support and resistance from last lookback periods
        lookback_lows = lows[i-lookback:i]
        lookback_highs = highs[i-lookback:i]

        support = min(lookback_lows) if lookback_lows else closes[i] * 0.95
        resistance = max(lookback_highs) if lookback_highs else closes[i] * 1.05

        if not in_trade:
            # Look for setup: price near support (within 2% of support)
            if closes[i] <= support * 1.02 and closes[i] >= support * 0.98:
                # Entry: close bounces above support next bar
                if i + 1 < len(closes) and closes[i+1] > support * 1.01:
                    in_trade = True
                    entry_idx = i + 1
                    entry_price = closes[i+1]
                    support_level = support
                    resistance_level = resistance
                    hold_days = 0
        else:
            hold_days += 1

            # Exit conditions
            should_exit = False
            exit_price = closes[i]

            # Exit 1: reaches resistance
            if closes[i] >= resistance_level * 0.99:
                should_exit = True
            # Exit 2: 30 days held
            elif hold_days >= 30:
                should_exit = True
            # Exit 3: close below support (stop loss)
            elif closes[i] < support_level * 0.98:
                should_exit = True

            if should_exit:
                points = exit_price - entry_price
                trades.append({
                    "entry_date": dates[entry_idx],
                    "entry_price": entry_price,
                    "exit_date": dates[i],
                    "exit_price": exit_price,
                    "points": points,
                    "days_held": (dates[i] - dates[entry_idx]).days,
                    "type": "resistance_hit" if closes[i] >= resistance_level * 0.99 else ("stop_hit" if closes[i] < support_level * 0.98 else "time_exit"),
                })
                in_trade = False

    return trades

def test_symbol(symbol, label):
    bars = load_bars(symbol, Path("backtest_data/daily_index"))

    print(f"\n{'='*80}")
    print(f"{label} ({symbol}) - Support/Resistance Bounce")
    print(f"Post-video period (2021-07-01 onwards)\n")

    trades = find_support_resistance_trades(bars)
    windowed = [t for t in trades if in_window(t["entry_date"], "2021-07-01", None)]

    if not windowed:
        print("No trades found")
        return 0

    # Manual summary
    net_points = sum(t["points"] for t in windowed)
    wins = [t for t in windowed if t["points"] > 0]
    win_pct = (len(wins) / len(windowed) * 100) if windowed else 0
    avg_points = net_points / len(windowed) if windowed else 0
    avg_days = sum(t["days_held"] for t in windowed) / len(windowed) if windowed else 0

    # Exit type breakdown
    resist_hits = len([t for t in windowed if t["type"] == "resistance_hit"])
    stop_hits = len([t for t in windowed if t["type"] == "stop_hit"])
    time_exits = len([t for t in windowed if t["type"] == "time_exit"])

    print(f"Trades: {len(windowed)}")
    print(f"Net Points: {net_points:.1f}")
    print(f"Win %: {win_pct:.1f}%")
    print(f"Avg/Trade: {avg_points:.1f} pts")
    print(f"Avg Days Held: {avg_days:.1f} days")
    print(f"  (Resistance hits: {resist_hits}, Stop hits: {stop_hits}, Time exits: {time_exits})")

    # Calculate annual profit at the size actually bought per trade
    years = window_years("2021-07-01", bars["date"][-1])
    yearly_trades = len(windowed) / years
    annual_profit = annual_dollars(windowed, symbol, years)

    print(f"\nAnnual Estimate (${position_size(symbol):.2f} per trade, no costs):")
    print(f"  Trades/year: {yearly_trades:.1f}")
    print(f"  Annual profit: ${annual_profit:.2f}/year")

    return annual_profit

# Test both indices
print(f"{'='*80}")
print(f"SUPPORT/RESISTANCE BOUNCE STRATEGY")
print(f"{'='*80}")

annual_sp = test_symbol("^GSPC", "S&P 500")
annual_nd = test_symbol("^IXIC", "Nasdaq 100")

print(f"\n{'='*80}")
print(f"COMBINED ANNUAL PROFIT: ${annual_sp + annual_nd:.2f}/year")
print(f"{'='*80}\n")
