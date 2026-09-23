"""Backtest Relative Strength strategy (Nasdaq/SPX ratio).

Entry: Nasdaq outperforming SPX (ratio above MA)
Exit: Nasdaq underperforming (ratio below MA)
Trade on Nasdaq when ratio is strong
"""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from trading_bot.backtest.rsi2_signals import simple_moving_average
from trading_bot.backtest.position_sizing import annual_dollars, position_size, window_years
from trading_bot.cli.rsi2_backtest import load_bars, in_window

def find_relative_strength_trades(spx_closes, ndx_closes, dates, ma_period=20):
    """Find trades based on Nasdaq/SPX relative strength."""

    # Calculate ratio
    ratio = []
    for i in range(len(spx_closes)):
        if spx_closes[i] > 0:
            ratio.append(ndx_closes[i] / spx_closes[i])
        else:
            ratio.append(None)

    # Calculate MA of ratio
    ratio_ma = simple_moving_average(ratio, ma_period)

    trades = []
    in_trade = False
    entry_idx = None
    entry_price = None
    hold_days = 0

    for i in range(ma_period, len(ratio)):
        if ratio[i] is None or ratio_ma[i] is None:
            continue

        if not in_trade:
            # Entry: ratio crosses above its MA (Nasdaq stronger)
            if (i > 0 and ratio[i-1] is not None and ratio_ma[i-1] is not None and
                ratio[i-1] <= ratio_ma[i-1] and ratio[i] > ratio_ma[i]):
                in_trade = True
                entry_idx = i
                entry_price = ndx_closes[i]
                hold_days = 0
        else:
            hold_days += 1

            # Exit: ratio crosses below its MA (Nasdaq weaker) or 60 days
            should_exit = False
            if (i > 0 and ratio[i-1] is not None and ratio_ma[i-1] is not None and
                ratio[i-1] >= ratio_ma[i-1] and ratio[i] < ratio_ma[i]):
                should_exit = True
            elif hold_days >= 60:
                should_exit = True

            if should_exit:
                exit_price = ndx_closes[i]
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

def test_relative_strength():
    spx_bars = load_bars("^GSPC", Path("backtest_data/daily_index"))
    ndx_bars = load_bars("^IXIC", Path("backtest_data/daily_index"))

    # Use Nasdaq dates as primary
    spx_closes = spx_bars["close"]
    ndx_closes = ndx_bars["close"]
    ndx_dates = ndx_bars["date"]

    # Align data (use shorter length)
    min_len = min(len(spx_closes), len(ndx_closes))
    spx_closes = spx_closes[:min_len]
    ndx_closes = ndx_closes[:min_len]
    ndx_dates = ndx_dates[:min_len]

    print(f"\n{'='*80}")
    print(f"RELATIVE STRENGTH (Nasdaq/SPX Ratio)")
    print(f"Post-video period (2021-07-01 onwards)")
    print(f"Trading Nasdaq when it outperforms S&P 500\n")

    trades = find_relative_strength_trades(spx_closes, ndx_closes, ndx_dates)
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

    print(f"Trades: {len(windowed)}")
    print(f"Net Points: {net_points:.1f}")
    print(f"Win %: {win_pct:.1f}%")
    print(f"Avg/Trade: {avg_points:.1f} pts")
    print(f"Avg Days Held: {avg_days:.1f} days")

    # Show individual trades
    print(f"\nIndividual Trades:")
    for i, t in enumerate(windowed[:5], 1):
        result = "WIN" if t["points"] > 0 else "LOSS"
        print(f"  {i}. {t['entry_date'].date()} -> {t['exit_date'].date()}: {t['points']:+.0f} pts ({result})")
    if len(windowed) > 5:
        print(f"  ... and {len(windowed) - 5} more trades")

    # Calculate annual profit at the size actually bought per trade (Nasdaq / SXRV)
    years = window_years("2021-07-01", ndx_dates[-1])
    yearly_trades = len(windowed) / years
    annual_profit = annual_dollars(windowed, "^IXIC", years)

    print(f"\nAnnual Estimate (${position_size('^IXIC'):.2f} per trade, no costs):")
    print(f"  Trades/year: {yearly_trades:.1f}")
    print(f"  Annual profit: ${annual_profit:.2f}/year")

    return annual_profit

# Test
print(f"{'='*80}")
print(f"RELATIVE STRENGTH STRATEGY BACKTEST")
print(f"{'='*80}")

annual = test_relative_strength()

print(f"\n{'='*80}")
print(f"Relative Strength: ${annual:.2f}/year (RSI2 + MA 30/90: run test_combined_strategies.py)")
print(f"{'='*80}\n")
