"""Test different SMA periods on RSI2 strategy."""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from trading_bot.backtest.rsi2_signals import find_rsi2_long_trades
from trading_bot.cli.rsi2_backtest import load_bars, in_window, summarize

def test_symbol(symbol: str, label: str):
    bars = load_bars(symbol, Path("backtest_data/daily_index"))

    print(f"\n{'='*80}")
    print(f"Testing different SMA periods on {label} ({symbol})")
    print(f"Post-video period (2021-07-01 onwards)\n")
    print(f"{'SMA':<10} {'Trades':<10} {'Net Pts':<12} {'Win %':<10} {'Avg/Trade':<12} {'Max DD':<10}")
    print("-" * 80)

    results = []

    for sma_period in [50, 75, 100, 125, 150, 175, 200, 250]:
        trades = find_rsi2_long_trades(
            bars,
            rsi_period=2,
            entry_level=10.0,
            exit_level=70.0,
            sma_period=sma_period,
            stop_points=225.0 if symbol == "^GSPC" else 125.0,  # Symbol-specific optimal
            exit_mode="first_profitable_close",
            min_hold_days=12,
            exit_timing="close",
        )

        windowed = [t for t in trades if in_window(t["entry_date"], "2021-07-01", None)]

        if not windowed:
            continue

        for t in windowed:
            t["net_points"] = t["points"]

        summary = summarize(windowed, bars, ("2021-07-01", None))

        marker = " <-- BASELINE" if sma_period == 200 else ""
        print(f"{sma_period:<10} {summary['trades']:<10} {summary['net_points']:<12.1f} {summary['win_pct']:<10.1f} {summary['avg_points']:<12.1f} {summary['max_drawdown_points']:<10.1f}{marker}")

        results.append({
            'sma': sma_period,
            'trades': summary['trades'],
            'net': summary['net_points'],
            'win_pct': summary['win_pct'],
            'avg': summary['avg_points'],
            'dd': summary['max_drawdown_points'],
        })

    if results:
        # Find best
        best = max(results, key=lambda x: x['net'])
        baseline = [r for r in results if r['sma'] == 200][0]
        improvement = best['net'] - baseline['net']
        improvement_pct = (improvement / abs(baseline['net'])) * 100 if baseline['net'] != 0 else 0

        print(f"\nOPTIMAL: SMA({best['sma']}) -> {best['net']:.1f} points ({best['trades']} trades)")
        if improvement > 0:
            print(f"vs baseline (SMA 200): +{improvement:.1f} points (+{improvement_pct:.1f}%)")
        else:
            print(f"vs baseline (SMA 200): {improvement:.1f} points ({improvement_pct:.1f}%)")

# Test both indices
test_symbol("^GSPC", "S&P 500")
test_symbol("^IXIC", "Nasdaq 100")
