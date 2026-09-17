"""Test different stop loss levels on Nasdaq 100."""

from pathlib import Path
from trading_bot.backtest.rsi2_signals import find_rsi2_long_trades
from trading_bot.cli.rsi2_backtest import load_bars, in_window, summarize

bars = load_bars("^IXIC", Path("backtest_data/daily_index"))

print("Testing different stop loss levels on RSI2 strategy")
print("Nasdaq 100 (^IXIC) - Post-video period (2021-07-01 onwards)\n")
print(f"{'Stop (pts)':<15} {'Trades':<10} {'Net Pts':<12} {'Win %':<10} {'Avg/Trade':<12} {'Max DD':<10}")
print("-" * 80)

results = []

for stop_pts in [100, 125, 150, 175, 200, 225, 250]:
    trades = find_rsi2_long_trades(
        bars,
        rsi_period=2,
        entry_level=10.0,
        exit_level=70.0,
        sma_period=200,
        stop_points=float(stop_pts),
        exit_mode="first_profitable_close",
        min_hold_days=12,
        exit_timing="close",
    )

    windowed = [t for t in trades if in_window(t["entry_date"], "2021-07-01", None)]

    if not windowed:
        continue

    # Add net_points field for summarize()
    for t in windowed:
        t["net_points"] = t["points"]

    summary = summarize(windowed, bars, ("2021-07-01", None))

    marker = " <-- BASELINE" if stop_pts == 200 else ""
    print(f"{stop_pts:<15} {summary['trades']:<10} {summary['net_points']:<12.1f} {summary['win_pct']:<10.1f} {summary['avg_points']:<12.1f} {summary['max_drawdown_points']:<10.1f}{marker}")

    results.append({
        'stop': stop_pts,
        'trades': summary['trades'],
        'net': summary['net_points'],
        'win_pct': summary['win_pct'],
        'avg': summary['avg_points'],
        'dd': summary['max_drawdown_points'],
    })

if results:
    # Find best
    best = max(results, key=lambda x: x['net'])
    baseline = [r for r in results if r['stop'] == 200][0]
    improvement = best['net'] - baseline['net']
    improvement_pct = (improvement / abs(baseline['net'])) * 100 if baseline['net'] != 0 else 0

    print(f"\nOPTIMAL: {best['stop']} points stop → {best['net']:.1f} points ({best['trades']} trades)")
    if improvement > 0:
        print(f"vs baseline (200pt): +{improvement:.1f} points (+{improvement_pct:.1f}%)")
    else:
        print(f"vs baseline (200pt): {improvement:.1f} points ({improvement_pct:.1f}%)")
