"""Quick entry RSI level sweep for RSI2 strategy"""
from pathlib import Path
import json
from trading_bot.backtest.rsi2_signals import find_rsi2_scale_in_trades
from trading_bot.cli.rsi2_fetch_data import DAILY_INDEX_DIR, safe_filename

symbol = "^GSPC"
data_dir = Path(DAILY_INDEX_DIR)
bars_path = data_dir / safe_filename(symbol)
bars_str = bars_path.read_text()
bars_dict = json.loads(bars_str)

# Filter to post_video window only (2021-07-01 onwards) for faster testing
from datetime import datetime
start_idx = next(i for i, d in enumerate(bars_dict["date"]) if datetime.fromisoformat(d).date() >= datetime(2021, 7, 1).date())
for key in bars_dict:
    bars_dict[key] = bars_dict[key][start_idx:]

print("Entry RSI Sweep - Post Video Window (2021-07-01 onwards)")
print("="*70)
print(f"{'Entry RSI':<12} {'Trades':<8} {'Win%':<8} {'Avg Pts':<10} {'Max DD':<10} {'Net Points'}")
print("="*70)

for entry_level in [8, 10, 12, 15, 20, 25, 30]:
    trades = find_rsi2_scale_in_trades(
        bars_dict,
        rsi_period=2,
        entry_level=entry_level,
        exit_level=70,
        sma_period=175,
        max_positions=1,
        first_dip=1,
        stop_pct=None,
        exit_timing="close",
        entry_timing="close",
    )

    if not trades:
        print(f"{entry_level:<12} {'0':<8} {'-':<8} {'-':<10} {'-':<10} 0")
        continue

    wins = sum(1 for t in trades if t["points"] > 0)
    total_points = sum(t["points"] for t in trades)
    avg_points = total_points / len(trades) if trades else 0

    # Simple max drawdown calculation
    equity = 1000
    equity_curve = [equity]
    for t in trades:
        equity += t["points"]
        equity_curve.append(equity)
    max_dd = max(equity_curve) - min(equity_curve) if equity_curve else 0

    win_pct = (wins / len(trades) * 100) if trades else 0

    print(f"{entry_level:<12} {len(trades):<8} {win_pct:<8.1f} {avg_points:<10.1f} {max_dd:<10.1f} {total_points:<10.1f}")

print("="*70)
print("Note: Using single position (non-scale-in) for clarity")
