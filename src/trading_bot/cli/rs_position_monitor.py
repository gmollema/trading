"""Monitor open Relative Strength positions.

Shows current P&L and exit signals (when death cross occurs on ratio).

Usage:
    python -m trading_bot.cli.rs_position_monitor
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import yfinance as yf

from trading_bot.backtest.rsi2_signals import simple_moving_average

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def load_trades() -> list[dict]:
    """Load open RS trades from trades_rs.csv."""
    trades_file = Path("trades_rs.csv")

    if not trades_file.exists():
        print("\n[INFO] trades_rs.csv not found - no open RS positions\n")
        return []

    trades = []
    with open(trades_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append({
                'entry_date': row['entry_date'],
                'entry_price': float(row['entry_price']),
                'entry_amount': float(row['entry_amount']),
            })

    return trades

def get_current_price(symbol: str) -> float | None:
    """Get current price for Nasdaq 100."""
    try:
        data = yf.download(symbol, period="1d", progress=False)
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except:
        pass
    return None

def get_rs_signal() -> str:
    """Check current RS signal (HOLD or SELL)."""
    try:
        spx_data = yf.download("^GSPC", period="100d", progress=False)
        ndx_data = yf.download("^IXIC", period="100d", progress=False)

        if spx_data.empty or ndx_data.empty or len(spx_data) < 20 or len(ndx_data) < 20:
            return "UNKNOWN"

        min_len = min(len(spx_data), len(ndx_data))
        spx_closes = [float(x) for x in spx_data["Close"].values[-min_len:]]
        ndx_closes = [float(x) for x in ndx_data["Close"].values[-min_len:]]

        # Calculate ratio
        ratio = []
        for i in range(len(spx_closes)):
            if spx_closes[i] > 0:
                ratio.append(ndx_closes[i] / spx_closes[i])
            else:
                ratio.append(None)

        ratio_ma = simple_moving_average(ratio, 20)

        if ratio[-1] is None or ratio_ma[-1] is None:
            return "UNKNOWN"

        # Check for death cross
        if (ratio[-2] is not None and ratio_ma[-2] is not None and
            ratio[-2] >= ratio_ma[-2] and ratio[-1] < ratio_ma[-1]):
            return "SELL"
        elif ratio[-1] > ratio_ma[-1]:
            return "HOLD"
        else:
            return "EXIT"
    except:
        return "UNKNOWN"

def check_position(trade: dict, current_price: float) -> dict:
    """Check position status."""
    entry_date = datetime.strptime(trade['entry_date'], '%Y-%m-%d')
    today = datetime.now()
    days_held = (today - entry_date).days

    # Calculate P&L
    price_change = current_price - trade['entry_price']
    pnl = (price_change / trade['entry_price']) * trade['entry_amount']
    pnl_pct = (price_change / trade['entry_price']) * 100

    return {
        'entry_price': trade['entry_price'],
        'current_price': current_price,
        'days_held': days_held,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
    }

def main():
    trades = load_trades()

    print(f"\n{'='*70}")
    print(f"RELATIVE STRENGTH POSITION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    if len(trades) == 0:
        print("[INFO] No open RS positions\n")
        return

    for trade in trades:
        current_price = get_current_price("^IXIC")

        if current_price is None:
            print(f"[ERROR] Could not fetch Nasdaq 100 price\n")
            continue

        position = check_position(trade, current_price)
        rs_signal = get_rs_signal()

        print(f"[POSITION] Nasdaq 100 (QQQ)")
        print(f"   Entry: ${position['entry_price']:.2f} ({position['days_held']} days ago)")
        print(f"   Current: ${position['current_price']:.2f}")
        print(f"   P&L: ${position['pnl']:+.2f} ({position['pnl_pct']:+.1f}%)")

        # Exit signal
        if rs_signal == "SELL":
            print(f"\n{RED}{'='*70}")
            print(f"{RED}{BOLD}[ACTION] SELL - DEATH CROSS (Ratio < MA){RESET}")
            print(f"{RED}{'='*70}{RESET}")
            print(f"{RED}Exit Nasdaq 100 (QQQ) at market price{RESET}")
        elif rs_signal == "HOLD":
            print(f"\n{GREEN}[ACTION] HOLD - Nasdaq outperforming")
            print(f"   Continue holding{RESET}")
        else:
            print(f"\n{CYAN}[ACTION] WATCH - RS unclear, monitor closely{RESET}")

        print()

if __name__ == "__main__":
    main()
