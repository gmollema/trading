"""Monitor open trading positions and show exit signals.

Usage:
    python -m trading_bot.cli.rsi2_position_monitor

Reads trades.csv (your open positions) and shows:
- Days held
- Current price and P&L
- SELL signal if 12+ days AND profitable
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import yfinance as yf

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def load_trades() -> list[dict]:
    """Load open trades from trades.csv."""
    trades_file = Path("trades.csv")

    if not trades_file.exists():
        print("\n[ERROR] trades.csv not found!")
        print("\nCreate trades.csv in your project folder with this format:")
        print("symbol,entry_date,entry_price,entry_amount,stop_loss")
        print("^GSPC,2026-09-18,7620.60,25.00,225")
        print("^IXIC,2026-09-18,26300.00,25.00,125")
        return []

    trades = []
    with open(trades_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append({
                'symbol': row['symbol'],
                'entry_date': row['entry_date'],
                'entry_price': float(row['entry_price']),
                'entry_amount': float(row['entry_amount']),
                'stop_loss': float(row['stop_loss']),
            })

    return trades

def get_current_price(symbol: str) -> float | None:
    """Get current price for a symbol."""
    try:
        data = yf.download(symbol, period="1d", progress=False)
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
    return None

def check_exit_signal(trade: dict, current_price: float) -> dict:
    """Check if position should be exited."""
    entry_date = datetime.strptime(trade['entry_date'], '%Y-%m-%d')
    today = datetime.now()
    days_held = (today - entry_date).days

    # Calculate P&L
    price_change = current_price - trade['entry_price']
    pnl = (price_change / trade['entry_price']) * trade['entry_amount']
    pnl_pct = (price_change / trade['entry_price']) * 100

    # Exit signal: 12+ days AND profitable
    should_exit = days_held >= 12 and pnl > 0

    # Check if stop hit
    stop_hit = price_change < -trade['stop_loss']

    return {
        'symbol': trade['symbol'],
        'entry_price': trade['entry_price'],
        'current_price': current_price,
        'days_held': days_held,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'should_exit': should_exit,
        'stop_hit': stop_hit,
    }

def main():
    trades = load_trades()

    if not trades:
        return

    print(f"\n{'='*70}")
    print(f"POSITION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    if len(trades) == 0:
        print("[INFO] No open positions\n")
        return

    for trade in trades:
        current_price = get_current_price(trade['symbol'])

        if current_price is None:
            print(f"[ERROR] Could not fetch price for {trade['symbol']}\n")
            continue

        result = check_exit_signal(trade, current_price)

        # Map symbols to labels
        label = "S&P 500 (SPY)" if trade['symbol'] == "^GSPC" else "Nasdaq 100 (QQQ)"

        print(f"[POSITION] {label} ({trade['symbol']})")
        print(f"   Entry: ${result['entry_price']:.2f} ({result['days_held']} days ago)")
        print(f"   Current: ${result['current_price']:.2f}")
        print(f"   P&L: ${result['pnl']:+.2f} ({result['pnl_pct']:+.1f}%)")

        # Exit signal
        if result['stop_hit']:
            print(f"\n{RED}{'='*70}")
            print(f"{RED}{BOLD}[ACTION] STOP HIT - SELL IMMEDIATELY{RESET}")
            print(f"{RED}{'='*70}{RESET}")
            print(f"{RED}Loss: ${result['pnl']:.2f}{RESET}")
        elif result['should_exit']:
            print(f"\n{GREEN}{'='*70}")
            print(f"{GREEN}{BOLD}[ACTION] SELL - 12+ DAYS AND PROFITABLE{RESET}")
            print(f"{GREEN}{'='*70}{RESET}")
            print(f"{GREEN}Profit: ${result['pnl']:.2f}{RESET}")
        elif result['days_held'] >= 12:
            print(f"\n{YELLOW}[ACTION] HOLD - 12+ days but not profitable yet")
            print(f"   Current loss: ${result['pnl']:.2f}{RESET}")
        else:
            days_left = 12 - result['days_held']
            print(f"\n{CYAN}[ACTION] HOLD - {days_left} more days until exit eligible{RESET}")

        print()

if __name__ == "__main__":
    main()
