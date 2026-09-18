"""Monitor open trading positions and show exit signals.

Usage:
    python -m trading_bot.cli.rsi2_position_monitor

Reads rsi2_open_positions.json (actual open positions) and shows:
- Days held
- Current price and P&L
- SELL signal if 12+ days AND profitable
"""

from __future__ import annotations

import json
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
    """Load actual open positions from rsi2_open_positions.json."""
    positions_file = Path("rsi2_open_positions.json")

    if not positions_file.exists():
        print("\n[INFO] No open positions file found")
        return []

    try:
        with open(positions_file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print("\n[ERROR] Invalid positions format")
                return []

            trades = []
            for pos in data:
                # Convert RSI2 position format to monitor format
                trades.append({
                    'symbol': pos.get('symbol', ''),
                    'entry_date': pos.get('entry_date', '').split('T')[0],  # Extract date from ISO string
                    'entry_price': float(pos.get('entry_price', 0)),
                    'entry_amount': float(pos.get('contracts', 1)) * float(pos.get('entry_price', 0)),
                    'stop_loss': 0,  # RSI2 uses dynamic stops, not fixed ones
                })
            return trades
    except (json.JSONDecodeError, OSError) as e:
        print(f"\n[ERROR] Failed to read positions: {e}")
        return []

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

    # Calculate P&L (for RSI2, entry_amount is already entry_price * contracts)
    price_change = current_price - trade['entry_price']
    contracts = trade['entry_amount'] / trade['entry_price'] if trade['entry_price'] > 0 else 1
    pnl = price_change * contracts
    pnl_pct = (price_change / trade['entry_price']) * 100

    # Exit signal: 12+ days AND profitable (RSI2 uses dynamic management)
    should_exit = days_held >= 12 and pnl > 0

    return {
        'symbol': trade['symbol'],
        'entry_price': trade['entry_price'],
        'current_price': current_price,
        'days_held': days_held,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'should_exit': should_exit,
        'stop_hit': False,
    }

def main():
    trades = load_trades()

    print(f"\n{'='*70}")
    print(f"POSITION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    if not trades:
        print("[INFO] No open positions\n")
        return

    for trade in trades:
        current_price = get_current_price(trade['symbol'])

        if current_price is None:
            print(f"[ERROR] Could not fetch price for {trade['symbol']}\n")
            continue

        result = check_exit_signal(trade, current_price)

        print(f"[POSITION] {result['symbol']}")
        print(f"   Entry: ${result['entry_price']:.2f} ({result['days_held']} days ago)")
        print(f"   Current: ${result['current_price']:.2f}")
        print(f"   P&L: ${result['pnl']:+.2f} ({result['pnl_pct']:+.1f}%)")

        # Exit signal
        if result['should_exit']:
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
