"""Monitor open MA 30/90 Crossover positions.

Shows current P&L and sell signals (when death cross occurs).

Usage:
    python -m trading_bot.cli.ma_position_monitor
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import csv
from datetime import datetime
from pathlib import Path

import yfinance as yf

from trading_bot.backtest.rsi2_signals import simple_moving_average

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def load_trades() -> list[dict]:
    """Load open trades from trades_ma.csv."""
    trades_file = Path("trades_ma.csv")

    if not trades_file.exists():
        print("\n[INFO] trades_ma.csv not found - no open MA positions\n")
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

def get_ma_signal(symbol: str) -> str:
    """Check current MA signal (HOLD or SELL)."""
    try:
        df = yf.download(symbol, period="100d", progress=False)
        if df.empty or len(df) < 90:
            return "UNKNOWN"

        df = df.reset_index()
        closes = [float(x) for x in df["Close"].values]

        from trading_bot.backtest.rsi2_signals import simple_moving_average
        ma_30 = simple_moving_average(closes, 30)
        ma_90 = simple_moving_average(closes, 90)

        if ma_30[-1] is None or ma_90[-1] is None:
            return "UNKNOWN"

        # Check for death cross
        if (ma_30[-2] >= ma_90[-2] and ma_30[-1] < ma_90[-1]):
            return "SELL"
        elif ma_30[-1] > ma_90[-1]:
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
        'symbol': trade['symbol'],
        'entry_price': trade['entry_price'],
        'current_price': current_price,
        'days_held': days_held,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
    }

def main():
    trades = load_trades()

    if len(trades) == 0:
        return

    for trade in trades:
        current_price = get_current_price(trade['symbol'])

        if current_price is None:
            print(f"[ERROR] Could not fetch price for {trade['symbol']}\n")
            continue

        position = check_position(trade, current_price)
        ma_signal = get_ma_signal(trade['symbol'])

        label = "S&P 500 (SPY)" if trade['symbol'] == "^GSPC" else "Nasdaq 100 (QQQ)"

        print(f"[POSITION] {label} ({trade['symbol']})")
        print(f"   Entry: ${position['entry_price']:.2f} ({position['days_held']} days ago)")
        print(f"   Current: ${position['current_price']:.2f}")
        print(f"   P&L: ${position['pnl']:+.2f} ({position['pnl_pct']:+.1f}%)")

        # Exit signal
        if ma_signal == "SELL":
            print(f"\n{RED}{'='*70}")
            print(f"{RED}{BOLD}[ACTION] SELL - DEATH CROSS (30-MA < 90-MA){RESET}")
            print(f"{RED}{'='*70}{RESET}")
            print(f"{RED}Exit now at market price{RESET}")
        elif ma_signal == "HOLD":
            print(f"\n{GREEN}[ACTION] HOLD - In uptrend, 30-MA > 90-MA")
            print(f"   Continue holding{RESET}")
        else:
            print(f"\n{YELLOW}[ACTION] WATCH - MAs unclear, monitor closely{RESET}")

        print()

if __name__ == "__main__":
    main()
