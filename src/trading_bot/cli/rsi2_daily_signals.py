"""Generate daily RSI2 trading signals for manual Trading 212 execution.

Usage:
    python -m trading_bot.cli.rsi2_daily_signals

Checks today's price action against RSI2 mean reversion rules for both
S&P 500 and Nasdaq 100. Output tells you exactly what to trade.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

from trading_bot.backtest.rsi2_signals import (
    wilder_rsi,
    simple_moving_average,
    get_optimal_entry_rsi,
    get_optimal_sma,
)

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

SYMBOLS = {
    "^GSPC": "S&P 500 (SPY)",
    "^IXIC": "Nasdaq 100 (QQQ)",
}

def fetch_recent_bars(symbol: str, days: int = 250) -> dict | None:
    """Fetch recent daily bars for a symbol."""
    try:
        df = yf.download(symbol, period=f"{days}d", progress=False)
        if df.empty or len(df) < 50:
            return None

        # Reset index to get date as column
        df = df.reset_index()

        return {
            "date": list(df["Date"].values),
            "open": [float(x.item()) if hasattr(x, 'item') else float(x) for x in df["Open"].values],
            "high": [float(x.item()) if hasattr(x, 'item') else float(x) for x in df["High"].values],
            "low": [float(x.item()) if hasattr(x, 'item') else float(x) for x in df["Low"].values],
            "close": [float(x.item()) if hasattr(x, 'item') else float(x) for x in df["Close"].values],
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def check_signal(symbol: str, bars: dict) -> dict:
    """Check if there's a valid buy signal for this symbol."""
    closes = bars["close"]

    # Calculate RSI(2)
    rsi = wilder_rsi(closes, period=2)

    # Calculate SMA with symbol-specific optimal period
    sma_period = get_optimal_sma(symbol)
    sma = simple_moving_average(closes, sma_period)

    # Get symbol-specific entry RSI level
    entry_level = get_optimal_entry_rsi(symbol)

    # Get latest values
    today_rsi = rsi[-1]
    today_close = closes[-1]
    today_sma = sma[-1]

    # Yesterday's values (to check for RSI cross)
    yesterday_rsi = rsi[-2] if len(rsi) > 1 else None

    # Check conditions
    price_above_sma = today_close > today_sma if today_sma else False
    rsi_crossed_below = (
        yesterday_rsi is not None and
        yesterday_rsi >= entry_level and
        today_rsi < entry_level
    )
    rsi_is_low = today_rsi < entry_level

    return {
        "symbol": symbol,
        "today_close": today_close,
        "today_rsi": today_rsi,
        "entry_level": entry_level,
        "sma_period": sma_period,
        "today_sma": today_sma,
        "price_above_sma": price_above_sma,
        "rsi_crossed_below": rsi_crossed_below,
        "rsi_is_low": rsi_is_low,
        "signal": (
            "BUY" if (rsi_crossed_below and price_above_sma)
            else "SETUP" if (rsi_is_low and price_above_sma)
            else "WAIT"
        ),
    }

def main():
    print(f"\n{'='*70}")
    print(f"RSI2 DAILY SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    for symbol, label in SYMBOLS.items():
        bars = fetch_recent_bars(symbol)
        if not bars or len(bars["close"]) < 50:
            print(f"[ERROR] {label}: Insufficient data\n")
            continue

        result = check_signal(symbol, bars)

        print(f"[PRICE] {label} ({symbol})")
        print(f"   Close: ${result['today_close']:.2f}")

        # Color-code RSI condition
        rsi_color = GREEN if result['rsi_is_low'] else RED
        print(f"   {rsi_color}RSI(2): {result['today_rsi']:.1f} (entry at < {result['entry_level']}){RESET}")

        print(f"   SMA({result['sma_period']}): ${result['today_sma']:.2f}")

        # Color-code Price > SMA condition
        sma_color = GREEN if result['price_above_sma'] else RED
        sma_text = "YES" if result['price_above_sma'] else "NO"
        print(f"   {sma_color}Price > SMA: {sma_text}{RESET}")

        # Signal output with color-coded decision
        if result["signal"] == "BUY":
            print(f"\n{GREEN}{'='*70}")
            print(f"{GREEN}{BOLD}>>> BUY SIGNAL <<<{RESET}")
            print(f"{GREEN}{'='*70}{RESET}")
            print(f"{GREEN}RSI crossed below {result['entry_level']}")
            print(f"Trade size: $25 on Trading 212")
            print(f"Entry: NOW{RESET}")
            print(f"{GREEN}{'='*70}{RESET}")
        elif result["signal"] == "SETUP":
            print(f"\n{YELLOW}{'='*70}")
            print(f"{YELLOW}{BOLD}>>> SETUP (WATCH) <<<{RESET}")
            print(f"{YELLOW}{'='*70}{RESET}")
            print(f"{YELLOW}RSI is low, waiting for cross")
            print(f"Monitor for entry tomorrow{RESET}")
            print(f"{YELLOW}{'='*70}{RESET}")
        else:
            print(f"\n{RED}{'='*70}")
            print(f"{RED}{BOLD}>>> WAIT - No signal <<<{RESET}")
            print(f"{RED}RSI not low enough or price below SMA{RESET}")
            print(f"{RED}{'='*70}{RESET}")

        print()

if __name__ == "__main__":
    main()
