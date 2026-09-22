"""Generate daily MA 30/90 Crossover trading signals.

Strategy: 30/90 Moving Average Crossover
Entry: 30-MA crosses above 90-MA (Golden Cross)
Exit: 30-MA crosses below 90-MA (Death Cross)

Usage:
    python -m trading_bot.cli.ma_crossover_signals
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from datetime import datetime
from pathlib import Path

import yfinance as yf
from trading_bot.util.notifier import notify

from trading_bot.backtest.rsi2_signals import simple_moving_average

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
        if df.empty or len(df) < 100:
            return None

        df = df.reset_index()

        return {
            "date": list(df["Date"].values),
            "open": [float(x) for x in df["Open"].values],
            "high": [float(x) for x in df["High"].values],
            "low": [float(x) for x in df["Low"].values],
            "close": [float(x) for x in df["Close"].values],
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def check_signal(symbol: str, bars: dict) -> dict:
    """Check if there's a valid buy/sell signal for this symbol."""
    closes = bars["close"]

    # Calculate MAs
    ma_30 = simple_moving_average(closes, 30)
    ma_90 = simple_moving_average(closes, 90)

    # Get latest values
    today_ma30 = ma_30[-1]
    today_ma90 = ma_90[-1]
    today_close = closes[-1]

    # Yesterday's values (to check for crossover)
    yesterday_ma30 = ma_30[-2] if len(ma_30) > 1 else None
    yesterday_ma90 = ma_90[-2] if len(ma_90) > 1 else None

    # Check conditions
    ma30_above_ma90 = today_ma30 > today_ma90 if (today_ma30 and today_ma90) else False
    golden_cross = (
        yesterday_ma30 is not None and yesterday_ma90 is not None and
        yesterday_ma30 <= yesterday_ma90 and
        today_ma30 > today_ma90
    )
    death_cross = (
        yesterday_ma30 is not None and yesterday_ma90 is not None and
        yesterday_ma30 >= yesterday_ma90 and
        today_ma30 < today_ma90
    )

    return {
        "symbol": symbol,
        "today_close": today_close,
        "today_ma30": today_ma30,
        "today_ma90": today_ma90,
        "ma30_above_ma90": ma30_above_ma90,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "signal": (
            "BUY" if golden_cross
            else "SELL" if death_cross
            else "HOLD" if ma30_above_ma90
            else "WAIT"
        ),
    }

def main():
    print(f"\n[MA 30/90]")

    for symbol, label in SYMBOLS.items():
        bars = fetch_recent_bars(symbol)
        if not bars or len(bars["close"]) < 100:
            print(f"[ERROR] {label}: Insufficient data\n")
            continue

        result = check_signal(symbol, bars)

        print(f"[PRICE] {label} ({symbol})")
        print(f"   Close: ${result['today_close']:.2f}")
        print(f"   30-MA: ${result['today_ma30']:.2f}")
        print(f"   90-MA: ${result['today_ma90']:.2f}")
        print(f"   Trend: {'UPTREND' if result['ma30_above_ma90'] else 'DOWNTREND'}")

        # Signal output
        if result["signal"] == "BUY":
            print(f"{GREEN}[BUY] Golden cross - 30-MA > 90-MA{RESET}")
            notify("MA 30/90 BUY Signal", f"{label}\nPrice: ${result['today_close']:.2f}\n30-MA: ${result['today_ma30']:.2f}")
        elif result["signal"] == "SELL":
            print(f"{RED}[SELL] Death cross - 30-MA < 90-MA{RESET}")
            notify("MA 30/90 SELL Signal", f"{label}\nPrice: ${result['today_close']:.2f}\n30-MA: ${result['today_ma30']:.2f}")
        elif result["signal"] == "HOLD":
            print(f"{YELLOW}[UPTREND] No cross signal{RESET}")
        else:
            print(f"{CYAN}[WAIT] Downtrend{RESET}")

        print()

if __name__ == "__main__":
    main()
