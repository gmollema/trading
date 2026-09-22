"""Daily Relative Strength (Nasdaq/SPX) signal generator.

Entry: Nasdaq outperforming SPX (ratio > MA)
Exit: Nasdaq underperforming (ratio < MA)
Trade on Nasdaq when ratio is strong

Usage:
    python -m trading_bot.cli.relative_strength_signals
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from datetime import datetime, timedelta
import yfinance as yf

from trading_bot.backtest.rsi2_signals import simple_moving_average

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def get_relative_strength_signal(ma_period: int = 20) -> dict:
    """Check Nasdaq/SPX relative strength ratio."""

    try:
        # Fetch 100 days of data for both indices
        spx_data = yf.download("^GSPC", period="100d", progress=False)
        ndx_data = yf.download("^IXIC", period="100d", progress=False)

        if spx_data.empty or ndx_data.empty or len(spx_data) < ma_period or len(ndx_data) < ma_period:
            return {"signal": "ERROR", "reason": "Insufficient data"}

        # Align data by taking minimum length
        min_len = min(len(spx_data), len(ndx_data))
        spx_closes = [float(x) for x in spx_data["Close"].values[-min_len:]]
        ndx_closes = [float(x) for x in ndx_data["Close"].values[-min_len:]]
        spx_dates = list(spx_data.index[-min_len:])

        # Calculate ratio
        ratio = []
        for i in range(len(spx_closes)):
            if spx_closes[i] > 0:
                ratio.append(ndx_closes[i] / spx_closes[i])
            else:
                ratio.append(None)

        # Calculate MA of ratio
        ratio_ma = simple_moving_average(ratio, ma_period)

        # Get current values
        current_ratio = ratio[-1]
        current_ratio_ma = ratio_ma[-1]
        prev_ratio = ratio[-2] if len(ratio) >= 2 else None
        prev_ratio_ma = ratio_ma[-2] if len(ratio_ma) >= 2 else None

        if current_ratio is None or current_ratio_ma is None:
            return {"signal": "ERROR", "reason": "Invalid ratio calculation"}

        # Determine signal
        signal = "WAIT"
        action = ""

        if prev_ratio is not None and prev_ratio_ma is not None:
            # Check for ratio crossing above MA (entry)
            if prev_ratio <= prev_ratio_ma and current_ratio > current_ratio_ma:
                signal = "BUY"
                action = "GOLDEN CROSS - Nasdaq stronger than SPX"
            # Check for ratio crossing below MA (exit)
            elif prev_ratio >= prev_ratio_ma and current_ratio < current_ratio_ma:
                signal = "SELL"
                action = "DEATH CROSS - Nasdaq weaker than SPX"
            # In uptrend
            elif current_ratio > current_ratio_ma:
                signal = "HOLD"
                action = f"In uptrend - Ratio above MA ({current_ratio:.4f} > {current_ratio_ma:.4f})"
            # In downtrend
            else:
                signal = "WAIT"
                action = f"In downtrend - Ratio below MA ({current_ratio:.4f} < {current_ratio_ma:.4f})"

        return {
            "signal": signal,
            "action": action,
            "current_ratio": current_ratio,
            "ratio_ma": current_ratio_ma,
            "ratio_distance": ((current_ratio - current_ratio_ma) / current_ratio_ma * 100),
        }

    except Exception as e:
        return {"signal": "ERROR", "reason": str(e)}

def main():
    print(f"\n[RELATIVE STRENGTH]")

    result = get_relative_strength_signal()

    if result["signal"] == "ERROR":
        print(f"[ERROR] {result['reason']}\n")
        return

    print(f"Ratio: {result['current_ratio']:.4f} / MA: {result['ratio_ma']:.4f} ({result['ratio_distance']:+.2f}%)")

    if result["signal"] == "BUY":
        print(f"{GREEN}[BUY] Golden cross - Nasdaq outperforming{RESET}")
    elif result["signal"] == "SELL":
        print(f"{RED}[SELL] Death cross - Nasdaq underperforming{RESET}")
    elif result["signal"] == "HOLD":
        print(f"{GREEN}[UPTREND] Ratio > MA, no cross signal{RESET}")
    else:
        print(f"{CYAN}[WAIT] Ratio < MA, Nasdaq underperforming{RESET}")

if __name__ == "__main__":
    main()
