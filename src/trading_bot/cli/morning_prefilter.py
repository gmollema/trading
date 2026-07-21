"""Morning gap/price prefilter for the S&P 500 universe.

Standalone CLI: screens the S&P 500 universe (trading_bot/data/sp500_tickers.py) via
yfinance for gap-up and minimum-price criteria, and writes the survivors
to watchlist.txt for downstream use by the bot.

This script does NOT implement the full D1-D3 / I1-I3 rule set from
rules.json. It is a coarse morning prefilter only (gap % and price floor).

Usage:
    python -m trading_bot.cli.morning_prefilter --dry-run
    python -m trading_bot.cli.morning_prefilter --min-gap 3.0 --min-price 3.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

from trading_bot.util.notifier import notify
from trading_bot.data.sp500_tickers import SP500_TICKERS

WATCHLIST_PATH = Path("watchlist.txt")
DEFAULT_MIN_GAP_PCT = 3.0
DEFAULT_MIN_PRICE = 3.0
MAX_SURVIVORS = 20
ET = ZoneInfo("America/New_York")

LOGS_DIR = Path("logs")
PREFILTER_ERRORS_LOG = LOGS_DIR / "prefilter_errors.log"

ALERT_FAILURE_RATIO_SEVERE = 0.95
ALERT_FAILURE_RATIO_DEGRADED = 0.30


def ibkr_to_yahoo(ticker: str) -> str:
    """Convert IBKR ticker format to Yahoo Finance format.

    Class-share tickers use a space in IBKR format ("BRK B") and a hyphen
    in Yahoo format ("BRK-B").
    """
    return ticker.replace(" ", "-")


def screen(min_gap_pct: float, min_price: float) -> dict:
    start_time = time.time()

    ibkr_tickers = list(SP500_TICKERS)
    yahoo_tickers = [ibkr_to_yahoo(t) for t in ibkr_tickers]
    yahoo_to_ibkr_map = {y: i for y, i in zip(yahoo_tickers, ibkr_tickers)}

    try:
        data = yf.download(
            tickers=" ".join(yahoo_tickers),
            period="2d",
            interval="1d",
            group_by="ticker",
            threads=5,  # NOTE: keep at 5, not 10 - higher concurrency exhausts
                        # file descriptors when run under Task Scheduler.
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": f"yf.download raised: {type(e).__name__}: {e}",
            "total_screened": len(ibkr_tickers),
            "elapsed_seconds": round(elapsed, 2),
        }

    total_screened = len(ibkr_tickers)

    if data is None or data.empty:
        elapsed = time.time() - start_time
        result = {
            "success": False,
            "error": "yf.download returned an empty dataframe",
            "total_screened": total_screened,
            "elapsed_seconds": round(elapsed, 2),
        }
        return result

    survivors = []  # list of dicts: ticker, gap_pct, today_open, yesterday_close
    below_gap = 0
    below_price = 0
    failed = 0

    for yahoo_ticker in yahoo_tickers:
        ibkr_ticker = yahoo_to_ibkr_map[yahoo_ticker]
        try:
            bars = data[yahoo_ticker]

            yesterday_close = bars.iloc[-2]["Close"]
            today_open = bars.iloc[-1]["Open"]
            today_close = bars.iloc[-1]["Close"]  # live intraday print during market hours
            today_high = bars.iloc[-1]["High"]
            today_low = bars.iloc[-1]["Low"]

            gap_pct = (today_close - yesterday_close) / yesterday_close * 100

            # NaN/inf guard: comparisons like "NaN < min_gap_pct" are always
            # False in Python, so bad data (e.g. missing prior close) would
            # otherwise silently slip past BOTH the below_price and below_gap
            # checks below and get treated as a passing survivor.
            if not (
                math.isfinite(today_close)
                and math.isfinite(yesterday_close)
                and math.isfinite(gap_pct)
            ):
                failed += 1
                continue

            if today_close < min_price:
                below_price += 1
                continue
            if gap_pct < min_gap_pct:
                below_gap += 1
                continue

            survivors.append(
                {
                    "ticker": ibkr_ticker,
                    "gap_pct": float(gap_pct),
                    "today_open": float(today_open),
                    "yesterday_close": float(yesterday_close),
                    "today_close": float(today_close),
                    "today_high": float(today_high),
                    "today_low": float(today_low),
                }
            )
        except (KeyError, IndexError, ValueError):
            failed += 1
            continue

    survivors.sort(key=lambda s: s["gap_pct"], reverse=True)
    survivors_before_cap = len(survivors)
    survivors = survivors[:MAX_SURVIVORS]

    elapsed = time.time() - start_time

    result = {
        "success": True,
        "total_screened": total_screened,
        "survivors": survivors,
        "survivors_count": len(survivors),
        "survivors_before_cap": survivors_before_cap,
        "below_gap": below_gap,
        "below_price": below_price,
        "failed": failed,
        "elapsed_seconds": round(elapsed, 2),
    }
    return result


def write_watchlist(
    survivors: list[dict],
    min_gap_pct: float,
    min_price: float,
    total_screened: int,
    survivors_before_cap: int | None = None,
) -> None:
    now_et = datetime.now(ET)
    timestamp_str = now_et.strftime("%Y-%m-%d %H:%M %Z")

    cap_note = (
        f"{len(survivors)} of {survivors_before_cap} that passed (capped at {MAX_SURVIVORS})"
        if survivors_before_cap is not None and survivors_before_cap > len(survivors)
        else f"{len(survivors)}"
    )

    lines = [
        f"# Auto-generated by morning_prefilter.py at {timestamp_str}",
        f"# Filters: gap >= {min_gap_pct}%, price >= ${min_price}",
        "# Source: yfinance (screening only); IBKR handles execution",
        f"# Survivors: {cap_note} (screened {total_screened} tickers)",
        "#",
        "# ticker  # gap +X.XX%  open $X.XX  prev $X.XX",
    ]

    for s in survivors:
        lines.append(
            f"{s['ticker']}  # gap +{s['gap_pct']:.2f}%  "
            f"open ${s['today_open']:.2f}  prev ${s['yesterday_close']:.2f}"
        )

    WATCHLIST_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP_PCT)
    parser.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = screen(args.min_gap, args.min_price)

    hhmm_et = datetime.now(ET).strftime("%H:%M")

    if not result.get("success"):
        print(json.dumps(result))
        if not args.dry_run:
            notify(f"Prefilter FAILED {hhmm_et} ET", result.get("error", "unknown error"), "high")
        sys.exit(1)

    survivors = result["survivors"]
    total = result["total_screened"]
    failed = result["failed"]
    survivors_before_cap = result["survivors_before_cap"]

    # Degradation tripwires
    if total > 0:
        failure_ratio = failed / total
        if failure_ratio >= ALERT_FAILURE_RATIO_SEVERE:
            alert_msg = "ALERT: Yahoo-wide failure suspected; check yfinance changelog"
            print(alert_msg, file=sys.stderr)
            if not args.dry_run:
                notify(f"Prefilter ALERT {hhmm_et} ET", alert_msg, "high")
        elif failure_ratio >= ALERT_FAILURE_RATIO_DEGRADED:
            alert_msg = f"ALERT: yfinance degradation (failed {failed}/{total})"
            print(alert_msg, file=sys.stderr)
            if not args.dry_run:
                notify(f"Prefilter ALERT {hhmm_et} ET", alert_msg, "high")

    watchlist_path_str = None
    if not args.dry_run:
        write_watchlist(survivors, args.min_gap, args.min_price, total, survivors_before_cap)
        watchlist_path_str = str(WATCHLIST_PATH)

    top_20_survivors = [f"{s['ticker']} (+{s['gap_pct']:.2f}%)" for s in survivors]

    summary = {
        "success": True,
        "total_screened": total,
        "survivors_count": len(survivors),
        "survivors_before_cap": survivors_before_cap,
        "below_gap": result["below_gap"],
        "below_price": result["below_price"],
        "failed": failed,
        "elapsed_seconds": result["elapsed_seconds"],
        "top_20_survivors": top_20_survivors,
        "watchlist_path": watchlist_path_str,
    }

    print(json.dumps(summary))

    if not args.dry_run:
        bullet_lines = "\n".join(f"- {s}" for s in top_20_survivors) if top_20_survivors else "- (none)"
        cap_note = f" ({survivors_before_cap} passed, capped)" if survivors_before_cap > len(survivors) else ""
        body = f"{result['survivors_count']}/{total} survivors in {result['elapsed_seconds']}s{cap_note}\n{bullet_lines}"
        notify(f"Prefilter {hhmm_et} ET", body, "default")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with PREFILTER_ERRORS_LOG.open("a") as f:
            f.write(f"--- {datetime.now(timezone.utc).isoformat()} ---\n")
            f.write(traceback.format_exc())
            f.write("\n")
        try:
            hhmm_et = datetime.now(ET).strftime("%H:%M")
            notify(f"Prefilter CRASHED {hhmm_et} ET", str(exc)[:500], "high")
        except Exception:
            pass
        sys.exit(1)