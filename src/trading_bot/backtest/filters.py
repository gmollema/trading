"""Pure D1-D3 / I2-I3 entry-filter evaluation for the backtest engine.

Ports cycle.evaluate_entry_filters's boolean logic
(src/trading_bot/cli/cycle.py:624) to operate on precomputed context (from
backtest.data) instead of a live yfinance snapshot.

I1 (above premarket high) is NOT evaluated here: the cached intraday bars
were fetched RTH-only (useRTH=True in backtest_fetch_data.py), so no
premarket bar exists for any ticker/day and I1 cannot be computed. I1 is
therefore always treated as passing -- a known, documented fidelity gap
versus the live bot, not an oversight.
"""

from __future__ import annotations


def evaluate_entry(daily_ctx: dict, intraday_ctx: dict, price: float, rules: dict) -> tuple[bool, list[str]]:
    """Evaluate D1-D3 (daily) and I2-I3 (intraday) filters for one bar.

    Args:
        daily_ctx: {"prior_day_high", "prior_day_close", "sma200"}. Any
            missing (None/NaN) value fails closed with "insufficient daily
            data" -- consistent with the live bot skipping tickers it can't
            yet compute 200 days of history for.
        intraday_ctx: {"today_hod", "rvol"}, as-of the bar being evaluated.
        price: the bar's close, used as "current price".
        rules: the parsed rules.json dict.

    Returns:
        (passed, reasons) -- reasons is empty iff passed is True.
    """
    if daily_ctx is None or any(
        _is_missing(daily_ctx.get(k)) for k in ("prior_day_high", "prior_day_close", "sma200")
    ):
        return False, ["insufficient daily data"]

    if intraday_ctx is None or _is_missing(intraday_ctx.get("today_hod")):
        return False, ["insufficient intraday data"]

    reasons: list[str] = []
    daily_filters = rules["daily_filters"]
    intraday_filters = rules["intraday_filters"]

    d1_pass = price > daily_ctx["prior_day_high"]
    if daily_filters.get("D1_above_prior_day_high") and not d1_pass:
        reasons.append("D1 fail: price not above prior day high")

    d2_pass = daily_ctx["prior_day_close"] > daily_ctx["sma200"]
    if daily_filters.get("D2_prior_close_above_sma200") and not d2_pass:
        reasons.append("D2 fail: prior close not above SMA200")

    gap_pct = (price - daily_ctx["prior_day_close"]) / daily_ctx["prior_day_close"] * 100
    d3_threshold = daily_filters.get("D3_min_gap_pct_from_prior_close", 0)
    d3_pass = gap_pct >= d3_threshold
    if not d3_pass:
        reasons.append(f"D3 fail: gap {gap_pct:.2f}% < {d3_threshold}%")

    i2_pass = price >= intraday_ctx["today_hod"]
    if intraday_filters.get("I2_above_today_hod") and not i2_pass:
        reasons.append("I2 fail: price not at/above today HOD")

    rvol = intraday_ctx.get("rvol")
    rvol_min = intraday_filters.get("I3_rvol_min", 0)
    i3_pass = not _is_missing(rvol) and rvol >= rvol_min
    if not i3_pass:
        rvol_str = f"{rvol:.2f}" if not _is_missing(rvol) else "N/A"
        reasons.append(f"I3 fail: rvol {rvol_str} < {rvol_min}")

    passed = bool(d1_pass and d2_pass and d3_pass and i2_pass and i3_pass)
    return passed, reasons


def _is_missing(value) -> bool:
    """True for None or NaN (NaN != NaN, so this needs no math/numpy import)."""
    if value is None:
        return True
    try:
        return value != value
    except TypeError:
        return True
