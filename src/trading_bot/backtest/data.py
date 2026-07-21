"""Data loading and vectorized context precomputation for the backtest engine.

Loads cached daily/intraday CSVs (written by backtest_fetch_data.py) and
precomputes, per symbol, the same daily (prior-day-high/close, SMA200) and
intraday (running HOD/LOD, cumulative volume, RVOL) context the live bot
computes on the fly in cycle.py -- but vectorized across the whole cached
history at once.

Invariant: every value at row t is computed using only bars up to and
including t. This mirrors the live bot's per-cycle snapshot semantics
(a yfinance call at time t can only ever see t and earlier) and is what
keeps the backtest from silently cheating via lookahead bias.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_bot.cli.backtest_fetch_data import DAILY_DIR, INTRADAY_DIR, safe_filename

SMA200_WINDOW = 200
ET = "America/New_York"


def load_daily(ticker: str, daily_dir: Path = DAILY_DIR) -> pd.DataFrame | None:
    """Load cached daily bars for `ticker`, tz-aware and sorted by date.

    Returns None if no cached file exists for this ticker.
    """
    path = daily_dir / safe_filename(ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(ET)
    return df.sort_values("Date").reset_index(drop=True)


def load_intraday(ticker: str, intraday_dir: Path = INTRADAY_DIR) -> pd.DataFrame | None:
    """Load cached 5-min intraday bars for `ticker`, tz-aware and sorted.

    Returns None if no cached file exists for this ticker.
    """
    path = intraday_dir / safe_filename(ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(ET)
    return df.sort_values("date").reset_index(drop=True)


def compute_daily_context(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add prior_day_high/prior_day_close/sma200 columns.

    Mirrors cycle.get_daily_context, which snapshots hist.iloc[-2] as "prior
    day" and averages the 200 trading days ending there (excluding today's
    own bar). In a fully-formed historical series (no partial "today" bar),
    that is exactly a 1-row shift: prior_day_* = yesterday's own values, and
    sma200 = the 200-day rolling mean of Close as of yesterday's close.
    """
    out = daily_df.copy()
    out["prior_day_high"] = out["High"].shift(1)
    out["prior_day_close"] = out["Close"].shift(1)
    out["sma200"] = out["Close"].rolling(SMA200_WINDOW).mean().shift(1)
    return out


def compute_intraday_context(intraday_df: pd.DataFrame, rvol_lookback_days: int) -> pd.DataFrame:
    """Add trading_date/running_hod/running_lod/running_cum_vol/rvol columns.

    RVOL denominator: mean cumulative volume at the SAME time-of-day over the
    trailing `rvol_lookback_days` trading days (cycle.get_intraday_context's
    "past_cum_volumes" logic), vectorized via a (date x time-of-day) pivot
    instead of live per-symbol yfinance calls.
    """
    out = intraday_df.copy()
    out["trading_date"] = out["date"].dt.date
    out["time_of_day"] = out["date"].dt.time

    grouped = out.groupby("trading_date", sort=False)
    out["running_hod"] = grouped["high"].cummax()
    out["running_lod"] = grouped["low"].cummin()
    out["running_cum_vol"] = grouped["volume"].cumsum()

    pivot = out.pivot_table(index="trading_date", columns="time_of_day", values="running_cum_vol")
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)

    # Forward-fill across time-of-day columns: an early-close day (e.g. a
    # holiday half day) has no bars past its close, but a later time-of-day
    # lookup on that day should still see "all volume traded so far" as its
    # contribution -- matching cycle.py's own "sum whatever bars exist up to
    # now_hm" behavior rather than silently contributing NaN.
    pivot_filled = pivot.ffill(axis=1)
    denom_pivot = pivot_filled.rolling(window=rvol_lookback_days, min_periods=1).mean().shift(1)

    denom_long = denom_pivot.reset_index().melt(
        id_vars="trading_date", var_name="time_of_day", value_name="rvol_denom"
    )
    out = out.merge(denom_long, on=["trading_date", "time_of_day"], how="left")

    out["rvol"] = out["running_cum_vol"] / out["rvol_denom"]
    out.loc[out["rvol_denom"].isna() | (out["rvol_denom"] <= 0), "rvol"] = float("nan")

    return out.drop(columns=["rvol_denom"])


def build_symbol_frame(
    ticker: str,
    rvol_lookback_days: int,
    daily_dir: Path = DAILY_DIR,
    intraday_dir: Path = INTRADAY_DIR,
) -> pd.DataFrame | None:
    """Load + fully precompute one symbol's per-bar backtest frame.

    Returns None if either the daily or intraday cache is missing for this
    ticker (the caller should simply skip that symbol).
    """
    daily_df = load_daily(ticker, daily_dir)
    intraday_df = load_intraday(ticker, intraday_dir)
    if daily_df is None or intraday_df is None:
        return None

    daily_ctx = compute_daily_context(daily_df)
    daily_ctx["trading_date"] = daily_ctx["Date"].dt.date
    daily_lookup = daily_ctx[["trading_date", "prior_day_high", "prior_day_close", "sma200"]]

    intraday_ctx = compute_intraday_context(intraday_df, rvol_lookback_days)
    merged = intraday_ctx.merge(daily_lookup, on="trading_date", how="left")
    merged["symbol"] = ticker
    return merged


def available_tickers(
    tickers: list[str], daily_dir: Path = DAILY_DIR, intraday_dir: Path = INTRADAY_DIR
) -> list[str]:
    """Filter `tickers` down to those with BOTH a cached daily and intraday
    file, preserving input order. Used to default the backtest CLI's
    universe to whatever's actually been downloaded."""
    return [
        t
        for t in tickers
        if (daily_dir / safe_filename(t)).exists() and (intraday_dir / safe_filename(t)).exists()
    ]
