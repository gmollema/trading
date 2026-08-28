"""Merge dated backtest_data fetch directories into one combined cache.

Why this exists: backtest_fetch_data.py's intraday step pulls IBKR's
`1 Y` window, so every fresh fetch silently drops whatever history has
aged past one year. The 5-min cache reaches back further than that (it
was first built when the then-current window covered earlier dates), and
that head is NOT re-fetchable from IBKR at any price. So a refresh must
be *merged* onto the existing cache, never written over it.

Usage (sources oldest-first, later sources win on conflicts):

    python -m trading_bot.cli.merge_backtest_data \
        --out backtest_data/intraday_5m_merged_2026-08-28 \
        backtest_data/intraday_5m_merged_2026-08-27 \
        backtest_data/intraday_5m_2026-08-27

Conflict rule: rows are keyed on their UTC instant and the LAST source
listed wins. That is deliberate -- the final bar of any fetch is often
captured mid-formation (a partial bar holding only part of the interval's
volume), and the newer fetch carries the settled version plus any late
consolidated-tape corrections.

Adjustment-basis guard (the reason this is not a plain concat):

  - Intraday bars come from IBKR whatToShow="TRADES", which is NOT
    dividend-adjusted -- two fetches agree on overlapping bars apart from
    late tape corrections and sub-cent rounding, so they normally splice
    cleanly. They ARE split-adjusted at fetch time, though, and that is
    not a footnote: a refetch after a split returns the whole series
    rescaled while the cached copy holds the raw prices that actually
    traded, putting the two on different bases. Confirmed on the first
    real use of this script -- MNST split 2:1 on 2026-07-15 and its
    overlap came back shifted by exactly -50.0000%.

    The cached side is the wrong one to keep. Raw prices carry a
    fabricated -50% single bar at the split, which a long-only strategy
    reads as a catastrophic gap and which poisons every order block and
    swing level derived from it. Rebase the cached head onto the
    refetch's adjusted scale (prices x ratio, volume / ratio) rather than
    splicing or discarding the head, which is not re-fetchable.

  - Daily bars come from yfinance with auto_adjust=True, which is back-
    adjusted for splits AND dividends. Every dividend paid between two
    fetches retroactively rescales every earlier row, so two daily fetches
    taken weeks apart sit on different price scales. Concatenating them
    splices those scales together and manufactures an overnight gap of up
    to ~1% at the seam -- precisely the artifact a gap-driven strategy
    would trade on. This script detects that and refuses by default.

Daily almost certainly does not need merging at all: the 2y yfinance
window reaches back far enough that SMA200 is warm well before the
intraday cache's own start date, and build_symbol_frame needs both, so
the intraday start is the binding constraint. Prefer promoting the newest
daily fetch directory wholesale over merging daily at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Two sources always disagree a little on overlapping rows, and the
# distinction that matters is WHY.
#
#   - Late consolidated-tape corrections and sub-cent rounding are noise:
#     unbiased, scattered in both directions, centred on zero.
#   - A dividend/split back-adjustment is systematic: it shifts nearly
#     every row older than the ex-date in ONE direction by a consistent
#     proportion.
#
# Counting how many rows differ does NOT separate these -- measured on the
# real cache, IBKR intraday refetches disagree on 23-62% of overlapping
# bars purely from rounding. What separates them cleanly is the MEDIAN
# SIGNED deviation: that noise sits at ~0.0001% with a near-even negative/
# positive split, while genuine yfinance daily re-adjustments sit at
# 0.086%-0.741% with 94-98% of rows moving the same way. Three orders of
# magnitude apart, so the threshold below sits comfortably in the gap. It
# doubles as a materiality floor: a shift under 0.01% is sub-cent on a
# $100 stock and not worth refusing a merge over.
REL_TOL = 1e-6
RESCALE_MEDIAN_REL = 1e-4

# The median is taken over the OLDEST slice of the overlap rather than all
# of it. A dividend part-way through the overlap leaves recent rows
# untouched (median 0) while still rescaling everything before it, and it
# is precisely those older rows whose scale the merge carries forward into
# the prepended history.
#
# The slice is floored at MIN_OVERLAP_ROWS (widening to the whole overlap
# when that is shorter): a median over two or three rows is not robust, and
# one late-corrected bar landing in a tiny slice would swing it past the
# threshold on its own.
OLDEST_SLICE_FRACTION = 0.10

# Below this many shared rows there is too little signal to judge, and one
# late-corrected bar can dominate the statistic. Real overlaps run to
# hundreds of daily rows or tens of thousands of intraday ones, so this
# floor never binds in practice. The trade-off is that a genuine rescale
# spanning fewer than this many shared rows goes unflagged -- acceptable,
# since such an overlap is too short to merge meaningfully anyway.
MIN_OVERLAP_ROWS = 20

# Timestamp column each fetcher writes.
DAILY_DATE_COL = "Date"
INTRADAY_DATE_COL = "date"


def detect_date_col(path: Path) -> str | None:
    """Return the timestamp column for a cache CSV, or None if unrecognised.

    yfinance writes a capital-D `Date` index; the IBKR intraday step writes
    a lowercase `date` column. That difference is the only reliable way to
    tell a daily cache file from an intraday one without trusting the
    directory name.
    """
    with path.open() as fh:
        header = fh.readline().strip()
    first = header.split(",")[0].lstrip("﻿")
    if first == DAILY_DATE_COL:
        return DAILY_DATE_COL
    if first == INTRADAY_DATE_COL:
        return INTRADAY_DATE_COL
    return None


def read_cache_csv(path: Path, date_col: str) -> pd.DataFrame:
    """Load one cache CSV with its timestamp parsed to UTC.

    Parsing with utc=True normalises the two on-disk conventions -- the
    fetch step writes ET offsets (-04:00/-05:00, varying with DST) while
    previously merged directories hold UTC -- onto one comparable instant.
    """
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    return df.sort_values(date_col).reset_index(drop=True)


def close_col(date_col: str) -> str:
    return "Close" if date_col == DAILY_DATE_COL else "close"


def compare_overlap(
    older: pd.DataFrame, newer: pd.DataFrame, date_col: str
) -> tuple[int, int, float]:
    """Return (overlapping rows, rows that disagree, median signed shift).

    The third value is the median of (newer - older) / older across the
    oldest OLDEST_SLICE_FRACTION of the shared rows, SIGNED. It is the
    statistic the adjustment-basis guard turns on: unbiased tape/rounding
    noise cancels to ~0, while a back-adjustment leaves a consistent
    non-zero offset. Comparison is on the close, which any split or
    dividend re-adjustment rescales uniformly.
    """
    col = close_col(date_col)
    a = older.set_index(date_col)[col]
    b = newer.set_index(date_col)[col]
    a = a[~a.index.duplicated(keep="last")]
    b = b[~b.index.duplicated(keep="last")]
    idx = a.index.intersection(b.index).sort_values()
    if len(idx) == 0:
        return 0, 0, 0.0
    denom = a.loc[idx].abs()
    signed = ((b.loc[idx] - a.loc[idx]) / denom.where(denom > 0)).dropna()
    if signed.empty:
        return len(idx), 0, 0.0
    n_diff = int((signed.abs() > REL_TOL).sum())
    k = min(len(signed), max(MIN_OVERLAP_ROWS, int(len(signed) * OLDEST_SLICE_FRACTION)))
    return len(idx), n_diff, float(signed.iloc[:k].median())


def merge_ticker(frames: list[pd.DataFrame], date_col: str) -> pd.DataFrame:
    """Concatenate sources and keep the last source's row per timestamp."""
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(date_col, kind="stable")
    combined = combined.drop_duplicates(subset=date_col, keep="last")
    return combined.sort_values(date_col).reset_index(drop=True)


def write_ticker(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge dated backtest_data fetch dirs, newest source winning per timestamp."
    )
    parser.add_argument(
        "sources", nargs="+", help="source dirs, OLDEST first; later sources win conflicts"
    )
    parser.add_argument("--out", required=True, help="output directory (must not already hold CSVs)")
    parser.add_argument(
        "--on-rescale",
        choices=("error", "warn", "skip"),
        default="error",
        help="what to do with a ticker whose sources sit on different price-adjustment "
        "bases (see module docstring): 'error' aborts before writing anything (default), "
        "'warn' merges it anyway, 'skip' leaves it out of the output",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would happen, write nothing"
    )
    parser.add_argument("--force", action="store_true", help="allow writing into a non-empty --out")
    args = parser.parse_args()

    sources = [Path(s) for s in args.sources]
    for s in sources:
        if not s.is_dir():
            print(f"error: source is not a directory: {s}", file=sys.stderr)
            return 2
    if len(sources) < 2:
        print("error: need at least two source directories to merge", file=sys.stderr)
        return 2

    out = Path(args.out)
    if out.exists() and any(out.glob("*.csv")) and not args.force and not args.dry_run:
        print(f"error: {out} already contains CSVs; pass --force to write into it", file=sys.stderr)
        return 2
    if out.resolve() in {s.resolve() for s in sources}:
        print("error: --out must not be one of the sources", file=sys.stderr)
        return 2

    filenames = sorted({p.name for s in sources for p in s.glob("*.csv")})
    print(f"{len(filenames)} tickers across {len(sources)} sources -> {out}")

    rescaled: list[str] = []
    unreadable: list[str] = []
    plan: list[tuple[Path, pd.DataFrame]] = []

    for name in filenames:
        present = [s / name for s in sources if (s / name).exists()]
        date_cols = {detect_date_col(p) for p in present}
        if None in date_cols or len(date_cols) != 1:
            unreadable.append(name)
            continue
        date_col = date_cols.pop()

        frames = [read_cache_csv(p, date_col) for p in present]

        # Compare each source against the one before it: a re-adjustment
        # between ANY consecutive pair contaminates the merged result.
        flagged = False
        for older, newer in zip(frames, frames[1:]):
            n_overlap, n_diff, median_shift = compare_overlap(older, newer, date_col)
            if n_overlap >= MIN_OVERLAP_ROWS and abs(median_shift) > RESCALE_MEDIAN_REL:
                if len(rescaled) < 10:
                    print(
                        f"  RESCALED {name}: oldest shared rows shifted by "
                        f"{median_shift * 100:+.4f}% ({n_diff}/{n_overlap} rows differ) "
                        "-- sources are on different adjustment bases"
                    )
                flagged = True
                break
        if flagged:
            rescaled.append(name)
            if args.on_rescale == "skip":
                continue

        plan.append((out / name, merge_ticker(frames, date_col)))

    if rescaled:
        print(
            f"\n{len(rescaled)} ticker(s) have sources on different price-adjustment bases. "
            "Merging them splices two price scales together and fabricates a gap at the seam."
        )
        if args.on_rescale == "error":
            print(
                "Refusing to write. This is the expected outcome for DAILY dirs (yfinance "
                "auto_adjust re-scales history on every dividend) -- promote the newest daily "
                "fetch wholesale instead of merging it. Pass --on-rescale warn to override.",
                file=sys.stderr,
            )
            return 1

    if unreadable:
        print(f"skipped {len(unreadable)} file(s) with an unrecognised header: {unreadable[:5]}")

    if args.dry_run:
        print(f"\ndry run: would write {len(plan)} file(s) to {out}")
        return 0

    for path, df in plan:
        write_ticker(df, path)

    print(f"\nwrote {len(plan)} file(s) to {out}")
    if rescaled and args.on_rescale == "warn":
        print(f"WARNING: {len(rescaled)} of them spliced across adjustment bases: {rescaled[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
