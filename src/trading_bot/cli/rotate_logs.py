"""Log and trade-history rotation for the IBKR paper-trading bot.

Safe to run at any time of day. Always exits 0, even when there is
nothing to rotate.

Usage:
    python -m trading_bot.cli.rotate_logs
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_bot.util.notifier import notify

LOGS_DIR = Path("logs")
ARCHIVE_DIR = LOGS_DIR / "archive"
TRADES_CSV_PATH = Path("trades.csv")
SAFETY_LOG_PATH = Path("safety-check-log.json")
ROTATE_LOGS_ERRORS_LOG = LOGS_DIR / "rotate_logs_errors.log"

ET = ZoneInfo("America/New_York")

TRADES_RETENTION_DAYS = 90
SAFETY_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


def rotate_generic_logs() -> int:
    """Move any logs/*.log or logs/*.jsonl file whose mtime's ET date is
    earlier than today into logs/archive/YYYY-MM-DD/ (dated by that
    file's own mtime), via atomic os.replace."""
    if not LOGS_DIR.exists():
        return 0

    today = datetime.now(ET).date()
    count = 0

    for f in LOGS_DIR.iterdir():
        if not f.is_file():
            continue
        if f.suffix not in (".log", ".jsonl"):
            continue

        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=ET)
        if mtime.date() >= today:
            continue

        date_str = mtime.strftime("%Y-%m-%d")
        dest_dir = ARCHIVE_DIR / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / f.name
        if dest_path.exists():
            # Avoid clobbering an already-archived file of the same name.
            dest_path = dest_dir / f"{f.stem}_{int(f.stat().st_mtime)}{f.suffix}"

        os.replace(f, dest_path)
        count += 1

    return count


def rotate_trades_csv() -> int:
    """Keep the last TRADES_RETENTION_DAYS days of rows in trades.csv;
    archive older rows to logs/archive/trades_YYYYMMDD.csv (today's date).
    Returns 1 if any rows were archived, else 0."""
    if not TRADES_CSV_PATH.exists():
        return 0

    with TRADES_CSV_PATH.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows or not fieldnames:
        return 0

    cutoff = datetime.now(ET) - timedelta(days=TRADES_RETENTION_DAYS)

    keep_rows = []
    old_rows = []
    for row in rows:
        ts_str = row.get("timestamp_iso", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            # Unparseable timestamp: keep the row rather than risk losing data.
            keep_rows.append(row)
            continue

        if ts >= cutoff:
            keep_rows.append(row)
        else:
            old_rows.append(row)

    if not old_rows:
        return 0

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now(ET).strftime("%Y%m%d")
    archive_path = ARCHIVE_DIR / f"trades_{today_str}.csv"

    file_exists = archive_path.exists()
    with archive_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in old_rows:
            writer.writerow(row)

    tmp_path = TRADES_CSV_PATH.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in keep_rows:
            writer.writerow(row)
    os.replace(tmp_path, TRADES_CSV_PATH)

    return 1


def rotate_safety_log() -> int:
    """If safety-check-log.json exceeds SAFETY_LOG_MAX_BYTES, archive the
    entire file with today's date prefix and start a fresh empty file."""
    if not SAFETY_LOG_PATH.exists():
        return 0

    size = SAFETY_LOG_PATH.stat().st_size
    if size <= SAFETY_LOG_MAX_BYTES:
        return 0

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now(ET).strftime("%Y%m%d")
    dest_path = ARCHIVE_DIR / f"safety-check-log_{today_str}.json"
    if dest_path.exists():
        dest_path = ARCHIVE_DIR / f"safety-check-log_{today_str}_{int(time.time())}.json"

    os.replace(SAFETY_LOG_PATH, dest_path)
    SAFETY_LOG_PATH.touch()  # start fresh

    return 1


def main() -> int:
    total = 0
    total += rotate_generic_logs()
    total += rotate_trades_csv()
    total += rotate_safety_log()

    print(f"Rotated {total} files to logs/archive/")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with ROTATE_LOGS_ERRORS_LOG.open("a") as f:
            f.write(f"--- {datetime.now(timezone.utc).isoformat()} ---\n")
            f.write(traceback.format_exc())
            f.write("\n")
        try:
            notify("Log Rotation CRASHED", str(exc)[:500], "high")
        except Exception:
            pass
        sys.exit(1)
