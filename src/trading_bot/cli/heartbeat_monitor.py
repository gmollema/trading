"""Dead-man's-switch monitor for the SMC paper-trading bot.

Deliberately independent of IBKR/TWS -- it only reads the local heartbeat
file (written by smc_cycle.py on every cycle that successfully connects
and completes, see util/heartbeat.py) and pushes a notification via
notify() over HTTP. It must keep working precisely when the thing it's
checking for (TWS being reachable) is broken.

Motivated by both bots going silent for 8 days (TWS itself was down,
WinError 1225) with nobody noticing until a manual log check -- neither
bot's own logging escalates "I haven't run successfully in hours," it
just logs each connection failure and waits for the next scheduled
cycle. This is the escalation that was missing.

Gap-and-go (cycle.py) is decommissioned and its scheduled task disabled,
so its heartbeat is intentionally never refreshed -- checking it here
would page forever about a bot that isn't supposed to be running.

Usage:
    python -m trading_bot.cli.heartbeat_monitor
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

SMC_HEARTBEAT_PATH = Path("smc_heartbeat.json")

# Cycles run every 5 min; this allows a few missed cycles (e.g. one
# transient, self-healing connection blip) before treating it as a real
# outage worth paging about, rather than alerting on every single retry.
STALE_THRESHOLD_MINUTES = 20

# Checked only Mon-Fri, 10:20-16:00 ET: 20 minutes after the 10:00 ET
# open gives each bot's first few cycles a chance to write a fresh
# heartbeat before comparing against (what would otherwise be) yesterday's
# leftover one -- avoids a false alarm right at the open.
CHECK_START_HHMM = "10:20"
CHECK_END_HHMM = "16:00"


def _in_check_window(now_et: datetime) -> bool:
    if now_et.weekday() >= 5:  # Saturday, Sunday
        return False
    hm = now_et.strftime("%H:%M")
    return CHECK_START_HHMM <= hm <= CHECK_END_HHMM


def check_one(name: str, path: Path) -> str | None:
    """Returns an alert message if `path`'s heartbeat is missing/stale,
    else None."""
    from trading_bot.util.heartbeat import read_heartbeat_age_minutes

    age_minutes = read_heartbeat_age_minutes(path)
    if age_minutes is None:
        return f"{name}: no heartbeat on record at {path} (bot may have never run, or the file is unreadable)"
    if age_minutes > STALE_THRESHOLD_MINUTES:
        return f"{name}: last successful cycle {age_minutes:.0f} min ago (threshold {STALE_THRESHOLD_MINUTES}min) -- check TWS/IBKR connectivity"
    return None


def main() -> int:
    now_et = datetime.now(ET)
    if not _in_check_window(now_et):
        print(f"Outside check window ({CHECK_START_HHMM}-{CHECK_END_HHMM} ET, Mon-Fri). Exiting.")
        return 0

    from trading_bot.util.notifier import notify

    alerts = [
        msg
        for msg in (check_one("SMC", SMC_HEARTBEAT_PATH),)
        if msg is not None
    ]

    if alerts:
        body = "\n".join(alerts)
        print(body)
        notify("Bot heartbeat ALERT", body, "high")
    else:
        print("SMC healthy.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
