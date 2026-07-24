"""Register Windows Task Scheduler entries for the IBKR paper-trading bot.

Creates 11 tasks (all HT_ prefixed), all running in the current user's
context (no admin elevation, no SYSTEM account). Safe to re-run - each
schtasks /create call uses /F to overwrite an existing task of the same
name.

Usage:
    python -m trading_bot.cli.setup_schedule
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VENV_PY = Path(".venv/Scripts/python.exe").resolve()
PROJECT_DIR = Path(".").resolve()

ET = ZoneInfo("America/New_York")
LOCAL_TZ = datetime.now().astimezone().tzinfo


def et_time_to_local_hhmm(hh: int, mm: int) -> str:
    """Convert an HH:MM ET wall-clock time (today's date, for DST
    correctness) to local machine time, returned as 'HH:MM'."""
    now_et = datetime.now(ET)
    et_dt = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0)
    local_dt = et_dt.astimezone(LOCAL_TZ)
    return local_dt.strftime("%H:%M")


def hide_task_window(task_name: str) -> None:
    """schtasks.exe has no /hidden flag, so toggle it via the Settings
    object afterward -- otherwise every run under the interactive logon
    flashes a visible cmd.exe console on screen."""
    ps_cmd = (
        f"$t = Get-ScheduledTask -TaskName '{task_name}'; "
        f"$t.Settings.Hidden = $true; "
        f"Set-ScheduledTask -TaskName '{task_name}' -Settings $t.Settings | Out-Null"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: failed to hide window for {task_name}: {result.stderr.strip()}", file=sys.stderr)


def run_schtasks_create(task_name: str, tr: str, schedule_args: list[str]) -> None:
    cmd = [
        "schtasks",
        "/create",
        "/tn", task_name,
        "/tr", tr,
        *schedule_args,
        "/F",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED to create {task_name}: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"Created {task_name}")
        hide_task_window(task_name)


def build_weekly_task(task_name: str, tr: str, hh: int, mm: int) -> None:
    local_hhmm = et_time_to_local_hhmm(hh, mm)
    schedule_args = [
        "/sc", "WEEKLY",
        "/D", "MON,TUE,WED,THU,FRI",
        "/ST", local_hhmm,
    ]
    run_schtasks_create(task_name, tr, schedule_args)


def build_cycle_task(task_name: str, tr: str, hh: int, mm: int) -> None:
    local_hhmm = et_time_to_local_hhmm(hh, mm)
    schedule_args = [
        "/sc", "MINUTE",
        "/MO", "5",
        "/ST", local_hhmm,
    ]
    run_schtasks_create(task_name, tr, schedule_args)


def quoted_tr(module: str) -> str:
    """Build a schtasks /tr value that (a) cd's into the project dir so all
    relative data paths (trades.csv, logs/, rules.json) resolve correctly,
    and (b) runs the given module with the venv's python via -m (requires
    the package to be installed in the venv: pip install -e .)."""
    return f'cmd /c cd /d "{PROJECT_DIR}" && "{VENV_PY}" -m {module}'


def main() -> None:
    if not VENV_PY.exists():
        print(f"ERROR: venv python not found at {VENV_PY}. Create the venv first.", file=sys.stderr)
        sys.exit(1)

    rotate_logs_tr = quoted_tr("trading_bot.cli.rotate_logs")
    prefilter_tr = quoted_tr("trading_bot.cli.morning_prefilter")
    cycle_tr = quoted_tr("trading_bot.cli.cycle")
    dashboard_tr = quoted_tr("trading_bot.cli.compute_perf")
    smc_prefilter_tr = quoted_tr("trading_bot.cli.smc_prefilter")
    smc_cycle_tr = quoted_tr("trading_bot.cli.smc_cycle")
    keepawake_tr = "powercfg /change standby-timeout-ac 0"

    # HT_LogRotate - 09:25 ET, Mon-Fri
    build_weekly_task("HT_LogRotate", rotate_logs_tr, 9, 25)

    # HT_KeepAwake - 09:30 ET, Mon-Fri, raw command
    build_weekly_task("HT_KeepAwake", keepawake_tr, 9, 30)

    # HT_Prefilter_01..07 - 09:55 through 12:55 ET, every 30 min, Mon-Fri
    prefilter_times = [(9, 55), (10, 25), (10, 55), (11, 25), (11, 55), (12, 25), (12, 55)]
    for i, (hh, mm) in enumerate(prefilter_times, start=1):
        task_name = f"HT_Prefilter_{i:02d}"
        build_weekly_task(task_name, prefilter_tr, hh, mm)

    # HT_Cycle - every 5 minutes starting 10:00 ET, runs daily
    # (cycle.py's own time_gate handles weekend/off-hours; it exits in
    # under 1 second outside market hours, so daily/every-5-min is cheap.)
    build_cycle_task("HT_Cycle", cycle_tr, 10, 0)

    # HT_Dashboard - 16:05 ET, Mon-Fri
    build_weekly_task("HT_Dashboard", dashboard_tr, 16, 5)

    # HT_SMC_Prefilter - 09:40 ET, Mon-Fri: builds smc_watchlist.txt
    # (prior close > SMA200 = the SMC daily_trend_filter, computed from
    # prior-day data so once per morning is exactly enough).
    build_weekly_task("HT_SMC_Prefilter", smc_prefilter_tr, 9, 40)

    # HT_SMC_Cycle - every 5 minutes starting 10:02 ET (staggered 2 min
    # after HT_Cycle so the two bots' yfinance/IBKR bursts don't collide).
    # smc_cycle.py's own fast gate exits in under a second off-hours.
    build_cycle_task("HT_SMC_Cycle", smc_cycle_tr, 10, 2)

    print()
    # schtasks /query /tn does NOT support wildcards (e.g. "HT_*") - it
    # expects an exact task name, so query everything and filter here.
    result = subprocess.run(
        ["schtasks", "/query", "/fo", "TABLE"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    else:
        lines = result.stdout.splitlines()
        header_lines = lines[:3] if len(lines) >= 3 else lines
        ht_lines = [line for line in lines if line.strip().startswith("HT_")]
        print("\n".join(header_lines))
        print("\n".join(ht_lines))

    print(
        "SCHEDULED: 13 tasks created (incl. HT_SMC_Prefilter/HT_SMC_Cycle for the SMC bot). "
        "The bots will fire on their own starting next market open. Watch Telegram."
    )


if __name__ == "__main__":
    main()