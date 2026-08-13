"""Register Windows Task Scheduler entries for the IBKR paper-trading bot.

Creates 14 tasks (all HT_ prefixed), all running in the current user's
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


def apply_post_create_settings(task_name: str) -> None:
    """schtasks.exe /create has no CLI flags for these -- toggle them via
    the Settings/Actions objects afterward:
      - Hidden = true: suppresses the console for the task's own process,
        but does NOT reliably suppress the brief console flash from a
        cmd.exe wrapper launching underneath it (a known Task Scheduler
        quirk) -- see quoted_tr's WorkingDirectory comment for the other
        half of that fix.
      - DisallowStartIfOnBatteries / StopIfGoingOnBatteries = false:
        schtasks /create defaults BOTH to true (its laptop-friendly
        default for background maintenance tasks), which silently skips
        every future trigger the moment this laptop is unplugged -- no
        error anywhere, since the task's process never even starts.
        Confirmed in practice: both trading bots stopped firing for over
        an hour with zero log entries once the laptop switched to
        battery power. An unattended trading bot needs to keep running
        regardless of power source.
      - Actions[0].WorkingDirectory = PROJECT_DIR: schtasks /tr has no
        flag for this, which is why quoted_tr used to route through
        `cmd /c cd /d ... &&` instead -- but that cmd.exe hop is itself
        the main source of the visible console flash on every run, even
        with Hidden=true set above. Setting it here instead lets
        quoted_tr call the venv's python.exe directly.
    """
    ps_cmd = (
        f"$t = Get-ScheduledTask -TaskName '{task_name}'; "
        f"$t.Settings.Hidden = $true; "
        f"$t.Settings.DisallowStartIfOnBatteries = $false; "
        f"$t.Settings.StopIfGoingOnBatteries = $false; "
        f"$t.Actions[0].WorkingDirectory = '{PROJECT_DIR}'; "
        f"Set-ScheduledTask -TaskName '{task_name}' -Settings $t.Settings -Action $t.Actions | Out-Null"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: failed to apply settings for {task_name}: {result.stderr.strip()}", file=sys.stderr)


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
        apply_post_create_settings(task_name)


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


def build_interval_task(task_name: str, tr: str, hh: int, mm: int, interval_minutes: int) -> None:
    local_hhmm = et_time_to_local_hhmm(hh, mm)
    schedule_args = [
        "/sc", "MINUTE",
        "/MO", str(interval_minutes),
        "/ST", local_hhmm,
    ]
    run_schtasks_create(task_name, tr, schedule_args)


def quoted_tr(module: str) -> str:
    """Build a schtasks /tr value that runs the given module with the
    venv's python via -m (requires the package to be installed in the
    venv: pip install -e .). Calls python.exe directly rather than
    routing through `cmd /c cd /d ... &&` -- that cmd.exe hop used to be
    how relative data paths (trades.csv, logs/, rules.json) resolved
    correctly, but launching a console app to launch another console app
    is exactly what causes Task Scheduler to flash a visible window even
    when Settings.Hidden is true. apply_post_create_settings sets the
    task Action's WorkingDirectory to PROJECT_DIR instead, which gets the
    same correct-relative-paths behavior without the extra hop."""
    return f'"{VENV_PY}" -m {module}'


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
    heartbeat_monitor_tr = quoted_tr("trading_bot.cli.heartbeat_monitor")
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

    # HT_HeartbeatMonitor - every 30 min starting 10:20 ET, daily (its own
    # check window, 10:20-16:00 ET Mon-Fri, is enforced inside
    # heartbeat_monitor.py, same "scheduling is coarse, the script's own
    # gate is precise" convention as HT_Cycle/HT_SMC_Cycle). Dead-man's
    # switch for both bots: pages if either hasn't completed a cycle
    # recently, independent of IBKR -- see heartbeat_monitor.py's docstring
    # for why this was added (both bots went silent for 8 days, unnoticed,
    # when TWS itself went down).
    build_interval_task("HT_HeartbeatMonitor", heartbeat_monitor_tr, 10, 20, 30)

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
        "SCHEDULED: 14 tasks created (incl. HT_SMC_Prefilter/HT_SMC_Cycle for the SMC bot, "
        "HT_HeartbeatMonitor as a dead-man's switch for both). "
        "The bots will fire on their own starting next market open. Watch Telegram."
    )


if __name__ == "__main__":
    main()