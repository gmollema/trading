"""Tear down all HT_* Windows Task Scheduler entries created by
setup_schedule.py, and restore default power settings.

Safe to re-run: if no HT_* tasks exist, it reports that and exits 0.

Usage:
    python -m trading_bot.cli.cleanup_schedule
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys


def list_ht_tasks() -> list[str]:
    result = subprocess.run(
        ["schtasks", "/query", "/fo", "CSV", "/nh"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: schtasks /query failed: {result.stderr.strip()}", file=sys.stderr)
        return []

    task_names = []
    reader = csv.reader(io.StringIO(result.stdout))
    for row in reader:
        if not row:
            continue
        # First CSV column is the TaskName, e.g. "\HT_Cycle"
        task_name = row[0].strip().lstrip("\\")
        if task_name.startswith("HT_"):
            task_names.append(task_name)

    return sorted(set(task_names))


def delete_task(task_name: str) -> bool:
    result = subprocess.run(
        ["schtasks", "/delete", "/tn", task_name, "/f"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"FAILED to delete {task_name}: {result.stderr.strip()}", file=sys.stderr)
        return False
    print(f"Deleted {task_name}")
    return True


def restore_power_settings() -> None:
    result = subprocess.run(
        ["powercfg", "/change", "standby-timeout-ac", "30"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: failed to restore power settings: {result.stderr.strip()}", file=sys.stderr)


def main() -> None:
    ht_tasks = list_ht_tasks()

    if not ht_tasks:
        print("Nothing to clean.")
        sys.exit(0)

    deleted_count = 0
    for task_name in ht_tasks:
        if delete_task(task_name):
            deleted_count += 1

    restore_power_settings()

    print(f"CLEANUP DONE: {deleted_count} tasks deleted; default power settings restored.")


if __name__ == "__main__":
    main()
