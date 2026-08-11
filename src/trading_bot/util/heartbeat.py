"""Shared heartbeat file read/write for the dead-man's-switch monitor.

Each cycle bot (cycle.py, smc_cycle.py) writes its own heartbeat file once
it has SUCCESSFULLY completed a cycle -- specifically, only after
connecting to IBKR and finishing that cycle's work, not merely because
the scheduled task fired. A stuck/absent heartbeat during market hours
therefore reflects a real failure (e.g. IBKR/TWS unreachable), not just
"nothing to trade this cycle" -- see cli/heartbeat_monitor.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_heartbeat(path: Path, status: str) -> None:
    """Atomically write {"timestamp_iso": <now UTC>, "status": status} to
    `path`. Swallows all errors -- a heartbeat write must never be the
    reason a cycle's actual trading logic fails or raises."""
    try:
        payload = {"timestamp_iso": datetime.now(timezone.utc).isoformat(), "status": status}
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload))
        tmp_path.replace(path)
    except Exception:
        pass


def read_heartbeat_age_minutes(path: Path) -> float | None:
    """Minutes since `path`'s last recorded timestamp_iso, or None if the
    file is missing, empty, or unparseable (treated by the monitor as
    "no successful heartbeat on record" -- at least as alarming as stale)."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        ts = datetime.fromisoformat(payload["timestamp_iso"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    age = datetime.now(timezone.utc) - ts
    return age.total_seconds() / 60.0
