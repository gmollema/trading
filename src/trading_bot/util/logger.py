"""Robust, fail-safe error logging for system subsystems."""

from datetime import datetime, timezone
from pathlib import Path
import traceback

LOGS_DIR = Path("logs")
NOTIFY_ERRORS_LOG = LOGS_DIR / "notify_errors.log"


def log_error(context: str, exc: Exception) -> None:
    """Logs the given exception (with traceback) to disk. Swallows all errors if disk write fails."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with NOTIFY_ERRORS_LOG.open("a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now(timezone.utc).isoformat()} [{context}] ---\n")
            f.writelines(traceback.format_exception(exc))
            f.write("\n")
    except Exception:
        # If filesystem is read-only or full, fail silently so the core app keeps running.
        pass