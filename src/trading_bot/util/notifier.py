"""Fire-and-forget push notifications (Telegram + optional ntfy fallback)."""

from __future__ import annotations

import html
import os
import requests
from dotenv import load_dotenv

# Import our new isolated logging component
from .logger import log_error

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
NOTIFY_URL = os.environ.get("NOTIFY_URL", "").strip()

REQUEST_TIMEOUT_SECS = 5

# Every call site already marks crashes/failures/alerts as priority="high"
# and routine trade events as the "default" priority, so that same signal
# doubles as the icon selector -- no need to touch each of the ~15 call
# sites across cycle.py/morning_prefilter.py/compute_perf.py/rotate_logs.py.
ICON_DEFAULT = "\U0001F4C8"  # chart increasing
ICON_HIGH_PRIORITY = "❗"  # red bold exclamation mark


def _send_telegram(title: str, body: str) -> None:
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    safe_title = html.escape(title)
    safe_body = html.escape(body)

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"<b>{safe_title}</b>\n{safe_body}",
        "parse_mode": "HTML",
    }
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECS)
        response.raise_for_status()
    except Exception as e:
        log_error("telegram", e)


def _send_ntfy(title: str, body: str, priority: str) -> None:
    if not NOTIFY_URL:
        return
    try:
        response = requests.post(
            NOTIFY_URL,
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            timeout=REQUEST_TIMEOUT_SECS,
        )
        response.raise_for_status()
    except Exception as e:
        log_error("ntfy", e)


def notify(title: str, body: str, priority: str = "default") -> None:
    """Best-effort push notification. Never raises."""
    icon = ICON_HIGH_PRIORITY if priority == "high" else ICON_DEFAULT
    titled = f"{icon} {title}"
    _send_telegram(titled, body)
    _send_ntfy(titled, body, priority)