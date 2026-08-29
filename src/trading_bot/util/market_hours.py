"""US regular-session bounds, as ET "HH:MM" strings.

Deliberately stdlib-free so the cheapest possible caller can use it.
cli/smc_cycle.py's fast time gate runs before every heavy import
specifically to exit in well under a second outside market hours, and it
needs these two values; importing smc_live for them would pull in pandas
and defeat the gate's whole purpose.

These are facts about the exchange, not strategy configuration. What a bot
does INSIDE the session -- when it will open a position, when it stops,
when it flattens -- belongs in smc_rules.json's time_filter, which is
bounded by these rather than duplicating them.
"""

from __future__ import annotations

MARKET_OPEN_ET = "09:30"
MARKET_CLOSE_ET = "16:00"
