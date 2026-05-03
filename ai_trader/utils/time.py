"""Time utilities for market-hour enforcement."""

from __future__ import annotations

from datetime import datetime, time, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    """Current time in IST."""
    return datetime.now(IST)


def is_market_open(
    current_time: datetime | None = None,
    market_open: time = time(9, 15),
    market_close: time = time(15, 30),
) -> bool:
    """Check if the Indian stock market is currently open.

    Excludes weekends. Does not account for holidays (extend with calendar).
    """
    now = current_time or now_ist()
    if now.weekday() >= 5:
        return False
    current = now.time()
    return market_open <= current <= market_close
