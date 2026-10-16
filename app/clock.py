"""One place for "what time is it, and which day is that?".

Everything is STORED in UTC (naive datetimes, as before) and everything a person SEES or types is in
their timezone (TIMEZONE in .env, default Asia/Kolkata). "Today", "yesterday", day/week/month buckets,
the hour chart and budget months all use the local day, so ₹1,500 spent at 01:30 on 1 November in India
counts in November, not in October (it is 20:00 on 31 October in UTC).

    now_local() / today_local()   the current local time / date
    to_local(utc_naive)           a stored time -> local time
    to_utc(local_naive)           a local time -> what gets stored / compared in SQL
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, tzinfo
from functools import lru_cache
from zoneinfo import ZoneInfo

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _zone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except Exception:  # unknown name, or no timezone database on this system (install `tzdata`)
        logger.warning("Unknown TIMEZONE %r: using UTC. Use a name like Asia/Kolkata.", name)
        return UTC


def tz() -> tzinfo:
    return _zone((settings.timezone or "UTC").strip() or "UTC")


def now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def now_local() -> datetime:
    return datetime.now(tz()).replace(tzinfo=None)


def today_local() -> date:
    return now_local().date()


def to_local(utc_naive: datetime) -> datetime:
    return utc_naive.replace(tzinfo=UTC).astimezone(tz()).replace(tzinfo=None)


def to_utc(local_naive: datetime) -> datetime:
    return local_naive.replace(tzinfo=tz()).astimezone(UTC).replace(tzinfo=None)
