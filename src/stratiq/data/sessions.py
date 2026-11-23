"""Trading sessions defined in local time, so daylight saving is handled by zoneinfo."""

from __future__ import annotations

from datetime import time

import pandas as pd

SESSIONS: dict[str, tuple[str, time, time]] = {
    "ASIA": ("Asia/Tokyo", time(9, 0), time(18, 0)),
    "LONDON": ("Europe/London", time(8, 0), time(17, 0)),
    "NEW_YORK": ("America/New_York", time(8, 0), time(17, 0)),
}


def in_session(index: pd.DatetimeIndex, name: str) -> pd.Series:
    """True for bars whose START time is in [open, close) of the session, in local time."""
    if name not in SESSIONS:
        raise ValueError(f"unknown session {name!r}; choose from {list(SESSIONS)}")
    if index.tz is None:
        raise ValueError("index must be timezone-aware")
    tz, start, end = SESSIONS[name]
    local = index.tz_convert(tz)
    minute = local.hour * 60 + local.minute
    lo = start.hour * 60 + start.minute
    hi = end.hour * 60 + end.minute
    return pd.Series((minute >= lo) & (minute < hi), index=index, name=name)


def label_sessions(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One boolean column per session. Sessions may overlap."""
    return pd.concat([in_session(index, n) for n in SESSIONS], axis=1)
