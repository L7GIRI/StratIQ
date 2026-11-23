from __future__ import annotations

import pandas as pd


def vwap(
    df: pd.DataFrame,
    tz: str = "America/New_York",
    reset_hour: int = 18,
) -> pd.Series:
    """Cumulative VWAP of typical price, reset at `reset_hour` local time each day.

    The trading-day key uses wall-clock time in `tz`, so daylight saving is handled.
    NaN while cumulative volume is zero.
    """
    if df.index.tz is None:
        raise ValueError("index must be timezone-aware")
    local = df.index.tz_convert(tz).tz_localize(None)
    day = pd.Series((local - pd.Timedelta(hours=reset_hour)).normalize(), index=df.index)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum()
    return (pv / vol.where(vol > 0)).rename("vwap")
