from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average, alpha = 2/(length+1), seeded with the first value."""
    if length < 1:
        raise ValueError("length must be >= 1")
    return series.ewm(span=length, adjust=False, min_periods=length).mean()
