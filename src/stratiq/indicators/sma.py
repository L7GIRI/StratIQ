from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average. NaN until `length` observations exist."""
    if length < 1:
        raise ValueError("length must be >= 1")
    return series.rolling(length, min_periods=length).mean()
