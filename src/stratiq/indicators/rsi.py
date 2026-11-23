from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder RSI in [0, 100]."""
    if length < 1:
        raise ValueError("length must be >= 1")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)
