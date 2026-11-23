"""Seeded synthetic OHLCV generator so tests never depend on downloads."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(
    n: int,
    seed: int = 0,
    start: str = "2024-01-01",
    freq: str = "1h",
    start_price: float = 2000.0,
    vol: float = 0.001,
) -> pd.DataFrame:
    """Random-walk OHLCV with a tz-aware UTC index. Always valid OHLC."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, vol, n)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick_up = np.abs(rng.normal(0.0, vol / 2, n)) * close
    wick_dn = np.abs(rng.normal(0.0, vol / 2, n)) * close
    high = np.maximum(open_, close) + wick_up
    low = np.minimum(open_, close) - wick_dn
    volume = rng.integers(100, 1000, n).astype(float)
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
