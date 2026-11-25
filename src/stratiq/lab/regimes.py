"""Causal, rule-based regimes. The label at bar t uses data up to the close of bar t only."""

from __future__ import annotations

from bisect import bisect_right, insort

import numpy as np
import pandas as pd

from stratiq.indicators import atr, ema


def expanding_percentile(series: pd.Series, min_periods: int = 100) -> pd.Series:
    """Percentile rank of each value among all values up to and including it."""
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    seen: list[float] = []
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        insort(seen, v)
        if len(seen) >= min_periods:
            out[i] = bisect_right(seen, v) / len(seen)
    return pd.Series(out, index=series.index, name="percentile")


def vol_regime(df: pd.DataFrame, atr_length: int = 14, min_periods: int = 100) -> pd.Series:
    a = atr(df["high"], df["low"], df["close"], atr_length)
    pct = expanding_percentile(a, min_periods)
    labels = np.where(
        pct.isna(),
        "UNKNOWN",
        np.where(pct <= 1 / 3, "LOW", np.where(pct >= 2 / 3, "HIGH", "MID")),
    )
    return pd.Series(labels, index=df.index, name="vol")


def trend_regime(
    df: pd.DataFrame,
    ema_length: int = 50,
    lookback: int = 10,
    atr_length: int = 14,
    threshold: float = 1.0,
) -> pd.Series:
    """EMA change over `lookback` bars, measured in ATRs."""
    e = ema(df["close"], ema_length)
    a = atr(df["high"], df["low"], df["close"], atr_length)
    slope = (e - e.shift(lookback)) / a
    labels = np.where(
        slope.isna(),
        "UNKNOWN",
        np.where(slope > threshold, "UP", np.where(slope < -threshold, "DOWN", "RANGE")),
    )
    return pd.Series(labels, index=df.index, name="trend")


def regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"vol": vol_regime(df), "trend": trend_regime(df)})
    out["regime"] = out["vol"] + "|" + out["trend"]
    return out
