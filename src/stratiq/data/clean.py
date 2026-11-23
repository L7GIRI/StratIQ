"""Cleaning and validation of raw OHLCV data, with a quality report."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
PRICES = ["open", "high", "low", "close"]


@dataclass
class QualityReport:
    rows_in: int
    rows_out: int
    duplicates_removed: int
    nan_removed: int
    invalid_ohlc_removed: int
    zero_volume_pct: float
    gap_count: int
    largest_gap: str
    first: str
    last: str

    def to_dict(self) -> dict:
        return asdict(self)


def _to_utc(index: pd.Index, daily: bool) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if daily:
        # Daily bars: keep the calendar date, do not shift it by a timezone conversion.
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return idx.normalize().tz_localize("UTC")
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def invalid_ohlc_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that violate OHLC logic or have non-positive prices/negative volume."""
    op, hi, lo, cl = df["open"], df["high"], df["low"], df["close"]
    bad = (df[PRICES] <= 0).any(axis=1)
    bad |= hi < np.maximum(op, cl)
    bad |= lo > np.minimum(op, cl)
    bad |= hi < lo
    bad |= df["volume"] < 0
    return bad


def clean(
    df: pd.DataFrame,
    *,
    daily: bool = False,
    max_gap: pd.Timedelta | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    rows_in = len(df)
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    missing = [c for c in OHLCV if c not in out.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = out[OHLCV].astype(float)

    out.index = _to_utc(out.index, daily)
    out.index.name = "timestamp"
    out = out.sort_index()

    dup = out.index.duplicated(keep="last")
    duplicates = int(dup.sum())
    out = out[~dup]

    nan_mask = out[PRICES].isna().any(axis=1)
    nan_removed = int(nan_mask.sum())
    out = out[~nan_mask]
    out["volume"] = out["volume"].fillna(0.0)

    bad = invalid_ohlc_mask(out)
    invalid = int(bad.sum())
    out = out[~bad]

    if max_gap is None:
        max_gap = pd.Timedelta("5D") if daily else pd.Timedelta("3D")
    deltas = out.index.to_series().diff().dropna()
    gap_count = int((deltas > max_gap).sum())
    largest = str(deltas.max()) if len(deltas) else "-"

    report = QualityReport(
        rows_in=rows_in,
        rows_out=len(out),
        duplicates_removed=duplicates,
        nan_removed=nan_removed,
        invalid_ohlc_removed=invalid,
        zero_volume_pct=round(float((out["volume"] == 0).mean() * 100), 2) if len(out) else 0.0,
        gap_count=gap_count,
        largest_gap=largest,
        first=str(out.index.min()) if len(out) else "-",
        last=str(out.index.max()) if len(out) else "-",
    )
    return out, report
