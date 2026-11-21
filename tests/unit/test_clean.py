import numpy as np
import pandas as pd
import pytest

from stratiq.data.clean import clean
from stratiq.data.synthetic import make_ohlcv


def test_clean_data_is_untouched():
    out, rep = clean(make_ohlcv(200, seed=1))
    assert len(out) == 200
    assert rep.duplicates_removed == 0
    assert rep.nan_removed == 0
    assert rep.invalid_ohlc_removed == 0
    assert rep.gap_count == 0


def test_duplicate_timestamps_removed():
    df = make_ohlcv(50, seed=1)
    df = pd.concat([df, df.iloc[[10]]])
    out, rep = clean(df)
    assert rep.duplicates_removed == 1
    assert len(out) == 50
    assert out.index.is_unique


def test_high_below_low_removed():
    df = make_ohlcv(50, seed=1)
    df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["low"] - 1.0
    out, rep = clean(df)
    assert rep.invalid_ohlc_removed == 1
    assert len(out) == 49


def test_high_below_close_removed():
    df = make_ohlcv(50, seed=1)
    df.iloc[7, df.columns.get_loc("high")] = df.iloc[7]["close"] - 0.5
    _, rep = clean(df)
    assert rep.invalid_ohlc_removed == 1


def test_nan_price_removed():
    df = make_ohlcv(50, seed=1)
    df.iloc[3, df.columns.get_loc("close")] = np.nan
    out, rep = clean(df)
    assert rep.nan_removed == 1
    assert len(out) == 49


def test_non_positive_price_removed():
    df = make_ohlcv(50, seed=1)
    df.iloc[4, df.columns.get_loc("open")] = -1.0
    _, rep = clean(df)
    assert rep.invalid_ohlc_removed == 1


def test_unsorted_input_is_sorted():
    out, _ = clean(make_ohlcv(50, seed=1).iloc[::-1])
    assert out.index.is_monotonic_increasing


def test_naive_index_becomes_utc():
    df = make_ohlcv(20, seed=1)
    df.index = df.index.tz_localize(None)
    out, _ = clean(df)
    assert str(out.index.tz) == "UTC"


def test_daily_new_york_midnight_keeps_calendar_date():
    df = make_ohlcv(2, seed=1, freq="1D")
    df.index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"]).tz_localize("America/New_York")
    out, _ = clean(df, daily=True)
    assert out.index[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert out.index[1] == pd.Timestamp("2024-01-03", tz="UTC")


def test_gap_detected():
    df = make_ohlcv(300, seed=1)
    df = df.drop(df.index[50:200])  # 151-hour hole, longer than 3 days
    _, rep = clean(df)
    assert rep.gap_count == 1


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        clean(make_ohlcv(10, seed=1).drop(columns=["volume"]))


def test_column_names_are_case_insensitive():
    df = make_ohlcv(20, seed=1)
    df.columns = [c.capitalize() for c in df.columns]
    out, _ = clean(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
