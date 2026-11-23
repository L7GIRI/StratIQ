import numpy as np
import pandas as pd
import pytest

from stratiq.data.synthetic import make_ohlcv
from stratiq.indicators import atr, ema, rsi, sma, true_range, vwap


def test_sma_hand_values():
    out = sma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert out.isna().tolist() == [True, True, False, False, False]
    assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_ema_hand_values():
    # alpha = 0.5: 1, 1.5, 2.25, 3.125, 4.0625
    out = ema(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert out.iloc[:2].isna().all()
    np.testing.assert_allclose(out.iloc[2:].to_numpy(), [2.25, 3.125, 4.0625])


def test_rsi_hand_values():
    # gains 1,0,1 and losses 0,1,0 with alpha 0.5 -> RSI 50 then 75
    out = rsi(pd.Series([1.0, 2.0, 1.0, 2.0]), 2)
    assert out.iloc[:2].isna().all()
    np.testing.assert_allclose(out.iloc[2:].to_numpy(), [50.0, 75.0])


def test_rsi_extremes():
    up = rsi(pd.Series(np.arange(1.0, 40.0)), 14).dropna()
    down = rsi(pd.Series(np.arange(40.0, 1.0, -1.0)), 14).dropna()
    assert (up == 100.0).all()
    assert (down == 0.0).all()


def test_rsi_bounded_on_random_data():
    out = rsi(make_ohlcv(500, seed=4)["close"], 14).dropna()
    assert ((out >= 0) & (out <= 100)).all()


def test_true_range_uses_gap():
    high = pd.Series([10.0, 15.0])
    low = pd.Series([8.0, 14.0])
    close = pd.Series([9.0, 14.5])
    assert true_range(high, low, close).tolist() == [2.0, 6.0]


def test_atr_hand_values():
    high = pd.Series([10.0, 11.0, 12.0])
    low = pd.Series([8.0, 9.0, 9.0])
    close = pd.Series([9.0, 10.0, 11.0])
    # TR = 2, 2, 3; alpha 0.5 -> 2, 2, 2.5
    out = atr(high, low, close, 2)
    assert np.isnan(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1:].to_numpy(), [2.0, 2.5])


def test_vwap_hand_values_and_daily_reset():
    idx = pd.DatetimeIndex(["2024-01-01 22:00", "2024-01-01 23:00", "2024-01-02 00:00"], tz="UTC")
    df = pd.DataFrame(
        {
            "open": [10.0, 20.0, 30.0],
            "high": [10.0, 20.0, 30.0],
            "low": [10.0, 20.0, 30.0],
            "close": [10.0, 20.0, 30.0],
            "volume": [1.0, 3.0, 5.0],
        },
        index=idx,
    )
    out = vwap(df, tz="UTC", reset_hour=0)
    np.testing.assert_allclose(out.to_numpy(), [10.0, 17.5, 30.0])


def test_vwap_nan_when_no_volume():
    df = make_ohlcv(5, seed=1)
    df["volume"] = 0.0
    assert vwap(df).isna().all()


def test_vwap_requires_tz():
    df = make_ohlcv(5, seed=1)
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError):
        vwap(df)


def test_bad_length_raises():
    s = pd.Series([1.0, 2.0, 3.0])
    for fn in (sma, ema, rsi):
        with pytest.raises(ValueError):
            fn(s, 0)


def test_indicators_use_no_future_data():
    """Values on the first 150 bars must not change when later bars are removed."""
    df = make_ohlcv(400, seed=7)
    head = df.iloc[:150]
    pairs = [
        (sma(df["close"], 20), sma(head["close"], 20)),
        (ema(df["close"], 20), ema(head["close"], 20)),
        (rsi(df["close"], 14), rsi(head["close"], 14)),
        (
            atr(df["high"], df["low"], df["close"], 14),
            atr(head["high"], head["low"], head["close"], 14),
        ),
        (vwap(df), vwap(head)),
    ]
    for full, truncated in pairs:
        pd.testing.assert_series_equal(full.iloc[:150], truncated, check_freq=False)
