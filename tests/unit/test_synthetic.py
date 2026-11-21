import pandas as pd

from stratiq.data.clean import invalid_ohlc_mask
from stratiq.data.synthetic import make_ohlcv


def test_length_and_utc_index():
    df = make_ohlcv(100, seed=1)
    assert len(df) == 100
    assert str(df.index.tz) == "UTC"


def test_same_seed_is_identical():
    pd.testing.assert_frame_equal(make_ohlcv(100, seed=5), make_ohlcv(100, seed=5))


def test_different_seed_differs():
    assert not make_ohlcv(100, seed=1)["close"].equals(make_ohlcv(100, seed=2)["close"])


def test_ohlc_is_always_valid():
    assert not invalid_ohlc_mask(make_ohlcv(2000, seed=3)).any()
