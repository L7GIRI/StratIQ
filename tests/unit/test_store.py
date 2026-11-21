import pandas as pd
import pytest

from stratiq.data.store import data_hash, load_dataset, save_dataset
from stratiq.data.synthetic import make_ohlcv


def test_roundtrip_preserves_data_and_hash(tmp_path):
    df = make_ohlcv(100, seed=1)
    info = save_dataset(df, "t", tmp_path, source="synthetic")
    loaded, meta = load_dataset("t", tmp_path)
    assert meta["sha256"] == info["sha256"] == data_hash(df)
    assert meta["source"] == "synthetic"
    assert meta["rows"] == 100
    pd.testing.assert_frame_equal(loaded, df, check_freq=False, check_index_type=False)


def test_hash_changes_when_data_changes():
    a = make_ohlcv(50, seed=1)
    b = a.copy()
    b.iloc[0, 0] += 0.01
    assert data_hash(a) != data_hash(b)


def test_tampered_file_is_detected(tmp_path):
    save_dataset(make_ohlcv(50, seed=1), "t", tmp_path)
    make_ohlcv(50, seed=2).to_parquet(tmp_path / "t.parquet")  # overwrite behind its back
    with pytest.raises(ValueError):
        load_dataset("t", tmp_path)
