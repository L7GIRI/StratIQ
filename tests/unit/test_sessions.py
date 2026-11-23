import pandas as pd
import pytest

from stratiq.data.sessions import in_session, label_sessions


def _check(name, stamp, expected):
    idx = pd.DatetimeIndex([stamp], tz="UTC")
    assert bool(in_session(idx, name).iloc[0]) is expected, (name, stamp)


def test_london_winter_open_is_08_utc():
    _check("LONDON", "2024-01-15 08:00", True)
    _check("LONDON", "2024-01-15 07:00", False)


def test_london_summer_open_is_07_utc():
    _check("LONDON", "2024-07-15 07:00", True)
    _check("LONDON", "2024-07-15 06:00", False)


def test_london_close_is_exclusive():
    _check("LONDON", "2024-07-15 15:00", True)
    _check("LONDON", "2024-07-15 16:00", False)  # 17:00 BST


def test_new_york_across_us_dst_change():
    # US clocks moved forward on 2024-03-10
    _check("NEW_YORK", "2024-03-08 13:00", True)
    _check("NEW_YORK", "2024-03-08 12:00", False)
    _check("NEW_YORK", "2024-03-11 12:00", True)
    _check("NEW_YORK", "2024-03-11 11:00", False)


def test_asia_has_no_dst_shift():
    _check("ASIA", "2024-01-15 00:00", True)
    _check("ASIA", "2024-07-15 00:00", True)
    _check("ASIA", "2024-07-15 09:00", False)


def test_naive_index_raises():
    with pytest.raises(ValueError):
        in_session(pd.DatetimeIndex(["2024-01-15 08:00"]), "LONDON")


def test_unknown_session_raises():
    with pytest.raises(ValueError):
        in_session(pd.DatetimeIndex(["2024-01-15"], tz="UTC"), "MARS")


def test_label_sessions_columns():
    idx = pd.date_range("2024-01-15", periods=24, freq="1h", tz="UTC")
    out = label_sessions(idx)
    assert list(out.columns) == ["ASIA", "LONDON", "NEW_YORK"]
    assert len(out) == 24
