import numpy as np
import pandas as pd

from stratiq.data.synthetic import make_ohlcv
from stratiq.lab import expanding_percentile, regimes, trend_regime, vol_regime


def test_expanding_percentile_hand_values():
    out = expanding_percentile(pd.Series([3.0, 1.0, 2.0]), min_periods=1)
    np.testing.assert_allclose(out.to_numpy(), [1.0, 0.5, 2 / 3])


def test_expanding_percentile_respects_min_periods_and_nan():
    out = expanding_percentile(pd.Series([1.0, np.nan, 2.0, 3.0]), min_periods=3)
    assert out.iloc[:3].isna().all()
    assert out.iloc[3] == 1.0


def test_labels_are_from_the_expected_sets():
    df = make_ohlcv(1500, seed=3, vol=0.004)
    r = regimes(df)
    assert set(r["vol"]) <= {"UNKNOWN", "LOW", "MID", "HIGH"}
    assert set(r["trend"]) <= {"UNKNOWN", "UP", "DOWN", "RANGE"}
    assert (r["vol"].iloc[:50] == "UNKNOWN").all()
    assert {"LOW", "MID", "HIGH"} <= set(r["vol"])


def test_regimes_are_causal_under_truncation():
    df = make_ohlcv(1200, seed=9, vol=0.004)
    full = regimes(df)
    head = regimes(df.iloc[:700])
    pd.testing.assert_frame_equal(full.iloc[:700], head, check_freq=False)


def test_trend_regime_detects_a_steady_rise():
    df = make_ohlcv(400, seed=1, vol=0.0005)
    df = df * 1.0
    ramp = pd.Series(np.linspace(2000.0, 3000.0, len(df)), index=df.index)
    df["close"], df["open"] = ramp, ramp
    df["high"], df["low"] = ramp + 1.0, ramp - 1.0
    assert (trend_regime(df).iloc[-50:] == "UP").all()


def test_vol_regime_flags_a_volatility_jump_as_high():
    df = make_ohlcv(600, seed=2, vol=0.0005)
    df.iloc[400:, df.columns.get_loc("high")] += 20.0
    df.iloc[400:, df.columns.get_loc("low")] -= 20.0
    assert (vol_regime(df).iloc[450:] == "HIGH").all()
