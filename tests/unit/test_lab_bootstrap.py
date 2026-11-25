import math

import numpy as np
import pandas as pd

from stratiq.engine import BacktestConfig, BacktestResult, Trade
from stratiq.lab import bootstrap_ci, per_regime_performance


def test_constant_values_have_a_degenerate_interval():
    est, lo, hi = bootstrap_ci([2.0] * 20)
    assert est == lo == hi == 2.0


def test_same_seed_same_interval():
    x = np.random.default_rng(0).normal(0, 1, 50)
    assert bootstrap_ci(x, seed=5) == bootstrap_ci(x, seed=5)


def test_interval_brackets_the_estimate_and_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = bootstrap_ci(rng.normal(0, 1, 20))
    big = bootstrap_ci(rng.normal(0, 1, 2000))
    assert small[1] <= small[0] <= small[2]
    assert (big[2] - big[1]) < (small[2] - small[1])


def test_empty_input_is_nan():
    assert all(math.isnan(v) for v in bootstrap_ci([]))


def test_trades_take_the_regime_of_their_signal_bar():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    frame = pd.DataFrame({"regime": ["X"] * 3 + ["Y"] * 7}, index=idx)
    trades = [
        Trade(idx[3], idx[5], 1, 1.0, 100.0, 102.0),  # signal bar idx[2] is X
        Trade(idx[4], idx[6], 1, 1.0, 100.0, 99.0),  # signal bar idx[3] is Y
        Trade(idx[7], idx[8], 1, 1.0, 100.0, 101.0),  # signal bar idx[6] is Y
    ]
    res = BacktestResult(pd.Series(0.0, index=idx), trades, BacktestConfig())
    out = per_regime_performance(res, frame, min_trades=2).set_index("regime")
    assert out.loc["X", "trades"] == 1
    assert out.loc["Y", "trades"] == 2
    assert out.loc["Y", "expectancy"] == 0.0
    assert out.loc["Y", "win_rate"] == 0.5


def test_small_groups_are_flagged_unreliable_and_have_no_interval():
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    frame = pd.DataFrame({"regime": ["A"] * 6}, index=idx)
    trades = [Trade(idx[2], idx[3], 1, 1.0, 100.0, 101.0)]
    res = BacktestResult(pd.Series(0.0, index=idx), trades, BacktestConfig())
    row = per_regime_performance(res, frame, min_trades=30).iloc[0]
    assert not row["reliable"]
    assert math.isnan(row["ci_low"]) and math.isnan(row["ci_high"])
