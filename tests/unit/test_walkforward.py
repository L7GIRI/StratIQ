import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stratiq.data.synthetic import make_ohlcv
from stratiq.engine import BacktestConfig
from stratiq.lab import walk_forward

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ema_cross.json"
BASE = json.loads(FIXTURE.read_text(encoding="utf-8"))
del BASE["session"]

GRID = {"fast": [10, 20], "slow": [40, 60]}
CFG = BacktestConfig(initial_cash=100_000.0, spread=0.2, slippage=0.1)


def make_spec(p: dict) -> dict:
    d = copy.deepcopy(BASE)
    d["indicators"][0]["params"]["length"] = p["fast"]
    d["indicators"][1]["params"]["length"] = p["slow"]
    return d


def run(df, train=500, test=250):
    return walk_forward(df, make_spec, GRID, train, test, CFG, min_trades=1)


def test_windows_tile_the_data_without_overlap():
    df = make_ohlcv(1500, seed=7, vol=0.004)
    wf = run(df)
    assert len(wf.windows) == 4
    starts = list(wf.windows["test_start"])
    assert starts == [df.index[i] for i in (500, 750, 1000, 1250)]
    assert len(wf.stitched_equity) == 4 * 250
    assert wf.stitched_equity.index.is_monotonic_increasing
    assert wf.stitched_equity.index.is_unique


def test_stitched_equity_starts_at_initial_cash():
    wf = run(make_ohlcv(1500, seed=7, vol=0.004))
    assert wf.stitched_equity.iloc[0] == pytest.approx(CFG.initial_cash)
    assert np.isfinite(wf.stitched_equity.to_numpy()).all()


def test_no_test_window_data_reaches_the_optimiser():
    df = make_ohlcv(1500, seed=7, vol=0.004)
    base = run(df)
    last_test_start = 1250
    altered = df.copy()
    cols = ["open", "high", "low", "close"]
    altered.iloc[last_test_start:, [altered.columns.get_loc(c) for c in cols]] *= 1.7
    other = run(altered)
    # every choice made before seeing the last test window must be unchanged
    assert list(base.windows["best_params"]) == list(other.windows["best_params"])
    pd.testing.assert_series_equal(base.windows["is_score"], other.windows["is_score"])
    pd.testing.assert_frame_equal(
        base.windows.iloc[:3].drop(columns=["best_params"]),
        other.windows.iloc[:3].drop(columns=["best_params"]),
    )


def test_runs_are_deterministic():
    df = make_ohlcv(1500, seed=7, vol=0.004)
    a, b = run(df), run(df)
    pd.testing.assert_frame_equal(a.windows, b.windows)
    pd.testing.assert_series_equal(a.stitched_equity, b.stitched_equity)


def test_degradation_is_a_finite_number():
    wf = run(make_ohlcv(1500, seed=7, vol=0.004))
    assert np.isfinite(wf.degradation)


def test_not_enough_data_is_rejected():
    with pytest.raises(ValueError):
        run(make_ohlcv(600, seed=1), train=500, test=250)
