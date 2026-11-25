import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stratiq.data.synthetic import make_ohlcv
from stratiq.engine import BacktestConfig
from stratiq.lab import (
    heatmap_figure,
    neighbor_share,
    sensitivity_summary,
    sensitivity_table,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ema_cross.json"
BASE = json.loads(FIXTURE.read_text(encoding="utf-8"))
del BASE["session"]
CFG = BacktestConfig(initial_cash=100_000.0, spread=0.2, slippage=0.1)


def make_spec(p: dict) -> dict:
    d = copy.deepcopy(BASE)
    d["indicators"][0]["params"]["length"] = p["fast"]
    d["indicators"][1]["params"]["length"] = p["slow"]
    return d


def grid_table(min_trades: int = 1) -> pd.DataFrame:
    df = make_ohlcv(1500, seed=7, vol=0.004)
    return sensitivity_table(
        df, make_spec, ("fast", [10, 20]), ("slow", [40, 60]), CFG, min_trades=min_trades
    )


def test_table_shape_labels_and_determinism():
    t = grid_table()
    assert t.shape == (2, 2)
    assert t.index.name == "slow" and t.columns.name == "fast"
    assert list(t.index) == [40, 60] and list(t.columns) == [10, 20]
    pd.testing.assert_frame_equal(t, grid_table())


def test_cells_with_too_few_trades_are_nan():
    assert grid_table(min_trades=10_000).isna().all().all()


def test_neighbor_share_hand_values():
    t = pd.DataFrame([[1.0, -1.0], [1.0, 1.0]])
    out = neighbor_share(t).to_numpy()
    np.testing.assert_allclose(out, [[2 / 3, 1.0], [2 / 3, 2 / 3]])


def test_summary_hand_values_and_nan_cells_are_ignored():
    t = pd.DataFrame([[1.0, -1.0], [1.0, 1.0]])
    s = sensitivity_summary(t)
    assert s["cells"] == 4
    assert s["profitable_share"] == 0.75
    assert math.isclose(s["neighbor_share_of_profitable"], 2 / 3)
    with_nan = pd.DataFrame([[1.0, np.nan], [np.nan, -1.0]])
    assert sensitivity_summary(with_nan)["cells"] == 2


def test_summary_of_an_all_nan_table_is_nan():
    s = sensitivity_summary(pd.DataFrame([[np.nan, np.nan]]))
    assert s["cells"] == 0 and math.isnan(s["profitable_share"])


def test_heatmap_figure_has_one_heatmap_trace():
    fig = heatmap_figure(grid_table(), "test")
    assert len(fig.data) == 1 and fig.data[0].type == "heatmap"
