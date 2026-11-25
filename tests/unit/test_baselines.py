import json
from pathlib import Path

import numpy as np
import pytest

from stratiq.analytics import compute_metrics
from stratiq.data.synthetic import make_ohlcv
from stratiq.dsl import approve, compile_spec, run_compiled, validate_spec
from stratiq.engine import BacktestConfig

STRATEGY_DIR = Path(__file__).parent.parent.parent / "strategies"
FILES = sorted(STRATEGY_DIR.glob("*.json"))
COSTS = BacktestConfig(spread=0.3, slippage=0.2, commission_per_unit=0.02)


def frame(timeframe: str):
    if timeframe == "1h":
        return make_ohlcv(2500, seed=5, vol=0.004)
    return make_ohlcv(1500, seed=5, freq="1D", vol=0.01)


def test_there_are_four_baselines():
    assert len(FILES) == 4


@pytest.mark.parametrize("path", FILES, ids=[f.stem for f in FILES])
def test_baseline_validates_compiles_and_runs(path):
    spec = validate_spec(json.loads(path.read_text(encoding="utf-8")))
    assert spec.approved_at is None  # files on disk are never pre-approved
    df = frame(spec.timeframe)
    res = run_compiled(compile_spec(approve(spec), df, COSTS), df)
    assert np.isfinite(res.equity.to_numpy()).all()
    assert compute_metrics(res)["trades"] == len(res.trades)


def test_baselines_trade_on_synthetic_data():
    total = 0
    for path in FILES:
        spec = validate_spec(json.loads(path.read_text(encoding="utf-8")))
        df = frame(spec.timeframe)
        total += len(run_compiled(compile_spec(approve(spec), df, COSTS), df).trades)
    assert total > 0
