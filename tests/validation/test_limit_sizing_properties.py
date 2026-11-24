import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stratiq.data.synthetic import make_ohlcv
from stratiq.engine import BacktestConfig, run_backtest


def scenario(seed: int, n: int = 200):
    df = make_ohlcv(n, seed=seed, vol=0.003)
    rng = np.random.default_rng(seed)
    sig = pd.Series(np.repeat(rng.integers(-1, 2, n // 10), 10), index=df.index)
    ends = pd.Series(rng.random(n) < 0.05, index=df.index)
    return df, sig, ends


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 5000),
    stop=st.floats(0.5, 15.0),
    offset=st.floats(0.0, 3.0),
    risk=st.floats(0.001, 0.05),
    target_r=st.one_of(st.none(), st.floats(0.5, 4.0)),
)
def test_limit_fills_never_worse_than_limit_and_sizing_respects_leverage(
    seed, stop, offset, risk, target_r
):
    df, sig, ends = scenario(seed)
    cfg = BacktestConfig(
        initial_cash=10_000.0,
        entry_mode="limit",
        limit_offset=offset,
        risk_pct=risk,
        target_r=target_r,
        commission_per_unit=0.1,
        spread=0.2,
        slippage=0.1,
    )
    res = run_backtest(df, sig, cfg, stop_distance=stop, session_end=ends)
    peak = max(float(res.equity.max()), cfg.initial_cash)
    for t in res.trades:
        assert t.units > 0
        ref_close = float(df["close"].iloc[df.index.get_loc(t.entry_time) - 1])
        if t.side == 1:
            assert t.entry_price <= ref_close - offset + 1e-9
        else:
            assert t.entry_price >= ref_close + offset - 1e-9
        assert t.units * t.entry_price <= cfg.max_leverage * peak * (1 + 1e-9) + 1e-6
    expected = cfg.initial_cash + sum(t.pnl for t in res.trades)
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-9, abs=1e-6)
    for a, b in zip(res.trades, res.trades[1:], strict=False):
        assert b.entry_time >= a.exit_time


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 5000), stop=st.floats(0.5, 15.0), offset=st.floats(0.0, 3.0))
def test_limit_runs_are_deterministic(seed, stop, offset):
    df, sig, ends = scenario(seed)
    cfg = BacktestConfig(initial_cash=10_000.0, entry_mode="limit", limit_offset=offset)
    a = run_backtest(df, sig, cfg, stop_distance=stop, session_end=ends)
    b = run_backtest(df, sig, cfg, stop_distance=stop, session_end=ends)
    assert a.trades == b.trades
    pd.testing.assert_series_equal(a.equity, b.equity)
