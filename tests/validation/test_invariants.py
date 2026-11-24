"""Accounting invariants and hypothesis property tests for the engine."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stratiq.data.synthetic import make_ohlcv
from stratiq.engine import BacktestConfig, run_backtest

REASONS = {"signal", "stop", "target", "session_end", "end"}


def scenario(seed: int, n: int = 200):
    df = make_ohlcv(n, seed=seed, vol=0.003)
    rng = np.random.default_rng(seed)
    blocks = rng.integers(-1, 2, n // 10)
    sig = pd.Series(np.repeat(blocks, 10), index=df.index)  # signals persist for 10 bars
    ends = pd.Series(rng.random(n) < 0.05, index=df.index)
    return df, sig, ends


def check_invariants(res, cfg: BacktestConfig) -> None:
    assert len(res.equity) > 0
    assert np.isfinite(res.equity.to_numpy()).all()
    trades = res.trades
    for t in trades:
        assert t.exit_time >= t.entry_time
        assert t.commission >= 0
        assert t.exit_reason in REASONS
        assert t.side in (-1, 1)
        if cfg.target_r is None:
            assert t.exit_reason != "target"
    for a, b in zip(trades, trades[1:], strict=False):
        assert b.entry_time >= a.exit_time  # one position at a time
    expected = cfg.initial_cash + sum(t.pnl for t in trades)
    assert res.equity.iloc[-1] == pytest.approx(expected, abs=1e-6)


PARAMS = dict(
    seed=st.integers(0, 5000),
    stop=st.floats(0.5, 15.0),
    target_r=st.one_of(st.none(), st.floats(0.5, 4.0)),
    comm=st.floats(0.0, 1.0),
    spread=st.floats(0.0, 1.0),
    slip=st.floats(0.0, 1.0),
)


@settings(max_examples=60, deadline=None)
@given(**PARAMS)
def test_invariants_hold_on_random_scenarios(seed, stop, target_r, comm, spread, slip):
    df, sig, ends = scenario(seed)
    cfg = BacktestConfig(
        initial_cash=10_000.0,
        units=2.0,
        commission_per_unit=comm,
        spread=spread,
        slippage=slip,
        target_r=target_r,
    )
    res = run_backtest(df, sig, cfg, stop_distance=stop, session_end=ends)
    check_invariants(res, cfg)


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 5000), stop=st.floats(0.5, 15.0), target_r=st.floats(0.5, 4.0))
def test_stops_never_profit_and_targets_never_lose_without_costs(seed, stop, target_r):
    df, sig, ends = scenario(seed)
    cfg = BacktestConfig(initial_cash=10_000.0, target_r=target_r)
    res = run_backtest(df, sig, cfg, stop_distance=stop, session_end=ends)
    for t in res.trades:
        move = t.side * (t.exit_price - t.entry_price)
        if t.exit_reason == "stop":
            assert move <= 1e-9
        if t.exit_reason == "target":
            assert move >= -1e-9


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 5000),
    stop=st.floats(0.5, 15.0),
    comm=st.floats(0.0, 1.0),
    spread=st.floats(0.0, 1.0),
    slip=st.floats(0.0, 1.0),
)
def test_costs_never_improve_results(seed, stop, comm, spread, slip):
    df, sig, ends = scenario(seed)
    free = run_backtest(
        df,
        sig,
        BacktestConfig(initial_cash=10_000.0, target_r=2.0),
        stop_distance=stop,
        session_end=ends,
    )
    paid = run_backtest(
        df,
        sig,
        BacktestConfig(
            initial_cash=10_000.0,
            target_r=2.0,
            commission_per_unit=comm,
            spread=spread,
            slippage=slip,
        ),
        stop_distance=stop,
        session_end=ends,
    )
    # costs change prices paid, not when trades trigger, so the trade list is the same
    assert [t.entry_time for t in free.trades] == [t.entry_time for t in paid.trades]
    assert paid.equity.iloc[-1] <= free.equity.iloc[-1] + 1e-6


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 5000), stop=st.floats(0.5, 15.0))
def test_same_inputs_give_identical_outputs(seed, stop):
    df, sig, ends = scenario(seed)
    cfg = BacktestConfig(initial_cash=10_000.0, target_r=2.0, spread=0.3)
    a = run_backtest(df, sig, cfg, stop_distance=stop, session_end=ends)
    b = run_backtest(df, sig, cfg, stop_distance=stop, session_end=ends)
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert a.trades == b.trades


def test_flat_signal_never_trades():
    df, _, _ = scenario(1)
    res = run_backtest(df, pd.Series(0, index=df.index), BacktestConfig(initial_cash=500.0))
    assert res.trades == []
    assert (res.equity == 500.0).all()


def test_constant_long_is_one_trade_from_second_open_to_last_close():
    df, _, _ = scenario(2)
    res = run_backtest(df, pd.Series(1, index=df.index), BacktestConfig(initial_cash=10_000.0))
    assert len(res.trades) == 1
    assert res.trades[0].pnl == pytest.approx(df["close"].iloc[-1] - df["open"].iloc[1])
