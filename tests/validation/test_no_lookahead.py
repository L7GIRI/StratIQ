"""Truncation test: removing future bars must not change anything that already happened."""

import numpy as np
import pandas as pd
import pytest

from stratiq.data.sessions import session_end_flags
from stratiq.data.synthetic import make_ohlcv
from stratiq.engine import BacktestConfig, run_backtest
from stratiq.indicators import atr, ema


def ema_cross_signals(df: pd.DataFrame) -> pd.Series:
    fast = ema(df["close"], 20)
    slow = ema(df["close"], 50)
    return pd.Series(np.where(fast > slow, 1, np.where(fast < slow, -1, 0)), index=df.index)


CFG = BacktestConfig(
    initial_cash=100_000.0,
    units=1.0,
    commission_per_unit=0.1,
    spread=0.2,
    slippage=0.1,
    target_r=2.0,
)


def _run(df: pd.DataFrame, flags: pd.Series):
    stop = 1.5 * atr(df["high"], df["low"], df["close"], 14)
    return run_backtest(df, ema_cross_signals(df), CFG, stop_distance=stop, session_end=flags)


@pytest.mark.parametrize("k", [80, 150, 250, 399])
def test_engine_results_identical_when_future_is_removed(k):
    df = make_ohlcv(400, seed=7, vol=0.004)
    # Session flags come from the clock, so compute them once on the full index and slice.
    flags = session_end_flags(df.index, "LONDON")

    full = _run(df, flags)
    head = _run(df.iloc[:k], flags.iloc[:k])

    assert len(full.trades) > 0  # the test is meaningless without trades

    # Equity before the last truncated bar must match (the last bar force-closes in `head`).
    pd.testing.assert_series_equal(
        full.equity.iloc[: k - 1], head.equity.iloc[: k - 1], check_freq=False
    )

    # Every trade that closed on its own must be identical in both runs.
    closed_head = [t for t in head.trades if t.exit_reason != "end"]
    closed_full = [t for t in full.trades if t.exit_time <= df.index[k - 1]]
    assert closed_head == closed_full


def _prefix_equal(make_signals, df: pd.DataFrame, k: int) -> bool:
    full = make_signals(df)
    head = make_signals(df.iloc[:k])
    try:
        pd.testing.assert_series_equal(full.iloc[:k], head, check_freq=False)
    except AssertionError:
        return False
    return True


def test_causal_signals_pass_the_truncation_check():
    df = make_ohlcv(400, seed=7)
    assert _prefix_equal(ema_cross_signals, df, 150)


def test_cheating_signals_are_caught_by_the_truncation_check():
    """Negative control: a strategy that peeks at the next close must FAIL the check."""

    def cheating(d: pd.DataFrame) -> pd.Series:
        peek = d["close"].shift(-1) - d["close"]
        return pd.Series(np.sign(peek).fillna(0).astype(int), index=d.index)

    df = make_ohlcv(400, seed=7)
    assert not _prefix_equal(cheating, df, 150)
