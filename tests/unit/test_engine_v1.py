import numpy as np
import pandas as pd
import pytest

from stratiq.data.synthetic import make_ohlcv
from stratiq.engine import BacktestConfig, run_backtest


def _five_bars() -> pd.DataFrame:
    opens = np.array([100.0, 102.0, 105.0, 104.0, 106.0])
    closes = np.array([101.0, 104.0, 103.0, 105.0, 105.0])
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 1.0,
            "low": np.minimum(opens, closes) - 1.0,
            "close": closes,
            "volume": 100.0,
        },
        index=idx,
    )


def test_hand_computed_five_bar_example():
    df = _five_bars()
    sig = pd.Series([1, 1, 0, -1, 0], index=df.index)
    res = run_backtest(df, sig, BacktestConfig(initial_cash=1000.0, units=1.0))

    assert len(res.trades) == 2
    long_t, short_t = res.trades

    assert (long_t.side, long_t.entry_price, long_t.exit_price) == (1, 102.0, 104.0)
    assert long_t.entry_time == df.index[1]
    assert long_t.exit_time == df.index[3]
    assert long_t.pnl == 2.0

    assert (short_t.side, short_t.entry_price, short_t.exit_price) == (-1, 106.0, 105.0)
    assert short_t.entry_time == df.index[4]
    assert short_t.exit_time == df.index[4]
    assert short_t.pnl == 1.0

    assert res.equity.tolist() == [1000.0, 1002.0, 1001.0, 1002.0, 1003.0]


def test_flip_closes_and_reopens_at_same_open():
    df = _five_bars()
    sig = pd.Series([1, -1, -1, -1, 0], index=df.index)
    res = run_backtest(df, sig, BacktestConfig(initial_cash=1000.0))
    long_t, short_t = res.trades
    assert (long_t.entry_price, long_t.exit_price, long_t.pnl) == (102.0, 105.0, 3.0)
    assert (short_t.side, short_t.entry_price) == (-1, 105.0)
    assert short_t.entry_time == long_t.exit_time == df.index[2]


def test_units_scale_pnl():
    df = _five_bars()
    sig = pd.Series([1, 1, 0, 0, 0], index=df.index)
    res = run_backtest(df, sig, BacktestConfig(initial_cash=1000.0, units=10.0))
    assert res.trades[0].pnl == 20.0
    assert res.equity.iloc[-1] == 1020.0


def test_no_signal_means_no_trades_and_flat_equity():
    df = _five_bars()
    res = run_backtest(df, pd.Series(0, index=df.index), BacktestConfig(initial_cash=500.0))
    assert res.trades == []
    assert (res.equity == 500.0).all()


def test_signal_on_last_bar_is_ignored():
    df = _five_bars()
    sig = pd.Series([0, 0, 0, 0, 1], index=df.index)
    assert run_backtest(df, sig).trades == []


def test_open_position_closed_at_last_close():
    df = _five_bars()
    sig = pd.Series([1, 1, 1, 1, 1], index=df.index)
    res = run_backtest(df, sig, BacktestConfig(initial_cash=1000.0))
    assert len(res.trades) == 1
    assert res.trades[0].exit_price == 105.0
    assert res.trades[0].exit_time == df.index[-1]


def test_invalid_signal_values_rejected():
    df = _five_bars()
    with pytest.raises(ValueError):
        run_backtest(df, pd.Series([2, 0, 0, 0, 0], index=df.index))


def test_mismatched_index_rejected():
    df = _five_bars()
    with pytest.raises(ValueError):
        run_backtest(df, pd.Series([0, 0, 0], index=df.index[:3]))


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        BacktestConfig(units=0)
    with pytest.raises(ValueError):
        BacktestConfig(initial_cash=-1)


def test_final_equity_equals_cash_plus_trade_pnl_on_random_data():
    df = make_ohlcv(500, seed=11)
    rng = np.random.default_rng(3)
    sig = pd.Series(rng.integers(-1, 2, len(df)), index=df.index)
    cfg = BacktestConfig(initial_cash=10_000.0, units=2.0)
    res = run_backtest(df, sig, cfg)
    assert res.equity.iloc[-1] == pytest.approx(cfg.initial_cash + sum(t.pnl for t in res.trades))
    assert all(t.exit_time >= t.entry_time for t in res.trades)
