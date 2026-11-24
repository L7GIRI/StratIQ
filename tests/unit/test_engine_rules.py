import numpy as np
import pandas as pd
import pytest

from stratiq.data.sessions import session_end_flags
from stratiq.data.synthetic import make_ohlcv
from stratiq.engine import BacktestConfig, run_backtest


def bars(opens, highs, lows, closes) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 100.0},
        index=idx,
    )


def sigs(df, values) -> pd.Series:
    return pd.Series(values, index=df.index)


CFG = BacktestConfig(initial_cash=1000.0)


def test_stop_hits_on_the_entry_bar():
    df = bars([100, 100, 101, 102], [101, 102, 103, 104], [99, 98, 100, 101], [100, 101, 102, 103])
    res = run_backtest(df, sigs(df, [1, 1, 1, 1]), CFG, stop_distance=2.0)
    assert len(res.trades) == 1  # no re-entry while the signal keeps holding
    t = res.trades[0]
    assert (t.entry_price, t.exit_price, t.exit_reason, t.pnl) == (100.0, 98.0, "stop", -2.0)
    assert t.exit_time == df.index[1]
    assert res.equity.iloc[-1] == 998.0


def test_gap_through_stop_fills_at_the_open():
    df = bars([100, 100, 95], [101, 103, 96], [99, 100, 94], [100, 102, 95])
    res = run_backtest(df, sigs(df, [1, 1, 1]), CFG, stop_distance=2.0)
    t = res.trades[0]
    assert (t.exit_price, t.exit_reason, t.pnl) == (95.0, "stop", -5.0)


def test_take_profit_at_r_multiple():
    cfg = BacktestConfig(initial_cash=1000.0, target_r=2.0)  # stop 98, target 104
    df = bars([100, 100], [101, 105], [99, 99], [100, 104])
    res = run_backtest(df, sigs(df, [1, 1]), cfg, stop_distance=2.0)
    t = res.trades[0]
    assert (t.exit_price, t.exit_reason, t.pnl) == (104.0, "target", 4.0)
    assert res.equity.tolist() == [1000.0, 1004.0]


def test_stop_wins_when_both_levels_are_hit_in_one_bar():
    cfg = BacktestConfig(initial_cash=1000.0, target_r=2.0)
    df = bars([100, 100], [101, 105], [99, 97], [100, 100])
    res = run_backtest(df, sigs(df, [1, 1]), cfg, stop_distance=2.0)
    t = res.trades[0]
    assert (t.exit_price, t.exit_reason, t.pnl) == (98.0, "stop", -2.0)


def test_short_stop():
    df = bars([100, 100], [101, 103], [99, 99], [100, 101])
    res = run_backtest(df, sigs(df, [-1, -1]), CFG, stop_distance=2.0)
    t = res.trades[0]
    assert (t.side, t.exit_price, t.exit_reason, t.pnl) == (-1, 102.0, "stop", -2.0)


def test_reentry_allowed_after_signal_changes():
    df = bars([100] * 5, [101] * 5, [99.5, 98, 99.5, 99.5, 99.5], [100] * 5)
    res = run_backtest(df, sigs(df, [1, 1, 0, 1, 1]), CFG, stop_distance=2.0)
    assert len(res.trades) == 2
    assert res.trades[0].exit_reason == "stop"
    assert res.trades[1].entry_time == df.index[4]


def test_costs_hand_computed():
    # fills cost slippage 0.5 + half spread 0.5 = 1.0: buy 101, sell 109
    df = bars([100, 100, 110], [100, 100, 110], [100, 100, 110], [100, 100, 110])
    cfg = BacktestConfig(
        initial_cash=1000.0, units=2.0, commission_per_unit=0.1, spread=1.0, slippage=0.5
    )
    res = run_backtest(df, sigs(df, [1, 0, 0]), cfg)
    t = res.trades[0]
    assert (t.entry_price, t.exit_price) == (101.0, 109.0)
    assert t.commission == pytest.approx(0.4)
    assert t.pnl == pytest.approx(15.6)
    assert res.equity.iloc[-1] == pytest.approx(1015.6)


def test_costs_never_improve_results():
    df = make_ohlcv(500, seed=21)
    rng = np.random.default_rng(5)
    s = sigs(df, rng.integers(-1, 2, len(df)))
    free = run_backtest(df, s, BacktestConfig(initial_cash=10_000.0))
    paid = run_backtest(
        df, s, BacktestConfig(initial_cash=10_000.0, commission_per_unit=0.5, spread=0.3)
    )
    assert paid.equity.iloc[-1] <= free.equity.iloc[-1]


def test_session_end_closes_position_at_close():
    px = [100, 101, 102, 103, 104]
    df = bars(px, px, px, px)
    flags = pd.Series([False, False, True, False, False], index=df.index)
    res = run_backtest(df, sigs(df, [1] * 5), CFG, session_end=flags)
    assert [t.exit_reason for t in res.trades] == ["session_end", "end"]
    assert [t.pnl for t in res.trades] == [1.0, 1.0]


def test_target_without_stop_is_rejected():
    df = bars([100, 100], [101, 101], [99, 99], [100, 100])
    with pytest.raises(ValueError):
        run_backtest(df, sigs(df, [1, 1]), BacktestConfig(target_r=2.0))


def test_nan_stop_distance_at_entry_is_rejected():
    df = bars([100, 100], [101, 101], [99, 99], [100, 100])
    with pytest.raises(ValueError):
        run_backtest(df, sigs(df, [1, 1]), CFG, stop_distance=pd.Series([np.nan, 1.0], df.index))


def test_session_end_flags_mark_last_in_session_bar():
    idx = pd.date_range("2024-01-15", periods=24, freq="1h", tz="UTC")
    flags = session_end_flags(idx, "LONDON")  # winter: 08:00-17:00 UTC
    assert flags[flags].index.tolist() == [pd.Timestamp("2024-01-15 16:00", tz="UTC")]
