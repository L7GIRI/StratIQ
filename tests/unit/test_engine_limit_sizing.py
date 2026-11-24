import numpy as np
import pandas as pd
import pytest

from stratiq.engine import BacktestConfig, run_backtest


def bars(opens, highs, lows, closes) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 100.0},
        index=idx,
    )


def sigs(df, values) -> pd.Series:
    return pd.Series(values, index=df.index)


LIMIT = BacktestConfig(initial_cash=1000.0, entry_mode="limit", limit_offset=1.0)


def long_limit_bars() -> pd.DataFrame:
    return bars(
        [100, 100, 101, 102],
        [101, 101, 102, 103],
        [99.5, 98.5, 100.5, 101.5],
        [100, 100.5, 101.5, 102.5],
    )


def test_long_limit_fills_at_limit_mid_bar():
    df = long_limit_bars()  # limit = 100 - 1 = 99; bar 1 low 98.5 reaches it
    res = run_backtest(df, sigs(df, [1, 1, 0, 0]), LIMIT)
    t = res.trades[0]
    assert (t.entry_price, t.exit_price, t.exit_reason) == (99.0, 102.0, "signal")
    assert t.entry_time == df.index[1]
    assert t.pnl == 3.0
    assert res.equity.iloc[-1] == 1003.0


def test_limit_never_reached_means_no_trade():
    df = bars([100] * 4, [101] * 4, [99.5] * 4, [100] * 4)
    res = run_backtest(df, sigs(df, [1, 1, 1, 1]), LIMIT)
    assert res.trades == []
    assert (res.equity == 1000.0).all()


def test_gap_through_limit_fills_at_the_open():
    df = bars([100, 97, 98], [101, 98, 99], [99, 96, 97], [100, 97.5, 98])
    res = run_backtest(df, sigs(df, [1, 0, 0]), LIMIT)
    t = res.trades[0]
    assert (t.entry_price, t.exit_price, t.pnl) == (97.0, 98.0, 1.0)


def test_short_limit_fills_at_limit():
    df = bars([100, 100, 99], [101, 101.5, 100], [99, 99.5, 98], [100, 100.5, 99.5])
    res = run_backtest(df, sigs(df, [-1, 0, 0]), LIMIT)  # limit = 100 + 1 = 101
    t = res.trades[0]
    assert (t.side, t.entry_price, t.exit_price, t.pnl) == (-1, 101.0, 99.0, 2.0)


def test_limit_fills_pay_no_slippage_but_exits_do():
    cfg = BacktestConfig(
        initial_cash=1000.0,
        entry_mode="limit",
        limit_offset=1.0,
        slippage=1.0,
        spread=1.0,
    )
    df = long_limit_bars()
    t = run_backtest(df, sigs(df, [1, 0, 0, 0]), cfg).trades[0]
    assert t.entry_price == 99.0  # no cost on the resting order
    assert t.exit_price == 99.5  # open 101 minus slippage 1.0 and half spread 0.5


def test_risk_sizing_units_and_stop_loss_is_one_percent():
    df = bars([100] * 3, [101, 101, 103], [99, 97, 99], [100, 100, 100])
    cfg = BacktestConfig(initial_cash=10_000.0, risk_pct=0.01)
    res = run_backtest(df, sigs(df, [1, 1, 1]), cfg, stop_distance=2.0)
    t = res.trades[0]
    assert t.units == pytest.approx(50.0)  # 100 risked / 2 stop distance
    assert (t.exit_reason, t.exit_price) == ("stop", 98.0)
    assert t.pnl == pytest.approx(-100.0)  # exactly 1% of equity
    assert res.equity.iloc[-1] == pytest.approx(9900.0)


def test_risk_sizing_winning_trade():
    df = bars([100] * 3, [101, 101, 103], [99, 99, 99], [100, 100, 102])
    cfg = BacktestConfig(initial_cash=10_000.0, risk_pct=0.01)
    res = run_backtest(df, sigs(df, [1, 1, 1]), cfg, stop_distance=2.0)
    assert res.trades[0].units == pytest.approx(50.0)
    assert res.trades[0].pnl == pytest.approx(100.0)  # closed at the last close, 102


def test_leverage_cap_limits_units():
    df = bars([100] * 3, [101] * 3, [99.5] * 3, [100] * 3)
    cfg = BacktestConfig(initial_cash=10_000.0, risk_pct=0.5, max_leverage=5.0)
    res = run_backtest(df, sigs(df, [1, 1, 1]), cfg, stop_distance=1.0)
    assert res.trades[0].units == pytest.approx(500.0)  # 10_000 * 5 / 100, not 5000


def test_risk_pct_requires_stop_distance():
    df = bars([100] * 3, [101] * 3, [99] * 3, [100] * 3)
    with pytest.raises(ValueError):
        run_backtest(df, sigs(df, [1, 1, 1]), BacktestConfig(risk_pct=0.01))


def test_bad_new_config_values_rejected():
    for kwargs in (
        {"limit_offset": -1.0},
        {"risk_pct": 0.0},
        {"risk_pct": 1.5},
        {"max_leverage": 0.0},
        {"entry_mode": "stop"},
    ):
        with pytest.raises(ValueError):
            BacktestConfig(**kwargs)


def test_config_is_stored_with_the_result_and_serialisable():
    df = long_limit_bars()
    res = run_backtest(df, sigs(df, [1, 0, 0, 0]), LIMIT)
    d = res.config.to_dict()
    assert d["entry_mode"] == "limit"
    assert d["limit_offset"] == 1.0
    assert np.isfinite(d["initial_cash"])
