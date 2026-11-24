import math

import numpy as np
import pandas as pd
import pytest

from stratiq.analytics import compute_metrics, drawdown_series
from stratiq.engine import BacktestConfig, BacktestResult, Trade


def make_trade(i: int, pnl: float) -> Trade:
    ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i)
    return Trade(ts, ts, 1, 1.0, 100.0, 100.0 + pnl)


def make_result(equity, pnls) -> BacktestResult:
    idx = pd.date_range("2024-01-01", periods=len(equity), freq="1D", tz="UTC")
    trades = [make_trade(i, p) for i, p in enumerate(pnls)]
    cfg = BacktestConfig(initial_cash=float(equity[0]))
    return BacktestResult(pd.Series(equity, index=idx, dtype=float), trades, cfg)


def test_drawdown_series_hand_values():
    dd = drawdown_series(pd.Series([100.0, 110.0, 99.0, 120.0]))
    np.testing.assert_allclose(dd.to_numpy(), [0.0, 0.0, -0.1, 0.0])


def test_hand_computed_trade_metrics():
    res = make_result([100, 110, 99, 120], [10, -11, 21])
    m = compute_metrics(res)
    assert m["trades"] == 3
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["gross_profit"] == 31.0
    assert m["gross_loss"] == 11.0
    assert m["profit_factor"] == pytest.approx(31 / 11)
    assert m["expectancy"] == pytest.approx(20 / 3)
    assert m["avg_win"] == 15.5
    assert m["avg_loss"] == -11.0
    assert m["net_profit"] == 20.0
    assert m["total_return"] == pytest.approx(0.2)
    assert m["max_drawdown"] == pytest.approx(0.1)
    assert m["max_consecutive_wins"] == 1
    assert m["max_consecutive_losses"] == 1


def test_sharpe_matches_hand_calculation():
    eq = [100.0, 110.0, 99.0, 120.0]
    res = make_result(eq, [])
    r = np.array([0.1, -0.1, 120 / 99 - 1])
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert compute_metrics(res)["sharpe"] == pytest.approx(expected)


def test_sortino_matches_hand_calculation():
    eq = [100.0, 110.0, 99.0, 120.0]
    r = np.array([0.1, -0.1, 120 / 99 - 1])
    downside = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    expected = r.mean() / downside * math.sqrt(252)
    assert compute_metrics(make_result(eq, []))["sortino"] == pytest.approx(expected)


def test_no_trades_does_not_crash():
    m = compute_metrics(make_result([100.0, 100.0, 100.0], []))
    assert m["trades"] == 0
    assert math.isnan(m["win_rate"])
    assert math.isnan(m["profit_factor"])
    assert math.isnan(m["expectancy"])
    assert m["total_return"] == 0.0
    assert m["max_drawdown"] == 0.0
    assert m["exposure"] == 0.0
    assert math.isnan(m["sharpe"])


def test_all_wins_has_infinite_profit_factor():
    m = compute_metrics(make_result([100, 105, 112], [5, 7]))
    assert m["profit_factor"] == math.inf
    assert m["win_rate"] == 1.0
    assert m["max_consecutive_wins"] == 2
    assert math.isnan(m["avg_loss"])


def test_all_losses():
    m = compute_metrics(make_result([100, 95, 90], [-5, -5]))
    assert m["profit_factor"] == 0.0
    assert m["win_rate"] == 0.0
    assert m["max_consecutive_losses"] == 2
    assert math.isnan(m["avg_win"])


def test_streaks():
    m = compute_metrics(make_result([100.0] * 8, [1, 1, 1, -1, -1, 1, -1]))
    assert m["max_consecutive_wins"] == 3
    assert m["max_consecutive_losses"] == 2


def test_exposure_counts_bars_in_market():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    trade = Trade(idx[2], idx[5], 1, 1.0, 100.0, 101.0)
    res = BacktestResult(pd.Series(100.0, index=idx), [trade], BacktestConfig(initial_cash=100.0))
    assert compute_metrics(res)["exposure"] == pytest.approx(4 / 10)


def test_calmar_and_annual_return_use_calendar_years():
    idx = pd.DatetimeIndex(["2024-01-01", "2025-01-01"], tz="UTC")
    eq = pd.Series([100.0, 121.0], index=idx)
    res = BacktestResult(eq, [], BacktestConfig(initial_cash=100.0))
    m = compute_metrics(res)
    assert m["annual_return"] == pytest.approx(0.21, abs=2e-3)  # 366 days in 2024
    assert math.isnan(m["calmar"])  # no drawdown


def test_hourly_equity_is_resampled_to_daily_for_sharpe():
    idx = pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC")
    eq = pd.Series(np.linspace(100.0, 130.0, 72), index=idx)
    res = BacktestResult(eq, [], BacktestConfig(initial_cash=100.0))
    daily_last = eq.groupby(eq.index.normalize()).last()
    r = daily_last.pct_change().dropna()
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert compute_metrics(res)["sharpe"] == pytest.approx(expected)
