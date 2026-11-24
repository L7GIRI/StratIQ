"""Performance metrics from an equity curve and a trade list.

Conventions:
- Sharpe/Sortino use daily-resampled equity returns (last equity of each UTC day),
  annualised with 252 days, risk-free rate 0.
- Drawdown is peak-to-trough as a fraction of the running peak.
- Profit factor is inf with no losing trades and NaN with no trades.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stratiq.engine.models import BacktestResult, Trade

TRADING_DAYS = 252


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fraction below the running peak (0 at highs, negative below)."""
    peak = equity.cummax()
    return equity / peak - 1.0


def daily_returns(equity: pd.Series) -> pd.Series:
    daily = equity.groupby(equity.index.normalize()).last()
    return daily.pct_change().dropna()


def _streaks(pnls: list[float]) -> tuple[int, int]:
    best_win = best_loss = cur_win = cur_loss = 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
        elif p < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = cur_loss = 0
        best_win = max(best_win, cur_win)
        best_loss = max(best_loss, cur_loss)
    return best_win, best_loss


def _exposure(equity: pd.Series, trades: list[Trade]) -> float:
    if len(equity) == 0 or not trades:
        return 0.0
    idx = equity.index
    in_market = np.zeros(len(idx), dtype=bool)
    for t in trades:
        lo = idx.searchsorted(t.entry_time, side="left")
        hi = idx.searchsorted(t.exit_time, side="right")
        in_market[lo:hi] = True
    return float(in_market.mean())


def compute_metrics(result: BacktestResult) -> dict:
    eq = result.equity
    trades = result.trades
    start = float(result.config.initial_cash)
    end = float(eq.iloc[-1])
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))
    n = len(pnls)

    if n == 0:
        profit_factor = math.nan
    elif gross_loss == 0:
        profit_factor = math.inf if gross_profit > 0 else math.nan
    else:
        profit_factor = gross_profit / gross_loss

    dd = drawdown_series(eq)
    max_dd = float(-dd.min()) if len(dd) else 0.0
    avg_dd = float(-dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    rets = daily_returns(eq)
    sharpe = sortino = math.nan
    if len(rets) >= 2:
        std = float(rets.std(ddof=1))
        if std > 0:
            sharpe = float(rets.mean() / std * math.sqrt(TRADING_DAYS))
        downside = float(np.sqrt(np.mean(np.minimum(rets.to_numpy(), 0.0) ** 2)))
        if downside > 0:
            sortino = float(rets.mean() / downside * math.sqrt(TRADING_DAYS))

    total_return = end / start - 1.0
    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400) if len(eq) > 1 else 0
    annual_return = (end / start) ** (1.0 / years) - 1.0 if years > 0 and end > 0 else math.nan
    calmar = annual_return / max_dd if max_dd > 0 and not math.isnan(annual_return) else math.nan

    win_streak, loss_streak = _streaks(pnls)
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "net_profit": end - start,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "trades": n,
        "win_rate": len(wins) / n if n else math.nan,
        "profit_factor": profit_factor,
        "expectancy": float(np.mean(pnls)) if n else math.nan,
        "avg_win": float(np.mean(wins)) if wins else math.nan,
        "avg_loss": float(np.mean(losses)) if losses else math.nan,
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "exposure": _exposure(eq, trades),
        "max_consecutive_wins": win_streak,
        "max_consecutive_losses": loss_streak,
    }
