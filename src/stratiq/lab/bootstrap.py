"""Bootstrap confidence intervals, so per-regime results are not over-read."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stratiq.engine.models import BacktestResult


def bootstrap_ci(
    values,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    stat=np.mean,
) -> tuple[float, float, float]:
    """Return (estimate, lower, upper) with a percentile bootstrap interval."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    stats = stat(x[idx], axis=1)
    return (
        float(stat(x)),
        float(np.quantile(stats, alpha / 2)),
        float(np.quantile(stats, 1 - alpha / 2)),
    )


def per_regime_performance(
    result: BacktestResult,
    regime_frame: pd.DataFrame,
    column: str = "regime",
    min_trades: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Trade P&L grouped by the regime of each trade's SIGNAL bar (the bar before the fill).

    Groups with fewer than `min_trades` trades are flagged unreliable and get no interval.
    """
    idx = regime_frame.index
    labels = []
    for t in result.trades:
        pos = idx.get_loc(t.entry_time)
        labels.append(regime_frame[column].iloc[pos - 1] if pos > 0 else "UNKNOWN")

    pnl_by_label: dict[str, list[float]] = {}
    for label, t in zip(labels, result.trades, strict=True):
        pnl_by_label.setdefault(label, []).append(t.pnl)

    rows = []
    for label in sorted(pnl_by_label):
        pnls = np.array(pnl_by_label[label])
        reliable = len(pnls) >= min_trades
        est, lo, hi = (
            bootstrap_ci(pnls, seed=seed) if reliable else (pnls.mean(), math.nan, math.nan)
        )
        rows.append(
            {
                column: label,
                "trades": len(pnls),
                "win_rate": float((pnls > 0).mean()),
                "expectancy": float(est),
                "ci_low": lo,
                "ci_high": hi,
                "reliable": reliable,
            }
        )
    cols = [column, "trades", "win_rate", "expectancy", "ci_low", "ci_high", "reliable"]
    return pd.DataFrame(rows, columns=cols)
