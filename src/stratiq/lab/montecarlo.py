"""Monte Carlo on closed trades: shuffle the order or resample with replacement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonteCarloResult:
    method: str
    initial_cash: float
    final_equity: np.ndarray
    max_drawdown: np.ndarray

    def summary(self) -> dict:
        def q(a: np.ndarray, p: float) -> float:
            return float(np.quantile(a, p))

        return {
            "method": self.method,
            "sims": len(self.final_equity),
            "final_p5": q(self.final_equity, 0.05),
            "final_p50": q(self.final_equity, 0.50),
            "final_p95": q(self.final_equity, 0.95),
            "max_dd_p50": q(self.max_drawdown, 0.50),
            "max_dd_p95": q(self.max_drawdown, 0.95),
            "prob_loss": float((self.final_equity < self.initial_cash).mean()),
        }


def monte_carlo(
    pnls,
    initial_cash: float = 100_000.0,
    n_sims: int = 2000,
    method: str = "shuffle",
    seed: int = 0,
) -> MonteCarloResult:
    x = np.asarray(pnls, dtype=float)
    if len(x) == 0:
        raise ValueError("no trades to simulate")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be > 0")
    if method not in ("shuffle", "resample"):
        raise ValueError("method must be 'shuffle' or 'resample'")

    rng = np.random.default_rng(seed)
    if method == "shuffle":
        paths = np.array([rng.permutation(x) for _ in range(n_sims)])
    else:
        paths = x[rng.integers(0, len(x), size=(n_sims, len(x)))]

    equity = initial_cash + np.cumsum(paths, axis=1)
    equity = np.concatenate([np.full((n_sims, 1), initial_cash), equity], axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    max_dd = ((peak - equity) / peak).max(axis=1)
    return MonteCarloResult(method, float(initial_cash), equity[:, -1], max_dd)
