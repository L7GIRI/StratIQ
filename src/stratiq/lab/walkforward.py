"""Walk-forward validation: optimise on train bars only, evaluate on the next unseen bars."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import pandas as pd

from stratiq.analytics import compute_metrics
from stratiq.dsl import SpecError, approve, compile_spec, run_compiled, validate_spec
from stratiq.engine import BacktestConfig, BacktestResult


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame
    stitched_equity: pd.Series

    @property
    def degradation(self) -> float:
        """Mean in-sample score minus mean out-of-sample score (positive = overfit)."""
        w = self.windows
        return float(w["is_score"].mean() - w["oos_score"].mean())


def _score(metrics: dict, objective: str, min_trades: int) -> float:
    value = metrics.get(objective, math.nan)
    if metrics["trades"] < min_trades or not isinstance(value, float) or not math.isfinite(value):
        return -math.inf
    return value


def _run(make_spec, params, data, base, zero_before: int = 0):
    try:
        spec = approve(validate_spec(make_spec(params)))
    except SpecError:
        return None
    compiled = compile_spec(spec, data, base)
    if zero_before > 0:
        signals = compiled.signals.copy()
        signals.iloc[:zero_before] = 0
        compiled = replace(compiled, signals=signals)
    return run_compiled(compiled, data)


def walk_forward(
    df: pd.DataFrame,
    make_spec: Callable[[dict], dict],
    grid: dict[str, list],
    train_bars: int,
    test_bars: int,
    base_config: BacktestConfig | None = None,
    objective: str = "total_return",
    min_trades: int = 5,
) -> WalkForwardResult:
    base = base_config or BacktestConfig()
    names = list(grid)
    combos = [dict(zip(names, v, strict=True)) for v in itertools.product(*grid.values())]
    n = len(df)
    if train_bars < 1 or test_bars < 1 or n < train_bars + test_bars:
        raise ValueError("not enough data for one train+test window")

    rows = []
    stitched: list[pd.Series] = []
    cash = base.initial_cash
    for start in range(0, n - train_bars - test_bars + 1, test_bars):
        train_end = start + train_bars
        test_end = train_end + test_bars
        train = df.iloc[start:train_end]  # the ONLY data the optimiser sees

        best_params, best_score = None, -math.inf
        for params in combos:
            res = _run(make_spec, params, train, base)
            if res is None:
                continue
            score = _score(compute_metrics(res), objective, min_trades)
            if score > best_score:  # strict: ties keep the first combination
                best_params, best_score = params, score
        if best_params is None:
            rows.append(_empty_row(df, start, train_end, test_end))
            continue

        full = _run(make_spec, best_params, df.iloc[start:test_end], base, zero_before=train_bars)
        eq = full.equity.iloc[train_bars:]
        oos = BacktestResult(eq, full.trades, full.config)
        m = compute_metrics(oos)
        stitched.append(eq / base.initial_cash * cash)
        cash = float(stitched[-1].iloc[-1])
        rows.append(
            {
                "train_start": df.index[start],
                "test_start": df.index[train_end],
                "test_end": df.index[test_end - 1],
                "best_params": best_params,
                "is_score": best_score,
                "oos_score": _score(m, objective, 0),
                "oos_trades": m["trades"],
                "oos_return": m["total_return"],
            }
        )
    equity = pd.concat(stitched) if stitched else pd.Series(dtype=float)
    return WalkForwardResult(pd.DataFrame(rows), equity.rename("stitched_equity"))


def _empty_row(df, start, train_end, test_end) -> dict:
    return {
        "train_start": df.index[start],
        "test_start": df.index[train_end],
        "test_end": df.index[test_end - 1],
        "best_params": None,
        "is_score": math.nan,
        "oos_score": math.nan,
        "oos_trades": 0,
        "oos_return": 0.0,
    }
