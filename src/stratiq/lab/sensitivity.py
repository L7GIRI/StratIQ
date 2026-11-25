"""Two-parameter sensitivity grid, heatmap and a neighbour-robustness measure."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from stratiq.analytics import compute_metrics
from stratiq.engine import BacktestConfig
from stratiq.lab.walkforward import _run


def sensitivity_table(
    df: pd.DataFrame,
    make_spec: Callable[[dict], dict],
    x: tuple[str, list],
    y: tuple[str, list],
    base_config: BacktestConfig | None = None,
    metric: str = "total_return",
    min_trades: int = 5,
) -> pd.DataFrame:
    """Rows are y values, columns are x values. Cells with too few trades are NaN."""
    base = base_config or BacktestConfig()
    (x_name, x_vals), (y_name, y_vals) = x, y
    data = []
    for yv in y_vals:
        row = []
        for xv in x_vals:
            res = _run(make_spec, {x_name: xv, y_name: yv}, df, base)
            if res is None:
                row.append(math.nan)
                continue
            m = compute_metrics(res)
            value = m.get(metric, math.nan)
            ok = m["trades"] >= min_trades and isinstance(value, float) and math.isfinite(value)
            row.append(value if ok else math.nan)
        data.append(row)
    return pd.DataFrame(
        data, index=pd.Index(y_vals, name=y_name), columns=pd.Index(x_vals, name=x_name)
    )


def neighbor_share(table: pd.DataFrame) -> pd.DataFrame:
    """For each cell, the share of its adjacent cells (up to 8, NaN ignored) that are > 0."""
    vals = table.to_numpy(dtype=float)
    rows, cols = vals.shape
    out = np.full(vals.shape, np.nan)
    for i in range(rows):
        for j in range(cols):
            neighbours = [
                vals[a, b]
                for a in range(max(0, i - 1), min(rows, i + 2))
                for b in range(max(0, j - 1), min(cols, j + 2))
                if (a, b) != (i, j) and not np.isnan(vals[a, b])
            ]
            if neighbours:
                out[i, j] = float(np.mean(np.array(neighbours) > 0))
    return pd.DataFrame(out, index=table.index, columns=table.columns)


def sensitivity_summary(table: pd.DataFrame) -> dict:
    vals = table.to_numpy(dtype=float)
    valid = ~np.isnan(vals)
    profitable = valid & (vals > 0)
    shares = neighbor_share(table).to_numpy(dtype=float)[profitable]
    shares = shares[~np.isnan(shares)]
    return {
        "cells": int(valid.sum()),
        "profitable_share": float(profitable.sum() / valid.sum()) if valid.any() else math.nan,
        "neighbor_share_of_profitable": float(shares.mean()) if len(shares) else math.nan,
    }


def heatmap_figure(table: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=table.to_numpy(dtype=float),
            x=[str(c) for c in table.columns],
            y=[str(i) for i in table.index],
            colorscale="RdYlGn",
            zmid=0,
            colorbar={"title": "metric"},
        )
    )
    fig.update_layout(
        title=title, xaxis_title=str(table.columns.name), yaxis_title=str(table.index.name)
    )
    return fig
