"""Monte Carlo and parameter sensitivity for the EMA-cross baseline on saved gold data."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from stratiq.data.store import load_dataset
from stratiq.dsl import approve, compile_spec, run_compiled, validate_spec
from stratiq.engine import BacktestConfig
from stratiq.lab import (
    heatmap_figure,
    monte_carlo,
    sensitivity_summary,
    sensitivity_table,
)

BASE = json.loads(Path("strategies/ema_cross_1h.json").read_text(encoding="utf-8"))
CFG = BacktestConfig(spread=0.30, slippage=0.20, commission_per_unit=0.02)
df, _ = load_dataset("gc_1h", "data/parquet")
pd.set_option("display.width", 200)

spec = approve(validate_spec(BASE))
res = run_compiled(compile_spec(spec, df, CFG), df)
pnls = [t.pnl for t in res.trades]
print(f"trades: {len(pnls)}")
if pnls:
    for method in ("shuffle", "resample"):
        print(monte_carlo(pnls, CFG.initial_cash, method=method).summary())


def make_spec(p: dict) -> dict:
    d = copy.deepcopy(BASE)
    d["indicators"][0]["params"]["length"] = p["fast"]
    d["indicators"][1]["params"]["length"] = p["slow"]
    return d


table = sensitivity_table(
    df,
    make_spec,
    ("fast", [10, 15, 20, 25, 30]),
    ("slow", [40, 50, 60, 80, 100]),
    CFG,
)
print(table.to_string(float_format=lambda v: f"{v:.4f}"))
print(sensitivity_summary(table))
Path("results").mkdir(exist_ok=True)
heatmap_figure(table, "EMA cross: total return by (fast, slow)").write_html(
    "results/sensitivity_ema.html", include_plotlyjs="cdn"
)
print("saved results/sensitivity_ema.html")
