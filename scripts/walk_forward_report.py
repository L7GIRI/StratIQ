"""Walk-forward on the saved GC=F hourly data for the EMA-cross baseline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from stratiq.data.store import load_dataset
from stratiq.engine import BacktestConfig
from stratiq.lab import walk_forward

BASE = json.loads(Path("strategies/ema_cross_1h.json").read_text(encoding="utf-8"))
CFG = BacktestConfig(spread=0.30, slippage=0.20, commission_per_unit=0.02)


def make_spec(p: dict) -> dict:
    d = copy.deepcopy(BASE)
    d["indicators"][0]["params"]["length"] = p["fast"]
    d["indicators"][1]["params"]["length"] = p["slow"]
    return d


df, _ = load_dataset("gc_1h", "data/parquet")
grid = {"fast": [10, 20, 30], "slow": [40, 50, 80]}
wf = walk_forward(df, make_spec, grid, train_bars=4000, test_bars=1000, base_config=CFG)

pd.set_option("display.width", 220)
print(wf.windows.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nin-sample minus out-of-sample (mean total_return): {wf.degradation:.4f}")
print(f"stitched OOS return: {wf.stitched_equity.iloc[-1] / wf.stitched_equity.iloc[0] - 1:.4f}")
