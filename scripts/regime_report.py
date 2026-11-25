"""Per-regime performance of each baseline on the saved gold data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stratiq.data.store import load_dataset
from stratiq.dsl import approve, compile_spec, run_compiled, validate_spec
from stratiq.engine import BacktestConfig
from stratiq.lab import per_regime_performance, regimes

BASE = BacktestConfig(spread=0.30, slippage=0.20, commission_per_unit=0.02)

pd.set_option("display.width", 200)
for path in sorted(Path("strategies").glob("*_1h.json")):
    spec = approve(validate_spec(json.loads(path.read_text(encoding="utf-8"))))
    df, _ = load_dataset("gc_1h", "data/parquet")
    res = run_compiled(compile_spec(spec, df, BASE), df)
    print(f"\n{spec.name}")
    table = per_regime_performance(res, regimes(df), column="vol")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
