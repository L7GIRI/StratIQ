"""Run every baseline strategy on the saved gold data and report the results as they are."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stratiq.analytics import compute_metrics, drawdown_series
from stratiq.data.store import load_dataset
from stratiq.dsl import approve, compile_spec, run_compiled, validate_spec
from stratiq.engine import BacktestConfig

DATASETS = {"1h": "gc_1h", "1d": "gc_1d"}


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def write_chart(result, title: str, path: Path) -> None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(
        go.Scatter(x=result.equity.index, y=result.equity.values, name="equity"), row=1, col=1
    )
    dd = drawdown_series(result.equity) * 100
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd.values, name="drawdown %", fill="tozeroy"), row=2, col=1
    )
    fig.update_layout(title=title, height=600)
    fig.write_html(str(path), include_plotlyjs="cdn")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", default="strategies")
    p.add_argument("--data", default="data/parquet")
    p.add_argument("--out", default="results")
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--spread", type=float, default=0.30)
    p.add_argument("--slippage", type=float, default=0.20)
    p.add_argument("--commission", type=float, default=0.02)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = BacktestConfig(
        initial_cash=args.cash,
        spread=args.spread,
        slippage=args.slippage,
        commission_per_unit=args.commission,
    )

    rows, report = [], []
    for path in sorted(Path(args.strategies).glob("*.json")):
        spec = validate_spec(json.loads(path.read_text(encoding="utf-8")))
        spec = approve(spec)  # the author reviewed these files; the app will ask for a click
        df, meta = load_dataset(DATASETS[spec.timeframe], args.data)
        compiled = compile_spec(spec, df, base)
        result = run_compiled(compiled, df)
        m = compute_metrics(result)

        result.trades_df().to_csv(out / f"{path.stem}_trades.csv", index=False)
        write_chart(result, spec.name, out / f"{path.stem}.html")
        report.append(
            {
                "strategy": spec.name,
                "timeframe": spec.timeframe,
                "data_sha256": meta["sha256"],
                "data_first": meta["first"],
                "data_last": meta["last"],
                "config": result.config.to_dict(),
                "metrics": m,
            }
        )
        rows.append(
            {
                "strategy": spec.name,
                "trades": m["trades"],
                "return": m["total_return"],
                "max_dd": m["max_drawdown"],
                "sharpe": m["sharpe"],
                "win_rate": m["win_rate"],
                "profit_factor": m["profit_factor"],
                "expectancy": m["expectancy"],
            }
        )

    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    (out / "baselines.json").write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    print(f"saved {out / 'baselines.json'} (metrics), plus trade CSVs and charts")


if __name__ == "__main__":
    main()
