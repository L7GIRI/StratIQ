# StratIQ

Research and educational LLM-assisted backtester for gold. **Not a trading system.**
A natural-language idea becomes a validated JSON strategy (never code), is approved by a
human, and runs on a small backtesting engine with explicit execution rules.

## What exists so far
- Data: GC=F hourly and daily via yfinance, cleaned, hashed, stored as Parquet (data is never committed).
- Indicators: SMA, EMA, RSI, ATR, daily-reset VWAP. Sessions are DST-safe.
- Engine: next-open fills, stop/target (stop first when both hit), costs, limit orders, risk sizing.
- Tests: golden hand-computed trades, a lookahead truncation test with a negative control,
  accounting invariants, hypothesis property tests.
- DSL: Pydantic models, semantic validator, approval gate, compiler checked against
  independent reference implementations.
- Metrics: return, drawdown, Sharpe, Sortino, Calmar, win rate, profit factor, expectancy.

## Results
See `results/baselines.json`. Costs are assumptions (spread 0.30, slippage 0.20, commission
0.02 per ounce), not measured values. Results go here after they are produced:

| Strategy | Trades | Return | Max DD | Sharpe |
|---|---|---|---|---|
| (fill from `python scripts/run_backtest.py`) | | | | |

## Limitations
- Free data: about 2.4 years of hourly bars; GC=F is a continuous futures series with roll artifacts.
- Market and limit orders only, one position at a time, fixed cost assumptions.
- Baselines are examples, not recommendations.

## Rules
Execution rules: `docs/execution_assumptions.md`. DSL: `docs/dsl_spec.md`.

## Progress
Days 1-12 done: scaffold, data, indicators, engine, validation tests, metrics, DSL, compiler, limit/risk, baselines.- Day 13: causal regimes (vol, trend) and bootstrap CIs per regime.
- Day 14: walk-forward (train-only optimisation, stitched out-of-sample equity).
