# StratIQ

Research and educational LLM-assisted backtester for gold. Not a trading system.

## Progress
- Day 1: scaffold, CI, data-source check.- Day 2: cleaning, Parquet store with hash, synthetic generator.
- Day 3: sessions (DST-safe) and indicators (SMA, EMA, RSI, ATR, VWAP) with hand-computed tests.
- Day 4: engine v1 (next-open fills, long/short, equity curve, trade log).
- Day 5: stop/target (gap and both-hit rules), costs, session-flat close.
- Day 6: validation tests (lookahead truncation, invariants, hypothesis properties).
- Day 7: metrics (return, drawdown, Sharpe, Sortino, Calmar, win rate, profit factor, expectancy).
- Day 8: strategy DSL (Pydantic models, semantic validator, approval gate).
- Day 9: compiler (spec to engine inputs) with reference-implementation tests.
