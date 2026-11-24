# StratIQ strategy DSL

A strategy is a JSON object validated by `stratiq.dsl.validate_spec`. It contains no code and
no free-form expressions. Anything outside this schema is rejected.

See `tests/fixtures/ema_cross.json` for a complete example.

| Field | Rule |
|---|---|
| instrument / instrument_type | XAUUSD=spot_fx, GC=F=futures_continuous, GLD=etf |
| timeframe | `1h` or `1d` (sessions and VWAP need `1h`) |
| entry_order | `MARKET` only (limit orders arrive on Day 10) |
| indicators | SMA, EMA, RSI, ATR (param `length` 1..500), VWAP (no params) |
| conditions | ops: CROSS_ABOVE, CROSS_BELOW, GT, LT, GTE, LTE; operands are indicator ids, price fields or numbers |
| entry_long / entry_short | `{"all": [conditions]}`, at least one of the two |
| stop_loss | ATR (indicator + multiplier) or FIXED (distance) |
| take_profit | R_MULTIPLE (needs stop_loss) |
| sizing | FIXED_UNITS |
| session | allowed sessions and `flat_at_session_end` |
| approved_at | set only by `approve()` after human review |

The validator rejects unknown indicator types, undefined references, out-of-range numbers,
conflicting conditions, unsupported order types and missing required fields.