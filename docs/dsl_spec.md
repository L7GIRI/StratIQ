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
## Compilation rules
- `compile_spec` refuses a spec without `approved_at`.
- An entry rule is an event on bar t. The target position is +1 on a long event and -1 on a
  short event, held until the opposite event, a stop, a target or a session end. If both
  events fire on the same bar, nothing changes.
- With a `session`, entries are allowed only inside the allowed sessions. With
  `flat_at_session_end`, the last in-session bar is blocked for entries and the target
  position resets to 0 there.
- Entries are blocked while the stop distance is NaN (indicator warm-up).
- Costs are not part of the spec. They come from the base `BacktestConfig`.
- There are no exit rules yet: a long-only spec holds until a stop, target or session end.

## Added on Day 11
- `entry_order`: `MARKET` or `LIMIT`. `LIMIT` requires `limit_offset` (price units from the
  signal bar's close). `MARKET` forbids it.
- `sizing`: `FIXED_UNITS` (ounces) or `RISK_PCT` (percent of equity risked per trade, at most 5,
  requires `stop_loss`).
- Baseline specs live in `strategies/`. Opening-range breakout is not expressible yet because
  the DSL has no opening-range indicator.

## Vocabulary (kept in sync with the code by tests/unit/test_docs_sync.py)
- Indicator types: SMA, EMA, RSI, ATR, VWAP
- Condition ops: CROSS_ABOVE, CROSS_BELOW, GT, LT, GTE, LTE
- Sessions: ASIA, LONDON, NEW_YORK
- Entry orders: MARKET, LIMIT. Sizing: FIXED_UNITS, RISK_PCT
