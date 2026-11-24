# Execution assumptions

These are the rules the engine follows. Tests in `tests/unit/test_engine_*.py` and
`tests/validation/` enforce them.

1. A signal on bar t uses data up to the close of bar t only.
2. The resulting order fills at the open of bar t+1. A signal on the last bar is ignored.
3. Stop distance is taken at the signal bar. After the fill,
   `stop = entry_open -/+ distance` and `target = entry_open +/- target_r * distance`.
4. Stop and target are active from the entry bar onward.
5. If a bar opens through a stop (or target), the fill is at the open, not at the level.
6. If one bar touches both stop and target, the stop is assumed to fill first (conservative).
7. After a stop or target exit, the same held signal does not re-enter. It must change first.
8. Every fill is adverse by `slippage + spread / 2`. Commission is per unit per side.
9. One position at a time. A flip closes and reopens at the same open.
10. `session_end` flags close the position at that bar's close. A position left open at
    the end of data is closed at the last close.
11. Equity is marked to market at each close.

## Known simplifications
- Flags from `session_end_flags` use the next bar's timestamp, so a gap in data can end a
  session early. This is schedule information, not price information.
- Fixed position size and market orders only (limit orders and ATR sizing come on Day 10).