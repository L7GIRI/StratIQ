"""Bar-based backtest loop. See docs/execution_assumptions.md.

Order of events inside bar i:
  A. gap check: an open position whose stop/target the OPEN gaps through exits at the open
  B. signal: target position decided at the close of bar i-1 is executed
       (market: at the open of bar i; limit: at the limit if bar i trades through it)
  C. intrabar: stop / target checked against the bar's low and high (stop first)
  D. session end: position closed at the close of a flagged bar
  E. equity marked to market at the close
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stratiq.engine.models import BacktestConfig, BacktestResult, Trade


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
    stop_distance: float | pd.Series | None = None,
    session_end: pd.Series | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    for col in ("open", "close"):
        if col not in df.columns:
            raise ValueError(f"df is missing column {col!r}")
    if len(signals) != len(df) or not signals.index.equals(df.index):
        raise ValueError("signals must have exactly the same index as df")

    sig = signals.fillna(0).to_numpy()
    if not np.isin(sig, (-1, 0, 1)).all():
        raise ValueError("signals must only contain -1, 0 or 1")
    sig = sig.astype(int)

    n = len(df)
    if stop_distance is not None or config.entry_mode == "limit":
        for col in ("high", "low"):
            if col not in df.columns:
                raise ValueError(f"df is missing column {col!r}")

    stop_arr = None
    if stop_distance is not None:
        if isinstance(stop_distance, pd.Series):
            if not stop_distance.index.equals(df.index):
                raise ValueError("stop_distance must have the same index as df")
            stop_arr = stop_distance.to_numpy(dtype=float)
        else:
            stop_arr = np.full(n, float(stop_distance))
    if config.target_r is not None and stop_arr is None:
        raise ValueError("target_r requires stop_distance")
    if config.risk_pct is not None and stop_arr is None:
        raise ValueError("risk_pct requires stop_distance")

    end_arr = None
    if session_end is not None:
        if not session_end.index.equals(df.index):
            raise ValueError("session_end must have the same index as df")
        end_arr = session_end.to_numpy(dtype=bool)

    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float) if "high" in df.columns else opens
    lows = df["low"].to_numpy(dtype=float) if "low" in df.columns else opens
    times = df.index
    cost_px = config.slippage + config.spread / 2.0
    use_limit = config.entry_mode == "limit"

    equity = np.empty(n)
    trades: list[Trade] = []
    pos = 0
    cur_units = config.units
    entry_fill = 0.0
    entry_idx = 0
    open_comm = 0.0
    stop = np.nan
    target = np.nan
    suppress = 0
    realized = 0.0

    def fill_price(raw: float, side: int, is_entry: bool) -> float:
        """Every market fill is adverse: buys pay more, sells receive less."""
        buying = (side == 1) == is_entry
        return raw + cost_px if buying else raw - cost_px

    def close_trade(exit_idx: int, raw_price: float, reason: str) -> None:
        nonlocal pos, realized, open_comm, stop, target, suppress
        px = fill_price(raw_price, pos, False)
        comm = open_comm + config.commission_per_unit * cur_units
        trade = Trade(
            times[entry_idx], times[exit_idx], pos, cur_units, entry_fill, px, comm, reason
        )
        realized += trade.pnl
        trades.append(trade)
        if reason in ("stop", "target"):
            suppress = pos
        pos = 0
        open_comm = 0.0
        stop = np.nan
        target = np.nan

    for i in range(n):
        if i > 0:
            # A. gap through protective levels at the open
            if pos == 1:
                if opens[i] <= stop:
                    close_trade(i, opens[i], "stop")
                elif opens[i] >= target:
                    close_trade(i, opens[i], "target")
            elif pos == -1:
                if opens[i] >= stop:
                    close_trade(i, opens[i], "stop")
                elif opens[i] <= target:
                    close_trade(i, opens[i], "target")

            # B. execute the target position decided at the previous close
            want = int(sig[i - 1])
            if suppress != 0 and want != suppress:
                suppress = 0
            if want != pos and not (suppress != 0 and want == suppress):
                if pos != 0:
                    close_trade(i, opens[i], "signal")
                if want != 0:
                    raw = None
                    if use_limit:
                        limit = closes[i - 1] - want * config.limit_offset
                        if want == 1:
                            if opens[i] <= limit:
                                raw = opens[i]  # gap through the limit: fill at the better open
                            elif lows[i] <= limit:
                                raw = limit
                        elif opens[i] >= limit:
                            raw = opens[i]
                        elif highs[i] >= limit:
                            raw = limit
                        fill = raw  # resting order: no slippage or spread
                    else:
                        raw = opens[i]
                        fill = fill_price(raw, want, True)

                    if raw is not None:
                        d = None
                        if stop_arr is not None:
                            d = stop_arr[i - 1]
                            if not np.isfinite(d) or d <= 0:
                                raise ValueError(f"invalid stop distance {d} at {times[i - 1]}")
                        size = config.units
                        if config.risk_pct is not None:
                            eq_prev = equity[i - 1]
                            size = eq_prev * config.risk_pct / d
                            size = min(size, eq_prev * config.max_leverage / raw)
                        if size > 0 and np.isfinite(size):
                            pos = want
                            cur_units = float(size)
                            entry_idx = i
                            entry_fill = fill
                            open_comm = config.commission_per_unit * cur_units
                            if d is not None:
                                stop = raw - pos * d
                                if config.target_r is not None:
                                    target = raw + pos * config.target_r * d

            # C. intrabar stop / target; stop is assumed to hit first
            if pos == 1:
                if lows[i] <= stop:
                    close_trade(i, stop, "stop")
                elif highs[i] >= target:
                    close_trade(i, target, "target")
            elif pos == -1:
                if highs[i] >= stop:
                    close_trade(i, stop, "stop")
                elif lows[i] <= target:
                    close_trade(i, target, "target")

        # D. flat at session end
        if end_arr is not None and pos != 0 and end_arr[i]:
            close_trade(i, closes[i], "session_end")

        # E. mark to market
        unrealized = pos * cur_units * (closes[i] - entry_fill) if pos != 0 else 0.0
        equity[i] = config.initial_cash + realized + unrealized - open_comm

    if pos != 0:
        close_trade(n - 1, closes[-1], "end")
        equity[-1] = config.initial_cash + realized

    return BacktestResult(pd.Series(equity, index=df.index, name="equity"), trades, config)
