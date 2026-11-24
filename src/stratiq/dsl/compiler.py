"""Compile an approved StrategySpec into engine inputs. No eval, no exec, no generated code."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from stratiq.data.sessions import in_session
from stratiq.dsl.models import Condition, ConditionGroup, Indicator, StrategySpec
from stratiq.dsl.validator import SpecError, ensure_approved, semantic_errors
from stratiq.engine import BacktestConfig, BacktestResult, run_backtest
from stratiq.indicators import atr, ema, rsi, sma, vwap

REQUIRED = ("open", "high", "low", "close", "volume")


@dataclass
class CompiledStrategy:
    signals: pd.Series
    stop_distance: float | pd.Series | None
    session_end: pd.Series | None
    config: BacktestConfig
    indicators: dict[str, pd.Series]


def compute_indicator(ind: Indicator, df: pd.DataFrame) -> pd.Series:
    if ind.type == "VWAP":
        return vwap(df)
    length = int(ind.params["length"])
    if ind.type == "SMA":
        return sma(df["close"], length)
    if ind.type == "EMA":
        return ema(df["close"], length)
    if ind.type == "RSI":
        return rsi(df["close"], length)
    return atr(df["high"], df["low"], df["close"], length)


def _operand(name: str, df: pd.DataFrame, inds: dict[str, pd.Series]) -> pd.Series:
    return inds[name] if name in inds else df[name]


def _eval_condition(c: Condition, df: pd.DataFrame, inds: dict[str, pd.Series]) -> np.ndarray:
    left = _operand(c.left, df, inds)
    if isinstance(c.right, str):
        right = _operand(c.right, df, inds)
    else:
        right = pd.Series(float(c.right), index=df.index)
    if c.op == "GT":
        out = left > right
    elif c.op == "LT":
        out = left < right
    elif c.op == "GTE":
        out = left >= right
    elif c.op == "LTE":
        out = left <= right
    elif c.op == "CROSS_ABOVE":
        out = (left > right) & (left.shift(1) <= right.shift(1))
    else:  # CROSS_BELOW
        out = (left < right) & (left.shift(1) >= right.shift(1))
    return out.to_numpy(dtype=bool)  # comparisons with NaN are False


def _eval_group(group: ConditionGroup | None, df: pd.DataFrame, inds) -> np.ndarray:
    if group is None:
        return np.zeros(len(df), dtype=bool)
    result = np.ones(len(df), dtype=bool)
    for c in group.all:
        result &= _eval_condition(c, df, inds)
    return result


def _hold_state(long_ev: np.ndarray, short_ev: np.ndarray, reset: np.ndarray) -> np.ndarray:
    """Target position: set by entry events, held otherwise, cleared on reset bars."""
    out = np.zeros(len(long_ev), dtype=int)
    pos = 0
    for i in range(len(out)):
        if reset[i]:
            pos = 0
        elif long_ev[i] and not short_ev[i]:
            pos = 1
        elif short_ev[i] and not long_ev[i]:
            pos = -1
        out[i] = pos
    return out


def compile_spec(
    spec: StrategySpec, df: pd.DataFrame, base_config: BacktestConfig | None = None
) -> CompiledStrategy:
    ensure_approved(spec)
    errors = semantic_errors(spec)
    if errors:
        raise SpecError(errors)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing columns: {missing}")
    if df.index.tz is None:
        raise ValueError("df index must be timezone-aware")

    n = len(df)
    inds = {i.id: compute_indicator(i, df) for i in spec.indicators}
    long_ev = _eval_group(spec.entry_long, df, inds)
    short_ev = _eval_group(spec.entry_short, df, inds)

    allowed = np.ones(n, dtype=bool)
    reset = np.zeros(n, dtype=bool)
    session_end = None
    if spec.session is not None:
        ins = np.zeros(n, dtype=bool)
        for name in spec.session.allowed:
            ins |= in_session(df.index, name).to_numpy()
        allowed &= ins
        if spec.session.flat_at_session_end:
            flags = ins & ~np.append(ins[1:], False)
            allowed &= ~flags
            reset = flags
            session_end = pd.Series(flags, index=df.index, name="session_end")

    stop: float | pd.Series | None = None
    if spec.stop_loss is not None:
        if spec.stop_loss.type == "ATR":
            stop = inds[spec.stop_loss.indicator] * spec.stop_loss.multiplier
            vals = stop.to_numpy(dtype=float)
            allowed &= np.isfinite(vals) & (vals > 0)
        else:
            stop = float(spec.stop_loss.distance)

    signals = _hold_state(long_ev & allowed, short_ev & allowed, reset)

    base = base_config or BacktestConfig()
    target_r = spec.take_profit.value if spec.take_profit is not None else None
    config = replace(base, units=float(spec.sizing.value), target_r=target_r)
    return CompiledStrategy(
        pd.Series(signals, index=df.index, name="signal"), stop, session_end, config, inds
    )


def run_compiled(compiled: CompiledStrategy, df: pd.DataFrame) -> BacktestResult:
    return run_backtest(
        df,
        compiled.signals,
        compiled.config,
        stop_distance=compiled.stop_distance,
        session_end=compiled.session_end,
    )
