"""Compiler tests. The reference strategies below are plain loops that do not use the engine,
the compiler or the indicators module, so they are an independent check of all three."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from stratiq.data.sessions import in_session
from stratiq.data.synthetic import make_ohlcv
from stratiq.dsl import SpecError, approve, compile_spec, run_compiled, validate_spec

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ema_cross.json"


# ---------------------------------------------------------------- reference implementations
def ref_ema(values, length):
    alpha = 2.0 / (length + 1)
    out = [math.nan] * len(values)
    prev = None
    for i, v in enumerate(values):
        prev = v if prev is None else (1 - alpha) * prev + alpha * v
        if i >= length - 1:
            out[i] = prev
    return out


def ref_rsi(values, length):
    n = len(values)
    out = [math.nan] * n
    alpha = 1.0 / length
    avg_gain = avg_loss = None
    for i in range(1, n):
        delta = values[i] - values[i - 1]
        gain, loss = max(delta, 0.0), max(-delta, 0.0)
        if avg_gain is None:
            avg_gain, avg_loss = gain, loss
        else:
            avg_gain = (1 - alpha) * avg_gain + alpha * gain
            avg_loss = (1 - alpha) * avg_loss + alpha * loss
        if i >= length:
            if avg_loss == 0:
                out[i] = 100.0 if avg_gain > 0 else math.nan
            else:
                out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def ref_run(opens, closes, long_ev, short_ev):
    """Hold the last event's side; fill at the next open; close what is open at the last close."""
    n = len(opens)
    pos = state = 0
    entry_i = entry_px = None
    trades = []
    for t in range(n - 1):
        if long_ev[t] and not short_ev[t]:
            state = 1
        elif short_ev[t] and not long_ev[t]:
            state = -1
        if state != pos:
            if pos != 0:
                trades.append((entry_i, t + 1, pos, entry_px, opens[t + 1]))
            if state != 0:
                entry_i, entry_px = t + 1, opens[t + 1]
            pos = state
    if pos != 0:
        trades.append((entry_i, n - 1, pos, entry_px, closes[n - 1]))
    return trades


def ref_ema_cross_events(closes, fast_len, slow_len):
    fast, slow = ref_ema(closes, fast_len), ref_ema(closes, slow_len)
    n = len(closes)
    long_ev, short_ev = [False] * n, [False] * n
    for t in range(1, n):
        vals = (fast[t], slow[t], fast[t - 1], slow[t - 1])
        if any(math.isnan(v) for v in vals):
            continue
        long_ev[t] = fast[t] > slow[t] and fast[t - 1] <= slow[t - 1]
        short_ev[t] = fast[t] < slow[t] and fast[t - 1] >= slow[t - 1]
    return long_ev, short_ev


def ref_rsi_events(closes, length, low, high):
    r = ref_rsi(closes, length)
    long_ev = [not math.isnan(v) and v < low for v in r]
    short_ev = [not math.isnan(v) and v > high for v in r]
    return long_ev, short_ev


# ---------------------------------------------------------------------------------- helpers
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def ema_spec() -> dict:
    d = fixture()
    for key in ("stop_loss", "take_profit", "session"):
        del d[key]
    return d


def rsi_spec() -> dict:
    return {
        "name": "RSI_Reversion",
        "instrument": "XAUUSD",
        "instrument_type": "spot_fx",
        "timeframe": "1h",
        "indicators": [{"id": "rsi14", "type": "RSI", "params": {"length": 14}}],
        "entry_long": {"all": [{"op": "LT", "left": "rsi14", "right": 30}]},
        "entry_short": {"all": [{"op": "GT", "left": "rsi14", "right": 70}]},
        "sizing": {"type": "FIXED_UNITS", "value": 1},
    }


def build(raw, df):
    comp = compile_spec(approve(validate_spec(raw)), df)
    return comp, run_compiled(comp, df)


def engine_tuples(res, idx):
    return [
        (idx.get_loc(t.entry_time), idx.get_loc(t.exit_time), t.side, t.entry_price, t.exit_price)
        for t in res.trades
    ]


# ------------------------------------------------------------------------ reference matches
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_compiled_ema_cross_equals_reference(seed):
    df = make_ohlcv(1500, seed=seed, vol=0.004)
    _, res = build(ema_spec(), df)
    closes = df["close"].to_numpy()
    long_ev, short_ev = ref_ema_cross_events(closes, 20, 50)
    expected = ref_run(df["open"].to_numpy(), closes, long_ev, short_ev)
    assert len(expected) >= 3
    assert engine_tuples(res, df.index) == expected


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_compiled_rsi_reversion_equals_reference(seed):
    df = make_ohlcv(3000, seed=seed + 10, vol=0.004)
    _, res = build(rsi_spec(), df)
    closes = df["close"].to_numpy()
    long_ev, short_ev = ref_rsi_events(closes, 14, 30, 70)
    expected = ref_run(df["open"].to_numpy(), closes, long_ev, short_ev)
    assert len(expected) >= 2
    assert engine_tuples(res, df.index) == expected


# -------------------------------------------------------------------- compiler behaviour
def test_unapproved_spec_cannot_be_compiled():
    df = make_ohlcv(100, seed=1)
    with pytest.raises(SpecError):
        compile_spec(validate_spec(ema_spec()), df)


def test_signals_are_valid_and_aligned_and_wait_for_warmup():
    df = make_ohlcv(600, seed=4, vol=0.004)
    comp, _ = build(fixture(), df)
    assert comp.signals.index.equals(df.index)
    assert set(comp.signals.unique()) <= {-1, 0, 1}
    assert (comp.signals.iloc[:50] == 0).all()  # EMA50 and ATR need warm-up


def test_target_and_stop_flow_into_the_config():
    df = make_ohlcv(300, seed=4)
    comp, _ = build(fixture(), df)
    assert comp.config.target_r == 2.0
    assert comp.config.units == 1.0
    assert isinstance(comp.stop_distance, pd.Series)
    fixed = fixture()
    fixed["stop_loss"] = {"type": "FIXED", "distance": 5.0}
    comp2, _ = build(fixed, df)
    assert comp2.stop_distance == 5.0


def test_compiled_signals_are_causal():
    raw = fixture()
    del raw["session"]  # session-end flags look one bar ahead on the clock, not on prices
    df = make_ohlcv(500, seed=7, vol=0.004)
    full = compile_spec(approve(validate_spec(raw)), df)
    head = compile_spec(approve(validate_spec(raw)), df.iloc[:200])
    pd.testing.assert_series_equal(full.signals.iloc[:200], head.signals, check_freq=False)
    pd.testing.assert_series_equal(
        full.stop_distance.iloc[:200], head.stop_distance, check_freq=False
    )


def test_session_rules_keep_every_position_inside_new_york_hours():
    df = make_ohlcv(1500, seed=3, vol=0.004)
    comp, res = build(fixture(), df)
    ins = in_session(df.index, "NEW_YORK")
    assert len(res.trades) > 0
    assert (comp.signals[~ins] == 0).all()
    assert (comp.signals[comp.session_end] == 0).all()
    for t in res.trades:
        assert ins.loc[t.entry_time]
        if t.exit_reason != "end":
            assert ins.loc[t.exit_time]
