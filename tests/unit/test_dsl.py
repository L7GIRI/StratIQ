import copy
import json
from pathlib import Path

import pytest

from stratiq.dsl import SpecError, StrategySpec, approve, ensure_approved, validate_spec

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ema_cross.json"


def good() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


RSI = {"id": "rsi14", "type": "RSI", "params": {"length": 14}}


def unknown_indicator(d):
    d["indicators"][0]["type"] = "MACD"


def undefined_reference(d):
    d["entry_long"]["all"][0]["right"] = "ema_slowest"


def out_of_range_target(d):
    d["take_profit"]["value"] = 100


def conflicting_bounds(d):
    d["indicators"].append(RSI)
    d["entry_long"] = {
        "all": [
            {"op": "GT", "left": "rsi14", "right": 70},
            {"op": "LT", "left": "rsi14", "right": 30},
        ]
    }


def conflicting_cross(d):
    d["entry_long"] = {
        "all": [
            {"op": "CROSS_ABOVE", "left": "ema_fast", "right": "ema_slow"},
            {"op": "CROSS_BELOW", "left": "ema_fast", "right": "ema_slow"},
        ]
    }


def limit_order(d):
    d["entry_order"] = "LIMIT"


def missing_sizing(d):
    del d["sizing"]


def extra_field_code_injection(d):
    d["expression"] = "__import__('os').system('whoami')"


def target_without_stop(d):
    del d["stop_loss"]


def duplicate_ids(d):
    d["indicators"].append(copy.deepcopy(d["indicators"][0]))


def no_entries(d):
    del d["entry_long"]
    del d["entry_short"]


def stop_references_non_atr(d):
    d["stop_loss"]["indicator"] = "ema_fast"


def stop_shape(d):
    d["stop_loss"] = {"type": "FIXED", "indicator": "atr14", "distance": 5}


def instrument_mismatch(d):
    d["instrument_type"] = "etf"


def session_on_daily(d):
    d["timeframe"] = "1d"


def zero_length(d):
    d["indicators"][0]["params"]["length"] = 0


def rsi_threshold_out_of_range(d):
    d["indicators"].append(RSI)
    d["entry_long"] = {"all": [{"op": "GT", "left": "rsi14", "right": 150}]}


def identical_entries(d):
    d["entry_short"] = copy.deepcopy(d["entry_long"])


def compares_with_itself(d):
    d["entry_long"]["all"][0]["right"] = "ema_fast"


CASES = [
    (unknown_indicator, "indicators.0.type"),
    (undefined_reference, "ema_slowest"),
    (out_of_range_target, "take_profit.value"),
    (conflicting_bounds, "conflicting"),
    (conflicting_cross, "conflicting"),
    (limit_order, "entry_order"),
    (missing_sizing, "sizing"),
    (extra_field_code_injection, "expression"),
    (target_without_stop, "take_profit"),
    (duplicate_ids, "duplicate"),
    (no_entries, "at least one entry rule"),
    (stop_references_non_atr, "ATR"),
    (stop_shape, "stop_loss"),
    (instrument_mismatch, "instrument_type"),
    (session_on_daily, "session"),
    (zero_length, "length"),
    (rsi_threshold_out_of_range, "RSI threshold"),
    (identical_entries, "identical"),
    (compares_with_itself, "itself"),
]


def test_good_spec_validates():
    spec = validate_spec(good())
    assert isinstance(spec, StrategySpec)
    assert spec.name == "EMA_Cross_20_50"
    assert spec.entry_order == "MARKET"
    assert spec.approved_at is None
    assert [i.id for i in spec.indicators] == ["ema_fast", "ema_slow", "atr14"]


@pytest.mark.parametrize(("mutate", "expected"), CASES, ids=[m.__name__ for m, _ in CASES])
def test_bad_specs_are_rejected_with_a_clear_message(mutate, expected):
    raw = good()
    mutate(raw)
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    assert expected in str(exc.value)
    assert exc.value.errors  # a non-empty list of messages


def test_all_semantic_errors_are_reported_together():
    raw = good()
    duplicate_ids(raw)
    undefined_reference(raw)
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    joined = " | ".join(exc.value.errors)
    assert "duplicate" in joined
    assert "ema_slowest" in joined
    assert len(exc.value.errors) >= 2


def test_json_string_input_is_accepted():
    assert validate_spec(FIXTURE.read_text(encoding="utf-8")).name == "EMA_Cross_20_50"


def test_invalid_json_string_is_a_spec_error():
    with pytest.raises(SpecError):
        validate_spec("{not json")


def test_long_only_and_short_only_specs_are_valid():
    long_only = good()
    del long_only["entry_short"]
    assert validate_spec(long_only).entry_short is None
    short_only = good()
    del short_only["entry_long"]
    assert validate_spec(short_only).entry_long is None


def test_fixed_stop_without_target_is_valid():
    raw = good()
    raw["stop_loss"] = {"type": "FIXED", "distance": 5.0}
    del raw["take_profit"]
    assert validate_spec(raw).stop_loss.distance == 5.0


def test_numeric_bounds_that_overlap_are_valid():
    raw = good()
    raw["indicators"].append(RSI)
    raw["entry_long"] = {
        "all": [
            {"op": "GT", "left": "rsi14", "right": 30},
            {"op": "LT", "left": "rsi14", "right": 70},
        ]
    }
    assert validate_spec(raw)


def test_approval_gate():
    spec = validate_spec(good())
    with pytest.raises(SpecError):
        ensure_approved(spec)
    approved = approve(spec)
    ensure_approved(approved)
    assert approved.approved_at is not None
    assert spec.approved_at is None  # the original is not mutated
