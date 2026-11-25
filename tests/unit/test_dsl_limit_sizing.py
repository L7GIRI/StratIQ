import json
from pathlib import Path

import pytest

from stratiq.data.synthetic import make_ohlcv
from stratiq.dsl import SpecError, approve, compile_spec, validate_spec

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ema_cross.json"


def good() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_limit_entry_with_offset_is_valid():
    raw = good()
    raw["entry_order"] = "LIMIT"
    raw["limit_offset"] = 2.0
    assert validate_spec(raw).limit_offset == 2.0


def test_limit_without_offset_is_rejected():
    raw = good()
    raw["entry_order"] = "LIMIT"
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    assert "limit_offset" in str(exc.value)


def test_offset_on_a_market_order_is_rejected():
    raw = good()
    raw["limit_offset"] = 1.0
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    assert "limit_offset" in str(exc.value)


def test_risk_pct_sizing_is_valid_with_a_stop():
    raw = good()
    raw["sizing"] = {"type": "RISK_PCT", "value": 1.0}
    assert validate_spec(raw).sizing.type == "RISK_PCT"


def test_risk_pct_without_a_stop_is_rejected():
    raw = good()
    raw["sizing"] = {"type": "RISK_PCT", "value": 1.0}
    del raw["stop_loss"]
    del raw["take_profit"]
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    assert "RISK_PCT requires stop_loss" in str(exc.value)


def test_risk_pct_above_five_is_rejected():
    raw = good()
    raw["sizing"] = {"type": "RISK_PCT", "value": 10}
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    assert "at most 5" in str(exc.value)


def test_limit_and_risk_flow_into_the_engine_config():
    raw = good()
    raw["entry_order"] = "LIMIT"
    raw["limit_offset"] = 2.0
    raw["sizing"] = {"type": "RISK_PCT", "value": 1.0}
    df = make_ohlcv(300, seed=2)
    cfg = compile_spec(approve(validate_spec(raw)), df).config
    assert cfg.entry_mode == "limit"
    assert cfg.limit_offset == 2.0
    assert cfg.risk_pct == pytest.approx(0.01)
