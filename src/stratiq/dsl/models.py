"""Strategy DSL: a strict, declarative description. No code, no free-form expressions."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IndicatorType = Literal["SMA", "EMA", "RSI", "ATR", "VWAP"]
Op = Literal["CROSS_ABOVE", "CROSS_BELOW", "GT", "LT", "GTE", "LTE"]
SessionName = Literal["ASIA", "LONDON", "NEW_YORK"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Indicator(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    type: IndicatorType
    params: dict[str, int | float] = Field(default_factory=dict)


class Condition(Strict):
    op: Op
    left: str
    right: str | float


class ConditionGroup(Strict):
    """All conditions must hold on the same bar."""

    all: list[Condition] = Field(min_length=1, max_length=5)


class StopLoss(Strict):
    type: Literal["ATR", "FIXED"]
    indicator: str | None = None
    multiplier: float | None = Field(default=None, gt=0, le=10)
    distance: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _shape(self) -> StopLoss:
        if self.type == "ATR":
            if self.indicator is None or self.multiplier is None or self.distance is not None:
                raise ValueError("ATR stop needs indicator and multiplier, and no distance")
        elif self.distance is None or self.indicator is not None or self.multiplier is not None:
            raise ValueError("FIXED stop needs distance only")
        return self


class TakeProfit(Strict):
    type: Literal["R_MULTIPLE"]
    value: float = Field(gt=0, le=20)


class Sizing(Strict):
    type: Literal["FIXED_UNITS", "RISK_PCT"]
    value: float = Field(gt=0, le=1000)


class SessionRule(Strict):
    allowed: list[SessionName] = Field(min_length=1)
    flat_at_session_end: bool = False


class StrategySpec(Strict):
    name: str = Field(min_length=1, max_length=60)
    instrument: Literal["XAUUSD", "GC=F", "GLD"]
    instrument_type: Literal["spot_fx", "futures_continuous", "etf"]
    timeframe: Literal["1h", "1d"]
    entry_order: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_offset: float | None = Field(default=None, ge=0, le=1000)
    indicators: list[Indicator] = Field(default_factory=list, max_length=12)
    entry_long: ConditionGroup | None = None
    entry_short: ConditionGroup | None = None
    stop_loss: StopLoss | None = None
    take_profit: TakeProfit | None = None
    sizing: Sizing
    session: SessionRule | None = None
    approved_at: datetime | None = None
