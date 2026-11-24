from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Execution assumptions. Costs are in price units per fill, commission per unit per side."""

    initial_cash: float = 100_000.0
    units: float = 1.0
    commission_per_unit: float = 0.0
    spread: float = 0.0  # full spread; half is paid on each fill
    slippage: float = 0.0  # adverse price move per fill
    target_r: float | None = None  # take-profit as a multiple of the stop distance

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be > 0")
        if self.units <= 0:
            raise ValueError("units must be > 0")
        if self.commission_per_unit < 0 or self.spread < 0 or self.slippage < 0:
            raise ValueError("costs must be >= 0")
        if self.target_r is not None and self.target_r <= 0:
            raise ValueError("target_r must be > 0")


@dataclass(frozen=True)
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int  # +1 long, -1 short
    units: float
    entry_price: float  # actual fill, including costs
    exit_price: float  # actual fill, including costs
    commission: float = 0.0  # round-trip total
    exit_reason: str = "signal"  # signal, stop, target, session_end, end

    @property
    def pnl(self) -> float:
        """Net P&L after commission (spread and slippage are already in the fill prices)."""
        return self.side * self.units * (self.exit_price - self.entry_price) - self.commission


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    config: BacktestConfig

    def trades_df(self) -> pd.DataFrame:
        cols = [
            "entry_time",
            "exit_time",
            "side",
            "units",
            "entry_price",
            "exit_price",
            "commission",
            "exit_reason",
            "pnl",
        ]
        rows = [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side,
                "units": t.units,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "commission": t.commission,
                "exit_reason": t.exit_reason,
                "pnl": t.pnl,
            }
            for t in self.trades
        ]
        return pd.DataFrame(rows, columns=cols)
