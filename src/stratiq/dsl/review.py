"""Human review: plain-English description and the approval gate."""

from __future__ import annotations

from collections.abc import Callable

from stratiq.dsl.models import StrategySpec
from stratiq.dsl.validator import approve

OPS = {
    "CROSS_ABOVE": "crosses above",
    "CROSS_BELOW": "crosses below",
    "GT": "is above",
    "LT": "is below",
    "GTE": "is at or above",
    "LTE": "is at or below",
}


def _rules(group) -> str:
    return " AND ".join(f"{c.left} {OPS[c.op]} {c.right}" for c in group.all)


def describe_spec(spec: StrategySpec) -> str:
    lines = [f"Strategy: {spec.name}", f"Market: {spec.instrument}, {spec.timeframe} bars"]
    for i in spec.indicators:
        if i.type == "VWAP":
            lines.append(f"Indicator {i.id}: daily-reset VWAP")
        else:
            lines.append(f"Indicator {i.id}: {i.type}({int(i.params['length'])})")
    if spec.entry_long is not None:
        lines.append(f"Go LONG when: {_rules(spec.entry_long)}")
    if spec.entry_short is not None:
        lines.append(f"Go SHORT when: {_rules(spec.entry_short)}")
    if spec.entry_order == "LIMIT":
        lines.append(
            f"Entry: limit order {spec.limit_offset:g} below (long) or above (short) the signal "
            "bar's close, valid for the next bar only; no fill if price never reaches it."
        )
    else:
        lines.append("Entry: market order at the open of the bar after the signal.")
    sl = spec.stop_loss
    if sl is None:
        lines.append("Stop-loss: none")
    elif sl.type == "ATR":
        lines.append(f"Stop-loss: {sl.multiplier:g} x {sl.indicator} from the entry price")
    else:
        lines.append(f"Stop-loss: {sl.distance:g} price units from the entry price")
    if spec.take_profit is not None:
        lines.append(f"Take-profit: {spec.take_profit.value:g}R (a multiple of the stop distance)")
    if spec.sizing.type == "RISK_PCT":
        lines.append(f"Size: risk {spec.sizing.value:g}% of equity per trade (size = risk / stop)")
    else:
        lines.append(f"Size: fixed {spec.sizing.value:g} ounce(s)")
    if spec.session is not None:
        names = ", ".join(spec.session.allowed)
        end = "closes at session end" if spec.session.flat_at_session_end else "no forced close"
        lines.append(f"Sessions: enters only during {names}; {end}")
    lines.append(
        "Exits: opposite signal, stop, target, or session end. Costs come from the run config."
    )
    return "\n".join(lines)


def request_approval(
    spec: StrategySpec,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> StrategySpec | None:
    """Show the rules and return an approved copy only if the human types exactly APPROVE."""
    print_fn(describe_spec(spec))
    print_fn(spec.model_dump_json(indent=2, exclude={"approved_at"}))
    answer = input_fn("Type APPROVE to approve (anything else rejects): ")
    if answer.strip() == "APPROVE":
        return approve(spec)
    return None
