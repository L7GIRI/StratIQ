"""Semantic validation. Shape errors come from Pydantic, meaning errors from here."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import ValidationError

from stratiq.dsl.models import Indicator, StrategySpec

PRICE_FIELDS = {"open", "high", "low", "close"}
INSTRUMENT_TYPES = {"XAUUSD": "spot_fx", "GC=F": "futures_continuous", "GLD": "etf"}


class SpecError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _param_errors(ind: Indicator, where: str) -> list[str]:
    if ind.type == "VWAP":
        return [f"{where}.params: VWAP takes no parameters"] if ind.params else []
    if set(ind.params) != {"length"}:
        return [f"{where}.params: {ind.type} needs exactly one parameter, length"]
    length = ind.params["length"]
    if isinstance(length, float) and not length.is_integer():
        return [f"{where}.params.length: must be a whole number"]
    if not 1 <= length <= 500:
        return [f"{where}.params.length: must be an integer between 1 and 500"]
    return []


def _bound_conflicts(name: str, conditions) -> list[str]:
    """Numeric bounds on the same operand that no value can satisfy (x > 70 and x < 30)."""
    lowers: dict[str, list[tuple[float, bool]]] = {}
    uppers: dict[str, list[tuple[float, bool]]] = {}
    for c in conditions:
        if isinstance(c.right, str):
            continue
        if c.op in ("GT", "GTE"):
            lowers.setdefault(c.left, []).append((float(c.right), c.op == "GT"))
        elif c.op in ("LT", "LTE"):
            uppers.setdefault(c.left, []).append((float(c.right), c.op == "LT"))
    errors = []
    for left, los in lowers.items():
        for lv, ls in los:
            for uv, us in uppers.get(left, []):
                if lv > uv or (lv == uv and (ls or us)):
                    errors.append(f"{name}: conflicting conditions on {left!r}")
    return errors


def _pair_conflicts(name: str, conditions) -> list[str]:
    """Opposite strict relations between the same two operands."""
    up_ops, down_ops = {"GT", "CROSS_ABOVE"}, {"LT", "CROSS_BELOW"}
    seen: dict[tuple[str, str], set[str]] = {}
    for c in conditions:
        if not isinstance(c.right, str) or c.op not in up_ops | down_ops:
            continue
        direction = "up" if c.op in up_ops else "down"
        a, b = c.left, c.right
        if a > b:  # canonical order, flipping direction when swapped
            a, b = b, a
            direction = "down" if direction == "up" else "up"
        seen.setdefault((a, b), set()).add(direction)
    return [
        f"{name}: conflicting conditions on {a!r} and {b!r}"
        for (a, b), dirs in seen.items()
        if len(dirs) == 2
    ]


def semantic_errors(spec: StrategySpec) -> list[str]:
    errors: list[str] = []
    ids = [i.id for i in spec.indicators]
    by_id = {i.id: i for i in spec.indicators}

    for dup in sorted({x for x in ids if ids.count(x) > 1}):
        errors.append(f"indicators: duplicate id {dup!r}")
    for k, ind in enumerate(spec.indicators):
        if ind.id in PRICE_FIELDS:
            errors.append(f"indicators[{k}].id: {ind.id!r} is a reserved price field name")
        errors += _param_errors(ind, f"indicators[{k}]")
        if ind.type == "VWAP" and spec.timeframe != "1h":
            errors.append(f"indicators[{k}]: VWAP requires timeframe '1h'")

    operands = set(ids) | PRICE_FIELDS
    groups = (("entry_long", spec.entry_long), ("entry_short", spec.entry_short))
    for name, group in groups:
        if group is None:
            continue
        for k, c in enumerate(group.all):
            where = f"{name}.all[{k}]"
            if c.left not in operands:
                errors.append(f"{where}.left: {c.left!r} is not a defined indicator or price field")
            if isinstance(c.right, str):
                if c.right not in operands:
                    errors.append(
                        f"{where}.right: {c.right!r} is not a defined indicator or price field"
                    )
                elif c.left == c.right:
                    errors.append(f"{where}: compares {c.left!r} with itself")
            else:
                left_ind = by_id.get(c.left)
                if left_ind and left_ind.type == "RSI" and not 0 <= c.right <= 100:
                    errors.append(f"{where}.right: RSI threshold must be between 0 and 100")
        errors += _bound_conflicts(name, group.all)
        errors += _pair_conflicts(name, group.all)

    if spec.entry_long is None and spec.entry_short is None:
        errors.append("entry_long/entry_short: at least one entry rule is required")
    elif (
        spec.entry_long is not None
        and spec.entry_short is not None
        and spec.entry_long.model_dump() == spec.entry_short.model_dump()
    ):
        errors.append("entry_long and entry_short are identical")

    if spec.stop_loss is not None and spec.stop_loss.type == "ATR":
        ref = by_id.get(spec.stop_loss.indicator or "")
        if ref is None or ref.type != "ATR":
            errors.append("stop_loss.indicator: must reference an ATR indicator")
    if spec.take_profit is not None and spec.stop_loss is None:
        errors.append("take_profit: requires stop_loss (target is a multiple of the stop distance)")

    if spec.session is not None and spec.timeframe != "1h":
        errors.append("session: requires timeframe '1h'")
    if spec.entry_order == "LIMIT" and spec.limit_offset is None:
        errors.append("limit_offset: required when entry_order is LIMIT")
    if spec.entry_order == "MARKET" and spec.limit_offset is not None:
        errors.append("limit_offset: only allowed when entry_order is LIMIT")
    if spec.sizing.type == "RISK_PCT":
        if spec.stop_loss is None:
            errors.append("sizing: RISK_PCT requires stop_loss")
        if spec.sizing.value > 5:
            errors.append("sizing.value: RISK_PCT must be at most 5 (percent of equity)")
    expected = INSTRUMENT_TYPES[spec.instrument]
    if spec.instrument_type != expected:
        errors.append(f"instrument_type: {spec.instrument} must be {expected}")
    return errors


def validate_spec(raw: dict | str) -> StrategySpec:
    """Parse and validate. Raises SpecError listing every problem found."""
    try:
        if isinstance(raw, str):
            spec = StrategySpec.model_validate_json(raw)
        else:
            spec = StrategySpec.model_validate(raw)
    except ValidationError as exc:
        raise SpecError(
            [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        ) from exc
    errors = semantic_errors(spec)
    if errors:
        raise SpecError(errors)
    return spec


def approve(spec: StrategySpec, when: datetime | None = None) -> StrategySpec:
    """Return a copy with approved_at set. Call this only after a human has reviewed the rules."""
    return spec.model_copy(update={"approved_at": when or datetime.now(UTC)})


def ensure_approved(spec: StrategySpec) -> None:
    if spec.approved_at is None:
        raise SpecError(["approved_at: strategy has not been approved by a human"])


__all__ = ["SpecError", "approve", "ensure_approved", "semantic_errors", "validate_spec"]
_ = json  # json is imported for callers that pass raw strings through this module
