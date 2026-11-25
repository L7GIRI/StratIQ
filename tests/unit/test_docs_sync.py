from pathlib import Path
from typing import get_args

from stratiq.dsl.models import IndicatorType, Op, SessionName, Sizing, StrategySpec

DOC = (Path(__file__).parents[2] / "docs" / "dsl_spec.md").read_text(encoding="utf-8")


def test_every_dsl_keyword_is_documented():
    sizing_types = get_args(Sizing.model_fields["type"].annotation)
    entry_orders = get_args(StrategySpec.model_fields["entry_order"].annotation)
    for word in (
        *get_args(IndicatorType),
        *get_args(Op),
        *get_args(SessionName),
        *sizing_types,
        *entry_orders,
    ):
        assert word in DOC, f"{word} is accepted by the DSL but missing from docs/dsl_spec.md"
