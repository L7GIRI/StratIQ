import json
from pathlib import Path

from stratiq.dsl import validate_spec
from stratiq.dsl.review import describe_spec, request_approval

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ema_cross.json"


def spec():
    return validate_spec(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_description_states_the_rules_in_plain_english():
    text = describe_spec(spec())
    for phrase in (
        "ema_fast crosses above ema_slow",
        "Go SHORT when",
        "1.5 x atr14",
        "2R",
        "fixed 1 ounce",
        "NEW_YORK",
        "closes at session end",
        "market order",
    ):
        assert phrase in text, phrase


def test_only_the_exact_word_approve_approves():
    s = spec()
    approved = request_approval(s, input_fn=lambda _: "APPROVE", print_fn=lambda _: None)
    assert approved is not None and approved.approved_at is not None
    assert s.approved_at is None  # the original is untouched
    for answer in ("approve", "yes", "y", "", "APPROVED", "no"):
        assert request_approval(s, input_fn=lambda _, a=answer: a, print_fn=lambda _: None) is None


def test_review_prints_description_and_json_but_not_approved_at():
    shown: list[str] = []
    request_approval(spec(), input_fn=lambda _: "no", print_fn=shown.append)
    joined = "\n".join(shown)
    assert "Go LONG when" in joined and '"name"' in joined
    assert "approved_at" not in joined
