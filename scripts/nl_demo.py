"""Try the NL -> DSL pipeline against your local Ollama model. Nothing is run or approved."""

from __future__ import annotations

import sys

from stratiq.llm import NL2DSLError, OllamaClient, nl_to_spec

text = " ".join(sys.argv[1:]) or (
    "Buy gold when the 20 EMA crosses above the 50 EMA and sell when it crosses below. "
    "Use a 1.5 ATR stop, a 2R target and risk 1% per trade on the hourly chart."
)
print("request:", text)
try:
    res = nl_to_spec(text, OllamaClient())
except NL2DSLError as exc:
    print(f"FAILED after {exc.attempts} attempts:")
    for e in exc.errors:
        print("  -", e)
    raise SystemExit(1) from exc

print(f"attempts: {res.attempts}")
if res.spec is None:
    print("The model needs clarification:")
    for q in res.questions:
        print("  ?", q)
else:
    print(res.spec.model_dump_json(indent=2, exclude={"approved_at"}))
    print("(not approved: approval is the Day 17 gate)")
