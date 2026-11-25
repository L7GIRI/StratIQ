"""Natural language -> validated spec -> plain-English review -> approval by typing APPROVE."""

from __future__ import annotations

import sys
from pathlib import Path

from stratiq.dsl.review import request_approval
from stratiq.llm import NL2DSLError, OllamaClient, nl_to_spec

text = (
    " ".join(sys.argv[1:])
    or "Buy gold when the 20 EMA crosses above the 50 EMA, stop 1.5 ATR, risk 1%."
)
try:
    res = nl_to_spec(text, OllamaClient())
except NL2DSLError as exc:
    raise SystemExit(f"no valid strategy: {exc}") from exc

if res.spec is None:
    print("The model needs clarification:")
    for q in res.questions:
        print("  ?", q)
    raise SystemExit(0)

approved = request_approval(res.spec)
if approved is None:
    print("Not approved. Nothing was saved or run.")
else:
    out = Path("results/approved")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{approved.name.replace(' ', '_')}.json"
    path.write_text(approved.model_dump_json(indent=2), encoding="utf-8")
    print(f"approved at {approved.approved_at}; saved {path}")
