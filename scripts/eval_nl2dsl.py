"""Run the 20-prompt evaluation against your local Ollama model and save the numbers."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from stratiq.llm import OllamaClient
from stratiq.llm.evaluate import evaluate_prompt, load_prompts, summarize

p = argparse.ArgumentParser()
p.add_argument("--model", default="qwen2.5:7b")
p.add_argument("--prompts", default="eval/nl2dsl_prompts.jsonl")
p.add_argument("--out", default="results/nl2dsl_eval.json")
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

items = load_prompts(args.prompts)[: args.limit]
chat = OllamaClient(model=args.model)
rows = []
for item in items:
    row = evaluate_prompt(item, chat)
    rows.append(row)
    ok = f"{sum(row['fields'].values())}/{len(row['fields'])}" if row["fields"] else "-"
    print(
        f"#{row['id']:>2} {row['category']:<11} {row['outcome']:<9} "
        f"attempts={row['attempts']} fields={ok} leaked={row['leaked']} {row['seconds']}s"
    )

summary = summarize(rows)
print(json.dumps(summary, indent=2))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(
    json.dumps(
        {
            "model": args.model,
            "date_utc": datetime.now(UTC).isoformat(),
            "summary": summary,
            "rows": rows,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"saved {args.out}")
