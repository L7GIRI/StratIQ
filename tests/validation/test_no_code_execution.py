"""The pipeline never executes model output: no dynamic-execution calls exist in src/stratiq."""

import ast
import json
import os
from pathlib import Path

from stratiq.llm import nl_to_spec

SRC = Path(__file__).parents[2] / "src" / "stratiq"
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}
BANNED_ATTRS = {"system", "popen", "Popen", "check_output", "check_call", "import_module"}
BANNED_ROOT_ATTRS = {("subprocess", "run"), ("subprocess", "call"), ("os", "exec")}
BANNED_MODULES = {"subprocess", "pickle", "marshal", "shelve"}


def findings(source: str, name: str = "<mem>") -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BANNED_CALLS:
                found.append(f"{name}:{node.lineno} calls {f.id}()")
            if isinstance(f, ast.Attribute):
                if f.attr in BANNED_ATTRS or f.attr in {"eval", "exec"}:
                    found.append(f"{name}:{node.lineno} calls .{f.attr}()")
                if isinstance(f.value, ast.Name) and (f.value.id, f.attr) in BANNED_ROOT_ATTRS:
                    found.append(f"{name}:{node.lineno} calls {f.value.id}.{f.attr}()")
        if isinstance(node, ast.Import):
            found += [
                f"{name}:{node.lineno} imports {a.name}"
                for a in node.names
                if a.name.split(".")[0] in BANNED_MODULES
            ]
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in BANNED_MODULES:
            found.append(f"{name}:{node.lineno} imports from {node.module}")
    return found


def test_scanner_negative_control_finds_planted_calls():
    bad = "import subprocess\nx = eval('1')\nos.system('whoami')\n__import__('os')\n"
    assert len(findings(bad)) == 4


def test_no_dynamic_execution_anywhere_in_the_package():
    files = list(SRC.rglob("*.py"))
    assert len(files) > 20  # the scan actually covered the package
    problems = []
    for path in files:
        problems += findings(path.read_text(encoding="utf-8"), str(path.relative_to(SRC)))
    assert problems == []


def test_code_like_model_output_stays_an_inert_string(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "system", lambda *a, **k: calls.append(a))
    draft = {
        "name": "__import__('os').system('echo PWNED')",
        "indicators": [{"id": "rsi14", "type": "RSI", "length": 14}],
        "long_rules": [{"op": "LT", "left": "rsi14", "right": 30}],
        "risk_pct": None,
        "units": 1,
    }
    res = nl_to_spec("anything", lambda m, s=None: json.dumps(draft))
    assert res.spec.name == "__import__('os').system('echo PWNED')"
    assert calls == []
    assert res.spec.approved_at is None
