# -*- coding: utf-8 -*-
"""Re-evaluate benchmark assertions against an existing run workspace.

用法: python scripts/reeval_assertions2.py <workspace> <task.yaml>
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

import yaml

from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runtime.code_runner import CodeRunner

ws = Path(sys.argv[1])
task_path = Path(sys.argv[2])
task = yaml.safe_load(task_path.read_text(encoding="utf-8"))

runner = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "kernel_reeval")
lib = RecipeLibrary(code_runner=runner)

assertions = []
for raw in task.get("assertions", []):
    item = dict(raw)
    if item.get("path") and not Path(str(item["path"])).is_absolute():
        item["path"] = str(ws / item["path"])
    assertions.append(item)

results = lib.run_assertions(assertions)
ok = 0
for r in results:
    mark = "OK  " if r.ok else "FAIL"
    ok += 1 if r.ok else 0
    print(f"[{mark}] {r.kind:<28} {r.detail[:130]}")
print(f"\n{ok}/{len(results)} 通过")
