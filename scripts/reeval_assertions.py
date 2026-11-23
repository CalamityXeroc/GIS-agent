# -*- coding: utf-8 -*-
"""Re-evaluate benchmark assertions against an existing run workspace."""
import sys, io, json
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

import yaml
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

ws = Path(sys.argv[1])
task = yaml.safe_load(open(r"E:\BST_lab\GIS_CLI\benchmark\tasks\am_task1_quality.yaml", encoding="utf-8"))

runner = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "kernel")
lib = RecipeLibrary(code_runner=runner)

assertions = []
for raw in task["assertions"]:
    item = dict(raw)
    if item.get("path") and not Path(item["path"]).is_absolute():
        item["path"] = str(ws / item["path"])
    assertions.append(item)

results = lib.run_assertions(assertions)
ok = 0
for r in results:
    mark = "OK  " if r.ok else "FAIL"
    ok += 1 if r.ok else 0
    print(f"[{mark}] {r.kind:<18} {r.target}\n       {r.detail}")
print(f"\n断言 {ok}/{len(results)} 通过")
runner.shutdown()
