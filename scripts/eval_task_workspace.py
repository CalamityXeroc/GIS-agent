# -*- coding: utf-8 -*-
"""Inspect a GDB and evaluate a task's assertions against it."""
import sys, io, json
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

import arcpy
import yaml
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

ws = Path(sys.argv[1])
task_file = Path(sys.argv[2])
gdb = ws / "output" / "result_data.gdb"

print(f"=== {gdb} ===")
if gdb.exists():
    for dirpath, dirnames, filenames in arcpy.da.Walk(str(gdb), datatype=["FeatureClass", "Table"]):
        for fn in filenames:
            p = dirpath + "\\" + fn
            try:
                desc = arcpy.Describe(p)
                sr = getattr(desc, "spatialReference", None)
                n = int(arcpy.management.GetCount(p)[0])
                fields = [f.name for f in arcpy.ListFields(p)]
                print(f"  {fn:<24} {str(getattr(desc,'shapeType','') or 'Table'):<9} n={n:<5} wkid={getattr(sr,'factoryCode','')} fields={fields[:8]}")
            except Exception as e:
                print(f"  {fn} ERR {str(e)[:60]}")

task = yaml.safe_load(task_file.read_text(encoding="utf-8"))
runner = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "kernel_eval")
lib = RecipeLibrary(code_runner=runner)
assertions = []
for raw in task.get("assertions", []):
    item = dict(raw)
    if item.get("path") and not Path(item["path"]).is_absolute():
        item["path"] = str(ws / item["path"])
    assertions.append(item)
print("\n=== 断言评估 ===")
results = lib.run_assertions(assertions)
ok = 0
for r in results:
    ok += 1 if r.ok else 0
    print(f"  [{'OK  ' if r.ok else 'FAIL'}] {r.kind:<24} {r.detail[:90]}")
print(f"断言 {ok}/{len(results)}")
runner.shutdown()
