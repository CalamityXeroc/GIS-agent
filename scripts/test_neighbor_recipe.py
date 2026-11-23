# -*- coding: utf-8 -*-
"""Run polygon_neighbor_stats recipe offline to sanity-check."""
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

ws = Path(sys.argv[1])
in_fc = sys.argv[2]
out_fc = sys.argv[3]
value_field = sys.argv[4] if len(sys.argv) > 4 else "Y2020"

runner = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "kernel_recipe")
lib = RecipeLibrary(code_runner=runner)
lib.load()

class Ctx:
    workspace = ws
    code_runner = runner

obs = lib.run("polygon_neighbor_stats", {
    "input_path": in_fc,
    "output_path": out_fc,
    "value_field": value_field,
    "mean_field": "NB_MEAN2020",
    "flag_field": "ABOVE_MEAN",
}, Ctx())
print("ok=", obs.ok)
print("summary=", obs.summary[:400])
if obs.data:
    print("result=", obs.data.get("result"))
runner.shutdown()
