# -*- coding: utf-8 -*-
"""Run the topology recipe offline against a projected county to sanity-check counts."""
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.engine.state import Observation

ws = Path(sys.argv[1])
in_county = sys.argv[2]
out = sys.argv[3]

runner = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "kernel_recipe")
lib = RecipeLibrary(code_runner=runner)
lib.load()

class Ctx:
    workspace = ws
    code_runner = runner

obs = lib.run("topology_repair_keep_attrs", {
    "input_path": in_county,
    "output_path": out,
    "key_field": "code",
    "keep": "LARGEST_AREA",
}, Ctx())
print("ok=", obs.ok)
print("summary=", obs.summary)
if obs.data:
    print("assertions=", [a for a in (obs.data.get("assertions") or [])])
    print("result=", obs.data.get("result"))
runner.shutdown()
