# -*- coding: utf-8 -*-
"""Debug: render and run csv_points_spatial_join in a FRESH workspace."""
import sys, io, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

SRC = Path(r"E:\BST_lab\GIS_CLI\workspace\input\13第十三届全国大学生GIS应用技能大赛（兰州交通大学）\数据\B上午\B上午\数据")
WS = Path(r"C:\Users\cheng\AppData\Local\Temp\bench2b_fresh")
shutil.rmtree(WS, ignore_errors=True)
(WS / "input").mkdir(parents=True)
(WS / "output").mkdir(parents=True)
for n in ("county.shp", "county.dbf", "county.shx", "county.prj", "county.cpg", "population.CSV"):
    if (SRC / n).exists():
        shutil.copy2(SRC / n, WS / "input" / n)

runner = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "kernel_dbg")
lib = RecipeLibrary(code_runner=runner)
lib.load()
recipe = lib.get("csv_points_spatial_join")
params = {
    "csv_path": str(WS / "input" / "population.CSV"),
    "lon_field": "CenterLon", "lat_field": "CenterLat",
    "target_path": str(WS / "input" / "county.shp"),
    "output_path": str(WS / "output" / "result_data.gdb" / "county_pop"),
    "source_srid": 4326, "target_srid": 4508,
}
code = recipe.render(params)
res = runner.run(code, timeout=600, workspace=str(WS))
print("ok=", res.ok)
print("stdout tail:", (res.stdout or "")[-600:])
if res.error:
    print("error msg:", res.error.get("message"))
runner.shutdown()
