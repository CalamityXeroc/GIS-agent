# -*- coding: utf-8 -*-
"""Deterministic offline validation of the full am_task2b pipeline via recipes.

Chain: csv_points_spatial_join -> polygon_neighbor_stats -> graduated_colors_map
then evaluate the task assertions. No LLM involved.
"""
import sys, io, shutil, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

import yaml
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

REPO = Path(r"E:\BST_lab\GIS_CLI")
DATA = REPO / "workspace/input/13第十三届全国大学生GIS应用技能大赛（兰州交通大学）/数据/B上午/B上午/数据"
WS = Path(r"C:\Users\cheng\AppData\Local\Temp") / ("bench2b_" + str(int(time.time())))
print("WS:", WS)
(WS / "input").mkdir(parents=True)
(WS / "output").mkdir(parents=True)
for name in ("county.shp", "county.dbf", "county.shx", "county.prj", "county.cpg", "population.CSV"):
    src = DATA / name
    if src.exists():
        shutil.copy2(src, WS / "input" / name)

runner = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "kernel_offline")
lib = RecipeLibrary(code_runner=runner)
lib.load()


class Ctx:
    workspace = WS
    code_runner = runner


gdb = str(WS / "output" / "result_data.gdb")
county = str(WS / "input" / "county.shp")
county_pop = gdb + "\\county_pop"
jpg = str(WS / "output" / "county_highlight.jpg")

steps = [
    ("csv_points_spatial_join", {
        "csv_path": str(WS / "input" / "population.CSV"),
        "lon_field": "CenterLon", "lat_field": "CenterLat",
        "target_path": county, "output_path": county_pop,
        "source_srid": 4326, "target_srid": 4508,
    }),
    ("polygon_neighbor_stats", {
        "input_path": county_pop, "output_path": county_pop,
        "value_field": "Y2020", "mean_field": "NB_MEAN2020", "flag_field": "ABOVE_MEAN",
    }),
    ("graduated_colors_map", {
        "input_path": county_pop, "field_name": "ABOVE_MEAN",
        "output_path": jpg, "title": "高于接壤区县均值分布图",
    }),
]
for rid, params in steps:
    obs = lib.run(rid, params, Ctx())
    print(f"[{'OK ' if obs.ok else 'FAIL'}] {rid}: {obs.summary[:200]}")

task = yaml.safe_load((REPO / "benchmark/tasks/am_task2b_neighbor_highlight.yaml").read_text(encoding="utf-8"))
assertions = []
for raw in task["assertions"]:
    item = dict(raw)
    if item.get("path") and not Path(item["path"]).is_absolute():
        item["path"] = str(WS / item["path"])
    assertions.append(item)
print("\n=== 断言评估 ===")
results = lib.run_assertions(assertions)
ok = 0
for r in results:
    ok += 1 if r.ok else 0
    print(f"  [{'OK  ' if r.ok else 'FAIL'}] {r.kind:<22} {r.detail[:80]}")
print(f"断言 {ok}/{len(results)}")
runner.shutdown()
