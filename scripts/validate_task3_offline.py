# -*- coding: utf-8 -*-
"""Deterministic offline validation of the am_task3a pipeline via recipes.

Chain: csv_points_spatial_join -> project_to_crs(city) -> aggregate_to_zones
       -> hotspot_analysis x3. No LLM involved.
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
WS = Path(r"C:\Users\cheng\AppData\Local\Temp") / ("bench3a_" + str(int(time.time())))
print("WS:", WS)
(WS / "input").mkdir(parents=True)
(WS / "output").mkdir(parents=True)
for stem in ("county", "city"):
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
        src = DATA / f"{stem}{ext}"
        if src.exists():
            shutil.copy2(src, WS / "input" / f"{stem}{ext}")
shutil.copy2(DATA / "population.CSV", WS / "input" / "population.CSV")

runner = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "kernel_offline")
lib = RecipeLibrary(code_runner=runner)
lib.load()


class Ctx:
    workspace = WS
    code_runner = runner


gdb = str(WS / "output" / "result_data.gdb")
county = str(WS / "input" / "county.shp")
city_src = str(WS / "input" / "city.shp")
county_pop = gdb + "\county_pop"
city_4508 = gdb + "\city_4508"
city_pop = gdb + "\city_pop"

steps = [
    ("csv_points_spatial_join", {
        "csv_path": str(WS / "input" / "population.CSV"),
        "lon_field": "CenterLon", "lat_field": "CenterLat",
        "target_path": county, "output_path": county_pop,
        "source_srid": 4326, "target_srid": 4508,
    }),
    ("project_to_crs", {
        "input_path": city_src, "output_path": city_4508, "target_srid": 4508,
    }),
    ("aggregate_to_zones", {
        "zones_path": city_4508, "source_path": county_pop,
        "output_path": city_pop, "stat_fields": "Y2011,Y2012,Y2013,Y2020",
    }),
    ("hotspot_analysis", {
        "input_path": city_pop, "field_name": "SUM_Y2011",
        "output_path": gdb + "\city_hot_2011",
    }),
    ("hotspot_analysis", {
        "input_path": city_pop, "field_name": "SUM_Y2012",
        "output_path": gdb + "\city_hot_2012",
    }),
    ("hotspot_analysis", {
        "input_path": city_pop, "field_name": "SUM_Y2013",
        "output_path": gdb + "\city_hot_2013",
    }),
]
for rid, params in steps:
    obs = lib.run(rid, params, Ctx())
    print(f"[{'OK ' if obs.ok else 'FAIL'}] {rid}: {obs.summary[:220]}")
    if obs.data and isinstance(obs.data, dict):
        res = obs.data.get("result")
        if isinstance(res, dict):
            print("      ->", {k: res[k] for k in list(res)[:6] if k in res})

task = yaml.safe_load((REPO / "benchmark/tasks/am_task3a_city_aggregate_hotspot.yaml").read_text(encoding="utf-8"))
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
    print(f"  [{'OK  ' if r.ok else 'FAIL'}] {r.kind:<22} {r.detail[:90]}")
print(f"断言 {ok}/{len(results)}")
runner.shutdown()
