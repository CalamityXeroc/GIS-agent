# -*- coding: utf-8 -*-
"""Deterministic offline validation of the am_task3b pipeline via recipes."""
import sys, io, shutil, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

import yaml
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

REPO = Path(r"E:\BST_lab\GIS_CLI")
DATA = REPO / "workspace/input/13第十三届全国大学生GIS应用技能大赛（兰州交通大学）/数据/B上午/B上午/数据"
WS = Path(r"C:\Users\cheng\AppData\Local\Temp") / ("bench3b_full_" + str(int(time.time())))
print("WS:", WS)
(WS / "input").mkdir(parents=True)
(WS / "output").mkdir(parents=True)
for stem in ("county", "city"):
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
        src = DATA / f"{stem}{ext}"
        if src.exists():
            shutil.copy2(src, WS / "input" / f"{stem}{ext}")
for name in ("population.CSV", "population_census.csv"):
    shutil.copy2(DATA / name, WS / "input" / name)

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
city_census = gdb + "\city_census"
sim_json = str(WS / "output" / "similarity_zhengzhou.json")

steps = [
    ("csv_points_spatial_join", {
        "csv_path": str(WS / "input" / "population.CSV"),
        "lon_field": "CenterLon", "lat_field": "CenterLat",
        "target_path": county, "output_path": county_pop,
        "source_srid": 4326, "target_srid": 4508,
    }),
    ("project_to_crs", {"input_path": city_src, "output_path": city_4508, "target_srid": 4508}),
    ("census_city_indicators", {
        "county_path": county_pop, "census_csv": str(WS / "input" / "population_census.csv"),
        "city_path": city_4508, "output_path": city_census, "pop_field": "Y2020",
    }),
    ("similarity_ranking", {
        "input_path": city_census, "fields": "OLD_PCT,FERT_PCT,SEX_RATIO",
        "reference_name": "郑州市", "output_json": sim_json, "top_n": 3,
    }),
    ("graduated_colors_map", {
        "input_path": city_census, "field_name": "OLD_PCT",
        "output_path": str(WS / "output" / "old_pct_map.jpg"), "title": "地市老年人占比分布图",
    }),
    ("graduated_colors_map", {
        "input_path": city_census, "field_name": "FERT_PCT",
        "output_path": str(WS / "output" / "fert_pct_map.jpg"), "title": "地市育龄女性占比分布图",
    }),
    ("graduated_colors_map", {
        "input_path": city_census, "field_name": "SEX_RATIO",
        "output_path": str(WS / "output" / "sex_ratio_map.jpg"), "title": "地市男女性别比分布图",
    }),
]
for rid, params in steps:
    obs = lib.run(rid, params, Ctx())
    print(f"[{'OK ' if obs.ok else 'FAIL'}] {rid}: {obs.summary[:180]}")

task = yaml.safe_load((REPO / "benchmark/tasks/am_task3b_census_indicators_similarity.yaml").read_text(encoding="utf-8"))
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
    print(f"  [{'OK  ' if r.ok else 'FAIL'}] {r.kind:<20} {r.detail[:80]}")
print(f"断言 {ok}/{len(results)}")
runner.shutdown()
