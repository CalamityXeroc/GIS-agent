# -*- coding: utf-8 -*-
"""离线验证 task3b: census 指标汇总到市 + 相似城市排序。"""
import sys, io, os, glob, shutil, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

REPO = Path(r"E:\BST_lab\GIS_CLI")
DATA = REPO / "workspace/input/13第十三届全国大学生GIS应用技能大赛（兰州交通大学）/数据/B上午/B上午/数据"

# 复用 task3a 已算好的 county_pop/city_4508（agent 运行产物）
runs = sorted(glob.glob(str(REPO / "benchmark/runs/2026*")))
src_gdb = None
for r in reversed(runs):
    cand = os.path.join(r, "am_task3a_city_aggregate_hotspot", "workspace", "output", "result_data.gdb")
    if os.path.isdir(cand):
        src_gdb = cand
        break
print("source gdb:", src_gdb)
assert src_gdb, "需要先跑 task3a 产出 county_pop/city_4508"

WS = Path(r"C:\Users\cheng\AppData\Local\Temp") / ("bench3b_" + str(int(time.time())))
(WS / "input").mkdir(parents=True)
(WS / "output").mkdir(parents=True)
shutil.copy2(DATA / "population_census.csv", WS / "input" / "population_census.csv")

runner = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "kernel_offline")
lib = RecipeLibrary(code_runner=runner)
lib.load()

class Ctx:
    workspace = WS
    code_runner = runner

out_gdb = str(WS / "output" / "result_data.gdb")
county_pop = os.path.join(src_gdb, "county_pop")
city_4508 = os.path.join(src_gdb, "city_4508")
city_census = out_gdb + "\city_census"
sim_json = str(WS / "output" / "similarity_zhengzhou.json")

steps = [
    ("census_city_indicators", {
        "county_path": county_pop, "census_csv": str(WS / "input" / "population_census.csv"),
        "city_path": city_4508, "output_path": city_census, "pop_field": "Y2020",
    }),
    ("similarity_ranking", {
        "input_path": city_census, "fields": "OLD_PCT,FERT_PCT,SEX_RATIO",
        "reference_name": "郑州市", "output_json": sim_json, "top_n": 3,
    }),
]
for rid, params in steps:
    obs = lib.run(rid, params, Ctx())
    print(f"[{'OK ' if obs.ok else 'FAIL'}] {rid}: {obs.summary[:200]}")
    res = (obs.data or {}).get("result")
    if isinstance(res, dict):
        for k in ("feature_count", "matched_counties", "most_similar", "ranking", "sample_cities"):
            if k in res:
                print("      ->", k, "=", str(res[k])[:200])
