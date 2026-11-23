# -*- coding: utf-8 -*-
"""用 task3a 的产出生成 2011-2013 热点/冷点地图（供展示）。"""
import sys, io, os, glob, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

REPO = Path(r"E:\BST_lab\GIS_CLI")
# 找 task3a 的产出 gdb
src_gdb = None
for r in sorted(glob.glob(str(REPO / "benchmark/runs/2026*")), reverse=True):
    cand = os.path.join(r, "am_task3a_city_aggregate_hotspot", "workspace", "output", "result_data.gdb")
    if os.path.isdir(cand):
        src_gdb = cand
        break
assert src_gdb, "未找到 task3a 产出"
print("source gdb:", src_gdb)

SHOW = REPO / "workspace" / "成果展示" / "AM任务三"
SHOW.mkdir(parents=True, exist_ok=True)

runner = CodeRunner(workspace=SHOW, state_dir=REPO / "workspace" / ".gis_agent" / "kernel_show")
lib = RecipeLibrary(code_runner=runner)
lib.load()

class Ctx:
    workspace = SHOW
    code_runner = runner

for year in (2011, 2012, 2013):
    out_jpg = str(SHOW / f"3a_热点冷点_{year}年.jpg")
    obs = lib.run("graduated_colors_map", {
        "input_path": os.path.join(src_gdb, f"city_hot_{year}"),
        "field_name": "Gi_Bin",
        "output_path": out_jpg,
        "title": f"{year}年地市人口热点/冷点分布图",
    }, Ctx())
    print(f"[{'OK ' if obs.ok else 'FAIL'}] {year}: {obs.summary[:120]}")

# 复制 3b 的成果
src3b = None
for r in sorted(glob.glob(str(REPO / "benchmark/runs/2026*")), reverse=True):
    cand = os.path.join(r, "am_task3b_census_indicators_similarity", "workspace", "output")
    if os.path.isdir(cand):
        src3b = cand
        break
if src3b:
    for name, alias in (("old_pct_map.jpg", "3b_老年人占比.jpg"),
                        ("fert_pct_map.jpg", "3b_育龄女性占比.jpg"),
                        ("sex_ratio_map.jpg", "3b_性别比.jpg"),
                        ("similarity_zhengzhou.json", "3b_与郑州最相似城市.json")):
        s = os.path.join(src3b, name)
        if os.path.exists(s):
            shutil.copy2(s, SHOW / alias)
            print("copied", alias)
# 复制 2b 成果
for r in sorted(glob.glob(str(REPO / "benchmark/runs/2026*")), reverse=True):
    s = os.path.join(r, "am_task2b_neighbor_highlight", "workspace", "output", "county_highlight.jpg")
    if os.path.exists(s):
        shutil.copy2(s, SHOW / "2b_高于邻接均值高亮图.jpg")
        print("copied 2b_高于邻接均值高亮图.jpg")
        break
runner.shutdown()
print("\n展示目录:", SHOW)
for f in sorted(os.listdir(SHOW)):
    p = SHOW / f
    print("  ", f, f"{p.stat().st_size//1024} KB")
