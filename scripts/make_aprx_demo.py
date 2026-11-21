# -*- coding: utf-8 -*-
"""生成制图工程示例：自包含 GDB + JPG + 可编辑 APRX（放入成果展示目录）。"""
import sys, io, os, glob, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

DEMO = Path(r"E:\BST_lab\GIS_CLI\workspace\成果展示\制图工程示例")
DEMO.mkdir(parents=True, exist_ok=True)

import arcpy
arcpy.env.overwriteOutput = True

# 1) 组装自包含数据（从已有 run 产出里挑两个图层拷进 demo.gdb）
runs = sorted(glob.glob(r"E:\BST_lab\GIS_CLI\benchmark\runs\2026*"))
src3b = src2b = None
for r in reversed(runs):
    c3 = os.path.join(r, "am_task3b_census_indicators_similarity", "workspace", "output", "result_data.gdb", "city_census")
    if arcpy.Exists(c3) and src3b is None:
        src3b = c3
    c2 = os.path.join(r, "am_task2b_neighbor_highlight", "workspace", "output", "result_data.gdb", "county_pop")
    if arcpy.Exists(c2) and src2b is None:
        src2b = c2
    if src3b and src2b:
        break
print("源: 3b =", src3b)
print("源: 2b =", src2b)

gdb = DEMO / "demo_data.gdb"
if gdb.exists():
    shutil.rmtree(gdb, ignore_errors=True)
arcpy.management.CreateFileGDB(str(DEMO), "demo_data.gdb")
arcpy.management.CopyFeatures(src3b, str(gdb / "city_census_2020"))
arcpy.management.CopyFeatures(src2b, str(gdb / "county_pop_2020"))
print("已建 demo_data.gdb:", arcpy.ListFeatureClasses(str(gdb)) or [f for f in os.listdir(str(gdb)) if not f.startswith("a00")][:0])

# 2) 生成两张专题图 + 工程
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary
runner = CodeRunner(workspace=DEMO, state_dir=DEMO / ".gis_agent_kernel")
lib = RecipeLibrary(code_runner=runner); lib.load()
class Ctx:
    workspace = DEMO
    code_runner = runner

jobs = [
    ("city_census_2020", "OLD_PCT", "地市老年人占比（2020）", "city_census_2020_老年人占比.jpg", "Greens"),
    ("county_pop_2020", "ABOVE_MEAN", "高于接壤区县均值区县（2020）", "county_2020_高于邻接均值.jpg", "Reds"),
]
for fc, fld, title, jpg, ramp in jobs:
    obs = lib.run("graduated_colors_map", {
        "input_path": str(gdb / fc), "field_name": fld,
        "output_path": str(DEMO / jpg), "title": title, "color_ramp_name": ramp,
    }, Ctx())
    res = (obs.data or {}).get("result") or {}
    print(f"[{'OK ' if obs.ok else 'FAIL'}] {jpg} | aprx={os.path.basename(str(res.get('aprx') or ''))} ({res.get('aprx_size')} bytes)")
runner.shutdown()

print("\n=== 示例目录内容 ===")
for p in sorted(DEMO.rglob("*")):
    if p.is_file() and not p.name.startswith(("a00", "timestamps", "gdb")):
        print(f"   {p.relative_to(DEMO)}  {p.stat().st_size//1024} KB")
