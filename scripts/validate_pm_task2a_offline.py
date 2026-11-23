# -*- coding: utf-8 -*-
"""PM 任务二a 离线确定性验证：相交 → 按社区融合 → 连接 → 人均绿地 → 指数归一化。

承接任务一的产物（community_elder/ELDER_COMM）。用于在 Agent 实测前确认数值与断言门槛。
"""
import sys, io, os, glob, shutil, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

import yaml
import arcpy
arcpy.env.overwriteOutput = True
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

REPO = Path(r"E:\BST_lab\GIS_CLI")
DATA = glob.glob(str(REPO / "workspace/input/**/B下午/B下午/数据"), recursive=True)[0]

# 承接最近一次 pm_task1 运行的工作区（模拟 carry_from）
runs = sorted(glob.glob(str(REPO / "benchmark/runs/*/pm_task1_elderly_raster/workspace")), reverse=True)
assert runs, "找不到任务一的工作区"
PREV = Path(runs[0])
WS = Path(r"C:\Users\cheng\AppData\Local\Temp") / ("pm2a_" + str(int(time.time())))
(WS / "input").mkdir(parents=True)
(WS / "output").mkdir(parents=True)
for name in ("district", "community"):
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
        f = Path(DATA) / f"{name}{ext}"
        if f.exists():
            shutil.copy2(f, WS / "input" / f.name)
for f in Path(DATA).glob("green_land.*"):
    shutil.copy2(f, WS / "input" / f.name)
for f in Path(DATA).glob("pop_density.tif*"):
    shutil.copy2(f, WS / "input" / f.name)
prev_out = PREV / "output"
if prev_out.exists():
    for item in prev_out.iterdir():
        target = WS / "output" / item.name
        if not target.exists():
            shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
print(f"承接任务一产物: {prev_out}")

runner = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "kernel")
lib = RecipeLibrary(code_runner=runner)
lib.load()


class Ctx:
    workspace = WS
    code_runner = runner


GDB = str(WS / "output" / "result_data.gdb")
I = lambda n: str(WS / "input" / n)
G = lambda n: GDB + "\\" + n

steps = [
    ("投影 green_land → 4548", "project_to_crs",
     {"input_path": I("green_land.shp"), "output_path": G("green_land_prj"), "target_srid": 4548}),
    ("相交 绿地 × 社区", "intersect_layers",
     {"input_path": G("green_land_prj"), "overlay_path": G("community_prj"), "output_path": G("green_inter")}),
    ("按社区(BSM)融合统计面积", "dissolve_with_stats",
     {"input_path": G("green_inter"), "dissolve_field": "BSM", "stat_fields": "Shape_Area SUM",
      "output_path": G("green_comm_stat")}),
    ("绿地面积挂回社区", "attribute_join_table",
     {"target_path": G("community_elder"), "target_key": "BSM", "join_table": G("green_comm_stat"),
      "join_key": "BSM", "join_fields": "SUM_Shape_Area", "output_path": G("community_green")}),
    ("绿地面积字段更名 GREEN_AREA", "field_calculate",
     {"input_path": G("community_green"), "field_name": "GREEN_AREA", "expression": "!SUM_Shape_Area!",
      "field_type": "DOUBLE"}),
    ("人均绿地面积 PER_CAPITA", "field_calculate",
     {"input_path": G("community_green"), "field_name": "PER_CAPITA",
      "expression": "None if (!GREEN_AREA! is None or !ELDER_COMM! in (None, 0)) else !GREEN_AREA!/!ELDER_COMM!",
      "field_type": "DOUBLE", "expression_type": "PYTHON3"}),
    ("人均绿地指数 GREEN_IDX（极值标准化）", "field_calculate",
     {"input_path": G("community_green"), "field_name": "GREEN_IDX",
      "expression": "None if !PER_CAPITA! is None else ((!PER_CAPITA! - gmin) / (gmax - gmin))",
      "field_type": "DOUBLE", "expression_type": "PYTHON3", "code_block": ""}),
]
for label, rid, params in steps:
    t0 = time.time()
    obs = lib.run(rid, params, Ctx())
    print(f"[{'OK ' if obs.ok else 'FAIL'}] {label:30s} {rid:22s} {time.time()-t0:6.1f}s | {str(obs.summary)[:100]}")

# 指数归一化需要先知道非 0 最小/最大值，这里用两步补算（Agent 也应这样做）
print("\n=== 极值标准化补算（非空非 0 的 PER_CAPITA）===")
stats = {}
vals = []
with arcpy.da.SearchCursor(G("community_green"), ["PER_CAPITA"]) as cur:
    for (v,) in cur:
        if v not in (None, 0):
            vals.append(float(v))
gmin, gmax = min(vals), max(vals)
print(f"  非空非 0 记录 {len(vals)} 条；min={gmin:.6f} max={gmax:.6f}")
code = f"""
import arcpy
arcpy.env.overwriteOutput = True
p = r'{G("community_green")}'
gmin, gmax = {gmin!r}, {gmax!r}
with arcpy.da.UpdateCursor(p, ['PER_CAPITA', 'GREEN_IDX']) as cur:
    for row in cur:
        v = row[0]
        row[1] = None if v in (None, 0) else (float(v) - gmin) / (gmax - gmin)
        cur.updateRow(row)
set_result({{'updated': len(vals), 'min': gmin, 'max': gmax}})
"""
res = runner.run(code, workspace=str(WS))
print(f"  [{'OK ' if res.ok else 'FAIL'}] 更新 GREEN_IDX | {(res.result or {})} | err={(res.error or {}).get('message','')[:80]}")

print("\n=== 关键数值核对 ===")
comm_n = lib._feature_count(G("community_green"))
idx_stats = {k: lib._field_stat(G("community_green"), "GREEN_IDX", k) for k in ("min", "max", "null")}
area_sum = lib._field_stat(G("community_green"), "GREEN_AREA", "sum")
inter_area = lib._field_stat(G("green_inter"), "Shape_Area", "sum")
print(f"  社区数 = {comm_n}")
print(f"  绿地面积合计(社区口径) = {area_sum:,.0f} m²")
print(f"  相交绿地面积合计        = {inter_area:,.0f} m²  (差 {abs(area_sum-inter_area)/inter_area*100:.4f}%)")
print(f"  GREEN_IDX min={idx_stats['min']:.6f} max={idx_stats['max']:.6f} 空值={idx_stats['null']:.0f}")

task = yaml.safe_load((REPO / "benchmark/tasks/pm_task2a_green_area_index.yaml").read_text(encoding="utf-8"))
assertions = []
for raw in task["assertions"]:
    item = dict(raw)
    if item.get("path") and not Path(item["path"]).is_absolute():
        item["path"] = str(WS / item["path"])
    assertions.append(item)
print("\n=== 基准断言评估 ===")
res_list = lib.run_assertions(assertions)
ok = 0
for r in res_list:
    ok += 1 if r.ok else 0
    print(f"  [{'OK  ' if r.ok else 'FAIL'}] {r.kind:<24} {r.detail[:80]}")
print(f"断言 {ok}/{len(res_list)}")
runner.shutdown()