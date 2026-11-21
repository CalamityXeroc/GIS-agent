# -*- coding: utf-8 -*-
"""PM 任务一离线确定性验证：数据集入库(4548) → 老年人口栅格分摊 → 社区尺度统计。

只跑配方不走模型，用于在 Agent 实测前确认基准断言的数值门槛正确。
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
DATA = None
for cand in glob.glob(str(REPO / "workspace/input/**/B下午/B下午/数据"), recursive=True):
    DATA = Path(cand)
    break
assert DATA, "找不到 PM 数据"

WS = Path(r"C:\Users\cheng\AppData\Local\Temp") / ("pm1_" + str(int(time.time())))
(WS / "input").mkdir(parents=True)
(WS / "output").mkdir(parents=True)
for name in ("district", "community"):
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
        f = DATA / f"{name}{ext}"
        if f.exists():
            shutil.copy2(f, WS / "input" / f.name)
for f in DATA.glob("pop_density.tif*"):
    shutil.copy2(f, WS / "input" / f.name)
print("WS:", WS)

runner = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "kernel")
lib = RecipeLibrary(code_runner=runner)
lib.load()


class Ctx:
    workspace = WS
    code_runner = runner


GDB = str(WS / "output" / "result_data.gdb")
I = lambda n: str(WS / "input" / n)
G = lambda n: GDB + "\\" + n
SRID = 4548

steps = [
    ("建结果库", "create_file_gdb", {"gdb_path": GDB}),
    ("投影 district → 4548", "project_to_crs", {"input_path": I("district.shp"), "output_path": G("district_prj"), "target_srid": SRID}),
    ("投影 community → 4548", "project_to_crs", {"input_path": I("community.shp"), "output_path": G("community_prj"), "target_srid": SRID}),
    ("投影栅格 → 4548 / 5m", "project_raster", {"input_path": I("pop_density.tif"), "output_path": G("pop_density_prj"), "target_srid": SRID, "cell_size": 5}),
    ("计算 ELDER_N（百分比×常住人口）", "field_calculate",
     {"input_path": G("district_prj"), "field_name": "ELDER_N",
      "expression": "float(!老年人![:-1])/100.0*float(!常住人!)", "field_type": "DOUBLE"}),
    ("街区老年人口 → 5m 栅格", "polygon_to_raster",
     {"input_path": G("district_prj"), "value_field": "ELDER_N", "output_path": G("elder_raster"), "cell_size": 5}),
    ("街区人口密度总值栅格", "zonal_statistics_raster",
     {"zone_path": G("district_prj"), "zone_field": "", "raster_path": G("pop_density_prj"),
      "output_path": G("zone_sum"), "statistics": "SUM"}),
    ("分摊计算", "raster_calculator",
     {"inputs": f"density={G('pop_density_prj')},zone_sum={G('zone_sum')},elder={G('elder_raster')}",
      "expression": "Con(IsNull(zone_sum), 0, Con(zone_sum == 0, 0, density / zone_sum * elder))",
      "output_path": G("elder_alloc")}),
    ("补缺（NoData→0）", "raster_calculator",
     {"inputs": f"alloc={G('elder_alloc')}", "expression": "Con(IsNull(alloc), 0, alloc)",
      "output_path": G("elder_fill")}),
    ("社区尺度分区统计（表）", "zonal_statistics",
     {"zone_path": G("community_prj"), "raster_path": G("elder_fill"), "output_table": G("community_elder_tbl"),
      "zone_field": "OBJECTID", "statistics": "SUM"}),
    ("统计表挂回社区", "attribute_join_table",
     {"target_path": G("community_prj"), "target_key": "OBJECTID", "join_table": G("community_elder_tbl"),
      "join_key": "OBJECTID", "join_fields": "SUM", "output_path": G("community_elder")}),
    ("统计字段更名/新增 ELDER_COMM", "field_calculate",
     {"input_path": G("community_elder"), "field_name": "ELDER_COMM",
      "expression": "!SUM!", "field_type": "DOUBLE"}),
]
for label, rid, params in steps:
    t0 = time.time()
    obs = lib.run(rid, params, Ctx())
    print(f"[{'OK ' if obs.ok else 'FAIL'}] {label:34s} {rid:26s} {time.time()-t0:5.1f}s | {str(obs.summary)[:90]}")

# 关键数值核对
print("\n=== 关键数值核对 ===")
total_elder = lib._field_stat(G("district_prj"), "ELDER_N", "sum")
info = lib._raster_stats(G("elder_fill")) or {}
rsum = info.get("sum")
print(f"  街区老年人口合计 = {total_elder:,.0f}")
if rsum is not None and total_elder:
    print(f"  分摊栅格合计     = {rsum:,.0f} | 误差 = {abs(rsum-total_elder)/total_elder*100:.3f}%")
else:
    print(f"  分摊栅格合计     = {rsum} | 街区合计={total_elder}（数据缺失，跳过误差）")
print(f"  栅格 NoData 数   = {info.get('nodata')}")
# 社区尺度字段（挂回后字段名可能带前缀，找出实际字段）
flds = [f.name for f in arcpy.ListFields(G("community_elder"))]
cand = [f for f in flds if "SUM" in f.upper()]
print(f"  社区层字段: {flds[-4:]}")
if cand:
    comm_fld = cand[-1]
    csum = lib._field_stat(G("community_elder"), comm_fld, "sum")
    cnull = lib._field_stat(G("community_elder"), comm_fld, "null")
    print(f"  社区尺度合计 = {csum:,.0f} | 空值单元 = {cnull:.0f}")

# 基准断言评估
task = yaml.safe_load((REPO / "benchmark/tasks/pm_task1_elderly_raster.yaml").read_text(encoding="utf-8"))
assertions = []
for raw in task["assertions"]:
    item = dict(raw)
    if item.get("path") and not Path(item["path"]).is_absolute():
        item["path"] = str(WS / item["path"])
    if not item.get("path"):
        item["path"] = ""
    item["path"] = item["path"].replace("\\", "\\")
    assertions.append(item)
print("\n=== 基准断言评估 ===")
res = lib.run_assertions(assertions)
ok = 0
for r in res:
    ok += 1 if r.ok else 0
    print(f"  [{'OK  ' if r.ok else 'FAIL'}] {r.kind:<22} {r.detail[:78]}")
print(f"断言 {ok}/{len(res)}")
runner.shutdown()