# -*- coding: utf-8 -*-
"""PM 任务二b 离线确定性验证：中心点 → 500m 缓冲 → 缓冲区老年人口分区统计 →
服务价值 → 前后 500 名标记 → 标杆/需整改名单。

承接任务二a 的工作区（community_green + elder_fill）。用于校准 Agent 实测前的断言门槛。
"""
import sys, io, glob, shutil, time
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

runs = sorted(glob.glob(str(REPO / "benchmark/runs/*/pm_task2a_green_area_index/workspace")), reverse=True)
assert runs, "找不到任务二a的工作区"
PREV = Path(runs[0])
WS = Path(r"C:\Users\cheng\AppData\Local\Temp") / ("pm2b_" + str(int(time.time())))
(WS / "input").mkdir(parents=True)
(WS / "output").mkdir(parents=True)
for name in ("district", "community", "green_land"):
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
        f = Path(DATA) / f"{name}{ext}"
        if f.exists():
            shutil.copy2(f, WS / "input" / f.name)
for f in Path(DATA).glob("pop_density.tif*"):
    shutil.copy2(f, WS / "input" / f.name)
prev_out = PREV / "output"
for item in prev_out.iterdir():
    target = WS / "output" / item.name
    if not target.exists():
        shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
print(f"承接任务二a产物: {prev_out}")

runner = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "kernel")
lib = RecipeLibrary(code_runner=runner)
lib.load()


class Ctx:
    workspace = WS
    code_runner = runner


GDB = str(WS / "output" / "result_data.gdb")
G = lambda n: GDB + "\\" + n

steps = [
    ("社区要素转点", "feature_to_centroid",
     {"input_path": G("community_green"), "output_path": G("comm_centroid"), "location": "CENTROID"}),
    ("500m 缓冲区（不融合）", "buffer_dissolve",
     {"input_path": G("comm_centroid"), "output_path": G("comm_buffer_500"), "distance": "500 Meters",
      "dissolve_field": "", "dissolve_type": "NONE"}),
    ("缓冲区老年人口求和（重叠安全）", "point_buffer_raster_sum",
     {"points_path": G("comm_centroid"), "id_field": "BSM", "raster_path": G("elder_fill"),
      "radius": 500, "output_table": G("buffer_elder_tbl")}),
    ("服务人口挂回社区", "attribute_join_table",
     {"target_path": G("community_green"), "target_key": "BSM", "join_table": G("buffer_elder_tbl"),
      "join_key": "BSM", "join_fields": "SUM_VALUE", "output_path": G("community_service")}),
    ("服务人数字段更名 SERVICE_POP", "field_calculate",
     {"input_path": G("community_service"), "field_name": "SERVICE_POP", "expression": "!SUM_VALUE!",
      "field_type": "DOUBLE"}),
    ("绿地服务价值 SERVICE_VALUE", "field_calculate",
     {"input_path": G("community_service"), "field_name": "SERVICE_VALUE",
      "expression": "None if (!SERVICE_POP! is None or !GREEN_AREA! in (None, 0)) else !SERVICE_POP!/!GREEN_AREA!",
      "field_type": "DOUBLE", "expression_type": "PYTHON3"}),
    ("人均绿地指数排名", "rank_top_bottom_flag",
     {"input_path": G("community_service"), "rank_field": "PER_CAPITA", "output_path": G("svc_rank_per"),
      "top_n": 500, "bottom_n": 500, "bottom_include_null": True, "second_field": "ELDER_COMM"}),
    ("服务价值排名", "rank_top_bottom_flag",
     {"input_path": G("svc_rank_per"), "rank_field": "SERVICE_VALUE", "output_path": G("svc_rank_all"),
      "top_n": 500, "bottom_n": 500, "bottom_include_null": True, "second_field": "ELDER_COMM"}),
]
SLOW = {"zonal_statistics": 2400.0}  # 缓冲区分区统计很慢（参考文档也"足足等了 8 分钟"）
for label, rid, params in steps:
    t0 = time.time()
    obs = lib.run(rid, params, Ctx(), timeout=SLOW.get(rid))
    print(f"[{'OK ' if obs.ok else 'FAIL'}] {label:30s} {rid:22s} {time.time()-t0:6.1f}s | {str(obs.summary)[:110]}")

# ---- 合并四个标记字段（两次排名结果 → TOP_PER/TOP_SVC/BOT_PER/BOT_SVC）
print("\n=== 合并四个标记字段 ===")
MERGE_CODE = """
import arcpy
arcpy.env.overwriteOutput = True
prev = r'__PREV__'
src = r'__SRC__'
out = r'__OUT__'
flags = {}
with arcpy.da.SearchCursor(prev, ['BSM', 'TOP_FLAG', 'BOTTOM_FLAG']) as cur:
    for bsm, t, b in cur:
        flags.setdefault(bsm, {}).update({'TOP_PER': t, 'BOT_PER': b})
with arcpy.da.SearchCursor(src, ['BSM', 'TOP_FLAG', 'BOTTOM_FLAG']) as cur:
    for bsm, t, b in cur:
        flags.setdefault(bsm, {}).update({'TOP_SVC': t, 'BOT_SVC': b})
have = {f.name for f in arcpy.ListFields(out)}
for f in ('TOP_PER', 'TOP_SVC', 'BOT_PER', 'BOT_SVC'):
    if f not in have:
        arcpy.management.AddField(out, f, 'SHORT')
cnt = {'TOP_PER': 0, 'TOP_SVC': 0, 'BOT_PER': 0, 'BOT_SVC': 0}
with arcpy.da.UpdateCursor(out, ['BSM', 'TOP_PER', 'TOP_SVC', 'BOT_PER', 'BOT_SVC']) as cur:
    for row in cur:
        v = flags.get(row[0], {})
        for i, f in enumerate(('TOP_PER', 'TOP_SVC', 'BOT_PER', 'BOT_SVC'), start=1):
            row[i] = int(v.get(f) or 0)
            cnt[f] += row[i]
        cur.updateRow(row)
set_result({'counts': cnt})
"""
code = (MERGE_CODE.replace("__PREV__", G("svc_rank_per"))
                   .replace("__SRC__", G("svc_rank_all"))
                   .replace("__OUT__", G("community_service")))
res = runner.run(code, workspace=str(WS))
print(f"[{'OK ' if res.ok else 'FAIL'}] 合并标记字段 | {(res.result or {}).get('counts')} | err={(res.error or {}).get('message','')[:80]}")

# ---- 导出标杆社区 / 需整改社区
LIST_CODE = """
import arcpy
arcpy.env.overwriteOutput = True
src = r'__SRC__'
out_bm = r'__OUT_BM__'
out_im = r'__OUT_IM__'
for out, where in ((out_bm, 'TOP_PER = 1 AND TOP_SVC = 1'), (out_im, 'BOT_PER = 1 AND BOT_SVC = 1')):
    if arcpy.Exists(out):
        arcpy.management.Delete(out)
    arcpy.analysis.Select(src, out, where)
    print(out.split(chr(92))[-1], '=', int(arcpy.management.GetCount(out)[0]))
set_result({'done': True})
"""
code2 = (LIST_CODE.replace("__SRC__", G("community_service"))
                  .replace("__OUT_BM__", G("标杆社区"))
                  .replace("__OUT_IM__", G("需整改社区")))
res2 = runner.run(code2, workspace=str(WS))
print(f"[{'OK ' if res2.ok else 'FAIL'}] 导出两个名单")
print((res2.stdout or "")[-400:])

print("\n=== 关键数值核对 ===")
buf_n = lib._feature_count(G("comm_buffer_500"))
buf_area = lib._field_stat(G("comm_buffer_500"), "Shape_Area", "mean")
svc_null = lib._field_stat(G("community_service"), "SERVICE_VALUE", "null")
svc_max = lib._field_stat(G("community_service"), "SERVICE_VALUE", "max")
pop_max = lib._field_stat(G("community_service"), "SERVICE_POP", "max")
counts = {f: lib._field_stat(G("community_service"), f, "sum") for f in ("TOP_PER", "TOP_SVC", "BOT_PER", "BOT_SVC")}
bm = lib._feature_count(G("标杆社区"))
im = lib._feature_count(G("需整改社区"))
print(f"  缓冲区数={buf_n} 平均面积={buf_area:,.0f} m2（理论 {3.14159*500*500:,.0f}）")
print(f"  SERVICE_VALUE 空值={svc_null:.0f} 最大={svc_max:.6f} | SERVICE_POP 最大={pop_max:,.0f}")
print("  标记计数:", {k: int(v) for k, v in counts.items()})
print(f"  标杆社区={bm} 需整改社区={im}")
print("  [参考文档] 标杆=34 需整改=500 服务价值空值=814")

task = yaml.safe_load((REPO / "benchmark/tasks/pm_task2b_service_value_rank.yaml").read_text(encoding="utf-8"))
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
    print(f"  [{'OK  ' if r.ok else 'FAIL'}] {r.kind:<24} {r.detail[:78]}")
print(f"断言 {ok}/{len(res_list)}")
runner.shutdown()