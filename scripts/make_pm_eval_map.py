# -*- coding: utf-8 -*-
"""PM 任务二收尾出图：标杆社区(蓝) / 需整改社区(绿) 分类设色图 + .aprx 工程。

复用任务二b 的运行工作区产物（community_service、标杆社区、需整改社区）。
"""
import sys, io, glob, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

import arcpy
arcpy.env.overwriteOutput = True
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

REPO = Path(r"E:\BST_lab\GIS_CLI")
runs = sorted(glob.glob(str(REPO / "benchmark/runs/*/pm_task2b_service_value_rank/workspace")), reverse=True)
assert runs, "找不到任务二b的工作区"
WS = Path(runs[0])
OUT_DIR = REPO / "workspace" / "成果展示" / "PM能力测试"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"承接工作区: {WS}")

runner = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "kernel_map")
lib = RecipeLibrary(code_runner=runner)
lib.load()


class Ctx:
    workspace = WS
    code_runner = runner


GDB = str(WS / "output" / "result_data.gdb")

# 1) 造分类字段：标杆社区 / 需整改社区 / 其他
BUILD = """
import arcpy
arcpy.env.overwriteOutput = True
GDB = r'__GDB__'
src = GDB + '\\\\community_service'
out = GDB + '\\\\community_eval'
if arcpy.Exists(out):
    arcpy.management.Delete(out)
arcpy.management.CopyFeatures(src, out)
if '类别' in [f.name for f in arcpy.ListFields(out)]:
    arcpy.management.DeleteField(out, '类别')
arcpy.management.AddField(out, '类别', 'TEXT', field_length=32)
bm = set()
with arcpy.da.SearchCursor(GDB + '\\\\标杆社区', ['BSM']) as cur:
    for (v,) in cur:
        bm.add(int(round(float(v))))
im = set()
with arcpy.da.SearchCursor(GDB + '\\\\需整改社区', ['BSM']) as cur:
    for (v,) in cur:
        im.add(int(round(float(v))))
cnt = {'标杆社区': 0, '需整改社区': 0, '其他': 0}
with arcpy.da.UpdateCursor(out, ['BSM', '类别']) as cur:
    for bsm, _ in cur:
        key = int(round(float(bsm))) if bsm is not None else -1
        label = '标杆社区' if key in bm else ('需整改社区' if key in im else '其他')
        cnt[label] += 1
        cur.updateRow([bsm, label])
set_result({'counts': cnt, 'output': out})
"""
res = runner.run(BUILD.replace("__GDB__", GDB.replace("\\", "\\\\")), workspace=str(WS))
print(f"[{'OK ' if res.ok else 'FAIL'}] 造分类字段 | {res.result} | {str((res.error or {}).get('message',''))[:80]}")

# 2) 出图（分类设色 + .aprx）
target = OUT_DIR / "pm_标杆社区与需整改社区.jpg"
t0 = time.time()
obs = lib.run("category_map", {
    "input_path": GDB + "\\community_eval",
    "category_field": "类别",
    "category_colors": "标杆社区:#1F77B4;需整改社区:#2CA02C",
    "other_color": "#E8E8E8",
    "output_path": str(target),
    "title": "社区绿地服务评价：标杆社区与需整改社区",
}, Ctx(), timeout=900)
print(f"[{'OK ' if obs.ok else 'FAIL'}] 出图 {time.time()-t0:.1f}s | {obs.summary[:160]}")
print("  result:", str((obs.data or {}).get('result'))[:300])
print("  stdout 尾:", str((obs.data or {}).get('stdout',''))[-300:])

# 3) 工程核验交给专用脚本：
#    python scripts/verify_map_aprx.py "<生成的 .aprx>"
print("
工程核验请运行: python scripts/verify_map_aprx.py \"%s\"" % target.with_suffix(".aprx"))

# 4) 程序化看图
IMG_CHECK = """
from PIL import Image
from collections import Counter
import numpy as np
im = Image.open(r'__IMG__').convert('RGB')
a = np.asarray(im).reshape(-1, 3)
# 统计是否出现设定的蓝/绿
def near(px, target, tol=40):
    return abs(int(px[0]) - target[0]) < tol and abs(int(px[1]) - target[1]) < tol and abs(int(px[2]) - target[2]) < tol
blue = sum(1 for p in a[::37] if near(p, (31, 119, 180)))
green = sum(1 for p in a[::37] if near(p, (44, 160, 44)))
print('抽样像素:', len(a[::37]), '| 蓝(标杆)=', blue, '| 绿(需整改)=', green)
print('白/浅灰占比:', round(float(((a > 235).all(axis=1)).mean()), 3))
set_result({'blue': blue, 'green': green})
"""
res3 = runner.run(IMG_CHECK.replace("__IMG__", str(target)), workspace=str(WS))
print("\n=== 程序化看图 ===")
print((res3.stdout or "")[-500:])
runner.shutdown()