# -*- coding: utf-8 -*-
"""最小复现：ArcGIS GP 工具在持久内核里是否触发 InitDatetime 致命错误。"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")
from pathlib import Path
from gis_cli.runtime.code_runner import CodeRunner

WS = Path(r"C:\Users\cheng\AppData\Local\Temp\min_ws")
WS.mkdir(parents=True, exist_ok=True)
r = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "k1")

print("步骤1: 内核里 import arcpy")
res = r.run("import arcpy\nset_result({'v': arcpy.GetInstallInfo()['Version']})", workspace=str(WS))
print("   ->", res.ok, (res.result or {}), str((res.error or {}).get("message", ""))[:80])

print("步骤2: 建要素类 + MakeFeatureLayer + SelectLayerByAttribute")
STEP2 = "\n".join([
    "import arcpy, os",
    "arcpy.env.overwriteOutput = True",
    "root = r'" + str(WS) + "'",
    "gdb = os.path.join(root, 't.gdb')",
    "if not arcpy.Exists(gdb):",
    "    arcpy.management.CreateFileGDB(root, 't.gdb')",
    "fc = os.path.join(gdb, 'pts')",
    "if arcpy.Exists(fc):",
    "    arcpy.management.Delete(fc)",
    "arcpy.management.CreateFeatureclass(gdb, 'pts', 'POINT', spatial_reference=arcpy.SpatialReference(4326))",
    "arcpy.management.AddField(fc, 'NAME', 'TEXT', field_length=16)",
    "with arcpy.da.InsertCursor(fc, ['SHAPE@XY', 'NAME']) as cur:",
    "    cur.insertRow([(116.0, 39.0), 'a'])",
    "    cur.insertRow([(117.0, 40.0), 'b'])",
    "lyr = arcpy.management.MakeFeatureLayer(fc, 'lyr').getOutput(0)",
    "arcpy.management.SelectLayerByAttribute(lyr, 'NEW_SELECTION', 'OBJECTID = 1')",
    "set_result({'selected': int(arcpy.management.GetCount(lyr)[0])})",
])
t0 = time.time()
res2 = r.run(STEP2, timeout=180, workspace=str(WS))
print(f"   -> {res2.ok} {time.time()-t0:.1f}s {res2.result} err={str((res2.error or {}).get('message',''))[:150]}")

print("步骤3: 再跑一次普通栅格/矢量工具（ArcGIS 是否整体损坏）")
res3 = r.run("import arcpy\nn = int(arcpy.management.GetCount(r'" + str(WS) + "\\t.gdb\\pts')[0])\nset_result({'n': n})",
             timeout=180, workspace=str(WS))
print("   ->", res3.ok, res3.result, str((res3.error or {}).get("message", ""))[:100])
r.shutdown()